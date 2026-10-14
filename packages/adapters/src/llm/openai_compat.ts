import { ContextOverflowError, type ChatChunk, type ChatModelPort, type ChatRequest, type EmbeddingPort } from "@rlm/core";
import { ThinkSplitter } from "./think";

const OVERFLOW_RE = /context.length|maximum context|too many tokens|context_length_exceeded|prompt is too long|exceeds the (model'?s )?context|input.*too long/i;

export interface OpenAICompatConfig {
  baseUrl: string;
  apiKey: string | null;
  model: string;
  /** Fail when the server sends nothing for this long. */
  stallMs?: number;
}

/** Aborts `ctl` when `touch()` isn't called for `ms`; the reason names the stall. */
function stallGuard(ctl: AbortController, ms: number | undefined, what: string): { touch: () => void; done: () => void } {
  if (!ms) return { touch: () => undefined, done: () => undefined };
  let timer: ReturnType<typeof setTimeout> | undefined;
  const touch = () => {
    clearTimeout(timer);
    timer = setTimeout(() => ctl.abort(new Error(`${what}: the server sent nothing for ${Math.round(ms / 1000)} s and looks stuck`)), ms);
  };
  touch();
  return { touch, done: () => clearTimeout(timer) };
}

function headers(key: string | null): Record<string, string> {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (key) h.Authorization = `Bearer ${key}`;
  return h;
}

function url(base: string, p: string): string {
  return `${base.replace(/\/+$/, "")}${p}`;
}

/** Servers sometimes echo the bearer token back in error bodies; never let it reach logs or the UI. */
export function redactKey(text: string, key: string | null): string {
  let out = text;
  if (key && key.length >= 4) out = out.split(key).join("[redacted]");
  return out.replace(/(Bearer\s+)[A-Za-z0-9._~+/=-]{8,}/gi, "$1[redacted]");
}

async function post(cfg: OpenAICompatConfig, p: string, body: unknown, signal?: AbortSignal): Promise<Response> {
  try {
    return await fetch(url(cfg.baseUrl, p), { method: "POST", headers: headers(cfg.apiKey), body: JSON.stringify(body), signal });
  } catch (e) {
    if ((e as Error).name === "AbortError" || signal?.aborted) throw e;
    const cause = (e as { cause?: { code?: string; message?: string } }).cause;
    throw new Error(`cannot reach ${url(cfg.baseUrl, p)}: ${cause?.code ?? cause?.message ?? (e as Error).message}`);
  }
}

export class ModelHttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ModelHttpError";
  }
}

export function httpError(what: string, status: number, body: string, cfg: OpenAICompatConfig): ModelHttpError {
  const text = redactKey(body, cfg.apiKey).slice(0, 500);
  if (status === 401 || status === 403) {
    const why = cfg.apiKey ? "the server rejected the API key" : "the server requires an API key and none is set for this provider";
    return new ModelHttpError(status, `${what} failed (${status}): ${why} (${cfg.baseUrl}). Check the key in Settings → Models → Providers.${text ? ` Server said: ${text}` : ""}`);
  }
  return new ModelHttpError(status, `${what} failed (${status}): ${text}`);
}

export class OpenAICompatChat implements ChatModelPort {
  constructor(
    readonly modelId: string,
    private readonly cfg: OpenAICompatConfig,
  ) {}

  async *stream(req: ChatRequest): AsyncIterable<ChatChunk> {
    const body: Record<string, unknown> = {
      model: this.cfg.model,
      messages: req.messages,
      max_tokens: req.maxTokens,
      temperature: req.temperature ?? 0.3,
      stream: true,
      stream_options: { include_usage: true },
    };
    if (req.stop?.length) body.stop = req.stop;
    if (req.enableThinking === false) body.chat_template_kwargs = { enable_thinking: false };
    const ctl = new AbortController();
    const onAbort = () => ctl.abort(req.signal?.reason);
    if (req.signal?.aborted) onAbort();
    req.signal?.addEventListener("abort", onAbort, { once: true });
    const guard = stallGuard(ctl, this.cfg.stallMs, `model request to ${this.cfg.baseUrl}`);
    const stalled = (e: unknown) => (ctl.signal.aborted && !req.signal?.aborted && ctl.signal.reason instanceof Error ? ctl.signal.reason : e);
    try {
      let res: Response;
      try {
        res = await post(this.cfg, "/chat/completions", body, ctl.signal);
      } catch (e) {
        throw stalled(e);
      }
      if (!res.ok || !res.body) {
        const text = await res.text().catch(() => "");
        if (res.status === 400 || res.status === 413) {
          if (OVERFLOW_RE.test(text)) throw new ContextOverflowError(text.slice(0, 500));
        }
        throw httpError("model request", res.status, text, this.cfg);
      }
      yield* this.parse(res.body.getReader(), guard.touch, stalled);
    } finally {
      guard.done();
      req.signal?.removeEventListener("abort", onAbort);
      if (!ctl.signal.aborted) ctl.abort();
    }
  }

  private async *parse(reader: ReadableStreamDefaultReader<Uint8Array>, touch: () => void, stalled: (e: unknown) => unknown): AsyncIterable<ChatChunk> {
    const splitter = new ThinkSplitter(false);
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      let r: Awaited<ReturnType<typeof reader.read>>;
      try {
        r = await reader.read();
      } catch (e) {
        throw stalled(e);
      }
      const { done, value } = r;
      if (done) break;
      touch();
      buf += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        let j: {
          choices?: Array<{ delta?: { content?: string | null; reasoning_content?: string | null; reasoning?: string | null } }>;
          usage?: { prompt_tokens?: number; completion_tokens?: number };
          error?: { message?: string };
        };
        try {
          j = JSON.parse(data);
        } catch {
          continue;
        }
        if (j.error) {
          const m = redactKey(j.error.message ?? "stream error", this.cfg.apiKey);
          if (OVERFLOW_RE.test(m)) throw new ContextOverflowError(m);
          throw new Error(m);
        }
        const d = j.choices?.[0]?.delta;
        const reasoning = d?.reasoning_content ?? d?.reasoning;
        if (reasoning) yield { type: "thinking", text: reasoning };
        if (d?.content) for (const part of splitter.push(d.content)) yield part;
        if (j.usage) yield { type: "usage", inputTokens: j.usage.prompt_tokens ?? 0, outputTokens: j.usage.completion_tokens ?? 0 };
      }
    }
    for (const part of splitter.flush()) yield part;
  }
}

export class OpenAICompatEmbed implements EmbeddingPort {
  readonly modelKey: string;
  constructor(
    private readonly cfg: OpenAICompatConfig,
    readonly dim: number,
  ) {
    this.modelKey = `${cfg.model}__${dim}`;
  }

  async embed(texts: string[], signal?: AbortSignal): Promise<number[][]> {
    const ctl = new AbortController();
    const onAbort = () => ctl.abort(signal?.reason);
    if (signal?.aborted) onAbort();
    signal?.addEventListener("abort", onAbort, { once: true });
    const guard = stallGuard(ctl, this.cfg.stallMs, `embedding request to ${this.cfg.baseUrl}`);
    try {
      const res = await post(this.cfg, "/embeddings", { model: this.cfg.model, input: texts }, ctl.signal);
      if (!res.ok) throw httpError("embedding request", res.status, await res.text().catch(() => ""), this.cfg);
      const j = (await res.json()) as { data: Array<{ embedding: number[]; index: number }> };
      return j.data.sort((a, b) => a.index - b.index).map((d) => d.embedding);
    } catch (e) {
      throw ctl.signal.aborted && !signal?.aborted && ctl.signal.reason instanceof Error ? ctl.signal.reason : e;
    } finally {
      guard.done();
      signal?.removeEventListener("abort", onAbort);
    }
  }
}

/** Best-effort context length detection for OpenAI-compatible servers. */
export async function detectOpenAICompatLimits(cfg: OpenAICompatConfig): Promise<{ contextLength: number | null; maxOutputTokens: number | null }> {
  let authError: ModelHttpError | null = null;
  try {
    const res = await fetch(url(cfg.baseUrl, "/models"), { headers: headers(cfg.apiKey), signal: AbortSignal.timeout(8000) });
    if (res.status === 401 || res.status === 403) authError = httpError("model listing", res.status, await res.text().catch(() => ""), cfg);
    if (res.ok) {
      const j = (await res.json()) as { data?: Array<Record<string, unknown>> };
      const m = j.data?.find((x) => x.id === cfg.model);
      if (m) {
        const top = m.top_provider as Record<string, unknown> | undefined;
        const meta = m.meta as Record<string, unknown> | undefined;
        const ctx = Number(m.max_model_len ?? m.context_length ?? m.context_window ?? top?.context_length ?? meta?.n_ctx ?? 0) || null;
        const out = Number(top?.max_completion_tokens ?? m.max_output_tokens ?? 0) || null;
        if (ctx) return { contextLength: ctx, maxOutputTokens: out };
      }
    }
  } catch {
    /* try llama.cpp next */
  }
  const root = cfg.baseUrl.replace(/\/v1\/?$/, "");
  try {
    const res = await fetch(`${root}/props`, { headers: headers(cfg.apiKey), signal: AbortSignal.timeout(8000) });
    if (res.ok) {
      const j = (await res.json()) as { default_generation_settings?: { n_ctx?: number }; n_ctx?: number };
      const ctx = Number(j.default_generation_settings?.n_ctx ?? j.n_ctx ?? 0) || null;
      if (ctx) return { contextLength: ctx, maxOutputTokens: null };
    }
  } catch {
    /* try ollama next */
  }
  try {
    const res = await fetch(`${root}/api/show`, { method: "POST", headers: headers(cfg.apiKey), body: JSON.stringify({ model: cfg.model }), signal: AbortSignal.timeout(8000) });
    if (res.ok) {
      authError = null;
      const j = (await res.json()) as { model_info?: Record<string, unknown> };
      const key = Object.keys(j.model_info ?? {}).find((k) => k.endsWith(".context_length"));
      const ctx = key ? Number(j.model_info![key]) : null;
      if (ctx) return { contextLength: ctx, maxOutputTokens: null };
    }
  } catch {
    /* unknown */
  }
  if (authError) throw authError;
  return { contextLength: null, maxOutputTokens: null };
}
