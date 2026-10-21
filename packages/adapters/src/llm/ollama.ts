import os from "node:os";
import { ContextOverflowError, type ChatChunk, type ChatModelPort, type ChatRequest } from "@rlm/core";
import { foldSystemPrompt } from "@rlm/contracts";
import { headers, httpError, redactKey, stallGuard, type OpenAICompatConfig } from "./openai_compat";
import { ThinkSplitter } from "./think";

/** Ollama's OpenAI-compatible endpoint cannot set the context window: the server then runs its default (often 4096
 * tokens) and silently drops the start of longer prompts, or a large window the user configured fills the RAM. The
 * native API takes `num_ctx`, so the IDE decides the window and sizes its prompts to it. */

export function ollamaRoot(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, "").replace(/\/v1$/, "");
}

const probes = new Map<string, { at: number; result: Promise<boolean>; settled?: boolean }>();
const NEGATIVE_TTL_MS = 5 * 60_000;

/** True when the provider's server is Ollama (it answers /api/version). A "no" is asked again after a few minutes,
 * in case Ollama was started after the IDE. */
export function isOllama(baseUrl: string, apiKey: string | null): Promise<boolean> {
  const root = ollamaRoot(baseUrl);
  const hit = probes.get(root);
  if (hit && (hit.settled !== false || Date.now() - hit.at < NEGATIVE_TTL_MS)) return hit.result;
  const result = fetch(`${root}/api/version`, { headers: headers(apiKey), signal: AbortSignal.timeout(2500) })
    .then(async (r) => r.ok && typeof ((await r.json()) as { version?: unknown }).version === "string")
    .catch(() => false);
  const entry: { at: number; result: Promise<boolean>; settled?: boolean } = { at: Date.now(), result };
  void result.then((v) => (entry.settled = v));
  probes.set(root, entry);
  return result;
}

export interface OllamaPlan {
  /** num_ctx sent with every request. */
  numCtx: number;
  /** The model's own window. */
  native: number | null;
  /** The model has a reasoning mode that `think` switches. */
  thinking: boolean;
  why: string;
  /** Shown to the user once: the weights alone need more memory than is free. */
  warning?: string;
}

export interface OllamaPlanOptions {
  cap: number;
  kvRamShare: number;
  /** The window the user configured for this model, if any. */
  configured: number | null;
  freeMem?: () => number;
}

const plans = new Map<string, Promise<OllamaPlan>>();

/** Picks num_ctx once per model and server: changing it makes Ollama reload the model. */
export function ollamaPlan(cfg: OpenAICompatConfig, o: OllamaPlanOptions): Promise<OllamaPlan> {
  const key = `${ollamaRoot(cfg.baseUrl)}|${cfg.model}|${o.cap}|${o.kvRamShare}|${o.configured ?? ""}`;
  let p = plans.get(key);
  if (!p) {
    p = computePlan(cfg, o);
    plans.set(key, p);
    p.catch(() => plans.delete(key));
  }
  return p;
}

const isLocalHost = (baseUrl: string) => {
  try {
    return ["localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0"].includes(new URL(baseUrl).hostname);
  } catch {
    return false;
  }
};

async function computePlan(cfg: OpenAICompatConfig, o: OllamaPlanOptions): Promise<OllamaPlan> {
  const root = ollamaRoot(cfg.baseUrl);
  const res = await fetch(`${root}/api/show`, { method: "POST", headers: headers(cfg.apiKey), body: JSON.stringify({ model: cfg.model }), signal: AbortSignal.timeout(10_000) });
  if (!res.ok) throw httpError("Ollama model info", res.status, await res.text().catch(() => ""), cfg);
  const show = (await res.json()) as { model_info?: Record<string, unknown>; parameters?: string; capabilities?: string[] };
  const info = show.model_info ?? {};
  const arch = String(info["general.architecture"] ?? "");
  const num = (k: string) => {
    const v = Number(info[`${arch}.${k}`]);
    return Number.isFinite(v) && v > 0 ? v : null;
  };
  const native = num("context_length");
  const fromModelfile = Number(/(?:^|\n)\s*num_ctx\s+(\d+)/.exec(show.parameters ?? "")?.[1] ?? 0) || null;
  let want = Math.min(o.configured ?? Infinity, fromModelfile ?? o.cap, native ?? Infinity);
  if (!Number.isFinite(want)) want = o.cap;
  let why = fromModelfile ? `Modelfile num_ctx ${fromModelfile}` : `cap ${o.cap}`;

  let warning: string | undefined;
  const layers = num("block_count");
  const heads = num("attention.head_count");
  const kvHeads = num("attention.head_count_kv") ?? heads;
  const embed = num("embedding_length");
  const keyLen = num("attention.key_length") ?? (embed && heads ? embed / heads : null);
  const valLen = num("attention.value_length") ?? keyLen;
  if (isLocalHost(cfg.baseUrl) && layers && kvHeads && keyLen && valLen) {
    const kvPerToken = layers * kvHeads * (keyLen + valLen) * 2;
    const loaded = await loadedSize(root, cfg);
    const weights = loaded === null ? await modelSize(root, cfg) : 0;
    const freeNow = (o.freeMem ?? os.freemem)();
    if (weights > freeNow * 0.9) {
      const gb = (n: number) => (n / 2 ** 30).toFixed(1);
      warning = `Ollama model ${cfg.model} needs about ${gb(weights)} GB but only ${gb(freeNow)} GB of memory is free: the computer will slow down badly. Close other programs or pick a smaller model or quantization.`;
    }
    const free = Math.max(0, freeNow - weights);
    const fit = Math.floor((free * o.kvRamShare) / kvPerToken);
    if (fit < want) {
      want = fit;
      why = `${Math.round(free / 2 ** 20)} MB free after the weights; ${Math.round(kvPerToken / 1024)} KB of KV cache per token`;
    }
  }
  const numCtx = Math.max(2048, Math.floor(want / 1024) * 1024);
  return { numCtx, native, thinking: (show.capabilities ?? []).includes("thinking"), why, warning };
}

async function modelSize(root: string, cfg: OpenAICompatConfig): Promise<number> {
  try {
    const r = await fetch(`${root}/api/tags`, { headers: headers(cfg.apiKey), signal: AbortSignal.timeout(5000) });
    const j = (await r.json()) as { models?: Array<{ name?: string; model?: string; size?: number }> };
    return j.models?.find((m) => sameModel(m.name ?? m.model ?? "", cfg.model))?.size ?? 0;
  } catch {
    return 0;
  }
}

/** Bytes the model already holds when it is loaded (its weights are then part of used memory), else null. */
async function loadedSize(root: string, cfg: OpenAICompatConfig): Promise<number | null> {
  try {
    const r = await fetch(`${root}/api/ps`, { headers: headers(cfg.apiKey), signal: AbortSignal.timeout(5000) });
    const j = (await r.json()) as { models?: Array<{ name?: string; model?: string; size?: number }> };
    return j.models?.find((m) => sameModel(m.name ?? m.model ?? "", cfg.model))?.size ?? null;
  } catch {
    return null;
  }
}

const withTag = (n: string) => (n.includes(":") ? n : `${n}:latest`);
const sameModel = (a: string, b: string) => withTag(a) === withTag(b);

export class OllamaChat implements ChatModelPort {
  constructor(
    readonly modelId: string,
    private readonly cfg: OpenAICompatConfig,
    private readonly plan: () => Promise<OllamaPlan>,
  ) {}

  private warned = false;

  async *stream(req: ChatRequest): AsyncIterable<ChatChunk> {
    const plan = await this.plan();
    if (plan.warning && !this.warned) {
      this.warned = true;
      yield { type: "notice", text: plan.warning };
    }
    const body: Record<string, unknown> = {
      model: this.cfg.model,
      messages: this.cfg.systemAsUser ? foldSystemPrompt(req.messages) : req.messages,
      stream: true,
      options: { num_ctx: plan.numCtx, num_predict: req.maxTokens, temperature: req.temperature ?? 0.3, ...(req.stop?.length ? { stop: req.stop } : {}) },
    };
    // Models without a reasoning mode reject `think`.
    if (plan.thinking && req.enableThinking !== undefined) body.think = req.enableThinking;
    const ctl = new AbortController();
    const onAbort = () => ctl.abort(req.signal?.reason);
    if (req.signal?.aborted) onAbort();
    req.signal?.addEventListener("abort", onAbort, { once: true });
    const guard = stallGuard(ctl, this.cfg.stallMs, `model request to ${this.cfg.baseUrl}`);
    const stalled = (e: unknown) => (ctl.signal.aborted && !req.signal?.aborted && ctl.signal.reason instanceof Error ? ctl.signal.reason : e);
    const root = ollamaRoot(this.cfg.baseUrl);
    try {
      let res: Response;
      try {
        res = await fetch(`${root}/api/chat`, { method: "POST", headers: headers(this.cfg.apiKey), body: JSON.stringify(body), signal: ctl.signal });
      } catch (e) {
        if ((e as Error).name === "AbortError" || req.signal?.aborted) throw stalled(e);
        const cause = (e as { cause?: { code?: string; message?: string } }).cause;
        throw new Error(`cannot reach ${root}/api/chat: ${cause?.code ?? cause?.message ?? (e as Error).message}`);
      }
      if (!res.ok || !res.body) {
        const text = await res.text().catch(() => "");
        if (/context|too long|exceed/i.test(text) && res.status === 400) throw new ContextOverflowError(redactKey(text, this.cfg.apiKey).slice(0, 500));
        const err = httpError("model request", res.status, text, this.cfg, res.headers.get("retry-after"));
        if (/failed to load|resource limitations|out of memory/i.test(text)) {
          err.message += ` Ollama could not load ${this.cfg.model} (window ${plan.numCtx} tokens) into memory. Other models it keeps loaded may hold the GPU or RAM: \`ollama ps\` lists them and \`ollama stop <model>\` frees one. A smaller window (Settings → Models → Ollama context cap) or a smaller quantization also helps.`;
        }
        throw err;
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
      if (r.done) break;
      buf += decoder.decode(r.value, { stream: true });
      let nl: number;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        let j: { message?: { content?: string; thinking?: string }; done?: boolean; prompt_eval_count?: number; eval_count?: number; error?: string };
        try {
          j = JSON.parse(line);
        } catch {
          continue;
        }
        touch();
        if (j.error) throw new Error(redactKey(j.error, this.cfg.apiKey));
        if (j.message?.thinking) yield { type: "thinking", text: j.message.thinking };
        if (j.message?.content) for (const part of splitter.push(j.message.content)) yield part;
        if (j.done) yield { type: "usage", inputTokens: j.prompt_eval_count ?? 0, outputTokens: j.eval_count ?? 0 };
      }
    }
    for (const part of splitter.flush()) yield part;
  }
}
