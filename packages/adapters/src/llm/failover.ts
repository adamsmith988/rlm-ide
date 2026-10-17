import { ContextOverflowError, ModelWindowError, estimateMessagesTokens, estimateTokens, type ChatChunk, type ChatMessage, type ChatModelPort, type ChatRequest, type ModelLimitsInfo } from "@rlm/core";
import type { FailoverSettings } from "@rlm/contracts";
import { ModelHttpError } from "./openai_compat";

export interface FailoverCandidate {
  id: string;
  label: string;
  /** Null when unknown; a candidate whose window clearly cannot hold the prompt is skipped while others remain. */
  contextLength: number | null;
  maxOutputTokens: number | null;
  port: () => Promise<ChatModelPort>;
}

export type FailureSeverity = "transient" | "hard";

export interface FailureInfo {
  severity: FailureSeverity;
  reason: string;
  retryAfterMs: number | null;
}

export class EmptyReplyError extends Error {
  constructor(
    model: string,
    readonly onlyThinking = false,
  ) {
    super(onlyThinking ? `${model} stopped after its reasoning without giving an answer` : `${model} returned an empty reply`);
    this.name = "EmptyReplyError";
  }
}

/** Some providers render a conversation that ends with an assistant message badly and the model answers with raw
 * chat-template markup instead of text. */
export class GarbledReplyError extends Error {
  constructor(model: string) {
    super(`${model} answered with chat-template markup instead of text`);
    this.name = "GarbledReplyError";
  }
}

const TEMPLATE_TOKEN = /<\|(?:start|end|im_start|im_end|im_sep|channel|message|return|call|constrain|eot_id|start_header_id|end_header_id|begin_of_text|endoftext)\|>/;

function short(text: string, max = 160): string {
  const one = text.replace(/\s+/g, " ").trim();
  return one.length > max ? `${one.slice(0, max - 1)}…` : one;
}

/** Hard failures will not clear up on a quick retry (bad key, unknown model, rejected request, out of credit). */
export function classifyFailure(e: unknown): FailureInfo {
  if (e instanceof ModelHttpError) {
    const s = e.status;
    const retryAfterMs = e.retryAfterMs;
    if (s === 401 || s === 403) return { severity: "hard", reason: `access denied (${s})`, retryAfterMs };
    if (s === 402) return { severity: "hard", reason: "out of credit or free quota (402)", retryAfterMs };
    if (s === 404) return { severity: "hard", reason: "model not found (404)", retryAfterMs };
    if (s === 400 || s === 422) return { severity: "hard", reason: `request rejected (${s}): ${short(e.message, 120)}`, retryAfterMs };
    if (s === 429) return { severity: "transient", reason: "rate limited (429)", retryAfterMs };
    if (s >= 500) return { severity: "transient", reason: `server error (${s})`, retryAfterMs };
    return { severity: "transient", reason: `HTTP ${s}`, retryAfterMs };
  }
  if (e instanceof EmptyReplyError) return { severity: "transient", reason: e.onlyThinking ? "no answer after reasoning" : "empty reply", retryAfterMs: null };
  if (e instanceof GarbledReplyError) return { severity: "transient", reason: "garbled reply", retryAfterMs: null };
  const msg = e instanceof Error ? e.message : String(e);
  if (/looks stuck/.test(msg)) return { severity: "transient", reason: "stalled", retryAfterMs: null };
  if (/cannot reach/.test(msg)) return { severity: "transient", reason: "unreachable", retryAfterMs: null };
  return { severity: "transient", reason: short(msg), retryAfterMs: null };
}

interface HealthEntry {
  until: number;
  fails: number;
  severity: FailureSeverity;
  reason: string;
}

/** Per-model cooldowns shared by every chain, so a model that just failed is skipped everywhere until it recovers. */
export class ModelHealth {
  private readonly entries = new Map<string, HealthEntry>();
  private readonly windows = new Map<string, number>();
  constructor(private readonly now: () => number = Date.now) {}

  /** The model refused a prompt of about `tokens` as too long, so its real window is smaller than configured. */
  learnWindow(id: string, tokens: number): void {
    this.windows.set(id, Math.min(this.windows.get(id) ?? Infinity, Math.floor(tokens * 0.9)));
  }

  learnedWindow(id: string): number | undefined {
    return this.windows.get(id);
  }

  available(id: string): boolean {
    const e = this.entries.get(id);
    return !e || e.until <= this.now();
  }

  get(id: string): HealthEntry | undefined {
    return this.entries.get(id);
  }

  fail(id: string, f: FailureInfo, cfg: FailoverSettings): number {
    const fails = (this.entries.get(id)?.fails ?? 0) + 1;
    const backoff = f.severity === "hard" ? cfg.hardFailureCooldownSec * 1000 : Math.min(cfg.maxCooldownSec * 1000, cfg.cooldownSec * 1000 * 2 ** (fails - 1));
    const wait = f.retryAfterMs !== null ? Math.max(1000, Math.min(f.retryAfterMs, cfg.hardFailureCooldownSec * 1000)) : backoff;
    this.entries.set(id, { until: this.now() + wait, fails, severity: f.severity, reason: f.reason });
    return wait;
  }

  ok(id: string): void {
    this.entries.delete(id);
  }

  clear(): void {
    this.entries.clear();
    this.windows.clear();
  }

  snapshot(): Array<{ id: string; untilMs: number; fails: number; severity: FailureSeverity; reason: string }> {
    const now = this.now();
    return [...this.entries].filter(([, e]) => e.until > now).map(([id, e]) => ({ id, untilMs: e.until - now, fails: e.fails, severity: e.severity, reason: e.reason }));
  }
}

/** A continuation is a new message, which never starts with whitespace, so a cut right after "31," comes back
 * as "37". Restores the space only where prose clearly needs one; mid-word, mid-number and code are left alone. */
export function seamSpace(prior: string, next: string): string {
  if (!next || /^\s/.test(next) || !/^[\p{L}\p{N}(“"'*_]/u.test(next)) return "";
  const last = prior.slice(-1);
  if (!/[,;:!?.]/.test(last)) return "";
  if ((prior.match(/```/g) ?? []).length % 2 === 1) return "";
  const line = prior.slice(prior.lastIndexOf("\n") + 1);
  if ((line.match(/`/g) ?? []).length % 2 === 1) return "";
  const before = prior.slice(-2, -1);
  if (/\d/.test(before) && (last === "." || last === ":") && /^\d/.test(next)) return "";
  if (/\d/.test(before) && last === "," && /^\d{3}(?!\d)/.test(next)) return "";
  if (last === "." && /^\p{Ll}/u.test(next)) return "";
  return " ";
}

/** The partial reply a backup continues from, cut back to before its last word. A continuation is a new message and
 * never starts with whitespace, so from a clean word boundary the backup writes that word whole (which the deduper
 * drops) instead of gluing the next word on, or guessing whether the cut split a word. */
export function resumePoint(partial: string): string {
  const m = /^([\s\S]*\s)(\S{1,40})$/.exec(partial);
  return m?.[1]?.trim() ? m[1] : partial;
}

const WORD_CHAR = /[\p{L}\p{N}_]/u;

/** Whether `next` opens with the last `k` characters of `word` as a whole token, as when a model takes the last word
 * of "sum(i*i" to be "i". A tail that starts or ends mid-identifier is a coincidence, not a repeat. */
function repeatsTail(word: string, next: string, k: number): boolean {
  if (!word.endsWith(next.slice(0, k))) return false;
  const before = word[word.length - k - 1] ?? "";
  const first = next[0] ?? "";
  const last = next[k - 1] ?? "";
  const after = next[k] ?? "";
  return !(WORD_CHAR.test(before) && WORD_CHAR.test(first)) && !(WORD_CHAR.test(last) && WORD_CHAR.test(after));
}

/** Drops what a continuing model repeats: the tail of the prior text (it resumes before the last word, see
 * `resumePoint`, so that word comes back with the seam's spacing), or the whole reply when it starts over. */
export class ContinuationDeduper {
  private buf = "";
  private mode: "probe" | "skip" | "pass" = "probe";
  private idx = 0;

  constructor(
    private readonly prior: string,
    private readonly probeChars = 160,
  ) {}

  push(text: string): string {
    if (this.mode === "pass") return text;
    if (this.mode === "skip") return this.skip(text);
    this.buf += text;
    return this.buf.length < this.probeChars ? "" : this.resolve();
  }

  flush(): string {
    return this.mode === "probe" ? this.resolve() : "";
  }

  private resolve(): string {
    const b = this.buf;
    this.buf = "";
    const head = this.prior.slice(0, 48);
    if (head.length >= 16 && b.length >= 16 && (b.startsWith(head) || head.startsWith(b))) {
      this.mode = "skip";
      this.idx = 0;
      return this.skip(b);
    }
    this.mode = "pass";
    for (let k = Math.min(b.length, this.prior.length); k >= 12; k--) if (this.prior.endsWith(b.slice(0, k))) return b.slice(k);
    const [, word = "", gap = ""] = /(\S+)(\s*)$/.exec(this.prior) ?? [];
    if (word && b.startsWith(word)) {
      const rest = b.slice(word.length);
      return gap ? rest.replace(/^[ \t]+/, "") : rest;
    }
    if (!gap) for (let k = Math.min(b.length, word.length - 1); k >= 1; k--) if (repeatsTail(word, b, k)) return b.slice(k);
    return seamSpace(this.prior, b) + b;
  }

  private skip(text: string): string {
    let i = 0;
    while (i < text.length && this.idx < this.prior.length && text[i] === this.prior[this.idx]) {
      i++;
      this.idx++;
    }
    if (i < text.length) {
      this.mode = "pass";
      return text.slice(i);
    }
    return "";
  }
}

export interface FailoverOptions {
  settings: () => FailoverSettings;
  health: ModelHealth;
  /** User message that asks the next model to continue a cut-off reply. */
  continuePrompt: () => string;
  now?: () => number;
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
  log?: (msg: string, meta: Record<string, unknown>) => void;
}

function abortableSleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason ?? new DOMException("stopped", "AbortError"));
    const t = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(t);
      reject(signal?.reason ?? new DOMException("stopped", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/** Room a model must have left after the prompt to give a useful answer. */
const MIN_ANSWER_TOKENS = 256;

/** Streams from the first healthy model of an ordered chain and moves down the chain when one fails, so a turn
 * keeps going through rate limits, outages, stalls and dropped streams. A model that fails mid-reply hands its
 * partial text to the next one, which continues it; consumers see one uninterrupted reply. */
export class FailoverChat implements ChatModelPort {
  readonly modelId: string;
  private announced = "";

  constructor(
    private readonly chain: FailoverCandidate[],
    private readonly o: FailoverOptions,
  ) {
    if (!chain.length) throw new Error("failover chain is empty");
    this.modelId = chain[0]!.id;
  }

  private now(): number {
    return (this.o.now ?? Date.now)();
  }

  /** Configured window, or the smaller one learned from a refused prompt; null = unknown. */
  private window(c: FailoverCandidate): number | null {
    const w = Math.min(c.contextLength ?? Infinity, this.o.health.learnedWindow(c.id) ?? Infinity);
    return Number.isFinite(w) ? w : null;
  }

  private fits(c: FailoverCandidate, tokens: number): boolean {
    const w = this.window(c);
    return !w || tokens + MIN_ANSWER_TOKENS < w;
  }

  /** First healthy model that can hold the prompt. Null while every model cools down. A prompt no healthy model can
   * hold is never sent (it would fail and wrongly mark a working model as broken): the caller compacts and retries. */
  private pick(tokens: number, tooSmall: Set<string>): FailoverCandidate | null {
    const healthy = this.chain.filter((c) => this.o.health.available(c.id));
    if (!healthy.length) return null;
    const fit = healthy.find((c) => !tooSmall.has(c.id) && this.fits(c, tokens));
    if (fit) return fit;
    const rest = healthy.filter((c) => !tooSmall.has(c.id));
    if (!rest.length) throw new ContextOverflowError("the prompt does not fit any available model");
    const widest = rest.reduce((a, c) => (this.window(c)! > this.window(a)! ? c : a));
    throw new ModelWindowError(`The conversation needs about ${tokens} tokens but the widest model available now, ${widest.label}, holds ${this.window(widest)}; compacting to fit`, this.window(widest)!);
  }

  /** The widest window among the models that can serve right now: the caller sizes its prompt to it. The chain then
   * picks, in order, the first model that holds the prompt. */
  activeLimits(): ModelLimitsInfo | null {
    const healthy = this.chain.filter((c) => this.o.health.available(c.id));
    const pool = healthy.length ? healthy : this.chain;
    if (pool.some((c) => !this.window(c))) return null;
    const widest = pool.reduce((a, c) => (this.window(c)! > this.window(a)! ? c : a));
    return { contextLength: this.window(widest), maxOutputTokens: widest.maxOutputTokens };
  }

  async *stream(req: ChatRequest): AsyncIterable<ChatChunk> {
    const cfg = this.o.settings();
    const sleep = this.o.sleep ?? abortableSleep;
    const primary = this.chain[0]!;
    let partial = "";
    let lastError: unknown = null;
    let lastReason = "";
    let current: FailoverCandidate | null = null;
    let waitedMs = 0;
    const tooSmall = new Set<string>();

    for (;;) {
      if (req.signal?.aborted) throw req.signal.reason ?? new DOMException("stopped", "AbortError");
      const tokens = estimateMessagesTokens(req.messages) + (partial ? estimateTokens(partial) + estimateMessagesTokens([{ content: this.o.continuePrompt() }]) : 0);
      const pick = this.pick(tokens, tooSmall);
      if (!pick) {
        const states = this.chain.map((c) => this.o.health.get(c.id));
        const allHard = states.every((s) => s?.severity === "hard");
        const wakeAt = Math.min(...states.map((s) => s?.until ?? 0));
        const wait = Math.max(250, wakeAt - this.now());
        if (allHard || waitedMs + wait > cfg.waitForRecoverySec * 1000) {
          if (lastError) throw lastError;
          const why = this.chain.map((c) => `${c.label}: ${this.o.health.get(c.id)?.reason ?? "unavailable"}`).join("; ");
          throw new Error(`every model in the failover chain is unavailable (${why}). Fix the cause in Settings → Models, which also clears the cooldowns.`);
        }
        const next = this.chain.find((c) => this.o.health.get(c.id)?.until === wakeAt) ?? primary;
        yield { type: "notice", text: `Every model in the chain is cooling down; retrying ${next.label} in ${Math.ceil(wait / 1000)} s` };
        await sleep(wait, req.signal);
        waitedMs += wait;
        continue;
      }

      if (current && current.id !== pick.id) {
        yield { type: "switch", from: current.label, to: pick.label, reason: lastReason, resumed: partial.length > 0 };
      } else if (!current && pick.id !== primary.id) {
        const key = `${primary.id}>${pick.id}`;
        if (this.announced !== key) {
          this.announced = key;
          const reason = this.o.health.available(primary.id) ? `the conversation is larger than ${primary.label}'s window` : `${primary.label} is cooling down: ${this.o.health.get(primary.id)?.reason ?? "unavailable"}`;
          yield { type: "switch", from: primary.label, to: pick.label, reason, resumed: false };
        }
      }
      if (!current && pick.id === primary.id) this.announced = "";
      current = pick;

      if (partial && !cfg.continueMidReply) throw lastError;
      const messages: ChatMessage[] = partial ? [...req.messages, { role: "assistant", content: resumePoint(partial) }, { role: "user", content: this.o.continuePrompt() }] : req.messages;
      const dedupe = partial ? new ContinuationDeduper(partial) : null;
      let gotText = false;
      let gotThinking = false;
      let raw = "";
      const win = this.window(pick);
      const maxTokens = Math.max(64, Math.min(req.maxTokens, pick.maxOutputTokens ?? Infinity, win ? win - tokens - 128 : Infinity));
      try {
        const port = await pick.port();
        for await (const c of port.stream({ ...req, messages, maxTokens })) {
          if (c.type === "text") {
            raw += c.text;
            // The deduper holds back the start of a continuation, so markup there never reaches the reply.
            if (dedupe && !gotText && TEMPLATE_TOKEN.test(raw)) throw new GarbledReplyError(pick.label);
            const out = dedupe ? dedupe.push(c.text) : c.text;
            if (!out) continue;
            partial += out;
            gotText = true;
            yield { type: "text", text: out };
          } else if (c.type === "usage" && dedupe) {
            // A continuation's prompt also carries the partial reply; counting it would skew the caller's token calibration.
            yield { ...c, inputTokens: 0 };
          } else {
            if (c.type === "thinking") gotThinking = true;
            yield c;
          }
        }
        if (dedupe && !raw.trim() && cfg.emptyReplyIsFailure) throw new EmptyReplyError(pick.label);
        const rest = dedupe?.flush();
        if (rest) {
          partial += rest;
          gotText = true;
          yield { type: "text", text: rest };
        }
        if (!partial && !gotText && cfg.emptyReplyIsFailure) throw new EmptyReplyError(pick.label, gotThinking);
        this.o.health.ok(pick.id);
        return;
      } catch (e) {
        if (req.signal?.aborted) throw e;
        if (e instanceof ContextOverflowError) {
          // Too big for this model, which is not broken: no cooldown. Its real window is now known to be smaller, so
          // the request moves to a model that holds it, or the caller compacts to fit (ModelWindowError from pick).
          this.o.health.learnWindow(pick.id, tokens);
          if (partial) tooSmall.add(pick.id);
          lastError = e;
          lastReason = partial ? "the continued reply does not fit its window" : "the prompt is larger than its real window";
          continue;
        }
        const f = classifyFailure(e);
        const wait = this.o.health.fail(pick.id, f, cfg);
        lastError = e;
        lastReason = f.reason;
        this.o.log?.("model failed; moving down the failover chain", { model: pick.id, reason: f.reason, severity: f.severity, cooldownMs: wait, partialChars: partial.length });
      }
    }
  }
}
