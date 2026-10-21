/** Shared local generation for FrogNano (Transformers.js). Runs in the hidden WebGPU window and
 * in the CPU fallback process; only `device` differs. */
import {
  AutoModelForCausalLM,
  AutoTokenizer,
  DynamicCache,
  InterruptableStoppingCriteria,
  TextStreamer,
  Tensor,
  env,
  type PreTrainedModel,
  type PreTrainedTokenizer,
} from "@huggingface/transformers";

/** Inside Electron the allocator ends the process on any single allocation of 1 GiB or more (plain Node does not).
 * onnxruntime's CPU arena grows in doubling regions and soon asks for one that large (see cpuSessionOptions), and the
 * attention scores of a pass are one float per head and (new token, total token) pair: 16 heads crashed reading a
 * 4144-token prompt at once and not 4040. Long prompts are read in chunks that keep that tensor at 256 MB. */
const BYTES_PER_PAIR_AND_HEAD = 4;
const PREFILL_BUFFER_BYTES = 2 ** 28;

/** Prompt tokens per forward pass when `totalTokens` will be in context. */
export function prefillChunk(totalTokens: number, heads: number): number {
  return Math.max(64, Math.floor(PREFILL_BUFFER_BYTES / (BYTES_PER_PAIR_AND_HEAD * Math.max(1, heads) * Math.max(1, totalTokens))));
}

/** onnxruntime session options for the CPU inside Electron: no memory arena, so every tensor is its own allocation of
 * its own size (the arena's doubling regions crash Electron's allocator well before memory runs out). */
export function cpuSessionOptions(threads: number): Record<string, unknown> {
  return { enableCpuMemArena: false, ...(threads > 0 ? { intraOpNumThreads: threads } : {}) };
}

/** present.* outputs of a forward pass under the names the next pass reads them from (as Transformers.js does). */
function cacheFrom(outputs: Record<string, Tensor>, into: DynamicCache | null): DynamicCache {
  const next: Record<string, Tensor> = {};
  for (const [name, t] of Object.entries(outputs)) {
    if (!name.startsWith("present")) continue;
    next[name.replace("present_ssm", "past_ssm").replace("present_conv", "past_conv").replace("present_recurrent", "past_recurrent").replace("present", "past_key_values")] = t;
  }
  if (!into) return new DynamicCache(next);
  into.update(next);
  return into;
}

export type InferenceRequest =
  | { type: "generate"; id: string; messages: Array<{ role: string; content: string }>; maxTokens: number; temperature: number; enableThinking: boolean; contextLimit: number }
  | { type: "abort"; id: string };

export type InferenceResponse =
  | { type: "chunk"; id: string; kind: "thinking" | "text"; text: string }
  | { type: "done"; id: string; inputTokens: number; outputTokens: number }
  | { type: "error"; id: string; message: string; overflow?: boolean };

export interface LoadOptions {
  /** Base URL or folder that contains the model folder. */
  modelRoot: string;
  modelDir: string;
  device: "webgpu" | "cpu";
  wasmPaths?: string;
  /** WebGPU adapter choice; must match the adapter that passed the shader-f16 probe. */
  powerPreference?: "high-performance" | "low-power";
  /** onnxruntime threads on the CPU (unset: its default, one per logical CPU). */
  cpuThreads?: number;
  onStatus?: (s: string) => void;
}

const GPU_FATAL = /OUT_OF_DEVICE_MEMORY|out of memory|Invalid Buffer|device (was )?lost|DeviceLost/i;
const GPU_OOM_MESSAGE =
  "The GPU ran out of memory while running the built-in model (another app may be using most of the GPU memory). " +
  "Free GPU memory, set Settings → Local inference → Device to CPU, or switch the agent to an API model.";

/** Splits streamed text into thinking/answer; the chat template opens `<think>` in the prompt. */
class Splitter {
  private buf = "";
  constructor(private inThink: boolean) {}
  push(s: string): Array<{ kind: "thinking" | "text"; text: string }> {
    this.buf += s;
    const out: Array<{ kind: "thinking" | "text"; text: string }> = [];
    for (;;) {
      const tag = this.inThink ? "</think>" : "<think>";
      const i = this.buf.indexOf(tag);
      if (i === -1) {
        let hold = 0;
        for (let n = Math.min(tag.length - 1, this.buf.length); n > 0; n--) if (tag.startsWith(this.buf.slice(-n))) { hold = n; break; }
        const emit = this.buf.slice(0, this.buf.length - hold);
        if (emit) out.push({ kind: this.inThink ? "thinking" : "text", text: emit });
        this.buf = this.buf.slice(this.buf.length - hold);
        return out;
      }
      const before = this.buf.slice(0, i);
      if (before) out.push({ kind: this.inThink ? "thinking" : "text", text: before });
      this.buf = this.buf.slice(i + tag.length);
      if (this.inThink) this.buf = this.buf.replace(/^\s+/, "");
      this.inThink = !this.inThink;
    }
  }
  flush(): Array<{ kind: "thinking" | "text"; text: string }> {
    const r = this.buf ? [{ kind: (this.inThink ? "thinking" : "text") as "thinking" | "text", text: this.buf }] : [];
    this.buf = "";
    return r;
  }
}

export class LocalGenerator {
  private tokenizer: PreTrainedTokenizer | null = null;
  private model: PreTrainedModel | null = null;
  private loading: Promise<void> | null = null;
  private readonly stops = new Map<string, InterruptableStoppingCriteria>();
  /** Generation is serialized: one KV cache, one request at a time. */
  private chain: Promise<void> = Promise.resolve();

  constructor(private readonly opts: LoadOptions) {}

  load(): Promise<void> {
    if (!this.loading) {
      this.loading = (async () => {
        env.allowRemoteModels = false;
        env.allowLocalModels = true;
        env.localModelPath = this.opts.modelRoot;
        if (this.opts.wasmPaths && env.backends.onnx.wasm) env.backends.onnx.wasm.wasmPaths = this.opts.wasmPaths;
        const webgpu = (env.backends.onnx as { webgpu?: { powerPreference?: string } }).webgpu;
        if (this.opts.powerPreference && webgpu) webgpu.powerPreference = this.opts.powerPreference;
        this.opts.onStatus?.("loading tokenizer");
        this.tokenizer = await AutoTokenizer.from_pretrained(this.opts.modelDir);
        const threads = this.opts.device === "cpu" && this.opts.cpuThreads ? this.opts.cpuThreads : 0;
        this.opts.onStatus?.(`loading model on ${this.opts.device}${threads ? ` (${threads} threads)` : ""}`);
        this.model = await AutoModelForCausalLM.from_pretrained(this.opts.modelDir, {
          dtype: "q4f16",
          device: this.opts.device,
          use_external_data_format: 2,
          ...(this.opts.device === "cpu" ? { session_options: cpuSessionOptions(threads) } : {}),
        } as Parameters<typeof AutoModelForCausalLM.from_pretrained>[1]);
        this.opts.onStatus?.("ready");
      })();
      this.loading.catch(() => (this.loading = null));
    }
    return this.loading;
  }

  handle(req: InferenceRequest, send: (r: InferenceResponse) => void): void {
    if (req.type === "abort") {
      this.stops.get(req.id)?.interrupt();
      return;
    }
    const stop = new InterruptableStoppingCriteria();
    this.stops.set(req.id, stop);
    this.chain = this.chain
      .then(() => this.generate(req, stop, send))
      .catch((e: unknown) => {
        const raw = e instanceof Error ? e.message : String(e);
        if (!GPU_FATAL.test(raw)) return send({ type: "error", id: req.id, message: raw });
        send({ type: "error", id: req.id, message: GPU_OOM_MESSAGE });
        // The WebGPU device is unusable after this; ask the host for a fresh runtime.
        send({ type: "error", id: "*", message: GPU_OOM_MESSAGE });
      })
      .finally(() => this.stops.delete(req.id));
  }

  private async generate(req: Extract<InferenceRequest, { type: "generate" }>, stop: InterruptableStoppingCriteria, send: (r: InferenceResponse) => void): Promise<void> {
    await this.load();
    const tok = this.tokenizer!;
    const model = this.model!;
    const inputs = tok.apply_chat_template(req.messages as never, {
      add_generation_prompt: true,
      return_dict: true,
      enable_thinking: req.enableThinking,
    } as never) as unknown as { input_ids: { dims: number[] }; attention_mask: unknown };
    const inputTokens = inputs.input_ids.dims[1] ?? 0;
    if (inputTokens + 16 >= req.contextLimit) {
      send({ type: "error", id: req.id, message: `prompt has ${inputTokens} tokens; local context limit is ${req.contextLimit}`, overflow: true });
      return;
    }
    const maxNew = Math.max(16, Math.min(req.maxTokens, req.contextLimit - inputTokens - 8));
    const splitter = new Splitter(req.enableThinking);
    let outputTokens = 0;
    let firstAt = 0;
    const streamer = new TextStreamer(tok, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text: string) => {
        for (const p of splitter.push(text)) send({ type: "chunk", id: req.id, kind: p.kind, text: p.text });
      },
      token_callback_function: () => {
        if (!outputTokens++) firstAt = performance.now();
      },
    } as never);
    const heads = Number((model.config as unknown as { num_attention_heads?: number }).num_attention_heads ?? 32);
    const chunk = prefillChunk(inputTokens, heads);
    let cache: DynamicCache | null = null;
    const ids = inputs.input_ids as unknown as Tensor;
    if (this.opts.device === "cpu" && inputTokens > chunk) {
      // The last token is left for generate(), which is given only that token: for this model it does not skip the
      // tokens a cache already holds and would read the whole prompt again in one pass.
      for (let start = 0; start < inputTokens - 1 && !stop.interrupted; start += chunk) {
        const end = Math.min(start + chunk, inputTokens - 1);
        const out = (await model.forward({
          input_ids: ids.slice(null, [start, end]),
          attention_mask: new Tensor("int64", new BigInt64Array(end).fill(1n), [1, end]),
          past_key_values: cache,
        })) as Record<string, Tensor>;
        cache = cacheFrom(out, cache);
        this.opts.onStatus?.(`reading the prompt: ${end} of ${inputTokens} tokens`);
      }
    }
    try {
      await model.generate({
        ...(inputs as object),
        ...(cache ? { past_key_values: cache, input_ids: ids.slice(null, [inputTokens - 1, inputTokens]) } : {}),
        max_new_tokens: maxNew,
        do_sample: req.temperature > 0,
        temperature: Math.max(0.01, req.temperature),
        top_p: 0.95,
        streamer,
        stopping_criteria: stop,
      } as never);
    } finally {
      await cache?.dispose();
    }
    for (const p of splitter.flush()) send({ type: "chunk", id: req.id, kind: p.kind, text: p.text });
    const secs = (performance.now() - firstAt) / 1000;
    if (outputTokens > 8 && secs > 0) this.opts.onStatus?.(`ready · ${((outputTokens - 1) / secs).toFixed(1)} tok/s on ${this.opts.device}`);
    if (stop.interrupted) {
      send({ type: "error", id: req.id, message: "generation stopped" });
      return;
    }
    send({ type: "done", id: req.id, inputTokens, outputTokens });
  }
}
