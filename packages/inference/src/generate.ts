/** Shared local generation for FrogNano (Transformers.js). Runs in the hidden WebGPU window and
 * in the CPU fallback process; only `device` differs. */
import {
  AutoModelForCausalLM,
  AutoTokenizer,
  InterruptableStoppingCriteria,
  TextStreamer,
  env,
  type PreTrainedModel,
  type PreTrainedTokenizer,
} from "@huggingface/transformers";

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
        this.opts.onStatus?.(`loading model on ${this.opts.device}`);
        this.model = await AutoModelForCausalLM.from_pretrained(this.opts.modelDir, {
          dtype: "q4f16",
          device: this.opts.device,
          use_external_data_format: 2,
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
    const streamer = new TextStreamer(tok, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text: string) => {
        for (const p of splitter.push(text)) send({ type: "chunk", id: req.id, kind: p.kind, text: p.text });
      },
      token_callback_function: () => {
        outputTokens++;
      },
    } as never);
    await model.generate({
      ...(inputs as object),
      max_new_tokens: maxNew,
      do_sample: req.temperature > 0,
      temperature: Math.max(0.01, req.temperature),
      top_p: 0.95,
      streamer,
      stopping_criteria: stop,
    } as never);
    for (const p of splitter.flush()) send({ type: "chunk", id: req.id, kind: p.kind, text: p.text });
    if (stop.interrupted) {
      send({ type: "error", id: req.id, message: "generation stopped" });
      return;
    }
    send({ type: "done", id: req.id, inputTokens, outputTokens });
  }
}
