import { ContextOverflowError, type ChatChunk, type ChatModelPort, type ChatRequest } from "@rlm/core";

/** Messages exchanged with the local inference host (hidden WebGPU window or CPU process). */
export type InferenceRequest =
  | { type: "generate"; id: string; messages: ChatRequest["messages"]; maxTokens: number; temperature: number; enableThinking: boolean; contextLimit: number }
  | { type: "abort"; id: string };

export type InferenceResponse =
  | { type: "chunk"; id: string; kind: "thinking" | "text"; text: string }
  | { type: "done"; id: string; inputTokens: number; outputTokens: number }
  | { type: "error"; id: string; message: string; overflow?: boolean };

export interface InferenceTransport {
  send(msg: InferenceRequest): void;
  onMessage(cb: (msg: InferenceResponse) => void): () => void;
  ready(): Promise<void>;
}

let seq = 0;

/** ChatModelPort that streams from the in-app local model through a message transport. */
export class LocalBridgeChat implements ChatModelPort {
  constructor(
    readonly modelId: string,
    private readonly transport: InferenceTransport,
    private readonly opts: { contextLimit: () => number; enableThinking: () => boolean },
  ) {}

  async *stream(req: ChatRequest): AsyncIterable<ChatChunk> {
    await this.transport.ready();
    const id = `gen_${++seq}_${Date.now().toString(36)}`;
    const queue: InferenceResponse[] = [];
    let wake: (() => void) | null = null;
    const off = this.transport.onMessage((m) => {
      if (m.id !== id) return;
      queue.push(m);
      wake?.();
    });
    const onAbort = () => this.transport.send({ type: "abort", id });
    req.signal?.addEventListener("abort", onAbort, { once: true });
    this.transport.send({
      type: "generate",
      id,
      messages: req.messages,
      maxTokens: req.maxTokens,
      temperature: req.temperature ?? 0.3,
      enableThinking: req.enableThinking ?? this.opts.enableThinking(),
      contextLimit: this.opts.contextLimit(),
    });
    try {
      for (;;) {
        if (!queue.length) await new Promise<void>((r) => (wake = r));
        wake = null;
        while (queue.length) {
          const m = queue.shift()!;
          if (m.type === "chunk") yield { type: m.kind, text: m.text };
          else if (m.type === "done") {
            yield { type: "usage", inputTokens: m.inputTokens, outputTokens: m.outputTokens };
            return;
          } else if (m.type === "error") {
            if (req.signal?.aborted) throw new DOMException("stopped", "AbortError");
            if (m.overflow) throw new ContextOverflowError(m.message);
            throw new Error(m.message);
          }
        }
        if (req.signal?.aborted) throw new DOMException("stopped", "AbortError");
      }
    } finally {
      off();
      req.signal?.removeEventListener("abort", onAbort);
    }
  }
}
