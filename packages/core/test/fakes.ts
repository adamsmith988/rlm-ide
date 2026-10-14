import path from "node:path";
import { readFileSync } from "node:fs";
import type { ChatChunk, ChatModelPort, ChatRequest, ClockPort, EmbeddingPort, LoggerPort, PromptsPort, StatusPort } from "../src";

export class ScriptedModel implements ChatModelPort {
  readonly calls: ChatRequest[] = [];
  constructor(
    readonly modelId: string,
    private readonly respond: (req: ChatRequest, n: number) => string,
  ) {}
  async *stream(req: ChatRequest): AsyncIterable<ChatChunk> {
    this.calls.push(req);
    const text = this.respond(req, this.calls.length - 1);
    for (let i = 0; i < text.length; i += 7) {
      if (req.signal?.aborted) throw new DOMException("stopped", "AbortError");
      yield { type: "text", text: text.slice(i, i + 7) };
    }
  }
}

/** Deterministic bag-of-words embedder: good enough for recall/isolation tests. */
export class HashEmbedder implements EmbeddingPort {
  readonly modelKey = "hash__256";
  readonly dim = 256;
  failing = false;
  async embed(texts: string[]): Promise<number[][]> {
    if (this.failing) throw new Error("embedder down");
    return texts.map((t) => {
      const v = new Array(this.dim).fill(0);
      for (const w of new Set(t.toLowerCase().match(/[a-z0-9_-]+/g) ?? [])) {
        let h = 0;
        for (const c of w) h = (h * 31 + c.charCodeAt(0)) >>> 0;
        v[h % this.dim] = 1;
      }
      return v;
    });
  }
}

export const prompts: PromptsPort = {
  get: (name) => readFileSync(path.join(__dirname, "../../../prompts", name), "utf8"),
};

export const clock: ClockPort = { now: () => Date.now() };
export const log: LoggerPort = { info: () => {}, warn: () => {}, error: () => {} };
export const status: StatusPort & { memoryState: string } = {
  memoryState: "",
  memory(s) {
    this.memoryState = s;
  },
  kernel() {},
};
