import { env, pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";
import type { EmbeddingPort } from "@rlm/core";

/** Texts per ONNX batch; bge-m3 attention cost grows with the longest text in a batch. */
const BATCH = 8;
/** bge-m3 accepts up to 8192 tokens; long chunks are truncated by the tokenizer. */
const MAX_LENGTH = 8192;

/** bge-m3 (int8 ONNX) in-process on CPU via onnxruntime-node. Dense CLS embedding, L2-normalized. */
export class BgeM3Embedder implements EmbeddingPort {
  readonly modelKey: string;
  private pipe: Promise<FeatureExtractionPipeline> | null = null;

  constructor(
    private readonly modelRoot: string,
    private readonly modelDir: string,
    readonly dim: number,
  ) {
    this.modelKey = `bge-m3__${dim}`;
  }

  private load(): Promise<FeatureExtractionPipeline> {
    if (!this.pipe) {
      env.allowRemoteModels = false;
      env.allowLocalModels = true;
      env.localModelPath = this.modelRoot;
      this.pipe = pipeline("feature-extraction", this.modelDir, { dtype: "q8", device: "cpu" } as never) as Promise<FeatureExtractionPipeline>;
      this.pipe.catch(() => (this.pipe = null)).catch(() => undefined);
    }
    return this.pipe;
  }

  async embed(texts: string[]): Promise<number[][]> {
    const p = await this.load();
    const out: number[][] = [];
    for (let i = 0; i < texts.length; i += BATCH) {
      const batch = texts.slice(i, i + BATCH);
      const t = await p(batch, { pooling: "cls", normalize: true, truncation: true, max_length: MAX_LENGTH } as never);
      out.push(...(t.tolist() as number[][]));
    }
    return out;
  }
}
