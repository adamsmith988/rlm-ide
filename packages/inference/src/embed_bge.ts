import { env, pipeline, type FeatureExtractionPipeline, type Tensor } from "@huggingface/transformers";
import type { EmbeddingPort } from "@rlm/core";
import { cpuSessionOptions } from "./generate";

/** Texts per ONNX batch; bge-m3 attention cost grows with the longest text in a batch. */
const BATCH = 8;
/** bge-m3's own limit. Its attention memory grows with the square of the length (16 heads x n² floats per text:
 * about 4 GB at 8192 tokens), so the IDE reads far less by default. */
const MODEL_MAX_LENGTH = 8192;

/** bge-m3 (int8 ONNX) in-process on CPU via onnxruntime-node. Dense CLS embedding, L2-normalized. */
export class BgeM3Embedder implements EmbeddingPort {
  readonly modelKey: string;
  private pipe: Promise<FeatureExtractionPipeline> | null = null;

  constructor(
    private readonly modelRoot: string,
    private readonly modelDir: string,
    readonly dim: number,
    private readonly maxTokens: () => number = () => 1024,
    /** onnxruntime threads (0: its default, one per logical CPU). */
    private readonly threads: () => Promise<number> = async () => 0,
  ) {
    this.modelKey = `bge-m3__${dim}`;
  }

  private load(): Promise<FeatureExtractionPipeline> {
    if (!this.pipe) {
      env.allowRemoteModels = false;
      env.allowLocalModels = true;
      env.localModelPath = this.modelRoot;
      this.pipe = this.threads().then(
        (n) => pipeline("feature-extraction", this.modelDir, { dtype: "q8", device: "cpu", session_options: cpuSessionOptions(n) } as never) as Promise<FeatureExtractionPipeline>,
      );
      this.pipe.catch(() => (this.pipe = null)).catch(() => undefined);
    }
    return this.pipe;
  }

  async embed(texts: string[]): Promise<number[][]> {
    const p = await this.load();
    const maxLength = Math.max(16, Math.min(MODEL_MAX_LENGTH, this.maxTokens()));
    const out: number[][] = [];
    for (let i = 0; i < texts.length; i += BATCH) {
      // Text past the limit would be cut by the tokenizer anyway; cutting it first saves tokenizing megabytes.
      const batch = texts.slice(i, i + BATCH).map((t) => (t.length > maxLength * 8 ? t.slice(0, maxLength * 8) : t));
      // The pipeline call ignores max_length (it truncates at the model's 8192), so tokenizer and model are called here.
      const inputs = p.tokenizer(batch, { padding: true, truncation: true, max_length: maxLength });
      const outputs = (await p.model(inputs)) as { last_hidden_state: Tensor };
      const cls = outputs.last_hidden_state.slice(null, 0).normalize(2, -1);
      out.push(...(cls.tolist() as number[][]));
    }
    return out;
  }
}
