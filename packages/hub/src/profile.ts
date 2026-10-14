import type { ModelProfileView } from "@rlm/contracts";
import type { RawModelInfo, RepoFile } from "./hf";

/** Everything here is read from the repo's own metadata; no model is known by name. */

const CHAT_PIPELINES = new Set(["text-generation", "text2text-generation", "image-text-to-text", "conversational", "any-to-any", "visual-question-answering", "image-to-text"]);
const EMBED_PIPELINES = new Set(["feature-extraction", "sentence-similarity"]);
const CTX_KEYS = ["max_position_embeddings", "n_positions", "max_sequence_length", "max_seq_len", "seq_length", "n_ctx", "model_max_length"];

const HF_WEIGHT = /(^|\/)[^/]*\.(safetensors|bin|pt|pth)$/i;
const OTHER_WEIGHT = /\.(h5|msgpack|ot|ckpt|npz|mlmodel|tflite|pte)$/i;

export const isGguf = (p: string) => /\.gguf$/i.test(p);
export const isOnnx = (p: string) => /\.onnx$/i.test(p);
const isOnnxData = (p: string) => /\.onnx_data$|\.onnx\.data$/i.test(p) || /(^|\/)onnx\/.*\.data$/i.test(p);
/** Weights transformers loads: *.safetensors, or PyTorch pickles when no safetensors exist. */
const isHfWeight = (p: string) => HF_WEIGHT.test(p) && !/(^|\/)(training_args|optimizer|scheduler|rng_state|scaler)[^/]*$/i.test(p) && !/(^|\/)original\//.test(p);

export function formatsOf(files: string[]): string[] {
  const f = new Set<string>();
  for (const p of files) {
    if (/\.safetensors$/i.test(p)) f.add("safetensors");
    else if (/\.(bin|pt|pth)$/i.test(p) && isHfWeight(p)) f.add("pytorch");
    else if (isGguf(p)) f.add("gguf");
    else if (isOnnx(p)) f.add("onnx");
    else if (/\.h5$/i.test(p)) f.add("tensorflow");
    else if (/\.msgpack$/i.test(p)) f.add("flax");
    else if (/openvino.*\.xml$/i.test(p)) f.add("openvino");
    else if (/\.(npz|mlx)$/i.test(p) || /(^|\/)mlx/i.test(p)) f.add("mlx");
  }
  return [...f];
}

export function contextFromConfig(cfg: Record<string, unknown> | null): number | null {
  if (!cfg) return null;
  const scopes = [cfg, cfg.text_config, cfg.llm_config, cfg.language_config].filter((x): x is Record<string, unknown> => !!x && typeof x === "object");
  for (const s of scopes)
    for (const k of CTX_KEYS) {
      const v = s[k];
      if (typeof v === "number" && Number.isInteger(v) && v > 0 && v < 10_000_000) return v;
    }
  return null;
}

export interface ProfileInput {
  info: RawModelInfo;
  files: RepoFile[];
  config: Record<string, unknown> | null;
  tokenizerConfig: Record<string, unknown> | null;
  sentenceBertConfig: Record<string, unknown> | null;
}

export function profileOf(x: ProfileInput): ModelProfileView {
  const paths = x.files.map((f) => f.path);
  const has = (re: RegExp) => paths.some((p) => re.test(p));
  const architectures = ((x.config?.architectures as string[] | undefined) ?? x.info.config?.architectures ?? []).filter((a) => typeof a === "string");
  const modelType = (x.config?.model_type as string | undefined) ?? x.info.config?.model_type ?? null;
  const pipelineTag = x.info.pipeline_tag ?? x.info.transformersInfo?.pipeline_tag ?? null;
  const libraryName = x.info.library_name ?? null;
  const ggufFiles = paths.filter(isGguf);
  const onnxFiles = paths.filter(isOnnx);
  const hasHfWeights = paths.some(isHfWeight) && !!(x.config || x.info.config);
  const sentenceTransformers = has(/(^|\/)modules\.json$/) || has(/(^|\/)sentence_bert_config\.json$/) || libraryName === "sentence-transformers";
  const notes: string[] = [];

  let kind: ModelProfileView["kind"] = "unknown";
  let kindReason = "";
  if (pipelineTag && EMBED_PIPELINES.has(pipelineTag)) [kind, kindReason] = ["embed", `pipeline tag "${pipelineTag}"`];
  else if (pipelineTag && CHAT_PIPELINES.has(pipelineTag)) [kind, kindReason] = ["chat", `pipeline tag "${pipelineTag}"`];
  else if (sentenceTransformers) [kind, kindReason] = ["embed", "sentence-transformers files"];
  else if (pipelineTag) [kind, kindReason] = ["other", `pipeline tag "${pipelineTag}" is neither text generation nor embeddings`];
  else if (architectures.some((a) => /For(CausalLM|ConditionalGeneration)$/.test(a))) [kind, kindReason] = ["chat", `architecture ${architectures.join(", ")}`];
  else if (architectures.some((a) => /(Model|ForMaskedLM)$/.test(a))) [kind, kindReason] = ["embed", `architecture ${architectures.join(", ")} (encoder)`];
  else if (ggufFiles.length) [kind, kindReason] = ["unknown", "GGUF files without a pipeline tag; choose chat or embeddings when deploying"];
  else kindReason = "the repo does not say what the model does; choose chat or embeddings when deploying";

  const chatTemplate =
    !!x.tokenizerConfig?.chat_template || !!x.info.config?.tokenizer_config?.chat_template || has(/(^|\/)chat_template\.(jinja|json)$/) || !!x.info.gguf?.chat_template;
  if (kind === "chat" && !chatTemplate && !ggufFiles.length) notes.push("No chat template: this looks like a base model; chats are sent as plain role-prefixed text.");

  let contextLength = contextFromConfig(x.config) ?? (typeof x.info.gguf?.context_length === "number" ? x.info.gguf.context_length : null);
  const stMax = x.sentenceBertConfig?.max_seq_length;
  if (kind === "embed" && typeof stMax === "number") contextLength = contextLength ? Math.min(contextLength, stMax) : stMax;
  const tokMax = x.tokenizerConfig?.model_max_length;
  if (!contextLength && typeof tokMax === "number" && tokMax < 10_000_000) contextLength = tokMax;

  const remoteCode = !!(x.config?.auto_map || (x.info.config as Record<string, unknown> | undefined)?.auto_map);
  if (remoteCode) notes.push("The model ships its own Python code (auto_map); transformers runs it only with “trust remote code”.");
  const qc = (x.config?.quantization_config as { quant_method?: string } | undefined) ?? x.info.config?.quantization_config;
  const quantizedWith = qc?.quant_method ?? null;
  if (quantizedWith) notes.push(`Published already quantized (${quantizedWith}).`);
  if (x.info.gated) notes.push(`Gated repo (${x.info.gated}): downloads need a token whose account accepted the license.`);
  const formats = formatsOf(paths);
  if (!formats.length) notes.push("No weight files found.");

  return {
    kind,
    kindReason,
    pipelineTag,
    libraryName,
    architectures,
    modelType,
    formats,
    ggufFiles,
    onnxFiles,
    hasHfWeights,
    contextLength,
    hasChatTemplate: chatTemplate,
    remoteCode,
    quantizedWith,
    parameters: x.info.safetensors?.total ?? x.info.gguf?.total ?? null,
    notes,
  };
}

/** Files fetched by default: one copy of the weights in the most useful format, plus everything else.
 * GGUF repos usually hold many quantizations of the same model, so the user picks which ones. */
export function defaultSelection(files: RepoFile[]): string[] {
  const paths = files.map((f) => f.path);
  const hasSafetensors = paths.some((p) => /\.safetensors$/i.test(p));
  const hasHf = paths.some(isHfWeight);
  const ggufs = paths.filter(isGguf);
  return paths.filter((p) => {
    if (isGguf(p)) return !hasHf && ggufs.length === 1;
    if (isOnnx(p) || isOnnxData(p)) return !hasHf && !ggufs.length;
    if (hasSafetensors && /\.(bin|pt|pth)$/i.test(p) && isHfWeight(p)) return false;
    if (/(^|\/)original\//.test(p) && hasHf) return false;
    if (OTHER_WEIGHT.test(p) && (hasHf || ggufs.length)) return false;
    return true;
  });
}
