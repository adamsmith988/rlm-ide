import { z } from "zod";

/** Provider kinds the ProviderRouter knows how to build adapters for. */
export const ProviderKind = z.enum(["local_onnx", "openai_compat"]);
export type ProviderKind = z.infer<typeof ProviderKind>;

export const ProviderConfig = z.object({
  id: z.string().min(1),
  kind: ProviderKind,
  label: z.string(),
  /** openai_compat only. Example: https://api.openai.com/v1 or http://localhost:11434/v1 */
  baseUrl: z.string().default(""),
  /** True when an API key is stored in the secret store under `provider:<id>`. */
  hasKey: z.boolean().default(false),
});
export type ProviderConfig = z.infer<typeof ProviderConfig>;

export const ModelConfig = z.object({
  id: z.string().min(1),
  providerId: z.string().min(1),
  /** Model name sent to the provider (or the local model folder name). */
  name: z.string().min(1),
  label: z.string(),
  kind: z.enum(["chat", "embed"]),
  /** Null until auto-detected or entered manually. Sending is blocked while null. */
  contextLength: z.number().int().positive().nullable().default(null),
  maxOutputTokens: z.number().int().positive().nullable().default(null),
  /** Embedding dimension (embed models only). */
  dim: z.number().int().positive().nullable().default(null),
});
export type ModelConfig = z.infer<typeof ModelConfig>;

export const ModelRole = z.enum(["agent", "summarizer", "embed"]);
export type ModelRole = z.infer<typeof ModelRole>;

/** Built-in local models shipped with or installed by the app. */
export const BUILTIN = {
  chatProviderId: "builtin",
  agentModelId: "builtin:frognano-4b",
  agentRepo: "RASMUS/FrogNano-4B-2609-ONNX",
  agentDir: "frognano-4b",
  embedModelId: "builtin:bge-m3",
  embedRepo: "Xenova/bge-m3",
  embedDir: "bge-m3",
  embedDim: 1024,
} as const;

export interface LocalModelStatus {
  id: string;
  installed: boolean;
  bytesOnDisk: number;
  expectedBytes: number;
  runtime: "webgpu" | "cpu" | "unavailable";
  detail: string;
}
