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
  /** Set when the IDE runs this provider's server itself. */
  managed: z.enum(["omniroute"]).optional(),
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
  /** The server drops the system role (some free web-playground providers); the agent's instructions, memory and
   * task board then travel at the top of the first user message instead. */
  systemAsUser: z.boolean().optional(),
});
export type ModelConfig = z.infer<typeof ModelConfig>;

/** Conservative token estimate. Code and non-English text tokenize denser than prose, so we
 * assume fewer characters per token than the usual ~4 to avoid underestimating. */
export const CHARS_PER_TOKEN = 3.2;

type Msg = { role: "system" | "user" | "assistant"; content: string };

/** Moves every system message into the first user message, for servers that ignore the system role. */
export function foldSystemPrompt<M extends Msg>(messages: M[]): Msg[] {
  const system = messages.filter((m) => m.role === "system").map((m) => m.content).join("\n\n");
  const rest: Msg[] = messages.filter((m) => m.role !== "system");
  if (!system) return rest;
  const preamble = `<instructions>\n${system}\n</instructions>`;
  if (rest[0]?.role === "user") return [{ role: "user", content: `${preamble}\n\n${rest[0].content}` }, ...rest.slice(1)];
  return [{ role: "user", content: preamble }, ...rest];
}

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
