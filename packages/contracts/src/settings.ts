import { z } from "zod";
import { ModelConfig, ProviderConfig, BUILTIN } from "./models";
import { HubSettings } from "./hub";

/** Every threshold used by the orchestrator lives here with a default and a reason.
 * Logic code never hard-codes these numbers. */

export const CompactSettings = z.object({
  /** Share of the context window the conversation history may use. */
  historyShare: z.number().min(0.1).max(0.95).default(0.6),
  /** Compaction fires when history reaches this share of the history budget. */
  triggerRatio: z.number().min(0.2).max(1).default(0.8),
  /** Share of the history budget kept verbatim as the recent tail. */
  keepRecentRatio: z.number().min(0.05).max(0.9).default(0.35),
  /** Output reserve is capped at this share of the window. */
  maxOutputRatio: z.number().min(0.05).max(0.5).default(0.25),
  /** Second trigger: compact when stored rows exceed this many turns per 1k tokens of history budget. */
  maxRowsPer1kHistory: z.number().min(0.1).default(4),
  /** A very long newest turn is still kept verbatim when it fits this share of the history budget. */
  newestLimitRatio: z.number().min(0.1).max(1).default(0.6),
  /** Max tokens the summarizer may produce for a checkpoint, as a share of max output. */
  summaryMaxShare: z.number().min(0.1).max(1).default(0.5),
});

export const RlmSettings = z.object({
  /** RLM transcript compaction trigger as a share of context length. */
  compactTriggerRatio: z.number().min(0.3).max(0.98).default(0.9),
  /** Max model/cell iterations within one turn. */
  maxSteps: z.number().int().min(1).max(1000).default(80),
  /** The model is told how many steps remain once this few are left, so it can leave the work resumable. */
  stepWarnRemaining: z.number().int().min(0).default(5),
  /** Shortest and longest delay the agent may schedule a wake-up for. */
  wakeupMinSec: z.number().int().min(5).default(30),
  wakeupMaxSec: z.number().int().min(60).default(86400),
  /** Wake-ups allowed in a row without a message from the user, so a runaway loop stops on its own. */
  maxAutoWakeups: z.number().int().min(0).default(200),
  /** An API model request fails when the server sends nothing for this long (slow prompt processing included). */
  modelStallSec: z.number().int().min(10).default(300),
  /** Per-cell execution timeout. */
  cellTimeoutSec: z.number().int().min(5).default(120),
  /** Cell output beyond this many chars is truncated with a marker. */
  cellOutputLimitChars: z.number().int().min(1024).default(16000),
  /** REPL outputs older than this many steps are truncated first when over budget. */
  oldOutputKeepChars: z.number().int().min(200).default(1500),
  /** After a turn that edited files, ask the model once to check its work against the request before answering. */
  selfReview: z.boolean().default(true),
  maxSubagentDepth: z.number().int().min(0).max(5).default(2),
  maxConcurrentChildren: z.number().int().min(1).max(16).default(3),
  /** Kernel memory cap (MB) for the Windows Job Object / Linux prlimit. */
  kernelMemoryMb: z.number().int().min(256).default(4096),
  /** Disable network inside the Linux bwrap jail. */
  sandboxNoNetwork: z.boolean().default(true),
});

export const MemorySettings = z.object({
  enabled: z.boolean().default(true),
  chunkChars: z.number().int().min(200).default(1200),
  chunkOverlap: z.number().int().min(0).default(150),
  maxChunksPerTurn: z.number().int().min(1).default(64),
  minScore: z.number().min(0).max(1).default(0.45),
  sessionLimit: z.number().int().min(0).default(6),
  userLimit: z.number().int().min(0).default(6),
  /** While the user has at most this many long-term facts, all of them are recalled (they are standing preferences). */
  alwaysRecallFacts: z.number().int().min(0).default(12),
  promptMaxChars: z.number().int().min(200).default(4000),
  factKnownLimit: z.number().int().min(0).default(12),
  factRequireUserGrounding: z.boolean().default(true),
  factDedupScore: z.number().min(0).max(1).default(0.88),
  historyKeep: z.number().int().min(0).default(5),
  /** "user" = user-wide long-term memory (source default), "workspace" = per project. */
  longTermScope: z.enum(["user", "workspace"]).default("user"),
});

export const ToolSettings = z.object({
  approvalMode: z.enum(["ask", "session", "workspace"]).default("ask"),
  /** Tool ids auto-approved for the current workspace. */
  workspaceAllow: z.array(z.string()).default([]),
  shellAllowPatterns: z.array(z.string()).default(["^git (status|diff|log)", "^ls\\b", "^pnpm (test|run|build|install)", "^npm (test|run|install)"]),
  shellDenyPatterns: z.array(z.string()).default(["rm -rf /", ":\\(\\)\\{", "mkfs", "dd if="]),
  autoApplyEdits: z.boolean().default(false),
  actionCheck: z.boolean().default(false),
  shellTimeoutSec: z.number().int().min(1).default(300),
});

export const LocalInferenceSettings = z.object({
  /** Cap for the local model's context, bounded by the memory available for the KV cache. */
  localContextCap: z.number().int().min(2048).default(32768),
  localMaxOutputTokens: z.number().int().min(256).default(4096),
  preferredDevice: z.enum(["auto", "webgpu", "cpu"]).default("auto"),
  enableThinking: z.boolean().default(true),
});

export const Settings = z.object({
  theme: z.enum(["dark", "light"]).default("dark"),
  fontSize: z.number().int().min(10).max(24).default(13),
  userId: z.string().default("local-user"),
  persona: z.string().default(""),
  providers: z.array(ProviderConfig).default([
    { id: BUILTIN.chatProviderId, kind: "local_onnx", label: "Built-in (local)", baseUrl: "", hasKey: false },
  ]),
  models: z.array(ModelConfig).default([
    {
      id: BUILTIN.agentModelId,
      providerId: BUILTIN.chatProviderId,
      name: BUILTIN.agentDir,
      label: "FrogNano 4B (local)",
      kind: "chat",
      contextLength: null,
      maxOutputTokens: null,
      dim: null,
    },
    {
      id: BUILTIN.embedModelId,
      providerId: BUILTIN.chatProviderId,
      name: BUILTIN.embedDir,
      label: "bge-m3 (local)",
      kind: "embed",
      contextLength: 8192,
      maxOutputTokens: null,
      dim: BUILTIN.embedDim,
    },
  ]),
  roles: z
    .object({
      agent: z.string().default(BUILTIN.agentModelId),
      summarizer: z.string().default(""),
      embed: z.string().default(BUILTIN.embedModelId),
    })
    .default({}),
  compact: CompactSettings.default({}),
  rlm: RlmSettings.default({}),
  memory: MemorySettings.default({}),
  tools: ToolSettings.default({}),
  local: LocalInferenceSettings.default({}),
  hub: HubSettings.default({}),
  autoUpdate: z.boolean().default(true),
  updateChannel: z.enum(["stable", "beta"]).default("stable"),
  recentWorkspaces: z.array(z.string()).default([]),
});
export type Settings = z.infer<typeof Settings>;
export type CompactSettings = z.infer<typeof CompactSettings>;
export type RlmSettings = z.infer<typeof RlmSettings>;
export type MemorySettings = z.infer<typeof MemorySettings>;
export type ToolSettings = z.infer<typeof ToolSettings>;

export function defaultSettings(): Settings {
  return Settings.parse({});
}

/** Points roles whose model was removed back at a model that exists, preferring the built-ins. */
export function repairRoles(s: Settings): Settings {
  const has = (id: string, kind: "chat" | "embed") => s.models.some((m) => m.id === id && m.kind === kind);
  const pick = (kind: "chat" | "embed", builtin: string) => (has(builtin, kind) ? builtin : (s.models.find((m) => m.kind === kind)?.id ?? builtin));
  const roles = { ...s.roles };
  if (!has(roles.agent, "chat")) roles.agent = pick("chat", BUILTIN.agentModelId);
  if (roles.summarizer && !has(roles.summarizer, "chat")) roles.summarizer = "";
  if (roles.embed && !has(roles.embed, "embed")) roles.embed = pick("embed", BUILTIN.embedModelId);
  const same = roles.agent === s.roles.agent && roles.summarizer === s.roles.summarizer && roles.embed === s.roles.embed;
  return same ? s : { ...s, roles };
}
