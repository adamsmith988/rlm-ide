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
  /** A cell's output is also kept under this share of the model's window, so one output never crowds out the rest. */
  cellOutputMaxShare: z.number().min(0.05).max(1).default(0.25),
  /** When the full system prompt (protocol and tool documentation) would take more than this share of the model's
   * window, the prompt lists tools compactly (signature and summary) and `help(tool)` in the REPL shows the rest. */
  toolDocsMaxShare: z.number().min(0.05).max(1).default(0.25),
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

/** How a role's model chain behaves when a model fails mid-turn. */
export const FailoverSettings = z.object({
  enabled: z.boolean().default(true),
  /** Cooldown after a transient failure (rate limit, 5xx, network, stall, empty reply) when the server gives no Retry-After. Doubles per consecutive failure. */
  cooldownSec: z.number().int().min(1).default(30),
  maxCooldownSec: z.number().int().min(10).default(900),
  /** Cooldown for failures a retry will not fix soon (bad key, unknown model, rejected request). */
  hardFailureCooldownSec: z.number().int().min(10).default(1800),
  /** When every model in the chain is cooling down, the turn waits for the first to recover instead of failing, for at most this long. */
  waitForRecoverySec: z.number().int().min(0).default(900),
  /** A model that fails after it started replying hands the partial reply to the next model, which continues it. */
  continueMidReply: z.boolean().default(true),
  /** A reply with no text and no reasoning counts as a failure (free gateways sometimes return empty turns). */
  emptyReplyIsFailure: z.boolean().default(true),
});

/** IDE-managed OmniRoute gateway (free models across many providers). */
export const OmniRouteSettings = z.object({
  /** Start the managed gateway with the app once it is installed. */
  autoStart: z.boolean().default(true),
  port: z.number().int().min(1024).max(65535).default(20128),
  /** Name of the gateway combo kept in sync with the chosen free models that honour system prompts, for other apps. */
  comboName: z.string().min(1).default("rlm-free"),
  /** Probe timeout when testing which free models answer. */
  probeTimeoutSec: z.number().int().min(5).default(90),
  /** How many working free models the auto-setup keeps in the chain. */
  maxChainModels: z.number().int().min(1).max(20).default(6),
  /** Web-playground providers accept far less input than the gateway advertises. The test sends a prompt this long;
   * a model that refuses it is narrowed down to its real limit, which becomes its context window. */
  windowProbeChars: z.number().int().min(4000).default(40000),
  /** A model whose real input limit is below this cannot hold the agent's instructions and work, so it is not chained. */
  minAgentInputChars: z.number().int().min(2000).default(12000),
  /** npm package spec installed into the app data folder. */
  packageSpec: z.string().min(1).default("omniroute@latest"),
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
      /** Tried in order when the agent model fails; the summarizer follows the same chain when it has no model of its own. */
      agentFallbacks: z.array(z.string()).default([]),
    })
    .default({}),
  failover: FailoverSettings.default({}),
  omniroute: OmniRouteSettings.default({}),
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
export type FailoverSettings = z.infer<typeof FailoverSettings>;
export type OmniRouteSettings = z.infer<typeof OmniRouteSettings>;

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
  roles.agentFallbacks = [...new Set(roles.agentFallbacks.filter((id) => id !== roles.agent && has(id, "chat")))];
  const sameFallbacks = roles.agentFallbacks.length === s.roles.agentFallbacks.length && roles.agentFallbacks.every((id, i) => id === s.roles.agentFallbacks[i]);
  const same = roles.agent === s.roles.agent && roles.summarizer === s.roles.summarizer && roles.embed === s.roles.embed && sameFallbacks;
  return same ? s : { ...s, roles };
}
