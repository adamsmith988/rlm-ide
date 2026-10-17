/** Stream events emitted by a turn. Every event carries a monotonically increasing `seq`
 * within its turn so a subscriber can resume after a renderer reload. */

export type TurnStatus = "running" | "done" | "error" | "cancelled";

export interface ToolCallInfo {
  callId: string;
  toolId: string;
  args: Record<string, unknown>;
}

export interface CheckpointInfo {
  summary: string;
  foldedTurns: number;
  keptTurns: number;
  compactedBy: "session" | "rlm" | "in_turn";
}

export interface RecalledMemory {
  id: string;
  text: string;
  memoryType: "short_term" | "long_term";
  score: number;
}

export type TurnEventBody =
  | { type: "turn_started"; chatId: string; userText: string; model: string; origin?: "user" | "wakeup" }
  | { type: "status"; text: string }
  /** The agent's failover chain moved to another model mid-turn; the turn keeps going. */
  | { type: "model_switch"; from: string; to: string; reason: string; resumed: boolean }
  | { type: "thinking_delta"; text: string }
  | { type: "text_delta"; text: string }
  /** The code cell the model is still writing: last lines only, throttled. Replaced by `code_cell` when complete. */
  | { type: "cell_draft"; lines: number; chars: number; tail: string; agentPath: string[] }
  /** Live generation progress for one model call, throttled. */
  | { type: "generation"; phase: "waiting" | "thinking" | "writing" | "writing_code"; outputTokens: number; tokPerSec: number; step: number }
  | { type: "code_cell"; cellId: string; code: string; agentPath: string[] }
  | { type: "cell_output"; cellId: string; stream: "stdout" | "stderr" | "result"; text: string }
  | { type: "cell_done"; cellId: string; ok: boolean; error?: string; durationMs: number }
  | { type: "tool_call"; call: ToolCallInfo; agentPath: string[] }
  | { type: "tool_result"; callId: string; ok: boolean; result: unknown; durationMs: number; deduped?: boolean }
  | { type: "approval_request"; approvalId: string; call: ToolCallInfo; preview?: string }
  | { type: "approval_resolved"; approvalId: string; approved: boolean }
  | { type: "compaction"; checkpoint: CheckpointInfo }
  | { type: "memory_recalled"; items: RecalledMemory[]; offline: boolean }
  | { type: "subagent_started"; agentId: string; parentPath: string[]; task: string }
  | { type: "subagent_done"; agentId: string; ok: boolean; result: string }
  /** All in prompt tokens of the next model call: the whole prompt, where compaction fires, the largest prompt that still leaves room to answer, and the window. */
  | { type: "context_usage"; usedTokens: number; triggerTokens: number; inputLimit: number; contextLength: number }
  | { type: "pending_edit"; editId: string; path: string; before: string; after: string }
  | { type: "turn_done"; finalText: string; usage?: { inputTokens: number; outputTokens: number } }
  | { type: "turn_error"; message: string }
  | { type: "turn_cancelled"; partialText: string };

export type TurnEvent = TurnEventBody & { turnId: string; seq: number; ts: number };

export type TurnEventType = TurnEventBody["type"];

export const TERMINAL_EVENT_TYPES: ReadonlySet<TurnEventType> = new Set([
  "turn_done",
  "turn_error",
  "turn_cancelled",
]);
