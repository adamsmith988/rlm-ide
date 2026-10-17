import type { CheckpointInfo, RecalledMemory, ToolCallInfo, TurnEvent } from "@rlm/contracts";

export type Block =
  | { kind: "text"; text: string }
  | { kind: "thinking"; text: string }
  | { kind: "cell"; cellId: string; code: string; output: string; ok?: boolean; error?: string; durationMs?: number; agentPath: string[] }
  | { kind: "tool"; call: ToolCallInfo; ok?: boolean; result?: unknown; durationMs?: number; deduped?: boolean; agentPath: string[] }
  | { kind: "approval"; approvalId: string; call: ToolCallInfo; preview?: string; resolved?: boolean; approved?: boolean }
  | { kind: "compaction"; checkpoint: CheckpointInfo }
  | { kind: "memory"; items: RecalledMemory[]; offline: boolean }
  | { kind: "subagent"; agentId: string; parentPath: string[]; task: string; done?: boolean; ok?: boolean; result?: string }
  | { kind: "edit"; editId: string; path: string; before: string; after: string }
  | { kind: "status"; text: string }
  | { kind: "switch"; from: string; to: string; reason: string; resumed: boolean }
  | { kind: "error"; text: string };

export interface UiMessage {
  id: string;
  turnId: string;
  role: "user" | "assistant";
  /** "wakeup": a user-role message the IDE wrote for a scheduled wake-up. */
  origin?: "user" | "wakeup";
  text: string;
  status: "running" | "done" | "error" | "cancelled";
  blocks: Block[];
  lastSeq: number;
  feedback?: "up" | "down" | null;
  usage?: { inputTokens: number; outputTokens: number };
  startedAt?: number;
  endedAt?: number;
  /** Live state of the current model call while the turn runs. */
  progress?: { phase: "waiting" | "thinking" | "writing" | "writing_code"; outputTokens: number; tokPerSec: number; step: number; ts: number } | null;
  /** The code cell being written right now (tail only). */
  draft?: { lines: number; chars: number; tail: string } | null;
}

export interface ContextUsage {
  usedTokens: number;
  triggerTokens: number;
  inputLimit: number;
  contextLength: number;
}

/** Pure reducer: applies one stream event to an assistant message. Used live and for replay. */
export function applyEvent(m: UiMessage, ev: TurnEvent): UiMessage {
  if (ev.seq <= m.lastSeq) return m;
  const blocks = [...m.blocks];
  const last = blocks[blocks.length - 1];
  const next: UiMessage = { ...m, blocks, lastSeq: ev.seq };
  if (ev.type === "turn_done" || ev.type === "turn_error" || ev.type === "turn_cancelled") {
    next.progress = null;
    next.draft = null;
  }
  switch (ev.type) {
    case "turn_started":
      next.startedAt = ev.ts;
      break;
    case "status":
      if (last?.kind === "status") blocks[blocks.length - 1] = { kind: "status", text: ev.text };
      else blocks.push({ kind: "status", text: ev.text });
      break;
    case "model_switch": {
      const b: Block = { kind: "switch", from: ev.from, to: ev.to, reason: ev.reason, resumed: ev.resumed };
      if (last?.kind === "status") blocks[blocks.length - 1] = b;
      else blocks.push(b);
      break;
    }
    case "thinking_delta":
      if (last?.kind === "thinking") blocks[blocks.length - 1] = { ...last, text: last.text + ev.text };
      else blocks.push({ kind: "thinking", text: ev.text });
      break;
    case "text_delta": {
      if (last?.kind === "text") {
        blocks[blocks.length - 1] = { ...last, text: last.text + ev.text };
        break;
      }
      // A reply continued by another model stays one text block; the switches and thoughts move above it.
      const j = resumedText(blocks);
      if (j >= 0) {
        const between = blocks.splice(j + 1);
        const t = blocks.pop() as Extract<Block, { kind: "text" }>;
        blocks.push(...between, { ...t, text: t.text + ev.text });
      } else blocks.push({ kind: "text", text: ev.text });
      break;
    }
    case "generation":
      next.progress = { phase: ev.phase, outputTokens: ev.outputTokens, tokPerSec: ev.tokPerSec, step: ev.step, ts: ev.ts };
      if (ev.phase !== "writing_code") next.draft = null;
      break;
    case "cell_draft":
      if (ev.agentPath.length === 0) next.draft = { lines: ev.lines, chars: ev.chars, tail: ev.tail };
      break;
    case "code_cell":
      next.draft = null;
      blocks.push({ kind: "cell", cellId: ev.cellId, code: ev.code, output: "", agentPath: ev.agentPath });
      break;
    case "cell_output": {
      const i = findLast(blocks, (b) => b.kind === "cell" && b.cellId === ev.cellId);
      if (i >= 0) {
        const b = blocks[i] as Extract<Block, { kind: "cell" }>;
        blocks[i] = { ...b, output: b.output + (ev.stream === "result" ? `${b.output ? "\n" : ""}→ ${ev.text}` : ev.text) };
      }
      break;
    }
    case "cell_done": {
      const i = findLast(blocks, (b) => b.kind === "cell" && b.cellId === ev.cellId);
      if (i >= 0) blocks[i] = { ...(blocks[i] as Extract<Block, { kind: "cell" }>), ok: ev.ok, error: ev.error, durationMs: ev.durationMs };
      break;
    }
    case "tool_call":
      blocks.push({ kind: "tool", call: ev.call, agentPath: ev.agentPath });
      break;
    case "tool_result": {
      const i = findLast(blocks, (b) => b.kind === "tool" && b.call.callId === ev.callId);
      if (i >= 0) blocks[i] = { ...(blocks[i] as Extract<Block, { kind: "tool" }>), ok: ev.ok, result: ev.result, durationMs: ev.durationMs, deduped: ev.deduped };
      break;
    }
    case "approval_request":
      blocks.push({ kind: "approval", approvalId: ev.approvalId, call: ev.call, preview: ev.preview });
      break;
    case "approval_resolved": {
      const i = findLast(blocks, (b) => b.kind === "approval" && b.approvalId === ev.approvalId);
      if (i >= 0) blocks[i] = { ...(blocks[i] as Extract<Block, { kind: "approval" }>), resolved: true, approved: ev.approved };
      break;
    }
    case "compaction":
      blocks.push({ kind: "compaction", checkpoint: ev.checkpoint });
      break;
    case "memory_recalled":
      if (ev.items.length || ev.offline) blocks.push({ kind: "memory", items: ev.items, offline: ev.offline });
      break;
    case "subagent_started":
      blocks.push({ kind: "subagent", agentId: ev.agentId, parentPath: ev.parentPath, task: ev.task });
      break;
    case "subagent_done": {
      const i = findLast(blocks, (b) => b.kind === "subagent" && b.agentId === ev.agentId);
      if (i >= 0) blocks[i] = { ...(blocks[i] as Extract<Block, { kind: "subagent" }>), done: true, ok: ev.ok, result: ev.result };
      break;
    }
    case "pending_edit":
      blocks.push({ kind: "edit", editId: ev.editId, path: ev.path, before: ev.before, after: ev.after });
      break;
    case "turn_done":
      next.status = "done";
      next.text = ev.finalText;
      next.usage = ev.usage;
      next.endedAt = ev.ts;
      next.blocks = blocks.filter((b) => b.kind !== "status");
      break;
    case "turn_error":
      next.status = "error";
      next.endedAt = ev.ts;
      next.blocks = [...blocks.filter((b) => b.kind !== "status"), { kind: "error", text: ev.message }];
      break;
    case "turn_cancelled":
      next.status = "cancelled";
      next.endedAt = ev.ts;
      next.blocks = blocks.filter((b) => b.kind !== "status");
      break;
    default:
      break;
  }
  return next;
}

/** Index of the text block a continued reply belongs to: only switches, thoughts and status lines follow it, and one
 * of those switches resumed the reply. -1 when the new text starts a new block. */
function resumedText(blocks: Block[]): number {
  let resumed = false;
  for (let i = blocks.length - 1; i >= 0; i--) {
    const b = blocks[i]!;
    if (b.kind === "text") return resumed ? i : -1;
    if (b.kind === "switch") resumed ||= b.resumed;
    else if (b.kind !== "thinking" && b.kind !== "status") return -1;
  }
  return -1;
}

function findLast<T>(arr: T[], pred: (x: T) => boolean): number {
  for (let i = arr.length - 1; i >= 0; i--) if (pred(arr[i]!)) return i;
  return -1;
}
