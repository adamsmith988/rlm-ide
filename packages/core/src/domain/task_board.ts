import { clipChars } from "./tokens";

export type StepStatus = "todo" | "doing" | "done" | "blocked" | "skipped";
export const STEP_STATUSES: readonly StepStatus[] = ["todo", "doing", "done", "blocked", "skipped"];

export interface TaskStep {
  text: string;
  status: StepStatus;
  note?: string;
}

export interface WakeupPlan {
  /** Epoch ms when the agent is woken. */
  at: number;
  reason: string;
  /** Workspace the chat was working in; the wake-up waits until that workspace is open. */
  workspace: string | null;
  scheduledAt: number;
}

/** The agent's own durable record of a job: survives compaction, restarts and wake-ups. */
export interface TaskBoard {
  chatId: string;
  goal: string;
  /** The user request the plan was made for, verbatim (clipped). */
  request: string;
  steps: TaskStep[];
  /** Newest last. Older entries are dropped; `loggedTotal` keeps the count. */
  log: Array<{ at: number; text: string }>;
  loggedTotal: number;
  wakeup: WakeupPlan | null;
  /** Wake-ups fired since the user last wrote in this chat. */
  autoWakeups: number;
  updatedAt: number;
}

export const BOARD_LOG_KEEP = 40;
export const BOARD_REQUEST_CHARS = 6000;

export function emptyBoard(chatId: string, now: number): TaskBoard {
  return { chatId, goal: "", request: "", steps: [], log: [], loggedTotal: 0, wakeup: null, autoWakeups: 0, updatedAt: now };
}

export function hasContent(b: TaskBoard | null): b is TaskBoard {
  return !!b && (!!b.goal || b.steps.length > 0 || b.log.length > 0 || !!b.wakeup);
}

/** Replacing the plan keeps the status and note of steps whose text did not change. */
export function replacePlan(b: TaskBoard, goal: string, steps: string[], request: string | null, now: number): TaskBoard {
  const old = new Map(b.steps.map((s) => [s.text.trim().toLowerCase(), s]));
  const next: TaskStep[] = steps
    .map((t) => t.trim())
    .filter(Boolean)
    .map((text) => {
      const prev = old.get(text.toLowerCase());
      return prev ? { ...prev, text } : { text, status: "todo" as const };
    });
  return { ...b, goal: goal.trim() || b.goal, request: request !== null ? clipChars(request.trim(), BOARD_REQUEST_CHARS) : b.request, steps: next, updatedAt: now };
}

export function appendLog(b: TaskBoard, text: string, now: number): TaskBoard {
  const log = [...b.log, { at: now, text: clipChars(text.trim(), 600) }];
  return { ...b, log: log.slice(-BOARD_LOG_KEEP), loggedTotal: b.loggedTotal + 1, updatedAt: now };
}

function ago(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 90) return `${s}s ago`;
  if (s < 5400) return `${Math.round(s / 60)}m ago`;
  return `${(s / 3600).toFixed(1)}h ago`;
}

const MARK: Record<StepStatus, string> = { todo: "[ ]", doing: "[>]", done: "[x]", blocked: "[!]", skipped: "[-]" };

/** The block the model sees in every request. */
export function renderBoard(b: TaskBoard, now: number, logLines = 12): string {
  const out: string[] = [];
  if (b.goal) out.push(`Goal: ${b.goal}`);
  if (b.request && b.request !== b.goal) out.push(`Original request (verbatim):\n"""\n${b.request}\n"""`);
  if (b.steps.length) {
    out.push("Steps (index, status, text):");
    b.steps.forEach((s, i) => out.push(`${i}. ${MARK[s.status]} ${s.text}${s.note ? ` — ${s.note}` : ""}`));
    const done = b.steps.filter((s) => s.status === "done" || s.status === "skipped").length;
    const next = b.steps.findIndex((s) => s.status === "doing" || s.status === "todo" || s.status === "blocked");
    out.push(`Progress: ${done}/${b.steps.length} done${next >= 0 ? `; next: step ${next}` : "; all steps finished"}`);
  }
  if (b.log.length) {
    const shown = b.log.slice(-logLines);
    const hidden = b.loggedTotal - shown.length;
    out.push(`Progress log (newest last${hidden > 0 ? `; ${hidden} older entries not shown` : ""}):`);
    for (const e of shown) out.push(`- (${ago(now - e.at)}) ${e.text}`);
  }
  if (b.wakeup) out.push(`Wake-up scheduled in ${Math.max(0, Math.round((b.wakeup.at - now) / 1000))}s: ${b.wakeup.reason}`);
  return out.join("\n");
}
