import type { RlmSettings } from "@rlm/contracts";
import { appendLog, emptyBoard, hasContent, renderBoard, replacePlan, STEP_STATUSES, type StepStatus, type TaskBoard } from "../domain/task_board";
import type { ToolDescriptor } from "../domain/tools";
import type { ClockPort, LoggerPort, TaskBoardStore, ToolImpl } from "../ports";

interface TurnContext {
  /** The user's message for this turn; null on wake-up turns. */
  request: string | null;
  workspace: string | null;
}

/** Per-chat task boards: the agent's durable plan, progress log and wake-up schedule. */
export class TaskBoards {
  private readonly cache = new Map<string, TaskBoard>();
  private readonly chains = new Map<string, Promise<unknown>>();
  private readonly turns = new Map<string, TurnContext>();
  private readonly listeners = new Set<(b: TaskBoard) => void>();

  constructor(
    private readonly d: {
      store: TaskBoardStore;
      clock: ClockPort;
      settings: () => RlmSettings;
      log: LoggerPort;
    },
  ) {}

  onChange(cb: (b: TaskBoard) => void): () => void {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }

  setTurnContext(chatId: string, ctx: TurnContext): void {
    this.turns.set(chatId, ctx);
  }

  async get(chatId: string): Promise<TaskBoard> {
    const hit = this.cache.get(chatId);
    if (hit) return hit;
    const loaded = (await this.d.store.load(chatId).catch(() => null)) ?? emptyBoard(chatId, this.d.clock.now());
    this.cache.set(chatId, loaded);
    return loaded;
  }

  /** Serialized read-modify-write per chat. */
  private mutate(chatId: string, fn: (b: TaskBoard) => TaskBoard): Promise<TaskBoard> {
    const prev = this.chains.get(chatId) ?? Promise.resolve();
    const next = prev.then(async () => {
      const b = fn(await this.get(chatId));
      this.cache.set(chatId, b);
      await this.d.store.save(b);
      for (const l of this.listeners) l(b);
      return b;
    });
    this.chains.set(chatId, next.catch(() => undefined));
    return next;
  }

  async render(chatId: string): Promise<string> {
    const b = await this.get(chatId);
    return hasContent(b) ? renderBoard(b, this.d.clock.now()) : "";
  }

  async plan(chatId: string, goal: string, steps: string[]): Promise<TaskBoard> {
    const request = this.turns.get(chatId)?.request ?? null;
    return this.mutate(chatId, (b) => replacePlan(b, goal, steps, request, this.d.clock.now()));
  }

  async setStep(chatId: string, index: number, status: StepStatus, note: string | null): Promise<TaskBoard> {
    return this.mutate(chatId, (b) => {
      if (!b.steps[index]) throw new Error(`no step ${index}; the board has ${b.steps.length} steps (0-${b.steps.length - 1}). Call task.plan first.`);
      const steps = b.steps.map((s, i) => (i === index ? { ...s, status, ...(note !== null ? { note: note.trim() || undefined } : {}) } : s));
      return { ...b, steps, updatedAt: this.d.clock.now() };
    });
  }

  async log(chatId: string, text: string): Promise<TaskBoard> {
    if (!text.trim()) throw new Error("text is empty");
    return this.mutate(chatId, (b) => appendLog(b, text, this.d.clock.now()));
  }

  async scheduleWakeup(chatId: string, afterSec: number, reason: string): Promise<TaskBoard> {
    const s = this.d.settings();
    if (!(afterSec >= s.wakeupMinSec && afterSec <= s.wakeupMaxSec)) throw new Error(`after_sec must be between ${s.wakeupMinSec} and ${s.wakeupMaxSec}`);
    if (!reason.trim()) throw new Error("reason is required: say what to check and do when you wake up");
    const now = this.d.clock.now();
    const workspace = this.turns.get(chatId)?.workspace ?? null;
    return this.mutate(chatId, (b) => ({ ...b, wakeup: { at: now + afterSec * 1000, reason: reason.trim(), workspace, scheduledAt: now }, updatedAt: now }));
  }

  async cancelWakeup(chatId: string): Promise<TaskBoard> {
    return this.mutate(chatId, (b) => ({ ...b, wakeup: null, updatedAt: this.d.clock.now() }));
  }

  /** Clears the due wake-up and counts it; returns the plan that fired, or null when it is gone. */
  takeWakeup(chatId: string): Promise<{ board: TaskBoard; fired: TaskBoard["wakeup"] }> {
    let fired: TaskBoard["wakeup"] = null;
    return this.mutate(chatId, (b) => {
      fired = b.wakeup;
      return fired ? { ...b, wakeup: null, autoWakeups: b.autoWakeups + 1, updatedAt: this.d.clock.now() } : b;
    }).then((board) => ({ board, fired }));
  }

  /** A message from the user resets the run of automatic wake-ups. */
  async noteUserTurn(chatId: string): Promise<void> {
    const b = await this.get(chatId);
    if (b.autoWakeups) await this.mutate(chatId, (x) => ({ ...x, autoWakeups: 0 }));
  }

  async pending(): Promise<TaskBoard[]> {
    const all = await this.d.store.list().catch(() => [] as TaskBoard[]);
    for (const b of all) if (!this.cache.has(b.chatId)) this.cache.set(b.chatId, b);
    return all.filter((b) => b.wakeup).map((b) => this.cache.get(b.chatId) ?? b);
  }

  async drop(chatId: string): Promise<void> {
    this.cache.delete(chatId);
    this.turns.delete(chatId);
    await this.d.store.delete(chatId).catch(() => undefined);
  }

  /** `task.*` tools: the model's handle on its own board. */
  tools(): ToolImpl[] {
    const tool = (d: Omit<ToolDescriptor, "group" | "sideEffects" | "timeoutSec">, run: ToolImpl["run"]): ToolImpl => ({
      descriptor: { group: "task", sideEffects: false, timeoutSec: 10, ...d },
      run,
    });
    const view = async (chatId: string) => renderBoard(await this.get(chatId), this.d.clock.now(), 40);
    return [
      tool(
        {
          id: "task.plan",
          description:
            "Write or replace the plan for the current job: a one-line goal and the ordered steps. The board is shown to you on every step, survives context compaction and wake-ups, and is your memory of what is done and what is next. Steps whose text is unchanged keep their status.",
          parameters: {
            type: "object",
            properties: { goal: { type: "string", description: "The job in one line" }, steps: { type: "array", items: { type: "string" }, description: "Ordered steps" } },
            required: ["goal", "steps"],
          },
          returns: "str (the board)",
        },
        async (a, ctx) => {
          await this.plan(ctx.chatId, String(a.goal), (a.steps as unknown[]).map(String));
          return view(ctx.chatId);
        },
      ),
      tool(
        {
          id: "task.step",
          description: "Set the status of a step on the board (by index) and optionally a short note: what was found, the output path, the error.",
          parameters: {
            type: "object",
            properties: {
              index: { type: "integer", description: "Step index from the board" },
              status: { type: "string", enum: [...STEP_STATUSES], description: "todo | doing | done | blocked | skipped" },
              note: { type: "string", description: "Short note kept on the step" },
            },
            required: ["index", "status"],
          },
          returns: "str (the board)",
        },
        async (a, ctx) => {
          await this.setStep(ctx.chatId, Number(a.index), a.status as StepStatus, a.note === undefined ? null : String(a.note));
          return view(ctx.chatId);
        },
      ),
      tool(
        {
          id: "task.log",
          description: "Add a line to the board's progress log: a result, a decision, a file written, an error and its fix, a job started (with its pid and log file).",
          parameters: { type: "object", properties: { text: { type: "string", description: "One short line" } }, required: ["text"] },
          returns: "str",
        },
        async (a, ctx) => {
          const b = await this.log(ctx.chatId, String(a.text));
          return `logged (${b.loggedTotal} entries)`;
        },
      ),
      tool(
        {
          id: "task.wake_me",
          description:
            "Schedule a wake-up for long jobs: the IDE starts a new turn in this chat after `after_sec` seconds with your board and `reason`, even if nobody is watching. After calling it, end your turn with a short status message (no repl block). Use it instead of waiting in a loop when a job needs more than a few minutes.",
          parameters: {
            type: "object",
            properties: {
              after_sec: { type: "integer", description: "Seconds from now" },
              reason: { type: "string", description: "What to check and do when you wake up (pids, log files, next step)" },
            },
            required: ["after_sec", "reason"],
          },
          returns: "str",
        },
        async (a, ctx) => {
          const b = await this.scheduleWakeup(ctx.chatId, Number(a.after_sec), String(a.reason));
          return `wake-up scheduled for ${new Date(b.wakeup!.at).toISOString()} (in ${Number(a.after_sec)}s). End your turn now with a short status for the user.`;
        },
      ),
      tool(
        {
          id: "task.cancel_wakeup",
          description: "Cancel the scheduled wake-up of this chat.",
          parameters: { type: "object", properties: {} },
          returns: "str",
        },
        async (_a, ctx) => {
          await this.cancelWakeup(ctx.chatId);
          return "no wake-up scheduled";
        },
      ),
      tool(
        {
          id: "task.board",
          description: "Return the full board, including the whole kept progress log.",
          parameters: { type: "object", properties: {} },
          returns: "str",
        },
        async (_a, ctx) => view(ctx.chatId),
      ),
    ];
  }
}
