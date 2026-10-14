import type { Settings, TurnEvent, TurnEventBody, ChatSummary, StoredMessage } from "@rlm/contracts";
import { latestCheckpoint, verbatimTurns, type StoredTurn } from "../domain/conversation";
import { clipChars } from "../domain/tokens";
import type {
  ClockPort,
  ConversationRepo,
  EditJournal,
  KernelFactory,
  LoggerPort,
  MemoryVectorStore,
  ModelCatalogPort,
  ModelRouterPort,
  PinnedFactsRepo,
  PromptsPort,
  SettingsPort,
  StatusPort,
  ToolRegistryPort,
  TranscriptSnapshotStore,
  WorkspacePort,
} from "../ports";
import { render } from "../ports";
import { MemoryService, newId, type MemoryScope } from "./memory_service";
import { RlmEngine } from "./rlm/engine";
import { SessionCompactor } from "./session_compactor";
import type { TaskBoards } from "./task_boards";
import type { TaskBoard } from "../domain/task_board";
import { ToolExecutor } from "./tool_executor";
import { TurnRunner } from "./turn_runner";

export interface EditorContextInput {
  activeFile?: string;
  selection?: string;
  openFiles?: string[];
  mentions?: string[];
}

export interface OrchestratorDeps {
  settings: SettingsPort;
  updateSettings: (patch: Partial<Settings>) => void;
  router: ModelRouterPort;
  catalog: ModelCatalogPort;
  conversations: ConversationRepo;
  memoryStore: MemoryVectorStore;
  pinned: PinnedFactsRepo;
  snapshots: TranscriptSnapshotStore;
  edits: EditJournal;
  kernels: KernelFactory;
  tools: ToolRegistryPort;
  prompts: PromptsPort;
  workspace: WorkspacePort;
  clock: ClockPort;
  log: LoggerPort;
  status: StatusPort;
  broadcast: (ev: TurnEvent) => void;
  /** Reads a workspace file for @mentions (confined to the workspace). */
  readMention: (path: string) => Promise<string>;
  /** Per-chat task boards and wake-ups; without it the agent has no board and cannot schedule itself. */
  boards?: TaskBoards;
}

/** Application facade: every use case the driving adapters (IPC, tests, evals) call. */
export class Orchestrator {
  readonly runner: TurnRunner;
  readonly executor: ToolExecutor;
  readonly engine: RlmEngine;
  readonly memory: MemoryService;
  readonly compactor: SessionCompactor;

  constructor(private readonly d: OrchestratorDeps) {
    this.runner = new TurnRunner(d.broadcast);
    const summarizer = () => d.router.chat("summarizer");
    this.executor = new ToolExecutor({
      registry: d.tools,
      approvals: this.runner,
      settings: () => d.settings.get().tools,
      persistWorkspaceAllow: (toolId) => {
        const s = d.settings.get();
        if (!s.tools.workspaceAllow.includes(toolId)) d.updateSettings({ tools: { ...s.tools, workspaceAllow: [...s.tools.workspaceAllow, toolId] } });
      },
      prompts: d.prompts,
      checker: summarizer,
    });
    this.memory = new MemoryService({
      store: d.memoryStore,
      embedder: () => d.router.embedder(),
      chat: summarizer,
      prompts: d.prompts,
      clock: d.clock,
      log: d.log,
      status: d.status,
    });
    this.compactor = new SessionCompactor({ prompts: d.prompts, summarizer });
    this.engine = new RlmEngine({
      prompts: d.prompts,
      kernels: d.kernels,
      executor: this.executor,
      tools: () => d.tools.list(),
      agentModel: () => d.router.chat("agent"),
      summarizer,
      snapshots: d.snapshots,
      log: d.log,
      status: d.status,
    });
  }

  scope(chatId: string): MemoryScope {
    return { userId: this.d.settings.get().userId, sessionId: chatId, workspaceId: this.d.workspace.root() };
  }

  // ---------- chats ----------

  listChats(): Promise<ChatSummary[]> {
    return this.d.conversations.listChats();
  }

  createChat(title: string): Promise<ChatSummary> {
    return this.d.conversations.createChat(title);
  }

  async deleteChat(chatId: string): Promise<void> {
    this.disarm(chatId);
    await this.d.boards?.drop(chatId);
    await this.engine.dropChat(chatId);
    await this.d.memoryStore.deleteAll({ user_id: this.d.settings.get().userId, memory_type: "short_term", session_id: chatId });
    await this.d.conversations.deleteChat(chatId);
  }

  async messages(chatId: string): Promise<StoredMessage[]> {
    const turns = await this.d.conversations.getTurns(chatId);
    const out: StoredMessage[] = [];
    for (const t of turns) {
      out.push({ id: `${t.id}:u`, chatId, turnId: t.id, role: "user", text: t.userText, createdAt: t.createdAt, status: "done", origin: t.meta.origin ?? "user" });
      const events = t.status === "running" ? this.runner.bufferedEvents(t.id) : ((await this.d.conversations.getTurnEvents(t.id)) as TurnEvent[]);
      out.push({ id: `${t.id}:a`, chatId, turnId: t.id, role: "assistant", text: t.assistantText, createdAt: t.createdAt, status: t.status, events, feedback: t.meta.feedback ?? null });
    }
    return out;
  }

  async checkpoint(chatId: string): Promise<{ summary: string; throughIdx: number } | null> {
    return latestCheckpoint(await this.d.conversations.getTurns(chatId));
  }

  // ---------- turns ----------

  async send(args: { chatId: string; text: string; editorContext: EditorContextInput; toolGroups?: string[]; origin?: "user" | "wakeup" }): Promise<{ turnId: string }> {
    if (this.runner.isActive(args.chatId)) throw new Error("A turn is already running in this chat. Stop it first.");
    const prior = await this.d.conversations.getTurns(args.chatId);
    const turn = await this.d.conversations.appendTurn({
      id: newId("turn"),
      chatId: args.chatId,
      userText: args.text,
      assistantText: "",
      status: "running",
      createdAt: this.d.clock.now(),
      meta: args.origin === "wakeup" ? { origin: "wakeup" } : {},
    });
    const { signal, emit } = this.runner.begin(args.chatId, turn.id);
    void this.runTurn(turn, prior, args, signal, emit);
    if (prior.length === 0) void this.d.conversations.renameChat(args.chatId, clipChars(args.text.split("\n")[0] ?? "New chat", 60).replace(/\n\[truncated.*$/, "…"));
    return { turnId: turn.id };
  }

  stop(turnId: string): boolean {
    return this.runner.stop(turnId);
  }

  // ---------- scheduled wake-ups ----------

  private readonly timers = new Map<string, ReturnType<typeof setTimeout>>();
  private schedulerOn = false;

  /** Arms a timer for every pending wake-up, including ones that fell due while the IDE was closed. */
  async startScheduler(): Promise<void> {
    const boards = this.d.boards;
    if (!boards || this.schedulerOn) return;
    this.schedulerOn = true;
    boards.onChange((b) => this.arm(b));
    for (const b of await boards.pending()) this.arm(b);
  }

  stopScheduler(): void {
    for (const t of this.timers.values()) clearTimeout(t);
    this.timers.clear();
    this.schedulerOn = false;
  }

  async board(chatId: string): Promise<TaskBoard | null> {
    return this.d.boards ? this.d.boards.get(chatId) : null;
  }

  async cancelWakeup(chatId: string): Promise<void> {
    await this.d.boards?.cancelWakeup(chatId);
  }

  private disarm(chatId: string): void {
    const t = this.timers.get(chatId);
    if (t) clearTimeout(t);
    this.timers.delete(chatId);
  }

  private arm(b: TaskBoard, delayMs?: number): void {
    this.disarm(b.chatId);
    if (!this.schedulerOn || !b.wakeup) return;
    const ms = delayMs ?? Math.max(0, b.wakeup.at - this.d.clock.now());
    const t = setTimeout(() => void this.fire(b.chatId).catch((e) => this.d.log.error("wake-up failed", String(e))), Math.min(ms, 2 ** 31 - 1));
    this.timers.set(b.chatId, t);
  }

  private async fire(chatId: string): Promise<void> {
    this.timers.delete(chatId);
    const boards = this.d.boards!;
    const b = await boards.get(chatId);
    const now = this.d.clock.now();
    if (!b.wakeup) return;
    if (b.wakeup.at > now + 1000) return this.arm(b);
    if (this.runner.isActive(chatId)) return this.arm(b, 15000);
    const root = this.d.workspace.root();
    if (!root || (b.wakeup.workspace && root !== b.wakeup.workspace)) {
      this.d.log.info(`wake-up for chat ${chatId} waits for workspace ${b.wakeup.workspace ?? "(any)"}`);
      return this.arm(b, root ? 60000 : 10000);
    }
    if (!(await this.d.conversations.listChats()).some((c) => c.id === chatId)) {
      await boards.drop(chatId);
      return;
    }
    const limit = this.d.settings.get().rlm.maxAutoWakeups;
    if (b.autoWakeups >= limit) {
      await boards.cancelWakeup(chatId);
      await boards.log(chatId, `wake-up not started: ${limit} wake-ups in a row without a message from the user`);
      return;
    }
    const { board, fired } = await boards.takeWakeup(chatId);
    if (!fired) return;
    const lateMin = Math.round((now - fired.at) / 60000);
    const text = render(this.d.prompts.get("rlm/wakeup.md"), {
      reason: fired.reason,
      scheduled: new Date(fired.scheduledAt).toString(),
      now: new Date(now).toString(),
      timing: lateMin >= 2 ? `${lateMin} minutes later than planned (the IDE was closed or busy)` : "on time",
      count: String(board.autoWakeups),
    }).trim();
    try {
      await this.send({ chatId, text, editorContext: {}, origin: "wakeup" });
    } catch (e) {
      await boards.log(chatId, `wake-up could not start a turn: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  /** Schedules a wake-up when the board still has open steps and none is pending; returns its delay in seconds.
   * A turn that ran out of steps without any board is unfinished by definition, so the request becomes the open step. */
  private async continueAfterStepLimit(chatId: string, maxSteps: number, request: string, lastText: string): Promise<number | null> {
    const boards = this.d.boards!;
    let b = await boards.get(chatId);
    if (b.wakeup || b.autoWakeups >= this.d.settings.get().rlm.maxAutoWakeups) return null;
    if (!b.steps.length) {
      b = await boards.plan(chatId, clipChars(request.trim(), 300) || "Finish the request", ["Finish the request (replace this placeholder with a real plan via task.plan)"]);
      if (lastText.trim()) b = await boards.log(chatId, `last status before the step limit: ${clipChars(lastText.trim(), 400)}`);
    }
    const open = b.steps.filter((x) => x.status === "todo" || x.status === "doing");
    if (!open.length || b.wakeup) return null;
    const delay = this.d.settings.get().rlm.wakeupMinSec;
    const next = b.steps.findIndex((x) => x.status === "todo" || x.status === "doing");
    await boards.scheduleWakeup(chatId, delay, `The previous turn used its whole budget of ${maxSteps} steps before the plan was finished. Check running processes and their logs, update the board, then continue with step ${next}: ${open[0]!.text}`);
    await boards.log(chatId, `turn stopped at the ${maxSteps}-step limit with ${open.length} open steps; continuation scheduled`);
    return delay;
  }

  private async runTurn(
    turn: StoredTurn,
    prior: StoredTurn[],
    args: { chatId: string; text: string; editorContext: EditorContextInput; toolGroups?: string[]; origin?: "user" | "wakeup" },
    signal: AbortSignal,
    emit: (e: TurnEventBody) => void,
  ): Promise<void> {
    const s = this.d.settings.get();
    const modelId = this.d.router.modelIdFor("agent");
    const wakeup = args.origin === "wakeup";
    const meta: StoredTurn["meta"] = { model: modelId, ...(wakeup ? { origin: "wakeup" as const } : {}) };
    let partial = "";
    const capture = (e: TurnEventBody) => {
      if (e.type === "text_delta") partial += e.text;
      emit(e);
    };
    this.executor.beginToolTurn(turn.id);
    try {
      emit({ type: "turn_started", chatId: args.chatId, userText: args.text, model: modelId, origin: wakeup ? "wakeup" : "user" });
      const root = this.d.workspace.root();
      if (!root) throw new Error("Open a folder first: the agent works inside a workspace.");
      const boards = this.d.boards;
      if (boards) {
        boards.setTurnContext(args.chatId, { request: wakeup ? null : args.text, workspace: root });
        if (!wakeup) await boards.noteUserTurn(args.chatId);
      }

      // 1. model limits
      const limits = await this.d.catalog.limits(modelId);
      if (!limits.contextLength) throw new Error(`The context length of model '${modelId}' is unknown. Set it in Settings → Models before sending.`);
      const modelLimits = { contextLength: limits.contextLength, maxOutputTokens: limits.maxOutputTokens ?? Math.floor(limits.contextLength / 4) };

      // 2. session workspace + session compaction
      const cp0 = latestCheckpoint(prior);
      const verbatim = verbatimTurns(prior, cp0);
      emit({ type: "status", text: "Preparing context" });
      const userText = await this.expandMentions(args.text, args.editorContext.mentions ?? []);
      const comp = await this.compactor.maybeCompact({ limits: modelLimits, settings: s.compact, checkpoint: cp0, verbatim, newUserText: userText, signal });
      if (comp.compacted && comp.checkpoint) {
        meta.selfCompact = { summary: comp.checkpoint.summary, compactedBy: "session", throughIdx: comp.checkpoint.throughIdx };
        await this.d.conversations.updateTurn(turn.id, { meta });
        emit({ type: "compaction", checkpoint: { summary: comp.checkpoint.summary, foldedTurns: comp.foldedTurns, keptTurns: comp.kept.length, compactedBy: "session" } });
      }
      // 3. preflight blocks
      const scope = this.scope(args.chatId);
      const recall = await this.memory.recall(userText, scope, s.memory);
      emit({ type: "memory_recalled", items: recall.items, offline: recall.offline });
      const pinned = await this.pinnedBlock(args.chatId, args.text, prior.length === 0);
      const editor = await this.editorBlock(args.editorContext);

      // 4. tool selection
      const all = this.d.tools.list();
      const groups = args.toolGroups && args.toolGroups.length ? new Set(args.toolGroups) : null;
      const toolIds = new Set(all.filter((t) => !groups || groups.has(t.group)).map((t) => t.id));
      const subagentToolIds = new Set(all.filter((t) => !t.sideEffects && t.group !== "task").map((t) => t.id));

      // 5. engine
      const conversationText = () => [...verbatim.map((t) => `User: ${t.userText}\nAssistant: ${t.assistantText}`), `User: ${args.text}`].join("\n\n");
      const out = await this.engine.runTurn({
        chatId: args.chatId,
        turnId: turn.id,
        turnIdx: turn.idx,
        userText,
        verbatim: comp.kept,
        checkpoint: comp.checkpoint,
        blocks: { memory: recall.block, pinned, editor, clock: new Date(this.d.clock.now()).toString(), persona: s.persona },
        limits: modelLimits,
        rlm: s.rlm,
        compact: s.compact,
        toolIds,
        subagentToolIds,
        workspaceRoot: root,
        signal,
        emit: capture,
        taskBlock: boards && toolIds.has("task.plan") ? () => boards.render(args.chatId) : undefined,
        toolScope: (agentPath) => ({
          turnId: turn.id,
          chatId: args.chatId,
          ctx: { workspaceRoot: root, turnId: turn.id, chatId: args.chatId, signal },
          conversation: conversationText,
          onCall: (call) => emit({ type: "tool_call", call, agentPath }),
          onResult: (callId, r) => emit({ type: "tool_result", callId, ok: r.ok, result: r.ok ? clipResult(r.result) : r.error, durationMs: r.durationMs, deduped: r.deduped }),
          onEditApplied: (edit) => {
            void this.d.edits.record({ ...edit, turnId: turn.id, createdAt: this.d.clock.now() });
            emit({ type: "pending_edit", editId: edit.editId, path: edit.path, before: edit.before, after: edit.after });
          },
        }),
      });

      // 6. a planned job that ran out of steps carries on by itself instead of waiting for "continue"
      if (boards && out.stoppedAtLimit && !signal.aborted) {
        const delay = await this.continueAfterStepLimit(args.chatId, s.rlm.maxSteps, args.text, out.stoppedAtLimit.lastText);
        if (delay !== null)
          out.finalText = `${out.stoppedAtLimit.lastText}\n\n(Stopped after ${s.rlm.maxSteps} steps with open steps on the task board; the IDE continues in ${delay} s. Cancel the wake-up on the task card to stop.)`.trim();
      }

      // 7. persist
      meta.workLog = out.workLog;
      await this.d.conversations.updateTurn(turn.id, { assistantText: out.finalText, status: "done", meta, events: this.runner.bufferedEvents(turn.id) });
      emit({ type: "turn_done", finalText: out.finalText, usage: out.usage });
      // Facts are learned only from what the user wrote, never from the IDE's own wake-up messages.
      void this.memory.saveTurn(scope, s.memory, args.text, out.finalText).then(() => (wakeup ? undefined : this.memory.learnFacts(scope, s.memory, args.text, out.finalText)));
    } catch (e) {
      const aborted = signal.aborted || (e instanceof Error && e.name === "AbortError");
      if (aborted) {
        emit({ type: "turn_cancelled", partialText: partial });
        await this.d.conversations.updateTurn(turn.id, { assistantText: partial, status: "cancelled", meta, events: this.runner.bufferedEvents(turn.id) });
      } else {
        const message = e instanceof Error ? e.message : String(e);
        this.d.log.error("turn failed", message);
        emit({ type: "turn_error", message });
        await this.d.conversations.updateTurn(turn.id, { assistantText: partial, status: "error", meta, events: this.runner.bufferedEvents(turn.id) });
      }
    } finally {
      this.executor.endToolTurn(turn.id);
      this.d.status.kernel("idle");
    }
  }

  private async expandMentions(text: string, mentions: string[]): Promise<string> {
    if (!mentions.length) return text;
    const parts: string[] = [text];
    for (const m of mentions.slice(0, 8)) {
      try {
        const body = await this.d.readMention(m);
        parts.push(`\n[attached file: ${m}]\n\`\`\`\n${clipChars(body, 20000)}\n\`\`\``);
      } catch (e) {
        parts.push(`\n[attached file: ${m} could not be read: ${String(e)}]`);
      }
    }
    return parts.join("\n");
  }

  /** Pinned facts are deterministic: the first message of a chat seeds the goal; users edit the rest. */
  private async pinnedBlock(chatId: string, text: string, firstTurn: boolean): Promise<string> {
    let facts = await this.d.pinned.list(chatId);
    if (firstTurn && !facts.some((f) => f.kind === "goal")) {
      await this.d.pinned.upsert({ chatId, kind: "goal", text: clipChars(text.trim(), 400) });
      facts = await this.d.pinned.list(chatId);
    }
    if (!facts.length) return "";
    const label = { goal: "Goal", constraint: "Constraint", decision: "Decision" } as const;
    return facts.map((f) => `- ${label[f.kind]}: ${f.text}`).join("\n");
  }

  private async editorBlock(ctx: EditorContextInput): Promise<string> {
    const git = await this.d.workspace.gitSummary().catch(() => "");
    return render(this.d.prompts.get("editor_context.md"), {
      workspace: this.d.workspace.root() ?? "(none)",
      active_file: ctx.activeFile ?? "(none)",
      open_files: (ctx.openFiles ?? []).slice(0, 12).join(", ") || "(none)",
      selection: ctx.selection ? clipChars(ctx.selection, 4000) : "(none)",
      git: git || "(not a git repository)",
    }).trim();
  }
}

function clipResult(r: unknown): unknown {
  const s = typeof r === "string" ? r : JSON.stringify(r);
  if (s && s.length > 4000) return clipChars(s, 4000);
  return r;
}
