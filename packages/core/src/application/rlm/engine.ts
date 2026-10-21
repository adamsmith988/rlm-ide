import type { RlmSettings, CompactSettings, TurnEventBody } from "@rlm/contracts";
import type { ModelLimits } from "../../domain/budget";
import { computeBudget, narrowLimits } from "../../domain/budget";
import type { Checkpoint, StoredTurn } from "../../domain/conversation";
import { estimateTokens, clipChars, charsForTokens } from "../../domain/tokens";
import type { ToolDescriptor } from "../../domain/tools";
import {
  ContextOverflowError,
  ModelWindowError,
  render,
  type ChatMessage,
  type ChatModelPort,
  type KernelFactory,
  type KernelSession,
  type PromptsPort,
  type TranscriptEntry,
  type TranscriptSnapshotStore,
  type LoggerPort,
  type StatusPort,
} from "../../ports";
import { summarizeInChunks } from "../session_compactor";
import { renderToolSurface } from "../tool_surface";
import type { ToolExecutor, TurnToolScope } from "../tool_executor";
import { newId } from "../memory_service";
import { CellFenceFilter, extractCells, retriedCell, retriedCellText, sameErrorHint, stripCells, stripThinking, syntaxHint, timeoutHint, unclosedCellAt, type FailedCell } from "./cells";
import { buildMessages, estimate, toUnits, truncateOldOutputs, type HostBlocks } from "./transcript";
import { ReplyGuard, type CutReason } from "./guard";

export interface RlmTurnInput {
  chatId: string;
  turnId: string;
  turnIdx: number;
  userText: string;
  /** Stored turns after the checkpoint (finished ones only) and the checkpoint itself. */
  verbatim: StoredTurn[];
  checkpoint: Checkpoint | null;
  blocks: HostBlocks;
  limits: ModelLimits;
  rlm: RlmSettings;
  compact: CompactSettings;
  toolIds: Set<string>;
  subagentToolIds: Set<string>;
  workspaceRoot: string;
  signal: AbortSignal;
  emit: (e: TurnEventBody) => void;
  toolScope: (agentPath: string[]) => Omit<TurnToolScope, "toolIds">;
  /** Renders the chat's task board; called before every root model call so updates show at once. */
  taskBlock?: () => Promise<string>;
}

export interface RlmTurnOutput {
  finalText: string;
  workLog: string;
  steps: number;
  usage: { inputTokens: number; outputTokens: number };
  /** Set when the turn ended on the step limit: the last reply, without the stop note. */
  stoppedAtLimit?: { lastText: string };
}

interface TurnBinding {
  input: RlmTurnInput;
  agentPath: string[];
  toolIds: Set<string>;
  children: Semaphore;
}

interface Session {
  chatId: string;
  entries: TranscriptEntry[];
  kernel: KernelSession | null;
  /** Folder the kernel was started in; a kernel never serves another project. */
  kernelRoot?: string;
  lastTurnIdx: number;
  depth: number;
  /** The kernel outlives turns; host requests are routed through the turn currently running. */
  binding?: TurnBinding;
  /** Tool notices raised while the current cell runs. */
  notices?: string[];
  /** The last cell that failed, to spot fixes made by re-sending the whole cell. */
  failedCell?: FailedCell;
  /** The model was already told not to re-send long failing cells; further copies are not run. */
  retryWarned?: boolean;
  /** Server-counted prompt tokens per estimated token; the character estimate undercounts some content. */
  tokenScale?: number;
}

const MAX_TOKEN_SCALE = 3;

/** Prompt tokens as the server will count them, as far as earlier replies tell. */
function measure(session: Session, messages: ChatMessage[]): number {
  return Math.ceil(estimate(messages) * (session.tokenScale ?? 1));
}

/** Learns from a server count; grows at once, shrinks slowly, so one light prompt cannot undo a heavy one. */
function calibrate(session: Session, estimated: number, counted: number): void {
  if (estimated < 200 || counted <= 0) return;
  const ratio = Math.min(MAX_TOKEN_SCALE, Math.max(1, (counted / estimated) * 1.03));
  const prev = session.tokenScale ?? 1;
  session.tokenScale = ratio > prev ? ratio : prev * 0.8 + ratio * 0.2;
}

/** "...your prompt contains at least 24577 input tokens..." and similar provider phrasings. */
function promptTokensFromOverflow(message: string): number | null {
  const m = /(\d{3,})\s+input tokens|prompt (?:contains|has|is)(?: at least)? (\d{3,})|(\d{3,}) tokens (?:in|for) the (?:prompt|input)/i.exec(message);
  const n = m ? Number(m[1] ?? m[2] ?? m[3]) : NaN;
  return Number.isFinite(n) ? n : null;
}

export interface RlmEngineDeps {
  prompts: PromptsPort;
  kernels: KernelFactory;
  executor: ToolExecutor;
  tools: () => ToolDescriptor[];
  agentModel: () => Promise<ChatModelPort>;
  summarizer: () => Promise<ChatModelPort>;
  snapshots: TranscriptSnapshotStore;
  log: LoggerPort;
  status: StatusPort;
}

/** Where in-turn compaction fires and the largest prompt that still leaves room for an answer. */
function promptLimits(input: RlmTurnInput): { trigger: number; inputLimit: number } {
  const inputLimit = input.limits.contextLength - Math.min(input.limits.maxOutputTokens, 1024);
  return { trigger: Math.min(Math.floor(input.limits.contextLength * input.rlm.compactTriggerRatio), inputLimit), inputLimit };
}

class Semaphore {
  private active = 0;
  private readonly waiters: Array<() => void> = [];
  constructor(private readonly max: number) {}
  async acquire(): Promise<() => void> {
    if (this.active >= this.max) await new Promise<void>((r) => this.waiters.push(r));
    this.active++;
    return () => {
      this.active--;
      this.waiters.shift()?.();
    };
  }
}

export class RlmEngine {
  private readonly sessions = new Map<string, Session>();

  constructor(private readonly deps: RlmEngineDeps) {}

  /** Memory ladder: warm session -> disk snapshot -> reseed from stored conversation. */
  private async acquireSession(input: RlmTurnInput): Promise<{ session: Session; restored: "warm" | "snapshot" | "reseed" }> {
    const lastStored = input.verbatim.length ? input.verbatim[input.verbatim.length - 1]!.idx : input.checkpoint?.throughIdx ?? -1;
    const warm = this.sessions.get(input.chatId);
    if (warm && warm.lastTurnIdx === lastStored) return { session: warm, restored: "warm" };

    const snap = await this.deps.snapshots.load(input.chatId).catch(() => null);
    if (snap && snap.throughTurnIdx === lastStored && !warm) {
      const session: Session = { chatId: input.chatId, entries: snap.entries, kernel: null, lastTurnIdx: lastStored, depth: 0 };
      session.entries.push({ kind: "note", turnIdx: input.turnIdx, text: this.deps.prompts.get("rlm/restart_note.md").trim() });
      this.sessions.set(input.chatId, session);
      return { session, restored: "snapshot" };
    }

    const entries: TranscriptEntry[] = [];
    if (input.checkpoint) entries.push({ kind: "checkpoint", text: input.checkpoint.summary });
    for (const t of input.verbatim) {
      entries.push({ kind: "user", text: t.userText, turnIdx: t.idx });
      if (t.assistantText) entries.push({ kind: "assistant", text: t.assistantText, turnIdx: t.idx });
    }
    if (input.verbatim.length || input.checkpoint) {
      entries.push({ kind: "note", turnIdx: input.turnIdx, text: this.deps.prompts.get("rlm/restart_note.md").trim() });
    }
    const session: Session = { chatId: input.chatId, entries, kernel: warm?.kernel ?? null, kernelRoot: warm?.kernelRoot, lastTurnIdx: lastStored, depth: 0 };
    this.sessions.set(input.chatId, session);
    return { session, restored: "reseed" };
  }

  async dropChat(chatId: string): Promise<void> {
    const s = this.sessions.get(chatId);
    this.sessions.delete(chatId);
    await s?.kernel?.shutdown().catch(() => undefined);
    await this.deps.snapshots.delete(chatId).catch(() => undefined);
  }

  async shutdownIdleKernels(busy: (chatId: string) => boolean): Promise<void> {
    for (const s of this.sessions.values()) {
      if (!s.kernel || busy(s.chatId)) continue;
      const k = s.kernel;
      s.kernel = null;
      await k.shutdown().catch(() => undefined);
    }
  }

  async shutdownAll(): Promise<void> {
    for (const s of this.sessions.values()) await s.kernel?.shutdown().catch(() => undefined);
    this.sessions.clear();
  }

  async runTurn(input: RlmTurnInput): Promise<RlmTurnOutput> {
    const { session, restored } = await this.acquireSession(input);
    if (restored !== "warm") input.emit({ type: "status", text: `RLM session restored (${restored})` });
    const children = new Semaphore(input.rlm.maxConcurrentChildren);
    session.entries.push({ kind: "user", text: await this.userEntryText(session, input, children), turnIdx: input.turnIdx });
    try {
      const out = await this.loop(session, input, [], input.toolIds, children);
      session.lastTurnIdx = input.turnIdx;
      return out;
    } finally {
      if (input.signal.aborted) {
        session.entries.push({ kind: "note", turnIdx: input.turnIdx, text: "The user stopped the previous turn. Partial work above may be incomplete." });
        session.lastTurnIdx = input.turnIdx;
      }
      await this.deps.snapshots
        .save({ chatId: session.chatId, throughTurnIdx: session.lastTurnIdx, entries: session.entries, savedAt: Date.now() })
        .catch((e) => this.deps.log.warn("snapshot save failed", String(e)));
    }
  }

  /** A message too large for a comfortable prompt goes into the REPL as `user_message`; the transcript gets its
   * start and end and how to work with it. A model handed a document that fills its window mostly reasons in
   * circles; slicing and searching it in code is the RLM way through. */
  private async userEntryText(session: Session, input: RlmTurnInput, children: Semaphore): Promise<string> {
    const text = input.userText;
    const tokens = estimateTokens(text);
    if (tokens <= input.limits.contextLength * input.rlm.largeInputShare) return text;
    try {
      const kernel = await this.ensureKernel(session, input, [], input.toolIds, children);
      const b64 = Buffer.from(text, "utf8").toString("base64");
      const r = await kernel.execute(`import base64 as _b64\nuser_message = _b64.b64decode("${b64}").decode("utf-8")\ndel _b64`, () => undefined, 60_000);
      if (!r.ok) throw new Error(r.error ?? "could not load the message into the REPL");
    } catch (e) {
      this.deps.log.warn("large message stays in the prompt", String(e));
      return text;
    }
    const preview = Math.max(500, Math.min(4000, charsForTokens(input.limits.contextLength * 0.04)));
    const head = text.slice(0, preview);
    const tail = text.slice(-Math.floor(preview / 3));
    input.emit({ type: "status", text: `Your message (~${tokens} tokens) was handed to the REPL as user_message; the agent reads it in parts` });
    return render(this.deps.prompts.get("rlm/large_input.md"), {
      chars: String(text.length),
      tokens: String(tokens),
      head,
      tail,
      omitted: String(Math.max(0, text.length - head.length - tail.length)),
    }).trim();
  }

  private systemPrompt(toolIds: Set<string>, depth: number, rlm: RlmSettings, compact = false): string {
    const tools = this.deps.tools().filter((t) => toolIds.has(t.id));
    return render(this.deps.prompts.get(depth === 0 ? "rlm/system.md" : "rlm/subagent.md"), {
      tool_surface: renderToolSurface(tools, compact),
      max_depth: String(rlm.maxSubagentDepth),
      can_spawn: depth < rlm.maxSubagentDepth ? "yes" : "no",
    });
  }

  private async ensureKernel(session: Session, input: RlmTurnInput, agentPath: string[], toolIds: Set<string>, children: Semaphore): Promise<KernelSession> {
    session.binding = { input, agentPath, toolIds, children };
    if (session.kernel?.isAlive() && session.kernelRoot === input.workspaceRoot) return session.kernel;
    await session.kernel?.shutdown().catch(() => undefined);
    const tools = this.deps.tools().filter((t) => toolIds.has(t.id));
    session.kernelRoot = input.workspaceRoot;
    session.kernel = await this.deps.kernels.create({
      cwd: input.workspaceRoot,
      tools,
      outputLimitChars: input.rlm.cellOutputLimitChars,
      onHostRequest: async (type, payload) => {
        const b = session.binding;
        if (!b) return { ok: false, error: "no active turn" };
        const { input, agentPath, toolIds, children } = b;
        if (type === "tool") {
          const base = input.toolScope(agentPath);
          const scope = { ...base, toolIds, ctx: { ...base.ctx, notice: (t: string) => (session.notices ??= []).push(t) } };
          const r = await this.deps.executor.execute(String(payload.tool_id ?? ""), payload.args ?? {}, scope);
          if (!r.ok) return { ok: false, error: r.error };
          return { ok: true, result: r.result };
        }
        if (type === "spawn") {
          if (session.depth >= input.rlm.maxSubagentDepth) return { ok: false, error: `sub-agent depth limit (${input.rlm.maxSubagentDepth}) reached` };
          const tasks = Array.isArray(payload.tasks) ? payload.tasks.map(String) : [];
          const requested = Array.isArray(payload.tool_ids) ? new Set(payload.tool_ids.map(String)) : null;
          const childTools = new Set([...input.subagentToolIds].filter((t) => toolIds.has(t) && (!requested || requested.has(t))));
          const results = await Promise.all(tasks.map((task) => this.runChild(session, input, agentPath, task, childTools, children)));
          return { ok: true, result: results };
        }
        return { ok: false, error: `unknown host request '${type}'` };
      },
    });
    return session.kernel;
  }

  private async runChild(parent: Session, input: RlmTurnInput, parentPath: string[], task: string, toolIds: Set<string>, children: Semaphore): Promise<string> {
    const release = await children.acquire();
    const agentId = newId("agent");
    const path = [...parentPath, agentId];
    input.emit({ type: "subagent_started", agentId, parentPath, task });
    const child: Session = { chatId: `${parent.chatId}/${agentId}`, entries: [{ kind: "user", text: task, turnIdx: input.turnIdx }], kernel: null, lastTurnIdx: input.turnIdx, depth: parent.depth + 1 };
    try {
      const out = await this.loop(child, input, path, toolIds, children);
      input.emit({ type: "subagent_done", agentId, ok: true, result: clipChars(out.finalText, 4000) });
      return out.finalText;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      input.emit({ type: "subagent_done", agentId, ok: false, result: msg });
      return `sub-agent failed: ${msg}`;
    } finally {
      await child.kernel?.shutdown().catch(() => undefined);
      release();
    }
  }

  private async loop(session: Session, turnInput: RlmTurnInput, agentPath: string[], toolIds: Set<string>, children: Semaphore): Promise<RlmTurnOutput> {
    const model = await this.deps.agentModel();
    let input = turnInput;
    const fullSys = this.systemPrompt(toolIds, session.depth, input.rlm);
    let compactSys: string | null = null;
    const resume = this.deps.prompts.get("rlm/compact_resume.md");
    const isRoot = agentPath.length === 0;
    const blocks: HostBlocks = isRoot ? { ...input.blocks } : { memory: "", pinned: "", editor: input.blocks.editor, clock: input.blocks.clock, persona: "" };
    const usage = { inputTokens: 0, outputTokens: 0 };
    const workLog: string[] = [];
    let lastText = "";
    let overflows = 0;
    /** Replies cut off in a row (reasoning too long, or repeating); the next request runs without reasoning. */
    let cuts = 0;
    let calmNext: CutReason | null = null;
    /** Cells run this turn and their last output, to spot a model running the same thing over and over. */
    const ran = new Map<string, { times: number; out: string }>();
    let edited = false;
    let reviewed = false;
    const editedPages = new Set<string>();
    const editedFiles = new Set<string>();

    for (let step = 0; step < input.rlm.maxSteps; step++) {
      if (input.signal.aborted) throw new DOMException("stopped", "AbortError");
      // A failover chain may be serving from a backup with a smaller window: the transcript is compacted to fit it
      // (summaries, task board and memory carry everything over) instead of being sent where it cannot fit.
      const limits = narrowLimits(turnInput.limits, model.activeLimits?.());
      if (limits.contextLength !== input.limits.contextLength) {
        this.deps.log.warn("agent context window changed", { from: input.limits.contextLength, to: limits.contextLength, step: step + 1 });
        if (isRoot && limits.contextLength < input.limits.contextLength) input.emit({ type: "status", text: `Fitting the conversation into the backup model's ${limits.contextLength}-token window` });
      }
      input = limits === turnInput.limits ? turnInput : { ...turnInput, limits };
      if (isRoot && input.taskBlock) blocks.task = await input.taskBlock().catch(() => blocks.task ?? "");
      const sys = estimateTokens(fullSys) > limits.contextLength * input.rlm.toolDocsMaxShare ? (compactSys ??= this.systemPrompt(toolIds, session.depth, input.rlm, true)) : fullSys;
      let messages = buildMessages(sys, resume, blocks, session.entries);
      messages = await this.fitBudget(session, input, sys, resume, blocks, messages, isRoot);
      const estimated = estimate(messages);
      const inputTokens = measure(session, messages);
      const maxTokens = Math.max(64, Math.min(input.limits.maxOutputTokens, input.limits.contextLength - inputTokens - 128));
      let counted = 0;
      if (isRoot) {
        const p = promptLimits(input);
        input.emit({ type: "context_usage", usedTokens: inputTokens, triggerTokens: p.trigger, inputLimit: p.inputLimit, contextLength: input.limits.contextLength });
      }

      let raw = "";
      const filter = new CellFenceFilter();
      const guard = new ReplyGuard({
        thinkingMaxChars: charsForTokens(Math.max(256, Math.min(input.rlm.thinkingMaxTokens, Math.floor(maxTokens * input.rlm.thinkingMaxShare)))),
        thinkingMaxMs: input.rlm.thinkingMaxSec * 1000,
        repeatLimit: input.rlm.repeatLimit,
      });
      let cutBy: CutReason | null = null;
      const stopper = new AbortController();
      const stopCut = () => stopper.abort();
      input.signal.addEventListener("abort", stopCut, { once: true });
      const enableThinking = calmNext ? false : undefined;
      const temperature = calmNext === "repeating" ? 0.7 : 0.3;
      calmNext = null;
      try {
        const started = Date.now();
        let outChars = 0;
        let lastProgress = 0;
        let phase: "waiting" | "thinking" | "writing" | "writing_code" = "waiting";
        const progress = (force = false) => {
          if (!isRoot) return;
          const now = Date.now();
          if (!force && now - lastProgress < 400) return;
          lastProgress = now;
          const outputTokens = Math.round(outChars / 4);
          const secs = Math.max(0.001, (now - started) / 1000);
          input.emit({ type: "generation", phase, outputTokens, tokPerSec: Math.round((outputTokens / secs) * 10) / 10, step: step + 1 });
          const draft = filter.draft();
          if (draft !== null) {
            const lines = draft.split("\n");
            input.emit({ type: "cell_draft", lines: lines.length, chars: draft.length, tail: lines.slice(-14).join("\n"), agentPath });
          }
        };
        progress(true);
        for await (const c of model.stream({ messages, maxTokens, signal: stopper.signal, temperature, enableThinking })) {
          if (c.type === "thinking") {
            outChars += c.text.length;
            phase = "thinking";
            if (isRoot) input.emit({ type: "thinking_delta", text: c.text });
            progress();
            if ((cutBy = guard.push("thinking", c.text))) break;
          } else if (c.type === "text") {
            raw += c.text;
            outChars += c.text.length;
            const visible = filter.push(c.text);
            phase = filter.draft() !== null ? "writing_code" : "writing";
            if (visible && isRoot) input.emit({ type: "text_delta", text: visible });
            progress();
            if ((cutBy = guard.push("text", c.text))) break;
          } else if (c.type === "usage") {
            counted += c.inputTokens;
            usage.inputTokens += c.inputTokens;
            usage.outputTokens += c.outputTokens;
          } else if (c.type === "switch") {
            this.deps.log.warn("agent model switched", { from: c.from, to: c.to, reason: c.reason, resumed: c.resumed, step: step + 1 });
            if (isRoot) input.emit({ type: "model_switch", from: c.from, to: c.to, reason: c.reason, resumed: c.resumed });
            else input.emit({ type: "status", text: `Sub-agent switched from ${c.from} to ${c.to} (${c.reason})` });
          } else if (c.type === "notice") {
            input.emit({ type: "status", text: c.text });
          }
        }
        const rest = filter.flush();
        if (rest && isRoot && !cutBy) input.emit({ type: "text_delta", text: rest });
        calibrate(session, estimated, counted);
        overflows = 0;
      } catch (e) {
        if (e instanceof ModelWindowError && overflows < 3) {
          overflows++;
          this.deps.log.warn("only smaller-window models are available; compacting for them", { window: e.contextLength, measured: inputTokens });
          step--;
          continue;
        }
        if (e instanceof ContextOverflowError && overflows < 3) {
          overflows++;
          const said = promptTokensFromOverflow(e.message);
          calibrate(session, estimated, said ?? Math.ceil(inputTokens * 1.25));
          this.deps.log.warn("prompt over the context window; recalibrated and compacting", { estimated, measured: inputTokens, serverCount: said, scale: session.tokenScale });
          await this.compactTranscript(session, input, isRoot, true);
          step--;
          continue;
        }
        // Closing a reply we cut off can surface as an abort from the transport.
        if (!(cutBy && !input.signal.aborted)) throw e;
      } finally {
        input.signal.removeEventListener("abort", stopCut);
        stopper.abort();
      }
      if (!usage.outputTokens) usage.outputTokens += estimateTokens(raw);
      if (!usage.inputTokens) usage.inputTokens += inputTokens;

      if (cutBy) {
        cuts++;
        const tokens = Math.round(guard.thinkingSoFar / 4);
        this.deps.log.warn("model reply cut off", { reason: cutBy, step: step + 1, thinkingTokens: tokens, cutsInARow: cuts });
        if (cuts > input.rlm.maxCutsInARow) {
          const why = cutBy === "repeating" ? "kept repeating itself" : "kept reasoning without acting";
          const advice = `The model ${why} ${cuts} times in a row, so this turn was stopped. A larger model, or a shorter, more specific request, usually helps.`;
          return { finalText: lastText ? `${lastText}\n\n(${advice})` : advice, workLog: workLog.join("\n"), steps: step + 1, usage };
        }
        if (isRoot) input.emit({ type: "status", text: cutBy === "repeating" ? "The model started repeating itself; asking it to take a different step" : `The model reasoned for ~${tokens} tokens without acting; asking it to act` });
        // Its partial reply never ran; the note alone tells it what happened.
        session.entries.push({
          kind: "note",
          text: render(this.deps.prompts.get(cutBy === "repeating" ? "rlm/cut_repeating.md" : "rlm/cut_thinking.md"), { tokens: String(tokens) }).trim(),
          turnIdx: input.turnIdx,
        });
        calmNext = cutBy;
        continue;
      }
      cuts = 0;

      let text = stripThinking(raw);
      const cutAt = unclosedCellAt(text);
      if (cutAt >= 0) {
        // Keep only the head of the cut-off cell: it never ran and would waste the window.
        const cut = text.slice(cutAt);
        text = `${text.slice(0, cutAt)}${clipChars(cut, 600)}\n[cell cut off at the output limit after ${cut.length} chars]`;
      }
      session.entries.push({ kind: "assistant", text, turnIdx: input.turnIdx });
      const cells = extractCells(text);
      const prose = stripCells(cutAt >= 0 ? text.slice(0, cutAt) : text);
      if (prose) lastText = prose;
      const cutNote: TranscriptEntry | null =
        cutAt < 0
          ? null
          : {
              kind: "note",
              text: `Your last reply hit the output limit (${maxTokens} tokens) before the \`\`\`repl cell was closed, so that cell did NOT run${cells.length ? " (the complete cells before it did)" : ""}. Split the work: write at most one file per cell, keep each cell well under the limit, and continue from where you stopped.`,
              turnIdx: input.turnIdx,
            };
      if (cutNote && isRoot) input.emit({ type: "status", text: "Reply hit the output limit inside a code cell; continuing in smaller steps" });
      if (cells.length === 0) {
        if (cutNote) {
          session.entries.push(cutNote);
          continue;
        }
        if (isRoot && edited && !reviewed && input.rlm.selfReview && step + 1 < input.rlm.maxSteps) {
          reviewed = true;
          input.emit({ type: "status", text: "Checking the work against the request" });
          const pages = [...editedPages].slice(0, 4);
          const webStep =
            pages.length && toolIds.has("web.check")
              ? `\nOpen ${pages.map((p) => `\`${p}\``).join(", ")} with \`web.check(path=..., click=[...])\`, clicking every main control (buttons, tiles, list items, play/next). Fix every error, and every click that leaves the page unchanged when it should do something, including on first load before anything is selected.\n`
              : "";
          const files = [...editedFiles].slice(0, 20);
          const checks =
            webStep +
            (blocks.memory
              ? `\nAlso check every file you wrote against the user's remembered preferences (file names, formatting, quotes, indentation) and rename or fix any that do not match.${files.length ? `\nFiles you wrote: ${files.map((f) => `\`${f}\``).join(", ")}` : ""}\n${blocks.memory}\n`
              : "");
          session.entries.push({ kind: "note", text: render(this.deps.prompts.get("rlm/self_review.md"), { request: clipChars(input.userText, 4000), checks }), turnIdx: input.turnIdx });
          continue;
        }
        return { finalText: prose || lastText, workLog: workLog.join("\n"), steps: step + 1, usage };
      }
      if (cells.some((c) => /\bfs\.(write_file|edit_file)\s*\(/.test(c))) edited = true;
      for (const c of cells) {
        for (const m of c.matchAll(/\bfs\.(?:write_file|edit_file)\s*\(\s*(?:path\s*=\s*)?["']([^"']+)["']/g)) {
          editedFiles.add(m[1]!);
          if (/\.html?$/i.test(m[1]!)) editedPages.add(m[1]!);
        }
      }

      const kernel = await this.ensureKernel(session, input, agentPath, toolIds, children);
      const outputs: string[] = [];
      for (const code of cells) {
        if (input.signal.aborted) break;
        const cellId = newId("cell");
        input.emit({ type: "code_cell", cellId, code, agentPath });
        const retry = retriedCell(session.failedCell, code);
        if (retry && session.retryWarned) {
          const text = retriedCellText(retry, true);
          input.emit({ type: "cell_output", cellId, stream: "stderr", text: `${text}\n` });
          input.emit({ type: "cell_done", cellId, ok: false, error: text, durationMs: 0 });
          outputs.push(text);
          workLog.push(`cell: ${clipChars(code.split("\n").slice(0, 3).join(" | "), 160)} -> not run (re-sent failing cell)`);
          break;
        }
        const seen = ran.get(code.trim());
        if (seen && seen.times >= input.rlm.maxIdenticalCells) {
          const text = `[not run] This exact cell already ran ${seen.times} times in this turn with the same output each time; running it again shows nothing new. Use that output and take a different step, or answer.`;
          input.emit({ type: "cell_output", cellId, stream: "stderr", text: `${text}\n` });
          input.emit({ type: "cell_done", cellId, ok: false, error: text, durationMs: 0 });
          outputs.push(text);
          workLog.push(`cell: ${clipChars(code.split("\n").slice(0, 3).join(" | "), 160)} -> not run (identical repeat)`);
          break;
        }
        this.deps.status.kernel("busy");
        const started = Date.now();
        const onAbort = () => void kernel.interrupt();
        input.signal.addEventListener("abort", onAbort, { once: true });
        const r = await kernel.execute(code, (o) => input.emit({ type: "cell_output", cellId, stream: o.stream, text: o.text }), input.rlm.cellTimeoutSec * 1000);
        input.signal.removeEventListener("abort", onAbort);
        this.deps.status.kernel("idle");
        const notices = session.notices?.splice(0) ?? [];
        const noticeText = notices.length ? `[IDE]\n${notices.join("\n")}` : "";
        if (noticeText) input.emit({ type: "cell_output", cellId, stream: "stderr", text: `${noticeText}\n` });
        input.emit({ type: "cell_done", cellId, ok: r.ok, error: r.error ?? undefined, durationMs: Date.now() - started });
        // A cell that did not parse never ran; sending it again corrected is the only way to fix it.
        const didNotParse = !r.ok && /^SyntaxError:/.test(r.error ?? "");
        const retryNote = retry && !didNotParse ? retriedCellText(retry, false) : "";
        if (retryNote) session.retryWarned = true;
        const repeated = !r.ok && r.error && session.failedCell ? sameErrorHint(session.failedCell.error, r.error) : "";
        if (r.ok) session.failedCell = undefined;
        else if (r.error && !didNotParse && !r.interrupted && !r.timedOut) session.failedCell = { code, error: r.error };
        const parts = [r.stdout, r.stderr ? `[stderr]\n${r.stderr}` : "", r.result ? `[result] ${r.result}` : "", r.error ? `[error]\n${r.error}` : "", r.error ? syntaxHint(code, r.error) : "", repeated, retryNote, noticeText, r.timedOut ? timeoutHint(input.rlm.cellTimeoutSec) : ""].filter(Boolean);
        let out = clipChars(parts.join("\n").trim() || "(no output)", Math.min(input.rlm.cellOutputLimitChars, charsForTokens(input.limits.contextLength * input.rlm.cellOutputMaxShare)));
        const times = seen && seen.out === out ? seen.times + 1 : 1;
        ran.set(code.trim(), { times, out });
        if (times > 1) out += `\n[hint] You already ran this exact cell ${times - 1} time(s) in this turn and got this same output. Running it again will not tell you more: use it and move on.`;
        outputs.push(out);
        workLog.push(`cell: ${clipChars(code.split("\n").slice(0, 3).join(" | "), 160)} -> ${clipChars(out.replace(/\n/g, " "), 200)}`);
        if (!r.ok && (r.interrupted || r.timedOut)) break;
      }
      session.entries.push({ kind: "repl", text: outputs.join("\n---\n"), turnIdx: input.turnIdx });
      if (cutNote) session.entries.push(cutNote);
      const left = input.rlm.maxSteps - step - 1;
      if (isRoot && input.taskBlock && !blocks.task && step + 1 === Math.max(6, Math.floor(input.rlm.maxSteps / 6)) && left > input.rlm.stepWarnRemaining)
        session.entries.push({
          kind: "note",
          text: `This job has taken ${step + 1} steps and has no task board yet. If it is not almost done, write the plan now with task.plan(goal=..., steps=[...]) and mark what is already done with task.step: older steps get compacted, and the board is how this job survives the step limit and continues in later turns.`,
          turnIdx: input.turnIdx,
        });
      if (left > 0 && left === input.rlm.stepWarnRemaining)
        session.entries.push({
          kind: "note",
          text: input.taskBlock
            ? blocks.task
              ? `${left} steps are left in this turn. Leave long jobs running in the background and bring the task board up to date now (task.step, task.log with pids, log files and results): if steps are still open when this turn ends, the IDE starts a new turn with your board to continue.`
              : `${left} steps are left in this turn and you have no task board. If the job is not finished, record it now: task.plan(goal=..., steps=[...]) with the steps done so far marked done (task.step) and the remaining ones open, and task.log the pids, files and results the next steps need. Open steps are continued automatically in a new turn; without a board the work stops here.`
            : `${left} steps are left in this turn. Leave long jobs running in the background, make sure they are resumable, and use the last steps to tell the user where things stand and how to continue.`,
          turnIdx: input.turnIdx,
        });
    }
    const final = `${lastText}\n\n(Stopped after ${input.rlm.maxSteps} steps; ask me to continue.)`.trim();
    return { finalText: final, workLog: workLog.join("\n"), steps: input.rlm.maxSteps, usage, stoppedAtLimit: { lastText } };
  }

  /** context_budget: truncate old REPL output, then compact, then fail readably. */
  private async fitBudget(session: Session, input: RlmTurnInput, sys: string, resume: string, blocks: HostBlocks, messages: ChatMessage[], isRoot: boolean): Promise<ChatMessage[]> {
    const { trigger, inputLimit } = promptLimits(input);
    const limit = (n: number) => n >= inputLimit;
    let msgs = messages;
    let tokens = measure(session, msgs);
    if (tokens < trigger && !limit(tokens)) return msgs;

    const t = truncateOldOutputs(session.entries, input.rlm.oldOutputKeepChars, 2);
    if (t.changed) {
      session.entries = t.entries;
      msgs = buildMessages(sys, resume, blocks, session.entries);
      tokens = measure(session, msgs);
      if (tokens < trigger && !limit(tokens)) return msgs;
    }
    await this.compactTranscript(session, input, isRoot, false);
    msgs = buildMessages(sys, resume, blocks, session.entries);
    tokens = measure(session, msgs);
    if (limit(tokens)) {
      // The window shrank under recent work (a switch to a smaller model): its outputs are clipped to the new window's
      // share too. The full values stay in the kernel's variables.
      for (let keep = charsForTokens(input.limits.contextLength * input.rlm.cellOutputMaxShare); limit(tokens) && keep >= 200; keep = Math.floor(keep / 2)) {
        const clipped = truncateOldOutputs(session.entries, keep, 0);
        if (!clipped.changed) continue;
        session.entries = clipped.entries;
        msgs = buildMessages(sys, resume, blocks, session.entries);
        tokens = measure(session, msgs);
      }
    }
    if (limit(tokens)) {
      throw new Error(`The request needs about ${tokens} tokens but the model context is ${input.limits.contextLength}. Shorten the message or raise the context length in Settings → Models.`);
    }
    return msgs;
  }

  /** RLM transcript compaction: verbatim tail + structured checkpoint, same format as the session
   * compactor. Units keep each code cell with its output. */
  private async compactTranscript(session: Session, input: RlmTurnInput, isRoot: boolean, aggressive: boolean): Promise<void> {
    const units = toUnits(session.entries);
    if (units.length < 2) return;
    const b = computeBudget(input.limits, input.compact);
    const keepBudget = aggressive ? Math.floor(b.keep / 2) : b.keep;
    let used = 0;
    let cut = units.length;
    for (let i = units.length - 1; i >= 0; i--) {
      const cost = Math.ceil(units[i]!.reduce((n, e) => n + estimateTokens(e.text), 0) * (session.tokenScale ?? 1));
      if (i < units.length - 1 && used + cost > keepBudget) break;
      used += cost;
      cut = i;
    }
    if (cut === 0) cut = 1;
    const fold = units.slice(0, cut).flat();
    let keep = units.slice(cut).flat();
    // The request being worked on stays verbatim: a summary of it loses the details the work depends on.
    const request = [...session.entries].reverse().find((e) => e.kind === "user");
    if (request && fold.includes(request)) {
      const maxChars = Math.max(2000, b.newestLimit * 4);
      keep = [{ ...request, text: request.text.length > maxChars ? clipChars(request.text, maxChars) : request.text }, ...keep];
    }
    const previous = [...session.entries].reverse().find((e) => e.kind === "checkpoint")?.text ?? null;
    const body: ChatMessage[] = fold.map((e) => ({
      role: e.kind === "assistant" ? "assistant" : "user",
      content: e.kind === "repl" ? `[REPL output]\n${clipChars(e.text, 4000)}` : e.kind === "note" ? `[host note] ${e.text}` : e.text,
    }));
    const summary = await summarizeInChunks({
      model: await this.deps.summarizer(),
      prompts: this.deps.prompts,
      previous,
      body,
      maxTokens: Math.max(512, Math.floor(b.outputReserve * input.compact.summaryMaxShare)),
      contextLength: input.limits.contextLength,
      signal: input.signal,
    });
    session.entries = [{ kind: "checkpoint", text: summary }, ...keep];
    if (isRoot) {
      input.emit({ type: "compaction", checkpoint: { summary, foldedTurns: fold.length, keptTurns: keep.length, compactedBy: aggressive ? "in_turn" : "rlm" } });
    }
  }
}
