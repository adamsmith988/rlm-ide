import type { CompactSettings } from "@rlm/contracts";
import { computeBudget, type HistoryBudget, type ModelLimits } from "../domain/budget";
import { CHECKPOINT_SECTIONS, normalizeCheckpoint, parseCheckpoint } from "../domain/checkpoint";
import { planCut, turnTokens, type Checkpoint, type StoredTurn } from "../domain/conversation";
import { clipChars, estimateTokens } from "../domain/tokens";
import { ContextOverflowError, type ChatMessage, type ChatModelPort, type PromptsPort } from "../ports";

export interface CompactionResult {
  compacted: boolean;
  checkpoint: Checkpoint | null;
  kept: StoredTurn[];
  foldedTurns: number;
  measuredTokens: number;
  budget: HistoryBudget;
  reason: "none" | "tokens" | "rows";
}

export interface SessionCompactorDeps {
  prompts: PromptsPort;
  summarizer: () => Promise<ChatModelPort>;
}

/** Builds the summarizer conversation: previous checkpoint, then the turns to fold, then the
 * checkpoint instruction. Shared by the session compactor and RLM transcript compaction so both
 * write the same format. */
export function buildSummarizerMessages(
  prompts: PromptsPort,
  previous: string | null,
  body: ChatMessage[],
): ChatMessage[] {
  const system = previous
    ? `${prompts.get("session_compact/system.md")}\n\nEarlier compacted summary:\n${previous}`
    : prompts.get("session_compact/system.md");
  const msgs: ChatMessage[] = [{ role: "system", content: system }];
  // Templates require a user message first; merge consecutive roles to keep alternation.
  for (const m of body) {
    const last = msgs[msgs.length - 1]!;
    if (last.role === m.role && last.role !== "system") last.content += `\n\n${m.content}`;
    else msgs.push({ ...m });
  }
  const last = msgs[msgs.length - 1]!;
  const instruction = prompts.get("session_compact/summarizer.md");
  if (last.role === "user") last.content += `\n\n---\n${instruction}`;
  else msgs.push({ role: "user", content: instruction });
  return msgs;
}

export async function runSummarizer(
  model: ChatModelPort,
  messages: ChatMessage[],
  maxTokens: number,
  signal?: AbortSignal,
): Promise<string> {
  let text = "";
  for await (const c of model.stream({ messages, maxTokens, temperature: 0.2, signal, enableThinking: false })) {
    if (c.type === "text") text += c.text;
  }
  return normalizeCheckpoint(text);
}

/** True when a checkpoint carries nothing (the summarizer returned empty or unusable text). */
export function isEmptyCheckpoint(text: string): boolean {
  const p = parseCheckpoint(text);
  return CHECKPOINT_SECTIONS.every((s) => !p[s] || /^\(none\)$/i.test(p[s].trim()));
}

/** Without a usable summary, keep a plain digest of what was folded rather than losing it. */
export function digestCheckpoint(previous: string | null, body: ChatMessage[]): string {
  const lines = body.map((m) => `- ${m.role}: ${clipChars(m.content.replace(/\s+/g, " ").trim(), 400)}`).slice(-60);
  const prev = previous ? parseCheckpoint(previous) : null;
  const tools = [prev?.["Tool Results"] && !/^\(none\)$/i.test(prev["Tool Results"]) ? prev["Tool Results"] : "", ...lines].filter(Boolean).join("\n");
  const merged = CHECKPOINT_SECTIONS.map((s) => `## ${s}\n${s === "Tool Results" ? tools : (prev?.[s] ?? "") || "(none)"}`).join("\n\n");
  return normalizeCheckpoint(merged);
}

/** Folds `body` into a checkpoint in as many summarizer calls as the window needs, carrying the
 * checkpoint forward; a chunk the summarizer rejects as too long is split and retried. */
export async function summarizeInChunks(a: {
  model: ChatModelPort;
  prompts: PromptsPort;
  previous: string | null;
  body: ChatMessage[];
  maxTokens: number;
  contextLength: number;
  signal?: AbortSignal;
}): Promise<string> {
  const overhead = estimateTokens(a.prompts.get("session_compact/system.md")) + estimateTokens(a.prompts.get("session_compact/summarizer.md")) + 256;
  let prev = a.previous;
  let i = 0;
  let shrink = 1;
  while (i < a.body.length) {
    const room = Math.max(512, Math.floor((a.contextLength - a.maxTokens - overhead - estimateTokens(prev ?? "")) / shrink));
    const chunk: ChatMessage[] = [];
    let used = 0;
    for (let j = i; j < a.body.length; j++) {
      const m = a.body[j]!;
      const cost = estimateTokens(m.content) + 8;
      if (chunk.length && used + cost > room) break;
      chunk.push(cost > room ? { ...m, content: clipChars(m.content, room * 3) } : m);
      used += Math.min(cost, room);
    }
    let next: string;
    try {
      next = await runSummarizer(a.model, buildSummarizerMessages(a.prompts, prev, chunk), a.maxTokens, a.signal);
    } catch (e) {
      if (e instanceof ContextOverflowError && shrink < 16) {
        shrink *= 2;
        continue;
      }
      throw e;
    }
    prev = isEmptyCheckpoint(next) ? digestCheckpoint(prev, chunk) : next;
    i += chunk.length;
    shrink = 1;
  }
  return prev ?? normalizeCheckpoint("");
}

export class SessionCompactor {
  constructor(private readonly deps: SessionCompactorDeps) {}

  measure(cp: Checkpoint | null, turns: StoredTurn[], newUserText: string): number {
    return (cp ? estimateTokens(cp.summary) : 0) + turns.reduce((n, t) => n + turnTokens(t), 0) + estimateTokens(newUserText);
  }

  async maybeCompact(args: {
    limits: ModelLimits;
    settings: CompactSettings;
    checkpoint: Checkpoint | null;
    verbatim: StoredTurn[];
    newUserText: string;
    signal?: AbortSignal;
    force?: boolean;
  }): Promise<CompactionResult> {
    const budget = computeBudget(args.limits, args.settings);
    const measured = this.measure(args.checkpoint, args.verbatim, args.newUserText);
    const overTokens = measured > budget.trigger;
    const overRows = args.verbatim.length > budget.maxRows;
    const base: CompactionResult = {
      compacted: false,
      checkpoint: args.checkpoint,
      kept: args.verbatim,
      foldedTurns: 0,
      measuredTokens: measured,
      budget,
      reason: "none",
    };
    if (!args.force && !overTokens && !overRows) return base;

    const keepBudget = Math.max(0, budget.keep - estimateTokens(args.newUserText));
    let { fold, keep } = planCut(args.verbatim, keepBudget, budget.newestLimit);
    if (overRows && keep.length > budget.maxRows / 2) {
      const n = keep.length - Math.floor(budget.maxRows / 2);
      fold = fold.concat(keep.slice(0, n));
      keep = keep.slice(n);
    }
    if (fold.length === 0) return base;

    const body: ChatMessage[] = [];
    for (const t of fold) {
      body.push({ role: "user", content: t.userText });
      const work = t.meta.workLog ? `\n\n[work log: code cells and tool results]\n${t.meta.workLog}` : "";
      if (t.assistantText || work) body.push({ role: "assistant", content: `${t.assistantText}${work}` });
    }
    const summary = await summarizeInChunks({
      model: await this.deps.summarizer(),
      prompts: this.deps.prompts,
      previous: args.checkpoint?.summary ?? null,
      body,
      maxTokens: Math.max(512, Math.floor(budget.outputReserve * args.settings.summaryMaxShare)),
      contextLength: args.limits.contextLength,
      signal: args.signal,
    });
    const throughIdx = fold[fold.length - 1]!.idx;
    const checkpoint = { summary, throughIdx };
    return {
      compacted: true,
      checkpoint,
      kept: keep,
      foldedTurns: fold.length,
      measuredTokens: this.measure(checkpoint, keep, args.newUserText),
      budget,
      reason: overTokens ? "tokens" : "rows",
    };
  }
}
