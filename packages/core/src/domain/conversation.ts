import { estimateTokens } from "./tokens";

export interface StoredTurn {
  id: string;
  chatId: string;
  idx: number;
  userText: string;
  assistantText: string;
  status: "running" | "done" | "error" | "cancelled";
  createdAt: number;
  meta: TurnMeta;
}

export interface TurnMeta {
  /** Present on the turn whose preflight produced a session checkpoint. */
  selfCompact?: {
    summary: string;
    compactedBy: "session";
    /** Every turn with idx <= throughIdx is folded into `summary`. */
    throughIdx: number;
  };
  model?: string;
  feedback?: "up" | "down" | null;
  /** Compact list of cells/tool results this turn; fed to the summarizer, not shown as chat text. */
  workLog?: string;
  /** "wakeup" when the IDE started the turn from a scheduled wake-up instead of a user message. */
  origin?: "user" | "wakeup";
}

export interface Checkpoint {
  summary: string;
  throughIdx: number;
}

export function turnTokens(t: Pick<StoredTurn, "userText" | "assistantText">): number {
  return estimateTokens(t.userText) + estimateTokens(t.assistantText) + 12;
}

/** The latest checkpoint stored in turn metadata, if any. */
export function latestCheckpoint(turns: readonly StoredTurn[]): Checkpoint | null {
  for (let i = turns.length - 1; i >= 0; i--) {
    const sc = turns[i]?.meta.selfCompact;
    if (sc) return { summary: sc.summary, throughIdx: sc.throughIdx };
  }
  return null;
}

/** The turns the engine receives verbatim: those after the checkpoint, finished or cancelled. */
export function verbatimTurns(turns: readonly StoredTurn[], cp: Checkpoint | null): StoredTurn[] {
  const from = cp ? cp.throughIdx : -1;
  return turns.filter((t) => t.idx > from && t.status !== "running");
}

export interface CutPlan {
  fold: StoredTurn[];
  keep: StoredTurn[];
}

/** Oldest whole turns fold, newest stay verbatim within `keepTokens`. The newest turn is kept even
 * when it alone exceeds `keepTokens`, as long as it fits `newestLimit`, so kept never collapses
 * to 0 for normal turns. */
export function planCut(turns: readonly StoredTurn[], keepTokens: number, newestLimit: number): CutPlan {
  const keep: StoredTurn[] = [];
  let used = 0;
  for (let i = turns.length - 1; i >= 0; i--) {
    const t = turns[i]!;
    const cost = turnTokens(t);
    if (keep.length === 0) {
      if (cost <= Math.max(keepTokens, newestLimit)) {
        keep.unshift(t);
        used += cost;
        continue;
      }
      break;
    }
    if (used + cost > keepTokens) break;
    keep.unshift(t);
    used += cost;
  }
  const fold = turns.slice(0, turns.length - keep.length);
  return { fold, keep };
}
