import type { CompactSettings } from "@rlm/contracts";

export interface ModelLimits {
  contextLength: number;
  maxOutputTokens: number;
}

/** The configured limits, narrowed to the window of the model actually serving now (never widened). */
export function narrowLimits(base: ModelLimits, active: { contextLength: number | null; maxOutputTokens: number | null } | null | undefined): ModelLimits {
  if (!active?.contextLength) return base;
  const contextLength = Math.min(base.contextLength, active.contextLength);
  return { contextLength, maxOutputTokens: Math.min(base.maxOutputTokens, active.maxOutputTokens ?? Math.floor(contextLength / 4)) };
}

export interface HistoryBudget {
  window: number;
  outputReserve: number;
  history: number;
  trigger: number;
  keep: number;
  newestLimit: number;
  maxRows: number;
}

/** Budget math from section 6.2 of the spec. Every ratio comes from settings. */
export function computeBudget(limits: ModelLimits, s: CompactSettings): HistoryBudget {
  const window = limits.contextLength;
  const outputReserve = Math.min(limits.maxOutputTokens, Math.floor(window * s.maxOutputRatio));
  const history = Math.floor(Math.max(0, window - outputReserve) * s.historyShare);
  const trigger = Math.floor(history * s.triggerRatio);
  const keep = Math.floor(history * s.keepRecentRatio);
  const newestLimit = Math.floor(history * s.newestLimitRatio);
  const maxRows = Math.max(4, Math.floor((history / 1000) * s.maxRowsPer1kHistory));
  return { window, outputReserve, history, trigger, keep, newestLimit, maxRows };
}
