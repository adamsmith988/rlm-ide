import type { CompactSettings } from "@rlm/contracts";

export interface ModelLimits {
  contextLength: number;
  maxOutputTokens: number;
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
