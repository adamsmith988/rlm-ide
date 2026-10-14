/** Line-level diff measures shared by the edit-instead-of-rewrite guards (files, kernel writes, retried cells). */

/** A full rewrite of an existing file this long that keeps most lines counts as a "small fix by rewrite". */
export const REWRITE_MIN_LINES = 30;
/** At most this share of lines changed. */
export const REWRITE_MAX_CHANGED = 0.3;

export interface RewriteStats {
  beforeLines: number;
  afterLines: number;
  changedLines: number;
  ratio: number;
  /** 1-based line ranges in the new content that differ from the old one. */
  ranges: Array<[number, number]>;
}

/** Line diff via LCS on line hashes (files here are small enough), reporting changed ranges of `after`. */
export function rewriteStats(before: string, after: string): RewriteStats {
  const a = before.split("\n");
  const b = after.split("\n");
  // Trim the common head and tail first; the LCS then only runs on the middle.
  let head = 0;
  while (head < a.length && head < b.length && a[head] === b[head]) head++;
  let tail = 0;
  while (tail < a.length - head && tail < b.length - head && a[a.length - 1 - tail] === b[b.length - 1 - tail]) tail++;
  const am = a.slice(head, a.length - tail);
  const bm = b.slice(head, b.length - tail);
  const keep = new Array<boolean>(bm.length).fill(false);
  if (am.length && bm.length && am.length * bm.length <= 4_000_000) {
    const dp: Uint32Array[] = Array.from({ length: am.length + 1 }, () => new Uint32Array(bm.length + 1));
    for (let i = am.length - 1; i >= 0; i--) {
      for (let j = bm.length - 1; j >= 0; j--) dp[i]![j] = am[i] === bm[j] ? dp[i + 1]![j + 1]! + 1 : Math.max(dp[i + 1]![j]!, dp[i]![j + 1]!);
    }
    let i = 0;
    let j = 0;
    while (i < am.length && j < bm.length) {
      if (am[i] === bm[j]) {
        keep[j] = true;
        i++;
        j++;
      } else if (dp[i + 1]![j]! >= dp[i]![j + 1]!) i++;
      else j++;
    }
  } else if (am.length && bm.length) {
    const pool = new Map<string, number>();
    for (const l of am) pool.set(l, (pool.get(l) ?? 0) + 1);
    bm.forEach((l, j) => {
      const n = pool.get(l) ?? 0;
      if (n > 0) {
        keep[j] = true;
        pool.set(l, n - 1);
      }
    });
  }
  const kept = head + tail + keep.filter(Boolean).length;
  const ranges: Array<[number, number]> = [];
  keep.forEach((k, j) => {
    if (k) return;
    const line = head + j + 1;
    const last = ranges[ranges.length - 1];
    if (last && line <= last[1] + 1) last[1] = line;
    else ranges.push([line, line]);
  });
  const removedOnly = keep.length === 0 && am.length > 0;
  if (removedOnly) ranges.push([head + 1, head + 1]);
  const changed = Math.max(a.length, b.length) - kept;
  return { beforeLines: a.length, afterLines: b.length, changedLines: changed, ratio: changed / Math.max(1, Math.max(a.length, b.length)), ranges };
}

export function isSmallFixByRewrite(s: RewriteStats): boolean {
  return s.beforeLines >= REWRITE_MIN_LINES && s.changedLines > 0 && s.ratio <= REWRITE_MAX_CHANGED;
}

export function describeRanges(r: Array<[number, number]>, max = 6): string {
  const parts = r.slice(0, max).map(([x, y]) => (x === y ? `${x}` : `${x}-${y}`));
  return `${parts.join(", ")}${r.length > max ? ` and ${r.length - max} more` : ""}`;
}
