/** Watches one model reply as it streams and says when to cut it off: reasoning past its budget, or a model caught in
 * a loop (the same passage over and over). Small local models do both, and at a few tokens per second either one
 * looks like a hang. */

export type CutReason = "thinking_budget" | "repeating";

const WINDOW = 12_000;
const CHECK_EVERY = 256;
const NEEDLE = 24;

export class ReplyGuard {
  private thinkingChars = 0;
  private thinkingSince = 0;
  private readonly buf = { thinking: "", text: "" };
  private readonly unchecked = { thinking: 0, text: 0 };

  constructor(private readonly o: { thinkingMaxChars: number; thinkingMaxMs?: number; repeatLimit: number; now?: () => number }) {}

  get thinkingSoFar(): number {
    return this.thinkingChars;
  }

  push(kind: "thinking" | "text", text: string): CutReason | null {
    if (kind === "thinking") {
      const now = (this.o.now ?? Date.now)();
      if (!this.thinkingChars) this.thinkingSince = now;
      this.thinkingChars += text.length;
      if (this.thinkingChars > this.o.thinkingMaxChars) return "thinking_budget";
      if (this.o.thinkingMaxMs !== undefined && now - this.thinkingSince > this.o.thinkingMaxMs) return "thinking_budget";
    }
    const b = (this.buf[kind] + text).slice(-WINDOW);
    this.buf[kind] = b;
    this.unchecked[kind] += text.length;
    if (this.unchecked[kind] < CHECK_EVERY) return null;
    this.unchecked[kind] = 0;
    if (repeatingTail(b, this.o.repeatLimit)) return "repeating";
    if (kind === "thinking" && repeatedSentence(b, this.o.repeatLimit)) return "repeating";
    return null;
  }
}

/** The text ends in one block repeated `times` times in a row. The block must hold real words (8+ distinct
 * characters), so long runs like `0, 0, 0, ` in a data table do not count. */
export function repeatingTail(s: string, times: number): boolean {
  if (s.length < NEEDLE * times) return false;
  const needle = s.slice(-NEEDLE);
  let from = s.length - NEEDLE - 1;
  for (let tries = 0; tries < 64 && from >= 0; tries++) {
    const at = s.lastIndexOf(needle, from);
    if (at < 0) return false;
    const period = s.length - NEEDLE - at;
    from = at - 1;
    if (period < 16 || period * times > s.length) continue;
    const block = s.slice(-period);
    if (new Set(block).size < 8 || period * times < 400) continue;
    let ok = true;
    for (let k = 2; k <= times && ok; k++) ok = s.slice(s.length - period * k, s.length - period * (k - 1)) === block;
    if (ok) return true;
  }
  return false;
}

/** Reasoning that keeps coming back to the same sentence ("Wait, let me re-check the file.") even with other text in
 * between. Numbers are ignored so "attempt 3" and "attempt 4" count as the same sentence. */
export function repeatedSentence(s: string, times: number): boolean {
  const counts = new Map<string, number>();
  for (const raw of s.split(/(?<=[.!?])\s+|\n+/)) {
    const k = raw.toLowerCase().replace(/\d+/g, "#").replace(/\s+/g, " ").trim();
    if (k.length < 30) continue;
    const n = (counts.get(k) ?? 0) + 1;
    if (n > times) return true;
    counts.set(k, n);
  }
  return false;
}
