/** The one checkpoint format shared by the session compactor, RLM transcript compaction and
 * the in-turn guard. */
export const CHECKPOINT_SECTIONS = [
  "Goal",
  "Constraints & Preferences",
  "Facts & Values",
  "Tool Results",
  "Requests answered",
  "Open requests",
  "Dead ends",
] as const;

export type CheckpointSection = (typeof CHECKPOINT_SECTIONS)[number];

export type ParsedCheckpoint = Record<CheckpointSection, string>;

export function parseCheckpoint(text: string): ParsedCheckpoint {
  const out = Object.fromEntries(CHECKPOINT_SECTIONS.map((s) => [s, ""])) as ParsedCheckpoint;
  let current: CheckpointSection | null = null;
  const buf: string[] = [];
  const flush = () => {
    if (current) out[current] = buf.join("\n").trim();
    buf.length = 0;
  };
  for (const line of text.split("\n")) {
    const m = /^#{1,3}\s*(.+?)\s*$/.exec(line);
    const heading = m?.[1];
    const match = heading ? CHECKPOINT_SECTIONS.find((s) => s.toLowerCase() === heading.toLowerCase()) : undefined;
    if (match) {
      flush();
      current = match;
    } else if (current) {
      buf.push(line);
    }
  }
  flush();
  return out;
}

/** Normalizes model output into the canonical section order. Missing sections become "(none)".
 * Text before the first heading is dropped (models sometimes add a preamble). */
export function normalizeCheckpoint(text: string): string {
  const stripped = text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
  const parsed = parseCheckpoint(stripped);
  const anySection = CHECKPOINT_SECTIONS.some((s) => parsed[s].length > 0);
  if (!anySection) {
    return CHECKPOINT_SECTIONS.map((s) => `## ${s}\n${s === "Facts & Values" ? stripped || "(none)" : "(none)"}`).join("\n\n");
  }
  return CHECKPOINT_SECTIONS.map((s) => `## ${s}\n${parsed[s] || "(none)"}`).join("\n\n");
}
