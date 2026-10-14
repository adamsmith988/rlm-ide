/** Split text into overlapping chunks, preferring paragraph / line boundaries. */
export function chunkText(text: string, chunkChars: number, overlap: number, maxChunks: number): string[] {
  const clean = text.trim();
  if (!clean) return [];
  if (clean.length <= chunkChars) return [clean];
  const chunks: string[] = [];
  let start = 0;
  const step = Math.max(1, chunkChars - overlap);
  while (start < clean.length && chunks.length < maxChunks) {
    let end = Math.min(clean.length, start + chunkChars);
    if (end < clean.length) {
      const window = clean.slice(start, end);
      const para = window.lastIndexOf("\n\n");
      const line = window.lastIndexOf("\n");
      const sentence = window.lastIndexOf(". ");
      const cut = [para, line, sentence].find((p) => p > chunkChars * 0.5);
      if (cut !== undefined && cut > 0) end = start + cut + 1;
    }
    chunks.push(clean.slice(start, end).trim());
    if (end >= clean.length) break;
    start = Math.max(start + 1, end - overlap, start + Math.min(step, end - start));
  }
  return chunks.filter((c) => c.length > 0);
}

export function cosine(a: readonly number[], b: readonly number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i]!;
    const y = b[i]!;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}
