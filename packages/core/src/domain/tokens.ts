import { CHARS_PER_TOKEN } from "@rlm/contracts";

/** Per-message framing overhead of chat templates (role markers, separators). */
const MESSAGE_OVERHEAD_TOKENS = 6;

export function estimateTokens(text: string): number {
  if (!text) return 0;
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

/** Characters that `estimateTokens` counts as at most `tokens`. */
export function charsForTokens(tokens: number): number {
  return Math.max(0, Math.floor(tokens * CHARS_PER_TOKEN));
}

export function estimateMessagesTokens(messages: ReadonlyArray<{ content: string }>): number {
  let total = 0;
  for (const m of messages) total += estimateTokens(m.content) + MESSAGE_OVERHEAD_TOKENS;
  return total;
}

/** Clip text to roughly `maxTokens`, keeping head and tail, with an explicit marker. */
export function clipToTokens(text: string, maxTokens: number): string {
  const maxChars = Math.max(0, Math.floor(maxTokens * CHARS_PER_TOKEN));
  if (text.length <= maxChars) return text;
  const head = Math.floor(maxChars * 0.6);
  const tail = maxChars - head;
  const omitted = text.length - head - tail;
  return `${text.slice(0, head)}\n...[truncated ${omitted} chars]...\n${text.slice(text.length - tail)}`;
}

export function clipChars(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  const omitted = text.length - maxChars;
  return `${text.slice(0, maxChars)}\n[truncated ${omitted} chars]`;
}
