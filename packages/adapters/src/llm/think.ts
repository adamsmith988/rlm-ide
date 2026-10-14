/** Splits a streamed completion into thinking and answer text. Handles `<think>...</think>` in
 * the content stream, including templates (like FrogNano's) that open the think block in the
 * prompt so the output starts already inside it. */
export class ThinkSplitter {
  private buf = "";
  private inThink: boolean;
  private sawAny = false;

  constructor(startInThinking = false) {
    this.inThink = startInThinking;
  }

  push(chunk: string): Array<{ type: "thinking" | "text"; text: string }> {
    this.buf += chunk;
    const out: Array<{ type: "thinking" | "text"; text: string }> = [];
    for (;;) {
      if (!this.sawAny) {
        const trimmed = this.buf.trimStart();
        if (trimmed.length < 7 && "<think>".startsWith(trimmed)) return out;
        if (trimmed.startsWith("<think>")) {
          this.inThink = true;
          this.buf = trimmed.slice(7);
        }
        this.sawAny = true;
      }
      const tag = this.inThink ? "</think>" : "<think>";
      const i = this.buf.indexOf(tag);
      if (i === -1) {
        const hold = partial(this.buf, tag);
        const emit = this.buf.slice(0, this.buf.length - hold);
        if (emit) out.push({ type: this.inThink ? "thinking" : "text", text: emit });
        this.buf = this.buf.slice(this.buf.length - hold);
        return out;
      }
      const before = this.buf.slice(0, i);
      if (before) out.push({ type: this.inThink ? "thinking" : "text", text: before });
      this.buf = this.buf.slice(i + tag.length);
      if (this.inThink) this.buf = this.buf.replace(/^\s+/, "");
      this.inThink = !this.inThink;
    }
  }

  flush(): Array<{ type: "thinking" | "text"; text: string }> {
    const rest = this.buf;
    this.buf = "";
    return rest ? [{ type: this.inThink ? "thinking" : "text", text: rest }] : [];
  }
}

function partial(s: string, token: string): number {
  for (let n = Math.min(token.length - 1, s.length); n > 0; n--) if (token.startsWith(s.slice(s.length - n))) return n;
  return 0;
}
