import { describeRanges, isSmallFixByRewrite, rewriteStats } from "../../domain/line_diff";

/** Executable cells are fenced as ```repl. Other fences (```python, ```ts) are shown, not run.
 * The closing fence must start a line, so backticks inside strings do not end a cell. */
const CELL_RE = /```repl[^\n]*\n([\s\S]*?)(?:^|\n)```[ \t]*(?=\n|$)/g;
const OPEN = "```repl";

export function extractCells(text: string): string[] {
  const cells: string[] = [];
  for (const m of text.matchAll(CELL_RE)) {
    const code = (m[1] ?? "").replace(/\s+$/, "");
    if (code.trim()) cells.push(code);
  }
  return cells;
}

export function stripCells(text: string): string {
  return text.replace(CELL_RE, "").replace(/\n{3,}/g, "\n\n").trim();
}

/** A ```repl fence that was opened but never closed, typically because the reply hit the output
 * limit. Returns the index of the fence, or -1. */
export function unclosedCellAt(text: string): number {
  const rest = text.replace(CELL_RE, (m) => " ".repeat(m.length));
  return rest.indexOf(OPEN);
}

/** Appended when a cell hits the time limit, so the model changes approach instead of retrying the same cell. */
export function timeoutHint(limitSec: number): string {
  return [
    `[cell timed out after ${limitSec}s] repl cells are for quick steps; this work needs longer than the cell limit, and running the same cell again will time out again.`,
    "[hint] Write the long-running work (downloads, training, builds, pipelines) to a script that logs its progress to a file, start it with shell.run(command=..., background=True), then wait for it with shell.wait(pid=...) (one call per cell, repeat while running is True) or read the log, until it finishes. Make the script resumable (skip steps whose outputs already exist).",
  ].join("\n");
}

export interface FailedCell {
  code: string;
  error: string;
}

export interface CellRetry {
  changed: number;
  total: number;
  where: string;
}

/** A long cell sent again with small changes after it failed, i.e. a fix made by re-sending everything. */
export function retriedCell(prev: FailedCell | undefined, code: string): CellRetry | null {
  if (!prev) return null;
  const st = rewriteStats(prev.code, code);
  return isSmallFixByRewrite(st) ? { changed: st.changedLines, total: st.afterLines, where: describeRanges(st.ranges) } : null;
}

export function retriedCellText(r: CellRetry, refused: boolean): string {
  const what = `this cell repeats the previous failing cell with only ${r.changed} of ${r.total} lines changed (lines ${r.where})`;
  const how = 'Put the code in a file once with fs.write_file(path=..., content=...), run the file (shell.run(command="python file.py") or exec(open("file.py").read())), and fix errors in it with fs.edit_file so the parts that work stay as they are.';
  return refused ? `[not run] ${what}. Long cells are not re-sent to fix them. ${how}` : `[IDE] ${what}. Do not re-send long cells to fix them. ${how} The next re-sent copy of a failing long cell will not be run.`;
}

const lastLine = (s: string) =>
  s
    .trim()
    .split("\n")
    .filter((l) => l.trim())
    .pop()
    ?.trim() ?? "";

/** Appended when a cell fails with the same error as the previous failure. */
export function sameErrorHint(prevError: string, error: string): string {
  const a = lastLine(prevError);
  if (!a || a !== lastLine(error)) return "";
  return `[hint] Same error as the previous attempt (${a.slice(0, 200)}); that change did not fix it. Read the whole error and the code it points at, work out the cause, and change that.`;
}

/** A hint appended to a failed cell's output when the model wrote another language as Python. */
export function syntaxHint(code: string, error: string): string {
  if (!/SyntaxError/.test(error)) return "";
  if (/^\s*(const|let|var|function|import .* from|export )\b|console\.log\(|require\(|=>/m.test(code)) {
    return "[hint] repl blocks run Python. Run JavaScript with shell.run(command=\"node file.js\") or write it to a file with fs.write_file first.";
  }
  if (/unterminated string literal/.test(error) && /content\s*=\s*["'](?!["'])/.test(code)) {
    return '[hint] Pass file contents as a triple-quoted string: content="""...""".';
  }
  return "";
}

/** Removes reasoning a model wrote inline: closed blocks, a close tag whose opener was in the prompt template, and a
 * block the output limit cut off before it closed. */
export function stripThinking(text: string): string {
  return text
    .replace(/<think>[\s\S]*?<\/think>/g, "")
    .replace(/^[\s\S]*?<\/think>/, "")
    .replace(/<think>[\s\S]*$/, "")
    .trim();
}

/** Streaming filter: passes prose through, swallows ```repl ... ``` fences (they are shown as
 * code cell cards instead). Holds back a small tail so a fence split across chunks is detected. */
export class CellFenceFilter {
  private buf = "";
  private inCell = false;
  private static readonly CLOSE = /(?:^|\n)```[ \t]*(?=\n|$)/;

  push(chunk: string): string {
    this.buf += chunk;
    let out = "";
    for (;;) {
      if (!this.inCell) {
        const i = this.buf.indexOf(OPEN);
        if (i === -1) {
          const hold = this.partialSuffix(this.buf, OPEN);
          out += this.buf.slice(0, this.buf.length - hold);
          this.buf = this.buf.slice(this.buf.length - hold);
          return out;
        }
        out += this.buf.slice(0, i);
        this.buf = this.buf.slice(i + OPEN.length);
        this.inCell = true;
      } else {
        const nl = this.buf.indexOf("\n");
        if (nl === -1) return out;
        // Wait for the character after a candidate fence so "```python" inside a cell is not a close.
        const m = CellFenceFilter.CLOSE.exec(this.buf.slice(nl));
        if (!m) return out;
        const end = nl + m.index + m[0].length;
        if (end >= this.buf.length) return out;
        this.buf = this.buf.slice(end);
        this.inCell = false;
      }
    }
  }

  /** The code of the cell currently being streamed, or null when outside a cell. */
  draft(): string | null {
    if (!this.inCell) return null;
    const nl = this.buf.indexOf("\n");
    return nl === -1 ? "" : this.buf.slice(nl + 1);
  }

  flush(): string {
    const rest = this.inCell ? "" : this.buf;
    this.buf = "";
    this.inCell = false;
    return rest;
  }

  private partialSuffix(s: string, token: string): number {
    for (let n = Math.min(token.length - 1, s.length); n > 0; n--) {
      if (token.startsWith(s.slice(s.length - n))) return n;
    }
    return 0;
  }
}
