import { spawn, type ChildProcess } from "node:child_process";
import { createWriteStream, type WriteStream } from "node:fs";

export interface RunOptions {
  cwd?: string;
  env?: Record<string, string>;
  onLine?: (line: string) => void;
  signal?: AbortSignal;
  logFile?: string;
  /** Characters of output kept for the result (default: the end, for error messages). */
  keepChars?: number;
}

export interface RunResult {
  code: number | null;
  tail: string;
}

const TAIL_CHARS = 6000;

/** Splits a child's stdout+stderr into lines (\r counts as a line end: progress bars redraw with it). */
export function pipeLines(child: ChildProcess, onLine: (line: string) => void, log?: WriteStream): void {
  for (const stream of [child.stdout, child.stderr]) {
    let buf = "";
    stream?.setEncoding("utf8");
    stream?.on("data", (d: string) => {
      log?.write(d);
      buf += d;
      const parts = buf.split(/\r\n|\n|\r/);
      buf = parts.pop() ?? "";
      for (const p of parts) if (p.trim()) onLine(p);
    });
    stream?.on("end", () => {
      if (buf.trim()) onLine(buf);
    });
  }
}

/** Runs a command to completion, keeping the end of its output for error messages. */
export function run(cmd: string, args: string[], o: RunOptions = {}): Promise<RunResult> {
  return new Promise((resolve) => {
    let tail = "";
    const log = o.logFile ? createWriteStream(o.logFile, { flags: "a" }) : undefined;
    log?.write(`\n$ ${cmd} ${args.join(" ")}\n`);
    let child: ChildProcess;
    try {
      child = spawn(cmd, args, { cwd: o.cwd, env: { ...process.env, ...o.env }, stdio: ["ignore", "pipe", "pipe"], signal: o.signal });
    } catch (e) {
      log?.end();
      resolve({ code: -1, tail: (e as Error).message });
      return;
    }
    pipeLines(
      child,
      (line) => {
        tail = (tail + line + "\n").slice(-(o.keepChars ?? TAIL_CHARS));
        o.onLine?.(line);
      },
      log,
    );
    child.on("error", (e) => {
      tail += `\n${e.message}`;
    });
    child.on("close", (code) => {
      log?.end();
      resolve({ code, tail });
    });
  });
}

export function lastLines(text: string, n = 12): string {
  return text.trim().split("\n").slice(-n).join("\n");
}
