import { spawn, type ChildProcess } from "node:child_process";

export interface ShellResult {
  exit_code: number | null;
  stdout: string;
  stderr: string;
  timed_out: boolean;
  pid?: number;
  running?: boolean;
}

export interface ProcessInfo {
  pid: number;
  command: string;
  running: boolean;
  exit_code: number | null;
  elapsed_sec: number;
  silent_sec: number;
  output_tail: string;
  note?: string;
}

/** Output kept per stream; the rest is cut from the middle so both head and tail survive. */
const SHELL_OUTPUT_CHARS = 12000;
/** A running background process silent this long gets a note asking the model to check it is not stuck. */
const STALL_NOTE_SEC = 300;

function clipMiddle(s: string, max: number): string {
  if (s.length <= max) return s;
  const head = Math.floor(max * 0.4);
  return `${s.slice(0, head)}\n...[${s.length - max} chars omitted]...\n${s.slice(s.length - (max - head))}`;
}

function shellCommand(command: string): { file: string; args: string[] } {
  if (process.platform === "win32") return { file: "powershell.exe", args: ["-NoProfile", "-NonInteractive", "-Command", command] };
  return { file: "/bin/bash", args: ["-lc", command] };
}

/** Tracks background processes (dev servers) so they can be listed, stopped and killed on exit. */
export class ProcessManager {
  private readonly procs = new Map<number, { child: ChildProcess; command: string; out: string[]; started: number; lastOutput: number }>();

  run(command: string, cwd: string, timeoutSec: number, signal: AbortSignal, env: NodeJS.ProcessEnv = process.env): Promise<ShellResult> {
    return new Promise((resolve) => {
      const { file, args } = shellCommand(command);
      const child = spawn(file, args, { cwd, env: { ...env, CI: "1", FORCE_COLOR: "0" }, detached: process.platform !== "win32" });
      let stdout = "";
      let stderr = "";
      let timedOut = false;
      child.stdout?.on("data", (d: Buffer) => (stdout += d.toString()));
      child.stderr?.on("data", (d: Buffer) => (stderr += d.toString()));
      const kill = () => this.killTree(child);
      const timer = setTimeout(() => {
        timedOut = true;
        kill();
      }, timeoutSec * 1000);
      const onAbort = () => kill();
      signal.addEventListener("abort", onAbort, { once: true });
      child.on("close", (code) => {
        clearTimeout(timer);
        signal.removeEventListener("abort", onAbort);
        resolve({ exit_code: code, stdout: clipMiddle(stdout, SHELL_OUTPUT_CHARS), stderr: clipMiddle(stderr, SHELL_OUTPUT_CHARS), timed_out: timedOut });
      });
      child.on("error", (e) => {
        clearTimeout(timer);
        resolve({ exit_code: -1, stdout, stderr: String(e), timed_out: false });
      });
    });
  }

  /** Start a long-running command and return its first output after `waitSec`. */
  start(command: string, cwd: string, waitSec: number): Promise<ShellResult> {
    const { file, args } = shellCommand(command);
    const child = spawn(file, args, { cwd, env: { ...process.env, FORCE_COLOR: "0", BROWSER: "none" }, detached: process.platform !== "win32" });
    const out: string[] = [];
    const push = (d: Buffer) => {
      const entry = child.pid ? this.procs.get(child.pid) : undefined;
      if (entry) entry.lastOutput = Date.now();
      out.push(d.toString());
      // Long jobs (training) print for hours; keep the head and a bounded tail.
      if (out.length > 4000) out.splice(200, out.length - 3000);
    };
    child.stdout?.on("data", push);
    child.stderr?.on("data", push);
    if (child.pid) this.procs.set(child.pid, { child, command, out, started: Date.now(), lastOutput: Date.now() });
    child.on("close", () => {
      /* keep entry so output can still be read */
    });
    return new Promise((resolve) => {
      setTimeout(() => {
        const text = out.join("");
        resolve({
          exit_code: child.exitCode,
          stdout: clipMiddle(text, SHELL_OUTPUT_CHARS),
          stderr: "",
          timed_out: false,
          pid: child.pid,
          running: child.exitCode === null,
        });
      }, waitSec * 1000);
    });
  }

  list(): ProcessInfo[] {
    return [...this.procs.entries()].map(([pid, p]) => {
      const running = p.child.exitCode === null && p.child.signalCode === null;
      const silent = Math.round((Date.now() - p.lastOutput) / 1000);
      return {
        pid,
        command: p.command,
        running,
        exit_code: p.child.exitCode,
        elapsed_sec: Math.round((Date.now() - p.started) / 1000),
        silent_sec: silent,
        output_tail: p.out.join("").slice(-2000),
        ...(running && silent >= STALL_NOTE_SEC
          ? { note: `no new output for ${silent}s. If it should be making progress, check its real progress (its log file, sizes of the files it writes); if nothing moves, stop it with shell.stop and fix or restart it.` }
          : {}),
      };
    });
  }

  /** Block until a background process exits or `timeoutSec` passes. */
  wait(pid: number, timeoutSec: number, signal?: AbortSignal): Promise<ProcessInfo | null> {
    const p = this.procs.get(pid);
    if (!p) return Promise.resolve(null);
    const snapshot = () => this.list().find((x) => x.pid === pid)!;
    if (p.child.exitCode !== null || p.child.signalCode !== null) return Promise.resolve(snapshot());
    return new Promise((resolve) => {
      const done = () => {
        clearTimeout(timer);
        p.child.off("close", done);
        signal?.removeEventListener("abort", done);
        resolve(snapshot());
      };
      const timer = setTimeout(done, Math.max(0, timeoutSec) * 1000);
      p.child.on("close", done);
      signal?.addEventListener("abort", done, { once: true });
    });
  }

  stop(pid: number): boolean {
    const p = this.procs.get(pid);
    if (!p) return false;
    this.killTree(p.child);
    this.procs.delete(pid);
    return true;
  }

  killAll(): void {
    for (const p of this.procs.values()) this.killTree(p.child);
    this.procs.clear();
  }

  private killTree(child: ChildProcess): void {
    if (!child.pid || child.exitCode !== null) return;
    try {
      if (process.platform === "win32") spawn("taskkill", ["/pid", String(child.pid), "/T", "/F"]);
      else process.kill(-child.pid, "SIGKILL");
    } catch {
      child.kill("SIGKILL");
    }
  }
}
