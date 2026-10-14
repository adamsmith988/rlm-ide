import { spawn, spawnSync, type ChildProcessWithoutNullStreams } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { createInterface } from "node:readline";
import type { CellOutput, CellResult, HostRequestHandler, KernelFactory, KernelSession, LoggerPort, ToolDescriptor } from "@rlm/core";
import { REWRITE_MAX_CHANGED, REWRITE_MIN_LINES } from "../tools/rewrite_guard";

/** Grace period after an interrupt before the worker is killed and restarted. */
const INTERRUPT_GRACE_MS = 3000;
const READY_TIMEOUT_MS = 20000;

export interface PythonKernelOptions {
  workerScript: string;
  /** Candidate interpreters, first existing wins (bundled first, then system). */
  pythonCandidates: string[];
  sandboxNoNetwork: () => boolean;
  memoryMb: () => number;
  log: LoggerPort;
}

type SandboxMode = "bwrap" | "landlock" | "jobobject" | "none";

export class PythonKernelFactory implements KernelFactory {
  private python: string | null = null;
  private sandbox: { mode: SandboxMode; detail: string } | null = null;

  constructor(private readonly opts: PythonKernelOptions) {}

  resolvePython(): string {
    if (this.python) return this.python;
    for (const c of this.opts.pythonCandidates) {
      if (!c) continue;
      if (path.isAbsolute(c) ? existsSync(c) : spawnSync(c, ["--version"]).status === 0) {
        this.python = c;
        return c;
      }
    }
    throw new Error("No Python interpreter found. The bundled runtime is missing; reinstall the app or set RLM_PYTHON.");
  }

  sandboxInfo(): { mode: SandboxMode; detail: string } {
    if (this.sandbox) return this.sandbox;
    if (process.platform === "win32") {
      this.sandbox = { mode: "jobobject", detail: "Windows: Job Object with kill-on-close and a memory cap. There is no filesystem jail. Tool approvals are the primary guard." };
    } else if (process.platform === "linux") {
      const probe = spawnSync("bwrap", ["--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--unshare-net", "true"], { timeout: 5000 });
      if (probe.status === 0) this.sandbox = { mode: "bwrap", detail: "Linux bubblewrap: read-only system, writable project folder only, network off by default." };
      else
        this.sandbox = {
          mode: "none",
          detail: `bubblewrap is unavailable (${probe.error ? "not installed" : (probe.stderr?.toString().trim() || "blocked by the system")}). The kernel runs unsandboxed. File tools stay confined to the project and still need approval.`,
        };
    } else {
      this.sandbox = { mode: "none", detail: "No sandbox on this platform. The kernel runs unsandboxed." };
    }
    return this.sandbox;
  }

  async create(o: { cwd: string; tools: ToolDescriptor[]; onHostRequest: HostRequestHandler; outputLimitChars: number }): Promise<KernelSession> {
    const k = new PythonKernel(this, o, this.opts);
    await k.start();
    return k;
  }

  command(cwd: string): { file: string; args: string[]; env: NodeJS.ProcessEnv } {
    const py = this.resolvePython();
    const env: NodeJS.ProcessEnv = { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8", RLM_MEMORY_MB: String(this.opts.memoryMb()) };
    const base = [py, "-u", this.opts.workerScript];
    const sb = this.sandboxInfo();
    if (sb.mode === "bwrap") {
      const args = ["--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--bind", cwd, cwd, "--die-with-parent", "--chdir", cwd];
      if (this.opts.sandboxNoNetwork()) args.push("--unshare-net");
      return { file: "bwrap", args: [...args, ...base], env };
    }
    return { file: base[0]!, args: base.slice(1), env };
  }
}

class PythonKernel implements KernelSession {
  readonly id = `k_${Math.random().toString(36).slice(2, 10)}`;
  private proc: ChildProcessWithoutNullStreams | null = null;
  private alive = false;
  private pending = new Map<string, { resolve: (r: CellResult) => void; onOutput: (o: CellOutput) => void }>();
  private waiters = new Map<string, () => void>();
  private cellSeq = 0;
  private currentCell: string | null = null;

  constructor(
    private readonly factory: PythonKernelFactory,
    private readonly o: { cwd: string; tools: ToolDescriptor[]; onHostRequest: HostRequestHandler; outputLimitChars: number },
    private readonly opts: PythonKernelOptions,
  ) {}

  isAlive(): boolean {
    return this.alive;
  }

  async start(): Promise<void> {
    const cmd = this.factory.command(this.o.cwd);
    const proc = spawn(cmd.file, cmd.args, { cwd: this.o.cwd, env: cmd.env, stdio: "pipe" });
    this.proc = proc;
    const ready = this.waitFor("ready", READY_TIMEOUT_MS);
    const rl = createInterface({ input: proc.stdout });
    rl.on("line", (line) => this.onLine(line));
    let stderrTail = "";
    proc.stderr.on("data", (d: Buffer) => (stderrTail = (stderrTail + d.toString()).slice(-4000)));
    proc.on("exit", (code) => {
      this.alive = false;
      for (const [id, p] of this.pending) {
        p.resolve({ ok: false, stdout: "", stderr: stderrTail, result: "", error: `kernel exited (code ${code})`, timedOut: false, interrupted: false });
        this.pending.delete(id);
      }
    });
    proc.on("error", (e) => this.opts.log.error("kernel spawn failed", String(e)));
    await ready.catch((e) => {
      throw new Error(`Python kernel did not start: ${String(e)} ${stderrTail}`);
    });
    this.alive = true;
    const boot = this.waitFor("bootstrapped", READY_TIMEOUT_MS);
    this.send({
      op: "bootstrap",
      output_limit: this.o.outputLimitChars,
      rewrite_guard: { min_lines: REWRITE_MIN_LINES, max_changed: REWRITE_MAX_CHANGED },
      tools: this.o.tools.map((t) => ({ id: t.id, params: Object.keys(t.parameters.properties ?? {}).sort((a, b) => Number((t.parameters.required ?? []).includes(b)) - Number((t.parameters.required ?? []).includes(a))), doc: t.description })),
    });
    await boot;
  }

  private waitFor(op: string, timeoutMs: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const t = setTimeout(() => {
        this.waiters.delete(op);
        reject(new Error(`timeout waiting for ${op}`));
      }, timeoutMs);
      this.waiters.set(op, () => {
        clearTimeout(t);
        resolve();
      });
    });
  }

  private send(obj: unknown): void {
    this.proc?.stdin.write(`${JSON.stringify(obj)}\n`);
  }

  private onLine(line: string): void {
    let msg: Record<string, unknown>;
    try {
      msg = JSON.parse(line) as Record<string, unknown>;
    } catch {
      return;
    }
    const op = String(msg.op);
    const w = this.waiters.get(op);
    if (w) {
      this.waiters.delete(op);
      w();
      return;
    }
    if (op === "stream") {
      this.pending.get(String(msg.id))?.onOutput({ stream: msg.stream === "stderr" ? "stderr" : "stdout", text: String(msg.text ?? "") });
    } else if (op === "exec_result") {
      const p = this.pending.get(String(msg.id));
      if (!p) return;
      this.pending.delete(String(msg.id));
      if (msg.result) p.onOutput({ stream: "result", text: String(msg.result) });
      p.resolve({
        ok: Boolean(msg.ok),
        stdout: String(msg.stdout ?? ""),
        stderr: String(msg.stderr ?? ""),
        result: String(msg.result ?? ""),
        error: (msg.error as string | null) ?? null,
        timedOut: false,
        interrupted: Boolean(msg.interrupted),
      });
    } else if (op === "host_request") {
      const id = String(msg.id);
      if (process.env.RLM_DEBUG_KERNEL) console.error("host_request", JSON.stringify(msg));
      this.o
        .onHostRequest(String(msg.type), (msg.payload as Record<string, unknown>) ?? {})
        .then((result) => this.send({ op: "host_response", id, ok: true, result }))
        .catch((e: unknown) => this.send({ op: "host_response", id, ok: false, error: e instanceof Error ? e.message : String(e) }));
    }
  }

  async execute(code: string, onOutput: (o: CellOutput) => void, timeoutMs: number): Promise<CellResult> {
    if (!this.alive) await this.start();
    const id = `c${++this.cellSeq}`;
    this.currentCell = id;
    const result = new Promise<CellResult>((resolve) => this.pending.set(id, { resolve, onOutput }));
    this.send({ op: "exec", id, code });
    let timer: ReturnType<typeof setTimeout> | undefined;
    let timedOut = false;
    const timeout = new Promise<CellResult>((resolve) => {
      timer = setTimeout(async () => {
        timedOut = true;
        await this.interrupt();
        const p = this.pending.get(id);
        this.pending.delete(id);
        p?.resolve({ ok: false, stdout: "", stderr: "", result: "", error: `cell timed out after ${Math.round(timeoutMs / 1000)}s`, timedOut: true, interrupted: true });
        resolve({ ok: false, stdout: "", stderr: "", result: "", error: `cell timed out after ${Math.round(timeoutMs / 1000)}s`, timedOut: true, interrupted: true });
      }, timeoutMs);
    });
    let r = await Promise.race([result, timeout]);
    clearTimeout(timer);
    this.currentCell = null;
    // The interrupt can settle the cell first (e.g. when the kernel had to be killed); it is still a timeout.
    if (timedOut && !r.timedOut) {
      const lost = this.alive ? "" : " The kernel did not stop, so it was restarted and all variables were lost.";
      r = { ...r, timedOut: true, error: `cell timed out after ${Math.round(timeoutMs / 1000)}s.${lost}` };
    }
    return r;
  }

  async interrupt(): Promise<void> {
    const cell = this.currentCell;
    if (!cell || !this.proc) return;
    this.send({ op: "interrupt" });
    if (process.platform !== "win32") this.proc.kill("SIGINT");
    await new Promise((r) => setTimeout(r, 50));
    const deadline = Date.now() + INTERRUPT_GRACE_MS;
    while (this.pending.has(cell) && Date.now() < deadline) await new Promise((r) => setTimeout(r, 100));
    if (this.pending.has(cell)) {
      this.opts.log.warn("kernel did not respond to interrupt; killing");
      this.proc.kill("SIGKILL");
      const p = this.pending.get(cell);
      this.pending.delete(cell);
      p?.resolve({ ok: false, stdout: "", stderr: "", result: "", error: "kernel killed after interrupt; variables were lost", timedOut: false, interrupted: true });
      this.alive = false;
    }
  }

  async reset(): Promise<void> {
    const done = this.waitFor("reset_done", 5000);
    this.send({ op: "reset" });
    await done;
  }

  async shutdown(): Promise<void> {
    if (!this.proc) return;
    this.send({ op: "shutdown" });
    const p = this.proc;
    setTimeout(() => p.kill("SIGKILL"), 1500).unref();
    this.alive = false;
  }
}
