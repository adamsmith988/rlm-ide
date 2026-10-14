import { utilityProcess, type MessagePortMain, type UtilityProcess } from "electron";
import path from "node:path";

export type OrchestratorEvent =
  | { kind: "turn-event"; ev: unknown }
  | { kind: "task-board"; board: unknown }
  | { kind: "status"; status: Record<string, unknown> }
  | { kind: "need-inference" }
  | { kind: "settings-patch"; patch: Record<string, unknown> }
  | { kind: "host-request"; id: number; type: string; args: Record<string, unknown> }
  | { kind: "log"; level: string; msg: string };

/** Runs the orchestrator (core + adapters) in a utilityProcess and exposes request/response RPC. */
export class OrchestratorHost {
  private child: UtilityProcess | null = null;
  private seq = 0;
  private readonly pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
  private init: Record<string, unknown> | null = null;
  private restarts = 0;

  constructor(private readonly onEvent: (e: OrchestratorEvent) => void) {}

  start(init: Record<string, unknown>): void {
    this.init = init;
    const child = utilityProcess.fork(path.join(__dirname, "orchestrator.js"), [], { serviceName: "rlm-orchestrator", stdio: "pipe" });
    this.child = child;
    child.stdout?.on("data", (d: Buffer) => process.stdout.write(`[orch] ${d}`));
    child.stderr?.on("data", (d: Buffer) => process.stderr.write(`[orch] ${d}`));
    child.on("message", (m: { kind: string; id?: number; ok?: boolean; value?: unknown; error?: string }) => {
      if (m.kind === "result" && m.id !== undefined) {
        const p = this.pending.get(m.id);
        if (!p) return;
        this.pending.delete(m.id);
        if (m.ok) p.resolve(m.value);
        else p.reject(new Error(m.error ?? "orchestrator error"));
      } else {
        this.onEvent(m as OrchestratorEvent);
      }
    });
    child.on("exit", (code) => {
      for (const p of this.pending.values()) p.reject(new Error(`orchestrator exited (${code})`));
      this.pending.clear();
      this.child = null;
      if (this.restarts < 3 && this.init) {
        this.restarts++;
        setTimeout(() => this.start(this.init!), 500);
      }
    });
    child.postMessage({ kind: "init", init });
  }

  post(msg: Record<string, unknown>, ports?: MessagePortMain[]): void {
    this.child?.postMessage(msg, ports);
  }

  call<T = unknown>(method: string, args: unknown): Promise<T> {
    if (!this.child) return Promise.reject(new Error("orchestrator not running"));
    const id = ++this.seq;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
      this.child!.postMessage({ kind: "call", id, method, args });
    });
  }

  stop(): void {
    this.init = null;
    this.child?.kill();
  }
}
