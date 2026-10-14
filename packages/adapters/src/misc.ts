import { readFileSync } from "node:fs";
import path from "node:path";
import type { LoggerPort, PromptsPort, WorkspacePort } from "@rlm/core";
import { ProcessManager } from "./tools/shell";

/** Prompts are `.md` files on disk, cached after first read. */
export class FilePrompts implements PromptsPort {
  private readonly cache = new Map<string, string>();
  constructor(private readonly dir: string) {}
  get(name: string): string {
    let t = this.cache.get(name);
    if (t === undefined) {
      t = readFileSync(path.join(this.dir, name), "utf8");
      this.cache.set(name, t);
    }
    return t;
  }
}

export class Workspace implements WorkspacePort {
  private current: string | null = null;
  constructor(private readonly pm: ProcessManager) {}
  set(root: string | null): void {
    this.current = root;
  }
  root(): string | null {
    return this.current;
  }
  async gitSummary(): Promise<string> {
    if (!this.current) return "";
    const r = await this.pm.run("git status --short --branch", this.current, 10, new AbortController().signal);
    if (r.exit_code !== 0) return "";
    const lines = r.stdout.trim().split("\n");
    const head = lines[0] ?? "";
    const changed = lines.length - 1;
    return `${head.replace(/^## /, "branch ")}; ${changed} changed file(s)${changed ? `: ${lines.slice(1, 15).map((l) => l.trim()).join(", ")}` : ""}`;
  }
}

export class ConsoleLogger implements LoggerPort {
  constructor(private readonly prefix = "[orchestrator]") {}
  info(msg: string, data?: unknown): void {
    console.log(this.prefix, msg, data ?? "");
  }
  warn(msg: string, data?: unknown): void {
    console.warn(this.prefix, msg, data ?? "");
  }
  error(msg: string, data?: unknown): void {
    console.error(this.prefix, msg, data ?? "");
  }
}
