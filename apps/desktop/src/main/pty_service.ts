import { createRequire } from "node:module";
import os from "node:os";

const require = createRequire(__filename);

interface IPty {
  pid: number;
  write(d: string): void;
  resize(c: number, r: number): void;
  kill(): void;
  onData(cb: (d: string) => void): void;
  onExit(cb: (e: { exitCode: number }) => void): void;
}

/** Integrated terminal sessions (node-pty). */
export class PtyService {
  private readonly ptys = new Map<string, IPty>();
  private n = 0;

  constructor(
    private readonly onData: (id: string, data: string) => void,
    private readonly onExit: (id: string, code: number) => void,
  ) {}

  create(cwd: string, cols: number, rows: number): string {
    const pty = require("node-pty") as { spawn(file: string, args: string[], opts: Record<string, unknown>): IPty };
    const shell = process.platform === "win32" ? "powershell.exe" : process.env.SHELL || "/bin/bash";
    const p = pty.spawn(shell, [], { name: "xterm-256color", cols, rows, cwd: cwd || os.homedir(), env: { ...process.env, TERM: "xterm-256color" } });
    const id = `pty${++this.n}`;
    this.ptys.set(id, p);
    p.onData((d) => this.onData(id, d));
    p.onExit((e) => {
      this.ptys.delete(id);
      this.onExit(id, e.exitCode);
    });
    return id;
  }

  write(id: string, data: string): void {
    this.ptys.get(id)?.write(data);
  }

  resize(id: string, cols: number, rows: number): void {
    this.ptys.get(id)?.resize(cols, rows);
  }

  kill(id: string): void {
    this.ptys.get(id)?.kill();
    this.ptys.delete(id);
  }

  killAll(): void {
    for (const p of this.ptys.values()) p.kill();
    this.ptys.clear();
  }
}
