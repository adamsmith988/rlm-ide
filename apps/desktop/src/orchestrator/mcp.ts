import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import type { McpServerConfig } from "@rlm/contracts";

export interface McpToolInfo {
  server: string;
  name: string;
  description: string;
}

interface Pending {
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
}

interface Conn {
  cfg: McpServerConfig;
  child: ChildProcessWithoutNullStreams;
  tools: McpToolInfo[];
  error: string | null;
  buf: Buffer;
  next: number;
  wait: Map<number, Pending>;
}

/** Speaks MCP over stdio (Content-Length frames) with the servers in Settings. */
export class McpHub {
  private conns = new Map<string, Conn>();
  private generation = 0;

  constructor(private readonly log: (m: string) => void) {}

  async sync(servers: McpServerConfig[]): Promise<void> {
    const gen = ++this.generation;
    const wanted = new Map(servers.filter((s) => s.enabled && s.command.trim()).map((s) => [s.id, s]));
    for (const [id, conn] of this.conns) {
      if (!wanted.has(id) || JSON.stringify(publicCfg(wanted.get(id)!)) !== JSON.stringify(publicCfg(conn.cfg))) this.stop(id);
    }
    await Promise.all(
      [...wanted.values()].map(async (cfg) => {
        if (this.conns.has(cfg.id) || gen !== this.generation) return;
        try {
          const conn = await this.start(cfg);
          if (gen !== this.generation) {
            conn.child.kill();
            return;
          }
          this.conns.set(cfg.id, conn);
        } catch (e) {
          this.log(`mcp ${cfg.name}: ${e instanceof Error ? e.message : String(e)}`);
          this.conns.set(cfg.id, dead(cfg, e instanceof Error ? e.message : String(e)));
        }
      }),
    );
  }

  status(): Array<{ id: string; name: string; error: string | null; tools: McpToolInfo[] }> {
    return [...this.conns.values()].map((c) => ({ id: c.cfg.id, name: c.cfg.name, error: c.error, tools: c.tools }));
  }

  async call(server: string, name: string, args: Record<string, unknown>): Promise<unknown> {
    const conn = [...this.conns.values()].find((c) => c.cfg.name === server || c.cfg.id === server);
    if (!conn) throw new Error(`no MCP server "${server}". Add one in Settings → Skills & MCP.`);
    if (conn.error || !conn.child.stdin.writable) throw new Error(conn.error ?? `${conn.cfg.name} is not running`);
    const result = (await this.rpc(conn, "tools/call", { name, arguments: args })) as { content?: Array<{ type?: string; text?: string }>; isError?: boolean };
    const text = (result.content ?? []).map((p) => p.text ?? "").join("\n");
    if (result.isError) throw new Error(text || `${name} failed`);
    return text || result;
  }

  stopAll(): void {
    for (const id of [...this.conns.keys()]) this.stop(id);
  }

  private stop(id: string): void {
    const conn = this.conns.get(id);
    this.conns.delete(id);
    if (!conn) return;
    for (const p of conn.wait.values()) p.reject(new Error("MCP server stopped"));
    conn.child.kill();
  }

  private async start(cfg: McpServerConfig): Promise<Conn> {
    const child = spawn(cfg.command, splitArgs(cfg.args), {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, ...parseEnv(cfg.env) },
      windowsHide: true,
    });
    const conn: Conn = { cfg, child, tools: [], error: null, buf: Buffer.alloc(0), next: 1, wait: new Map() };
    child.stdout.on("data", (d: Buffer) => this.onData(conn, d));
    child.stderr.on("data", (d: Buffer) => this.log(`mcp ${cfg.name}: ${d.toString("utf8").trim().slice(0, 300)}`));
    child.on("exit", () => {
      conn.error = conn.error ?? "server exited";
    });
    try {
      await this.rpc(conn, "initialize", {
        protocolVersion: "2024-11-05",
        capabilities: {},
        clientInfo: { name: "rlm-ide", version: "0.1.0" },
      });
      this.notify(conn, "notifications/initialized", {});
      const listed = (await this.rpc(conn, "tools/list", {})) as { tools?: Array<{ name: string; description?: string }> };
      conn.tools = (listed.tools ?? []).map((t) => ({ server: cfg.name, name: t.name, description: t.description ?? "" }));
      return conn;
    } catch (e) {
      child.kill();
      throw e;
    }
  }

  private notify(conn: Conn, method: string, params: unknown): void {
    this.write(conn, { jsonrpc: "2.0", method, params });
  }

  private rpc(conn: Conn, method: string, params: unknown): Promise<unknown> {
    const id = conn.next++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        conn.wait.delete(id);
        reject(new Error(`${conn.cfg.name} ${method} timed out`));
      }, 20000);
      conn.wait.set(id, {
        resolve: (v) => {
          clearTimeout(timer);
          resolve(v);
        },
        reject: (e) => {
          clearTimeout(timer);
          reject(e);
        },
      });
      this.write(conn, { jsonrpc: "2.0", id, method, params });
    });
  }

  private write(conn: Conn, msg: unknown): void {
    const body = Buffer.from(JSON.stringify(msg));
    conn.child.stdin.write(`Content-Length: ${body.length}\r\n\r\n`);
    conn.child.stdin.write(body);
  }

  private onData(conn: Conn, d: Buffer): void {
    conn.buf = Buffer.concat([conn.buf, d]);
    for (;;) {
      const headerEnd = conn.buf.indexOf("\r\n\r\n");
      if (headerEnd < 0) return;
      const header = conn.buf.subarray(0, headerEnd).toString("utf8");
      const m = /Content-Length:\s*(\d+)/i.exec(header);
      if (!m) {
        conn.buf = conn.buf.subarray(headerEnd + 4);
        continue;
      }
      const len = Number(m[1]);
      const start = headerEnd + 4;
      if (conn.buf.length < start + len) return;
      const raw = conn.buf.subarray(start, start + len).toString("utf8");
      conn.buf = conn.buf.subarray(start + len);
      let msg: { id?: number; result?: unknown; error?: { message?: string } };
      try {
        msg = JSON.parse(raw) as typeof msg;
      } catch {
        continue;
      }
      if (msg.id === undefined) continue;
      const pending = conn.wait.get(msg.id);
      if (!pending) continue;
      conn.wait.delete(msg.id);
      if (msg.error) pending.reject(new Error(msg.error.message ?? "MCP error"));
      else pending.resolve(msg.result);
    }
  }
}

function dead(cfg: McpServerConfig, error: string): Conn {
  return {
    cfg,
    child: { stdin: { writable: false }, kill() {}, stdout: { on() {} }, stderr: { on() {} }, on() {} } as unknown as ChildProcessWithoutNullStreams,
    tools: [],
    error,
    buf: Buffer.alloc(0),
    next: 1,
    wait: new Map(),
  };
}

function publicCfg(c: McpServerConfig): Omit<McpServerConfig, "id"> {
  return { name: c.name, command: c.command, args: c.args, env: c.env, enabled: c.enabled };
}

/** Splits a command-line argument string, keeping quoted pieces together. */
export function splitArgs(text: string): string[] {
  const out: string[] = [];
  const re = /"([^"]*)"|'([^']*)'|(\S+)/g;
  for (const m of text.matchAll(re)) out.push(m[1] ?? m[2] ?? m[3] ?? "");
  return out.filter(Boolean);
}

export function parseEnv(text: string): Record<string, string> {
  const env: Record<string, string> = {};
  for (const line of text.split(/\r?\n/)) {
    const t = line.trim();
    if (!t || t.startsWith("#")) continue;
    const i = t.indexOf("=");
    if (i > 0) env[t.slice(0, i).trim()] = t.slice(i + 1);
  }
  return env;
}
