import net from "node:net";
import ssh2 from "ssh2";
import { blobFingerprint } from "./adapters/ssh_server";

const { Client, utils } = ssh2;

/** How an IDE reaches a hub. */
export type Transport =
  /** Direct HTTP on the LAN or VPN. */
  | { kind: "http"; url: string }
  /** The hub's built-in SSH server (key auth); the API goes through an SSH tunnel. */
  | { kind: "ssh"; host: string; port: number; hostFingerprint?: string | null }
  /** Any SSH server (user@host) that can reach the hub at hubHost:hubPort, e.g. a team server. */
  | { kind: "jump"; host: string; port: number; username: string; hubHost: string; hubPort: number; hostFingerprint?: string | null };

export class HubHttpError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

/** JSON API client. */
export class HubClient {
  constructor(
    public baseUrl: string,
    public token: string | null,
  ) {}

  async request<T>(method: string, path: string, body?: unknown, timeoutMs = 30000): Promise<T> {
    const r = await fetch(`${this.baseUrl}${path}`, {
      method,
      headers: { ...(this.token ? { Authorization: `Bearer ${this.token}` } : {}), ...(body !== undefined ? { "Content-Type": "application/json" } : {}) },
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(timeoutMs),
    });
    const text = await r.text();
    if (!r.ok) {
      let msg = text;
      try {
        msg = (JSON.parse(text) as { error?: string }).error ?? text;
      } catch {
        /* plain text */
      }
      throw new HubHttpError(msg || `${r.status} ${r.statusText}`, r.status);
    }
    const ct = r.headers.get("content-type") ?? "";
    return (ct.includes("json") ? JSON.parse(text) : text) as T;
  }

  /** Server-sent events with automatic reconnect. Returns a stop function. */
  events(onEvent: (type: string, data: unknown) => void, onState?: (connected: boolean) => void): () => void {
    let stopped = false;
    let ac: AbortController | null = null;
    const loop = async () => {
      let delay = 1000;
      while (!stopped) {
        ac = new AbortController();
        try {
          const r = await fetch(`${this.baseUrl}/api/events`, { headers: { Authorization: `Bearer ${this.token}` }, signal: ac.signal });
          if (!r.ok || !r.body) throw new Error(`events ${r.status}`);
          onState?.(true);
          delay = 1000;
          const reader = r.body.getReader();
          const dec = new TextDecoder();
          let buf = "";
          for (;;) {
            const { value, done } = await reader.read();
            if (done) break;
            buf += dec.decode(value, { stream: true });
            let i: number;
            while ((i = buf.indexOf("\n\n")) >= 0) {
              const chunk = buf.slice(0, i);
              buf = buf.slice(i + 2);
              let type = "message";
              let data = "";
              for (const line of chunk.split("\n")) {
                if (line.startsWith("event:")) type = line.slice(6).trim();
                else if (line.startsWith("data:")) data += line.slice(5).trim();
              }
              if (data) {
                try {
                  onEvent(type, JSON.parse(data));
                } catch {
                  /* ignore malformed */
                }
              }
            }
          }
        } catch {
          /* reconnect below */
        }
        onState?.(false);
        if (stopped) break;
        await new Promise((r) => setTimeout(r, delay));
        delay = Math.min(delay * 2, 15000);
      }
    };
    void loop();
    return () => {
      stopped = true;
      ac?.abort();
    };
  }
}

export interface Tunnel {
  localPort: number;
  /** Host key fingerprint the SSH server presented. */
  fingerprint: string;
  close(): void;
  onClose(fn: () => void): void;
}

export interface TunnelAuth {
  username: string;
  privateKey?: string;
  password?: string;
}

/**
 * Opens an SSH connection and a local TCP port that forwards to `target` through it.
 * `expectFingerprint` pins the server's host key; without it, the key seen is returned for the caller to pin.
 */
export function openTunnel(opts: { host: string; port: number; auth: TunnelAuth; target: { host: string; port: number }; expectFingerprint?: string | null; timeoutMs?: number }): Promise<Tunnel> {
  return new Promise((resolve, reject) => {
    const conn = new Client();
    let seen = "";
    let mismatch: Error | null = null;
    let settled = false;
    const closers: Array<() => void> = [];
    const fail = (e: Error) => {
      if (settled) return;
      settled = true;
      conn.end();
      reject(mismatch ?? e);
    };
    conn.on("error", (e) => fail(e));
    conn.on("close", () => closers.forEach((f) => f()));
    conn.on("ready", () => {
      const server = net.createServer((sock) => {
        conn.forwardOut("127.0.0.1", 0, opts.target.host, opts.target.port, (err, ch) => {
          if (err) return sock.destroy();
          sock.pipe(ch).pipe(sock);
          sock.on("error", () => ch.close());
          ch.on("error", () => sock.destroy());
        });
      });
      server.on("error", (e) => fail(e));
      server.listen(0, "127.0.0.1", () => {
        settled = true;
        conn.on("close", () => server.close());
        resolve({
          localPort: (server.address() as net.AddressInfo).port,
          fingerprint: seen,
          close: () => {
            server.close();
            conn.end();
          },
          onClose: (fn) => closers.push(fn),
        });
      });
    });
    conn.connect({
      host: opts.host,
      port: opts.port,
      username: opts.auth.username,
      privateKey: opts.auth.privateKey,
      password: opts.auth.password,
      readyTimeout: opts.timeoutMs ?? 15000,
      keepaliveInterval: 15000,
      hostVerifier: (key: Buffer) => {
        seen = blobFingerprint(key);
        if (opts.expectFingerprint && opts.expectFingerprint !== seen) {
          mismatch = new Error(`host key mismatch for ${opts.host}: expected ${opts.expectFingerprint}, got ${seen}. Someone may be intercepting the connection.`);
          return false;
        }
        return true;
      },
    });
  });
}

/** An ed25519 identity for git over SSH and hub authentication. */
export function generateIdentity(comment: string): { privateKey: string; publicKey: string } {
  const kp = utils.generateKeyPairSync("ed25519", { comment });
  return { privateKey: kp.private, publicKey: kp.public.trim() };
}

/** URL of a project as a git remote for a connection. */
export function remoteUrl(t: Transport, hubId: string, group: string, project: string): string {
  if (t.kind === "http") return `${t.url.replace(/\/+$/, "")}/git/${group}/${project}.git`;
  if (t.kind === "ssh") return `ssh://git@${t.host.includes(":") ? `[${t.host}]` : t.host}:${t.port}/${group}/${project}.git`;
  return `rlm-hub://${hubId}/git/${group}/${project}.git`;
}

/** Prefix every remote of a connection starts with; used to find the connection of a workspace. */
export function remotePrefix(t: Transport, hubId: string): string {
  if (t.kind === "http") return `${t.url.replace(/\/+$/, "")}/git/`;
  if (t.kind === "ssh") return `ssh://git@${t.host.includes(":") ? `[${t.host}]` : t.host}:${t.port}/`;
  return `rlm-hub://${hubId}/`;
}

/**
 * Git `-c` options and environment that authenticate one git command against a hub.
 * `apiBase` is the reachable HTTP base (direct URL or the local tunnel port).
 */
export function gitAuth(t: Transport, hubId: string, token: string, apiBase: string, keyFile: string | null, knownHostsFile: string | null): { config: string[]; env: Record<string, string> } {
  const base = apiBase.replace(/\/+$/, "") + "/";
  const header = `Authorization: Bearer ${token}`;
  if (t.kind === "http") return { config: [`http.${base}.extraHeader=${header}`], env: { GIT_TERMINAL_PROMPT: "0" } };
  if (t.kind === "jump") return { config: [`url.${base}.insteadOf=rlm-hub://${hubId}/`, `http.${base}.extraHeader=${header}`], env: { GIT_TERMINAL_PROMPT: "0" } };
  const q = (s: string) => `"${s.replace(/\\/g, "/").replace(/"/g, '\\"')}"`;
  const parts = ["ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes"];
  if (keyFile) parts.push("-i", q(keyFile));
  if (knownHostsFile) parts.push("-o", `UserKnownHostsFile=${q(knownHostsFile)}`, "-o", "StrictHostKeyChecking=accept-new");
  return { config: [], env: { GIT_TERMINAL_PROMPT: "0", GIT_SSH_COMMAND: parts.join(" ") } };
}
