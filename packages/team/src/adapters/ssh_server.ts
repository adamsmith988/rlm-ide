import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { chmodSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import net from "node:net";
import ssh2 from "ssh2";
import type { User } from "../domain/types";
import type { GitHost } from "../ports";
import type { TeamService } from "../service";

const { Server, utils } = ssh2;

export interface SshHubOptions {
  host: string;
  port: number;
  hostKeyFile: string;
  /** Port of the hub's HTTP server on this machine; tunnels may only reach it. */
  httpPort: () => number;
  git: GitHost;
  isValidInvite: (code: string) => boolean;
  log?: (msg: string) => void;
}

export function loadOrCreateHostKey(file: string): string {
  if (existsSync(file)) return readFileSync(file, "utf8");
  const kp = utils.generateKeyPairSync("ed25519");
  writeFileSync(file, kp.private, { mode: 0o600 });
  try {
    chmodSync(file, 0o600);
  } catch {
    /* Windows */
  }
  return kp.private;
}

/** SHA256 fingerprint of a raw SSH public key blob. */
export function blobFingerprint(blob: Buffer): string {
  return `SHA256:${createHash("sha256").update(blob).digest("base64").replace(/=+$/, "")}`;
}

const GIT_CMD = /^(git-upload-pack|git-receive-pack|git upload-pack|git receive-pack) '?\/?([^/'\s]+)\/([^/'\s]+?)(?:\.git)?\/?'?$/;

type Session = { user: User | null; joinOnly: boolean };

/** SSH transport: git over SSH and port forwarding to the hub API, with key or invite authentication. */
export class SshHub {
  private server: InstanceType<typeof Server>;
  readonly hostKey: string;
  readonly fingerprint: string;

  constructor(
    private svc: TeamService,
    private opts: SshHubOptions,
  ) {
    this.hostKey = loadOrCreateHostKey(opts.hostKeyFile);
    const parsed = utils.parseKey(this.hostKey);
    if (parsed instanceof Error) throw parsed;
    this.fingerprint = blobFingerprint(parsed.getPublicSSH());
    this.server = new Server({ hostKeys: [this.hostKey], ident: "RLM-Team-Hub" }, (client) => this.onClient(client));
    this.server.on("error", (e: Error) => opts.log?.(`ssh server error: ${e.message}`));
  }

  listen(): Promise<{ port: number }> {
    return new Promise((resolve, reject) => {
      this.server.once("error", reject);
      this.server.listen(this.opts.port, this.opts.host, () => resolve({ port: (this.server.address() as net.AddressInfo).port }));
    });
  }

  close(): Promise<void> {
    return new Promise((r) => this.server.close(() => r()));
  }

  private onClient(client: ssh2.Connection): void {
    const session: Session = { user: null, joinOnly: false };
    client.on("error", () => undefined);
    client.on("authentication", (ctx) => {
      if (ctx.method === "publickey") {
        const user = this.svc.userByKeyFingerprint(blobFingerprint(ctx.key.data));
        if (!user) return ctx.reject(["publickey", "password"]);
        if (!ctx.signature) return ctx.accept();
        const stored = this.svc.me(user).keys.find((k) => k.fingerprint === blobFingerprint(ctx.key.data));
        const pk = stored ? utils.parseKey(stored.publicKey) : null;
        if (!pk || pk instanceof Error || !ctx.blob || !pk.verify(ctx.blob, ctx.signature, ctx.hashAlgo)) return ctx.reject(["publickey", "password"]);
        session.user = user;
        return ctx.accept();
      }
      if (ctx.method === "password" && ctx.username === "join" && this.opts.isValidInvite(ctx.password)) {
        session.joinOnly = true;
        return ctx.accept();
      }
      if (ctx.method === "password" && ctx.username === "token") {
        try {
          session.user = this.svc.authenticate(ctx.password);
          return ctx.accept();
        } catch {
          /* fall through */
        }
      }
      ctx.reject(["publickey", "password"]);
    });

    client.on("ready", () => {
      client.on("tcpip", (accept, reject, info) => {
        const local = ["127.0.0.1", "localhost", "::1"].includes(info.destIP) && info.destPort === this.opts.httpPort();
        if (!local && info.destIP !== "hub") return reject();
        const ch = accept();
        const sock = net.connect(this.opts.httpPort(), "127.0.0.1");
        ch.pipe(sock).pipe(ch);
        sock.on("error", () => ch.close());
        ch.on("error", () => sock.destroy());
      });
      client.on("session", (acceptSession) => {
        const s = acceptSession();
        const env: Record<string, string> = {};
        s.on("env", (accept, _reject, info) => {
          if (info.key === "GIT_PROTOCOL") env.GIT_PROTOCOL = info.val;
          accept?.();
        });
        s.on("pty", (accept) => accept?.());
        s.on("shell", (accept) => {
          const ch = accept();
          ch.write(`Hi${session.user ? ` ${session.user.name}` : ""}! This is an RLM Team Hub. It serves git (git clone ssh://...) and API tunnels, not a shell.\r\n`);
          ch.exit(0);
          ch.end();
        });
        s.on("exec", (accept, reject, info) => {
          if (session.joinOnly || !session.user) return reject?.();
          const m = GIT_CMD.exec(info.command.trim());
          if (!m) {
            const ch = accept();
            ch.stderr.write(`unsupported command: ${info.command}\n`);
            ch.exit(1);
            return ch.end();
          }
          void this.runGit(accept, session.user, m[1]!.includes("receive") ? "receive-pack" : "upload-pack", m[2]!, m[3]!, env);
        });
      });
    });
  }

  private async runGit(accept: () => ssh2.ServerChannel, user: User, service: "upload-pack" | "receive-pack", g: string, p: string, extraEnv: Record<string, string>): Promise<void> {
    const ch = accept();
    const fail = (msg: string) => {
      ch.stderr.write(`RLM Hub: ${msg}\n`);
      ch.exit(1);
      ch.end();
    };
    let repo: string;
    const env: NodeJS.ProcessEnv = { ...process.env, ...extraEnv, GIT_COMMITTER_NAME: user.name, GIT_COMMITTER_EMAIL: user.email };
    try {
      repo = this.svc.access(user, g, p).repo;
      if (service === "receive-pack") {
        const pol = await this.svc.pushPolicy(user, g, p);
        if (!pol.allowed) return fail(pol.reason ?? "push not allowed");
        if (pol.protectedBranch) env.RLM_PROTECTED_BRANCH = pol.protectedBranch;
      }
    } catch (e) {
      return fail(e instanceof Error ? e.message : String(e));
    }
    const before = service === "receive-pack" ? await this.opts.git.refs(repo) : null;
    const child = spawn("git", [service, repo], { env, windowsHide: true });
    ch.pipe(child.stdin);
    child.stdout.pipe(ch, { end: false });
    child.stderr.pipe(ch.stderr, { end: false });
    child.on("close", async (code) => {
      if (before) await this.svc.recordPush(user, g, p, before, await this.opts.git.refs(repo), "ssh").catch((e: unknown) => this.opts.log?.(`push record failed: ${String(e)}`));
      ch.exit(code ?? 1);
      ch.end();
    });
    child.on("error", (e) => fail(e.message));
  }
}