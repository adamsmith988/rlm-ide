import { chmodSync, existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { randomUUID } from "node:crypto";
import {
  BeaconListener,
  generateIdentity,
  gitAuth,
  HubClient,
  openTunnel,
  remotePrefix,
  remoteUrl,
  runGit,
  startHub,
  type RunningHub,
  type Transport,
  type Tunnel,
} from "@rlm/team";
import type { TeamConnectInput, TeamHostConfig, TeamStateView } from "@rlm/contracts";
import type { SecretStore } from "./settings_store";

interface Connection {
  id: string;
  label: string;
  hubId: string;
  hubName: string;
  transport: Transport;
  userId: string;
  userName: string;
  /** For jump hosts: log in with the stored password instead of the IDE key. */
  jumpPassword?: boolean;
}

interface Persisted {
  profile: { name: string; email: string };
  host: TeamHostConfig;
  connections: Connection[];
}

interface Runtime {
  base: string;
  client: HubClient;
  tunnel: Tunnel | null;
  stopEvents: () => void;
  online: boolean;
}

const DEFAULT_HOST: TeamHostConfig = { enabled: false, name: "", lan: true, httpPort: 7420, sshPort: 7422, discovery: true };
const LOCAL = "local";

/** The IDE side of team hubs: hosting one, connecting to others, and git against them. */
export class TeamManager {
  private dir: string;
  private data: Persisted;
  private hub: RunningHub | null = null;
  private hostError: string | null = null;
  private runtimes = new Map<string, Runtime>();
  private connecting = new Map<string, Promise<Runtime>>();
  private errors = new Map<string, string>();
  private listener: BeaconListener | null = null;

  constructor(
    userData: string,
    private secrets: SecretStore,
    private emit: (channel: "team:event" | "team:state", payload: unknown) => void,
    private log: (msg: string) => void,
  ) {
    this.dir = path.join(userData, "team");
    mkdirSync(this.dir, { recursive: true });
    const file = path.join(this.dir, "state.json");
    let raw: Partial<Persisted> = {};
    if (existsSync(file)) {
      try {
        raw = JSON.parse(readFileSync(file, "utf8")) as Partial<Persisted>;
      } catch {
        raw = {};
      }
    }
    this.data = {
      profile: raw.profile ?? { name: os.userInfo().username, email: "" },
      host: { ...DEFAULT_HOST, ...raw.host },
      connections: raw.connections ?? [],
    };
  }

  // ---- identity -----------------------------------------------------------------------------------

  private get keyFile() {
    return path.join(this.dir, "id_ed25519");
  }
  private get knownHosts() {
    return path.join(this.dir, "known_hosts");
  }

  /** The IDE's SSH identity, created on first use. The private key never leaves this machine. */
  private identity(): { privateKey: string; publicKey: string } {
    const pub = `${this.keyFile}.pub`;
    if (existsSync(this.keyFile) && existsSync(pub)) return { privateKey: readFileSync(this.keyFile, "utf8"), publicKey: readFileSync(pub, "utf8").trim() };
    const id = generateIdentity(`${this.data.profile.email || this.data.profile.name}@rlm-ide`);
    writeFileSync(this.keyFile, id.privateKey, { mode: 0o600 });
    try {
      chmodSync(this.keyFile, 0o600);
    } catch {
      /* Windows */
    }
    writeFileSync(pub, `${id.publicKey}\n`);
    return id;
  }

  private save(): void {
    const file = path.join(this.dir, "state.json");
    writeFileSync(`${file}.tmp`, JSON.stringify(this.data, null, 2));
    renameSync(`${file}.tmp`, file);
  }

  private changed(): void {
    this.emit("team:state", this.state());
  }

  state(): TeamStateView {
    if (!this.listener) this.startDiscovery();
    return {
      profile: this.data.profile,
      publicKey: this.identity().publicKey,
      host: this.data.host,
      hosting: this.hub
        ? { hubId: this.hub.hubId, httpPort: this.hub.httpPort, sshPort: this.hub.sshPort, sshFingerprint: this.hub.sshFingerprint, addresses: this.hub.addresses }
        : null,
      hostError: this.hostError,
      connections: this.data.connections.map((c) => ({
        id: c.id,
        label: c.label,
        hubId: c.hubId,
        hubName: c.hubName,
        transport: c.transport,
        userName: c.userName,
        status: this.runtimes.get(c.id)?.online ? "online" : this.connecting.has(c.id) ? "connecting" : this.errors.has(c.id) ? "error" : "offline",
        error: this.errors.get(c.id) ?? null,
      })),
      discovered: (this.listener?.list() ?? []).filter((d) => d.hubId !== this.hub?.hubId),
    };
  }

  setProfile(p: { name: string; email: string }): TeamStateView {
    const name = p.name.trim();
    const email = p.email.trim().toLowerCase();
    if (!name || !/^[^@\s]+@[^@\s]+$/.test(email)) throw new Error("enter your name and a valid email; teammates see them on commits and bug reports");
    this.data.profile = { name, email };
    this.save();
    this.changed();
    return this.state();
  }

  private startDiscovery(): void {
    this.listener = new BeaconListener(() => this.changed());
    void this.listener.start();
  }

  // ---- hosting ------------------------------------------------------------------------------------

  async setHost(cfg: TeamHostConfig): Promise<TeamStateView> {
    this.data.host = { ...DEFAULT_HOST, ...cfg };
    this.save();
    await this.stopHub();
    if (this.data.host.enabled) await this.startHosting();
    this.changed();
    return this.state();
  }

  /** Called at startup; brings the hub back if it was on. */
  /** `host: false` in every copy of the IDE but the first: the first one already serves the team hub on its ports. */
  async boot(opts: { host?: boolean } = {}): Promise<void> {
    if (this.data.host.enabled && opts.host !== false) await this.startHosting();
    else if (this.data.host.enabled) this.hostError = "hosted by the first RLM IDE window on this computer";
    for (const c of this.data.connections) void this.runtime(c.id).catch(() => undefined);
  }

  private async startHosting(): Promise<void> {
    const h = this.data.host;
    this.hostError = null;
    if (!this.data.profile.email) {
      this.hostError = "set your name and email first";
      return;
    }
    try {
      this.hub = await startHub({
        dataDir: path.join(this.dir, "hub"),
        name: h.name.trim() || `${this.data.profile.name}'s hub`,
        owner: this.data.profile,
        bindHost: h.lan ? "0.0.0.0" : "127.0.0.1",
        httpPort: h.httpPort,
        sshPort: h.sshPort,
        discovery: h.discovery,
        log: this.log,
      });
    } catch (e) {
      this.hostError = e instanceof Error ? e.message : String(e);
      return;
    }
    // The host's own IDE is a client like any other, over loopback.
    const transport: Transport = { kind: "http", url: `http://127.0.0.1:${this.hub.httpPort}` };
    const existing = this.data.connections.find((c) => c.id === LOCAL);
    const conn: Connection = { id: LOCAL, label: "This machine", hubId: this.hub.hubId, hubName: h.name.trim() || `${this.data.profile.name}'s hub`, transport, userId: this.hub.ownerId, userName: this.data.profile.name };
    this.data.connections = existing ? this.data.connections.map((c) => (c.id === LOCAL ? conn : c)) : [conn, ...this.data.connections];
    this.secrets.set(`team:${LOCAL}`, this.hub.ownerToken);
    this.save();
    this.dropRuntime(LOCAL);
    const rt = await this.runtime(LOCAL).catch(() => null);
    // Lets the host use git over SSH on its own hub too.
    await rt?.client.request("POST", "/api/me/keys", { publicKey: this.identity().publicKey }).catch(() => undefined);
  }

  private async stopHub(): Promise<void> {
    this.dropRuntime(LOCAL);
    await this.hub?.close().catch(() => undefined);
    this.hub = null;
  }

  // ---- connections --------------------------------------------------------------------------------

  private async openBase(t: Transport, auth: { token?: string | null; joinCode?: string; password?: string; username?: string }): Promise<{ base: string; tunnel: Tunnel | null }> {
    if (t.kind === "http") return { base: t.url.replace(/\/+$/, ""), tunnel: null };
    const known = t.hostFingerprint ?? null;
    if (t.kind === "ssh") {
      const tunnelAuth = auth.joinCode ? { username: "join", password: auth.joinCode } : { username: "git", privateKey: this.identity().privateKey };
      const tunnel = await openTunnel({ host: t.host, port: t.port, auth: tunnelAuth, target: { host: "hub", port: 0 }, expectFingerprint: known });
      return { base: `http://127.0.0.1:${tunnel.localPort}`, tunnel };
    }
    const tunnel = await openTunnel({
      host: t.host,
      port: t.port,
      auth: auth.password ? { username: t.username, password: auth.password } : { username: t.username, privateKey: this.identity().privateKey },
      target: { host: t.hubHost, port: t.hubPort },
      expectFingerprint: known,
    });
    return { base: `http://127.0.0.1:${tunnel.localPort}`, tunnel };
  }

  /** Joins a hub with an invite code, or signs in with an existing token. */
  async connect(input: TeamConnectInput): Promise<TeamStateView> {
    if (!this.data.profile.email) throw new Error("set your name and email first");
    const t = input.transport as Transport;
    if (!input.code && !input.token) throw new Error("enter an invite code or an access token");
    const password = t.kind === "jump" ? input.password : undefined;
    const { base, tunnel } = await this.openBase(t, { joinCode: t.kind === "ssh" && input.code ? input.code : undefined, password });
    try {
      const pinned: Transport = tunnel && t.kind !== "http" ? { ...t, hostFingerprint: tunnel.fingerprint } : t;
      const anon = new HubClient(base, null);
      const info = await anon.request<{ hubId: string; name: string }>("GET", "/api/info", undefined, 10000);
      const same = this.data.connections.find((c) => c.hubId === info.hubId);
      const prevToken = same ? this.secrets.get(`team:${same.id}`) : null;
      let token = input.token?.trim() || prevToken;
      if (input.code) {
        const r = await new HubClient(base, prevToken).request<{ token: string | null }>("POST", "/api/join", { code: input.code.trim(), name: this.data.profile.name, email: this.data.profile.email, publicKey: this.identity().publicKey });
        token = r.token ?? prevToken;
      }
      if (!token) throw new Error("the hub did not return an access token");
      const me = await new HubClient(base, token).request<{ id: string; name: string }>("GET", "/api/me");
      await new HubClient(base, token).request("POST", "/api/me/keys", { publicKey: this.identity().publicKey }).catch(() => undefined);
      const conn: Connection = {
        id: same?.id ?? randomUUID(),
        label: input.label?.trim() || (t.kind === "http" ? t.url : t.kind === "ssh" ? `ssh ${t.host}:${t.port}` : `${t.username}@${t.host}`),
        hubId: info.hubId,
        hubName: info.name,
        transport: same && same.id === LOCAL ? same.transport : pinned,
        userId: me.id,
        userName: me.name,
        jumpPassword: t.kind === "jump" && !!password,
      };
      this.secrets.set(`team:${conn.id}`, token);
      if (password) this.secrets.set(`teamjump:${conn.id}`, password);
      this.data.connections = same ? this.data.connections.map((c) => (c.id === conn.id ? conn : c)) : [...this.data.connections, conn];
      this.save();
    } finally {
      tunnel?.close();
    }
    for (const c of this.data.connections) if (c.hubId) this.dropRuntime(c.id);
    for (const c of this.data.connections) void this.runtime(c.id).catch(() => undefined);
    this.changed();
    return this.state();
  }

  disconnect(id: string): TeamStateView {
    this.dropRuntime(id);
    if (id !== LOCAL) {
      this.data.connections = this.data.connections.filter((c) => c.id !== id);
      this.secrets.set(`team:${id}`, "");
      this.secrets.set(`teamjump:${id}`, "");
      this.save();
    }
    this.changed();
    return this.state();
  }

  async reconnect(id: string): Promise<TeamStateView> {
    this.dropRuntime(id);
    await this.runtime(id).catch(() => undefined);
    this.changed();
    return this.state();
  }

  private conn(id: string): Connection {
    const c = this.data.connections.find((x) => x.id === id);
    if (!c) throw new Error("unknown team connection");
    return c;
  }

  private dropRuntime(id: string): void {
    const rt = this.runtimes.get(id);
    if (!rt) return;
    rt.stopEvents();
    rt.tunnel?.close();
    this.runtimes.delete(id);
  }

  /** A live API session for a connection, opening the tunnel if the transport needs one. */
  private runtime(id: string): Promise<Runtime> {
    const have = this.runtimes.get(id);
    if (have) return Promise.resolve(have);
    const pending = this.connecting.get(id);
    if (pending) return pending;
    const p = (async () => {
      const c = this.conn(id);
      const token = this.secrets.get(`team:${id}`);
      if (!token) throw new Error("no access token stored for this hub; join it again");
      const { base, tunnel } = await this.openBase(c.transport, { password: c.jumpPassword ? (this.secrets.get(`teamjump:${id}`) ?? undefined) : undefined });
      const client = new HubClient(base, token);
      await client.request("GET", "/api/me", undefined, 10000);
      const rt: Runtime = { base, client, tunnel, online: true, stopEvents: () => undefined };
      rt.stopEvents = client.events(
        (type, data) => this.emit("team:event", { connId: id, type, data }),
        (online) => {
          if (rt.online !== online) {
            rt.online = online;
            this.changed();
          }
        },
      );
      tunnel?.onClose(() => {
        if (this.runtimes.get(id) === rt) {
          this.dropRuntime(id);
          this.errors.set(id, "connection closed");
          this.changed();
        }
      });
      this.runtimes.set(id, rt);
      this.errors.delete(id);
      return rt;
    })();
    this.connecting.set(id, p);
    this.changed();
    p.catch((e: unknown) => this.errors.set(id, e instanceof Error ? e.message : String(e))).finally(() => {
      this.connecting.delete(id);
      this.changed();
    });
    return p;
  }

  /** Proxies one hub API call for the renderer. Only /api paths, always with this connection's token. */
  async call(id: string, method: string, apiPath: string, body?: unknown): Promise<unknown> {
    if (!/^\/api\/[\w\-./%?=&:+@]*$/.test(apiPath) || apiPath.includes("..")) throw new Error("bad hub path");
    const rt = await this.runtime(id);
    return rt.client.request(method, apiPath, body);
  }

  // ---- git against hubs ---------------------------------------------------------------------------

  private async gitFor(id: string): Promise<{ config: string[]; env: Record<string, string> }> {
    const c = this.conn(id);
    const token = this.secrets.get(`team:${id}`) ?? "";
    const base = c.transport.kind === "ssh" ? "" : (await this.runtime(id)).base;
    if (c.transport.kind === "ssh") this.identity();
    const a = gitAuth(c.transport, c.hubId, token, base, this.keyFile, this.knownHosts);
    return { config: a.config.flatMap((x) => ["-c", x]), env: a.env };
  }

  /** Commits in hub repos carry the hub identity, so bug reports can find their author. Keeps a repo's own setting. */
  private async adoptIdentity(cwd: string): Promise<void> {
    if ((await runGit(["config", "--local", "user.email"], { cwd })).stdout.trim()) return;
    await runGit(["config", "--local", "user.name", this.data.profile.name], { cwd });
    await runGit(["config", "--local", "user.email", this.data.profile.email], { cwd });
  }

  private identityArgs(cwd: string): Promise<string[]> {
    return runGit(["config", "user.email"], { cwd }).then((r) => (r.stdout.trim() ? [] : ["-c", `user.name=${this.data.profile.name}`, "-c", `user.email=${this.data.profile.email}`]));
  }

  /** Which hub project a workspace belongs to, from its git remotes. */
  async workspaceProject(root: string): Promise<{ connId: string; group: string; project: string; remote: string } | null> {
    const r = await runGit(["remote", "-v"], { cwd: root });
    for (const line of r.stdout.split("\n")) {
      const [name, url] = line.split(/\s+/);
      if (!name || !url) continue;
      for (const c of this.data.connections) {
        const prefix = remotePrefix(c.transport, c.hubId);
        if (!url.startsWith(prefix)) continue;
        const m = /^(?:git\/)?([^/]+)\/([^/]+?)(?:\.git)?\/?$/.exec(url.slice(prefix.length));
        if (m) return { connId: c.id, group: m[1]!, project: m[2]!, remote: name };
      }
    }
    return null;
  }

  async git(root: string, op: "push" | "pull" | "fetch", opts: { branch?: string }): Promise<string> {
    const wp = await this.workspaceProject(root);
    const head = (await runGit(["rev-parse", "--abbrev-ref", "HEAD"], { cwd: root })).stdout.trim();
    const branch = opts.branch || head;
    // Remotes that are not on a team hub use the user's own git credentials.
    const auth = wp ? await this.gitFor(wp.connId) : { config: [], env: {} };
    const remote = wp?.remote ?? "origin";
    const args =
      op === "push" ? ["push", "--porcelain", "-u", remote, `HEAD:refs/heads/${branch}`]
      : op === "pull" ? [...(await this.identityArgs(root)), "pull", "--no-rebase", "--no-edit", remote, branch]
      : ["fetch", "--prune", remote];
    const r = await runGit([...auth.config, ...args], { cwd: root, env: auth.env });
    if (r.code !== 0) throw new Error(cleanGitError(r.stderr || r.stdout));
    return (r.stdout + r.stderr).trim();
  }

  async clone(id: string, group: string, project: string, parent: string): Promise<string> {
    const c = this.conn(id);
    const dest = path.join(parent, project);
    if (existsSync(dest)) throw new Error(`${dest} already exists`);
    const auth = await this.gitFor(id);
    const r = await runGit([...auth.config, "clone", remoteUrl(c.transport, c.hubId, group, project), dest], { env: auth.env });
    if (r.code !== 0) throw new Error(cleanGitError(r.stderr));
    await this.adoptIdentity(dest);
    return dest;
  }

  /** Publishes the open workspace as a new hub project and pushes it. */
  async share(root: string, id: string, group: string, p: { name: string; description?: string }): Promise<{ group: string; project: string }> {
    const c = this.conn(id);
    if (!existsSync(path.join(root, ".git"))) {
      const init = await runGit(["init", "-q", "-b", "main"], { cwd: root });
      if (init.code !== 0) throw new Error(cleanGitError(init.stderr));
    }
    await this.adoptIdentity(root);
    const ident = await this.identityArgs(root);
    if ((await runGit(["rev-parse", "--verify", "-q", "HEAD"], { cwd: root })).code !== 0) {
      await runGit(["add", "-A"], { cwd: root });
      const commit = await runGit([...ident, "commit", "-q", "--allow-empty", "-m", "Initial commit"], { cwd: root });
      if (commit.code !== 0) throw new Error(cleanGitError(commit.stderr));
    }
    const branch = (await runGit(["rev-parse", "--abbrev-ref", "HEAD"], { cwd: root })).stdout.trim() || "main";
    const rt = await this.runtime(id);
    const created = await rt.client.request<{ slug: string }>("POST", `/api/groups/${encodeURIComponent(group)}/projects`, { name: p.name, description: p.description, defaultBranch: branch });
    const url = remoteUrl(c.transport, c.hubId, group, created.slug);
    const remotes = (await runGit(["remote"], { cwd: root })).stdout.split("\n").filter(Boolean);
    const name = remotes.includes("origin") ? "hub" : "origin";
    if (remotes.includes(name)) await runGit(["remote", "set-url", name, url], { cwd: root });
    else await runGit(["remote", "add", name, url], { cwd: root });
    const auth = await this.gitFor(id);
    const push = await runGit([...auth.config, "push", "-u", name, `HEAD:refs/heads/${branch}`], { cwd: root, env: auth.env });
    if (push.code !== 0) throw new Error(cleanGitError(push.stderr));
    return { group, project: created.slug };
  }

  /** Files a bug from the editor, adding what only this machine knows: the local blame and branch. */
  async reportBug(root: string | null, input: Record<string, unknown> & { connId?: string; group?: string; project?: string; file?: string; line?: number }): Promise<unknown> {
    let { connId, group, project } = input;
    const extra: Record<string, unknown> = {};
    if (root) {
      const wp = await this.workspaceProject(root);
      if (wp && !connId) ({ connId, group, project } = wp);
      const branch = (await runGit(["rev-parse", "--abbrev-ref", "HEAD"], { cwd: root })).stdout.trim();
      if (branch && !input.ref) extra.ref = branch;
      if (input.file && input.line && !input.commit) {
        const b = await runGit(["blame", "--porcelain", "-L", `${input.line},${input.line}`, "--", String(input.file)], { cwd: root });
        const sha = b.stdout.split(" ")[0];
        if (b.code === 0 && sha && !/^0+$/.test(sha)) extra.commit = sha;
      }
    }
    if (!connId || !group || !project) throw new Error("pick the hub project this bug belongs to");
    const env = { os: `${os.type()} ${os.release()} (${os.arch()})`, ...(input.environment as Record<string, string> | undefined) };
    const { connId: _c, group: _g, project: _p, ...rest } = input;
    return this.call(connId, "POST", `/api/projects/${encodeURIComponent(group)}/${encodeURIComponent(project)}/bugs`, { ...rest, ...extra, environment: env });
  }

  async close(): Promise<void> {
    for (const id of [...this.runtimes.keys()]) this.dropRuntime(id);
    this.listener?.stop();
    await this.hub?.close().catch(() => undefined);
  }
}

function cleanGitError(s: string): string {
  const lines = s.split("\n").map((l) => l.replace(/^remote:\s*/, "").trim()).filter((l) => l && !/^(hint|To |error: failed to push)/.test(l));
  return lines.slice(-6).join("\n") || "git failed";
}
