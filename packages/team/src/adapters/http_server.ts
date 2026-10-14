import { spawn } from "node:child_process";
import http from "node:http";
import type { AddressInfo } from "node:net";
import path from "node:path";
import { TeamError, type User } from "../domain/types";
import type { HubEvent, TeamService } from "../service";
import type { GitHost } from "../ports";

export interface HttpHubOptions {
  host: string;
  port: number;
  reposRoot: string;
  git: GitHost;
  /** Extra public info (SSH port and host key fingerprint). */
  publicInfo: () => Record<string, unknown>;
  log?: (msg: string) => void;
}

type Handler = (ctx: { req: http.IncomingMessage; res: http.ServerResponse; params: string[]; query: URLSearchParams; body: () => Promise<Record<string, unknown>>; user: () => User; maybeUser: () => User | null }) => Promise<unknown>;

const MAX_BODY = 5 * 1024 * 1024;

function bearer(req: http.IncomingMessage): string | null {
  const h = req.headers.authorization ?? "";
  if (/^bearer /i.test(h)) return h.slice(7).trim();
  if (/^basic /i.test(h)) {
    const [, pass] = Buffer.from(h.slice(6), "base64").toString("utf8").split(/:(.*)/s);
    return pass || null;
  }
  return null;
}

function readBody(req: http.IncomingMessage): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (d: Buffer) => {
      size += d.length;
      if (size > MAX_BODY) {
        reject(new TeamError("request body too large"));
        req.destroy();
      } else chunks.push(d);
    });
    req.on("end", () => {
      if (!size) return resolve({});
      try {
        const v = JSON.parse(Buffer.concat(chunks).toString("utf8")) as unknown;
        resolve(v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {});
      } catch {
        reject(new TeamError("body is not JSON"));
      }
    });
    req.on("error", reject);
  });
}

const str = (v: unknown) => (typeof v === "string" ? v : v === undefined || v === null ? undefined : String(v));
const num = (v: string | undefined) => {
  const x = Number(v);
  if (!Number.isInteger(x) || x <= 0) throw new TeamError("bad number", 404);
  return x;
};

/** The hub's HTTP face: JSON API under /api, git smart HTTP under /git, live events on /api/events. */
export class HttpHub {
  private server: http.Server;
  private routes: Array<{ method: string; re: RegExp; fn: Handler }> = [];
  private sse = new Set<{ user: User; res: http.ServerResponse; groups: Set<string> }>();
  private off: () => void;

  constructor(
    private svc: TeamService,
    private opts: HttpHubOptions,
  ) {
    this.defineRoutes();
    this.server = http.createServer((req, res) => void this.handle(req, res));
    this.server.keepAliveTimeout = 30_000;
    this.off = svc.on((e) => this.broadcast(e));
  }

  listen(): Promise<{ port: number }> {
    return new Promise((resolve, reject) => {
      this.server.once("error", reject);
      this.server.listen(this.opts.port, this.opts.host, () => resolve({ port: (this.server.address() as AddressInfo).port }));
    });
  }

  close(): Promise<void> {
    this.off();
    for (const c of this.sse) c.res.end();
    this.sse.clear();
    return new Promise((r) => {
      this.server.close(() => r());
      this.server.closeAllConnections?.();
    });
  }

  private route(method: string, pattern: string, fn: Handler): void {
    const re = new RegExp(`^${pattern.replace(/:(\w+)/g, "([^/]+)")}$`);
    this.routes.push({ method, re, fn });
  }

  private async handle(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    const url = new URL(req.url ?? "/", "http://hub");
    try {
      if (url.pathname.startsWith("/git/")) return await this.git(req, res, url);
      if (url.pathname === "/api/events" && req.method === "GET") return this.events(req, res);
      for (const r of this.routes) {
        if (r.method !== req.method) continue;
        const m = r.re.exec(url.pathname);
        if (!m) continue;
        let cachedUser: User | null | undefined;
        const token = bearer(req);
        const value = await r.fn({
          req,
          res,
          params: m.slice(1).map(decodeURIComponent),
          query: url.searchParams,
          body: () => readBody(req),
          user: () => (cachedUser ??= this.svc.authenticate(token)),
          maybeUser: () => (token ? (cachedUser ??= this.svc.authenticate(token)) : null),
        });
        if (res.headersSent) return;
        if (typeof value === "string") {
          res.writeHead(200, { "Content-Type": "text/markdown; charset=utf-8" });
          res.end(value);
        } else {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify(value ?? null));
        }
        return;
      }
      throw new TeamError(`no route ${req.method} ${url.pathname}`, 404);
    } catch (e) {
      const status = e instanceof TeamError ? e.status : 500;
      if (status === 500) this.opts.log?.(`hub error ${req.method} ${url.pathname}: ${e instanceof Error ? e.stack : String(e)}`);
      if (!res.headersSent) {
        res.writeHead(status, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: e instanceof Error ? e.message : String(e) }));
      } else res.end();
    }
  }

  private defineRoutes(): void {
    const S = this.svc;
    this.route("GET", "/api/info", async () => ({ ...S.info(), ...this.opts.publicInfo(), api: 1 }));
    this.route("POST", "/api/join", async (c) => {
      const b = await c.body();
      return S.join(String(b.code ?? ""), { name: str(b.name) ?? "", email: str(b.email) ?? "", publicKey: str(b.publicKey) }, c.maybeUser());
    });
    this.route("GET", "/api/me", async (c) => S.me(c.user()));
    this.route("PATCH", "/api/me", async (c) => {
      const b = await c.body();
      return S.updateMe(c.user(), { name: str(b.name), email: str(b.email) });
    });
    this.route("POST", "/api/me/keys", async (c) => S.addKey(c.user(), String((await c.body()).publicKey ?? "")));
    this.route("DELETE", "/api/me/keys/:id", async (c) => S.deleteKey(c.user(), c.params[0]!));
    this.route("GET", "/api/me/bugs", async (c) => S.myBugs(c.user()));
    this.route("GET", "/api/notifications", async (c) => S.notifications(c.user(), c.query.get("unread") === "1"));
    this.route("POST", "/api/notifications/read", async (c) => {
      const b = await c.body();
      S.markRead(c.user(), Array.isArray(b.ids) ? b.ids.map(String) : "all");
      return true;
    });

    this.route("GET", "/api/groups", async (c) => S.groups(c.user()));
    this.route("POST", "/api/groups", async (c) => {
      const b = await c.body();
      return S.createGroup(c.user(), { name: String(b.name ?? ""), slug: str(b.slug), description: str(b.description) });
    });
    this.route("GET", "/api/groups/:g", async (c) => S.group(c.user(), c.params[0]!));
    this.route("GET", "/api/groups/:g/activity", async (c) => S.feed(c.user(), c.params[0]!, c.query.get("project") ?? undefined));
    this.route("POST", "/api/groups/:g/invites", async (c) => {
      const b = await c.body();
      return S.createInvite(c.user(), c.params[0]!, { role: str(b.role) as never, uses: b.uses ? Number(b.uses) : undefined, ttlHours: b.ttlHours ? Number(b.ttlHours) : undefined });
    });
    this.route("DELETE", "/api/groups/:g/invites/:code", async (c) => S.revokeInvite(c.user(), c.params[0]!, c.params[1]!));
    this.route("PATCH", "/api/groups/:g/members/:uid", async (c) => S.setRole(c.user(), c.params[0]!, c.params[1]!, String((await c.body()).role) as never));
    this.route("DELETE", "/api/groups/:g/members/:uid", async (c) => S.removeMember(c.user(), c.params[0]!, c.params[1]!));
    this.route("POST", "/api/groups/:g/projects", async (c) => {
      const b = await c.body();
      return S.createProject(c.user(), c.params[0]!, {
        name: String(b.name ?? ""), slug: str(b.slug), description: str(b.description), defaultBranch: str(b.defaultBranch),
        protectDefault: typeof b.protectDefault === "boolean" ? b.protectDefault : undefined,
      });
    });

    const P = "/api/projects/:g/:p";
    this.route("GET", P, async (c) => S.projectDetail(c.user(), c.params[0]!, c.params[1]!));
    this.route("PATCH", P, async (c) => {
      const b = await c.body();
      return S.updateProject(c.user(), c.params[0]!, c.params[1]!, { name: str(b.name), description: str(b.description), defaultBranch: str(b.defaultBranch), protectDefault: typeof b.protectDefault === "boolean" ? b.protectDefault : undefined });
    });
    this.route("GET", `${P}/tree`, async (c) => S.tree(c.user(), c.params[0]!, c.params[1]!, c.query.get("ref") ?? "HEAD", c.query.get("path") ?? ""));
    this.route("GET", `${P}/file`, async (c) => ({ text: await S.file(c.user(), c.params[0]!, c.params[1]!, c.query.get("ref") ?? "HEAD", c.query.get("path") ?? "") }));
    this.route("GET", `${P}/commits/:sha`, async (c) => S.commit(c.user(), c.params[0]!, c.params[1]!, c.params[2]!));

    this.route("GET", `${P}/issues`, async (c) => S.issues(c.user(), c.params[0]!, c.params[1]!, { kind: c.query.get("kind") ?? undefined, state: c.query.get("state") ?? undefined, assignee: c.query.get("assignee") ?? undefined }));
    this.route("POST", `${P}/issues`, async (c) => {
      const b = await c.body();
      return S.createIssue(c.user(), c.params[0]!, c.params[1]!, { title: String(b.title ?? ""), body: str(b.body), labels: Array.isArray(b.labels) ? b.labels.map(String) : undefined, assigneeId: str(b.assigneeId) ?? null });
    });
    this.route("GET", `${P}/issues/:n`, async (c) => S.issue(c.user(), c.params[0]!, c.params[1]!, num(c.params[2])));
    this.route("GET", `${P}/issues/:n/report.md`, async (c) => (await S.issue(c.user(), c.params[0]!, c.params[1]!, num(c.params[2]))).body);
    this.route("PATCH", `${P}/issues/:n`, async (c) => {
      const b = await c.body();
      return S.updateIssue(c.user(), c.params[0]!, c.params[1]!, num(c.params[2]), {
        title: str(b.title), body: str(b.body), state: str(b.state) as never, labels: Array.isArray(b.labels) ? b.labels.map(String) : undefined,
        assigneeId: "assigneeId" in b ? (str(b.assigneeId) ?? null) : undefined,
      });
    });
    this.route("POST", `${P}/issues/:n/comments`, async (c) => S.comment(c.user(), c.params[0]!, c.params[1]!, num(c.params[2]), String((await c.body()).body ?? "")));
    this.route("POST", `${P}/bugs`, async (c) => {
      const b = await c.body();
      const env = b.environment && typeof b.environment === "object" ? Object.fromEntries(Object.entries(b.environment as Record<string, unknown>).map(([k, v]) => [k, String(v)])) : undefined;
      return S.reportBug(c.user(), c.params[0]!, c.params[1]!, {
        title: String(b.title ?? ""), summary: str(b.summary), severity: str(b.severity) as never, error: str(b.error), file: str(b.file), line: b.line ? Number(b.line) : undefined,
        ref: str(b.ref), commit: str(b.commit), steps: Array.isArray(b.steps) ? b.steps.map(String) : undefined, expected: str(b.expected), actual: str(b.actual),
        environment: env, logs: str(b.logs), notes: str(b.notes), assigneeId: str(b.assigneeId) ?? null,
      });
    });
    this.route("POST", `${P}/issues/:n/status`, async (c) => {
      const b = await c.body();
      return S.setBugStatus(c.user(), c.params[0]!, c.params[1]!, num(c.params[2]), { status: String(b.status) as never, fixCommit: str(b.fixCommit), comment: str(b.comment) });
    });

    this.route("GET", `${P}/pulls`, async (c) => S.pulls(c.user(), c.params[0]!, c.params[1]!, c.query.get("state") ?? undefined));
    this.route("POST", `${P}/pulls`, async (c) => {
      const b = await c.body();
      return S.createPull(c.user(), c.params[0]!, c.params[1]!, { title: String(b.title ?? ""), body: str(b.body), source: String(b.source ?? ""), target: str(b.target) });
    });
    this.route("GET", `${P}/pulls/:n`, async (c) => S.pull(c.user(), c.params[0]!, c.params[1]!, num(c.params[2])));
    this.route("POST", `${P}/pulls/:n/reviews`, async (c) => {
      const b = await c.body();
      return S.review(c.user(), c.params[0]!, c.params[1]!, num(c.params[2]), { state: String(b.state ?? "comment") as never, body: str(b.body) });
    });
    this.route("POST", `${P}/pulls/:n/merge`, async (c) => S.merge(c.user(), c.params[0]!, c.params[1]!, num(c.params[2])));
    this.route("POST", `${P}/pulls/:n/close`, async (c) => S.closePull(c.user(), c.params[0]!, c.params[1]!, num(c.params[2]), (await c.body()).reopen === true));
  }

  // ---- live events --------------------------------------------------------------------------------

  private events(req: http.IncomingMessage, res: http.ServerResponse): void {
    const url = new URL(req.url ?? "/", "http://hub");
    const user = this.svc.authenticate(bearer(req) ?? url.searchParams.get("token"));
    res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive", "X-Accel-Buffering": "no" });
    res.write(`event: hello\ndata: ${JSON.stringify({ userId: user.id })}\n\n`);
    const client = { user, res, groups: this.svc.groupIdsOf(user.id) };
    this.sse.add(client);
    const ping = setInterval(() => res.write(": ping\n\n"), 25_000);
    req.on("close", () => {
      clearInterval(ping);
      this.sse.delete(client);
    });
  }

  private broadcast(e: HubEvent): void {
    for (const c of this.sse) {
      if (e.type === "changed" && e.what === "members") c.groups = this.svc.groupIdsOf(c.user.id);
      const visible = e.type === "notification" ? e.userId === c.user.id : c.groups.has(e.groupId);
      if (visible) c.res.write(`event: ${e.type}\ndata: ${JSON.stringify(e)}\n\n`);
    }
  }

  // ---- git smart HTTP -----------------------------------------------------------------------------

  private async git(req: http.IncomingMessage, res: http.ServerResponse, url: URL): Promise<void> {
    const m = /^\/git\/([^/]+)\/([^/]+?)\.git(\/.*)$/.exec(url.pathname);
    if (!m) throw new TeamError("not a repository path", 404);
    const [, g, p, rest] = m as unknown as [string, string, string, string];
    const token = bearer(req);
    if (!token) {
      res.writeHead(401, { "WWW-Authenticate": 'Basic realm="RLM Team Hub"', "Content-Type": "text/plain" });
      res.end("authentication required: use your hub token as the password\n");
      return;
    }
    const user = this.svc.authenticate(token);
    const { repo } = this.svc.access(user, g, p);
    const pushing = rest === "/git-receive-pack" || url.searchParams.get("service") === "git-receive-pack";
    const env: NodeJS.ProcessEnv = {};
    if (pushing) {
      const pol = await this.svc.pushPolicy(user, g, p);
      if (!pol.allowed) throw new TeamError(pol.reason ?? "push not allowed", 403);
      if (pol.protectedBranch) env.RLM_PROTECTED_BRANCH = pol.protectedBranch;
    }
    const before = rest === "/git-receive-pack" ? await this.opts.git.refs(repo) : null;
    await runHttpBackend(req, res, {
      ...env,
      GIT_PROJECT_ROOT: path.dirname(path.dirname(repo)),
      GIT_HTTP_EXPORT_ALL: "1",
      PATH_INFO: `/${g}/${p}.git${rest}`,
      REQUEST_METHOD: req.method ?? "GET",
      QUERY_STRING: url.search.slice(1),
      CONTENT_TYPE: req.headers["content-type"] ?? "",
      REMOTE_USER: user.email,
      REMOTE_ADDR: req.socket.remoteAddress ?? "",
      HTTP_CONTENT_ENCODING: req.headers["content-encoding"] ?? "",
      GIT_PROTOCOL: String(req.headers["git-protocol"] ?? ""),
      GIT_COMMITTER_NAME: user.name,
      GIT_COMMITTER_EMAIL: user.email,
    });
    if (before) await this.svc.recordPush(user, g, p, before, await this.opts.git.refs(repo), "http").catch((e: unknown) => this.opts.log?.(`push record failed: ${String(e)}`));
  }
}

/** Runs `git http-backend` as a CGI script for one request. */
function runHttpBackend(req: http.IncomingMessage, res: http.ServerResponse, env: NodeJS.ProcessEnv): Promise<void> {
  return new Promise((resolve, reject) => {
    const cgi = spawn("git", ["http-backend"], { env: { ...process.env, ...env }, windowsHide: true });
    let headerDone = false;
    let buf = Buffer.alloc(0);
    cgi.stdout.on("data", (d: Buffer) => {
      if (headerDone) return void res.write(d);
      buf = Buffer.concat([buf, d]);
      let idx = buf.indexOf("\r\n\r\n");
      let sepLen = 4;
      if (idx < 0) {
        idx = buf.indexOf("\n\n");
        sepLen = 2;
      }
      if (idx < 0) return;
      const head = buf.subarray(0, idx).toString("utf8");
      const body = buf.subarray(idx + sepLen);
      headerDone = true;
      let status = 200;
      const headers: Record<string, string> = {};
      for (const line of head.split(/\r?\n/)) {
        const i = line.indexOf(":");
        if (i < 0) continue;
        const k = line.slice(0, i).trim();
        const v = line.slice(i + 1).trim();
        if (k.toLowerCase() === "status") status = Number(v.split(" ")[0]) || 200;
        else headers[k] = v;
      }
      res.writeHead(status, headers);
      if (body.length) res.write(body);
    });
    const err: Buffer[] = [];
    cgi.stderr.on("data", (d: Buffer) => err.push(d));
    cgi.on("error", reject);
    cgi.on("close", (code) => {
      if (!headerDone) {
        res.writeHead(500, { "Content-Type": "text/plain" });
        res.end(Buffer.concat(err).toString() || `git http-backend exited ${code}`);
      } else res.end();
      resolve();
    });
    req.pipe(cgi.stdin);
  });
}
