import { mkdtempSync, rmSync, writeFileSync, readFileSync, chmodSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { BeaconListener, BeaconSender, generateIdentity, gitAuth, HubClient, openTunnel, remoteUrl, runGit, startHub, type RunningHub } from "../src";

const ISOLATED = { GIT_CONFIG_GLOBAL: "/dev/null", GIT_CONFIG_NOSYSTEM: "1" };

async function git(cwd: string, args: string[], env: Record<string, string> = {}) {
  const r = await runGit(args, { cwd, env: { ...ISOLATED, ...env } });
  return r;
}

async function ok(cwd: string, args: string[], env: Record<string, string> = {}) {
  const r = await git(cwd, args, env);
  if (r.code !== 0) throw new Error(`git ${args.join(" ")} failed: ${r.stderr}`);
  return r.stdout.trim();
}

const as = (name: string, email: string) => ["-c", `user.name=${name}`, "-c", `user.email=${email}`];

describe("team hub end to end", () => {
  let dir: string;
  let hub: RunningHub;
  let base: string;
  let alice: HubClient;
  let bob: HubClient;
  let carol: HubClient;
  let bobKey: { privateKey: string; publicKey: string };
  let bobKeyFile: string;
  let knownHosts: string;
  const bobEvents: Array<{ type: string; data: any }> = [];
  let stopBobEvents: () => void = () => undefined;

  beforeAll(async () => {
    dir = mkdtempSync(path.join(os.tmpdir(), "rlm-hub-"));
    hub = await startHub({ dataDir: path.join(dir, "hub"), name: "Test Hub", owner: { name: "Alice", email: "alice@team.dev" }, bindHost: "127.0.0.1", httpPort: 0, sshPort: 0, discovery: false });
    base = `http://127.0.0.1:${hub.httpPort}`;
    alice = new HubClient(base, hub.ownerToken);
    bobKey = generateIdentity("bob@laptop");
    bobKeyFile = path.join(dir, "bob_ed25519");
    writeFileSync(bobKeyFile, bobKey.privateKey, { mode: 0o600 });
    chmodSync(bobKeyFile, 0o600);
    knownHosts = path.join(dir, "known_hosts");
  });

  afterAll(async () => {
    stopBobEvents();
    await hub?.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it("serves public info without auth and rejects bad tokens", async () => {
    const info = await new HubClient(base, null).request<any>("GET", "/api/info");
    expect(info).toMatchObject({ name: "Test Hub", sshPort: hub.sshPort, api: 1 });
    expect(info.sshFingerprint).toMatch(/^SHA256:/);
    await expect(new HubClient(base, "nope").request("GET", "/api/me")).rejects.toMatchObject({ status: 401 });
  });

  it("creates a group and lets people join with invite codes", async () => {
    await alice.request("POST", "/api/groups", { name: "Core Team" });
    const devInvite = await alice.request<any>("POST", "/api/groups/core-team/invites", { role: "developer" });
    const repInvite = await alice.request<any>("POST", "/api/groups/core-team/invites", { role: "reporter" });

    const b = await new HubClient(base, null).request<any>("POST", "/api/join", { code: devInvite.code, name: "Bob", email: "bob@team.dev", publicKey: bobKey.publicKey });
    expect(b.role).toBe("developer");
    bob = new HubClient(base, b.token);
    const c = await new HubClient(base, null).request<any>("POST", "/api/join", { code: repInvite.code, name: "Carol", email: "carol@team.dev" });
    carol = new HubClient(base, c.token);

    const g = await bob.request<any>("GET", "/api/groups/core-team");
    expect(g.members.map((m: any) => `${m.name}:${m.role}`).sort()).toEqual(["Alice:owner", "Bob:developer", "Carol:reporter"]);
    await expect(carol.request("POST", "/api/groups/core-team/invites", {})).rejects.toMatchObject({ status: 403 });

    stopBobEvents = bob.events((type, data) => bobEvents.push({ type, data }));
  });

  it("pushes and clones over HTTP with the hub token", async () => {
    await alice.request("POST", "/api/groups/core-team/projects", { name: "app" });
    const work = path.join(dir, "alice");
    const url = remoteUrl({ kind: "http", url: base }, hub.hubId, "core-team", "app");
    const auth = gitAuth({ kind: "http", url: base }, hub.hubId, hub.ownerToken, base, null, null);
    const cfg = auth.config.flatMap((c) => ["-c", c]);
    await ok(dir, [...cfg, "clone", "-q", url, work], auth.env);
    writeFileSync(path.join(work, "app.js"), ["function add(a, b) {", "  return a + b;", "}", "", "module.exports = { add };", ""].join("\n"));
    await ok(work, ["add", "."]);
    await ok(work, [...as("Alice", "alice@team.dev"), "commit", "-q", "-m", "add()"]);
    await ok(work, [...cfg, "push", "-q", "origin", "HEAD:main"], auth.env);

    // Without credentials git is refused.
    const anon = await git(dir, ["ls-remote", url]);
    expect(anon.code).not.toBe(0);

    const detail = await alice.request<any>("GET", "/api/projects/core-team/app");
    expect(detail.commits[0].subject).toBe("add()");
    expect(detail.pushes[0]).toMatchObject({ transport: "http", ref: "refs/heads/main" });
  });

  it("does git over SSH with the member's key and protects the default branch", async () => {
    const work = path.join(dir, "bob");
    const t = { kind: "ssh" as const, host: "127.0.0.1", port: hub.sshPort! };
    const url = remoteUrl(t, hub.hubId, "core-team", "app");
    const auth = gitAuth(t, hub.hubId, "", "", bobKeyFile, knownHosts);
    await ok(dir, ["clone", "-q", url, work], auth.env);
    expect(readFileSync(path.join(work, "app.js"), "utf8")).toContain("return a + b");
    expect(readFileSync(knownHosts, "utf8")).toContain("ssh-ed25519");

    writeFileSync(path.join(work, "app.js"), ["function add(a, b) {", "  return a - b;", "}", "", "module.exports = { add };", ""].join("\n"));
    await ok(work, [...as("Bob", "bob@team.dev"), "commit", "-qam", "simplify add"]);
    const denied = await git(work, ["push", "origin", "HEAD:main"], auth.env);
    expect(denied.code).not.toBe(0);
    expect(denied.stderr).toContain("main is protected");
    await ok(work, ["push", "-q", "origin", "HEAD:refs/heads/bob/simplify"], auth.env);

    // A commit authored by someone outside the group but pushed by Bob.
    writeFileSync(path.join(work, "util.js"), ["exports.half = (x) => x / 3;", ""].join("\n"));
    await ok(work, ["add", "."]);
    await ok(work, [...as("Contractor", "contractor@elsewhere.dev"), "commit", "-qm", "half()"]);
    await ok(work, ["push", "-q", "origin", "HEAD:refs/heads/bob/simplify"], auth.env);

    const detail = await bob.request<any>("GET", "/api/projects/core-team/app");
    expect(detail.branches.map((b: any) => b.name).sort()).toEqual(["bob/simplify", "main"]);
    expect(detail.pushes[0]).toMatchObject({ transport: "ssh" });
  });

  it("reviews and merges a pull request on the hub", async () => {
    const pr = await bob.request<any>("POST", "/api/projects/core-team/app/pulls", { title: "Simplify add", source: "bob/simplify" });
    const view = await bob.request<any>("GET", `/api/projects/core-team/app/pulls/${pr.number}`);
    expect(view.mergeable).toBe(true);
    expect(view.commits).toHaveLength(2);
    expect(view.diff).toContain("-  return a + b;");
    await expect(bob.request("POST", `/api/projects/core-team/app/pulls/${pr.number}/merge`)).rejects.toMatchObject({ status: 403 });
    await expect(bob.request("POST", `/api/projects/core-team/app/pulls/${pr.number}/reviews`, { state: "approve" })).rejects.toMatchObject({ status: 403 });
    await alice.request("POST", `/api/projects/core-team/app/pulls/${pr.number}/reviews`, { state: "approve", body: "LGTM" });
    const merged = await bob.request<any>("POST", `/api/projects/core-team/app/pulls/${pr.number}/merge`);
    expect(merged.state).toBe("merged");
    const file = await carol.request<any>("GET", "/api/projects/core-team/app/file?path=app.js&ref=main");
    expect(file.text).toContain("return a - b");
  });

  it("routes a bug report to the author of the blamed line as a full markdown report", async () => {
    const bug = await carol.request<any>("POST", "/api/projects/core-team/app/bugs", {
      title: "add() subtracts",
      summary: "Totals are wrong on the checkout page.",
      severity: "high",
      error: "AssertionError: expected 5, got -1\n    at test (cart.test.js:12:3)",
      file: "app.js",
      line: 2,
      steps: ["Open checkout", "Add two items"],
      expected: "add(2, 3) === 5",
      actual: "add(2, 3) === -1",
      environment: { os: "Linux", node: "22" },
    });
    expect(bug.assignee.name).toBe("Bob");
    expect(bug.routedBy).toMatch(/^wrote commit/);
    const md = await carol.request<string>("GET", `/api/projects/core-team/app/issues/${bug.number}/report.md`);
    for (const s of ["# Bug #", "add() subtracts", "## Error", "AssertionError", "## Code at `app.js:2`", "> 2 |", "return a - b", "## Origin", "simplify add", "Merged through pull request", "Pushed to the hub by** Bob", "## Steps to reproduce", "## Expected vs actual", "## Environment", "## For the developer"])
      expect(md).toContain(s);

    await new Promise((r) => setTimeout(r, 300));
    expect(bobEvents.some((e) => e.type === "notification" && e.data.notification?.kind === "bug_assigned")).toBe(true);
    const mine = await bob.request<any[]>("GET", "/api/me/bugs");
    expect(mine.map((b) => b.number)).toContain(bug.number);

    const fixed = await bob.request<any>("POST", `/api/projects/core-team/app/issues/${bug.number}/status`, { status: "fixed", fixCommit: "abc1234", comment: "Restored the plus sign." });
    expect(fixed.state).toBe("closed");
    const inbox = await carol.request<any[]>("GET", "/api/notifications");
    expect(inbox.some((n) => n.kind === "bug_status" && n.body.includes("Restored"))).toBe(true);
  });

  it("routes to whoever pushed a commit when its author is not a member", async () => {
    const bug = await carol.request<any>("POST", "/api/projects/core-team/app/bugs", { title: "half() is wrong", file: "util.js", line: 1, error: "half(4) returned 1.33" });
    expect(bug.assignee.name).toBe("Bob");
    expect(bug.routedBy).toMatch(/^pushed commit/);
  });

  it("tunnels the API over the built-in SSH server with key, token, and invite logins", async () => {
    const viaKey = await openTunnel({ host: "127.0.0.1", port: hub.sshPort!, auth: { username: "git", privateKey: bobKey.privateKey }, target: { host: "hub", port: 0 }, expectFingerprint: hub.sshFingerprint });
    try {
      const me = await new HubClient(`http://127.0.0.1:${viaKey.localPort}`, bob.token).request<any>("GET", "/api/me");
      expect(me.name).toBe("Bob");
    } finally {
      viaKey.close();
    }

    const viaToken = await openTunnel({ host: "127.0.0.1", port: hub.sshPort!, auth: { username: "token", password: carol.token! }, target: { host: "hub", port: 0 } });
    expect(viaToken.fingerprint).toBe(hub.sshFingerprint);
    viaToken.close();

    const invite = await alice.request<any>("POST", "/api/groups/core-team/invites", { role: "developer" });
    const viaInvite = await openTunnel({ host: "127.0.0.1", port: hub.sshPort!, auth: { username: "join", password: invite.code }, target: { host: "hub", port: 0 } });
    try {
      const d = await new HubClient(`http://127.0.0.1:${viaInvite.localPort}`, null).request<any>("POST", "/api/join", { code: invite.code, name: "Dan", email: "dan@remote.dev" });
      expect(d.token).toBeTruthy();
    } finally {
      viaInvite.close();
    }

    await expect(openTunnel({ host: "127.0.0.1", port: hub.sshPort!, auth: { username: "join", password: "wrong" }, target: { host: "hub", port: 0 } })).rejects.toThrow();
    await expect(openTunnel({ host: "127.0.0.1", port: hub.sshPort!, auth: { username: "git", privateKey: bobKey.privateKey }, target: { host: "hub", port: 0 }, expectFingerprint: "SHA256:forged" })).rejects.toThrow(/host key mismatch/);
  });

  it("finds hubs on the LAN through beacons", async () => {
    const port = 40000 + Math.floor(Math.random() * 20000);
    const listener = new BeaconListener(undefined, port);
    await listener.start();
    const sender = new BeaconSender(() => ({ rlmHub: 1, hubId: hub.hubId, name: "Test Hub", httpPort: hub.httpPort, sshPort: hub.sshPort, sshFingerprint: hub.sshFingerprint }), port, 200);
    sender.start();
    try {
      for (let i = 0; i < 20 && !listener.list().length; i++) await new Promise((r) => setTimeout(r, 100));
      expect(listener.list()[0]).toMatchObject({ hubId: hub.hubId, httpPort: hub.httpPort, sshPort: hub.sshPort });
    } finally {
      sender.stop();
      listener.stop();
    }
  });
});
