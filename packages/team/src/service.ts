import { createHash, randomBytes, randomUUID } from "node:crypto";
import { renderBugReport, type BugContext, type BugReportInput } from "./domain/report";
import {
  assertBranch,
  assertSlug,
  atLeast,
  slugify,
  TeamError,
  type Activity,
  type BugStatus,
  type Comment,
  type Group,
  type Issue,
  type Notification,
  type NotificationKind,
  type Project,
  type PullRequest,
  type ReviewState,
  type Role,
  type Severity,
  type User,
} from "./domain/types";
import type { GitHost, TeamStore } from "./ports";

export type HubEvent =
  | { type: "notification"; userId: string; notification: Notification }
  | { type: "activity"; groupId: string; activity: Activity }
  | { type: "changed"; groupId: string; projectId: string | null; what: "issues" | "pulls" | "code" | "members" | "projects" };

export interface UserView {
  id: string;
  name: string;
  email: string;
}

const now = () => Date.now();
const id = () => randomUUID();
export const hashToken = (t: string) => createHash("sha256").update(t).digest("hex");
const newToken = () => `rlmt_${randomBytes(24).toString("base64url")}`;
/** 12 characters from an alphabet without look-alikes (no 0/O, 1/I/L), easy to read out loud. */
const INVITE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789";
const newInviteCode = () =>
  [...randomBytes(12)]
    .map((b) => INVITE_ALPHABET[b % INVITE_ALPHABET.length])
    .join("")
    .replace(/(.{4})(?=.)/g, "$1-");
const normalizeInvite = (code: string) => code.trim().toUpperCase().replace(/[\s_]+/g, "-");

/** SHA256 fingerprint of an OpenSSH public key line, as `ssh-keygen -l` prints it. */
export function sshFingerprint(publicKey: string): string {
  const parts = publicKey.trim().split(/\s+/);
  if (parts.length < 2 || !/^(ssh-(ed25519|rsa)|ecdsa-sha2-nistp\d+|sk-)/.test(parts[0]!)) throw new TeamError("not an OpenSSH public key");
  const blob = Buffer.from(parts[1]!, "base64");
  if (!blob.length) throw new TeamError("not an OpenSSH public key");
  return `SHA256:${createHash("sha256").update(blob).digest("base64").replace(/=+$/, "")}`;
}

/** Every team use case, with its permission check. Transports (HTTP, SSH) call into this. */
export class TeamService {
  private listeners = new Set<(e: HubEvent) => void>();

  constructor(
    private store: TeamStore,
    private git: GitHost,
  ) {}

  on(fn: (e: HubEvent) => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private emit(e: HubEvent): void {
    for (const fn of this.listeners) {
      try {
        fn(e);
      } catch {
        /* a broken listener must not break the use case */
      }
    }
  }

  // ---- hub identity and accounts ----------------------------------------------------------------

  info(): { hubId: string; name: string; bootstrapped: boolean } {
    return { hubId: this.store.meta("hubId") ?? "", name: this.store.meta("hubName") ?? "RLM Team Hub", bootstrapped: this.store.meta("ownerId") !== null };
  }

  /** First start: creates the hub owner. Later starts: returns a fresh owner token (host machine only). */
  bootstrap(hubName: string, owner: { name: string; email: string }): { user: User; token: string; hubId: string } {
    if (!this.store.meta("hubId")) this.store.setMeta("hubId", id());
    this.store.setMeta("hubName", hubName);
    const token = newToken();
    const ownerId = this.store.meta("ownerId");
    if (ownerId) {
      const user = this.store.userById(ownerId)!;
      this.store.setUserToken(user.id, hashToken(token));
      return { user, token, hubId: this.store.meta("hubId")! };
    }
    const user: User = { id: id(), name: owner.name.trim() || "Owner", email: owner.email.trim().toLowerCase(), createdAt: now() };
    this.store.insertUser(user, hashToken(token));
    this.store.setMeta("ownerId", user.id);
    return { user, token, hubId: this.store.meta("hubId")! };
  }

  isHubOwner(u: User): boolean {
    return this.store.meta("ownerId") === u.id;
  }

  authenticate(token: string | null | undefined): User {
    if (!token) throw new TeamError("sign in required", 401);
    const u = this.store.userByTokenHash(hashToken(token));
    if (!u) throw new TeamError("invalid or revoked token", 401);
    return u;
  }

  inviteValid(code: string): boolean {
    const inv = this.store.invite(normalizeInvite(code));
    return !!inv && inv.expiresAt > now() && inv.usesLeft > 0;
  }

  userByKeyFingerprint(fp: string): User | null {
    const k = this.store.keyByFingerprint(fp);
    return k ? this.store.userById(k.userId) : null;
  }

  view(userId: string | null): UserView | null {
    if (!userId) return null;
    const u = this.store.userById(userId);
    return u ? { id: u.id, name: u.name, email: u.email } : { id: userId, name: "unknown", email: "" };
  }

  me(actor: User): UserView & { keys: Array<{ id: string; fingerprint: string; publicKey: string }>; hubOwner: boolean } {
    return { ...this.view(actor.id)!, keys: this.store.keysOf(actor.id).map((k) => ({ id: k.id, fingerprint: k.fingerprint, publicKey: k.publicKey })), hubOwner: this.isHubOwner(actor) };
  }

  updateMe(actor: User, p: { name?: string; email?: string }): UserView {
    const next = { ...actor, name: p.name?.trim() || actor.name, email: p.email?.trim().toLowerCase() || actor.email };
    this.store.updateUser(next);
    return this.view(next.id)!;
  }

  addKey(actor: User, publicKey: string): { id: string; fingerprint: string } {
    const fingerprint = sshFingerprint(publicKey);
    const existing = this.store.keyByFingerprint(fingerprint);
    if (existing) {
      if (existing.userId !== actor.id) throw new TeamError("this key belongs to another user", 409);
      return { id: existing.id, fingerprint };
    }
    const k = { id: id(), userId: actor.id, publicKey: publicKey.trim(), fingerprint, createdAt: now() };
    this.store.insertKey(k);
    return { id: k.id, fingerprint };
  }

  deleteKey(actor: User, keyId: string): void {
    this.store.deleteKey(keyId, actor.id);
  }

  // ---- groups and membership ---------------------------------------------------------------------

  private groupFor(actor: User, slug: string, min: Role = "reporter"): { group: Group; role: Role } {
    const group = this.store.groupBySlug(slug);
    if (!group) throw new TeamError(`group ${slug} not found`, 404);
    const m = this.store.member(group.id, actor.id);
    const role = m?.role ?? (this.isHubOwner(actor) ? "owner" : null);
    if (!role) throw new TeamError(`you are not a member of ${slug}`, 403);
    if (!atLeast(role, min)) throw new TeamError(`${min} role required in ${slug}`, 403);
    return { group, role };
  }

  createGroup(actor: User, p: { name: string; slug?: string; description?: string }): Group {
    const slug = assertSlug(p.slug?.trim() || slugify(p.name));
    if (this.store.groupBySlug(slug)) throw new TeamError(`group ${slug} already exists`, 409);
    const g: Group = { id: id(), slug, name: p.name.trim() || slug, description: p.description?.trim() ?? "", createdAt: now() };
    this.store.insertGroup(g);
    this.store.upsertMember({ groupId: g.id, userId: actor.id, role: "owner", joinedAt: now() });
    this.activity(g.id, null, actor.id, "created", `created group ${g.name}`, g.slug);
    return g;
  }

  groups(actor: User): Array<Group & { role: Role; projects: number; members: number }> {
    return this.store.groupsOf(actor.id).map((g) => ({ ...g, projects: this.store.projectsOf(g.id).length, members: this.store.members(g.id).length }));
  }

  group(actor: User, slug: string) {
    const { group, role } = this.groupFor(actor, slug);
    const members = this.store.members(group.id).map((m) => ({ ...this.view(m.userId)!, role: m.role, joinedAt: m.joinedAt }));
    const projects = this.store.projectsOf(group.id).map((p) => this.projectView(p));
    const invites = atLeast(role, "maintainer") ? this.store.invitesOf(group.id).filter((i) => i.expiresAt > now() && i.usesLeft > 0) : [];
    return { ...group, role, members, projects, invites };
  }

  createInvite(actor: User, slug: string, p: { role?: Role; uses?: number; ttlHours?: number }) {
    const { group, role } = this.groupFor(actor, slug, "maintainer");
    const want = p.role ?? "developer";
    if (!atLeast(role, want)) throw new TeamError("you cannot invite with a role above your own", 403);
    const inv = { code: newInviteCode(), groupId: group.id, role: want, createdBy: actor.id, expiresAt: now() + Math.max(1, p.ttlHours ?? 72) * 3600_000, usesLeft: Math.max(1, Math.min(100, p.uses ?? 10)) };
    this.store.insertInvite(inv);
    return inv;
  }

  revokeInvite(actor: User, slug: string, code: string): void {
    const { group } = this.groupFor(actor, slug, "maintainer");
    const inv = this.store.invite(normalizeInvite(code));
    if (inv?.groupId === group.id) this.store.deleteInvite(inv.code);
  }

  /** Redeems an invite. New people get an account and a token; signed-in people just join the group. */
  join(code: string, who: { name: string; email: string; publicKey?: string }, actor?: User | null): { user: UserView; token: string | null; group: Group; role: Role } {
    const inv = this.store.invite(normalizeInvite(code));
    if (!inv || inv.expiresAt < now() || inv.usesLeft <= 0) throw new TeamError("invite code is invalid or expired", 403);
    const group = this.store.groupById(inv.groupId)!;
    let user = actor ?? null;
    let token: string | null = null;
    if (!user) {
      const email = who.email.trim().toLowerCase();
      if (!who.name.trim() || !/^\S+@\S+$/.test(email)) throw new TeamError("name and email are required");
      if (this.store.userByEmail(email)) throw new TeamError(`${email} already has an account on this hub; sign in with its token or ask an owner`, 409);
      user = { id: id(), name: who.name.trim(), email, createdAt: now() };
      token = newToken();
      this.store.insertUser(user, hashToken(token));
    }
    if (who.publicKey?.trim()) this.addKey(user, who.publicKey);
    const existing = this.store.member(group.id, user.id);
    if (!existing || !atLeast(existing.role, inv.role)) this.store.upsertMember({ groupId: group.id, userId: user.id, role: inv.role, joinedAt: existing?.joinedAt ?? now() });
    this.store.useInvite(inv.code);
    this.activity(group.id, null, user.id, "joined", `${user.name} joined as ${inv.role}`, group.slug);
    this.emit({ type: "changed", groupId: group.id, projectId: null, what: "members" });
    return { user: this.view(user.id)!, token, group, role: existing && atLeast(existing.role, inv.role) ? existing.role : inv.role };
  }

  setRole(actor: User, slug: string, userId: string, role: Role): void {
    const { group, role: mine } = this.groupFor(actor, slug, "maintainer");
    const m = this.store.member(group.id, userId);
    if (!m) throw new TeamError("not a member", 404);
    if (!atLeast(mine, role) || !atLeast(mine, m.role)) throw new TeamError("you cannot change a role above your own", 403);
    if (m.role === "owner" && role !== "owner" && this.store.members(group.id).filter((x) => x.role === "owner").length === 1) throw new TeamError("a group needs at least one owner", 409);
    this.store.upsertMember({ ...m, role });
    this.emit({ type: "changed", groupId: group.id, projectId: null, what: "members" });
  }

  removeMember(actor: User, slug: string, userId: string): void {
    const { group, role: mine } = userId === actor.id ? this.groupFor(actor, slug) : this.groupFor(actor, slug, "maintainer");
    const m = this.store.member(group.id, userId);
    if (!m) return;
    if (userId !== actor.id && !atLeast(mine, m.role)) throw new TeamError("you cannot remove someone with a higher role", 403);
    if (m.role === "owner" && this.store.members(group.id).filter((x) => x.role === "owner").length === 1) throw new TeamError("a group needs at least one owner", 409);
    this.store.removeMember(group.id, userId);
    this.emit({ type: "changed", groupId: group.id, projectId: null, what: "members" });
  }

  // ---- projects -----------------------------------------------------------------------------------

  private projectView(p: Project) {
    return {
      ...p,
      openIssues: this.store.issues(p.id, { kind: "issue", state: "open" }).length,
      openBugs: this.store.issues(p.id, { kind: "bug", state: "open" }).length,
      openPulls: this.store.pulls(p.id, "open").length,
    };
  }

  /** Resolves "group/project" for an actor with at least `min` role. */
  access(actor: User, groupSlug: string, projectSlug: string, min: Role = "reporter"): { group: Group; project: Project; role: Role; repo: string } {
    const { group, role } = this.groupFor(actor, groupSlug, min);
    const project = this.store.project(group.id, projectSlug);
    if (!project) throw new TeamError(`project ${groupSlug}/${projectSlug} not found`, 404);
    return { group, project, role, repo: this.git.path(group.slug, project.slug) };
  }

  async createProject(actor: User, groupSlug: string, p: { name: string; slug?: string; description?: string; defaultBranch?: string; protectDefault?: boolean }) {
    const { group } = this.groupFor(actor, groupSlug, "developer");
    const slug = assertSlug(p.slug?.trim() || slugify(p.name));
    if (this.store.project(group.id, slug)) throw new TeamError(`project ${groupSlug}/${slug} already exists`, 409);
    const defaultBranch = assertBranch(p.defaultBranch?.trim() || "main");
    await this.git.create(group.slug, slug, defaultBranch);
    const project: Project = { id: id(), groupId: group.id, slug, name: p.name.trim() || slug, description: p.description?.trim() ?? "", defaultBranch, protectDefault: p.protectDefault ?? true, createdAt: now() };
    this.store.insertProject(project);
    this.activity(group.id, project.id, actor.id, "created", `created project ${project.name}`, `${group.slug}/${slug}`);
    this.emit({ type: "changed", groupId: group.id, projectId: project.id, what: "projects" });
    return this.projectView(project);
  }

  updateProject(actor: User, g: string, p: string, patch: { name?: string; description?: string; defaultBranch?: string; protectDefault?: boolean }) {
    const { project } = this.access(actor, g, p, "maintainer");
    const next = {
      ...project,
      name: patch.name?.trim() || project.name,
      description: patch.description ?? project.description,
      defaultBranch: patch.defaultBranch ? assertBranch(patch.defaultBranch) : project.defaultBranch,
      protectDefault: patch.protectDefault ?? project.protectDefault,
    };
    this.store.updateProject(next);
    return this.projectView(next);
  }

  async projectDetail(actor: User, g: string, p: string) {
    const { project, role, repo, group } = this.access(actor, g, p);
    const branches = await this.git.branches(repo);
    const head = branches.find((b) => b.name === project.defaultBranch) ?? branches[0];
    const commits = head ? await this.git.log(repo, head.name, 30) : [];
    return { ...this.projectView(project), group: group.slug, role, branches, commits, pushes: this.store.pushes(project.id, 20).map((x) => ({ ...x, user: this.view(x.userId) })), empty: branches.length === 0 };
  }

  async tree(actor: User, g: string, p: string, ref: string, dir: string) {
    const { repo } = this.access(actor, g, p);
    return this.git.tree(repo, ref, dir);
  }

  async file(actor: User, g: string, p: string, ref: string, file: string) {
    const { repo } = this.access(actor, g, p);
    const text = await this.git.show(repo, ref, file);
    if (text === null) throw new TeamError(`${file} not found at ${ref}`, 404);
    return text;
  }

  async commit(actor: User, g: string, p: string, sha: string) {
    const { repo } = this.access(actor, g, p);
    const c = await this.git.commit(repo, sha);
    if (!c) throw new TeamError("commit not found", 404);
    const parent = c.parents[0];
    const d = parent ? await this.git.diff(repo, parent, c.sha, 400_000) : { diff: "", truncated: false, files: [] };
    return { ...c, ...d };
  }

  /** Who may update which refs. Called by both git transports before receive-pack runs. */
  async pushPolicy(actor: User, g: string, p: string): Promise<{ allowed: boolean; protectedBranch: string | null; reason?: string }> {
    const { project, role, repo } = this.access(actor, g, p);
    if (!atLeast(role, "developer")) return { allowed: false, protectedBranch: null, reason: "reporters cannot push" };
    // The first push seeds the default branch, whoever makes it.
    const empty = !(await this.git.resolve(repo, `refs/heads/${project.defaultBranch}`));
    return { allowed: true, protectedBranch: project.protectDefault && !empty && !atLeast(role, "maintainer") ? project.defaultBranch : null };
  }

  /** Records what a push changed (refs before/after) and announces it. */
  async recordPush(actor: User, g: string, p: string, before: Map<string, string>, after: Map<string, string>, transport: "http" | "ssh"): Promise<void> {
    const { group, project, repo } = this.access(actor, g, p);
    const changed = [...new Set([...before.keys(), ...after.keys()])].filter((r) => before.get(r) !== after.get(r) && r.startsWith("refs/heads/"));
    for (const ref of changed) {
      const b = before.get(ref) ?? "";
      const a = after.get(ref) ?? "";
      let commits: string[] = [];
      if (a) {
        const others = [...before.values()].filter((v) => v && v !== a);
        const range = b ? `${b}..${a}` : others.length ? `${a} --not ${others.join(" ")}` : a;
        commits = (await this.git.log(repo, range, 500)).map((c) => c.sha);
      }
      this.store.insertPush({ id: id(), projectId: project.id, userId: actor.id, ref, before: b, after: a, commits, transport, createdAt: now() });
      const branch = ref.replace("refs/heads/", "");
      const summary = !a ? `deleted branch ${branch}` : `pushed ${commits.length} commit${commits.length === 1 ? "" : "s"} to ${branch}`;
      this.activity(group.id, project.id, actor.id, "pushed", summary, `${group.slug}/${project.slug}@${branch}`);
    }
    if (changed.length) this.emit({ type: "changed", groupId: group.id, projectId: project.id, what: "code" });
  }

  // ---- issues and bugs ----------------------------------------------------------------------------

  private issueView(i: Issue) {
    return { ...i, author: this.view(i.authorId), assignee: this.view(i.assigneeId) };
  }

  issues(actor: User, g: string, p: string, f: { kind?: string; state?: string; assignee?: string }) {
    const { project } = this.access(actor, g, p);
    return this.store.issues(project.id, { kind: f.kind, state: f.state, assigneeId: f.assignee === "me" ? actor.id : f.assignee }).map((i) => this.issueView(i));
  }

  issue(actor: User, g: string, p: string, n: number) {
    const { project } = this.access(actor, g, p);
    const i = this.store.issue(project.id, n);
    if (!i) throw new TeamError(`#${n} not found`, 404);
    return { ...this.issueView(i), comments: this.store.comments("issue", i.id).map((c) => ({ ...c, author: this.view(c.authorId) })) };
  }

  /** Bugs assigned to the actor across every group, newest first. */
  myBugs(actor: User) {
    const out: Array<ReturnType<TeamService["issueView"]> & { group: string; project: string }> = [];
    for (const g of this.store.groupsOf(actor.id))
      for (const p of this.store.projectsOf(g.id))
        for (const i of this.store.issues(p.id, { kind: "bug", assigneeId: actor.id })) out.push({ ...this.issueView(i), group: g.slug, project: p.slug });
    return out.sort((a, b) => b.updatedAt - a.updatedAt);
  }

  createIssue(actor: User, g: string, p: string, d: { title: string; body?: string; labels?: string[]; assigneeId?: string | null }) {
    const { group, project } = this.access(actor, g, p);
    if (!d.title?.trim()) throw new TeamError("title is required");
    if (d.assigneeId && !this.store.member(group.id, d.assigneeId)) throw new TeamError("assignee is not a member of this group");
    const t = now();
    const issue: Issue = {
      id: id(), projectId: project.id, number: this.store.nextNumber(project.id), kind: "issue", title: d.title.trim(), body: d.body ?? "", state: "open",
      labels: (d.labels ?? []).map((l) => l.trim()).filter(Boolean), authorId: actor.id, assigneeId: d.assigneeId ?? null, severity: null, bugStatus: null,
      culpritCommit: null, routedBy: null, fixCommit: null, createdAt: t, updatedAt: t,
    };
    this.store.insertIssue(issue);
    const link = `${group.slug}/${project.slug}#${issue.number}`;
    this.activity(group.id, project.id, actor.id, "opened", `opened issue #${issue.number} ${issue.title}`, link);
    if (issue.assigneeId && issue.assigneeId !== actor.id) this.notify(issue.assigneeId, "issue_assigned", `#${issue.number} ${issue.title}`, `${actor.name} assigned you an issue in ${group.slug}/${project.slug}`, link);
    this.emit({ type: "changed", groupId: group.id, projectId: project.id, what: "issues" });
    return this.issueView(issue);
  }

  updateIssue(actor: User, g: string, p: string, n: number, d: { title?: string; body?: string; state?: "open" | "closed"; labels?: string[]; assigneeId?: string | null }) {
    const { group, project, role } = this.access(actor, g, p);
    const i = this.store.issue(project.id, n);
    if (!i) throw new TeamError(`#${n} not found`, 404);
    if (i.authorId !== actor.id && i.assigneeId !== actor.id && !atLeast(role, "developer")) throw new TeamError("only the author, the assignee or developers can edit this", 403);
    if (d.assigneeId && !this.store.member(group.id, d.assigneeId)) throw new TeamError("assignee is not a member of this group");
    const next: Issue = {
      ...i,
      title: d.title?.trim() || i.title,
      body: d.body ?? i.body,
      state: d.state ?? i.state,
      labels: d.labels ?? i.labels,
      assigneeId: d.assigneeId === undefined ? i.assigneeId : d.assigneeId,
      updatedAt: now(),
    };
    this.store.updateIssue(next);
    const link = `${group.slug}/${project.slug}#${n}`;
    if (next.assigneeId && next.assigneeId !== i.assigneeId && next.assigneeId !== actor.id)
      this.notify(next.assigneeId, next.kind === "bug" ? "bug_assigned" : "issue_assigned", `#${n} ${next.title}`, `${actor.name} assigned this ${next.kind} to you`, link);
    if (next.state !== i.state) this.activity(group.id, project.id, actor.id, next.state === "closed" ? "closed" : "reopened", `${next.state === "closed" ? "closed" : "reopened"} #${n} ${next.title}`, link);
    this.emit({ type: "changed", groupId: group.id, projectId: project.id, what: "issues" });
    return this.issueView(next);
  }

  comment(actor: User, g: string, p: string, n: number, body: string) {
    const { group, project } = this.access(actor, g, p);
    const i = this.store.issue(project.id, n);
    if (!i) throw new TeamError(`#${n} not found`, 404);
    if (!body.trim()) throw new TeamError("comment is empty");
    const c: Comment = { id: id(), targetKind: "issue", targetId: i.id, authorId: actor.id, body: body.trim(), review: null, createdAt: now() };
    this.store.insertComment(c);
    this.store.updateIssue({ ...i, updatedAt: now() });
    const link = `${group.slug}/${project.slug}#${n}`;
    for (const uid of new Set([i.authorId, i.assigneeId].filter((x): x is string => !!x && x !== actor.id)))
      this.notify(uid, "comment", `#${n} ${i.title}`, `${actor.name}: ${body.trim().slice(0, 200)}`, link);
    this.emit({ type: "changed", groupId: group.id, projectId: project.id, what: "issues" });
    return { ...c, author: this.view(actor.id) };
  }

  /** Files a bug as a complete markdown report and routes it to whoever introduced the code. */
  async reportBug(actor: User, g: string, p: string, input: BugReportInput & { assigneeId?: string | null }) {
    const { group, project, repo } = this.access(actor, g, p);
    if (!input.title?.trim()) throw new TeamError("title is required");
    // The reporter's branch may exist only on their machine; fall back to the default branch.
    let ref = input.ref?.trim() || project.defaultBranch;
    let refSha = await this.git.resolve(repo, ref);
    if (!refSha && ref !== project.defaultBranch) {
      ref = project.defaultBranch;
      refSha = await this.git.resolve(repo, ref);
    }
    const number = this.store.nextNumber(project.id);
    const ctx: BugContext = {
      project: `${group.slug}/${project.slug}`, number, reporter: { name: actor.name, email: actor.email }, createdAt: now(), ref,
      blame: null, excerpt: null, commitInfo: null, routedTo: null, pull: null, pushedBy: null,
    };

    if (input.file && refSha) {
      const text = await this.git.show(repo, refSha, input.file);
      if (text !== null && input.line) {
        const lines = text.split("\n");
        const start = Math.max(1, input.line - 6);
        ctx.excerpt = { startLine: start, lines: lines.slice(start - 1, Math.min(lines.length, input.line + 6)) };
        ctx.blame = await this.git.blame(repo, refSha, input.file, input.line);
      }
    }
    const given = input.commit?.trim() ? await this.git.resolve(repo, input.commit.trim()) : null;
    const culprit = given ?? ctx.blame?.commit ?? null;
    if (culprit) {
      const c = await this.git.commit(repo, culprit);
      if (c) ctx.commitInfo = { commit: c.sha, author: c.author, email: c.email, time: c.time, subject: c.subject, files: c.files };
      const push = this.store.pushContaining(project.id, culprit);
      if (push) ctx.pushedBy = { name: this.view(push.userId)!.name, at: push.createdAt };
      const pr = await this.pullIntroducing(repo, project.id, culprit);
      if (pr) ctx.pull = { number: pr.number, title: pr.title, author: this.view(pr.authorId)!.name };
    }

    // Routing: explicit choice, then the commit author, then whoever pushed it, then the PR author.
    let assigneeId: string | null = null;
    let routedBy: string | null = null;
    const isMember = (uid: string | null | undefined): uid is string => !!uid && !!this.store.member(group.id, uid);
    if (input.assigneeId) {
      if (!isMember(input.assigneeId)) throw new TeamError("assignee is not a member of this group");
      assigneeId = input.assigneeId;
      routedBy = "chosen by the reporter";
    } else if (ctx.commitInfo) {
      const author = this.store.userByEmail(ctx.commitInfo.email.toLowerCase());
      const push = this.store.pushContaining(project.id, ctx.commitInfo.commit);
      const pr = ctx.pull ? this.store.pull(project.id, ctx.pull.number) : null;
      if (isMember(author?.id)) [assigneeId, routedBy] = [author!.id, `wrote commit ${ctx.commitInfo.commit.slice(0, 10)}`];
      else if (isMember(push?.userId)) [assigneeId, routedBy] = [push!.userId, `pushed commit ${ctx.commitInfo.commit.slice(0, 10)}`];
      else if (isMember(pr?.authorId)) [assigneeId, routedBy] = [pr!.authorId, `opened pull request !${pr!.number}`];
    }
    if (assigneeId) {
      const u = this.view(assigneeId)!;
      ctx.routedTo = { name: u.name, email: u.email, via: routedBy! };
    }

    const t = now();
    const issue: Issue = {
      id: id(), projectId: project.id, number, kind: "bug", title: input.title.trim(), body: renderBugReport(input, ctx), state: "open",
      labels: ["bug"], authorId: actor.id, assigneeId, severity: (input.severity ?? "medium") as Severity, bugStatus: "new",
      culpritCommit: ctx.commitInfo?.commit ?? null, routedBy, fixCommit: null, createdAt: t, updatedAt: t,
    };
    this.store.insertIssue(issue);
    const link = `${group.slug}/${project.slug}#${number}`;
    this.activity(group.id, project.id, actor.id, "reported", `reported bug #${number} ${issue.title}`, link);
    const recipients = assigneeId ? [assigneeId] : this.store.members(group.id).filter((m) => atLeast(m.role, "maintainer")).map((m) => m.userId);
    for (const uid of recipients)
      if (uid !== actor.id)
        this.notify(uid, "bug_assigned", `Bug #${number}: ${issue.title}`, assigneeId ? `${actor.name} reported a bug in code you ${routedBy}.` : `${actor.name} reported a bug nobody could be traced to; please triage.`, link);
    this.emit({ type: "changed", groupId: group.id, projectId: project.id, what: "issues" });
    return this.issueView(issue);
  }

  private async pullIntroducing(repo: string, projectId: string, sha: string): Promise<PullRequest | null> {
    const direct = this.store.pullByMergeCommit(projectId, sha);
    if (direct) return direct;
    for (const pr of this.store.pulls(projectId, "merged").slice(0, 50)) {
      if (!pr.mergeCommit) continue;
      const head = await this.git.resolve(repo, `${pr.mergeCommit}^2`);
      const base = await this.git.resolve(repo, `${pr.mergeCommit}^1`);
      if (head && base && (await this.git.isAncestor(repo, sha, head)) && !(await this.git.isAncestor(repo, sha, base))) return pr;
    }
    return null;
  }

  setBugStatus(actor: User, g: string, p: string, n: number, d: { status: BugStatus; fixCommit?: string; comment?: string }) {
    const { group, project, role } = this.access(actor, g, p);
    const i = this.store.issue(project.id, n);
    if (!i || i.kind !== "bug") throw new TeamError(`bug #${n} not found`, 404);
    if (i.assigneeId !== actor.id && !atLeast(role, "developer")) throw new TeamError("only the assignee or developers can change a bug's status", 403);
    const closed = d.status === "fixed" || d.status === "wont_fix";
    const next: Issue = { ...i, bugStatus: d.status, fixCommit: d.fixCommit?.trim() || i.fixCommit, state: closed ? "closed" : "open", updatedAt: now() };
    this.store.updateIssue(next);
    const label = d.status.replace("_", " ");
    if (d.comment?.trim() || d.fixCommit) {
      const body = [`**Status: ${label}**${d.fixCommit ? ` — fixed in \`${d.fixCommit.slice(0, 12)}\`` : ""}`, d.comment?.trim() ?? ""].filter(Boolean).join("\n\n");
      this.store.insertComment({ id: id(), targetKind: "issue", targetId: i.id, authorId: actor.id, body, review: null, createdAt: now() });
    }
    const link = `${group.slug}/${project.slug}#${n}`;
    if (i.authorId !== actor.id) this.notify(i.authorId, "bug_status", `Bug #${n} is ${label}`, `${actor.name}: ${d.comment?.trim().slice(0, 200) || label}`, link);
    this.activity(group.id, project.id, actor.id, "bug_status", `marked bug #${n} ${label}`, link);
    this.emit({ type: "changed", groupId: group.id, projectId: project.id, what: "issues" });
    return this.issueView(next);
  }

  // ---- pull requests ------------------------------------------------------------------------------

  private pullView(pr: PullRequest) {
    const reviews = this.store.comments("pull", pr.id).filter((c) => c.review);
    const latest = new Map<string, ReviewState>();
    for (const r of reviews) if (r.review !== "comment") latest.set(r.authorId, r.review!);
    return { ...pr, author: this.view(pr.authorId), mergedByUser: this.view(pr.mergedBy), approvals: [...latest.values()].filter((v) => v === "approve").length, changesRequested: [...latest.values()].filter((v) => v === "request_changes").length };
  }

  pulls(actor: User, g: string, p: string, state?: string) {
    const { project } = this.access(actor, g, p);
    return this.store.pulls(project.id, state).map((x) => this.pullView(x));
  }

  async createPull(actor: User, g: string, p: string, d: { title: string; body?: string; source: string; target?: string }) {
    const { group, project, repo } = this.access(actor, g, p, "developer");
    const source = assertBranch(d.source);
    const target = assertBranch(d.target || project.defaultBranch);
    if (source === target) throw new TeamError("source and target branches are the same");
    if (!(await this.git.resolve(repo, `refs/heads/${source}`))) throw new TeamError(`branch ${source} does not exist on the hub; push it first`, 404);
    if (!(await this.git.resolve(repo, `refs/heads/${target}`))) throw new TeamError(`branch ${target} does not exist`, 404);
    if (this.store.pulls(project.id, "open").some((x) => x.sourceBranch === source && x.targetBranch === target)) throw new TeamError(`an open pull request for ${source} → ${target} already exists`, 409);
    const t = now();
    const pr: PullRequest = { id: id(), projectId: project.id, number: this.store.nextNumber(project.id), title: d.title?.trim() || source, body: d.body ?? "", sourceBranch: source, targetBranch: target, authorId: actor.id, state: "open", mergedBy: null, mergeCommit: null, createdAt: t, updatedAt: t };
    this.store.insertPull(pr);
    const link = `${group.slug}/${project.slug}!${pr.number}`;
    this.activity(group.id, project.id, actor.id, "opened", `opened pull request !${pr.number} ${pr.title}`, link);
    for (const m of this.store.members(group.id)) if (m.userId !== actor.id && atLeast(m.role, "maintainer")) this.notify(m.userId, "pull_opened", `!${pr.number} ${pr.title}`, `${actor.name} wants to merge ${source} into ${target}`, link);
    this.emit({ type: "changed", groupId: group.id, projectId: project.id, what: "pulls" });
    return this.pullView(pr);
  }

  async pull(actor: User, g: string, p: string, n: number) {
    const { project, repo } = this.access(actor, g, p);
    const pr = this.store.pull(project.id, n);
    if (!pr) throw new TeamError(`!${n} not found`, 404);
    const comments = this.store.comments("pull", pr.id).map((c) => ({ ...c, author: this.view(c.authorId) }));
    let commits: Awaited<ReturnType<GitHost["log"]>> = [];
    let diff = { diff: "", truncated: false, files: [] as Array<{ path: string; added: number; removed: number }> };
    let mergeable: boolean | null = null;
    if (pr.state === "open") {
      const base = await this.git.resolve(repo, `refs/heads/${pr.targetBranch}`);
      const head = await this.git.resolve(repo, `refs/heads/${pr.sourceBranch}`);
      if (base && head) {
        commits = await this.git.log(repo, `${base}..${head}`, 200);
        const mb = (await this.git.mergeBase(repo, base, head)) ?? base;
        diff = await this.git.diff(repo, mb, head, 600_000);
        mergeable = (await this.git.isAncestor(repo, base, head)) || "sha" in (await this.git.merge(repo, base, head, "probe", { name: "probe", email: "probe@hub" }));
      }
    } else if (pr.mergeCommit) {
      const base = await this.git.resolve(repo, `${pr.mergeCommit}^1`);
      if (base) {
        commits = await this.git.log(repo, `${base}..${pr.mergeCommit}`, 200);
        diff = await this.git.diff(repo, base, pr.mergeCommit, 600_000);
      }
    }
    return { ...this.pullView(pr), comments, commits, ...diff, mergeable };
  }

  review(actor: User, g: string, p: string, n: number, d: { state: ReviewState; body?: string }) {
    const { group, project } = this.access(actor, g, p, d.state === "comment" ? "reporter" : "developer");
    const pr = this.store.pull(project.id, n);
    if (!pr) throw new TeamError(`!${n} not found`, 404);
    if (d.state !== "comment" && pr.authorId === actor.id) throw new TeamError("you cannot approve or reject your own pull request", 403);
    const c: Comment = { id: id(), targetKind: "pull", targetId: pr.id, authorId: actor.id, body: d.body?.trim() ?? "", review: d.state, createdAt: now() };
    this.store.insertComment(c);
    this.store.updatePull({ ...pr, updatedAt: now() });
    const link = `${group.slug}/${project.slug}!${n}`;
    const verb = d.state === "approve" ? "approved" : d.state === "request_changes" ? "requested changes on" : "commented on";
    if (pr.authorId !== actor.id) this.notify(pr.authorId, "review", `!${n} ${pr.title}`, `${actor.name} ${verb} your pull request${d.body?.trim() ? `: ${d.body.trim().slice(0, 160)}` : ""}`, link);
    this.activity(group.id, project.id, actor.id, "reviewed", `${verb} !${n} ${pr.title}`, link);
    this.emit({ type: "changed", groupId: group.id, projectId: project.id, what: "pulls" });
    return { ...c, author: this.view(actor.id) };
  }

  async merge(actor: User, g: string, p: string, n: number) {
    const { group, project, role, repo } = this.access(actor, g, p, "developer");
    const pr = this.store.pull(project.id, n);
    if (!pr) throw new TeamError(`!${n} not found`, 404);
    if (pr.state !== "open") throw new TeamError(`!${n} is already ${pr.state}`, 409);
    const view = this.pullView(pr);
    if (!atLeast(role, "maintainer") && !(pr.authorId === actor.id && view.approvals > 0)) throw new TeamError("merging needs a maintainer, or the author after an approval", 403);
    if (view.changesRequested > 0 && !atLeast(role, "maintainer")) throw new TeamError("changes were requested; address them first", 409);
    const base = await this.git.resolve(repo, `refs/heads/${pr.targetBranch}`);
    const head = await this.git.resolve(repo, `refs/heads/${pr.sourceBranch}`);
    if (!base || !head) throw new TeamError("a branch of this pull request no longer exists", 409);
    const before = await this.git.refs(repo);
    const r = await this.git.merge(repo, base, head, `Merge pull request !${n} from ${pr.sourceBranch}\n\n${pr.title}`, { name: actor.name, email: actor.email }, `refs/heads/${pr.targetBranch}`);
    if ("conflicts" in r) throw new TeamError(`merge conflicts in: ${r.conflicts.join(", ") || "unknown files"}. Merge ${pr.targetBranch} into ${pr.sourceBranch}, resolve, and push.`, 409);
    const next: PullRequest = { ...pr, state: "merged", mergedBy: actor.id, mergeCommit: r.sha, updatedAt: now() };
    this.store.updatePull(next);
    await this.recordPush(actor, g, p, before, await this.git.refs(repo), "http");
    const link = `${group.slug}/${project.slug}!${n}`;
    this.activity(group.id, project.id, actor.id, "merged", `merged !${n} ${pr.title} into ${pr.targetBranch}`, link);
    if (pr.authorId !== actor.id) this.notify(pr.authorId, "pull_merged", `!${n} ${pr.title} was merged`, `${actor.name} merged your pull request into ${pr.targetBranch}`, link);
    this.emit({ type: "changed", groupId: group.id, projectId: project.id, what: "pulls" });
    return this.pullView(next);
  }

  closePull(actor: User, g: string, p: string, n: number, reopen = false) {
    const { group, project, role } = this.access(actor, g, p);
    const pr = this.store.pull(project.id, n);
    if (!pr) throw new TeamError(`!${n} not found`, 404);
    if (pr.state === "merged") throw new TeamError("already merged", 409);
    if (pr.authorId !== actor.id && !atLeast(role, "maintainer")) throw new TeamError("only the author or a maintainer can close this", 403);
    const next: PullRequest = { ...pr, state: reopen ? "open" : "closed", updatedAt: now() };
    this.store.updatePull(next);
    this.emit({ type: "changed", groupId: group.id, projectId: project.id, what: "pulls" });
    return this.pullView(next);
  }

  // ---- inbox and activity -------------------------------------------------------------------------

  notifications(actor: User, unreadOnly = false) {
    return this.store.notifications(actor.id, unreadOnly, 200);
  }

  markRead(actor: User, ids: string[] | "all") {
    this.store.markRead(actor.id, ids);
  }

  feed(actor: User, g: string, p?: string) {
    const { group } = this.groupFor(actor, g);
    const project = p ? this.store.project(group.id, p) : null;
    return this.store.activity(group.id, project?.id ?? null, 100).map((a) => ({ ...a, actor: this.view(a.actorId) }));
  }

  /** Group ids the user may see events for. */
  groupIdsOf(userId: string): Set<string> {
    return new Set(this.store.groupsOf(userId).map((g) => g.id));
  }

  private notify(userId: string, kind: NotificationKind, title: string, body: string, link: string): void {
    const n: Notification = { id: id(), userId, kind, title, body, link, read: false, createdAt: now() };
    this.store.insertNotification(n);
    this.emit({ type: "notification", userId, notification: n });
  }

  private activity(groupId: string, projectId: string | null, actorId: string, verb: string, summary: string, link: string): void {
    const a: Activity = { id: id(), groupId, projectId, actorId, verb, summary, link, createdAt: now() };
    this.store.insertActivity(a);
    this.emit({ type: "activity", groupId, activity: a });
  }
}
