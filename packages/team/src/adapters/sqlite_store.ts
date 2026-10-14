import { DatabaseSync } from "node:sqlite";
import { mkdirSync } from "node:fs";
import path from "node:path";
import type { Activity, Comment, Group, Invite, Issue, Membership, Notification, Project, PullRequest, PushRecord, Role, SshKey, User } from "../domain/types";
import type { TeamStore } from "../ports";

const SCHEMA = `
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL UNIQUE, token_hash TEXT NOT NULL, created_at INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS users_token ON users(token_hash);
CREATE TABLE IF NOT EXISTS ssh_keys (id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE, public_key TEXT NOT NULL, fingerprint TEXT NOT NULL UNIQUE, created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS groups_ (id TEXT PRIMARY KEY, slug TEXT NOT NULL UNIQUE, name TEXT NOT NULL, description TEXT NOT NULL, created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS members (group_id TEXT NOT NULL REFERENCES groups_(id) ON DELETE CASCADE, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE, role TEXT NOT NULL, joined_at INTEGER NOT NULL, PRIMARY KEY (group_id, user_id));
CREATE TABLE IF NOT EXISTS invites (code TEXT PRIMARY KEY, group_id TEXT NOT NULL REFERENCES groups_(id) ON DELETE CASCADE, role TEXT NOT NULL, created_by TEXT NOT NULL, expires_at INTEGER NOT NULL, uses_left INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, group_id TEXT NOT NULL REFERENCES groups_(id) ON DELETE CASCADE, slug TEXT NOT NULL, name TEXT NOT NULL, description TEXT NOT NULL,
  default_branch TEXT NOT NULL, protect_default INTEGER NOT NULL, next_number INTEGER NOT NULL DEFAULT 1, created_at INTEGER NOT NULL, UNIQUE (group_id, slug));
CREATE TABLE IF NOT EXISTS issues (id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE, number INTEGER NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL,
  state TEXT NOT NULL, labels TEXT NOT NULL, author_id TEXT NOT NULL, assignee_id TEXT, severity TEXT, bug_status TEXT, culprit_commit TEXT, routed_by TEXT, fix_commit TEXT,
  created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, UNIQUE (project_id, number));
CREATE TABLE IF NOT EXISTS pulls (id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE, number INTEGER NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL,
  source_branch TEXT NOT NULL, target_branch TEXT NOT NULL, author_id TEXT NOT NULL, state TEXT NOT NULL, merged_by TEXT, merge_commit TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
  UNIQUE (project_id, number));
CREATE TABLE IF NOT EXISTS comments (id TEXT PRIMARY KEY, target_kind TEXT NOT NULL, target_id TEXT NOT NULL, author_id TEXT NOT NULL, body TEXT NOT NULL, review TEXT, created_at INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS comments_target ON comments(target_kind, target_id);
CREATE TABLE IF NOT EXISTS pushes (id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE, user_id TEXT NOT NULL, ref TEXT NOT NULL, before_sha TEXT NOT NULL, after_sha TEXT NOT NULL,
  commits TEXT NOT NULL, transport TEXT NOT NULL, created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS push_commits (push_id TEXT NOT NULL REFERENCES pushes(id) ON DELETE CASCADE, project_id TEXT NOT NULL, sha TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS push_commits_sha ON push_commits(project_id, sha);
CREATE TABLE IF NOT EXISTS notifications (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL, link TEXT NOT NULL, read INTEGER NOT NULL, created_at INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS notifications_user ON notifications(user_id, created_at);
CREATE TABLE IF NOT EXISTS activity (id TEXT PRIMARY KEY, group_id TEXT NOT NULL, project_id TEXT, actor_id TEXT NOT NULL, verb TEXT NOT NULL, summary TEXT NOT NULL, link TEXT NOT NULL, created_at INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS activity_group ON activity(group_id, created_at);
`;

type Row = Record<string, unknown>;
const s = (v: unknown) => (v === null || v === undefined ? null : String(v));
const n = (v: unknown) => Number(v);

const toUser = (r: Row): User => ({ id: String(r.id), name: String(r.name), email: String(r.email), createdAt: n(r.created_at) });
const toGroup = (r: Row): Group => ({ id: String(r.id), slug: String(r.slug), name: String(r.name), description: String(r.description), createdAt: n(r.created_at) });
const toProject = (r: Row): Project => ({
  id: String(r.id), groupId: String(r.group_id), slug: String(r.slug), name: String(r.name), description: String(r.description),
  defaultBranch: String(r.default_branch), protectDefault: !!n(r.protect_default), createdAt: n(r.created_at),
});
const toIssue = (r: Row): Issue => ({
  id: String(r.id), projectId: String(r.project_id), number: n(r.number), kind: r.kind as Issue["kind"], title: String(r.title), body: String(r.body),
  state: r.state as Issue["state"], labels: JSON.parse(String(r.labels)) as string[], authorId: String(r.author_id), assigneeId: s(r.assignee_id),
  severity: s(r.severity) as Issue["severity"], bugStatus: s(r.bug_status) as Issue["bugStatus"], culpritCommit: s(r.culprit_commit), routedBy: s(r.routed_by),
  fixCommit: s(r.fix_commit), createdAt: n(r.created_at), updatedAt: n(r.updated_at),
});
const toPull = (r: Row): PullRequest => ({
  id: String(r.id), projectId: String(r.project_id), number: n(r.number), title: String(r.title), body: String(r.body), sourceBranch: String(r.source_branch),
  targetBranch: String(r.target_branch), authorId: String(r.author_id), state: r.state as PullRequest["state"], mergedBy: s(r.merged_by), mergeCommit: s(r.merge_commit),
  createdAt: n(r.created_at), updatedAt: n(r.updated_at),
});
const toPush = (r: Row): PushRecord => ({
  id: String(r.id), projectId: String(r.project_id), userId: String(r.user_id), ref: String(r.ref), before: String(r.before_sha), after: String(r.after_sha),
  commits: JSON.parse(String(r.commits)) as string[], transport: r.transport as PushRecord["transport"], createdAt: n(r.created_at),
});

export class SqliteTeamStore implements TeamStore {
  private db: DatabaseSync;

  constructor(file: string) {
    if (file !== ":memory:") mkdirSync(path.dirname(file), { recursive: true });
    this.db = new DatabaseSync(file);
    this.db.exec("PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON; PRAGMA busy_timeout = 5000;");
    this.db.exec(SCHEMA);
  }

  close(): void {
    this.db.close();
  }

  private get(sql: string, ...a: Array<string | number | null>): Row | null {
    return (this.db.prepare(sql).get(...a) as Row | undefined) ?? null;
  }
  private all(sql: string, ...a: Array<string | number | null>): Row[] {
    return this.db.prepare(sql).all(...a) as Row[];
  }
  private run(sql: string, ...a: Array<string | number | null>): void {
    this.db.prepare(sql).run(...a);
  }

  meta(key: string) {
    return s(this.get("SELECT value FROM meta WHERE key = ?", key)?.value);
  }
  setMeta(key: string, value: string) {
    this.run("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", key, value);
  }

  insertUser(u: User, tokenHash: string) {
    this.run("INSERT INTO users (id, name, email, token_hash, created_at) VALUES (?, ?, ?, ?, ?)", u.id, u.name, u.email, tokenHash, u.createdAt);
  }
  updateUser(u: User) {
    this.run("UPDATE users SET name = ?, email = ? WHERE id = ?", u.name, u.email, u.id);
  }
  userById(id: string) {
    const r = this.get("SELECT * FROM users WHERE id = ?", id);
    return r ? toUser(r) : null;
  }
  userByTokenHash(hash: string) {
    const r = this.get("SELECT * FROM users WHERE token_hash = ?", hash);
    return r ? toUser(r) : null;
  }
  userByEmail(email: string) {
    const r = this.get("SELECT * FROM users WHERE email = ?", email.toLowerCase());
    return r ? toUser(r) : null;
  }
  setUserToken(userId: string, tokenHash: string) {
    this.run("UPDATE users SET token_hash = ? WHERE id = ?", tokenHash, userId);
  }
  listUsers(ids: string[]) {
    return ids.map((i) => this.userById(i)).filter((u): u is User => !!u);
  }

  insertKey(k: SshKey) {
    this.run("INSERT INTO ssh_keys (id, user_id, public_key, fingerprint, created_at) VALUES (?, ?, ?, ?, ?)", k.id, k.userId, k.publicKey, k.fingerprint, k.createdAt);
  }
  keysOf(userId: string) {
    return this.all("SELECT * FROM ssh_keys WHERE user_id = ? ORDER BY created_at", userId).map((r) => ({ id: String(r.id), userId: String(r.user_id), publicKey: String(r.public_key), fingerprint: String(r.fingerprint), createdAt: n(r.created_at) }));
  }
  keyByFingerprint(fp: string) {
    const r = this.get("SELECT * FROM ssh_keys WHERE fingerprint = ?", fp);
    return r ? { id: String(r.id), userId: String(r.user_id), publicKey: String(r.public_key), fingerprint: String(r.fingerprint), createdAt: n(r.created_at) } : null;
  }
  deleteKey(id: string, userId: string) {
    this.run("DELETE FROM ssh_keys WHERE id = ? AND user_id = ?", id, userId);
  }

  insertGroup(g: Group) {
    this.run("INSERT INTO groups_ (id, slug, name, description, created_at) VALUES (?, ?, ?, ?, ?)", g.id, g.slug, g.name, g.description, g.createdAt);
  }
  groupBySlug(slug: string) {
    const r = this.get("SELECT * FROM groups_ WHERE slug = ?", slug);
    return r ? toGroup(r) : null;
  }
  groupById(id: string) {
    const r = this.get("SELECT * FROM groups_ WHERE id = ?", id);
    return r ? toGroup(r) : null;
  }
  groupsOf(userId: string) {
    return this.all("SELECT g.*, m.role FROM groups_ g JOIN members m ON m.group_id = g.id WHERE m.user_id = ? ORDER BY g.name", userId).map((r) => ({ ...toGroup(r), role: r.role as Role }));
  }

  upsertMember(m: Membership) {
    this.run("INSERT INTO members (group_id, user_id, role, joined_at) VALUES (?, ?, ?, ?) ON CONFLICT(group_id, user_id) DO UPDATE SET role = excluded.role", m.groupId, m.userId, m.role, m.joinedAt);
  }
  removeMember(groupId: string, userId: string) {
    this.run("DELETE FROM members WHERE group_id = ? AND user_id = ?", groupId, userId);
  }
  member(groupId: string, userId: string) {
    const r = this.get("SELECT * FROM members WHERE group_id = ? AND user_id = ?", groupId, userId);
    return r ? { groupId, userId, role: r.role as Role, joinedAt: n(r.joined_at) } : null;
  }
  members(groupId: string) {
    return this.all("SELECT * FROM members WHERE group_id = ? ORDER BY joined_at", groupId).map((r) => ({ groupId, userId: String(r.user_id), role: r.role as Role, joinedAt: n(r.joined_at) }));
  }

  insertInvite(i: Invite) {
    this.run("INSERT INTO invites (code, group_id, role, created_by, expires_at, uses_left) VALUES (?, ?, ?, ?, ?, ?)", i.code, i.groupId, i.role, i.createdBy, i.expiresAt, i.usesLeft);
  }
  invite(code: string) {
    const r = this.get("SELECT * FROM invites WHERE code = ?", code);
    return r ? { code, groupId: String(r.group_id), role: r.role as Role, createdBy: String(r.created_by), expiresAt: n(r.expires_at), usesLeft: n(r.uses_left) } : null;
  }
  useInvite(code: string) {
    this.run("UPDATE invites SET uses_left = uses_left - 1 WHERE code = ?", code);
  }
  invitesOf(groupId: string) {
    return this.all("SELECT * FROM invites WHERE group_id = ? ORDER BY expires_at DESC", groupId).map((r) => ({
      code: String(r.code), groupId, role: r.role as Role, createdBy: String(r.created_by), expiresAt: n(r.expires_at), usesLeft: n(r.uses_left),
    }));
  }
  deleteInvite(code: string) {
    this.run("DELETE FROM invites WHERE code = ?", code);
  }

  insertProject(p: Project) {
    this.run(
      "INSERT INTO projects (id, group_id, slug, name, description, default_branch, protect_default, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
      p.id, p.groupId, p.slug, p.name, p.description, p.defaultBranch, p.protectDefault ? 1 : 0, p.createdAt,
    );
  }
  updateProject(p: Project) {
    this.run("UPDATE projects SET name = ?, description = ?, default_branch = ?, protect_default = ? WHERE id = ?", p.name, p.description, p.defaultBranch, p.protectDefault ? 1 : 0, p.id);
  }
  project(groupId: string, slug: string) {
    const r = this.get("SELECT * FROM projects WHERE group_id = ? AND slug = ?", groupId, slug);
    return r ? toProject(r) : null;
  }
  projectById(id: string) {
    const r = this.get("SELECT * FROM projects WHERE id = ?", id);
    return r ? toProject(r) : null;
  }
  projectsOf(groupId: string) {
    return this.all("SELECT * FROM projects WHERE group_id = ? ORDER BY name", groupId).map(toProject);
  }

  nextNumber(projectId: string) {
    const r = this.get("UPDATE projects SET next_number = next_number + 1 WHERE id = ? RETURNING next_number - 1 AS n", projectId);
    return n(r?.n ?? 1);
  }
  insertIssue(i: Issue) {
    this.run(
      `INSERT INTO issues (id, project_id, number, kind, title, body, state, labels, author_id, assignee_id, severity, bug_status, culprit_commit, routed_by, fix_commit, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      i.id, i.projectId, i.number, i.kind, i.title, i.body, i.state, JSON.stringify(i.labels), i.authorId, i.assigneeId, i.severity, i.bugStatus, i.culpritCommit, i.routedBy, i.fixCommit, i.createdAt, i.updatedAt,
    );
  }
  updateIssue(i: Issue) {
    this.run(
      "UPDATE issues SET title = ?, body = ?, state = ?, labels = ?, assignee_id = ?, severity = ?, bug_status = ?, fix_commit = ?, updated_at = ? WHERE id = ?",
      i.title, i.body, i.state, JSON.stringify(i.labels), i.assigneeId, i.severity, i.bugStatus, i.fixCommit, i.updatedAt, i.id,
    );
  }
  issue(projectId: string, number: number) {
    const r = this.get("SELECT * FROM issues WHERE project_id = ? AND number = ?", projectId, number);
    return r ? toIssue(r) : null;
  }
  issues(projectId: string, f: { kind?: string; state?: string; assigneeId?: string }) {
    const where = ["project_id = ?"];
    const args: string[] = [projectId];
    if (f.kind) (where.push("kind = ?"), args.push(f.kind));
    if (f.state) (where.push("state = ?"), args.push(f.state));
    if (f.assigneeId) (where.push("assignee_id = ?"), args.push(f.assigneeId));
    return this.all(`SELECT * FROM issues WHERE ${where.join(" AND ")} ORDER BY number DESC`, ...args).map(toIssue);
  }

  insertPull(p: PullRequest) {
    this.run(
      "INSERT INTO pulls (id, project_id, number, title, body, source_branch, target_branch, author_id, state, merged_by, merge_commit, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
      p.id, p.projectId, p.number, p.title, p.body, p.sourceBranch, p.targetBranch, p.authorId, p.state, p.mergedBy, p.mergeCommit, p.createdAt, p.updatedAt,
    );
  }
  updatePull(p: PullRequest) {
    this.run("UPDATE pulls SET title = ?, body = ?, state = ?, merged_by = ?, merge_commit = ?, updated_at = ? WHERE id = ?", p.title, p.body, p.state, p.mergedBy, p.mergeCommit, p.updatedAt, p.id);
  }
  pull(projectId: string, number: number) {
    const r = this.get("SELECT * FROM pulls WHERE project_id = ? AND number = ?", projectId, number);
    return r ? toPull(r) : null;
  }
  pulls(projectId: string, state?: string) {
    return (state ? this.all("SELECT * FROM pulls WHERE project_id = ? AND state = ? ORDER BY number DESC", projectId, state) : this.all("SELECT * FROM pulls WHERE project_id = ? ORDER BY number DESC", projectId)).map(toPull);
  }
  pullByMergeCommit(projectId: string, sha: string) {
    const r = this.get("SELECT * FROM pulls WHERE project_id = ? AND merge_commit = ?", projectId, sha);
    return r ? toPull(r) : null;
  }

  insertComment(c: Comment) {
    this.run("INSERT INTO comments (id, target_kind, target_id, author_id, body, review, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)", c.id, c.targetKind, c.targetId, c.authorId, c.body, c.review, c.createdAt);
  }
  comments(kind: "issue" | "pull", targetId: string) {
    return this.all("SELECT * FROM comments WHERE target_kind = ? AND target_id = ? ORDER BY created_at", kind, targetId).map((r) => ({
      id: String(r.id), targetKind: kind, targetId, authorId: String(r.author_id), body: String(r.body), review: s(r.review) as Comment["review"], createdAt: n(r.created_at),
    }));
  }

  insertPush(p: PushRecord) {
    this.run(
      "INSERT INTO pushes (id, project_id, user_id, ref, before_sha, after_sha, commits, transport, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
      p.id, p.projectId, p.userId, p.ref, p.before, p.after, JSON.stringify(p.commits), p.transport, p.createdAt,
    );
    const ins = this.db.prepare("INSERT INTO push_commits (push_id, project_id, sha) VALUES (?, ?, ?)");
    for (const c of p.commits) ins.run(p.id, p.projectId, c);
  }
  pushes(projectId: string, limit: number) {
    return this.all("SELECT * FROM pushes WHERE project_id = ? ORDER BY created_at DESC LIMIT ?", projectId, limit).map(toPush);
  }
  pushContaining(projectId: string, sha: string) {
    const r = this.get("SELECT p.* FROM push_commits c JOIN pushes p ON p.id = c.push_id WHERE c.project_id = ? AND c.sha = ? ORDER BY p.created_at ASC LIMIT 1", projectId, sha);
    return r ? toPush(r) : null;
  }

  insertNotification(x: Notification) {
    this.run("INSERT INTO notifications (id, user_id, kind, title, body, link, read, created_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)", x.id, x.userId, x.kind, x.title, x.body, x.link, x.createdAt);
  }
  notifications(userId: string, unreadOnly: boolean, limit: number) {
    return this.all(`SELECT * FROM notifications WHERE user_id = ? ${unreadOnly ? "AND read = 0" : ""} ORDER BY created_at DESC LIMIT ?`, userId, limit).map((r) => ({
      id: String(r.id), userId, kind: r.kind as Notification["kind"], title: String(r.title), body: String(r.body), link: String(r.link), read: !!n(r.read), createdAt: n(r.created_at),
    }));
  }
  markRead(userId: string, ids: string[] | "all") {
    if (ids === "all") this.run("UPDATE notifications SET read = 1 WHERE user_id = ?", userId);
    else for (const i of ids) this.run("UPDATE notifications SET read = 1 WHERE user_id = ? AND id = ?", userId, i);
  }

  insertActivity(a: Activity) {
    this.run("INSERT INTO activity (id, group_id, project_id, actor_id, verb, summary, link, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", a.id, a.groupId, a.projectId, a.actorId, a.verb, a.summary, a.link, a.createdAt);
  }
  activity(groupId: string, projectId: string | null, limit: number) {
    const rows = projectId
      ? this.all("SELECT * FROM activity WHERE group_id = ? AND project_id = ? ORDER BY created_at DESC LIMIT ?", groupId, projectId, limit)
      : this.all("SELECT * FROM activity WHERE group_id = ? ORDER BY created_at DESC LIMIT ?", groupId, limit);
    return rows.map((r) => ({ id: String(r.id), groupId, projectId: s(r.project_id), actorId: String(r.actor_id), verb: String(r.verb), summary: String(r.summary), link: String(r.link), createdAt: n(r.created_at) }));
  }
}
