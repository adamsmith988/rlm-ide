import type {
  Activity,
  Comment,
  Group,
  Invite,
  Issue,
  Membership,
  Notification,
  Project,
  PullRequest,
  PushRecord,
  Role,
  SshKey,
  User,
} from "./domain/types";

/** Persistence for the hub. One implementation: SQLite. */
export interface TeamStore {
  meta(key: string): string | null;
  setMeta(key: string, value: string): void;

  insertUser(u: User, tokenHash: string): void;
  updateUser(u: User): void;
  userById(id: string): User | null;
  userByTokenHash(hash: string): User | null;
  userByEmail(email: string): User | null;
  setUserToken(userId: string, tokenHash: string): void;
  listUsers(ids: string[]): User[];

  insertKey(k: SshKey): void;
  keysOf(userId: string): SshKey[];
  keyByFingerprint(fp: string): SshKey | null;
  deleteKey(id: string, userId: string): void;

  insertGroup(g: Group): void;
  groupBySlug(slug: string): Group | null;
  groupById(id: string): Group | null;
  groupsOf(userId: string): Array<Group & { role: Role }>;

  upsertMember(m: Membership): void;
  removeMember(groupId: string, userId: string): void;
  member(groupId: string, userId: string): Membership | null;
  members(groupId: string): Membership[];

  insertInvite(i: Invite): void;
  invite(code: string): Invite | null;
  useInvite(code: string): void;
  invitesOf(groupId: string): Invite[];
  deleteInvite(code: string): void;

  insertProject(p: Project): void;
  updateProject(p: Project): void;
  project(groupId: string, slug: string): Project | null;
  projectById(id: string): Project | null;
  projectsOf(groupId: string): Project[];

  nextNumber(projectId: string): number;
  insertIssue(i: Issue): void;
  updateIssue(i: Issue): void;
  issue(projectId: string, number: number): Issue | null;
  issues(projectId: string, f: { kind?: string; state?: string; assigneeId?: string }): Issue[];

  insertPull(p: PullRequest): void;
  updatePull(p: PullRequest): void;
  pull(projectId: string, number: number): PullRequest | null;
  pulls(projectId: string, state?: string): PullRequest[];
  pullByMergeCommit(projectId: string, sha: string): PullRequest | null;

  insertComment(c: Comment): void;
  comments(kind: "issue" | "pull", targetId: string): Comment[];

  insertPush(p: PushRecord): void;
  pushes(projectId: string, limit: number): PushRecord[];
  pushContaining(projectId: string, sha: string): PushRecord | null;

  insertNotification(n: Notification): void;
  notifications(userId: string, unreadOnly: boolean, limit: number): Notification[];
  markRead(userId: string, ids: string[] | "all"): void;

  insertActivity(a: Activity): void;
  activity(groupId: string, projectId: string | null, limit: number): Activity[];
}

export interface CommitInfo {
  sha: string;
  parents: string[];
  author: string;
  email: string;
  time: number;
  subject: string;
}

export interface BlameLine {
  commit: string;
  author: string;
  email: string;
  time: number;
  summary: string;
}

/** Bare repositories on the hub's disk. */
export interface GitHost {
  create(groupSlug: string, projectSlug: string, defaultBranch: string): Promise<string>;
  path(groupSlug: string, projectSlug: string): string;
  refs(repo: string): Promise<Map<string, string>>;
  branches(repo: string): Promise<Array<{ name: string; sha: string; time: number; subject: string }>>;
  log(repo: string, range: string, limit: number): Promise<CommitInfo[]>;
  commit(repo: string, sha: string): Promise<(CommitInfo & { files: string[] }) | null>;
  resolve(repo: string, ref: string): Promise<string | null>;
  diff(repo: string, base: string, head: string, maxBytes: number): Promise<{ diff: string; truncated: boolean; files: Array<{ path: string; added: number; removed: number }> }>;
  blame(repo: string, ref: string, file: string, line: number): Promise<BlameLine | null>;
  show(repo: string, ref: string, file: string): Promise<string | null>;
  tree(repo: string, ref: string, dir: string): Promise<Array<{ name: string; kind: "file" | "dir"; size: number }>>;
  /** Merges head into base without a work tree. Moves `updateRef` (compare-and-swap on base) when given;
   * otherwise it only tells whether the merge is clean. */
  merge(repo: string, base: string, head: string, message: string, author: { name: string; email: string }, updateRef?: string): Promise<{ sha: string } | { conflicts: string[] }>;
  mergeBase(repo: string, a: string, b: string): Promise<string | null>;
  isAncestor(repo: string, ancestor: string, of: string): Promise<boolean>;
}
