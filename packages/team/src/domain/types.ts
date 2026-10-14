/** Team hub domain: people, groups, projects and the work items around them. Pure data, no I/O. */

export type Role = "owner" | "maintainer" | "developer" | "reporter";

export const ROLE_RANK: Record<Role, number> = { reporter: 1, developer: 2, maintainer: 3, owner: 4 };

export interface User {
  id: string;
  name: string;
  email: string;
  createdAt: number;
}

export interface SshKey {
  id: string;
  userId: string;
  /** OpenSSH public key line ("ssh-ed25519 AAAA... comment"). */
  publicKey: string;
  fingerprint: string;
  createdAt: number;
}

export interface Group {
  id: string;
  slug: string;
  name: string;
  description: string;
  createdAt: number;
}

export interface Membership {
  groupId: string;
  userId: string;
  role: Role;
  joinedAt: number;
}

export interface Invite {
  code: string;
  groupId: string;
  role: Role;
  createdBy: string;
  expiresAt: number;
  usesLeft: number;
}

export interface Project {
  id: string;
  groupId: string;
  slug: string;
  name: string;
  description: string;
  defaultBranch: string;
  /** Only maintainers may push to the default branch; others go through pull requests. */
  protectDefault: boolean;
  createdAt: number;
}

export type IssueKind = "issue" | "bug";
export type IssueState = "open" | "closed";
export type Severity = "low" | "medium" | "high" | "critical";
export type BugStatus = "new" | "acknowledged" | "in_progress" | "fixed" | "wont_fix";

export interface Issue {
  id: string;
  projectId: string;
  number: number;
  kind: IssueKind;
  title: string;
  /** Markdown. For bugs this is the full generated report. */
  body: string;
  state: IssueState;
  labels: string[];
  authorId: string;
  assigneeId: string | null;
  severity: Severity | null;
  bugStatus: BugStatus | null;
  /** Commit the bug was traced to, and how the assignee was chosen. */
  culpritCommit: string | null;
  routedBy: string | null;
  fixCommit: string | null;
  createdAt: number;
  updatedAt: number;
}

export type PullState = "open" | "merged" | "closed";

export interface PullRequest {
  id: string;
  projectId: string;
  number: number;
  title: string;
  body: string;
  sourceBranch: string;
  targetBranch: string;
  authorId: string;
  state: PullState;
  mergedBy: string | null;
  mergeCommit: string | null;
  createdAt: number;
  updatedAt: number;
}

export type ReviewState = "approve" | "request_changes" | "comment";

export interface Comment {
  id: string;
  /** "issue" or "pull" */
  targetKind: "issue" | "pull";
  targetId: string;
  authorId: string;
  body: string;
  /** For reviews on pull requests. */
  review: ReviewState | null;
  createdAt: number;
}

export interface PushRecord {
  id: string;
  projectId: string;
  userId: string;
  ref: string;
  before: string;
  after: string;
  commits: string[];
  transport: "http" | "ssh";
  createdAt: number;
}

export type NotificationKind = "bug_assigned" | "issue_assigned" | "comment" | "review" | "pull_opened" | "pull_merged" | "push" | "mention" | "bug_status";

export interface Notification {
  id: string;
  userId: string;
  kind: NotificationKind;
  title: string;
  body: string;
  /** Where it points: "group/project#12" or "group/project!3". */
  link: string;
  read: boolean;
  createdAt: number;
}

export interface Activity {
  id: string;
  groupId: string;
  projectId: string | null;
  actorId: string;
  verb: string;
  summary: string;
  link: string;
  createdAt: number;
}

export class TeamError extends Error {
  constructor(
    message: string,
    readonly status: 400 | 401 | 403 | 404 | 409 = 400,
  ) {
    super(message);
  }
}

export function atLeast(role: Role | null | undefined, min: Role): boolean {
  return !!role && ROLE_RANK[role] >= ROLE_RANK[min];
}

const SLUG_RE = /^[a-z0-9][a-z0-9._-]{0,62}$/;

export function slugify(name: string): string {
  return (
    name
      .toLowerCase()
      .normalize("NFKD")
      .replace(/[^\w\s.-]/g, "")
      .trim()
      .replace(/[\s_]+/g, "-")
      .replace(/^[.-]+/, "")
      .slice(0, 63) || "project"
  );
}

export function assertSlug(slug: string): string {
  if (!SLUG_RE.test(slug) || slug.endsWith(".git")) throw new TeamError(`invalid name "${slug}": use lowercase letters, digits, '-', '_' or '.'`);
  return slug;
}

export function assertBranch(name: string): string {
  if (!/^[\w][\w./-]{0,200}$/.test(name) || name.includes("..") || name.endsWith(".lock") || name.endsWith("/")) throw new TeamError(`invalid branch name "${name}"`);
  return name;
}
