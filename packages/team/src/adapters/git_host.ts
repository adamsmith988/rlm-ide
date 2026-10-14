import { spawn } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import type { BlameLine, CommitInfo, GitHost } from "../ports";

export interface GitResult {
  code: number;
  stdout: string;
  stderr: string;
}

export function runGit(args: string[], opts: { cwd?: string; env?: NodeJS.ProcessEnv; input?: string; maxBytes?: number } = {}): Promise<GitResult> {
  return new Promise((resolve, reject) => {
    const p = spawn("git", args, { cwd: opts.cwd, env: { ...process.env, GIT_TERMINAL_PROMPT: "0", LC_ALL: "C", ...opts.env }, windowsHide: true });
    const out: Buffer[] = [];
    const err: Buffer[] = [];
    let size = 0;
    const max = opts.maxBytes ?? 64 * 1024 * 1024;
    p.stdout.on("data", (d: Buffer) => {
      size += d.length;
      if (size <= max) out.push(d);
    });
    p.stderr.on("data", (d: Buffer) => err.push(d));
    p.on("error", reject);
    p.on("close", (code) => resolve({ code: code ?? 1, stdout: Buffer.concat(out).toString("utf8"), stderr: Buffer.concat(err).toString("utf8") }));
    if (opts.input !== undefined) p.stdin.end(opts.input);
    else p.stdin.end();
  });
}

/** Rejects pushes to a protected branch. The transports set RLM_PROTECTED_BRANCH for the pusher. */
const PRE_RECEIVE = `#!/bin/sh
# Installed by RLM Team Hub.
[ -z "$RLM_PROTECTED_BRANCH" ] && exit 0
while read old new ref; do
  if [ "$ref" = "refs/heads/$RLM_PROTECTED_BRANCH" ]; then
    echo "RLM Hub: $RLM_PROTECTED_BRANCH is protected. Push a branch and open a pull request." >&2
    exit 1
  fi
done
exit 0
`;

const SEP = "\x1f";
const REC = "\x1e";

export class CliGitHost implements GitHost {
  constructor(private root: string) {
    mkdirSync(root, { recursive: true });
  }

  path(groupSlug: string, projectSlug: string): string {
    return path.join(this.root, groupSlug, `${projectSlug}.git`);
  }

  async create(groupSlug: string, projectSlug: string, defaultBranch: string): Promise<string> {
    const repo = this.path(groupSlug, projectSlug);
    if (!existsSync(repo)) {
      mkdirSync(path.dirname(repo), { recursive: true });
      const r = await runGit(["init", "--bare", "-q", `--initial-branch=${defaultBranch}`, repo]);
      if (r.code !== 0) throw new Error(r.stderr.trim() || "git init failed");
    }
    await runGit(["config", "http.receivepack", "true"], { cwd: repo });
    await runGit(["config", "receive.advertisePushOptions", "true"], { cwd: repo });
    const hook = path.join(repo, "hooks", "pre-receive");
    mkdirSync(path.dirname(hook), { recursive: true });
    writeFileSync(hook, PRE_RECEIVE);
    try {
      chmodSync(hook, 0o755);
    } catch {
      /* Windows: Git for Windows runs hooks through its own sh */
    }
    return repo;
  }

  async refs(repo: string): Promise<Map<string, string>> {
    const r = await runGit(["for-each-ref", "--format=%(refname) %(objectname)"], { cwd: repo });
    const m = new Map<string, string>();
    for (const line of r.stdout.split("\n")) {
      const [ref, sha] = line.split(" ");
      if (ref && sha) m.set(ref, sha);
    }
    return m;
  }

  async branches(repo: string) {
    const r = await runGit(["for-each-ref", "--sort=-committerdate", `--format=%(refname:short)${SEP}%(objectname)${SEP}%(committerdate:unix)${SEP}%(subject)`, "refs/heads"], { cwd: repo });
    return r.stdout
      .split("\n")
      .filter(Boolean)
      .map((l) => {
        const [name, sha, time, subject] = l.split(SEP);
        return { name: name!, sha: sha!, time: Number(time) * 1000, subject: subject ?? "" };
      });
  }

  async log(repo: string, range: string, limit: number): Promise<CommitInfo[]> {
    const r = await runGit(["log", `--format=%H${SEP}%P${SEP}%an${SEP}%ae${SEP}%at${SEP}%s${REC}`, "-n", String(limit), ...range.split(/\s+/).filter(Boolean), "--"], { cwd: repo });
    if (r.code !== 0) return [];
    return r.stdout
      .split(REC)
      .map((s) => s.trim())
      .filter(Boolean)
      .map((s) => {
        const [sha, parents, author, email, time, subject] = s.split(SEP);
        return { sha: sha!, parents: parents ? parents.split(" ") : [], author: author!, email: email!, time: Number(time) * 1000, subject: subject ?? "" };
      });
  }

  async commit(repo: string, sha: string) {
    const [c] = await this.log(repo, `${sha} -1`, 1);
    if (!c) return null;
    const f = await runGit(["diff-tree", "--no-commit-id", "--name-only", "-r", "--root", c.sha], { cwd: repo });
    return { ...c, files: f.stdout.split("\n").filter(Boolean) };
  }

  async resolve(repo: string, ref: string): Promise<string | null> {
    if (!ref || ref.startsWith("-")) return null;
    const r = await runGit(["rev-parse", "--verify", "--quiet", `${ref}^{commit}`], { cwd: repo });
    return r.code === 0 ? r.stdout.trim() : null;
  }

  async diff(repo: string, base: string, head: string, maxBytes: number) {
    const [d, n] = await Promise.all([
      runGit(["diff", "--no-color", "--no-ext-diff", "-M", base, head], { cwd: repo, maxBytes: maxBytes + 1 }),
      runGit(["diff", "--numstat", "-M", base, head], { cwd: repo }),
    ]);
    const files = n.stdout
      .split("\n")
      .filter(Boolean)
      .map((l) => {
        const [a, r, ...p] = l.split("\t");
        return { path: p.join("\t"), added: a === "-" ? 0 : Number(a), removed: r === "-" ? 0 : Number(r) };
      });
    const truncated = Buffer.byteLength(d.stdout) > maxBytes;
    return { diff: truncated ? d.stdout.slice(0, maxBytes) : d.stdout, truncated, files };
  }

  async blame(repo: string, ref: string, file: string, line: number): Promise<BlameLine | null> {
    const r = await runGit(["blame", "--porcelain", "-L", `${line},${line}`, ref, "--", file], { cwd: repo });
    if (r.code !== 0) return null;
    const lines = r.stdout.split("\n");
    const sha = lines[0]?.split(" ")[0];
    if (!sha || /^0+$/.test(sha)) return null;
    const get = (k: string) => lines.find((l) => l.startsWith(`${k} `))?.slice(k.length + 1) ?? "";
    return { commit: sha, author: get("author"), email: get("author-mail").replace(/^<|>$/g, ""), time: Number(get("author-time")) * 1000, summary: get("summary") };
  }

  async show(repo: string, ref: string, file: string): Promise<string | null> {
    if (file.startsWith("-") || file.includes("..")) return null;
    const r = await runGit(["show", `${ref}:${file.replace(/^\/+/, "")}`], { cwd: repo, maxBytes: 8 * 1024 * 1024 });
    return r.code === 0 ? r.stdout : null;
  }

  async tree(repo: string, ref: string, dir: string) {
    const clean = dir.replace(/^\/+|\/+$/g, "");
    if (clean.includes("..")) return [];
    const r = await runGit(["ls-tree", "-l", clean ? `${ref}:${clean}` : `${ref}^{tree}`], { cwd: repo });
    if (r.code !== 0) return [];
    return r.stdout
      .split("\n")
      .filter(Boolean)
      .map((l) => {
        const [meta, name] = l.split("\t");
        const [, type, , size] = meta!.split(/\s+/);
        return { name: name!, kind: (type === "tree" ? "dir" : "file") as "file" | "dir", size: size === "-" ? 0 : Number(size) };
      })
      .sort((a, b) => (a.kind === b.kind ? a.name.localeCompare(b.name) : a.kind === "dir" ? -1 : 1));
  }

  async mergeBase(repo: string, a: string, b: string): Promise<string | null> {
    const r = await runGit(["merge-base", a, b], { cwd: repo });
    return r.code === 0 ? r.stdout.trim() : null;
  }

  async isAncestor(repo: string, ancestor: string, of: string): Promise<boolean> {
    return (await runGit(["merge-base", "--is-ancestor", ancestor, of], { cwd: repo })).code === 0;
  }

  async merge(repo: string, base: string, head: string, message: string, author: { name: string; email: string }, updateRef?: string) {
    const mt = await runGit(["merge-tree", "--write-tree", "--name-only", "--no-messages", base, head], { cwd: repo });
    const lines = mt.stdout.split("\n").filter(Boolean);
    if (mt.code === 1) return { conflicts: lines.slice(1) };
    if (mt.code !== 0 || !lines[0]) throw new Error(mt.stderr.trim() || "git merge-tree failed (git 2.38 or newer is required)");
    if (!updateRef) return { sha: "" };
    const env = { GIT_AUTHOR_NAME: author.name, GIT_AUTHOR_EMAIL: author.email, GIT_COMMITTER_NAME: author.name, GIT_COMMITTER_EMAIL: author.email };
    const ct = await runGit(["commit-tree", lines[0], "-p", base, "-p", head, "-F", "-"], { cwd: repo, env, input: message });
    if (ct.code !== 0) throw new Error(ct.stderr.trim() || "git commit-tree failed");
    const sha = ct.stdout.trim();
    const ur = await runGit(["update-ref", "-m", `merge ${head}`, updateRef, sha, base], { cwd: repo });
    if (ur.code !== 0) throw new Error(`the target branch moved during the merge; try again (${ur.stderr.trim()})`);
    return { sha };
  }
}
