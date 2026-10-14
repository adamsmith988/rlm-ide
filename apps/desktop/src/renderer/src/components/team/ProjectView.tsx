import { useEffect, useMemo, useState, type ReactNode } from "react";
import { call, errText } from "../../api";
import { useApp } from "../../store";
import { Markdown } from "../../ui/Markdown";
import { IAgent, IBack, ICopy, IDownload, IFile, IFolder, IPlus } from "../../ui/icons";
import { ago, atLeast, hub, useTeam, type ProjectTab } from "./teamStore";

interface Person {
  id: string;
  name: string;
  email: string;
}

interface Detail {
  name: string;
  description: string;
  defaultBranch: string;
  protectDefault: boolean;
  role: string;
  empty: boolean;
  openIssues: number;
  openBugs: number;
  openPulls: number;
  branches: Array<{ name: string; sha: string; time: number; subject: string }>;
  commits: Array<{ sha: string; author: string; email: string; time: number; subject: string }>;
  pushes: Array<{ id: string; ref: string; commits: string[]; transport: string; createdAt: number; user: Person | null }>;
}

interface IssueRow {
  number: number;
  kind: "issue" | "bug";
  title: string;
  body: string;
  state: "open" | "closed";
  labels: string[];
  severity: string | null;
  bugStatus: string | null;
  routedBy: string | null;
  culpritCommit: string | null;
  fixCommit: string | null;
  author: Person | null;
  assignee: Person | null;
  createdAt: number;
  updatedAt: number;
}

interface CommentRow {
  id: string;
  body: string;
  review: string | null;
  createdAt: number;
  author: Person | null;
}

interface PullRow {
  number: number;
  title: string;
  body: string;
  state: "open" | "merged" | "closed";
  sourceBranch: string;
  targetBranch: string;
  author: Person | null;
  mergedByUser: Person | null;
  mergeCommit: string | null;
  approvals: number;
  changesRequested: number;
  createdAt: number;
}

interface Props {
  connId: string;
  group: string;
  project: string;
  tab?: ProjectTab;
  item?: number;
}

export function ProjectView({ connId, group, project, tab: initialTab, item: initialItem }: Props) {
  const { tick, go, reportBug } = useTeam();
  const toast = useApp((s) => s.toast);
  const [d, setD] = useState<Detail | null>(null);
  const [tab, setTab] = useState<ProjectTab>(initialTab ?? "code");
  const [item, setItem] = useState<number | null>(initialItem ?? null);
  const [members, setMembers] = useState<Person[]>([]);
  const [me, setMe] = useState<Person | null>(null);
  const [linked, setLinked] = useState(false);
  const base = `/api/projects/${group}/${project}`;

  useEffect(() => {
    setTab(initialTab ?? "code");
    setItem(initialItem ?? null);
  }, [initialTab, initialItem]);
  useEffect(() => {
    void hub<Detail>(connId, "GET", base).then(setD, (e) => toast(errText(e), "error"));
  }, [connId, base, tick]);
  useEffect(() => {
    void hub<{ members: Person[] }>(connId, "GET", `/api/groups/${group}`).then((g) => setMembers(g.members), () => undefined);
    void hub<Person>(connId, "GET", "/api/me").then(setMe, () => undefined);
    void call<{ connId: string; group: string; project: string } | null>("team:workspaceProject", {}).then(
      (w) => setLinked(!!w && w.connId === connId && w.group === group && w.project === project),
      () => undefined,
    );
  }, [connId, group, project]);

  if (!d) return null;
  const ctx: Ctx = { connId, group, project, base, detail: d, members, me, linked };
  const tabs: Array<[ProjectTab, string]> = [
    ["code", "Code"],
    ["issues", `Issues ${d.openIssues || ""}`],
    ["bugs", `Bugs ${d.openBugs || ""}`],
    ["pulls", `Pull requests ${d.openPulls || ""}`],
    ["activity", "Activity"],
  ];
  return (
    <div className="mx-auto max-w-[1000px] px-10 py-8" data-testid="team-project">
      <div className="flex items-start gap-4">
        <div className="min-w-0 flex-1">
          <button className="smallcaps hover:underline" onClick={() => go({ kind: "group", connId, group })}>
            {group}
          </button>
          <h1 className="heading m-0 mt-1 text-[36px] leading-none">{d.name}</h1>
          {d.description && <div className="mt-2 text-[13px] text-dim">{d.description}</div>}
          <div className="mt-2 text-[11.5px] text-faint">
            you are {d.role} · default branch <span className="mono">{d.defaultBranch}</span>
            {d.protectDefault ? " (protected: developers merge through pull requests)" : ""}
            {linked ? " · linked to the open folder" : ""}
          </div>
        </div>
        <div className="flex gap-2">
          <CloneButton ctx={ctx} />
          <button className="btn btn-sm btn-primary" onClick={() => reportBug({ connId, group, project })} data-testid="project-report-bug">
            Report a bug
          </button>
        </div>
      </div>
      <div className="seg mt-5">
        {tabs.map(([t, l]) => (
          <button
            key={t}
            className={tab === t ? "on" : ""}
            data-testid={`project-tab-${t}`}
            onClick={() => {
              setTab(t);
              setItem(null);
            }}
          >
            {l}
          </button>
        ))}
      </div>
      <div className="mt-5">
        {tab === "code" && <CodeTab ctx={ctx} />}
        {(tab === "issues" || tab === "bugs") &&
          (item ? <IssueDetail ctx={ctx} n={item} back={() => setItem(null)} /> : <IssueList ctx={ctx} kind={tab === "bugs" ? "bug" : "issue"} open={setItem} />)}
        {tab === "pulls" && (item ? <PullDetail ctx={ctx} n={item} back={() => setItem(null)} /> : <PullList ctx={ctx} open={setItem} />)}
        {tab === "activity" && <ActivityTab ctx={ctx} />}
      </div>
    </div>
  );
}

interface Ctx {
  connId: string;
  group: string;
  project: string;
  base: string;
  detail: Detail;
  members: Person[];
  me: Person | null;
  linked: boolean;
}

function useAction() {
  const toast = useApp((s) => s.toast);
  const [busy, setBusy] = useState(false);
  const run = async <T,>(fn: () => Promise<T>, ok?: string) => {
    setBusy(true);
    try {
      const v = await fn();
      if (ok) toast(ok);
      useTeam.setState({ tick: useTeam.getState().tick + 1 });
      return v;
    } catch (e) {
      toast(errText(e), "error");
      return undefined;
    } finally {
      setBusy(false);
    }
  };
  return { busy, run };
}

function CloneButton({ ctx }: { ctx: Ctx }) {
  const { busy, run } = useAction();
  if (ctx.linked) return null;
  return (
    <button
      className="btn btn-sm"
      disabled={busy}
      data-testid="project-clone"
      onClick={() =>
        void run(async () => {
          const parent = await call<string | null>("dialog:openFolder", {});
          if (!parent) return;
          const dest = await call<string>("team:clone", { connId: ctx.connId, group: ctx.group, project: ctx.project, parent });
          await useApp.getState().openWorkspace(dest);
        }, undefined)
      }
    >
      {busy ? "Cloning…" : "Clone and open"}
    </button>
  );
}

// ---- code ---------------------------------------------------------------------------------------

function CodeTab({ ctx }: { ctx: Ctx }) {
  const d = ctx.detail;
  const [ref, setRef] = useState(d.branches.find((b) => b.name === d.defaultBranch)?.name ?? d.branches[0]?.name ?? "");
  const [dir, setDir] = useState("");
  const [entries, setEntries] = useState<Array<{ name: string; kind: "file" | "dir"; size: number }>>([]);
  const [file, setFile] = useState<{ path: string; text: string } | null>(null);
  const [commit, setCommit] = useState<string | null>(null);
  const [commits, setCommits] = useState(d.commits);
  const tick = useTeam((s) => s.tick);

  useEffect(() => {
    if (!ref) return;
    void hub<typeof entries>(ctx.connId, "GET", `${ctx.base}/tree?ref=${encodeURIComponent(ref)}&path=${encodeURIComponent(dir)}`).then(setEntries, () => setEntries([]));
  }, [ref, dir, tick]);
  useEffect(() => setCommits(d.commits), [d]);

  if (d.empty)
    return (
      <div className="card p-5 text-[13px]">
        <div className="heading text-[20px]">This project is empty</div>
        <div className="mt-1 text-dim">Open a folder in the IDE and use “Share open folder” on the group page, or push to it:</div>
        <pre className="mono selectable mt-3 rounded-[10px] bg-surface-2 p-3 text-[12px]">git push -u &lt;hub remote&gt; HEAD:{d.defaultBranch}</pre>
      </div>
    );
  if (commit) return <CommitView ctx={ctx} sha={commit} back={() => setCommit(null)} />;

  const crumbs = dir ? dir.split("/") : [];
  return (
    <div className="grid grid-cols-[1fr_320px] gap-5">
      <div className="min-w-0">
        <div className="mb-2 flex items-center gap-2 text-[12.5px]">
          <select className="input shrink-0" style={{ width: 200 }} value={ref} onChange={(e) => (setRef(e.target.value), setDir(""), setFile(null))} data-testid="code-branch">
            {d.branches.map((b) => (
              <option key={b.name} value={b.name}>
                {b.name}
              </option>
            ))}
          </select>
          <button className="hover:underline" onClick={() => (setDir(""), setFile(null))}>
            {ctx.project}
          </button>
          {crumbs.map((c, i) => (
            <span key={i}>
              /{" "}
              <button className="hover:underline" onClick={() => (setDir(crumbs.slice(0, i + 1).join("/")), setFile(null))}>
                {c}
              </button>
            </span>
          ))}
          {file && <span>/ {file.path.split("/").pop()}</span>}
        </div>
        {file ? (
          <div className="team-md">
            <Markdown text={`\`\`\`${file.path.split(".").pop() ?? ""}\n${file.text.replace(/```/g, "ˋˋˋ")}\n\`\`\``} previews={false} />
          </div>
        ) : (
          <div className="card overflow-hidden" data-testid="code-tree">
            {entries.map((e) => (
              <button
                key={e.name}
                className="flex w-full items-center gap-2 border-b border-line px-3 py-[7px] text-left text-[12.5px] last:border-b-0 hover:bg-surface-2"
                onClick={() => {
                  const p = dir ? `${dir}/${e.name}` : e.name;
                  if (e.kind === "dir") setDir(p);
                  else void hub<{ text: string }>(ctx.connId, "GET", `${ctx.base}/file?ref=${encodeURIComponent(ref)}&path=${encodeURIComponent(p)}`).then((r) => setFile({ path: p, text: r.text }));
                }}
              >
                {e.kind === "dir" ? <IFolder size={14} /> : <IFile size={14} />}
                <span>{e.name}</span>
                {e.kind === "file" && <span className="ml-auto text-[11px] text-faint">{e.size < 1024 ? `${e.size} B` : `${(e.size / 1024).toFixed(1)} KB`}</span>}
              </button>
            ))}
            {!entries.length && <div className="px-3 py-3 text-[12px] text-faint">Empty folder.</div>}
          </div>
        )}
      </div>
      <div className="min-w-0">
        <div className="smallcaps mb-2">Commits on {d.defaultBranch}</div>
        <div className="flex flex-col gap-1">
          {commits.slice(0, 20).map((c) => (
            <button key={c.sha} className="rounded-[10px] px-2.5 py-2 text-left hover:bg-surface-2" onClick={() => setCommit(c.sha)}>
              <div className="truncate text-[12.5px] font-medium">{c.subject}</div>
              <div className="text-[11px] text-faint">
                <span className="mono">{c.sha.slice(0, 8)}</span> · {c.author} · {ago(c.time)}
              </div>
            </button>
          ))}
        </div>
        {d.pushes.length > 0 && (
          <>
            <div className="smallcaps mb-2 mt-5">Recent pushes</div>
            {d.pushes.slice(0, 8).map((p) => (
              <div key={p.id} className="px-2.5 py-1 text-[11.5px] text-dim">
                <b className="text-ink">{p.user?.name ?? "someone"}</b> pushed {p.commits.length} to <span className="mono">{p.ref.replace("refs/heads/", "")}</span> over {p.transport} · {ago(p.createdAt)}
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}

function CommitView({ ctx, sha, back }: { ctx: Ctx; sha: string; back: () => void }) {
  const [c, setC] = useState<{ sha: string; subject: string; author: string; email: string; time: number; diff: string; truncated: boolean; files: Array<{ path: string; added: number; removed: number }> } | null>(null);
  useEffect(() => {
    void hub<typeof c>(ctx.connId, "GET", `${ctx.base}/commits/${sha}`).then(setC, () => undefined);
  }, [sha]);
  if (!c) return null;
  return (
    <div>
      <BackLink onClick={back} label="Code" />
      <h2 className="heading m-0 text-[24px]">{c.subject}</h2>
      <div className="mt-1 text-[12px] text-dim">
        <span className="mono">{c.sha.slice(0, 12)}</span> · {c.author} &lt;{c.email}&gt; · {new Date(c.time).toLocaleString()}
      </div>
      <DiffView diff={c.diff} truncated={c.truncated} />
    </div>
  );
}

function BackLink({ onClick, label }: { onClick: () => void; label: string }) {
  return (
    <button className="mb-3 flex items-center gap-1 text-[12px] text-dim hover:text-ink" onClick={onClick}>
      <IBack size={13} /> {label}
    </button>
  );
}

/** Unified diff, split per file, with added and removed lines marked. */
export function DiffView({ diff, truncated }: { diff: string; truncated?: boolean }) {
  const files = useMemo(() => {
    const out: Array<{ name: string; lines: string[] }> = [];
    for (const line of diff.split("\n")) {
      if (line.startsWith("diff --git ")) out.push({ name: line.replace(/^diff --git a\/(.+?) b\/.*$/, "$1"), lines: [] });
      else if (out.length && !/^(index |--- |\+\+\+ |new file mode|deleted file mode|similarity index|rename (from|to) )/.test(line)) out[out.length - 1]!.lines.push(line);
    }
    return out;
  }, [diff]);
  if (!diff.trim()) return <div className="mt-4 text-[12.5px] text-dim">No changes.</div>;
  return (
    <div className="mt-4 flex flex-col gap-3" data-testid="diff-view">
      {files.map((f) => (
        <div key={f.name} className="card overflow-hidden">
          <div className="mono border-b border-line bg-surface-2 px-3 py-1.5 text-[12px] font-semibold">{f.name}</div>
          <pre className="mono m-0 overflow-x-auto text-[11.5px] leading-[1.55]">
            {f.lines.map((l, i) => (
              <div
                key={i}
                className="px-3"
                style={{
                  background: l.startsWith("+") ? "color-mix(in srgb, var(--ok) 14%, transparent)" : l.startsWith("-") ? "color-mix(in srgb, var(--err) 14%, transparent)" : undefined,
                  color: l.startsWith("@@") ? "var(--text-faint)" : undefined,
                }}
              >
                {l || " "}
              </div>
            ))}
          </pre>
        </div>
      ))}
      {truncated && <div className="text-[12px] text-faint">The diff is too large to show completely.</div>}
    </div>
  );
}

// ---- issues and bugs ----------------------------------------------------------------------------

const SEVERITY_COLOR: Record<string, string> = { critical: "var(--err)", high: "var(--err)", medium: "var(--warn)", low: "var(--text-faint)" };
const BUG_STATUSES = ["new", "acknowledged", "in_progress", "fixed", "wont_fix"] as const;

function IssueList({ ctx, kind, open }: { ctx: Ctx; kind: "issue" | "bug"; open: (n: number) => void }) {
  const tick = useTeam((s) => s.tick);
  const reportBug = useTeam((s) => s.reportBug);
  const [state, setState] = useState<"open" | "closed">("open");
  const [mine, setMine] = useState(false);
  const [rows, setRows] = useState<IssueRow[]>([]);
  const [creating, setCreating] = useState(false);
  useEffect(() => {
    void hub<IssueRow[]>(ctx.connId, "GET", `${ctx.base}/issues?kind=${kind}&state=${state}${mine ? "&assignee=me" : ""}`).then(setRows, () => setRows([]));
  }, [kind, state, mine, tick]);
  return (
    <div>
      <div className="mb-3 flex items-center gap-2">
        <div className="seg">
          {(["open", "closed"] as const).map((s) => (
            <button key={s} className={state === s ? "on" : ""} onClick={() => setState(s)}>
              {s === "open" ? "Open" : "Closed"}
            </button>
          ))}
        </div>
        <label className="ml-2 flex items-center gap-1.5 text-[12px]">
          <input type="checkbox" checked={mine} onChange={(e) => setMine(e.target.checked)} /> Assigned to me
        </label>
        <div className="flex-1" />
        {kind === "issue" ? (
          <button className="btn btn-sm btn-primary" onClick={() => setCreating(true)} data-testid="issue-new">
            <IPlus size={13} /> New issue
          </button>
        ) : (
          <button className="btn btn-sm btn-primary" onClick={() => reportBug({ connId: ctx.connId, group: ctx.group, project: ctx.project })}>
            <IPlus size={13} /> Report a bug
          </button>
        )}
      </div>
      {creating && <NewIssue ctx={ctx} done={(n) => (setCreating(false), n && open(n))} />}
      <div className="flex flex-col gap-2" data-testid={`${kind}-list`}>
        {rows.map((r) => (
          <button key={r.number} className="card flex items-start gap-3 px-4 py-3 text-left" onClick={() => open(r.number)}>
            <span className="mono pt-[1px] text-[12px] text-faint">#{r.number}</span>
            <div className="min-w-0 flex-1">
              <div className="text-[13.5px] font-semibold">{r.title}</div>
              <div className="mt-0.5 text-[11.5px] text-faint">
                {r.author?.name} · {ago(r.createdAt)}
                {r.assignee ? ` · assigned to ${r.assignee.name}` : " · unassigned"}
                {r.routedBy ? ` (${r.routedBy})` : ""}
              </div>
            </div>
            {r.kind === "bug" && (
              <div className="flex shrink-0 flex-col items-end gap-1">
                <span className="smallcaps" style={{ color: SEVERITY_COLOR[r.severity ?? "medium"] }}>
                  {r.severity}
                </span>
                <span className="text-[11px] text-dim">{r.bugStatus?.replace("_", " ")}</span>
              </div>
            )}
          </button>
        ))}
        {!rows.length && <div className="py-8 text-center text-[12.5px] text-dim">{kind === "bug" ? "No bugs here." : "No issues here."}</div>}
      </div>
    </div>
  );
}

function NewIssue({ ctx, done }: { ctx: Ctx; done: (n?: number) => void }) {
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [assignee, setAssignee] = useState("");
  const { busy, run } = useAction();
  return (
    <div className="card mb-3 p-4">
      <input className="input" autoFocus placeholder="Title" value={title} onChange={(e) => setTitle(e.target.value)} data-testid="issue-title" />
      <textarea className="input mt-2 h-[110px] resize-y" placeholder="Describe it (markdown)" value={body} onChange={(e) => setBody(e.target.value)} />
      <div className="mt-2 flex items-center gap-2">
        <select className="input shrink-0" style={{ width: 220 }} value={assignee} onChange={(e) => setAssignee(e.target.value)}>
          <option value="">Unassigned</option>
          {ctx.members.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name}
            </option>
          ))}
        </select>
        <div className="flex-1" />
        <button className="btn btn-sm btn-ghost" onClick={() => done()}>
          Cancel
        </button>
        <button
          className="btn btn-sm btn-primary"
          disabled={busy || !title.trim()}
          data-testid="issue-submit"
          onClick={() => void run(() => hub<{ number: number }>(ctx.connId, "POST", `${ctx.base}/issues`, { title, body, assigneeId: assignee || undefined }), "Issue opened").then((r) => r && done(r.number))}
        >
          Open issue
        </button>
      </div>
    </div>
  );
}

/** "## Code at `src/app.ts:42`" in a bug report. */
function reportLocation(body: string): { file: string; line: number } | null {
  const m = /## Code at `([^`]+?):(\d+)`/.exec(body);
  return m ? { file: m[1]!, line: Number(m[2]) } : null;
}

function IssueDetail({ ctx, n, back }: { ctx: Ctx; n: number; back: () => void }) {
  const tick = useTeam((s) => s.tick);
  const app = useApp();
  const [i, setI] = useState<(IssueRow & { comments: CommentRow[] }) | null>(null);
  const [comment, setComment] = useState("");
  const [status, setStatus] = useState<string>("");
  const [fixCommit, setFixCommit] = useState("");
  const { busy, run } = useAction();
  useEffect(() => {
    void hub<IssueRow & { comments: CommentRow[] }>(ctx.connId, "GET", `${ctx.base}/issues/${n}`).then((x) => {
      setI(x);
      setStatus(x.bugStatus ?? "");
    });
  }, [n, tick]);
  if (!i) return null;
  const bug = i.kind === "bug";
  const loc = bug ? reportLocation(i.body) : null;
  const mineToFix = bug && i.assignee?.id === ctx.me?.id;

  const saveMd = () => {
    const url = URL.createObjectURL(new Blob([i.body], { type: "text/markdown" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `${ctx.project}-bug-${n}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };
  const askAgent = async () => {
    if (!ctx.linked) app.toast("Open this project's folder first (Clone and open) so the agent can edit it.", "error");
    else {
      app.set({ agent: true, activity: "explorer" });
      await app.newChat();
      await app.sendMessage(`A teammate reported this bug in ${ctx.group}/${ctx.project} (#${n}) and it was routed to me. Find the root cause, fix it, and add a test that fails without the fix.\n\n${i.body}`, loc ? [loc.file] : []);
    }
  };

  return (
    <div>
      <BackLink onClick={back} label={bug ? "Bugs" : "Issues"} />
      <div className="flex items-start gap-3">
        <h2 className="heading m-0 flex-1 text-[26px]">
          {i.title} <span className="text-faint">#{n}</span>
        </h2>
        <span className="smallcaps mt-2">{i.state}</span>
      </div>
      <div className="mt-1 text-[12px] text-dim">
        {bug ? "Reported" : "Opened"} by {i.author?.name} · {ago(i.createdAt)}
        {i.assignee ? ` · assigned to ${i.assignee.name}` : " · unassigned"}
        {i.routedBy ? ` because they ${i.routedBy}` : ""}
      </div>

      {bug && (
        <div className="card mt-4 flex flex-wrap items-center gap-2 p-3" data-testid="bug-actions">
          {loc && ctx.linked && (
            <button className="btn btn-sm" onClick={() => (app.set({ activity: "explorer" }), void app.openFile(loc.file, { line: loc.line }))}>
              Open {loc.file}:{loc.line}
            </button>
          )}
          <button className="btn btn-sm" onClick={() => void askAgent()} data-testid="bug-ask-agent">
            <IAgent size={13} /> Ask the agent to fix
          </button>
          <button className="btn btn-sm" onClick={saveMd}>
            <IDownload size={13} /> Save .md
          </button>
          <button className="btn btn-sm" onClick={() => (navigator.clipboard.writeText(i.body), app.toast("Report copied"))}>
            <ICopy size={13} /> Copy
          </button>
          <div className="flex-1" />
          <span className="smallcaps" style={{ color: SEVERITY_COLOR[i.severity ?? "medium"] }}>
            {i.severity}
          </span>
        </div>
      )}

      <div className="card team-md selectable mt-4 p-5" data-testid="issue-body">
        {i.body.trim() ? <Markdown text={i.body} previews={false} /> : <span className="text-dim">No description.</span>}
      </div>

      <div className="mt-5 flex flex-col gap-2">
        {i.comments.map((c) => (
          <div key={c.id} className="card px-4 py-3">
            <div className="mb-1 text-[11.5px] text-faint">
              <b className="text-ink">{c.author?.name}</b> · {ago(c.createdAt)}
            </div>
            <div className="team-md selectable">
              <Markdown text={c.body} previews={false} />
            </div>
          </div>
        ))}
      </div>

      {bug && (
        <div className="card mt-4 p-4" data-testid="bug-feedback">
          <div className="smallcaps mb-2">{mineToFix ? "Your fix and feedback" : "Status and feedback"}</div>
          <div className="seg">
            {BUG_STATUSES.map((s) => (
              <button key={s} className={status === s ? "on" : ""} onClick={() => setStatus(s)} data-testid={`bug-status-${s}`}>
                {s.replace("_", " ")}
              </button>
            ))}
          </div>
          {status === "fixed" && <input className="input mono mt-2" placeholder="Fix commit or pull request (optional)" value={fixCommit} onChange={(e) => setFixCommit(e.target.value)} />}
          <textarea className="input mt-2 h-[80px] resize-y" placeholder="Feedback for the reporter: what caused it, how it was fixed, or what you need" value={comment} onChange={(e) => setComment(e.target.value)} data-testid="bug-comment" />
          <div className="mt-2 flex justify-end">
            <button
              className="btn btn-sm btn-primary"
              disabled={busy || (status === i.bugStatus && !comment.trim())}
              data-testid="bug-send-status"
              onClick={() =>
                void run(async () => {
                  if (status !== i.bugStatus || fixCommit) await hub(ctx.connId, "POST", `${ctx.base}/issues/${n}/status`, { status, fixCommit: fixCommit || undefined, comment: comment || undefined });
                  else await hub(ctx.connId, "POST", `${ctx.base}/issues/${n}/comments`, { body: comment });
                  setComment("");
                  setFixCommit("");
                }, "Sent to the reporter")
              }
            >
              Send
            </button>
          </div>
        </div>
      )}

      {!bug && (
        <div className="card mt-4 p-4">
          <textarea className="input h-[80px] resize-y" placeholder="Comment (markdown)" value={comment} onChange={(e) => setComment(e.target.value)} />
          <div className="mt-2 flex items-center gap-2">
            <select
              className="input shrink-0" style={{ width: 220 }}
              value={i.assignee?.id ?? ""}
              onChange={(e) => void run(() => hub(ctx.connId, "PATCH", `${ctx.base}/issues/${n}`, { assigneeId: e.target.value || null }))}
            >
              <option value="">Unassigned</option>
              {ctx.members.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
            <div className="flex-1" />
            <button className="btn btn-sm" disabled={busy} onClick={() => void run(() => hub(ctx.connId, "PATCH", `${ctx.base}/issues/${n}`, { state: i.state === "open" ? "closed" : "open" }))}>
              {i.state === "open" ? "Close issue" : "Reopen"}
            </button>
            <button
              className="btn btn-sm btn-primary"
              disabled={busy || !comment.trim()}
              onClick={() => void run(() => hub(ctx.connId, "POST", `${ctx.base}/issues/${n}/comments`, { body: comment })).then(() => setComment(""))}
            >
              Comment
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ---- pull requests ------------------------------------------------------------------------------

function PullList({ ctx, open }: { ctx: Ctx; open: (n: number) => void }) {
  const tick = useTeam((s) => s.tick);
  const [state, setState] = useState<"open" | "merged" | "closed">("open");
  const [rows, setRows] = useState<PullRow[]>([]);
  const [creating, setCreating] = useState(false);
  useEffect(() => {
    void hub<PullRow[]>(ctx.connId, "GET", `${ctx.base}/pulls?state=${state}`).then(setRows, () => setRows([]));
  }, [state, tick]);
  return (
    <div>
      <div className="mb-3 flex items-center gap-2">
        <div className="seg">
          {(["open", "merged", "closed"] as const).map((s) => (
            <button key={s} className={state === s ? "on" : ""} onClick={() => setState(s)}>
              {s[0]!.toUpperCase() + s.slice(1)}
            </button>
          ))}
        </div>
        <div className="flex-1" />
        {atLeast(ctx.detail.role, "developer") && (
          <button className="btn btn-sm btn-primary" onClick={() => setCreating(true)} data-testid="pull-new">
            <IPlus size={13} /> New pull request
          </button>
        )}
      </div>
      {creating && <NewPull ctx={ctx} done={(n) => (setCreating(false), n && open(n))} />}
      <div className="flex flex-col gap-2" data-testid="pull-list">
        {rows.map((r) => (
          <button key={r.number} className="card flex items-start gap-3 px-4 py-3 text-left" onClick={() => open(r.number)}>
            <span className="mono pt-[1px] text-[12px] text-faint">!{r.number}</span>
            <div className="min-w-0 flex-1">
              <div className="text-[13.5px] font-semibold">{r.title}</div>
              <div className="mt-0.5 text-[11.5px] text-faint">
                <span className="mono">
                  {r.sourceBranch} → {r.targetBranch}
                </span>{" "}
                · {r.author?.name} · {ago(r.createdAt)}
              </div>
            </div>
            <div className="shrink-0 text-right text-[11px] text-dim">
              {r.approvals > 0 && <div>{r.approvals} approval{r.approvals === 1 ? "" : "s"}</div>}
              {r.changesRequested > 0 && <div className="text-[var(--err)]">changes requested</div>}
            </div>
          </button>
        ))}
        {!rows.length && <div className="py-8 text-center text-[12.5px] text-dim">No pull requests here.</div>}
      </div>
    </div>
  );
}

function NewPull({ ctx, done }: { ctx: Ctx; done: (n?: number) => void }) {
  const others = ctx.detail.branches.filter((b) => b.name !== ctx.detail.defaultBranch);
  const [source, setSource] = useState(others[0]?.name ?? "");
  const [target, setTarget] = useState(ctx.detail.defaultBranch);
  const [title, setTitle] = useState(others[0]?.subject ?? "");
  const [body, setBody] = useState("");
  const { busy, run } = useAction();
  if (!others.length)
    return (
      <div className="card mb-3 p-4 text-[12.5px] text-dim">
        Push a branch first: in Source control create a branch, commit, and press Push. Then open the pull request here.
        <button className="btn btn-sm btn-ghost ml-2" onClick={() => done()}>
          OK
        </button>
      </div>
    );
  return (
    <div className="card mb-3 p-4">
      <div className="flex items-center gap-2 text-[12.5px]">
        <select className="input shrink-0" style={{ width: 220 }} value={source} onChange={(e) => (setSource(e.target.value), setTitle(ctx.detail.branches.find((b) => b.name === e.target.value)?.subject ?? title))} data-testid="pull-source">
          {others.map((b) => (
            <option key={b.name}>{b.name}</option>
          ))}
        </select>
        <span>into</span>
        <select className="input shrink-0" style={{ width: 220 }} value={target} onChange={(e) => setTarget(e.target.value)}>
          {ctx.detail.branches.map((b) => (
            <option key={b.name}>{b.name}</option>
          ))}
        </select>
      </div>
      <input className="input mt-2" placeholder="Title" value={title} onChange={(e) => setTitle(e.target.value)} data-testid="pull-title" />
      <textarea className="input mt-2 h-[100px] resize-y" placeholder="What changed and why (markdown)" value={body} onChange={(e) => setBody(e.target.value)} />
      <div className="mt-2 flex justify-end gap-2">
        <button className="btn btn-sm btn-ghost" onClick={() => done()}>
          Cancel
        </button>
        <button
          className="btn btn-sm btn-primary"
          disabled={busy || !source || !title.trim()}
          data-testid="pull-submit"
          onClick={() => void run(() => hub<{ number: number }>(ctx.connId, "POST", `${ctx.base}/pulls`, { title, body, source, target }), "Pull request opened").then((r) => r && done(r.number))}
        >
          Open pull request
        </button>
      </div>
    </div>
  );
}

function PullDetail({ ctx, n, back }: { ctx: Ctx; n: number; back: () => void }) {
  const tick = useTeam((s) => s.tick);
  const [p, setP] = useState<(PullRow & { comments: CommentRow[]; commits: Detail["commits"]; diff: string; truncated: boolean; files: Array<{ path: string; added: number; removed: number }>; mergeable: boolean | null }) | null>(null);
  const [tab, setTab] = useState<"conversation" | "commits" | "files">("conversation");
  const [text, setText] = useState("");
  const { busy, run } = useAction();
  useEffect(() => {
    void hub<typeof p>(ctx.connId, "GET", `${ctx.base}/pulls/${n}`).then(setP, () => undefined);
  }, [n, tick]);
  if (!p) return null;
  const mine = p.author?.id === ctx.me?.id;
  const canMerge = p.state === "open" && (atLeast(ctx.detail.role, "maintainer") || (mine && p.approvals > 0));
  const review = (state: "approve" | "request_changes" | "comment") =>
    void run(() => hub(ctx.connId, "POST", `${ctx.base}/pulls/${n}/reviews`, { state, body: text }), state === "approve" ? "Approved" : state === "request_changes" ? "Changes requested" : "Commented").then(() => setText(""));
  return (
    <div data-testid="pull-detail">
      <BackLink onClick={back} label="Pull requests" />
      <div className="flex items-start gap-3">
        <h2 className="heading m-0 flex-1 text-[26px]">
          {p.title} <span className="text-faint">!{n}</span>
        </h2>
        <span className="smallcaps mt-2">{p.state}</span>
      </div>
      <div className="mt-1 text-[12px] text-dim">
        {p.author?.name} wants to merge <span className="mono">{p.sourceBranch}</span> into <span className="mono">{p.targetBranch}</span> · {p.files.length} files ·{" "}
        {p.files.reduce((a, f) => a + f.added, 0)}+ {p.files.reduce((a, f) => a + f.removed, 0)}−
        {p.state === "merged" && p.mergedByUser ? ` · merged by ${p.mergedByUser.name}` : ""}
      </div>
      {p.state === "open" && (
        <div className="card mt-4 flex items-center gap-3 p-3 text-[12.5px]" data-testid="pull-merge-box">
          <span>
            {p.mergeable === false ? "Conflicts with the target branch. Merge it into your branch, resolve, and push." : p.mergeable ? "No conflicts." : "Checking…"} {p.approvals} approval
            {p.approvals === 1 ? "" : "s"}
            {p.changesRequested ? ` · ${p.changesRequested} requested changes` : ""}
          </span>
          <div className="flex-1" />
          {(mine || atLeast(ctx.detail.role, "maintainer")) && (
            <button className="btn btn-sm" disabled={busy} onClick={() => void run(() => hub(ctx.connId, "POST", `${ctx.base}/pulls/${n}/close`, {}), "Closed")}>
              Close
            </button>
          )}
          <button
            className="btn btn-sm btn-primary"
            disabled={busy || !canMerge || p.mergeable === false}
            title={canMerge ? "" : "Needs a maintainer, or an approval if you are the author"}
            data-testid="pull-merge"
            onClick={() => void run(() => hub(ctx.connId, "POST", `${ctx.base}/pulls/${n}/merge`), "Merged")}
          >
            Merge
          </button>
        </div>
      )}
      <div className="seg mt-4">
        {(["conversation", "commits", "files"] as const).map((t) => (
          <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)} data-testid={`pull-tab-${t}`}>
            {t === "conversation" ? "Conversation" : t === "commits" ? `Commits (${p.commits.length})` : `Files (${p.files.length})`}
          </button>
        ))}
      </div>
      {tab === "conversation" && (
        <div className="mt-4">
          {p.body.trim() && (
            <div className="card team-md selectable p-4">
              <Markdown text={p.body} previews={false} />
            </div>
          )}
          <div className="mt-3 flex flex-col gap-2">
            {p.comments.map((c) => (
              <div key={c.id} className="card px-4 py-3">
                <div className="mb-1 text-[11.5px] text-faint">
                  <b className="text-ink">{c.author?.name}</b> {c.review === "approve" ? "approved" : c.review === "request_changes" ? "requested changes" : "commented"} · {ago(c.createdAt)}
                </div>
                {c.body && (
                  <div className="team-md selectable">
                    <Markdown text={c.body} previews={false} />
                  </div>
                )}
              </div>
            ))}
          </div>
          <div className="card mt-3 p-4">
            <textarea className="input h-[80px] resize-y" placeholder="Leave a review (markdown)" value={text} onChange={(e) => setText(e.target.value)} />
            <div className="mt-2 flex justify-end gap-2">
              <button className="btn btn-sm" disabled={busy || !text.trim()} onClick={() => review("comment")}>
                Comment
              </button>
              {!mine && atLeast(ctx.detail.role, "developer") && (
                <>
                  <button className="btn btn-sm btn-danger" disabled={busy} onClick={() => review("request_changes")}>
                    Request changes
                  </button>
                  <button className="btn btn-sm btn-primary" disabled={busy} onClick={() => review("approve")} data-testid="pull-approve">
                    Approve
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      )}
      {tab === "commits" && (
        <div className="mt-4 flex flex-col gap-1">
          {p.commits.map((c) => (
            <div key={c.sha} className="rounded-[10px] px-3 py-2 hover:bg-surface-2">
              <div className="text-[12.5px] font-medium">{c.subject}</div>
              <div className="text-[11px] text-faint">
                <span className="mono">{c.sha.slice(0, 8)}</span> · {c.author} · {ago(c.time)}
              </div>
            </div>
          ))}
        </div>
      )}
      {tab === "files" && <DiffView diff={p.diff} truncated={p.truncated} />}
    </div>
  );
}

// ---- activity -----------------------------------------------------------------------------------

function ActivityTab({ ctx }: { ctx: Ctx }) {
  const tick = useTeam((s) => s.tick);
  const [rows, setRows] = useState<Array<{ id: string; verb: string; summary: string; createdAt: number; actor: Person | null }>>([]);
  useEffect(() => {
    void hub<typeof rows>(ctx.connId, "GET", `/api/groups/${ctx.group}/activity?project=${ctx.project}`).then(setRows, () => undefined);
  }, [tick]);
  return (
    <div className="flex flex-col">
      {rows.map((a) => (
        <Row key={a.id}>
          <b>{a.actor?.name ?? "someone"}</b> {a.summary}
          <span className="ml-auto shrink-0 text-[11px] text-faint">{ago(a.createdAt)}</span>
        </Row>
      ))}
      {!rows.length && <div className="text-[12.5px] text-dim">No activity yet.</div>}
    </div>
  );
}

function Row({ children }: { children: ReactNode }) {
  return <div className="flex items-center gap-1.5 border-b border-line py-2 text-[12.5px]">{children}</div>;
}
