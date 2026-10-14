import { useEffect, useState } from "react";
import type { GitBranchInfo, GitFileStatus } from "@rlm/contracts";
import { call, errText } from "../api";
import { useApp } from "../store";
import { IBug, IPlus, IPull, IPush, IRefresh, ITeam } from "../ui/icons";
import { PanelHeader } from "./Explorer";
import { useTeam } from "./team/teamStore";

const LABEL: Record<string, string> = { M: "modified", A: "added", D: "deleted", R: "renamed", "?": "untracked", U: "conflict" };

export function GitPanel() {
  const { git, treeVersion, openDiff, toast, refreshGit, set } = useApp();
  const teamTick = useTeam((s) => s.tick);
  const [files, setFiles] = useState<GitFileStatus[]>([]);
  const [branches, setBranches] = useState<GitBranchInfo[]>([]);
  const [linked, setLinked] = useState<{ connId: string; group: string; project: string } | null>(null);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [newBranch, setNewBranch] = useState<string | null>(null);

  const load = () => {
    void call<GitFileStatus[]>("git:status", {}).then(setFiles, () => setFiles([]));
    void call<GitBranchInfo[]>("git:branches", {}).then(setBranches, () => setBranches([]));
    void call<typeof linked>("team:workspaceProject", {}).then(setLinked, () => setLinked(null));
  };
  useEffect(load, [treeVersion, git.dirty, git.branch, teamTick]);

  const run = async (label: string, fn: () => Promise<unknown>, ok?: string) => {
    setBusy(label);
    try {
      await fn();
      if (ok) toast(ok);
      await refreshGit();
      load();
    } catch (e) {
      toast(errText(e), "error");
    } finally {
      setBusy(null);
    }
  };

  const show = async (f: GitFileStatus) => {
    const before = f.code.includes("?") || f.code.includes("A") ? "" : await call<string>("git:show", { path: f.path });
    const after = f.code.includes("D") ? "" : (await call<{ content: string }>("fs:readFile", { path: f.path }).catch(() => ({ content: "" }))).content;
    openDiff(f.path, before, after);
  };

  if (!git.branch)
    return (
      <div className="flex h-full flex-col">
        <PanelHeader title="Source control" />
        <div className="px-3 text-[12.5px] text-dim">This folder is not a git repository.</div>
        <button className="btn btn-sm mx-3 mt-3" onClick={() => set({ activity: "team" })}>
          <ITeam size={13} /> Share it on a team hub
        </button>
      </div>
    );

  const current = branches.find((b) => b.current);
  return (
    <div className="flex h-full flex-col" data-testid="git-panel">
      <PanelHeader title="Source control">
        <button className="iconbtn" title="Report a bug to the team" onClick={() => useTeam.getState().reportBug({})}>
          <IBug size={14} />
        </button>
        <button className="iconbtn" title="Refresh" onClick={load}>
          <IRefresh size={14} />
        </button>
      </PanelHeader>

      <div className="px-3 pb-2">
        {newBranch === null ? (
          <div className="flex items-center gap-1.5">
            <select
              className="input h-[28px] flex-1 py-0 text-[12px]"
              value={current?.name ?? git.branch}
              data-testid="git-branch"
              onChange={(e) => void run("checkout", () => call("git:checkout", { branch: e.target.value, create: false }), `Switched to ${e.target.value}`)}
            >
              {branches.map((b) => (
                <option key={b.name} value={b.name}>
                  {b.name}
                  {b.ahead ? ` ↑${b.ahead}` : ""}
                  {b.behind ? ` ↓${b.behind}` : ""}
                </option>
              ))}
              {!branches.length && <option>{git.branch}</option>}
            </select>
            <button className="iconbtn" title="New branch" onClick={() => setNewBranch("")} data-testid="git-new-branch">
              <IPlus size={14} />
            </button>
          </div>
        ) : (
          <div className="flex items-center gap-1.5">
            <input
              className="input h-[28px] flex-1 text-[12px]"
              autoFocus
              placeholder="feature/my-change"
              value={newBranch}
              data-testid="git-new-branch-name"
              onChange={(e) => setNewBranch(e.target.value.replace(/\s+/g, "-"))}
              onKeyDown={(e) => {
                if (e.key === "Escape") setNewBranch(null);
                if (e.key === "Enter" && newBranch.trim())
                  void run("checkout", () => call("git:checkout", { branch: newBranch.trim(), create: true }), `Created ${newBranch.trim()}`).then(() => setNewBranch(null));
              }}
            />
            <button className="btn btn-sm btn-ghost" onClick={() => setNewBranch(null)}>
              Cancel
            </button>
          </div>
        )}
        <div className="mt-2 grid grid-cols-3 gap-1.5">
          <button className="btn btn-sm" disabled={!!busy} onClick={() => void run("pull", () => call("team:git", { op: "pull" }), "Pulled")} data-testid="git-pull">
            <IPull size={12} /> {busy === "pull" ? "…" : `Pull${current?.behind ? ` ${current.behind}` : ""}`}
          </button>
          <button className="btn btn-sm" disabled={!!busy} onClick={() => void run("push", () => call("team:git", { op: "push" }), "Pushed")} data-testid="git-push">
            <IPush size={12} /> {busy === "push" ? "…" : `Push${current?.ahead ? ` ${current.ahead}` : ""}`}
          </button>
          <button className="btn btn-sm" disabled={!!busy} onClick={() => void run("fetch", () => call("team:git", { op: "fetch" }))}>
            {busy === "fetch" ? "…" : "Fetch"}
          </button>
        </div>
        {linked ? (
          <button
            className="mt-2 flex w-full items-center gap-1.5 text-left text-[11.5px] text-dim hover:text-ink"
            onClick={() => {
              set({ activity: "team" });
              useTeam.getState().go({ kind: "project", ...linked });
            }}
          >
            <ITeam size={12} /> Team hub: {linked.group}/{linked.project}
          </button>
        ) : (
          <button className="mt-2 flex w-full items-center gap-1.5 text-left text-[11.5px] text-faint hover:text-ink" onClick={() => set({ activity: "team" })}>
            <ITeam size={12} /> Not on a team hub yet
          </button>
        )}
      </div>

      <div className="px-3 pb-3">
        <textarea className="input h-[64px] resize-none" placeholder={`Commit message (on ${git.branch})`} value={msg} onChange={(e) => setMsg(e.target.value)} data-testid="git-message" />
        <button
          className="btn btn-primary btn-sm mt-2 w-full"
          disabled={!msg.trim() || !files.length || !!busy}
          data-testid="git-commit"
          onClick={() => void run("commit", () => call("git:commit", { message: msg, paths: [] }), "Committed").then(() => setMsg(""))}
        >
          Commit all ({files.length})
        </button>
      </div>
      <div className="smallcaps px-3 pb-1">Changes</div>
      <div className="min-h-0 flex-1 overflow-auto pb-2">
        {files.map((f) => {
          const c = f.code.trim()[0] ?? "?";
          return (
            <button key={f.path} className="flex w-full items-center gap-2 px-3 py-[4px] text-left text-[12.5px] hover:bg-surface-2" onClick={() => void show(f)} title={LABEL[c] ?? f.code}>
              <span className="truncate">{f.path.split("/").pop()}</span>
              <span className="truncate text-[11px] text-faint">{f.path.split("/").slice(0, -1).join("/")}</span>
              <span className="mono ml-auto shrink-0 text-[11px]" style={{ color: c === "D" ? "var(--err)" : c === "?" || c === "A" ? "var(--ok)" : "var(--warn)" }}>
                {c === "?" ? "U" : c}
              </span>
            </button>
          );
        })}
        {!files.length && <div className="px-3 text-[12px] text-faint">No changes.</div>}
      </div>
    </div>
  );
}
