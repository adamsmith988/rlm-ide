import { useEffect, useState, type ReactNode } from "react";
import { call, errText } from "../../api";
import { useApp } from "../../store";
import { IClose } from "../../ui/icons";
import { hub, useTeam } from "./teamStore";

interface Target {
  connId: string;
  hubName: string;
  group: string;
  project: string;
  name: string;
}

/** Files a bug as a full markdown report; the hub routes it to whoever wrote or pushed the code. */
export function BugDialog() {
  const { bugDraft, state, go } = useTeam();
  const app = useApp();
  const [targets, setTargets] = useState<Target[]>([]);
  const [target, setTarget] = useState("");
  const [members, setMembers] = useState<Array<{ id: string; name: string }>>([]);
  const [f, setF] = useState({
    title: "",
    severity: "medium" as "low" | "medium" | "high" | "critical",
    summary: "",
    error: "",
    file: "",
    line: "",
    steps: "",
    expected: "",
    actual: "",
    logs: "",
    notes: "",
    assigneeId: "",
  });
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState<{ number: number; assignee: { name: string } | null; routedBy: string | null; t: Target } | null>(null);

  useEffect(() => {
    if (!bugDraft) return;
    setSent(null);
    const tab = app.tabs.find((t) => t.id === app.activeTab && t.kind === "file");
    const file = bugDraft.file ?? tab?.path ?? "";
    const line = bugDraft.line ?? (tab && file === tab.path ? app.cursorLine : undefined);
    const diag = bugDraft.error ?? app.diagnostics.find((d) => d.path === file && (!line || Math.abs(d.line - line) <= 1))?.message;
    setF((x) => ({ ...x, title: "", summary: "", steps: "", expected: "", actual: "", logs: "", notes: "", assigneeId: "", file, line: line ? String(line) : "", error: diag ?? "" }));
    void (async () => {
      const list: Target[] = [];
      for (const c of state?.connections.filter((x) => x.status === "online") ?? []) {
        const groups = await hub<Array<{ slug: string }>>(c.id, "GET", "/api/groups").catch(() => []);
        for (const g of groups) {
          const d = await hub<{ projects: Array<{ slug: string; name: string }> }>(c.id, "GET", `/api/groups/${g.slug}`).catch(() => ({ projects: [] }));
          for (const p of d.projects) list.push({ connId: c.id, hubName: c.hubName, group: g.slug, project: p.slug, name: p.name });
        }
      }
      setTargets(list);
      const ws = await call<{ connId: string; group: string; project: string } | null>("team:workspaceProject", {}).catch(() => null);
      const pick = bugDraft.connId ? `${bugDraft.connId}|${bugDraft.group}|${bugDraft.project}` : ws ? `${ws.connId}|${ws.group}|${ws.project}` : list[0] ? `${list[0].connId}|${list[0].group}|${list[0].project}` : "";
      setTarget(pick);
    })();
  }, [bugDraft]);

  const t = targets.find((x) => `${x.connId}|${x.group}|${x.project}` === target);
  useEffect(() => {
    if (!t) return;
    void hub<{ members: Array<{ id: string; name: string }> }>(t.connId, "GET", `/api/groups/${t.group}`).then((g) => setMembers(g.members), () => setMembers([]));
  }, [target]);

  if (!bugDraft) return null;
  const close = () => useTeam.setState({ bugDraft: null });
  const set = (k: keyof typeof f) => (e: { target: { value: string } }) => setF({ ...f, [k]: e.target.value });

  const submit = async () => {
    if (!t) return;
    setBusy(true);
    try {
      const r = await call<{ number: number; assignee: { name: string } | null; routedBy: string | null }>("team:reportBug", {
        connId: t.connId,
        group: t.group,
        project: t.project,
        title: f.title.trim(),
        severity: f.severity,
        summary: f.summary || undefined,
        error: f.error || undefined,
        file: f.file || undefined,
        line: f.line ? Number(f.line) : undefined,
        steps: f.steps.split("\n").map((s) => s.replace(/^\s*(\d+[.)]|[-*])\s*/, "").trim()).filter(Boolean),
        expected: f.expected || undefined,
        actual: f.actual || undefined,
        logs: f.logs || undefined,
        notes: f.notes || undefined,
        assigneeId: f.assigneeId || undefined,
        environment: { ide: "RLM IDE", workspace: app.workspace?.name ?? "" },
      });
      setSent({ ...r, t });
    } catch (e) {
      app.toast(errText(e), "error");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-6" onMouseDown={(e) => e.target === e.currentTarget && close()}>
      <div className="card flex max-h-full w-[760px] flex-col overflow-hidden" data-testid="bug-dialog">
        <div className="flex items-center border-b border-line px-5 py-3">
          <div className="flex-1">
            <div className="heading text-[22px]">Report a bug</div>
            <div className="text-[12px] text-dim">Sent as a complete markdown report to whoever wrote or pushed the code at that line, found with git blame and push history.</div>
          </div>
          <button className="iconbtn" onClick={close}>
            <IClose size={15} />
          </button>
        </div>
        {sent ? (
          <div className="p-6 text-[13px]" data-testid="bug-sent">
            <div className="heading text-[24px]">Bug #{sent.number} filed</div>
            <div className="mt-2 text-dim">
              {sent.assignee ? (
                <>
                  Sent to <b className="text-ink">{sent.assignee.name}</b>
                  {sent.routedBy ? `, who ${sent.routedBy}` : ""}. They were notified and will reply on the report.
                </>
              ) : (
                "Nobody could be traced from the code, so the group's maintainers were notified to triage it."
              )}
            </div>
            <div className="mt-5 flex gap-2">
              <button
                className="btn btn-primary btn-sm"
                onClick={() => {
                  close();
                  app.set({ activity: "team" });
                  go({ kind: "project", connId: sent.t.connId, group: sent.t.group, project: sent.t.project, tab: "bugs", item: sent.number });
                }}
              >
                View report
              </button>
              <button className="btn btn-sm" onClick={close}>
                Done
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="grid min-h-0 flex-1 grid-cols-2 gap-3 overflow-y-auto px-5 py-4">
              <F label="Project" wide>
                <select className="input" value={target} onChange={(e) => setTarget(e.target.value)} data-testid="bug-target">
                  {!targets.length && <option value="">Join a hub and a project first</option>}
                  {targets.map((x) => (
                    <option key={`${x.connId}|${x.group}|${x.project}`} value={`${x.connId}|${x.group}|${x.project}`}>
                      {x.hubName} · {x.group}/{x.name}
                    </option>
                  ))}
                </select>
              </F>
              <F label="Title" wide>
                <input className="input" autoFocus value={f.title} onChange={set("title")} placeholder="What is broken, in one line" data-testid="bug-title" />
              </F>
              <F label="Severity">
                <select className="input" value={f.severity} onChange={set("severity")}>
                  {["low", "medium", "high", "critical"].map((s) => (
                    <option key={s}>{s}</option>
                  ))}
                </select>
              </F>
              <F label="Send to">
                <select className="input" value={f.assigneeId} onChange={set("assigneeId")}>
                  <option value="">Automatic: the author of that code</option>
                  {members.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
                </select>
              </F>
              <F label="File">
                <input className="input mono" value={f.file} onChange={set("file")} placeholder="src/app.ts" data-testid="bug-file" />
              </F>
              <F label="Line">
                <input className="input mono" value={f.line} onChange={set("line")} placeholder="42" data-testid="bug-line" />
              </F>
              <F label="Summary" wide>
                <textarea className="input h-[60px] resize-y" value={f.summary} onChange={set("summary")} placeholder="What happens and where" />
              </F>
              <F label="Error message or stack trace" wide>
                <textarea className="input mono h-[80px] resize-y text-[11.5px]" value={f.error} onChange={set("error")} data-testid="bug-error" />
              </F>
              <F label="Steps to reproduce (one per line)" wide>
                <textarea className="input h-[70px] resize-y" value={f.steps} onChange={set("steps")} />
              </F>
              <F label="Expected">
                <textarea className="input h-[54px] resize-y" value={f.expected} onChange={set("expected")} />
              </F>
              <F label="Actual">
                <textarea className="input h-[54px] resize-y" value={f.actual} onChange={set("actual")} />
              </F>
              <F label="Logs" wide>
                <textarea className="input mono h-[60px] resize-y text-[11.5px]" value={f.logs} onChange={set("logs")} />
              </F>
              <F label="Notes" wide>
                <textarea className="input h-[50px] resize-y" value={f.notes} onChange={set("notes")} />
              </F>
            </div>
            <div className="flex items-center gap-2 border-t border-line px-5 py-3">
              <span className="text-[11.5px] text-faint">The report includes the code around the line, the commit and pull request that introduced it, and who pushed it.</span>
              <div className="flex-1" />
              <button className="btn btn-sm btn-ghost" onClick={close}>
                Cancel
              </button>
              <button className="btn btn-sm btn-primary" disabled={busy || !t || !f.title.trim()} onClick={() => void submit()} data-testid="bug-submit">
                {busy ? "Sending…" : "Send report"}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function F({ label, children, wide }: { label: string; children: ReactNode; wide?: boolean }) {
  return (
    <label className={`flex flex-col gap-1 ${wide ? "col-span-2" : ""}`}>
      <span className="smallcaps">{label}</span>
      {children}
    </label>
  );
}
