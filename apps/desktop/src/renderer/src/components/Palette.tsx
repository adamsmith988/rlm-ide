import { useEffect, useMemo, useRef, useState } from "react";
import { call, errText } from "../api";
import { useApp } from "../store";
import { useTeam } from "./team/teamStore";
import { IFile, ISpark } from "../ui/icons";

interface Item {
  id: string;
  label: string;
  hint?: string;
  run: () => void;
}

function fuzzy(text: string, q: string): number {
  if (!q) return 1;
  const t = text.toLowerCase();
  let ti = 0;
  let score = 0;
  let streak = 0;
  for (const ch of q.toLowerCase()) {
    const i = t.indexOf(ch, ti);
    if (i < 0) return 0;
    streak = i === ti ? streak + 1 : 0;
    score += 1 + streak * 2 - Math.min(3, (i - ti) * 0.1);
    ti = i + 1;
  }
  const base = t.split("/").pop() ?? t;
  if (base.includes(q.toLowerCase())) score += 10;
  return score;
}

export function Palette() {
  const st = useApp();
  const mode = st.palette!;
  const [q, setQ] = useState(mode === "commands" ? ">" : "");
  const [files, setFiles] = useState<string[]>([]);
  const [sel, setSel] = useState(0);
  const list = useRef<HTMLDivElement>(null);
  const close = () => st.set({ palette: null });

  useEffect(() => {
    if (st.workspace) void call<string[]>("fs:listFiles", { limit: 20000 }).then(setFiles, () => undefined);
  }, []);

  const commands: Item[] = useMemo(
    () => [
      { id: "open", label: "Open folder…", run: async () => {
          const root = await call<string | null>("dialog:openFolder", {});
          if (root) await st.openWorkspace(root);
        } },
      { id: "chat", label: "New chat", hint: "Ctrl Shift N", run: () => void st.newChat() },
      { id: "agent", label: "Focus agent", hint: "Ctrl L", run: () => (st.set({ agent: true }), setTimeout(() => document.getElementById("composer")?.focus(), 30)) },
      { id: "sidebar", label: "Toggle sidebar", hint: "Ctrl B", run: () => st.set({ sidebar: !st.sidebar }) },
      { id: "panel", label: "Toggle terminal panel", hint: "Ctrl J", run: () => st.set({ bottom: !st.bottom }) },
      { id: "theme", label: `Switch to ${st.settings?.theme === "dark" ? "light" : "dark"} theme`, run: () => void st.updateSettings({ theme: st.settings?.theme === "dark" ? "light" : "dark" }) },
      { id: "save", label: "Save file", hint: "Ctrl S", run: () => st.activeTab && void st.saveTab(st.activeTab) },
      { id: "settings", label: "Open settings", hint: "Ctrl ,", run: () => st.set({ activity: "settings" }) },
      { id: "models", label: "Manage models", run: () => st.set({ activity: "settings" }) },
      { id: "memory", label: "Open memory", run: () => st.set({ activity: "memory" }) },
      { id: "team", label: "Open team hub", run: () => st.set({ activity: "team" }) },
      { id: "hub", label: "Open models hub (download and deploy Hugging Face models)", run: () => st.set({ activity: "hub" }) },
      { id: "bug", label: "Report a bug to the team", run: () => useTeam.getState().reportBug({}) },
      { id: "push", label: "Git: push to team hub", run: () => void call<string>("team:git", { op: "push" }).then(() => st.toast("Pushed"), (e) => st.toast(errText(e), "error")) },
      { id: "pull", label: "Git: pull from team hub", run: () => void call<string>("team:git", { op: "pull" }).then(() => (st.toast("Pulled"), void st.refreshGit()), (e) => st.toast(errText(e), "error")) },
      { id: "search", label: "Search in files", run: () => st.set({ activity: "search", sidebar: true }) },
      { id: "git", label: "Source control", run: () => st.set({ activity: "git", sidebar: true }) },
      { id: "chats", label: "Show chats", run: () => st.set({ activity: "chats", sidebar: true }) },
      { id: "stop", label: "Stop agent", run: () => void st.stopTurn() },
      { id: "closeall", label: "Close all editors", run: () => st.set({ tabs: st.tabs.filter((t) => t.content !== t.saved), activeTab: null }) },
    ],
    [st],
  );

  const isCmd = q.startsWith(">");
  const query = isCmd ? q.slice(1).trim() : q.trim();
  const items: Item[] = useMemo(() => {
    if (isCmd)
      return commands
        .map((c) => ({ c, s: fuzzy(c.label, query) }))
        .filter((x) => x.s > 0)
        .sort((a, b) => b.s - a.s)
        .map((x) => x.c);
    return files
      .map((f) => ({ f, s: fuzzy(f, query) }))
      .filter((x) => x.s > 0)
      .sort((a, b) => b.s - a.s || a.f.length - b.f.length)
      .slice(0, 60)
      .map(({ f }) => ({ id: f, label: f.split("/").pop()!, hint: f, run: () => void st.openFile(f) }));
  }, [isCmd, query, files, commands]);

  useEffect(() => setSel(0), [q]);
  useEffect(() => {
    list.current?.children[sel]?.scrollIntoView({ block: "nearest" });
  }, [sel]);

  const run = (it: Item | undefined) => {
    if (!it) return;
    close();
    it.run();
  };

  return (
    <div className="fixed inset-0 z-50 flex justify-center bg-black/30 pt-[12vh]" onMouseDown={close}>
      <div className="card h-fit w-[620px] max-w-[90vw] overflow-hidden fade-in" style={{ boxShadow: "var(--shadow-lg)" }} onMouseDown={(e) => e.stopPropagation()}>
        <input
          autoFocus
          className="w-full border-b border-line bg-transparent px-4 py-3 text-[14px] outline-none"
          placeholder={isCmd ? "Run a command" : "Go to file  ( > for commands )"}
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape") close();
            else if (e.key === "ArrowDown") (e.preventDefault(), setSel(Math.min(items.length - 1, sel + 1)));
            else if (e.key === "ArrowUp") (e.preventDefault(), setSel(Math.max(0, sel - 1)));
            else if (e.key === "Enter") run(items[sel]);
          }}
        />
        <div ref={list} className="max-h-[50vh] overflow-y-auto py-1">
          {items.map((it, i) => (
            <button
              key={it.id}
              className="flex w-full items-center gap-3 px-4 py-[7px] text-left text-[13px]"
              style={{ background: i === sel ? "var(--accent)" : undefined, color: i === sel ? "var(--accent-ink)" : undefined }}
              onMouseEnter={() => setSel(i)}
              onClick={() => run(it)}
            >
              {isCmd ? <ISpark size={14} className="opacity-60" /> : <IFile size={14} className="opacity-60" />}
              <span className="truncate">{it.label}</span>
              {it.hint && <span className="ml-auto truncate pl-4 text-[11.5px] opacity-60">{it.hint}</span>}
            </button>
          ))}
          {!items.length && <div className="px-4 py-3 text-[12.5px] text-faint">{isCmd ? "No matching command" : st.workspace ? "No matching files" : "Open a folder first"}</div>}
        </div>
      </div>
    </div>
  );
}
