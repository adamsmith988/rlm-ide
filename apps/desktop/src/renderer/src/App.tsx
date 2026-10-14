import { useEffect, useRef, useState, type ReactNode } from "react";
import { api, call } from "./api";
import { useApp } from "./store";
import { TitleBar } from "./components/TitleBar";
import { ActivityBar } from "./components/ActivityBar";
import { StatusBar } from "./components/StatusBar";
import { Welcome } from "./components/Welcome";
import { Explorer } from "./components/Explorer";
import { SearchPanel } from "./components/SearchPanel";
import { GitPanel } from "./components/GitPanel";
import { ChatsPanel } from "./components/ChatsPanel";
import { EditorArea } from "./components/EditorArea";
import { BottomPanel } from "./components/BottomPanel";
import { AgentPanel } from "./components/agent/AgentPanel";
import { MemoryScreen } from "./components/MemoryScreen";
import { SettingsScreen } from "./components/settings/SettingsScreen";
import { Palette } from "./components/Palette";
import { Toasts } from "./components/Toasts";
import { BrowserPanel, useBrowserSync } from "./components/BrowserPanel";
import { TeamScreen } from "./components/team/TeamScreen";
import { HubScreen } from "./components/hub/HubScreen";
import { BugDialog } from "./components/team/BugDialog";
import { wireTeam } from "./components/team/teamStore";

export function App() {
  const s = useApp();
  const [ready, setReady] = useState(false);
  useBrowserSync();

  useEffect(() => {
    const st = useApp.getState();
    void st
      .loadSettings()
      .finally(() => setReady(true))
      .then(() => call<string | null>("app:takeAutoOpen", {}))
      .then((root) => (root ? useApp.getState().openWorkspace(root) : undefined));
    const offs = [
      api.onTurnEvent((ev) => useApp.getState().onTurnEvent(ev)),
      api.onTaskBoard((b) => useApp.getState().onTaskBoard(b)),
      api.onStatus((status) => useApp.setState({ status: status as never })),
      api.onSettings((settings) => {
        useApp.setState({ settings });
        document.documentElement.dataset.theme = settings.theme;
      }),
      api.onFsChanged(() => {
        clearTimeout(fsTimer);
        fsTimer = setTimeout(() => {
          const a = useApp.getState();
          a.set({ treeVersion: a.treeVersion + 1 });
          void a.refreshGit();
        }, 250);
      }),
      wireTeam((title, body) => useApp.getState().toast(`${title}: ${body}`)),
    ];
    return () => offs.forEach((off) => off());
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;
      if (!mod) return;
      const a = useApp.getState();
      const k = e.key.toLowerCase();
      if (k === "p" && e.shiftKey) a.set({ palette: "commands" });
      else if (k === "p") a.set({ palette: "files" });
      else if (k === "s" && a.activeTab) void a.saveTab(a.activeTab);
      else if (k === "b") a.set({ sidebar: !a.sidebar });
      else if (k === "j" || k === "`") a.set({ bottom: !a.bottom });
      else if (k === "l") {
        a.set({ agent: true });
        setTimeout(() => document.getElementById("composer")?.focus(), 30);
      } else if (k === "n" && e.shiftKey) void a.newChat();
      else if (k === "," ) a.set({ activity: "settings" });
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Push editor state to the orchestrator (debounced) so the agent sees the active file and selection.
  const tabs = s.tabs;
  const activeTab = s.activeTab;
  const selection = s.selection;
  useEffect(() => {
    const t = setTimeout(() => {
      const tab = tabs.find((x) => x.id === activeTab);
      api.pushEditorState({
        activeFile: tab?.kind === "file" ? tab.path : null,
        selection,
        openFiles: tabs.filter((x) => x.kind === "file").map((x) => x.path),
        dirty: tabs.filter((x) => x.content !== x.saved).map((x) => x.path),
      });
    }, 300);
    return () => clearTimeout(t);
  }, [tabs, activeTab, selection]);

  if (!ready) return <div className="h-full bg-bg" />;

  const screen = s.activity === "memory" || s.activity === "settings" || s.activity === "team" || s.activity === "hub";
  return (
    <div className="flex h-full flex-col">
      <TitleBar />
      {!s.workspace && !screen ? (
        <Welcome />
      ) : (
        <div className="flex min-h-0 flex-1">
          <ActivityBar />
          {screen ? (
            <div className="min-w-0 flex-1 overflow-hidden">{s.activity === "memory" ? <MemoryScreen /> : s.activity === "team" ? <TeamScreen /> : s.activity === "hub" ? <HubScreen /> : <SettingsScreen />}</div>
          ) : (
            <>
              {s.sidebar && (
                <Resizable side="right" initial={264} min={180} max={520}>
                  <aside className="flex h-full flex-col border-r border-line bg-surface">
                    {s.activity === "explorer" && <Explorer />}
                    {s.activity === "search" && <SearchPanel />}
                    {s.activity === "git" && <GitPanel />}
                    {s.activity === "chats" && <ChatsPanel />}
                  </aside>
                </Resizable>
              )}
              <main className="flex min-w-0 flex-1 flex-col">
                <div className="min-h-0 flex-1">
                  <EditorArea />
                </div>
                {s.bottom && (
                  <ResizableV initial={220} min={100} max={600}>
                    <BottomPanel />
                  </ResizableV>
                )}
              </main>
              {s.browser && (
                <Resizable side="left" initial={560} min={340} max={1200}>
                  <BrowserPanel />
                </Resizable>
              )}
              {s.agent && (
                <Resizable side="left" initial={420} min={320} max={760}>
                  <AgentPanel />
                </Resizable>
              )}
            </>
          )}
        </div>
      )}
      <StatusBar />
      {s.palette && <Palette />}
      <BugDialog />
      <Toasts />
    </div>
  );
}

let fsTimer: ReturnType<typeof setTimeout> | undefined;

function Resizable({ side, initial, min, max, children }: { side: "left" | "right"; initial: number; min: number; max: number; children: ReactNode }) {
  const [w, setW] = useState(initial);
  const drag = useRef<{ x: number; w: number } | null>(null);
  useEffect(() => {
    const move = (e: MouseEvent) => {
      if (!drag.current) return;
      const dx = e.clientX - drag.current.x;
      setW(Math.max(min, Math.min(max, drag.current.w + (side === "right" ? dx : -dx))));
    };
    const up = () => {
      drag.current = null;
      document.body.style.cursor = "";
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
  }, [side, min, max]);
  const handle = (
    <div
      className="w-1 shrink-0 cursor-col-resize hover:bg-line-strong"
      onMouseDown={(e) => {
        drag.current = { x: e.clientX, w };
        document.body.style.cursor = "col-resize";
      }}
    />
  );
  return (
    <div className="flex h-full shrink-0" style={{ width: w }}>
      {side === "left" && handle}
      <div className="min-w-0 flex-1">{children}</div>
      {side === "right" && handle}
    </div>
  );
}

function ResizableV({ initial, min, max, children }: { initial: number; min: number; max: number; children: ReactNode }) {
  const [h, setH] = useState(initial);
  const drag = useRef<{ y: number; h: number } | null>(null);
  useEffect(() => {
    const move = (e: MouseEvent) => {
      if (drag.current) setH(Math.max(min, Math.min(max, drag.current.h - (e.clientY - drag.current.y))));
    };
    const up = () => (drag.current = null);
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
  }, [min, max]);
  return (
    <div className="flex shrink-0 flex-col" style={{ height: h }}>
      <div className="h-1 shrink-0 cursor-row-resize border-t border-line hover:bg-line-strong" onMouseDown={(e) => (drag.current = { y: e.clientY, h })} />
      <div className="min-h-0 flex-1">{children}</div>
    </div>
  );
}
