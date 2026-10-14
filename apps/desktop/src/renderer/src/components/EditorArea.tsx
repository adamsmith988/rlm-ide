import { useEffect, useRef } from "react";
import { call, errText } from "../api";
import { languageFor, monaco } from "../monaco";
import { pendingLine, useApp, type Tab } from "../store";
import { IClose, IGlobe } from "../ui/icons";
import { Circles } from "../ui/LineArt";
import { useTeam } from "./team/teamStore";

const models = new Map<string, monaco.editor.ITextModel>();

function modelFor(tab: Tab): monaco.editor.ITextModel {
  let m = models.get(tab.path);
  if (!m || m.isDisposed()) {
    m = monaco.editor.createModel(tab.content, languageFor(tab.path), monaco.Uri.file(`/workspace/${tab.path}`));
    models.set(tab.path, m);
  } else if (m.getValue() !== tab.content) {
    m.pushEditOperations([], [{ range: m.getFullModelRange(), text: tab.content }], () => null);
  }
  return m;
}

monaco.editor.onDidChangeMarkers(() => {
  const markers = monaco.editor.getModelMarkers({});
  useApp.getState().set({
    diagnostics: markers.map((mk) => ({
      path: mk.resource.path.replace(/^\/workspace\//, ""),
      line: mk.startLineNumber,
      severity: mk.severity === monaco.MarkerSeverity.Error ? "error" : mk.severity === monaco.MarkerSeverity.Warning ? "warning" : "info",
      message: mk.message,
    })),
  });
});

export function EditorArea() {
  const { tabs, activeTab, set, closeTab } = useApp();
  const tab = tabs.find((t) => t.id === activeTab);

  useEffect(() => {
    const open = new Set(tabs.filter((t) => t.kind === "file").map((t) => t.path));
    for (const [p, m] of models) if (!open.has(p)) (m.dispose(), models.delete(p));
  }, [tabs]);

  return (
    <div className="flex h-full flex-col bg-bg">
      {tabs.length > 0 && (
        <div className="flex h-[36px] shrink-0 items-end gap-0.5 overflow-x-auto border-b border-line bg-bg px-2">
          {tabs.map((t) => {
            const on = t.id === activeTab;
            const dirty = t.content !== t.saved;
            return (
              <div
                key={t.id}
                className="group flex h-[30px] max-w-[220px] shrink-0 cursor-pointer items-center gap-2 rounded-t-[10px] px-3 text-[12.5px]"
                style={{ background: on ? "var(--surface)" : "transparent", color: on ? "var(--text)" : "var(--text-dim)", boxShadow: on ? "inset 0 1px 0 var(--border), 1px 0 0 var(--border), -1px 0 0 var(--border)" : undefined }}
                onClick={() => set({ activeTab: t.id })}
                onMouseDown={(e) => e.button === 1 && closeTab(t.id)}
                title={t.path}
              >
                <span className={`truncate ${t.kind === "diff" ? "italic" : ""}`}>{t.title}</span>
                <button
                  className="flex h-4 w-4 items-center justify-center rounded"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (dirty && !confirm(`Discard unsaved changes to ${t.title}?`)) return;
                    closeTab(t.id);
                  }}
                >
                  {dirty ? <span className="h-2 w-2 rounded-full bg-ink group-hover:hidden" /> : null}
                  <IClose size={12} className={dirty ? "hidden group-hover:block" : "opacity-0 group-hover:opacity-100"} />
                </button>
              </div>
            );
          })}
          <span className="flex-1" />
          {tab?.kind === "file" && /\.(html?|svg)$/i.test(tab.path) && (
            <button
              className="btn btn-sm mb-[4px] shrink-0"
              title="Open this file in the built-in browser (saves first)"
              onClick={() => {
                const a = useApp.getState();
                void (tab.content !== tab.saved ? a.saveTab(tab.id) : Promise.resolve()).then(() => a.openInBrowser(tab.path));
              }}
            >
              <IGlobe size={12} /> Preview
            </button>
          )}
        </div>
      )}
      <div className="relative min-h-0 flex-1">
        {!tab && <EmptyEditor />}
        {tab?.kind === "file" && (tab.binary ? <div className="p-6 text-dim">Binary file — not shown.</div> : <CodeEditor tab={tab} />)}
        {tab?.kind === "diff" && <DiffView tab={tab} />}
      </div>
    </div>
  );
}

function EmptyEditor() {
  const keys: Array<[string, string]> = [
    ["Quick open", "Ctrl P"],
    ["Commands", "Ctrl Shift P"],
    ["Ask the agent", "Ctrl L"],
    ["New chat", "Ctrl Shift N"],
    ["Toggle terminal", "Ctrl J"],
  ];
  return (
    <div className="relative flex h-full items-center justify-center overflow-hidden">
      <Circles count={2} />
      <div className="relative text-center">
        <div className="heading text-[34px]">Nothing open</div>
        <div className="mt-1 text-dim">Pick a file, or ask the agent to build something.</div>
        <div className="mt-6 inline-grid grid-cols-[auto_auto] gap-x-6 gap-y-1.5 text-left text-[12px]">
          {keys.map(([l, k]) => (
            <div key={l} className="contents">
              <span className="text-dim">{l}</span>
              <span className="mono text-faint">{k}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function editorOptions(fontSize: number): monaco.editor.IStandaloneEditorConstructionOptions {
  return {
    fontFamily: "JetBrains Mono, ui-monospace, monospace",
    fontSize,
    lineHeight: Math.round(fontSize * 1.6),
    minimap: { enabled: false },
    scrollBeyondLastLine: false,
    smoothScrolling: true,
    renderLineHighlight: "line",
    padding: { top: 12 },
    automaticLayout: true,
    tabSize: 2,
    bracketPairColorization: { enabled: false },
    guides: { indentation: true },
    stickyScroll: { enabled: false },
    fixedOverflowWidgets: true,
  };
}

function CodeEditor({ tab }: { tab: Tab }) {
  const el = useRef<HTMLDivElement>(null);
  const ed = useRef<monaco.editor.IStandaloneCodeEditor | null>(null);
  const theme = useApp((s) => s.settings?.theme ?? "dark");
  const fontSize = useApp((s) => s.settings?.fontSize ?? 13);

  useEffect(() => {
    const editor = monaco.editor.create(el.current!, { ...editorOptions(fontSize), theme: theme === "dark" ? "rlm-dark" : "rlm-light" });
    ed.current = editor;
    const subs = [
      editor.onDidChangeModelContent(() => {
        const m = editor.getModel();
        if (!m) return;
        const path = m.uri.path.replace(/^\/workspace\//, "");
        const st = useApp.getState();
        const t = st.tabs.find((x) => x.kind === "file" && x.path === path);
        if (t && t.content !== m.getValue()) st.setTabContent(t.id, m.getValue());
      }),
      editor.onDidChangeCursorSelection((e) => {
        const m = editor.getModel();
        const text = m ? m.getValueInRange(e.selection) : "";
        useApp.getState().set({ selection: text.slice(0, 8000), cursorLine: e.selection.positionLineNumber });
      }),
      editor.addAction({
        id: "rlm.reportBug",
        label: "Report a bug on this line to the team",
        contextMenuGroupId: "9_team",
        run: (ed) => {
          const m = ed.getModel();
          if (!m) return;
          const file = m.uri.path.replace(/^\/workspace\//, "");
          const line = ed.getPosition()?.lineNumber ?? 1;
          const diag = useApp.getState().diagnostics.find((d) => d.path === file && Math.abs(d.line - line) <= 1);
          useTeam.getState().reportBug({ file, line, error: diag?.message });
        },
      }),
    ];
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
      const st = useApp.getState();
      if (st.activeTab) void st.saveTab(st.activeTab);
    });
    return () => {
      subs.forEach((s) => s.dispose());
      editor.dispose();
    };
  }, []);

  useEffect(() => {
    monaco.editor.setTheme(theme === "dark" ? "rlm-dark" : "rlm-light");
  }, [theme]);
  useEffect(() => {
    ed.current?.updateOptions({ fontSize, lineHeight: Math.round(fontSize * 1.6) });
  }, [fontSize]);

  useEffect(() => {
    const editor = ed.current;
    if (!editor) return;
    const m = modelFor(tab);
    if (editor.getModel() !== m) editor.setModel(m);
    if (pendingLine.path === tab.path && pendingLine.line) {
      const line = pendingLine.line;
      Object.assign(pendingLine, { path: "", line: 0 });
      editor.revealLineInCenter(line);
      editor.setPosition({ lineNumber: line, column: 1 });
      editor.focus();
    }
  }, [tab.path, tab.content]);

  return <div ref={el} className="h-full w-full" />;
}

function DiffView({ tab }: { tab: Tab }) {
  const el = useRef<HTMLDivElement>(null);
  const theme = useApp((s) => s.settings?.theme ?? "dark");
  const fontSize = useApp((s) => s.settings?.fontSize ?? 13);
  const { closeTab, toast, set, treeVersion } = useApp();

  useEffect(() => {
    const d = monaco.editor.createDiffEditor(el.current!, {
      ...editorOptions(fontSize),
      readOnly: true,
      renderSideBySide: true,
      theme: theme === "dark" ? "rlm-dark" : "rlm-light",
      originalEditable: false,
    });
    const lang = languageFor(tab.path);
    const original = monaco.editor.createModel(tab.diff?.before ?? "", lang);
    const modified = monaco.editor.createModel(tab.diff?.after ?? "", lang);
    d.setModel({ original, modified });
    return () => {
      d.dispose();
      original.dispose();
      modified.dispose();
    };
  }, [tab.id, tab.diff?.before, tab.diff?.after, theme, fontSize]);

  const resolve = async (accept: boolean) => {
    try {
      await call("edits:resolve", { editId: tab.diff!.editId!, accept });
      toast(accept ? "Edit kept" : "Edit reverted");
      set({ treeVersion: treeVersion + 1 });
      closeTab(tab.id);
    } catch (e) {
      toast(errText(e), "error");
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex h-[40px] shrink-0 items-center gap-3 border-b border-line px-4">
        <span className="smallcaps">Diff</span>
        <span className="mono truncate text-[12px] text-dim">{tab.path}</span>
        <div className="flex-1" />
        {tab.diff?.editId && (
          <>
            <button className="btn btn-sm" onClick={() => void resolve(false)}>
              Revert
            </button>
            <button className="btn btn-primary btn-sm" onClick={() => void resolve(true)}>
              Keep
            </button>
          </>
        )}
      </div>
      <div ref={el} className="min-h-0 flex-1" />
    </div>
  );
}
