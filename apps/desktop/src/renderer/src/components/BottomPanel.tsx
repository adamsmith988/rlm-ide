import { useEffect, useRef, useState } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import "@xterm/xterm/css/xterm.css";
import { api, call, errText } from "../api";
import { useApp } from "../store";
import { IClose, IPlus } from "../ui/icons";

export function BottomPanel() {
  const { bottomTab, set, diagnostics } = useApp();
  const [terms, setTerms] = useState<number[]>([1]);
  const [activeTerm, setActiveTerm] = useState(1);
  const tabsDef: Array<[typeof bottomTab, string]> = [
    ["terminal", "Terminal"],
    ["problems", `Problems${diagnostics.length ? ` · ${diagnostics.length}` : ""}`],
  ];
  return (
    <div className="flex h-full flex-col bg-bg">
      <div className="flex h-[32px] shrink-0 items-center gap-1 px-3">
        {tabsDef.map(([id, label]) => (
          <button
            key={id}
            className="rounded-full px-3 py-[3px] text-[11px] font-semibold uppercase tracking-[0.1em]"
            style={{ background: bottomTab === id ? "var(--accent)" : "transparent", color: bottomTab === id ? "var(--accent-ink)" : "var(--text-dim)" }}
            onClick={() => set({ bottomTab: id })}
          >
            {label}
          </button>
        ))}
        {bottomTab === "terminal" && (
          <div className="ml-3 flex items-center gap-1">
            {terms.map((t, i) => (
              <button
                key={t}
                className="group flex items-center gap-1 rounded-full border px-2 py-[1px] text-[11px]"
                style={{ borderColor: t === activeTerm ? "var(--border-strong)" : "transparent", color: t === activeTerm ? "var(--text)" : "var(--text-dim)" }}
                onClick={() => setActiveTerm(t)}
              >
                shell {i + 1}
                {terms.length > 1 && (
                  <IClose
                    size={10}
                    className="opacity-0 group-hover:opacity-100"
                    onClick={(e) => {
                      e.stopPropagation();
                      const next = terms.filter((x) => x !== t);
                      setTerms(next);
                      if (activeTerm === t) setActiveTerm(next[0]!);
                    }}
                  />
                )}
              </button>
            ))}
            <button
              className="iconbtn h-6 w-6"
              title="New terminal"
              onClick={() => {
                const id = Math.max(...terms) + 1;
                setTerms([...terms, id]);
                setActiveTerm(id);
              }}
            >
              <IPlus size={13} />
            </button>
          </div>
        )}
        <div className="flex-1" />
        <button className="iconbtn h-6 w-6" title="Close panel" onClick={() => set({ bottom: false })}>
          <IClose size={13} />
        </button>
      </div>
      <div className="relative min-h-0 flex-1">
        {terms.map((t) => (
          <div key={t} className="absolute inset-0 px-3 pb-1" style={{ visibility: bottomTab === "terminal" && t === activeTerm ? "visible" : "hidden" }}>
            <XTerm visible={bottomTab === "terminal" && t === activeTerm} />
          </div>
        ))}
        {bottomTab === "problems" && <Problems />}
      </div>
    </div>
  );
}

function Problems() {
  const { diagnostics, openFile } = useApp();
  if (!diagnostics.length) return <div className="px-4 py-2 text-[12px] text-faint">No problems in open files.</div>;
  return (
    <div className="absolute inset-0 overflow-auto px-2 text-[12px]">
      {diagnostics.map((d, i) => (
        <button key={i} className="flex w-full gap-2 rounded px-2 py-[3px] text-left hover:bg-surface-2" onClick={() => void openFile(d.path, { line: d.line })}>
          <span style={{ color: d.severity === "error" ? "var(--err)" : "var(--warn)" }}>{d.severity === "error" ? "●" : "▲"}</span>
          <span className="truncate">{d.message}</span>
          <span className="mono ml-auto shrink-0 text-faint">
            {d.path}:{d.line}
          </span>
        </button>
      ))}
    </div>
  );
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function XTerm({ visible }: { visible: boolean }) {
  const el = useRef<HTMLDivElement>(null);
  const fit = useRef<FitAddon | null>(null);
  const termRef = useRef<Terminal | null>(null);
  const theme = useApp((s) => s.settings?.theme ?? "dark");
  const root = useApp((s) => s.workspace?.root);

  useEffect(() => {
    const term = new Terminal({
      fontFamily: "JetBrains Mono, ui-monospace, monospace",
      fontSize: 12.5,
      cursorBlink: true,
      allowProposedApi: true,
      scrollback: 5000,
      theme: { background: cssVar("--bg"), foreground: cssVar("--text"), cursor: cssVar("--text"), selectionBackground: cssVar("--surface-3") },
    });
    const f = new FitAddon();
    term.loadAddon(f);
    term.open(el.current!);
    fit.current = f;
    termRef.current = term;
    let id: string | null = null;
    let disposed = false;
    const offData = api.onPtyData((m) => m.id === id && term.write(m.data));
    const offExit = api.onPtyExit((m) => m.id === id && term.write(`\r\n\x1b[2m[process exited with code ${m.code}]\x1b[0m\r\n`));
    try {
      f.fit();
    } catch {
      /* hidden */
    }
    call<string>("pty:create", { cols: Math.max(term.cols, 20), rows: Math.max(term.rows, 5) }).then(
      (pid) => {
        if (disposed) void call("pty:kill", { id: pid });
        else id = pid;
      },
      (e) => term.write(`\x1b[31m${errText(e)}\x1b[0m\r\n`),
    );
    const sub = term.onData((d) => id && void call("pty:write", { id, data: d }));
    const ro = new ResizeObserver(() => {
      try {
        f.fit();
        if (id) void call("pty:resize", { id, cols: term.cols, rows: term.rows });
      } catch {
        /* not laid out */
      }
    });
    ro.observe(el.current!);
    return () => {
      disposed = true;
      ro.disconnect();
      sub.dispose();
      offData();
      offExit();
      if (id) void call("pty:kill", { id });
      term.dispose();
    };
  }, [root]);

  useEffect(() => {
    const t = termRef.current;
    if (t) t.options.theme = { background: cssVar("--bg"), foreground: cssVar("--text"), cursor: cssVar("--text"), selectionBackground: cssVar("--surface-3") };
  }, [theme]);

  useEffect(() => {
    if (visible) setTimeout(() => (fit.current?.fit(), termRef.current?.focus()), 20);
  }, [visible]);

  return <div ref={el} className="h-full w-full" />;
}
