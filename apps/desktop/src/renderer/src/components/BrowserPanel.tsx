import { useEffect, useRef, useState } from "react";
import type { BrowserElementRef, BrowserScreenshot, BrowserState } from "@rlm/contracts";
import { api, call, errText } from "../api";
import { useApp } from "../store";
import { IBack, ICamera, IClose, IExternal, IForward, IGlobe, IInspect, IPick, IRefresh, IStop } from "../ui/icons";

let attachSeq = 0;

/** The built-in browser. The page itself is a native view drawn by main over the `host` box. */
export function BrowserPanel() {
  const st = useApp((s) => s.browserState);
  const overlay = useApp((s) => !!s.palette);
  const { set, toast, attach, workspace } = useApp();
  const [input, setInput] = useState(st.url);
  const [editing, setEditing] = useState(false);
  const host = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!editing) setInput(displayUrl(st.url));
  }, [st.url, editing]);

  // Keep the native view glued to the host box (layout changes do not always resize it).
  useEffect(() => {
    let last = "";
    let raf = 0;
    const tick = () => {
      const el = host.current;
      if (el) {
        const r = el.getBoundingClientRect();
        const visible = !overlay;
        const sig = `${r.x},${r.y},${r.width},${r.height},${visible}`;
        if (sig !== last) {
          last = sig;
          api.browserBounds({ x: r.x, y: r.y, width: r.width, height: r.height }, visible);
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      api.browserBounds({ x: 0, y: 0, width: 0, height: 0 }, false);
    };
  }, [overlay]);

  const cmd = (c: "back" | "forward" | "reload" | "stop" | "devtools" | "close" | "external" | "cancelPick") => void call("browser:command", { cmd: c }).catch((e) => toast(errText(e), "error"));
  const go = (url: string) => {
    if (!url.trim()) return;
    setEditing(false);
    void call("browser:navigate", { url }).catch((e) => toast(errText(e), "error"));
  };
  const screenshot = async () => {
    try {
      const r = await call<BrowserScreenshot>("browser:screenshot", {});
      attach({ id: `a${++attachSeq}`, kind: "screenshot", image: r.url, pageUrl: r.pageUrl, title: r.title, width: r.width, height: r.height });
      toast("Screenshot attached to the chat");
    } catch (e) {
      toast(errText(e), "error");
    }
  };
  const pick = async () => {
    if (st.picking) return cmd("cancelPick");
    try {
      const r = await call<BrowserElementRef | null>("browser:pick", {});
      if (r) attach({ id: `a${++attachSeq}`, kind: "element", image: r.image, pageUrl: r.url, title: r.title, selector: r.selector, tag: r.tag, text: r.text, html: r.html });
    } catch (e) {
      toast(errText(e), "error");
    }
  };

  return (
    <section className="flex h-full flex-col border-l border-line bg-bg" data-testid="browser-panel">
      <div className="flex h-10 shrink-0 items-center gap-1 px-2">
        <button className="iconbtn" title="Back" disabled={!st.canGoBack} style={{ opacity: st.canGoBack ? 1 : 0.35 }} onClick={() => cmd("back")}>
          <IBack size={15} />
        </button>
        <button className="iconbtn" title="Forward" disabled={!st.canGoForward} style={{ opacity: st.canGoForward ? 1 : 0.35 }} onClick={() => cmd("forward")}>
          <IForward size={15} />
        </button>
        <button className="iconbtn" title={st.loading ? "Stop" : "Reload"} onClick={() => cmd(st.loading ? "stop" : "reload")}>
          {st.loading ? <IStop size={13} /> : <IRefresh size={14} />}
        </button>
        <form
          className="mx-1 flex h-[28px] min-w-0 flex-1 items-center gap-2 rounded-full border-[1.5px] border-line-strong bg-surface px-3 focus-within:border-ink"
          onSubmit={(e) => {
            e.preventDefault();
            go(input);
          }}
        >
          {st.loading ? <span className="pulse-dot" /> : <IGlobe size={13} className="shrink-0 text-faint" />}
          <input
            id="browser-url"
            className="min-w-0 flex-1 bg-transparent text-[12px] outline-none placeholder:text-faint"
            value={input}
            placeholder="Search or enter a URL or project file"
            spellCheck={false}
            onFocus={(e) => {
              setEditing(true);
              e.currentTarget.select();
            }}
            onBlur={() => setEditing(false)}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                setInput(displayUrl(st.url));
                e.currentTarget.blur();
              }
            }}
          />
        </form>
        <button className="iconbtn" title="Screenshot to chat" disabled={!st.open} onClick={() => void screenshot()}>
          <ICamera size={15} />
        </button>
        <button
          className="iconbtn"
          title={st.picking ? "Cancel picking (Esc)" : "Pick an element to reference in chat"}
          disabled={!st.open}
          style={st.picking ? { background: "var(--text)", color: "var(--bg)" } : undefined}
          onClick={() => void pick()}
        >
          <IPick size={15} />
        </button>
        <button className="iconbtn" title="Inspect (DevTools)" disabled={!st.open} style={st.devtools ? { background: "var(--text)", color: "var(--bg)" } : undefined} onClick={() => cmd("devtools")}>
          <IInspect size={15} />
        </button>
        <button className="iconbtn" title="Open in system browser" disabled={!/^https?:/.test(st.url)} onClick={() => cmd("external")}>
          <IExternal size={14} />
        </button>
        <button
          className="iconbtn"
          title="Close browser"
          onClick={() => {
            cmd("close");
            set({ browser: false });
          }}
        >
          <IClose size={14} />
        </button>
      </div>
      {(st.agent || st.picking || st.error) && (
        <div className="flex shrink-0 items-center gap-2 border-y border-line px-3 py-1.5 text-[11.5px]" data-testid="browser-banner">
          {st.agent ? (
            <>
              <span className="rounded-full bg-ink px-2 py-[1px] text-[10px] font-semibold uppercase tracking-[0.08em] text-bg">Agent</span>
              <span className="pulse-dot" />
              <span className="truncate">{st.agent}</span>
            </>
          ) : st.picking ? (
            <>
              <IPick size={13} />
              <span>Click an element on the page to reference it in the chat. Esc cancels.</span>
            </>
          ) : (
            <span className="truncate text-err">{st.error}</span>
          )}
        </div>
      )}
      <div ref={host} className="relative min-h-0 flex-1 bg-surface">
        {!st.open && <StartPage onGo={go} hasWorkspace={!!workspace} />}
      </div>
    </section>
  );
}

function displayUrl(url: string): string {
  if (url.startsWith("rlm-preview://")) return "Chat preview";
  return url;
}

function StartPage({ onGo, hasWorkspace }: { onGo: (u: string) => void; hasWorkspace: boolean }) {
  const [pages, setPages] = useState<string[]>([]);
  useEffect(() => {
    if (!hasWorkspace) return;
    void call<string[]>("fs:listFiles", { limit: 5000 })
      .then((files) => setPages(files.filter((f) => /\.html?$/i.test(f) && !/node_modules|\/dist\//.test(f)).slice(0, 8)))
      .catch(() => undefined);
  }, [hasWorkspace]);
  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 p-6 text-center">
      <div className="relative flex h-[120px] w-[120px] items-center justify-center rounded-full border-[1.5px] border-ink">
        <div className="absolute inset-[14px] rounded-full border border-line-strong" />
        <IGlobe size={44} strokeWidth={1.3} />
      </div>
      <div className="heading text-[22px]">Browse, inspect, reference</div>
      <div className="max-w-[300px] text-[12px] text-dim">Open a site or a page from your project. Screenshot it or pick an element to send to the agent; the agent can drive this browser too.</div>
      {pages.length > 0 && (
        <div className="flex max-w-[360px] flex-wrap justify-center gap-1.5">
          {pages.map((p) => (
            <button key={p} className="rounded-full border-[1.5px] border-ink px-3 py-[3px] text-[11.5px] hover:bg-ink hover:text-bg" onClick={() => onGo(p)}>
              {p}
            </button>
          ))}
        </div>
      )}
      <div className="flex gap-1.5">
        {["localhost:3000", "localhost:5173", "developer.mozilla.org"].map((u) => (
          <button key={u} className="btn btn-sm" onClick={() => onGo(u)}>
            {u}
          </button>
        ))}
      </div>
    </div>
  );
}

export function useBrowserSync(): void {
  useEffect(
    () =>
      api.onBrowserState((st: BrowserState) => {
        const a = useApp.getState();
        a.set({ browserState: st, ...(st.agent && !a.browser ? { browser: true } : {}) });
      }),
    [],
  );
}
