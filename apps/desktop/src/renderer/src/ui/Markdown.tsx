import { Fragment, memo, useEffect, useRef, useState, type ReactNode } from "react";
import { call } from "../api";
import { useApp } from "../store";
import { ICheck, ICode, ICopy, IExternal, IGlobe } from "./icons";

/**
 * Safe markdown renderer: never injects message HTML into the app. Fenced ```html / ```svg / ```canvas
 * blocks render in a sandboxed, offline frame; ```mermaid renders as a diagram.
 * `previews=false` shows those blocks as code (used for user messages).
 */
export const Markdown = memo(function Markdown({ text, previews = true }: { text: string; previews?: boolean }) {
  return <div className="md selectable">{renderBlocks(text, previews)}</div>;
});

const LIST_RE = /^(\s*)([-*+]|\d+[.)])\s+/;
const TABLE_SEP_RE = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

function renderBlocks(src: string, previews: boolean): ReactNode[] {
  const lines = src.replace(/\r\n/g, "\n").split("\n");
  const out: ReactNode[] = [];
  let i = 0;
  let key = 0;
  while (i < lines.length) {
    const line = lines[i]!;
    const fence = /^\s{0,3}(`{3,}|~{3,})\s*([\w+#.-]*)/.exec(line);
    if (fence) {
      const marker = fence[1]!;
      const body: string[] = [];
      i++;
      let closed = false;
      while (i < lines.length) {
        if (lines[i]!.trim().startsWith(marker) && lines[i]!.trim().replace(/[`~]/g, "") === "") {
          closed = true;
          i++;
          break;
        }
        body.push(lines[i++]!);
      }
      out.push(<FencedBlock key={key++} lang={(fence[2] || "").toLowerCase()} code={body.join("\n")} closed={closed} previews={previews} />);
      continue;
    }
    const h = /^(#{1,6})\s+(.*?)\s*#*$/.exec(line);
    if (h) {
      const level = Math.min(h[1]!.length, 4);
      const Tag = `h${level}` as "h1" | "h2" | "h3" | "h4";
      out.push(<Tag key={key++}>{inline(h[2]!)}</Tag>);
      i++;
      continue;
    }
    if (/^\s{0,3}([-*_])(\s*\1){2,}\s*$/.test(line)) {
      out.push(<hr key={key++} />);
      i++;
      continue;
    }
    if (line.includes("|") && i + 1 < lines.length && TABLE_SEP_RE.test(lines[i + 1]!)) {
      const head = cells(line);
      const align = cells(lines[i + 1]!).map((c) => (/^:-+:$/.test(c) ? "center" : /-:$/.test(c) ? "right" : "left") as "left" | "center" | "right");
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && lines[i]!.includes("|") && lines[i]!.trim()) rows.push(cells(lines[i++]!));
      out.push(
        <div key={key++} className="md-table">
          <table>
            <thead>
              <tr>
                {head.map((c, j) => (
                  <th key={j} style={{ textAlign: align[j] }}>
                    {inline(c)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, ri) => (
                <tr key={ri}>
                  {head.map((_, j) => (
                    <td key={j} style={{ textAlign: align[j] }}>
                      {inline(r[j] ?? "")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      continue;
    }
    if (LIST_RE.test(line)) {
      const ordered = /^\s*\d+[.)]/.test(line);
      const start = ordered ? Number(/\d+/.exec(line)![0]) : undefined;
      const items: string[] = [];
      while (i < lines.length && LIST_RE.test(lines[i]!)) {
        let item = lines[i]!.replace(LIST_RE, "");
        i++;
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]!) && !LIST_RE.test(lines[i]!)) item += "\n" + lines[i++]!.trim();
        items.push(item);
      }
      const L = ordered ? "ol" : "ul";
      out.push(
        <L key={key++} start={start !== 1 ? start : undefined} className={items.some((t) => /^\[[ xX]\]\s/.test(t)) ? "tasks" : undefined}>
          {items.map((it, j) => {
            const task = /^\[([ xX])\]\s+([\s\S]*)$/.exec(it);
            if (task)
              return (
                <li key={j} className="task">
                  <span className={`taskbox ${task[1] !== " " ? "on" : ""}`}>{task[1] !== " " && <ICheck size={10} />}</span>
                  <span>{inline(task[2]!)}</span>
                </li>
              );
            return <li key={j}>{inline(it)}</li>;
          })}
        </L>,
      );
      continue;
    }
    if (/^\s*>\s?/.test(line)) {
      const q: string[] = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i]!)) q.push(lines[i++]!.replace(/^\s*>\s?/, ""));
      out.push(<blockquote key={key++}>{renderBlocks(q.join("\n"), previews)}</blockquote>);
      continue;
    }
    if (!line.trim()) {
      i++;
      continue;
    }
    const para: string[] = [];
    while (
      i < lines.length &&
      lines[i]!.trim() &&
      !/^\s{0,3}(`{3,}|~{3,}|#{1,6}\s|>\s?)/.test(lines[i]!) &&
      !LIST_RE.test(lines[i]!) &&
      !(lines[i]!.includes("|") && i + 1 < lines.length && TABLE_SEP_RE.test(lines[i + 1]!))
    )
      para.push(lines[i++]!);
    out.push(<p key={key++}>{inline(para.join("\n"))}</p>);
  }
  return out;
}

function cells(row: string): string[] {
  return row
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split(/(?<!\\)\|/)
    .map((c) => c.trim().replace(/\\\|/g, "|"));
}

const SAFE_IMG = /^(rlm-attach:\/\/img\/[\w-]+\.png|data:image\/(png|jpeg|gif|webp);base64,[\w+/=]+)$/;
const LINK = /^(https?:\/\/|file:\/\/|rlm-preview:\/\/)/;

function inline(s: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re =
    /(`+)([\s\S]*?[^`])\1(?!`)|!\[([^\]]*)\]\(([^)\s]+)\)|\[([^\]]+)\]\(([^)\s]+)\)|\*\*([^*]+)\*\*|__([^_]+)__|~~([^~]+)~~|(?<![\w*])\*([^*\s][^*]*?)\*(?!\w)|(?<!\w)_([^_\s][^_]*?)_(?!\w)|(https?:\/\/[^\s<>()]+[^\s<>().,;:!?'"])/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(s))) {
    if (m.index > last) out.push(<Fragment key={k++}>{withBreaks(s.slice(last, m.index))}</Fragment>);
    if (m[1]) out.push(<code key={k++}>{m[2]!.trim()}</code>);
    else if (m[4] !== undefined) {
      out.push(SAFE_IMG.test(m[4]) ? <img key={k++} src={m[4]} alt={m[3]} className="md-img" /> : <Fragment key={k++}>{m[0]}</Fragment>);
    } else if (m[6] !== undefined) out.push(LINK.test(m[6]) ? <Link key={k++} href={m[6]}>{inline(m[5]!)}</Link> : <Fragment key={k++}>{m[5]}</Fragment>);
    else if (m[7] || m[8]) out.push(<strong key={k++}>{inline(m[7] ?? m[8]!)}</strong>);
    else if (m[9]) out.push(<del key={k++}>{inline(m[9])}</del>);
    else if (m[10] || m[11]) out.push(<em key={k++}>{inline(m[10] ?? m[11]!)}</em>);
    else if (m[12]) out.push(<Link key={k++} href={m[12]}>{m[12]}</Link>);
    last = m.index + m[0].length;
  }
  if (last < s.length) out.push(<Fragment key={k++}>{withBreaks(s.slice(last))}</Fragment>);
  return out;
}

/** Links open in the built-in browser; Ctrl/Cmd-click opens the system browser. */
function Link({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a
      href={href}
      title={href}
      onClick={(e) => {
        e.preventDefault();
        if ((e.ctrlKey || e.metaKey) && /^https?:/.test(href)) window.open(href, "_blank");
        else void useApp.getState().openInBrowser(href);
      }}
    >
      {children}
    </a>
  );
}

function withBreaks(s: string): ReactNode {
  const parts = s.split("\n");
  return parts.map((p, i) => (
    <Fragment key={i}>
      {p}
      {i < parts.length - 1 && <br />}
    </Fragment>
  ));
}

// ---- code blocks ---------------------------------------------------------------------------------

const PREVIEW_KINDS: Record<string, "html" | "svg" | "canvas"> = { html: "html", htm: "html", svg: "svg", canvas: "canvas" };

function FencedBlock({ lang, code, closed, previews }: { lang: string; code: string; closed: boolean; previews: boolean }) {
  if (previews && closed && lang === "mermaid") return <MermaidBlock code={code} />;
  const kind = PREVIEW_KINDS[lang] ?? (lang === "xml" && /^\s*<svg[\s>]/.test(code) ? "svg" : isCanvasScript(lang, code) ? "canvas" : undefined);
  if (previews && closed && kind && code.trim()) return <PreviewBlock code={code} kind={kind} lang={lang} />;
  return <CodeBlock lang={lang} code={code} />;
}

/** Browser JavaScript that only draws on a 2D canvas runs as a canvas preview. */
function isCanvasScript(lang: string, code: string): boolean {
  const draws = /getContext\(\s*["'`]2d["'`]/.test(code) || /\bctx\.(beginPath|arc|fillRect|strokeRect|fill|stroke|moveTo|lineTo|fillText|drawImage|ellipse|rect)\s*\(/.test(code);
  return /^(js|javascript)$/.test(lang) && draws && !/\b(require\(|import\s|export\s|fetch\(|process\.|XMLHttpRequest)/.test(code);
}

function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      className="md-btn"
      title="Copy"
      onClick={() => {
        void navigator.clipboard.writeText(text);
        setDone(true);
        setTimeout(() => setDone(false), 1200);
      }}
    >
      {done ? <ICheck size={12} /> : <ICopy size={12} />}
      {done ? "Copied" : "Copy"}
    </button>
  );
}

function CodeBlock({ lang, code, extra }: { lang: string; code: string; extra?: ReactNode }) {
  return (
    <div className="md-code">
      <div className="md-code-head">
        <span className="mono">{lang || "text"}</span>
        <span className="flex-1" />
        {extra}
        <CopyButton text={code} />
      </div>
      <pre data-lang={lang || undefined}>
        <code>{code}</code>
      </pre>
    </div>
  );
}

function PreviewBlock({ code, kind, lang }: { code: string; kind: "html" | "svg" | "canvas"; lang: string }) {
  const [view, setView] = useState<"preview" | "code">("preview");
  const [url, setUrl] = useState<string | null>(null);
  const [height, setHeight] = useState(kind === "svg" ? 200 : 320);
  const [error, setError] = useState<string | null>(null);
  const frame = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    let alive = true;
    call<string>("preview:create", { source: code, kind }).then(
      (u) => alive && setUrl(u),
      (e: unknown) => alive && setError(String(e)),
    );
    return () => {
      alive = false;
    };
  }, [code, kind]);

  useEffect(() => {
    const onMsg = (e: MessageEvent) => {
      if (e.source !== frame.current?.contentWindow) return;
      const d = e.data as { rlmPreview?: boolean; height?: number };
      if (d?.rlmPreview && typeof d.height === "number") setHeight(Math.max(60, Math.min(640, d.height)));
    };
    window.addEventListener("message", onMsg);
    return () => window.removeEventListener("message", onMsg);
  }, []);

  const toggle = (
    <div className="md-seg">
      <button className={view === "preview" ? "on" : ""} onClick={() => setView("preview")}>
        <IGlobe size={11} /> Preview
      </button>
      <button className={view === "code" ? "on" : ""} onClick={() => setView("code")}>
        <ICode size={11} /> Code
      </button>
    </div>
  );
  const open = url && (
    <button className="md-btn" title="Open in the built-in browser" onClick={() => void useApp.getState().openInBrowser(url)}>
      <IExternal size={12} /> Open
    </button>
  );
  if (view === "code")
    return (
      <CodeBlock
        lang={lang}
        code={code}
        extra={
          <>
            {toggle}
            {open}
          </>
        }
      />
    );
  return (
    <div className="md-code" data-testid="md-preview">
      <div className="md-code-head">
        <span className="mono">{lang}</span>
        <span className="flex-1" />
        {toggle}
        {open}
        <CopyButton text={code} />
      </div>
      {error ? (
        <div className="px-3 py-2 text-[12px] text-err">{error}</div>
      ) : url ? (
        <iframe ref={frame} src={url} title={`${lang} preview`} sandbox="allow-scripts" className="md-frame" style={{ height }} />
      ) : (
        <div className="md-frame" style={{ height }} />
      )}
    </div>
  );
}

// ---- mermaid -------------------------------------------------------------------------------------

type MermaidApi = (typeof import("mermaid"))["default"];
let mermaidReady: Promise<MermaidApi> | null = null;
let mermaidTheme = "";
let mermaidSeq = 0;

function themeVars(dark: boolean): Record<string, string> {
  const ink = dark ? "#f4f4f2" : "#111111";
  const bg = dark ? "#141415" : "#ffffff";
  const soft = dark ? "#1c1c1e" : "#f2f2f0";
  return {
    background: bg,
    primaryColor: bg,
    primaryTextColor: ink,
    primaryBorderColor: ink,
    secondaryColor: soft,
    secondaryTextColor: ink,
    secondaryBorderColor: ink,
    tertiaryColor: soft,
    tertiaryTextColor: ink,
    tertiaryBorderColor: ink,
    lineColor: ink,
    textColor: ink,
    mainBkg: bg,
    nodeBorder: ink,
    clusterBkg: soft,
    clusterBorder: ink,
    titleColor: ink,
    edgeLabelBackground: bg,
    actorBkg: bg,
    actorBorder: ink,
    actorTextColor: ink,
    actorLineColor: ink,
    signalColor: ink,
    signalTextColor: ink,
    labelBoxBkgColor: bg,
    labelBoxBorderColor: ink,
    labelTextColor: ink,
    loopTextColor: ink,
    noteBkgColor: soft,
    noteTextColor: ink,
    noteBorderColor: ink,
    pie1: ink,
    pie2: dark ? "#9a9a9f" : "#6b6b70",
    pie3: dark ? "#3a3a3e" : "#bdbdb8",
    pieStrokeColor: bg,
    pieTitleTextColor: ink,
    pieLegendTextColor: ink,
    fontFamily: "Inter, system-ui, sans-serif",
  };
}

async function getMermaid(): Promise<MermaidApi> {
  const theme = document.documentElement.dataset.theme ?? "dark";
  if (!mermaidReady) mermaidReady = import("mermaid").then((m) => m.default);
  const mermaid = await mermaidReady;
  if (mermaidTheme !== theme) {
    mermaid.initialize({ startOnLoad: false, securityLevel: "strict", theme: "base", themeVariables: themeVars(theme === "dark"), flowchart: { curve: "basis" } });
    mermaidTheme = theme;
  }
  return mermaid;
}

const svgCache = new Map<string, string>();

function MermaidBlock({ code }: { code: string }) {
  const theme = useApp((s) => s.settings?.theme ?? "dark");
  const cacheKey = `${theme}\n${code}`;
  const [svg, setSvg] = useState<string | null>(svgCache.get(cacheKey) ?? null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<"diagram" | "code">("diagram");

  useEffect(() => {
    if (svgCache.has(cacheKey)) {
      setSvg(svgCache.get(cacheKey)!);
      return;
    }
    let alive = true;
    void (async () => {
      try {
        const mermaid = await getMermaid();
        const { svg } = await mermaid.render(`mmd-${++mermaidSeq}`, code);
        svgCache.set(cacheKey, svg);
        if (alive) {
          setSvg(svg);
          setError(null);
        }
      } catch (e) {
        document.querySelectorAll('[id^="dmmd-"]').forEach((n) => n.remove());
        if (alive) setError(e instanceof Error ? e.message.split("\n").slice(0, 3).join(" ") : String(e));
      }
    })();
    return () => {
      alive = false;
    };
  }, [cacheKey, code]);

  const toggle = (
    <div className="md-seg">
      <button className={view === "diagram" ? "on" : ""} onClick={() => setView("diagram")}>
        Diagram
      </button>
      <button className={view === "code" ? "on" : ""} onClick={() => setView("code")}>
        <ICode size={11} /> Code
      </button>
    </div>
  );
  if (view === "code" || error)
    return (
      <div>
        {error && <div className="mb-1 text-[11.5px] text-err">Diagram error: {error}</div>}
        <CodeBlock lang="mermaid" code={code} extra={toggle} />
      </div>
    );
  return (
    <div className="md-code" data-testid="md-mermaid">
      <div className="md-code-head">
        <span className="mono">mermaid</span>
        <span className="flex-1" />
        {toggle}
        <CopyButton text={code} />
      </div>
      {/* Mermaid output is sanitized by mermaid itself (securityLevel: strict). */}
      <div className="md-mermaid" dangerouslySetInnerHTML={{ __html: svg ?? "" }} />
    </div>
  );
}
