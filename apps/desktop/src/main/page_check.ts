import { BrowserWindow, session } from "electron";
import { pathToFileURL } from "node:url";

export interface PageCheckArgs {
  path: string;
  waitMs: number;
  clicks: string[];
}

export interface PageReport {
  loaded: boolean;
  loadError: string | null;
  errors: string[];
  warnings: string[];
  missingFiles: string[];
  blockedRequests: string[];
  title: string;
  text: string;
  counts: Record<string, number>;
  clicks: Array<{ selector: string; result: string; textChanged: boolean }>;
  /** Text that is nearly invisible against its background, checked after the clicks. */
  unreadableText: string[];
}

const PARTITION = "rlm-page-check";
let configured = false;

/** The checker session is in-memory and offline: workspace pages may only load local files. */
function checkerSession(): Electron.Session {
  const s = session.fromPartition(PARTITION, { cache: false });
  if (!configured) {
    configured = true;
    s.setPermissionRequestHandler((_wc, _perm, cb) => cb(false));
  }
  return s;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Page-side: visible text whose color nearly matches its background (WCAG contrast below 1.6). */
const UNREADABLE_TEXT_JS = `(() => {
  const parse = (c) => { const m = /rgba?\\(([^)]+)\\)/.exec(c); if (!m) return null; const [r, g, b, a = 1] = m[1].split(/[\\s,\\/]+/).filter(Boolean).map(Number); return { r, g, b, a }; };
  const lum = ({ r, g, b }) => { const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }; return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
  const bgOf = (el) => { for (let e = el; e; e = e.parentElement) { const s = getComputedStyle(e); if (s.backgroundImage !== "none") return null; const c = parse(s.backgroundColor); if (c && c.a > 0.5) return c; } return { r: 255, g: 255, b: 255, a: 1 }; };
  const out = [];
  for (const el of document.querySelectorAll("body *")) {
    if (out.length >= 5) break;
    if (!el.getClientRects().length || ![...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim())) continue;
    const st = getComputedStyle(el);
    if (st.visibility === "hidden" || Number(st.opacity) === 0) continue;
    const fg = parse(st.color), bg = bgOf(el);
    if (!fg || !bg || fg.a < 0.3) continue;
    const l1 = lum(fg), l2 = lum(bg);
    const ratio = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
    if (ratio < 1.6) out.push('"' + el.textContent.trim().slice(0, 30) + '" (' + el.tagName.toLowerCase() + (el.id ? "#" + el.id : "") + (el.classList[0] ? "." + el.classList[0] : "") + ") is " + st.color + " on " + "rgb(" + bg.r + ", " + bg.g + ", " + bg.b + ")");
  }
  return out;
})()`;

let queue: Promise<unknown> = Promise.resolve();

/** Loads a workspace HTML page in a hidden sandboxed window and reports what a developer would see in DevTools.
 * Checks run one at a time because they share the session's request hooks. */
export function checkPage(args: PageCheckArgs): Promise<PageReport> {
  const run = queue.then(() => checkOne(args));
  queue = run.catch(() => undefined);
  return run;
}

async function checkOne(args: PageCheckArgs): Promise<PageReport> {
  const report: PageReport = { loaded: false, loadError: null, errors: [], warnings: [], missingFiles: [], blockedRequests: [], title: "", text: "", counts: {}, clicks: [], unreadableText: [] };
  const ses = checkerSession();
  ses.webRequest.onBeforeRequest((d, cb) => {
    const ok = /^(file|data|blob|devtools|chrome-extension):/.test(d.url);
    if (!ok) report.blockedRequests.push(d.url);
    cb({ cancel: !ok });
  });
  ses.webRequest.onErrorOccurred((d) => {
    if (d.url.startsWith("file:") && /FILE_NOT_FOUND/.test(d.error)) report.missingFiles.push(decodeURIComponent(d.url.replace(/^file:\/\//, "")));
  });

  const win = new BrowserWindow({
    show: false,
    width: 1280,
    height: 860,
    webPreferences: { partition: PARTITION, sandbox: true, contextIsolation: true, nodeIntegration: false, javascript: true, webSecurity: true },
  });
  win.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  win.webContents.on("will-navigate", (e) => e.preventDefault());
  win.webContents.on("console-message", (...a: unknown[]) => {
    const ev = a[0] as { level?: string | number; message?: string; lineNumber?: number; sourceId?: string };
    const level = ev.level ?? (a[1] as number);
    const message = ev.message ?? String(a[2] ?? "");
    const line = ev.lineNumber ?? (a[3] as number);
    const src = (ev.sourceId ?? String(a[4] ?? "")).split("/").pop();
    if (/Electron Security Warning/.test(message)) return;
    const text = `${message}${src ? ` (${src}:${line})` : ""}`;
    if (level === "error" || level === 3) report.errors.push(text);
    else if (level === "warning" || level === 2) report.warnings.push(text);
  });

  const deadline = setTimeout(() => win.destroy(), 20000);
  try {
    try {
      await win.loadURL(pathToFileURL(args.path).toString());
      report.loaded = true;
    } catch (e) {
      report.loadError = e instanceof Error ? e.message : String(e);
      return report;
    }
    await sleep(args.waitMs);
    const snap = () =>
      win.webContents.executeJavaScript(
        `({ title: document.title, text: (document.body?.innerText ?? "").replace(/\\n{3,}/g, "\\n\\n").slice(0, 1200),
            counts: { buttons: document.querySelectorAll("button,[role=button]").length, links: document.querySelectorAll("a[href]").length,
                      inputs: document.querySelectorAll("input,select,textarea").length, images: document.images.length,
                      elements: document.querySelectorAll("body *").length } })`,
      ) as Promise<{ title: string; text: string; counts: Record<string, number> }>;
    let s = await snap();
    for (const selector of args.clicks.slice(0, 10)) {
      const before = s.text;
      const result = (await win.webContents.executeJavaScript(
        `(() => { const el = document.querySelector(${JSON.stringify(selector)}); if (!el) return "no element matches"; el.click(); return "clicked"; })()`,
      )) as string;
      await sleep(400);
      s = await snap();
      report.clicks.push({ selector, result, textChanged: s.text !== before });
    }
    report.title = s.title;
    report.text = s.text;
    report.counts = s.counts;
    report.unreadableText = (await win.webContents.executeJavaScript(UNREADABLE_TEXT_JS).catch(() => [])) as string[];
    return report;
  } finally {
    clearTimeout(deadline);
    if (!win.isDestroyed()) win.destroy();
    ses.webRequest.onBeforeRequest(null);
    ses.webRequest.onErrorOccurred(null);
  }
}
