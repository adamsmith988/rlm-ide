import { BrowserWindow, session, shell, WebContentsView } from "electron";
import { existsSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import type { BrowserElementRef as ElementRef, BrowserState } from "@rlm/contracts";
import type { PreviewService } from "./preview_service";

export const BROWSER_PARTITION = "persist:rlm-browser";

export interface Bounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Runs in an isolated world: page scripts cannot see or tamper with it. */
const HELPERS = `
window.__rlm = window.__rlm || (() => {
  const cssEsc = (s) => (window.CSS && CSS.escape ? CSS.escape(s) : s.replace(/[^\\w-]/g, "\\\\$&"));
  const unique = (sel) => { try { return document.querySelectorAll(sel).length === 1; } catch { return false; } };
  const selectorFor = (el) => {
    if (!(el instanceof Element)) return "";
    if (el.id && unique("#" + cssEsc(el.id))) return "#" + cssEsc(el.id);
    for (const a of ["data-testid", "data-test", "name", "aria-label"]) {
      const v = el.getAttribute(a);
      if (v) { const s = el.tagName.toLowerCase() + "[" + a + "=" + JSON.stringify(v) + "]"; if (unique(s)) return s; }
    }
    const parts = [];
    for (let e = el; e && e.nodeType === 1 && e !== document.documentElement; e = e.parentElement) {
      let part = e.tagName.toLowerCase();
      if (e.id && unique("#" + cssEsc(e.id))) { parts.unshift("#" + cssEsc(e.id)); break; }
      const cls = [...e.classList].filter((c) => /^[a-zA-Z][\\w-]*$/.test(c)).slice(0, 2);
      if (cls.length) part += "." + cls.map(cssEsc).join(".");
      const sib = e.parentElement ? [...e.parentElement.children].filter((c) => c.tagName === e.tagName) : [];
      if (sib.length > 1) part += ":nth-of-type(" + (sib.indexOf(e) + 1) + ")";
      parts.unshift(part);
      if (unique(parts.join(" > "))) break;
    }
    return parts.join(" > ");
  };
  const visible = (el) => { const r = el.getBoundingClientRect(); if (r.width < 1 || r.height < 1) return false; const s = getComputedStyle(el); return s.visibility !== "hidden" && s.display !== "none" && Number(s.opacity) > 0.05; };
  const nameOf = (el) => (el.getAttribute("aria-label") || el.getAttribute("title") || el.getAttribute("placeholder") || el.innerText || el.value || el.getAttribute("alt") || "").replace(/\\s+/g, " ").trim().slice(0, 80);
  const describe = (el) => {
    const r = el.getBoundingClientRect();
    return { selector: selectorFor(el), tag: el.tagName.toLowerCase(), text: (el.innerText || el.textContent || "").replace(/\\s+/g, " ").trim().slice(0, 300),
      html: el.outerHTML.slice(0, 2000), rect: { x: Math.round(r.left), y: Math.round(r.top), width: Math.round(r.width), height: Math.round(r.height) } };
  };
  let refs = [];
  const INTERACTIVE = "a[href],button,input,select,textarea,summary,[role=button],[role=link],[role=tab],[role=checkbox],[role=menuitem],[onclick],[contenteditable=''],[contenteditable=true],[tabindex]:not([tabindex='-1'])";
  const snapshot = (maxText) => {
    refs = [...document.querySelectorAll(INTERACTIVE)].filter(visible).slice(0, 150);
    const elements = refs.map((el, i) => {
      const role = el.getAttribute("role") || el.tagName.toLowerCase();
      const type = el.tagName === "INPUT" ? "[" + (el.type || "text") + "]" : "";
      const val = "value" in el && el.value && el.tagName !== "BUTTON" ? " value=" + JSON.stringify(String(el.value).slice(0, 40)) : "";
      const state = (el.disabled ? " disabled" : "") + (el.checked ? " checked" : "") + (el.getAttribute("aria-selected") === "true" || el.getAttribute("aria-pressed") === "true" ? " selected" : "");
      return "e" + (i + 1) + " " + role + type + " " + JSON.stringify(nameOf(el)) + val + state;
    });
    const headings = [...document.querySelectorAll("h1,h2,h3")].filter(visible).slice(0, 20).map((h) => h.tagName.toLowerCase() + ": " + h.innerText.trim().slice(0, 80));
    return { url: location.href, title: document.title, headings, elements, text: (document.body ? document.body.innerText : "").replace(/\\n{3,}/g, "\\n\\n").slice(0, maxText),
      scroll: { y: Math.round(scrollY), height: document.documentElement.scrollHeight, viewport: innerHeight } };
  };
  const resolve = (target) => {
    if (/^e\\d+$/.test(target)) { const el = refs[Number(target.slice(1)) - 1]; if (el && el.isConnected) return el; throw new Error("ref " + target + " is stale; call browser.snapshot() again"); }
    const el = document.querySelector(target);
    if (!el) throw new Error("no element matches " + target);
    return el;
  };
  const point = (target) => {
    const el = resolve(target);
    el.scrollIntoView({ block: "center", inline: "center" });
    const r = el.getBoundingClientRect();
    return { x: Math.round(r.left + r.width / 2), y: Math.round(r.top + r.height / 2), name: nameOf(el), tag: el.tagName.toLowerCase() };
  };
  const focus = (target, clear) => {
    const el = resolve(target);
    el.scrollIntoView({ block: "center" });
    el.focus();
    if (clear && "value" in el) { el.value = ""; el.dispatchEvent(new Event("input", { bubbles: true })); }
    if (clear && el.isContentEditable) el.textContent = "";
    return { tag: el.tagName.toLowerCase(), name: nameOf(el) };
  };
  const pick = () => new Promise((done) => {
    const box = document.createElement("div");
    const label = document.createElement("div");
    box.style.cssText = "position:fixed;z-index:2147483647;pointer-events:none;border:2px solid #000;background:rgba(0,0,0,.08);box-shadow:0 0 0 1px #fff;border-radius:3px;transition:all 60ms";
    label.style.cssText = "position:fixed;z-index:2147483647;pointer-events:none;background:#000;color:#fff;font:11px/1.4 ui-monospace,monospace;padding:2px 6px;border-radius:10px;max-width:60vw;overflow:hidden;white-space:nowrap;text-overflow:ellipsis";
    document.documentElement.append(box, label);
    let current = null;
    const move = (e) => {
      const el = document.elementFromPoint(e.clientX, e.clientY);
      if (!el || el === current) return;
      current = el;
      const r = el.getBoundingClientRect();
      Object.assign(box.style, { left: r.left + "px", top: r.top + "px", width: r.width + "px", height: r.height + "px" });
      label.textContent = selectorFor(el) + "  " + Math.round(r.width) + "×" + Math.round(r.height);
      Object.assign(label.style, { left: Math.max(0, r.left) + "px", top: Math.max(0, r.top - 20) + "px" });
    };
    const stop = (v) => { removeEventListener("mousemove", move, true); removeEventListener("click", click, true); removeEventListener("keydown", key, true); removeEventListener("mousedown", block, true); removeEventListener("mouseup", block, true); box.remove(); label.remove(); done(v); };
    const block = (e) => { e.preventDefault(); e.stopPropagation(); };
    const click = (e) => { block(e); const el = document.elementFromPoint(e.clientX, e.clientY); stop(el ? describe(el) : null); };
    const key = (e) => { if (e.key === "Escape") { block(e); stop(null); } };
    addEventListener("mousemove", move, true);
    addEventListener("mousedown", block, true);
    addEventListener("mouseup", block, true);
    addEventListener("click", click, true);
    addEventListener("keydown", key, true);
    window.__rlmCancelPick = () => stop(null);
  });
  return { snapshot, point, focus, pick, describe, resolve };
})();
`;

const WORLD = 1077;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** The in-app browser: a WebContentsView laid over the panel area the renderer reserves for it. */
export class BrowserService {
  private view: WebContentsView | null = null;
  private devView: WebContentsView | null = null;
  private bounds: Bounds = { x: 0, y: 0, width: 0, height: 0 };
  private visible = false;
  private console: Array<{ level: string; text: string }> = [];
  private st: BrowserState = { open: false, url: "", title: "", loading: false, canGoBack: false, canGoForward: false, devtools: false, picking: false, agent: null, error: null };

  constructor(
    private win: () => BrowserWindow | null,
    private preview: PreviewService,
    private emit: (state: BrowserState) => void,
    private workspaceRoot: () => string | null,
  ) {}

  state(): BrowserState {
    return { ...this.st };
  }

  private update(p: Partial<BrowserState>): void {
    const wc = this.view?.webContents;
    this.st = {
      ...this.st,
      ...(wc && !wc.isDestroyed()
        ? { url: wc.getURL(), title: wc.getTitle(), loading: wc.isLoading(), canGoBack: wc.navigationHistory.canGoBack(), canGoForward: wc.navigationHistory.canGoForward() }
        : {}),
      ...p,
    };
    this.emit(this.state());
  }

  private ensure(): WebContentsView {
    if (this.view && !this.view.webContents.isDestroyed()) return this.view;
    const ses = session.fromPartition(BROWSER_PARTITION);
    this.preview.registerOn(ses.protocol);
    ses.setPermissionRequestHandler((_wc, perm, cb) => cb(perm === "clipboard-sanitized-write" || perm === "fullscreen"));
    const view = new WebContentsView({
      webPreferences: { partition: BROWSER_PARTITION, sandbox: true, contextIsolation: true, nodeIntegration: false, webSecurity: true, spellcheck: false },
    });
    view.setBackgroundColor("#ffffff");
    const wc = view.webContents;
    wc.setWindowOpenHandler(({ url }) => {
      if (/^(https?|file|rlm-preview):/.test(url)) void this.navigate(url);
      return { action: "deny" };
    });
    const refresh = () => this.update({});
    wc.on("did-start-loading", () => this.update({ loading: true, error: null }));
    wc.on("did-stop-loading", refresh);
    wc.on("did-navigate", () => {
      this.console = [];
      refresh();
    });
    wc.on("did-navigate-in-page", refresh);
    wc.on("page-title-updated", refresh);
    wc.on("did-fail-load", (_e, code, desc, url, isMain) => {
      if (isMain && code !== -3) this.update({ error: `${desc} (${code}) loading ${url}` });
    });
    wc.on("devtools-closed", () => this.closeDevtoolsView());
    wc.on("console-message", (...a: unknown[]) => {
      const ev = a[0] as { level?: string | number; message?: string; lineNumber?: number; sourceId?: string };
      const level = String(ev.level ?? a[1] ?? "info");
      const message = ev.message ?? String(a[2] ?? "");
      if (/Electron Security Warning/.test(message)) return;
      const src = (ev.sourceId ?? String(a[4] ?? "")).split("/").pop();
      this.console.push({ level: level === "3" ? "error" : level === "2" ? "warning" : level, text: `${message}${src ? ` (${src}:${ev.lineNumber ?? a[3]})` : ""}` });
      if (this.console.length > 200) this.console.splice(0, this.console.length - 200);
    });
    wc.on("render-process-gone", (_e, d) => this.update({ error: `page crashed (${d.reason})` }));
    this.view = view;
    return view;
  }

  /** Accepts URLs, workspace-relative paths, host names and search words. */
  resolveUrl(input: string): string {
    const raw = input.trim();
    if (/^(https?|file|rlm-preview|about):/i.test(raw)) return raw;
    const root = this.workspaceRoot();
    if (root && !/\s/.test(raw)) {
      const abs = path.resolve(root, raw);
      if (abs.startsWith(path.resolve(root)) && existsSync(abs)) return pathToFileURL(abs).toString();
    }
    if (path.isAbsolute(raw) && existsSync(raw)) return pathToFileURL(raw).toString();
    if (/^(localhost|127\.0\.0\.1|\[::1\])(:\d+)?(\/|$)/.test(raw)) return `http://${raw}`;
    if (/^[\w-]+(\.[\w-]+)+(:\d+)?(\/\S*)?$/.test(raw)) return `https://${raw}`;
    return `https://duckduckgo.com/?q=${encodeURIComponent(raw)}`;
  }

  async navigate(input: string): Promise<BrowserState> {
    const url = this.resolveUrl(input);
    const view = this.ensure();
    this.update({ open: true, error: null });
    this.layout();
    try {
      await view.webContents.loadURL(url);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (!/ERR_ABORTED/.test(msg)) this.update({ error: msg });
    }
    this.update({});
    return this.state();
  }

  back(): void {
    this.view?.webContents.navigationHistory.goBack();
  }
  forward(): void {
    this.view?.webContents.navigationHistory.goForward();
  }
  reload(): void {
    this.view?.webContents.reload();
  }
  stop(): void {
    this.view?.webContents.stop();
  }

  /** The renderer reports where the page area is; the view follows it. */
  setBounds(b: Bounds, visible: boolean): void {
    this.bounds = { x: Math.round(b.x), y: Math.round(b.y), width: Math.max(0, Math.round(b.width)), height: Math.max(0, Math.round(b.height)) };
    this.visible = visible;
    this.layout();
  }

  private layout(): void {
    const win = this.win();
    if (!win || !this.view) return;
    const show = this.visible && this.st.open && this.bounds.width > 20 && this.bounds.height > 20;
    const children = win.contentView.children;
    for (const v of [this.view, this.devView]) {
      if (!v) continue;
      const attached = children.includes(v);
      if (show && !attached) win.contentView.addChildView(v);
      if (!show && attached) win.contentView.removeChildView(v);
    }
    if (!show) return;
    const { x, y, width, height } = this.bounds;
    if (this.devView) {
      const pageH = Math.round(height * 0.55);
      this.view.setBounds({ x, y, width, height: pageH });
      this.devView.setBounds({ x, y: y + pageH + 1, width, height: height - pageH - 1 });
    } else this.view.setBounds({ x, y, width, height });
  }

  toggleDevtools(): void {
    const wc = this.view?.webContents;
    if (!wc) return;
    if (this.devView) {
      wc.closeDevTools();
      this.closeDevtoolsView();
      return;
    }
    const dev = new WebContentsView();
    this.devView = dev;
    wc.setDevToolsWebContents(dev.webContents);
    wc.openDevTools({ mode: "detach", activate: false });
    this.update({ devtools: true });
    this.layout();
  }

  private closeDevtoolsView(): void {
    const dev = this.devView;
    if (!dev) return;
    this.devView = null;
    const win = this.win();
    if (win && win.contentView.children.includes(dev)) win.contentView.removeChildView(dev);
    if (!dev.webContents.isDestroyed()) dev.webContents.close();
    this.update({ devtools: false });
    this.layout();
  }

  close(): void {
    this.closeDevtoolsView();
    this.update({ open: false, picking: false });
    this.layout();
  }

  openExternal(): void {
    if (/^https?:/.test(this.st.url)) void shell.openExternal(this.st.url);
  }

  private async run<T>(code: string): Promise<T> {
    const wc = this.ensure().webContents;
    if (!this.st.open || !wc.getURL()) throw new Error("the browser has no page open; call browser.navigate(url) first");
    return (await wc.executeJavaScriptInIsolatedWorld(WORLD, [{ code: `${HELPERS};${code}` }], true)) as T;
  }

  async screenshot(rect?: Bounds): Promise<{ url: string; file: string; width: number; height: number; pageUrl: string; title: string }> {
    const wc = this.ensure().webContents;
    if (!this.st.open) throw new Error("the browser has no page open");
    const img = await wc.capturePage(rect && rect.width > 0 && rect.height > 0 ? rect : undefined);
    const size = img.getSize();
    const saved = this.preview.saveImage(img.toPNG());
    return { ...saved, width: size.width, height: size.height, pageUrl: wc.getURL(), title: wc.getTitle() };
  }

  /** Lets the user click an element; returns its selector, markup and a cropped screenshot. */
  async pickElement(): Promise<ElementRef | null> {
    const wc = this.ensure().webContents;
    if (!this.st.open) throw new Error("open a page first");
    this.update({ picking: true });
    wc.focus();
    try {
      const el = await this.run<Omit<ElementRef, "url" | "title" | "image"> | null>("window.__rlm.pick()");
      if (!el) return null;
      await sleep(60);
      const pad = 6;
      const crop = { x: Math.max(0, el.rect.x - pad), y: Math.max(0, el.rect.y - pad), width: el.rect.width + pad * 2, height: el.rect.height + pad * 2 };
      const shot = await this.screenshot(crop).catch(() => null);
      return { ...el, url: wc.getURL(), title: wc.getTitle(), image: shot?.url ?? null };
    } finally {
      this.update({ picking: false });
    }
  }

  cancelPick(): void {
    void this.run("window.__rlmCancelPick && window.__rlmCancelPick()").catch(() => undefined);
  }

  // ---- agent control -------------------------------------------------------------------------

  private async agentStep<T>(label: string, fn: () => Promise<T>): Promise<T> {
    this.update({ agent: label });
    try {
      return await fn();
    } finally {
      this.update({ agent: null });
    }
  }

  private async settle(ms = 500): Promise<void> {
    const wc = this.view?.webContents;
    await sleep(ms);
    for (let i = 0; i < 20 && wc?.isLoading(); i++) await sleep(250);
  }

  private consoleSince(n: number): string[] {
    return this.console.slice(n).filter((c) => c.level === "error" || c.level === "warning").map((c) => `${c.level}: ${c.text}`).slice(-10);
  }

  async agent(op: string, a: Record<string, unknown>): Promise<unknown> {
    switch (op) {
      case "navigate":
        return this.agentStep(`Opening ${String(a.url)}`, async () => {
          const before = this.console.length;
          const st = await this.navigate(String(a.url));
          await this.settle(Number(a.wait_ms ?? 600));
          const snap = await this.run<Record<string, unknown>>(`window.__rlm.snapshot(1500)`).catch(() => null);
          return { url: st.url, title: this.view?.webContents.getTitle() ?? "", error: st.error, console: this.consoleSince(before), ...(snap ? { text: snap.text, elements: snap.elements } : {}) };
        });
      case "snapshot":
        return this.agentStep("Reading the page", async () => ({
          ...(await this.run<Record<string, unknown>>(`window.__rlm.snapshot(${Math.min(20000, Number(a.max_text ?? 4000))})`)),
          console: this.consoleSince(Math.max(0, this.console.length - 20)),
        }));
      case "click":
        return this.agentStep(`Clicking ${String(a.target)}`, async () => {
          const before = this.console.length;
          const urlBefore = this.st.url;
          const p = await this.run<{ x: number; y: number; name: string; tag: string }>(`window.__rlm.point(${JSON.stringify(String(a.target))})`);
          const wc = this.view!.webContents;
          await sleep(80);
          wc.sendInputEvent({ type: "mouseMove", x: p.x, y: p.y });
          wc.sendInputEvent({ type: "mouseDown", x: p.x, y: p.y, button: "left", clickCount: 1 });
          wc.sendInputEvent({ type: "mouseUp", x: p.x, y: p.y, button: "left", clickCount: 1 });
          await this.settle();
          return { clicked: `${p.tag} "${p.name}"`, url: wc.getURL(), navigated: wc.getURL() !== urlBefore, console: this.consoleSince(before) };
        });
      case "type":
        return this.agentStep(`Typing into ${String(a.target)}`, async () => {
          const before = this.console.length;
          const wc = this.view!.webContents;
          const f = await this.run<{ tag: string; name: string }>(`window.__rlm.focus(${JSON.stringify(String(a.target))}, ${a.clear !== false})`);
          wc.focus();
          await wc.insertText(String(a.text ?? ""));
          if (a.submit) {
            wc.sendInputEvent({ type: "keyDown", keyCode: "Enter" });
            wc.sendInputEvent({ type: "char", keyCode: "\r" });
            wc.sendInputEvent({ type: "keyUp", keyCode: "Enter" });
          }
          await this.settle(a.submit ? 700 : 200);
          return { typed_into: `${f.tag} "${f.name}"`, url: wc.getURL(), console: this.consoleSince(before) };
        });
      case "press":
        return this.agentStep(`Pressing ${String(a.key)}`, async () => {
          const wc = this.ensure().webContents;
          const key = String(a.key);
          wc.sendInputEvent({ type: "keyDown", keyCode: key });
          if (key.length === 1) wc.sendInputEvent({ type: "char", keyCode: key });
          wc.sendInputEvent({ type: "keyUp", keyCode: key });
          await this.settle(300);
          return { pressed: key, url: wc.getURL() };
        });
      case "scroll":
        return this.agentStep("Scrolling", async () => {
          const dy = Number(a.dy ?? 600);
          return this.run(`(scrollBy(0, ${dy}), { y: Math.round(scrollY), height: document.documentElement.scrollHeight })`);
        });
      case "eval":
        return this.agentStep("Running script in the page", async () => {
          const wc = this.ensure().webContents;
          const code = String(a.code);
          const v = (await wc.executeJavaScript(`(async () => { const __r = await (async () => (${code}))(); try { return JSON.parse(JSON.stringify(__r ?? null)); } catch { return String(__r); } })()`, true)) as unknown;
          const s = JSON.stringify(v);
          return s && s.length > 8000 ? s.slice(0, 8000) + "…[truncated]" : v;
        });
      case "screenshot":
        return this.agentStep("Taking a screenshot", async () => {
          const r = await this.screenshot();
          return { image: r.url, width: r.width, height: r.height, url: r.pageUrl, title: r.title };
        });
      case "console":
        return this.console.slice(-Number(a.limit ?? 40)).map((c) => `${c.level}: ${c.text}`);
      default:
        throw new Error(`unknown browser op ${op}`);
    }
  }
}
