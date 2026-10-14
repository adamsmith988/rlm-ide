import { protocol } from "electron";
import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

/**
 * In-memory documents for chat previews (```html / ```svg / ```canvas) served on `rlm-preview://doc/<id>`,
 * and image attachments (browser screenshots) served on `rlm-attach://img/<id>.png`.
 * Preview documents run scripts but can never reach the network or the IDE.
 */
const PREVIEW_CSP =
  "default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval' blob: data:; style-src 'unsafe-inline' data:; img-src data: blob:; font-src data:; media-src data: blob:; connect-src 'none'; form-action 'none'; base-uri 'none'";

const MAX_DOCS = 64;

/** Reports the document height to the embedding chat so the frame can size itself. */
const SIZE_REPORTER = `<script>(()=>{const measure=()=>{const html=document.documentElement,b=document.body,hs=getComputedStyle(html);let h=html.getBoundingClientRect().bottom+parseFloat(hs.marginBottom||"0");if(b){const s=getComputedStyle(b);h=Math.max(h,b.getBoundingClientRect().bottom+parseFloat(s.marginBottom||"0")+parseFloat(hs.paddingBottom||"0"));for(const el of b.querySelectorAll("*")){const r=el.getBoundingClientRect();if(r.bottom>h&&getComputedStyle(el).position!=="fixed")h=r.bottom}}return h+scrollY+2};const post=()=>{try{parent.postMessage({rlmPreview:true,height:Math.ceil(measure())},"*")}catch{}};addEventListener("load",post);const ro=new ResizeObserver(post);ro.observe(document.documentElement);document.body&&ro.observe(document.body);new MutationObserver(post).observe(document.documentElement,{subtree:true,childList:true,characterData:true});setTimeout(post,50)})()</script>`;

export class PreviewService {
  private docs = new Map<string, string>();

  constructor(private attachDir: string) {}

  static registerSchemes(): Array<Electron.CustomScheme> {
    return [
      { scheme: "rlm-preview", privileges: { standard: true, secure: true } },
      { scheme: "rlm-attach", privileges: { standard: true, secure: true, supportFetchAPI: true } },
    ];
  }

  private registered = new WeakSet<Electron.Protocol>();

  register(): void {
    this.registerOn(protocol);
  }

  /** Protocol handlers are per session; the in-app browser has its own. */
  registerOn(proto: Electron.Protocol): void {
    if (this.registered.has(proto)) return;
    this.registered.add(proto);
    proto.handle("rlm-preview", (req) => {
      const id = new URL(req.url).pathname.replace(/^\/+/, "");
      const html = this.docs.get(id);
      if (html === undefined) return new Response("preview expired", { status: 404 });
      return new Response(html, {
        status: 200,
        headers: { "Content-Type": "text/html; charset=utf-8", "Content-Security-Policy": PREVIEW_CSP, "X-Content-Type-Options": "nosniff" },
      });
    });
    proto.handle("rlm-attach", (req) => {
      const name = path.basename(new URL(req.url).pathname);
      const file = path.join(this.attachDir, name);
      if (!/^[\w-]+\.png$/.test(name) || !existsSync(file)) return new Response("not found", { status: 404 });
      return new Response(readFileSync(file), { status: 200, headers: { "Content-Type": "image/png" } });
    });
  }

  /** Stores a preview document and returns its URL. `kind` wraps svg/canvas snippets into a page. */
  create(source: string, kind: "html" | "svg" | "canvas" = "html"): string {
    const id = randomUUID();
    this.docs.set(id, wrap(source, kind));
    while (this.docs.size > MAX_DOCS) this.docs.delete(this.docs.keys().next().value!);
    return `rlm-preview://doc/${id}`;
  }

  /** Saves a PNG and returns its attachment URL. */
  saveImage(png: Buffer): { url: string; file: string } {
    mkdirSync(this.attachDir, { recursive: true });
    const name = `${randomUUID()}.png`;
    const file = path.join(this.attachDir, name);
    writeFileSync(file, png);
    return { url: `rlm-attach://img/${name}`, file };
  }

  attachmentPath(url: string): string | null {
    const m = /^rlm-attach:\/\/img\/([\w-]+\.png)$/.exec(url);
    return m ? path.join(this.attachDir, m[1]!) : null;
  }
}

const BASE_STYLE = `<style>html{margin:0;padding:0;background:#fff}body{margin:0;padding:12px;color:#111;font:14px/1.5 system-ui,sans-serif}</style>`;

function wrap(source: string, kind: "html" | "svg" | "canvas"): string {
  if (kind === "svg") return `<!doctype html><meta charset="utf-8">${BASE_STYLE}<style>svg{max-width:100%;height:auto}</style>${source}${SIZE_REPORTER}`;
  if (kind === "canvas") {
    return `<!doctype html><meta charset="utf-8">${BASE_STYLE}<style>canvas{max-width:100%;border:1px solid #ddd;border-radius:8px}</style>
<canvas id="canvas" width="640" height="360"></canvas>
<pre id="err" style="color:#b00;white-space:pre-wrap"></pre>
<script>var canvas=document.getElementById("canvas");var ctx=canvas.getContext("2d");
(()=>{const byId=document.getElementById.bind(document);document.getElementById=(id)=>byId(id)||(/canvas|board|stage|chart|game/i.test(String(id))?canvas:null)})();
addEventListener("error",e=>{document.getElementById("err").textContent=String(e.message)});</script>
<script>{
${source.replace(/<\/script/gi, "<\\/script")}
}</script>${SIZE_REPORTER}`;
  }
  const full = /<html[\s>]/i.test(source) || /<!doctype/i.test(source);
  if (!full) return `<!doctype html><meta charset="utf-8">${BASE_STYLE}${source}${SIZE_REPORTER}`;
  return /<\/body>/i.test(source) ? source.replace(/<\/body>/i, `${SIZE_REPORTER}</body>`) : source + SIZE_REPORTER;
}
