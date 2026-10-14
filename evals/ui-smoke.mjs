/**
 * UI smoke test for chat rendering and the built-in browser, against the built app and a real model.
 *   node evals/ui-smoke.mjs        (EVAL_API_URL / EVAL_API_MODEL as in run.mjs)
 */
import { execFileSync } from "node:child_process";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { IdeApp, REPO } from "./lib/app.mjs";
import { Cdp, sleep } from "./lib/cdp.mjs";

const API_MODEL = process.env.EVAL_API_MODEL ?? "Qwen3.5-4B";
const API_URL = process.env.EVAL_API_URL ?? "http://localhost:8100/v1";
const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
const dir = path.join(REPO, "evals/results", `ui-${stamp}`);
const ws = path.join(dir, "ws");

const PAGE = `<!doctype html>
<html><head><meta charset="utf-8"><title>Counter Demo</title>
<style>body{font-family:sans-serif;padding:40px} button{font-size:18px;padding:8px 18px;border-radius:999px;border:2px solid #111;background:#fff}</style></head>
<body><h1>Counter Demo</h1><p>Count: <span id="count">0</span></p>
<button id="inc">Increment</button> <button id="reset">Reset</button>
<script>let n=0;const c=document.getElementById('count');inc.onclick=()=>{c.textContent=++n};reset.onclick=()=>{n=0;c.textContent=0};console.log('counter ready')</script>
</body></html>`;

const settings = {
  providers: [
    { id: "builtin", kind: "local_onnx", label: "Built-in (local)", baseUrl: "", hasKey: false },
    { id: "eval-api", kind: "openai_compat", label: "Eval API", baseUrl: API_URL, hasKey: false },
  ],
  models: [
    { id: "builtin:bge-m3", providerId: "builtin", name: "bge-m3", label: "bge-m3 (local)", kind: "embed", contextLength: 8192, maxOutputTokens: null, dim: 1024 },
    { id: `eval-api:${API_MODEL}`, providerId: "eval-api", name: API_MODEL, label: API_MODEL, kind: "chat", contextLength: 32768, maxOutputTokens: 8192, dim: null },
  ],
  roles: { agent: `eval-api:${API_MODEL}`, summarizer: "", embed: "builtin:bge-m3" },
  tools: { workspaceAllow: ["*"], autoApplyEdits: true },
  autoUpdate: false,
};

const checks = [];
const check = (name, ok, detail = "") => {
  checks.push({ name, ok: !!ok, detail });
  console.log(`  ${ok ? "PASS" : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
};

/** Any CDP target (page or out-of-process iframe) whose URL contains `urlPart`. */
async function pageTarget(port, urlPart, timeoutMs = 15000) {
  const until = Date.now() + timeoutMs;
  for (;;) {
    const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    const t = list.find((p) => p.url.includes(urlPart) && p.webSocketDebuggerUrl);
    if (t) {
      const c = new Cdp(t.webSocketDebuggerUrl);
      await c.open();
      return c;
    }
    if (Date.now() > until) throw new Error(`no target ${urlPart}`);
    await sleep(300);
  }
}

async function waitFor(fn, timeoutMs = 15000, every = 250) {
  const until = Date.now() + timeoutMs;
  for (;;) {
    const v = await fn().catch(() => null);
    if (v) return v;
    if (Date.now() > until) return null;
    await sleep(every);
  }
}

rmSync(dir, { recursive: true, force: true });
mkdirSync(path.join(ws, "site"), { recursive: true });
writeFileSync(path.join(ws, "site/index.html"), PAGE);
writeFileSync(path.join(dir, "package.json"), '{"type":"commonjs"}');
execFileSync("git", ["init", "-q"], { cwd: ws });

const app = new IdeApp({ userData: path.join(dir, "userdata"), workspace: ws, settings, logFile: path.join(dir, "app.log") });
const port = app.opts.port;
try {
  await app.start();
  const cdp = app.cdp;

  console.log("▶ chat rendering");
  const r1 = await app.ask(
    "Do not run any code or tools; answer directly in markdown. Give me: (1) a markdown table of three planets with columns Planet, Moons, Rings; (2) a mermaid flowchart (```mermaid) of a login flow with 4 nodes; (3) a ```html snippet with a button that counts clicks; (4) a task list with one done and one open item; (5) a ```canvas block (plain JavaScript using the predefined `ctx`) that draws a black circle.",
  );
  check("render turn finished", r1.status === "done", r1.status);
  writeFileSync(path.join(dir, "render-answer.md"), r1.text);
  await sleep(2500);
  const dom = await cdp.eval(`(() => ({
    tables: document.querySelectorAll('.md-table table').length,
    mermaid: document.querySelectorAll('[data-testid=md-mermaid] svg').length,
    mermaidErr: [...document.querySelectorAll('.text-err')].map(e => e.innerText).join(' | ').slice(0, 300),
    previews: document.querySelectorAll('[data-testid=md-preview] iframe').length,
    previewSrc: document.querySelector('[data-testid=md-preview] iframe')?.src ?? '',
    tasks: document.querySelectorAll('.md li.task').length,
    codeHeads: document.querySelectorAll('.md-code-head').length,
    canvas: [...document.querySelectorAll('[data-testid=md-preview]')].filter(n => /^(canvas|js|javascript)$/.test(n.querySelector('.md-code-head .mono')?.textContent ?? '') && n.querySelector('iframe')).length,
  }))()`);
  check("canvas preview rendered", dom.canvas > 0);
  if (dom.canvas > 0) {
    await sleep(800);
    const targets = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).filter((t) => t.url.startsWith("rlm-preview://"));
    let drawn = null;
    for (const t of targets) {
      const c = new Cdp(t.webSocketDebuggerUrl);
      await c.open();
      const v = await c
        .eval(`(() => { const c = document.querySelector('canvas'); if (!c) return null; const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++; return { painted: n, err: document.getElementById('err')?.textContent ?? '' }; })()`)
        .catch(() => null);
      c.close();
      if (v) drawn = v;
    }
    check("canvas preview draws pixels", drawn && drawn.painted > 100 && !drawn.err, JSON.stringify(drawn));
  }
  check("markdown table rendered", dom.tables > 0, JSON.stringify(dom));
  check("mermaid diagram rendered", dom.mermaid > 0, dom.mermaidErr);
  check("html preview frame rendered", dom.previews > 0 && dom.previewSrc.startsWith("rlm-preview://"), dom.previewSrc);
  check("task list rendered", dom.tasks >= 2);
  if (dom.previews) {
    const pv = await waitFor(() => pageTarget(port, "rlm-preview://", 3000), 8000);
    if (pv) {
      await pv.send("Runtime.enable");
      const btn = await pv.eval(`(() => { const b = document.querySelector('button'); if (!b) return null; const before = document.body.innerText; b.click(); b.click(); return { before: before.slice(0, 80), after: document.body.innerText.slice(0, 80) }; })()`);
      check("preview scripts run (button reacts)", btn && btn.before !== btn.after, JSON.stringify(btn));
      const net = await pv.eval(`fetch('https://example.com').then(() => 'reached', () => 'blocked')`);
      check("preview is offline", net === "blocked", net);
      const parent = await pv.eval(`(() => { try { return String(parent.document.title); } catch { return 'isolated'; } })()`);
      check("preview cannot reach the IDE", parent === "isolated", parent);
      const frameH = await cdp.eval(`document.querySelector('[data-testid=md-preview] iframe').getBoundingClientRect().height`);
      const docH = await pv.eval(`Math.ceil(document.body.getBoundingClientRect().bottom)`);
      check("preview frame sized to its content", Math.abs(frameH - Math.max(60, Math.min(640, docH))) < 40, `frame=${frameH} content=${docH}`);
      pv.close();
    } else check("preview target found", false);
  }
  await app.screenshot(path.join(dir, "1-rendering.png"));

  console.log("▶ browser panel");
  await cdp.click('[data-testid="toggle-browser"]');
  await sleep(500);
  check("browser panel opens", await cdp.eval(`!!document.querySelector('[data-testid=browser-panel]')`));
  await cdp.typeInto("#browser-url", "site/index.html", { enter: true });
  const st = await waitFor(async () => {
    const s = await cdp.invoke("browser:state");
    return s.url.endsWith("site/index.html") && !s.loading ? s : null;
  });
  check("navigates to a project file", !!st, JSON.stringify(st));
  const page = await waitFor(() => pageTarget(port, "site/index.html", 3000), 8000);
  check("page target available", !!page);
  if (page) {
    await page.send("Runtime.enable");
    await page.screenshot(path.join(dir, "2-browser-page.png"));
  }
  await cdp.click('button[title="Screenshot to chat"]');
  const shot = await waitFor(() => cdp.eval(`document.querySelector('[data-testid=attachments]')?.innerText.includes('Screenshot') || null`));
  check("screenshot attaches to chat", !!shot);

  await cdp.click('button[title^="Pick an element"]');
  await sleep(600);
  const picking = await cdp.invoke("browser:state");
  check("picker active", picking.picking);
  if (page) {
    const rect = await page.eval(`(() => { const r = document.getElementById('inc').getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; })()`);
    await page.send("Input.dispatchMouseEvent", { type: "mouseMoved", x: rect.x, y: rect.y });
    await sleep(150);
    await page.send("Input.dispatchMouseEvent", { type: "mousePressed", x: rect.x, y: rect.y, button: "left", clickCount: 1 });
    await page.send("Input.dispatchMouseEvent", { type: "mouseReleased", x: rect.x, y: rect.y, button: "left", clickCount: 1 });
  }
  const el = await waitFor(() => cdp.eval(`(() => { const t = document.querySelector('[data-testid=attachments]')?.innerText ?? ''; return t.includes('#inc') ? t : null; })()`));
  check("picked element attaches with selector #inc", !!el, el ?? "");
  const count = page ? await page.eval(`document.getElementById('count').textContent`) : null;
  check("picking does not click the page", count === "0", count);

  await cdp.click('button[title="Inspect (DevTools)"]');
  const dev = await waitFor(async () => (await cdp.invoke("browser:state")).devtools || null, 5000);
  const devTarget = await waitFor(async () => {
    const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    return list.find((t) => t.url.startsWith("devtools://")) ?? null;
  }, 8000);
  check("devtools open docked", !!dev && !!devTarget, devTarget?.url?.slice(0, 60) ?? "");
  await sleep(1200);
  await app.screenshot(path.join(dir, "3-browser-panel.png"));
  if (devTarget) {
    const dt = await pageTarget(port, "devtools://", 3000).catch(() => null);
    await dt?.screenshot(path.join(dir, "3-devtools.png")).catch(() => undefined);
    dt?.close();
  }
  await cdp.click('button[title="Inspect (DevTools)"]');
  await sleep(500);

  console.log("▶ attachments in a message");
  const r2 = await app.ask("What does the referenced button do on this page? Answer in one sentence without tools.");
  check("attachment turn finished", r2.status === "done", r2.text.slice(0, 160));
  const ua = await cdp.eval(`(() => { const n = [...document.querySelectorAll('[data-testid=user-attachments]')].pop(); return n ? { imgs: n.querySelectorAll('img').length, text: n.innerText.slice(0, 120) } : null; })()`);
  check("user message shows attachments with images", ua && ua.imgs >= 2, JSON.stringify(ua));
  const imgOk = await cdp.eval(`Promise.all([...document.querySelectorAll('[data-testid=user-attachments] img')].map(i => i.decode().then(() => i.naturalWidth > 0, () => false))).then(a => a.length > 0 && a.every(Boolean))`);
  check("attachment images load", imgOk);
  check("model understood the element", /increment|count|increase|add/i.test(r2.text), r2.text.slice(0, 160));

  console.log("▶ agent drives the browser");
  await app.newChat();
  const r3 = await app.ask(
    "Use the built-in browser tools (browser.navigate, browser.click, browser.snapshot): open site/index.html, click the Increment button three times, then read the page and tell me the count shown.",
  );
  writeFileSync(path.join(dir, "agent-browser-turn.json"), JSON.stringify(r3, null, 2));
  const used = [...new Set(r3.events.filter((e) => e.type === "tool_call").map((e) => e.call.toolId))];
  check("agent used browser tools", used.includes("browser.navigate") && used.includes("browser.click"), used.join(", "));
  const finalCount = page ? await page.eval(`document.getElementById('count').textContent`).catch(() => null) : null;
  check("page state changed by agent clicks", finalCount === "3", `count=${finalCount}`);
  check("agent reports the count", /\b3\b|three/i.test(r3.text), r3.text.slice(0, 200));
  await app.screenshot(path.join(dir, "4-agent-browser.png"));
  await page?.screenshot(path.join(dir, "4-page-after-agent.png")).catch(() => undefined);
  page?.close();
} catch (e) {
  check("no harness error", false, String(e?.stack ?? e).slice(0, 600));
} finally {
  await app.stop();
}
const passed = checks.filter((c) => c.ok).length;
writeFileSync(path.join(dir, "summary.json"), JSON.stringify({ passed, total: checks.length, checks }, null, 2));
console.log(`\n${passed}/${checks.length} checks passed → ${dir}`);
process.exit(passed === checks.length ? 0 : 1);
