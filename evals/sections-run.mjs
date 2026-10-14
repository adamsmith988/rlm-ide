/**
 * Health check of every section of the IDE, in the dark and the light theme, with screenshots.
 * Sets up real content first (a git workspace, a chat on a real model, memories, a hosted team hub
 * with a group, a shared project and a filed bug), then visits each section, checks that it shows
 * what it should, and that the renderer logged no errors or exceptions.
 *
 *   node evals/sections-run.mjs      (EVAL_API_URL / EVAL_API_MODEL; SHOTS_DIR for the screenshots)
 */
import { execFileSync } from "node:child_process";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { IdeApp, REPO } from "./lib/app.mjs";
import { sleep } from "./lib/cdp.mjs";

const API_MODEL = process.env.EVAL_API_MODEL ?? "Qwen3.5-4B";
const API_URL = process.env.EVAL_API_URL ?? "http://localhost:8100/v1";
const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
const dir = path.join(REPO, "evals/results", `sections-${stamp}`);
const shots = process.env.SHOTS_DIR ?? path.join(dir, "shots");
const ws = path.join(dir, "cart");
rmSync(dir, { recursive: true, force: true });
mkdirSync(ws, { recursive: true });
mkdirSync(shots, { recursive: true });

// A small project with history and an uncommitted change, so Source control has something to show.
const git = (...a) => execFileSync("git", a, { cwd: ws, encoding: "utf8" });
writeFileSync(path.join(ws, "README.md"), "# Cart\n\nTotals and discounts for the checkout page.\n");
mkdirSync(path.join(ws, "src"));
writeFileSync(path.join(ws, "src/cart.js"), "function total(items) {\n  return items.reduce((sum, i) => sum + i.price * i.qty, 0);\n}\n\nfunction discounted(items, rate) {\n  return total(items) * (1 - rate);\n}\n\nmodule.exports = { total, discounted };\n");
writeFileSync(path.join(ws, "src/cart.test.js"), "const { total } = require('./cart');\nconsole.assert(total([{ price: 10, qty: 2 }]) === 20);\n");
git("init", "-q", "-b", "main");
git("-c", "user.name=Alice", "-c", "user.email=alice@team.dev", "add", ".");
git("-c", "user.name=Alice", "-c", "user.email=alice@team.dev", "commit", "-q", "-m", "Cart totals and discounts");
writeFileSync(path.join(ws, "src/cart.js"), "function total(items) {\n  return items.reduce((sum, i) => sum + i.price * i.qty, 0);\n}\n\nfunction discounted(items, rate) {\n  if (rate < 0 || rate > 1) throw new RangeError('rate must be between 0 and 1');\n  return total(items) * (1 - rate);\n}\n\nmodule.exports = { total, discounted };\n");

const settings = {
  providers: [
    { id: "builtin", kind: "local_onnx", label: "Built-in (local)", baseUrl: "", hasKey: false },
    { id: "eval-api", kind: "openai_compat", label: "vLLM (local)", baseUrl: API_URL, hasKey: false },
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
const check = (group, name, ok, detail = "") => {
  checks.push({ group, name, ok: !!ok, detail: String(detail).slice(0, 300) });
  console.log(`  ${ok ? "PASS" : "FAIL"} [${group}] ${name}${detail ? ` — ${String(detail).slice(0, 200)}` : ""}`);
};
const note = (s) => console.log(`▶ ${s}`);

const app = new IdeApp({ userData: path.join(dir, "userdata"), workspace: ws, settings, logFile: path.join(dir, "app.log") });
const tid = (t) => `[data-testid="${t}"]`;
const evalJs = (js) => app.cdp.eval(js);
async function waitFor(js, ms = 15000) {
  const end = Date.now() + ms;
  for (;;) {
    const v = await evalJs(js).catch(() => null);
    if (v) return v;
    if (Date.now() > end) return null;
    await sleep(250);
  }
}
async function click(testid, ms = 10000) {
  const ok = await waitFor(`(() => { const el = document.querySelector('${tid(testid)}'); if (!el || el.disabled) return false; el.click(); return true; })()`, ms);
  if (!ok) throw new Error(`could not click ${testid}`);
}
async function type(testid, text) {
  await evalJs(`(() => { const el = document.querySelector('${tid(testid)}'); el.focus(); el.select?.(); })()`);
  await app.cdp.send("Input.insertText", { text });
}
const clickText = (selector, text) =>
  evalJs(`(() => { const el = [...document.querySelectorAll(${JSON.stringify(selector)})].find((e) => e.innerText.trim() === ${JSON.stringify(text)}); if (!el) return false; el.click(); return true; })()`);
const bodyText = () => evalJs("document.body.innerText");

async function setup() {
  note("start the IDE on a git workspace");
  await app.start();
  await app.cdp.send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });
  await sleep(800);
  const size = await evalJs("`${innerWidth}x${innerHeight}`");
  check("window", "the window is sized for screenshots", /^14\d\dx(8|9)\d\d$/.test(size), size);

  note("a real chat turn");
  const turn = await app.ask("In two short sentences: what does src/cart.js do, and what changed in the uncommitted edit?", { timeoutMs: 8 * 60_000 });
  check("chat", "a turn on the agent model finishes with an answer", turn.status === "done" && turn.text.length > 20, `${turn.status}: ${turn.text.slice(0, 140)}`);

  note("memories");
  for (const [text, category] of [
    ["Prefers pnpm and strict TypeScript.", "preference"],
    ["The checkout service deploys every Tuesday.", "fact"],
    ["Discount rates are fractions between 0 and 1, never percentages.", "fact"],
  ])
    await app.cdp.invoke("memory:add", { text, category });

  note("team hub: host, group, shared project, bug report");
  const httpPort = 20000 + Math.floor(Math.random() * 20000);
  await click("activity-team");
  await type("team-name", "Alice");
  await type("team-email", "alice@team.dev");
  await click("team-save-profile");
  await sleep(300);
  await app.cdp.invoke("team:host", { enabled: true, name: "Alice's hub", lan: false, httpPort, sshPort: httpPort + 1, discovery: false });
  const hosting = await waitFor(`document.querySelector('${tid("team-host-info")}')?.innerText`);
  check("team", "the hub runs and shows how to connect", hosting?.includes(`:${httpPort}`), hosting?.split("\n")[0]);
  await type("team-group-name", "Checkout");
  await click("team-create-group");
  await click("group-tab-projects");
  await click("group-share-workspace", 15000);
  const project = await waitFor(`document.querySelector('${tid("team-project")}')?.innerText`, 30000);
  check("team", "the open folder is shared as a group project", project?.includes("cart"), project?.split("\n")[0]);
  await click("project-report-bug");
  await waitFor(`document.querySelector('${tid("bug-dialog")}')`);
  await type("bug-title", "Negative discount rates are accepted");
  await type("bug-file", "src/cart.js");
  await type("bug-line", "5");
  await type("bug-error", "discounted([{price: 10, qty: 1}], -0.5) returned 15");
  await click("bug-submit");
  const sent = await waitFor(`document.querySelector('${tid("bug-sent")}')?.innerText`, 20000);
  check("bugs", "a bug report is filed and routed to the line's author", !!sent && /sent/i.test(sent), sent?.replace(/\s+/g, " ").slice(0, 140));
  await evalJs(`[...document.querySelectorAll('${tid("bug-dialog")} button')].find((b) => b.innerText.trim().toLowerCase() === 'done')?.click()`);
  const closed = await waitFor(`!document.querySelector('${tid("bug-dialog")}')`, 5000);
  check("bugs", "the bug dialog closes with Done", closed);
}

/** Each section: how to open it and what it must show. */
const SECTIONS = [
  {
    id: "explorer",
    open: async () => {
      await click("activity-explorer");
      await sleep(400);
      await clickText("[data-path], .tree-row, button, div", "cart.js");
    },
    ok: async () => /README\.md/.test(await bodyText()) && /cart\.js/.test(await bodyText()),
  },
  {
    id: "search",
    open: async () => {
      await click("activity-search");
      await sleep(300);
      await evalJs(`(() => { const el = document.querySelector('input[placeholder="Search in files"]'); el.focus(); el.select(); })()`);
      await app.cdp.send("Input.insertText", { text: "discounted" });
      await sleep(1200);
    },
    ok: async () => /cart\.js/.test(await evalJs("document.querySelector('nav + div, aside')?.innerText ?? document.body.innerText")),
  },
  { id: "source-control", open: () => click("activity-git"), ok: async () => /cart\.js/.test((await waitFor(`document.querySelector('${tid("git-panel")}')?.innerText`)) ?? "") },
  { id: "chats", open: () => click("activity-chats"), ok: async () => (await app.cdp.invoke("chat:list")).length > 0 && /cart/i.test(await bodyText()) },
  {
    id: "team-project",
    open: () => click("activity-team"),
    ok: async () => /cart/.test((await waitFor(`document.querySelector('${tid("team-project")}')?.innerText`)) ?? ""),
  },
  {
    id: "bug-reports",
    open: async () => {
      await click("activity-team");
      await click("project-tab-bugs").catch(() => undefined);
      await sleep(800);
    },
    ok: async () => /Negative discount rates/.test((await waitFor(`document.querySelector('${tid("bug-list")}')?.innerText`)) ?? ""),
  },
  {
    id: "models-hub",
    open: () => click("activity-hub"),
    ok: async () => {
      const t = await waitFor(`(() => { const s = document.querySelector('${tid("hub-screen")}'); const p = document.querySelector('${tid("tools-python")}'); return s && p ? p.innerText : null; })()`);
      if (!/Python/.test(t ?? "")) console.log(`  models-hub saw: ${JSON.stringify(t)} · body: ${(await bodyText()).slice(0, 200).replace(/\s+/g, " ")}`);
      return /Python/.test(t ?? "");
    },
  },
  { id: "memory", open: () => click("activity-memory"), ok: async () => /strict TypeScript/.test(await bodyText()) && /Tuesday/.test(await bodyText()) },
  { id: "settings", open: () => click("activity-settings"), ok: async () => /Settings/.test(await bodyText()) && new RegExp(API_MODEL.replace(/[.]/g, "\\.")).test(await bodyText()) },
];

async function tour(theme) {
  note(`${theme} theme`);
  await app.cdp.invoke("settings:update", { patch: { theme } });
  await sleep(800);
  const applied = await evalJs("getComputedStyle(document.body).backgroundColor");
  for (const s of SECTIONS) {
    const before = app.cdp.console.length;
    let ok = false;
    let err = "";
    try {
      await s.open();
      await sleep(900);
      ok = await s.ok();
    } catch (e) {
      err = e.message;
    }
    const crashed = /Something went wrong|Unexpected Application Error/i.test(await bodyText());
    const logged = app.cdp.console.slice(before).filter((c) => c.type !== "warning");
    check(theme, `${s.id} renders its content`, ok && !crashed, err || (crashed ? "error screen" : ""));
    check(theme, `${s.id} logs no renderer errors`, logged.length === 0, logged.map((c) => c.text).join(" | "));
    await app.screenshot(path.join(shots, `${theme}-${s.id}.png`));
  }
  return applied;
}

let fatal = null;
try {
  await setup();
  const dark = await tour("dark");
  const light = await tour("light");
  check("theme", "dark and light use different backgrounds", dark !== light, `${dark} vs ${light}`);
  await app.cdp.invoke("settings:update", { patch: { theme: "dark" } });
} catch (e) {
  fatal = e;
  console.error(e);
  await app.screenshot(path.join(dir, "fatal.png")).catch(() => undefined);
} finally {
  await app.stop();
}
const passed = checks.filter((c) => c.ok).length;
writeFileSync(path.join(dir, "results.json"), JSON.stringify({ checks, fatal: fatal ? String(fatal.stack ?? fatal) : null }, null, 2));
console.log(`\n${passed}/${checks.length} checks passed${fatal ? ` · stopped early: ${fatal.message}` : ""}\nresults: ${path.relative(REPO, dir)}\nscreenshots: ${shots}`);
process.exit(fatal || passed !== checks.length ? 1 : 0);
