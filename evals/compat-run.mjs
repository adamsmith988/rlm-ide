/**
 * API-key compatibility eval. Brings up several model servers on different runtimes, each
 * protected by its own API key, then drives the built IDE like a user (Settings → Models →
 * providers, keys, models, Detect, Test, roles, chat) and checks that the key system behaves:
 * right key works, wrong/missing key gives a clear error, every server only ever sees its own
 * key, rotation and removal work, and no key leaks into settings, logs, the database or the UI.
 *
 * Nothing here is known to the agent: the IDE gets no special settings beyond an empty profile,
 * and every provider/model/key is entered through the Settings UI.
 *
 *   node evals/compat-run.mjs
 *   env: EVAL_CONTROL_URL (keyless vLLM, default http://127.0.0.1:8100/v1), EVAL_CONTROL_MODEL
 */
import { execFileSync, spawn } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, openSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { IdeApp, REPO } from "./lib/app.mjs";
import { sleep } from "./lib/cdp.mjs";

const CONTROL_URL = process.env.EVAL_CONTROL_URL ?? "http://127.0.0.1:8100/v1";
const CONTROL_MODEL = process.env.EVAL_CONTROL_MODEL ?? "Qwen3.5-4B";
const RT = path.join(REPO, "evals/.runtimes");
const PY = path.join(RT, "venv/bin/python");
const GGUF = path.join(RT, "models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
const GATEWAY = path.join(REPO, "evals/compat/key_gateway.py");
const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
const dir = path.join(REPO, "evals/results", `compat-${stamp}`);
const ws = path.join(dir, "ws");
const userData = path.join(dir, "userdata");
rmSync(dir, { recursive: true, force: true });
mkdirSync(ws, { recursive: true });

const newKey = (tag) => `sk-${tag}-${randomBytes(18).toString("hex")}`;
const fp = (k) => (k ? createHash("sha256").update(k).digest("hex").slice(0, 12) : "");

const checks = [];
const check = (group, name, ok, detail = "") => {
  checks.push({ group, name, ok: !!ok, detail: String(detail).slice(0, 400) });
  console.log(`  ${ok ? "PASS" : "FAIL"} [${group}] ${name}${detail ? ` — ${String(detail).slice(0, 200)}` : ""}`);
};
const note = (s) => {
  console.log(`▶ ${s}`);
  appendFileSync(path.join(dir, "progress.log"), `${new Date().toISOString()} ${s}\n`);
};

// ---- runtimes -----------------------------------------------------------------------------------

const procs = [];
function start(name, cmd, args, env = {}) {
  const log = openSync(path.join(dir, `${name}.log`), "a");
  const p = spawn(cmd, args, { stdio: ["ignore", log, log], env: { ...process.env, ...env } });
  procs.push({ name, p });
  return p;
}

async function stopAll() {
  for (const { p } of procs.reverse()) if (p.exitCode === null) p.kill("SIGTERM");
  const until = Date.now() + 30000;
  while (procs.some(({ p }) => p.exitCode === null) && Date.now() < until) await sleep(500);
  for (const { p } of procs) if (p.exitCode === null) p.kill("SIGKILL");
}

async function waitHttp(url, key, timeoutMs, name) {
  const until = Date.now() + timeoutMs;
  for (;;) {
    try {
      const r = await fetch(url, { headers: key ? { Authorization: `Bearer ${key}` } : {}, signal: AbortSignal.timeout(3000) });
      if (r.status === 200) return true;
    } catch {
      /* not up */
    }
    const dead = procs.find((x) => x.name === name && x.p.exitCode !== null);
    if (dead) throw new Error(`${name} exited (${dead.p.exitCode}); see ${name}.log`);
    if (Date.now() > until) throw new Error(`${name} not ready at ${url}`);
    await sleep(1000);
  }
}

const gatewayLog = (name) => path.join(dir, `${name}-requests.jsonl`);
function gateway(name, port, upstream, { keyFile, tap }) {
  const args = [GATEWAY, "--port", String(port), "--upstream", upstream, "--log", gatewayLog(name)];
  if (keyFile) args.push("--key-file", keyFile);
  if (tap) args.push("--tap");
  return start(`${name}-gw`, "python3", args);
}
const requests = (name) =>
  existsSync(gatewayLog(name))
    ? readFileSync(gatewayLog(name), "utf8")
        .split("\n")
        .filter(Boolean)
        .map((l) => JSON.parse(l))
    : [];

/** Each runtime: the server's own key and the URL the IDE is told to use (a logging tap or gate). */
const R = {
  vllm: { label: "vLLM SmolLM2", key: newKey("vllm"), port: 8201, upstreamPort: 8211, model: "smollm2-135m", kind: "chat" },
  llama: { label: "llama.cpp Qwen2.5", key: newKey("llama"), port: 8202, upstreamPort: 8212, model: "qwen2.5-0.5b-gguf", kind: "chat", ctx: 4096 },
  ollama: { label: "Ollama gated", key: newKey("ollama"), port: 8203, model: "compat-qwen2.5-0.5b-cpu", embed: "nomic-embed-text", kind: "chat" },
  control: { label: "Agent vLLM gated", key: newKey("agent"), port: 8204, model: CONTROL_MODEL, kind: "chat" },
};
const WRONG = newKey("wrong");
const ROTATED = newKey("ollama2");
const allKeys = () => [R.vllm.key, R.llama.key, R.ollama.key, R.control.key, WRONG, ROTATED];
for (const r of Object.values(R)) r.base = `http://127.0.0.1:${r.port}/v1`;

async function bringUp() {
  note("starting runtimes");
  start("vllm", "vllm", ["serve", "HuggingFaceTB/SmolLM2-135M-Instruct", "--served-model-name", R.vllm.model, "--api-key", R.vllm.key, "--host", "127.0.0.1", "--port", String(R.vllm.upstreamPort), "--gpu-memory-utilization", process.env.EVAL_VLLM_GPU_UTIL ?? "0.04", "--max-model-len", "2048", "--max-num-seqs", "4", "--enforce-eager"]);
  gateway("vllm", R.vllm.port, `http://127.0.0.1:${R.vllm.upstreamPort}`, { tap: true });

  start("llama", PY, ["-m", "llama_cpp.server", "--model", GGUF, "--model_alias", R.llama.model, "--api_key", R.llama.key, "--host", "127.0.0.1", "--port", String(R.llama.upstreamPort), "--n_ctx", "4096", "--n_gpu_layers", "0"]);
  gateway("llama", R.llama.port, `http://127.0.0.1:${R.llama.upstreamPort}`, { tap: true });

  // The GPU is mostly taken by the agent's vLLM; pin the small Ollama chat model to the CPU.
  const modelfile = path.join(dir, "Modelfile");
  writeFileSync(modelfile, "FROM qwen2.5:0.5b\nPARAMETER num_gpu 0\n");
  execFileSync("ollama", ["create", R.ollama.model, "-f", modelfile], { stdio: "ignore" });

  R.ollama.keyFile = path.join(dir, "ollama-keys.txt");
  writeFileSync(R.ollama.keyFile, `${R.ollama.key}\n`, { mode: 0o600 });
  gateway("ollama", R.ollama.port, "http://127.0.0.1:11434", { keyFile: R.ollama.keyFile });

  R.control.keyFile = path.join(dir, "agent-keys.txt");
  writeFileSync(R.control.keyFile, `${R.control.key}\n`, { mode: 0o600 });
  gateway("control", R.control.port, CONTROL_URL.replace(/\/v1\/?$/, ""), { keyFile: R.control.keyFile });

  await waitHttp(`http://127.0.0.1:${R.llama.upstreamPort}/v1/models`, R.llama.key, 120000, "llama");
  await waitHttp(`${R.ollama.base}/models`, R.ollama.key, 30000, "ollama-gw");
  await waitHttp(`${R.control.base}/models`, R.control.key, 30000, "control-gw");
  await waitHttp(`http://127.0.0.1:${R.vllm.upstreamPort}/v1/models`, R.vllm.key, 420000, "vllm");
  await waitHttp(`${R.vllm.base}/models`, R.vllm.key, 10000, "vllm-gw");
  await waitHttp(`${R.llama.base}/models`, R.llama.key, 10000, "llama-gw");

  // The servers themselves must enforce their keys, or the IDE checks below prove nothing.
  for (const [name, r] of Object.entries(R)) {
    const none = await fetch(`${r.base}/models`).then((x) => x.status);
    const wrong = await fetch(`${r.base}/models`, { headers: { Authorization: `Bearer ${WRONG}` } }).then((x) => x.status);
    check("servers", `${name} rejects missing and wrong keys`, none === 401 && wrong === 401, `no key → ${none}, wrong key → ${wrong}`);
  }
}

// ---- UI driving ---------------------------------------------------------------------------------

let app;
const page = (js) => app.cdp.eval(js);

const HELPERS = `(() => {
  if (window.__h) return true;
  window.__toasts = [];
  new MutationObserver(() => {
    for (const el of document.querySelectorAll('[role=status]')) {
      if (el.__seen) continue;
      el.__seen = true;
      window.__toasts.push({ text: el.innerText, error: el.style.background.includes('--err') });
    }
  }).observe(document.body, { childList: true, subtree: true });
  window.confirm = () => true;
  const setter = (el) => Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), 'value').set;
  window.__h = {
    btn(root, text) { const t = text.toLowerCase(); return [...(root ?? document).querySelectorAll('button')].find((b) => b.innerText.trim().toLowerCase() === t || b.title === text) ?? null; },
    fill(el, v) { setter(el).call(el, v); el.dispatchEvent(new Event('input', { bubbles: true })); },
    choose(el, v) { setter(el).call(el, v); el.dispatchEvent(new Event('change', { bubbles: true })); },
    enter(el) { el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true })); },
    row(label) { return [...document.querySelectorAll('.text-\\\\[13px\\\\].font-medium')].find((d) => d.innerText.trim() === label)?.closest('.flex.items-center') ?? null; },
  };
  return true;
})()`;

async function ui(js) {
  await page(HELPERS);
  return page(`(() => { const h = window.__h; ${js} })()`);
}

async function openModels() {
  if (!(await ui(`return !!h.btn(document, 'Add provider') || !!document.querySelector('.smallcaps') && [...document.querySelectorAll('.smallcaps')].some((e) => e.innerText.trim().toLowerCase() === 'add provider')`))) {
    await app.cdp.click('button[title^="Settings"]');
    await sleep(500);
    await ui(`h.btn(document, 'Models')?.click()`);
    await sleep(400);
  }
}

/** Runs `action` and returns the first new toast whose text matches `match`. */
async function toastAfter(action, match, timeoutMs = 90000) {
  const n = await ui(`return window.__toasts.length`);
  await action();
  const until = Date.now() + timeoutMs;
  for (;;) {
    const t = (await ui(`return window.__toasts.slice(${n})`)).filter((x) => match.test(x.text));
    if (t.length) return t[0];
    if (Date.now() > until) return { text: `(no toast matching ${match} within ${timeoutMs / 1000} s)`, error: true };
    await sleep(300);
  }
}

async function waitButton(rowSel, text, timeoutMs) {
  const until = Date.now() + timeoutMs;
  while (!(await ui(`return !!h.btn(${rowSel}, ${JSON.stringify(text)}) && !h.btn(${rowSel}, ${JSON.stringify(text)}).disabled`))) {
    if (Date.now() > until) throw new Error(`button ${text} in ${rowSel} not available after ${timeoutMs / 1000} s`);
    await sleep(500);
  }
}

const settingsNow = () => app.cdp.invoke("settings:get");

async function addProvider(r, key) {
  await openModels();
  const t = await toastAfter(
    () =>
      ui(`
      const card = [...document.querySelectorAll('.card')].find((c) => c.querySelector('.smallcaps')?.innerText.trim().toLowerCase() === 'add provider');
      h.btn(card, 'Custom').click();
      return new Promise((res) => setTimeout(() => {
        const [label, url, key] = card.querySelectorAll('input');
        h.fill(label, ${JSON.stringify(r.label)});
        h.fill(url, ${JSON.stringify(r.base)});
        h.fill(key, ${JSON.stringify(key ?? "")});
        setTimeout(() => { h.btn(card, 'Add').click(); res(true); }, 150);
      }, 150));`),
    /^Added /,
  );
  const s = await settingsNow();
  r.pid = s.providers.find((p) => p.label === r.label)?.id;
  return t;
}

async function addModel(r, name, kind) {
  await openModels();
  await ui(`h.btn(document, 'Add API model')?.click()`);
  await sleep(300);
  await ui(`
    const input = document.querySelector('input[placeholder^="model name"]');
    const card = input.closest('.card');
    const [prov, k] = card.querySelectorAll('select');
    h.choose(prov, ${JSON.stringify(r.pid)});
    h.choose(k, ${JSON.stringify(kind)});
    h.fill(input, ${JSON.stringify(name)});
    return new Promise((res) => setTimeout(() => { h.btn(card, 'Add').click(); res(true); }, 200));`);
  await sleep(500);
  return `${r.pid}:${name}`;
}

const rowJs = (id) => `document.querySelector('[data-model=${JSON.stringify(JSON.stringify(id)).slice(1, -1)}]')`;
const TOAST_FOR = {
  Detect: /^Context \d+|did not report a context length|failed|cannot reach|stuck/,
  Test: null,
};
async function modelButton(id, text) {
  const label = (await settingsNow()).models.find((m) => m.id === id)?.label ?? id;
  await waitButton(rowJs(id), text, 330000);
  const match = TOAST_FOR[text] ?? new RegExp(`^${label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}: `);
  return toastAfter(() => ui(`h.btn(${rowJs(id)}, ${JSON.stringify(text)}).click()`), match, 330000);
}

async function setNum(id, index, value) {
  await ui(`const el = ${rowJs(id)}.querySelectorAll('input')[${index}]; h.fill(el, ${JSON.stringify(String(value))}); return new Promise((res) => setTimeout(() => { h.enter(el); res(true); }, 100));`);
  await sleep(400);
}

async function setKey(r, key) {
  await openModels();
  return toastAfter(async () => {
    await ui(`document.querySelector('[data-provider=${JSON.stringify(r.pid)}] [data-testid=key-edit]').click()`);
    await sleep(250);
    await ui(`const el = document.querySelector('[data-provider=${JSON.stringify(r.pid)}] input[type=password]'); h.fill(el, ${JSON.stringify(key)}); return new Promise((res) => setTimeout(() => { h.enter(el); res(true); }, 100));`);
  }, /^Key (saved|removed|not saved)/);
}

const removeKey = (r) => toastAfter(() => ui(`document.querySelector('[data-provider=${JSON.stringify(r.pid)}] [data-testid=key-remove]').click()`), /^Key (saved|removed|not saved)/);

async function setRole(label, value) {
  await openModels();
  await ui(`h.choose(h.row(${JSON.stringify(label)}).querySelector('select'), ${JSON.stringify(value)})`);
  await sleep(500);
}

const secretsFile = () => {
  const f = path.join(userData, "secrets.json");
  return existsSync(f) ? JSON.parse(readFileSync(f, "utf8")) : {};
};

function scanForKeys(root, keys) {
  const hits = [];
  const walk = (p) => {
    const st = statSync(p);
    if (st.isDirectory()) return readdirSync(p).forEach((n) => walk(path.join(p, n)));
    if (st.size > 200 * 1024 * 1024) return;
    const buf = readFileSync(p);
    for (const k of keys) if (buf.includes(k) || buf.includes(Buffer.from(k, "utf16le"))) hits.push(`${path.relative(dir, p)} contains ${k.slice(0, 8)}…`);
  };
  if (existsSync(root)) walk(root);
  return hits;
}

const closeSettings = async () => {
  await app.cdp.click('button[title="Explorer"]').catch(() => undefined);
  await sleep(300);
};

// ---- scenario -----------------------------------------------------------------------------------

const report = { stamp, runtimes: {}, checks, notes: [] };
try {
  await bringUp();
  for (const [n, r] of Object.entries(R)) report.runtimes[n] = { label: r.label, url: r.base, model: r.model, key_fp: fp(r.key) };

  note("launching IDE with an empty profile");
  app = new IdeApp({ userData, workspace: ws, settings: { autoUpdate: false, tools: { workspaceAllow: ["*"], autoApplyEdits: true } }, logFile: path.join(dir, "app.log") });
  await app.start();
  const info = await app.cdp.invoke("app:info");
  report.secretsBackend = info.secretsBackend;
  check("store", "secure storage available", info.secretsBackend !== "unavailable", `safeStorage backend: ${info.secretsBackend}`);
  if (info.secretsBackend === "basic_text") report.notes.push("safeStorage fell back to basic_text: keys are obfuscated with a fixed password, not protected by an OS keyring.");

  note("adding providers and models through Settings → Models");
  for (const [n, r] of Object.entries(R)) {
    const t = await addProvider(r, r.key);
    check("setup", `${n}: provider added with key`, r.pid && !t.error, t.text);
    r.modelId = await addModel(r, r.model, "chat");
    if (r.embed) r.embedId = await addModel(r, r.embed, "embed");
  }
  let s = await settingsNow();
  check("setup", "all models present in settings", Object.values(R).every((r) => s.models.some((m) => m.id === r.modelId)), s.models.map((m) => m.id).join(", "));
  check("store", "settings mark every provider as keyed", Object.values(R).every((r) => s.providers.find((p) => p.id === r.pid)?.hasKey === true));

  note("Detect + Test each model with its own key");
  for (const [n, r] of Object.entries(R)) {
    const d = await modelButton(r.modelId, "Detect");
    if (/enter it manually/.test(d.text)) {
      report.notes.push(`${n}: the server does not publish its context length; entered ${r.ctx} manually as a user would`);
      check("detect", `${n}: Detect says the server reports no context length`, !/401|403|rejected/.test(d.text), d.text);
      await setNum(r.modelId, 0, r.ctx);
    } else check("detect", `${n}: Detect reads the context length`, !d.error && /Context \d+/.test(d.text), d.text);
    const t = await modelButton(r.modelId, "Test");
    check("test", `${n}: Test gets a reply`, !t.error && /first token/.test(t.text), t.text);
  }
  await setNum(R.ollama.embedId, 0, 768);
  const et = await modelButton(R.ollama.embedId, "Test");
  check("test", "ollama embeddings: Test returns vectors", !et.error && /dim 768/.test(et.text), et.text);

  note("negative: wrong key, missing key, then restore");
  let t = await setKey(R.vllm, WRONG);
  t = await modelButton(R.vllm.modelId, "Test");
  check("negative", "vllm: wrong key → clear rejection", t.error && /401/.test(t.text) && /rejected the API key/.test(t.text), t.text);
  check("leak", "vllm: rejection message does not echo the key", !t.text.includes(WRONG));
  t = await modelButton(R.vllm.modelId, "Detect");
  check("negative", "vllm: Detect with wrong key reports the key problem", t.error && /rejected the API key/.test(t.text), t.text);
  t = await removeKey(R.vllm);
  s = await settingsNow();
  check("store", "Remove key clears hasKey and the stored secret", s.providers.find((p) => p.id === R.vllm.pid)?.hasKey === false && !secretsFile()[`provider:${R.vllm.pid}`], t.text);
  t = await modelButton(R.vllm.modelId, "Test");
  check("negative", "vllm: missing key → says a key is required", t.error && /requires an API key and none is set/.test(t.text), t.text);
  await setKey(R.vllm, R.vllm.key);
  t = await modelButton(R.vllm.modelId, "Test");
  check("negative", "vllm: restored key works again", !t.error && /first token/.test(t.text), t.text);

  note("rotation: the Ollama gate revokes the old key and accepts a new one");
  writeFileSync(R.ollama.keyFile, `${ROTATED}\n`);
  t = await modelButton(R.ollama.modelId, "Test");
  check("rotation", "old key rejected after the server rotates", t.error && /rejected the API key/.test(t.text), t.text);
  await setKey(R.ollama, ROTATED);
  t = await modelButton(R.ollama.modelId, "Test");
  check("rotation", "new key works without restarting the IDE", !t.error && /first token/.test(t.text), t.text);
  t = await modelButton(R.ollama.embedId, "Test");
  check("rotation", "embeddings use the new key too", !t.error && /dim 768/.test(t.text), t.text);
  R.ollama.key = ROTATED;

  note("agent chat through a key-protected server");
  await setNum(R.control.modelId, 1, 8192);
  await setRole("Agent", R.control.modelId);
  await setRole("Embeddings", R.ollama.embedId);
  await setRole("Summarizer", R.llama.modelId);
  s = await settingsNow();
  check("roles", "roles point at the API models", s.roles.agent === R.control.modelId && s.roles.embed === R.ollama.embedId && s.roles.summarizer === R.llama.modelId, JSON.stringify(s.roles));
  await closeSettings();
  const ok = await app.ask("Create a file named add.py that prints the sum of 17 and 25, run it with python, and tell me the number it printed.", { timeoutMs: 10 * 60 * 1000, onApproval: async () => true });
  writeFileSync(path.join(dir, "agent-answer.md"), ok.text);
  const fileOk = existsSync(path.join(ws, "add.py"));
  check("chat", "agent turn through the gated server finishes", ok.status === "done" && /42/.test(ok.text), `${ok.status}: ${ok.text.slice(0, 160)}`);
  check("chat", "agent created and ran the file", fileOk);
  const agentReqs = requests("control").filter((q) => q.path.endsWith("/chat/completions"));
  check("chat", "agent requests carried the agent key", agentReqs.length > 0 && agentReqs.every((q) => q.authorized && q.key_fp === fp(R.control.key)), `${agentReqs.length} chat requests`);
  const embReqs = requests("ollama").filter((q) => q.path.endsWith("/embeddings") && q.t * 1000 > Date.now() - ok.durationMs - 60000);
  report.notes.push(`memory embeddings during the agent turn: ${embReqs.length} request(s) to the gated Ollama embedder`);

  note("negative chat: the agent's key is wrong");
  await setKey(R.control, WRONG);
  await closeSettings();
  await app.newChat();
  const bad = await app.ask("Say hello.", { timeoutMs: 120000 });
  const blob = JSON.stringify(bad);
  const dom = await page(`document.body.innerText`);
  check("chat", "turn fails with a clear key error", bad.status !== "done" && /rejected the API key/.test(blob), `${bad.status}: ${(blob.match(/[^"]*rejected the API key[^"]*/) ?? [""])[0].slice(0, 200)}`);
  check("chat", "the key error is visible in the chat UI", /rejected the API key/.test(dom));
  check("leak", "chat error does not echo the key", !blob.includes(WRONG) && !dom.includes(WRONG));
  await setKey(R.control, R.control.key);

  note("isolation: which key reached which server");
  for (const [n, r] of Object.entries(R)) {
    const seen = new Set(requests(n).map((q) => q.key_fp));
    const others = Object.entries(R)
      .filter(([m]) => m !== n)
      .map(([, o]) => fp(o.key));
    const foreign = [...seen].filter((x) => others.includes(x));
    check("isolation", `${n}: never received another server's key`, foreign.length === 0, `fingerprints seen: ${[...seen].map((x) => x || "(none)").join(", ")}`);
  }

  note("removing a provider purges its key and repairs roles");
  await openModels();
  const llamaPid = R.llama.pid;
  await ui(`document.querySelector('[data-provider=${JSON.stringify(llamaPid)}] button[title="Remove provider"]').click()`);
  await sleep(800);
  s = await settingsNow();
  check("store", "provider and its models are gone", !s.providers.some((p) => p.id === llamaPid) && !s.models.some((m) => m.providerId === llamaPid));
  check("store", "its key is purged from secrets.json", !secretsFile()[`provider:${llamaPid}`], Object.keys(secretsFile()).join(", "));
  check("store", "summarizer role no longer points at the removed model", s.roles.summarizer === "", JSON.stringify(s.roles));
  check("store", "secrets.json holds exactly the live API providers", JSON.stringify(Object.keys(secretsFile()).sort()) === JSON.stringify(s.providers.filter((p) => p.hasKey).map((p) => `provider:${p.id}`).sort()));

  note("leak scan");
  const domAll = await page(`document.documentElement.outerHTML + JSON.stringify(localStorage) + JSON.stringify(sessionStorage)`);
  const domHits = allKeys().filter((k) => domAll.includes(k));
  check("leak", "no key in the renderer DOM or web storage", domHits.length === 0, domHits.map((k) => k.slice(0, 8)).join(", "));
  const settingsBlob = JSON.stringify(await settingsNow());
  check("leak", "no key in settings returned to the renderer", !allKeys().some((k) => settingsBlob.includes(k)));
  report.toasts = await ui(`return window.__toasts`);
  await app.stop();
  app = null;
  const fileHits = scanForKeys(userData, allKeys()).concat(scanForKeys(path.join(dir, "app.log"), allKeys()));
  check("leak", "no plaintext key anywhere in the profile (settings, db, logs, secrets.json)", fileHits.length === 0, fileHits.join("; "));
} catch (e) {
  check("run", "harness error", false, e.stack ?? String(e));
  if (app) await app.screenshot(path.join(dir, "error.png")).catch(() => undefined);
} finally {
  if (app) report.toasts = await ui(`return window.__toasts`).catch(() => undefined);
  if (app) await app.stop();
  await stopAll();
  const pass = checks.filter((c) => c.ok).length;
  report.summary = { pass, total: checks.length };
  writeFileSync(path.join(dir, "report.json"), JSON.stringify(report, null, 2));
  console.log(`\n${pass}/${checks.length} checks passed → ${path.relative(REPO, dir)}/report.json`);
}
