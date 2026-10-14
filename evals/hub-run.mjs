/**
 * Models hub end-to-end eval, driven through the built IDE like a user:
 * look up repos (gated, missing, real), pick files, download with progress and speed, deploy a chat
 * model with llama.cpp and an embedding model with transformers, test them, make them the agent and
 * the memory embedder, chat with the agent, restart the IDE (deployments come back), delete a model,
 * and check that no deployment key is written anywhere in plain text.
 *
 *   node evals/hub-run.mjs
 *   env: HUB_EVAL_LLAMA_DIR (an existing llama.cpp build; otherwise the eval builds one via Tools)
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { IdeApp, REPO } from "./lib/app.mjs";
import { sleep } from "./lib/cdp.mjs";

const CHAT_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF";
const CHAT_FILE = "qwen2.5-0.5b-instruct-q8_0.gguf";
const EMBED_REPO = "sentence-transformers/all-MiniLM-L6-v2";
const LLAMA_DIR = process.env.HUB_EVAL_LLAMA_DIR ?? path.join(REPO, "evals/.runtimes/hub-live/tools/llama.cpp");

const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
const dir = path.join(REPO, "evals/results", `hub-${stamp}`);
const ws = path.join(dir, "ws");
const userData = path.join(dir, "userdata");
rmSync(dir, { recursive: true, force: true });
mkdirSync(ws, { recursive: true });
writeFileSync(path.join(ws, "README.md"), "# scratch workspace for the models hub eval\n");

const checks = [];
const check = (group, name, ok, detail = "") => {
  checks.push({ group, name, ok: !!ok, detail: String(detail).slice(0, 400) });
  console.log(`  ${ok ? "PASS" : "FAIL"} [${group}] ${name}${detail ? ` — ${String(detail).slice(0, 220)}` : ""}`);
};
const note = (s) => console.log(`▶ ${s}`);

const settings = existsSync(path.join(LLAMA_DIR, "build/bin/llama-server")) ? { hub: { llamaCppDir: LLAMA_DIR } } : {};
let app;
const shot = (name) => app.screenshot(path.join(dir, `${name}.png`)).catch(() => undefined);

const HELPERS = `(() => {
  if (window.__h) return true;
  window.__toasts = [];
  new MutationObserver(() => {
    for (const el of document.querySelectorAll('[role=status]')) {
      if (el.__seen) continue;
      el.__seen = true;
      window.__toasts.push(el.innerText);
    }
  }).observe(document.body, { childList: true, subtree: true });
  window.confirm = () => true;
  const setter = (el) => Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el), 'value').set;
  window.__h = {
    q: (s, root) => (root ?? document).querySelector(s),
    fill(el, v) { setter(el).call(el, v); el.dispatchEvent(new Event('input', { bubbles: true })); },
    choose(el, v) { setter(el).call(el, v); el.dispatchEvent(new Event('change', { bubbles: true })); },
    options(el) { return [...el.options].map((o) => o.value); },
  };
  return true;
})()`;
const ui = async (js) => {
  await app.cdp.eval(HELPERS);
  return app.cdp.eval(`(() => { const h = window.__h; ${js} })()`);
};
async function until(js, ms, what) {
  const end = Date.now() + ms;
  for (;;) {
    const v = await ui(js).catch(() => null);
    if (v) return v;
    if (Date.now() > end) {
      await shot(`stuck-${what.replace(/\W+/g, "-")}`);
      throw new Error(`timed out: ${what}`);
    }
    await sleep(1000);
  }
}
const toasts = () => ui("return window.__toasts.splice(0)");

async function openHub() {
  await app.cdp.click('button[title="Models hub"]');
  await until("return !!h.q('[data-testid=hub-screen]')", 15000, "hub screen");
}

async function lookup(repo) {
  await ui(`h.fill(h.q('[data-testid=hub-repo]'), ${JSON.stringify(repo)}); return true;`);
  await app.cdp.click("[data-testid=hub-lookup]");
  return until(
    `const card = h.q('[data-testid=hub-repo-card]'); const err = h.q('[data-testid=hub-error]');
     if (err) return { error: err.innerText };
     if (card && card.dataset.repo === ${JSON.stringify(repo)}) return { repo: card.dataset.repo, access: h.q('[data-testid=hub-access]')?.innerText ?? null,
       files: [...document.querySelectorAll('[data-testid=hub-files] input[data-file]')].map((i) => ({ path: i.dataset.file, checked: i.checked })),
       canDownload: !h.q('[data-testid=hub-download]')?.disabled };
     return null;`,
    60000,
    `lookup ${repo}`,
  );
}

/** Starts the download from the repo card and watches the job row until it finishes. */
async function download(repo) {
  await app.cdp.click("[data-testid=hub-download]");
  const seen = { percents: [], speeds: [] };
  const end = Date.now() + 60 * 60_000;
  for (;;) {
    const row = await ui(`const r = [...document.querySelectorAll('[data-job]')].find((x) => x.dataset.repo === ${JSON.stringify(repo)});
      return r ? { status: r.dataset.status, pct: r.querySelector('[data-testid=job-percent]')?.innerText, speed: r.querySelector('[data-testid=job-speed]')?.innerText ?? null, text: r.innerText } : null;`);
    if (row) {
      if (row.pct && seen.percents.at(-1) !== row.pct) seen.percents.push(row.pct);
      if (row.speed) seen.speeds.push(row.speed);
      if (["done", "error"].includes(row.status)) return { ...row, ...seen };
    }
    if (Date.now() > end) throw new Error(`download of ${repo} did not finish`);
    await sleep(1000);
  }
}

/** Opens the deploy form of a library model, fills it and returns the new deployment's row once it settles. */
async function deploy(repo, { target, source, quant, kind }) {
  const before = await app.cdp.invoke("hub:state");
  await ui(`const row = [...document.querySelectorAll('[data-model]')].find((r) => r.innerText.includes(${JSON.stringify(repo)})); row.querySelector('[data-testid=model-deploy]').click(); return true;`);
  await until("return !!h.q('[data-testid=deploy-form] [data-target]')", 30000, "deploy form");
  await ui(`h.q('[data-testid=deploy-form] [data-target=${target}]').click(); return true;`);
  await sleep(500);
  const form = await ui(`return { sources: h.options(h.q('[data-testid=deploy-source]')), kinds: h.options(h.q('[data-testid=deploy-kind]')) };`);
  const src = form.sources.find((s) => s === source || s.startsWith(source)) ?? form.sources[0];
  await ui(`h.choose(h.q('[data-testid=deploy-source]'), ${JSON.stringify(src)}); return true;`);
  await sleep(300);
  const quants = await ui("return h.options(h.q('[data-testid=deploy-quant]'))");
  await ui(`h.choose(h.q('[data-testid=deploy-quant]'), ${JSON.stringify(quant)}); h.choose(h.q('[data-testid=deploy-kind]'), ${JSON.stringify(kind)}); return true;`);
  await sleep(300);
  await app.cdp.click("[data-testid=deploy-go]");
  const id = await until(
    `return [...document.querySelectorAll('[data-deployment]')].map((d) => d.dataset.deployment).find((x) => !${JSON.stringify(before.deployments.map((d) => d.id))}.includes(x)) ?? null`,
    30000,
    "deployment row",
  );
  const status = await until(`const d = h.q('[data-deployment="${id}"]'); return d && ['running', 'error'].includes(d.dataset.status) ? d.dataset.status : null;`, 40 * 60_000, `deployment of ${repo}`);
  const dep = (await app.cdp.invoke("hub:state")).deployments.find((d) => d.id === id);
  return { id, status, dep, form: { ...form, quants } };
}

const depState = async (id) => (await app.cdp.invoke("hub:state")).deployments.find((d) => d.id === id);

async function main() {
  app = await new IdeApp({ userData, workspace: ws, settings, logFile: path.join(dir, "app.log") }).start();
  note("open the Models hub");
  await openHub();
  await shot("01-hub");
  await until("return window.rlm.invoke('hub:state', {}).then((s) => !s.tools.probing && (s.tools.python || s.tools.pythonError))", 120000, "python probe");
  const tools = (await app.cdp.invoke("hub:state")).tools;
  check("tools", "Python is detected", !!tools.python, tools.python ? `${tools.python.version} · torch ${tools.python.packages.torch} · transformers ${tools.python.packages.transformers}` : tools.pythonError);
  if (!tools.llamaCpp.server) {
    note("building llama.cpp from the Tools section");
    await app.cdp.click("[data-testid=llama-build-cpu]");
    await until("return !h.q('[data-testid=tools-installing]') && (window.rlm ? true : false)", 60 * 60_000, "llama.cpp build");
  }
  const t2 = (await app.cdp.invoke("hub:state")).tools.llamaCpp;
  check("tools", "llama.cpp is found with its quantization types", !!t2.server && t2.quantTypes.length > 15, `${t2.version} · ${t2.quantTypes.length} types`);

  note("gated and missing repos");
  const gated = await lookup("google/embeddinggemma-300m");
  check("lookup", "gated repo explains that a token is needed and blocks the download", /token/i.test(gated.access ?? "") && !gated.canDownload, gated.access);
  const missing = await lookup("no-such-org-xyz/no-such-model-123");
  check("lookup", "missing repo shows a clear error", /not found|gated|private/i.test(missing.error ?? ""), missing.error);

  note(`download ${CHAT_FILE} from ${CHAT_REPO}`);
  const chat = await lookup(CHAT_REPO);
  const ggufs = chat.files.filter((f) => f.path.endsWith(".gguf"));
  check("lookup", "GGUF repo lists every quantization file and preselects none of them", ggufs.length > 5 && ggufs.every((f) => !f.checked), `${ggufs.length} GGUF files`);
  await ui(`h.q('input[data-file="${CHAT_FILE}"]').click(); return true;`);
  await shot("02-chat-files");
  const dl = await download(CHAT_REPO);
  check("download", "chat model download finishes", dl.status === "done", dl.text.slice(0, 200));
  check("download", "progress and speed are shown while downloading", dl.percents.length > 3 && dl.speeds.some((s) => /\/s/.test(s)), `${dl.percents.length} progress steps, e.g. ${dl.speeds.at(-1) ?? "none"}`);

  note("deploy the chat model with llama.cpp");
  const c = await deploy(CHAT_REPO, { target: "llamacpp", source: `gguf:${CHAT_FILE}`, quant: "as-is", kind: "chat" });
  await shot("03-chat-deployed");
  check("deploy", "llama.cpp deployment runs", c.status === "running", c.dep?.error ?? `${c.dep?.servedName} at ${c.dep?.baseUrl}`);
  check("deploy", "deploy form offers the file as-is and the llama.cpp quantization types", c.form.quants.includes("as-is") && c.form.quants.includes("Q4_K_M"), c.form.quants.slice(0, 8).join(" "));
  if (c.status !== "running") throw new Error("chat deployment failed");
  const noKey = await fetch(`${c.dep.baseUrl}/models`).then((r) => r.status);
  check("security", "the deployment refuses requests without its key", noKey === 401, `HTTP ${noKey}`);

  await ui(`h.q('[data-deployment="${c.id}"] [data-action=test]').click(); return true;`);
  const testToast = await until("const t = window.__toasts.find((x) => /^(OK|Failed):/.test(x)); return t ?? null;", 120000, "test toast");
  check("deploy", "Test button reaches the model", /^OK:/.test(testToast), testToast);
  await toasts();

  await ui(`h.q('[data-deployment="${c.id}"] [data-role=agent]').click(); return true;`);
  await sleep(1000);
  let s = await app.cdp.invoke("settings:get");
  check("roles", "Use as agent makes it the agent model", s.roles.agent === c.dep.settingsModelId, s.roles.agent);
  const provider = s.providers.find((p) => p.id === `hub-${c.id}`);
  check("roles", "the deployment is a provider with its key in the keychain", provider?.hasKey === true && provider.baseUrl === c.dep.baseUrl, JSON.stringify(provider ?? null));

  note("chat with the hub model as the agent");
  await app.cdp.click('button[title="Explorer"]').catch(() => undefined);
  await sleep(500);
  // llama-server does not log request paths; each completion it serves logs "processing task".
  const tasks = async () => ((await app.cdp.invoke("hub:log", { id: c.id })).match(/processing task/g) ?? []).length;
  const tasksBefore = await tasks();
  const turn = await app.ask("What is 17 + 25? Reply with just the number.", { timeoutMs: 10 * 60_000 });
  await shot("04-chat");
  const served = (await tasks()) - tasksBefore;
  check("agent", "a chat turn completes on the hub model", turn.status === "done" && turn.text.trim().length > 0, `${turn.status}: ${turn.text.slice(0, 160)}`);
  check("agent", "the turn's requests went to the hub deployment", served > 0, `${served} completions served by the deployment during the turn`);
  check("agent", "the answer is right", /\b42\b/.test(turn.text), turn.text.slice(0, 80));

  note(`download and deploy ${EMBED_REPO} as the memory embedder`);
  await openHub();
  const emb = await lookup(EMBED_REPO);
  const picked = emb.files.filter((f) => f.checked).map((f) => f.path);
  check("lookup", "embedding repo preselects one weight format", picked.filter((p) => /\.(safetensors|bin|onnx|h5|msgpack|ot)$/.test(p)).length === 1, picked.join(", ").slice(0, 200));
  const edl = await download(EMBED_REPO);
  check("download", "embedding model download finishes", edl.status === "done", edl.text.slice(0, 160));
  const e = await deploy(EMBED_REPO, { target: "transformers", source: "weights", quant: "none", kind: "embed" });
  check("deploy", "transformers embedding deployment runs", e.status === "running" && e.dep.dim === 384, e.dep?.error ?? `${e.dep.dim}-dim`);
  await ui(`h.q('[data-deployment="${e.id}"] [data-role=embed]').click(); return true;`);
  await sleep(1000);
  s = await app.cdp.invoke("settings:get");
  check("roles", "Use for memory makes it the embedder", s.roles.embed === e.dep.settingsModelId, s.roles.embed);
  for (const text of ["The user prefers tabs over spaces for indentation.", "The production database runs PostgreSQL 16.", "Deploys go out every Tuesday after the standup."])
    await app.cdp.invoke("memory:add", { text, category: "preference" });
  const stats = await app.cdp.invoke("memory:stats");
  const elog = await app.cdp.invoke("hub:log", { id: e.id });
  check("memory", "memories are embedded by the hub embedder", /POST \/v1\/embeddings 200/.test(elog) && JSON.stringify(stats.collections).includes("384"), `${(elog.match(/POST \/v1\/embeddings 200/g) ?? []).length} embedding requests · collections ${JSON.stringify(stats.collections)}`);
  await app.cdp.click('button[title="Explorer"]').catch(() => undefined);
  await app.newChat().catch(() => undefined);
  const recallTurn = await app.ask("Which indentation should I use when I add code here? One short sentence.", { timeoutMs: 10 * 60_000 });
  const recalled = recallTurn.events.find((ev) => ev.type === "memory_recalled");
  // A handful of facts are all recalled as standing facts; the hub embedder's similarity is the score.
  const ranked = [...(recalled?.items ?? [])].sort((a, b) => b.score - a.score);
  check(
    "memory",
    "recall scores the matching memory highest with the hub embedder",
    !recalled?.offline && /tabs/i.test(ranked[0]?.text ?? "") && ranked[0].score > (ranked[1]?.score ?? 0),
    ranked.map((m) => `${m.score.toFixed(3)} ${m.text.slice(0, 40)}`).join(" | "),
  );
  console.log(`  info: the 0.5B agent answered "${recallTurn.text.slice(0, 120)}" with the recalled facts in its prompt`);
  await openHub();
  await shot("05-embed");

  note("restart the IDE; running deployments come back");
  await app.stop();
  app = await new IdeApp({ userData, workspace: ws, settings, logFile: path.join(dir, "app-2.log") }).start();
  await openHub();
  const back = await until(
    `const ids = ${JSON.stringify([c.id, e.id])}; const rows = ids.map((i) => h.q('[data-deployment="' + i + '"]')); return rows.every((r) => r && ['running', 'error'].includes(r.dataset.status)) ? rows.map((r) => r.dataset.status) : null;`,
    15 * 60_000,
    "restored deployments",
  );
  await shot("06-restored");
  check("restart", "both deployments are running again after a restart", back.every((x) => x === "running"), back.join(", "));
  const cBack = await depState(c.id);
  check("restart", "the restored deployment kept its provider, key and role", (await app.cdp.invoke("settings:get")).roles.agent === cBack.settingsModelId && (await fetch(`${cBack.baseUrl}/models`).then((r) => r.status)) === 401, cBack.baseUrl);
  await app.cdp.click('button[title="Explorer"]').catch(() => undefined);
  await app.newChat().catch(() => undefined);
  const turn2 = await app.ask("Name the capital of France in one word.", { timeoutMs: 10 * 60_000 });
  check("restart", "the agent answers after the restart", turn2.status === "done" && /paris/i.test(turn2.text), turn2.text.slice(0, 120));

  note("delete the embedding model");
  await openHub();
  await ui(`h.q('[data-deployment="${e.id}"] [data-action=stop]').click(); return true;`);
  await until(`return h.q('[data-deployment="${e.id}"]')?.dataset.status === 'stopped'`, 30000, "stop");
  const modelDir = (await app.cdp.invoke("hub:state")).models.find((m) => m.repo === EMBED_REPO).dir;
  await ui(`const row = [...document.querySelectorAll('[data-model]')].find((r) => r.innerText.includes(${JSON.stringify(EMBED_REPO)})); row.querySelector('[data-testid=model-delete]').click(); return true;`);
  await until(`return ![...document.querySelectorAll('[data-model]')].some((r) => r.innerText.includes(${JSON.stringify(EMBED_REPO)}))`, 30000, "model removed");
  s = await app.cdp.invoke("settings:get");
  check("delete", "model, deployment, provider and files are gone", !(await depState(e.id)) && !s.providers.some((p) => p.id === `hub-${e.id}`) && !existsSync(modelDir), modelDir);
  check("delete", "the memory role falls back to a model that exists", s.models.some((m) => m.id === s.roles.embed && m.kind === "embed"), s.roles.embed);
  await shot("07-deleted");

  note("no deployment key in plain text on disk");
  const keyish = /rlm-hub-[0-9a-f]{48}/;
  const leaks = [];
  const walk = (d) => {
    for (const n of readdirSync(d)) {
      const p = path.join(d, n);
      const st = statSync(p);
      if (st.isDirectory()) {
        if (n === "models" || n === "node_modules") continue;
        walk(p);
      } else if (st.size < 20_000_000 && keyish.test(readFileSync(p, "latin1"))) leaks.push(path.relative(userData, p));
    }
  };
  walk(userData);
  check("security", "no hub API key appears in plain text under the profile", leaks.length === 0, leaks.join(", "));
  const keyFiles = readdirSync(path.join(userData, "hub/deployments"), { recursive: true }).filter((f) => String(f).endsWith("api-key"));
  check("security", "key files for llama.cpp are removed once the server has read them", keyFiles.length === 0, keyFiles.join(", "));
}

let fatal = null;
try {
  await main();
} catch (err) {
  fatal = err;
  console.error(err);
  await shot("fatal");
} finally {
  await app?.stop();
}
const passed = checks.filter((c) => c.ok).length;
writeFileSync(path.join(dir, "results.json"), JSON.stringify({ checks, fatal: fatal ? String(fatal.stack ?? fatal) : null }, null, 2));
console.log(`\n${passed}/${checks.length} checks passed${fatal ? ` · stopped early: ${fatal.message}` : ""}\nresults: ${path.relative(REPO, dir)}`);
process.exit(fatal || passed !== checks.length ? 1 : 0);
