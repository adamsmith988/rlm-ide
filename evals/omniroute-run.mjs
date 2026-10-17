/**
 * Free models end-to-end eval, driven through the built IDE like a user:
 * install and start the managed OmniRoute gateway from Settings → Models, let Set up pick the free models
 * that answer, chat with the agent on them, then put a fault-injecting proxy in front of the chain and check
 * that a reply cut off mid-stream is finished by the next model and that a rate limit switches silently,
 * restart the IDE (the gateway comes back), and check the gateway is local-only and key-protected.
 *
 *   node evals/omniroute-run.mjs
 *   env: OMNI_EVAL_USERDATA (profile reused between runs so the 3 GB install happens once;
 *        default evals/.runtimes/omniroute-live)
 */
import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { IdeApp, REPO } from "./lib/app.mjs";
import { sleep } from "./lib/cdp.mjs";

const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
const dir = path.join(REPO, "evals/results", `omniroute-${stamp}`);
const ws = path.join(dir, "ws");
const userData = process.env.OMNI_EVAL_USERDATA ?? path.join(REPO, "evals/.runtimes/omniroute-live");
rmSync(dir, { recursive: true, force: true });
mkdirSync(ws, { recursive: true });
mkdirSync(userData, { recursive: true });
writeFileSync(path.join(ws, "README.md"), "# scratch workspace for the free models eval\n");

const checks = [];
const check = (group, name, ok, detail = "") => {
  checks.push({ group, name, ok: !!ok, detail: String(detail).slice(0, 400) });
  console.log(`  ${ok ? "PASS" : "FAIL"} [${group}] ${name}${detail ? ` — ${String(detail).slice(0, 240)}` : ""}`);
};
const skipped = [];
/** A check that cannot apply with the models available (e.g. no free model follows the cell protocol). */
const skip = (group, name, why) => {
  skipped.push({ group, name, why });
  console.log(`  SKIP [${group}] ${name} — ${String(why).slice(0, 240)}`);
};
const note = (s) => console.log(`▶ ${s}`);

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
  window.__h = { q: (s, root) => (root ?? document).querySelector(s) };
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
const state = () => app.cdp.invoke("omniroute:state");
const settingsNow = () => app.cdp.invoke("settings:get");
const clickTid = (tid) => ui(`const b = h.q('[data-testid=${tid}]'); if (!b || b.disabled) return false; b.click(); return true;`);

async function openFreeModels() {
  await app.cdp.click("[data-testid=activity-settings]");
  await until("return !!h.q('[data-testid=omniroute-card]')", 15000, "free models card");
  await ui("h.q('[data-testid=omniroute-card]').scrollIntoView({ block: 'start' }); return true;");
}

async function backToChat() {
  await app.cdp.click('button[title="Explorer"]').catch(() => undefined);
  await sleep(400);
}

/** OpenAI-compatible proxy in front of one free model that misbehaves on demand. */
function faultProxy(gatewayKey, port, model) {
  const p = {
    mode: "pass",
    cutAfter: 40,
    requests: 0,
    cuts: 0,
    limited: 0,
    partial: "",
    errors: 0,
  };
  p.server = http.createServer((req, res) => {
    let body = "";
    req.on("data", (d) => (body += d));
    req.on("end", () =>
      relay(req, res, body).catch(() => {
        p.errors++;
        res.socket?.destroy();
      }),
    );
  });
  async function relay(req, res, body) {
    if (!req.url.endsWith("/chat/completions")) {
      res.writeHead(404).end();
      return;
    }
    p.requests++;
    if (p.mode === "429") {
      p.limited++;
      res.writeHead(429, {
        "Content-Type": "application/json",
        "Retry-After": "120",
      });
      res.end(
        JSON.stringify({
          error: { message: "rate limited by the fault proxy" },
        }),
      );
      return;
    }
    const j = JSON.parse(body);
    const up = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${gatewayKey}`,
      },
      body: JSON.stringify({ ...j, model }),
    }).catch((e) => ({
      ok: false,
      status: 502,
      text: async () => String(e),
      headers: new Headers(),
    }));
    if (!up.ok || !up.body) {
      res.writeHead(up.status, { "Content-Type": "application/json" }).end(await up.text());
      return;
    }
    res.writeHead(200, {
      "Content-Type": up.headers.get("content-type") ?? "text/event-stream",
    });
    const cut = p.mode === "cut-once" || (p.mode === "cut-nth" && p.requests === p.cutOn);
    let content = "";
    let buf = "";
    const dec = new TextDecoder();
    for await (const chunk of up.body) {
      const text = dec.decode(chunk, { stream: true });
      if (cut) {
        buf += text;
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          const m = /^data: (.*)$/.exec(line.trim());
          if (m && m[1] !== "[DONE]") {
            try {
              content += JSON.parse(m[1]).choices?.[0]?.delta?.content ?? "";
            } catch {
              /* partial JSON line */
            }
          }
          if (!line.trim() && content.length >= p.cutAfter) {
            p.cuts++;
            p.partial = content;
            p.mode = "pass";
            // Cut only once the client has every event it was sent, so `partial` is exactly what the IDE saw.
            await new Promise((resolve) => res.write(line + "\n", resolve));
            res.socket?.destroy();
            return;
          }
          res.write(line + "\n");
        }
      } else res.write(text);
    }
    res.end();
  }
  return new Promise((resolve) => p.server.listen(0, "127.0.0.1", () => resolve(Object.assign(p, { port: p.server.address().port }))));
}

function lanAddress() {
  for (const list of Object.values(os.networkInterfaces())) for (const a of list ?? []) if (a.family === "IPv4" && !a.internal) return a.address;
  return null;
}

/** An extra key for the eval's proxy, made with the gateway's own CLI like any local tool would. */
function evalKey(port) {
  const r = spawnSync(
    process.execPath,
    [path.join(userData, "omniroute/runtime/node_modules/omniroute/bin/omniroute.mjs"), "api", "api-keys", "post-api-keys", "--body", JSON.stringify({ name: "eval proxy" }), "--output", "json"],
    {
      env: {
        ...process.env,
        DATA_DIR: path.join(userData, "omniroute/data"),
        OMNIROUTE_CLI_SKIP_DEFAULT_DATA_ENV: "1",
        OMNIROUTE_BASE_URL: `http://127.0.0.1:${port}`,
        NO_COLOR: "1",
      },
      encoding: "utf8",
    },
  );
  const lines = r.stdout.split("\n");
  return JSON.parse(lines.slice(lines.findIndex((l) => l.startsWith("{"))).join("\n")).key;
}

const MEMORY_FACT = "The user's deploy target is the staging cluster BLUEJAY.";
const REPL_PROMPT = "First run a REPL cell that computes sum(i*i for i in range(1, 101)) and prints it. After you see its output, reply with one sentence that states the number and my deploy target.";
const PRIMES = "2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47";
const PRIME_PROMPT = "Without running any code, write the first 15 prime numbers separated by commas and spaces on one line, then one short sentence on why 1 is not prime.";

async function main() {
  app = await new IdeApp({
    userData,
    workspace: ws,
    logFile: path.join(dir, "app.log"),
  }).start();
  await app.cdp.send("Emulation.setDeviceMetricsOverride", {
    width: 1440,
    height: 900,
    deviceScaleFactor: 1,
    mobile: false,
  });
  note("open Settings → Models → Free models");
  await openFreeModels();
  await shot("01-free-models");
  let v = await state();
  check("node", "a supported Node.js is found", v.node.ok || v.phase !== "not_installed", v.node.detail);

  if (v.phase === "not_installed") {
    note("install OmniRoute from the card (one-time, ~3 GB)");
    await clickTid("omniroute-install");
    const t0 = Date.now();
    await until("return window.rlm.invoke('omniroute:state', {}).then((s) => !s.busy && s.phase !== 'installing')", 45 * 60_000, "install");
    v = await state();
    check("install", "install finishes from the UI", !!v.version && v.phase === "stopped", `${v.version ?? v.error} in ${Math.round((Date.now() - t0) / 60000)} min`);
  }
  if (v.phase === "starting") {
    note("the gateway is starting with the IDE");
    await until("return window.rlm.invoke('omniroute:state', {}).then((s) => s.phase !== 'starting')", 5 * 60_000, "auto start");
    v = await state();
  }
  if (v.phase !== "running") {
    note("start the gateway");
    await until("const b = h.q('[data-testid=omniroute-start]'); if (b && !b.disabled) { b.click(); return true; } return false;", 30000, "start button");
    await until("return window.rlm.invoke('omniroute:state', {}).then((s) => s.phase === 'running' || s.phase === 'error')", 5 * 60_000, "gateway start");
  }
  v = await state();
  check("start", "the gateway runs", v.phase === "running", v.error ?? v.baseUrl);
  if (v.phase !== "running") throw new Error(v.error);
  await shot("02-running");

  const port = Number(new URL(v.baseUrl).port);
  note("security");
  const anon = await fetch(`${v.baseUrl}/models`).then(
    (r) => r.status,
    () => 0,
  );
  check("security", "the gateway refuses requests without a key", anon === 401 || anon === 403, `HTTP ${anon}`);
  const anonMgmt = await fetch(`http://127.0.0.1:${port}/api/providers`).then(
    (r) => r.status,
    () => 0,
  );
  check("security", "the management API needs a login", anonMgmt === 401 || anonMgmt === 403 || anonMgmt === 307 || anonMgmt === 302, `HTTP ${anonMgmt}`);
  const lan = lanAddress();
  const lanStatus = lan
    ? await fetch(`http://${lan}:${port}/api/health`, {
        signal: AbortSignal.timeout(3000),
      }).then(
        (r) => r.status,
        () => "refused",
      )
    : "no LAN address";
  check("security", "the gateway is not reachable from the network", lanStatus === "refused" || lanStatus === "no LAN address", `${lan}:${port} → ${lanStatus}`);
  let s = await settingsNow();
  check(
    "security",
    "the provider is registered with its key in the keychain",
    s.providers.some((p) => p.id === "omniroute" && p.hasKey && p.managed === "omniroute"),
    JSON.stringify(s.providers.find((p) => p.id === "omniroute") ?? null),
  );

  note("Set up free models (tests which free models answer, builds the chain)");
  await until("return !!h.q('[data-testid=omniroute-providers]')", 60000, "provider list");
  await shot("03-providers");
  const t1 = Date.now();
  await until("const b = h.q('[data-testid=omniroute-setup]'); if (b && !b.disabled) { b.click(); return true; } return false;", 60000, "setup button");
  await sleep(1500);
  await until("return window.rlm.invoke('omniroute:state', {}).then((s) => !s.busy)", 20 * 60_000, "setup");
  v = await state();
  const chain = v.models.filter((m) => m.chainIndex !== null).sort((a, b) => a.chainIndex - b.chainIndex);
  const tested = v.models.filter((m) => m.ok !== null);
  check(
    "setup",
    "Set up finds working free models and chains them",
    chain.length >= 2,
    `${chain.length} chained of ${tested.length} tested in ${Math.round((Date.now() - t1) / 1000)} s: ${chain.map((m) => `${m.id} (${((m.latencyMs ?? 0) / 1000).toFixed(1)} s)`).join(", ")}`,
  );
  check(
    "setup",
    "consecutive chain entries come from different providers where possible",
    chain.length < 2 || chain[0].provider !== chain[1].provider || new Set(chain.map((m) => m.provider)).size === 1,
    chain.map((m) => m.provider).join(" → "),
  );
  s = await settingsNow();
  const chainIds = chain.map((m) => `omniroute:${m.id}`);
  const primary = s.models.find((m) => m.id === chainIds[0]);
  check(
    "setup",
    "the first chain model is the agent and the rest are its backups, in order",
    s.roles.agent === chainIds[0] && chainIds.slice(1).every((id, i) => s.roles.agentFallbacks[i] === id),
    `${s.roles.agent} → ${s.roles.agentFallbacks.join(", ")}`,
  );
  const lost = tested.filter((m) => m.ok === false && /forget its memory/.test(m.error ?? ""));
  check(
    "setup",
    "every chained model receives the agent's instructions and memory",
    chain.every((m) => m.ok && m.systemAsUser !== null),
    `${chain.map((m) => `${m.id}: ${m.systemAsUser ? "in user message" : "system prompt"}`).join(", ")}; rejected for losing them: ${lost.map((m) => m.id).join(", ") || "none"}`,
  );
  check(
    "setup",
    "models whose provider drops system prompts get them in the user message",
    chain.every((m) => !!s.models.find((x) => x.id === `omniroute:${m.id}`)?.systemAsUser === !!m.systemAsUser),
    chain
      .filter((m) => m.systemAsUser)
      .map((m) => m.id)
      .join(", ") || "none needed",
  );
  check(
    "setup",
    "a chained model with a hidden input limit gets its measured window, not the advertised one",
    chain.every((m) => !m.inputLimitChars || s.models.find((x) => x.id === `omniroute:${m.id}`).contextLength < (m.contextLength ?? Infinity)),
    chain.map((m) => `${m.id}: ${m.inputLimitChars ? `${m.inputLimitChars} chars → ${s.models.find((x) => x.id === `omniroute:${m.id}`).contextLength} tokens` : `${m.contextLength} tokens`}`).join(", "),
  );
  const toolUsers = chain.filter((m) => m.agentReady);
  check("setup", "every chained model was tested on the agent's cell protocol", chain.every((m) => m.agentReady !== null), chain.map((m) => `${m.id}: ${m.agentReady ? "uses its tools" : "answers without them"}`).join(", "));
  const warned = await ui("return !!h.q('[data-testid=omniroute-chat-only]')");
  check("setup", "the chain warns when no free model reliably uses the agent's tools", warned === (toolUsers.length === 0), warned ? "warning shown" : "no warning");
  await ui("h.q('[data-testid=omniroute-chain]').scrollIntoView({ block: 'start' }); return true;");
  await shot("04-chain");
  await ui("h.q('[data-testid=fallback-chain]').scrollIntoView({ block: 'center' }); return true;");
  await shot("05-fallbacks");

  note("chat with the agent on free models");
  await backToChat();
  const t = await app.ask("What is 17 + 25? Reply with just the number.", {
    timeoutMs: 10 * 60_000,
  });
  await shot("06-free-chat");
  check("agent", "a turn completes on free models", t.status === "done" && /\b42\b/.test(t.text), `${t.status} in ${Math.round(t.durationMs / 1000)} s: ${t.text.slice(0, 100)}`);

  note("RLM on free models: REPL cells and memory recall");
  const facts = await app.cdp.invoke("memory:list", { query: "" });
  if (!facts.some((f) => f.text === MEMORY_FACT)) await app.cdp.invoke("memory:add", { text: MEMORY_FACT, category: "preference" });
  await app.newChat().catch(() => undefined);
  const rlm = await app.ask(REPL_PROMPT, { timeoutMs: 10 * 60_000 });
  await shot("06b-rlm-free");
  const cellsRun = rlm.events.filter((e) => e.type === "cell_done");
  const recalled = rlm.events.find((e) => e.type === "memory_recalled");
  const usage = rlm.events.filter((e) => e.type === "context_usage");
  check("rlm", "the agent's prompt is sized to the model's real window", usage.length > 0 && usage.every((u) => u.usedTokens < u.contextLength && u.contextLength <= primary.contextLength), usage.map((u) => `${u.usedTokens}/${u.contextLength}`).join(" "));
  if (toolUsers.length)
    check("rlm", "the agent ran its work in the REPL on free models", cellsRun.some((c) => c.ok), `${cellsRun.length} cells, ok: ${cellsRun.filter((c) => c.ok).length}`);
  else skip("rlm", "the agent ran its work in the REPL on free models", `no chained model follows the cell protocol; ${cellsRun.length} cells`);
  check(
    "rlm",
    "memory recall found the stored fact",
    (recalled?.items ?? []).some((m) => m.text.includes("BLUEJAY")),
    (recalled?.items ?? []).map((m) => m.text.slice(0, 50)).join(" | "),
  );
  check("rlm", "the answer uses the remembered fact", rlm.status === "done" && /BLUEJAY/i.test(rlm.text), `${rlm.status}: ${rlm.text.slice(0, 160)}`);
  if (toolUsers.length) check("rlm", "the answer uses the REPL result", /338,?350/.test(rlm.text), rlm.text.slice(0, 160));
  else skip("rlm", "the answer uses the REPL result", rlm.text.slice(0, 120));

  note("fault injection: a primary that drops the stream mid-reply");
  const proxy = await faultProxy(evalKey(port), port, chain[0].id);
  await app.cdp.invoke("settings:update", {
    patch: {
      providers: [
        ...s.providers.filter((p) => p.id !== "flaky"),
        {
          id: "flaky",
          kind: "openai_compat",
          label: "Flaky primary",
          baseUrl: `http://127.0.0.1:${proxy.port}/v1`,
          hasKey: false,
        },
      ],
      models: [
        ...s.models.filter((m) => m.providerId !== "flaky"),
        {
          id: "flaky:primary",
          providerId: "flaky",
          name: "primary",
          label: "Flaky primary",
          kind: "chat",
          contextLength: primary.contextLength,
          maxOutputTokens: primary.maxOutputTokens,
          dim: null,
          ...(primary.systemAsUser ? { systemAsUser: true } : {}),
        },
      ],
    },
  });
  await app.cdp.invoke("settings:update", {
    patch: { roles: { agent: "flaky:primary", agentFallbacks: chainIds } },
  });
  proxy.mode = "cut-once";
  await app.newChat().catch(() => undefined);
  const cut = await app.ask(PRIME_PROMPT, { timeoutMs: 10 * 60_000 });
  await shot("07-cut-mid-reply");
  const sw = cut.events.filter((e) => e.type === "model_switch");
  const shown = await ui("return [...document.querySelectorAll('[data-testid=model-switch]')].map((e) => e.innerText)");
  check("failover", "the proxy really cut a reply mid-stream", proxy.cuts === 1, `cut after ${proxy.partial.length} chars: ${JSON.stringify(proxy.partial.slice(0, 80))}`);
  check(
    "failover",
    "the turn moved to the next model and continued the reply",
    sw.some((e) => e.resumed && e.from === "Flaky primary"),
    JSON.stringify(sw),
  );
  check("failover", "the turn still finished with the right answer, said once", cut.status === "done" && cut.text.split(PRIMES).length === 2, `${cut.status}: ${cut.text.slice(0, 200)}`);
  check(
    "failover",
    "the chat shows the switch",
    shown.some((x) => /continued the reply/.test(x)),
    shown.join(" | "),
  );

  note("fault injection: a rate-limited primary (HTTP 429 before any output)");
  await app.cdp.invoke("settings:update", {
    patch: { failover: { cooldownSec: s.failover.cooldownSec + 1 } },
  });
  proxy.mode = "429";
  await app.newChat().catch(() => undefined);
  const t0 = Date.now();
  const lim = await app.ask("Name the capital of France in one word.", {
    timeoutMs: 10 * 60_000,
  });
  await shot("08-rate-limited");
  const sw2 = lim.events.filter((e) => e.type === "model_switch");
  check(
    "failover",
    "a rate limit switches to the backup without stopping",
    lim.status === "done" && /paris/i.test(lim.text) && proxy.limited >= 1,
    `${lim.status} in ${Math.round((Date.now() - t0) / 1000)} s, ${proxy.limited} limited: ${lim.text.slice(0, 80)}`,
  );
  check(
    "failover",
    "the switch names the reason and does not resume",
    sw2.some((e) => !e.resumed && /rate limited/.test(e.reason)),
    JSON.stringify(sw2),
  );
  check("failover", "the cooled-down primary is skipped for the rest of the turn", proxy.limited === 1, `${proxy.limited} requests hit the limited primary`);

  await app.cdp.invoke("settings:update", { patch: { failover: { cooldownSec: s.failover.cooldownSec + 2 } } });
  proxy.cuts = 0;
  await app.newChat().catch(() => undefined);
  if (toolUsers.length) {
    note("fault injection mid-job: the step after a REPL cell is cut off; the backup must remember the cell and the memory");
    proxy.mode = "cut-nth";
    proxy.cutOn = proxy.requests + 2;
    proxy.cutAfter = 25;
    const job = await app.ask(REPL_PROMPT, { timeoutMs: 10 * 60_000 });
    await shot("09-cut-mid-job");
    const jobCells = job.events.filter((e) => e.type === "cell_done");
    const jobSw = job.events.filter((e) => e.type === "model_switch");
    check("memory", "the second step of the job was cut mid-reply after a REPL cell ran", proxy.cuts === 1 && jobCells.length >= 1, `cuts ${proxy.cuts}, cells: ${jobCells.length}, partial ${JSON.stringify(proxy.partial.slice(0, 60))}`);
    check("memory", "the backup continued the job instead of restarting it", jobSw.some((e) => e.resumed) && jobCells.filter((c) => c.ok).length <= 2, `${JSON.stringify(jobSw.map((e) => `${e.from}→${e.to} (${e.reason}${e.resumed ? ", resumed" : ""})`))}, ${jobCells.length} cells`);
    check("memory", "code continued across the cut runs without error", jobCells.length >= 1 && jobCells.every((c) => c.ok), jobCells.map((c) => (c.ok ? "ok" : `failed: ${String(c.error ?? "").slice(0, 80)}`)).join(" | "));
    check("memory", "after the switch the answer still has the REPL result and the remembered fact", job.status === "done" && /338,?350/.test(job.text) && /BLUEJAY/i.test(job.text), `${job.status}: ${job.text.slice(0, 160)}`);
  } else {
    note("fault injection: a reply that must use the memory is cut off; the backup continues it with the memory");
    proxy.mode = "cut-once";
    proxy.cutAfter = 12;
    const job = await app.ask("In two sentences: first say hello, then tell me my deploy target.", { timeoutMs: 10 * 60_000 });
    await shot("09-cut-memory");
    const jobSw = job.events.filter((e) => e.type === "model_switch");
    check("memory", "the reply was cut mid-stream and continued on the backup", proxy.cuts === 1 && jobSw.some((e) => e.resumed), `partial ${JSON.stringify(proxy.partial)}; ${JSON.stringify(jobSw.map((e) => `${e.from}→${e.to} (${e.reason})`))}`);
    check("memory", "after the switch the answer still has the remembered fact", job.status === "done" && /BLUEJAY/i.test(job.text), `${job.status}: ${job.text.slice(0, 160)}`);
  }
  proxy.server.close();
  await app.cdp.invoke("settings:update", {
    patch: {
      roles: { agent: chainIds[0], agentFallbacks: chainIds.slice(1) },
      failover: { cooldownSec: s.failover.cooldownSec },
    },
  });
  s = await settingsNow();
  await app.cdp.invoke("settings:update", {
    patch: {
      providers: s.providers.filter((p) => p.id !== "flaky"),
      models: s.models.filter((m) => m.providerId !== "flaky"),
    },
  });

  note("restart the IDE; the gateway comes back on its own");
  await app.stop();
  const down = await fetch(`http://127.0.0.1:${port}/api/health`, {
    signal: AbortSignal.timeout(3000),
  }).then(
    () => false,
    () => true,
  );
  check("restart", "closing the IDE stops the gateway", down, down ? "port closed" : "still answering");
  app = await new IdeApp({
    userData,
    workspace: ws,
    logFile: path.join(dir, "app-2.log"),
  }).start();
  await until("return window.rlm.invoke('omniroute:state', {}).then((s) => s.phase === 'running' || s.phase === 'error')", 5 * 60_000, "auto start");
  v = await state();
  check("restart", "the gateway starts with the IDE", v.phase === "running", v.error ?? "running");
  await app.newChat().catch(() => undefined);
  const t2 = await app.ask("What is 6 times 7? Reply with just the number.", {
    timeoutMs: 10 * 60_000,
  });
  check("restart", "the agent answers on free models after the restart", t2.status === "done" && /\b42\b/.test(t2.text), t2.text.slice(0, 80));

  note("no gateway key in plain text in the IDE's own files");
  const leaks = [];
  const walk = (d) => {
    for (const n of readdirSync(d)) {
      const p = path.join(d, n);
      const st = statSync(p);
      if (st.isDirectory()) {
        // The gateway's own data and code are its business; the IDE's files must not hold keys.
        if (p === path.join(userData, "omniroute") || n === "node_modules") continue;
        walk(p);
      } else if (st.size < 20_000_000 && /\bsk-[A-Za-z0-9_-]{20,}/.test(readFileSync(p, "latin1"))) leaks.push(path.relative(userData, p));
    }
  };
  walk(userData);
  check("security", "no API key appears in plain text in the IDE profile", leaks.length === 0, leaks.join(", "));
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
writeFileSync(path.join(dir, "results.json"), JSON.stringify({ checks, skipped, fatal: fatal ? String(fatal.stack ?? fatal) : null }, null, 2));
console.log(`\n${passed}/${checks.length} checks passed${skipped.length ? `, ${skipped.length} skipped` : ""}${fatal ? ` · stopped early: ${fatal.message}` : ""}\nresults: ${path.relative(REPO, dir)}`);
process.exit(fatal || passed !== checks.length ? 1 : 0);
