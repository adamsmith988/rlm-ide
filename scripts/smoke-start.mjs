#!/usr/bin/env node
// Start-up test: launches the IDE with a throwaway profile and workspace and checks that it really works:
// the window opens the workspace, the terminal (node-pty) runs a command, memory (embedder + sqlite-vec) stores
// and recalls a fact, and an agent turn runs a Python REPL cell in the bundled runtime. The agent talks to a
// local mock model, so no network, keys or downloads are needed.
// Usage: node scripts/smoke-start.mjs [--exe "path/to/RLM IDE.exe"] [--out dir] [--timeout-sec 300] [--platform win32]
import http from "node:http";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { IdeApp, REPO } from "../evals/lib/app.mjs";
import { sleep } from "../evals/lib/cdp.mjs";

const arg = (name, fallback) => (process.argv.includes(`--${name}`) ? process.argv[process.argv.indexOf(`--${name}`) + 1] : fallback);
const exe = arg("exe", null);
const out = path.resolve(arg("out", path.join(REPO, "evals", "results", `smoke-${new Date().toISOString().replace(/[:.]/g, "-")}`)));
const timeoutMs = Number(arg("timeout-sec", "300")) * 1000;
// The platform the app runs on; differs from this script's when a Windows build runs under Wine.
const platform = arg("platform", process.platform);
const version = JSON.parse(readFileSync(path.join(REPO, "apps", "desktop", "package.json"), "utf8")).version;

const TASK = "RLM_SMOKE_TASK: run one REPL cell that prints a marker, then report it.";
const PY_MARK = "RLM_SMOKE_PY_OK";
const MEMORY_FACT = "RLM smoke test fact: the build colour is TEAL";

const checks = [];
const check = (name, ok, detail = "") => {
  checks.push({ name, ok: !!ok, detail: String(detail).slice(0, 400) });
  console.log(`  ${ok ? "PASS" : "FAIL"} ${name}${detail ? ` — ${String(detail).slice(0, 200)}` : ""}`);
};

/** OpenAI-compatible mock: asks for one REPL cell, then answers once the cell's output comes back. */
function mockModel() {
  const seen = { requests: 0, sawOutput: false };
  const server = http.createServer((req, res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      if (!req.url.endsWith("/chat/completions")) {
        res.writeHead(404).end();
        return;
      }
      seen.requests++;
      const text = JSON.stringify(JSON.parse(body || "{}").messages ?? []);
      let reply = "ok";
      if (text.includes(PY_MARK)) {
        seen.sawOutput = true;
        reply = `Smoke test passed: the REPL printed ${PY_MARK}.`;
      } else if (text.includes("RLM_SMOKE_TASK")) {
        reply = '```repl\nimport platform, sys\nprint("RLM_SMOKE_" + "PY_OK", 6 * 7, platform.system(), sys.version.split()[0])\n```';
      }
      res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache" });
      for (let i = 0; i < reply.length; i += 24) res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: reply.slice(i, i + 24) } }] })}\n\n`);
      res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: "stop" }] })}\n\n`);
      res.end("data: [DONE]\n\n");
    });
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve({ server, seen, port: server.address().port })));
}

mkdirSync(out, { recursive: true });
const scratch = mkdtempSync(path.join(os.tmpdir(), "rlm-smoke-"));
const userData = path.join(scratch, "profile");
const workspace = path.join(scratch, "workspace");
mkdirSync(workspace, { recursive: true });
writeFileSync(path.join(workspace, "README.md"), "# Smoke test workspace\n");

console.log(`▶ start-up test of ${exe ? `packaged app ${exe}` : "the development build"}`);
const mock = await mockModel();
const app = new IdeApp({ userData, workspace, exe, logFile: path.join(out, "app.log") });
let fatal = null;
const deadline = setTimeout(() => {
  console.error(`start-up test timed out after ${timeoutMs / 1000} s`);
  app.stop().finally(() => process.exit(2));
}, timeoutMs);
try {
  const t0 = Date.now();
  await app.start();
  check("the window opens the workspace and the composer is ready", true, `${((Date.now() - t0) / 1000).toFixed(1)} s`);

  const info = await app.cdp.invoke("app:info");
  check("the app reports its version and platform", info.version === version && info.platform === platform, `${info.version} on ${info.platform}, sandbox ${info.sandbox?.mode}`);

  await app.cdp.eval("window.__smokePty = ''; window.rlm.onPtyData((m) => { window.__smokePty += m.data; }); true");
  const ptyId = await app.cdp.invoke("pty:create", { cols: 120, rows: 30 });
  await sleep(1500);
  // Windows: the first line works in PowerShell, the second in cmd.exe (the fallback shell); each prints the marker
  // only when actually run, never as the echoed command.
  for (const line of platform === "win32" ? ["echo RLM_SMOKE_PTY_$(6*7)", "echo RLM_SMOKE_PTY_^42"] : ["echo RLM_SMOKE_PTY_$((6*7))"]) {
    await app.cdp.invoke("pty:write", { id: ptyId, data: `${line}\r` });
    await sleep(500);
  }
  let pty = "";
  for (let i = 0; i < 40 && !pty.includes("RLM_SMOKE_PTY_42"); i++) {
    await sleep(500);
    pty = await app.cdp.eval("window.__smokePty");
  }
  check("the terminal runs a shell command", pty.includes("RLM_SMOKE_PTY_42"), pty.replace(/\x1b\][^\x07\x1b]*(\x07|\x1b\\)/g, "").replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "").trim().slice(-160));
  await app.cdp.invoke("pty:kill", { id: ptyId }).catch(() => undefined);

  await app.cdp.invoke("memory:add", { text: MEMORY_FACT, category: "preference" });
  const facts = await app.cdp.invoke("memory:list", { query: "build colour" });
  check("memory stores and recalls a fact", facts.some((f) => f.text === MEMORY_FACT), `${facts.length} found`);

  const s = await app.cdp.invoke("settings:get");
  await app.cdp.invoke("settings:update", {
    patch: {
      providers: [...s.providers.filter((p) => p.id !== "smoke"), { id: "smoke", kind: "openai_compat", label: "Smoke mock", baseUrl: `http://127.0.0.1:${mock.port}/v1`, hasKey: false }],
      models: [...s.models.filter((m) => m.providerId !== "smoke"), { id: "smoke:mock", providerId: "smoke", name: "mock", label: "Smoke mock", kind: "chat", contextLength: 32768, maxOutputTokens: 2048, dim: null }],
    },
  });
  await app.cdp.invoke("settings:update", { patch: { roles: { agent: "smoke:mock", agentFallbacks: [] } } });
  const turn = await app.ask(TASK, { timeoutMs: Math.max(60_000, timeoutMs - (Date.now() - t0) - 30_000), onApproval: async () => true });
  const cells = turn.events.filter((e) => e.type === "cell_done");
  check("an agent turn runs a Python REPL cell in the bundled runtime", cells.some((c) => c.ok) && mock.seen.sawOutput, `${cells.length} cells, ok: ${cells.filter((c) => c.ok).length}${cells.find((c) => !c.ok)?.error ? `, error: ${cells.find((c) => !c.ok).error}` : ""}`);
  check("the turn finishes with the answer", turn.status === "done" && turn.text.includes(PY_MARK), `${turn.status}: ${turn.text.slice(0, 120)}`);
  await app.screenshot(path.join(out, "smoke.png")).catch(() => undefined);
  const errors = app.cdp.console.filter((c) => c.type === "exception");
  check("no uncaught errors in the window", errors.length === 0, errors.map((e) => e.text).join(" | "));
} catch (e) {
  fatal = e;
  console.error(`  FAIL ${e.message}`);
  await app.screenshot(path.join(out, "smoke-failed.png")).catch(() => undefined);
} finally {
  clearTimeout(deadline);
  await app.stop();
  mock.server.close();
}
await sleep(1000);
try {
  rmSync(scratch, { recursive: true, force: true, maxRetries: 5, retryDelay: 500 });
} catch {
  /* Windows can hold a file briefly after the app exits; the OS temp folder is cleaned eventually */
}
const passed = checks.filter((c) => c.ok).length;
const ok = !fatal && passed === checks.length && checks.length > 0;
writeFileSync(path.join(out, "smoke.json"), JSON.stringify({ target: exe ?? "dev build", ok, checks, fatal: fatal ? String(fatal.stack ?? fatal) : null }, null, 2));
console.log(`${ok ? "start-up test passed" : "start-up test FAILED"}: ${passed}/${checks.length} checks · ${path.relative(process.cwd(), out) || out}`);
process.exit(ok ? 0 : 1);
