#!/usr/bin/env node
/**
 * Dogfood eval runner. Launches the built RLM IDE per scenario with an isolated profile and
 * workspace, types the prompts into the agent composer, then scores what the agent produced.
 *
 *   node evals/run.mjs [--scenarios quotes-app,fix-bugs] [--model api|local] [--tag name]
 *
 * Env: EVAL_API_URL (default http://localhost:8100/v1), EVAL_API_MODEL (default Qwen3.5-4B),
 *      EVAL_API_KEY, EVAL_CONTEXT (default 32768), EVAL_JUDGE_URL / EVAL_JUDGE_MODEL / EVAL_JUDGE_KEY.
 */
import { execFileSync } from "node:child_process";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { IdeApp, REPO, turnMetrics } from "./lib/app.mjs";
import { sleep } from "./lib/cdp.mjs";
import { filesFor } from "./lib/checks.mjs";
import { judge } from "./lib/judge.mjs";
import { renderPage } from "./lib/render.mjs";

const ALL = ["quotes-app", "music-app", "fix-bugs", "add-feature", "memory", "long-session"];

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : fallback;
}

const scenarioIds = arg("scenarios", ALL.join(",")).split(",");
const modelKind = arg("model", "api");
const tag = arg("tag", "");
const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19) + (tag ? `-${tag}` : "");
const outRoot = path.join(REPO, "evals/results", stamp);

const API_MODEL = process.env.EVAL_API_MODEL ?? "Qwen3.5-4B";
const API_URL = process.env.EVAL_API_URL ?? "http://localhost:8100/v1";

function settingsFor(scenario) {
  const ctx = scenario.contextLength ?? Number(process.env.EVAL_CONTEXT ?? 32768);
  const builtinModels = [
    { id: "builtin:frognano-4b", providerId: "builtin", name: "frognano-4b", label: "FrogNano 4B (local)", kind: "chat", contextLength: null, maxOutputTokens: null, dim: null },
    { id: "builtin:bge-m3", providerId: "builtin", name: "bge-m3", label: "bge-m3 (local)", kind: "embed", contextLength: 8192, maxOutputTokens: null, dim: 1024 },
  ];
  const apiModel = { id: `eval-api:${API_MODEL}`, providerId: "eval-api", name: API_MODEL, label: `${API_MODEL} (eval)`, kind: "chat", contextLength: ctx, maxOutputTokens: Math.min(8192, Math.floor(ctx / 4)), dim: null };
  const ask = scenario.approvals === "ask";
  return {
    providers: [
      { id: "builtin", kind: "local_onnx", label: "Built-in (local)", baseUrl: "", hasKey: false },
      { id: "eval-api", kind: "openai_compat", label: "Eval API", baseUrl: API_URL, hasKey: !!process.env.EVAL_API_KEY },
    ],
    models: [...builtinModels, apiModel],
    roles: { agent: modelKind === "local" ? "builtin:frognano-4b" : apiModel.id, summarizer: "", embed: "builtin:bge-m3" },
    local: { localContextCap: scenario.contextLength ?? 32768 },
    tools: { workspaceAllow: ask ? [] : ["*"], autoApplyEdits: !ask },
    autoUpdate: false,
  };
}

function seedWorkspace(ws, seed = {}) {
  rmSync(ws, { recursive: true, force: true });
  mkdirSync(ws, { recursive: true });
  for (const [rel, content] of Object.entries(seed)) {
    mkdirSync(path.dirname(path.join(ws, rel)), { recursive: true });
    writeFileSync(path.join(ws, rel), content);
  }
  const git = (...a) => execFileSync("git", a, { cwd: ws, stdio: "ignore" });
  git("init", "-q");
  git("add", "-A");
  git("-c", "user.email=eval@rlm", "-c", "user.name=eval", "commit", "-q", "--allow-empty", "-m", "seed");
}

async function runScenario(id) {
  const scenario = (await import(`./scenarios/${id}.mjs`)).default;
  const dir = path.join(outRoot, id);
  const ws = path.join(dir, "ws");
  mkdirSync(dir, { recursive: true });
  seedWorkspace(ws, scenario.seed);
  console.log(`\n▶ ${id}: ${scenario.title}`);

  const env = process.env.EVAL_API_KEY ? { RLM_API_KEY: process.env.EVAL_API_KEY, RLM_API_PROVIDER: "eval-api" } : {};
  Object.assign(process.env, env);
  const app = new IdeApp({ userData: path.join(dir, "userdata"), workspace: ws, settings: settingsFor(scenario), logFile: path.join(dir, "app.log") });
  const results = [];
  let checks = [];
  let render = null;
  let error = null;
  try {
    await app.start();
    for (const [i, turn] of scenario.turns.entries()) {
      if (turn.waitBeforeMs) await sleep(turn.waitBeforeMs);
      if (turn.newChat) await app.newChat();
      const r = await app.ask(turn.prompt, { onApproval: scenario.approvals === "ask" ? async () => true : undefined });
      results.push(r);
      await app.screenshot(path.join(dir, `ide-turn-${i + 1}.png`));
      const m = turnMetrics(r);
      console.log(`  turn ${i + 1}: ${m.status} in ${m.durationSec}s, ${m.cells} cells (${m.cellErrors} failed), ${m.toolCalls} tools (${m.toolErrors} failed), ${m.approvals} approvals, ${m.compactions} compactions`);
    }
    if (scenario.render) render = await renderPage(path.join(ws, scenario.render.entry), path.join(dir, "app.png"), scenario.render.probe);
    checks = await scenario.checks(ws, { results, render, app });
  } catch (e) {
    error = String(e?.stack ?? e);
    console.log(`  ERROR ${error}`);
  } finally {
    await app.stop();
  }

  let verdict = null;
  if (scenario.judge && !error) {
    const files = filesFor(ws, scenario.judge.files).filter((f) => !f.path.startsWith("hidden"));
    verdict = files.length
      ? await judge({ task: scenario.judge.task, rubric: scenario.judge.rubric, files, render, answer: results[results.length - 1]?.text ?? "" })
      : { scores: scenario.judge.rubric.map(() => 1), mean: 1, issues: ["no files were produced"], summary: "Nothing was produced.", model: "n/a" };
  }
  const passed = checks.filter((c) => c.pass).length;
  for (const c of checks) console.log(`  ${c.pass ? "✓" : "✗"} ${c.name}${c.pass || !c.detail ? "" : ` — ${c.detail}`}`);
  if (verdict) console.log(`  judge ${verdict.mean?.toFixed(2) ?? "n/a"} [${verdict.scores.join(", ")}] ${verdict.summary}`);

  const summary = {
    id,
    title: scenario.title,
    model: modelKind === "local" ? "builtin:frognano-4b" : API_MODEL,
    error,
    checks,
    passed,
    total: checks.length,
    judge: verdict,
    turns: results.map((r) => ({ ...turnMetrics(r), answer: r.text.slice(0, 2000) })),
    render: render && { loaded: render.loaded, console: render.console, errors: render.errors, probe: render.probe, counts: render.counts },
  };
  writeFileSync(path.join(dir, "result.json"), JSON.stringify({ ...summary, events: results.map((r) => r.events) }, null, 2));
  return summary;
}

mkdirSync(outRoot, { recursive: true });
// Workspaces must not inherit evals/package.json ("type": "module"); a user's project would not.
writeFileSync(path.join(REPO, "evals/results/package.json"), '{ "type": "commonjs" }\n');
const rows = [];
for (const id of scenarioIds) rows.push(await runScenario(id));

const sum = (f) => rows.reduce((n, r) => n + f(r), 0);
const lines = [
  `# Eval run ${stamp}`,
  "",
  `Agent model: ${rows[0]?.model ?? "?"}. Judge: ${rows.find((r) => r.judge)?.judge?.model ?? "none"}.`,
  "",
  "| Scenario | Checks | Judge (1-5) | Time | Cells (failed) | Tool calls (failed) | Compactions |",
  "| --- | --- | --- | --- | --- | --- | --- |",
  ...rows.map((r) => {
    const t = r.turns;
    const s = (k) => t.reduce((n, x) => n + x[k], 0);
    return `| ${r.id} | ${r.error ? "error" : `${r.passed}/${r.total}`} | ${r.judge?.mean?.toFixed(2) ?? "–"} | ${Math.round(s("durationSec"))}s | ${s("cells")} (${s("cellErrors")}) | ${s("toolCalls")} (${s("toolErrors")}) | ${s("compactions")} |`;
  }),
  "",
  `**Total checks passed: ${sum((r) => r.passed)}/${sum((r) => r.total)}**`,
  "",
  ...rows.flatMap((r) => [
    `## ${r.id}`,
    "",
    ...r.checks.map((c) => `- ${c.pass ? "✓" : "✗"} ${c.name}${c.pass || !c.detail ? "" : ` — ${c.detail}`}`),
    ...(r.judge ? ["", `Judge: ${r.judge.summary}`, ...r.judge.issues.map((i) => `- ${i}`)] : []),
    ...(r.error ? ["", "```", r.error, "```"] : []),
    "",
  ]),
];
writeFileSync(path.join(outRoot, "summary.md"), lines.join("\n"));
console.log(`\nSummary: ${path.join(outRoot, "summary.md")}`);
