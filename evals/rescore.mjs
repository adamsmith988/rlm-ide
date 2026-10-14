#!/usr/bin/env node
/** Re-runs render probes and deterministic checks on an existing results folder (no agent, no judge).
 *   node evals/rescore.mjs evals/results/<stamp> [--scenarios music-app] */
import { readFileSync } from "node:fs";
import path from "node:path";
import { renderPage } from "./lib/render.mjs";

const dir = path.resolve(process.argv[2] ?? "");
const only = process.argv.includes("--scenarios") ? process.argv[process.argv.indexOf("--scenarios") + 1].split(",") : null;
const app = { cdp: { invoke: async () => [] } };

for (const id of only ?? ["quotes-app", "music-app", "fix-bugs", "add-feature", "memory", "long-session"]) {
  const scenario = (await import(`./scenarios/${id}.mjs`)).default;
  const ws = path.join(dir, id, "ws");
  let saved;
  try {
    saved = JSON.parse(readFileSync(path.join(dir, id, "result.json"), "utf8"));
  } catch {
    continue;
  }
  const results = (saved.events ?? []).map((events, i) => ({ events, text: saved.turns?.[i]?.answer ?? "", status: saved.turns?.[i]?.status }));
  const render = scenario.render ? await renderPage(path.join(ws, scenario.render.entry), path.join(dir, id, "app-rescore.png"), scenario.render.probe) : null;
  const checks = await scenario.checks(ws, { results, render, app });
  console.log(`${id}: ${checks.filter((c) => c.pass).length}/${checks.length}`);
  for (const c of checks) if (!c.pass) console.log(`  ✗ ${c.name}${c.detail ? ` — ${c.detail}` : ""}`);
}
