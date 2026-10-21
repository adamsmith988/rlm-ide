// Measures the built-in FrogNano model in the real app: which device it runs on, why, and tokens per second.
//   node evals/frognano-speed.mjs [--device auto|webgpu|cpu] [--exe "C:\...\RLM IDE.exe"] [--min-tps 10]
// Without --exe it runs the dev build (pnpm --filter @rlm/desktop build). On Windows pass the installed exe.
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { IdeApp } from "./lib/app.mjs";
import { sleep } from "./lib/cdp.mjs";

const arg = (name, fallback) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : fallback;
};
const device = arg("device", "auto");
const minTps = Number(arg("min-tps", "10"));
const out = mkdtempSync(path.join(tmpdir(), "rlm-frog-"));
const ws = path.join(out, "ws");
mkdirSync(ws, { recursive: true });
writeFileSync(path.join(ws, "README.md"), "speed check\n");

const app = new IdeApp({
  userData: path.join(out, "profile"),
  workspace: ws,
  logFile: path.join(out, "app.log"),
  exe: arg("exe", null),
  settings: {
    providers: [{ id: "builtin", kind: "local_onnx", label: "Built-in (local)", baseUrl: "", hasKey: false }],
    models: [
      { id: "builtin:frognano-4b", providerId: "builtin", name: "frognano-4b", label: "FrogNano 4B (local)", kind: "chat", contextLength: null, maxOutputTokens: null, dim: null },
      { id: "builtin:bge-m3", providerId: "builtin", name: "bge-m3", label: "bge-m3 (local)", kind: "embed", contextLength: 8192, maxOutputTokens: null, dim: 1024 },
    ],
    roles: { agent: "builtin:frognano-4b", summarizer: "", embed: "builtin:bge-m3" },
    local: { preferredDevice: device, enableThinking: false, localMaxOutputTokens: 256, localContextCap: 8192 },
    autoUpdate: false,
  },
});

const status = () =>
  app.cdp.eval(`(() => { const b = document.querySelector('[data-testid=status-agent-model]'); return JSON.stringify({ text: b?.innerText ?? '', title: b?.title ?? '' }); })()`).then(JSON.parse);

let code = 1;
try {
  await app.start();
  const turn = app.ask("Write two sentences about frogs. Answer directly, do not run any code.", { timeoutMs: Number(arg("timeout", "480")) * 1000 });
  let last = "";
  let s = { text: "", title: "" };
  for (let done = false; !done; ) {
    done = await Promise.race([turn.then(() => true), sleep(3000).then(() => false)]);
    s = await status();
    if (s.text !== last) console.log(`  status: ${s.text}${s.title ? `  [${s.title}]` : ""}`);
    last = s.text;
  }
  const r = await turn;
  console.log(`reply (${r.status}, ${Math.round(r.durationMs / 1000)} s): ${JSON.stringify(r.text.slice(0, 160))}`);
  const evs = await app.cdp.invoke("chat:subscribe", { turnId: r.turnId, lastSeq: -1 }).catch(() => r.events);
  const counts = {};
  for (const e of evs) counts[e.type] = (counts[e.type] ?? 0) + 1;
  console.log(`events: ${JSON.stringify(counts)}`);
  for (const e of evs.filter((e) => !/delta|chunk/.test(e.type)).slice(-12)) console.log(`  ${e.type}: ${JSON.stringify(e).slice(0, 220)}`);
  const tps = Number(/([\d.]+) tok\/s/.exec(s.text)?.[1] ?? NaN);
  const on = /on (webgpu|cpu)/.exec(s.text)?.[1] ?? "unknown";
  console.log(`device: ${on}  speed: ${Number.isFinite(tps) ? `${tps} tok/s` : "not reported"}  detail: ${s.title || "-"}`);
  code = r.status === "done" && Number.isFinite(tps) && (on === "cpu" || tps >= minTps) ? 0 : 1;
  console.log(code ? "FAIL" : "PASS");
} catch (e) {
  console.log(`FAIL ${e instanceof Error ? e.message : e}`);
} finally {
  await app.stop();
  console.log(`log: ${path.join(out, "app.log")}`);
}
process.exit(code);
