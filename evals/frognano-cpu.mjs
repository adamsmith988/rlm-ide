// Times the built-in model on the CPU outside the app: load, prompt processing and generation speed.
//   node --experimental-transform-types evals/frognano-cpu.mjs [promptTokens]
// Run it with Electron's own Node to match the app (its allocator ends the process on 2 GB allocations):
//   ELECTRON_RUN_AS_NODE=1 <electron> --experimental-transform-types evals/frognano-cpu.mjs 8000
import path from "node:path";
import { fileURLToPath } from "node:url";
import { cpuThreads } from "../packages/inference/src/cpu.ts";
import { LocalGenerator } from "../packages/inference/src/generate.ts";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../resources/models");
const promptTokens = Number(process.argv[2] ?? 1500);
const t0 = performance.now();
const gen = new LocalGenerator({
  modelRoot: root + path.sep,
  modelDir: "frognano-4b",
  device: "cpu",
  cpuThreads: await cpuThreads(Number(process.env.THREADS ?? 0)),
  onStatus: (s) => console.log(`  [${((performance.now() - t0) / 1000).toFixed(1)} s] ${s}`),
});
await gen.load();
console.log(`${process.versions.electron ? `electron ${process.versions.electron}` : `node ${process.version}`}`);
const filler = "The project keeps its notes in docs and its code in src. ".repeat(Math.ceil(promptTokens / 12));
const msgs = [
  { role: "system", content: `${filler}\nThe deploy target is BLUEJAY.` },
  { role: "user", content: "What is the deploy target? Then write one sentence about frogs." },
];
const t1 = performance.now();
let first = 0;
let text = "";
let result;
await new Promise((resolve) =>
  gen.handle({ type: "generate", id: "a", messages: msgs, enableThinking: false, temperature: 0, maxTokens: 48, contextLimit: 32768 }, (r) => {
    if (r.type === "chunk") {
      if (!first) first = performance.now();
      text += r.text;
    }
    if (r.type === "done" || r.type === "error") {
      result = r;
      resolve();
    }
  }),
);
const end = performance.now();
console.log(result);
console.log(`prompt processing: ${((first - t1) / 1000).toFixed(1)} s, generation: ${((end - first) / 1000).toFixed(1)} s`);
console.log(JSON.stringify(text));
const ok = result?.type === "done" && /BLUEJAY/i.test(text);
console.log(ok ? "PASS" : "FAIL");
process.exit(ok ? 0 : 1);
