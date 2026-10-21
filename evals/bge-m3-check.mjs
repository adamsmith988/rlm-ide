// Checks the built-in bge-m3 embedder on real weights: shape, normalisation, agreement with the reference pipeline,
// semantic and cross-language similarity, and bounded memory on a huge input.
// Usage: npx jiti evals/bge-m3-check.mjs [modelsDir]
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { BgeM3Embedder } from "../packages/inference/src/embed_bge.ts";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const transformers = createRequire(path.join(root, "packages/inference/package.json")).resolve("@huggingface/transformers");
const { pipeline } = await import(pathToFileURL(transformers).href);
const models = path.resolve(process.argv[2] ?? path.join(root, "resources/models"));
const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok });
  console.log(`${ok ? "PASS" : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
};
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

const e = new BgeM3Embedder(models, "bge-m3", 1024, () => 1024);
const texts = [
  "The cat is sleeping on the sofa.",
  "A kitten naps on the couch.",
  "Quarterly revenue grew by five percent.",
  "Where is the train station?",
  "Wo ist der Bahnhof?",
  "ایستگاه قطار کجاست؟",
  "def add(a, b): return a + b",
  "function that sums two numbers in Python",
];
const v = await e.embed(texts);
check("one 1024-dimensional vector per text", v.length === texts.length && v.every((x) => x.length === 1024), `${v.length} x ${v[0]?.length}`);
check("vectors are unit length", v.every((x) => Math.abs(Math.sqrt(dot(x, x)) - 1) < 1e-3));

const ref = await pipeline("feature-extraction", path.join(models, "bge-m3"), { dtype: "q8", device: "cpu", local_files_only: true });
// Same batch as above: int8 dynamic quantisation makes results depend slightly on the padding a batch needs.
const r = (await ref(texts, { pooling: "cls", normalize: true })).tolist();
const agree = Math.min(...r.map((x, i) => dot(x, v[i])));
check("matches the reference pipeline on short texts", agree > 0.9999, `min cosine ${agree.toFixed(6)}`);

const sim = (a, b) => dot(v[a], v[b]);
check("paraphrases score above unrelated text", sim(0, 1) > sim(0, 2) + 0.15, `cat/kitten ${sim(0, 1).toFixed(3)}, cat/revenue ${sim(0, 2).toFixed(3)}`);
check("German and Persian translations match the English question", sim(3, 4) > 0.75 && sim(3, 5) > 0.7 && sim(3, 4) > sim(3, 0) + 0.2, `de ${sim(3, 4).toFixed(3)}, fa ${sim(3, 5).toFixed(3)}, unrelated ${sim(3, 0).toFixed(3)}`);
check("code matches its description", sim(6, 7) > sim(6, 2) + 0.15, `code/desc ${sim(6, 7).toFixed(3)}, code/revenue ${sim(6, 2).toFixed(3)}`);

const before = process.memoryUsage().rss;
let peak = before;
const timer = setInterval(() => (peak = Math.max(peak, process.memoryUsage().rss)), 20);
const huge = Array.from({ length: 6000 }, (_, i) => `Line ${i}: the deployment log shows request ${i} served in ${i % 97} ms.`).join("\n");
const t0 = Date.now();
const [h] = await e.embed([huge, huge.slice(0, 2000)]);
clearInterval(timer);
const grewMb = Math.round((peak - before) / 2 ** 20);
check("a 400 KB message embeds with bounded memory", h.length === 1024 && grewMb < 1500, `${huge.length} chars in ${Date.now() - t0} ms, RSS grew ${grewMb} MB`);

const failed = results.filter((x) => !x.ok).length;
console.log(`${results.length - failed}/${results.length} passed`);
process.exit(failed ? 1 : 0);
