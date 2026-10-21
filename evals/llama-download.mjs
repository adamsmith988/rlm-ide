// Live check of the prebuilt llama.cpp download: node --experimental-transform-types evals/llama-download.mjs [cpu|auto|gpu] [model.gguf]
// Downloads into a temp hub folder, runs llama-server --version, and with a GGUF serves it and asks one question.
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";
import { Toolchain } from "../packages/hub/src/toolchain.ts";

const want = process.argv[2] ?? "cpu";
const gguf = process.argv[3] ?? "";
const dir = mkdtempSync(path.join(tmpdir(), "llama-dl-"));
let ok = 0;
let bad = 0;
const check = (name, pass, detail = "") => {
  pass ? ok++ : bad++;
  console.log(`${pass ? "PASS" : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
};

const tc = new Toolchain({
  dir,
  scriptsDir: dir,
  basePythons: () => [],
  configuredPython: () => "",
  configuredLlamaCpp: () => "",
  llamaRelease: () => ({ repo: "ggml-org/llama.cpp", release: process.env.LLAMA_RELEASE ?? "" }),
  onChange: () => {},
});

let server = null;
try {
  const t0 = Date.now();
  const timer = setInterval(() => {
    const log = tc.installing?.log.split("\n").filter(Boolean).at(-1);
    if (log) process.stdout.write(`  … ${log}\n`);
  }, 5000);
  try {
    await tc.downloadLlamaCpp(want);
  } finally {
    clearInterval(timer);
  }
  const l = await tc.llamaCpp();
  check("download finished", true, `${Math.round((Date.now() - t0) / 1000)} s`);
  check("llama-server found in the IDE's own folder", !!l.server && l.server.startsWith(dir), l.server ?? "");
  check("llama-quantize found", !!l.quantize, l.quantize ?? "");
  check("llama-server --version runs", !!l.version && !l.error, l.version ?? l.error ?? "");
  check("backend recorded", !!l.backend, String(l.backend));
  check("converter script beside it", !!l.convert, l.convert ?? "");
  check("converter knows architectures", l.convertArchitectures.includes("LlamaForCausalLM"), `${l.convertArchitectures.length} architectures`);
  check("quantization types listed", l.quantTypes.some((q) => q.id === "Q4_K_M"), `${l.quantTypes.length} types`);

  if (gguf && l.server) {
    const port = 18000 + Math.floor(Math.random() * 1000);
    server = spawn(l.server, ["-m", gguf, "--port", String(port), "-c", "2048", "-ngl", want === "cpu" ? "0" : "999"], { env: { ...process.env, ...l.env }, stdio: ["ignore", "pipe", "pipe"] });
    let out = "";
    server.stdout.on("data", (d) => (out += d));
    server.stderr.on("data", (d) => (out += d));
    let up = false;
    for (let i = 0; i < 120 && !up; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      up = await fetch(`http://127.0.0.1:${port}/health`).then((r) => r.ok, () => false);
      if (server.exitCode !== null) break;
    }
    check("serves the model", up, up ? "" : out.split("\n").slice(-6).join(" | "));
    if (up) {
      const t1 = Date.now();
      const r = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: [{ role: "user", content: "What is 2+3? Answer with just the number." }], max_tokens: 64 }),
      }).then((r) => r.json());
      const text = r.choices?.[0]?.message?.content ?? "";
      const tps = r.timings?.predicted_per_second;
      check("answers a question", /5/.test(text), `${JSON.stringify(text.slice(0, 60))} in ${Date.now() - t1} ms${tps ? `, ${tps.toFixed(1)} tok/s` : ""}`);
      const offload = /offloaded [1-9]\d*\/\d+ layers to GPU|(?:CUDA|Vulkan|Metal)\d* model buffer size\s*=\s*[\d.]+ MiB/.exec(out);
      if (want !== "cpu") check("weights on the GPU", !!offload, offload?.[0] ?? "no GPU buffer line");
    }
  }
} catch (e) {
  check("download", false, e instanceof Error ? e.message : String(e));
  if (tc.installing) console.log(tc.installing.log.split("\n").slice(-15).join("\n"));
} finally {
  if (server && server.exitCode === null) server.kill();
  rmSync(dir, { recursive: true, force: true, maxRetries: 3 });
}
console.log(`\n${ok} passed, ${bad} failed`);
process.exit(bad ? 1 : 0);
