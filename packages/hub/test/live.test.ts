/**
 * Live check of the hub against huggingface.co and real runtimes (no Electron).
 * Runs only with HUB_LIVE=1. HUB_LIVE_DIR keeps downloads and builds between runs.
 *   HUB_LIVE=1 HUB_LIVE_ONLY=transformers-chat npx vitest run packages/hub/test/live.test.ts
 */
import { existsSync, mkdirSync } from "node:fs";
import path from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { defaultSettings, type HubDeployRequest, type HubStateView, type Settings } from "@rlm/contracts";
import { HubService } from "../src";

const LIVE = process.env.HUB_LIVE === "1";
const ONLY = (process.env.HUB_LIVE_ONLY ?? "").split(",").filter(Boolean);
const ROOT = path.resolve(__dirname, "../../..");
const DIR = process.env.HUB_LIVE_DIR ?? path.join(ROOT, "evals/.runtimes/hub-live");

const settings: Settings = defaultSettings();
settings.hub.startTimeoutSec = 600;
// Only the restart test restores; otherwise every run would bring back the previous run's servers.
settings.hub.restoreDeployments = false;
const secrets = new Map<string, string>();
let last: HubStateView | null = null;
mkdirSync(DIR, { recursive: true });
const service = () =>
  new HubService({
    dir: DIR,
    scriptsDir: path.join(ROOT, "python/rlm_hub"),
    basePythons: () => ["python3"],
    settings: () => settings,
    updateSettings: (p) => Object.assign(settings, p),
    secrets: { get: (k) => secrets.get(k) ?? null, set: (k, v) => (v ? secrets.set(k, v) : secrets.delete(k)) },
    emit: (s) => (last = s),
    log: (m) => console.log(`[hub] ${m}`),
  });
let hub = service();

async function until<T>(fn: () => T | undefined | null | false, ms: number, what: string): Promise<T> {
  const end = Date.now() + ms;
  for (;;) {
    const v = fn();
    if (v) return v;
    if (Date.now() > end) throw new Error(`timed out waiting for ${what}`);
    await new Promise((r) => setTimeout(r, 500));
  }
}

async function fetchModel(repo: string, pick?: (paths: string[], defaults: string[]) => string[]): Promise<string> {
  const info = await hub.lookup(repo);
  const files = pick ? pick(info.files.map((f) => f.path), info.defaultSelection) : info.defaultSelection;
  // A run that was interrupted leaves a paused job; finish it before asking for more.
  for (const j of hub.view().jobs.filter((x) => x.repo === repo && x.status === "paused")) {
    hub.jobAction(j.id, "resume");
    await until(() => hub.view().jobs.find((x) => x.id === j.id && ["done", "error"].includes(x.status)), 30 * 60_000, `resumed download of ${repo}`);
  }
  const job = await hub.download(repo, "", files);
  const done = await until(() => hub.view().jobs.find((j) => j.id === job.id && ["done", "error"].includes(j.status)), 30 * 60_000, `download of ${repo}`);
  expect(done.error).toBeNull();
  return hub.view().models.find((m) => m.repo === repo)!.id;
}

async function deployAndCheck(req: Omit<HubDeployRequest, "dtype" | "contextLength" | "trustRemoteCode" | "label"> & Partial<HubDeployRequest>) {
  const dep = await hub.deploy({ dtype: "auto", contextLength: 0, trustRemoteCode: false, label: "", ...req } as HubDeployRequest);
  const end = await until(() => hub.view().deployments.find((d) => d.id === dep.id && ["running", "error"].includes(d.status)), 40 * 60_000, `deployment ${dep.servedName}`);
  if (end.status === "error") console.log(end.error, "\n", end.logTail);
  expect(end.status).toBe("running");
  const key = secrets.get(`provider:hub-${dep.id}`)!;
  expect(key).toMatch(/^rlm-hub-/);
  // Key enforcement: no key and a wrong key are refused.
  for (const auth of [undefined, "Bearer wrong-key-0000000000"]) {
    const r = await fetch(`${end.baseUrl}/models`, { headers: auth ? { Authorization: auth } : {} });
    expect(r.status).toBe(401);
  }
  const p = settings.providers.find((x) => x.id === `hub-${dep.id}`)!;
  expect(p).toMatchObject({ kind: "openai_compat", hasKey: true, baseUrl: end.baseUrl });
  expect(settings.models.some((m) => m.id === end.settingsModelId && m.kind === req.kind)).toBe(true);
  const t = await hub.test(dep.id);
  console.log(`[live] ${end.servedName}: ${t.detail}`);
  expect(t.ok).toBe(true);
  return end;
}

const run = (name: string) => (LIVE && (!ONLY.length || ONLY.includes(name)) ? it : it.skip);

describe("hub live", () => {
  afterAll(() => hub.close());

  it.runIf(LIVE)("boots and probes the tools", async () => {
    await hub.boot();
    expect(hub.view().tools.python?.packages.torch).toBeTruthy();
  }, 120_000);

  run("transformers-chat")("downloads a chat model and serves it with transformers", async () => {
    const id = await fetchModel("HuggingFaceTB/SmolLM2-135M-Instruct");
    const targets = await hub.targets(id);
    const tf = targets.find((t) => t.id === "transformers")!;
    expect(tf.available).toBe(true);
    await deployAndCheck({ modelId: id, target: "transformers", source: "weights", quant: "none", kind: "chat", device: "cpu" });
  }, 3_600_000);

  run("transformers-embed")("downloads an embedding model and serves it with transformers", async () => {
    const id = await fetchModel("sentence-transformers/all-MiniLM-L6-v2");
    const m = hub.view().models.find((x) => x.id === id)!;
    expect(m.profile.kind).toBe("embed");
    const d = await deployAndCheck({ modelId: id, target: "transformers", source: "weights", quant: "none", kind: "embed", device: "cpu" });
    expect(d.dim).toBe(384);
  }, 3_600_000);

  run("bnb")("quantizes on load with bitsandbytes 4-bit on the GPU", async () => {
    const id = await fetchModel("HuggingFaceTB/SmolLM2-135M-Instruct");
    const tf = (await hub.targets(id)).find((t) => t.id === "transformers")!;
    expect(tf.quantizations.weights!.map((q) => q.id)).toContain("bnb-4bit-nf4");
    await deployAndCheck({ modelId: id, target: "transformers", source: "weights", quant: "bnb-4bit-nf4", kind: "chat", device: "cuda" });
  }, 3_600_000);

  run("llama-build")("builds llama.cpp and reads its quantization types and converter", async () => {
    const tools = await hub.toolsAction("installLlamaCpp", [], "cpu");
    expect(tools.llamaCpp.server).toBeTruthy();
    expect(tools.llamaCpp.quantTypes.length).toBeGreaterThan(15);
    expect(tools.llamaCpp.convertArchitectures).toBeGreaterThan(50);
    console.log(`[live] llama.cpp ${tools.llamaCpp.version}: ${tools.llamaCpp.quantTypes.map((q) => q.id).join(" ")}`);
  }, 3_600_000);

  run("gguf-chat")("converts a chat model to GGUF, quantizes to Q4_K_M and serves it", async () => {
    const id = await fetchModel("HuggingFaceTB/SmolLM2-135M-Instruct");
    const l = (await hub.targets(id)).find((t) => t.id === "llamacpp")!;
    expect(l.needs).toEqual([]);
    expect(l.sources.map((s) => s.id)).toContain("convert");
    await deployAndCheck({ modelId: id, target: "llamacpp", source: "convert", quant: "Q4_K_M", kind: "chat", device: "cpu" });
  }, 3_600_000);

  run("gguf-embed")("converts an embedding model to GGUF F16 and serves embeddings", async () => {
    const id = await fetchModel("sentence-transformers/all-MiniLM-L6-v2");
    const d = await deployAndCheck({ modelId: id, target: "llamacpp", source: "convert", quant: "F16", kind: "embed", device: "cpu" });
    expect(d.dim).toBe(384);
  }, 3_600_000);

  run("gguf-imatrix")("takes the importance-matrix route for types that need one", async () => {
    const id = await fetchModel("HuggingFaceTB/SmolLM2-135M-Instruct");
    const d = await deployAndCheck({ modelId: id, target: "llamacpp", source: "convert", quant: "IQ2_XXS", kind: "chat", device: "cpu" });
    expect(d.steps.map((s) => s.label)).toContain("compute importance matrix");
  }, 3_600_000);

  run("gguf-repo")("serves a file from a GGUF-only repo as downloaded and re-quantized", async () => {
    const id = await fetchModel("Qwen/Qwen2.5-0.5B-Instruct-GGUF", (paths, defaults) => {
      const ggufs = paths.filter((p) => p.endsWith(".gguf"));
      expect(defaults.filter((p) => p.endsWith(".gguf"))).toEqual([]);
      const smallest = ggufs.find((p) => /q8_0/i.test(p)) ?? ggufs[0]!;
      return [...defaults, smallest];
    });
    const l = (await hub.targets(id)).find((t) => t.id === "llamacpp")!;
    const src = l.sources.find((s) => s.id.startsWith("gguf:"))!.id;
    expect(l.quantizations[src]![0]!.id).toBe("as-is");
    await deployAndCheck({ modelId: id, target: "llamacpp", source: src, quant: "as-is", kind: "chat", device: "cpu" });
    await deployAndCheck({ modelId: id, target: "llamacpp", source: src, quant: "Q4_K_M", kind: "chat", device: "cpu" });
  }, 3_600_000);

  run("onnx-tools")("installs optimum into its own environment without touching the main one", async () => {
    const before = (await hub.toolsAction("probe", [], "cpu")).python?.packages.transformers;
    const tools = await hub.toolsAction("installPackages", ["optimum-onnx[onnxruntime]"], "cpu", "onnx");
    expect(tools.onnxPython?.own).toBe(true);
    expect(tools.onnxPython?.onnxExportModelTypes).toBeGreaterThan(20);
    expect(tools.onnxPython?.onnxQuantizers.length).toBeGreaterThan(0);
    expect(tools.python?.packages.transformers).toBe(before);
    console.log(`[live] onnx env: transformers ${tools.onnxPython?.packages.transformers}, main keeps ${before}`);
  }, 3_600_000);

  run("onnx-repo")("serves the repo's own ONNX files with ONNX Runtime", async () => {
    const id = await fetchModel("HuggingFaceTB/SmolLM2-135M-Instruct", (paths, defaults) => [...defaults, ...paths.filter((p) => p === "onnx/model.onnx")]);
    const o = (await hub.targets(id)).find((t) => t.id === "onnx")!;
    expect(o.needs).toEqual([]);
    await deployAndCheck({ modelId: id, target: "onnx", source: "onnx:onnx/model.onnx", quant: "none", kind: "chat", device: "cpu" });
  }, 3_600_000);

  run("onnx-export")("exports an embedding model to ONNX, quantizes to int8 and serves it", async () => {
    const id = await fetchModel("sentence-transformers/all-MiniLM-L6-v2");
    const o = (await hub.targets(id)).find((t) => t.id === "onnx")!;
    expect(o.sources.map((s) => s.id)).toContain("export");
    const d = await deployAndCheck({ modelId: id, target: "onnx", source: "export", quant: "dynamic-int8", kind: "embed", device: "cpu" });
    expect(d.dim).toBe(384);
    // Same meaning as the full-precision model: export + int8 must not change what the vectors say.
    const ref = await deployAndCheck({ modelId: id, target: "transformers", source: "weights", quant: "none", kind: "embed", device: "cpu" });
    const texts = ["how do I reset my password", "steps to recover a forgotten login", "the weather in Paris is sunny"];
    const embed = async (dep: typeof d) => {
      const r = await fetch(`${dep.baseUrl}/embeddings`, {
        method: "POST",
        headers: { Authorization: `Bearer ${secrets.get(`provider:hub-${dep.id}`)}`, "Content-Type": "application/json" },
        body: JSON.stringify({ model: dep.servedName, input: texts }),
      });
      return ((await r.json()) as { data: Array<{ embedding: number[] }> }).data.map((x) => x.embedding);
    };
    const cos = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i]!, 0) / Math.sqrt(a.reduce((s, v) => s + v * v, 0) * b.reduce((s, v) => s + v * v, 0));
    const [q, f] = [await embed(d), await embed(ref)];
    const agreement = q.map((v, i) => cos(v, f[i]!));
    console.log(`[live] int8 ONNX vs full precision cosine: ${agreement.map((x) => x.toFixed(4)).join(", ")}`);
    for (const a of agreement) expect(a).toBeGreaterThan(0.95);
    expect(cos(q[0]!, q[1]!)).toBeGreaterThan(cos(q[0]!, q[2]!));
  }, 3_600_000);

  run("pause")("pauses a real download and resumes it where it stopped", async () => {
    const repo = "sentence-transformers/paraphrase-MiniLM-L3-v2";
    const info = await hub.lookup(repo);
    const job = await hub.download(repo, "", info.defaultSelection);
    const moving = await until(() => hub.view().jobs.find((j) => j.id === job.id && j.bytesDone > 5_000_000 && j.speedBps > 0), 5 * 60_000, "download progress");
    console.log(`[live] ${repo}: ${(moving.bytesDone / 1e6).toFixed(1)} of ${(moving.bytesTotal / 1e6).toFixed(1)} MB at ${(moving.speedBps / 1e6).toFixed(2)} MB/s, eta ${moving.etaSec}s`);
    hub.jobAction(job.id, "pause");
    const paused = await until(() => hub.view().jobs.find((j) => j.id === job.id && j.status === "paused"), 30_000, "pause");
    if (paused.bytesDone < paused.bytesTotal) {
      hub.jobAction(job.id, "resume");
      const resumed = await until(() => hub.view().jobs.find((j) => j.id === job.id && j.status !== "paused"), 30_000, "resume");
      expect(resumed.bytesDone).toBeGreaterThanOrEqual(paused.bytesDone - 1_000_000);
    }
    const done = await until(() => hub.view().jobs.find((j) => j.id === job.id && ["done", "error"].includes(j.status)), 10 * 60_000, "finish");
    expect(done.error).toBeNull();
    expect(done.bytesDone).toBe(done.bytesTotal);
  }, 900_000);

  run("restore")("restores a running deployment after the IDE restarts", async () => {
    const id = await fetchModel("sentence-transformers/all-MiniLM-L6-v2");
    const before = await deployAndCheck({ modelId: id, target: "llamacpp", source: "convert", quant: "Q8_0", kind: "embed", device: "cpu" });
    hub.close();
    settings.hub.restoreDeployments = true;
    hub = service();
    try {
      await hub.boot();
      const back = await until(() => hub.view().deployments.find((d) => d.id === before.id && ["running", "error"].includes(d.status)), 10 * 60_000, "restore");
      expect(back.status).toBe("running");
      expect(back.steps.some((s) => s.status === "skipped")).toBe(true);
      const t = await hub.test(back.id);
      console.log(`[live] restored ${back.servedName}: ${t.detail}`);
      expect(t.ok).toBe(true);
    } finally {
      settings.hub.restoreDeployments = false;
    }
  }, 3_600_000);

  run("delete")("deletes a model with its deployments, provider and files", async () => {
    const repo = "sentence-transformers/paraphrase-MiniLM-L3-v2";
    const id = await fetchModel(repo);
    const dep = await deployAndCheck({ modelId: id, target: "transformers", source: "weights", quant: "none", kind: "embed", device: "cpu" });
    const dir = hub.view().models.find((m) => m.id === id)!.dir;
    expect(() => hub.deleteModel(id)).toThrow(/stop its deployments first/);
    hub.stop(dep.id);
    hub.deleteModel(id);
    expect(hub.view().models.some((m) => m.id === id)).toBe(false);
    expect(hub.view().deployments.some((d) => d.id === dep.id)).toBe(false);
    expect(settings.providers.some((p) => p.id === `hub-${dep.id}`)).toBe(false);
    expect(settings.models.some((m) => m.id === dep.settingsModelId)).toBe(false);
    expect(secrets.has(`provider:hub-${dep.id}`)).toBe(false);
    expect(existsSync(dir)).toBe(false);
    await expect(fetch(`${dep.baseUrl}/models`, { signal: AbortSignal.timeout(3000) })).rejects.toThrow();
  }, 900_000);

  run("access")("explains gated and missing repos instead of failing later", async () => {
    const gated = await hub.lookup("google/embeddinggemma-300m");
    expect(gated.gated).toBeTruthy();
    expect(gated.access).toBe("needs-token");
    expect(gated.profile.kind).toBe("embed");
    await expect(hub.download("google/embeddinggemma-300m", "", gated.defaultSelection)).rejects.toThrow(/needs a Hugging Face token/);
    await expect(hub.lookup("no-such-org-xyz/no-such-model-xyz")).rejects.toThrow(/not found|gated/);
  }, 120_000);
});
