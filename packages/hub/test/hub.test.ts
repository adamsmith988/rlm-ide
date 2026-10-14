import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import http from "node:http";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { defaultSettings, type HubModelView } from "@rlm/contracts";
import { DownloadManager, HfClient, deployTargets, defaultSelection, parseConvertArchitectures, parseConvertOutTypes, parseQuantTypes, parseRepoRef, profileOf, planDeployment } from "../src";
import type { LlamaCpp, PythonProbe } from "../src/toolchain";

const sha = (b: Buffer) => createHash("sha256").update(b).digest("hex");

describe("repo references", () => {
  it("accepts ids and huggingface.co URLs, with revisions", () => {
    expect(parseRepoRef("google/embeddinggemma-300m")).toEqual({ repo: "google/embeddinggemma-300m", revision: "main" });
    expect(parseRepoRef("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/v1.0/sub")).toEqual({ repo: "Qwen/Qwen2.5-0.5B-Instruct", revision: "v1.0" });
    expect(parseRepoRef("hf.co/org/name", "refs/pr/3")).toEqual({ repo: "org/name", revision: "refs/pr/3" });
    expect(parseRepoRef("gpt2")).toEqual({ repo: "gpt2", revision: "main" });
    expect(() => parseRepoRef("../etc/passwd")).toThrow(/not a model repo/);
    expect(() => parseRepoRef("a/b/c")).toThrow();
  });
});

describe("model profile from repo metadata", () => {
  const f = (p: string, size = 10, lfs = false) => ({ path: p, size, sha256: lfs ? "x".repeat(64) : null });

  it("reads a chat model: pipeline, template, context, formats", () => {
    const p = profileOf({
      info: { id: "o/chat", pipeline_tag: "text-generation", library_name: "transformers", config: { architectures: ["LlamaForCausalLM"], model_type: "llama" } },
      files: [f("config.json"), f("model.safetensors", 1e8, true), f("onnx/model.onnx", 1e8, true), f("tokenizer_config.json")],
      config: { architectures: ["LlamaForCausalLM"], model_type: "llama", max_position_embeddings: 8192 },
      tokenizerConfig: { chat_template: "{% for m in messages %}{% endfor %}" },
      sentenceBertConfig: null,
    });
    expect(p).toMatchObject({ kind: "chat", modelType: "llama", contextLength: 8192, hasChatTemplate: true, hasHfWeights: true, onnxFiles: ["onnx/model.onnx"] });
    expect(p.formats.sort()).toEqual(["onnx", "safetensors"]);
  });

  it("reads an embedding model from sentence-transformers files and caps its context", () => {
    const p = profileOf({
      info: { id: "o/emb", library_name: "sentence-transformers" },
      files: [f("config.json"), f("modules.json"), f("sentence_bert_config.json"), f("pytorch_model.bin", 1e7, true)],
      config: { architectures: ["BertModel"], model_type: "bert", max_position_embeddings: 512 },
      tokenizerConfig: null,
      sentenceBertConfig: { max_seq_length: 256 },
    });
    expect(p).toMatchObject({ kind: "embed", contextLength: 256, formats: ["pytorch"] });
  });

  it("marks non-text models as other and reads nested text configs", () => {
    const asr = profileOf({ info: { id: "o/asr", pipeline_tag: "automatic-speech-recognition" }, files: [f("model.safetensors", 5, true), f("config.json")], config: { model_type: "whisper" }, tokenizerConfig: null, sentenceBertConfig: null });
    expect(asr.kind).toBe("other");
    const vlm = profileOf({ info: { id: "o/vlm" }, files: [f("config.json"), f("model.safetensors", 5, true)], config: { architectures: ["Gemma3ForConditionalGeneration"], text_config: { max_position_embeddings: 131072 } }, tokenizerConfig: null, sentenceBertConfig: null });
    expect(vlm).toMatchObject({ kind: "chat", contextLength: 131072 });
    expect(vlm.notes.join(" ")).toMatch(/No chat template/);
  });

  it("notes remote code, pre-quantized weights and gating", () => {
    const p = profileOf({
      info: { id: "o/q", gated: "manual", pipeline_tag: "text-generation" },
      files: [f("config.json"), f("model.safetensors", 5, true)],
      config: { auto_map: { AutoModel: "x.Y" }, quantization_config: { quant_method: "gptq" } },
      tokenizerConfig: null,
      sentenceBertConfig: null,
    });
    expect(p.remoteCode).toBe(true);
    expect(p.quantizedWith).toBe("gptq");
    expect(p.notes.join(" ")).toMatch(/Gated repo/);
  });

  it("selects one copy of the weights by default; GGUF repos let the user pick", () => {
    const mixed = [f("config.json"), f("model.safetensors"), f("pytorch_model.bin"), f("tf_model.h5"), f("onnx/model.onnx"), f("original/consolidated.pth"), f("README.md")];
    expect(defaultSelection(mixed).sort()).toEqual(["README.md", "config.json", "model.safetensors"]);
    const ggufRepo = [f("README.md"), f("m-Q4_K_M.gguf"), f("m-Q8_0.gguf")];
    expect(defaultSelection(ggufRepo)).toEqual(["README.md"]);
    expect(defaultSelection([f("README.md"), f("only.gguf")])).toEqual(["README.md", "only.gguf"]);
    const onnxOnly = [f("config.json"), f("onnx/model.onnx"), f("onnx/model.onnx_data")];
    expect(defaultSelection(onnxOnly).length).toBe(3);
  });
});

describe("llama.cpp introspection", () => {
  it("parses quantization types from llama-quantize --help", () => {
    const help = `usage: llama-quantize ...
Allowed quantization types:
   2  or  Q4_0    :  4.34G, +0.4685 ppl @ Llama-3-8B
  15  or  Q4_K_M  :  4.58G, +0.1754 ppl @ Llama-3-8B
  19  or  IQ2_XXS :  2.06 bpw quantization
   0  or  F32     : 26.00G              @ 7B
          COPY    : only copy tensors, no quantizing`;
    expect(parseQuantTypes(help).map((q) => q.id)).toEqual(["Q4_0", "Q4_K_M", "IQ2_XXS", "F32"]);
  });

  it("reads registered architectures and outtypes from the converter source", () => {
    const src = `
@ModelBase.register("LlamaForCausalLM", "MistralForCausalLM")
class LlamaModel(TextModel): pass
@ModelBase.register('BertModel')
class BertModel(TextModel): pass
parser.add_argument(
    "--outtype", type=str, choices=["f32", "f16", "bf16", "q8_0", "tq1_0", "auto"], default="f16",
)`;
    expect(parseConvertArchitectures(src)).toEqual(["BertModel", "LlamaForCausalLM", "MistralForCausalLM"]);
    expect(parseConvertOutTypes(src)).toEqual(["f32", "f16", "bf16", "q8_0", "tq1_0", "auto"]);
  });
});

describe("deploy targets follow the model and the installed tools", () => {
  const model = (over: Partial<HubModelView["profile"]>, files: string[]): HubModelView => ({
    id: "o__m",
    repo: "o/m",
    revision: "main",
    sha: "s",
    dir: "/tmp/none",
    bytes: 1,
    files,
    downloadedAt: 0,
    profile: { kind: "chat", kindReason: "", pipelineTag: null, libraryName: null, architectures: ["LlamaForCausalLM"], modelType: "llama", formats: [], ggufFiles: [], onnxFiles: [], hasHfWeights: true, contextLength: 4096, hasChatTemplate: true, remoteCode: false, quantizedWith: null, parameters: null, notes: [], ...over },
  });
  const probe: PythonProbe = { python: "3.10", executable: "/py", packages: { torch: "2", transformers: "5", bitsandbytes: "0.4", optimum: null, onnxruntime: "1.2" }, cuda: true, gpu: null, transformersModelTypes: ["llama", "bert"], onnxExportModelTypes: [], onnxQuantizers: ["dynamic-int8", "int4-weight"] };
  const llama: LlamaCpp = { dir: "/l", server: "/l/llama-server", quantize: "/l/llama-quantize", imatrix: "/l/llama-imatrix", convert: "/l/convert_hf_to_gguf.py", backend: "cpu", version: "1", quantTypes: [{ id: "Q4_K_M", label: "Q4_K_M" }, { id: "F16", label: "F16" }, { id: "COPY", label: "COPY" }], convertArchitectures: ["LlamaForCausalLM"], convertOutTypes: ["f32", "f16", "bf16", "q8_0", "auto"] };
  const tools = { python: "/py", probe, onnxPython: "/py", onnxProbe: probe, llama, serveScript: "serve.py", onnxScript: "onnx_tools.py", contextCap: 2048 };

  it("offers every target that fits, with quantizations from the tools", () => {
    const t = deployTargets(model({}, ["config.json", "model.safetensors"]), tools);
    const by = Object.fromEntries(t.map((x) => [x.id, x]));
    expect(by.llamacpp!.available).toBe(true);
    expect(by.llamacpp!.quantizations.convert!.map((q) => q.id)).toEqual(["Q4_K_M", "F16"]);
    expect(by.transformers!.quantizations.weights!.map((q) => q.id)).toEqual(["none", "bnb-8bit", "bnb-4bit-nf4", "bnb-4bit-fp4"]);
    expect(by.transformers!.devices).toEqual(["cpu", "cuda"]);
    expect(by.onnx!.needs).toContain("onnx:optimum-onnx[onnxruntime]");
  });

  it("explains unconvertible architectures, unknown model types and non-text models", () => {
    const odd = deployTargets(model({ architectures: ["NewArchForCausalLM"], modelType: "newarch" }, ["config.json", "model.safetensors"]), tools);
    expect(odd.find((x) => x.id === "llamacpp")!.reason).toMatch(/no support for NewArchForCausalLM/);
    expect(odd.find((x) => x.id === "transformers")!.reason).toMatch(/does not know model type "newarch"/);
    const asr = deployTargets(model({ kind: "other", kindReason: 'pipeline tag "automatic-speech-recognition"' }, ["config.json", "model.safetensors"]), tools);
    expect(asr.every((x) => !x.available)).toBe(true);
  });

  it("serves a downloaded GGUF as-is or re-quantized, and caps the context", () => {
    const m = model({ ggufFiles: ["a-Q8_0.gguf", "b.gguf"], hasHfWeights: false, contextLength: 32768 }, ["a-Q8_0.gguf"]);
    const t = deployTargets(m, tools).find((x) => x.id === "llamacpp")!;
    expect(t.sources.map((s) => s.id)).toEqual(["gguf:a-Q8_0.gguf"]);
    expect(t.quantizations["gguf:a-Q8_0.gguf"]![0]!.id).toBe("as-is");
    const dir = mkdtempSync(path.join(tmpdir(), "hub-plan-"));
    writeFileSync(path.join(dir, "a-Q8_0.gguf"), "x");
    const plan = planDeployment({ ...m, dir }, { modelId: m.id, target: "llamacpp", source: "gguf:a-Q8_0.gguf", quant: "as-is", kind: "chat", device: "cpu", dtype: "auto", contextLength: 0, trustRemoteCode: false, label: "" }, tools, dir);
    expect(plan.steps).toEqual([]);
    expect(plan.contextLength).toBe(2048);
    const serve = plan.serve(1234, "k".repeat(20), "served", "/keys/api-key");
    expect(serve.args).toEqual(expect.arrayContaining(["--port", "1234", "--alias", "served", "-c", "2048", "--jinja", "--api-key-file", "/keys/api-key"]));
    expect(serve.args.join(" ")).not.toContain("k".repeat(20));
  });

  it("converts then quantizes, with an importance matrix as the fallback route", () => {
    const dir = mkdtempSync(path.join(tmpdir(), "hub-plan-"));
    writeFileSync(path.join(dir, "README.md"), "# A model\nSome words about it.");
    const m = { ...model({}, ["config.json", "model.safetensors", "README.md"]), dir };
    const plan = planDeployment(m, { modelId: m.id, target: "llamacpp", source: "convert", quant: "Q4_K_M", kind: "chat", device: "cpu", dtype: "auto", contextLength: 0, trustRemoteCode: false, label: "" }, tools, dir);
    expect(plan.steps.map((s) => s.label)).toEqual(["convert to GGUF (auto)", "quantize to Q4_K_M"]);
    expect(plan.steps[1]!.recover!("error: this quantization requires an importance matrix")!.map((s) => s.label)).toEqual(["compute importance matrix", "quantize to Q4_K_M"]);
    expect(plan.steps[1]!.recover!("out of memory")).toBeNull();
    const direct = planDeployment(m, { modelId: m.id, target: "llamacpp", source: "convert", quant: "F16", kind: "embed", device: "cpu", dtype: "auto", contextLength: 512, trustRemoteCode: false, label: "" }, tools, dir);
    expect(direct.steps.map((s) => s.label)).toEqual(["convert to GGUF (f16)"]);
    expect(direct.serve(1, "k".repeat(20), "e", "/keys/api-key").args).toEqual(expect.arrayContaining(["--embeddings", "-c", "512"]));
  });
});

describe("downloader", () => {
  const files: Record<string, Buffer> = {
    "config.json": Buffer.from('{"model_type":"llama"}'),
    "model.safetensors": Buffer.alloc(3_000_000, 7),
    "sub/tokenizer.json": Buffer.from("{}"),
  };
  let server: http.Server;
  let base = "";
  let corrupt = false;
  let slow = false;
  const auth: string[] = [];

  beforeAll(async () => {
    server = http.createServer((req, res) => {
      auth.push(req.headers.authorization ?? "");
      const u = new URL(req.url!, "http://x");
      if (u.pathname === "/api/models/o/m") return res.end(JSON.stringify({ id: "o/m", sha: "abc123", pipeline_tag: "text-generation" }));
      if (u.pathname === "/api/models/o/m/tree/abc123") {
        const entries = Object.entries(files).map(([p, b]) => ({ type: "file", path: p, size: b.length, ...(p.endsWith(".safetensors") ? { lfs: { oid: sha(b), size: b.length } } : {}) }));
        // Two pages, like the Hub's pagination.
        if (!u.searchParams.get("cursor")) {
          res.setHeader("Link", `<${base}/api/models/o/m/tree/abc123?recursive=true&cursor=2>; rel="next"`);
          return res.end(JSON.stringify(entries.slice(0, 2)));
        }
        return res.end(JSON.stringify(entries.slice(2)));
      }
      const m = /^\/o\/m\/resolve\/abc123\/(.+)$/.exec(u.pathname);
      if (m) {
        const name = decodeURIComponent(m[1]!);
        let body = files[name];
        if (!body) return void res.writeHead(404).end();
        if (corrupt && name.endsWith(".safetensors")) body = Buffer.alloc(body.length, 1);
        const range = /bytes=(\d+)-/.exec(req.headers.range ?? "");
        const start = range ? Number(range[1]) : 0;
        res.writeHead(range ? 206 : 200, { "Content-Length": body.length - start });
        if (!slow) return res.end(body.subarray(start));
        let at = start;
        const t = setInterval(() => {
          if (at >= body!.length || res.destroyed) {
            clearInterval(t);
            return res.end();
          }
          res.write(body!.subarray(at, at + 65536));
          at += 65536;
        }, 20);
        return;
      }
      if (u.pathname.startsWith("/api/models/gated/m")) return void res.writeHead(401).end(JSON.stringify({ error: "Access to model gated/m is restricted." }));
      res.writeHead(404).end();
    });
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", () => r()));
    base = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  });
  afterAll(() => server.close());

  it("lists paginated trees with sizes and LFS hashes, sending the token", async () => {
    const hf = new HfClient(() => "hf_test", base);
    const tree = await hf.tree("o/m", "abc123");
    expect(tree.map((t) => t.path)).toEqual(["config.json", "model.safetensors", "sub/tokenizer.json"]);
    expect(tree[1]!.sha256).toBe(sha(files["model.safetensors"]!));
    expect(auth.at(-1)).toBe("Bearer hf_test");
    await expect(new HfClient(() => null, base).info("gated/m", "main")).rejects.toThrow(/need a Hugging Face token/);
  });

  const runJob = async (dest: string, opts: { pauseAfterMs?: number } = {}) => {
    const hf = new HfClient(() => null, base);
    const tree = await hf.tree("o/m", "abc123");
    const views: Array<{ status: string; bytesDone: number; speedBps: number }> = [];
    let finished: (v: { status: string; error: string | null; bytesDone: number; bytesTotal: number }) => void;
    const done = new Promise<{ status: string; error: string | null; bytesDone: number; bytesTotal: number }>((r) => (finished = r));
    const dm = new DownloadManager(hf, { concurrency: () => 2, onChange: (v) => views.push(v), onFinished: (v) => finished(v) });
    dm.start({ id: "j", repo: "o/m", revision: "main", sha: "abc123", dest, files: tree });
    if (opts.pauseAfterMs) setTimeout(() => dm.pause("j"), opts.pauseAfterMs);
    return { result: await done, views };
  };

  it("downloads, verifies sha256, and reports progress and speed", async () => {
    const dest = mkdtempSync(path.join(tmpdir(), "hub-dl-"));
    slow = true;
    const { result, views } = await runJob(dest);
    slow = false;
    expect(result.status).toBe("done");
    expect(result.bytesDone).toBe(result.bytesTotal);
    expect(readFileSync(path.join(dest, "model.safetensors")).equals(files["model.safetensors"]!)).toBe(true);
    expect(existsSync(path.join(dest, "sub/tokenizer.json"))).toBe(true);
    expect(views.some((v) => v.speedBps > 0)).toBe(true);
  });

  it("pauses and resumes from the partial file with a Range request", async () => {
    const dest = mkdtempSync(path.join(tmpdir(), "hub-dl-"));
    slow = true;
    const first = await runJob(dest, { pauseAfterMs: 300 });
    expect(first.result.status).toBe("paused");
    const part = path.join(dest, "model.safetensors.part");
    expect(existsSync(part)).toBe(true);
    const second = await runJob(dest);
    slow = false;
    expect(second.result.status).toBe("done");
    expect(second.views[0]!.bytesDone).toBeGreaterThan(0);
    expect(readFileSync(path.join(dest, "model.safetensors")).equals(files["model.safetensors"]!)).toBe(true);
  });

  it("rejects a corrupted file by checksum and removes the partial", async () => {
    const dest = mkdtempSync(path.join(tmpdir(), "hub-dl-"));
    corrupt = true;
    const { result } = await runJob(dest);
    corrupt = false;
    expect(result.status).toBe("error");
    expect(result.error).toMatch(/checksum mismatch/);
    expect(existsSync(path.join(dest, "model.safetensors.part"))).toBe(false);
  });

  it("refuses repo paths that escape the model folder", async () => {
    const dest = mkdtempSync(path.join(tmpdir(), "hub-dl-"));
    mkdirSync(dest, { recursive: true });
    const hf = new HfClient(() => null, base);
    let err = "";
    const done = new Promise<void>((r) => {
      const dm = new DownloadManager(hf, { concurrency: () => 1, onChange: () => undefined, onFinished: (v) => ((err = v.error ?? ""), r()) });
      try {
        dm.start({ id: "evil", repo: "o/m", revision: "main", sha: "abc123", dest, files: [{ path: "../../escape.txt", size: 1, sha256: null }] });
      } catch (e) {
        err = (e as Error).message;
        r();
      }
    });
    await done;
    expect(err).toMatch(/outside the model folder/);
  });
});

it("hub settings have defaults", () => {
  expect(defaultSettings().hub).toMatchObject({ downloadConcurrency: 4, contextCap: 32768, restoreDeployments: true });
});
