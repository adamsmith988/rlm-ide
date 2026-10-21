import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import type { DeployTargetView, HubDeployRequest, HubKind, HubModelView, HubOption } from "@rlm/contracts";
import type { LlamaCpp, PythonProbe } from "./toolchain";

export type ModelRecord = HubModelView;

export interface StepSpec {
  label: string;
  cmd: string;
  args: string[];
  env?: Record<string, string>;
  cwd?: string;
  /** Skip when this file already exists (an earlier deployment produced it). */
  output?: string;
  /** Replacement steps when this one fails with the given output (e.g. a quantization that needs an importance matrix). */
  recover?: (tail: string) => StepSpec[] | null;
}

export interface ServeSpec {
  cmd: string;
  args: string[];
  env: Record<string, string>;
}

export interface Plan {
  steps: StepSpec[];
  artifact: string;
  contextLength: number;
  /** `keyFile` holds the same key, readable by the owner only, for servers that take keys from a file. */
  serve: (port: number, key: string, servedName: string, keyFile: string) => ServeSpec;
}

export interface ToolsSnapshot {
  python: string;
  probe: PythonProbe | null;
  onnxPython: string;
  onnxProbe: PythonProbe | null;
  llama: LlamaCpp;
  serveScript: string;
  onnxScript: string;
  contextCap: number;
}

const TARGET_LABELS: Record<DeployTargetView["id"], string> = {
  llamacpp: "llama.cpp (GGUF)",
  transformers: "Transformers",
  onnx: "ONNX Runtime",
};

function kindsFor(m: ModelRecord): HubKind[] {
  if (m.profile.kind === "chat") return ["chat", "embed"];
  if (m.profile.kind === "embed") return ["embed", "chat"];
  if (m.profile.kind === "unknown") return ["chat", "embed"];
  return [];
}

const downloaded = (m: ModelRecord, files: string[]) => files.filter((f) => m.files.includes(f));
const hasDownloadedHfWeights = (m: ModelRecord) => m.profile.hasHfWeights && m.files.some((f) => /\.(safetensors|bin|pt|pth)$/i.test(f)) && m.files.includes("config.json");

export function deployTargets(m: ModelRecord, t: ToolsSnapshot): DeployTargetView[] {
  const kinds = kindsFor(m);
  const other = m.profile.kind === "other" ? `Not a chat or embedding model (${m.profile.kindReason}); the IDE serves chat and embedding models.` : "";
  const out: DeployTargetView[] = [];
  const py = t.probe;

  // llama.cpp: serve downloaded GGUF files, or convert the Hugging Face weights.
  {
    const l = t.llama;
    const ggufs = downloaded(m, m.profile.ggufFiles);
    const sources: HubOption[] = ggufs.map((g) => ({ id: `gguf:${g}`, label: `${g} (as downloaded)` }));
    let reason = other;
    const convertible = hasDownloadedHfWeights(m) && (!l.convert || m.profile.architectures.some((a) => l.convertArchitectures.includes(a)));
    if (convertible) sources.push({ id: "convert", label: "Convert the Hugging Face weights to GGUF", detail: l.convert ? undefined : "convertibility is checked once llama.cpp is installed" });
    else if (hasDownloadedHfWeights(m) && l.convert && !reason) reason = `llama.cpp's converter has no support for ${m.profile.architectures.join(", ") || "this architecture"}.`;
    const needs = l.server && l.quantize && (l.convert || !sources.some((s) => s.id === "convert")) ? [] : ["llamacpp"];
    const quantTypes = l.quantTypes.filter((q) => q.id !== "COPY");
    const quantizations: Record<string, HubOption[]> = {};
    for (const s of sources)
      quantizations[s.id] = s.id === "convert" ? quantTypes : [{ id: "as-is", label: "as downloaded" }, ...quantTypes.map((q) => ({ ...q, detail: `re-quantize: ${q.detail ?? ""}` }))];
    if (!sources.length && !reason) reason = "No GGUF files downloaded and no Hugging Face weights to convert.";
    out.push({
      id: "llamacpp",
      label: TARGET_LABELS.llamacpp,
      available: !other && sources.length > 0,
      reason,
      needs,
      sources,
      quantizations,
      // The GPU first: it is what the user installed the GPU build for.
      devices: l.backend === "cuda" ? ["cuda", "cpu"] : l.backend === "vulkan" || l.backend === "metal" ? ["gpu", "cpu"] : ["cpu"],
      kinds,
      dtypes: [],
    });
  }

  // transformers: the Hugging Face weights as they are, optionally quantized on load with bitsandbytes.
  {
    const needs: string[] = [];
    if (!py) needs.push("python");
    else for (const p of ["torch", "transformers"]) if (!py.packages[p]) needs.push(`python:${p}`);
    let reason = other;
    const knownType = !py || !m.profile.modelType || py.transformersModelTypes.includes(m.profile.modelType) || m.profile.remoteCode;
    if (!reason && !hasDownloadedHfWeights(m)) reason = "Needs the Hugging Face weights (safetensors or PyTorch) and config.json.";
    else if (!reason && !knownType) reason = `The installed transformers ${py?.packages.transformers ?? ""} does not know model type "${m.profile.modelType}"; upgrade transformers.`;
    const quants: HubOption[] = [{ id: "none", label: m.profile.quantizedWith ? `as published (${m.profile.quantizedWith})` : "none (full precision)" }];
    if (!m.profile.quantizedWith && py?.packages.bitsandbytes && py.cuda)
      quants.push(
        { id: "bnb-8bit", label: "bitsandbytes 8-bit (LLM.int8)", detail: "GPU" },
        { id: "bnb-4bit-nf4", label: "bitsandbytes 4-bit NF4", detail: "GPU" },
        { id: "bnb-4bit-fp4", label: "bitsandbytes 4-bit FP4", detail: "GPU" },
      );
    out.push({
      id: "transformers",
      label: TARGET_LABELS.transformers,
      available: !reason,
      reason: reason || (quants.length === 1 && !m.profile.quantizedWith ? "On-load quantization (bitsandbytes 8/4-bit) needs a CUDA GPU and the bitsandbytes package." : ""),
      needs,
      sources: hasDownloadedHfWeights(m) ? [{ id: "weights", label: "Hugging Face weights" }] : [],
      quantizations: { weights: quants },
      devices: py?.cuda ? ["cpu", "cuda"] : ["cpu"],
      kinds,
      dtypes: ["auto", "float32", "float16", "bfloat16"],
    });
  }

  // ONNX Runtime: downloaded ONNX files, or an export of the Hugging Face weights; ONNX quantization on top.
  {
    const py = t.onnxProbe;
    const needs: string[] = [];
    if (!py) needs.push("python");
    else {
      if (!py.packages.optimum && !py.packages["optimum-onnx"]) needs.push("onnx:optimum-onnx[onnxruntime]");
      if (!py.packages.onnxruntime && !py.packages["onnxruntime-gpu"]) needs.push("onnx:onnxruntime");
    }
    const onnx = downloaded(m, m.profile.onnxFiles);
    const sources: HubOption[] = onnx.map((f) => ({ id: `onnx:${f}`, label: `${f} (as downloaded)` }));
    const exportable = hasDownloadedHfWeights(m) && (!py || needs.length > 0 || !m.profile.modelType || py.onnxExportModelTypes.includes(m.profile.modelType));
    if (exportable) sources.push({ id: "export", label: "Export the Hugging Face weights to ONNX", detail: needs.length ? "exportability is checked once optimum is installed" : undefined });
    const methods: HubOption[] = [{ id: "none", label: "none" }, ...(py?.onnxQuantizers ?? []).map((q) => ({ id: q, label: q }))];
    const quantizations: Record<string, HubOption[]> = {};
    for (const s of sources) quantizations[s.id] = methods;
    let reason = other;
    if (!reason && !sources.length) reason = hasDownloadedHfWeights(m) ? `optimum cannot export model type "${m.profile.modelType}" to ONNX.` : "No ONNX files downloaded and no Hugging Face weights to export.";
    out.push({
      id: "onnx",
      label: TARGET_LABELS.onnx,
      available: !reason,
      reason,
      needs,
      sources,
      quantizations,
      devices: py?.cuda && py.packages["onnxruntime-gpu"] ? ["cpu", "cuda"] : ["cpu"],
      kinds,
      dtypes: [],
    });
  }
  return out;
}

const slug = (s: string) => s.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "").toLowerCase();

/** Calibration text for an importance matrix, from the repo's own documents. */
function calibrationFile(m: ModelRecord, workDir: string): string {
  const texts: string[] = [];
  for (const f of m.files.filter((x) => /\.(md|txt)$/i.test(x)).slice(0, 20)) {
    try {
      texts.push(readFileSync(path.join(m.dir, f), "utf8"));
    } catch {
      /* skip */
    }
  }
  let text = texts.join("\n\n").replace(/```[\s\S]*?```/g, " ");
  while (text.length && text.length < 20000) text += `\n\n${text}`;
  const p = path.join(workDir, "calibration.txt");
  writeFileSync(p, text || "The quick brown fox jumps over the lazy dog. ".repeat(500));
  return p;
}

export function planDeployment(m: ModelRecord, req: HubDeployRequest, t: ToolsSnapshot, workDir: string): Plan {
  const cache = path.join(m.dir, ".rlm");
  const ctx = req.contextLength || Math.min(m.profile.contextLength ?? 4096, t.contextCap);
  const base = slug(path.basename(m.repo));

  if (req.target === "llamacpp") {
    const l = t.llama;
    if (!l.server) throw new Error("llama.cpp is not installed (Models hub → Tools)");
    const steps: StepSpec[] = [];
    let artifact: string;
    const quantize = (src: string, out: string, type: string, requant: boolean, imatrix?: string): StepSpec => ({
      label: `quantize to ${type}`,
      cmd: l.quantize!,
      env: l.env,
      args: [...(requant ? ["--allow-requantize"] : []), ...(imatrix ? ["--imatrix", imatrix] : []), src, out, type],
      output: out,
      recover: imatrix || !l.imatrix ? undefined : (tail) => (/imatrix|importance matrix/i.test(tail) ? withImatrix(src, out, type, requant) : null),
    });
    const withImatrix = (src: string, out: string, type: string, requant: boolean): StepSpec[] => {
      const dat = path.join(cache, `${base}.imatrix.dat`);
      return [
        { label: "compute importance matrix", cmd: l.imatrix!, env: l.env, args: ["-m", src, "-f", calibrationFile(m, workDir), "-o", dat, "-c", "256", "--chunks", "64"], output: dat },
        quantize(src, out, type, requant, dat),
      ];
    };
    if (req.source === "convert") {
      if (!l.convert || !l.quantize) throw new Error("llama.cpp's converter is not installed");
      const direct = l.convertOutTypes.find((o) => o.toLowerCase() === req.quant.toLowerCase());
      const outtype = direct ?? (l.convertOutTypes.includes("auto") ? "auto" : "f16");
      const converted = path.join(cache, `${base}-${direct ?? outtype}.gguf`);
      steps.push({ label: `convert to GGUF (${direct ?? outtype})`, cmd: t.python, args: [l.convert, m.dir, "--outfile", converted, "--outtype", outtype], output: converted });
      if (direct) artifact = converted;
      else {
        artifact = path.join(cache, `${base}-${req.quant}.gguf`);
        steps.push(quantize(converted, artifact, req.quant, false));
      }
    } else if (req.source.startsWith("gguf:")) {
      const src = path.join(m.dir, req.source.slice(5));
      if (!existsSync(src)) throw new Error(`${req.source.slice(5)} is not downloaded`);
      if (req.quant === "as-is") artifact = src;
      else {
        if (!l.quantize) throw new Error("llama-quantize is not installed");
        artifact = path.join(cache, `${slug(path.basename(src, ".gguf"))}-requant-${req.quant}.gguf`);
        steps.push(quantize(src, artifact, req.quant, true));
      }
    } else throw new Error(`unknown llama.cpp source ${req.source}`);
    return {
      steps,
      artifact,
      contextLength: ctx,
      serve: (port, _key, served, keyFile) => ({
        cmd: l.server!,
        args: [
          "-m", artifact,
          "--api-key-file", keyFile,
          "--host", "127.0.0.1",
          "--port", String(port),
          "--alias", served,
          "-c", String(ctx),
          ...(req.kind === "embed" ? ["--embeddings", "-b", String(ctx), "-ub", String(ctx)] : ["--jinja"]),
          ...(req.device === "cpu" ? ["-ngl", "0"] : ["-ngl", "999"]),
        ],
        env: { ...l.env },
      }),
    };
  }

  if (req.target === "transformers") {
    return {
      steps: [],
      artifact: m.dir,
      contextLength: ctx,
      serve: (port, key, served) => ({
        cmd: t.python,
        args: [
          t.serveScript,
          "--model-dir", m.dir,
          "--backend", "transformers",
          "--kind", req.kind,
          "--served-name", served,
          "--port", String(port),
          "--device", req.device,
          "--dtype", req.dtype || "auto",
          "--quant", req.quant,
          "--max-context", String(ctx),
          ...(req.trustRemoteCode ? ["--trust-remote-code"] : []),
        ],
        env: { RLM_HUB_API_KEY: key, PYTHONUNBUFFERED: "1" },
      }),
    };
  }

  // onnx
  const steps: StepSpec[] = [];
  let dir: string;
  let file = "";
  if (req.source === "export") {
    const exported = path.join(cache, `onnx-export-${req.kind}`);
    steps.push({ label: "export to ONNX", cmd: t.onnxPython, args: [t.onnxScript, "export", "--src", m.dir, "--out", exported, "--kind", req.kind, ...(req.trustRemoteCode ? ["--trust-remote-code"] : [])], output: path.join(exported, "config.json") });
    dir = exported;
    if (req.quant !== "none") {
      const q = path.join(cache, `onnx-export-${req.kind}-${req.quant}`);
      steps.push({ label: `ONNX quantization (${req.quant})`, cmd: t.onnxPython, args: [t.onnxScript, "quantize", "--src", exported, "--out", q, "--method", req.quant], output: path.join(q, "config.json") });
      dir = q;
    }
  } else if (req.source.startsWith("onnx:")) {
    file = req.source.slice(5);
    if (!existsSync(path.join(m.dir, file))) throw new Error(`${file} is not downloaded`);
    dir = m.dir;
    if (req.quant !== "none") {
      const q = path.join(cache, `onnx-${slug(file)}-${req.quant}`);
      steps.push({ label: `ONNX quantization (${req.quant})`, cmd: t.onnxPython, args: [t.onnxScript, "quantize", "--src", m.dir, "--onnx-file", file, "--out", q, "--method", req.quant], output: path.join(q, file) });
      dir = q;
    }
  } else throw new Error(`unknown ONNX source ${req.source}`);
  return {
    steps,
    artifact: file ? path.join(dir, file) : dir,
    contextLength: ctx,
    serve: (port, key, served) => ({
      cmd: t.onnxPython,
      args: [
        t.serveScript,
        "--model-dir", dir,
        "--backend", "onnx",
        "--kind", req.kind,
        "--served-name", served,
        "--port", String(port),
        "--device", req.device,
        "--max-context", String(ctx),
        ...(file ? ["--onnx-file", file] : []),
        ...(req.trustRemoteCode ? ["--trust-remote-code"] : []),
      ],
      env: { RLM_HUB_API_KEY: key, PYTHONUNBUFFERED: "1" },
    }),
  };
}

/** Bytes of a folder, skipping nothing (used for the library list). */
export function folderBytes(dir: string): number {
  let n = 0;
  const walk = (d: string) => {
    for (const e of readdirSync(d, { withFileTypes: true })) {
      const p = path.join(d, e.name);
      if (e.isDirectory()) walk(p);
      else if (e.isFile()) n += statSync(p).size;
    }
  };
  if (existsSync(dir)) walk(dir);
  return n;
}
