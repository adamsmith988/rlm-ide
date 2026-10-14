import { existsSync, mkdirSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import type { HubOption, HubPythonEnv, HubPythonView, HubToolsView } from "@rlm/contracts";
import { lastLines, run } from "./proc";

const exe = (name: string) => (process.platform === "win32" ? `${name}.exe` : name);
const LLAMA_REPO = "https://github.com/ggml-org/llama.cpp";

export interface PythonProbe {
  python: string;
  executable: string;
  packages: Record<string, string | null>;
  cuda: boolean;
  gpu: { name: string; freeMb: number; totalMb: number } | null;
  transformersModelTypes: string[];
  onnxExportModelTypes: string[];
  onnxQuantizers: string[];
}

export interface LlamaCpp {
  dir: string | null;
  server: string | null;
  quantize: string | null;
  imatrix: string | null;
  convert: string | null;
  backend: "cpu" | "cuda" | null;
  version: string | null;
  quantTypes: HubOption[];
  convertArchitectures: string[];
  convertOutTypes: string[];
}

/** "  15  or  Q4_K_M  :  4.58G, +0.1754 ppl @ Llama-3-8B" -> { id: "Q4_K_M", detail: "4.58G, ..." } */
export function parseQuantTypes(help: string): HubOption[] {
  const out: HubOption[] = [];
  const seen = new Set<string>();
  for (const line of help.split("\n")) {
    const m = /^\s*\d+\s+or\s+([A-Za-z0-9_]+)\s*:\s*(.*)$/.exec(line);
    if (!m || seen.has(m[1]!)) continue;
    seen.add(m[1]!);
    out.push({ id: m[1]!, label: m[1]!, detail: m[2]!.trim() });
  }
  return out;
}

/** Architectures the converter registers: @ModelBase.register("LlamaForCausalLM", ...) (older: @Model.register). */
export function parseConvertArchitectures(source: string): string[] {
  const out = new Set<string>();
  for (const m of source.matchAll(/@(?:ModelBase|Model|TextModel|MmprojModel)\.register\(([^)]*)\)/g))
    for (const s of m[1]!.matchAll(/["']([A-Za-z0-9_]+)["']/g)) out.add(s[1]!);
  return [...out].sort();
}

/** argparse choices of --outtype in the converter. */
export function parseConvertOutTypes(source: string): string[] {
  const m = /"--outtype"[^)]*?choices=\[([^\]]*)\]/s.exec(source);
  return m ? [...m[1]!.matchAll(/["']([A-Za-z0-9_]+)["']/g)].map((x) => x[1]!) : ["f32", "f16", "bf16", "q8_0"];
}

export class Toolchain {
  private probeCache: PythonProbe | null = null;
  private probeError: string | null = null;
  private onnxProbeCache: PythonProbe | null = null;
  private llamaCache: LlamaCpp | null = null;
  installing: { tool: string; log: string } | null = null;

  constructor(
    private readonly o: {
      dir: string;
      scriptsDir: string;
      basePythons: () => string[];
      configuredPython: () => string;
      configuredLlamaCpp: () => string;
      onChange: () => void;
    },
  ) {
    mkdirSync(o.dir, { recursive: true });
  }

  venvDir(env: HubPythonEnv = "main"): string {
    return path.join(this.o.dir, env === "main" ? "venv" : `venv-${env}`);
  }

  private venvPython(env: HubPythonEnv = "main"): string {
    const dir = this.venvDir(env);
    return process.platform === "win32" ? path.join(dir, "Scripts", "python.exe") : path.join(dir, "bin", "python");
  }

  /** The Python deployments run with: configured, else the IDE's own venv, else the first base Python that works.
   *  The ONNX environment falls back to the main one until it is created. */
  python(env: HubPythonEnv = "main"): string {
    if (env !== "main" && existsSync(this.venvPython(env))) return this.venvPython(env);
    const configured = this.o.configuredPython();
    if (configured) return configured;
    if (existsSync(this.venvPython())) return this.venvPython();
    return this.probeCache?.executable ?? this.o.basePythons()[0] ?? "python3";
  }

  ownPython(env: HubPythonEnv = "main"): boolean {
    return env === "main" ? !this.o.configuredPython() && existsSync(this.venvPython()) : existsSync(this.venvPython(env));
  }

  /** Probe of the environment ONNX runs in. */
  async probeOnnx(force = false): Promise<PythonProbe | null> {
    if (!existsSync(this.venvPython("onnx"))) {
      this.onnxProbeCache = null;
      return this.probe(force);
    }
    if (this.onnxProbeCache && !force) return this.onnxProbeCache;
    const r = await run(this.venvPython("onnx"), [this.script("probe.py")], { keepChars: 4_000_000 });
    const line = r.tail.trim().split("\n").reverse().find((l) => l.startsWith("{"));
    this.onnxProbeCache = r.code === 0 && line ? (JSON.parse(line) as PythonProbe) : null;
    this.o.onChange();
    return this.onnxProbeCache;
  }

  script(name: string): string {
    return path.join(this.o.scriptsDir, name);
  }

  private probing: Promise<PythonProbe | null> | null = null;

  async probe(force = false): Promise<PythonProbe | null> {
    if (this.probeCache && !force) return this.probeCache;
    if (this.probing) return this.probing;
    this.probing = this.probeNow();
    this.o.onChange();
    try {
      return await this.probing;
    } finally {
      this.probing = null;
      this.o.onChange();
    }
  }

  private async probeNow(): Promise<PythonProbe | null> {
    const candidates = this.o.configuredPython() ? [this.o.configuredPython()] : existsSync(this.venvPython()) ? [this.venvPython()] : this.o.basePythons();
    let lastErr = "no Python found";
    for (const py of candidates) {
      const r = await run(py, [this.script("probe.py")], { keepChars: 4_000_000 });
      const line = r.tail.trim().split("\n").reverse().find((l) => l.startsWith("{"));
      if (r.code === 0 && line) {
        this.probeCache = JSON.parse(line) as PythonProbe;
        this.probeError = null;
        this.o.onChange();
        return this.probeCache;
      }
      lastErr = `${py}: ${lastLines(r.tail, 4)}`;
    }
    this.probeCache = null;
    this.probeError = lastErr;
    this.o.onChange();
    return null;
  }

  cached(): PythonProbe | null {
    return this.probeCache;
  }

  private async task<T>(tool: string, fn: (log: (l: string) => void) => Promise<T>): Promise<T> {
    if (this.installing) throw new Error(`${this.installing.tool} is being installed; wait for it to finish`);
    this.installing = { tool, log: "" };
    let last = 0;
    const log = (l: string) => {
      this.installing!.log = (this.installing!.log + l + "\n").slice(-8000);
      if (Date.now() - last > 500) {
        last = Date.now();
        this.o.onChange();
      }
    };
    this.o.onChange();
    try {
      return await fn(log);
    } finally {
      this.installing = null;
      this.o.onChange();
    }
  }

  /** An IDE-owned venv that sees the base Python's packages (torch is large); what it installs shadows them only inside it. */
  async setupPython(env: HubPythonEnv = "main"): Promise<void> {
    await this.task(`python environment (${env})`, async (log) => {
      if (!existsSync(this.venvPython(env))) {
        const base = this.probeCache?.executable ?? this.o.basePythons()[0];
        if (!base) throw new Error("no Python found to create the environment from");
        log(`creating ${this.venvDir(env)} from ${base}`);
        const r = await run(base, ["-m", "venv", "--system-site-packages", this.venvDir(env)], { onLine: log });
        if (r.code !== 0) throw new Error(`venv failed: ${lastLines(r.tail)}`);
      }
    });
    await (env === "main" ? this.probe(true) : this.probeOnnx(true));
  }

  async pipInstall(packages: string[], env: HubPythonEnv = "main"): Promise<void> {
    if (!packages.length) return;
    const needsVenv = env === "main" ? !this.o.configuredPython() && !existsSync(this.venvPython()) : !existsSync(this.venvPython(env));
    if (needsVenv) await this.setupPython(env);
    await this.task(`pip (${env}): ${packages.join(" ")}`, async (log) => {
      const r = await run(this.python(env), ["-m", "pip", "install", "--disable-pip-version-check", ...packages], { onLine: log });
      if (r.code !== 0) throw new Error(`pip install failed: ${lastLines(r.tail)}`);
    });
    await (env === "main" ? this.probe(true) : this.probeOnnx(true));
  }

  private findLlamaBin(name: string, dirs: string[]): string | null {
    for (const d of dirs) {
      for (const sub of ["", "bin", path.join("build", "bin"), path.join("build", "bin", "Release")]) {
        const p = path.join(d, sub, exe(name));
        if (existsSync(p)) return p;
      }
    }
    for (const d of (process.env.PATH ?? "").split(path.delimiter)) {
      const p = path.join(d, exe(name));
      if (d && existsSync(p)) return p;
    }
    return null;
  }

  get ownLlamaDir(): string {
    return path.join(this.o.dir, "llama.cpp");
  }

  async llamaCpp(force = false): Promise<LlamaCpp> {
    if (this.llamaCache && !force) return this.llamaCache;
    const dirs = [this.o.configuredLlamaCpp(), this.ownLlamaDir].filter(Boolean);
    const server = this.findLlamaBin("llama-server", dirs);
    const quantize = this.findLlamaBin("llama-quantize", dirs);
    const imatrix = this.findLlamaBin("llama-imatrix", dirs);
    const dir = dirs.find((d) => existsSync(d)) ?? null;
    const convert = dirs.map((d) => path.join(d, "convert_hf_to_gguf.py")).find((p) => existsSync(p)) ?? null;
    let version: string | null = null;
    let backend: LlamaCpp["backend"] = null;
    if (server) {
      const r = await run(server, ["--version"]);
      version = /version:\s*(\S+.*)/.exec(r.tail)?.[1]?.trim() ?? lastLines(r.tail, 1);
      const binDir = path.dirname(server);
      const libs = existsSync(binDir) ? readdirSync(binDir).join(" ") : "";
      backend = /cuda/i.test(libs) || /CUDA/.test(r.tail) ? "cuda" : "cpu";
    }
    let quantTypes: HubOption[] = [];
    if (quantize) quantTypes = parseQuantTypes((await run(quantize, ["--help"], { keepChars: 1_000_000 })).tail);
    let convertArchitectures: string[] = [];
    let convertOutTypes: string[] = [];
    if (convert) {
      const src = readFileSync(convert, "utf8");
      const pkg = path.join(path.dirname(convert), "conversion");
      const modules = existsSync(pkg)
        ? readdirSync(pkg).filter((f) => f.endsWith(".py")).map((f) => readFileSync(path.join(pkg, f), "utf8"))
        : [];
      convertArchitectures = parseConvertArchitectures([src, ...modules].join("\n"));
      convertOutTypes = parseConvertOutTypes(src);
    }
    this.llamaCache = { dir, server, quantize, imatrix, convert, backend, version, quantTypes, convertArchitectures, convertOutTypes };
    this.o.onChange();
    return this.llamaCache;
  }

  /** Clones llama.cpp and builds the server, quantizer and imatrix tools; installs the converter's missing Python packages. */
  async installLlamaCpp(backend: "cpu" | "cuda"): Promise<void> {
    await this.task(`llama.cpp (${backend})`, async (log) => {
      const dir = this.ownLlamaDir;
      if (!existsSync(path.join(dir, "CMakeLists.txt"))) {
        log(`cloning ${LLAMA_REPO}`);
        const r = await run("git", ["clone", "--depth", "1", LLAMA_REPO, dir], { onLine: log });
        if (r.code !== 0) throw new Error(`git clone failed: ${lastLines(r.tail)}`);
      }
      const build = path.join(dir, "build");
      const cfg = ["-S", dir, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", `-DGGML_CUDA=${backend === "cuda" ? "ON" : "OFF"}`];
      log(`cmake ${cfg.join(" ")}`);
      let r = await run("cmake", cfg, { onLine: log });
      if (r.code !== 0) throw new Error(`cmake configure failed: ${lastLines(r.tail)}`);
      const jobs = String(Math.max(1, (await import("node:os")).cpus().length));
      r = await run("cmake", ["--build", build, "--config", "Release", "-j", jobs, "--target", "llama-server", "llama-quantize", "llama-imatrix"], { onLine: log });
      if (r.code !== 0) throw new Error(`build failed: ${lastLines(r.tail)}`);
      log("build done");
    });
    await this.convertRequirements();
    await this.llamaCpp(true);
  }

  /** Installs only the converter requirements this Python lacks (torch and transformers are usually there already). */
  async convertRequirements(): Promise<string[]> {
    const dir = (await this.llamaCpp(true)).dir;
    if (!dir) return [];
    const names = new Set<string>();
    const read = (f: string) => {
      if (!existsSync(f)) return;
      for (const raw of readFileSync(f, "utf8").split("\n")) {
        const line = raw.trim();
        if (!line || line.startsWith("#") || line.startsWith("--")) continue;
        if (line.startsWith("-r")) {
          read(path.join(path.dirname(f), line.slice(2).trim()));
          continue;
        }
        const name = /^([A-Za-z0-9_.\-]+)/.exec(line)?.[1];
        if (name) names.add(name.toLowerCase());
      }
    };
    read(path.join(dir, "requirements", "requirements-convert_hf_to_gguf.txt"));
    names.delete("gguf");
    if (!(await this.probe(true))) return [];
    const r = await run(this.python(), [
      "-c",
      "import importlib.metadata as m, json, sys\nmissing = []\nfor n in json.loads(sys.argv[1]):\n    try: m.version(n)\n    except m.PackageNotFoundError: missing.append(n)\nprint(json.dumps(missing))",
      JSON.stringify([...names]),
    ]);
    const missing = r.code === 0 ? (JSON.parse(r.tail.trim().split("\n").pop() ?? "[]") as string[]) : [];
    if (missing.length) await this.pipInstall(missing);
    return missing;
  }

  view(hfToken: boolean): HubToolsView {
    const l = this.llamaCache;
    const pyView = (p: PythonProbe | null, env: HubPythonEnv): HubPythonView | null =>
      p
        ? {
            path: p.executable,
            version: p.python,
            own: this.ownPython(env),
            packages: p.packages,
            cuda: p.cuda,
            gpu: p.gpu,
            transformersModelTypes: p.transformersModelTypes.length,
            onnxExportModelTypes: p.onnxExportModelTypes.length,
            onnxQuantizers: p.onnxQuantizers,
          }
        : null;
    return {
      python: pyView(this.probeCache, "main"),
      pythonError: this.probeError,
      probing: !!this.probing,
      onnxPython: pyView(this.onnxProbeCache, "onnx"),
      llamaCpp: {
        dir: l?.dir ?? null,
        server: l?.server ?? null,
        quantize: l?.quantize ?? null,
        convert: l?.convert ?? null,
        backend: l?.backend ?? null,
        version: l?.version ?? null,
        quantTypes: l?.quantTypes ?? [],
        convertArchitectures: l?.convertArchitectures.length ?? 0,
      },
      cudaToolkit: this.cudaToolkit(),
      installing: this.installing,
      hfToken,
    };
  }

  cudaToolkit(): boolean {
    const dirs = [...(process.env.PATH ?? "").split(path.delimiter), "/usr/local/cuda/bin", process.env.CUDA_PATH ? path.join(process.env.CUDA_PATH, "bin") : ""];
    return dirs.some((d) => d && existsSync(path.join(d, exe("nvcc"))));
  }
}
