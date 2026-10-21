import { randomBytes, randomUUID } from "node:crypto";
import { spawn, type ChildProcess } from "node:child_process";
import { chmodSync, createWriteStream, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync, renameSync } from "node:fs";
import net from "node:net";
import path from "node:path";
import type {
  DeployTargetView,
  HubDeployRequest,
  HubDeploymentView,
  HubJobView,
  HubModelView,
  HubRepoView,
  HubStateView,
  HubPythonEnv,
  ModelConfig,
  ProviderConfig,
  Settings,
} from "@rlm/contracts";
import { deployTargets, folderBytes, planDeployment, type StepSpec, type ToolsSnapshot } from "./deploy";
import { DownloadManager, type JobSpec } from "./download";
import { HfClient, HubError, parseRepoRef, type RepoFile } from "./hf";
import { lastLines, pipeLines, run } from "./proc";
import { defaultSelection, profileOf } from "./profile";
import { Toolchain } from "./toolchain";

export interface HubDeps {
  /** userData/hub */
  dir: string;
  /** python/rlm_hub */
  scriptsDir: string;
  basePythons: () => string[];
  settings: () => Settings;
  updateSettings: (patch: Partial<Settings>) => void;
  secrets: { get(name: string): string | null; set(name: string, value: string): void };
  emit: (state: HubStateView) => void;
  log: (msg: string) => void;
  hfBase?: string;
}

interface StoredJob {
  view: HubJobView;
  spec: JobSpec;
  profile: HubModelView["profile"];
}

interface StoredDeployment extends HubDeploymentView {
  /** Was serving when the IDE closed; restarted on the next launch. */
  wasRunning?: boolean;
  port?: number;
}

interface Persisted {
  jobs: StoredJob[];
  deployments: StoredDeployment[];
}

const HF_TOKEN = "hf:token";
const LOG_TAIL = 5000;
const MODEL_META = path.join(".rlm", "model.json");

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const s = net.createServer();
    s.unref();
    s.on("error", reject);
    s.listen(0, "127.0.0.1", () => {
      const port = (s.address() as net.AddressInfo).port;
      s.close(() => resolve(port));
    });
  });
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const dirName = (repo: string) => repo.replace("/", "__");

export class HubService {
  readonly hf: HfClient;
  readonly tools: Toolchain;
  private readonly downloads: DownloadManager;
  private data: Persisted = { jobs: [], deployments: [] };
  private models = new Map<string, HubModelView>();
  private procs = new Map<string, ChildProcess>();
  private busy = new Set<string>();
  private emitTimer: NodeJS.Timeout | null = null;
  private saveTimer: NodeJS.Timeout | null = null;
  private closing = false;
  private follower = false;
  private readonly shutdown = new AbortController();

  constructor(private readonly d: HubDeps) {
    mkdirSync(this.modelsDir, { recursive: true });
    mkdirSync(this.deployDir, { recursive: true });
    this.hf = new HfClient(() => d.secrets.get(HF_TOKEN), d.hfBase);
    this.tools = new Toolchain({
      dir: path.join(d.dir, "tools"),
      scriptsDir: d.scriptsDir,
      basePythons: d.basePythons,
      configuredPython: () => d.settings().hub.python,
      configuredLlamaCpp: () => d.settings().hub.llamaCppDir,
      llamaRelease: () => ({ repo: d.settings().hub.llamaCppRepo, release: d.settings().hub.llamaCppRelease }),
      onChange: () => this.changed(),
    });
    this.downloads = new DownloadManager(this.hf, {
      concurrency: () => d.settings().hub.downloadConcurrency,
      onChange: (v) => this.jobChanged(v),
      onFinished: (v, spec) => this.jobFinished(v, spec),
    });
  }

  get modelsDir(): string {
    return path.join(this.d.dir, "models");
  }
  get deployDir(): string {
    return path.join(this.d.dir, "deployments");
  }
  private get stateFile(): string {
    return path.join(this.d.dir, "state.json");
  }

  // ---- lifecycle ------------------------------------------------------------------------------

  /** `follower`: another copy of the IDE on this profile runs the deployments; this one shows them and changes nothing. */
  async boot(opts: { follower?: boolean } = {}): Promise<void> {
    this.follower = !!opts.follower;
    if (existsSync(this.stateFile)) {
      try {
        this.data = { jobs: [], deployments: [], ...(JSON.parse(readFileSync(this.stateFile, "utf8")) as Partial<Persisted>) };
      } catch (e) {
        this.d.log(`hub state unreadable, starting empty: ${(e as Error).message}`);
      }
    }
    if (this.follower) {
      this.scanModels();
      await Promise.all([this.tools.probe().catch(() => null), this.tools.llamaCpp().catch(() => null)]);
      this.changed();
      return;
    }
    for (const j of this.data.jobs) if (j.view.status === "running" || j.view.status === "queued" || j.view.status === "verifying") j.view.status = "paused";
    this.scanModels();
    const restore = this.data.deployments.filter((x) => x.wasRunning);
    for (const dep of this.data.deployments) {
      if (dep.status !== "error") dep.status = "stopped";
      dep.wasRunning = false;
    }
    this.save();
    await Promise.all([this.tools.probe().catch(() => null), this.tools.llamaCpp().catch(() => null)]);
    await this.tools.probeOnnx().catch(() => null);
    if (this.d.settings().hub.restoreDeployments) for (const dep of restore) void this.start(dep.id).catch((e) => this.d.log(`restore ${dep.id}: ${(e as Error).message}`));
    this.changed();
  }

  close(): void {
    this.closing = true;
    this.shutdown.abort();
    this.downloads.stopAll();
    for (const dep of this.data.deployments) if (this.procs.has(dep.id) || this.busy.has(dep.id)) dep.wasRunning = true;
    for (const [, p] of this.procs) p.kill();
    this.procs.clear();
    this.saveNow();
  }

  // ---- state ----------------------------------------------------------------------------------

  get isFollower(): boolean {
    return this.follower;
  }

  view(): HubStateView {
    if (this.follower) {
      try {
        this.data = { jobs: [], deployments: [], ...(JSON.parse(readFileSync(this.stateFile, "utf8")) as Partial<Persisted>) };
      } catch {
        /* the leader is rewriting it; show the last state */
      }
    }
    return {
      jobs: this.data.jobs.map((j) => j.view),
      models: [...this.models.values()].sort((a, b) => b.downloadedAt - a.downloadedAt),
      deployments: this.data.deployments.map(({ wasRunning: _w, port: _p, ...v }) => v),
      tools: this.tools.view(this.hf.hasToken()),
    };
  }

  private changed(): void {
    if (this.emitTimer) return;
    this.emitTimer = setTimeout(() => {
      this.emitTimer = null;
      this.d.emit(this.view());
    }, 200);
  }

  private save(): void {
    if (this.saveTimer) return;
    this.saveTimer = setTimeout(() => this.saveNow(), 500);
  }

  private saveNow(): void {
    if (this.saveTimer) clearTimeout(this.saveTimer);
    this.saveTimer = null;
    if (this.follower) return;
    const tmp = `${this.stateFile}.tmp`;
    writeFileSync(tmp, JSON.stringify(this.data));
    renameSync(tmp, this.stateFile);
  }

  private scanModels(): void {
    this.models.clear();
    for (const name of existsSync(this.modelsDir) ? readdirSync(this.modelsDir) : []) {
      const meta = path.join(this.modelsDir, name, MODEL_META);
      if (!existsSync(meta)) continue;
      try {
        const m = JSON.parse(readFileSync(meta, "utf8")) as HubModelView;
        this.models.set(m.id, { ...m, dir: path.join(this.modelsDir, name) });
      } catch {
        /* a broken record is skipped; the files stay */
      }
    }
  }

  // ---- lookup & download ----------------------------------------------------------------------

  async lookup(input: string, revision = ""): Promise<HubRepoView> {
    const { repo, revision: rev } = parseRepoRef(input, revision);
    const info = await this.hf.info(repo, rev);
    const sha = info.sha ?? rev;
    const files = await this.hf.tree(repo, sha);
    const json = async (f: string) => {
      if (!files.some((x) => x.path === f)) return null;
      const t = await this.hf.text(repo, sha, f);
      try {
        return t ? (JSON.parse(t) as Record<string, unknown>) : null;
      } catch {
        return null;
      }
    };
    const [config, tokenizerConfig, sentenceBertConfig] = await Promise.all([json("config.json"), json("tokenizer_config.json"), json("sentence_bert_config.json")]);
    const profile = profileOf({ info, files, config, tokenizerConfig, sentenceBertConfig });
    let access: HubRepoView["access"] = "ok";
    let accessDetail = "";
    const probeFile = files.find((f) => f.path === "config.json") ?? files.find((f) => !f.path.startsWith(".")) ?? files[0];
    if (probeFile && (info.gated || info.private)) {
      const can = await this.hf.canDownload(repo, sha, probeFile.path);
      if (!can.ok) {
        access = this.hf.hasToken() ? "needs-approval" : "needs-token";
        accessDetail = this.hf.hasToken()
          ? `HTTP ${can.status}: the token's account has not been granted access; accept the license on https://huggingface.co/${repo}`
          : `HTTP ${can.status}: this ${info.gated ? "gated" : "private"} repo needs a Hugging Face token (Models hub → Tools).`;
      }
    }
    return {
      repo,
      revision: rev,
      sha,
      gated: info.gated ?? false,
      private: !!info.private,
      access,
      accessDetail,
      tags: info.tags ?? [],
      files: files.map((f) => ({ path: f.path, size: f.size, lfs: !!f.sha256 })),
      totalBytes: files.reduce((n, f) => n + f.size, 0),
      defaultSelection: defaultSelection(files),
      profile,
    };
  }

  async download(input: string, revision: string, selected: string[]): Promise<HubJobView> {
    const view = await this.lookup(input, revision);
    if (view.access !== "ok") throw new HubError(view.accessDetail, 401);
    const want = new Set(selected);
    const all = await this.hf.tree(view.repo, view.sha);
    const files: RepoFile[] = all.filter((f) => want.has(f.path));
    const unknown = selected.filter((s) => !all.some((f) => f.path === s));
    if (unknown.length) throw new Error(`not in the repo: ${unknown.slice(0, 5).join(", ")}`);
    let dest = path.join(this.modelsDir, dirName(view.repo));
    const existing = this.models.get(dirName(view.repo));
    if (existing && existing.sha !== view.sha) dest = path.join(this.modelsDir, `${dirName(view.repo)}@${view.sha.slice(0, 10)}`);
    const active = this.data.jobs.find((j) => j.spec.dest === dest && ["running", "queued", "paused", "verifying"].includes(j.view.status));
    if (active) throw new Error(`${view.repo} is already downloading; resume or cancel that download first`);
    const spec: JobSpec = { id: randomUUID().slice(0, 12), repo: view.repo, revision: view.revision, sha: view.sha, dest, files };
    const job = this.downloads.start(spec);
    this.data.jobs.unshift({ view: job, spec, profile: view.profile });
    this.save();
    this.changed();
    return job;
  }

  jobAction(id: string, action: "pause" | "resume" | "cancel" | "remove"): HubJobView | null {
    const j = this.data.jobs.find((x) => x.view.id === id);
    if (!j) throw new Error(`no download ${id}`);
    if (action === "pause") this.downloads.pause(id);
    else if (action === "cancel") {
      if (!this.downloads.cancel(id)) j.view.status = "cancelled";
      this.dropPartials(j.spec);
    } else if (action === "resume") {
      if (!["paused", "error", "cancelled"].includes(j.view.status)) return j.view;
      j.view = this.downloads.start(j.spec, j.view);
    } else {
      if (this.downloads.isRunning(id)) throw new Error("pause or cancel the download first");
      this.data.jobs = this.data.jobs.filter((x) => x !== j);
    }
    this.save();
    this.changed();
    return j.view;
  }

  private dropPartials(spec: JobSpec): void {
    for (const f of spec.files) rmSync(path.join(spec.dest, `${f.path}.part`), { force: true });
  }

  private jobChanged(v: HubJobView): void {
    const j = this.data.jobs.find((x) => x.view.id === v.id);
    if (j) j.view = v;
    this.changed();
  }

  private jobFinished(v: HubJobView, spec: JobSpec): void {
    const j = this.data.jobs.find((x) => x.view.id === v.id);
    if (j) j.view = v;
    if (v.status === "cancelled") this.dropPartials(spec);
    if (v.status === "done" && j) {
      const id = path.basename(spec.dest);
      const prev = this.models.get(id);
      const files = [...new Set([...(prev?.sha === spec.sha ? prev.files : []), ...spec.files.map((f) => f.path)])].sort();
      const m: HubModelView = { id, repo: spec.repo, revision: spec.revision, sha: spec.sha, dir: spec.dest, bytes: 0, files, downloadedAt: Date.now(), profile: j.profile };
      m.bytes = folderBytes(spec.dest);
      mkdirSync(path.join(spec.dest, ".rlm"), { recursive: true });
      writeFileSync(path.join(spec.dest, MODEL_META), JSON.stringify(m, null, 2));
      this.models.set(id, m);
      this.d.log(`downloaded ${spec.repo} (${spec.files.length} files) to ${spec.dest}`);
    }
    if (!this.closing) this.save();
    this.changed();
  }

  deleteModel(id: string): void {
    const m = this.models.get(id);
    if (!m) throw new Error(`no model ${id}`);
    const live = this.data.deployments.filter((x) => x.modelId === id && this.procs.has(x.id));
    if (live.length) throw new Error(`stop its deployments first (${live.map((x) => x.servedName).join(", ")})`);
    for (const dep of this.data.deployments.filter((x) => x.modelId === id)) this.removeDeployment(dep.id);
    rmSync(m.dir, { recursive: true, force: true });
    this.models.delete(id);
    this.save();
    this.changed();
  }

  // ---- deployment -----------------------------------------------------------------------------

  private async snapshot(): Promise<ToolsSnapshot> {
    const [probe, onnxProbe, llama] = await Promise.all([this.tools.probe(), this.tools.probeOnnx(), this.tools.llamaCpp()]);
    return { python: this.tools.python(), probe, onnxPython: this.tools.python("onnx"), onnxProbe, llama, serveScript: this.tools.script("serve.py"), onnxScript: this.tools.script("onnx_tools.py"), contextCap: this.d.settings().hub.contextCap };
  }

  async targets(modelId: string): Promise<DeployTargetView[]> {
    const m = this.models.get(modelId);
    if (!m) throw new Error(`no model ${modelId}`);
    return deployTargets(m, await this.snapshot());
  }

  async deploy(req: HubDeployRequest): Promise<HubDeploymentView> {
    const m = this.models.get(req.modelId);
    if (!m) throw new Error(`no model ${req.modelId}`);
    const t = (await this.targets(req.modelId)).find((x) => x.id === req.target);
    if (!t) throw new Error(`unknown target ${req.target}`);
    if (!t.available) throw new Error(t.reason);
    if (t.needs.length) throw new Error(`install first: ${t.needs.join(", ")}`);
    if (!t.sources.some((s) => s.id === req.source)) throw new Error(`source ${req.source} is not available for ${t.label}`);
    if (!(t.quantizations[req.source] ?? []).some((q) => q.id === req.quant)) throw new Error(`quantization ${req.quant} is not available for this source`);
    if (!t.kinds.includes(req.kind)) throw new Error(`${t.label} cannot serve this model as ${req.kind}`);
    if (!t.devices.includes(req.device)) throw new Error(`device ${req.device} is not available for ${t.label}`);
    const id = randomUUID().slice(0, 8);
    const quantTag = req.quant === "none" || req.quant === "as-is" ? (req.source.startsWith("gguf:") ? path.basename(req.source, ".gguf") : req.target) : req.quant;
    const servedName = `${path.basename(m.repo)}-${quantTag}`.replace(/[^A-Za-z0-9._-]+/g, "-").toLowerCase();
    const dep: StoredDeployment = {
      id,
      modelId: m.id,
      repo: m.repo,
      request: req,
      status: "preparing",
      steps: [],
      artifact: null,
      baseUrl: null,
      servedName,
      providerId: null,
      settingsModelId: null,
      contextLength: null,
      dim: null,
      error: null,
      logTail: "",
      createdAt: Date.now(),
    };
    this.data.deployments.unshift(dep);
    this.save();
    this.changed();
    void this.start(id).catch(() => undefined);
    return dep;
  }

  private logFile(id: string): string {
    return path.join(this.deployDir, `${id}.log`);
  }

  private appendLog(dep: StoredDeployment, line: string): void {
    dep.logTail = (dep.logTail + line + "\n").slice(-LOG_TAIL);
    this.changed();
  }

  /** Runs one planned step shown at dep.steps[index]; returns how many step rows it used (recovery adds rows). */
  private async runStep(dep: StoredDeployment, s: StepSpec, index: number): Promise<number> {
    const step = dep.steps[index]!;
    if (s.output && existsSync(s.output)) {
      step.status = "skipped";
      step.detail = `reused ${path.basename(s.output)}`;
      this.changed();
      return 1;
    }
    step.status = "running";
    this.changed();
    if (s.output) mkdirSync(path.dirname(s.output), { recursive: true });
    const tmpOut = s.output && path.extname(s.output) ? `${s.output}.partial${path.extname(s.output)}` : null;
    const args = tmpOut ? s.args.map((a) => (a === s.output ? tmpOut : a)) : s.args;
    const r = await run(s.cmd, args, {
      cwd: s.cwd,
      env: s.env,
      signal: this.shutdown.signal,
      logFile: this.logFile(dep.id),
      onLine: (l) => {
        step.detail = l.slice(0, 200);
        this.appendLog(dep, l);
      },
    });
    if (r.code === 0 && tmpOut && existsSync(tmpOut)) renameSync(tmpOut, s.output!);
    if (r.code === 0) {
      step.status = "done";
      step.detail = undefined;
      return 1;
    }
    if (tmpOut) rmSync(tmpOut, { force: true });
    const alt = s.recover?.(r.tail);
    if (alt?.length) {
      step.status = "skipped";
      step.detail = `${lastLines(r.tail, 1).slice(0, 160)} → taking another route`;
      dep.steps.splice(index + 1, 0, ...alt.map((a) => ({ label: a.label, status: "todo" as const })));
      let used = 1;
      for (const a of alt) used += await this.runStep(dep, a, index + used);
      return used;
    }
    step.status = "error";
    throw new Error(`${s.label} failed (exit ${r.code}):\n${lastLines(r.tail, 15)}`);
  }

  /** Prepares (convert / quantize / export, skipping work already done) and starts the server. */
  async start(id: string): Promise<HubDeploymentView> {
    const dep = this.data.deployments.find((x) => x.id === id);
    if (!dep) throw new Error(`no deployment ${id}`);
    if (this.procs.has(id) || this.busy.has(id)) return dep;
    const m = this.models.get(dep.modelId);
    if (!m) throw new Error(`model ${dep.modelId} was deleted`);
    this.busy.add(id);
    dep.error = null;
    dep.status = "preparing";
    this.changed();
    try {
      const tools = await this.snapshot();
      const plan = planDeployment(m, dep.request, tools, this.deployDir);
      dep.steps = [...plan.steps.map((s) => ({ label: s.label, status: "todo" as const })), { label: "start server", status: "todo" as const }, { label: "check the endpoint", status: "todo" as const }];
      this.changed();
      let at = 0;
      for (const s of plan.steps) at += await this.runStep(dep, s, at);
      dep.artifact = plan.artifact;
      dep.contextLength = plan.contextLength;
      const serveStep = dep.steps[dep.steps.length - 2]!;
      const checkStep = dep.steps[dep.steps.length - 1]!;
      serveStep.status = "running";
      dep.status = "starting";
      this.changed();
      const key = this.d.secrets.get(`provider:hub-${id}`) ?? `rlm-hub-${randomBytes(24).toString("hex")}`;
      const port = await freePort();
      if (this.closing) throw new Error("the IDE is shutting down");
      const keyDir = path.join(this.deployDir, id);
      mkdirSync(keyDir, { recursive: true, mode: 0o700 });
      const keyFile = path.join(keyDir, "api-key");
      writeFileSync(keyFile, `${key}\n`, { mode: 0o600 });
      chmodSync(keyFile, 0o600);
      const spec = plan.serve(port, key, dep.servedName, keyFile);
      const log = createWriteStream(this.logFile(id), { flags: "a" });
      log.write(`\n$ ${spec.cmd} ${spec.args.join(" ")}\n`);
      const child = spawn(spec.cmd, spec.args, { env: { ...process.env, ...spec.env }, stdio: ["ignore", "pipe", "pipe"] });
      this.procs.set(id, child);
      let exited: string | null = null;
      pipeLines(child, (l) => this.appendLog(dep, l), log);
      child.on("error", (e) => (exited = e.message));
      child.on("exit", (code, sig) => {
        log.end();
        exited = `server exited (${sig ?? code})`;
        if (this.procs.get(id) === child) this.procs.delete(id);
        if (!this.closing && dep.status === "running") {
          dep.status = "error";
          dep.error = `${exited}:\n${lastLines(dep.logTail, 12)}`;
          this.save();
        }
        this.changed();
      });
      const baseUrl = `http://127.0.0.1:${port}/v1`;
      const deadline = Date.now() + this.d.settings().hub.startTimeoutSec * 1000;
      let ready = false;
      while (Date.now() < deadline && !exited && !this.closing) {
        try {
          const res = await fetch(`${baseUrl}/models`, { headers: { Authorization: `Bearer ${key}` }, signal: AbortSignal.timeout(5000) });
          if (res.ok) {
            ready = true;
            break;
          }
        } catch {
          /* still loading */
        }
        await sleep(1000);
      }
      rmSync(keyFile, { force: true });
      if (!ready) {
        child.kill();
        throw new Error(`${exited ?? `the server did not answer within ${this.d.settings().hub.startTimeoutSec} s`}:\n${lastLines(dep.logTail, 15)}`);
      }
      const open = await fetch(`${baseUrl}/models`, { signal: AbortSignal.timeout(5000) }).then((r) => r.status, () => 0);
      if (open !== 401 && open !== 403) {
        child.kill();
        throw new Error(`the server answered without the API key (HTTP ${open}); refusing to expose it unauthenticated`);
      }
      serveStep.status = "done";
      checkStep.status = "running";
      dep.baseUrl = baseUrl;
      dep.port = port;
      this.changed();
      const check = await this.probeEndpoint(baseUrl, key, dep);
      if (!check.ok) {
        child.kill();
        checkStep.status = "error";
        throw new Error(`the endpoint answered /models but failed a ${dep.request.kind} request: ${check.detail}`);
      }
      checkStep.status = "done";
      checkStep.detail = check.detail;
      this.register(dep, key);
      dep.status = "running";
      this.d.log(`deployment ${dep.servedName} serving at ${baseUrl}`);
      return dep;
    } catch (e) {
      if (this.closing) {
        this.procs.get(id)?.kill();
        dep.status = "stopped";
        for (const s of dep.steps) if (s.status === "running") s.status = "todo";
        throw new Error("the IDE is shutting down");
      }
      dep.status = "error";
      dep.error = e instanceof Error ? e.message : String(e);
      for (const s of dep.steps) if (s.status === "running") s.status = "error";
      throw e;
    } finally {
      this.busy.delete(id);
      this.save();
      this.changed();
    }
  }

  /** One real request of the deployment's kind; also learns the context and embedding size. */
  private async probeEndpoint(baseUrl: string, key: string, dep: StoredDeployment): Promise<{ ok: boolean; detail: string }> {
    const headers = { Authorization: `Bearer ${key}`, "Content-Type": "application/json" };
    try {
      const models = (await (await fetch(`${baseUrl}/models`, { headers })).json()) as { data?: Array<{ max_model_len?: number; meta?: { n_ctx_train?: number } }> };
      const reported = models.data?.[0]?.max_model_len;
      if (typeof reported === "number" && reported > 0) dep.contextLength = reported;
      const started = Date.now();
      if (dep.request.kind === "embed") {
        const res = await fetch(`${baseUrl}/embeddings`, { method: "POST", headers, body: JSON.stringify({ model: dep.servedName, input: ["dimension probe", "a second sentence"] }), signal: AbortSignal.timeout(120_000) });
        const body = await res.text();
        if (!res.ok) return { ok: false, detail: `HTTP ${res.status} ${body.slice(0, 300)}` };
        const j = JSON.parse(body) as { data?: Array<{ embedding: number[] }> };
        const dim = j.data?.[0]?.embedding?.length ?? 0;
        if (!dim || j.data?.length !== 2) return { ok: false, detail: "no embeddings in the reply" };
        dep.dim = dim;
        return { ok: true, detail: `${dim}-dim embeddings in ${Date.now() - started} ms` };
      }
      const res = await fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        headers,
        body: JSON.stringify({ model: dep.servedName, messages: [{ role: "user", content: "Reply with one short sentence: what are you?" }], max_tokens: 32, temperature: 0 }),
        signal: AbortSignal.timeout(300_000),
      });
      const body = await res.text();
      if (!res.ok) return { ok: false, detail: `HTTP ${res.status} ${body.slice(0, 300)}` };
      const j = JSON.parse(body) as { choices?: Array<{ message?: { content?: string; reasoning_content?: string } }> };
      const text = (j.choices?.[0]?.message?.content ?? j.choices?.[0]?.message?.reasoning_content ?? "").trim();
      if (!j.choices?.length) return { ok: false, detail: "no choices in the reply" };
      return { ok: true, detail: `replied in ${Date.now() - started} ms: ${JSON.stringify(text.slice(0, 120))}` };
    } catch (e) {
      return { ok: false, detail: (e as Error).message };
    }
  }

  /** The deployment becomes a provider + model in Settings → Models, its key in the secret store. */
  private register(dep: StoredDeployment, key: string): void {
    const s = this.d.settings();
    const providerId = `hub-${dep.id}`;
    const modelId = `${providerId}:${dep.servedName}`;
    const label = dep.request.label || `${dep.repo} · ${dep.servedName.slice(path.basename(dep.repo).length + 1) || dep.request.target}`;
    this.d.secrets.set(`provider:${providerId}`, key);
    const provider: ProviderConfig = { id: providerId, kind: "openai_compat", label: `Hub: ${label}`, baseUrl: dep.baseUrl!, hasKey: true };
    const model: ModelConfig = {
      id: modelId,
      providerId,
      name: dep.servedName,
      label,
      kind: dep.request.kind,
      contextLength: dep.contextLength,
      maxOutputTokens: dep.request.kind === "chat" && dep.contextLength ? Math.max(256, Math.floor(dep.contextLength / 4)) : null,
      dim: dep.dim,
    };
    this.d.updateSettings({
      providers: [...s.providers.filter((p) => p.id !== providerId), provider],
      models: [...s.models.filter((x) => x.id !== modelId), model],
    });
    dep.providerId = providerId;
    dep.settingsModelId = modelId;
  }

  stop(id: string): HubDeploymentView {
    const dep = this.data.deployments.find((x) => x.id === id);
    if (!dep) throw new Error(`no deployment ${id}`);
    const p = this.procs.get(id);
    dep.status = "stopped";
    if (p) {
      this.procs.delete(id);
      p.kill();
    }
    this.save();
    this.changed();
    return dep;
  }

  removeDeployment(id: string): void {
    const dep = this.data.deployments.find((x) => x.id === id);
    if (!dep) return;
    this.stop(id);
    const s = this.d.settings();
    if (dep.providerId) this.d.updateSettings({ providers: s.providers.filter((p) => p.id !== dep.providerId), models: s.models.filter((m) => m.providerId !== dep.providerId) });
    this.d.secrets.set(`provider:hub-${id}`, "");
    rmSync(this.logFile(id), { force: true });
    rmSync(path.join(this.deployDir, id), { recursive: true, force: true });
    this.data.deployments = this.data.deployments.filter((x) => x.id !== id);
    this.save();
    this.changed();
  }

  async test(id: string): Promise<{ ok: boolean; detail: string }> {
    const dep = this.data.deployments.find((x) => x.id === id);
    if (!dep) throw new Error(`no deployment ${id}`);
    const key = this.d.secrets.get(`provider:hub-${id}`);
    if (!dep.baseUrl || !this.procs.has(id) || !key) return { ok: false, detail: "not running" };
    return this.probeEndpoint(dep.baseUrl, key, dep);
  }

  readLog(id: string): string {
    const f = this.logFile(id);
    if (!existsSync(f)) return "";
    const t = readFileSync(f, "utf8");
    return t.length > 200_000 ? t.slice(-200_000) : t;
  }

  // ---- tools & token --------------------------------------------------------------------------

  setToken(token: string): boolean {
    this.d.secrets.set(HF_TOKEN, token.trim());
    this.changed();
    return this.hf.hasToken();
  }

  async toolsAction(
    action: "probe" | "setupPython" | "installPackages" | "installLlamaCpp" | "downloadLlamaCpp",
    packages: string[],
    backend: "cpu" | "cuda" | "gpu" | "auto",
    env: HubPythonEnv = "main",
  ): Promise<HubStateView["tools"]> {
    if (action === "probe") await Promise.all([this.tools.probe(true), this.tools.llamaCpp(true)]).then(() => this.tools.probeOnnx(true));
    else if (action === "setupPython") await this.tools.setupPython(env);
    else if (action === "installPackages") await this.tools.pipInstall(packages, env);
    else if (action === "downloadLlamaCpp") await this.tools.downloadLlamaCpp(backend === "cpu" ? "cpu" : backend === "auto" ? "auto" : "gpu");
    else await this.tools.installLlamaCpp(backend === "cuda" ? "cuda" : "cpu");
    return this.view().tools;
  }
}
