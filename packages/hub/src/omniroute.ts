import { randomBytes, randomInt } from "node:crypto";
import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { createWriteStream, existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  CHARS_PER_TOKEN,
  foldSystemPrompt,
  OMNIROUTE_AUTO_MODEL_ID,
  type ModelConfig,
  type OmniRouteConnectionView,
  type OmniRouteModelView,
  type OmniRouteProviderView,
  type OmniRouteView,
  type ProviderConfig,
  type Settings,
} from "@rlm/contracts";
import { lastLines, pipeLines } from "./proc";

export const OMNIROUTE_PROVIDER_ID = "omniroute";
const KEY_SECRET = `provider:${OMNIROUTE_PROVIDER_ID}`;
const DASHBOARD_SECRET = "omniroute:dashboard";
/** Used only until an installed package declares its own `engines.node`. */
const FALLBACK_ENGINES = ">=22.22.2 <23 || >=24.0.0 <27";
const LOG_TAIL = 6000;

export interface OmniRouteDeps {
  /** userData/omniroute */
  dir: string;
  settings: () => Settings;
  updateSettings: (patch: Partial<Settings>) => void;
  /** True in every copy of the IDE but the first on this profile: the first one runs the gateway. */
  follower?: () => boolean;
  secrets: {
    get(name: string): string | null;
    set(name: string, value: string): void;
  };
  emit: (view: OmniRouteView) => void;
  log: (msg: string) => void;
  /** The inference key changed without a settings change; the agent must reload provider keys. */
  keysChanged: () => void;
  /** Extra places to look for a Node.js binary, checked first. */
  nodeCandidates?: () => string[];
  /** The agent's system prompt template (`rlm/system.md`), for testing whether a model follows its cell protocol. */
  agentPrompt?: () => string | null;
}

// ---------- pure helpers (unit tested) ----------

/** "Prompt too long (max 6000 characters)" and similar. */
const MAX_CHARS_RE = /max(?:imum)?\s*(?:of\s*)?(\d{3,})\s*char/i;
/** Input-size refusals as providers word them (DuckDuckGo's arrives as a 429 "ERR_INPUT_LIMIT"). */
const INPUT_LIMIT_RE = /ERR_INPUT_LIMIT|too long|context.length|maximum context|too many tokens|input limit/i;

/** Failures that mean "busy", not "too big". */
const BUSY_RE = /\b429\b|rate.?limit|too many requests|quota|timed? ?out|timeout|aborted/i;

type Holds = { ok: true } | { ok: false; maxChars: number | null; sizeLimit: boolean; error: string };

/** Tasks only a REPL cell can do, so a model that follows the agent protocol must write one. */
const AGENT_TASKS = ["Run a REPL cell that prints the result of 17 * 23.", "How many files are in the project root folder?"];
const AGENT_TEST_TOOLS = [
  "### fs",
  "- `fs.list_dir(path: str = \".\")` -> list[str]: List a directory.",
  "- `fs.read_file(path: str, start_line: int = 1, end_line: int = None)` -> str: Read a text file.",
  "",
  "### shell",
  "- `shell.run(command: str, timeout_sec: int = 120, background: bool = False)` -> dict(exit_code, stdout, stderr): Run a shell command in the project root.",
].join("\n");

/** The agent's system prompt with a short tool list, as a small-window model would get it. */
export function agentTestPrompt(template: string): string {
  return template.replace("{{tool_surface}}", AGENT_TEST_TOOLS).replace("{{can_spawn}}", "no").replace("{{max_depth}}", "0");
}

function testWord(): string {
  return `${randomBytes(3).toString("hex").toUpperCase()}-${randomInt(10, 99)}`;
}

/** Neutral numbered notes to pad a window test: nothing repeats verbatim and nothing reads like an instruction. */
export function windowFiller(chars: number): string {
  let s = "";
  for (let i = 1; s.length < chars; i++) s += `Note ${i}: module ${i % 37} keeps its configuration in config/${i % 11}.json and its tests in test/${i % 13}.spec.ts.\n`;
  return s.slice(0, chars);
}

/** The CLI prints env-loading banners before its JSON; the JSON starts on the first line beginning with { or [. */
export function parseCliJson<T = unknown>(stdout: string): T {
  const lines = stdout.split(/\r?\n/);
  const i = lines.findIndex((l) => l.startsWith("{") || l.startsWith("["));
  if (i < 0) throw new Error(`no JSON in CLI output: ${lastLines(stdout, 4)}`);
  return JSON.parse(lines.slice(i).join("\n")) as T;
}

function semver(v: string): [number, number, number] | null {
  const m = /(\d+)\.(\d+)\.(\d+)/.exec(v);
  return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : null;
}

function cmp(a: [number, number, number], b: [number, number, number]): number {
  return a[0] - b[0] || a[1] - b[1] || a[2] - b[2];
}

/** Evaluates npm-style ranges made of `>=x <y` comparator sets joined by `||`. */
export function satisfiesEngines(version: string, range: string): boolean {
  const v = semver(version);
  if (!v) return false;
  return range.split("||").some((set) =>
    set
      .trim()
      .split(/\s+/)
      .filter(Boolean)
      .every((c) => {
        const m = /^(>=|<=|>|<|=)?v?(\d+(?:\.\d+){0,2})$/.exec(c);
        if (!m) return false;
        const parts = m[2]!.split(".").map(Number);
        const t: [number, number, number] = [parts[0] ?? 0, parts[1] ?? 0, parts[2] ?? 0];
        const d = cmp(v, t);
        switch (m[1]) {
          case ">=":
            return d >= 0;
          case ">":
            return d > 0;
          case "<=":
            return d <= 0;
          case "<":
            return d < 0;
          default:
            return d === 0;
        }
      }),
  );
}

export interface GatewayModel {
  id: string;
  owned_by?: string;
  name?: string;
  context_length?: number;
  max_output_tokens?: number;
  output_modalities?: string[];
  capabilities?: { tool_calling?: boolean };
}

/** Text chat models of free providers, best first within each provider (tool calling, then window size). */
export function freeChatCandidates(models: GatewayModel[], freeProviders: Set<string>, perProvider: number): GatewayModel[] {
  const usable = models.filter((m) => {
    if (!m.owned_by || m.owned_by === "combo" || !freeProviders.has(m.owned_by)) return false;
    if (m.output_modalities) return m.output_modalities.includes("text");
    return !!m.context_length;
  });
  const byProvider = new Map<string, GatewayModel[]>();
  for (const m of usable) byProvider.set(m.owned_by!, [...(byProvider.get(m.owned_by!) ?? []), m]);
  const out: GatewayModel[] = [];
  for (const list of byProvider.values()) out.push(...list.sort(rankModels).slice(0, perProvider));
  return out;
}

/** Rough strength from the model's name alone (gateways report no quality signal): flagship models rarely state a
 * parameter count, while small ones do ("20b", "8B") or say so ("mini", "nano"). 2 = large or unknown, 0 = small. */
export function capabilityTier(id: string): number {
  const name = id.toLowerCase();
  const sizes = [...name.matchAll(/(?:^|[^a-z0-9.])(\d+(?:\.\d+)?)b(?![a-z0-9])/g)].map((m) => Number(m[1]));
  const size = sizes.length ? Math.max(...sizes) : null;
  let tier = size === null || size >= 70 ? 2 : size >= 30 ? 1 : 0;
  if (/(?:^|[^a-z])(mini|nano|tiny|lite|small|micro)(?:[^a-z]|$)/.test(name)) tier -= 1;
  return Math.max(0, tier);
}

function rankModels(a: GatewayModel, b: GatewayModel): number {
  return (
    capabilityTier(b.id) - capabilityTier(a.id) ||
    Number(!!b.capabilities?.tool_calling) - Number(!!a.capabilities?.tool_calling) ||
    (b.context_length ?? 0) - (a.context_length ?? 0) ||
    a.id.localeCompare(b.id)
  );
}

/** Orders working models so consecutive entries come from different providers: one provider going down
 * (they fail together) then costs a single step of the chain. */
export function interleaveByProvider<T extends { provider: string }>(ranked: T[], max: number): T[] {
  const queues = new Map<string, T[]>();
  for (const m of ranked) queues.set(m.provider, [...(queues.get(m.provider) ?? []), m]);
  const out: T[] = [];
  while (out.length < max && [...queues.values()].some((q) => q.length)) for (const q of queues.values()) if (q.length && out.length < max) out.push(q.shift()!);
  return out;
}

// ---------- service ----------

interface Persisted {
  models: OmniRouteModelView[];
}

interface CliResult {
  code: number | null;
  stdout: string;
  stderr: string;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export class OmniRouteService {
  private child: ChildProcess | null = null;
  private stopping = false;
  private closing = false;
  private view: OmniRouteView;
  private nodeBin: string | null = null;
  private emitTimer: NodeJS.Timeout | null = null;
  private freeApiKeyProviders = new Set<string>();

  constructor(private readonly d: OmniRouteDeps) {
    const port = d.settings().omniroute.port;
    this.view = {
      phase: "not_installed",
      version: null,
      node: { path: null, version: null, ok: false, detail: "not checked" },
      baseUrl: `http://127.0.0.1:${port}/v1`,
      dashboardUrl: `http://127.0.0.1:${port}/`,
      busy: null,
      error: null,
      logTail: "",
      providers: [],
      connections: [],
      models: this.load().models,
      comboName: d.settings().omniroute.comboName,
    };
  }

  private get runtimeDir(): string {
    return path.join(this.d.dir, "runtime");
  }
  private get dataDir(): string {
    return path.join(this.d.dir, "data");
  }
  private get browsersDir(): string {
    return path.join(this.d.dir, "browsers");
  }
  private get pkgDir(): string {
    return path.join(this.runtimeDir, "node_modules", "omniroute");
  }
  private get bin(): string {
    const pkg = this.pkgJson();
    const rel = typeof pkg.bin === "string" ? pkg.bin : (pkg.bin?.omniroute ?? path.join("bin", "omniroute.mjs"));
    return path.join(this.pkgDir, rel);
  }
  private pkgJson(): {
    version?: string;
    bin?: string | Record<string, string>;
    engines?: { node?: string };
  } {
    try {
      return JSON.parse(readFileSync(path.join(this.pkgDir, "package.json"), "utf8")) as ReturnType<OmniRouteService["pkgJson"]>;
    } catch {
      return {};
    }
  }
  private get logFile(): string {
    return path.join(this.d.dir, "omniroute.log");
  }
  private port(): number {
    return this.d.settings().omniroute.port;
  }

  private load(): Persisted {
    try {
      return JSON.parse(readFileSync(path.join(this.d.dir, "state.json"), "utf8")) as Persisted;
    } catch {
      return { models: [] };
    }
  }

  private save(): void {
    mkdirSync(this.d.dir, { recursive: true });
    writeFileSync(path.join(this.d.dir, "state.json"), JSON.stringify({ models: this.view.models } satisfies Persisted, null, 2));
  }

  private changed(): void {
    if (this.emitTimer) return;
    this.emitTimer = setTimeout(() => {
      this.emitTimer = null;
      this.d.emit(this.state());
    }, 100);
  }

  private set(patch: Partial<OmniRouteView>): void {
    this.view = { ...this.view, ...patch };
    this.changed();
  }

  private appendLog(line: string): void {
    this.view.logTail = (this.view.logTail + line + "\n").slice(-LOG_TAIL);
    this.changed();
  }

  state(): OmniRouteView {
    const port = this.port();
    return {
      ...this.view,
      baseUrl: `http://127.0.0.1:${port}/v1`,
      dashboardUrl: `http://127.0.0.1:${port}/`,
      comboName: this.d.settings().omniroute.comboName,
    };
  }

  installedVersion(): string | null {
    return this.pkgJson().version ?? null;
  }

  private engines(): string {
    return this.pkgJson().engines?.node ?? FALLBACK_ENGINES;
  }

  /** Finds a Node.js the gateway supports: explicit candidates, PATH, then version managers. */
  detectNode(): OmniRouteView["node"] {
    const exe = process.platform === "win32" ? "node.exe" : "node";
    const home = os.homedir();
    const list: string[] = [...(this.d.nodeCandidates?.() ?? [])];
    if (process.env.RLM_NODE) list.push(process.env.RLM_NODE);
    for (const dir of (process.env.PATH ?? "").split(path.delimiter)) if (dir) list.push(path.join(dir, exe));
    for (const root of [path.join(home, ".nvm", "versions", "node"), path.join(home, ".local", "share", "fnm", "node-versions")]) {
      try {
        for (const v of readdirSync(root).sort().reverse()) list.push(path.join(root, v, "bin", exe), path.join(root, v, "installation", "bin", exe));
      } catch {
        /* no version manager */
      }
    }
    list.push(path.join(home, ".volta", "bin", exe), path.join(home, ".local", "bin", exe), "/usr/local/bin/node", "/usr/bin/node", "/opt/homebrew/bin/node");
    if (process.platform === "win32") list.push(path.join(process.env.ProgramFiles ?? "C:\\Program Files", "nodejs", exe));
    const range = this.engines();
    const seen: string[] = [];
    for (const p of [...new Set(list)]) {
      if (!existsSync(p)) continue;
      const r = spawnSync(p, ["-v"], {
        encoding: "utf8",
        timeout: 5000,
        env: this.baseEnv(),
      });
      const version = r.stdout?.trim();
      if (r.status !== 0 || !version) continue;
      seen.push(`${version} at ${p}`);
      if (satisfiesEngines(version, range)) {
        this.nodeBin = p;
        return { path: p, version, ok: true, detail: `Node.js ${version}` };
      }
    }
    this.nodeBin = null;
    return {
      path: null,
      version: null,
      ok: false,
      detail: seen.length
        ? `OmniRoute needs Node.js ${range}; found ${seen.join(", ")}. Install a supported Node.js and press Retry.`
        : `OmniRoute needs Node.js ${range}, which was not found. Install it from nodejs.org and press Retry.`,
    };
  }

  private baseEnv(): Record<string, string> {
    const env: Record<string, string> = {};
    for (const [k, v] of Object.entries(process.env)) if (v !== undefined && !k.startsWith("ELECTRON_") && k !== "NODE_OPTIONS") env[k] = v;
    return env;
  }

  /** Environment shared by the CLI and the server: our own data dir, no colour, the browser cache inside the app folder. */
  private env(): Record<string, string> {
    const env = this.baseEnv();
    if (this.nodeBin) env.PATH = `${path.dirname(this.nodeBin)}${path.delimiter}${env.PATH ?? ""}`;
    return {
      ...env,
      DATA_DIR: this.dataDir,
      OMNIROUTE_CLI_SKIP_DEFAULT_DATA_ENV: "1",
      PLAYWRIGHT_BROWSERS_PATH: this.browsersDir,
      NO_COLOR: "1",
      FORCE_COLOR: "0",
      OMNIROUTE_BASE_URL: `http://127.0.0.1:${this.port()}`,
    };
  }

  private npmCommand(): { cmd: string; pre: string[] } {
    const dir = path.dirname(this.nodeBin!);
    for (const cli of [path.join(dir, "..", "lib", "node_modules", "npm", "bin", "npm-cli.js"), path.join(dir, "node_modules", "npm", "bin", "npm-cli.js")])
      if (existsSync(cli)) return { cmd: this.nodeBin!, pre: [cli] };
    return { cmd: process.platform === "win32" ? "npm.cmd" : "npm", pre: [] };
  }

  private exec(cmd: string, args: string[], o: { cwd?: string; stdin?: string; timeoutMs?: number; log?: boolean } = {}): Promise<CliResult> {
    return new Promise((resolve) => {
      let stdout = "";
      let stderr = "";
      let child: ChildProcess;
      try {
        child = spawn(cmd, args, {
          cwd: o.cwd,
          env: this.env(),
          stdio: ["pipe", "pipe", "pipe"],
          shell: process.platform === "win32" && cmd.endsWith(".cmd"),
        });
      } catch (e) {
        resolve({ code: -1, stdout: "", stderr: (e as Error).message });
        return;
      }
      const timer = o.timeoutMs ? setTimeout(() => child.kill(), o.timeoutMs) : null;
      child.stdout?.setEncoding("utf8");
      child.stderr?.setEncoding("utf8");
      child.stdout?.on("data", (s: string) => {
        stdout += s;
        if (o.log) for (const l of s.split(/\r?\n|\r/)) if (l.trim()) this.appendLog(l);
      });
      child.stderr?.on("data", (s: string) => {
        stderr += s;
        if (o.log) for (const l of s.split(/\r?\n|\r/)) if (l.trim()) this.appendLog(l);
      });
      child.on("error", (e) => (stderr += `\n${e.message}`));
      child.on("close", (code) => {
        if (timer) clearTimeout(timer);
        resolve({ code, stdout, stderr });
      });
      child.stdin?.end(o.stdin ?? "");
    });
  }

  /** Runs the bundled `omniroute` CLI against our server; it authenticates with its per-install local token. */
  private async cli<T = unknown>(args: string[], o: { stdin?: string; json?: boolean; timeoutMs?: number } = {}): Promise<T> {
    if (!this.nodeBin) throw new Error("Node.js for OmniRoute was not found");
    const r = await this.exec(this.nodeBin, [this.bin, ...args], {
      stdin: o.stdin,
      timeoutMs: o.timeoutMs ?? 60_000,
    });
    const out = r.stdout + (r.stderr ? `\n${r.stderr}` : "");
    if (r.code !== 0) throw new Error(`omniroute ${args.slice(0, 2).join(" ")} failed: ${lastLines(out.replace(/.*Loaded env from.*\n?/g, ""), 4)}`);
    return (o.json === false ? (r.stdout as unknown) : parseCliJson<T>(r.stdout)) as T;
  }

  async boot(): Promise<void> {
    this.set({ node: this.detectNode(), version: this.installedVersion() });
    if (!this.view.version) return this.set({ phase: "not_installed" });
    this.set({ phase: "stopped" });
    const s = this.d.settings();
    const registered = s.providers.some((p) => p.id === OMNIROUTE_PROVIDER_ID && p.managed === "omniroute");
    if (!s.omniroute.autoStart || !registered || !this.view.node.ok) return;
    if (this.d.follower?.()) {
      // The first copy of the IDE starts the gateway; wait for it instead of racing it for the port.
      for (let i = 0; i < 60 && !this.closing; i++) {
        if (await this.adopt()) return;
        await sleep(2000);
      }
      return;
    }
    await this.start().catch((e: unknown) => this.d.log(`auto-start failed: ${String(e)}`));
  }

  /** Uses a gateway another copy of the IDE runs on our port with our key; it stays theirs to stop. */
  private async adopt(): Promise<boolean> {
    const key = this.d.secrets.get(KEY_SECRET);
    if (!key || !(await this.health())) return false;
    const ok = await fetch(`http://127.0.0.1:${this.port()}/v1/models`, {
      headers: { Authorization: `Bearer ${key}` },
      signal: AbortSignal.timeout(10_000),
    }).then(
      (r) => r.ok,
      () => false,
    );
    if (!ok) return false;
    this.register();
    this.set({ phase: "running", busy: null, error: null, version: this.installedVersion() });
    void this.refreshCatalog().catch((e: unknown) => this.d.log(`catalog: ${String(e)}`));
    return true;
  }

  async install(): Promise<OmniRouteView> {
    if (this.view.busy) throw new Error(`busy: ${this.view.busy}`);
    const node = this.detectNode();
    this.set({ node });
    if (!node.ok) throw new Error(node.detail);
    this.set({
      phase: "installing",
      busy: "Installing OmniRoute (this downloads about 3 GB once)",
      error: null,
      logTail: "",
    });
    try {
      mkdirSync(this.runtimeDir, { recursive: true });
      const pj = path.join(this.runtimeDir, "package.json");
      if (!existsSync(pj)) writeFileSync(pj, JSON.stringify({ name: "rlm-omniroute-runtime", private: true }, null, 2));
      const npm = this.npmCommand();
      const spec = this.d.settings().omniroute.packageSpec;
      const r = await this.exec(npm.cmd, [...npm.pre, "install", "--no-audit", "--no-fund", "--loglevel=error", spec], { cwd: this.runtimeDir, log: true });
      if (r.code !== 0 || !this.installedVersion()) throw new Error(`npm install ${spec} failed:\n${lastLines(r.stderr || r.stdout, 12)}`);
      this.set({
        node: this.detectNode(),
        busy: "Installing the headless browser some keyless providers need",
      });
      const pw = path.join(this.runtimeDir, "node_modules", "playwright", "cli.js");
      if (existsSync(pw)) {
        const b = await this.exec(this.nodeBin!, [pw, "install", "chromium"], {
          cwd: this.runtimeDir,
          log: true,
        });
        if (b.code !== 0) this.appendLog(`browser install failed; browser-based free providers will not work:\n${lastLines(b.stderr || b.stdout, 6)}`);
      }
      this.set({
        phase: "stopped",
        version: this.installedVersion(),
        busy: null,
      });
    } catch (e) {
      this.set({
        phase: this.installedVersion() ? "stopped" : "not_installed",
        busy: null,
        error: e instanceof Error ? e.message : String(e),
      });
      throw e;
    }
    // Ready to pick right away: the gateway runs and the "OmniRoute" model gets its tested chain in the background.
    await this.start().catch((e: unknown) => this.d.log(`start after install failed: ${String(e)}`));
    if (this.view.phase === "running" && !this.d.settings().omniroute.chain.length) {
      void this.setup([], false).catch((e: unknown) => this.d.log(`free model setup after install failed: ${String(e)}`));
    }
    return this.state();
  }

  private async health(timeoutMs = 3000): Promise<boolean> {
    try {
      const r = await fetch(`http://127.0.0.1:${this.port()}/api/health`, {
        signal: AbortSignal.timeout(timeoutMs),
      });
      return r.ok;
    } catch {
      return false;
    }
  }

  async start(): Promise<OmniRouteView> {
    if (this.child && this.view.phase === "running") return this.state();
    if (!this.installedVersion()) throw new Error("OmniRoute is not installed");
    if (!this.nodeBin) this.set({ node: this.detectNode() });
    if (!this.nodeBin) throw new Error(this.view.node.detail);
    if (this.d.follower?.() && (await this.adopt())) return this.state();
    this.set({ phase: "starting", busy: "Starting the gateway", error: null });
    try {
      // A server left over from a crash holds our data dir; the CLI stops it by its pid file.
      await this.cli(["stop"], { json: false, timeoutMs: 20_000 }).catch(() => undefined);
      if (await this.health())
        throw new Error(
          `port ${this.port()} is used by another program (maybe your own OmniRoute). Change the port in Settings → Models → Free models, or add your server as a provider with its API key.`,
        );
      let password = this.d.secrets.get(DASHBOARD_SECRET);
      if (!password) {
        password = randomBytes(18).toString("base64url");
        this.d.secrets.set(DASHBOARD_SECRET, password);
      }
      mkdirSync(this.dataDir, { recursive: true, mode: 0o700 });
      const log = createWriteStream(this.logFile, { flags: "a" });
      const args = [this.bin, "serve", "--port", String(this.port()), "--no-open", "--no-tray", "--log"];
      log.write(`\n$ node ${args.join(" ")}\n`);
      this.stopping = false;
      const child = spawn(this.nodeBin, args, {
        env: {
          ...this.env(),
          PORT: String(this.port()),
          OMNIROUTE_SERVER_HOST: "127.0.0.1",
          API_HOST: "127.0.0.1",
          REQUIRE_API_KEY: "true",
          INITIAL_PASSWORD: password,
        },
        stdio: ["ignore", "pipe", "pipe"],
      });
      this.child = child;
      let exited: string | null = null;
      pipeLines(child, (l) => this.appendLog(l), log);
      child.on("error", (e) => (exited = e.message));
      child.on("exit", (code, sig) => {
        log.end();
        exited = `gateway exited (${sig ?? code})`;
        if (this.child === child) this.child = null;
        if (!this.stopping && !this.closing)
          this.set({
            phase: "error",
            error: `${exited}:\n${lastLines(this.view.logTail, 10)}`,
          });
      });
      const deadline = Date.now() + 120_000;
      while (!(await this.health()) && !exited && Date.now() < deadline && !this.closing) await sleep(1000);
      if (exited || !(await this.health())) throw new Error(`${exited ?? "the gateway did not become ready in 120 s"}:\n${lastLines(this.view.logTail, 10)}`);
      const open = await fetch(`http://127.0.0.1:${this.port()}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: "probe", messages: [] }),
        signal: AbortSignal.timeout(10_000),
      }).then(
        (r) => r.status,
        () => 0,
      );
      if (open !== 401 && open !== 403) throw new Error(`the gateway answered without an API key (HTTP ${open}); refusing to run it unauthenticated`);
      await this.ensureKey();
      this.register();
      this.set({
        phase: "running",
        busy: null,
        version: this.installedVersion(),
      });
      void this.refreshCatalog().catch((e: unknown) => this.d.log(`catalog: ${String(e)}`));
      return this.state();
    } catch (e) {
      await this.stop().catch(() => undefined);
      this.set({
        phase: "error",
        busy: null,
        error: e instanceof Error ? e.message : String(e),
      });
      throw e;
    }
  }

  /** The IDE's inference key lives in the secret store as the provider key, so the model adapter sends it. */
  private async ensureKey(): Promise<string> {
    const existing = this.d.secrets.get(KEY_SECRET);
    if (existing) {
      const r = await fetch(`http://127.0.0.1:${this.port()}/v1/models`, {
        headers: { Authorization: `Bearer ${existing}` },
        signal: AbortSignal.timeout(10_000),
      }).then(
        (x) => x.status,
        () => 0,
      );
      if (r === 200) return existing;
    }
    const created = await this.cli<{ key?: string }>(["api", "api-keys", "post-api-keys", "--body", JSON.stringify({ name: "RLM IDE" }), "--output", "json"]);
    if (!created.key) throw new Error("the gateway did not return an API key");
    this.d.secrets.set(KEY_SECRET, created.key);
    this.d.keysChanged();
    return created.key;
  }

  private key(): string {
    const k = this.d.secrets.get(KEY_SECRET);
    if (!k) throw new Error("the OmniRoute API key is missing; restart the gateway");
    return k;
  }

  private register(): void {
    const s = this.d.settings();
    const provider: ProviderConfig = {
      id: OMNIROUTE_PROVIDER_ID,
      kind: "openai_compat",
      label: "OmniRoute (free models)",
      baseUrl: `http://127.0.0.1:${this.port()}/v1`,
      hasKey: true,
      managed: "omniroute",
    };
    const cur = s.providers.find((p) => p.id === OMNIROUTE_PROVIDER_ID);
    const patch: Partial<Settings> = {};
    if (!(cur && cur.baseUrl === provider.baseUrl && cur.hasKey && cur.managed === "omniroute")) {
      patch.providers = [...s.providers.filter((p) => p.id !== OMNIROUTE_PROVIDER_ID), provider];
    }
    // Chains set up before the "OmniRoute" model existed live only in the gateway state.
    const chain = s.omniroute.chain.length
      ? s.omniroute.chain
      : this.view.models
          .filter((m) => m.chainIndex !== null && s.models.some((x) => x.id === memberId(m.id)))
          .sort((a, b) => a.chainIndex! - b.chainIndex!)
          .map((m) => memberId(m.id));
    if (chain.join("|") !== s.omniroute.chain.join("|")) patch.omniroute = { ...s.omniroute, chain };
    const auto = autoModel(s, chain);
    const curAuto = s.models.find((m) => m.id === OMNIROUTE_AUTO_MODEL_ID);
    if (!curAuto || JSON.stringify(curAuto) !== JSON.stringify(auto)) patch.models = [...s.models.filter((m) => m.id !== OMNIROUTE_AUTO_MODEL_ID), auto];
    if (Object.keys(patch).length) this.d.updateSettings(patch);
  }

  async stop(): Promise<OmniRouteView> {
    this.stopping = true;
    const child = this.child;
    // A gateway adopted from another copy of the IDE keeps serving it; this copy only lets go of it.
    const adopted = !child && !!this.d.follower?.() && this.view.phase === "running";
    if (!adopted && this.nodeBin && this.installedVersion()) await this.cli(["stop"], { json: false, timeoutMs: 20_000 }).catch(() => undefined);
    if (child && child.exitCode === null) {
      child.kill();
      await Promise.race([new Promise((r) => child.once("exit", r)), sleep(5000)]);
    }
    this.child = null;
    if (!this.closing)
      this.set({
        phase: this.installedVersion() ? "stopped" : "not_installed",
        busy: null,
      });
    return this.state();
  }

  /** Synchronous best effort for app quit. */
  close(): void {
    this.closing = true;
    if (!this.child) return;
    if (this.nodeBin)
      spawnSync(this.nodeBin, [this.bin, "stop"], {
        env: this.env(),
        timeout: 15_000,
      });
    if (this.child.exitCode === null) this.child.kill();
    this.child = null;
  }

  private requireRunning(): void {
    if (this.view.phase !== "running") throw new Error("start the OmniRoute gateway first");
  }

  async refreshCatalog(): Promise<OmniRouteView> {
    this.requireRunning();
    type Avail = {
      providers: Array<{
        id: string;
        name: string;
        website?: string;
        hasFree?: boolean;
        deprecated?: boolean;
      }>;
    };
    const [apiKey, noauth, conns] = await Promise.all([
      this.cli<Avail>(["providers", "available", "--category", "api-key", "--json"]),
      this.cli<Avail>(["providers", "available", "--category", "noauth", "--json"]),
      this.cli<{
        providers: Array<{
          id: string;
          provider: string;
          name: string;
          isActive: boolean;
          testStatus: string;
          lastError: string | null;
        }>;
      }>(["providers", "list", "--json"]),
    ]);
    const connections: OmniRouteConnectionView[] = conns.providers.map((c) => ({
      id: c.id,
      provider: c.provider,
      name: c.name,
      active: c.isActive,
      status: c.testStatus,
      lastError: c.lastError,
    }));
    const connected = new Set(connections.filter((c) => c.active).map((c) => c.provider));
    this.freeApiKeyProviders = new Set(apiKey.providers.filter((p) => p.hasFree && !p.deprecated).map((p) => p.id));
    const providers: OmniRouteProviderView[] = [
      ...noauth.providers
        .filter((p) => p.hasFree && !p.deprecated)
        .map((p) => ({
          id: p.id,
          name: p.name,
          website: p.website ?? null,
          keyless: true,
          connected: true,
        })),
      ...apiKey.providers
        .filter((p) => p.hasFree && !p.deprecated)
        .map((p) => ({
          id: p.id,
          name: p.name,
          website: p.website ?? null,
          keyless: false,
          connected: connected.has(p.id),
        })),
    ];
    this.set({ providers, connections });
    return this.state();
  }

  async connect(provider: string, key: string): Promise<OmniRouteView> {
    this.requireRunning();
    const args = ["providers", "add", provider, "--yes", "--json"];
    await this.cli(key ? [...args, "--credential-stdin"] : [...args, "--no-credential"], { stdin: key ? `${key}\n` : "" });
    return this.refreshCatalog();
  }

  async disconnect(connectionId: string): Promise<OmniRouteView> {
    this.requireRunning();
    await this.cli(["providers", "remove", connectionId, "--yes"], {
      json: false,
    });
    return this.refreshCatalog();
  }

  private async gatewayModels(): Promise<GatewayModel[]> {
    const r = await fetch(`http://127.0.0.1:${this.port()}/v1/models`, {
      headers: { Authorization: `Bearer ${this.key()}` },
      signal: AbortSignal.timeout(20_000),
    });
    if (!r.ok) throw new Error(`model list failed (${r.status})`);
    return ((await r.json()) as { data: GatewayModel[] }).data;
  }

  private async freeProviders(): Promise<Set<string>> {
    if (!this.view.providers.length) await this.refreshCatalog();
    return new Set(this.view.providers.filter((p) => p.connected).map((p) => p.id));
  }

  private toView(m: GatewayModel, prev?: OmniRouteModelView): OmniRouteModelView {
    return {
      id: m.id,
      provider: m.owned_by ?? m.id.split("/")[0]!,
      name: m.name ?? m.id,
      contextLength: m.context_length ?? null,
      maxOutputTokens: m.max_output_tokens ?? null,
      toolCalling: !!m.capabilities?.tool_calling,
      ok: prev?.ok ?? null,
      latencyMs: prev?.latencyMs ?? null,
      error: prev?.error ?? null,
      systemAsUser: prev?.systemAsUser ?? null,
      inputLimitChars: prev?.inputLimitChars ?? null,
      agentReady: prev?.agentReady ?? null,
      chainIndex: prev?.chainIndex ?? null,
    };
  }

  private async ask(id: string, messages: Array<{ role: "system" | "user" | "assistant"; content: string }>): Promise<{ text: string; error: string | null }> {
    try {
      const r = await fetch(`http://127.0.0.1:${this.port()}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${this.key()}`,
        },
        body: JSON.stringify({ model: id, messages, max_tokens: 400 }),
        signal: AbortSignal.timeout(this.d.settings().omniroute.probeTimeoutSec * 1000),
      });
      const body = (await r.json().catch(() => ({}))) as {
        choices?: Array<{ message?: { content?: string } }>;
        error?: { message?: string };
      };
      if (!r.ok)
        return {
          text: "",
          error: (body.error?.message ?? `HTTP ${r.status}`).slice(0, 200),
        };
      const text = body.choices?.[0]?.message?.content ?? "";
      return { text, error: text.trim() ? null : "empty reply" };
    } catch (e) {
      return { text: "", error: e instanceof Error ? e.message : String(e) };
    }
  }

  /** The agent's protocol, memory and task board travel in the system prompt, and some free providers silently drop
   * it. A random word that only the system prompt carries shows whether it arrives, natively or folded into the
   * user message; a model that loses it both ways would forget everything, so it fails the test. Then the real input
   * limit is measured, because web-playground providers accept far less than the gateway advertises. */
  private async probeOne(m: GatewayModel): Promise<Pick<OmniRouteModelView, "ok" | "latencyMs" | "error" | "systemAsUser" | "inputLimitChars" | "agentReady">> {
    const t0 = Date.now();
    const word = testWord();
    const messages = [
      { role: "system" as const, content: `The secret word is ${word}. When asked for the secret word, reply with only that word.` },
      { role: "user" as const, content: "What is the secret word? Reply with only the word." },
    ];
    const failed = (error: string) => ({ ok: false, latencyMs: Date.now() - t0, error, systemAsUser: null, inputLimitChars: null, agentReady: null });
    const native = await this.ask(m.id, messages);
    if (native.error) return failed(native.error);
    let systemAsUser = false;
    if (!native.text.includes(word)) {
      const folded = await this.ask(m.id, foldSystemPrompt(messages));
      if (folded.error || !folded.text.includes(word)) return failed(`ignores the agent's instructions, so it would forget its memory and tools (said: ${(folded.text || native.text).trim().slice(0, 80)})`);
      systemAsUser = true;
    }
    const latencyMs = Date.now() - t0;
    const inputLimitChars = await this.measureInput(m, systemAsUser);
    const min = this.d.settings().omniroute.minAgentInputChars;
    if (inputLimitChars !== null && inputLimitChars < min)
      return { ok: false, latencyMs, error: `accepts only about ${inputLimitChars} characters of input; the agent's instructions and work need at least ${min}`, systemAsUser, inputLimitChars, agentReady: null };
    return { ok: true, latencyMs, error: null, systemAsUser, inputLimitChars, agentReady: await this.followsAgentProtocol(m.id, systemAsUser) };
  }

  /** Given the agent's real instructions, does the model do work by writing `repl` cells, every time? Web-chat models
   * often answer from memory or say they cannot see the project instead. Null when there is no prompt to test with. */
  private async followsAgentProtocol(id: string, systemAsUser: boolean): Promise<boolean | null> {
    const template = this.d.agentPrompt?.();
    if (!template) return null;
    const system = agentTestPrompt(template);
    for (const task of AGENT_TASKS) {
      const messages = [
        { role: "system" as const, content: system },
        { role: "user" as const, content: task },
      ];
      const r = await this.ask(id, systemAsUser ? foldSystemPrompt(messages) : messages);
      if (r.error || !/```repl\s*\n/.test(r.text)) return false;
    }
    return true;
  }

  /** Does the model take a prompt of `chars` and still see both its first and last line? */
  private async holds(id: string, chars: number, systemAsUser: boolean): Promise<Holds> {
    const first = testWord();
    const last = testWord();
    const messages = [
      { role: "system" as const, content: `The first test word is ${first}.\n${windowFiller(chars)}\nThe last test word is ${last}.` },
      { role: "user" as const, content: "Reply with the first test word and the last test word, nothing else." },
    ];
    const r = await this.ask(id, systemAsUser ? foldSystemPrompt(messages) : messages);
    if (r.error) {
      const max = MAX_CHARS_RE.exec(r.error);
      return { ok: false, maxChars: max ? Number(max[1]) : null, sizeLimit: !!max || INPUT_LIMIT_RE.test(r.error), error: r.error };
    }
    // A provider that silently cuts long prompts loses one of the words.
    return r.text.includes(first) && r.text.includes(last) ? { ok: true } : { ok: false, maxChars: null, sizeLimit: true, error: "lost part of the prompt" };
  }

  /** `holds`, retried once when the failure does not name the size. Gateways often relay an input limit as a bare 502,
   * so a long prompt that fails twice while a short one still works right after counts as a size refusal. A rate limit
   * or timeout never does: that is the provider being busy. */
  private async holdsSized(id: string, chars: number, systemAsUser: boolean): Promise<Holds> {
    const r = await this.holds(id, chars, systemAsUser);
    if (r.ok || r.sizeLimit) return r;
    const again = await this.holds(id, chars, systemAsUser);
    if (again.ok || again.sizeLimit || BUSY_RE.test(again.error)) return again;
    const control = await this.holds(id, Math.min(1000, Math.floor(chars / 4)), systemAsUser);
    if (!control.ok) return again;
    this.d.log(`${id} fails a ${chars}-character prompt twice but takes a short one; counting it as an input limit (${again.error})`);
    return { ...again, sizeLimit: true };
  }

  /** Real input limit in characters, or null when the model takes the full test prompt (its advertised window then
   * stands). */
  private async measureInput(m: GatewayModel, systemAsUser: boolean): Promise<number | null> {
    const s = this.d.settings().omniroute;
    let hi = Math.min(s.windowProbeChars, m.context_length ? Math.floor(m.context_length * CHARS_PER_TOKEN * 0.8) : Infinity);
    let r = await this.holdsSized(m.id, hi, systemAsUser);
    if (r.ok) return null;
    let lo = 0;
    for (;;) {
      if (r.maxChars) return Math.floor(r.maxChars * 0.9);
      if (lo) return lo;
      const next = Math.floor(hi / 2);
      if (next < s.minAgentInputChars / 2) return next;
      r = await this.holdsSized(m.id, next, systemAsUser);
      if (r.ok) {
        lo = next;
        const mid = Math.floor((lo + hi) / 2);
        r = await this.holdsSized(m.id, mid, systemAsUser);
        if (r.ok) return mid;
      } else hi = next;
    }
  }

  /** Tests models with a tiny request, two at a time per provider (browser-backed providers share one session). */
  async probe(ids: string[]): Promise<OmniRouteView> {
    this.requireRunning();
    if (this.view.busy) throw new Error(`busy: ${this.view.busy}`);
    const all = await this.gatewayModels();
    const prev = new Map(this.view.models.map((m) => [m.id, m]));
    let targets: GatewayModel[];
    if (ids.length) targets = ids.map((id) => all.find((m) => m.id === id)).filter((m): m is GatewayModel => !!m);
    else {
      const free = await this.freeProviders();
      const s = this.d.settings().omniroute;
      targets = freeChatCandidates(all, free, Math.max(3, Math.ceil((s.maxChainModels * 3) / Math.max(1, free.size))));
    }
    // A full test round replaces every earlier result: a model it did not test is untested, not "worked last time".
    const views = new Map<string, OmniRouteModelView>(this.view.models.map((m) => [m.id, ids.length ? m : { ...m, ok: null, error: null, latencyMs: null }]));
    for (const m of targets)
      views.set(m.id, {
        ...this.toView(m, prev.get(m.id)),
        ok: null,
        error: null,
        latencyMs: null,
        systemAsUser: null,
        inputLimitChars: null,
        agentReady: null,
      });
    let done = 0;
    this.set({
      models: [...views.values()],
      busy: `Testing ${targets.length} free models (0/${targets.length})`,
    });
    const byProvider = new Map<string, GatewayModel[]>();
    for (const m of targets) byProvider.set(m.owned_by ?? "", [...(byProvider.get(m.owned_by ?? "") ?? []), m]);
    try {
      await Promise.all(
        [...byProvider.values()].map(async (list) => {
          const lanes = [list.filter((_, i) => i % 2 === 0), list.filter((_, i) => i % 2 === 1)];
          await Promise.all(
            lanes.map(async (lane) => {
              for (const m of lane) {
                if (this.closing) return;
                const r = await this.probeOne(m);
                views.set(m.id, { ...views.get(m.id)!, ...r });
                done++;
                this.set({
                  models: [...views.values()],
                  busy: `Testing ${targets.length} free models (${done}/${targets.length})`,
                });
              }
            }),
          );
        }),
      );
    } finally {
      this.set({ busy: null });
      this.save();
    }
    return this.state();
  }

  /** Makes the given models (or the best working ones) the agent's failover chain. The IDE switches between them
   * itself, sending each the prompt in the shape it understands and resuming cut-off replies. The gateway combo is
   * kept for other apps, with only the models that honour a system prompt. */
  /** Tests free models and saves the chain behind the "OmniRoute" model. `selectAgent` also makes it the agent's model. */
  async setup(ids: string[], selectAgent = true): Promise<OmniRouteView> {
    this.requireRunning();
    let chosen: OmniRouteModelView[];
    if (ids.length) {
      const all = await this.gatewayModels();
      const untested = ids.filter((id) => this.view.models.find((m) => m.id === id)?.systemAsUser == null);
      if (untested.length) await this.probe(untested);
      const prev = new Map(this.view.models.map((m) => [m.id, m]));
      chosen = ids
        .map((id) => all.find((m) => m.id === id))
        .filter((m): m is GatewayModel => !!m)
        .map((m) => this.toView(m, prev.get(m.id)));
      if (!chosen.length) throw new Error("none of those models exist in the gateway");
      chosen = chosen.filter((m) => m.ok);
      if (!chosen.length) throw new Error("none of those models passed the test; see the errors next to them");
    } else {
      await this.probe([]);
      const working = this.view.models
        .filter((m) => m.ok)
        .sort(
          (a, b) =>
            // A model that does its work in cells comes first; one that answers without its tools is a last resort.
            Number(b.agentReady !== false) - Number(a.agentReady !== false) ||
            // A model that holds the full test prompt does real agent work; a cramped one compacts every few steps.
            Number(a.inputLimitChars !== null) - Number(b.inputLimitChars !== null) ||
            capabilityTier(b.id) - capabilityTier(a.id) ||
            Number(!!a.systemAsUser) - Number(!!b.systemAsUser) ||
            Number(b.toolCalling) - Number(a.toolCalling) ||
            (b.inputLimitChars ?? Infinity) - (a.inputLimitChars ?? Infinity) ||
            (b.contextLength ?? 0) - (a.contextLength ?? 0) ||
            (a.latencyMs ?? 0) - (b.latencyMs ?? 0),
        );
      chosen = interleaveByProvider(working, this.d.settings().omniroute.maxChainModels);
      if (!chosen.length) throw new Error("no free model passed the test. Connect a free provider with a key (Groq, Gemini, Mistral, OpenRouter…) and try again.");
    }
    this.set({ busy: "Saving the free model chain" });
    try {
      const combo = this.d.settings().omniroute.comboName;
      await this.cli(["combo", "delete", combo, "--yes"], {
        json: false,
      }).catch(() => undefined);
      const native = chosen.filter((m) => !m.systemAsUser);
      if (native.length) await this.cli(["combo", "create", combo, "--strategy", "priority", "--models", native.map((m) => m.id).join(",")], { json: false });
      this.applyToSettings(chosen, selectAgent);
      const order = new Map(chosen.map((m, i) => [m.id, i]));
      const known = new Map(this.view.models.map((m) => [m.id, m]));
      for (const m of chosen) if (!known.has(m.id)) known.set(m.id, m);
      const models = [...known.values()].map((m) => ({
        ...m,
        chainIndex: order.get(m.id) ?? null,
      }));
      models.sort((a, b) => (a.chainIndex ?? Infinity) - (b.chainIndex ?? Infinity));
      this.set({ models, busy: null });
      this.save();
      return this.state();
    } catch (e) {
      this.set({
        busy: null,
        error: e instanceof Error ? e.message : String(e),
      });
      throw e;
    }
  }

  private applyToSettings(chosen: OmniRouteModelView[], selectAgent: boolean): void {
    const s = this.d.settings();
    const pid = OMNIROUTE_PROVIDER_ID;
    const members: ModelConfig[] = chosen.map((m) => {
      let ctx = m.contextLength ?? s.local.localContextCap;
      let max = Math.max(256, Math.min(m.maxOutputTokens ?? Math.floor(ctx / 4), Math.floor(ctx / 4)));
      if (m.inputLimitChars) {
        // The measured limit counts input only: the window is that much input plus room for the reply.
        const input = Math.floor(m.inputLimitChars / CHARS_PER_TOKEN);
        max = Math.max(256, Math.min(m.maxOutputTokens ?? input, input));
        ctx = Math.min(ctx, input + max);
      }
      return {
        id: memberId(m.id),
        providerId: pid,
        name: m.id,
        label: `${m.name} (free)`,
        kind: "chat",
        contextLength: ctx,
        maxOutputTokens: max,
        dim: null,
        ...(m.systemAsUser ? { systemAsUser: true } : {}),
      };
    });
    const chain = members.map((m) => m.id);
    const models = [...s.models.filter((m) => m.providerId !== pid), ...members];
    const isMember = (id: string) => id.startsWith(`${pid}:`);
    // The user's own models stay behind the free chain as the last resort.
    const previous = [s.roles.agent, ...s.roles.agentFallbacks].filter((id) => !isMember(id));
    const keep = (id: string) => !isMember(id) || id === OMNIROUTE_AUTO_MODEL_ID || chain.includes(id);
    const roles = selectAgent
      ? { ...s.roles, agent: OMNIROUTE_AUTO_MODEL_ID, agentFallbacks: [...new Set(previous)] }
      : {
          ...s.roles,
          agent: keep(s.roles.agent) ? s.roles.agent : OMNIROUTE_AUTO_MODEL_ID,
          agentFallbacks: s.roles.agentFallbacks.filter(keep),
        };
    this.d.updateSettings({
      models: [...models, autoModel(s, chain, members)],
      omniroute: { ...s.omniroute, chain },
      roles: { ...roles, summarizer: keep(roles.summarizer) ? roles.summarizer : "" },
    });
  }
}

function memberId(gatewayModelId: string): string {
  return `${OMNIROUTE_PROVIDER_ID}:${gatewayModelId}`;
}

/** The "OmniRoute" model the pickers list. Its limits are the first chain model's; the router expands it into the chain. */
function autoModel(s: Settings, chain: string[], members: ModelConfig[] = s.models): ModelConfig {
  const first = members.find((m) => m.id === chain[0]);
  return {
    id: OMNIROUTE_AUTO_MODEL_ID,
    providerId: OMNIROUTE_PROVIDER_ID,
    name: s.omniroute.comboName,
    label: "OmniRoute (free, switches automatically)",
    kind: "chat",
    contextLength: first?.contextLength ?? s.local.localContextCap,
    maxOutputTokens: first?.maxOutputTokens ?? s.local.localMaxOutputTokens,
    dim: null,
  };
}
