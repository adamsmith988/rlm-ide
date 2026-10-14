import { app, BrowserWindow, dialog, ipcMain, net, protocol, shell } from "electron";
import { existsSync, statSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";
import { BUILTIN, IpcSchemas, type HubDeployRequest, type IpcChannel, type LocalModelStatus, type Settings } from "@rlm/contracts";
import { bytesOnDisk, downloadModel, importModel, isInstalled, loadManifest, totalBytes, type ModelManifest } from "@rlm/inference/src/model_manager";
import { FsService } from "./fs_service";
import { InferenceHost } from "./inference_host";
import { OrchestratorHost, type OrchestratorEvent } from "./orchestrator_host";
import { BrowserService } from "./browser_service";
import { checkPage, type PageCheckArgs } from "./page_check";
import { PreviewService } from "./preview_service";
import { paths } from "./paths";
import { PtyService } from "./pty_service";
import { SecretStore, SettingsStore } from "./settings_store";
import { TeamManager } from "./team_manager";
import { HubService } from "@rlm/hub";

// WebGPU for the built-in model (Linux needs Vulkan enabled explicitly).
app.commandLine.appendSwitch("enable-unsafe-webgpu");
if (process.platform === "linux") {
  app.commandLine.appendSwitch("enable-features", "Vulkan,VulkanFromANGLE,DefaultANGLEVulkan");
  // Dawn hides shader-f16 on NVIDIA Vulkan by default; the q4f16 model needs it.
  app.commandLine.appendSwitch("enable-dawn-features", "vulkan_enable_f16_on_nvidia");
}
if (process.env.RLM_USER_DATA) app.setPath("userData", process.env.RLM_USER_DATA);

protocol.registerSchemesAsPrivileged([
  { scheme: "rlm-model", privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true, corsEnabled: true } },
  ...PreviewService.registerSchemes(),
]);

let mainWindow: BrowserWindow | null = null;
const settings = new SettingsStore(path.join(app.getPath("userData"), "settings.json"));
const secrets = new SecretStore(path.join(app.getPath("userData"), "secrets.json"));
const send = (channel: string, payload: unknown) => mainWindow?.webContents.send(channel, payload);
const fsService = new FsService((p) => send("fs:changed", { path: p }));
const pty = new PtyService(
  (id, data) => send("pty:data", { id, data }),
  (id, code) => send("pty:exit", { id, code }),
);
let status: Record<string, unknown> = { memory: "online", kernel: "idle", model: "", contextLength: null };
let manifest: ModelManifest = {};
let autoOpen: string | null = process.env.RLM_OPEN_WORKSPACE ?? null;
const downloads = new Map<string, AbortController>();
const preview = new PreviewService(path.join(app.getPath("userData"), "attachments"));
const browser = new BrowserService(
  () => mainWindow,
  preview,
  (st) => send("browser:state", st),
  () => fsService.current(),
);
const team = new TeamManager(app.getPath("userData"), secrets, send, (msg) => console.warn(`[team] ${msg}`));
const hub = new HubService({
  dir: path.join(app.getPath("userData"), "hub"),
  scriptsDir: paths.hubScripts,
  basePythons: () => paths.pythonCandidates(),
  settings: () => settings.get(),
  updateSettings: (patch) => settings.update(patch as Record<string, unknown>),
  secrets,
  emit: (s) => send("hub:state", s),
  log: (msg) => console.warn(`[hub] ${msg}`),
});

function needWorkspace(): string {
  const root = fsService.current();
  if (!root) throw new Error("open a folder first");
  return root;
}

const orch = new OrchestratorHost((e: OrchestratorEvent) => {
  if (e.kind === "turn-event") send("turn:event", e.ev);
  else if (e.kind === "task-board") send("task:board", e.board);
  else if (e.kind === "status") {
    status = { ...status, ...e.status };
    send("status", status);
  } else if (e.kind === "need-inference") void inference.ensure();
  else if (e.kind === "settings-patch") settings.update(e.patch);
  else if (e.kind === "host-request") {
    const reply = (ok: boolean, value: unknown, error?: string) => orch.post({ kind: "host-response", id: e.id, ok, value, error });
    let work: Promise<unknown>;
    if (e.type === "check-page") work = checkPage(e.args as unknown as PageCheckArgs);
    else if (e.type.startsWith("browser:")) {
      work = browser.agent(e.type.slice("browser:".length), e.args);
    } else return reply(false, null, `unknown host request ${e.type}`);
    work.then(
      (v) => reply(true, v),
      (err: unknown) => reply(false, null, err instanceof Error ? err.message : String(err)),
    );
  }
});

const inference = new InferenceHost(
  orch,
  () => settings.get(),
  (s) => {
    status = { ...status, inference: s };
    send("status", status);
  },
  (win) => loadPage(win, "inference.html"),
);

function loadPage(win: BrowserWindow, page: string): void {
  const devUrl = process.env.ELECTRON_RENDERER_URL;
  if (!app.isPackaged && devUrl) void win.loadURL(`${devUrl}/${page}`);
  else void win.loadFile(path.join(__dirname, "../renderer", page));
}

function pushSettingsToOrchestrator(s: Settings): void {
  orch.post({ kind: "settings", settings: s, keys: secrets.all() });
}

function createWindow(): void {
  const s = settings.get();
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    frame: false,
    titleBarStyle: "hidden",
    backgroundColor: s.theme === "dark" ? "#0B0B0C" : "#F2F2F0",
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "../preload/index.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
      spellcheck: false,
    },
  });
  mainWindow.once("ready-to-show", () => mainWindow?.show());
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:/.test(url)) void shell.openExternal(url);
    return { action: "deny" };
  });
  mainWindow.webContents.on("will-navigate", (e, url) => {
    if (!url.startsWith(process.env.ELECTRON_RENDERER_URL ?? "file:")) e.preventDefault();
  });
  mainWindow.on("closed", () => (mainWindow = null));
  loadPage(mainWindow, "index.html");
}

function registerModelProtocol(): void {
  protocol.handle("rlm-model", async (req) => {
    const u = new URL(req.url);
    const rel = decodeURIComponent(u.pathname).replace(/^\/+/, "");
    let file: string | null = null;
    if (u.host === "models") {
      for (const root of paths.modelDirs) {
        const p = path.resolve(root, rel);
        if (p.startsWith(path.resolve(root)) && existsSync(p) && statSync(p).isFile()) {
          file = p;
          break;
        }
      }
    } else if (u.host === "ort") {
      const dist = paths.ortWebDist();
      if (dist) {
        const p = path.resolve(dist, rel);
        if (p.startsWith(dist) && existsSync(p)) file = p;
      }
    }
    if (!file) return new Response("not found", { status: 404 });
    const res = await net.fetch(pathToFileURL(file).toString());
    const headers = new Headers(res.headers);
    headers.set("Access-Control-Allow-Origin", "*");
    headers.set("Content-Length", String(statSync(file).size));
    if (file.endsWith(".wasm")) headers.set("Content-Type", "application/wasm");
    if (file.endsWith(".mjs") || file.endsWith(".js")) headers.set("Content-Type", "text/javascript");
    return new Response(res.body, { status: 200, headers });
  });
}

async function modelStatus(): Promise<LocalModelStatus[]> {
  const out: LocalModelStatus[] = [];
  for (const [id, entry] of Object.entries(manifest)) {
    const root = paths.modelDirs.find((r) => isInstalled(r, entry));
    out.push({
      id,
      installed: !!root,
      bytesOnDisk: root ? totalBytes(entry) : bytesOnDisk(paths.downloadDir, entry),
      expectedBytes: totalBytes(entry),
      runtime: id === BUILTIN.agentModelId ? (inference.state.device === "none" ? "unavailable" : inference.state.device) : "cpu",
      detail: id === BUILTIN.agentModelId ? `${inference.state.status}${inference.state.detail ? `: ${inference.state.detail}` : ""}` : root ? `installed in ${root}` : "not installed",
    });
  }
  return out;
}

function gitInfo(): { branch: string | null; dirty: number } {
  const root = fsService.current();
  if (!root) return { branch: null, dirty: 0 };
  const r = spawnSync("git", ["status", "--porcelain", "--branch"], { cwd: root, encoding: "utf8", timeout: 5000 });
  if (r.status !== 0) return { branch: null, dirty: 0 };
  const lines = r.stdout.trim().split("\n");
  const branch = (lines[0] ?? "").replace(/^## /, "").split("...")[0] ?? null;
  return { branch, dirty: Math.max(0, lines.length - 1) };
}

function gitStatus(): Array<{ path: string; code: string }> {
  const root = fsService.current();
  if (!root) return [];
  const r = spawnSync("git", ["status", "--porcelain=v1", "-uall"], { cwd: root, encoding: "utf8", timeout: 5000 });
  if (r.status !== 0) return [];
  return r.stdout
    .split("\n")
    .filter(Boolean)
    .map((l) => ({ code: l.slice(0, 2), path: l.slice(3).split(" -> ").pop()!.replace(/^"|"$/g, "") }));
}

/** Channels the orchestrator process answers; main forwards them after validation. */
const ORCH_CHANNELS = new Set<IpcChannel>([
  "chat:list",
  "chat:create",
  "chat:rename",
  "chat:delete",
  "chat:messages",
  "chat:send",
  "chat:stop",
  "chat:subscribe",
  "chat:activeTurn",
  "chat:approve",
  "chat:feedback",
  "chat:checkpoint",
  "task:board",
  "task:cancelWakeup",
  "pinned:list",
  "pinned:upsert",
  "pinned:delete",
  "memory:list",
  "memory:update",
  "memory:add",
  "memory:delete",
  "memory:forgetAll",
  "memory:export",
  "memory:import",
  "memory:stats",
  "edits:list",
  "edits:resolve",
  "models:detect",
  "models:test",
]);

async function handle(channel: IpcChannel, payload: Record<string, unknown>): Promise<unknown> {
  if (ORCH_CHANNELS.has(channel)) return orch.call(channel, payload);
  switch (channel) {
    case "app:info":
      return {
        version: app.getVersion(),
        platform: process.platform,
        userDataDir: app.getPath("userData"),
        sandbox: await orch.call("kernel:sandbox", {}).catch(() => ({ mode: "none", detail: "unknown" })),
        secretsBackend: secrets.backend(),
      };
    case "app:takeAutoOpen": {
      const root = autoOpen;
      autoOpen = null;
      return root;
    }
    case "dialog:openFolder": {
      const r = await dialog.showOpenDialog(mainWindow!, { properties: ["openDirectory", "createDirectory"] });
      return r.canceled ? null : r.filePaths[0];
    }
    case "workspace:open": {
      const root = path.resolve(String(payload.root));
      if (!existsSync(root) || !statSync(root).isDirectory()) throw new Error(`not a folder: ${root}`);
      fsService.open(root);
      await orch.call("workspace:set", { root });
      const recent = [root, ...settings.get().recentWorkspaces.filter((r) => r !== root)].slice(0, 10);
      settings.update({ recentWorkspaces: recent });
      return { root, name: path.basename(root) };
    }
    case "fs:readDir":
      return fsService.readDir(String(payload.path));
    case "fs:readFile":
      return fsService.readFile(String(payload.path));
    case "fs:writeFile":
      return fsService.writeFile(String(payload.path), String(payload.content));
    case "fs:create":
      return fsService.create(String(payload.path), payload.kind as "file" | "dir");
    case "fs:rename":
      return fsService.rename(String(payload.from), String(payload.to));
    case "fs:delete":
      return fsService.delete(String(payload.path));
    case "fs:search":
      return fsService.search(String(payload.query), Number(payload.maxResults));
    case "fs:listFiles":
      return fsService.listFiles(Number(payload.limit));
    case "pty:create":
      return pty.create(fsService.current() ?? "", Number(payload.cols), Number(payload.rows));
    case "pty:write":
      return pty.write(String(payload.id), String(payload.data));
    case "pty:resize":
      return pty.resize(String(payload.id), Number(payload.cols), Number(payload.rows));
    case "pty:kill":
      return pty.kill(String(payload.id));
    case "settings:get":
      return settings.get();
    case "settings:update":
      return settings.update(payload.patch as Record<string, unknown>);
    case "secrets:setKey": {
      const s = settings.get();
      const key = String(payload.key ?? "").trim();
      if (!s.providers.some((p) => p.id === payload.providerId && p.kind === "openai_compat")) {
        if (key) throw new Error("unknown provider");
        return true;
      }
      secrets.set(`provider:${String(payload.providerId)}`, key);
      settings.update({ providers: s.providers.map((p) => (p.id === payload.providerId ? { ...p, hasKey: !!key } : p)) });
      return true;
    }
    case "git:info":
      return gitInfo();
    case "git:status":
      return gitStatus();
    case "git:show": {
      const root = fsService.current();
      if (!root) return "";
      const rel = String(payload.path).replace(/\\/g, "/");
      const r = spawnSync("git", ["show", `HEAD:./${rel}`], { cwd: root, encoding: "utf8", timeout: 5000, maxBuffer: 32 * 1024 * 1024 });
      return r.status === 0 ? r.stdout : "";
    }
    case "git:commit": {
      const root = fsService.current();
      if (!root) throw new Error("no workspace");
      const files = (payload.paths as string[]).length ? ["--", ...(payload.paths as string[])] : ["-A"];
      const add = spawnSync("git", ["add", ...files], { cwd: root, encoding: "utf8", timeout: 15000 });
      if (add.status !== 0) throw new Error(add.stderr.trim() || "git add failed");
      // Fall back to the team profile when git has no identity configured.
      const hasIdentity = spawnSync("git", ["config", "user.email"], { cwd: root, encoding: "utf8" }).stdout.trim();
      const profile = team.state().profile;
      const ident = hasIdentity || !profile.email ? [] : ["-c", `user.name=${profile.name}`, "-c", `user.email=${profile.email}`];
      const c = spawnSync("git", [...ident, "commit", "-m", String(payload.message)], { cwd: root, encoding: "utf8", timeout: 30000 });
      if (c.status !== 0) throw new Error((c.stderr || c.stdout).trim() || "git commit failed");
      return c.stdout.trim();
    }
    case "git:branches": {
      const root = fsService.current();
      if (!root) return [];
      const r = spawnSync("git", ["for-each-ref", "--sort=-committerdate", "--format=%(HEAD)\x1f%(refname:short)\x1f%(upstream:short)\x1f%(upstream:track,nobracket)", "refs/heads"], { cwd: root, encoding: "utf8", timeout: 5000 });
      if (r.status !== 0) return [];
      return r.stdout
        .split("\n")
        .filter(Boolean)
        .map((l) => {
          const [head, name, upstream, track] = l.split("\x1f");
          return { name, current: head === "*", upstream: upstream || null, ahead: Number(/ahead (\d+)/.exec(track ?? "")?.[1] ?? 0), behind: Number(/behind (\d+)/.exec(track ?? "")?.[1] ?? 0) };
        });
    }
    case "git:checkout": {
      const root = needWorkspace();
      const branch = String(payload.branch);
      const r = spawnSync("git", payload.create ? ["checkout", "-b", branch] : ["checkout", branch], { cwd: root, encoding: "utf8", timeout: 30000 });
      if (r.status !== 0) throw new Error(r.stderr.trim() || "git checkout failed");
      return true;
    }
    case "hub:state":
      return hub.view();
    case "hub:lookup":
      return hub.lookup(String(payload.repo), String(payload.revision ?? ""));
    case "hub:download":
      return hub.download(String(payload.repo), String(payload.revision ?? ""), payload.files as string[]);
    case "hub:job":
      return hub.jobAction(String(payload.id), payload.action as "pause" | "resume" | "cancel" | "remove");
    case "hub:deleteModel":
      return hub.deleteModel(String(payload.modelId));
    case "hub:targets":
      return hub.targets(String(payload.modelId));
    case "hub:deploy":
      return hub.deploy(payload as unknown as HubDeployRequest);
    case "hub:deployment": {
      const id = String(payload.id);
      if (payload.action === "start") return hub.start(id);
      if (payload.action === "stop") return hub.stop(id);
      if (payload.action === "test") return hub.test(id);
      return hub.removeDeployment(id);
    }
    case "hub:tools":
      return hub.toolsAction(payload.action as "probe", payload.packages as string[], payload.backend as "cpu" | "cuda", payload.env as "main" | "onnx");
    case "hub:setToken":
      return hub.setToken(String(payload.token));
    case "hub:log":
      return hub.readLog(String(payload.id));
    case "team:state":
      return team.state();
    case "team:profile":
      return team.setProfile(payload as { name: string; email: string });
    case "team:host":
      return team.setHost(payload as never);
    case "team:connect":
      return team.connect(payload as never);
    case "team:disconnect":
      return team.disconnect(String(payload.connId));
    case "team:reconnect":
      return team.reconnect(String(payload.connId));
    case "team:call":
      return team.call(String(payload.connId), String(payload.method), String(payload.path), payload.body);
    case "team:git":
      return team.git(needWorkspace(), payload.op as "push" | "pull" | "fetch", { branch: payload.branch as string | undefined });
    case "team:workspaceProject": {
      const root = fsService.current();
      return root ? team.workspaceProject(root) : null;
    }
    case "team:clone":
      return team.clone(String(payload.connId), String(payload.group), String(payload.project), String(payload.parent));
    case "team:share":
      return team.share(needWorkspace(), String(payload.connId), String(payload.group), { name: String(payload.name), description: payload.description as string | undefined });
    case "team:reportBug":
      return team.reportBug(fsService.current(), payload as never);
    case "models:status":
      return modelStatus();
    case "models:install": {
      const id = String(payload.modelId);
      const entry = manifest[id];
      if (!entry) throw new Error(`unknown model ${id}`);
      if (downloads.has(id)) return "already downloading";
      const ac = new AbortController();
      downloads.set(id, ac);
      void downloadModel(id, entry, paths.downloadDir, (p) => send("models:progress", p), ac.signal)
        .catch((e: unknown) => send("models:progress", { modelId: id, received: 0, total: totalBytes(entry), file: "", done: true, error: String(e) }))
        .finally(() => {
          downloads.delete(id);
          inference.stop();
          orch.post({ kind: "models-changed" });
        });
      return "started";
    }
    case "preview:create":
      return preview.create(String(payload.source), payload.kind as "html" | "svg" | "canvas");
    case "browser:navigate":
      return browser.navigate(String(payload.url));
    case "browser:state":
      return browser.state();
    case "browser:screenshot":
      return browser.screenshot();
    case "browser:pick":
      return browser.pickElement();
    case "browser:command": {
      const cmd = String(payload.cmd);
      if (cmd === "back") browser.back();
      else if (cmd === "forward") browser.forward();
      else if (cmd === "reload") browser.reload();
      else if (cmd === "stop") browser.stop();
      else if (cmd === "devtools") browser.toggleDevtools();
      else if (cmd === "close") browser.close();
      else if (cmd === "external") browser.openExternal();
      else if (cmd === "cancelPick") browser.cancelPick();
      return browser.state();
    }
    case "models:import": {
      const id = String(payload.modelId);
      const entry = manifest[id];
      if (!entry) throw new Error(`unknown model ${id}`);
      await importModel(entry, String(payload.sourceDir), paths.downloadDir);
      orch.post({ kind: "models-changed" });
      return true;
    }
    default:
      throw new Error(`unhandled channel ${channel}`);
  }
}

function registerIpc(): void {
  ipcMain.handle("rpc", async (_e, channel: string, payload: unknown) => {
    if (!(channel in IpcSchemas)) throw new Error(`unknown channel ${channel}`);
    const parsed = IpcSchemas[channel as IpcChannel].parse(payload ?? {});
    return handle(channel as IpcChannel, parsed as Record<string, unknown>);
  });
  ipcMain.on("editor:state", (_e, state: unknown) => orch.post({ kind: "editor-state", state }));
  ipcMain.on("browser:bounds", (_e, b: unknown, visible: unknown) => {
    const r = b as { x?: unknown; y?: unknown; width?: unknown; height?: unknown } | null;
    if (!r || ![r.x, r.y, r.width, r.height].every((n) => typeof n === "number" && Number.isFinite(n))) return;
    browser.setBounds(r as { x: number; y: number; width: number; height: number }, visible === true);
  });
  ipcMain.on("window:minimize", () => mainWindow?.minimize());
  ipcMain.on("window:maximize", () => (mainWindow?.isMaximized() ? mainWindow.unmaximize() : mainWindow?.maximize()));
  ipcMain.on("window:close", () => mainWindow?.close());
}

app.whenReady().then(async () => {
  manifest = await loadManifest(paths.manifest).catch(() => ({}));
  registerModelProtocol();
  preview.register();
  registerIpc();
  orch.start({
    dbPath: paths.db,
    snapshotsDir: paths.snapshots,
    taskBoardsDir: path.join(paths.userData, "task-boards"),
    promptsDir: paths.prompts,
    workerScript: paths.workerScript,
    pythonCandidates: paths.pythonCandidates(),
    modelDirs: paths.modelDirs,
    vecExtension: paths.vecExtension(),
    settings: settings.get(),
    keys: secrets.all(),
  });
  settings.onChange((s) => {
    secrets.pruneProviders(s.providers.map((p) => p.id));
    pushSettingsToOrchestrator(s);
    send("settings", s);
  });
  createWindow();
  void team.boot();
  void hub.boot().catch((e: unknown) => console.warn(`[hub] boot failed: ${String(e)}`));
  if (app.isPackaged && settings.get().autoUpdate) {
    import("electron-updater")
      .then(({ autoUpdater }) => {
        autoUpdater.channel = settings.get().updateChannel === "beta" ? "beta" : "latest";
        void autoUpdater.checkForUpdatesAndNotify();
      })
      .catch(() => undefined);
  }
});

let stopped = false;
function stopServices(): void {
  if (stopped) return;
  stopped = true;
  pty.killAll();
  orch.stop();
  inference.stop();
  hub.close();
  void team.close();
}

app.on("window-all-closed", () => {
  stopServices();
  if (process.platform !== "darwin") app.quit();
});
app.on("will-quit", stopServices);
for (const sig of ["SIGTERM", "SIGINT", "SIGHUP"] as const) process.on(sig, () => app.quit());
