import { z } from "zod";
import type { TurnEvent } from "./events";
import type { Settings } from "./settings";
import type { LocalModelStatus } from "./models";
import { HubDeployRequest, type HubStateView } from "./hub";
import type { OmniRouteView } from "./omniroute";

/** Every renderer -> main invoke channel and its payload schema. Main validates each payload. */
export const IpcSchemas = {
  "app:info": z.object({}),
  /** Returns the folder passed via RLM_OPEN_WORKSPACE once, then null. */
  "app:takeAutoOpen": z.object({}),
  "dialog:openFolder": z.object({}),
  /** Starts another copy of the IDE (another window), optionally on a folder. */
  "app:newWindow": z.object({ root: z.string().min(1).optional() }),
  "workspace:open": z.object({ root: z.string().min(1) }),
  "fs:readDir": z.object({ path: z.string() }),
  "fs:readFile": z.object({ path: z.string() }),
  "fs:writeFile": z.object({ path: z.string(), content: z.string() }),
  "fs:create": z.object({ path: z.string(), kind: z.enum(["file", "dir"]) }),
  "fs:rename": z.object({ from: z.string(), to: z.string() }),
  "fs:delete": z.object({ path: z.string() }),
  "fs:search": z.object({ query: z.string().min(1), maxResults: z.number().int().positive().max(2000).default(300) }),
  "fs:listFiles": z.object({ limit: z.number().int().positive().max(50000).default(20000) }),
  "pty:create": z.object({ cols: z.number().int().positive(), rows: z.number().int().positive() }),
  "pty:write": z.object({ id: z.string(), data: z.string() }),
  "pty:resize": z.object({ id: z.string(), cols: z.number().int().positive(), rows: z.number().int().positive() }),
  "pty:kill": z.object({ id: z.string() }),
  "settings:get": z.object({}),
  "settings:update": z.object({ patch: z.record(z.unknown()) }),
  "secrets:setKey": z.object({ providerId: z.string().min(1), key: z.string() }),
  "git:info": z.object({}),
  "git:status": z.object({}),
  "git:show": z.object({ path: z.string() }),
  "git:commit": z.object({ message: z.string().min(1), paths: z.array(z.string()).default([]) }),
  "chat:list": z.object({}),
  "chat:create": z.object({ title: z.string().default("New chat") }),
  "chat:rename": z.object({ chatId: z.string(), title: z.string() }),
  "chat:delete": z.object({ chatId: z.string() }),
  "chat:messages": z.object({ chatId: z.string() }),
  "chat:send": z.object({
    chatId: z.string(),
    text: z.string().min(1),
    editorContext: z
      .object({
        activeFile: z.string().optional(),
        selection: z.string().optional(),
        openFiles: z.array(z.string()).default([]),
        mentions: z.array(z.string()).default([]),
      })
      .default({}),
    toolGroups: z.array(z.string()).optional(),
  }),
  "chat:stop": z.object({ turnId: z.string() }),
  "chat:subscribe": z.object({ turnId: z.string(), lastSeq: z.number().int().default(-1) }),
  "chat:activeTurn": z.object({ chatId: z.string() }),
  "chat:approve": z.object({ approvalId: z.string(), approved: z.boolean(), remember: z.enum(["once", "session", "workspace"]).default("once") }),
  "chat:feedback": z.object({ messageId: z.string(), value: z.enum(["up", "down", "none"]) }),
  "chat:checkpoint": z.object({ chatId: z.string() }),
  "task:board": z.object({ chatId: z.string() }),
  "task:cancelWakeup": z.object({ chatId: z.string() }),
  "pinned:list": z.object({ chatId: z.string() }),
  "pinned:upsert": z.object({ chatId: z.string(), id: z.string().optional(), kind: z.enum(["goal", "constraint", "decision"]), text: z.string().min(1) }),
  "pinned:delete": z.object({ id: z.string() }),
  "memory:list": z.object({ query: z.string().default("") }),
  "memory:update": z.object({ id: z.string(), text: z.string().min(1) }),
  "memory:add": z.object({ text: z.string().min(1), category: z.string().default("preference") }),
  "memory:delete": z.object({ id: z.string() }),
  "memory:forgetAll": z.object({}),
  "memory:export": z.object({}),
  "memory:import": z.object({ json: z.string() }),
  "memory:stats": z.object({}),
  "edits:list": z.object({}),
  "edits:resolve": z.object({ editId: z.string(), accept: z.boolean() }),
  "models:status": z.object({}),
  "models:install": z.object({ modelId: z.string() }),
  "models:import": z.object({ modelId: z.string(), sourceDir: z.string() }),
  "models:detect": z.object({ modelId: z.string() }),
  "models:test": z.object({ modelId: z.string() }),
  /** Stores a chat preview document; returns its rlm-preview:// URL. */
  "preview:create": z.object({ source: z.string().max(5_000_000), kind: z.enum(["html", "svg", "canvas"]).default("html") }),
  "browser:navigate": z.object({ url: z.string().min(1) }),
  "browser:command": z.object({ cmd: z.enum(["back", "forward", "reload", "stop", "devtools", "close", "external", "cancelPick"]) }),
  "browser:state": z.object({}),
  "browser:screenshot": z.object({}),
  "browser:pick": z.object({}),
  "git:branches": z.object({}),
  "git:checkout": z.object({ branch: z.string().min(1).max(200).regex(/^[\w./-]+$/), create: z.boolean().default(false) }),
  "hub:state": z.object({}),
  /** Repo id ("org/name"), a huggingface.co URL, optionally with a revision. */
  "hub:lookup": z.object({ repo: z.string().min(1).max(300), revision: z.string().max(200).default("") }),
  "hub:download": z.object({ repo: z.string().min(1).max(300), revision: z.string().max(200).default(""), files: z.array(z.string().max(1000)).min(1).max(20000) }),
  "hub:job": z.object({ id: z.string().min(1), action: z.enum(["pause", "resume", "cancel", "remove"]) }),
  "hub:deleteModel": z.object({ modelId: z.string().min(1) }),
  "hub:targets": z.object({ modelId: z.string().min(1) }),
  "hub:deploy": HubDeployRequest,
  "hub:deployment": z.object({ id: z.string().min(1), action: z.enum(["start", "stop", "remove", "test"]) }),
  "hub:tools": z.object({
    action: z.enum(["probe", "setupPython", "installPackages", "installLlamaCpp", "downloadLlamaCpp"]),
    packages: z.array(z.string().regex(/^[A-Za-z0-9_.\-\[\],<>=!~ ]+$/).max(200)).max(50).default([]),
    /** Source builds: cpu or cuda. Downloads: auto (best for this computer), gpu, or cpu. */
    backend: z.enum(["cpu", "cuda", "gpu", "auto"]).default("cpu"),
    env: z.enum(["main", "onnx"]).default("main"),
  }),
  "hub:setToken": z.object({ token: z.string().max(500) }),
  "hub:log": z.object({ id: z.string().min(1) }),
  "omniroute:state": z.object({}),
  "omniroute:install": z.object({}),
  "omniroute:service": z.object({ action: z.enum(["start", "stop", "restart"]) }),
  "omniroute:catalog": z.object({}),
  /** Adds a free provider to the gateway. The key goes to the gateway over stdin and is never stored by the IDE. */
  "omniroute:connect": z.object({ provider: z.string().min(1).max(80).regex(/^[a-z0-9][a-z0-9._-]*$/), key: z.string().max(4000).default("") }),
  "omniroute:disconnect": z.object({ connectionId: z.string().min(1).max(100) }),
  /** Tests which free models answer. Empty = the auto-picked candidates. */
  "omniroute:probe": z.object({ models: z.array(z.string().min(1).max(300)).max(60).default([]) }),
  /** Makes these models, in this order, the agent and its backups (empty = test and auto-pick). */
  "omniroute:setup": z.object({ models: z.array(z.string().min(1).max(300)).max(20).default([]) }),
  /** Opens the gateway dashboard and copies its password to the clipboard. */
  "omniroute:dashboard": z.object({}),
  "team:state": z.object({}),
  "team:profile": z.object({ name: z.string().max(100), email: z.string().max(200) }),
  "team:host": z.object({
    enabled: z.boolean(),
    name: z.string().max(80).default(""),
    lan: z.boolean().default(true),
    httpPort: z.number().int().min(0).max(65535).default(7420),
    sshPort: z.number().int().min(0).max(65535).nullable().default(7422),
    discovery: z.boolean().default(true),
  }),
  "team:connect": z.object({
    transport: z.discriminatedUnion("kind", [
      z.object({ kind: z.literal("http"), url: z.string().regex(/^https?:\/\/[^\s]+$/) }),
      z.object({ kind: z.literal("ssh"), host: z.string().min(1).max(255), port: z.number().int().min(1).max(65535) }),
      z.object({
        kind: z.literal("jump"),
        host: z.string().min(1).max(255),
        port: z.number().int().min(1).max(65535),
        username: z.string().min(1).max(100),
        hubHost: z.string().min(1).max(255).default("127.0.0.1"),
        hubPort: z.number().int().min(1).max(65535).default(7420),
      }),
    ]),
    code: z.string().max(100).optional(),
    token: z.string().max(200).optional(),
    password: z.string().max(500).optional(),
    label: z.string().max(80).optional(),
  }),
  "team:disconnect": z.object({ connId: z.string().min(1) }),
  "team:reconnect": z.object({ connId: z.string().min(1) }),
  "team:call": z.object({ connId: z.string().min(1), method: z.enum(["GET", "POST", "PATCH", "DELETE"]), path: z.string().min(1).max(2000), body: z.unknown().optional() }),
  /** push/pull/fetch the open workspace against its hub remote. */
  "team:git": z.object({ op: z.enum(["push", "pull", "fetch"]), branch: z.string().max(200).optional() }),
  /** The hub project the open workspace is linked to (from its git remotes), or null. */
  "team:workspaceProject": z.object({}),
  "team:clone": z.object({ connId: z.string().min(1), group: z.string().min(1), project: z.string().min(1), parent: z.string().min(1) }),
  "team:share": z.object({ connId: z.string().min(1), group: z.string().min(1), name: z.string().min(1).max(80), description: z.string().max(2000).optional() }),
  "team:reportBug": z
    .object({
      connId: z.string().optional(),
      group: z.string().optional(),
      project: z.string().optional(),
      title: z.string().min(1).max(200),
      summary: z.string().max(20000).optional(),
      severity: z.enum(["low", "medium", "high", "critical"]).default("medium"),
      error: z.string().max(50000).optional(),
      file: z.string().max(1000).optional(),
      line: z.number().int().positive().optional(),
      steps: z.array(z.string().max(2000)).max(50).optional(),
      expected: z.string().max(5000).optional(),
      actual: z.string().max(5000).optional(),
      logs: z.string().max(200000).optional(),
      notes: z.string().max(20000).optional(),
      assigneeId: z.string().optional(),
      environment: z.record(z.string()).optional(),
    }),
} as const;

export type IpcChannel = keyof typeof IpcSchemas;
export type IpcRequest<C extends IpcChannel> = z.input<(typeof IpcSchemas)[C]>;

export interface DirEntry {
  name: string;
  path: string;
  kind: "file" | "dir";
}

export interface ChatSummary {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  /** Project folder the chat belongs to; chats are listed, recalled and run only inside it. */
  workspaceId: string | null;
}

export interface StoredMessage {
  id: string;
  chatId: string;
  turnId: string;
  role: "user" | "assistant";
  text: string;
  createdAt: number;
  status: "done" | "error" | "cancelled" | "running";
  events?: TurnEvent[];
  feedback?: "up" | "down" | null;
  /** User-role messages the IDE wrote itself (scheduled wake-ups). */
  origin?: "user" | "wakeup";
}

export interface TaskBoardView {
  chatId: string;
  goal: string;
  steps: Array<{ text: string; status: "todo" | "doing" | "done" | "blocked" | "skipped"; note?: string }>;
  log: Array<{ at: number; text: string }>;
  wakeup: { at: number; reason: string } | null;
  autoWakeups: number;
  updatedAt: number;
}

export interface MemoryFactView {
  id: string;
  text: string;
  category: string;
  memoryType: "short_term" | "long_term";
  createdAt: number;
  updatedAt: number;
  workspaceId: string | null;
  versions: number;
}

export interface PinnedFactView {
  id: string;
  chatId: string;
  kind: "goal" | "constraint" | "decision";
  text: string;
}

export interface PendingEditView {
  editId: string;
  turnId: string;
  path: string;
  before: string;
  after: string;
  createdAt: number;
}

export interface SearchHit {
  path: string;
  line: number;
  text: string;
}

export interface GitInfo {
  branch: string | null;
  dirty: number;
}

export interface GitFileStatus {
  path: string;
  /** Two-letter porcelain code, e.g. " M", "??", "A ". */
  code: string;
}

export interface AppInfo {
  version: string;
  platform: string;
  userDataDir: string;
  sandbox: { mode: "bwrap" | "landlock" | "jobobject" | "none"; detail: string };
}

export interface OrchestratorStatus {
  memory: "online" | "offline" | "disabled";
  kernel: "idle" | "busy" | "down";
  model: string;
  contextLength: number | null;
}

/** The API exposed on `window.rlm` by the preload script. */
export interface RlmApi {
  invoke<C extends IpcChannel>(channel: C, payload: IpcRequest<C>): Promise<unknown>;
  onTurnEvent(cb: (ev: TurnEvent) => void): () => void;
  onPtyData(cb: (msg: { id: string; data: string }) => void): () => void;
  onPtyExit(cb: (msg: { id: string; code: number }) => void): () => void;
  onFsChanged(cb: (msg: { path: string }) => void): () => void;
  onStatus(cb: (s: OrchestratorStatus) => void): () => void;
  onModelProgress(cb: (p: { modelId: string; received: number; total: number; file: string; done: boolean; error?: string }) => void): () => void;
  onSettings(cb: (s: Settings) => void): () => void;
  onBrowserState(cb: (s: BrowserState) => void): () => void;
  onTeamState(cb: (s: TeamStateView) => void): () => void;
  onHubState(cb: (s: HubStateView) => void): () => void;
  onOmniRouteState(cb: (s: OmniRouteView) => void): () => void;
  /** A chat's task board changed (plan, progress, wake-up). */
  onTaskBoard(cb: (b: TaskBoardView) => void): () => void;
  /** Live hub events: notification, activity, changed. */
  onTeamEvent(cb: (e: { connId: string; type: string; data: unknown }) => void): () => void;
  /** Where the browser page should be drawn (window coordinates); fire-and-forget. */
  browserBounds(bounds: { x: number; y: number; width: number; height: number }, visible: boolean): void;
  window: { minimize(): void; maximize(): void; close(): void; platform: string };
}

export interface BrowserState {
  open: boolean;
  url: string;
  title: string;
  loading: boolean;
  canGoBack: boolean;
  canGoForward: boolean;
  devtools: boolean;
  picking: boolean;
  agent: string | null;
  error: string | null;
}

export interface BrowserScreenshot {
  url: string;
  file: string;
  width: number;
  height: number;
  pageUrl: string;
  title: string;
}

export interface BrowserElementRef {
  selector: string;
  tag: string;
  text: string;
  html: string;
  rect: { x: number; y: number; width: number; height: number };
  url: string;
  title: string;
  image: string | null;
}

export type TeamHostConfig = z.output<(typeof IpcSchemas)["team:host"]>;
export type TeamConnectInput = z.output<(typeof IpcSchemas)["team:connect"]>;
export type TeamTransport = TeamConnectInput["transport"] & { hostFingerprint?: string | null };

export interface TeamStateView {
  profile: { name: string; email: string };
  publicKey: string;
  host: TeamHostConfig;
  hosting: { hubId: string; httpPort: number; sshPort: number | null; sshFingerprint: string | null; addresses: string[] } | null;
  hostError: string | null;
  connections: Array<{ id: string; label: string; hubId: string; hubName: string; transport: TeamTransport; userName: string; status: "online" | "connecting" | "offline" | "error"; error: string | null }>;
  discovered: Array<{ hubId: string; name: string; address: string; httpPort: number; sshPort: number | null; sshFingerprint: string | null }>;
}

export interface GitBranchInfo {
  name: string;
  current: boolean;
  upstream: string | null;
  ahead: number;
  behind: number;
}

export type { LocalModelStatus };
