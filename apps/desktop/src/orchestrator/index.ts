/** Orchestrator utilityProcess: the composition root that wires core ports to adapters.
 * No Electron imports besides the parentPort messaging. */
import { existsSync, readFileSync } from "node:fs";
import { promises as fsp } from "node:fs";
import path from "node:path";
import { BUILTIN, Settings, type ModelConfig, type TurnEvent } from "@rlm/contracts";
import { Orchestrator, TaskBoards, newId, type StatusPort, type TaskBoard } from "@rlm/core";
import {
  ConsoleLogger,
  createBuiltinTools,
  FilePrompts,
  FileSnapshotStore,
  FileTaskBoardStore,
  LocalBridgeChat,
  openDb,
  ProcessManager,
  ProviderRouter,
  PythonKernelFactory,
  resolveInside,
  SqliteConversationRepo,
  SqliteEditJournal,
  SqliteMemoryStore,
  SqlitePinnedFacts,
  ToolRegistry,
  Workspace,
  type InferenceRequest,
  type InferenceResponse,
  type InferenceTransport,
} from "@rlm/adapters";
import { BgeM3Embedder } from "@rlm/inference/src/embed_bge";

interface ParentPort {
  on(ev: "message", cb: (e: { data: Record<string, unknown>; ports: MessagePortLike[] }) => void): void;
  postMessage(m: unknown): void;
}
interface MessagePortLike {
  on(ev: "message", cb: (e: { data: unknown }) => void): void;
  postMessage(m: unknown): void;
  start(): void;
  close(): void;
}

const parent = (process as unknown as { parentPort: ParentPort }).parentPort;
const post = (m: unknown) => parent.postMessage(m);
const log = new ConsoleLogger();

let settings: Settings = Settings.parse({});
let keys: Record<string, string> = {};
let editorState: Record<string, unknown> = {};

/** Requests that only the main process can serve (it owns BrowserWindows). */
const hostPending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
let hostSeq = 0;
function hostRequest(type: string, args: Record<string, unknown>): Promise<unknown> {
  const id = ++hostSeq;
  return new Promise((resolve, reject) => {
    hostPending.set(id, { resolve, reject });
    post({ kind: "host-request", id, type, args });
  });
}

/** Transport to the inference host; the port arrives (and may be replaced) from main. */
class PortTransport implements InferenceTransport {
  private port: MessagePortLike | null = null;
  private readonly listeners = new Set<(m: InferenceResponse) => void>();
  private waiters: Array<{ resolve: () => void; reject: (e: Error) => void }> = [];
  private ready_ = false;
  private requested = false;

  attach(port: MessagePortLike): void {
    this.port?.close();
    this.port = port;
    this.ready_ = false;
    this.requested = false;
    port.on("message", (e) => {
      const m = e.data as InferenceResponse | { type: "ready" } | { type: "status"; text: string };
      if (m.type === "ready") {
        this.ready_ = true;
        for (const w of this.waiters) w.resolve();
        this.waiters = [];
        return;
      }
      if (m.type === "status") return;
      if (m.type === "error" && m.id === "*") {
        this.fail(m.message);
        this.reset();
        return;
      }
      for (const l of this.listeners) l(m as InferenceResponse);
    });
    port.start();
  }

  fail(reason: string): void {
    for (const w of this.waiters) w.reject(new Error(reason));
    this.waiters = [];
    this.requested = false;
  }

  reset(): void {
    this.port?.close();
    this.port = null;
    this.ready_ = false;
  }

  ready(): Promise<void> {
    if (this.port && this.ready_) return Promise.resolve();
    // A port that is attached but still loading will post "ready"; only ask main when there is none.
    if (!this.port && !this.requested) {
      this.requested = true;
      post({ kind: "need-inference" });
    }
    return new Promise((resolve, reject) => this.waiters.push({ resolve, reject }));
  }

  send(msg: InferenceRequest): void {
    this.port?.postMessage(msg);
  }

  onMessage(cb: (m: InferenceResponse) => void): () => void {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  }
}

const transport = new PortTransport();
let orch: Orchestrator | null = null;
let router: ProviderRouter | null = null;
let memoryStore: SqliteMemoryStore | null = null;
let edits: SqliteEditJournal | null = null;
let workspace: Workspace | null = null;
let pinned: SqlitePinnedFacts | null = null;
let kernels: PythonKernelFactory | null = null;
let repo: SqliteConversationRepo | null = null;
let modelDirs: string[] = [];
const pm = new ProcessManager();

function findModelRoot(dir: string): string | null {
  return modelDirs.find((r) => existsSync(path.join(r, dir, "config.json"))) ?? null;
}

function init(i: Record<string, unknown>): void {
  settings = Settings.parse(i.settings);
  keys = (i.keys as Record<string, string>) ?? {};
  modelDirs = i.modelDirs as string[];
  const db = openDb(String(i.dbPath), (i.vecExtension as string | null) ?? null);
  log.info(`database ready (sqlite-vec ${db.vecLoaded ? "loaded" : "unavailable; using JS vector math"})`);
  repo = new SqliteConversationRepo(db);
  const recovered = repo.recoverInterrupted();
  if (recovered) log.warn(`marked ${recovered} interrupted turn(s) as cancelled`);
  memoryStore = new SqliteMemoryStore(db);
  edits = new SqliteEditJournal(db);
  pinned = new SqlitePinnedFacts(db);
  workspace = new Workspace(pm);
  const embedders = new Map<string, BgeM3Embedder>();

  router = new ProviderRouter({
    settings: () => settings,
    apiKey: (pid) => keys[`provider:${pid}`] ?? null,
    localChat: (model: ModelConfig) =>
      new LocalBridgeChat(model.id, transport, {
        contextLimit: () => settings.local.localContextCap,
        enableThinking: () => settings.local.enableThinking,
      }),
    localEmbed: async (model: ModelConfig) => {
      const root = findModelRoot(model.name);
      if (!root) return null;
      let e = embedders.get(model.id);
      if (!e) {
        e = new BgeM3Embedder(root, model.name, model.dim ?? BUILTIN.embedDim);
        embedders.set(model.id, e);
      }
      return e;
    },
    localNativeContext: (model: ModelConfig) => {
      const root = findModelRoot(model.name);
      if (!root) return null;
      try {
        const cfg = JSON.parse(readFileSync(path.join(root, model.name, "config.json"), "utf8")) as Record<string, unknown>;
        const tc = (cfg.text_config as Record<string, unknown> | undefined) ?? cfg;
        return Number(tc.max_position_embeddings ?? cfg.max_position_embeddings ?? 32768);
      } catch {
        return null;
      }
    },
  });

  kernels = new PythonKernelFactory({
    workerScript: String(i.workerScript),
    pythonCandidates: i.pythonCandidates as string[],
    sandboxNoNetwork: () => settings.rlm.sandboxNoNetwork,
    memoryMb: () => settings.rlm.kernelMemoryMb,
    log,
  });

  const status: StatusPort = {
    memory: (s) => post({ kind: "status", status: { memory: s } }),
    kernel: (s) => post({ kind: "status", status: { kernel: s } }),
  };

  const boards = new TaskBoards({
    store: new FileTaskBoardStore(String(i.taskBoardsDir ?? path.join(path.dirname(String(i.snapshotsDir)), "task-boards"))),
    clock: { now: () => Date.now() },
    settings: () => settings.rlm,
    log,
  });
  boards.onChange((b) => post({ kind: "task-board", board: boardView(b) }));

  const tools = new ToolRegistry([
    ...createBuiltinTools({
      processes: pm,
      shellTimeoutSec: () => settings.tools.shellTimeoutSec,
      cellTimeoutSec: () => settings.rlm.cellTimeoutSec,
      editorState: () => editorState as never,
      searchSession: async (chatId, q, limit) => repo!.searchTurns(chatId, q, limit),
      checkPage: (args) => hostRequest("check-page", args as unknown as Record<string, unknown>) as never,
      browser: (op, args) => hostRequest(`browser:${op}`, args),
    }),
    ...boards.tools(),
  ]);

  orch = new Orchestrator({
    settings: { get: () => settings },
    updateSettings: (patch) => post({ kind: "settings-patch", patch }),
    router,
    catalog: router,
    conversations: repo,
    memoryStore,
    pinned,
    snapshots: new FileSnapshotStore(String(i.snapshotsDir)),
    edits,
    kernels,
    tools,
    prompts: new FilePrompts(String(i.promptsDir)),
    workspace,
    clock: { now: () => Date.now() },
    log,
    status,
    broadcast: (ev: TurnEvent) => post({ kind: "turn-event", ev }),
    readMention: async (p) => {
      const root = workspace!.root();
      if (!root) throw new Error("no workspace");
      return fsp.readFile(resolveInside(root, p), "utf8");
    },
    boards,
  });
  void orch.startScheduler();
  publishModel();
}

function boardView(b: TaskBoard) {
  return { chatId: b.chatId, goal: b.goal, steps: b.steps, log: b.log.slice(-20), wakeup: b.wakeup ? { at: b.wakeup.at, reason: b.wakeup.reason } : null, autoWakeups: b.autoWakeups, updatedAt: b.updatedAt };
}

function publishModel(): void {
  if (!router) return;
  const id = router.modelIdFor("agent");
  void router
    .limits(id)
    .then((l) => post({ kind: "status", status: { model: settings.models.find((m) => m.id === id)?.label ?? id, contextLength: l.contextLength } }))
    .catch(() => post({ kind: "status", status: { model: id, contextLength: null } }));
}

function scope() {
  return { user_id: settings.userId };
}

type Handler = (a: Record<string, unknown>) => Promise<unknown> | unknown;

const methods: Record<string, Handler> = {
  "workspace:set": (a) => workspace!.set(String(a.root)),
  "kernel:sandbox": () => kernels!.sandboxInfo(),
  "chat:list": () => orch!.listChats(),
  "chat:create": (a) => orch!.createChat(String(a.title ?? "New chat")),
  "chat:rename": (a) => repo!.renameChat(String(a.chatId), String(a.title)),
  "chat:delete": (a) => orch!.deleteChat(String(a.chatId)),
  "chat:messages": (a) => orch!.messages(String(a.chatId)),
  "chat:send": (a) =>
    orch!.send({
      chatId: String(a.chatId),
      text: String(a.text),
      editorContext: (a.editorContext as Record<string, never>) ?? {},
      toolGroups: a.toolGroups as string[] | undefined,
    }),
  "chat:stop": (a) => orch!.stop(String(a.turnId)),
  "chat:subscribe": (a) => orch!.runner.replay(String(a.turnId), Number(a.lastSeq ?? -1)),
  "chat:activeTurn": (a) => orch!.runner.activeTurn(String(a.chatId)),
  "chat:approve": (a) => {
    const ok = orch!.runner.resolveApproval(String(a.approvalId), Boolean(a.approved), (a.remember as "once") ?? "once");
    return ok;
  },
  "chat:feedback": (a) => repo!.setFeedback(String(a.messageId).replace(/:a$/, ""), a.value === "none" ? null : (a.value as "up" | "down")),
  "chat:checkpoint": (a) => orch!.checkpoint(String(a.chatId)),
  "task:board": async (a) => {
    const b = await orch!.board(String(a.chatId));
    return b ? boardView(b) : null;
  },
  "task:cancelWakeup": (a) => orch!.cancelWakeup(String(a.chatId)),
  "pinned:list": (a) => pinned!.list(String(a.chatId)),
  "pinned:upsert": (a) => pinned!.upsert({ id: a.id as string | undefined, chatId: String(a.chatId), kind: a.kind as "goal", text: String(a.text) }),
  "pinned:delete": (a) => pinned!.delete(String(a.id)),
  "memory:list": async (a) => {
    const pts = await memoryStore!.list({ ...scope(), memory_type: "long_term" }, String(a.query ?? "") || undefined);
    return pts.map((p) => ({
      id: p.id,
      text: p.payload.text,
      category: p.payload.category,
      memoryType: p.payload.memory_type,
      createdAt: p.payload.created_at,
      updatedAt: p.payload.updated_at,
      workspaceId: p.payload.workspace_id,
      versions: p.payload.history.length,
    }));
  },
  "memory:add": async (a) => {
    const p = await orch!.memory.addFact({ userId: settings.userId, sessionId: "", workspaceId: workspace!.root() }, String(a.text), String(a.category ?? "preference"));
    return p.id;
  },
  "memory:update": (a) => orch!.memory.updateFact(String(a.id), String(a.text), { ...scope(), memory_type: "long_term" }, settings.memory),
  "memory:delete": (a) => memoryStore!.delete([String(a.id)], { ...scope(), memory_type: "long_term" }),
  "memory:forgetAll": async () => {
    await memoryStore!.deleteAll({ ...scope(), memory_type: "long_term" });
    await memoryStore!.deleteAll({ ...scope(), memory_type: "short_term" });
    return true;
  },
  "memory:export": async () => JSON.stringify((await memoryStore!.list({ ...scope(), memory_type: "long_term" })).map((p) => ({ text: p.payload.text, category: p.payload.category, history: p.payload.history })), null, 2),
  "memory:import": async (a) => {
    const arr = JSON.parse(String(a.json)) as Array<{ text: string; category?: string }>;
    for (const f of arr) if (f.text) await orch!.memory.addFact({ userId: settings.userId, sessionId: "", workspaceId: workspace!.root() }, f.text, f.category ?? "imported");
    return arr.length;
  },
  "memory:stats": async () => {
    const facts = (await memoryStore!.list({ ...scope(), memory_type: "long_term" })).length;
    const chats = (await orch!.listChats()).length;
    return { facts, chats, collections: memoryStore!.collections() };
  },
  "edits:list": () => edits!.list(),
  "edits:resolve": async (a) => {
    const e = await edits!.get(String(a.editId));
    if (!e) throw new Error("edit not found");
    if (!a.accept) {
      const root = workspace!.root();
      if (!root) throw new Error("no workspace");
      const abs = resolveInside(root, e.path);
      const current = await fsp.readFile(abs, "utf8").catch(() => "");
      if (current !== e.after) throw new Error(`${e.path} changed since this edit; revert it manually`);
      if (e.before === "" && current !== "") await fsp.rm(abs);
      else await fsp.writeFile(abs, e.before, "utf8");
      await edits!.setState(e.editId, "reverted");
    } else {
      await edits!.setState(e.editId, "applied");
    }
    return true;
  },
  "models:detect": async (a) => {
    const id = String(a.modelId);
    const l = await router!.detect(id);
    if (l.contextLength) {
      post({
        kind: "settings-patch",
        patch: { models: settings.models.map((m) => (m.id === id ? { ...m, contextLength: l.contextLength, maxOutputTokens: l.maxOutputTokens ?? m.maxOutputTokens } : m)) },
      });
    }
    return l;
  },
  "models:test": async (a) => {
    const id = String(a.modelId);
    const m = settings.models.find((x) => x.id === id);
    if (!m) throw new Error("unknown model");
    const signal = AbortSignal.timeout(120_000);
    if (m.kind === "embed") {
      const e = await router!.embedderFor(id);
      if (!e) throw new Error("embedder unavailable");
      const t0 = Date.now();
      const [v] = await e.embed(["hello world"], signal);
      if (m.dim && v && v.length !== m.dim) return { ok: false, detail: `server returned ${v.length}-dim vectors but the model is configured as ${m.dim}-dim` };
      return { ok: true, detail: `dim ${v?.length}, ${Date.now() - t0} ms` };
    }
    const chat = await router!.chatFor(id);
    const t0 = Date.now();
    let text = "";
    let first = 0;
    try {
      for await (const c of chat.stream({ messages: [{ role: "user", content: "Reply with the single word: ready" }], maxTokens: 64, enableThinking: false, temperature: 0, signal })) {
        if (c.type === "text") {
          if (!first) first = Date.now() - t0;
          text += c.text;
        }
      }
    } catch (e) {
      if (signal.aborted) throw new Error(`no complete reply within ${Math.round((Date.now() - t0) / 1000)} s${first ? ` (first token after ${first} ms)` : " (no tokens yet; the server may still be loading the model)"}`);
      throw e;
    }
    return { ok: true, detail: `"${text.trim().slice(0, 60)}" (first token ${first} ms, total ${Date.now() - t0} ms)` };
  },
};

parent.on("message", (e) => {
  const m = e.data;
  try {
    switch (m.kind) {
      case "init":
        init(m.init as Record<string, unknown>);
        break;
      case "settings":
        settings = Settings.parse(m.settings);
        keys = (m.keys as Record<string, string>) ?? keys;
        router?.invalidate();
        publishModel();
        break;
      case "host-response": {
        const p = hostPending.get(m.id as number);
        hostPending.delete(m.id as number);
        if (m.ok) p?.resolve(m.value);
        else p?.reject(new Error(String(m.error ?? "host request failed")));
        break;
      }
      case "editor-state":
        editorState = (m.state as Record<string, unknown>) ?? {};
        break;
      case "inference-port": {
        const port = e.ports[0];
        if (port) transport.attach(port);
        break;
      }
      case "inference-unavailable":
        transport.fail(String(m.reason));
        break;
      case "models-changed":
        transport.reset();
        router?.invalidate();
        publishModel();
        break;
      case "call": {
        const id = m.id as number;
        const fn = methods[String(m.method)];
        if (!fn) {
          post({ kind: "result", id, ok: false, error: `unknown method ${String(m.method)}` });
          break;
        }
        Promise.resolve()
          .then(() => fn((m.args as Record<string, unknown>) ?? {}))
          .then((value) => post({ kind: "result", id, ok: true, value: value ?? null }))
          .catch((err: unknown) => post({ kind: "result", id, ok: false, error: err instanceof Error ? err.message : String(err) }));
        break;
      }
    }
  } catch (err) {
    log.error("message handling failed", String(err));
  }
});

process.on("exit", () => pm.killAll());
for (const sig of ["SIGTERM", "SIGINT", "SIGHUP"] as const) process.on(sig, () => process.exit(0));
void newId;
