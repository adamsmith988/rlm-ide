import { create } from "zustand";
import type { BrowserState, ChatSummary, OrchestratorStatus, Settings, StoredMessage, TaskBoardView, TurnEvent } from "@rlm/contracts";
import { call, errText } from "./api";
import { applyEvent, type ContextUsage, type UiMessage } from "./chat/model";

/** Last segment of an absolute folder path (Windows or POSIX separators). */
export const baseName = (p: string): string => p.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || p;

export type Activity = "explorer" | "search" | "git" | "chats" | "team" | "hub" | "memory" | "settings";

export interface Tab {
  id: string;
  path: string;
  title: string;
  kind: "file" | "diff" | "settings" | "memory";
  content: string;
  saved: string;
  binary?: boolean;
  diff?: { before: string; after: string; editId?: string };
}

/** Something from the browser the user attached to the next message. */
export type Attachment =
  | { id: string; kind: "screenshot"; image: string; pageUrl: string; title: string; width: number; height: number }
  | { id: string; kind: "element"; image: string | null; pageUrl: string; title: string; selector: string; tag: string; text: string; html: string };

interface Toast {
  id: number;
  text: string;
  tone: "info" | "error";
}

export interface AppState {
  settings: Settings | null;
  workspace: { root: string; name: string } | null;
  activity: Activity;
  sidebar: boolean;
  bottom: boolean;
  agent: boolean;
  bottomTab: "terminal" | "problems" | "output";
  tabs: Tab[];
  activeTab: string | null;
  selection: string;
  cursorLine: number;
  diagnostics: Array<{ path: string; line: number; severity: string; message: string }>;
  chats: ChatSummary[];
  chatId: string | null;
  messages: Record<string, UiMessage[]>;
  context: Record<string, ContextUsage>;
  boards: Record<string, TaskBoardView | null>;
  status: OrchestratorStatus & { inference?: { device: string; status: string; detail: string } };
  git: { branch: string | null; dirty: number };
  palette: null | "commands" | "files";
  toasts: Toast[];
  treeVersion: number;
  browser: boolean;
  browserState: BrowserState;
  attachments: Attachment[];

  set(p: Partial<AppState>): void;
  openInBrowser(url: string): Promise<void>;
  attach(a: Attachment): void;
  detach(id: string): void;
  toast(text: string, tone?: "info" | "error"): void;
  loadSettings(): Promise<void>;
  updateSettings(patch: Record<string, unknown>): Promise<void>;
  openWorkspace(root: string): Promise<void>;
  /** Folder picker, then opens the folder here (or in another window with `newWindow`). */
  pickFolder(opts?: { newWindow?: boolean }): Promise<void>;
  /** Starts another IDE window, on `root` if given. */
  newWindow(root?: string): Promise<void>;
  openFile(path: string, opts?: { line?: number }): Promise<void>;
  openDiff(path: string, before: string, after: string, editId?: string): void;
  closeTab(id: string): void;
  saveTab(id: string): Promise<void>;
  setTabContent(id: string, content: string): void;
  refreshChats(): Promise<void>;
  selectChat(id: string | null): Promise<void>;
  newChat(): Promise<string>;
  /** Resolves false when the message could not be sent (the composer keeps the text). */
  sendMessage(text: string, mentions: string[]): Promise<boolean>;
  stopTurn(): Promise<void>;
  onTurnEvent(ev: TurnEvent): void;
  onTaskBoard(b: TaskBoardView): void;
  loadBoard(chatId: string): Promise<void>;
  refreshGit(): Promise<void>;
}

let toastId = 0;

/** Attachments travel inside the message text so the model reads them and the transcript shows them. */
function attachmentMarkdown(a: Attachment): string {
  const page = `[${(a.title || a.pageUrl).replace(/[[\]]/g, "")}](${a.pageUrl})`;
  if (a.kind === "screenshot") return `**Browser screenshot** of ${page} (${a.width}×${a.height})\n\n![screenshot](${a.image})`;
  const fence = a.html.includes("```") ? "~~~" : "```";
  return [
    `**Referenced element** \`${a.selector}\` on ${page}`,
    a.text ? `Text: "${a.text.slice(0, 200)}"` : "",
    a.image ? `![element](${a.image})` : "",
    `${fence}html\n${a.html}\n${fence}`,
  ]
    .filter(Boolean)
    .join("\n\n");
}
export const pendingLine: { path: string; line: number } | { path: ""; line: 0 } = { path: "", line: 0 };

function toUi(m: StoredMessage): UiMessage {
  let ui: UiMessage = { id: m.id, turnId: m.turnId, role: m.role, origin: m.origin, text: m.text, status: m.status, blocks: [], lastSeq: -1, feedback: m.feedback ?? null };
  if (m.role === "assistant" && m.events) for (const ev of m.events) ui = applyEvent(ui, ev);
  if (m.role === "assistant") {
    ui.status = m.status;
    if (!ui.blocks.some((b) => b.kind === "text") && m.text) ui.blocks.push({ kind: "text", text: m.text });
  }
  return ui;
}

export const useApp = create<AppState>((set, get) => ({
  settings: null,
  workspace: null,
  activity: "explorer",
  sidebar: true,
  bottom: true,
  agent: true,
  bottomTab: "terminal",
  tabs: [],
  activeTab: null,
  selection: "",
  cursorLine: 1,
  diagnostics: [],
  chats: [],
  chatId: null,
  messages: {},
  context: {},
  boards: {},
  status: { memory: "online", kernel: "idle", model: "", contextLength: null },
  git: { branch: null, dirty: 0 },
  palette: null,
  toasts: [],
  treeVersion: 0,
  browser: false,
  browserState: { open: false, url: "", title: "", loading: false, canGoBack: false, canGoForward: false, devtools: false, picking: false, agent: null, error: null },
  attachments: [],

  set: (p) => set(p),

  async openInBrowser(url) {
    set({ browser: true });
    try {
      await call("browser:navigate", { url });
    } catch (e) {
      get().toast(errText(e), "error");
    }
  },

  attach(a) {
    set({ attachments: [...get().attachments, a], agent: true });
    setTimeout(() => document.getElementById("composer")?.focus(), 30);
  },

  detach(id) {
    set({ attachments: get().attachments.filter((a) => a.id !== id) });
  },

  toast(text, tone = "info") {
    const id = ++toastId;
    set({ toasts: [...get().toasts, { id, text, tone }] });
    setTimeout(() => set({ toasts: get().toasts.filter((t) => t.id !== id) }), tone === "error" ? 7000 : 3500);
  },

  async loadSettings() {
    const s = await call<Settings>("settings:get", {});
    set({ settings: s });
    document.documentElement.dataset.theme = s.theme;
  },

  async updateSettings(patch) {
    const s = await call<Settings>("settings:update", { patch });
    set({ settings: s });
    document.documentElement.dataset.theme = s.theme;
  },

  async openWorkspace(root) {
    try {
      const ws = await call<{ root: string; name: string }>("workspace:open", { root });
      set({ workspace: ws, tabs: [], activeTab: null, activity: "explorer", treeVersion: get().treeVersion + 1 });
      await get().loadSettings();
      await get().refreshChats();
      await get().refreshGit();
      // Chats are per project: the list now holds only this folder's chats, and none may stay selected from another.
      await get().selectChat(get().chats[0]?.id ?? null);
    } catch (e) {
      get().toast(errText(e), "error");
    }
  },

  async pickFolder(opts) {
    try {
      const root = await call<string | null>("dialog:openFolder", {});
      if (!root) return;
      if (opts?.newWindow) await get().newWindow(root);
      else await get().openWorkspace(root);
    } catch (e) {
      get().toast(errText(e), "error");
    }
  },

  async newWindow(root) {
    try {
      await call<boolean>("app:newWindow", root ? { root } : {});
      get().toast(root ? `Opening ${baseName(root)} in a new window…` : "Opening a new window…");
    } catch (e) {
      get().toast(errText(e), "error");
    }
  },

  async openFile(path, opts) {
    const existing = get().tabs.find((t) => t.kind === "file" && t.path === path);
    if (opts?.line) Object.assign(pendingLine, { path, line: opts.line });
    if (existing) {
      set({ activeTab: existing.id });
      return;
    }
    try {
      const r = await call<{ content: string; binary: boolean }>("fs:readFile", { path });
      const tab: Tab = { id: `file:${path}`, path, title: path.split("/").pop() ?? path, kind: "file", content: r.content, saved: r.content, binary: r.binary };
      set({ tabs: [...get().tabs, tab], activeTab: tab.id });
    } catch (e) {
      get().toast(errText(e), "error");
    }
  },

  openDiff(path, before, after, editId) {
    const id = `diff:${editId ?? path}`;
    const tabs = get().tabs.filter((t) => t.id !== id);
    const tab: Tab = { id, path, title: `${path.split("/").pop()} (AI edit)`, kind: "diff", content: after, saved: after, diff: { before, after, editId } };
    set({ tabs: [...tabs, tab], activeTab: id });
  },

  closeTab(id) {
    const tabs = get().tabs;
    const i = tabs.findIndex((t) => t.id === id);
    const next = tabs.filter((t) => t.id !== id);
    set({ tabs: next, activeTab: get().activeTab === id ? (next[Math.min(i, next.length - 1)]?.id ?? null) : get().activeTab });
  },

  async saveTab(id) {
    const t = get().tabs.find((x) => x.id === id);
    if (!t || t.kind !== "file") return;
    try {
      await call("fs:writeFile", { path: t.path, content: t.content });
      set({ tabs: get().tabs.map((x) => (x.id === id ? { ...x, saved: x.content } : x)) });
      void get().refreshGit();
    } catch (e) {
      get().toast(errText(e), "error");
    }
  },

  setTabContent(id, content) {
    set({ tabs: get().tabs.map((t) => (t.id === id ? { ...t, content } : t)) });
  },

  async refreshChats() {
    const chats = await call<ChatSummary[]>("chat:list", {});
    set({ chats });
  },

  async selectChat(id) {
    set({ chatId: id });
    if (!id) return;
    void get().loadBoard(id);
    const msgs = await call<StoredMessage[]>("chat:messages", { chatId: id });
    set({ messages: { ...get().messages, [id]: msgs.map(toUi) } });
    const active = await call<string | null>("chat:activeTurn", { chatId: id });
    if (active) {
      const list = get().messages[id] ?? [];
      const msg = list.find((m) => m.turnId === active && m.role === "assistant");
      const replay = await call<TurnEvent[]>("chat:subscribe", { turnId: active, lastSeq: msg?.lastSeq ?? -1 });
      for (const ev of replay) get().onTurnEvent(ev);
    } else {
      const list = get().messages[id] ?? [];
      if (list.some((m) => m.status === "running")) {
        set({ messages: { ...get().messages, [id]: list.map((m) => (m.status === "running" ? { ...m, status: "cancelled" } : m)) } });
      }
    }
  },

  async newChat() {
    const c = await call<ChatSummary>("chat:create", { title: "New chat" });
    await get().refreshChats();
    set({ chatId: c.id, messages: { ...get().messages, [c.id]: [] }, agent: true });
    return c.id;
  },

  async sendMessage(typed, mentions) {
    const tab = get().tabs.find((t) => t.id === get().activeTab);
    const attachments = get().attachments;
    const text = attachments.length ? `${typed || "See the attached browser context."}\n\n${attachments.map(attachmentMarkdown).join("\n\n")}` : typed;
    try {
      const chatId = get().chatId ?? (await get().newChat());
      const { turnId } = await call<{ turnId: string }>("chat:send", {
        chatId,
        text,
        editorContext: {
          activeFile: tab?.kind === "file" ? tab.path : undefined,
          selection: get().selection || undefined,
          openFiles: get().tabs.filter((t) => t.kind === "file").map((t) => t.path),
          mentions,
        },
      });
      const list = get().messages[chatId] ?? [];
      const user: UiMessage = { id: `${turnId}:u`, turnId, role: "user", text, status: "done", blocks: [], lastSeq: -1 };
      const assistant: UiMessage = { id: `${turnId}:a`, turnId, role: "assistant", text: "", status: "running", blocks: [], lastSeq: -1, startedAt: Date.now() };
      const existing = list.filter((m) => m.turnId !== turnId);
      const buffered = list.find((m) => m.turnId === turnId && m.role === "assistant");
      set({ messages: { ...get().messages, [chatId]: [...existing, user, buffered ?? assistant] }, attachments: get().attachments.filter((a) => !attachments.includes(a)) });
      void get().refreshChats();
      return true;
    } catch (e) {
      get().toast(`Could not send: ${errText(e)}`, "error");
      return false;
    }
  },

  async stopTurn() {
    const chatId = get().chatId;
    if (!chatId) return;
    const running = (get().messages[chatId] ?? []).find((m) => m.role === "assistant" && m.status === "running");
    if (running) await call("chat:stop", { turnId: running.turnId });
  },

  onTurnEvent(ev) {
    const msgs = get().messages;
    for (const [chatId, list] of Object.entries(msgs)) {
      const i = list.findIndex((m) => m.turnId === ev.turnId && m.role === "assistant");
      if (i < 0) continue;
      const next = [...list];
      next[i] = applyEvent(next[i]!, ev);
      const patch: Partial<AppState> = { messages: { ...msgs, [chatId]: next } };
      if (ev.type === "context_usage") patch.context = { ...get().context, [chatId]: ev };
      set(patch);
      if (ev.type === "pending_edit") {
        const t = get().tabs.find((x) => x.kind === "file" && x.path === ev.path);
        if (t && t.content === t.saved) set({ tabs: get().tabs.map((x) => (x.id === t.id ? { ...x, content: ev.after, saved: ev.after } : x)) });
        set({ treeVersion: get().treeVersion + 1 });
      }
      if (ev.type === "turn_done" || ev.type === "turn_error" || ev.type === "turn_cancelled") void get().refreshGit();
      return;
    }
    if (ev.type === "turn_started" && msgs[ev.chatId]) {
      // A turn this window did not send (a scheduled wake-up, another client): show its message too.
      const list = msgs[ev.chatId]!;
      const user: UiMessage[] = list.some((m) => m.turnId === ev.turnId && m.role === "user")
        ? []
        : [{ id: `${ev.turnId}:u`, turnId: ev.turnId, role: "user", origin: ev.origin, text: ev.userText, status: "done", blocks: [], lastSeq: -1 }];
      const assistant = applyEvent({ id: `${ev.turnId}:a`, turnId: ev.turnId, role: "assistant", text: "", status: "running", blocks: [], lastSeq: -1, startedAt: Date.now() }, ev);
      set({ messages: { ...msgs, [ev.chatId]: [...list, ...user, assistant] } });
    }
    if (ev.type === "turn_started" && ev.origin === "wakeup" && ev.chatId !== get().chatId) {
      get().toast("A scheduled wake-up started a turn in another chat");
    }
  },

  onTaskBoard(b) {
    set({ boards: { ...get().boards, [b.chatId]: b } });
  },

  async loadBoard(chatId) {
    const b = await call<TaskBoardView | null>("task:board", { chatId }).catch(() => null);
    set({ boards: { ...get().boards, [chatId]: b } });
  },

  async refreshGit() {
    try {
      set({ git: await call("git:info", {}) });
    } catch {
      /* not a repo */
    }
  },
}));
