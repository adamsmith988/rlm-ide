/** Ports: the only way the core talks to the outside world. Adapters implement these. */
import type { StoredTurn, TurnMeta } from "../domain/conversation";
import type { ToolDescriptor } from "../domain/tools";
import type { TaskBoard } from "../domain/task_board";
import type { Settings, ChatSummary, PinnedFactView } from "@rlm/contracts";

// ---------- LLM ----------

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  maxTokens: number;
  temperature?: number;
  stop?: string[];
  signal?: AbortSignal;
  enableThinking?: boolean;
}

export type ChatChunk =
  | { type: "thinking"; text: string }
  | { type: "text"; text: string }
  | { type: "usage"; inputTokens: number; outputTokens: number };

export interface ChatModelPort {
  readonly modelId: string;
  stream(req: ChatRequest): AsyncIterable<ChatChunk>;
}

/** Thrown by chat adapters when the provider rejects a request for length. */
export class ContextOverflowError extends Error {
  constructor(message = "context length exceeded") {
    super(message);
    this.name = "ContextOverflowError";
  }
}

export interface EmbeddingPort {
  /** `<model>__<dim>`; used to pick the vector table so vector spaces never mix. */
  readonly modelKey: string;
  readonly dim: number;
  embed(texts: string[], signal?: AbortSignal): Promise<number[][]>;
}

export interface ModelLimitsInfo {
  contextLength: number | null;
  maxOutputTokens: number | null;
}

export interface ModelCatalogPort {
  limits(modelId: string): Promise<ModelLimitsInfo>;
}

/** Resolves the adapter for a role at turn time (settings may change between turns). */
export interface ModelRouterPort {
  chat(role: "agent" | "summarizer"): Promise<ChatModelPort>;
  embedder(): Promise<EmbeddingPort | null>;
  modelIdFor(role: "agent" | "summarizer" | "embed"): string;
}

// ---------- persistence ----------

export interface ConversationRepo {
  createChat(title: string): Promise<ChatSummary>;
  listChats(): Promise<ChatSummary[]>;
  renameChat(chatId: string, title: string): Promise<void>;
  deleteChat(chatId: string): Promise<void>;
  getTurns(chatId: string): Promise<StoredTurn[]>;
  appendTurn(turn: Omit<StoredTurn, "idx">): Promise<StoredTurn>;
  updateTurn(turnId: string, patch: Partial<Pick<StoredTurn, "assistantText" | "status">> & { meta?: TurnMeta; events?: unknown[] }): Promise<void>;
  getTurnEvents(turnId: string): Promise<unknown[]>;
  setFeedback(turnId: string, value: "up" | "down" | null): Promise<void>;
}

export interface MemoryPayload {
  user_id: string;
  session_id: string | null;
  memory_type: "short_term" | "long_term";
  kind: "chunk" | "fact";
  category: string;
  workspace_id: string | null;
  text: string;
  created_at: number;
  updated_at: number;
  /** Prior versions of a long-term fact (newest first). */
  history: string[];
}

export interface MemoryPoint {
  id: string;
  vector: number[] | null;
  payload: MemoryPayload;
}

/** Every operation is filtered. Required fields guarantee user isolation. */
export interface MemoryFilter {
  user_id: string;
  memory_type?: "short_term" | "long_term";
  session_id?: string | null;
  workspace_id?: string | null;
  kind?: "chunk" | "fact";
}

export interface MemoryVectorStore {
  upsert(modelKey: string, points: MemoryPoint[]): Promise<void>;
  search(modelKey: string, vector: number[], filter: MemoryFilter, limit: number): Promise<Array<MemoryPoint & { score: number }>>;
  list(filter: MemoryFilter, textQuery?: string): Promise<MemoryPoint[]>;
  get(id: string, filter: MemoryFilter): Promise<MemoryPoint | null>;
  delete(ids: string[], filter: MemoryFilter): Promise<void>;
  deleteAll(filter: MemoryFilter): Promise<void>;
}

export interface PinnedFactsRepo {
  list(chatId: string): Promise<PinnedFactView[]>;
  upsert(f: Omit<PinnedFactView, "id"> & { id?: string }): Promise<PinnedFactView>;
  delete(id: string): Promise<void>;
}

export interface TranscriptSnapshot {
  chatId: string;
  /** Last stored turn idx the transcript covers. */
  throughTurnIdx: number;
  entries: TranscriptEntry[];
  savedAt: number;
}

export type TranscriptEntry =
  | { kind: "checkpoint"; text: string }
  | { kind: "user"; text: string; turnIdx: number }
  | { kind: "assistant"; text: string; turnIdx: number }
  | { kind: "repl"; text: string; turnIdx: number }
  | { kind: "note"; text: string; turnIdx: number };

export interface TranscriptSnapshotStore {
  load(chatId: string): Promise<TranscriptSnapshot | null>;
  save(s: TranscriptSnapshot): Promise<void>;
  delete(chatId: string): Promise<void>;
}

export interface TaskBoardStore {
  load(chatId: string): Promise<TaskBoard | null>;
  save(b: TaskBoard): Promise<void>;
  delete(chatId: string): Promise<void>;
  list(): Promise<TaskBoard[]>;
}

export interface PendingEditRecord {
  editId: string;
  turnId: string;
  path: string;
  before: string;
  after: string;
  createdAt: number;
  state: "applied" | "reverted";
}

export interface EditJournal {
  record(e: Omit<PendingEditRecord, "state">): Promise<void>;
  list(): Promise<PendingEditRecord[]>;
  setState(editId: string, state: PendingEditRecord["state"]): Promise<void>;
  get(editId: string): Promise<PendingEditRecord | null>;
}

// ---------- kernel ----------

export interface CellOutput {
  stream: "stdout" | "stderr" | "result";
  text: string;
}

export interface CellResult {
  ok: boolean;
  stdout: string;
  stderr: string;
  result: string;
  error: string | null;
  timedOut: boolean;
  interrupted: boolean;
}

export type HostRequestHandler = (type: string, payload: Record<string, unknown>) => Promise<unknown>;

export interface KernelSession {
  readonly id: string;
  execute(code: string, onOutput: (o: CellOutput) => void, timeoutMs: number): Promise<CellResult>;
  interrupt(): Promise<void>;
  reset(): Promise<void>;
  shutdown(): Promise<void>;
  isAlive(): boolean;
}

export interface KernelFactory {
  create(opts: {
    cwd: string;
    tools: ToolDescriptor[];
    onHostRequest: HostRequestHandler;
    outputLimitChars: number;
  }): Promise<KernelSession>;
  sandboxInfo(): { mode: "bwrap" | "landlock" | "jobobject" | "none"; detail: string };
}

// ---------- tools ----------

export interface ToolContext {
  workspaceRoot: string;
  turnId: string;
  chatId: string;
  signal: AbortSignal;
  /** Reaches the model in the cell output even when it does not print the tool's return value. */
  notice?: (text: string) => void;
}

export interface ToolImpl {
  descriptor: ToolDescriptor;
  run(args: Record<string, unknown>, ctx: ToolContext): Promise<unknown>;
  /** For file-editing tools: compute the resulting content without writing (for diff previews). */
  preview?(args: Record<string, unknown>, ctx: ToolContext): Promise<{ path: string; before: string; after: string }>;
  /** Runs before the approval prompt; a returned string refuses the call with that message. */
  precheck?(args: Record<string, unknown>, ctx: ToolContext): Promise<string | null>;
}

export interface ToolRegistryPort {
  list(): ToolDescriptor[];
  get(id: string): ToolImpl | undefined;
}

// ---------- misc ----------

export interface PromptsPort {
  /** Loads a prompt `.md` by relative name, e.g. `rlm/system.md`. */
  get(name: string): string;
}

export interface ClockPort {
  now(): number;
}

export interface SettingsPort {
  get(): Settings;
}

export interface WorkspacePort {
  root(): string | null;
  /** Short git summary for the editor context block. */
  gitSummary(): Promise<string>;
}

export interface LoggerPort {
  info(msg: string, data?: unknown): void;
  warn(msg: string, data?: unknown): void;
  error(msg: string, data?: unknown): void;
}

export interface StatusPort {
  memory(state: "online" | "offline" | "disabled"): void;
  kernel(state: "idle" | "busy" | "down"): void;
}

export function render(template: string, vars: Record<string, string>): string {
  return template.replace(/\{\{(\w+)\}\}/g, (_, k: string) => vars[k] ?? "");
}
