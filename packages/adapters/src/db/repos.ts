import { promises as fs } from "node:fs";
import path from "node:path";
import type { ChatSummary, PinnedFactView } from "@rlm/contracts";
import {
  cosine,
  newId,
  type ConversationRepo,
  type EditJournal,
  type MemoryFilter,
  type MemoryPoint,
  type MemoryVectorStore,
  type PendingEditRecord,
  type PinnedFactsRepo,
  type StoredTurn,
  type TranscriptSnapshot,
  type TranscriptSnapshotStore,
  type TaskBoard,
  type TaskBoardStore,
  type TurnMeta,
} from "@rlm/core";
import type { Db } from "./sqlite";

type Row = Record<string, unknown>;

export class SqliteConversationRepo implements ConversationRepo {
  constructor(private readonly db: Db) {}

  /** Turns left "running" by a crashed or killed process can never finish; mark them cancelled. */
  recoverInterrupted(): number {
    const r = this.db.raw.prepare("UPDATE turns SET status = 'cancelled' WHERE status = 'running'").run();
    return Number(r.changes);
  }

  async createChat(title: string, workspaceId: string | null): Promise<ChatSummary> {
    const now = Date.now();
    const c = { id: newId("chat"), title, createdAt: now, updatedAt: now, workspaceId };
    this.db.raw.prepare("INSERT INTO chats (id, title, created_at, updated_at, workspace_id) VALUES (?, ?, ?, ?, ?)").run(c.id, c.title, now, now, workspaceId);
    return c;
  }

  async listChats(workspaceId?: string | null): Promise<ChatSummary[]> {
    const rows =
      workspaceId === undefined
        ? this.db.raw.prepare("SELECT * FROM chats ORDER BY updated_at DESC").all()
        : workspaceId === null
          ? this.db.raw.prepare("SELECT * FROM chats WHERE workspace_id IS NULL ORDER BY updated_at DESC").all()
          : this.db.raw.prepare("SELECT * FROM chats WHERE workspace_id = ? ORDER BY updated_at DESC").all(workspaceId);
    return (rows as Row[]).map(toChat);
  }

  async getChat(chatId: string): Promise<ChatSummary | null> {
    const r = this.db.raw.prepare("SELECT * FROM chats WHERE id = ?").get(chatId) as Row | undefined;
    return r ? toChat(r) : null;
  }

  async setChatWorkspace(chatId: string, workspaceId: string): Promise<void> {
    this.db.raw.prepare("UPDATE chats SET workspace_id = ? WHERE id = ?").run(workspaceId, chatId);
  }

  async renameChat(chatId: string, title: string): Promise<void> {
    this.db.raw.prepare("UPDATE chats SET title = ? WHERE id = ?").run(title, chatId);
  }

  async deleteChat(chatId: string): Promise<void> {
    this.db.raw.prepare("DELETE FROM chats WHERE id = ?").run(chatId);
  }

  async getTurns(chatId: string): Promise<StoredTurn[]> {
    return (this.db.raw.prepare("SELECT id, chat_id, idx, user_text, assistant_text, status, created_at, meta FROM turns WHERE chat_id = ? ORDER BY idx").all(chatId) as Row[]).map(toTurn);
  }

  async appendTurn(turn: Omit<StoredTurn, "idx">): Promise<StoredTurn> {
    const row = this.db.raw.prepare("SELECT COALESCE(MAX(idx), -1) + 1 AS n FROM turns WHERE chat_id = ?").get(turn.chatId) as { n: number };
    const idx = Number(row.n);
    this.db.raw
      .prepare("INSERT INTO turns (id, chat_id, idx, user_text, assistant_text, status, created_at, meta) VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
      .run(turn.id, turn.chatId, idx, turn.userText, turn.assistantText, turn.status, turn.createdAt, JSON.stringify(turn.meta));
    this.db.raw.prepare("UPDATE chats SET updated_at = ? WHERE id = ?").run(Date.now(), turn.chatId);
    return { ...turn, idx };
  }

  async updateTurn(turnId: string, patch: Partial<Pick<StoredTurn, "assistantText" | "status">> & { meta?: TurnMeta; events?: unknown[] }): Promise<void> {
    const sets: string[] = [];
    const vals: Array<string | number> = [];
    if (patch.assistantText !== undefined) (sets.push("assistant_text = ?"), vals.push(patch.assistantText));
    if (patch.status !== undefined) (sets.push("status = ?"), vals.push(patch.status));
    if (patch.meta !== undefined) (sets.push("meta = ?"), vals.push(JSON.stringify(patch.meta)));
    if (patch.events !== undefined) (sets.push("events = ?"), vals.push(JSON.stringify(patch.events)));
    if (!sets.length) return;
    this.db.raw.prepare(`UPDATE turns SET ${sets.join(", ")} WHERE id = ?`).run(...vals, turnId);
  }

  async getTurnEvents(turnId: string): Promise<unknown[]> {
    const r = this.db.raw.prepare("SELECT events FROM turns WHERE id = ?").get(turnId) as Row | undefined;
    return r ? (JSON.parse(String(r.events)) as unknown[]) : [];
  }

  async setFeedback(turnId: string, value: "up" | "down" | null): Promise<void> {
    const r = this.db.raw.prepare("SELECT meta FROM turns WHERE id = ?").get(turnId) as Row | undefined;
    if (!r) return;
    const meta = JSON.parse(String(r.meta)) as TurnMeta;
    meta.feedback = value;
    this.db.raw.prepare("UPDATE turns SET meta = ? WHERE id = ?").run(JSON.stringify(meta), turnId);
  }

  /** Plain-text search over this chat's stored turns (for the session.read tool). */
  searchTurns(chatId: string, query: string, limit: number): Array<{ turn: number; role: string; excerpt: string }> {
    const words = query.toLowerCase().split(/\s+/).filter(Boolean);
    const out: Array<{ turn: number; role: string; excerpt: string; score: number }> = [];
    for (const r of this.db.raw.prepare("SELECT idx, user_text, assistant_text FROM turns WHERE chat_id = ? ORDER BY idx").all(chatId) as Row[]) {
      for (const [role, text] of [["user", String(r.user_text)], ["assistant", String(r.assistant_text)]] as const) {
        const lower = text.toLowerCase();
        const score = words.filter((w) => lower.includes(w)).length;
        if (!score) continue;
        const pos = Math.max(0, lower.indexOf(words.find((w) => lower.includes(w))!) - 200);
        out.push({ turn: Number(r.idx), role, excerpt: text.slice(pos, pos + 600), score });
      }
    }
    return out.sort((a, b) => b.score - a.score).slice(0, limit).map(({ score: _s, ...rest }) => rest);
  }
}

function toChat(r: Row): ChatSummary {
  return {
    id: String(r.id),
    title: String(r.title),
    createdAt: Number(r.created_at),
    updatedAt: Number(r.updated_at),
    workspaceId: r.workspace_id == null ? null : String(r.workspace_id),
  };
}

function toTurn(r: Row): StoredTurn {
  return {
    id: String(r.id),
    chatId: String(r.chat_id),
    idx: Number(r.idx),
    userText: String(r.user_text),
    assistantText: String(r.assistant_text),
    status: String(r.status) as StoredTurn["status"],
    createdAt: Number(r.created_at),
    meta: JSON.parse(String(r.meta)) as TurnMeta,
  };
}

export class SqlitePinnedFacts implements PinnedFactsRepo {
  constructor(private readonly db: Db) {}
  async list(chatId: string): Promise<PinnedFactView[]> {
    return (this.db.raw.prepare("SELECT * FROM pinned_facts WHERE chat_id = ? ORDER BY CASE kind WHEN 'goal' THEN 0 WHEN 'constraint' THEN 1 ELSE 2 END, created_at").all(chatId) as Row[]).map((r) => ({
      id: String(r.id),
      chatId: String(r.chat_id),
      kind: String(r.kind) as PinnedFactView["kind"],
      text: String(r.text),
    }));
  }
  async upsert(f: Omit<PinnedFactView, "id"> & { id?: string }): Promise<PinnedFactView> {
    const id = f.id ?? newId("pin");
    this.db.raw
      .prepare("INSERT INTO pinned_facts (id, chat_id, kind, text, created_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET kind = excluded.kind, text = excluded.text")
      .run(id, f.chatId, f.kind, f.text, Date.now());
    return { id, chatId: f.chatId, kind: f.kind, text: f.text };
  }
  async delete(id: string): Promise<void> {
    this.db.raw.prepare("DELETE FROM pinned_facts WHERE id = ?").run(id);
  }
}

export class SqliteEditJournal implements EditJournal {
  constructor(private readonly db: Db) {}
  async record(e: Omit<PendingEditRecord, "state">): Promise<void> {
    this.db.raw
      .prepare("INSERT INTO edits (edit_id, turn_id, path, before_text, after_text, created_at, state) VALUES (?, ?, ?, ?, ?, ?, 'applied')")
      .run(e.editId, e.turnId, e.path, e.before, e.after, e.createdAt);
  }
  async list(workspaceId?: string): Promise<PendingEditRecord[]> {
    const rows =
      workspaceId === undefined
        ? this.db.raw.prepare("SELECT * FROM edits WHERE state = 'applied' ORDER BY created_at DESC LIMIT 200").all()
        : this.db.raw
            .prepare(
              `SELECT e.* FROM edits e JOIN turns t ON t.id = e.turn_id JOIN chats c ON c.id = t.chat_id
               WHERE e.state = 'applied' AND c.workspace_id = ? ORDER BY e.created_at DESC LIMIT 200`,
            )
            .all(workspaceId);
    return (rows as Row[]).map(toEdit);
  }
  async setState(editId: string, state: PendingEditRecord["state"]): Promise<void> {
    this.db.raw.prepare("UPDATE edits SET state = ? WHERE edit_id = ?").run(state, editId);
  }
  async get(editId: string): Promise<PendingEditRecord | null> {
    const r = this.db.raw.prepare("SELECT * FROM edits WHERE edit_id = ?").get(editId) as Row | undefined;
    return r ? toEdit(r) : null;
  }
}

function toEdit(r: Row): PendingEditRecord {
  return {
    editId: String(r.edit_id),
    turnId: String(r.turn_id),
    path: String(r.path),
    before: String(r.before_text),
    after: String(r.after_text),
    createdAt: Number(r.created_at),
    state: String(r.state) as PendingEditRecord["state"],
  };
}

/** Prefix for vector tables; memory operations only ever touch tables with this prefix. */
export const MEMORY_TABLE_PREFIX = "orchestrator_memory";

export class SqliteMemoryStore implements MemoryVectorStore {
  private readonly ensured = new Set<string>();
  constructor(private readonly db: Db) {}

  static tableFor(modelKey: string): string {
    return `${MEMORY_TABLE_PREFIX}__${modelKey.replace(/[^A-Za-z0-9_]/g, "_")}`;
  }

  collections(): string[] {
    return (this.db.raw.prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE ?").all(`${MEMORY_TABLE_PREFIX}__%`) as Row[]).map((r) => String(r.name));
  }

  private ensure(modelKey: string): string {
    const t = SqliteMemoryStore.tableFor(modelKey);
    if (!this.ensured.has(t)) {
      this.db.raw.exec(`CREATE TABLE IF NOT EXISTS "${t}" (id TEXT PRIMARY KEY REFERENCES memory(id) ON DELETE CASCADE, embedding BLOB NOT NULL)`);
      this.ensured.add(t);
    }
    return t;
  }

  private where(f: MemoryFilter, alias = "m"): { sql: string; vals: Array<string | null> } {
    const parts = [`${alias}.user_id = ?`];
    const vals: Array<string | null> = [f.user_id];
    if (f.memory_type) (parts.push(`${alias}.memory_type = ?`), vals.push(f.memory_type));
    if (f.kind) (parts.push(`${alias}.kind = ?`), vals.push(f.kind));
    if (f.session_id !== undefined) {
      if (f.session_id === null) parts.push(`${alias}.session_id IS NULL`);
      else (parts.push(`${alias}.session_id = ?`), vals.push(f.session_id));
    }
    if (f.workspace_id !== undefined) {
      if (f.workspace_id === null) parts.push(`${alias}.workspace_id IS NULL`);
      else if (f.include_global) (parts.push(`(${alias}.workspace_id = ? OR ${alias}.workspace_id IS NULL)`), vals.push(f.workspace_id));
      else (parts.push(`${alias}.workspace_id = ?`), vals.push(f.workspace_id));
    }
    return { sql: parts.join(" AND "), vals };
  }

  async upsert(modelKey: string, points: MemoryPoint[]): Promise<void> {
    const raw = this.db.raw;
    raw.exec("BEGIN");
    try {
      const ins = raw.prepare(
        `INSERT INTO memory (id, user_id, session_id, memory_type, kind, category, workspace_id, text, created_at, updated_at, history)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET text = excluded.text, category = excluded.category, updated_at = excluded.updated_at, history = excluded.history`,
      );
      for (const p of points) {
        const q = p.payload;
        ins.run(p.id, q.user_id, q.session_id, q.memory_type, q.kind, q.category, q.workspace_id, q.text, q.created_at, q.updated_at, JSON.stringify(q.history));
        if (p.vector && modelKey) {
          const t = this.ensure(modelKey);
          raw.prepare(`INSERT INTO "${t}" (id, embedding) VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET embedding = excluded.embedding`).run(p.id, toBlob(p.vector));
        }
      }
      raw.exec("COMMIT");
    } catch (e) {
      raw.exec("ROLLBACK");
      throw e;
    }
  }

  async search(modelKey: string, vector: number[], filter: MemoryFilter, limit: number): Promise<Array<MemoryPoint & { score: number }>> {
    const t = this.ensure(modelKey);
    const w = this.where(filter);
    if (this.db.vecLoaded) {
      const rows = this.db.raw
        .prepare(`SELECT m.*, v.embedding AS emb, 1 - vec_distance_cosine(v.embedding, ?) AS score FROM memory m JOIN "${t}" v ON v.id = m.id WHERE ${w.sql} ORDER BY score DESC LIMIT ?`)
        .all(toBlob(vector), ...w.vals, limit) as Row[];
      return rows.map((r) => ({ ...toPoint(r), score: Number(r.score) }));
    }
    const rows = this.db.raw.prepare(`SELECT m.*, v.embedding AS emb FROM memory m JOIN "${t}" v ON v.id = m.id WHERE ${w.sql}`).all(...w.vals) as Row[];
    return rows
      .map((r) => ({ ...toPoint(r), score: cosine(vector, fromBlob(r.emb as Uint8Array)) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  async list(filter: MemoryFilter, textQuery?: string): Promise<MemoryPoint[]> {
    const w = this.where(filter);
    let sql = `SELECT m.* FROM memory m WHERE ${w.sql}`;
    const vals: Array<string | null> = [...w.vals];
    if (textQuery) (sql += " AND m.text LIKE ?"), vals.push(`%${textQuery}%`);
    sql += " ORDER BY m.updated_at DESC LIMIT 1000";
    return (this.db.raw.prepare(sql).all(...vals) as Row[]).map(toPoint);
  }

  async get(id: string, filter: MemoryFilter): Promise<MemoryPoint | null> {
    const w = this.where(filter);
    const r = this.db.raw.prepare(`SELECT m.* FROM memory m WHERE m.id = ? AND ${w.sql}`).get(id, ...w.vals) as Row | undefined;
    if (!r) return null;
    const p = toPoint(r);
    for (const t of this.collections()) {
      const v = this.db.raw.prepare(`SELECT embedding FROM "${t}" WHERE id = ?`).get(id) as Row | undefined;
      if (v) {
        p.vector = fromBlob(v.embedding as Uint8Array);
        break;
      }
    }
    return p;
  }

  async delete(ids: string[], filter: MemoryFilter): Promise<void> {
    const w = this.where(filter, "memory");
    for (const id of ids) this.db.raw.prepare(`DELETE FROM memory WHERE id = ? AND ${w.sql}`).run(id, ...w.vals);
  }

  async deleteAll(filter: MemoryFilter): Promise<void> {
    const w = this.where(filter, "memory");
    this.db.raw.prepare(`DELETE FROM memory WHERE ${w.sql}`).run(...w.vals);
  }
}

function toPoint(r: Row): MemoryPoint {
  return {
    id: String(r.id),
    vector: null,
    payload: {
      user_id: String(r.user_id),
      session_id: r.session_id == null ? null : String(r.session_id),
      memory_type: String(r.memory_type) as "short_term" | "long_term",
      kind: String(r.kind) as "chunk" | "fact",
      category: String(r.category),
      workspace_id: r.workspace_id == null ? null : String(r.workspace_id),
      text: String(r.text),
      created_at: Number(r.created_at),
      updated_at: Number(r.updated_at),
      history: JSON.parse(String(r.history ?? "[]")) as string[],
    },
  };
}

function toBlob(v: number[]): Uint8Array {
  return new Uint8Array(new Float32Array(v).buffer);
}

function fromBlob(b: Uint8Array): number[] {
  return Array.from(new Float32Array(b.buffer, b.byteOffset, Math.floor(b.byteLength / 4)));
}

/** One JSON file per chat board. */
export class FileTaskBoardStore implements TaskBoardStore {
  constructor(private readonly dir: string) {}
  private file(chatId: string): string {
    return path.join(this.dir, `${chatId.replace(/[^A-Za-z0-9_-]/g, "_")}.json`);
  }
  async load(chatId: string): Promise<TaskBoard | null> {
    try {
      return JSON.parse(await fs.readFile(this.file(chatId), "utf8")) as TaskBoard;
    } catch {
      return null;
    }
  }
  async save(b: TaskBoard): Promise<void> {
    await fs.mkdir(this.dir, { recursive: true });
    const tmp = `${this.file(b.chatId)}.${process.pid}.tmp`;
    await fs.writeFile(tmp, JSON.stringify(b));
    await fs.rename(tmp, this.file(b.chatId));
  }
  async delete(chatId: string): Promise<void> {
    await fs.rm(this.file(chatId), { force: true });
  }
  async list(): Promise<TaskBoard[]> {
    const names = await fs.readdir(this.dir).catch(() => [] as string[]);
    const out: TaskBoard[] = [];
    for (const n of names) {
      if (!n.endsWith(".json")) continue;
      try {
        out.push(JSON.parse(await fs.readFile(path.join(this.dir, n), "utf8")) as TaskBoard);
      } catch {
        /* skip a corrupt board */
      }
    }
    return out;
  }
}

export class FileSnapshotStore implements TranscriptSnapshotStore {
  constructor(private readonly dir: string) {}
  private file(chatId: string): string {
    return path.join(this.dir, `${chatId.replace(/[^A-Za-z0-9_-]/g, "_")}.json`);
  }
  async load(chatId: string): Promise<TranscriptSnapshot | null> {
    try {
      return JSON.parse(await fs.readFile(this.file(chatId), "utf8")) as TranscriptSnapshot;
    } catch {
      return null;
    }
  }
  async save(s: TranscriptSnapshot): Promise<void> {
    await fs.mkdir(this.dir, { recursive: true });
    const tmp = `${this.file(s.chatId)}.tmp`;
    await fs.writeFile(tmp, JSON.stringify(s));
    await fs.rename(tmp, this.file(s.chatId));
  }
  async delete(chatId: string): Promise<void> {
    await fs.rm(this.file(chatId), { force: true });
  }
}
