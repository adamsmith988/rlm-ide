import type { MemorySettings, RecalledMemory } from "@rlm/contracts";
import { chunkText } from "../domain/chunking";
import { clipChars } from "../domain/tokens";
import type {
  ChatModelPort,
  ClockPort,
  EmbeddingPort,
  LoggerPort,
  MemoryFilter,
  MemoryPoint,
  MemoryVectorStore,
  PromptsPort,
  StatusPort,
} from "../ports";
import { render } from "../ports";

export interface MemoryScope {
  userId: string;
  sessionId: string;
  workspaceId: string | null;
}

export interface RecallResult {
  block: string;
  items: RecalledMemory[];
  offline: boolean;
}

type FactOp =
  | { op: "add"; text: string; category?: string }
  | { op: "update"; id: string; text: string }
  | { op: "delete"; id: string };

let idCounter = 0;
export function newId(prefix: string): string {
  idCounter = (idCounter + 1) % 1_000_000;
  return `${prefix}_${Date.now().toString(36)}${idCounter.toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

export class MemoryService {
  private readonly userLocks = new Map<string, Promise<void>>();

  constructor(
    private readonly deps: {
      store: MemoryVectorStore;
      embedder: () => Promise<EmbeddingPort | null>;
      chat: () => Promise<ChatModelPort>;
      prompts: PromptsPort;
      clock: ClockPort;
      log: LoggerPort;
      status: StatusPort;
    },
  ) {}

  private longTermFilter(scope: MemoryScope, s: MemorySettings): MemoryFilter {
    const f: MemoryFilter = { user_id: scope.userId, memory_type: "long_term" };
    if (s.longTermScope === "workspace") f.workspace_id = scope.workspaceId;
    return f;
  }

  private async embedOne(text: string): Promise<{ emb: EmbeddingPort; vec: number[] } | null> {
    const emb = await this.deps.embedder();
    if (!emb) return null;
    const [vec] = await emb.embed([text]);
    if (!vec) return null;
    return { emb, vec };
  }

  /** Recall never throws: on embedding failure the turn runs without memory. */
  async recall(query: string, scope: MemoryScope, s: MemorySettings): Promise<RecallResult> {
    if (!s.enabled) {
      this.deps.status.memory("disabled");
      return { block: "", items: [], offline: false };
    }
    try {
      const hit = await this.embedOne(query);
      if (!hit) throw new Error("no embedder");
      const { emb, vec } = hit;
      const [short, long, allFacts] = await Promise.all([
        s.sessionLimit > 0
          ? this.deps.store.search(emb.modelKey, vec, { user_id: scope.userId, memory_type: "short_term", session_id: scope.sessionId }, s.sessionLimit)
          : [],
        s.userLimit > 0 ? this.deps.store.search(emb.modelKey, vec, this.longTermFilter(scope, s), s.userLimit) : [],
        s.alwaysRecallFacts > 0 ? this.deps.store.list(this.longTermFilter(scope, s)) : [],
      ]);
      const standing = allFacts.length <= s.alwaysRecallFacts ? allFacts : [];
      const scored = new Map(long.map((p) => [p.id, p.score]));
      const items: RecalledMemory[] = [
        ...standing.map((p) => ({ id: p.id, text: p.payload.text, memoryType: p.payload.memory_type, score: scored.get(p.id) ?? 1 })),
        ...[...long, ...short]
          .filter((p) => p.score >= s.minScore && !standing.some((f) => f.id === p.id))
          .map((p) => ({ id: p.id, text: p.payload.text, memoryType: p.payload.memory_type, score: p.score })),
      ];
      this.deps.status.memory("online");
      return { block: this.renderBlock(items, s.promptMaxChars), items, offline: false };
    } catch (e) {
      this.deps.log.warn("memory recall offline", String(e));
      this.deps.status.memory("offline");
      return { block: "", items: [], offline: true };
    }
  }

  renderBlock(items: RecalledMemory[], maxChars: number): string {
    if (items.length === 0) return "";
    const facts = items.filter((i) => i.memoryType === "long_term").map((i) => `- ${i.text}`);
    const chunks = items.filter((i) => i.memoryType === "short_term").map((i) => `- ${i.text.replace(/\n+/g, " ")}`);
    const parts: string[] = [];
    if (facts.length) parts.push(`Known about the user (long-term memory):\n${facts.join("\n")}`);
    if (chunks.length) parts.push(`Earlier in this chat (recalled excerpts):\n${chunks.join("\n")}`);
    return clipChars(parts.join("\n\n"), maxChars);
  }

  /** Short-term save: chunk the whole turn so facts buried in long pastes stay retrievable. */
  async saveTurn(scope: MemoryScope, s: MemorySettings, userText: string, assistantText: string): Promise<void> {
    if (!s.enabled) return;
    try {
      const emb = await this.deps.embedder();
      if (!emb) return;
      const chunks = [
        ...chunkText(`User: ${userText}`, s.chunkChars, s.chunkOverlap, s.maxChunksPerTurn),
        ...chunkText(`Assistant: ${assistantText}`, s.chunkChars, s.chunkOverlap, Math.max(1, Math.floor(s.maxChunksPerTurn / 4))),
      ];
      if (!chunks.length) return;
      const vecs = await emb.embed(chunks);
      const now = this.deps.clock.now();
      await this.deps.store.upsert(
        emb.modelKey,
        chunks.map((text, i) => ({
          id: newId("mem"),
          vector: vecs[i] ?? null,
          payload: {
            user_id: scope.userId,
            session_id: scope.sessionId,
            memory_type: "short_term",
            kind: "chunk",
            category: "turn",
            workspace_id: scope.workspaceId,
            text,
            created_at: now,
            updated_at: now,
            history: [],
          },
        })),
      );
      this.deps.status.memory("online");
    } catch (e) {
      this.deps.log.warn("memory save offline", String(e));
      this.deps.status.memory("offline");
    }
  }

  /** Long-term distillation, serialized per user so concurrent chats do not race. */
  learnFacts(scope: MemoryScope, s: MemorySettings, userText: string, assistantText: string): Promise<void> {
    if (!s.enabled) return Promise.resolve();
    const prev = this.userLocks.get(scope.userId) ?? Promise.resolve();
    const next = prev
      .catch(() => undefined)
      .then(() => this.learnFactsLocked(scope, s, userText, assistantText))
      .catch((e) => this.deps.log.warn("fact distillation failed", String(e)));
    this.userLocks.set(scope.userId, next);
    return next;
  }

  private async learnFactsLocked(scope: MemoryScope, s: MemorySettings, userText: string, assistantText: string): Promise<void> {
    const hit = await this.embedOne(userText);
    if (!hit) return;
    const { emb, vec } = hit;
    const filter = this.longTermFilter(scope, s);
    const known = await this.deps.store.search(emb.modelKey, vec, filter, s.factKnownLimit);
    const knownText = known.length ? known.map((k) => `[${k.id}] ${k.payload.text}`).join("\n") : "(none)";
    const prompt = render(this.deps.prompts.get("memory/fact_distill.md"), {
      known_facts: knownText,
      user_message: clipChars(userText, 6000),
      assistant_message: clipChars(assistantText, 3000),
      require_grounding: s.factRequireUserGrounding ? "yes" : "no",
    });
    const model = await this.deps.chat();
    let out = "";
    for await (const c of model.stream({ messages: [{ role: "user", content: prompt }], maxTokens: 800, temperature: 0, enableThinking: false })) {
      if (c.type === "text") out += c.text;
    }
    const ops = parseFactOps(out);
    const knownIds = new Set(known.map((k) => k.id));
    for (const op of ops) {
      if (op.op === "delete" && knownIds.has(op.id)) {
        await this.deps.store.delete([op.id], filter);
      } else if (op.op === "update" && knownIds.has(op.id)) {
        await this.updateFact(op.id, op.text, filter, s, emb);
      } else if (op.op === "add") {
        const [v] = await emb.embed([op.text]);
        if (!v) continue;
        const near = await this.deps.store.search(emb.modelKey, v, filter, 1);
        const top = near[0];
        if (top && top.score >= s.factDedupScore) {
          await this.updateFact(top.id, op.text, filter, s, emb, v);
        } else {
          await this.addFact(scope, op.text, op.category ?? "general", emb, v);
        }
      }
    }
  }

  async addFact(scope: MemoryScope, text: string, category: string, emb?: EmbeddingPort, vec?: number[]): Promise<MemoryPoint> {
    const now = this.deps.clock.now();
    let e = emb;
    let v = vec;
    if (!e) e = (await this.deps.embedder()) ?? undefined;
    if (e && !v) v = (await e.embed([text]))[0];
    const point: MemoryPoint = {
      id: newId("fact"),
      vector: v ?? null,
      payload: {
        user_id: scope.userId,
        session_id: null,
        memory_type: "long_term",
        kind: "fact",
        category,
        workspace_id: scope.workspaceId,
        text,
        created_at: now,
        updated_at: now,
        history: [],
      },
    };
    await this.deps.store.upsert(e?.modelKey ?? "", [point]);
    return point;
  }

  async updateFact(id: string, text: string, filter: MemoryFilter, s: MemorySettings, emb?: EmbeddingPort, vec?: number[]): Promise<void> {
    const existing = await this.deps.store.get(id, filter);
    if (!existing) return;
    let e = emb;
    if (!e) e = (await this.deps.embedder()) ?? undefined;
    const v = vec ?? (e ? (await e.embed([text]))[0] : undefined);
    const history = [existing.payload.text, ...existing.payload.history].slice(0, s.historyKeep);
    await this.deps.store.upsert(e?.modelKey ?? "", [
      { id, vector: v ?? existing.vector, payload: { ...existing.payload, text, history, updated_at: this.deps.clock.now() } },
    ]);
  }
}

export function parseFactOps(text: string): FactOp[] {
  const cleaned = text.replace(/<think>[\s\S]*?<\/think>/g, "");
  const m = /\[[\s\S]*\]/.exec(cleaned);
  if (!m) return [];
  try {
    const arr = JSON.parse(m[0]) as unknown;
    if (!Array.isArray(arr)) return [];
    const ops: FactOp[] = [];
    for (const raw of arr) {
      if (!raw || typeof raw !== "object") continue;
      const o = raw as Record<string, unknown>;
      const text = typeof o.text === "string" ? o.text.trim() : "";
      const id = typeof o.id === "string" ? o.id : "";
      if (o.op === "add" && text) ops.push({ op: "add", text, category: typeof o.category === "string" ? o.category : undefined });
      else if (o.op === "update" && id && text) ops.push({ op: "update", id, text });
      else if (o.op === "delete" && id) ops.push({ op: "delete", id });
    }
    return ops;
  } catch {
    return [];
  }
}
