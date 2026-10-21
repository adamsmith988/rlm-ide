import { mkdtempSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import type * as Sqlite from "node:sqlite";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { defaultSettings, type Settings, type TurnEvent } from "@rlm/contracts";
import { Orchestrator, type ChatChunk, type ChatModelPort, type ChatRequest, type EmbeddingPort } from "@rlm/core";
import { HashEmbedder, ScriptedModel, clock, log, prompts, status } from "../../core/test/fakes";
import {
  createBuiltinTools,
  FileSnapshotStore,
  openDb,
  ProcessManager,
  PythonKernelFactory,
  SqliteConversationRepo,
  SqliteEditJournal,
  SqliteMemoryStore,
  SqlitePinnedFacts,
  ToolRegistry,
  Workspace,
  ThinkSplitter,
  resolveInside,
} from "../src";

const ROOT = path.resolve(__dirname, "../../..");

interface Harness {
  orch: Orchestrator;
  events: TurnEvent[];
  agent: { model: ChatModelPort };
  summarizer: { model: ChatModelPort };
  embedder: HashEmbedder;
  settings: Settings;
  ws: string;
  workspace: Workspace;
  memory: SqliteMemoryStore;
  run(chatId: string, text: string, approve?: boolean): Promise<TurnEvent[]>;
}

function harness(): Harness {
  const ws = mkdtempSync(path.join(tmpdir(), "rlm-ws-"));
  const data = mkdtempSync(path.join(tmpdir(), "rlm-data-"));
  const db = openDb(path.join(data, "app.db"), null);
  const settings: Settings = defaultSettings();
  settings.rlm.sandboxNoNetwork = true;
  const pm = new ProcessManager();
  const workspace = new Workspace(pm);
  workspace.set(ws);
  const repo = new SqliteConversationRepo(db);
  const tools = new ToolRegistry(
    createBuiltinTools({ processes: pm, shellTimeoutSec: () => 30, editorState: () => ({}), searchSession: async (c, q, l) => repo.searchTurns(c, q, l) }),
  );
  const embedder = new HashEmbedder();
  const agent: { model: ChatModelPort } = { model: new ScriptedModel("agent", () => "hello") };
  const summarizer: { model: ChatModelPort } = { model: new ScriptedModel("sum", () => "[]") };
  const memory = new SqliteMemoryStore(db);
  const events: TurnEvent[] = [];
  const orch: Orchestrator = new Orchestrator({
    settings: { get: () => settings },
    updateSettings: (p) => Object.assign(settings, p),
    router: {
      chat: async (role) => (role === "agent" ? agent.model : summarizer.model),
      embedder: async () => embedder as EmbeddingPort,
      modelIdFor: () => "test-model",
    },
    catalog: { limits: async () => ({ contextLength: 32000, maxOutputTokens: 2000 }) },
    conversations: repo,
    memoryStore: memory,
    pinned: new SqlitePinnedFacts(db),
    snapshots: new FileSnapshotStore(path.join(data, "snap")),
    edits: new SqliteEditJournal(db),
    kernels: new PythonKernelFactory({
      workerScript: path.join(ROOT, "python/rlm_worker/worker.py"),
      pythonCandidates: [process.env.RLM_PYTHON ?? "", "python3", "python"],
      sandboxNoNetwork: () => true,
      memoryMb: () => 2048,
      log,
    }),
    tools,
    prompts,
    workspace,
    clock,
    log,
    status,
    broadcast: (e) => {
      events.push(e);
      if (e.type === "approval_request") setTimeout(() => orch.runner.resolveApproval(e.approvalId, h.autoApprove, "once"), 5);
    },
    readMention: async (p) => readFileSync(resolveInside(ws, p), "utf8"),
  });
  const h = {
    orch,
    events,
    agent,
    summarizer,
    embedder,
    settings,
    ws,
    workspace,
    memory,
    autoApprove: true,
    async run(chatId: string, text: string, approve = true) {
      h.autoApprove = approve;
      const { turnId } = await orch.send({ chatId, text, editorContext: {} });
      for (let i = 0; i < 300; i++) {
        const done = events.find((e) => e.turnId === turnId && (e.type === "turn_done" || e.type === "turn_error" || e.type === "turn_cancelled"));
        if (done) return events.filter((e) => e.turnId === turnId);
        await new Promise((r) => setTimeout(r, 50));
      }
      throw new Error(`turn did not finish: ${JSON.stringify(events.filter((e) => e.turnId === turnId).map((e) => e.type))}`);
    },
  };
  return h;
}

describe("RLM engine end to end (scripted model, real kernel + tools)", () => {
  let h: Harness;
  beforeAll(() => {
    h = harness();
    writeFileSync(path.join(h.ws, "notes.txt"), "alpha\nbeta\ngamma\n");
  });
  afterAll(async () => h.orch.engine.shutdownAll());

  it("runs code cells, calls tools as kernel callables, persists variables across turns", async () => {
    const chat = await h.orch.createChat("t");
    h.agent.model = new ScriptedModel("agent", (req, n) => {
      if (n === 0) return "Reading.\n```repl\nlines = fs.read_file(path='notes.txt').splitlines()\nprint(len(lines))\n```";
      if (n === 1) return "There are 3 lines.";
      if (n === 2) return "```repl\nprint(lines[1].upper())\n```";
      return "Second line upper: BETA";
    });
    const ev1 = await h.run(chat.id, "how many lines in notes.txt?");
    expect(ev1.find((e) => e.type === "tool_call")).toBeTruthy();
    const out = ev1.filter((e) => e.type === "cell_output").map((e) => (e as { text: string }).text).join("");
    expect(out).toContain("3");
    expect((ev1.find((e) => e.type === "turn_done") as { finalText: string }).finalText).toBe("There are 3 lines.");
    const text = ev1.filter((e) => e.type === "text_delta").map((e) => (e as { text: string }).text).join("");
    expect(text).not.toContain("```repl");

    const ev2 = await h.run(chat.id, "uppercase the second line");
    const out2 = ev2.filter((e) => e.type === "cell_output").map((e) => (e as { text: string }).text).join("");
    expect(out2).toContain("BETA");
  });

  it("refuses tools outside tool_ids host-side and blocks path escapes", async () => {
    const chat = await h.orch.createChat("t2");
    h.agent.model = new ScriptedModel("agent", (_r, n) =>
      n === 0
        ? "```repl\ntry:\n    fs.read_file(path='../../etc/passwd')\nexcept ToolError as e:\n    print('ERR', e)\nr = host_request if 'host_request' in dir() else None\nprint(r)\n```"
        : "done",
    );
    const ev = await h.run(chat.id, "try escaping");
    const out = ev.filter((e) => e.type === "cell_output").map((e) => (e as { text: string }).text).join("");
    expect(out).toContain("outside the workspace");
    // Direct executor check: unknown / disallowed tool ids are refused host-side.
    const r = await h.orch.executor.execute("fs.read_file", { path: "notes.txt" }, {
      turnId: "x",
      chatId: chat.id,
      toolIds: new Set(["search.grep"]),
      ctx: { workspaceRoot: h.ws, turnId: "x", chatId: chat.id, signal: new AbortController().signal },
      conversation: () => "",
    });
    expect(r.ok).toBe(false);
    expect(r.error).toContain("not available");
  });

  it("asks approval for edits, applies on approve, records a reviewable diff, rejects cleanly", async () => {
    const chat = await h.orch.createChat("t3");
    h.agent.model = new ScriptedModel("agent", (_r, n) =>
      n === 0 ? "```repl\nprint(fs.edit_file(path='notes.txt', old_string='beta', new_string='BETA!'))\n```" : "edited",
    );
    const ev = await h.run(chat.id, "edit beta", true);
    expect(ev.some((e) => e.type === "approval_request")).toBe(true);
    expect(ev.some((e) => e.type === "pending_edit")).toBe(true);
    expect(readFileSync(path.join(h.ws, "notes.txt"), "utf8")).toContain("BETA!");

    h.agent.model = new ScriptedModel("agent", (_r, n) =>
      n === 0 ? "```repl\ntry:\n    fs.write_file(path='x.txt', content='no')\nexcept ToolError as e:\n    print('rejected:', e)\n```" : "ok",
    );
    const ev2 = await h.run(chat.id, "write x", false);
    const out = ev2.filter((e) => e.type === "cell_output").map((e) => (e as { text: string }).text).join("");
    expect(out).toContain("rejected");
  });

  it("de-duplicates identical side-effecting calls within a turn", async () => {
    const chat = await h.orch.createChat("t4");
    mkdirSync(path.join(h.ws, "d"), { recursive: true });
    h.agent.model = new ScriptedModel("agent", (_r, n) =>
      n === 0
        ? "```repl\na = fs.write_file(path='d/once.txt', content='1')\nb = fs.write_file(path='d/once.txt', content='1')\nprint(a == b)\n```"
        : "ok",
    );
    const ev = await h.run(chat.id, "write twice");
    const results = ev.filter((e) => e.type === "tool_result");
    expect(results.some((e) => (e as { deduped?: boolean }).deduped)).toBe(true);
    expect(ev.filter((e) => e.type === "approval_request")).toHaveLength(1);
  });

  it("re-runs a repeated command after an edit instead of returning the stale result", async () => {
    const chat = await h.orch.createChat("t4a");
    h.agent.model = new ScriptedModel("agent", (_r, n) =>
      n === 0
        ? "```repl\nfs.write_file(path='v.txt', content='one')\na = shell.run(command='cat v.txt')['stdout']\nfs.write_file(path='v.txt', content='two')\nb = shell.run(command='cat v.txt')['stdout']\nprint(a, b)\n```"
        : "ok",
    );
    const ev = await h.run(chat.id, "check rerun");
    const out = ev.filter((e) => e.type === "cell_output").map((e) => (e as { text: string }).text).join("");
    expect(out).toContain("one two");
  });

  it("after edits, asks the model once to check its work against the request", async () => {
    const chat = await h.orch.createChat("t4b");
    const seen: string[] = [];
    h.agent.model = new ScriptedModel("agent", (req, n) => {
      seen.push(req.messages[req.messages.length - 1]!.content);
      if (n === 0) return "```repl\nfs.write_file(path='app/a.txt', content='v1')\n```";
      if (n === 1) return "Wrote it.";
      if (n === 2) return "```repl\nfs.write_file(path='app/b.txt', content='v2')\n```";
      return "Wrote app/a.txt and app/b.txt.";
    });
    const ev = await h.run(chat.id, "write a.txt and b.txt in app/");
    expect(seen[2]).toContain("check your work against the request");
    expect(seen[2]).toContain("write a.txt and b.txt in app/");
    expect(readFileSync(path.join(h.ws, "app/b.txt"), "utf8")).toBe("v2");
    expect((ev.find((e) => e.type === "turn_done") as { finalText: string }).finalText).toBe("Wrote app/a.txt and app/b.txt.");
    expect(seen).toHaveLength(4);
  });

  it("reports syntax errors in files the agent writes, even when it does not print the result", async () => {
    const chat = await h.orch.createChat("t4d");
    const seen: string[] = [];
    h.agent.model = new ScriptedModel("agent", (req, n) => {
      seen.push(req.messages[req.messages.length - 1]!.content);
      if (n === 0) return "```repl\nfs.write_file(path='web/app.js', content=\"const s = [{ t: 'Sweet Child O' Mine' }];\\n\")\nfs.write_file(path='web/ok.json', content='{\"a\": 1}')\n```";
      if (n === 1) return "```repl\nfs.edit_file(path='web/app.js', old_string=\"O' Mine\", new_string=\"O\\\\' Mine\")\n```";
      return "fixed";
    });
    await h.run(chat.id, "make web/app.js");
    expect(seen[1]).toContain("[IDE]");
    expect(seen[1]).toMatch(/web\/app\.js has a syntax error at line 1/);
    expect(seen[1]).not.toContain("ok.json is not valid");
    expect(seen[2]).not.toContain("syntax error");
  });

  it("recovers when a reply is cut off inside a cell", async () => {
    const chat = await h.orch.createChat("t4c");
    const seen: string[] = [];
    h.agent.model = new ScriptedModel("agent", (req, n) => {
      seen.push(req.messages.map((m) => m.content).join("\n"));
      if (n === 0) return "Writing.\n```repl\nfs.write_file(path='big.txt', content='''aaaa";
      if (n === 1) return "```repl\nprint('small step')\n```";
      return "done";
    });
    const ev = await h.run(chat.id, "write a big file");
    expect(seen[1]).toContain("did NOT run");
    expect(ev.filter((e) => e.type === "code_cell")).toHaveLength(1);
    expect(ev.find((e) => e.type === "turn_done")).toBeTruthy();
  });

  it("Stop interrupts a runaway cell and keeps partial output", async () => {
    const chat = await h.orch.createChat("t5");
    h.agent.model = new ScriptedModel("agent", () => "Looping.\n```repl\nwhile True:\n    pass\n```");
    const { turnId } = await h.orch.send({ chatId: chat.id, text: "loop", editorContext: {} });
    for (let i = 0; i < 100 && !h.events.some((e) => e.turnId === turnId && e.type === "code_cell"); i++) await new Promise((r) => setTimeout(r, 50));
    await new Promise((r) => setTimeout(r, 300));
    h.orch.stop(turnId);
    for (let i = 0; i < 200 && !h.events.some((e) => e.turnId === turnId && e.type === "turn_cancelled"); i++) await new Promise((r) => setTimeout(r, 50));
    const cancelled = h.events.find((e) => e.turnId === turnId && e.type === "turn_cancelled") as { partialText: string } | undefined;
    expect(cancelled).toBeTruthy();
    expect(cancelled!.partialText).toContain("Looping.");
  });

  it("resubscribe replays buffered events after a reload", async () => {
    const chat = await h.orch.createChat("t6");
    h.agent.model = new ScriptedModel("agent", () => "Just text.");
    const ev = await h.run(chat.id, "hi");
    const turnId = ev[0]!.turnId;
    const replay = h.orch.runner.replay(turnId, -1);
    expect(replay[0]!.type).toBe("turn_started");
    expect(replay[replay.length - 1]!.type).toBe("turn_done");
    expect(h.orch.runner.replay(turnId, replay[replay.length - 1]!.seq)).toHaveLength(0);
  });
});

/** A model that streams `kind` chunks from `next(i)` until stopped, or a fixed reply. */
class RunawayModel implements ChatModelPort {
  readonly calls: ChatRequest[] = [];
  constructor(
    readonly modelId: string,
    private readonly plan: (req: ChatRequest, n: number) => { kind: "thinking" | "text"; next: (i: number) => string } | string,
  ) {}
  async *stream(req: ChatRequest): AsyncIterable<ChatChunk> {
    this.calls.push(req);
    const p = this.plan(req, this.calls.length - 1);
    if (typeof p === "string") {
      yield { type: "text", text: p };
      return;
    }
    for (let i = 0; i < 100_000; i++) {
      if (req.signal?.aborted) throw new DOMException("stopped", "AbortError");
      yield { type: p.kind, text: p.next(i) };
      if (i % 50 === 0) await new Promise((r) => setTimeout(r, 0));
    }
    throw new Error("the guard never cut this reply off");
  }
}

const lastPrompt = (req: ChatRequest) => req.messages.map((m) => m.content).join("\n");
const words = ["file", "parse", "index", "maybe", "token", "check", "branch", "value", "layout", "router", "cache", "query"];
/** Reasoning that never repeats: a different word sequence every chunk. */
const freshThought = (i: number) => `${words[i % 12]} ${words[(i * 7) % 11]} ${words[(i * 5 + 3) % 12]}${"xyzw"[i % 4]!.repeat((i % 5) + 1)}. `;

describe("runaway replies", () => {
  let h: Harness;
  beforeAll(() => {
    h = harness();
    h.settings.rlm.thinkingMaxTokens = 400;
  });
  afterAll(async () => h.orch.engine.shutdownAll());

  it("cuts off reasoning past its budget and has the model act without reasoning", async () => {
    const chat = await h.orch.createChat("think");
    const m = new RunawayModel("agent", (req, n) =>
      n === 0 ? { kind: "thinking", next: freshThought } : n === 1 ? "```repl\nprint(2 + 2)\n```" : "It is 4.",
    );
    h.agent.model = m;
    const ev = await h.run(chat.id, "what is 2 + 2?");
    expect(ev.some((e) => e.type === "status" && /reasoned for ~\d+ tokens without acting/.test((e as { text: string }).text))).toBe(true);
    expect(lastPrompt(m.calls[1]!)).toContain("so the IDE stopped you");
    expect(m.calls[1]!.enableThinking).toBe(false);
    expect(m.calls[2]!.enableThinking).toBeUndefined();
    expect((ev.find((e) => e.type === "turn_done") as { finalText: string }).finalText).toBe("It is 4.");
  });

  it("cuts off a reply that repeats itself and retries warmer", async () => {
    const chat = await h.orch.createChat("repeat");
    const m = new RunawayModel("agent", (_req, n) => (n === 0 ? { kind: "text", next: () => "Let me look at the configuration file once more. " } : "Done."));
    h.agent.model = m;
    const ev = await h.run(chat.id, "fix the config");
    expect(ev.some((e) => e.type === "status" && /repeating itself/.test((e as { text: string }).text))).toBe(true);
    expect(m.calls[1]!.temperature).toBe(0.7);
    expect(lastPrompt(m.calls[1]!)).toContain("kept repeating the same passage");
    expect((ev.find((e) => e.type === "turn_done") as { finalText: string }).finalText).toBe("Done.");
  });

  it("catches reasoning that circles back to the same sentence", async () => {
    const chat = await h.orch.createChat("circle");
    const m = new RunawayModel("agent", (_req, n) =>
      n === 0 ? { kind: "thinking", next: (i) => (i % 3 === 0 ? "Wait, I should re-read the user request again carefully. " : freshThought(i)) } : "Answer.",
    );
    h.agent.model = m;
    h.settings.rlm.thinkingMaxTokens = 100_000;
    const ev = await h.run(chat.id, "explain");
    h.settings.rlm.thinkingMaxTokens = 400;
    expect(ev.some((e) => e.type === "status" && /repeating itself/.test((e as { text: string }).text))).toBe(true);
  });

  it("stops the turn with an explanation when the model keeps looping", async () => {
    const chat = await h.orch.createChat("stuck");
    h.agent.model = new RunawayModel("agent", () => ({ kind: "text", next: () => "I am going around in circles here, again. " }));
    const ev = await h.run(chat.id, "go");
    const done = ev.find((e) => e.type === "turn_done") as { finalText: string };
    expect(done.finalText).toMatch(/kept repeating itself 4 times in a row/);
  });

  it("does not keep running the same cell for the same output", async () => {
    const chat = await h.orch.createChat("same-cell");
    h.agent.model = new ScriptedModel("agent", (req) => (/\[not run\]/.test(lastPrompt(req)) ? "Nothing changes; stopping." : "```repl\nprint('same')\n```"));
    const ev = await h.run(chat.id, "poll");
    const out = ev.filter((e) => e.type === "cell_output").map((e) => (e as { text: string }).text).join("");
    expect(out).toContain("[not run] This exact cell already ran 3 times");
    expect(ev.filter((e) => e.type === "code_cell")).toHaveLength(4);
    expect((ev.find((e) => e.type === "turn_done") as { finalText: string }).finalText).toBe("Nothing changes; stopping.");
  });

  it("hands a message too large for the prompt to the REPL", async () => {
    const chat = await h.orch.createChat("large");
    const filler = (n: number) => Array.from({ length: n }, (_, i) => `Line ${i}: ${words[i % 12]} ${words[(i * 5) % 12]}.`).join("\n");
    const big = `${filler(1500)}\nNEEDLE-7731 is the deploy key name.\n${filler(1500)}`;
    let first = "";
    h.agent.model = new ScriptedModel("agent", (req, n) => {
      if (n === 0) {
        first = lastPrompt(req);
        return "```repl\nprint(user_message.find('NEEDLE-7731') > 0, len(user_message))\n```";
      }
      return "Found it.";
    });
    const ev = await h.run(chat.id, big);
    expect(first).toContain("All of it is in the REPL variable `user_message`");
    expect(first).not.toContain("NEEDLE-7731");
    const out = ev.filter((e) => e.type === "cell_output").map((e) => (e as { text: string }).text).join("");
    expect(out).toContain(`True ${big.length}`);
  });
});

describe("memory", () => {
  let h: Harness;
  beforeAll(() => {
    h = harness();
  });
  afterAll(async () => h.orch.engine.shutdownAll());

  it("recalls a fact buried in a long paste later in the chat, isolated per session", async () => {
    const s = h.settings.memory;
    s.minScore = 0.2;
    const scopeA = { userId: "u1", sessionId: "A", workspaceId: null };
    const longLog = Array.from({ length: 80 }, (_, i) => `log line ${i} nothing interesting here`).join("\n") + "\nDATABASE_URL uses port 6543 on host db-internal\n" + "more noise\n".repeat(50);
    await h.orch.memory.saveTurn(scopeA, s, longLog, "noted");
    const r = await h.orch.memory.recall("which port does DATABASE_URL use on db-internal", scopeA, s);
    expect(r.items.some((i) => i.text.includes("6543"))).toBe(true);
    const other = await h.orch.memory.recall("which port does DATABASE_URL use on db-internal", { ...scopeA, sessionId: "B" }, s);
    expect(other.items.some((i) => i.text.includes("6543"))).toBe(false);
    const otherUser = await h.orch.memory.recall("which port does DATABASE_URL use", { ...scopeA, userId: "u2" }, s);
    expect(otherUser.items).toHaveLength(0);
  });

  it("long-term facts cross chats, dedup updates, delete removes from recall", async () => {
    const s = h.settings.memory;
    s.minScore = 0.2;
    const a = { userId: "u1", sessionId: "A", workspaceId: null };
    const f = await h.orch.memory.addFact(a, "Prefers pnpm over npm for installs", "tooling");
    const inB = await h.orch.memory.recall("should I use pnpm or npm for installs", { ...a, sessionId: "B" }, s);
    expect(inB.items.some((i) => i.id === f.id)).toBe(true);
    await h.memory.delete([f.id], { user_id: "u1", memory_type: "long_term" });
    const after = await h.orch.memory.recall("should I use pnpm or npm for installs", { ...a, sessionId: "B" }, s);
    expect(after.items.some((i) => i.id === f.id)).toBe(false);
  });

  it("recalls every standing preference while there are few, even for unrelated requests", async () => {
    const s = { ...h.settings.memory, minScore: 0.99, alwaysRecallFacts: 3 };
    const a = { userId: "u2", sessionId: "A", workspaceId: null };
    const f = await h.orch.memory.addFact(a, "Uses 2-space indentation", "style");
    const r = await h.orch.memory.recall("write a title case helper", { ...a, sessionId: "B" }, s);
    expect(r.items.map((i) => i.id)).toContain(f.id);
    expect(r.block).toContain("Uses 2-space indentation");
    for (const t of ["Likes tabs in Go", "Uses pnpm", "Name is Amanda"]) await h.orch.memory.addFact(a, t, "general");
    const many = await h.orch.memory.recall("write a title case helper", { ...a, sessionId: "B" }, s);
    expect(many.items).toHaveLength(0);
  });

  it("turns still run with the embedder down", async () => {
    h.embedder.failing = true;
    const chat = await h.orch.createChat("m");
    h.agent.model = new ScriptedModel("agent", () => "fine");
    const ev = await h.run(chat.id, "hello");
    expect(ev.find((e) => e.type === "memory_recalled")).toMatchObject({ offline: true });
    expect(ev.some((e) => e.type === "turn_done")).toBe(true);
    h.embedder.failing = false;
  });

  it("vector tables are prefixed and isolated", () => {
    expect(h.memory.collections().every((c) => c.startsWith("orchestrator_memory__"))).toBe(true);
  });
});

const cellText = (ev: TurnEvent[]) =>
  ev
    .filter((e) => e.type === "cell_output")
    .map((e) => (e as { text: string }).text)
    .join("");

describe("project isolation", () => {
  let h: Harness;
  let other: string;
  beforeAll(() => {
    h = harness();
    other = mkdtempSync(path.join(tmpdir(), "rlm-ws2-"));
  });
  afterAll(async () => h.orch.engine.shutdownAll());

  it("lists, runs and continues chats only inside their own project", async () => {
    const a = await h.orch.createChat("in A");
    expect(a.workspaceId).toBe(h.ws);
    h.agent.model = new ScriptedModel("agent", (_r, n) => (n === 0 ? "```repl\nsecret = 41 + 1\nimport os\nprint(os.getcwd())\n```" : "ok"));
    expect(cellText(await h.run(a.id, "remember a number"))).toContain(path.basename(h.ws));

    h.workspace.set(other);
    await h.orch.workspaceChanged();
    expect((await h.orch.listChats()).map((c) => c.id)).not.toContain(a.id);
    await expect(h.orch.send({ chatId: a.id, text: "hi", editorContext: {} })).rejects.toThrow(/belongs to the project/);
    const b = await h.orch.createChat("in B");
    expect((await h.orch.listChats()).map((c) => c.id)).toEqual([b.id]);
    h.agent.model = new ScriptedModel("agent", (_r, n) => (n === 0 ? "```repl\nimport os\nprint(os.getcwd(), 'secret' in globals())\n```" : "ok"));
    const inB = cellText(await h.run(b.id, "where am I"));
    expect(inB).toContain(path.basename(other));
    expect(inB).toContain("False");

    h.workspace.set(h.ws);
    await h.orch.workspaceChanged();
    expect((await h.orch.listChats()).map((c) => c.id)).toEqual([a.id]);
    // The folder switch ended A's idle kernel; its next turn starts a fresh one in A.
    h.agent.model = new ScriptedModel("agent", (_r, n) => (n === 0 ? "```repl\nimport os\nprint(os.getcwd(), 'secret' in globals())\n```" : "ok"));
    const back = cellText(await h.run(a.id, "again"));
    expect(back).toContain(path.basename(h.ws));
    expect(back).toContain("False");
  });

  it("keeps project facts in their project, facts about the user everywhere, and chat excerpts in their chat", async () => {
    const s = { ...h.settings.memory, minScore: 0.2 };
    expect(s.longTermScope).toBe("workspace");
    const inA = { userId: "iso", sessionId: "c1", workspaceId: h.ws };
    const inB = { ...inA, sessionId: "c2", workspaceId: other };
    await h.orch.memory.addFact(inA, "The billing service deploys with Docker Compose on Hetzner", "stack", undefined, undefined, "project");
    await h.orch.memory.addFact(inA, "Prefers pnpm over npm for installs", "tooling");
    const q = "how does the billing service deploy, and should I use pnpm or npm for installs?";
    const a = await h.orch.memory.recall(q, inA, s);
    expect(a.items.map((i) => i.text)).toEqual(expect.arrayContaining(["The billing service deploys with Docker Compose on Hetzner", "Prefers pnpm over npm for installs"]));
    expect(a.block).toContain("Known about this project");
    expect(a.block).toContain("Known about the user");
    const b = await h.orch.memory.recall(q, inB, s);
    expect(b.items.map((i) => i.text)).toContain("Prefers pnpm over npm for installs");
    expect(b.items.some((i) => i.text.includes("Hetzner"))).toBe(false);

    await h.orch.memory.saveTurn(inA, s, "The staging password rotation happens on Fridays", "ok");
    const sameChat = await h.orch.memory.recall("when does the staging password rotation happen", inA, s);
    expect(sameChat.items.some((i) => i.memoryType === "short_term")).toBe(true);
    const newChat = await h.orch.memory.recall("when does the staging password rotation happen", { ...inA, sessionId: "c3" }, s);
    expect(newChat.items.some((i) => i.memoryType === "short_term")).toBe(false);
  });

  it("learns only short facts the user stated, filed as user or project facts", async () => {
    const s = { ...h.settings.memory };
    const scope = { userId: "learn", sessionId: "c1", workspaceId: h.ws };
    const long = `The user asked to fix the login button, so the agent edited styles.css, re-ran the whole test suite twice, checked the button in the browser and confirmed the login button now looks right on every page ${"again ".repeat(10)}`;
    const sum = new ScriptedModel("sum", () =>
      JSON.stringify([
        { op: "add", about: "project", text: "The API runs on Node 22 and is tested with vitest", category: "stack" },
        { op: "add", about: "user", text: "Name is Amanda", category: "identity" },
        { op: "add", about: "user", text: "Likes Haskell and category theory" },
        { op: "add", about: "project", text: "```js\nconst a = 1\n```" },
        { op: "add", about: "project", text: long },
      ]),
    );
    h.summarizer.model = sum;
    await h.orch.memory.learnFacts(scope, s, "I'm Amanda. Our API runs on Node 22 and we test with vitest. Please fix the login button.", "Fixed.");
    const facts = await h.memory.list({ user_id: "learn", memory_type: "long_term" });
    expect(facts.map((f) => [f.payload.text, f.payload.workspace_id]).sort()).toEqual([
      ["Name is Amanda", null],
      ["The API runs on Node 22 and is tested with vitest", h.ws],
    ]);
    const calls = sum.calls.length;
    await h.orch.memory.learnFacts(scope, s, "ok thanks", "Done.");
    expect(sum.calls.length).toBe(calls);
  });
});

describe("database migration", () => {
  it("files existing chats under the project their turns were saved in", async () => {
    const file = path.join(mkdtempSync(path.join(tmpdir(), "rlm-mig-")), "old.db");
    const { DatabaseSync } = process.getBuiltinModule("node:sqlite") as typeof Sqlite;
    const raw = new DatabaseSync(file);
    raw.exec(readFileSync(path.join(ROOT, "packages/adapters/src/db/migrations/001_init.sql"), "utf8"));
    raw.exec("CREATE TABLE schema_version (version INTEGER NOT NULL); INSERT INTO schema_version VALUES (1);");
    raw.exec("INSERT INTO chats (id, title, created_at, updated_at) VALUES ('c1', 'a', 1, 2), ('c2', 'b', 1, 1);");
    raw.exec(
      "INSERT INTO memory (id, user_id, session_id, memory_type, kind, category, workspace_id, text, created_at, updated_at) VALUES ('m1', 'u', 'c1', 'short_term', 'chunk', 'turn', '/p/a', 'x', 1, 1);",
    );
    raw.close();
    const repo = new SqliteConversationRepo(openDb(file, null));
    expect(await repo.getChat("c1")).toMatchObject({ workspaceId: "/p/a" });
    expect(await repo.getChat("c2")).toMatchObject({ workspaceId: null });
    expect((await repo.listChats("/p/a")).map((c) => c.id)).toEqual(["c1"]);
    expect((await repo.listChats(null)).map((c) => c.id)).toEqual(["c2"]);
  });
});

describe("think splitter", () => {
  it("handles templates that open <think> in the prompt", () => {
    const s = new ThinkSplitter(true);
    const parts = [...s.push("plan it</thi"), ...s.push("nk>\n\nAnswer"), ...s.flush()];
    expect(parts.filter((p) => p.type === "thinking").map((p) => p.text).join("")).toBe("plan it");
    expect(parts.filter((p) => p.type === "text").map((p) => p.text).join("")).toBe("Answer");
  });
  it("detects an explicit <think> at the start", () => {
    const s = new ThinkSplitter(false);
    const parts = [...s.push("<thi"), ...s.push("nk>x</think>y"), ...s.flush()];
    expect(parts).toEqual([{ type: "thinking", text: "x" }, { type: "text", text: "y" }]);
  });
});
