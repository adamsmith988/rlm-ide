import { mkdtempSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { defaultSettings, type Settings, type TurnEvent } from "@rlm/contracts";
import { Orchestrator, type ChatModelPort, type EmbeddingPort } from "@rlm/core";
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
  embedder: HashEmbedder;
  settings: Settings;
  ws: string;
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
  const summarizer = new ScriptedModel("sum", () => "[]");
  const memory = new SqliteMemoryStore(db);
  const events: TurnEvent[] = [];
  const orch: Orchestrator = new Orchestrator({
    settings: { get: () => settings },
    updateSettings: (p) => Object.assign(settings, p),
    router: {
      chat: async (role) => (role === "agent" ? agent.model : summarizer),
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
    embedder,
    settings,
    ws,
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
