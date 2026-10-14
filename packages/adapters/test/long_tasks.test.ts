import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { defaultSettings, type Settings, type TurnEvent } from "@rlm/contracts";
import { ContextOverflowError, Orchestrator, TaskBoards, type ChatChunk, type ChatModelPort, type ChatRequest } from "@rlm/core";
import { HashEmbedder, ScriptedModel, clock, log, prompts, status } from "../../core/test/fakes";
import {
  createBuiltinTools,
  FileSnapshotStore,
  FileTaskBoardStore,
  openDb,
  ProcessManager,
  PythonKernelFactory,
  SqliteConversationRepo,
  SqliteEditJournal,
  SqliteMemoryStore,
  SqlitePinnedFacts,
  ToolRegistry,
  Workspace,
} from "../src";
import { errorLocations, rewriteStats, isSmallFixByRewrite } from "../src/tools/rewrite_guard";

const ROOT = path.resolve(__dirname, "../../..");

function harness(limits = { contextLength: 32000, maxOutputTokens: 2000 }) {
  const ws = mkdtempSync(path.join(tmpdir(), "rlm-lt-ws-"));
  const data = mkdtempSync(path.join(tmpdir(), "rlm-lt-data-"));
  const db = openDb(path.join(data, "app.db"), null);
  const settings: Settings = defaultSettings();
  settings.rlm.sandboxNoNetwork = true;
  settings.rlm.wakeupMinSec = 1;
  settings.tools.workspaceAllow = ["*"];
  const pm = new ProcessManager();
  const workspace = new Workspace(pm);
  workspace.set(ws);
  const repo = new SqliteConversationRepo(db);
  const boards = new TaskBoards({ store: new FileTaskBoardStore(path.join(data, "boards")), clock, settings: () => settings.rlm, log });
  const tools = new ToolRegistry([
    ...createBuiltinTools({ processes: pm, shellTimeoutSec: () => 30, cellTimeoutSec: () => settings.rlm.cellTimeoutSec, editorState: () => ({}), searchSession: async () => [] }),
    ...boards.tools(),
  ]);
  const models: { agent: ChatModelPort; summarizer: ChatModelPort } = {
    agent: new ScriptedModel("agent", () => "hello"),
    summarizer: new ScriptedModel("sum", () => ""),
  };
  const events: TurnEvent[] = [];
  const orch = new Orchestrator({
    settings: { get: () => settings },
    updateSettings: (p) => Object.assign(settings, p),
    router: { chat: async (role) => (role === "agent" ? models.agent : models.summarizer), embedder: async () => new HashEmbedder(), modelIdFor: () => "test-model" },
    catalog: { limits: async () => limits },
    conversations: repo,
    memoryStore: new SqliteMemoryStore(db),
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
    broadcast: (e) => events.push(e),
    readMention: async () => "",
    boards,
  });
  const waitDone = async (turnId: string, ms = 20000) => {
    for (let t = 0; t < ms; t += 50) {
      const done = events.find((e) => e.turnId === turnId && (e.type === "turn_done" || e.type === "turn_error" || e.type === "turn_cancelled"));
      if (done) return done;
      await new Promise((r) => setTimeout(r, 50));
    }
    throw new Error(`turn ${turnId} did not finish: ${events.filter((e) => e.turnId === turnId).map((e) => e.type).join(",")}`);
  };
  return { ws, data, settings, boards, orch, events, models, waitDone, pm };
}

const system = (req: ChatRequest) => req.messages.find((m) => m.role === "system")?.content ?? "";
const transcript = (req: ChatRequest) => req.messages.filter((m) => m.role !== "system").map((m) => m.content).join("\n");

describe("task board and scheduled wake-ups", () => {
  const h = harness();
  afterAll(async () => {
    h.orch.stopScheduler();
    await h.orch.engine.shutdownAll();
  });

  it("keeps the plan on a persisted board, shows it every step, and wakes the agent up with it", async () => {
    const chat = await h.orch.createChat("long job");
    const seen: string[] = [];
    h.models.agent = new ScriptedModel("agent", (req, n) => {
      seen.push(system(req));
      if (n === 0)
        return [
          "Planning.",
          "```repl",
          "print(task.plan(goal='Train the toy model', steps=['download', 'train', 'report']))",
          "task.step(index=0, status='done', note='models/toy')",
          "task.log(text='training started, pid 4242, log train.log')",
          "print(task.wake_me(after_sec=1, reason='check train.log; when it says DONE write the report'))",
          "```",
        ].join("\n");
      if (n === 1) return "Training is running; I will check back in a second.";
      if (n === 2) return "```repl\ntask.step(index=1, status='done')\ntask.step(index=2, status='done', note='report.md')\nprint('ok')\n```";
      return "All steps are done: report.md is written.";
    });
    await h.orch.startScheduler();
    const { turnId } = await h.orch.send({ chatId: chat.id, text: "Please train the toy model on the synthetic set and report the loss. Secret code: KESTREL-77.", editorContext: {} });
    expect((await h.waitDone(turnId)).type).toBe("turn_done");

    // The board was rendered into the system message right after task.plan ran.
    expect(seen[1]).toContain("Your task board");
    expect(seen[1]).toContain("Goal: Train the toy model");
    expect(seen[1]).toContain("KESTREL-77");
    expect(seen[1]).toContain("[x] download");
    expect(seen[1]).toContain("training started, pid 4242");
    const boardFiles = readFileSync(path.join(h.data, "boards", `${chat.id.replace(/[^A-Za-z0-9_-]/g, "_")}.json`), "utf8");
    expect(JSON.parse(boardFiles).wakeup.reason).toContain("train.log");

    // The wake-up fires on its own and starts a turn marked as coming from the IDE.
    let wake: TurnEvent | undefined;
    for (let t = 0; t < 8000 && !wake; t += 100) {
      wake = h.events.find((e) => e.type === "turn_started" && e.chatId === chat.id && e.origin === "wakeup");
      await new Promise((r) => setTimeout(r, 100));
    }
    expect(wake).toBeTruthy();
    expect((await h.waitDone(wake!.turnId)).type).toBe("turn_done");
    const wakeSystem = seen[2]!;
    const wakeMsgs = (h.models.agent as ScriptedModel).calls[2]!;
    expect(transcript(wakeMsgs)).toContain("Scheduled wake-up #1");
    expect(transcript(wakeMsgs)).toContain("check train.log");
    expect(wakeSystem).toContain("KESTREL-77");
    expect(wakeSystem).toContain("Progress: 1/3 done; next: step 1");

    const board = await h.boards.get(chat.id);
    expect(board.wakeup).toBeNull();
    expect(board.autoWakeups).toBe(1);
    expect(board.steps.map((s) => s.status)).toEqual(["done", "done", "done"]);
    const msgs = await h.orch.messages(chat.id);
    expect(msgs.filter((m) => m.role === "user").map((m) => m.origin)).toEqual(["user", "wakeup"]);

    // A message from the user resets the run of automatic wake-ups.
    h.models.agent = new ScriptedModel("agent", () => "noted");
    const t3 = await h.orch.send({ chatId: chat.id, text: "thanks", editorContext: {} });
    await h.waitDone(t3.turnId);
    expect((await h.boards.get(chat.id)).autoWakeups).toBe(0);
  }, 40000);

  it("rejects wake-ups outside the allowed range and stops after the auto wake-up limit", async () => {
    const chat = await h.orch.createChat("limits");
    await expect(h.boards.scheduleWakeup(chat.id, 999999, "too late")).rejects.toThrow(/between/);
    h.settings.rlm.maxAutoWakeups = 0;
    await h.boards.scheduleWakeup(chat.id, 1, "should not fire");
    await new Promise((r) => setTimeout(r, 2500));
    expect(h.events.some((e) => e.type === "turn_started" && e.chatId === chat.id)).toBe(false);
    const b = await h.boards.get(chat.id);
    expect(b.wakeup).toBeNull();
    expect(b.log.at(-1)?.text).toMatch(/wake-ups in a row/);
    h.settings.rlm.maxAutoWakeups = 200;
  }, 10000);

  it("restores a wake-up that fell due while the IDE was closed", async () => {
    const chat = await h.orch.createChat("restart");
    h.orch.stopScheduler();
    await h.boards.scheduleWakeup(chat.id, 1, "resume after restart");
    await new Promise((r) => setTimeout(r, 1500));
    expect(h.events.some((e) => e.type === "turn_started" && e.chatId === chat.id)).toBe(false);
    h.models.agent = new ScriptedModel("agent", () => "resumed");
    await h.orch.startScheduler();
    let started: TurnEvent | undefined;
    for (let t = 0; t < 5000 && !started; t += 100) {
      started = h.events.find((e) => e.type === "turn_started" && e.chatId === chat.id);
      await new Promise((r) => setTimeout(r, 100));
    }
    expect(started && started.type === "turn_started" && started.origin).toBe("wakeup");
    await h.waitDone(started!.turnId);
  }, 15000);
});

describe("in-turn compaction keeps the job", () => {
  it("keeps the request verbatim and the board after folding, even when the summarizer returns nothing", async () => {
    const h = harness({ contextLength: 16000, maxOutputTokens: 1500 });
    h.settings.rlm.cellOutputLimitChars = 12000;
    h.settings.rlm.oldOutputKeepChars = 12000;
    const chat = await h.orch.createChat("compact");
    const calls: ChatRequest[] = [];
    h.models.agent = new ScriptedModel("agent", (req, n) => {
      calls.push(req);
      if (n === 0) return "```repl\ntask.plan(goal='Crunch the logs', steps=['read', 'crunch', 'answer'])\nprint('planned')\n```";
      if (n < 7) return `\`\`\`repl\nprint('chunk ${n} ' + 'x' * 11000)\n\`\`\``;
      return "Answer: the code word was ORCHID-19.";
    });
    const request = "Read the logs and tell me the code word. Constraint: never delete files. Code word: ORCHID-19.";
    const { turnId } = await h.orch.send({ chatId: chat.id, text: request, editorContext: {} });
    expect((await h.waitDone(turnId, 40000)).type).toBe("turn_done");
    const compactions = h.events.filter((e) => e.turnId === turnId && e.type === "compaction");
    expect(compactions.length).toBeGreaterThan(0);
    const after = calls[calls.length - 1]!;
    expect(transcript(after)).toContain(request);
    expect(system(after)).toContain("Goal: Crunch the logs");
    // The empty summary fell back to a digest instead of a checkpoint of "(none)".
    expect(system(after)).toContain("## Earlier work (checkpoint)");
    expect(system(after)).toMatch(/chunk \d/);
    // The meter reports the real prompt against the real window.
    const usage = h.events.filter((e) => e.turnId === turnId && e.type === "context_usage") as Array<Extract<TurnEvent, { type: "context_usage" }>>;
    for (const u of usage) {
      expect(u.contextLength).toBe(16000);
      expect(u.triggerTokens).toBe(Math.floor(16000 * h.settings.rlm.compactTriggerRatio));
      expect(u.usedTokens).toBeLessThanOrEqual(u.inputLimit);
    }
    await h.orch.engine.shutdownAll();
  }, 60000);
});

/** A server whose tokenizer counts more tokens than the character estimate and enforces its window like vLLM. */
class DenseTokenizerModel implements ChatModelPort {
  readonly modelId = "dense";
  readonly requests: Array<{ counted: number; maxTokens: number }> = [];
  overflows = 0;
  private n = 0;
  constructor(private ctx: number, private reply: (n: number) => string) {}
  async *stream(req: ChatRequest): AsyncIterable<ChatChunk> {
    const counted = Math.ceil(req.messages.reduce((t, m) => t + m.content.length, 0) / 2.2);
    if (counted + req.maxTokens > this.ctx) {
      this.overflows++;
      throw new ContextOverflowError(`This model's maximum context length is ${this.ctx} tokens. However, you requested ${req.maxTokens} output tokens and your prompt contains at least ${counted} input tokens, for a total of at least ${counted + req.maxTokens} tokens.`);
    }
    this.requests.push({ counted, maxTokens: req.maxTokens });
    yield { type: "text", text: this.reply(this.n++) };
    yield { type: "usage", inputTokens: counted, outputTokens: 20 };
  }
}

describe("prompt size as the server counts it", () => {
  it("learns the server's token count and keeps prompt + output inside the window", async () => {
    const h = harness({ contextLength: 16000, maxOutputTokens: 4000 });
    h.settings.rlm.cellOutputLimitChars = 9000;
    h.settings.rlm.oldOutputKeepChars = 9000;
    const chat = await h.orch.createChat("dense");
    const model = new DenseTokenizerModel(16000, (n) => (n < 9 ? `\`\`\`repl\nprint('${"█"}' * 8000 + ' part ${n}')\n\`\`\`` : "All parts printed."));
    h.models.agent = model;
    const { turnId } = await h.orch.send({ chatId: chat.id, text: "print the parts", editorContext: {} });
    const end = await h.waitDone(turnId, 60000);
    expect(end.type).toBe("turn_done");
    for (const r of model.requests) expect(r.counted + r.maxTokens).toBeLessThanOrEqual(16000);
    // At most the first oversized request is refused; after that the estimate follows the server.
    expect(model.overflows).toBeLessThanOrEqual(1);
    const usage = h.events.filter((e) => e.turnId === turnId && e.type === "context_usage") as Array<Extract<TurnEvent, { type: "context_usage" }>>;
    const last = usage[usage.length - 1]!;
    const lastReq = model.requests[model.requests.length - 1]!;
    // The meter shows what the server counts, not the raw character estimate.
    expect(last.usedTokens).toBeGreaterThanOrEqual(lastReq.counted * 0.95);
    await h.orch.engine.shutdownAll();
  }, 90000);
});

describe("fixing code by editing, not rewriting", () => {
  const pyFile = Array.from({ length: 60 }, (_, i) => `x${i} = ${i}`).join("\n");

  it("measures how much of a file a rewrite changes", () => {
    const fixed = pyFile.replace("x10 = 10", "x10 = 'ten'");
    const st = rewriteStats(pyFile, fixed);
    expect(st.changedLines).toBe(1);
    expect(st.ranges).toEqual([[11, 11]]);
    expect(isSmallFixByRewrite(st)).toBe(true);
    const rewritten = Array.from({ length: 60 }, (_, i) => `y${i} = ${i * 2}`).join("\n");
    expect(isSmallFixByRewrite(rewriteStats(pyFile, rewritten))).toBe(false);
  });

  it("warns on the first small-fix rewrite and refuses the next one unless overwrite=True", async () => {
    const ws = mkdtempSync(path.join(tmpdir(), "rlm-rw-"));
    writeFileSync(path.join(ws, "train.py"), pyFile);
    const notices: string[] = [];
    const ctx = { workspaceRoot: ws, turnId: "t", chatId: "c1", signal: new AbortController().signal, notice: (t: string) => notices.push(t) };
    const tools = createBuiltinTools({ processes: new ProcessManager(), shellTimeoutSec: () => 30, editorState: () => ({}), searchSession: async () => [] });
    const write = tools.find((t) => t.descriptor.id === "fs.write_file")!;
    const v1 = pyFile.replace("x10 = 10", "x10 = 'ten'");
    expect(await write.precheck!({ path: "train.py", content: v1 }, ctx)).toBeNull();
    const r1 = String(await write.run({ path: "train.py", content: v1 }, ctx));
    expect(r1).toMatch(/only 1 changed \(lines 11\)/);
    expect(notices.join(" ")).toMatch(/fs\.edit_file/);
    const v2 = v1.replace("x20 = 20", "x20 = 'twenty'");
    const refusal = await write.precheck!({ path: "train.py", content: v2 }, ctx);
    expect(refusal).toMatch(/not written.*fs\.edit_file/s);
    expect(await write.precheck!({ path: "train.py", content: v2, overwrite: true }, ctx)).toBeNull();
    // Another chat is not affected, and a real rewrite is never refused.
    expect(await write.precheck!({ path: "train.py", content: v2 }, { ...ctx, chatId: "c2" })).toBeNull();
    expect(await write.precheck!({ path: "train.py", content: "print('new program')\n" }, ctx)).toBeNull();
    expect(existsSync(path.join(ws, "train.py"))).toBe(true);
  });

  it("shows the closest text when fs.edit_file cannot find old_string", async () => {
    const ws = mkdtempSync(path.join(tmpdir(), "rlm-ed-"));
    writeFileSync(path.join(ws, "a.py"), "def f():\n    total = 0\n    for i in range(10):\n        total += i\n    return total\n");
    const ctx = { workspaceRoot: ws, turnId: "t", chatId: "c", signal: new AbortController().signal };
    const edit = createBuiltinTools({ processes: new ProcessManager(), shellTimeoutSec: () => 30, editorState: () => ({}), searchSession: async () => [] }).find((t) => t.descriptor.id === "fs.edit_file")!;
    await expect(edit.run({ path: "a.py", old_string: "for i in range(10):\n  total += i", new_string: "x" }, ctx)).rejects.toThrow(/closest text[\s\S]*3\|     for i in range\(10\)/);
  });

  it("points at the failing line of a script inside the project", () => {
    const root = "/proj";
    const py = 'Traceback (most recent call last):\n  File "/proj/train.py", line 88, in <module>\n    main()\n  File "/proj/lib/data.py", line 12, in load\n    x = y\n  File "/usr/lib/python3/site-packages/torch/x.py", line 5, in z\nNameError: y';
    expect(errorLocations(py, root)).toEqual([
      { path: "lib/data.py", line: 12 },
      { path: "train.py", line: 88 },
    ]);
    const js = "TypeError: x is undefined\n    at run (/proj/src/app.js:41:7)\n    at /proj/node_modules/lib/i.js:3:1";
    expect(errorLocations(js, root)).toEqual([{ path: "src/app.js", line: 41 }]);
    expect(errorLocations("src/main.ts:7:3 - error TS2322", root)).toEqual([{ path: "src/main.ts", line: 7 }]);
  });

  it("adds the hint to a failed shell.run", async () => {
    const ws = mkdtempSync(path.join(tmpdir(), "rlm-sh-"));
    writeFileSync(path.join(ws, "boom.py"), "def main():\n    raise ValueError('bad')\n\nmain()\n");
    const notices: string[] = [];
    const ctx = { workspaceRoot: ws, turnId: "t", chatId: "c", signal: new AbortController().signal, notice: (t: string) => notices.push(t) };
    const run = createBuiltinTools({ processes: new ProcessManager(), shellTimeoutSec: () => 30, editorState: () => ({}), searchSession: async () => [] }).find((t) => t.descriptor.id === "shell.run")!;
    const r = (await run.run({ command: `${process.env.RLM_PYTHON || "python3"} boom.py` }, ctx)) as { exit_code: number; hint?: string };
    expect(r.exit_code).not.toBe(0);
    expect(r.hint).toMatch(/boom\.py line 2/);
    expect(notices[0]).toMatch(/fs\.edit_file/);
  });
});

describe("step budget", () => {
  const h = harness();
  afterAll(async () => {
    h.orch.stopScheduler();
    await h.orch.engine.shutdownAll();
  });

  it("continues a planned job in a new turn when a turn runs out of steps", async () => {
    h.settings.rlm.maxSteps = 3;
    h.settings.rlm.stepWarnRemaining = 1;
    const chat = await h.orch.createChat("budget");
    let wokeUp = false;
    h.models.agent = new ScriptedModel("agent", (req, n) => {
      if (n === 0) return "```repl\ntask.plan(goal='long job', steps=['prepare', 'train', 'report'])\nprint('planned')\n```";
      if (transcript(req).includes("Scheduled wake-up")) {
        wokeUp = true;
        return "```repl\nfor i in range(3): task.step(index=i, status='done')\nprint('finished')\n```\n";
      }
      return "```repl\nprint('working')\n```";
    });
    await h.orch.startScheduler();
    const { turnId } = await h.orch.send({ chatId: chat.id, text: "do the long job", editorContext: {} });
    expect((await h.waitDone(turnId)).type).toBe("turn_done");
    const msgs = await h.orch.messages(chat.id);
    expect(msgs.find((m) => m.turnId === turnId && m.role === "assistant")?.text).toMatch(/Stopped after 3 steps with open steps on the task board; the IDE continues in 1 s/);
    const board = await h.boards.get(chat.id);
    expect(board.log.map((l) => l.text).join(" ")).toMatch(/3-step limit with 3 open steps/);

    let wake: TurnEvent | undefined;
    for (let t = 0; t < 8000 && !wake; t += 100) {
      wake = h.events.find((e) => e.type === "turn_started" && e.chatId === chat.id && e.origin === "wakeup");
      await new Promise((r) => setTimeout(r, 100));
    }
    expect(wake).toBeTruthy();
    await h.waitDone(wake!.turnId);
    expect(wokeUp).toBe(true);
    expect((await h.boards.get(chat.id)).steps.every((s) => s.status === "done")).toBe(true);
    // Nothing is open any more, so running out of steps again does not schedule another turn.
    expect((await h.boards.get(chat.id)).wakeup).toBeNull();
  });

  it("asks for a plan before the limit when a long job has no board", async () => {
    h.settings.rlm.maxSteps = 4;
    h.settings.rlm.stepWarnRemaining = 2;
    const chat = await h.orch.createChat("no board");
    h.models.agent = new ScriptedModel("agent", (req) =>
      transcript(req).includes("you have no task board")
        ? "```repl\ntask.plan(goal='job', steps=['a', 'b'])\ntask.step(index=0, status='done')\nprint('recorded')\n```"
        : "```repl\nprint('working')\n```",
    );
    const { turnId } = await h.orch.send({ chatId: chat.id, text: "do the job", editorContext: {} });
    await h.waitDone(turnId);
    const board = await h.boards.get(chat.id);
    expect(board.steps.map((s) => s.status)).toEqual(["done", "todo"]);
    expect(board.wakeup).not.toBeNull();
    await h.boards.cancelWakeup(chat.id);
  });

  it("keeps an unplanned job going when the model never writes a board", async () => {
    h.settings.rlm.maxSteps = 3;
    h.settings.rlm.stepWarnRemaining = 1;
    const chat = await h.orch.createChat("never plans");
    h.models.agent = new ScriptedModel("agent", () => "Still on it.\n```repl\nprint('working')\n```");
    const request = "Convert every CSV in data/ to parquet and report the row counts.";
    const { turnId } = await h.orch.send({ chatId: chat.id, text: request, editorContext: {} });
    await h.waitDone(turnId);
    const board = await h.boards.get(chat.id);
    expect(board.goal).toBe(request);
    expect(board.steps).toHaveLength(1);
    expect(board.steps[0]!.status).toBe("todo");
    expect(board.log.map((l) => l.text).join("\n")).toMatch(/last status before the step limit: Still on it/);
    expect(board.wakeup).not.toBeNull();
    await h.boards.cancelWakeup(chat.id);
  });
});

describe("fixing a long failing cell by re-sending it", () => {
  const h = harness();
  afterAll(async () => {
    await h.orch.engine.shutdownAll();
  });
  const longCell = (tag: string, value: string) =>
    ["```repl", `print("RAN-${tag}")`, ...Array.from({ length: 36 }, (_, i) => `v${i} = ${i}`), `data = ${value}`, "a, b = data", "```"].join("\n");

  it("warns on the first re-sent copy, flags a repeated error, and does not run the next copy", async () => {
    const chat = await h.orch.createChat("retry");
    h.models.agent = new ScriptedModel("agent", (_req, n) => {
      if (n === 0) return longCell("1", "[1, 2, 3]");
      if (n === 1) return longCell("2", "[4, 5, 6]");
      if (n === 2) return longCell("3", "[7, 8, 9]");
      return "I will move the code into a file and fix it there.";
    });
    const { turnId } = await h.orch.send({ chatId: chat.id, text: "unpack the data", editorContext: {} });
    expect((await h.waitDone(turnId)).type).toBe("turn_done");
    const out = h.events.filter((e) => e.turnId === turnId && e.type === "cell_output").map((e) => (e as { text: string }).text).join("");
    expect(out).toContain("RAN-1");
    expect(out).toContain("RAN-2");
    expect(out).not.toContain("RAN-3");
    const done = h.events.filter((e) => e.turnId === turnId && e.type === "cell_done") as Array<{ ok: boolean; error?: string }>;
    expect(done.map((d) => d.ok)).toEqual([false, false, false]);
    expect(done[2]!.error).toMatch(/^\[not run\] this cell repeats the previous failing cell with only 2 of 39 lines changed \(lines 1, 38\).*fs\.edit_file/s);

    const calls = (h.models.agent as ScriptedModel).calls;
    const afterSecond = transcript(calls[2]!);
    expect(afterSecond).toContain("[IDE] this cell repeats the previous failing cell");
    expect(afterSecond).toMatch(/\[hint\] Same error as the previous attempt \(ValueError: too many values to unpack/);
    expect(transcript(calls[3]!)).toContain("[not run]");
  });

  it("lets a cell that did not parse be sent again corrected", async () => {
    const chat = await h.orch.createChat("syntax");
    const broken = (tag: string) => ["```repl", `print("SYN-${tag}")`, ...Array.from({ length: 36 }, (_, i) => `w${i} = ${i}`), "x = (1,", "```"].join("\n");
    const fixed = (tag: string) => ["```repl", `print("SYN-${tag}")`, ...Array.from({ length: 36 }, (_, i) => `w${i} = ${i}`), "x = (1,)", "```"].join("\n");
    h.models.agent = new ScriptedModel("agent", (_req, n) => (n === 0 ? broken("1") : n === 1 ? broken("2") : n === 2 ? fixed("3") : "done"));
    const { turnId } = await h.orch.send({ chatId: chat.id, text: "go", editorContext: {} });
    expect((await h.waitDone(turnId)).type).toBe("turn_done");
    const out = h.events.filter((e) => e.turnId === turnId && e.type === "cell_output").map((e) => (e as { text: string }).text).join("");
    expect(out).toContain("SYN-3");
    expect(out).not.toContain("[not run]");
  });
});
