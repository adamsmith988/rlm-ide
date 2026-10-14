import { describe, expect, it } from "vitest";
import { defaultSettings } from "@rlm/contracts";
import {
  CellFenceFilter,
  computeBudget,
  extractCells,
  normalizeCheckpoint,
  parseCheckpoint,
  planCut,
  SessionCompactor,
  stripCells,
  toUnits,
  type StoredTurn,
} from "../src";
import { prompts, ScriptedModel } from "./fakes";

const turn = (idx: number, user: string, assistant = "ok", meta: StoredTurn["meta"] = {}): StoredTurn => ({
  id: `t${idx}`,
  chatId: "c",
  idx,
  userText: user,
  assistantText: assistant,
  status: "done",
  createdAt: idx,
  meta,
});

describe("budget", () => {
  it("derives every number from limits and settings", () => {
    const b = computeBudget({ contextLength: 10000, maxOutputTokens: 4000 }, defaultSettings().compact);
    expect(b.outputReserve).toBe(2500);
    expect(b.history).toBe(4500);
    expect(b.trigger).toBe(3600);
    expect(b.keep).toBe(1575);
  });
});

describe("planCut", () => {
  it("never collapses kept turns to zero when the newest turn fits newestLimit", () => {
    const turns = [turn(0, "a"), turn(1, "x".repeat(3000))];
    const { keep, fold } = planCut(turns, 100, 2000);
    expect(keep.map((t) => t.idx)).toEqual([1]);
    expect(fold.map((t) => t.idx)).toEqual([0]);
  });

  it("folds the oldest whole turns", () => {
    const turns = Array.from({ length: 10 }, (_, i) => turn(i, "y".repeat(320)));
    const { keep, fold } = planCut(turns, 400, 400);
    expect(fold.length + keep.length).toBe(10);
    expect(keep[keep.length - 1]!.idx).toBe(9);
    expect(fold[0]!.idx).toBe(0);
  });
});

describe("SessionCompactor", () => {
  const limits = { contextLength: 4000, maxOutputTokens: 1000 };
  const settings = defaultSettings().compact;

  it("does nothing under the trigger", async () => {
    const c = new SessionCompactor({ prompts, summarizer: async () => new ScriptedModel("s", () => "") });
    const r = await c.maybeCompact({ limits, settings, checkpoint: null, verbatim: [turn(0, "hi")], newUserText: "hello" });
    expect(r.compacted).toBe(false);
  });

  it("updates from the previous checkpoint and carries facts", async () => {
    const summarizer = new ScriptedModel("s", (req) => {
      const sys = req.messages[0]!.content;
      const carried = sys.includes("CODE-7731") ? "- deploy code is CODE-7731\n" : "";
      const body = req.messages.map((m) => m.content).join("\n");
      const newFacts = body.match(/port is \d+/) ? `- ${body.match(/port is \d+/)![0]}\n` : "";
      return `## Goal\nbuild app\n## Facts & Values\n${carried}${newFacts}`;
    });
    const c = new SessionCompactor({ prompts, summarizer: async () => summarizer });
    const verbatim = Array.from({ length: 12 }, (_, i) => turn(i, i === 1 ? "the port is 5173 " + "z".repeat(900) : "w".repeat(900)));
    const r = await c.maybeCompact({
      limits,
      settings,
      checkpoint: { summary: "## Facts & Values\n- deploy code is CODE-7731", throughIdx: -1 },
      verbatim,
      newUserText: "next",
    });
    expect(r.compacted).toBe(true);
    expect(r.kept.length).toBeGreaterThan(0);
    expect(r.checkpoint!.summary).toContain("CODE-7731");
    expect(r.checkpoint!.summary).toContain("port is 5173");
    // canonical 7 sections, in order
    const parsed = parseCheckpoint(r.checkpoint!.summary);
    expect(Object.keys(parsed)).toHaveLength(7);
    expect(r.checkpoint!.summary.indexOf("## Goal")).toBeLessThan(r.checkpoint!.summary.indexOf("## Dead ends"));
    // the summarizer saw the previous checkpoint and the instruction last
    const req = summarizer.calls[0]!;
    expect(req.messages[0]!.content).toContain("Earlier compacted summary");
    expect(req.messages[req.messages.length - 1]!.content).toContain("Write the checkpoint now.");
  });

  it("fires on row count too", async () => {
    const small = defaultSettings().compact;
    const c = new SessionCompactor({ prompts, summarizer: async () => new ScriptedModel("s", () => "## Goal\nx") });
    const verbatim = Array.from({ length: 60 }, (_, i) => turn(i, "q"));
    const r = await c.maybeCompact({ limits: { contextLength: 4000, maxOutputTokens: 500 }, settings: small, checkpoint: null, verbatim, newUserText: "q" });
    expect(r.reason).toBe("rows");
    expect(r.kept.length).toBeGreaterThan(0);
    expect(r.kept.length).toBeLessThan(60);
  });
});

describe("checkpoint format", () => {
  it("normalizes model output into the canonical sections", () => {
    const n = normalizeCheckpoint("<think>hmm</think>Sure!\n## Facts & Values\n- a = 1\n## Goal\nship it");
    expect(n.startsWith("## Goal\nship it")).toBe(true);
    expect(n).toContain("## Dead ends\n(none)");
    expect(n).not.toContain("Sure!");
  });
});

describe("RLM transcript units", () => {
  it("never separates a code cell from its output", () => {
    const units = toUnits([
      { kind: "user", text: "u", turnIdx: 0 },
      { kind: "assistant", text: "```repl\nx\n```", turnIdx: 0 },
      { kind: "repl", text: "out", turnIdx: 0 },
      { kind: "assistant", text: "done", turnIdx: 0 },
    ]);
    expect(units.map((u) => u.map((e) => e.kind))).toEqual([["user"], ["assistant", "repl"], ["assistant"]]);
  });
});

describe("cells", () => {
  it("extracts only repl fences", () => {
    const t = "Let me look.\n```repl\nprint(1)\n```\nand\n```python\nnot_run()\n```";
    expect(extractCells(t)).toEqual(["print(1)"]);
    expect(stripCells(t)).toContain("not_run()");
  });

  it("filters repl fences from a stream split at arbitrary points", () => {
    const text = "Reading.\n```repl\nprint(fs.read_file(path='a'))\n```\nDone.";
    for (const size of [1, 2, 3, 5, 8]) {
      const f = new CellFenceFilter();
      let out = "";
      for (let i = 0; i < text.length; i += size) out += f.push(text.slice(i, i + size));
      out += f.flush();
      expect(out).toBe("Reading.\n\nDone.");
    }
  });
});
