import { describe, expect, it } from "vitest";
import { ReplyGuard, repeatedSentence, repeatingTail } from "../src/application/rlm/guard";
import { stripThinking } from "../src/application/rlm/cells";

describe("reply guard", () => {
  it("spots a passage repeated at the end of a reply", () => {
    const loop = "Let me open the settings file and check the port again. ";
    expect(repeatingTail(`Intro text. ${loop.repeat(8)}`, 4)).toBe(true);
    expect(repeatingTail(`Intro text. ${loop.repeat(3)}`, 4)).toBe(false);
  });

  it("leaves repetitive data and normal code alone", () => {
    expect(repeatingTail(`data = [${"0, ".repeat(2000)}0]`, 4)).toBe(false);
    const code = Array.from({ length: 80 }, (_, i) => `row_${i} = compute(${i}, scale=${i * 3})`).join("\n");
    expect(repeatingTail(code, 4)).toBe(false);
  });

  it("spots reasoning that keeps coming back to one sentence, numbers aside", () => {
    const s = Array.from({ length: 6 }, (_, i) => `Attempt ${i}: Wait, I should re-read the request again. Then thought ${i * 13} about layout ${i}.`).join("\n");
    expect(repeatedSentence(s, 4)).toBe(true);
    expect(repeatedSentence(s, 6)).toBe(false);
  });

  it("cuts reasoning at its budget", () => {
    const g = new ReplyGuard({ thinkingMaxChars: 1000, repeatLimit: 4 });
    let cut = null;
    for (let i = 0; i < 100 && !cut; i++) cut = g.push("thinking", `thought number ${i} is different. `.slice(0, 20));
    expect(cut).toBe("thinking_budget");
    expect(g.thinkingSoFar).toBeGreaterThan(1000);
  });

  it("cuts slow reasoning at its time limit", () => {
    let t = 0;
    const g = new ReplyGuard({ thinkingMaxChars: 1e9, thinkingMaxMs: 60_000, repeatLimit: 4, now: () => t });
    expect(g.push("thinking", "a slow start")).toBeNull();
    t = 59_000;
    expect(g.push("thinking", " still going")).toBeNull();
    t = 61_000;
    expect(g.push("thinking", " and on")).toBe("thinking_budget");
  });

  it("does not count answer text against the reasoning budget", () => {
    const g = new ReplyGuard({ thinkingMaxChars: 100, repeatLimit: 4 });
    const lines = Array.from({ length: 200 }, (_, i) => `const v${i} = load("file-${i}.json");\n`);
    expect(lines.map((l) => g.push("text", l)).filter(Boolean)).toEqual([]);
  });
});

describe("stripThinking", () => {
  it("drops closed, prompt-opened and cut-off reasoning", () => {
    expect(stripThinking("<think>a</think>Answer")).toBe("Answer");
    expect(stripThinking("reasoning</think>Answer")).toBe("Answer");
    expect(stripThinking("Answer<think>never closed because the limit hit")).toBe("Answer");
    expect(stripThinking("<think>only reasoning")).toBe("");
  });
});
