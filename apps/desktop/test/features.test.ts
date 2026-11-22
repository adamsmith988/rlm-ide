import { describe, expect, it } from "vitest";
import { applyEvent, settleApprovals, type UiMessage } from "../src/renderer/src/chat/model";
import { parseEnv, splitArgs } from "../src/orchestrator/mcp";

function msg(): UiMessage {
  return { id: "m", turnId: "t", role: "assistant", text: "", status: "running", blocks: [], lastSeq: 0 };
}

describe("skipped approvals", () => {
  it("disables an unanswered approval when the turn ends", () => {
    const call = { callId: "c", toolId: "fs.write_file", args: {} };
    let m = applyEvent(msg(), { type: "approval_request", approvalId: "a1", call, turnId: "t", seq: 1, ts: 1 });
    m = applyEvent(m, { type: "turn_done", finalText: "done", usage: { inputTokens: 1, outputTokens: 1 }, turnId: "t", seq: 2, ts: 2 });
    const approval = m.blocks.find((b) => b.kind === "approval");
    expect(approval).toMatchObject({ resolved: true, approved: false, skipped: true });
  });

  it("keeps an answered approval as the user left it", () => {
    const blocks = settleApprovals([{ kind: "approval", approvalId: "a", call: { callId: "c", toolId: "x", args: {} }, resolved: true, approved: true }]);
    expect(blocks[0]).toMatchObject({ approved: true, resolved: true });
    expect(blocks[0]?.kind === "approval" && blocks[0].skipped).toBeFalsy();
  });
});

describe("sub-agent stream", () => {
  it("keeps the sub-agent's words on its card", () => {
    let m = applyEvent(msg(), { type: "subagent_started", agentId: "s1", parentPath: [], task: "fix the tests", turnId: "t", seq: 1, ts: 1 });
    m = applyEvent(m, { type: "subagent_delta", agentId: "s1", text: "reading ", turnId: "t", seq: 2, ts: 2 });
    m = applyEvent(m, { type: "subagent_delta", agentId: "s1", text: "the file", turnId: "t", seq: 3, ts: 3 });
    const sub = m.blocks.find((b) => b.kind === "subagent");
    expect(sub).toMatchObject({ log: "reading the file" });
    expect(sub?.kind === "subagent" && sub.done).toBeFalsy();
  });
});

describe("mcp command lines", () => {
  it("keeps quoted arguments together and skips comments in env", () => {
    expect(splitArgs(`-y "@scope/pkg" --flag`)).toEqual(["-y", "@scope/pkg", "--flag"]);
    expect(parseEnv("# secret\n\nFOO=bar\nBAZ=a=b")).toEqual({ FOO: "bar", BAZ: "a=b" });
  });
});
