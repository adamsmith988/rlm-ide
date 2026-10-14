import { describe, expect, it } from "vitest";
import { ToolSettings } from "@rlm/contracts";
import { ToolExecutor, type TurnToolScope } from "../src/application/tool_executor";
import type { ToolImpl } from "../src/ports";

function setup() {
  const runs: string[] = [];
  const mk = (id: string, extra: Partial<ToolImpl["descriptor"]> = {}): ToolImpl => ({
    descriptor: { id, description: id, parameters: { type: "object", properties: { x: { type: "string" } } }, returns: "str", sideEffects: true, timeoutSec: 5, group: "t", ...extra },
    run: async (a) => {
      runs.push(`${id}:${String(a.x ?? "")}`);
      return "ok";
    },
  });
  const tools = [mk("t.write"), mk("t.click", { repeatable: true })];
  const exec = new ToolExecutor({
    registry: { list: () => tools.map((t) => t.descriptor), get: (id) => tools.find((t) => t.descriptor.id === id) },
    approvals: { request: async () => ({ approved: true, remember: "once" }) },
    settings: () => ToolSettings.parse({ workspaceAllow: ["*"] }),
    persistWorkspaceAllow: () => undefined,
    prompts: { get: () => "" } as never,
    checker: async () => {
      throw new Error("unused");
    },
  });
  const scope: TurnToolScope = {
    turnId: "t1",
    chatId: "c1",
    toolIds: new Set(tools.map((t) => t.descriptor.id)),
    ctx: { workspaceRoot: "/tmp", chatId: "c1", signal: new AbortController().signal } as never,
    conversation: () => "",
  };
  exec.beginToolTurn("t1");
  return { exec, scope, runs };
}

describe("ToolExecutor de-duplication", () => {
  it("collapses an immediate repeat of a side-effecting call", async () => {
    const { exec, scope, runs } = setup();
    await exec.execute("t.write", { x: "a" }, scope);
    const r = await exec.execute("t.write", { x: "a" }, scope);
    expect(r.deduped).toBe(true);
    expect(runs).toEqual(["t.write:a"]);
  });

  it("runs every repeat of a repeatable call, and a repeatable call resets the memory", async () => {
    const { exec, scope, runs } = setup();
    await exec.execute("t.write", { x: "a" }, scope);
    for (let i = 0; i < 3; i++) expect((await exec.execute("t.click", { x: "b" }, scope)).deduped).toBeUndefined();
    await exec.execute("t.write", { x: "a" }, scope);
    expect(runs).toEqual(["t.write:a", "t.click:b", "t.click:b", "t.click:b", "t.write:a"]);
  });
});
