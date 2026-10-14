import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { createBuiltinTools, ProcessManager } from "../src";

const ws = mkdtempSync(path.join(tmpdir(), "rlm-shell-"));
const ctx = { workspaceRoot: ws, signal: new AbortController().signal } as never;

function tools(cellTimeoutSec?: number) {
  const pm = new ProcessManager();
  const all = createBuiltinTools({ processes: pm, shellTimeoutSec: () => 300, cellTimeoutSec: cellTimeoutSec ? () => cellTimeoutSec : undefined, editorState: () => ({}), searchSession: async () => [] });
  const get = (id: string) => all.find((t) => t.descriptor.id === id)!;
  return { pm, get };
}

describe.skipIf(process.platform === "win32")("shell tools", () => {
  it("shell.wait returns when a background process exits", async () => {
    const { pm, get } = tools();
    const started = (await get("shell.run").run({ command: "sleep 1; echo finished", background: true, wait_sec: 0 }, ctx)) as { pid: number; running: boolean };
    expect(started.running).toBe(true);
    const t0 = Date.now();
    const r = (await get("shell.wait").run({ pid: started.pid, timeout_sec: 20 }, ctx)) as { running: boolean; exit_code: number; output_tail: string };
    expect(Date.now() - t0).toBeLessThan(5000);
    expect(r.running).toBe(false);
    expect(r.exit_code).toBe(0);
    expect(r.output_tail).toContain("finished");
    pm.killAll();
  });

  it("shell.wait gives up at its timeout while the process keeps running", async () => {
    const { pm, get } = tools();
    const started = (await get("shell.run").run({ command: "sleep 30", background: true, wait_sec: 0 }, ctx)) as { pid: number };
    const r = (await get("shell.wait").run({ pid: started.pid, timeout_sec: 1 }, ctx)) as { running: boolean };
    expect(r.running).toBe(true);
    pm.killAll();
  });

  it("foreground commands stay inside the cell time limit", async () => {
    const { get } = tools(20);
    const t0 = Date.now();
    const r = (await get("shell.run").run({ command: "sleep 30", timeout_sec: 60 }, ctx)) as { timed_out: boolean; note?: string };
    expect(Date.now() - t0).toBeLessThan(10000);
    expect(r.timed_out).toBe(true);
    expect(r.note).toMatch(/background=True/);
  }, 15000);
});
