import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { applyEvent, settleApprovals, type UiMessage } from "../src/renderer/src/chat/model";
import { McpHub, parseEnv, splitArgs } from "../src/orchestrator/mcp";

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

describe("mcp stdio", () => {
  const dir = mkdtempSync(path.join(tmpdir(), "mcp-frames-"));
  const ndjson = path.join(dir, "ndjson.mjs");
  const lsp = path.join(dir, "lsp.mjs");
  writeFileSync(
    ndjson,
    `let buf = "";
process.stdin.on("data", (d) => {
  buf += d;
  let i;
  while ((i = buf.indexOf("\\n")) >= 0) {
    const line = buf.slice(0, i).trim();
    buf = buf.slice(i + 1);
    if (!line.startsWith("{")) continue;
    let msg;
    try { msg = JSON.parse(line); } catch { continue; }
    if (!msg.id) continue;
    const result = msg.method === "tools/list" ? { tools: [{ name: "ping", description: "p" }] }
      : msg.method === "tools/call" ? { content: [{ type: "text", text: "nd-pong" }] }
      : { protocolVersion: "2024-11-05", capabilities: {}, serverInfo: { name: "nd", version: "1" } };
    process.stdout.write(JSON.stringify({ jsonrpc: "2.0", id: msg.id, result }) + "\\n");
  }
});`,
  );
  writeFileSync(
    lsp,
    `import { Buffer } from "node:buffer";
let buf = Buffer.alloc(0);
process.stdin.on("data", (d) => {
  buf = Buffer.concat([buf, d]);
  for (;;) {
    const headerEnd = buf.indexOf("\\r\\n\\r\\n");
    if (headerEnd < 0) return;
    const header = buf.subarray(0, headerEnd).toString("utf8");
    const m = /Content-Length:\\s*(\\d+)/i.exec(header);
    if (!m) { buf = buf.subarray(headerEnd + 4); continue; }
    const len = Number(m[1]);
    const start = headerEnd + 4;
    if (buf.length < start + len) return;
    const msg = JSON.parse(buf.subarray(start, start + len).toString("utf8"));
    buf = buf.subarray(start + len);
    if (!msg.id) continue;
    const result = msg.method === "tools/list" ? { tools: [{ name: "ping", description: "p" }] }
      : msg.method === "tools/call" ? { content: [{ type: "text", text: "lsp-pong" }] }
      : { protocolVersion: "2024-11-05", capabilities: {}, serverInfo: { name: "lsp", version: "1" } };
    const body = Buffer.from(JSON.stringify({ jsonrpc: "2.0", id: msg.id, result }));
    process.stdout.write(\`Content-Length: \${body.length}\\r\\n\\r\\n\`);
    process.stdout.write(body);
  }
});`,
  );

  async function use(name: string, script: string, expectText: string) {
    const hub = new McpHub(() => undefined);
    try {
      await hub.sync([{ id: name, name, command: process.execPath, args: script, env: "", enabled: true }]);
      expect(hub.status()[0]?.tools.map((t) => t.name)).toEqual(["ping"]);
      expect(await hub.call(name, "ping", {})).toBe(expectText);
    } finally {
      hub.stopAll();
    }
  }

  it("talks to a newline-framed server and a Content-Length server", async () => {
    await use("nd", ndjson, "nd-pong");
    await use("lsp", lsp, "lsp-pong");
  });

  afterAll(() => rmSync(dir, { recursive: true, force: true }));
});
