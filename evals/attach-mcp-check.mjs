/**
 * Launch the built IDE, attach a project file and a pasted image, then call the
 * ModelScope MCP server that was installed for this check.
 *   node evals/attach-mcp-check.mjs
 */
import { createServer } from "node:http";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { IdeApp, REPO } from "./lib/app.mjs";
import { sleep } from "./lib/cdp.mjs";

const SERVER = "/tmp/modelscope-mcp-venv/bin/modelscope-mcp-server";
const PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
const dir = path.join(REPO, "evals/results", `attach-mcp-${stamp}`);
const ws = path.join(dir, "ws");
const checks = [];
const check = (name, ok, detail = "") => {
  checks.push({ name, ok: !!ok, detail: String(detail ?? "").slice(0, 500) });
  console.log(`  ${ok ? "PASS" : "FAIL"} ${name}${detail ? ` — ${String(detail).replace(/\s+/g, " ").slice(0, 220)}` : ""}`);
};

function sse(res, text) {
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: text } }] })}\n\n`);
  res.end("data: [DONE]\n\n");
}

function textOf(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map((p) => (typeof p === "string" ? p : p?.text || "")).join("\n");
  return "";
}

function replyFor(messages) {
  const visible = messages.filter((m) => m.role !== "system").map((m) => textOf(m.content)).join("\n");
  if (visible.includes("MCP_SEARCH") && visible.includes("Qwen")) return "MCP tools finished.";
  if (visible.includes("MCP_CHECK_PROMPT")) {
    return [
      "```repl",
      "listed = mcp.list()",
      'print("MCP_LIST", listed)',
      'env = mcp.call(server="modelscope", name="get_environment_info", arguments={})',
      'print("MCP_ENV", env)',
      'found = mcp.call(server="modelscope", name="search_models", arguments={"query": "qwen", "limit": 1})',
      'print("MCP_SEARCH", found)',
      "```",
    ].join("\n");
  }
  if (visible.includes("ATTACH_FILE_MARKER")) return "Saw the attached file.";
  return "ok";
}

const server = createServer((req, res) => {
  if (req.url === "/v1/models") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ data: [{ id: "feature-check", max_model_len: 8192 }] }));
    return;
  }
  if (req.url !== "/v1/chat/completions") {
    res.writeHead(404);
    res.end();
    return;
  }
  const chunks = [];
  req.on("data", (c) => chunks.push(c));
  req.on("end", () => {
    let messages = [];
    try {
      messages = JSON.parse(Buffer.concat(chunks).toString("utf8")).messages ?? [];
    } catch {
      /* empty */
    }
    sse(res, replyFor(messages));
  });
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;

rmSync(dir, { recursive: true, force: true });
mkdirSync(ws, { recursive: true });
writeFileSync(path.join(ws, "attach-me.txt"), "ATTACH_FILE_MARKER\nThe attached note is ready.\n");

const settings = {
  providers: [
    { id: "builtin", kind: "local_onnx", label: "Built-in (local)", baseUrl: "", hasKey: false },
    { id: "eval-api", kind: "openai_compat", label: "Eval API", baseUrl: `http://127.0.0.1:${port}/v1`, hasKey: true },
  ],
  models: [
    { id: "builtin:bge-m3", providerId: "builtin", name: "bge-m3", label: "bge-m3 (local)", kind: "embed", contextLength: 8192, maxOutputTokens: null, dim: 1024 },
    { id: "eval-api:feature-check", providerId: "eval-api", name: "feature-check", label: "feature-check", kind: "chat", contextLength: 8192, maxOutputTokens: 1024, dim: null },
  ],
  roles: { agent: "eval-api:feature-check", summarizer: "", embed: "builtin:bge-m3" },
  tools: { workspaceAllow: ["mcp.call"], autoApplyEdits: false, approvalMode: "ask" },
  mcp: [{ id: "modelscope", name: "modelscope", command: SERVER, args: "", env: "", enabled: true }],
  autoUpdate: false,
};
process.env.RLM_API_KEY = "sk-feature-check";
process.env.RLM_API_PROVIDER = "eval-api";

const app = new IdeApp({ userData: path.join(dir, "userdata"), workspace: ws, settings, logFile: path.join(dir, "app.log") });

async function waitFor(fn, timeoutMs = 20000, every = 300) {
  const until = Date.now() + timeoutMs;
  for (;;) {
    const last = await fn().catch(() => null);
    if (last) return last;
    if (Date.now() > until) return null;
    await sleep(every);
  }
}

async function setComposer(cdp, value) {
  await cdp.eval(`(() => {
    const el = document.getElementById("composer");
    const set = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
    set.call(el, ${JSON.stringify(value)});
    el.dispatchEvent(new Event("input", { bubbles: true }));
  })()`);
}

async function main() {
  await app.start();
  const cdp = app.cdp;
  const shot = (name) => cdp.screenshot(path.join(dir, `${name}.png`)).catch(() => undefined);

  console.log("▶ attach a file and an image");
  await setComposer(cdp, "/attach-me");
  const fileRow = await waitFor(async () => {
    const label = await cdp.eval(`(() => {
      const b = [...document.querySelectorAll('[data-testid=slash-menu] button')].find((x) => x.textContent.includes("attach-me.txt"));
      if (!b) return "";
      const key = Object.keys(b).find((k) => k.startsWith("__reactProps$"));
      b[key].onMouseDown({ preventDefault() {} });
      return b.textContent.trim();
    })()`);
    return label || null;
  }, 8000);
  check("the attach menu can pick a project file", !!fileRow, fileRow || "file row did not appear");
  const chip = await waitFor(async () => {
    const text = await cdp.eval(`document.querySelector('[data-testid=attachments]')?.innerText ?? ""`);
    return text.includes("attach-me.txt") ? text : null;
  }, 4000);
  check("the file shows up as an attachment chip", !!chip, chip || "no chip");

  const pasted = await cdp.eval(`(() => new Promise((resolve) => {
    const b64 = ${JSON.stringify(PNG)};
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const file = new File([bytes], "pasted.png", { type: "image/png" });
    const dt = new DataTransfer();
    dt.items.add(file);
    const el = document.getElementById("composer");
    el.dispatchEvent(new ClipboardEvent("paste", { bubbles: true, cancelable: true, clipboardData: dt }));
    setTimeout(() => resolve(document.querySelector('[data-testid=attachments] img')?.getAttribute("src")?.slice(0, 30) ?? ""), 400);
  }))()`);
  check("pasting an image attaches it", String(pasted).startsWith("data:image/png"), String(pasted));
  await shot("attachments");

  await setComposer(cdp, "Read the attachment");
  await cdp.click('button[title^="Send"]');
  const sent = await waitFor(async () => {
    const chats = await cdp.invoke("chat:list", {});
    const chatId = chats[0]?.id;
    if (!chatId) return null;
    const msgs = await cdp.invoke("chat:messages", { chatId });
    const user = msgs.find((m) => m.role === "user");
    return user?.text?.includes("Attached image") || user?.text?.includes("ATTACH_FILE_MARKER") ? user.text : null;
  }, 15000);
  check("the sent message contains the file text", !!sent?.includes("ATTACH_FILE_MARKER"), sent?.slice(0, 180) ?? "message missing");
  check("the sent message contains the pasted image", !!sent?.includes("Attached image") && !!sent?.includes("pasted.png") && !!sent?.includes("data:image/png"), sent?.includes("pasted.png") ? "pasted.png is in the message" : "image markdown missing");
  const answered = await waitFor(async () => {
    const text = await cdp.eval(`document.body.innerText`);
    return text.includes("Saw the attached file") ? text : null;
  }, 30000);
  check("the agent received the attachment", !!answered);

  console.log("▶ ModelScope MCP server");
  await cdp.click('[data-testid=activity-settings]');
  await waitFor(async () => cdp.eval(`!!document.querySelector('[data-testid="settings-skills"]')`), 8000);
  await cdp.click('[data-testid=settings-skills]');
  const mcpUi = await waitFor(async () => {
    const values = await cdp.eval(`[...document.querySelectorAll("input, textarea")].map((el) => el.value).join("\\n")`);
    return values.includes("modelscope-mcp-server") && values.includes("modelscope") ? values : null;
  }, 8000);
  check("settings has the ModelScope server attached", !!mcpUi, mcpUi ? "command and name are filled in" : "server fields were not on the skills screen");
  await cdp.click('[data-testid=activity-explorer]');
  await sleep(300);

  await setComposer(cdp, "MCP_CHECK_PROMPT");
  await cdp.click('button[title^="Send"]');
  const output = await waitFor(async () => {
    const text = await cdp.eval(`document.querySelector('[data-testid=agent-output] pre')?.innerText ?? ""`);
    return text.includes("MCP_SEARCH") && text.includes("Qwen") && text.includes("modelscope.cn") ? text : null;
  }, 90000);
  await shot("mcp-output");
  check("the agent listed the ModelScope server and read its environment", !!output?.includes("modelscope.cn") && !!output?.includes("0.2.1"), output?.slice(output.indexOf("MCP_ENV"), output.indexOf("MCP_ENV") + 180) || "no environment result");
  check("the agent searched ModelScope and got a Qwen model", !!output?.includes("MCP_SEARCH") && !!output?.includes("Qwen"), output?.slice(output.indexOf("MCP_SEARCH"), output.indexOf("MCP_SEARCH") + 180) || "no search result");

  writeFileSync(path.join(dir, "checks.json"), JSON.stringify({ passed: checks.filter((c) => c.ok).length, failed: checks.filter((c) => !c.ok).length, checks }, null, 2));
  console.log(`\n${checks.filter((c) => c.ok).length}/${checks.length} passed`);
  if (checks.some((c) => !c.ok)) process.exitCode = 1;
}

try {
  await main();
} catch (e) {
  console.error(e);
  process.exitCode = 1;
} finally {
  await app.stop().catch(() => undefined);
  server.close();
}
