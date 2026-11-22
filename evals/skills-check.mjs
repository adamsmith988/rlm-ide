/**
 * Launch the built IDE with a real project skill (SKILL.md + format + script)
 * and check the agent loads and follows it.
 *   node evals/skills-check.mjs
 */
import { createServer } from "node:http";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { IdeApp, REPO } from "./lib/app.mjs";
import { sleep } from "./lib/cdp.mjs";

const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
const dir = path.join(REPO, "evals/results", `skills-${stamp}`);
const ws = path.join(dir, "ws");
const checks = [];
const check = (name, ok, detail = "") => {
  checks.push({ name, ok: !!ok, detail: String(detail ?? "").slice(0, 600) });
  console.log(`  ${ok ? "PASS" : "FAIL"} ${name}${detail ? ` — ${String(detail).replace(/\s+/g, " ").slice(0, 240)}` : ""}`);
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
  if (visible.includes("SKILL_RECEIPT_OK") && visible.includes("receipts/today.txt")) return "Wrote receipts/today.txt using the receipt skill.";
  if (visible.includes("SKILL_RECEIPT_PROMPT") || visible.includes("Skill: receipt")) {
    return [
      "```repl",
      "listed = skills.list()",
      'print("SKILL_LIST", listed)',
      'spec = skills.read(name="receipt")',
      'print("SKILL_BODY", spec["body"][:400] if isinstance(spec, dict) else spec)',
      'fmt = skills.read_file(name="receipt", path="format.md")',
      'print("SKILL_FMT", fmt)',
      'stamp = shell.run(command="python .rlm/skills/receipt/scripts/stamp.py")',
      'print("SKILL_STAMP", stamp)',
      'note = "board lunch"',
      'date = (stamp.get("stdout") if isinstance(stamp, dict) else str(stamp)).strip().splitlines()[-1]',
      'body = f"SKILL_RECEIPT_OK\\ndate: {date}\\nnote: {note}\\n"',
      'print(fs.write_file(path="receipts/today.txt", content=body))',
      "```",
    ].join("\n");
  }
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
mkdirSync(path.join(ws, ".rlm", "skills", "receipt", "scripts"), { recursive: true });
writeFileSync(
  path.join(ws, ".rlm", "skills", "receipt", "SKILL.md"),
  `---
name: receipt
description: When the user asks for a receipt, write receipts/today.txt using format.md and scripts/stamp.py. Never answer with only chat text.
---
# Receipt

Follow every step. Do not skip the extra files.

1. Call skills.read_file(name="receipt", path="format.md") and use that layout exactly.
2. Run the stamp script from the project root:
   \`shell.run(command="python .rlm/skills/receipt/scripts/stamp.py")\`
3. Write \`receipts/today.txt\` with fs.write_file.
4. Line 1 must be \`SKILL_RECEIPT_OK\`.
5. Reply with only the path you wrote.
`,
);
writeFileSync(
  path.join(ws, ".rlm", "skills", "receipt", "format.md"),
  `# Receipt format

Line 1: SKILL_RECEIPT_OK
Line 2: date: <UTC stamp from scripts/stamp.py>
Line 3: note: <the user's note, one line>
`,
);
writeFileSync(
  path.join(ws, ".rlm", "skills", "receipt", "scripts", "stamp.py"),
  `from datetime import datetime, timezone
print(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
`,
);

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
  tools: { workspaceAllow: ["*"], autoApplyEdits: true, approvalMode: "workspace" },
  skills: [
    {
      id: "house-style",
      name: "house-style",
      description: "Writing conventions for notes in this workspace",
      body: "When you add a Markdown note, use ATX headings, wrap near 88 characters, and end with a line that starts with Sources:. Search the repo before inventing an API.",
      enabled: true,
    },
  ],
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

  const catalog = await waitFor(async () => {
    const list = await cdp.invoke("skills:list", {});
    return Array.isArray(list) && list.some((s) => s.name === "receipt") && list.some((s) => s.name === "house-style") ? list : null;
  }, 8000);
  check("project and user skills are listed", !!catalog, JSON.stringify(catalog));
  const receipt = catalog?.find((s) => s.name === "receipt");
  check("the receipt skill includes format.md and the stamp script", !!receipt?.files?.includes("format.md") && !!receipt?.files?.includes("scripts/stamp.py"), JSON.stringify(receipt?.files));

  await cdp.click('[data-testid=activity-settings]');
  await waitFor(async () => cdp.eval(`!!document.querySelector('[data-testid="settings-skills"]')`), 8000);
  await cdp.click('[data-testid=settings-skills]');
  const settingsUi = await waitFor(async () => {
    const info = await cdp.eval(`(() => ({
      text: document.body.innerText.toLowerCase(),
      project: [...document.querySelectorAll('[data-testid=project-skill]')].map((n) => n.innerText).join("\\n"),
      values: [...document.querySelectorAll("input, textarea")].map((el) => el.value).join("\\n"),
    }))()`);
    return info.text.includes("add skill") && info.project.toLowerCase().includes("receipt") ? info : null;
  }, 6000);
  check("settings shows the project skill and the house-style skill", !!settingsUi && settingsUi.values.includes("house-style") && settingsUi.values.includes("ATX headings"), settingsUi?.project?.slice(0, 180));
  await cdp.click('[data-testid=activity-explorer]');
  await sleep(300);

  await setComposer(cdp, "/rec");
  const menu = await waitFor(async () => {
    const text = await cdp.eval(`document.querySelector('[data-testid=slash-menu]')?.innerText ?? ""`);
    return /receipt/i.test(text) ? text : null;
  }, 6000);
  check("the slash menu lists the receipt skill", !!menu, menu?.replace(/\s+/g, " ").slice(0, 180));
  await cdp.eval(`(() => {
    const b = [...document.querySelectorAll('[data-testid=slash-menu] button')].find((x) => /receipt/i.test(x.textContent));
    if (!b) return;
    const key = Object.keys(b).find((k) => k.startsWith("__reactProps$"));
    b[key].onMouseDown({ preventDefault() {} });
  })()`);
  const chip = await waitFor(async () => {
    const text = await cdp.eval(`document.querySelector('[data-testid=skill-chip]')?.innerText ?? ""`);
    return text.includes("receipt") ? text : null;
  }, 4000);
  check("choosing the skill pins it on the composer", !!chip, chip || "no chip");

  await setComposer(cdp, "SKILL_RECEIPT_PROMPT Write a receipt for board lunch.");
  await cdp.click('button[title^="Send"]');
  const written = await waitFor(async () => {
    const file = await cdp.invoke("fs:readFile", { path: "receipts/today.txt" }).catch(() => null);
    return file?.content?.includes("SKILL_RECEIPT_OK") && file.content.includes("board lunch") ? file.content : null;
  }, 90000);
  await shot("skill-run");
  const trace = await waitFor(async () => {
    const chats = await cdp.invoke("chat:list", {});
    const chatId = chats[0]?.id;
    if (!chatId) return null;
    const msgs = await cdp.invoke("chat:messages", { chatId });
    const blob = JSON.stringify(msgs);
    return blob.includes("skills.read") && blob.includes("skills.read_file") ? blob : null;
  }, 8000);
  check("the agent called skills.read and skills.read_file", !!trace?.includes("skills.read") && !!trace?.includes("skills.read_file") && !!trace?.includes("format.md"), "skills.read + skills.read_file(format.md)");
  check("the agent loaded the skill procedure", !!trace?.includes("Follow every step") || !!trace?.includes("Never answer with only chat text"), "procedure text in the skill result");
  check("the agent ran the skill's stamp script", !!written && /date: 20\d{2}-\d{2}-\d{2}T/.test(written), written || "no stamp in the file");
  check("receipts/today.txt matches the skill format", !!written, written || "file missing");

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
