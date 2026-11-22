/**
 * Launch the built IDE and check the chat, browser, git, skills, sub-agent,
 * approval, and project-memory behavior against the running window.
 *   node evals/feature-check.mjs
 */
import { createServer } from "node:http";
import { execFileSync } from "node:child_process";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { IdeApp, REPO } from "./lib/app.mjs";
import { sleep } from "./lib/cdp.mjs";

const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
const dir = path.join(REPO, "evals/results", `features-${stamp}`);
const ws = path.join(dir, "ws");
const other = path.join(dir, "other");
const checks = [];
const check = (name, ok, detail = "") => {
  checks.push({ name, ok: !!ok, detail: String(detail ?? "") });
  console.log(`  ${ok ? "PASS" : "FAIL"} ${name}${detail ? ` — ${detail}` : ""}`);
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
  // The system prompt mentions "[REPL output]" as instructions. Only the conversation after it counts.
  const visible = messages.filter((m) => m.role !== "system").map((m) => textOf(m.content)).join("\n");
  const last = textOf([...messages].reverse().find((m) => m.role === "user")?.content);
  if (last.includes("subagent finished the marker") && !last.includes("FEATURE_CHECK_PROMPT")) return "subagent finished the marker";
  if (visible.includes("FEATURE_OUTPUT")) return "All done. FEATURE_DONE";
  if (visible.includes("FEATURE_CHECK_PROMPT")) {
    return [
      "```repl",
      'print("FEATURE_OUTPUT")',
      'print(spawn_agents(tasks=["Reply with exactly this sentence and no code: subagent finished the marker"])[0])',
      'fs.write_file(path="out.txt", content="x")',
      "```",
    ].join("\n");
  }
  return "All done.";
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
const api = `http://127.0.0.1:${port}/v1`;

rmSync(dir, { recursive: true, force: true });
mkdirSync(path.join(ws, "site"), { recursive: true });
mkdirSync(other, { recursive: true });
writeFileSync(
  path.join(ws, "site/index.html"),
  `<!doctype html><html><head><title>Counter Demo</title></head><body><h1>Counter Demo</h1><script>console.log("counter ready")</script></body></html>`,
);
writeFileSync(path.join(ws, "keep.txt"), "hello\n");
writeFileSync(path.join(ws, "gone.txt"), "gone\n");
execFileSync("git", ["init", "-q"], { cwd: ws });
execFileSync("git", ["add", "keep.txt", "gone.txt"], { cwd: ws });
execFileSync("git", ["-c", "user.email=feature@check.local", "-c", "user.name=Feature Check", "commit", "-qm", "base"], { cwd: ws });
writeFileSync(path.join(ws, "keep.txt"), "changed\n");
writeFileSync(path.join(ws, "added.txt"), "new\n");
rmSync(path.join(ws, "gone.txt"));

const settings = {
  providers: [
    { id: "builtin", kind: "local_onnx", label: "Built-in (local)", baseUrl: "", hasKey: false },
    { id: "eval-api", kind: "openai_compat", label: "Eval API", baseUrl: api, hasKey: true },
  ],
  models: [
    { id: "builtin:bge-m3", providerId: "builtin", name: "bge-m3", label: "bge-m3 (local)", kind: "embed", contextLength: 8192, maxOutputTokens: null, dim: 1024 },
    { id: "eval-api:feature-check", providerId: "eval-api", name: "feature-check", label: "feature-check", kind: "chat", contextLength: 8192, maxOutputTokens: 1024, dim: null },
  ],
  roles: { agent: "eval-api:feature-check", summarizer: "", embed: "builtin:bge-m3" },
  tools: { workspaceAllow: [], autoApplyEdits: false, approvalMode: "ask" },
  skills: [
    {
      id: "house-style",
      name: "house-style",
      description: "Writing conventions for notes in this workspace",
      body: "When you add a Markdown note, use ATX headings, wrap near 88 characters, and end with a line that starts with Sources:. Search the repo before inventing an API.",
      enabled: true,
    },
  ],
  mcp: [{ id: "mcp-check", name: "checkmcp", command: "/bin/true", args: "", env: "", enabled: false }],
  autoUpdate: false,
};
process.env.RLM_API_KEY = "sk-feature-check";
process.env.RLM_API_PROVIDER = "eval-api";

const app = new IdeApp({ userData: path.join(dir, "userdata"), workspace: ws, settings, logFile: path.join(dir, "app.log") });

async function waitFor(fn, timeoutMs = 20000, every = 300) {
  const until = Date.now() + timeoutMs;
  let last = null;
  for (;;) {
    last = await fn().catch((e) => String(e));
    if (last) return last;
    if (Date.now() > until) return null;
    await sleep(every);
  }
}

async function main() {
  await app.start();
  const cdp = app.cdp;
  const shot = (name) => cdp.screenshot(path.join(dir, `${name}.png`)).catch(() => undefined);

  console.log("▶ composer, git, settings");
  const chrome = await cdp.eval(`(() => ({
    attach: !!document.querySelector('[data-testid=attach]'),
    exportMd: !!document.querySelector('button[title="Export this chat as Markdown"]'),
    backup: !!document.querySelector('button[title="Backup this chat as JSON"]'),
    agentTab: [...document.querySelectorAll("button")].some((b) => b.textContent.trim() === "Agent"),
  }))()`);
  check("attach button is in the composer", chrome.attach);
  check("export and backup buttons are in the chat header", chrome.exportMd && chrome.backup);
  check("agent output tab is in the bottom panel", chrome.agentTab);

  await cdp.eval(`(() => {
    const el = document.getElementById("composer");
    const set = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
    set.call(el, "/");
    el.dispatchEvent(new Event("input", { bubbles: true }));
  })()`);
  await sleep(200);
  const slash = await cdp.eval(`(() => {
    const menu = document.querySelector('[data-testid=slash-menu]');
    return menu ? menu.innerText : "";
  })()`);
  check("attach menu lists plan, files, and MCP", /Plan/.test(slash) && /Files/.test(slash) && /MCP/.test(slash), slash.replace(/\s+/g, " ").slice(0, 180));
  await cdp.eval(`(() => {
    const el = document.getElementById("composer");
    const set = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
    set.call(el, "");
    el.dispatchEvent(new Event("input", { bubbles: true }));
  })()`);

  const git = await waitFor(async () => {
    const marks = await cdp.eval(`(() => Object.fromEntries([...document.querySelectorAll("[data-git]")].map((n) => [n.textContent.trim(), n.getAttribute("data-git")])))()`);
    return marks["keep.txt"] && marks["added.txt"] && marks["gone.txt"] ? marks : null;
  }, 8000);
  check("modified file is marked changed", git?.["keep.txt"] === "mod", JSON.stringify(git));
  check("new file is green", git?.["added.txt"] === "add", JSON.stringify(git));
  check("deleted file is red in the sidebar", git?.["gone.txt"] === "del", JSON.stringify(git));

  await cdp.click('[data-testid=activity-settings]');
  const skillsProbe = await waitFor(async () => {
    const probe = await cdp.eval(`(() => {
      const quoted = document.querySelector('[data-testid="settings-skills"]');
      const bare = document.querySelector('[data-testid=settings-skills]');
      const byText = [...document.querySelectorAll("button")].find((b) => b.textContent.trim() === "Skills & MCP");
      return {
        quoted: !!quoted,
        bare: !!bare,
        text: !!byText,
        heading: document.querySelector("h1")?.textContent ?? "",
        buttons: [...document.querySelectorAll("button")].map((b) => b.textContent.trim()).filter(Boolean).slice(0, 24),
      };
    })()`);
    return probe.quoted || probe.bare || probe.text || probe.heading === "Settings" ? probe : null;
  }, 8000);
  if (skillsProbe?.quoted || skillsProbe?.bare || skillsProbe?.text) {
    await cdp.eval(`(() => {
      const el = document.querySelector('[data-testid="settings-skills"]')
        || [...document.querySelectorAll("button")].find((b) => b.textContent.trim() === "Skills & MCP");
      el?.click();
    })()`);
  }
  const skillsUi = await waitFor(async () => {
    const info = await cdp.eval(`(() => ({
      text: document.body.innerText.toLowerCase(),
      values: [...document.querySelectorAll("input, textarea")].map((el) => el.value).join("\\n"),
    }))()`);
    return info.text.includes("add skill") ? info : null;
  }, 4000);
  if (!skillsUi) await shot("skills-miss");
  const saved = await cdp.invoke("settings:get", {}).catch((e) => ({ error: String(e) }));
  const text = skillsUi?.text ?? "";
  const values = skillsUi?.values ?? "";
  check(
    "settings shows the skill and MCP server",
    !!skillsUi && values.includes("house-style") && values.includes("checkmcp") && text.includes("add mcp server"),
    skillsUi
      ? `saved skills=${JSON.stringify((saved.skills ?? []).map((s) => s.name))}`
      : `settings screen did not show the skills section; probe=${JSON.stringify(skillsProbe)}`,
  );
  await cdp.eval(`(() => { const b = [...document.querySelectorAll("button")].find((x) => x.textContent.trim() === "Memory"); b?.click(); })()`);
  await sleep(200);
  const memoryUi = await cdp.eval(`document.body.innerText`);
  check("memory settings say projects stay separate", memoryUi.includes("Each project keeps its own chats") && memoryUi.includes("Always recall"));
  await cdp.click('[data-testid=activity-explorer]');
  await sleep(300);

  console.log("▶ browser");
  await cdp.click('[data-testid=toggle-browser]');
  await sleep(400);
  await cdp.invoke("browser:navigate", { url: "site/index.html" });
  await sleep(500);
  const opened = await cdp.invoke("browser:tab", { action: "new" });
  check("browser opens a second tab", Array.isArray(opened.tabs) && opened.tabs.length >= 2, `tabs=${opened.tabs?.length}`);
  const first = opened.tabs?.find((t) => t.url.includes("index.html")) ?? opened.tabs?.[0];
  if (first) await cdp.invoke("browser:tab", { action: "select", id: first.id });
  const consoleLines = await waitFor(async () => {
    const lines = await cdp.invoke("browser:console", {});
    return lines.some((l) => String(l.text).includes("counter ready")) ? lines : null;
  }, 8000);
  check("browser console captured the page log", !!consoleLines, JSON.stringify(consoleLines)?.slice(0, 240));
  await cdp.click('[data-testid=browser-console-toggle]');
  await sleep(200);
  const consoleUi = await cdp.eval(`document.querySelector('[data-testid=browser-console]')?.innerText ?? ""`);
  check("console drawer shows the log and a copy button", consoleUi.includes("counter ready") && !!await cdp.eval(`!!document.querySelector('[data-testid=browser-console] button[title="Copy console"]')`), consoleUi.slice(0, 160));
  await cdp.click('[data-testid=toggle-browser]');

  console.log("▶ running turn");
  await cdp.eval(`(() => {
    const el = document.getElementById("composer");
    const set = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
    set.call(el, "FEATURE_CHECK_PROMPT");
    el.dispatchEvent(new Event("input", { bubbles: true }));
  })()`);
  await cdp.click('button[title^="Send"]');
  const running = await waitFor(async () => {
    const ui = await cdp.eval(`(() => ({
      output: document.querySelector('[data-testid=agent-output] pre')?.innerText ?? "",
      sub: document.querySelector('[data-testid=subagent]')?.innerText ?? "",
      approval: !!document.querySelector('[data-testid=approval] button:not([disabled])'),
      copyUser: !!document.querySelector('button[title="Copy message"]'),
    }))()`);
    return ui.output.includes("FEATURE_OUTPUT") && ui.sub.includes("subagent finished the marker") && ui.approval ? ui : null;
  }, 120000);
  check("agent output shows the running cell", !!running?.output.includes("FEATURE_OUTPUT") || !!running?.output.includes("print"), running?.output?.slice(0, 180) ?? "timed out");
  check("sub-agent card shows the running task", !!running?.sub.includes("Sub-agent"), running?.sub?.slice(0, 180) ?? "timed out");
  check("approval buttons are enabled while the question is open", !!running?.approval);
  check("the user message has a copy button", !!running?.copyUser);

  if (running) {
    const chatId = (await cdp.invoke("chat:list", {}))[0]?.id;
    const turnId = chatId ? await cdp.invoke("chat:activeTurn", { chatId }) : null;
    if (turnId) await cdp.invoke("chat:stop", { turnId });
    const skipped = await waitFor(async () => {
      const ui = await cdp.eval(`(() => {
        const card = document.querySelector('[data-testid=approval]');
        const buttons = [...(card?.querySelectorAll("button") ?? [])];
        return { text: card?.innerText ?? "", disabled: buttons.length > 0 && buttons.every((b) => b.disabled), copyAnswer: !!document.querySelector('button[title="Copy answer"]') };
      })()`);
      return ui.disabled && /Skipped/.test(ui.text) ? ui : null;
    }, 10000);
    check("stopping the turn skips the question and disables its buttons", !!skipped, skipped?.text?.slice(0, 120) ?? "still active");
    check("the agent answer has a copy button", !!skipped?.copyAnswer);
    const exported = chatId ? await cdp.invoke("chat:export", { chatId }).catch((e) => ({ error: String(e) })) : { error: "no chat" };
    check("chat export contains the prompt and a JSON backup", typeof exported.markdown === "string" && exported.markdown.includes("FEATURE_CHECK_PROMPT") && exported.backup?.includes("FEATURE_CHECK_PROMPT"));
    await cdp.click('[data-testid=subagent-chat]');
    await sleep(300);
    const subchat = await cdp.eval(`document.querySelector('[data-testid=subchat]')?.innerText ?? ""`);
    check("sub-chat opens for that sub-agent", subchat.includes("Sub-chat") && subchat.includes("subagent finished the marker"), subchat.slice(0, 160));
  } else {
    check("stopping the turn skips the question and disables its buttons", false, "turn never reached the approval");
    check("the agent answer has a copy button", false, "turn never finished");
    check("chat export contains the prompt and a JSON backup", false, "no finished chat");
    check("sub-chat opens for that sub-agent", false, "no sub-agent card");
    await shot("turn-stuck");
  }

  console.log("▶ project isolation");
  const chatA = (await cdp.invoke("chat:list", {}))[0]?.id;
  await cdp.invoke("memory:import", { json: JSON.stringify([{ text: "PROJECT_A_ONLY fact", about: "project" }, { text: "USER_GLOBAL fact", about: "user" }]) });
  await cdp.invoke("workspace:open", { root: other });
  const chatsB = await cdp.invoke("chat:list", {});
  const factsB = await cdp.invoke("memory:list", {});
  const leakedMessages = await cdp.invoke("chat:messages", { chatId: chatA }).then(() => true).catch(() => false);
  const leakedExport = await cdp.invoke("chat:export", { chatId: chatA }).then(() => true).catch(() => false);
  check("the other project does not list this project's chat", !chatsB.some((c) => c.id === chatA), `chats=${chatsB.length}`);
  check("the other project cannot read or export this project's chat", !leakedMessages && !leakedExport);
  check("a project fact does not appear in the other project", !factsB.some((f) => f.text === "PROJECT_A_ONLY fact"), factsB.map((f) => f.text).join(", "));
  check("a fact about the user is still there", factsB.some((f) => f.text === "USER_GLOBAL fact"));
  await cdp.invoke("memory:forgetAll", {});
  const afterForgetB = await cdp.invoke("memory:list", {});
  check("clearing the other project keeps the fact about the user", afterForgetB.some((f) => f.text === "USER_GLOBAL fact") && !afterForgetB.some((f) => f.text === "PROJECT_A_ONLY fact"));
  await cdp.invoke("workspace:open", { root: ws });
  const factsA = await cdp.invoke("memory:list", {});
  check("the first project's fact is still there after the other project was cleared", factsA.some((f) => f.text === "PROJECT_A_ONLY fact") && factsA.some((f) => f.text === "USER_GLOBAL fact"), factsA.map((f) => `${f.text}:${f.workspaceId ? "project" : "user"}`).join(", "));
  await cdp.invoke("memory:forgetAll", {});
  const clearedA = await cdp.invoke("memory:list", {});
  check("clearing this project removes its fact and keeps the fact about the user", !clearedA.some((f) => f.text === "PROJECT_A_ONLY fact") && clearedA.some((f) => f.text === "USER_GLOBAL fact"), clearedA.map((f) => f.text).join(", "));

  await shot("final");
  const failed = checks.filter((c) => !c.ok);
  writeFileSync(path.join(dir, "checks.json"), JSON.stringify({ passed: checks.length - failed.length, failed: failed.length, checks }, null, 2));
  console.log(`\n${checks.length - failed.length}/${checks.length} passed`);
  if (failed.length) process.exitCode = 1;
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
