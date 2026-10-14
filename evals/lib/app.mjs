import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { createWriteStream, existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Cdp, sleep } from "./cdp.mjs";

const here = path.dirname(fileURLToPath(import.meta.url));
export const REPO = path.resolve(here, "../..");
export const DESKTOP = path.join(REPO, "apps/desktop");

export function electronBinary() {
  return createRequire(path.join(DESKTOP, "package.json"))("electron");
}

/** Launches the built IDE (apps/desktop/out) with an isolated profile and drives it over CDP. */
export class IdeApp {
  constructor({ userData, workspace, settings, port = 9400 + Math.floor(Math.random() * 400), logFile }) {
    this.opts = { userData, workspace, settings, port, logFile };
  }

  async start() {
    if (!existsSync(path.join(DESKTOP, "out/main/index.js"))) throw new Error("build the app first: pnpm --filter @rlm/desktop build");
    const args = [`--remote-debugging-port=${this.opts.port}`, "."];
    if (process.platform === "linux") args.unshift("--no-sandbox");
    this.proc = spawn(electronBinary(), args, {
      cwd: DESKTOP,
      env: {
        ...process.env,
        RLM_USER_DATA: this.opts.userData,
        RLM_OPEN_WORKSPACE: this.opts.workspace,
        RLM_SETTINGS_JSON: JSON.stringify(this.opts.settings ?? {}),
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
    if (this.opts.logFile) {
      const log = createWriteStream(this.opts.logFile);
      this.proc.stdout.pipe(log);
      this.proc.stderr.pipe(log);
    }
    this.cdp = await Cdp.connect(this.opts.port, "index.html", 60000);
    await this.cdp.send("Runtime.enable");
    // Wait until the workspace is open and the composer is ready.
    for (let i = 0; i < 100; i++) {
      const ok = await this.cdp.eval("!!document.getElementById('composer') && !document.getElementById('composer').disabled").catch(() => false);
      if (ok) return this;
      await sleep(300);
    }
    throw new Error("IDE did not open the workspace");
  }

  async newChat() {
    await this.cdp.click('button[title^="New chat"]');
    await sleep(400);
  }

  /** Types a prompt into the composer like a user and waits for the turn to finish. */
  async ask(prompt, { timeoutMs = 15 * 60 * 1000, onApproval } = {}) {
    const before = await this.cdp.invoke("chat:list");
    // The window is visible on the desktop; if Settings or Memory got opened, go back like a user would.
    if (!(await this.cdp.eval("!!document.getElementById('composer')"))) {
      await this.cdp.click('button[title="Explorer"]').catch(() => undefined);
      await sleep(400);
    }
    await this.cdp.typeInto("#composer", prompt, { enter: true });
    const t0 = Date.now();
    let chatId = null;
    let turnId = null;
    let retried = false;
    while (!turnId) {
      await sleep(300);
      const chats = await this.cdp.invoke("chat:list");
      chatId = chats[0]?.id ?? null;
      if (chatId) turnId = await this.cdp.invoke("chat:activeTurn", { chatId });
      if (!turnId && chatId) {
        const msgs = await this.cdp.invoke("chat:messages", { chatId });
        const last = msgs[msgs.length - 1];
        if (last?.role === "assistant" && last.text !== undefined && msgs.length && last.createdAt >= t0 - 2000 && last.status !== "running") {
          turnId = last.turnId;
        }
      }
      if (!turnId && !retried && Date.now() - t0 > 5000) {
        // Enter can be lost if the window was busy; a user would press Send.
        retried = true;
        await this.cdp.click('button[title^="Send"]').catch(() => undefined);
      }
      if (Date.now() - t0 > 30000) {
        const ui = await this.cdp
          .eval(`JSON.stringify({ composer: document.getElementById('composer')?.value?.slice(0, 80), disabled: document.getElementById('composer')?.disabled, placeholder: document.getElementById('composer')?.placeholder, toasts: [...document.querySelectorAll('[role=status], .toast')].map((t) => t.innerText).join(' | ') })`)
          .catch((e) => String(e));
        if (this.opts.logFile) await this.screenshot(this.opts.logFile.replace(/\.log$/, "-stuck.png")).catch(() => undefined);
        throw new Error(`turn did not start (chats before: ${before.length}) ui=${ui} console=${JSON.stringify(this.cdp.console.slice(-5))}`);
      }
    }
    const handled = new Set();
    for (;;) {
      const active = await this.cdp.invoke("chat:activeTurn", { chatId });
      if (!active) break;
      if (onApproval) {
        const evs = await this.cdp.invoke("chat:subscribe", { turnId, lastSeq: -1 });
        const resolved = new Set(evs.filter((e) => e.type === "approval_resolved").map((e) => e.approvalId));
        for (const e of evs) {
          if (e.type === "approval_request" && !resolved.has(e.approvalId) && !handled.has(e.approvalId)) {
            handled.add(e.approvalId);
            const approved = await onApproval(e);
            await this.cdp.invoke("chat:approve", { approvalId: e.approvalId, approved, remember: "once" });
          }
        }
      }
      if (Date.now() - t0 > timeoutMs) {
        await this.cdp.invoke("chat:stop", { turnId });
        break;
      }
      await sleep(1000);
    }
    await sleep(500);
    const msgs = await this.cdp.invoke("chat:messages", { chatId });
    const msg = msgs.find((m) => m.turnId === turnId && m.role === "assistant");
    return { chatId, turnId, status: msg?.status, text: msg?.text ?? "", events: msg?.events ?? [], durationMs: Date.now() - t0 };
  }

  async screenshot(file) {
    await this.cdp.screenshot(file);
  }

  async stop() {
    this.cdp?.close();
    if (this.proc && this.proc.exitCode === null) {
      this.proc.kill("SIGTERM");
      await Promise.race([new Promise((r) => this.proc.once("exit", r)), sleep(5000)]);
      if (this.proc.exitCode === null) this.proc.kill("SIGKILL");
    }
  }
}

/** Summarize a turn's event stream into metrics. */
export function turnMetrics(r) {
  const ev = r.events;
  const count = (t) => ev.filter((e) => e.type === t).length;
  const toolResults = ev.filter((e) => e.type === "tool_result");
  const cells = ev.filter((e) => e.type === "cell_done");
  return {
    status: r.status,
    durationSec: Math.round(r.durationMs / 100) / 10,
    cells: cells.length,
    cellErrors: cells.filter((c) => !c.ok).length,
    toolCalls: count("tool_call"),
    toolErrors: toolResults.filter((t) => !t.ok).length,
    approvals: count("approval_request"),
    compactions: count("compaction"),
    edits: count("pending_edit"),
    subagents: count("subagent_started"),
    thinkingChars: ev.filter((e) => e.type === "thinking_delta").reduce((n, e) => n + e.text.length, 0),
    answerChars: r.text.length,
    toolsUsed: [...new Set(ev.filter((e) => e.type === "tool_call").map((e) => e.call.toolId))],
  };
}
