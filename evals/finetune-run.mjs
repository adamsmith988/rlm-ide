/**
 * Long agent task, hands off: the agent downloads a small Hugging Face model, schedules a pipeline
 * (load + smoke test, synthetic dataset, fine-tune, evaluate), fixes its own errors, and ends with a
 * canvas in the chat. This script only types the prompt, accepts approvals like a user clicking Allow,
 * and records what happens. It never edits the workspace or talks to the agent again.
 *   node evals/finetune-run.mjs        (EVAL_API_URL / EVAL_API_MODEL as in run.mjs)
 */
import { execFileSync } from "node:child_process";
import { appendFileSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { IdeApp, REPO, turnMetrics } from "./lib/app.mjs";

const API_MODEL = process.env.EVAL_API_MODEL ?? "Qwen3.5-4B";
const API_URL = process.env.EVAL_API_URL ?? "http://localhost:8100/v1";
const MODEL_REPO = process.env.FT_MODEL ?? "HuggingFaceTB/SmolLM2-135M-Instruct";
const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
const dir = path.join(REPO, "evals/results", `finetune-${stamp}`);
const ws = path.join(dir, "ws");
const target = path.join(ws, "models", MODEL_REPO.split("/")[1]);

const settings = {
  providers: [
    { id: "builtin", kind: "local_onnx", label: "Built-in (local)", baseUrl: "", hasKey: false },
    { id: "eval-api", kind: "openai_compat", label: "Eval API", baseUrl: API_URL, hasKey: false },
  ],
  models: [
    { id: "builtin:bge-m3", providerId: "builtin", name: "bge-m3", label: "bge-m3 (local)", kind: "embed", contextLength: 8192, maxOutputTokens: null, dim: 1024 },
    { id: `eval-api:${API_MODEL}`, providerId: "eval-api", name: API_MODEL, label: API_MODEL, kind: "chat", contextLength: 32768, maxOutputTokens: 8192, dim: null },
  ],
  roles: { agent: `eval-api:${API_MODEL}`, summarizer: "", embed: "builtin:bge-m3" },
  tools: { workspaceAllow: ["*"], autoApplyEdits: true },
  autoUpdate: false,
};

const PROMPT = `Download the Hugging Face model ${MODEL_REPO} (https://huggingface.co/${MODEL_REPO}) into this directory: ${target}

Then set this up as a scheduled task that runs on its own once the download has finished:
1. load the downloaded model and run a test on it,
2. design a synthetic dataset for this model,
3. fine-tune the model on that dataset.

Wait for the fine-tuning to finish, then look at the results and compare the fine-tuned model with the original. If you hit errors at any step, fix them and keep going until the whole pipeline has completed.

When everything is done, create a canvas in the chat that shows what was done and the results.`;

rmSync(dir, { recursive: true, force: true });
mkdirSync(ws, { recursive: true });
execFileSync("git", ["init", "-q"], { cwd: ws });
writeFileSync(path.join(dir, "prompt.md"), PROMPT);
const progress = path.join(dir, "progress.log");
const log = (s) => {
  const line = `[${new Date().toISOString().slice(11, 19)}] ${s}`;
  console.log(line);
  appendFileSync(progress, `${line}\n`);
};

const app = new IdeApp({ userData: path.join(dir, "userdata"), workspace: ws, settings, logFile: path.join(dir, "app.log") });
let shots = 0;
let lastSeq = -1;
let watchedTurn = null;
let timer;
const handledApprovals = new Set();
const approve = async (e) => {
  log(`approval: ${String(e.summary ?? e.toolId ?? e.title ?? "").slice(0, 200)} -> allow`);
  return true;
};
try {
  await app.start();
  log(`started; workspace ${ws}`);
  timer = setInterval(async () => {
    try {
      const chats = await app.cdp.invoke("chat:list");
      const chatId = chats[0]?.id;
      const turnId = chatId && (await app.cdp.invoke("chat:activeTurn", { chatId }));
      if (!turnId) return;
      if (turnId !== watchedTurn) {
        watchedTurn = turnId;
        lastSeq = -1;
        log(`watching turn ${turnId}`);
      }
      const evs = await app.cdp.invoke("chat:subscribe", { turnId, lastSeq });
      for (const e of evs) {
        lastSeq = Math.max(lastSeq, e.seq ?? lastSeq);
        if (e.type === "tool_call") log(`tool ${e.call.toolId} ${JSON.stringify(e.call.args).slice(0, 220)}`);
        else if (e.type === "tool_result") log(`  -> ${e.ok ? "ok" : "ERR"} ${JSON.stringify(e.result ?? "").replace(/\\n/g, " ").slice(0, 240)}`);
        else if (e.type === "code_cell") log(`cell: ${String(e.code ?? "").replace(/\s+/g, " ").slice(0, 200)}`);
        else if (e.type === "cell_output") log(`  out: ${String(e.text ?? "").replace(/\s+/g, " ").slice(0, 200)}`);
        else if (e.type === "cell_done") log(`cell ${e.ok ? "ok" : "ERR"} ${String(e.error ?? "").replace(/\s+/g, " ").slice(0, 200)}`);
        else if (e.type === "status") log(`status ${String(e.text).slice(0, 160)}`);
        else if (e.type === "subagent_started") log(`subagent: ${String(e.task).slice(0, 160)}`);
      }
      if (++shots % 4 === 0) await app.screenshot(path.join(dir, `progress-${String(shots / 4).padStart(3, "0")}.png`));
    } catch {
      /* window busy */
    }
  }, 30000);

  const deadline = Date.now() + Number(process.env.FT_HOURS ?? 3) * 3600 * 1000;
  const first = await app.ask(PROMPT, { timeoutMs: deadline - Date.now(), onApproval: approve });
  const chatId = first.chatId;
  const boardLine = (b) => (b ? `board: ${b.steps.filter((s) => s.status === "done").length}/${b.steps.length} done; wake-up ${b.wakeup ? `at ${new Date(b.wakeup.at).toISOString().slice(11, 19)} (${b.wakeup.reason.slice(0, 120)})` : "none"}` : "board: none");
  log(`turn 1 ${first.status}; ${boardLine(await app.cdp.invoke("task:board", { chatId }))}`);

  // The agent may schedule wake-ups; keep watching like a user who leaves the IDE open.
  for (;;) {
    const active = await app.cdp.invoke("chat:activeTurn", { chatId });
    const board = await app.cdp.invoke("task:board", { chatId });
    if (!active && !board?.wakeup) break;
    if (Date.now() > deadline) {
      log("deadline reached; stopping");
      if (active) await app.cdp.invoke("chat:stop", { turnId: active });
      break;
    }
    if (active) {
      const evs = await app.cdp.invoke("chat:subscribe", { turnId: active, lastSeq: -1 });
      const resolved = new Set(evs.filter((e) => e.type === "approval_resolved").map((e) => e.approvalId));
      for (const e of evs) {
        if (e.type === "approval_request" && !resolved.has(e.approvalId) && !handledApprovals.has(e.approvalId)) {
          handledApprovals.add(e.approvalId);
          await app.cdp.invoke("chat:approve", { approvalId: e.approvalId, approved: await approve(e), remember: "once" });
        }
      }
    }
    await new Promise((res) => setTimeout(res, 5000));
  }
  clearInterval(timer);

  const msgs = await app.cdp.invoke("chat:messages", { chatId });
  const turns = msgs.filter((m) => m.role === "assistant");
  const origins = new Map(msgs.filter((m) => m.role === "user").map((m) => [m.turnId, m.origin ?? "user"]));
  const span = (evs) => (evs.length ? (evs[evs.length - 1].ts ?? 0) - (evs[0].ts ?? 0) : 0);
  const perTurn = turns.map((t) => ({ origin: origins.get(t.turnId) ?? "user", ...turnMetrics({ ...t, events: t.events ?? [], text: t.text ?? "", durationMs: span(t.events ?? []) }) }));
  for (const [i, m] of perTurn.entries()) log(`turn ${i + 1} (${m.origin}) ${m.status}; cells ${m.cells} (${m.cellErrors} errors), tools ${m.toolCalls} (${m.toolErrors} errors), compactions ${m.compactions}`);
  const last = turns[turns.length - 1];
  const finalBoard = await app.cdp.invoke("task:board", { chatId });
  log(boardLine(finalBoard));
  writeFileSync(path.join(dir, "answer.md"), last?.text ?? "");
  writeFileSync(path.join(dir, "events.json"), JSON.stringify(turns.map((t) => ({ turnId: t.turnId, origin: origins.get(t.turnId), events: t.events })), null, 1));
  writeFileSync(path.join(dir, "metrics.json"), JSON.stringify({ turns: perTurn, wakeups: perTurn.filter((t) => t.origin === "wakeup").length }, null, 2));
  writeFileSync(path.join(dir, "board.json"), JSON.stringify(finalBoard, null, 2));

  // Let previews render, then capture the end of the chat.
  await new Promise((res) => setTimeout(res, 4000));
  await app.cdp.eval(`(() => { const s = [...document.querySelectorAll("*")].filter((e) => e.scrollHeight > e.clientHeight + 20 && getComputedStyle(e).overflowY !== "visible"); s.forEach((e) => (e.scrollTop = e.scrollHeight)); })()`);
  await new Promise((res) => setTimeout(res, 1500));
  await app.screenshot(path.join(dir, "final-chat.png"));
  const frames = await app.cdp.eval(`[...document.querySelectorAll("iframe")].map((f) => ({ src: f.src.slice(0, 60), h: f.clientHeight }))`);
  writeFileSync(path.join(dir, "previews.json"), JSON.stringify(frames, null, 2));
  log(`previews in chat: ${JSON.stringify(frames)}`);
} catch (e) {
  log(`runner error: ${e instanceof Error ? e.stack : String(e)}`);
} finally {
  clearInterval(timer);
  await app.stop();
  log(`done; results in ${dir}`);
}
