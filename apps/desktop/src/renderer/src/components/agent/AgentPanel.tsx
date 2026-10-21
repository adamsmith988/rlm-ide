import { useEffect, useMemo, useRef, useState } from "react";
import { agentPickerModels } from "@rlm/contracts";
import { call } from "../../api";
import { selectAgentModel } from "../../models";
import type { UiMessage } from "../../chat/model";
import { useApp } from "../../store";
import { ICopy, IDown, IPlus, IUp } from "../../ui/icons";
import { PersonHeadphones } from "../../ui/LineArt";
import { Markdown } from "../../ui/Markdown";
import { BlockView, groupBlocks } from "./Blocks";
import { Composer } from "./Composer";
import { ContextMeter } from "./ContextMeter";
import { PinnedCard } from "./PinnedCard";
import { TaskCard } from "./TaskCard";

const EMPTY: UiMessage[] = [];

const STARTERS = [
  "Explain this codebase",
  "Find and fix a bug",
  "Write tests for the open file",
  "Build a small app from scratch",
];

export function AgentPanel() {
  const { chatId, chats, newChat, workspace, context, status, settings } = useApp();
  const messages = useApp((s) => (s.chatId ? (s.messages[s.chatId] ?? EMPTY) : EMPTY));
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const running = messages.some((m) => m.role === "assistant" && m.status === "running");
  const chat = chats.find((c) => c.id === chatId);
  const doneCount = messages.filter((m) => m.status !== "running").length;

  useEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [messages]);

  // If the final event of a turn was missed (reload, crashed host), stop showing it as running.
  useEffect(() => {
    if (!running || !chatId) return;
    const t = setInterval(() => {
      void call<string | null>("chat:activeTurn", { chatId }).then((active) => {
        if (!active) void useApp.getState().selectChat(chatId);
      });
    }, 4000);
    return () => clearInterval(t);
  }, [running, chatId]);

  const chatModels = settings ? agentPickerModels(settings) : [];
  return (
    <section className="flex h-full flex-col border-l border-line bg-bg">
      <div className="flex h-10 shrink-0 items-center gap-2 px-3">
        <span className="heading min-w-0 flex-1 truncate text-[16px]">{chat?.title ?? "Agent"}</span>
        <select
          className="max-w-[150px] cursor-pointer truncate rounded-full border border-line bg-surface px-2 py-[2px] text-[11px] text-dim outline-none hover:border-line-strong"
          value={settings?.roles.agent ?? ""}
          title="Agent model"
          onChange={(e) => void selectAgentModel(e.target.value)}
        >
          {chatModels.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
        <button className="iconbtn" title="New chat (Ctrl+Shift+N)" onClick={() => void newChat()}>
          <IPlus size={15} />
        </button>
      </div>
      {chatId && <PinnedCard chatId={chatId} version={doneCount} />}
      {chatId && <TaskCard chatId={chatId} />}
      <div
        ref={scroller}
        className="min-h-0 flex-1 overflow-y-auto px-4 pb-4"
        onScroll={(e) => {
          const el = e.currentTarget;
          stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
        }}
      >
        {messages.length === 0 ? (
          <Empty disabled={!workspace} />
        ) : (
          messages.map((m) => (m.role === "user" ? <UserMsg key={m.id} m={m} /> : <AssistantMsg key={m.id} m={m} />))
        )}
      </div>
      <div className="shrink-0 px-3 pb-3">
        <div className="mb-2.5">
          <ContextMeter usage={chatId ? context[chatId] : undefined} contextLength={status.contextLength} />
        </div>
        <Composer running={running} disabledReason={workspace ? null : "Open a folder to start"} />
      </div>
    </section>
  );
}

function Empty({ disabled }: { disabled: boolean }) {
  const send = useApp((s) => s.sendMessage);
  return (
    <div className="flex h-full flex-col items-center justify-center py-8 text-center fade-in">
      <PersonHeadphones width={150} />
      <div className="heading mt-4 text-[26px]">What are we building?</div>
      <div className="mt-1 max-w-[280px] text-[12.5px] text-dim">The agent works in a live Python REPL with your workspace tools, and asks before it changes anything.</div>
      {!disabled && (
        <div className="mt-6 grid w-full max-w-[340px] grid-cols-2 gap-2">
          {STARTERS.map((s) => (
            <button key={s} className="tile" style={{ minHeight: 64, fontSize: 10 }} onClick={() => void send(s, [])}>
              {s}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

const ATTACHED_RE = /\n\n(\*\*(?:Browser screenshot|Referenced element)\*\*[\s\S]*)$/;

function UserMsg({ m }: { m: UiMessage }) {
  if (m.origin === "wakeup") {
    const note = /Your note for this wake-up: (.*)/.exec(m.text)?.[1] ?? "";
    return (
      <div className="mt-4 flex items-center gap-2 text-[11.5px] text-dim fade-in" data-testid="wakeup-msg" title={m.text}>
        <span className="h-px flex-1 bg-line" />
        <span className="font-medium text-ink">Scheduled wake-up</span>
        {note && <span className="max-w-[60%] truncate text-faint">{note}</span>}
        <span className="h-px flex-1 bg-line" />
      </div>
    );
  }
  const split = ATTACHED_RE.exec(m.text);
  const typed = split ? m.text.slice(0, split.index) : m.text;
  return (
    <div className="mt-4 flex flex-col items-end gap-1.5 fade-in">
      <div className="selectable max-w-[88%] whitespace-pre-wrap rounded-[18px] rounded-br-[6px] bg-ink px-4 py-2.5 text-[13px] text-bg">{typed}</div>
      {split && (
        <div className="max-w-[88%] rounded-[14px] border-[1.5px] border-ink bg-surface px-3 py-2 text-[12px]" data-testid="user-attachments">
          <Markdown text={split[1]!} previews={false} />
        </div>
      )}
    </div>
  );
}

const PHASE_LABEL = { waiting: "Waiting for the model", thinking: "Thinking", writing: "Writing", writing_code: "Writing code" } as const;

/** What the agent is doing right now, updated every second so a long step never looks frozen. */
function LiveActivity({ m }: { m: UiMessage }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  const last = m.blocks[m.blocks.length - 1];
  const runningCell = last?.kind === "cell" && last.ok === undefined;
  const runningTool = [...m.blocks].reverse().find((b) => b.kind === "tool" && b.ok === undefined) as Extract<UiMessage["blocks"][number], { kind: "tool" }> | undefined;
  const waitingApproval = m.blocks.some((b) => b.kind === "approval" && !b.resolved);
  const p = m.progress;
  const secs = (from?: number) => (from ? `${Math.max(0, Math.round((now - from) / 1000))}s` : "");
  let label: string;
  let detail = "";
  if (waitingApproval) label = "Waiting for your approval";
  else if (runningTool) {
    label = `Running ${runningTool.call.toolId}`;
  } else if (runningCell) label = "Running code";
  else if (p) {
    label = PHASE_LABEL[p.phase];
    detail = p.phase === "waiting" ? "" : `${p.outputTokens.toLocaleString()} tokens · ${p.tokPerSec} tok/s`;
  } else label = "Starting";
  const quiet = p && !runningCell && !runningTool && !waitingApproval && now - p.ts > 20000;
  return (
    <div className="flex flex-col gap-1.5" data-testid="live-activity">
      {m.draft && (
        <div className="overflow-hidden rounded-[12px] border-[1.5px] border-ink">
          <div className="flex items-center gap-2 border-b border-line px-3 py-1.5 text-[11px]">
            <span className="pulse-dot" />
            <span className="font-medium">Writing code cell</span>
            <span className="text-faint">
              {m.draft.lines} lines · {(m.draft.chars / 1024).toFixed(1)} KB
            </span>
          </div>
          <pre className="draft-tail m-0 max-h-[180px] overflow-hidden px-3 py-2 font-code text-[11px] leading-[1.45] text-dim">{m.draft.tail}</pre>
        </div>
      )}
      <div className="flex items-center gap-2 text-[11.5px] text-dim">
        <span className="pulse-dot" />
        <span className="font-medium text-ink">{label}</span>
        {detail && <span className="text-faint">{detail}</span>}
        {p && p.step > 1 && <span className="text-faint">· step {p.step}</span>}
        <span className="ml-auto tabular-nums text-faint">{secs(m.startedAt)}</span>
      </div>
      {quiet && <div className="text-[11px] text-faint">No output from the model for {secs(p.ts)}. It is still working; large models can pause before long outputs.</div>}
    </div>
  );
}

function AssistantMsg({ m }: { m: UiMessage }) {
  const toast = useApp((s) => s.toast);
  const [fb, setFb] = useState(m.feedback ?? null);
  const running = m.status === "running";
  const finalText = m.text || m.blocks.filter((b) => b.kind === "text").map((b) => (b as { text: string }).text).join("\n");
  const feedback = async (v: "up" | "down") => {
    const next = fb === v ? null : v;
    setFb(next);
    await call("chat:feedback", { messageId: m.id, value: next ?? "none" }).catch(() => undefined);
  };
  const hasText = m.blocks.some((b) => b.kind === "text");
  const grouped = useMemo(() => groupBlocks(m.blocks), [m.blocks]);
  return (
    <div className="mt-4 flex flex-col gap-2 fade-in">
      {grouped.map((b, i) => (
        <BlockView key={i} block={b} running={running} last={i === grouped.length - 1} />
      ))}
      {running && <LiveActivity m={m} />}
      {!running && !hasText && m.text && <Markdown text={m.text} />}
      {m.status === "cancelled" && <div className="text-[11.5px] text-faint">Stopped.</div>}
      {!running && (
        <div className="flex items-center gap-1 text-faint">
          <button className="iconbtn h-6 w-6" title="Good answer" style={{ color: fb === "up" ? "var(--text)" : undefined }} onClick={() => void feedback("up")}>
            <IUp size={13} />
          </button>
          <button className="iconbtn h-6 w-6" title="Bad answer" style={{ color: fb === "down" ? "var(--text)" : undefined }} onClick={() => void feedback("down")}>
            <IDown size={13} />
          </button>
          <button
            className="iconbtn h-6 w-6"
            title="Copy answer"
            onClick={() => {
              void navigator.clipboard.writeText(finalText);
              toast("Copied");
            }}
          >
            <ICopy size={13} />
          </button>
          <span className="ml-1 text-[10.5px]">
            {m.startedAt && m.endedAt ? `${((m.endedAt - m.startedAt) / 1000).toFixed(1)}s` : ""}
            {m.usage ? ` · ${m.usage.inputTokens + m.usage.outputTokens} tokens` : ""}
          </span>
        </div>
      )}
    </div>
  );
}
