import { useEffect, useState } from "react";
import type { TaskBoardView } from "@rlm/contracts";
import { call, errText } from "../../api";
import { useApp } from "../../store";

const MARK = { todo: "○", doing: "◐", done: "●", blocked: "!", skipped: "–" } as const;

function countdown(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

/** The agent's task board: plan, progress and the next scheduled wake-up. */
export function TaskCard({ chatId }: { chatId: string }) {
  const board = useApp((s) => s.boards[chatId]);
  const toast = useApp((s) => s.toast);
  const [open, setOpen] = useState(false);
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!board?.wakeup) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [board?.wakeup]);

  if (!board || (!board.steps.length && !board.wakeup)) return null;
  const done = board.steps.filter((s) => s.status === "done" || s.status === "skipped").length;
  const current = board.steps.find((s) => s.status === "doing") ?? board.steps.find((s) => s.status === "todo" || s.status === "blocked");
  const cancel = async () => {
    try {
      await call("task:cancelWakeup", { chatId } satisfies { chatId: string });
    } catch (e) {
      toast(errText(e), "error");
    }
  };
  return (
    <div className="mx-3 mb-2 rounded-[14px] border-[1.5px] border-ink px-3 py-2" data-testid="task-card">
      <button className="flex w-full items-center gap-2 text-left" onClick={() => setOpen(!open)}>
        <span className="text-[9.5px] font-bold uppercase tracking-[0.16em] text-dim">Task</span>
        {board.steps.length > 0 && (
          <span className="mono text-[11px] text-faint" data-testid="task-progress">
            {done}/{board.steps.length}
          </span>
        )}
        <span className="min-w-0 flex-1 truncate text-[12px]">{current?.text ?? board.goal}</span>
        <span className="text-[10.5px] text-faint">{open ? "Hide" : "Show"}</span>
      </button>
      {board.steps.length > 0 && (
        <div className="mt-1.5 h-[3px] rounded-full bg-surface-3">
          <div className="h-full rounded-full bg-ink transition-[width]" style={{ width: `${(done / board.steps.length) * 100}%` }} />
        </div>
      )}
      {board.wakeup && (
        <div className="mt-1.5 flex items-center gap-2 text-[11px]" data-testid="task-wakeup">
          <span className="pulse-dot" />
          <span className="text-dim">
            Wakes up in <span className="mono text-ink">{countdown(board.wakeup.at - now)}</span>
          </span>
          <span className="min-w-0 flex-1 truncate text-faint" title={board.wakeup.reason}>
            {board.wakeup.reason}
          </span>
          <button className="text-[10.5px] text-faint hover:text-ink" onClick={() => void cancel()}>
            Cancel
          </button>
        </div>
      )}
      {open && (
        <div className="mt-2 border-t border-line pt-2">
          {board.goal && <div className="mb-1 text-[12px] font-medium">{board.goal}</div>}
          {board.steps.map((s, i) => (
            <div key={i} className="flex items-start gap-2 py-[1px] text-[12px]">
              <span className="mono w-3 shrink-0 text-center text-dim">{MARK[s.status]}</span>
              <span className={`flex-1 ${s.status === "done" || s.status === "skipped" ? "text-faint line-through" : ""}`}>
                {s.text}
                {s.note && <span className="text-faint"> — {s.note}</span>}
              </span>
            </div>
          ))}
          {board.log.length > 0 && (
            <div className="mt-2">
              <div className="smallcaps mb-1" style={{ fontSize: 9.5 }}>
                Progress log
              </div>
              {board.log.slice(-6).map((e, i) => (
                <div key={i} className="truncate text-[11px] text-dim" title={e.text}>
                  {new Date(e.at).toLocaleTimeString()} · {e.text}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
