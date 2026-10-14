import { TERMINAL_EVENT_TYPES, type TurnEvent, type TurnEventBody, type ToolCallInfo } from "@rlm/contracts";
import type { ApprovalRequester } from "./tool_executor";
import { newId } from "./memory_service";

/** Upper bound on buffered events per turn. Deltas are coalesced, so this is rarely reached. */
const MAX_BUFFERED_EVENTS = 5000;
/** How long a finished turn's buffer stays available for late resubscribes. */
const FINISHED_RETENTION_MS = 10 * 60 * 1000;

interface TurnState {
  turnId: string;
  chatId: string;
  seq: number;
  events: TurnEvent[];
  done: boolean;
  abort: AbortController;
}

interface PendingApproval {
  turnId: string;
  resolve: (d: { approved: boolean; remember: "once" | "session" | "workspace" }) => void;
}

/** Turns run as background tasks that outlive any UI subscription. Only `stop` cancels. */
export class TurnRunner implements ApprovalRequester {
  private readonly turns = new Map<string, TurnState>();
  private readonly activeByChat = new Map<string, string>();
  private readonly approvals = new Map<string, PendingApproval>();

  constructor(private readonly broadcast: (ev: TurnEvent) => void) {}

  begin(chatId: string, turnId: string): { signal: AbortSignal; emit: (e: TurnEventBody) => void } {
    const st: TurnState = { turnId, chatId, seq: -1, events: [], done: false, abort: new AbortController() };
    this.turns.set(turnId, st);
    this.activeByChat.set(chatId, turnId);
    return { signal: st.abort.signal, emit: (e) => this.emit(turnId, e) };
  }

  isActive(chatId: string): boolean {
    const id = this.activeByChat.get(chatId);
    return !!id && !this.turns.get(id)?.done;
  }

  emit(turnId: string, body: TurnEventBody): void {
    const st = this.turns.get(turnId);
    if (!st || st.done) return;
    st.seq++;
    const ev = { ...body, turnId, seq: st.seq, ts: Date.now() } as TurnEvent;
    this.buffer(st, ev);
    this.broadcast(ev);
    if (TERMINAL_EVENT_TYPES.has(body.type)) this.finish(st);
  }

  private buffer(st: TurnState, ev: TurnEvent): void {
    if (ev.type === "generation" || ev.type === "cell_draft") {
      // Progress snapshots: only the latest matters for replay.
      const i = st.events.findLastIndex((e) => e.type === ev.type);
      if (i >= 0) st.events.splice(i, 1);
      st.events.push(ev);
      return;
    }
    const last = st.events[st.events.length - 1];
    if (last && last.type === ev.type) {
      if ((ev.type === "text_delta" || ev.type === "thinking_delta") && (last.type === "text_delta" || last.type === "thinking_delta")) {
        st.events[st.events.length - 1] = { ...last, text: last.text + ev.text, seq: ev.seq, ts: ev.ts };
        return;
      }
      if (ev.type === "cell_output" && last.type === "cell_output" && last.cellId === ev.cellId && last.stream === ev.stream) {
        st.events[st.events.length - 1] = { ...last, text: last.text + ev.text, seq: ev.seq, ts: ev.ts };
        return;
      }
    }
    st.events.push(ev);
    if (st.events.length > MAX_BUFFERED_EVENTS) {
      const firstDelta = st.events.findIndex((e) => e.type === "cell_output" || e.type === "thinking_delta");
      st.events.splice(firstDelta >= 0 ? firstDelta : 1, 1);
    }
  }

  private finish(st: TurnState): void {
    st.done = true;
    if (this.activeByChat.get(st.chatId) === st.turnId) this.activeByChat.delete(st.chatId);
    for (const [id, p] of this.approvals) {
      if (p.turnId === st.turnId) {
        p.resolve({ approved: false, remember: "once" });
        this.approvals.delete(id);
      }
    }
    setTimeout(() => this.turns.delete(st.turnId), FINISHED_RETENTION_MS).unref?.();
  }

  /** Events after `lastSeq` for a resubscribing renderer. */
  replay(turnId: string, lastSeq: number): TurnEvent[] {
    const st = this.turns.get(turnId);
    if (!st) return [];
    return st.events.filter((e) => e.seq > lastSeq);
  }

  bufferedEvents(turnId: string): TurnEvent[] {
    return this.turns.get(turnId)?.events ?? [];
  }

  activeTurn(chatId: string): string | null {
    const id = this.activeByChat.get(chatId);
    return id && !this.turns.get(id)?.done ? id : null;
  }

  stop(turnId: string): boolean {
    const st = this.turns.get(turnId);
    if (!st || st.done) return false;
    st.abort.abort();
    return true;
  }

  request(turnId: string, call: ToolCallInfo, preview: string | undefined, signal: AbortSignal) {
    const approvalId = newId("appr");
    return new Promise<{ approved: boolean; remember: "once" | "session" | "workspace" }>((resolve) => {
      const onAbort = () => {
        this.approvals.delete(approvalId);
        resolve({ approved: false, remember: "once" });
      };
      if (signal.aborted) return onAbort();
      signal.addEventListener("abort", onAbort, { once: true });
      this.approvals.set(approvalId, {
        turnId,
        resolve: (d) => {
          signal.removeEventListener("abort", onAbort);
          this.emit(turnId, { type: "approval_resolved", approvalId, approved: d.approved });
          resolve(d);
        },
      });
      this.emit(turnId, { type: "approval_request", approvalId, call, preview });
    });
  }

  resolveApproval(approvalId: string, approved: boolean, remember: "once" | "session" | "workspace"): boolean {
    const p = this.approvals.get(approvalId);
    if (!p) return false;
    this.approvals.delete(approvalId);
    p.resolve({ approved, remember });
    return true;
  }
}
