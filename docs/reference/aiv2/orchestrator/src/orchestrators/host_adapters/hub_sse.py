"""Map generic orchestrator event payloads → hub chat SSE event dicts.

Hub-chat consumer only. Engines emit via HostPorts.events; this adapter
translates for the existing chat timeline UI.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .hub_final_answer import incomplete_notice


class HubEventBridge:
    """Map RLM-style on_event payloads → hub SSE event dicts."""

    def __init__(self, *, log_cursor: int = 0) -> None:
        self._reason_len = 0
        self._content_len = 0
        # On durable continue, seed past the already-shown terminal so the FE
        # does not replay ``[kernel] ready`` / bootstrap lines every message.
        self._log_n = max(0, int(log_cursor or 0))
        self._thinking_open = False
        self._answer_started = False
        self.collected_thinking = ""
        self.collected_answer = ""

    def seed_log_cursor(self, n: int) -> None:
        """Advance the terminal cursor (call when reusing a live RLM session)."""
        self._log_n = max(self._log_n, max(0, int(n or 0)))

    def seed_from_live_session(self, runtime_session: Any) -> None:
        """Skip already-shown logs / answer / reasoning from a prior turn.

        Answer cursors are seeded so a stale pre-stream snapshot does not
        replay the previous reply. When the new LLM stream starts the session
        emits ``stream_begin`` → :meth:`begin_live_stream` resets *answer*
        cursors only so length-continuation chunks are not swallowed — the
        terminal/thinking block stays one continuous piece for the turn.
        """
        if runtime_session is None:
            return
        try:
            self.seed_log_cursor(len(getattr(runtime_session, "log_lines", None) or []))
        except Exception:
            pass
        # Progress snapshots before the next stream_begin still carry the prior
        # turn's live buffer, which can differ from the stamped last_content.
        try:
            prev_answer = str(getattr(runtime_session, "last_content", "") or "")
            self._content_len = max(self._content_len, len(prev_answer))
        except Exception:
            pass
        try:
            snap = runtime_session.live_progress_snapshot()
            self._content_len = max(self._content_len, len(str(snap.get("last_content") or "")))
        except Exception:
            pass
        try:
            prev_reason = str(getattr(runtime_session, "last_reasoning", "") or "")
            self._reason_len = max(self._reason_len, len(prev_reason))
        except Exception:
            pass

    def begin_live_stream(self) -> None:
        """Reset answer/reasoning stream cursors for a fresh LLM call.

        Keep the thinking/terminal phase open so ``thinking_start`` is not
        re-emitted mid-turn (that fragmented the hub UI into many cards).
        """
        self._content_len = 0
        self._reason_len = 0
        self.collected_answer = ""
        self._answer_started = False

    def _ensure_thinking(self, out: List[Dict[str, Any]]) -> None:
        if not self._thinking_open:
            out.append({"type": "thinking_start"})
            self._thinking_open = True

    def _append_thinking(self, out: List[Dict[str, Any]], piece: str) -> None:
        text = str(piece or "")
        if not text:
            return
        self._ensure_thinking(out)
        out.append({"type": "thinking", "content": text})
        self.collected_thinking += text

    def _drain_new_log_lines(
        self, logs: Any, out: List[Dict[str, Any]]
    ) -> None:
        if not isinstance(logs, list):
            return
        if len(logs) <= self._log_n:
            return
        for line in logs[self._log_n :]:
            text = str(line or "").strip()
            if not text:
                continue
            chunk = text if text.endswith("\n") else text + "\n"
            self._append_thinking(out, chunk)
        self._log_n = len(logs)

    def map_event(self, ev: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(ev, dict):
            return []
        kind = str(ev.get("kind") or "")
        out: List[Dict[str, Any]] = []

        if kind == "stream_begin":
            self.begin_live_stream()
            return out

        if kind == "stream_progress":
            rlm = ev.get("rlm") if isinstance(ev.get("rlm"), dict) else {}
            reasoning = str(rlm.get("last_reasoning") or rlm.get("reasoning") or "")
            content = str(rlm.get("last_content") or "")
            # content_len=0 is meaningful (new stream empty buffer). Never use
            # ``or len(content)`` — that replays the prior turn's last_content.
            _rl = ev.get("reasoning_len")
            _cl = ev.get("content_len")
            reason_len = int(_rl) if _rl is not None else len(reasoning)
            content_len = int(_cl) if _cl is not None else len(content)

            if reason_len > self._reason_len and reasoning:
                piece = reasoning[self._reason_len :]
                self._reason_len = reason_len
                if piece:
                    self._append_thinking(out, piece)

            if content_len > self._content_len and content:
                piece = content[self._content_len :]
                self._content_len = content_len
                if piece:
                    if self._thinking_open:
                        out.append({"type": "thinking_end"})
                        self._thinking_open = False
                    if not self._answer_started:
                        out.append({"type": "answer_start", "step": 1})
                        self._answer_started = True
                    # Only ``answer`` — ``/chat/stream/full`` + FE both treat
                    # answer deltas as visible tokens; emitting ``token`` too
                    # doubles the final reply in the UI.
                    out.append({"type": "answer", "content": piece})
                    self.collected_answer += piece

            self._drain_new_log_lines(rlm.get("log_lines"), out)
            return out

        if kind == "llm_reconnect":
            # Prefer the session terminal line (via rlm.log_lines) so we do not
            # emit a duplicate thinking chunk alongside stream_progress.
            rlm = ev.get("rlm") if isinstance(ev.get("rlm"), dict) else {}
            logs = rlm.get("log_lines") if isinstance(rlm.get("log_lines"), list) else None
            before = self._log_n
            if logs is not None:
                self._drain_new_log_lines(logs, out)
            if self._log_n == before:
                preview = str(ev.get("preview") or "").strip()
                if not preview:
                    n = ev.get("attempt") or "?"
                    total = ev.get("max_attempts") or "?"
                    wait = ev.get("wait_sec")
                    if wait is not None:
                        preview = f"model gateway 503 — retry {n}/{total} in {wait}s"
                    else:
                        preview = f"model gateway 503 — retry {n}/{total}"
                if preview:
                    self._append_thinking(out, f"[reconnecting] {preview}\n")
            return out

        if kind in ("ipython", "ipython_result"):
            preview = str(
                ev.get("code") or ev.get("preview") or ev.get("result") or ev.get("error") or ""
            )[:2000]
            if preview:
                label = "ipython" if kind == "ipython" else "ipython_result"
                self._append_thinking(out, f"[{label}]\n{preview}\n")
            return out

        if kind in ("tool_start", "tool_result"):
            args = ev.get("args") if isinstance(ev.get("args"), dict) else {}
            mapped: Dict[str, Any] = {
                "type": kind,
                "tool_id": ev.get("tool_id"),
                "tool_name": ev.get("tool_id"),
                "run_id": ev.get("run_id"),
                "args": args,
                "arguments": args,
                "step": ev.get("step", 0),
            }
            if kind == "tool_result":
                mapped.update(
                    ok=bool(ev.get("ok")),
                    summary=ev.get("summary") or "",
                    error=ev.get("error"),
                    execution_time=ev.get("execution_time", 0),
                )
            out.append(mapped)
            rlm = ev.get("rlm") if isinstance(ev.get("rlm"), dict) else {}
            self._drain_new_log_lines(rlm.get("log_lines"), out)
            return out

        if kind in ("run_error", "shutdown") and (
            ev.get("error") or ev.get("notice") or ev.get("detail")
        ):
            notice = str(
                ev.get("notice") or ev.get("error") or ev.get("detail") or ""
            ).strip()
            if notice:
                out.append({"type": "error", "message": notice[:2000]})
                # AF/hub: never end mid-work with an empty answer card.
                if not str(self.collected_answer or "").strip():
                    if not self._answer_started:
                        out.append({"type": "answer_start", "step": 1})
                        self._answer_started = True
                    out.append({"type": "answer", "content": notice})
                    self.collected_answer = notice
            return out

        if kind == "context_overflow_incomplete":
            notice = str(
                ev.get("notice")
                or ev.get("error")
                or ev.get("detail")
                or "Context window full — compaction could not continue."
            ).strip()
            if notice:
                out.append({"type": "error", "message": notice[:2000]})
                if not self._answer_started:
                    out.append({"type": "answer_start", "step": 1})
                    self._answer_started = True
                out.append({"type": "answer", "content": notice})
                self.collected_answer = notice
            return out

        if kind == "autonomous_limit_exhausted":
            reason = str(ev.get("reason") or "").strip()
            notice = incomplete_notice(reason)
            if not str(self.collected_answer or "").strip():
                out.append({"type": "error", "message": notice[:2000]})
                if not self._answer_started:
                    out.append({"type": "answer_start", "step": 1})
                    self._answer_started = True
                out.append({"type": "answer", "content": notice})
                self.collected_answer = notice
            return out

        if kind in ("llm_call", "assistant_turn", "kernel_ready", "file", "compact"):
            detail = str(ev.get("summary") or ev.get("path") or kind)
            out.append(
                {
                    "type": "status",
                    "message": f"rlm:{kind}",
                    "detail": detail[:500],
                }
            )
            # Same snapshot often carries the new terminal line — drain once so
            # the hub timeline stays one continuous thinking block.
            rlm = ev.get("rlm") if isinstance(ev.get("rlm"), dict) else {}
            self._drain_new_log_lines(rlm.get("log_lines"), out)
        return out

    def close_phases(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if self._thinking_open:
            out.append({"type": "thinking_end"})
            self._thinking_open = False
        return out
