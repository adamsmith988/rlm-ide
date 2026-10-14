"""Hub SSE event building for controller engines (process log + answer + done).

Event shapes match what ``main.py`` relays for every chat engine:
``start``/``thinking*``/``progress``/``answer*``/``warning``/``error``/``done``.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Dict, List, Optional

from .llm_roles import gateway_llm_stream


class HubEmitter:
    def __init__(self, engine_id: str, session_id: str) -> None:
        self.engine_id = engine_id
        self.session_id = session_id
        self.trace_id = str(uuid.uuid4())
        self._thinking_open = False
        self._answer_open = False
        self.thinking_parts: List[str] = []
        self.answer_parts: List[str] = []

    def start(self) -> Dict[str, Any]:
        return {
            "type": "start",
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "engine": self.engine_id,
            "timestamp": _dt.datetime.utcnow().isoformat(),
        }

    def log(self, text: str) -> List[Dict[str, Any]]:
        """One process-log line (rendered in the thinking panel)."""
        out: List[Dict[str, Any]] = []
        if not self._thinking_open:
            self._thinking_open = True
            out.append({"type": "thinking_start"})
        chunk = str(text or "")
        if not chunk.endswith("\n"):
            chunk += "\n"
        self.thinking_parts.append(chunk)
        out.append({"type": "thinking", "content": chunk})
        return out

    def thinking_delta(self, text: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not self._thinking_open:
            self._thinking_open = True
            out.append({"type": "thinking_start"})
        self.thinking_parts.append(str(text or ""))
        out.append({"type": "thinking", "content": str(text or "")})
        return out

    def progress(
        self,
        message: str,
        *,
        progress: Optional[float] = None,
        status: str = "running",
        **extra: Any,
    ) -> Dict[str, Any]:
        ev: Dict[str, Any] = {
            "type": "progress",
            "engine": self.engine_id,
            "message": message,
            "status": status,
        }
        if progress is not None:
            ev["progress"] = round(float(progress), 4)
        ev.update(extra)
        return ev

    def close_thinking(self) -> List[Dict[str, Any]]:
        if not self._thinking_open:
            return []
        self._thinking_open = False
        return [{"type": "thinking_end"}]

    def answer(self, text: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not text:
            return out
        if self._thinking_open:
            self._thinking_open = False
            out.append({"type": "thinking_end"})
        if not self._answer_open:
            self._answer_open = True
            out.append({"type": "answer_start", "step": 1})
        self.answer_parts.append(text)
        out.append({"type": "answer", "content": text})
        return out

    @property
    def streamed_answer(self) -> str:
        return "".join(self.answer_parts)

    @property
    def thinking(self) -> str:
        return "".join(self.thinking_parts)

    def finish_answer(self, final_text: str) -> List[Dict[str, Any]]:
        """Stream whatever of ``final_text`` the UI has not painted yet."""
        prev = self.streamed_answer
        if final_text == prev:
            return []
        if prev and final_text.startswith(prev):
            return self.answer(final_text[len(prev) :])
        if prev:
            # The UI appends deltas; a rewritten answer is replaced by done.final_answer.
            return []
        return self.answer(final_text)

    def done(
        self,
        final_answer: str,
        *,
        tokens_prompt: Optional[int] = None,
        tokens_completion: Optional[int] = None,
        tool_runs: Optional[List[Dict[str, Any]]] = None,
        extensions: Optional[Dict[str, Any]] = None,
        status: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        ev: Dict[str, Any] = {
            "type": "done",
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "engine": self.engine_id,
            "final_answer": final_answer,
            "answer": final_answer,
            "thinking": self.thinking or None,
            "tool_runs": list(tool_runs or []),
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "extensions": dict(extensions or {}),
        }
        if status:
            ev["status"] = status
        if error:
            ev["error"] = str(error)[:2000]
        return ev

    def fail(self, message: str, *, extensions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        msg = f"{self.engine_id} orchestrator error: {message}"
        out: List[Dict[str, Any]] = [{"type": "error", "message": msg}]
        out.extend(self.answer(msg))
        out.append(self.done(msg, extensions=extensions, status="error", error=message))
        return out


@dataclass
class TaskOutcome:
    value: Any = None
    error: Optional[BaseException] = None


async def run_with_events(
    coro: Awaitable[Any],
    queue: "asyncio.Queue[Dict[str, Any]]",
    outcome: TaskOutcome,
) -> AsyncIterator[Dict[str, Any]]:
    """Run ``coro`` and yield events it puts on ``queue`` until it finishes."""
    task = asyncio.ensure_future(coro)
    try:
        while True:
            getter = asyncio.ensure_future(queue.get())
            done, _ = await asyncio.wait({task, getter}, return_when=asyncio.FIRST_COMPLETED)
            if getter in done:
                yield getter.result()
                continue
            getter.cancel()
            break
        while not queue.empty():
            yield queue.get_nowait()
        try:
            outcome.value = task.result()
        except BaseException as exc:  # noqa: BLE001 — surfaced to the caller via outcome
            outcome.error = exc
    finally:
        if not task.done():
            task.cancel()


@dataclass
class StreamedReply:
    text: str = ""
    reasoning: str = ""
    error: str = ""
    raw_parts: List[str] = field(default_factory=list)


async def stream_llm(
    emitter: HubEmitter,
    messages: List[Dict[str, Any]],
    reply: StreamedReply,
    *,
    as_answer: bool,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
    sampling: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Stream one completion. ``as_answer`` paints content live as the answer;
    otherwise content goes to the process log and the caller post-processes it."""
    async for ev in gateway_llm_stream(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        sampling=sampling,
    ):
        et = ev.get("type")
        if et == "thinking":
            piece = str(ev.get("content") or "")
            reply.reasoning += piece
            for out in emitter.thinking_delta(piece):
                yield out
        elif et == "answer":
            piece = str(ev.get("content") or "")
            reply.raw_parts.append(piece)
            events = emitter.answer(piece) if as_answer else emitter.thinking_delta(piece)
            for out in events:
                yield out
        elif et == "error":
            reply.error = str(ev.get("content") or ev.get("message") or "llm_error")
    reply.text = "".join(reply.raw_parts)
