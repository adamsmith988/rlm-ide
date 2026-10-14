"""Run one turn of any registered chat engine and collect its result.

Controllers (SRLM peers, RRSI policy trials) use this to drive other drop-ins
without knowing which ones exist. Engines whose manifest declares a durable
``session_binding`` get an isolated engine session per call so parallel trials
never share a kernel or memory.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from ..chat_engine_resolve import chat_stream_manifest, stream_agent_for_id, wraps_engines
from ..host_adapters.session_binding import read_session_binding

EventSink = Callable[[Dict[str, Any]], None]


@dataclass
class EngineTurn:
    engine_id: str
    text: str = ""
    thinking: str = ""
    tool_runs: List[Dict[str, Any]] = field(default_factory=list)
    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    error: str = ""
    elapsed_sec: float = 0.0
    # The engine's context checkpoint (``done.self_compact``), for a wrapper that serves this turn.
    self_compact: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.text.strip())

    @property
    def tokens(self) -> Optional[int]:
        if self.tokens_prompt is None and self.tokens_completion is None:
            return None
        return int(self.tokens_prompt or 0) + int(self.tokens_completion or 0)


def validate_inner_engine(engine_id: str, *, caller_id: str) -> None:
    oid = str(engine_id or "").strip()
    if not oid:
        raise ValueError("inner engine id is empty")
    if oid == caller_id:
        raise ValueError(f"{caller_id} cannot run itself as an inner engine")
    manifest = chat_stream_manifest(oid)
    if manifest is None:
        raise ValueError(f"inner engine {oid!r} is not an available chat engine")
    if wraps_engines(manifest):
        raise ValueError(f"inner engine {oid!r} is itself a controller; nesting is not allowed")


def _isolated_session_id(engine_id: str, session_id: str, suffix: str) -> str:
    binding = read_session_binding(chat_stream_manifest(engine_id))
    if binding is not None and binding.is_durable and suffix:
        return f"{session_id}__{suffix}"
    return session_id


async def run_engine_turn(
    engine_id: str,
    engine_kwargs: Mapping[str, Any],
    *,
    orchestrator_config: Optional[Mapping[str, Any]] = None,
    max_steps: Optional[int] = None,
    session_suffix: str = "",
    timeout_sec: Optional[float] = None,
    on_event: Optional[EventSink] = None,
) -> EngineTurn:
    fn = stream_agent_for_id(engine_id)
    kwargs: Dict[str, Any] = dict(engine_kwargs)
    kwargs["session_id"] = _isolated_session_id(
        engine_id, str(kwargs.get("session_id") or ""), session_suffix
    )
    turn = EngineTurn(engine_id=engine_id)
    answer_parts: List[str] = []
    thinking_parts: List[str] = []
    started = time.monotonic()

    async def _consume() -> None:
        async for ev in fn(
            max_steps=max_steps,
            orchestrator_config=dict(orchestrator_config or {}),
            **kwargs,
        ):
            if not isinstance(ev, dict):
                continue
            et = ev.get("type")
            if et == "answer":
                answer_parts.append(str(ev.get("content") or ""))
            elif et == "thinking":
                thinking_parts.append(str(ev.get("content") or ""))
            elif et == "tool_result":
                turn.tool_runs.append(
                    {"tool_id": ev.get("tool_name"), "ok": bool(ev.get("ok", False))}
                )
            elif et == "error" and not turn.error:
                turn.error = str(ev.get("message") or "engine_error")
            elif et == "done":
                if ev.get("final_answer"):
                    turn.text = str(ev.get("final_answer"))
                if ev.get("thinking"):
                    turn.thinking = str(ev.get("thinking"))
                if isinstance(ev.get("tool_runs"), list) and ev.get("tool_runs"):
                    turn.tool_runs = list(ev.get("tool_runs"))
                turn.tokens_prompt = ev.get("tokens_prompt")
                turn.tokens_completion = ev.get("tokens_completion")
                if isinstance(ev.get("self_compact"), dict):
                    turn.self_compact = ev["self_compact"]
                if ev.get("status") in ("error",) and ev.get("error") and not turn.error:
                    turn.error = str(ev.get("error"))
            if on_event is not None:
                on_event(ev)

    try:
        if timeout_sec and timeout_sec > 0:
            await asyncio.wait_for(_consume(), timeout=timeout_sec)
        else:
            await _consume()
    except asyncio.TimeoutError:
        turn.error = f"timeout after {timeout_sec}s"
    except Exception as exc:  # noqa: BLE001 — a failed trial is data, not a crash
        turn.error = f"{type(exc).__name__}: {exc}"
    if not turn.text:
        turn.text = "".join(answer_parts)
    if not turn.thinking:
        turn.thinking = "".join(thinking_parts)
    turn.elapsed_sec = round(time.monotonic() - started, 3)
    return turn
