"""Policy for tools that change something outside the chat (``"side_effects": true`` in tool.json).

Every engine reaches tools through ``ToolExecutor.execute``; this module sits in front of it:

* a short check asks the gateway model whether the user asked for that action now, with
  those values (prompt: ``prompts/action_check.md``), so a model cannot act on guesses,
  recalled memory or a request the user put on hold;
* within one chat turn, identical calls run once and share the result, so engines that
  sample several answers (or judge several trials) do not repeat a real action;
* a turn context can mark runs as rehearsals (replays of earlier requests), whose
  side-effect calls are simulated instead of performed.

The chat host starts the turn context (:func:`begin_tool_turn`); it is inherited by every
task the turn spawns. Calls outside a chat turn (direct tool API) are not checked.
"""

from __future__ import annotations

import asyncio
import contextvars
import datetime as dt
import json
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterator, List, Mapping, Optional

from ..config import settings

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class ToolTurn:
    key: str
    conversation: List[Dict[str, str]]
    request_context: Mapping[str, Any] = field(default_factory=dict)
    rehearsal: bool = False
    observations: List[Dict[str, Any]] = field(default_factory=list)
    inflight: Dict[str, "asyncio.Future[Dict[str, Any]]"] = field(default_factory=dict)


_TURN: contextvars.ContextVar[Optional[ToolTurn]] = contextvars.ContextVar("tool_turn", default=None)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(p.get("text") or "") for p in content if isinstance(p, Mapping))
    return str(content or "")


def begin_tool_turn(key: str, chat_history: Any, user_message: str,
                    request_context: Optional[Mapping[str, Any]] = None,
                    summary: str = "") -> ToolTurn:
    """Start the tool context of one chat turn in the current task (and the tasks it spawns).

    ``summary`` is the summary of turns no longer in ``chat_history``; it is kept ahead of
    the recent messages so values agreed earlier in a long chat still count.
    """
    convo = [
        {"role": str(m.get("role")), "content": _text(m.get("content"))}
        for m in (chat_history or []) if isinstance(m, Mapping) and m.get("role") in ("user", "assistant")
    ]
    convo.append({"role": "user", "content": str(user_message or "")})
    if str(summary or "").strip():
        convo.insert(0, {"role": "summary of earlier turns", "content": str(summary)})
    turn = ToolTurn(key=str(key or ""), conversation=convo, request_context=dict(request_context or {}))
    _TURN.set(turn)
    return turn


def current_turn() -> Optional[ToolTurn]:
    return _TURN.get()


@contextmanager
def use_turn(turn: Optional[ToolTurn]) -> Iterator[None]:
    """Run a block under ``turn`` (for callbacks served by tasks older than the turn)."""
    if turn is None:
        yield
        return
    token = _TURN.set(turn)
    try:
        yield
    finally:
        _TURN.reset(token)


@contextmanager
def rehearsal() -> Iterator[None]:
    """Side-effect calls inside this block are simulated (replays of earlier requests)."""
    turn = _TURN.get()
    token = _TURN.set(replace(turn, rehearsal=True) if turn else
                      ToolTurn(key="", conversation=[], rehearsal=True))
    try:
        yield
    finally:
        _TURN.reset(token)


def side_effecting(meta: Optional[Mapping[str, Any]]) -> bool:
    return bool(isinstance(meta, Mapping) and meta.get("side_effects") is True)


def call_key(tool_id: str, args: Mapping[str, Any]) -> str:
    def norm(v: Any) -> Any:
        if isinstance(v, str):
            return " ".join(v.split()).casefold()
        if isinstance(v, Mapping):
            return {str(k): norm(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return v

    canon = {k: norm(v) for k, v in dict(args or {}).items() if v not in (None, "")}
    canon = {k: (int(v) if isinstance(v, str) and v.isdigit() else v) for k, v in canon.items()}
    return f"{tool_id}:{json.dumps(canon, sort_keys=True, ensure_ascii=False, default=str)}"


@lru_cache(maxsize=4)
def _prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if limit <= 0 or len(text) <= limit else text[: limit - 1] + "…"


def _today(turn: Optional[ToolTurn]) -> str:
    from ..context.turn_guidance import clock_block

    return clock_block((turn.request_context if turn else {}) or {}, dt.datetime.now(dt.timezone.utc))


def render_check(tool_id: str, args: Mapping[str, Any], meta: Mapping[str, Any],
                 turn: Optional[ToolTurn], user_message: str) -> List[Dict[str, str]]:
    per = int(settings.TOOL_ACTION_CHECK_MESSAGE_CHARS or 0)
    convo = list(turn.conversation) if turn else [{"role": "user", "content": str(user_message or "")}]
    head = [m for m in convo[:1] if m["role"] not in ("user", "assistant")]
    convo = convo[len(head):]
    keep = int(settings.TOOL_ACTION_CHECK_HISTORY_MESSAGES or 0)
    if keep > 0:
        convo = convo[-keep:]
    lines = [f"{m['role']}: {m['content']}" for m in head]
    lines += [f"{m['role']}: {_clip(m['content'], per)}" for m in convo]
    obs = [
        f"- {o['tool']}({json.dumps(o['args'], ensure_ascii=False, default=str)}) -> {_clip(o['result'], per)}"
        for o in (turn.observations if turn else [])
    ]
    prompt = _prompt("action_check.md").format(
        clock=_today(turn),
        conversation="\n".join(lines) or "(empty)",
        observations="\n".join(obs) or "(none)",
        tool=tool_id,
        description=_clip(meta.get("description") or "", per),
        arguments=json.dumps(dict(args or {}), ensure_ascii=False, default=str),
    )
    return [{"role": "user", "content": prompt}]


_VERDICT_LINE = re.compile(r"VERDICT\s*:\s*\**\s*(YES|NO)\b", re.I)
_VERDICT = re.compile(r"\b(YES|NO)\b", re.I)


async def check_action(tool_id: str, args: Mapping[str, Any], meta: Mapping[str, Any],
                       turn: Optional[ToolTurn], user_message: str,
                       llm: Optional[Callable[..., Awaitable[Any]]] = None) -> Optional[str]:
    """None when the action may run, else the reason it may not."""
    from ..orchestrators._shared.llm_roles import gateway_llm

    call = llm or gateway_llm()
    reply = await call(render_check(tool_id, args, meta, turn, user_message), temperature=0.0,
                       max_tokens=int(settings.TOOL_ACTION_CHECK_MAX_TOKENS), enable_thinking=False)
    text = str(getattr(reply, "text", "") or "").strip()
    m = None
    if getattr(reply, "ok", False):
        verdicts = list(_VERDICT_LINE.finditer(text))
        m = verdicts[-1] if verdicts else _VERDICT.match(text)
    if not m:
        log.warning("action check unavailable for %s: %s", tool_id, getattr(reply, "error", "") or text[:120])
        return None if settings.TOOL_ACTION_CHECK_ON_ERROR == "allow" else "the confirmation check is unavailable"
    if m.group(1).upper() == "YES":
        return None
    after = [ln.strip(" .:-—*`") for ln in text[m.end():].split("\n")]
    reason = next((ln for ln in after[:2] if ln), "") or "the user has not asked for this action with these values"
    return _clip(reason, 300)


def _observe(turn: Optional[ToolTurn], tool_id: str, args: Mapping[str, Any], result: Any) -> None:
    if turn is None:
        return
    body = json.dumps(result, ensure_ascii=False, default=str) if not isinstance(result, str) else result
    turn.observations.append({"tool": tool_id, "args": dict(args or {}), "result": body})
    limit = int(settings.TOOL_ACTION_CHECK_OBSERVATIONS or 0)
    if limit > 0 and len(turn.observations) > limit:
        del turn.observations[:-limit]


async def guarded(tool_id: str, args: Mapping[str, Any], meta: Optional[Mapping[str, Any]],
                  user_message: str, run: Callable[[], Awaitable[Dict[str, Any]]],
                  llm: Optional[Callable[..., Awaitable[Any]]] = None) -> Dict[str, Any]:
    turn = _TURN.get()
    if not (settings.TOOL_SIDE_EFFECTS_POLICY and side_effecting(meta)):
        result = await run()
        _observe(turn, tool_id, args, result)
        return result
    if turn is not None and turn.rehearsal:
        return {"ok": True, "simulated": True, "tool_id": tool_id,
                "message": settings.TOOL_SIDE_EFFECTS_REHEARSAL_MESSAGE}
    if turn is None and not user_message:
        return await run()

    key = call_key(tool_id, args)
    pending = turn.inflight.get(key) if turn is not None else None
    if pending is not None:
        shared = await asyncio.shield(pending)
        if shared is not None:
            return dict(shared)
        # the first caller failed or was cancelled: this caller runs it itself
        return await guarded(tool_id, args, meta, user_message, run, llm)
    fut: "asyncio.Future[Optional[Dict[str, Any]]]" = asyncio.get_running_loop().create_future()
    if turn is not None:
        turn.inflight[key] = fut
    try:
        reason = None
        if settings.TOOL_ACTION_CHECK_ENABLED:
            reason = await check_action(tool_id, args, meta or {}, turn, user_message, llm)
        if reason:
            log.info("side-effect tool %s held: %s", tool_id, reason)
            result = {"ok": False, "error": "action_not_confirmed", "tool_id": tool_id,
                      "message": settings.TOOL_ACTION_HELD_TEMPLATE.format(reason=reason)}
        else:
            result = await run()
            _observe(turn, tool_id, args, result)
        fut.set_result(result)
        return result
    except BaseException:
        if turn is not None and turn.inflight.get(key) is fut:
            turn.inflight.pop(key, None)
        if not fut.done():
            fut.set_result(None)
        raise


__all__ = [
    "ToolTurn",
    "begin_tool_turn",
    "call_key",
    "check_action",
    "current_turn",
    "guarded",
    "rehearsal",
    "render_check",
    "side_effecting",
    "use_turn",
]
