"""Per-turn guidance blocks every chat engine receives as ``<name>_prompt_block`` kwargs.

- ``clock``: today's date/time, so relative dates ("tomorrow") resolve to real days.
- ``persona``: the chat session's own ``system_prompt`` (set when the chat was created).
- ``loop_guard``: the user has sent essentially the same message several times in a row;
  the engine must stop repeating itself and offer escalation.

Every text is a setting template; an empty template disables its block.
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
from typing import Any, Dict, List, Mapping, Optional

from ..config import settings

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _zone(name: str) -> Optional[_dt.tzinfo]:
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 — unknown zone name
        return None


def clock_block(request_context: Optional[Mapping[str, Any]] = None, now: Optional[_dt.datetime] = None) -> str:
    template = (settings.HOST_CLOCK_TEMPLATE or "").strip()
    if not template:
        return ""
    ctx = request_context if isinstance(request_context, Mapping) else {}
    requested = str(ctx.get(settings.HOST_CLOCK_CONTEXT_KEY or "") or "").strip()
    name = requested if requested and _zone(requested) else (settings.HOST_CLOCK_TIMEZONE or "UTC")
    tz = _zone(name) or _dt.timezone.utc
    moment = (now or _dt.datetime.now(_dt.timezone.utc)).astimezone(tz)
    return template.format(
        date=moment.strftime("%Y-%m-%d"),
        weekday=moment.strftime("%A"),
        time=moment.strftime("%H:%M"),
        timezone=name,
    )


def persona_block(system_prompt: Any) -> str:
    text = str(system_prompt or "").strip()
    template = (settings.HOST_PERSONA_TEMPLATE or "").strip()
    if not text or not template:
        return ""
    limit = int(settings.HOST_PERSONA_MAX_CHARS or 0)
    if limit > 0 and len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return template.format(instructions=text)


def _words(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(str(text or ""))}


def _similar(a: str, b: str, threshold: float) -> bool:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= threshold


def _text(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
    return str(content or "")


def repeated_user_turns(user_message: str, history: List[Dict[str, Any]]) -> int:
    """How many consecutive user turns (this one included) say essentially the same thing."""
    threshold = float(settings.USER_LOOP_SIMILARITY or 0)
    count = 1
    for msg in reversed(history or []):
        if msg.get("role") != "user":
            continue
        if not _similar(user_message, _text(msg.get("content")), threshold):
            break
        count += 1
    return count


def loop_guard_block(user_message: str, history: List[Dict[str, Any]]) -> str:
    template = (settings.USER_LOOP_TEMPLATE or "").strip()
    repeats = int(settings.USER_LOOP_REPEATS or 0)
    if not template or repeats < 2:
        return ""
    count = repeated_user_turns(user_message, history)
    return template.format(count=count) if count >= repeats else ""


def build_guidance_blocks(
    *,
    user_message: str,
    chat_history: List[Dict[str, Any]],
    session_record: Optional[Mapping[str, Any]] = None,
    request_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    record = session_record if isinstance(session_record, Mapping) else {}
    blocks = {
        "clock": clock_block(request_context),
        "persona": persona_block(record.get("system_prompt")),
        "loop_guard": loop_guard_block(user_message, chat_history),
    }
    return {k: v for k, v in blocks.items() if v}


__all__ = [
    "build_guidance_blocks",
    "clock_block",
    "loop_guard_block",
    "persona_block",
    "repeated_user_turns",
]
