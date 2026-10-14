"""Minimal schema helpers (vendored from agent_flow.agent_schema — no plugin import)."""

from __future__ import annotations

from typing import Any, List


def normalize_run_lifetime(raw: Any) -> str:
    """Return ``one_shot``, ``heartbeat``, or ``chat``."""
    v = str(raw or "one_shot").strip().lower().replace("-", "_")
    if v in ("heartbeat", "always_on", "alwayson", "resident", "daemon_loop"):
        return "heartbeat"
    if v in ("chat", "conversational", "multi_turn", "multiturn"):
        return "chat"
    return "one_shot"


def is_resident_run_lifetime(raw: Any) -> bool:
    """True when the daemon should keep the session warm (no idle eviction)."""
    return normalize_run_lifetime(raw) in ("heartbeat", "chat")


def is_turn_wait_run_lifetime(raw: Any) -> bool:
    """True when the active turn should wait for children / fan-in."""
    return normalize_run_lifetime(raw) in ("one_shot", "chat")


def normalize_tool_id_list(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    seen = set()
    for item in raw:
        tid = str(item or "").strip()
        if not tid or tid in seen:
            continue
        seen.add(tid)
        out.append(tid)
    return out


def coerce_bool(raw: Any, *, default: bool = False) -> bool:
    """Parse widget/JSON booleans; None/'' keep ``default``."""
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return bool(raw)
    s = str(raw).strip().lower()
    if s == "":
        return default
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return default


def agent_model_is_explicit(raw: Any) -> bool:
    """True when model is set and not an auto/default alias."""
    m = str(raw or "").strip().lower()
    return bool(m) and m not in ("auto", "default", "")


def clamp_int(raw: Any, *, default: int, min_v: int, max_v: int) -> int:
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = default
    return max(min_v, min(max_v, n))
