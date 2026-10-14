"""Turn-intent vocabulary (``tool_intents.json``) and tool matching by tool.json metadata.

Tools are never named here: a tool qualifies for an intent through the
``category`` / ``capabilities`` / ``plugin_id`` the tool-runner serves in its
manifest.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

_BUNDLED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tool_intents.json")


@lru_cache(maxsize=1)
def intents() -> Dict[str, Any]:
    from ..config import settings

    path = str(getattr(settings, "TOOL_INTENTS_FILE", "") or "").strip() or _BUNDLED
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _words(values: Any) -> List[str]:
    return [str(v) for v in values or [] if isinstance(v, str) and v and not v.startswith("_")]


def markers(intent: str) -> List[str]:
    return _words((intents().get("markers") or {}).get(intent))


def capabilities(tool: Mapping[str, Any]) -> Set[str]:
    return {str(c).strip().lower() for c in tool.get("capabilities") or [] if str(c).strip()}


def category(tool: Mapping[str, Any]) -> str:
    return str(tool.get("category") or "").strip().lower()


def matches(tool: Mapping[str, Any], rule: Optional[Mapping[str, Any]]) -> bool:
    """True when the tool's category or any capability is listed in ``rule``."""
    if not isinstance(rule, Mapping):
        return False
    cats = {c.lower() for c in _words(rule.get("categories"))}
    caps = {c.lower() for c in _words(rule.get("capabilities"))}
    return bool((cats and category(tool) in cats) or (caps & capabilities(tool)))


def session_capabilities_for(intent: str) -> Optional[Set[str]]:
    """Capabilities a session tool needs for ``intent``; ``None`` = every session tool."""
    table = intents().get("session_tools") or {}
    if intent not in table or intent.startswith("_"):
        return None
    return {c.lower() for c in _words(table[intent])}


def session_min_files(intent: str, tool: Mapping[str, Any]) -> int:
    table = (intents().get("session_tools_min_files") or {}).get(intent) or {}
    caps = capabilities(tool)
    return max([int(n) for cap, n in table.items() if cap.lower() in caps] or [0])


def mention_gate(tool: Mapping[str, Any]) -> Optional[List[str]]:
    """Mention words that unlock a gated tool, or ``None`` when the tool is not gated."""
    for group in (intents().get("mention_gated") or {}).get("groups") or []:
        if matches(tool, group):
            return [m.lower() for m in _words(group.get("mentions"))]
    return None


def mentioned(message: str, words: Iterable[str]) -> bool:
    text = (message or "").lower()
    return any(w in text for w in words)


__all__ = [
    "capabilities",
    "category",
    "intents",
    "markers",
    "matches",
    "mention_gate",
    "mentioned",
    "session_capabilities_for",
    "session_min_files",
]
