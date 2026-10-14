"""
Tool loop detection and mistake budget for native tools engine.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Set

log = logging.getLogger(__name__)


def _canonical_tool_key(tool_name: str, tool_args: Dict[str, Any]) -> str:
    try:
        args_str = json.dumps(tool_args or {}, sort_keys=True, ensure_ascii=False)
    except Exception:
        args_str = str(tool_args)
    return f"{tool_name}:{args_str}"


LOOP_BLOCKED_TOOL_REPAIR = (
    "[Tool loop blocked — mandatory]\n"
    "You repeated the same tool call (or the same tool with no progress). "
    "That call was NOT executed again.\n\n"
    "Rules for your next message:\n"
    "1. Do NOT call the same tool with the same arguments again.\n"
    "2. Do NOT restate the same plan/monologue — answer the user directly.\n"
    "3. Use evidence already in tool results above (filenames, file_id, read_file output).\n"
    "4. If the user asked about names/people IN a file you already read, answer from that "
    "content or call `session_read_file` with a different offset — do not search for unrelated keywords.\n"
    "5. If only one file exists in the workspace, say so clearly instead of searching for another file."
)


class ToolLoopGuard:
    """Track repeated identical tool calls within one user turn."""

    def __init__(self, *, block_at: int = 2) -> None:
        # 1st identical call runs; 2nd+ identical calls are blocked (Continue-style).
        self.block_at = max(2, block_at)
        self._counts: Dict[str, int] = {}
        self._blocked_keys: Set[str] = set()

    def record(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Return ``ok`` (execute) or ``block`` (skip execution)."""
        key = _canonical_tool_key(tool_name, tool_args)
        if key in self._blocked_keys:
            return "block"
        n = self._counts.get(key, 0) + 1
        self._counts[key] = n
        if n >= self.block_at:
            self._blocked_keys.add(key)
            return "block"
        return "ok"

    def is_blocked(self, tool_name: str, tool_args: Dict[str, Any]) -> bool:
        return _canonical_tool_key(tool_name, tool_args) in self._blocked_keys

    def blocked_message(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        args_preview = ""
        try:
            args_preview = json.dumps(tool_args or {}, ensure_ascii=False)[:120]
        except Exception:
            args_preview = str(tool_args)[:120]
        return (
            f"[LOOP_BLOCKED] `{tool_name}` with args {args_preview} was blocked — "
            "identical call repeated too many times with no new information.\n"
            "Do NOT retry this call. Answer from existing tool results or use a different tool/args."
        )

    def repair_user_message(self, user_message: Optional[str] = None) -> str:
        base = LOOP_BLOCKED_TOOL_REPAIR
        um = (user_message or "").strip()
        if um:
            return f"{base}\n\nUser question (answer this, do not loop tools):\n{um[:500]}"
        return base


class MistakeBudget:
    """Limit repair retries before honest synthesis."""

    def __init__(self, *, max_mistakes: int = 2) -> None:
        self.max_mistakes = max_mistakes
        self.count = 0

    def record_mistake(self) -> None:
        self.count += 1

    def exhausted(self) -> bool:
        return self.count >= self.max_mistakes


__all__ = [
    "LOOP_BLOCKED_TOOL_REPAIR",
    "MistakeBudget",
    "ToolLoopGuard",
]
