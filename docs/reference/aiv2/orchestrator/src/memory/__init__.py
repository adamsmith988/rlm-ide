"""Chat memory: Qdrant vectors + llm-gateway embeddings (short-term per chat, long-term per user)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..config import settings
from .client import (
    PAIRED_ROLE,
    AdvancedMemoryClient,
    MemoryCategory,
    MemoryClient,
    MemoryScope,
    qdrant_base_url,
)

__all__ = [
    "AdvancedMemoryClient",
    "MemoryCategory",
    "MemoryClient",
    "MemoryScope",
    "format_memories_for_prompt",
    "format_memories_summary",
    "format_memory_prompt_block",
    "get_advanced_memory_client",
    "get_memory_client",
    "qdrant_base_url",
]

_memory_client: Optional[MemoryClient] = None
_advanced_memory_client: Optional[AdvancedMemoryClient] = None


def get_memory_client() -> MemoryClient:
    global _memory_client
    if _memory_client is None:
        _memory_client = MemoryClient()
    return _memory_client


def get_advanced_memory_client() -> AdvancedMemoryClient:
    global _advanced_memory_client
    if _advanced_memory_client is None:
        _advanced_memory_client = AdvancedMemoryClient()
    return _advanced_memory_client


DEFAULT_BLOCK_HEADER = """## Recalled memory
Retrieved from this user's memory store by relevance to the current message. Use an item only \
when it is relevant; the current conversation and attached files take precedence when they conflict. \
Never fill a tool or action argument (dates, counts, amounts, destinations, ids) from this memory \
alone: if the user has not given or confirmed that value in this conversation, ask for it. \
Do not mention this memory store unless asked."""


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if limit <= 0 or len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def format_memory_prompt_block(
    memories: Dict[str, List[Dict[str, Any]]] | List[Dict[str, Any]],
    *,
    max_chars: Optional[int] = None,
    item_chars: Optional[int] = None,
) -> str:
    """Prompt section for engines; empty string when nothing was recalled."""
    if isinstance(memories, list):
        memories = {
            "user": [m for m in memories if m.get("source") == "user"],
            "session": [m for m in memories if m.get("source") != "user"],
        }
    budget = int(max_chars if max_chars is not None else settings.MEMORY_PROMPT_MAX_CHARS or 3000)
    per_item = int(item_chars if item_chars is not None else settings.MEMORY_PROMPT_ITEM_CHARS or 600)
    header = (settings.MEMORY_PROMPT_HEADER or "").strip() or DEFAULT_BLOCK_HEADER
    sections: List[str] = []
    used = len(header)
    seen: set[str] = set()
    for key, title in (
        ("user", settings.MEMORY_PROMPT_USER_TITLE),
        ("session", settings.MEMORY_PROMPT_SESSION_TITLE),
    ):
        lines: List[str] = []
        for mem in memories.get(key) or []:
            text = _clip(mem.get("content", ""), per_item)
            if not text or text in seen:
                continue
            role = (mem.get("metadata") or {}).get("role")
            tagged = role and role != PAIRED_ROLE and key == "session"
            line = f"- ({role}) {text}" if tagged else f"- {text}"
            if used + len(line) + len(title) + 2 > budget:
                break
            seen.add(text)
            lines.append(line)
            used += len(line) + 1
        if lines:
            sections.append(title + "\n" + "\n".join(lines))
            used += len(title) + 2
    return header + "\n\n" + "\n\n".join(sections) if sections else ""


def format_memories_for_prompt(
    memories: Dict[str, List[Dict[str, Any]]] | List[Dict[str, Any]],
    include_session: bool = True,
    include_user: bool = True,
    max_per_section: int = 5,
) -> str:
    """Legacy formatter kept for callers of the old API."""
    if not memories:
        return ""
    if isinstance(memories, list):
        memories = {"session": memories}
    picked = {
        "session": (memories.get("session") or [])[:max_per_section] if include_session else [],
        "user": (memories.get("user") or [])[:max_per_section] if include_user else [],
    }
    return format_memory_prompt_block(picked)


def format_memories_summary(memories: Dict[str, List]) -> str:
    session_count = len(memories.get("session", []))
    user_count = len(memories.get("user", []))
    if session_count == 0 and user_count == 0:
        return "📭 حافظه‌ای یافت نشد"
    return f"📊 حافظه: {session_count} مورد از این گفتگو، {user_count} مورد کلی"
