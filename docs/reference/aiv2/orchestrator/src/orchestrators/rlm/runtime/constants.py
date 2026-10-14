"""Shared constants for the RLM runtime."""

from __future__ import annotations

import os
import re

HOST_REQUEST_TYPES = frozenset(
    {
        "rlm.run",
        "rlm.list_subagents",
        "rlm.delete_subagent",
        "rlm.find_models",
        "rlm.look",
        "agent_message.send",
        "agent_message.inbox",
        "agent_message.list_agents",
        "agent_observe.list",
        "agent_observe.get",
        "agent_observe.recent",
        "pools.search",
        "pools.resolve_file",
        "pools.list",
        "tools.call",
        "tools.describe",
        "harness.get",
        "harness.list",
        "harness.upsert",
        "harness.delete",
        "goal.get",
        "goal.create",
        "goal.complete",
        "rlm_heartbeat.list",
        "rlm_heartbeat.create",
        "rlm_heartbeat.delete",
        "compact.status",
        "compact.run",
        "refine.status",
        "refine.run",
    }
)

SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_]+")


def safe_ident(name: str, *, fallback: str = "item") -> str:
    text = SAFE_NAME_RE.sub("_", (name or "").strip())[:64].strip("_")
    if not text:
        return fallback
    if text[0].isdigit():
        text = f"n_{text}"
    return text


def idle_eviction_minutes() -> float:
    raw = os.environ.get("AIV2_RLM_IDLE_EVICTION_MINUTES", "60")
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 60.0
