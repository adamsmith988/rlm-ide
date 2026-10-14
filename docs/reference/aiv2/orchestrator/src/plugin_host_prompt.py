"""
Per-plugin host-assistant prompt text (plugin.json ``llm_host_prompt``).

The tool-runner owns modules/plugins and serves each manifest at
``GET /plugins/{id}``; the orchestrator never reads plugin folders.
``load_llm_host_prompt`` fetches it (cached for ``PLUGIN_HOST_PROMPT_TTL_SEC``)
during the turn preflight; ``get_llm_host_prompt`` reads that cache for the
synchronous prompt builders of every engine.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

from .config import settings

log = logging.getLogger(__name__)

_CACHE: Dict[str, Tuple[float, str]] = {}


def _pid(plugin_id: Optional[str]) -> str:
    return str(plugin_id or "").strip()


async def load_llm_host_prompt(plugin_id: Optional[str], *, force_refresh: bool = False) -> str:
    """Fetch the plugin's ``llm_host_prompt`` from the tool-runner (cached)."""
    pid = _pid(plugin_id)
    if not pid:
        return ""
    hit = _CACHE.get(pid)
    ttl = float(getattr(settings, "PLUGIN_HOST_PROMPT_TTL_SEC", 300.0) or 0.0)
    if hit and not force_refresh and time.monotonic() - hit[0] < ttl:
        return hit[1]
    from .tools.executor import get_tool_executor

    meta = await get_tool_executor().get_plugin_manifest(pid)
    if meta is None:
        # Tool-runner unreachable: keep the last known prompt rather than dropping the page role.
        return hit[1] if hit else ""
    raw = meta.get("llm_host_prompt")
    text = raw.strip() if isinstance(raw, str) else ""
    _CACHE[pid] = (time.monotonic(), text)
    return text


def get_llm_host_prompt(plugin_id: Optional[str]) -> str:
    """The prompt loaded by :func:`load_llm_host_prompt` for this turn, or ``""``."""
    hit = _CACHE.get(_pid(plugin_id))
    return hit[1] if hit else ""


def clear_llm_host_prompt_cache() -> None:
    _CACHE.clear()
