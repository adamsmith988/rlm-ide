"""Per-turn host prompt blocks (RAG, memory, …) for engines with a persistent transcript.

The host workspace exposes each block as an engine kwarg named ``<name>_prompt_block``.
Engines that keep a live system message swap these sections in place every turn
(marker-delimited), so they never accumulate in the transcript.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Tuple

BLOCK_SUFFIX = "_prompt_block"
_MARK = "host-context"
_SECTION_RE = re.compile(
    rf"\n*<!-- {_MARK}:(?P<name>[\w.-]+) -->\n.*?\n<!-- /{_MARK}:(?P=name) -->\n*", re.S
)
_RUN_RE = re.compile(rf"(?:{_SECTION_RE.pattern})+", re.S)


def prompt_blocks(kwargs: Mapping[str, Any], selection: Any = "*") -> Dict[str, str]:
    """``{name: text}`` for every non-empty ``*_prompt_block`` kwarg allowed by ``selection``.

    ``selection``: ``"*"`` (all), a comma-separated string or list of names, or empty (none).
    """
    if isinstance(selection, str):
        wanted = {s.strip() for s in selection.split(",") if s.strip()}
    else:
        wanted = {str(s).strip() for s in (selection or []) if str(s).strip()}
    out: Dict[str, str] = {}
    for key, value in kwargs.items():
        if not key.endswith(BLOCK_SUFFIX) or not isinstance(value, str) or not value.strip():
            continue
        name = key[: -len(BLOCK_SUFFIX)]
        if "*" in wanted or name in wanted:
            out[name] = value.strip()
    return out


def page_prompt_block(active_plugin: Any = None, host_page_prompt: Any = None) -> str:
    """The page the chat runs on: the plugin's ``llm_host_prompt`` plus the client page context."""
    from src.plugin_host_prompt import get_llm_host_prompt

    plugin_text = get_llm_host_prompt(str(active_plugin or "").strip() or None)
    page_text = str(host_page_prompt or "").strip()
    return "\n\n".join(x for x in (plugin_text, page_text) if x)


def render_blocks(blocks: Mapping[str, str]) -> str:
    return "\n\n".join(
        f"<!-- {_MARK}:{name} -->\n{text}\n<!-- /{_MARK}:{name} -->"
        for name, text in sorted(blocks.items())
    )


def replace_host_context(system_text: str, blocks: Mapping[str, str]) -> str:
    """Drop every previous host-context section, then append this turn's blocks."""
    text = _RUN_RE.sub("\n\n", str(system_text or "")).strip()
    rendered = render_blocks(blocks)
    if rendered:
        text = f"{text}\n\n{rendered}" if text else rendered
    return text


def split_host_context(system_text: str) -> Tuple[str, str]:
    """``(text without host-context sections, the sections joined in order)``."""
    text = str(system_text or "")
    sections = [m.group(0).strip() for m in _SECTION_RE.finditer(text)]
    return _RUN_RE.sub("\n\n", text).strip(), "\n\n".join(sections)


__all__ = [
    "BLOCK_SUFFIX",
    "page_prompt_block",
    "prompt_blocks",
    "render_blocks",
    "replace_host_context",
    "split_host_context",
]
