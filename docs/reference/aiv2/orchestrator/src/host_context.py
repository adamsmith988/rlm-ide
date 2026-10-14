"""
Page-aware chat host context (floating AI assistant dock / page mini-chat).

Pages send an envelope under ChatRequest.context; which keys count as page
envelopes is configured (``HOST_PAGE_CONTEXT_KEYS``), never coded per plugin.
- active_plugin comes from explicit client fields only (no surface/path maps).
- The HOST PAGE CONTEXT block renders whatever the envelope carries, generically.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Tuple

CHAT_MENTIONS_CONTEXT_KEY = "chat_mentions_v1"
# Envelope fields the host itself reads; everything else is page content.
ACTIVE_PLUGIN_FIELD = "active_plugin"
ENGINE_FIELD = "orchestrator_id"
PAGE_FIELD = "surface"
_PROTOCOL_FIELDS = frozenset({"schema_version", ACTIVE_PLUGIN_FIELD, ENGINE_FIELD})

_MAX_PROMPT_CHARS = 3500
_MAX_FIELD_CHARS = 800
_MAX_MENTION_ROWS = 12
_HINTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "chat_mentions_hints.json")


def _as_dict(value: Any) -> Optional[Dict[str, Any]]:
    return value if isinstance(value, dict) else None


def page_context_keys() -> List[str]:
    """Context keys that carry a page envelope, highest priority first."""
    from .config import settings

    raw = str(getattr(settings, "HOST_PAGE_CONTEXT_KEYS", "") or "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def host_envelopes(context: Optional[Mapping[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """``(key, envelope)`` for every page envelope present, in priority order."""
    if not isinstance(context, Mapping):
        return []
    out: List[Tuple[str, Dict[str, Any]]] = []
    for key in page_context_keys():
        env = _as_dict(context.get(key))
        if env is not None:
            out.append((key, env))
    return out


def extract_host_envelopes(context: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [env for _key, env in host_envelopes(context)]


@lru_cache(maxsize=1)
def _mention_hints() -> Dict[str, Any]:
    with open(_HINTS_FILE, encoding="utf-8") as fh:
        return json.load(fh)


def _hint_lines(name: str) -> List[str]:
    value = _mention_hints().get(name)
    if isinstance(value, str):
        return [value] if value else []
    return [str(v) for v in value or [] if str(v)]


def build_mentions_prompt(context: Optional[Mapping[str, Any]]) -> str:
    """Structured @ mentions from FE — prefer exact artifact_id / case_id over filenames."""
    if not isinstance(context, Mapping):
        return ""
    mentions = _as_dict(context.get(CHAT_MENTIONS_CONTEXT_KEY))
    if not mentions:
        return ""

    files = mentions.get("files") if isinstance(mentions.get("files"), list) else []
    cases = mentions.get("cases") if isinstance(mentions.get("cases"), list) else []
    pages = mentions.get("pages") if isinstance(mentions.get("pages"), list) else []
    if not files and not cases and not pages:
        return ""

    lines = [*_hint_lines("header"), f"Source: {CHAT_MENTIONS_CONTEXT_KEY}"]
    if files:
        lines.append("files:")
        for row in files[:_MAX_MENTION_ROWS]:
            if not isinstance(row, dict):
                continue
            fid = str(row.get("id") or "").strip()
            label = str(row.get("label") or "").strip() or "?"
            path = str(row.get("path") or "").strip()
            mime = str(row.get("mime_type") or "").strip()
            bit = f"- {label}"
            if fid:
                bit += f" (artifact_id={fid})"
            if mime:
                bit += f" mime={mime}"
            if path:
                bit += f" path={path}"
            lines.append(bit)
            lines.extend(_hint_lines("file"))
    if cases:
        lines.append("cases:")
        for row in cases[:_MAX_MENTION_ROWS]:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("id") or "").strip()
            label = str(row.get("label") or "").strip() or "?"
            if not cid:
                continue
            lines.append(f"- {label} (case_id={cid})")
            outputs = row.get("outputs") if isinstance(row.get("outputs"), list) else []
            if outputs:
                handles = [str(o["handle"]) for o in outputs[:12] if isinstance(o, dict) and o.get("handle")]
                if handles:
                    lines.append(f"  → Selected outputs: {', '.join(handles)}")
                lines.extend(_hint_lines("case_with_outputs"))
            else:
                lines.extend(_hint_lines("case_without_outputs"))
    if pages:
        lines.append("pages:")
        for row in pages[:_MAX_MENTION_ROWS]:
            if not isinstance(row, dict):
                continue
            label = str(row.get("label") or "").strip() or "?"
            href = str(row.get("href") or row.get("id") or "").strip()
            lines.append(f"- {label}" + (f" ({href})" if href else ""))
    return "\n".join(lines)


def resolve_active_plugin(
    context: Optional[Mapping[str, Any]],
    *,
    explicit: Optional[str] = None,
) -> Optional[str]:
    """Prefer explicit context.active_plugin / envelope fields — never path maps."""
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if isinstance(context, Mapping):
        raw = context.get(ACTIVE_PLUGIN_FIELD)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    for env in extract_host_envelopes(context):
        nested = env.get(ACTIVE_PLUGIN_FIELD)
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def _compact_json(obj: Any, limit: int = _MAX_FIELD_CHARS) -> str:
    try:
        text = json.dumps(obj, ensure_ascii=False, default=str, separators=(",", ":"))
    except Exception:
        text = str(obj)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _format_envelope(key: str, envelope: Mapping[str, Any]) -> str:
    """Render any page envelope: scalars as ``- field: value``, structures as compact JSON."""
    lines = [*_hint_lines("page_header"), f"Source: {key}", *_hint_lines("page_guidance")]
    for field, value in envelope.items():
        if field in _PROTOCOL_FIELDS or value is None or value == "" or value == [] or value == {}:
            continue
        if isinstance(value, (dict, list)):
            lines.append(f"- {field}: {_compact_json(value)}")
        else:
            lines.append(f"- {field}: {value}")
    return "\n".join(lines)


def build_host_page_prompt(context: Optional[Mapping[str, Any]]) -> str:
    """Bounded prompt section: mentions + what the user currently sees."""
    parts: List[str] = []
    mentions = build_mentions_prompt(context)
    if mentions:
        parts.append(mentions)

    envelopes = host_envelopes(context)
    if envelopes:
        parts.append(_format_envelope(*envelopes[0]))

    if not parts:
        return ""

    text = "\n\n".join(parts).strip()
    if len(text) > _MAX_PROMPT_CHARS:
        return text[: _MAX_PROMPT_CHARS - 3] + "..."
    return text


def resolve_host_context(
    context: Optional[Mapping[str, Any]],
) -> Tuple[Optional[str], str]:
    """Return (active_plugin, host_page_prompt)."""
    plugin = resolve_active_plugin(context)
    prompt = build_host_page_prompt(context)
    return plugin, prompt
