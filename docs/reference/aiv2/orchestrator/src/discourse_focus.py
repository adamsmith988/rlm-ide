"""
Discourse Focus — tracks which session file(s) the conversation is about.

Resolves deictic references («این»، «this»، «it») using recent tool activity,
attachments, and optional LLM/heuristic file-ref resolution.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set

log = logging.getLogger(__name__)

_FOCUS_TOOLS = frozenset({
    "session_attach",
    "session_read",
    "session_read_file",
    "vision_info",
    "image_meta",
    "image_faces",
    "image_search",
    "audio_transcribe",
    "file_meta",
    "file_identify",
})

_LB_BEFORE = r"(?<![\w\u0600-\u06FF])"
_LB_AFTER = r"(?![\w\u0600-\u06FF])"

_DEICTIC_PATTERN = re.compile(
    _LB_BEFORE
    + r"(?:"
    r"این(?:\s+رو|\s+و)?|همین(?:\s+رو|\s+و)?|همون(?:\s+رو|\s+و)?"
    r"|رو(?:ی)?\s*این|با\s*این|روش|توش|داخلش"
    r"|this\s+one|this\b|that\b|it\b|do\s+this|to\s+this|on\s+this"
    r")"
    + _LB_AFTER,
    re.IGNORECASE | re.UNICODE,
)

_IMPLICIT_ACTION_PATTERN = re.compile(
    r"(?:کارش|کار\s*کن|انجام\s*بده|تحلیل\s*کن|بگو|نشون\s*بده|describe|analyze|process)\s*$",
    re.IGNORECASE | re.UNICODE,
)


def has_deictic_reference(text: str) -> bool:
    """True when the user likely refers to a recently discussed file without naming it."""
    if not text or not text.strip():
        return False
    body = text.strip()
    if _DEICTIC_PATTERN.search(body):
        return True
    # «این کارش بکن» — deictic + action verb, no explicit «عکس/فایل»
    if re.search(r"\b(?:این|همین|this)\b", body, re.I) and _IMPLICIT_ACTION_PATTERN.search(body):
        return True
    return False


def message_implies_file_context(text: str) -> bool:
    """Broader than message_references_files — includes deictic-only refs."""
    if not text:
        return False
    from .conversation_state import message_references_files

    return bool(message_references_files(text) or has_deictic_reference(text))


@dataclass
class DiscourseFocus:
    """Pointer to the file(s) currently under discussion."""

    active_artifact_ids: List[str] = field(default_factory=list)
    last_visual_id: Optional[str] = None
    last_text_file_id: Optional[str] = None
    recent_tool_targets: List[Dict[str, Any]] = field(default_factory=list)
    focus_reason: str = ""
    updated_turn: int = 0
    disambiguation_candidates: List[Dict[str, Any]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.active_artifact_ids and not self.last_visual_id and not self.last_text_file_id

    def primary_artifact_id(self) -> Optional[str]:
        if self.active_artifact_ids:
            return self.active_artifact_ids[0]
        return self.last_visual_id or self.last_text_file_id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "DiscourseFocus":
        if not isinstance(data, dict):
            return cls()
        return cls(
            active_artifact_ids=list(data.get("active_artifact_ids") or []),
            last_visual_id=data.get("last_visual_id"),
            last_text_file_id=data.get("last_text_file_id"),
            recent_tool_targets=list(data.get("recent_tool_targets") or []),
            focus_reason=str(data.get("focus_reason") or ""),
            updated_turn=int(data.get("updated_turn") or 0),
            disambiguation_candidates=list(data.get("disambiguation_candidates") or []),
        )


def _artifact_id_from_registry_entry(entry: Dict[str, Any]) -> Optional[str]:
    if not isinstance(entry, dict):
        return None
    aid = entry.get("artifact_id") or entry.get("id") or entry.get("file_id")
    return str(aid).strip() if aid else None


def _is_visual_entry(entry: Dict[str, Any]) -> bool:
    ftype = str(entry.get("type") or entry.get("media_type") or "").lower()
    mime = str(entry.get("mime_type") or "").lower()
    return ftype in {"image", "video"} or mime.startswith("image/") or mime.startswith("video/")


def _registry_by_id(
    file_registry: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for entry in file_registry or []:
        aid = _artifact_id_from_registry_entry(entry)
        if aid:
            out[aid.lower()] = entry
    return out


def extract_focus_from_history(
    chat_history: List[Dict[str, Any]],
    *,
    max_targets: int = 8,
) -> DiscourseFocus:
    """Rebuild discourse focus from recent turns (tool_calls + attachments)."""
    focus = DiscourseFocus()
    targets: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    turn = 0

    for row in chat_history or []:
        if not isinstance(row, dict):
            continue
        role = row.get("role")
        if role == "user":
            turn += 1
            for att in row.get("attachments") or []:
                if not isinstance(att, dict):
                    continue
                aid = _artifact_id_from_registry_entry(att)
                if aid and aid.lower() not in seen_ids:
                    seen_ids.add(aid.lower())
                    targets.append({
                        "artifact_id": aid,
                        "name": att.get("name") or att.get("filename") or "unnamed",
                        "tool": "user_attachment",
                        "turn": turn,
                    })

        if role != "assistant":
            continue

        for tc in row.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            name = (tc.get("name") or "").strip()
            if name not in _FOCUS_TOOLS:
                continue
            args = tc.get("arguments") if "arguments" in tc else tc.get("args")
            if isinstance(args, str):
                try:
                    import json
                    args = json.loads(args)
                except Exception:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            aid = str(
                args.get("file_id")
                or args.get("artifact_id")
                or args.get("filename")
                or args.get("name")
                or ""
            ).strip()
            if not aid:
                continue
            key = aid.lower()
            if key not in seen_ids:
                seen_ids.add(key)
                targets.append({
                    "artifact_id": aid,
                    "name": args.get("filename") or args.get("name") or aid,
                    "tool": name,
                    "turn": turn,
                })

        for tr in row.get("tool_results") or []:
            if not isinstance(tr, dict):
                continue
            tname = (tr.get("name") or tr.get("tool_id") or "").strip()
            if tname not in _FOCUS_TOOLS:
                continue
            args = tr.get("arguments") or tr.get("args") or {}
            if isinstance(args, dict):
                aid = str(
                    args.get("file_id")
                    or args.get("artifact_id")
                    or args.get("filename")
                    or ""
                ).strip()
                if aid and aid.lower() not in seen_ids:
                    seen_ids.add(aid.lower())
                    targets.append({
                        "artifact_id": aid,
                        "name": args.get("filename") or aid,
                        "tool": tname,
                        "turn": turn,
                    })

    focus.recent_tool_targets = targets[-max_targets:]
    focus.updated_turn = turn

    if targets:
        last = targets[-1]
        aid = str(last.get("artifact_id") or "")
        focus.active_artifact_ids = [aid] if aid else []
        focus.focus_reason = f"recent_{last.get('tool', 'tool')}_turn_{last.get('turn', turn)}"
        focus.updated_turn = int(last.get("turn") or turn)

    return focus


def enrich_focus_from_registry(
    focus: DiscourseFocus,
    file_registry: List[Dict[str, Any]],
) -> DiscourseFocus:
    """Set last_visual_id / last_text_file_id from registry + recent targets."""
    by_id = _registry_by_id(file_registry)
    for target in reversed(focus.recent_tool_targets):
        aid = str(target.get("artifact_id") or "").lower()
        entry = by_id.get(aid)
        if not entry:
            # filename match
            for reg in file_registry:
                name = str(reg.get("name") or reg.get("filename") or "").lower()
                if name and name == aid:
                    entry = reg
                    aid = (_artifact_id_from_registry_entry(reg) or aid).lower()
                    break
        if not entry:
            continue
        real_id = _artifact_id_from_registry_entry(entry)
        if not real_id:
            continue
        if _is_visual_entry(entry):
            focus.last_visual_id = real_id
        else:
            focus.last_text_file_id = real_id
        if real_id not in focus.active_artifact_ids:
            focus.active_artifact_ids = [real_id]
        focus.focus_reason = focus.focus_reason or f"registry_match_{target.get('tool')}"
        break
    return focus


def resolve_focus_for_message(
    *,
    user_message: str,
    file_registry: List[Dict[str, Any]],
    chat_history: List[Dict[str, Any]],
    stored_focus: Optional[DiscourseFocus] = None,
    llm_intent: Optional[Any] = None,
) -> DiscourseFocus:
    """
    Pick the artifact(s) the user is referring to this turn.
    Combines history-derived focus, stored focus, and optional FileReferenceIntent.
    """
    from .conversation_state import resolve_artifacts_from_intent

    focus = stored_focus or extract_focus_from_history(chat_history)
    focus = enrich_focus_from_registry(focus, file_registry)

    deictic = has_deictic_reference(user_message)
    if not deictic and not message_implies_file_context(user_message):
        return focus

    # LLM/heuristic intent from conversation_state
    if llm_intent is not None and getattr(llm_intent, "target", "none") != "none":
        resolved = resolve_artifacts_from_intent(llm_intent, file_registry)
        if resolved:
            ids = [_artifact_id_from_registry_entry(r) for r in resolved]
            ids = [i for i in ids if i]
            if ids:
                focus.active_artifact_ids = ids[:2]
                focus.focus_reason = f"smart_resolve_{getattr(llm_intent, 'resolved_by', 'intent')}"
                if len(resolved) == 1 and _is_visual_entry(resolved[0]):
                    focus.last_visual_id = ids[0]
                elif len(resolved) == 1:
                    focus.last_text_file_id = ids[0]
                return focus
            if len(resolved) > 1:
                focus.disambiguation_candidates = [
                    {
                        "artifact_id": _artifact_id_from_registry_entry(r),
                        "name": r.get("name") or r.get("filename"),
                        "type": r.get("type"),
                    }
                    for r in resolved[:5]
                ]
                return focus

    # Deictic: prefer last visual if recent topic was image
    primary = focus.primary_artifact_id()
    if primary:
        focus.active_artifact_ids = [primary]
        focus.focus_reason = focus.focus_reason or "deictic_primary_focus"
        return focus

    # Fallback: most recently touched file in registry
    if file_registry and deictic:
        last = file_registry[-1]
        aid = _artifact_id_from_registry_entry(last)
        if aid:
            focus.active_artifact_ids = [aid]
            focus.focus_reason = "deictic_fallback_latest_registry"
            if _is_visual_entry(last):
                focus.last_visual_id = aid
            else:
                focus.last_text_file_id = aid
    return focus


def registry_entries_for_focus(
    focus: DiscourseFocus,
    file_registry: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Map focus artifact ids to full registry rows."""
    if not focus.active_artifact_ids:
        return []
    by_id = _registry_by_id(file_registry)
    out: List[Dict[str, Any]] = []
    for aid in focus.active_artifact_ids:
        entry = by_id.get(str(aid).lower())
        if entry:
            out.append(entry)
    return out


def artifacts_for_focus(
    focus: DiscourseFocus,
    file_registry: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build artifact dicts suitable for multimodal attach."""
    artifacts: List[Dict[str, Any]] = []
    for entry in registry_entries_for_focus(focus, file_registry):
        if not _is_visual_entry(entry):
            continue
        path = entry.get("path") or entry.get("storage_path")
        aid = _artifact_id_from_registry_entry(entry)
        if not path and not aid:
            continue
        artifacts.append({
            "id": aid,
            "artifact_id": aid,
            "path": path,
            "storage_path": path,
            "mime_type": entry.get("mime_type"),
            "type": entry.get("type"),
            "name": entry.get("name") or entry.get("filename"),
            "filename": entry.get("name") or entry.get("filename"),
        })
    return artifacts


def focus_is_fresh(focus: DiscourseFocus, current_turn: int, *, max_turn_gap: int = 3) -> bool:
    if focus.is_empty():
        return False
    gap = current_turn - int(focus.updated_turn or 0)
    return gap <= max_turn_gap


def format_active_focus_block(focus: DiscourseFocus, file_registry: List[Dict[str, Any]]) -> str:
    """Human-readable block for system prompt / workspace runtime."""
    if focus.is_empty():
        return ""
    lines = [
        "ACTIVE DISCOURSE FOCUS (what the user likely means by «this/این»):",
    ]
    by_id = _registry_by_id(file_registry)
    for aid in focus.active_artifact_ids[:2]:
        entry = by_id.get(str(aid).lower()) or {}
        name = entry.get("name") or entry.get("filename") or aid
        ftype = entry.get("type") or "file"
        lines.append(f"- `{name}` ({ftype}) artifact_id=`{aid}`")
    if focus.focus_reason:
        lines.append(f"- resolved via: {focus.focus_reason}")
    if focus.disambiguation_candidates:
        lines.append("- AMBIGUOUS: multiple files match — ask user to pick:")
        for i, c in enumerate(focus.disambiguation_candidates[:4], 1):
            lines.append(f"  {chr(64 + i)}) `{c.get('name')}` id=`{c.get('artifact_id')}`")
    return "\n".join(lines)


def update_focus_from_tool_run(
    focus: DiscourseFocus,
    *,
    tool_name: str,
    tool_args: Dict[str, Any],
    ok: bool,
    turn: int,
) -> DiscourseFocus:
    """Update focus after a successful file-oriented tool execution."""
    if not ok or tool_name not in _FOCUS_TOOLS:
        return focus
    aid = str(
        tool_args.get("file_id")
        or tool_args.get("artifact_id")
        or tool_args.get("filename")
        or tool_args.get("name")
        or ""
    ).strip()
    if not aid:
        return focus
    focus.active_artifact_ids = [aid]
    focus.updated_turn = turn
    focus.focus_reason = f"tool_{tool_name}"
    target = {"artifact_id": aid, "name": tool_args.get("filename") or aid, "tool": tool_name, "turn": turn}
    focus.recent_tool_targets = (focus.recent_tool_targets + [target])[-8:]
    return focus


__all__ = [
    "DiscourseFocus",
    "artifacts_for_focus",
    "enrich_focus_from_registry",
    "extract_focus_from_history",
    "focus_is_fresh",
    "format_active_focus_block",
    "has_deictic_reference",
    "message_implies_file_context",
    "registry_entries_for_focus",
    "resolve_focus_for_message",
    "update_focus_from_tool_run",
]
