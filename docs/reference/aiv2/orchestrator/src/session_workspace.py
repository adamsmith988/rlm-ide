"""Canonical per-request session workspace for the Orchestrator.

Inspired by the Continue extension's separation between ``constructMessages``
(state assembly) and ``streamResponse`` (execution). Both phases there read
from a single canonical Redux slice. The AIv2 Orchestrator previously
re-implemented that assembly in three different endpoints (`/chat`,
`/chat/stream`, `/chat/stream/full`) with subtle drift between them.

This module unifies that assembly:

* ``SessionWorkspaceState`` — a single, typed view of everything the agent
  loop needs (history, registry, attachments, pinned facts, memory, etc.).
* ``build_session_workspace`` — async builder that fans out every I/O fetch
  in parallel (``asyncio.gather``), then runs the deterministic, in-memory
  decisions (auto-attach, topic-change clearing) once. This collapses the
  three duplicated preflight blocks into a single tested path and removes
  the serial waterfall that previously cost ~hundreds of ms per request.

The state object exposes ``to_engine_kwargs()`` so the three chat endpoints
in :mod:`main` can forward it directly to the streaming/non-streaming agent
entry points without re-listing every keyword.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import settings
from .schemas import ChatRequest

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers re-used by the workspace builder (kept module-local for clarity).
# ---------------------------------------------------------------------------


def _registry_file_category(mime: str, filename: str = "") -> str:
    """Map mime/filename to registry ``type`` (image, video, …)."""
    from .chat_history import ChatHistoryClient

    if mime:
        cat = ChatHistoryClient._mime_to_category(mime)
        if cat != "unknown":
            return cat
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            return ChatHistoryClient._mime_to_category(guessed)
    return "unknown"


def _registry_entry_from_artifact(
    art: Dict[str, Any],
    *,
    source: str = "fresh_turn",
) -> Optional[Dict[str, Any]]:
    """Build a registry row from an upload/artifact dict (same turn or attach)."""
    if not isinstance(art, dict):
        return None
    path = (
        art.get("path")
        or art.get("storage_path")
        or art.get("file_path")
    )
    aid = art.get("artifact_id") or art.get("id")
    if not path and not aid:
        return None
    name = (
        art.get("name")
        or art.get("filename")
        or art.get("original_filename")
        or "unknown"
    )
    mime = str(art.get("mime_type") or art.get("content_type") or "")
    ftype = str(art.get("type") or art.get("media_type") or "").strip().lower()
    if not ftype or ftype == "unknown":
        ftype = _registry_file_category(mime, str(name))
    if not mime and path:
        guessed, _ = mimetypes.guess_type(str(path))
        mime = guessed or mime
        if ftype == "unknown" and mime:
            ftype = _registry_file_category(mime, str(name))
    return {
        "ordinal": 0,
        "turn": 0,
        "name": name,
        "path": str(path or ""),
        "mime_type": mime,
        "type": ftype,
        "artifact_id": aid,
        "knowledge": [],
        "source": source,
    }


def _append_registry_entry(
    registry: List[Dict[str, Any]],
    entry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Append if not already present (by artifact_id or path)."""
    aid = str(entry.get("artifact_id") or entry.get("id") or "")
    path = str(entry.get("path") or entry.get("storage_path") or "")
    for existing in registry:
        ex_aid = str(existing.get("artifact_id") or existing.get("id") or "")
        ex_path = str(existing.get("path") or existing.get("storage_path") or "")
        if aid and ex_aid == aid:
            return registry
        if path and ex_path == path:
            return registry
    out = list(registry)
    row = dict(entry)
    row["ordinal"] = len(out) + 1
    out.append(row)
    return out


def finalize_file_registry_for_llm(
    registry: List[Dict[str, Any]],
    *,
    fresh_artifacts: Optional[List[Dict[str, Any]]] = None,
    storage_artifacts: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Registry used for tool selection must include uploads + storage metadata."""
    merged = list(registry or [])
    for art in fresh_artifacts or []:
        row = _registry_entry_from_artifact(art, source="fresh_turn")
        if row:
            merged = _append_registry_entry(merged, row)
    return _merge_registry_with_storage_artifacts(merged, storage_artifacts or [])


def _count_registry_images(registry: List[Dict[str, Any]]) -> int:
    n = 0
    for entry in registry or []:
        if not isinstance(entry, dict):
            continue
        ft = str(entry.get("type") or "").lower()
        mime = str(entry.get("mime_type") or "").lower()
        if ft == "image" or mime.startswith("image/"):
            n += 1
    return n


def _visual_artifacts_from_file_registry(
    file_registry: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Project session registry rows to artifact-like dicts (visual only)."""
    artifacts: List[Dict[str, Any]] = []
    for entry in file_registry or []:
        if not isinstance(entry, dict):
            continue
        file_type = str(entry.get("type") or "").lower()
        mime_type = str(entry.get("mime_type") or "").lower()
        if file_type not in {"image", "video"} and not (
            mime_type.startswith("image/") or mime_type.startswith("video/")
        ):
            continue
        path = entry.get("path") or entry.get("storage_path")
        artifact_id = entry.get("artifact_id") or entry.get("id") or entry.get("file_id")
        if not path and not artifact_id:
            continue
        artifacts.append(
            {
                "id": artifact_id,
                "artifact_id": artifact_id,
                "path": path,
                "storage_path": path,
                "mime_type": entry.get("mime_type"),
                "type": entry.get("type"),
                "name": entry.get("name") or entry.get("filename") or entry.get("original_filename"),
                "filename": entry.get("name") or entry.get("filename") or entry.get("original_filename"),
            }
        )
    return artifacts


def _visual_artifacts_from_attachments(
    attachments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalize previous user-message attachments to visual artifacts."""
    artifacts: List[Dict[str, Any]] = []
    for entry in attachments or []:
        if not isinstance(entry, dict):
            continue
        file_type = str(entry.get("type") or entry.get("media_type") or "").lower()
        mime_type = str(entry.get("mime_type") or entry.get("content_type") or "").lower()
        if file_type not in {"image", "video"} and not (
            mime_type.startswith("image/") or mime_type.startswith("video/")
        ):
            continue
        path = entry.get("path") or entry.get("storage_path")
        artifact_id = entry.get("artifact_id") or entry.get("id") or entry.get("file_id")
        if not path and not artifact_id:
            continue
        artifacts.append(
            {
                "id": artifact_id,
                "artifact_id": artifact_id,
                "path": path,
                "storage_path": path,
                "mime_type": entry.get("mime_type") or entry.get("content_type"),
                "type": entry.get("type") or entry.get("media_type"),
                "name": entry.get("name") or entry.get("filename") or entry.get("original_filename"),
                "filename": entry.get("name") or entry.get("filename") or entry.get("original_filename"),
            }
        )
    return artifacts


def _merge_visual_artifacts(
    attachments_visual: List[Dict[str, Any]],
    registry_visual: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Deduplicate (oldest-first) the union of visual artifacts from both
    sources. ``get_session_attachments`` returns newest-first; we reverse it
    so the merged list matches the registry ordering (oldest first)."""
    merged: List[Dict[str, Any]] = []
    seen: set = set()

    def _add(items: List[Dict[str, Any]]) -> None:
        for item in items:
            if not isinstance(item, dict):
                continue
            key = str(
                item.get("artifact_id")
                or item.get("id")
                or item.get("path")
                or item.get("storage_path")
                or item.get("name")
                or ""
            ).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)

    _add(list(reversed(attachments_visual or [])))
    _add(registry_visual or [])
    return merged


def _request_storage_auth_context(request: ChatRequest) -> Dict[str, Any]:
    """Build the user-scoped auth context the chat_history/storage clients
    expect (mirrors ``main._request_storage_auth_context``)."""
    ctx = request.context if isinstance(request.context, dict) else {}
    out: Dict[str, Any] = {"user_id": request.user_id}
    if ctx.get("group_id"):
        out["group_id"] = ctx["group_id"]
    if ctx.get("case_id"):
        out["case_id"] = ctx["case_id"]
    return out


def _context_rule_policies(context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Mirror ``main._context_rule_policies``."""
    if not context:
        return None
    rp = context.get("rule_policies")
    return rp if isinstance(rp, dict) else None


def _last_user_message_text(chat_history: List[Dict[str, Any]]) -> str:
    for msg in reversed(chat_history or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content") or ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text") or ""))
                return " ".join(parts).strip()
    return ""


def _runtime_entry_key(entry: Dict[str, Any]) -> str:
    return str(
        entry.get("artifact_id")
        or entry.get("id")
        or entry.get("path")
        or entry.get("storage_path")
        or entry.get("name")
        or ""
    ).strip()


def _runtime_candidate_strings(entry: Dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for value in (
        entry.get("artifact_id"),
        entry.get("id"),
        entry.get("file_id"),
        entry.get("name"),
        entry.get("filename"),
        entry.get("original_filename"),
        entry.get("path"),
        entry.get("storage_path"),
    ):
        if not value:
            continue
        text = str(value).strip()
        if not text:
            continue
        out.add(text.lower())
        out.add(os.path.basename(text).lower())
    return out


def _merge_registry_with_storage_artifacts(
    file_registry: List[Dict[str, Any]],
    storage_artifacts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Combine chronological session registry with storage metadata."""
    by_id: Dict[str, Dict[str, Any]] = {}
    by_path: Dict[str, Dict[str, Any]] = {}
    for art in storage_artifacts or []:
        if not isinstance(art, dict):
            continue
        aid = str(art.get("id") or art.get("artifact_id") or "").strip()
        spath = str(art.get("storage_path") or art.get("path") or "").strip()
        if aid:
            by_id[aid] = art
        if spath:
            by_path[spath] = art

    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for entry in file_registry or []:
        if not isinstance(entry, dict):
            continue
        aid = str(entry.get("artifact_id") or entry.get("id") or "").strip()
        spath = str(entry.get("path") or entry.get("storage_path") or "").strip()
        storage_art = by_id.get(aid) if aid else None
        if storage_art is None and spath:
            storage_art = by_path.get(spath)
        merged_entry = dict(storage_art or {})
        merged_entry.update(entry)
        if storage_art:
            merged_entry.setdefault("id", storage_art.get("id"))
            merged_entry.setdefault("storage_path", storage_art.get("storage_path"))
            merged_entry.setdefault("original_filename", storage_art.get("original_filename"))
            merged_entry.setdefault("file_size_bytes", storage_art.get("file_size_bytes"))
            merged_entry.setdefault("metadata", storage_art.get("metadata") or {})
            merged_entry.setdefault("media_type", storage_art.get("media_type"))
            merged_entry.setdefault("mime_type", storage_art.get("mime_type"))
        key = _runtime_entry_key(merged_entry)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(merged_entry)

    # Include storage-only session artifacts too (e.g. session_attach links).
    for art in storage_artifacts or []:
        if not isinstance(art, dict):
            continue
        key = _runtime_entry_key(art)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(
            {
                "artifact_id": art.get("id") or art.get("artifact_id"),
                "id": art.get("id") or art.get("artifact_id"),
                "name": art.get("original_filename") or art.get("filename") or art.get("name") or "unnamed",
                "path": art.get("storage_path") or art.get("path"),
                "storage_path": art.get("storage_path") or art.get("path"),
                "mime_type": art.get("mime_type"),
                "type": art.get("media_type") or "file",
                "file_size_bytes": art.get("file_size_bytes"),
                "metadata": art.get("metadata") or {},
                "knowledge": [],
                "folder_path": art.get("folder_path") or "",
            }
        )
    return merged


def _user_query_terms(text: str) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for tok in re.findall(r"[\w.\-]{2,}", text or "", flags=re.UNICODE):
        t = tok.strip().lower()
        if len(t) < 2:
            continue
        if t in {"this", "that", "file", "image", "video", "audio", "please"}:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:12]


def _extract_text_snippet(extracted_text: str, query_terms: List[str], limit: int = 700) -> str:
    text = (extracted_text or "").strip()
    if not text:
        return ""
    lower = text.lower()
    for term in query_terms:
        idx = lower.find(term.lower())
        if idx >= 0:
            start = max(0, idx - 180)
            end = min(len(text), idx + 420)
            snippet = text[start:end].strip()
            return snippet[:limit]
    return text[:limit]


def _build_workspace_runtime(
    *,
    user_message: str,
    file_registry: List[Dict[str, Any]],
    storage_artifacts: List[Dict[str, Any]],
    active_artifacts: List[Dict[str, Any]],
    wants_file: bool,
    discourse_focus: Optional[Any] = None,
) -> Dict[str, Any]:
    """Create a Continue-like passive workspace runtime for the current turn."""
    from .discourse_focus import (
        DiscourseFocus,
        format_active_focus_block,
        registry_entries_for_focus,
    )

    merged_entries = _merge_registry_with_storage_artifacts(file_registry, storage_artifacts)
    query_lc = (user_message or "").lower()
    query_terms = _user_query_terms(user_message)

    file_index: List[Dict[str, Any]] = []
    selected_files: List[Dict[str, Any]] = []
    text_snippets: List[Dict[str, Any]] = []
    visual_targets: List[Dict[str, Any]] = []
    seen_selected: set[str] = set()

    focus_obj = discourse_focus if isinstance(discourse_focus, DiscourseFocus) else DiscourseFocus.from_dict(
        discourse_focus if isinstance(discourse_focus, dict) else None
    )
    focus_ids = {str(a).lower() for a in (focus_obj.active_artifact_ids or [])}

    active_keys = {
        str(a.get("artifact_id") or a.get("id") or a.get("path") or "").strip()
        for a in (active_artifacts or [])
        if isinstance(a, dict)
    }

    for entry in merged_entries:
        name = entry.get("name") or entry.get("original_filename") or "unnamed"
        artifact_id = entry.get("artifact_id") or entry.get("id")
        path = entry.get("path") or entry.get("storage_path")
        mime_type = entry.get("mime_type") or ""
        file_type = entry.get("type") or entry.get("media_type") or "file"
        size_bytes = entry.get("file_size_bytes") or entry.get("size_bytes")
        file_index.append(
            {
                "artifact_id": artifact_id,
                "name": name,
                "type": file_type,
                "mime_type": mime_type,
                "size_bytes": size_bytes,
                "turn": entry.get("turn"),
                "ordinal": entry.get("ordinal"),
            }
        )

        exact_match = any(
            cand and cand in query_lc
            for cand in _runtime_candidate_strings(entry)
        )
        active_match = bool(active_keys & {str(artifact_id or ""), str(path or "")})
        focus_match = bool(artifact_id and str(artifact_id).lower() in focus_ids)
        if exact_match or active_match or focus_match:
            key = _runtime_entry_key(entry)
            if key and key not in seen_selected:
                seen_selected.add(key)
                reason = "discourse_focus" if focus_match else (
                    "exact_reference" if exact_match else "active_context"
                )
                selected_files.append(
                    {
                        "artifact_id": artifact_id,
                        "name": name,
                        "path": path,
                        "type": file_type,
                        "mime_type": mime_type,
                        "knowledge": entry.get("knowledge") or [],
                        "match_reason": reason,
                    }
                )

    # Discourse focus entries not yet in selected_files
    for entry in registry_entries_for_focus(focus_obj, file_registry):
        key = _runtime_entry_key(entry)
        if key and key not in seen_selected:
            seen_selected.add(key)
            selected_files.insert(
                0,
                {
                    "artifact_id": entry.get("artifact_id") or entry.get("id"),
                    "name": entry.get("name") or entry.get("filename") or "unnamed",
                    "path": entry.get("path") or entry.get("storage_path"),
                    "type": entry.get("type") or entry.get("media_type") or "file",
                    "mime_type": entry.get("mime_type") or "",
                    "knowledge": entry.get("knowledge") or [],
                    "match_reason": "discourse_focus",
                },
            )

    if wants_file and not selected_files and len(merged_entries) == 1:
        entry = merged_entries[0]
        selected_files.append(
            {
                "artifact_id": entry.get("artifact_id") or entry.get("id"),
                "name": entry.get("name") or entry.get("original_filename") or "unnamed",
                "path": entry.get("path") or entry.get("storage_path"),
                "type": entry.get("type") or entry.get("media_type") or "file",
                "mime_type": entry.get("mime_type") or "",
                "knowledge": entry.get("knowledge") or [],
                "match_reason": "single_session_file",
            }
        )

    selected_ids = {
        str(sf.get("artifact_id") or "").strip()
        for sf in selected_files
        if str(sf.get("artifact_id") or "").strip()
    }
    selected_paths = {
        str(sf.get("path") or "").strip()
        for sf in selected_files
        if str(sf.get("path") or "").strip()
    }
    allow_inline_content = bool(selected_ids or selected_paths) or (wants_file and len(merged_entries) == 1)

    for entry in merged_entries:
        key = _runtime_entry_key(entry)
        entry_id = str(entry.get("artifact_id") or entry.get("id") or "").strip()
        entry_path = str(entry.get("path") or entry.get("storage_path") or "").strip()
        if selected_ids or selected_paths:
            if not (
                (entry_id and entry_id in selected_ids)
                or (entry_path and entry_path in selected_paths)
            ):
                continue
        elif not allow_inline_content:
            continue
        meta = entry.get("metadata") or {}
        if isinstance(meta, dict):
            text_extract = meta.get("text_extract") if isinstance(meta.get("text_extract"), dict) else {}
            extracted_text = text_extract.get("extracted_text") if isinstance(text_extract, dict) else ""
            if isinstance(extracted_text, str) and extracted_text.strip():
                snippet = _extract_text_snippet(extracted_text, query_terms)
                if snippet:
                    text_snippets.append(
                        {
                            "artifact_id": entry.get("artifact_id") or entry.get("id"),
                            "name": entry.get("name") or entry.get("original_filename") or "unnamed",
                            "snippet": snippet,
                            "source": "metadata.text_extract",
                        }
                    )
        mime = str(entry.get("mime_type") or "").lower()
        ftype = str(entry.get("type") or entry.get("media_type") or "").lower()
        if ftype in {"image", "video"} or mime.startswith("image/") or mime.startswith("video/"):
            visual_targets.append(
                {
                    "id": entry.get("artifact_id") or entry.get("id"),
                    "artifact_id": entry.get("artifact_id") or entry.get("id"),
                    "path": entry.get("path") or entry.get("storage_path"),
                    "storage_path": entry.get("path") or entry.get("storage_path"),
                    "mime_type": entry.get("mime_type"),
                    "type": entry.get("type") or entry.get("media_type"),
                    "name": entry.get("name") or entry.get("original_filename") or "unnamed",
                    "filename": entry.get("name") or entry.get("original_filename") or "unnamed",
                }
            )

    # Tiered catalog: Active Focus / Recent Touched / compact index
    recent_touched: List[Dict[str, Any]] = []
    seen_recent: set[str] = set()
    for target in reversed(focus_obj.recent_tool_targets or []):
        aid = str(target.get("artifact_id") or "").lower()
        if not aid or aid in seen_recent:
            continue
        for entry in merged_entries:
            eid = str(entry.get("artifact_id") or entry.get("id") or "").lower()
            ename = str(entry.get("name") or entry.get("filename") or "").lower()
            if eid == aid or ename == aid:
                seen_recent.add(aid)
                recent_touched.append({
                    "artifact_id": entry.get("artifact_id") or entry.get("id"),
                    "name": entry.get("name") or entry.get("filename"),
                    "type": entry.get("type") or "file",
                    "last_tool": target.get("tool"),
                })
                break
        if len(recent_touched) >= 5:
            break

    active_focus_rows = [
        {
            "artifact_id": sf.get("artifact_id"),
            "name": sf.get("name"),
            "type": sf.get("type"),
            "match_reason": sf.get("match_reason"),
        }
        for sf in selected_files[:2]
    ]

    compact_index = file_index[:20]
    overflow = max(0, len(file_index) - len(compact_index))

    return {
        "available_files_count": len(file_index),
        "file_index": compact_index,
        "file_index_overflow": overflow,
        "tiered_catalog": {
            "active_focus": active_focus_rows,
            "recent_touched": recent_touched,
            "compact_index": compact_index,
            "overflow_count": overflow,
        },
        "active_focus_block": format_active_focus_block(focus_obj, file_registry),
        "discourse_focus": focus_obj.to_dict(),
        "selected_files": selected_files[:3],
        "text_snippets": text_snippets[:2],
        "visual_targets": visual_targets[:2],
        "query_terms": query_terms,
        "prefer_runtime_first": bool(wants_file and (selected_files or text_snippets)),
        "needs_session_files_pagination": len(file_index) > 20,
    }


# ---------------------------------------------------------------------------
# Canonical state object
# ---------------------------------------------------------------------------


@dataclass
class SessionWorkspaceState:
    """Single source of truth for one chat request's pre-loop context.

    The dataclass intentionally mirrors the kwargs each agent entry point
    (``stream_agent``, ``stream_agent_cline``, ``run_react_agent``,
    ``stream_agent_for_full_endpoint``, ``run_planning_agent_*``) expects,
    so ``to_engine_kwargs()`` can forward without re-typing them.
    """

    # ---- Request identity ----
    session_id: str
    user_id: str
    user_message: str
    rule_policies: Optional[Dict[str, Any]] = None
    storage_auth_context: Dict[str, Any] = field(default_factory=dict)

    # ---- Fresh inputs from the user this turn ----
    fresh_artifacts: List[Dict[str, Any]] = field(default_factory=list)
    has_fresh_artifacts: bool = False

    # ---- Fetched session context (parallelised) ----
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
    history_summary: str = ""
    file_registry: List[Dict[str, Any]] = field(default_factory=list)
    previous_attachments: List[Dict[str, Any]] = field(default_factory=list)
    memory_context: List[Dict[str, Any]] = field(default_factory=list)
    memory_prompt_block: str = ""
    memory_available: bool = True
    rag_context: List[Dict[str, Any]] = field(default_factory=list)
    rag_meta: Dict[str, Any] = field(default_factory=dict)
    rag_prompt_block: str = ""
    session_record: Dict[str, Any] = field(default_factory=dict)
    # Per-turn guidance (clock, persona, loop guard) → ``<name>_prompt_block`` kwargs.
    guidance_blocks: Dict[str, str] = field(default_factory=dict)

    # ---- Derived (post-fetch) state ----
    conversation_state: Any = None
    pinned_facts: Optional[Dict[str, Any]] = None
    discourse_focus: Any = None
    task_checklist_state: Dict[str, Any] = field(default_factory=dict)
    workspace_runtime: Dict[str, Any] = field(default_factory=dict)
    active_plugin: Optional[str] = None
    host_page_prompt: str = ""
    # Raw ChatRequest.context (for case mention materialization / RLM pool).
    request_context: Optional[Dict[str, Any]] = None
    # Resolved case-mention inventory (files + tool evidence texts).
    case_inventory: Optional[Dict[str, Any]] = None
    # SelfCompact trajectory: checkpoint summary + messages after it + state.
    self_compact: Optional[Dict[str, Any]] = None
    # Stored rows behind ``self_compact["messages"]`` (session compaction cuts on turns).
    trajectory_rows: List[Dict[str, Any]] = field(default_factory=list)
    # This turn's session checkpoint; stored with the turn (see ``turn_self_compact``).
    session_compaction: Optional[Dict[str, Any]] = None
    # Long-input digest meta when ``user_message`` was replaced by a digest
    # (the stored message stays the original request text).
    long_input: Optional[Dict[str, Any]] = None

    # ---- Artifacts the agent loop will actually consume ----
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    auto_loaded_artifact_count: int = 0
    auto_load_reason: Optional[str] = None
    wants_file: bool = False
    topic_changed_cleared: bool = False

    # ---- Telemetry ----
    preflight_ms: float = 0.0
    preflight_breakdown: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Kwarg adapters for the various agent entry points.
    # ------------------------------------------------------------------

    async def apply_long_input(
        self, orchestrator_config: Optional[Dict[str, Any]] = None, *, llm: Any = None
    ) -> Optional[Dict[str, Any]]:
        """Swap an oversized ``user_message`` for the parallel-reader digest."""
        if self.long_input is not None:
            return self.long_input
        from .context.long_input import LongInputReader

        t0 = time.monotonic()
        try:
            reader = LongInputReader.for_turn(orchestrator_config, llm=llm)
            if reader is None or not reader.needed(self.user_message):
                return None
            result = await reader.read(self.user_message)
        except Exception as exc:  # noqa: BLE001 — the engine still gets the original text
            log.warning("session_workspace: long input reading failed: %s", exc)
            return None
        finally:
            self.preflight_breakdown["long_input"] = (time.monotonic() - t0) * 1000.0
        if result is None:
            return None
        self.user_message = result.message
        self.long_input = result.meta()
        log.info(
            "session_workspace: long input %s chars → %s-char digest (%s parts, %s failed, %s tasks)",
            result.original_chars, len(result.message), result.parts, result.failed_parts, len(result.tasks),
        )
        return self.long_input

    async def apply_session_compaction(
        self, orchestrator_config: Optional[Dict[str, Any]] = None, *, llm: Any = None
    ) -> Optional[Dict[str, Any]]:
        """Size the conversation for this turn's model before any engine runs.

        Returns the ``context_compacted`` event when the history was compacted.
        """
        if self.self_compact is None:
            return None
        from .context.session_compactor import OWNER, SessionCompactSettings, compact_session, summarizer_path

        try:
            st = SessionCompactSettings.resolve(orchestrator_config)
        except Exception as exc:  # noqa: BLE001 — a bad override leaves the engines in charge
            log.warning("session_workspace: session compaction settings invalid: %s", exc)
            return None
        if not st.enabled:
            return None
        self.self_compact["owner"] = OWNER
        self.self_compact["owner_summarizer"] = summarizer_path(st)
        if self.session_compaction is not None:
            return None
        from .chat_history import get_chat_history_client

        client = get_chat_history_client()
        if llm is None:
            from .llm.gateway_llm_client import get_gateway_llm_client

            llm = get_gateway_llm_client()
        t0 = time.monotonic()
        try:
            result = await compact_session(
                self.self_compact,
                self.trajectory_rows,
                user_message=self.user_message,
                orchestrator_config=orchestrator_config,
                llm_client=llm,
                expand=client._recent_dicts_for_stream,  # noqa: SLF001
                rows_limit=settings.SELF_COMPACT_HISTORY_LIMIT,
            )
        except Exception as exc:  # noqa: BLE001 — the engines' window guards still apply
            log.warning("session_workspace: session compaction failed: %s", exc)
            return None
        finally:
            self.preflight_breakdown["session_compaction"] = (time.monotonic() - t0) * 1000.0
        if result is None:
            return None
        kept = client._recent_dicts_for_stream(result.kept_rows)  # noqa: SLF001
        self.self_compact.update(
            summary=result.summary,
            messages=kept,
            rounds=sum(1 for r in result.kept_rows if r.get("role") == "assistant"),
        )
        self.trajectory_rows = list(result.kept_rows)
        self.chat_history = list(kept)
        self.history_summary = result.summary
        self.session_compaction = result.payload()
        return result.event()

    def turn_self_compact(self, engine_payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """``metadata.self_compact`` for this turn: session checkpoint + the engine's own."""
        from .context.session_compactor import merge_turn_payload

        return merge_turn_payload(self.session_compaction, engine_payload)

    def to_engine_kwargs(self) -> Dict[str, Any]:
        """Kwargs for the ReAct/Cline engines (``stream_agent``,
        ``stream_agent_cline``, ``run_react_agent``,
        ``stream_agent_for_full_endpoint``)."""
        return {
            "user_message": self.user_message,
            "session_id": self.session_id,
            "artifacts": self.artifacts,
            "user_id": self.user_id,
            "chat_history": self.chat_history,
            "has_fresh_artifacts": self.has_fresh_artifacts,
            "conversation_state": self.conversation_state,
            "history_summary": self.history_summary,
            "file_registry": self.file_registry,
            "pinned_facts": self.pinned_facts,
            "workspace_runtime": self.workspace_runtime,
            "discourse_focus": (
                self.discourse_focus.to_dict()
                if hasattr(self.discourse_focus, "to_dict")
                else self.discourse_focus
            ),
            "task_checklist_state": self.task_checklist_state,
            "rule_policies": self.rule_policies,
            "active_plugin": self.active_plugin,
            "host_page_prompt": self.host_page_prompt or None,
            "rag_context": self.rag_context,
            "rag_prompt_block": self.rag_prompt_block,
            "memory_prompt_block": self.memory_prompt_block,
            "request_context": self.request_context,
            "case_inventory": self.case_inventory,
            **{f"{name}_prompt_block": text for name, text in self.guidance_blocks.items()},
        }

    def to_hub_engine_kwargs(self) -> Dict[str, Any]:
        """Kwargs for registry chat engines (``stream_entry`` functions take ``**kwargs``)."""
        kw = self.to_engine_kwargs()
        kw["self_compact"] = self.self_compact
        return kw

    def to_planning_kwargs(self) -> Dict[str, Any]:
        """Kwargs for the Plan-Execute-Critic agent (run_planning_agent /
        run_planning_agent_stream)."""
        kw = self.to_engine_kwargs()
        kw["memory_context"] = self.memory_context
        # Planning entrypoint does not accept native RAG / host prompt kwargs yet.
        kw.pop("rag_context", None)
        kw.pop("rag_prompt_block", None)
        kw.pop("memory_prompt_block", None)
        for name in self.guidance_blocks:
            kw.pop(f"{name}_prompt_block", None)
        kw.pop("host_page_prompt", None)
        kw.pop("request_context", None)
        kw.pop("case_inventory", None)
        return kw

    def to_legacy_graph_kwargs(self) -> Dict[str, Any]:
        """Kwargs for the legacy ``run_agent_graph`` / ``run_agent_graph_stream``
        path that takes a slightly thinner signature (no pinned_facts/policies)."""
        return {
            "user_message": self.user_message,
            "session_id": self.session_id,
            "artifacts": self.artifacts,
            "memory_context": self.memory_context,
            "user_id": self.user_id,
            "chat_history": self.chat_history,
        }

    def telemetry(self) -> Dict[str, Any]:
        return {
            "preflight_ms": round(self.preflight_ms, 1),
            "preflight_breakdown_ms": {
                k: round(v, 1) for k, v in self.preflight_breakdown.items()
            },
            "history_messages": len(self.chat_history),
            "history_summary_chars": len(self.history_summary or ""),
            "file_registry_count": len(self.file_registry),
            "previous_attachments_count": len(self.previous_attachments),
            "memory_context_count": len(self.memory_context),
            "rag_context_count": len(self.rag_context),
            "rag_meta": self.rag_meta,
            "artifacts_count": len(self.artifacts),
            "auto_loaded_artifact_count": self.auto_loaded_artifact_count,
            "auto_load_reason": self.auto_load_reason,
            "wants_file": self.wants_file,
            "topic_changed_cleared": self.topic_changed_cleared,
            "has_fresh_artifacts": self.has_fresh_artifacts,
            "workspace_runtime_selected_files": len(self.workspace_runtime.get("selected_files") or []),
            "workspace_runtime_snippets": len(self.workspace_runtime.get("text_snippets") or []),
            "workspace_runtime_runtime_first": bool(self.workspace_runtime.get("prefer_runtime_first")),
            "long_input": self.long_input,
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


async def _safe_call(coro, label: str, breakdown: Dict[str, float]):
    """Wrap an awaitable so a single failing fetch never poisons the gather
    result; record its latency in ``breakdown`` for telemetry."""
    t0 = time.monotonic()
    try:
        result = await coro
        return result, None
    except Exception as exc:  # noqa: BLE001 — defensive fan-out
        log.warning("session_workspace: %s failed: %s", label, exc)
        return None, str(exc)
    finally:
        breakdown[label] = (time.monotonic() - t0) * 1000.0


async def build_session_workspace(
    request: ChatRequest,
    *,
    llm_client: Any,
    enable_memory: bool = True,
) -> SessionWorkspaceState:
    """Assemble a :class:`SessionWorkspaceState` for ``request``.

    All I/O-bound fetches (chat history, file registry, previous attachments,
    long-term memory) are issued concurrently via ``asyncio.gather``. After
    the gather settles we run the deterministic, in-memory decisions
    (auto-attach merge, topic-change reset) and finally extract pinned facts
    (which depends on the gathered history + registry, so it runs in a
    second gather alongside any remaining work).
    """

    t_wall = time.monotonic()
    breakdown: Dict[str, float] = {}

    fresh_artifacts = [a.model_dump() for a in (request.artifacts or [])]
    has_fresh_artifacts = bool(fresh_artifacts)
    storage_auth_context = _request_storage_auth_context(request)
    rule_policies = _context_rule_policies(request.context)
    ctx = request.context if isinstance(request.context, dict) else {}
    from .host_context import resolve_host_context

    active_plugin, host_page_prompt = resolve_host_context(ctx)
    if active_plugin:
        from .plugin_host_prompt import load_llm_host_prompt

        try:
            await load_llm_host_prompt(active_plugin)
        except Exception as exc:  # noqa: BLE001 — the page prompt is optional context
            log.warning("session_workspace: plugin host prompt for %s unavailable: %s", active_plugin, exc)

    state = SessionWorkspaceState(
        session_id=request.session_id,
        user_id=request.user_id,
        user_message=request.message,
        rule_policies=rule_policies,
        storage_auth_context=storage_auth_context,
        fresh_artifacts=fresh_artifacts,
        has_fresh_artifacts=has_fresh_artifacts,
        artifacts=list(fresh_artifacts),
        active_plugin=active_plugin,
        host_page_prompt=host_page_prompt,
        request_context=dict(ctx) if ctx else None,
    )

    # ------------------------------------------------------------------
    # Stage A — share a single ``/messages`` + ``/artifacts`` round-trip
    # across all three derivation paths (history, attachments, registry).
    # ------------------------------------------------------------------
    raw_messages: List[Dict[str, Any]] = []
    raw_storage_artifacts: List[Dict[str, Any]] = []

    if settings.CHAT_HISTORY_ENABLED:
        from .chat_history import get_chat_history_client

        client = get_chat_history_client()

        fetch_limit = settings.CHAT_HISTORY_LIMIT
        keep_trajectory = settings.SELF_COMPACT_ENABLED or settings.SESSION_COMPACT_ENABLED
        if keep_trajectory:
            fetch_limit = max(fetch_limit, settings.SELF_COMPACT_HISTORY_LIMIT)

        async def _fetch_messages_once() -> List[Dict[str, Any]]:
            return await client.get_messages(request.session_id, fetch_limit)

        async def _fetch_storage_artifacts_once() -> List[Dict[str, Any]]:
            arts = await client._request(
                "GET",
                f"/chat/sessions/{request.session_id}/artifacts",
                params={"limit": 500, "offset": 0},
                auth_context=storage_auth_context,
            )
            return arts if isinstance(arts, list) else []

        raw_results = await asyncio.gather(
            _safe_call(_fetch_messages_once(), "messages", breakdown),
            _safe_call(_fetch_storage_artifacts_once(), "storage_artifacts", breakdown),
            return_exceptions=False,
        )
        (msgs_val, msgs_err), (arts_val, arts_err) = raw_results
        if msgs_err:
            state.warnings.append(f"messages_fetch_failed: {msgs_err}")
        else:
            raw_messages = list(msgs_val or [])
            if keep_trajectory:
                try:
                    from .context.self_compact import split_trajectory

                    traj = split_trajectory(raw_messages)
                    state.trajectory_rows = list(traj["messages"])
                    traj["messages"] = client._recent_dicts_for_stream(traj["messages"])  # noqa: SLF001
                    state.self_compact = traj
                except Exception as exc:  # noqa: BLE001
                    log.warning("session_workspace: self_compact trajectory skipped: %s", exc)
            raw_messages = raw_messages[-settings.CHAT_HISTORY_LIMIT:]
        if arts_err:
            state.warnings.append(f"storage_artifacts_fetch_failed: {arts_err}")
        else:
            raw_storage_artifacts = list(arts_val or [])

        try:
            from .chat_history import extract_agent_state_from_messages

            agent_state = extract_agent_state_from_messages(raw_messages)
            if isinstance(agent_state.get("task_checklist"), dict):
                state.task_checklist_state = dict(agent_state["task_checklist"])
            checkpoint = agent_state.get("conversation_summary")
            if isinstance(checkpoint, str) and checkpoint.strip():
                state.history_summary = checkpoint.strip()
        except Exception as exc:  # noqa: BLE001
            log.debug("session_workspace: agent state from messages skipped: %s", exc)

    # Now derive everything from the cached raw lists — these calls are
    # CPU-only or run the lazy-summary background task.
    fetch_specs: List[Tuple[Any, str, bool]] = []

    if settings.CHAT_HISTORY_ENABLED:
        from .chat_history import (
            get_chat_history_for_stream,
            get_session_attachments,
            get_session_file_registry,
        )

        fetch_specs.append((
            get_chat_history_for_stream(
                session_id=request.session_id,
                llm_client=llm_client,
                limit=settings.CHAT_HISTORY_LIMIT,
                prefetched_messages=raw_messages,
                lazy_summary=True,
            ),
            "history",
            True,
        ))
        fetch_specs.append((
            get_session_file_registry(
                session_id=request.session_id,
                limit=settings.CHAT_HISTORY_LIMIT,
                auth_context=storage_auth_context,
                prefetched_messages=raw_messages,
                prefetched_storage_artifacts=raw_storage_artifacts,
            ),
            "file_registry",
            True,
        ))
        async def _fetch_session_record() -> Dict[str, Any]:
            row = await client._request(  # noqa: SLF001
                "GET", f"/chat/sessions/{request.session_id}", auth_context=storage_auth_context,
            )
            return row if isinstance(row, dict) else {}

        fetch_specs.append((_fetch_session_record(), "session_record", True))
        if not has_fresh_artifacts:
            fetch_specs.append((
                get_session_attachments(
                    session_id=request.session_id,
                    limit=settings.CHAT_HISTORY_LIMIT,
                    auth_context=storage_auth_context,
                    prefetched_messages=raw_messages,
                ),
                "previous_attachments",
                True,
            ))

    if enable_memory and getattr(request, "use_memory", False) and settings.MEMORY_ENABLED:
        try:
            from .memory import get_advanced_memory_client, MemoryScope

            memory_client = get_advanced_memory_client()
            fetch_specs.append((
                memory_client.get_relevant_memories(
                    query=request.message,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    scope=MemoryScope.COMBINED,
                    limit=settings.MEMORY_SESSION_LIMIT,
                ),
                "memory",
                True,
            ))
        except Exception as exc:  # noqa: BLE001
            log.warning("session_workspace: memory client init failed: %s", exc)

    if settings.NATIVE_RAG_ENABLED and getattr(request, "use_rag", True):
        try:
            from .rag import retrieve_for_prompt

            fetch_specs.append((
                retrieve_for_prompt(
                    request.message,
                    user_id=request.user_id,
                    auth_context=storage_auth_context,
                ),
                "rag",
                True,
            ))
        except Exception as exc:  # noqa: BLE001
            log.warning("session_workspace: rag client init failed: %s", exc)

    if fetch_specs:
        results = await asyncio.gather(
            *(_safe_call(coro, label, breakdown) for coro, label, _enabled in fetch_specs),
            return_exceptions=False,
        )
        for (_, label, _enabled), (value, error) in zip(fetch_specs, results):
            if error:
                state.warnings.append(f"{label}_fetch_failed: {error}")
                if label == "memory":
                    state.memory_available = False
                continue
            if label == "history":
                hist, summary = value if isinstance(value, tuple) else ([], "")
                state.chat_history = hist or []
                state.history_summary = summary or ""
            elif label == "file_registry":
                state.file_registry = finalize_file_registry_for_llm(
                    value or [],
                    fresh_artifacts=state.fresh_artifacts,
                    storage_artifacts=raw_storage_artifacts,
                )
                n_img = _count_registry_images(state.file_registry)
                log.info(
                    "session_workspace file_registry: total=%d images=%d "
                    "(fresh=%d storage_prefetch=%d)",
                    len(state.file_registry),
                    n_img,
                    len(state.fresh_artifacts),
                    len(raw_storage_artifacts),
                )
                if n_img == 0 and (
                    state.fresh_artifacts
                    or raw_storage_artifacts
                ):
                    state.warnings.append(
                        "file_registry_no_images_detected: uploads or storage "
                        "artifacts present but no image/* type in registry — "
                        "image_meta may be omitted from LLM tools"
                    )
            elif label == "previous_attachments":
                state.previous_attachments = value or []
            elif label == "session_record":
                state.session_record = value or {}
            elif label == "memory":
                from .memory import format_memory_prompt_block

                mem = value if isinstance(value, dict) else {}
                state.memory_context = mem.get("combined", [])
                state.memory_prompt_block = format_memory_prompt_block(mem)
                err = mem.get("error")
                state.memory_available = not err
                if err:
                    state.warnings.append(f"memory_unavailable: {err}")
            elif label == "rag":
                from .rag import RagResult

                if isinstance(value, RagResult):
                    state.rag_context = list(value.hits or [])
                    state.rag_prompt_block = value.prompt_block or ""
                    state.rag_meta = dict(value.meta or {})
                    reason = (value.meta or {}).get("reason")
                    if reason and reason not in {
                        "ok",
                        "ok_ann_only",
                        "no_results",
                        "no_results_after_trim",
                        "empty_query",
                        "disabled",
                    }:
                        state.warnings.append(f"rag_soft_fail: {reason}")
                elif isinstance(value, dict):
                    state.rag_context = list(value.get("hits") or [])
                    state.rag_prompt_block = str(value.get("prompt_block") or "")
                    state.rag_meta = dict(value.get("meta") or {})

    try:
        from .context.turn_guidance import build_guidance_blocks

        state.guidance_blocks = build_guidance_blocks(
            user_message=request.message,
            chat_history=state.chat_history,
            session_record=state.session_record,
            request_context=ctx,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("session_workspace: guidance blocks skipped: %s", exc)

    # ------------------------------------------------------------------
    # Stage B — derived decisions that depend on the fetched context.
    # ------------------------------------------------------------------
    t_decisions = time.monotonic()

    # Decide whether the user is talking about a file at all.
    try:
        from .discourse_focus import (
            DiscourseFocus,
            artifacts_for_focus,
            enrich_focus_from_registry,
            extract_focus_from_history,
            focus_is_fresh,
            has_deictic_reference,
            message_implies_file_context,
            resolve_focus_for_message,
        )
        from .conversation_state import smart_resolve_file_reference

        last_user_text = _last_user_message_text(state.chat_history)
        state.wants_file = bool(
            message_implies_file_context(request.message)
            or message_implies_file_context(last_user_text)
        )

        ctx_discourse = None
        if isinstance(getattr(request, "context", None), dict):
            ctx_discourse = (request.context or {}).get("discourse_focus")
        stored_focus = DiscourseFocus.from_dict(ctx_discourse)
        if stored_focus.is_empty() and raw_messages:
            try:
                from .chat_history import extract_agent_state_from_messages

                persisted = extract_agent_state_from_messages(raw_messages).get("discourse_focus")
                if persisted:
                    stored_focus = DiscourseFocus.from_dict(persisted)
            except Exception:
                pass
        if stored_focus.is_empty():
            stored_focus = extract_focus_from_history(state.chat_history)

        llm_intent = None
        if state.wants_file and state.file_registry and (
            has_deictic_reference(request.message)
            or not any(
                str(c).lower() in (request.message or "").lower()
                for entry in state.file_registry
                for c in _runtime_candidate_strings(entry)
                if c
            )
        ):
            llm_intent = await smart_resolve_file_reference(
                request.message,
                state.file_registry,
                llm_client=llm_client,
            )

        state.discourse_focus = resolve_focus_for_message(
            user_message=request.message,
            file_registry=state.file_registry,
            chat_history=state.chat_history,
            stored_focus=stored_focus,
            llm_intent=llm_intent,
        )
        state.discourse_focus = enrich_focus_from_registry(
            state.discourse_focus, state.file_registry
        ) if state.discourse_focus else stored_focus
    except Exception as exc:  # noqa: BLE001
        log.warning("session_workspace: discourse focus failed: %s", exc)
        state.wants_file = False
        state.discourse_focus = None

    # Auto-attach visual artifacts when the user mentions a file but did not
    # upload a fresh one this turn.
    if not has_fresh_artifacts and state.file_registry and state.wants_file:
        focus_artifacts: List[Dict[str, Any]] = []
        if state.discourse_focus is not None:
            try:
                focus_artifacts = artifacts_for_focus(
                    state.discourse_focus, state.file_registry
                )
            except Exception:
                focus_artifacts = []

        if focus_artifacts:
            state.artifacts = focus_artifacts[:2]
            state.auto_load_reason = "discourse_focus visual attach"
        else:
            previous_visual = _visual_artifacts_from_attachments(state.previous_attachments)
            registry_visual = _visual_artifacts_from_file_registry(state.file_registry)
            merged_visual = _merge_visual_artifacts(previous_visual, registry_visual)

            if merged_visual and len(merged_visual) <= 2:
                state.artifacts = merged_visual
                state.auto_load_reason = "attachments+registry merged"
            elif previous_visual:
                state.artifacts = list(reversed(previous_visual))
                state.auto_load_reason = "previous attachments (carry forward)"
            elif len(registry_visual) == 1:
                state.artifacts = registry_visual
                state.auto_load_reason = "single registry visual"
            elif registry_visual and state.discourse_focus is not None:
                # Use discourse focus instead of skipping when multiple visuals
                fa = artifacts_for_focus(state.discourse_focus, state.file_registry)
                if fa:
                    state.artifacts = fa[:2]
                    state.auto_load_reason = "discourse_focus resolved among multiple visuals"
                else:
                    state.auto_load_reason = "skipped: multiple registry visuals require explicit selection"
            elif registry_visual:
                state.auto_load_reason = "skipped: multiple registry visuals require explicit selection"

        state.auto_loaded_artifact_count = len(state.artifacts)
        if state.auto_loaded_artifact_count and state.auto_load_reason:
            log.info(
                "session_workspace: auto-loaded %d visual artifact(s) — %s",
                state.auto_loaded_artifact_count,
                state.auto_load_reason,
            )

    # Conversation state analysis (deterministic, cheap, sync).
    try:
        from .conversation_state import analyze_conversation

        state.conversation_state = analyze_conversation(
            chat_history=state.chat_history,
            current_message=request.message,
            has_file_context=has_fresh_artifacts,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("session_workspace: analyze_conversation failed: %s", exc)
        state.conversation_state = None

    # If topic changed and the only attachments we carry are auto-loaded
    # carry-overs, drop them so unrelated topics don't drag the previous
    # image back in — unless discourse focus is still fresh.
    _skip_clear = False
    if state.discourse_focus is not None:
        try:
            from .discourse_focus import focus_is_fresh
            turn = getattr(state.conversation_state, "turn_number", 0) or len(
                [m for m in state.chat_history if isinstance(m, dict) and m.get("role") == "user"]
            )
            _skip_clear = focus_is_fresh(state.discourse_focus, turn)
        except Exception:
            _skip_clear = False

    if (
        state.conversation_state is not None
        and getattr(state.conversation_state, "topic_changed", False)
        and not has_fresh_artifacts
        and state.artifacts
        and not state.wants_file
        and not _skip_clear
    ):
        log.info("session_workspace: topic changed — clearing auto-loaded artifacts")
        state.artifacts = []
        state.auto_loaded_artifact_count = 0
        state.topic_changed_cleared = True

    state.workspace_runtime = _build_workspace_runtime(
        user_message=request.message,
        file_registry=state.file_registry,
        storage_artifacts=raw_storage_artifacts,
        active_artifacts=state.artifacts,
        wants_file=state.wants_file,
        discourse_focus=state.discourse_focus,
    )

    # If the runtime resolved explicit visual targets more narrowly than the
    # broad carry-forward merge, prefer the exact selection so the model sees
    # the intended image/video instead of a random recent artifact.
    if (
        not has_fresh_artifacts
        and state.wants_file
        and state.workspace_runtime.get("visual_targets")
    ):
        visual_targets = state.workspace_runtime.get("visual_targets") or []
        if 0 < len(visual_targets) <= 2 and (
            not state.artifacts or len(visual_targets) < len(state.artifacts)
        ):
            state.artifacts = list(visual_targets)
            state.auto_loaded_artifact_count = len(state.artifacts)
            state.auto_load_reason = "workspace runtime exact visual target"

    breakdown["decisions"] = (time.monotonic() - t_decisions) * 1000.0

    # ------------------------------------------------------------------
    # Stage C — pinned facts (depends on history+registry, may call LLM).
    # ------------------------------------------------------------------
    if settings.CHAT_HISTORY_ENABLED:
        try:
            from .pinned_facts import extract_pinned_facts

            t_pf = time.monotonic()
            facts_obj = await extract_pinned_facts(
                chat_history=state.chat_history,
                current_message=request.message,
                file_registry=state.file_registry,
                llm_client=llm_client,
                conversation_state=state.conversation_state,
            )
            breakdown["pinned_facts"] = (time.monotonic() - t_pf) * 1000.0
            if facts_obj and not facts_obj.is_empty():
                state.pinned_facts = facts_obj.to_dict()
        except Exception as exc:  # noqa: BLE001
            log.warning("session_workspace: pinned_facts extraction failed: %s", exc)

    state.preflight_breakdown = breakdown
    state.preflight_ms = (time.monotonic() - t_wall) * 1000.0

    # ------------------------------------------------------------------
    # Stage D — @case mention outputs → durable bindings + RLM case pool.
    # Mentions only arrive on the pick turn; later turns restore from disk.
    # Case files are NOT merged into Session Files / file_registry by default
    # (AIV2_CHAT_CASE_SESSION_ATTACH_CAP defaults to 0).
    # ------------------------------------------------------------------
    try:
        from .orchestrators.host_adapters.case_mentions import (
            case_files_for_session_registry,
            case_mentions_prompt_addon,
            inventory_has_cases,
            restore_or_build_case_inventory,
        )

        t_case = time.monotonic()
        case_inv = await restore_or_build_case_inventory(
            context=ctx,
            user_id=str(request.user_id or ""),
            session_id=str(request.session_id or ""),
        )
        breakdown["case_mentions"] = (time.monotonic() - t_case) * 1000.0
        if inventory_has_cases(case_inv):
            state.case_inventory = case_inv
            addon = case_mentions_prompt_addon(case_inv)
            if addon:
                state.host_page_prompt = (
                    f"{state.host_page_prompt}\n\n{addon}".strip()
                    if state.host_page_prompt
                    else addon
                )
            # Optional legacy attach path — off by default (cap=0).
            reg_rows = case_files_for_session_registry(case_inv)
            if reg_rows:
                seen = {
                    str(r.get("artifact_id") or r.get("id") or "").strip()
                    for r in (state.file_registry or [])
                    if isinstance(r, dict)
                }
                for row in reg_rows:
                    aid = str(row.get("artifact_id") or "").strip()
                    if not aid or aid in seen:
                        continue
                    seen.add(aid)
                    state.file_registry.append(row)
                if not state.artifacts and reg_rows:
                    state.artifacts = list(reg_rows[:8])
                    state.auto_load_reason = "case_mention outputs"
                    state.auto_loaded_artifact_count = len(state.artifacts)
    except Exception as exc:  # noqa: BLE001
        log.warning("session_workspace: case mention materialize failed: %s", exc)

    log.info(
        "session_workspace ready: hist=%d files=%d artifacts=%d auto=%d wants_file=%s preflight=%.1fms",
        len(state.chat_history),
        len(state.file_registry),
        len(state.artifacts),
        state.auto_loaded_artifact_count,
        state.wants_file,
        state.preflight_ms,
    )

    return state


async def _noop_list() -> List[Dict[str, Any]]:
    """Awaitable that returns an empty list — used so the parallel gather
    can keep a uniform shape when previous-attachment fetch is skipped."""
    return []


__all__ = [
    "SessionWorkspaceState",
    "build_session_workspace",
]
