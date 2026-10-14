"""
Cline-style streaming agent for AIv2.

This module is an end-to-end replacement for the LangGraph-based ``stream_agent``
in ``streaming.py``. The point is to switch the wire format between Qwen and
the orchestrator from "OpenAI native function-calling + recovery hacks" to
"XML inside content + simple stream parser", the way Cline does it.

What this engine does, in plain English:

1. Loads the tool manifest once and builds the Cline-style system prompt
   (every tool documented as `## tool_name` / Parameters / Usage XML).
   **Note:** This path duplicates tool docs in prompt XML — unlike
   ``native_tools_engine``, which uses only the API ``tools`` parameter.
   Out of scope for the tools-SSOT refactor unless ``CLINE_XML_ENGINE_ENABLED``.
2. Reconstructs the chat history as plain alternating user/assistant
   messages — no ``tool_calls`` / ``ToolMessage`` round-trips. Tool calls
   from earlier turns are preserved verbatim in the assistant text (as the
   raw ``<tool_name>...</tool_name>``) and tool results were stored as the
   following synthetic ``user`` message ``[Tool Result — name] ...``.
3. Streams the LLM response (``chat_stream``, no ``tools=`` argument), feeds
   chunks into ``ClineXmlStreamParser``, and forwards parser events to the
   SSE stream as the existing UI expects (``answer``, ``thinking``,
   ``tool_call_start``, …).
4. When the parser emits ``tool_call``, the engine stops the LLM stream,
   executes the tool via ``ToolExecutor``, then appends a new user message
   ``[Tool Result — name]\\n<channel.llm text>`` and loops back to the LLM.
5. When the parser emits ``attempt_completion``, the engine surfaces the
   ``<result>`` body as the final answer and exits the loop.

Why this is robust where the previous engine was not:

* The wire is plain text plus one XML envelope. There is no
  ``delta.tool_calls`` / ``arguments`` / ``args`` ambiguity to recover from.
* Qwen3.x never reaches the "assistant with `content=\"\\n\\n\"` and
  `tool_calls`" template path that the gateway closes prematurely — every
  assistant message has the XML inline as visible content.
* One tool per turn is enforced by the parser, so we never have to drop
  orphan tool messages.
* The system block is a single string, so the chat template never sees
  multiple ``system`` messages.

The engine is intentionally narrow: it does session-files lookup, file_context
detection, multimodal embedding (image/video pixels), and tool execution
through the same adapters as ``streaming.py``. It deliberately skips the
rule engine, conversation-state classifier, pinned facts, and history
summary — those can be re-added later as data injected into
``environment_details`` rather than as separate system blocks.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import time as _time
import uuid
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

from ...config import settings
from ...auth import resolve_allowed_tools
from ...llm.gateway_llm_client import get_gateway_llm_client
from ...tools.executor import (
    SESSION_ATTACH_HISTORY_GUIDANCE,
    get_tool_executor,
    session_attach_stale_error_text,
)
from ...tools.llm_tool_selection import (
    select_llm_tools,
    selection_debug_summary,
)
from ...utils.file_context import detect_file_context
from ...utils.multimodal import build_multimodal_content
from ...prompts.cline_xml import (
    build_cline_system_prompt,
    known_tool_names,
)
from ..cline_parser import ClineXmlStreamParser, build_param_types_index

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_llm_summary(tool_result: Dict[str, Any]) -> str:
    """Pull the human-readable summary out of ToolExecutor's nested envelope.

    The shape returned by ``ToolExecutor.execute`` is::

        {"ok": bool, "data": {...}, "channels": {...}, "tool_id": str, "error"?: str}

    Prefer top-level ``channels.llm`` (flattened tool-runner shape), then nested
    ``data`` / ``data.result`` envelopes. If nothing readable is found we serialise
    the payload as JSON so the model can still reason about what happened.
    """
    if not isinstance(tool_result, dict):
        return str(tool_result)

    def _from_channels(ch: Any) -> Optional[str]:
        if not isinstance(ch, dict):
            return None
        for key in ("llm", "text", "summary"):
            v = ch.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, dict):
                for sub in ("text", "summary", "value"):
                    sv = v.get(sub)
                    if isinstance(sv, str) and sv.strip():
                        return sv.strip()
        llm = ch.get("llm")
        if isinstance(llm, dict) and llm:
            # Structured model-facing payload: prefer it over the full data
            # envelope, which may carry UI-only (localized) fields.
            try:
                return json.dumps(llm, ensure_ascii=False, default=str)
            except Exception:
                return None
        return None

    if not tool_result.get("ok", True):
        err = tool_result.get("error") or "tool_failed"
        top = _from_channels(tool_result.get("channels"))
        if top:
            return top
        data = tool_result.get("data")
        if isinstance(data, dict):
            nested = _from_channels(data.get("channels"))
            if nested:
                return nested
            inner = data.get("result")
            if isinstance(inner, dict):
                nested = _from_channels(inner.get("channels"))
                if nested:
                    return nested
        return (
            f"[ERROR] {err}. The tool did not return usable data for this exact request. "
            "Do not repeat the same tool call with the same arguments; either try a different "
            "file identifier/name or finish with the best explanation from the available evidence."
        )

    top = _from_channels(tool_result.get("channels"))
    if top:
        return top

    data = tool_result.get("data")
    if isinstance(data, dict):
        for layer in (data, data.get("result") if isinstance(data.get("result"), dict) else {}):
            if not isinstance(layer, dict):
                continue
            nested = _from_channels(layer.get("channels"))
            if nested:
                return nested
            for key in ("summary", "message", "text", "content"):
                v = layer.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()

    try:
        return json.dumps(tool_result.get("data") or tool_result, ensure_ascii=False, default=str)[:4000]
    except Exception:
        return str(tool_result)[:4000]


def _format_tool_result_message(tool_name: str, ok: bool, body: str) -> str:
    """Render the synthetic user message that follows a tool execution."""
    status = "OK" if ok else "ERROR"
    return f"[Tool Result — {tool_name} | {status}]\n{body}"


def _cap_llm_text(text: str, limit: Optional[int] = None) -> str:
    """Cap tool/LLM text fed back into the model; default = STREAM_TOOL_RESULT_MAX_CHARS."""
    if limit is None:
        limit = max(4_000, int(getattr(settings, "STREAM_TOOL_RESULT_MAX_CHARS", 24_000) or 24_000))
    t = (text or "").strip()
    return t[:limit] if len(t) > limit else t


def _normalized_tool_summary(
    tool_name: str,
    ok: bool,
    summary_text: str,
) -> str:
    t = (summary_text or "").strip()
    if t and t not in {"{}", "[]", "null", "None"}:
        return t
    if ok:
        return (
            f"[NO_DIRECT_OUTPUT] Tool `{tool_name}` executed successfully but returned no "
            "user-visible text or concise summary. Treat this as no direct evidence from "
            "that tool for the current request."
        )
    return (
        f"[ERROR] Tool `{tool_name}` failed and did not return a useful error summary. "
        "Treat this as no usable result from that tool call."
    )


def _tool_runs_digest_for_llm(tool_runs: List[Dict[str, Any]], limit: int = 6) -> str:
    lines: List[str] = []
    for i, tr in enumerate(tool_runs[-limit:], 1):
        tool_name = str(tr.get("tool_id") or tr.get("name") or "unknown_tool")
        ok = bool(tr.get("ok", (tr.get("result") or {}).get("ok")))
        summary = _normalized_tool_summary(
            tool_name,
            ok,
            str(tr.get("text_for_llm") or tr.get("summary") or ""),
        )
        lines.append(f"{i}. {tool_name} | {'OK' if ok else 'ERROR'}\n{summary}")
    return "\n\n".join(lines).strip()


def _fallback_final_answer_from_tool_runs(
    tool_runs: List[Dict[str, Any]],
) -> str:
    if not tool_runs:
        return (
            "I could not complete a response for this request. "
            "Please try rephrasing or adding a bit more detail."
        )
    last = tool_runs[-1]
    last_text = str(last.get("text_for_llm") or last.get("summary") or "").strip()
    if last_text and last_text not in {"{}", "[]", "null", "None"}:
        return last_text
    digest = _tool_runs_digest_for_llm(tool_runs, limit=4)
    if digest:
        return digest
    return (
        "I could not complete a response for this request. "
        "Please try rephrasing or adding a bit more detail."
    )


def _tool_result_sse_event(
    *,
    step: int,
    tool_id: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    ok: bool,
    summary_text: str,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Single canonical LLM text for SSE — avoids repeating the same body in many keys."""
    body = _cap_llm_text(summary_text)
    preview = body[:500] if len(body) > 500 else body
    ev: Dict[str, Any] = {
        "type": "tool_result",
        "step": step,
        "tool_id": tool_id,
        "tool_name": tool_name,
        "arguments": tool_args,
        "args": tool_args,
        "query": _display_query_from_args(tool_args),
        "ok": ok,
        "text_for_llm": body,
        "summary": preview,
    }
    if error:
        ev["error"] = error
    return ev


def _slim_resolution_for_export(res: Any) -> Any:
    if not isinstance(res, dict):
        return res
    picked = res.get("picked")
    slim_picked = None
    if isinstance(picked, dict):
        props = picked.get("properties") if isinstance(picked.get("properties"), dict) else {}
        slim_picked = {
            "element_id": picked.get("element_id"),
            "labels": picked.get("labels"),
            "name": props.get("name"),
            "elementKey": props.get("elementKey"),
        }
    return {
        "query": res.get("query"),
        "candidate_count": res.get("candidate_count"),
        "picked": slim_picked,
    }


def _canonical_tool_call_key(tool_name: str, tool_args: Dict[str, Any]) -> str:
    """Stable key for detecting repeated failed calls."""
    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): scrub(v) for k, v in sorted(value.items())}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        if isinstance(value, str):
            return re.sub(r"\s+", " ", value.strip())
        return value

    try:
        return f"{tool_name}:{json.dumps(scrub(tool_args or {}), ensure_ascii=False, sort_keys=True, default=str)}"
    except Exception:
        return f"{tool_name}:{str(tool_args)}"


def _display_query_from_args(tool_args: Dict[str, Any]) -> Optional[Any]:
    """Best-effort query/prompt value for live tool cards."""
    if not isinstance(tool_args, dict):
        return None
    for key in ("query", "queries", "text", "question", "prompt", "filename", "path", "file_id", "artifact_id"):
        value = tool_args.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _registry_entry_from_attach(file_info: Dict[str, Any]) -> Dict[str, Any]:
    """Build a file_registry row from session_attach tool output."""
    name = (
        file_info.get("name")
        or file_info.get("filename")
        or file_info.get("original_filename")
        or "unknown"
    )
    path = file_info.get("path") or file_info.get("storage_path") or ""
    mime = file_info.get("mime_type") or file_info.get("content_type") or ""
    ftype = file_info.get("type") or file_info.get("media_type") or "unknown"
    aid = file_info.get("artifact_id") or file_info.get("id")
    return {
        "ordinal": 0,
        "turn": 0,
        "name": name,
        "path": path,
        "mime_type": mime,
        "type": ftype,
        "artifact_id": aid,
        "knowledge": [],
        "source": "session_attach",
    }


def _extract_attach_file_from_result(tool_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull attached file metadata from ToolExecutor envelope."""
    if not isinstance(tool_result, dict) or not tool_result.get("ok"):
        return None
    data = tool_result.get("data")
    if not isinstance(data, dict):
        return None
    inner = data.get("result") if isinstance(data.get("result"), dict) else data
    payload = inner.get("data") if isinstance(inner.get("data"), dict) else inner
    file_info = payload.get("file") if isinstance(payload, dict) else None
    if isinstance(file_info, dict) and (file_info.get("path") or file_info.get("id")):
        return file_info
    return None


def _append_registry_entry(
    registry: List[Dict[str, Any]],
    entry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Append a file to registry if not already present (by artifact_id or path)."""
    aid = str(entry.get("artifact_id") or entry.get("id") or "")
    path = str(entry.get("path") or "")
    for existing in registry:
        if aid and str(existing.get("artifact_id") or existing.get("id") or "") == aid:
            return registry
        if path and str(existing.get("path") or "") == path:
            return registry
    out = list(registry)
    entry = dict(entry)
    entry["ordinal"] = len(out) + 1
    out.append(entry)
    return out


def _detect_file_context_from_registry(
    registry: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not registry:
        return None
    try:
        rows = [dict(item) for item in registry if isinstance(item, dict)]
        return detect_file_context(rows) if rows else None
    except Exception as exc:
        log.warning("detect_file_context on workspace registry failed: %s", exc)
        return None


def _registry_entry_lookup_keys(entry: Dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for raw in (
        entry.get("artifact_id"),
        entry.get("id"),
        entry.get("file_id"),
        entry.get("path"),
        entry.get("storage_path"),
        entry.get("name"),
        entry.get("filename"),
        entry.get("original_filename"),
    ):
        if not raw:
            continue
        text = str(raw).strip()
        if not text:
            continue
        keys.add(text)
        keys.add(text.lower())
        base = text.rsplit("/", 1)[-1]
        keys.add(base)
        keys.add(base.lower())
    return keys


def _visual_file_context_keys(file_context: Optional[Dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    if not isinstance(file_context, dict):
        return keys
    files = file_context.get("all_files") or [file_context]
    image_count = 0
    video_count = 0
    ordered = list(reversed(files))
    for item in ordered:
        if not isinstance(item, dict):
            continue
        file_type = str(item.get("type") or "").lower()
        mime_type = str(item.get("mime_type") or "").lower()
        is_image = file_type == "image" or mime_type.startswith("image/")
        is_video = file_type == "video" or mime_type.startswith("video/")
        if not (is_image or is_video):
            continue
        if is_image:
            if image_count >= 2:
                continue
            image_count += 1
        if is_video:
            if video_count >= 1:
                continue
            video_count += 1
        key = (
            item.get("artifact_id")
            or item.get("id")
            or item.get("file_id")
            or item.get("path")
            or item.get("name")
        )
        if key:
            keys.add(str(key))
    return keys


def _visual_file_labels(file_context: Optional[Dict[str, Any]], limit: int = 3) -> List[str]:
    labels: List[str] = []
    if not isinstance(file_context, dict):
        return labels
    files = file_context.get("all_files") or [file_context]
    for item in files:
        if not isinstance(item, dict):
            continue
        label = (
            item.get("name")
            or item.get("filename")
            or item.get("original_filename")
            or item.get("path")
            or item.get("artifact_id")
            or item.get("id")
            or item.get("file_id")
        )
        if not label:
            continue
        text = str(label).strip()
        if text and text not in labels:
            labels.append(text)
        if len(labels) >= limit:
            break
    return labels


def _visual_attachment_missing(
    file_context: Optional[Dict[str, Any]],
    injected_multimodal_keys: set[str],
) -> bool:
    expected_keys = _visual_file_context_keys(file_context)
    return bool(expected_keys) and not expected_keys.issubset(injected_multimodal_keys)


def _missing_visual_attachment_notice(file_context: Optional[Dict[str, Any]]) -> str:
    labels = _visual_file_labels(file_context)
    label_text = ", ".join(labels) if labels else "the selected visual file"
    return (
        "[Visual Attachment Missing]\n"
        "A visual file is selected in the current session context, but its actual image/video bytes "
        "were not attached to this model turn. "
        f"Selected file(s): {label_text}. "
        "Do not claim that you saw, inspected, or analyzed the visual contents. "
        "If the user asks what is visible in the media, say clearly that the media was not attached "
        "for this turn and ask for an explicit retry or exact file selection."
    )


def _final_answer_for_missing_visual_attachment(
    file_context: Optional[Dict[str, Any]],
    tool_runs: List[Dict[str, Any]],
) -> str:
    labels = _visual_file_labels(file_context)
    label_text = ", ".join(labels) if labels else "the selected visual file"
    answer = (
        "I do not have the actual image/video bytes for the selected visual file in this turn, "
        "so I cannot honestly claim that I saw its contents. "
        f"Selected file(s): {label_text}. "
        "Please retry after explicitly selecting or re-attaching the exact visual file."
    )
    if tool_runs:
        answer += "\n\nAvailable non-visual evidence:\n" + _tool_runs_digest_for_llm(tool_runs, limit=3)
    return answer


def _is_visual_registry_entry(entry: Dict[str, Any]) -> bool:
    file_type = str(entry.get("type") or "").lower()
    mime_type = str(entry.get("mime_type") or "").lower()
    return file_type in {"image", "video"} or mime_type.startswith(("image/", "video/"))


def _match_registry_entry_from_tool_args(
    tool_args: Dict[str, Any],
    registry: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(tool_args, dict) or not registry:
        return None
    wanted: set[str] = set()
    for key in ("file_id", "artifact_id", "path", "file_path", "filename", "name"):
        value = tool_args.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        wanted.add(text)
        wanted.add(text.lower())
        base = text.rsplit("/", 1)[-1]
        wanted.add(base)
        wanted.add(base.lower())
    if not wanted:
        return None
    for entry in registry:
        if isinstance(entry, dict) and wanted & _registry_entry_lookup_keys(entry):
            return dict(entry)
    return None


def _refresh_cline_workspace(
    messages: List[Dict[str, Any]],
    *,
    tools_manifest: List[Dict[str, Any]],
    file_registry: List[Dict[str, Any]],
    file_context: Optional[Dict[str, Any]],
    session_id: str,
    pinned_facts: Optional[Dict[str, Any]] = None,
    history_summary: Optional[str] = None,
    active_plugin: Optional[str] = None,
    rag_context: Optional[List[Dict[str, Any]]] = None,
    rag_prompt_block: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], set, Dict[str, Any], Dict[str, Any]]:
    """Rebuild env + system prompt after session workspace changes."""
    env_details = {
        "session_id": session_id,
        "current_time": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "session_files": [
            {
                "artifact_id": f.get("artifact_id") or f.get("id") or f.get("file_id"),
                "name": f.get("name") or f.get("filename") or f.get("original_filename"),
                "type": f.get("type") or f.get("mime_type"),
                "size_bytes": f.get("size_bytes") or f.get("size") or f.get("file_size_bytes"),
            }
            for f in file_registry
            if isinstance(f, dict)
        ],
    }
    extras: Dict[str, Any] = {}
    if pinned_facts:
        try:
            extras["pinned_facts"] = json.dumps(pinned_facts, ensure_ascii=False)[:1500]
        except Exception:
            pass
    if history_summary:
        extras["history_summary"] = str(history_summary)[:1500]
    if extras:
        env_details["extras"] = extras

    selected_tools = select_llm_tools(
        tools_manifest,
        current_file_context=file_context,
        session_file_registry=file_registry,
        active_plugin=active_plugin,
    )
    system_prompt = build_cline_system_prompt(
        selected_tools,
        env_details=env_details,
        multimodal_enabled=True,
        rag_context=rag_context,
        rag_prompt_block=rag_prompt_block,
    )
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = system_prompt
    return selected_tools, set(known_tool_names(selected_tools)), build_param_types_index(selected_tools), env_details


async def _build_environment_details(
    session_id: str,
    file_registry: Optional[List[Dict[str, Any]]] = None,
    pinned_facts: Optional[Dict[str, Any]] = None,
    history_summary: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the env block the system prompt embeds in every turn.

    We deliberately keep this lightweight (no extra HTTP calls): the
    file_registry list comes from the request handler which already
    fetched session files once.
    """
    env: Dict[str, Any] = {
        "session_id": session_id,
        "current_time": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "session_files": [
            {
                "artifact_id": f.get("artifact_id") or f.get("id") or f.get("file_id"),
                "name": f.get("name") or f.get("filename") or f.get("original_filename"),
                "type": f.get("type") or f.get("mime_type"),
                "size_bytes": f.get("size_bytes") or f.get("size") or f.get("file_size_bytes"),
            }
            for f in (file_registry or [])
            if isinstance(f, dict)
        ],
    }

    extras: Dict[str, Any] = {}
    if pinned_facts:
        try:
            extras["pinned_facts"] = json.dumps(pinned_facts, ensure_ascii=False)[:1500]
        except Exception:
            pass
    if history_summary:
        extras["history_summary"] = str(history_summary)[:1500]
    if extras:
        env["extras"] = extras

    return env


def _rebuild_message_stack(
    chat_history: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Translate AIv2 chat history rows into a flat alternating list.

    Old rows have legacy shapes (``tool_calls`` on assistant, ``role=tool``
    follow-ups). We collapse them so the new engine sees pure
    user/assistant turns:

    * Old assistant turn with ``tool_calls`` → keep the textual content as
      assistant message, then synthesise a follow-up user message
      ``[Tool Result — name] {content}`` from the matching ``role=tool`` row.
    * Old user turn → keep verbatim.

    This is one-way — once the new engine writes a turn back to history it
    will already be in the cline-XML shape.
    """
    out: List[Dict[str, Any]] = []
    pending_tool_summaries: Dict[str, str] = {}
    if not chat_history:
        return out

    # First pass: index tool messages by tool_call_id.
    for msg in chat_history:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            tid = str(msg.get("tool_call_id") or "")
            content = msg.get("content") or ""
            if tid and isinstance(content, str):
                pending_tool_summaries[tid] = content

    for msg in chat_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "user":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                out.append({"role": "user", "content": content})
            elif isinstance(content, list):
                out.append({"role": "user", "content": content})
            continue
        if role == "assistant":
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []
            if isinstance(content, str):
                out.append({"role": "assistant", "content": content})
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                name = fn.get("name") or tc.get("name") or "tool"
                tid = str(tc.get("id") or "")
                summary = pending_tool_summaries.get(tid, "")
                if summary:
                    out.append(
                        {
                            "role": "user",
                            "content": _format_tool_result_message(name, True, summary),
                        }
                    )
            continue
        # Skip role=tool here; already folded into the synthetic user turn.
    return out


# ---------------------------------------------------------------------------
# Evidence loop (staged, transparent reasoning trail)
# ---------------------------------------------------------------------------
#
# Inspired by Continue's "incremental ReAct" cycle: think → small action →
# observe → accumulate evidence → repeat. The Cline engine already runs this
# shape internally, but it does not surface the structured trail to the UI or
# downstream consumers. The evidence ledger fills that gap by recording one
# typed record per loop event (plan / act / observe / attach / conclude). The
# ledger is emitted both as live ``evidence_step`` SSE events and as a final
# ``evidence_ledger`` array on the ``done`` event, giving the UI a stable hook
# for transparency panels and enabling deterministic post-hoc inspection of
# why a particular answer was produced.

_EVIDENCE_PHASES = {"plan", "act", "observe", "attach", "conclude"}


def _evidence_record(
    *,
    step: int,
    phase: str,
    kind: str,
    summary: str,
    ok: Optional[bool] = None,
    tool_name: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
    files: Optional[List[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one typed entry for the evidence ledger.

    ``summary`` is capped so the ledger stays cheap to ship over SSE; the full
    tool result is still available inside ``accumulated_tool_runs``.
    """
    phase_safe = phase if phase in _EVIDENCE_PHASES else "observe"
    rec: Dict[str, Any] = {
        "step": int(step),
        "phase": phase_safe,
        "kind": kind,
        "summary": (summary or "").strip()[:600],
        "timestamp": _dt.datetime.utcnow().isoformat(),
    }
    if ok is not None:
        rec["ok"] = bool(ok)
    if tool_name:
        rec["tool_name"] = tool_name
    if args is not None:
        rec["args"] = args
    if files:
        rec["files"] = list(files)
    if meta:
        rec["meta"] = meta
    return rec


def _evidence_files_from_context(
    file_context: Optional[Dict[str, Any]],
) -> List[str]:
    """Best-effort extraction of file identifiers from an active context."""
    if not isinstance(file_context, dict):
        return []
    out: List[str] = []
    for key in ("name", "filename", "path", "storage_path", "artifact_id", "id"):
        v = file_context.get(key)
        if v and isinstance(v, str):
            out.append(v)
    files = file_context.get("files")
    if isinstance(files, list):
        for f in files:
            if isinstance(f, dict):
                for key in ("name", "filename", "path", "storage_path", "artifact_id", "id"):
                    v = f.get(key)
                    if v and isinstance(v, str):
                        out.append(v)
    # Dedupe preserving order.
    seen: set = set()
    deduped: List[str] = []
    for item in out:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _evidence_files_from_tool_args(args: Optional[Dict[str, Any]]) -> List[str]:
    if not isinstance(args, dict):
        return []
    out: List[str] = []
    for key in (
        "filename",
        "file",
        "file_path",
        "path",
        "image",
        "image_path",
        "artifact_id",
        "file_id",
        "name",
    ):
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    return out


def _summarize_evidence_for_done(
    ledger: List[Dict[str, Any]],
    *,
    tool_runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compact aggregate suitable for embedding into the ``done`` event."""
    by_phase: Dict[str, int] = {}
    distinct_tools: List[str] = []
    distinct_files: List[str] = []
    failed = 0
    seen_tools: set = set()
    seen_files: set = set()
    for rec in ledger:
        phase = rec.get("phase", "observe")
        by_phase[phase] = by_phase.get(phase, 0) + 1
        tn = rec.get("tool_name")
        if tn and tn not in seen_tools:
            seen_tools.add(tn)
            distinct_tools.append(tn)
        for fn in rec.get("files") or []:
            if fn not in seen_files:
                seen_files.add(fn)
                distinct_files.append(fn)
        if rec.get("ok") is False:
            failed += 1
    return {
        "steps_seen": max((r.get("step", 0) for r in ledger), default=0),
        "by_phase": by_phase,
        "tools_invoked": distinct_tools,
        "files_touched": distinct_files,
        "tool_failures": failed,
        "tool_runs_count": len(tool_runs),
    }


# ---------------------------------------------------------------------------
# Main streaming entrypoint
# ---------------------------------------------------------------------------


async def stream_agent_cline(
    user_message: str,
    session_id: str,
    artifacts: Optional[List[Dict[str, Any]]] = None,
    user_id: str = "default",
    max_steps: Optional[int] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    has_fresh_artifacts: bool = False,
    conversation_state: Any = None,
    history_summary: str = "",
    file_registry: Optional[List[Dict[str, Any]]] = None,
    pinned_facts: Optional[Dict[str, Any]] = None,
    rule_policies: Optional[Dict[str, Any]] = None,
    active_plugin: Optional[str] = None,
    rag_context: Optional[List[Dict[str, Any]]] = None,
    rag_prompt_block: str = "",
    **_kwargs: Any,
) -> AsyncGenerator[Dict[str, Any], None]:
    """SSE-shaped generator: ``start`` → many UI events → ``done`` (or ``error``).

    The yielded shapes intentionally match what ``stream_agent`` (LangGraph
    version) used to produce, so ``main.py:generate_stream`` does not need
    to change between the two engines.
    """
    trace_id = str(uuid.uuid4())
    effective_max_steps = int(max_steps if max_steps is not None else settings.MAX_STEPS)
    t_engine_start = _time.monotonic()
    step_timings: List[Dict[str, Any]] = []
    rag_context = list(rag_context or [])
    rag_prompt_block = (rag_prompt_block or "").strip()

    yield {
        "type": "start",
        "trace_id": trace_id,
        "session_id": session_id,
        "engine": "cline_xml",
        "timestamp": _dt.datetime.utcnow().isoformat(),
    }

    llm_client = get_gateway_llm_client()
    tool_executor = get_tool_executor()

    # ------------------------------------------------------------------
    # File context & tool catalog
    # ------------------------------------------------------------------
    # ``main.py`` passes two kinds of artifacts here:
    #
    # * fresh uploads from the current user turn (``has_fresh_artifacts=True``)
    # * auto-loaded attachments from earlier session turns when the user asks
    #   about "this image/video" again (``has_fresh_artifacts=False``)
    #
    # Both must be eligible for direct multimodal embedding. The previous
    # Cline engine only embedded fresh uploads, so follow-up prompts like
    # "What is the image metadata?" reached the LLM as plain text even though main.py
    # had correctly resolved the session artifact.
    file_context: Optional[Dict[str, Any]] = None
    if artifacts:
        try:
            file_context = detect_file_context(artifacts)
        except Exception as exc:
            log.warning("detect_file_context failed: %s", exc)
            file_context = None
        if file_context:
            yield {"type": "file_detected", "file_context": file_context}

    active_file_context = file_context

    try:
        tools_manifest = await tool_executor.get_tools_manifest()
    except Exception as exc:
        log.error("Failed to fetch tools manifest: %s", exc, exc_info=True)
        tools_manifest = []

    tools_manifest = await resolve_allowed_tools(tools_manifest, user_id)

    env_details = await _build_environment_details(
        session_id=session_id,
        file_registry=file_registry,
        pinned_facts=pinned_facts,
        history_summary=history_summary,
    )

    selected_tools = select_llm_tools(
        tools_manifest,
        current_file_context=active_file_context,
        session_file_registry=file_registry,
        active_plugin=active_plugin,
    )
    log.info(
        "📦 Cline XML tools selected for LLM: %s/%s %s",
        len(selected_tools),
        len(tools_manifest or []),
        [t.get("id") or t.get("name") for t in selected_tools],
    )
    log.debug("📦 Cline XML tool selection detail: %s", selection_debug_summary(selected_tools))

    tool_names_set = set(known_tool_names(selected_tools))
    param_index = build_param_types_index(selected_tools)

    system_prompt = build_cline_system_prompt(
        selected_tools,
        env_details=env_details,
        multimodal_enabled=True,
        rag_context=rag_context,
        rag_prompt_block=rag_prompt_block,
    )

    # Mutable copy — grows when session_attach links DB files mid-turn.
    workspace_registry: List[Dict[str, Any]] = list(file_registry or [])

    # ------------------------------------------------------------------
    # Build the message stack
    # ------------------------------------------------------------------
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(_rebuild_message_stack(chat_history))

    # Current user message — possibly multimodal.
    user_content: Any = user_message
    injected_multimodal_keys: set[str] = set()
    if active_file_context:
        try:
            user_content = await build_multimodal_content(
                user_message,
                active_file_context,
                auth_context={
                    "user_id": user_id,
                },
            )
            if isinstance(user_content, list):
                injected_multimodal_keys = _visual_file_context_keys(active_file_context)
                yield {"type": "vision_attached", "parts": len(user_content)}
        except Exception as exc:
            log.warning("build_multimodal_content failed: %s", exc)
            user_content = user_message
    if _visual_attachment_missing(active_file_context, injected_multimodal_keys):
        if isinstance(user_content, str):
            user_content = (
                f"{user_content}\n\n{_missing_visual_attachment_notice(active_file_context)}"
            )
        log.warning(
            "Visual file context present but no multimodal bytes were attached for the current user turn."
        )
        yield {
            "type": "warning",
            "step": 0,
            "message": "A visual file is selected, but its bytes were not attached to the model for this turn.",
        }
    messages.append({"role": "user", "content": user_content})

    # ------------------------------------------------------------------
    # ReAct-style loop with evidence ledger
    # ------------------------------------------------------------------
    accumulated_tool_runs: List[Dict[str, Any]] = []
    evidence_ledger: List[Dict[str, Any]] = []
    final_answer: Optional[str] = None
    final_assistant_text: str = ""
    failed_call_counts: Dict[str, int] = {}
    last_failure_summary: Optional[str] = None
    empty_answer_retry_count = 0

    # Seed the ledger with what the engine already knows before the LLM ever
    # speaks — this is the "ground evidence" the loop will build on.
    if active_file_context:
        seed_files = _evidence_files_from_context(active_file_context)
        seed_rec = _evidence_record(
            step=0,
            phase="attach",
            kind="initial_file_context",
            summary=(
                f"Detected initial file context with {len(seed_files)} reference(s)."
                if seed_files else "Detected initial file context."
            ),
            files=seed_files,
        )
        evidence_ledger.append(seed_rec)
        yield {"type": "evidence_step", **seed_rec}

    if injected_multimodal_keys:
        attach_rec = _evidence_record(
            step=0,
            phase="attach",
            kind="vision_attached",
            summary=f"Bytes for {len(injected_multimodal_keys)} visual file(s) embedded in the first user turn.",
            files=sorted(injected_multimodal_keys),
        )
        evidence_ledger.append(attach_rec)
        yield {"type": "evidence_step", **attach_rec}

    for step in range(1, effective_max_steps + 1):
        step_started_at = _time.monotonic()
        step_llm_ms = 0.0
        step_tool_ms = 0.0
        # ``plan`` phase: the model is about to think and decide its next action.
        plan_rec = _evidence_record(
            step=step,
            phase="plan",
            kind="llm_turn",
            summary=f"Starting LLM reasoning round {step}.",
            meta={
                "tools_so_far": len(accumulated_tool_runs),
                "max_steps": effective_max_steps,
            },
        )
        evidence_ledger.append(plan_rec)
        yield {"type": "evidence_step", **plan_rec}

        yield {"type": "llm_start", "step": step}
        t_llm_start = _time.monotonic()

        parser = ClineXmlStreamParser(
            tool_names=tool_names_set,
            param_types_by_tool=param_index,
        )
        collected = ""
        thinking_emitted = False
        answer_emitted = False
        pending_tool: Optional[Dict[str, Any]] = None

        def _drain_events(events_in: List[Any]) -> Optional[Dict[str, Any]]:
            """Forward parser events to the SSE stream. Returns the
            tool/completion event if one was emitted, else None.

            Side effects: appends to the outer ``_outgoing`` list and
            mutates the ``thinking_emitted`` / ``answer_emitted`` flags.
            """
            nonlocal thinking_emitted, answer_emitted
            terminal: Optional[Dict[str, Any]] = None
            for pe in events_in:
                if pe.kind == "text_delta":
                    if not pe.text:
                        continue
                    if not answer_emitted:
                        _outgoing.append({"type": "answer_start", "step": step})
                        answer_emitted = True
                    _outgoing.append({"type": "answer", "content": pe.text})
                elif pe.kind == "thinking_delta":
                    # Cline-style <thinking> is private scratchpad. Do not
                    # stream it to the UI; persisted turns already omit it,
                    # so this keeps live and refreshed views consistent.
                    continue
                elif pe.kind == "tool_call":
                    _outgoing.append(
                        {
                            "type": "tool_call_start",
                            "step": step,
                            "index": 0,
                            "tool_id": f"call_{step}",
                            "tool_name": pe.tool_name,
                            "arguments": pe.arguments,
                            "args": pe.arguments,
                        }
                    )
                    terminal = {"kind": "tool_call", "name": pe.tool_name, "args": pe.arguments, "raw": pe.raw_xml}
                elif pe.kind == "attempt_completion":
                    if not answer_emitted:
                        _outgoing.append({"type": "answer_start", "step": step})
                        answer_emitted = True
                    if pe.result:
                        _outgoing.append({"type": "answer", "content": pe.result})
                    terminal = {"kind": "attempt_completion", "result": pe.result, "raw": pe.raw_xml}
                elif pe.kind == "parse_error":
                    log.warning("cline parser error step=%d: %s", step, pe.text)
                    terminal = {"kind": "parse_error", "message": pe.text}
            return terminal

        _outgoing: List[Dict[str, Any]] = []
        terminal_event: Optional[Dict[str, Any]] = None
        try:
            async for llm_event in llm_client.chat_stream(messages=messages):
                etype = llm_event.get("type")
                if etype == "thinking_start":
                    # Native model reasoning is internal for the Cline engine.
                    continue
                if etype == "thinking":
                    # Keep live UI aligned with persisted history: do not
                    # surface raw reasoning/thinking tokens.
                    continue
                if etype == "thinking_end":
                    # We close thinking lazily (after parser finishes) so a
                    # late text chunk does not race ahead of the close.
                    continue
                if etype == "answer_start":
                    # Defer to parser to decide when actual user-visible
                    # prose starts.
                    continue
                if etype == "answer":
                    chunk = llm_event.get("content", "") or ""
                    if not chunk:
                        continue
                    collected += chunk
                    parsed = parser.feed(chunk)
                    if parsed:
                        terminal_event = _drain_events(parsed)
                        for ev in _outgoing:
                            yield ev
                        _outgoing.clear()
                    if terminal_event is not None:
                        break
                    continue
                if etype == "error":
                    msg = llm_event.get("content") or "llm_error"
                    log.warning("LLM stream error: %s", msg)
                    yield {"type": "error", "message": msg, "step": step}
                    final_answer = (
                        "A model communication error occurred. Please try again."
                    )
                    return
                # done / unknown — ignore; we wait for the stream generator to exit
        except Exception as exc:
            log.error("LLM streaming failed step=%d: %s", step, exc, exc_info=True)
            yield {"type": "error", "message": str(exc), "step": step}
            final_answer = (
                "A model communication error occurred. Please try again."
            )
            return

        # Flush any tail content.
        if terminal_event is None:
            tail = parser.flush()
            if tail:
                terminal_event = _drain_events(tail)
                for ev in _outgoing:
                    yield ev
                _outgoing.clear()

        if thinking_emitted:
            yield {"type": "thinking_end", "step": step}

        # Persist the raw assistant content so the next iteration sees it.
        if collected:
            messages.append({"role": "assistant", "content": collected})

        step_llm_ms = (_time.monotonic() - t_llm_start) * 1000.0
        yield {"type": "llm_done", "step": step, "llm_ms": round(step_llm_ms, 1)}

        if terminal_event is None:
            # The model neither called a tool nor used attempt_completion.
            # Treat whatever text it produced as the final answer.
            cleaned = (collected or "").strip()
            if cleaned:
                final_answer = cleaned
                final_assistant_text = collected
                break

            if empty_answer_retry_count < 1 and step < effective_max_steps:
                empty_answer_retry_count += 1
                digest = _tool_runs_digest_for_llm(accumulated_tool_runs) if accumulated_tool_runs else ""
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[No Final Answer Produced]\n"
                            "The previous assistant turn produced no visible user-facing answer.\n\n"
                            "You MUST repair this now.\n"
                            "- If you already have enough information, answer the user directly in the user's language.\n"
                            "- If you still need information, emit exactly one valid XML tool block.\n"
                            "- Do not output XML placeholders, code fences, or an empty message.\n"
                            + (
                                "\nTool results digest:\n" + digest
                                if digest
                                else "\nNo tool results were produced in the previous turn."
                            )
                        ),
                    }
                )
                yield {
                    "type": "warning",
                    "step": step,
                    "message": "Model returned no visible answer; retrying once with repair instructions.",
                }
                continue

            final_answer = _fallback_final_answer_from_tool_runs(accumulated_tool_runs)
            final_assistant_text = collected
            break

        if terminal_event["kind"] == "attempt_completion":
            final_answer = (terminal_event.get("result") or "").strip()
            final_assistant_text = collected
            break

        if terminal_event["kind"] == "parse_error":
            if messages and messages[-1].get("role") == "assistant" and messages[-1].get("content") == collected:
                messages.pop()
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "[Tool Parse Error]\n"
                        f"{terminal_event.get('message') or 'malformed_xml'}\n\n"
                        "Your previous tool XML was malformed, so no tool was executed. "
                        "Emit exactly one valid XML tool block using the documented "
                        "parameter tags, or finish with <attempt_completion>."
                    ),
                }
            )
            yield {
                "type": "warning",
                "step": step,
                "message": "Malformed tool XML; asking model to retry without executing.",
            }
            continue

        # tool_call branch
        tool_name = terminal_event["name"]
        tool_args = terminal_event.get("args") or {}
        tool_id = f"call_{step}"
        call_key = _canonical_tool_call_key(tool_name, tool_args)

        if failed_call_counts.get(call_key, 0) >= 2:
            blocked_summary = (
                f"[RETRY_BLOCKED] The exact same tool call `{tool_name}` with the same arguments "
                "has already failed twice in this conversation turn. Treat this as no usable "
                "result for that query; do not call it again. Explain the failure to the user "
                "or try a meaningfully different tool/filename/file_id."
            )
            last_failure_summary = blocked_summary
            blocked_body = _cap_llm_text(blocked_summary)
            accumulated_tool_runs.append(
                {
                    "id": tool_id,
                    "tool_id": tool_name,
                    "args": tool_args,
                    "ok": False,
                    "result": {
                        "ok": False,
                        "error": "repeat_failed_tool_call_blocked",
                        "tool_id": tool_name,
                        "data": {"previous_failures": failed_call_counts.get(call_key, 0)},
                    },
                    "text_for_llm": blocked_body,
                    "summary": blocked_body[:500],
                    "step": step,
                    "skipped": True,
                }
            )
            yield _tool_result_sse_event(
                step=step,
                tool_id=tool_id,
                tool_name=tool_name,
                tool_args=tool_args,
                ok=False,
                summary_text=blocked_summary,
                error="repeat_failed_tool_call_blocked",
            )
            messages.append(
                {
                    "role": "user",
                    "content": _format_tool_result_message(tool_name, False, blocked_summary),
                }
            )
            continue

        # ``act`` phase: the model has chosen a tool. Record the decision
        # before we execute so the ledger reflects causality even if the
        # subsequent execution fails or hangs.
        act_rec = _evidence_record(
            step=step,
            phase="act",
            kind="tool_call",
            summary=f"Calling tool `{tool_name}`.",
            tool_name=tool_name,
            args=tool_args,
            files=_evidence_files_from_tool_args(tool_args),
        )
        evidence_ledger.append(act_rec)
        yield {"type": "evidence_step", **act_rec}

        yield {
            "type": "tool_start",
            "step": step,
            "tool_id": tool_id,
            "tool_name": tool_name,
            "arguments": tool_args,
            "args": tool_args,
            "query": _display_query_from_args(tool_args),
        }

        selected_registry_entry = _match_registry_entry_from_tool_args(tool_args, workspace_registry)
        if selected_registry_entry:
            if _is_visual_registry_entry(selected_registry_entry):
                exact_context = _detect_file_context_from_registry([selected_registry_entry])
                if exact_context:
                    active_file_context = exact_context
            else:
                active_file_context = None

        t_tool_start = _time.monotonic()
        try:
            result = await tool_executor.execute(
                tool_name,
                tool_args,
                file_context=active_file_context,
                session_id=session_id,
                user_id=user_id,
            )
        except Exception as exc:
            log.error("Tool %s execution exception: %s", tool_name, exc, exc_info=True)
            result = {"ok": False, "error": f"execution_failed: {exc}", "tool_id": tool_name}
        if tool_name == "session_attach":
            result = await tool_executor.ensure_session_attach_recovered(
                result,
                tool_args=tool_args,
                session_id=session_id,
                user_id=user_id,
            )
        step_tool_ms += (_time.monotonic() - t_tool_start) * 1000.0

        ok = bool(result.get("ok"))
        summary_text = _normalized_tool_summary(
            tool_name,
            ok,
            _extract_llm_summary(result),
        )
        if ok:
            failed_call_counts.pop(call_key, None)
        else:
            failed_call_counts[call_key] = failed_call_counts.get(call_key, 0) + 1
            last_failure_summary = summary_text

        # Keys match the shape ``chat_history.save_chat_turn`` expects
        # (``tool_id`` for the tool name, ``args`` for the input dict).
        llm_body = _cap_llm_text(summary_text)
        accumulated_tool_runs.append(
            {
                "id": tool_id,
                "tool_id": tool_name,
                "args": tool_args,
                "ok": ok,
                "result": result,
                "text_for_llm": llm_body,
                "summary": llm_body[:500],
                "step": step,
            }
        )

        yield _tool_result_sse_event(
            step=step,
            tool_id=tool_id,
            tool_name=tool_name,
            tool_args=tool_args,
            ok=ok,
            summary_text=summary_text,
        )

        # ``observe`` phase: record the evidence we just gathered.
        observe_rec = _evidence_record(
            step=step,
            phase="observe",
            kind="tool_result",
            summary=summary_text,
            ok=ok,
            tool_name=tool_name,
            args=tool_args,
            files=_evidence_files_from_tool_args(tool_args),
        )
        evidence_ledger.append(observe_rec)
        yield {"type": "evidence_step", **observe_rec}

        if ok and tool_name == "session_attach":
            attached_file = _extract_attach_file_from_result(result)
            if attached_file:
                attached_entry = _registry_entry_from_attach(attached_file)
                workspace_registry = _append_registry_entry(
                    workspace_registry,
                    attached_entry,
                )
                attached_context = _detect_file_context_from_registry([attached_entry])
                if attached_context:
                    active_file_context = (
                        attached_context if _is_visual_registry_entry(attached_entry) else None
                    )
                selected_tools, tool_names_set, param_index, env_details = _refresh_cline_workspace(
                    messages,
                    tools_manifest=tools_manifest,
                    file_registry=workspace_registry,
                    file_context=active_file_context,
                    session_id=session_id,
                    pinned_facts=pinned_facts,
                    history_summary=history_summary,
                    active_plugin=active_plugin,
                    rag_context=rag_context,
                    rag_prompt_block=rag_prompt_block,
                )
                log.info(
                    "📎 session_attach refreshed workspace: %s files, %s LLM tools",
                    len(workspace_registry),
                    len(selected_tools),
                )
                yield {
                    "type": "session_file_attached",
                    "file": attached_file,
                    "registry_size": len(workspace_registry),
                }
                attach_rec = _evidence_record(
                    step=step,
                    phase="attach",
                    kind="session_file_attached",
                    summary=(
                        f"Linked file `{attached_file.get('name') or attached_file.get('path') or attached_file.get('id')}` into the session workspace."
                    ),
                    files=_evidence_files_from_context(attached_file)
                    or [str(attached_file.get('id') or attached_file.get('artifact_id') or '')],
                    meta={"registry_size": len(workspace_registry)},
                )
                evidence_ledger.append(attach_rec)
                yield {"type": "evidence_step", **attach_rec}

        # Feed the result back to the model as the next user turn.
        next_user_content: Any = _format_tool_result_message(tool_name, ok, summary_text)
        current_context_keys = _visual_file_context_keys(active_file_context)
        if (
            active_file_context
            and current_context_keys
            and not current_context_keys.issubset(injected_multimodal_keys)
        ):
            try:
                multimodal_tool_result = await build_multimodal_content(
                    next_user_content,
                    active_file_context,
                    auth_context={"user_id": user_id},
                )
                if isinstance(multimodal_tool_result, list):
                    next_user_content = multimodal_tool_result
                    injected_multimodal_keys |= current_context_keys
                    yield {"type": "vision_attached", "parts": len(multimodal_tool_result)}
                    attach_rec = _evidence_record(
                        step=step,
                        phase="attach",
                        kind="vision_attached",
                        summary=(
                            f"Embedded {len(multimodal_tool_result)} multimodal part(s) for the next model turn."
                        ),
                        files=_evidence_files_from_context(active_file_context),
                    )
                    evidence_ledger.append(attach_rec)
                    yield {"type": "evidence_step", **attach_rec}
            except Exception as exc:
                log.warning("build_multimodal_content for tool result failed: %s", exc)
        if _visual_attachment_missing(active_file_context, injected_multimodal_keys):
            if isinstance(next_user_content, str):
                next_user_content = (
                    f"{next_user_content}\n\n{_missing_visual_attachment_notice(active_file_context)}"
                )
            log.warning(
                "Visual file context present but no multimodal bytes were attached after tool result step=%s.",
                step,
            )
            yield {
                "type": "warning",
                "step": step,
                "message": "A visual file is selected, but its bytes were not attached to the model for the next turn.",
            }

        messages.append(
            {
                "role": "user",
                "content": next_user_content,
            }
        )
        step_total_ms = (_time.monotonic() - step_started_at) * 1000.0
        step_timings.append({
            "step": step,
            "llm_ms": round(step_llm_ms, 1),
            "tool_ms": round(step_tool_ms, 1),
            "total_ms": round(step_total_ms, 1),
        })
        # Loop continues for the next LLM round.

    if final_answer is None:
        if last_failure_summary:
            not_found = session_attach_stale_error_text(last_failure_summary)
            recovered_guidance = (
                _fallback_final_answer_from_tool_runs(accumulated_tool_runs)
                if not_found
                else None
            )
            if recovered_guidance and not session_attach_stale_error_text(recovered_guidance):
                final_answer = recovered_guidance
            elif not_found:
                final_answer = SESSION_ATTACH_HISTORY_GUIDANCE
            else:
                final_answer = (
                    "I could not obtain a usable result from the tools. "
                    f"Latest tool error: {last_failure_summary}"
                )
        else:
            final_answer = (
                (final_assistant_text or "").strip()
                or _fallback_final_answer_from_tool_runs(accumulated_tool_runs)
            )
    if session_attach_stale_error_text(final_answer or ""):
        alt = _fallback_final_answer_from_tool_runs(accumulated_tool_runs)
        final_answer = (
            alt
            if alt and not session_attach_stale_error_text(alt)
            else SESSION_ATTACH_HISTORY_GUIDANCE
        )
    if _visual_attachment_missing(active_file_context, injected_multimodal_keys):
        log.warning(
            "Overriding final answer because visual media was not attached to the model turn."
        )
        final_answer = _final_answer_for_missing_visual_attachment(
            active_file_context,
            accumulated_tool_runs,
        )
        final_assistant_text = final_answer

    # ------------------------------------------------------------------
    # Strip heavy channels from tool_runs before exporting (parity with the
    # old ``stream_agent``).
    # ------------------------------------------------------------------
    def _strip_channels(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Slim tool_runs for SSE clients; neo4j/rawdata stay intact for downstream writers."""
        stripped = []
        for tr in runs:
            tr = dict(tr)
            tr.pop("output", None)
            try:
                result = dict(tr.get("result") or {})
                data = result.get("data") if isinstance(result.get("data"), dict) else {}
                inner = data.get("result") if isinstance(data.get("result"), dict) else {}
                channels = inner.get("channels") if isinstance(inner.get("channels"), dict) else None
                if isinstance(channels, dict) and channels:
                    inner = dict(inner)
                    inner["channels"] = {
                        "llm": channels.get("llm"),
                        "neo4j": channels.get("neo4j"),
                        "rawdata": channels.get("rawdata"),
                        "metadata": channels.get("metadata"),
                    }
                    tool_data = inner.get("data")
                    if isinstance(tool_data, dict) and isinstance(tool_data.get("resolutions"), list):
                        tool_data = dict(tool_data)
                        tool_data["resolutions"] = [
                            _slim_resolution_for_export(r) for r in tool_data["resolutions"]
                        ]
                        inner["data"] = tool_data
                    data = dict(data)
                    data["result"] = inner
                    result["data"] = data
                    tr["result"] = result
            except Exception:
                pass
            stripped.append(tr)
        return stripped

    # ``conclude`` phase: record the final synthesis the loop produced so the
    # ledger ends with a clear cause-and-effect trail.
    conclude_rec = _evidence_record(
        step=max((r.get("step", 0) for r in evidence_ledger), default=0),
        phase="conclude",
        kind="final_answer",
        summary=(final_answer or "")[:600],
        meta={
            "tool_runs": len(accumulated_tool_runs),
            "tool_failures": sum(1 for tr in accumulated_tool_runs if not tr.get("ok")),
            "visual_attached": sorted(injected_multimodal_keys) if injected_multimodal_keys else [],
        },
    )
    evidence_ledger.append(conclude_rec)
    yield {"type": "evidence_step", **conclude_rec}

    evidence_summary = _summarize_evidence_for_done(
        evidence_ledger,
        tool_runs=accumulated_tool_runs,
    )

    engine_total_ms = (_time.monotonic() - t_engine_start) * 1000.0
    perf_summary = {
        "engine_total_ms": round(engine_total_ms, 1),
        "rounds": len(step_timings),
        "llm_total_ms": round(sum(s.get("llm_ms", 0.0) for s in step_timings), 1),
        "tool_total_ms": round(sum(s.get("tool_ms", 0.0) for s in step_timings), 1),
        "per_step": step_timings,
    }
    log.info(
        "⏱️ cline_engine done: total=%.1fms rounds=%d llm=%.1fms tools=%.1fms",
        perf_summary["engine_total_ms"],
        perf_summary["rounds"],
        perf_summary["llm_total_ms"],
        perf_summary["tool_total_ms"],
    )

    yield {
        "type": "done",
        "trace_id": trace_id,
        "final_answer": final_answer,
        "thinking": None,  # surfaced live, no aggregate
        "tool_runs": _strip_channels(accumulated_tool_runs),
        "evidence_ledger": evidence_ledger,
        "evidence_summary": evidence_summary,
        "perf_summary": perf_summary,
        "engine": "cline_xml",
        "assistant_raw": final_assistant_text,
        "timestamp": _dt.datetime.utcnow().isoformat(),
    }
