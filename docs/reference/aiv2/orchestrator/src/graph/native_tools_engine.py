"""
Native OpenAI `tool_calls` streaming engine for AIv2.

This engine is the high-performance sibling of ``cline_engine``. Where the
Cline-XML engine is constrained to **one tool per LLM turn** (the wire is
``<tool_name>...</tool_name>`` inside the assistant text and the parser stops
the stream the moment that closing tag appears), this engine uses the OpenAI
``tools=[...] / tool_choice="auto"`` channel:

* The model may emit *any number* of ``tool_calls`` in a single streaming
  response.
* The engine executes them **in parallel** via ``asyncio.gather`` and feeds
  each result back as a ``role="tool"`` message keyed by ``tool_call_id``.
* The conversation then loops with a single follow-up LLM round that either
  emits more ``tool_calls`` or produces the final answer.

For a turn that needs ``N`` tools, this collapses ``N+1`` LLM rounds (Cline)
into ``2`` rounds (decide + finalize). On wall-clock terms that's the gap
between AIv2 and Continue's "ریز ریز فکر میکنه و به نتیجه میرسه" behaviour.

The yielded SSE event shapes are intentionally identical to those produced
by ``stream_agent_cline`` so ``main.py``/``generate_stream`` and the React
frontend continue to work unchanged. Concretely we emit:

    start → file_detected → evidence_step* → llm_start →
    answer_start? → answer* → (tool_call_start → tool_start → tool_result)+ →
    evidence_step* → llm_done → … → done

Only the *engine label* on ``start``/``done`` is different (``native_tools``)
so downstream telemetry can tell the two engines apart.

The heavy lifting helpers — env details, registry detection, multimodal
embedding, missing-vision guard rails, evidence ledger, tool-result envelope
flattening, repeat-call blocking — are re-used **as-is** from ``cline_engine``
to keep behaviour aligned and avoid drift between the two paths.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import re
import time as _time
import unicodedata
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from ..config import settings
from ..auth import resolve_allowed_tools
from ..llm.gateway_llm_client import get_gateway_llm_client
from ..tools.executor import (
    SESSION_ATTACH_HISTORY_GUIDANCE,
    get_tool_executor,
    session_attach_stale_error_text,
)
from ..session_tool_resolve import (
    extract_attach_filename_hint,
    resolve_read_file_args,
    should_preloop_session_read,
    user_wants_case_lookup,
    user_wants_file_discovery,
    user_wants_file_read,
)
from ..guardrails import ToolLoopGuard
from ..llm.gateway_llm_client import (
    _strip_looping_assistant_content,
)
from ..tools.llm_tool_selection import (
    page_allows_tool,
    select_llm_tools_for_turn,
    selection_debug_summary,
    _tool_name as _manifest_tool_name,
)
from ..utils.file_context import detect_file_context
from ..utils.multimodal import build_multimodal_content
from ..utils.message_compile import compile_messages_continue_style, estimate_messages_chars
from ..prompts.cline_xml import _format_env_details  # type: ignore[attr-defined]

# Re-use Cline engine helpers so behaviour stays aligned across the two paths.
from .cline_engine import (
    _append_registry_entry,
    _build_environment_details,
    _canonical_tool_call_key,
    _cap_llm_text,
    _detect_file_context_from_registry,
    _display_query_from_args,
    _evidence_files_from_context,
    _evidence_files_from_tool_args,
    _evidence_record,
    _extract_attach_file_from_result,
    _extract_llm_summary,
    _fallback_final_answer_from_tool_runs,
    _final_answer_for_missing_visual_attachment,
    _format_tool_result_message,
    _is_visual_registry_entry,
    _match_registry_entry_from_tool_args,
    _missing_visual_attachment_notice,
    _normalized_tool_summary,
    _registry_entry_from_attach,
    _slim_resolution_for_export,
    _summarize_evidence_for_done,
    _tool_result_sse_event,
    _tool_runs_digest_for_llm,
    _visual_attachment_missing,
    _visual_file_context_keys,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builder — native tools edition (no XML tool docs)
# ---------------------------------------------------------------------------

_NATIVE_HEADER = (
    "You are AIv2, a highly skilled assistant operating inside a session-scoped "
    "workspace. You can search a knowledge graph, read uploaded files, analyse "
    "images and audio, and write detailed answers in the user's language."
)

_NATIVE_TOOL_USE_RULES = """====

TOOL USE

Tool definitions for this turn are supplied **only** via the API `tools` parameter
(OpenAI function-calling). Read each function's `description` and `parameters`
there before calling. This system prompt does not duplicate tool catalogs.

You may call **multiple tools** in one assistant response — emit them in `tool_calls`
and the orchestrator executes them in parallel, returning results keyed by
`tool_call_id`.

Rules:
- Call a tool only when you need its output. Do not repeat the same tool with
  the same arguments twice in a row.
- For file tools, never invent absolute paths; use filenames/file_ids from
  SESSION ENVIRONMENT as each tool's parameters require.
- Do not invent tool names — only names present in the API `tools` list.
- Arguments MUST be valid JSON.
- Never output standalone JSON argument blobs or fenced ```json parameter-only blocks.
  Emit real native tool_calls, or answer in prose.
- After tool results: call more tools if needed, or write the final answer."""

_NATIVE_RULES_BLOCK = """====

RULES

- Match the user's language exactly. Persian → Persian, English → English,
  mixed → dominant language of the latest user turn.
- The user's latest message is the priority. Use chat history as context
  only — never repeat earlier answers verbatim.
- Iterate: call as many tools as you need (in parallel when independent),
  then write the final answer.
- Quote concrete evidence from tool outputs (numbers, file ids, names);
  never invent data.
- If you genuinely cannot fulfill the request after honest effort, say so
  explicitly and state what is missing."""

_NATIVE_MULTIMODAL_NOTE = """====

VISION (native multimodal)

When the user attaches an image or video, pixels are in the `user` message as
`image_url` / `video_url` parts. Describe/OCR from perception directly — do not
call a tool only to "view" attached media. Use API `tools` when indexing or
reading non-attached session files (see each tool's description)."""

_NATIVE_RUNTIME_NOTE = """====

WORKSPACE RUNTIME

The WORKSPACE RUNTIME block is passive context already resolved from the current
session. It may contain exact file targets, inline text snippets, and active
visual targets. Read it first.

If the runtime block already has enough evidence, answer directly. Otherwise
use a tool from the API `tools` list — do not claim tools are unavailable."""


def _format_workspace_runtime_block(runtime: Optional[Dict[str, Any]]) -> str:
    if not isinstance(runtime, dict):
        return ""

    lines: List[str] = [
        "====",
        "",
        "WORKSPACE RUNTIME THIS TURN",
        "",
        "Treat this as passive, already-available session context.",
        "Prefer it before calling tools, but tools remain available every turn.",
        "",
    ]

    available_count = int(runtime.get("available_files_count") or 0)
    lines.append(f"- Available session files: {available_count}")
    if runtime.get("needs_session_files_pagination"):
        lines.append(
            "- More than 20 files in session: use `session_files(session_id=..., limit=20, offset=0)` "
            "to paginate before picking a file_id."
        )

    focus_block = str(runtime.get("active_focus_block") or "").strip()
    if focus_block:
        lines.append("")
        lines.append(focus_block)

    tiered = runtime.get("tiered_catalog") or {}
    if isinstance(tiered, dict):
        active = tiered.get("active_focus") or []
        recent = tiered.get("recent_touched") or []
        if active:
            lines.append("- Active focus (under discussion now):")
            for item in active[:2]:
                if isinstance(item, dict):
                    lines.append(
                        f"  - `{item.get('name')}` id=`{item.get('artifact_id')}` ({item.get('type')})"
                    )
        if recent:
            lines.append("- Recently touched by tools:")
            for item in recent[:5]:
                if isinstance(item, dict):
                    lines.append(
                        f"  - `{item.get('name')}` id=`{item.get('artifact_id')}` via {item.get('last_tool')}"
                    )
        overflow = int(tiered.get("overflow_count") or 0)
        if overflow > 0:
            lines.append(f"- (+{overflow} more files — call session_files with offset/limit)")

    selected_files = runtime.get("selected_files") or []
    if selected_files:
        lines.append("- Exact resolved targets:")
        for item in selected_files[:3]:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or "unnamed"
            path = item.get("path") or ""
            match_reason = item.get("match_reason") or "resolved"
            type_label = item.get("type") or "file"
            lines.append(f"  - `{name}` ({type_label}, reason={match_reason}) path=`{path}`")
            knowledge = item.get("knowledge") or []
            for fact in knowledge[:2]:
                if not isinstance(fact, dict):
                    continue
                tool = str(fact.get("tool") or "tool").strip()
                summary = " ".join(str(fact.get("summary") or "").split())
                if summary:
                    lines.append(f"    - prior {tool}: {summary[:220]}")

    text_snippets = runtime.get("text_snippets") or []
    if text_snippets:
        lines.append("- Inline text evidence:")
        for item in text_snippets[:2]:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or "unnamed"
            snippet = str(item.get("snippet") or "").strip()
            if not snippet:
                continue
            lines.append(f"  - `{name}` snippet:")
            lines.append(f"    {snippet[:700]}")

    visual_targets = runtime.get("visual_targets") or []
    if visual_targets:
        lines.append("- Active visual targets:")
        for item in visual_targets[:2]:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or "unnamed"
            path = item.get("path") or ""
            lines.append(f"  - `{name}` path=`{path}`")

    file_index = runtime.get("file_index") or []
    if file_index:
        lines.append("- Compact file index (ordinal | name | type):")
        for item in file_index[:20]:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or "unnamed"
            type_label = item.get("type") or "file"
            ordinal = item.get("ordinal")
            aid = item.get("artifact_id") or "?"
            lines.append(f"  - #{ordinal or '?'} `{name}` ({type_label}) id=`{aid}`")

    return "\n".join(lines).rstrip()


def _message_text_content(msg: Dict[str, Any]) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return " ".join(parts).strip()
    return ""


def _is_affirmative_followup(text: str) -> bool:
    s = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not s:
        return False
    affirmatives = {
        "yes", "ok", "okay", "sure", "do it", "go ahead",
        "بله", "آره", "اره", "اوکی", "باشه", "خوبه", "بزن بریم",
    }
    return s in affirmatives or s in {f"{x}." for x in affirmatives}


def _pending_confirmed_artifact_id(
    user_message: str,
    chat_history: Optional[List[Dict[str, Any]]],
) -> Optional[str]:
    """Infer `artifact_id` when the user confirms the previous suggestion.

    Pattern:
    assistant: "Do you want artifact_id X?"
    user: "yes"
    """
    if not _is_affirmative_followup(user_message):
        return None
    if not chat_history:
        return None
    for row in reversed(chat_history):
        if not isinstance(row, dict) or row.get("role") != "assistant":
            continue
        text = _message_text_content(row)
        if not text:
            continue
        text_lc = text.lower()
        if "artifact_id" not in text_lc:
            continue
        if not any(sig in text_lc for sig in ("do you want", "would you like", "آیا می", "انتخاب", "choose", "pick")):
            continue
        ids = re.findall(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", text_lc)
        if len(ids) == 1:
            return ids[0]
        return None
    return None


def _explicit_or_confirmed_attach_artifact_id(
    user_message: str,
    chat_history: Optional[List[Dict[str, Any]]],
) -> Optional[str]:
    text = user_message or ""
    ids = re.findall(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", text)
    if len(ids) == 1:
        return ids[0].lower()

    actionish = any(
        token in text.lower()
        for token in ("attach", "connect", "use this", "select this", "متصل", "وصل", "همین", "این", "بله", "اوکی", "باشه")
    )
    if actionish:
        return _pending_confirmed_artifact_id(user_message, chat_history)
    return None


_SESSION_READ_TOOL_NAMES = frozenset({
    "session_read_file",
    "session_read",
    "plugin_session_read",
})


async def _openai_tools_for_selected(
    tool_executor: Any,
    selected_tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Prefer canonical JSON Schema from tool-runner; fallback to manifest parse."""
    ids = [_manifest_tool_name(t) for t in selected_tools if _manifest_tool_name(t)]
    if ids:
        canonical = await tool_executor.get_openai_tools_for_ids(ids)
        if len(canonical) >= len(ids):
            return canonical
    return tool_executor.get_openai_tools_format(selected_tools)


def _parse_line_range_from_user_message(text: str) -> Optional[Dict[str, int]]:
    """Extract start_line/end_line when the user asks for a specific slice."""
    s = text or ""
    m = re.search(r"خط\s*(?:و)?\s*(\d+)\s*(?:تا|الی|-)\s*(\d+)", s, re.I)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        if start > end:
            start, end = end, start
        return {"start_line": start, "end_line": end}
    m = re.search(r"lines?\s*(\d+)\s*(?:to|-)\s*(\d+)", s, re.I)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        if start > end:
            start, end = end, start
        return {"start_line": start, "end_line": end}
    m = re.search(r"(?:خط|line)\s*(\d+)\b", s, re.I)
    if m:
        n = int(m.group(1))
        return {"start_line": n, "end_line": n}
    return None


def _offered_tool_names(tools: Optional[List[Dict[str, Any]]]) -> set:
    """Names of the tools offered to the model; empty disables the unknown-tool gate."""
    if not settings.NATIVE_REJECT_UNOFFERED_TOOLS:
        return set()
    names = set()
    for spec in tools or []:
        fn = spec.get("function") if isinstance(spec, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.add(str(fn["name"]))
    return names


def _unknown_tool_message(name: str, offered: set) -> str:
    return settings.NATIVE_UNKNOWN_TOOL_TEMPLATE.format(
        name=name, available=", ".join(sorted(offered))
    )


def _inject_preloop_tool_into_messages(
    messages: List[Dict[str, Any]],
    run: Dict[str, Any],
) -> None:
    """Wire orchestrator-executed pre-loop tools into the LLM message stack."""
    tid = str(run.get("id") or f"preloop_{uuid.uuid4().hex[:12]}")
    tname = str(run.get("tool_id") or "tool")
    targs = run.get("args") if isinstance(run.get("args"), dict) else {}
    body = _cap_llm_text(str(run.get("text_for_llm") or run.get("summary") or ""))
    try:
        args_str = json.dumps(targs, ensure_ascii=False)
    except Exception:
        args_str = "{}"
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tid,
                    "type": "function",
                    "function": {"name": tname, "arguments": args_str},
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tid,
            "name": tname,
            "content": body,
        }
    )


def _primary_registry_file_for_read(
    registry: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    if not registry:
        return None
    textish: List[Dict[str, Any]] = []
    for entry in registry:
        if not isinstance(entry, dict):
            continue
        mime = str(entry.get("mime_type") or "").lower()
        ftype = str(entry.get("type") or entry.get("media_type") or "").lower()
        name = str(entry.get("name") or entry.get("filename") or "").lower()
        if (
            ftype in {"text", "document", "csv"}
            or mime.startswith("text/")
            or name.endswith((".txt", ".csv", ".md", ".json", ".log"))
        ):
            textish.append(entry)
    if textish:
        return textish[-1]
    return registry[-1] if isinstance(registry[-1], dict) else None


def _user_facing_llm_error(raw_msg: str) -> str:
    raw = (raw_msg or "").strip()
    if "chat_model_unavailable" in raw:
        return "سرویس چت موقتاً در دسترس نیست. لطفاً بعداً تلاش کنید."
    if "400" in raw or "Bad Request" in raw:
        return (
            "درخواست به مدل رد شد (معمولاً به‌خاطر تاریخچهٔ نامعتبر یا حذف پیام کاربر). "
            "یک سشن جدید باز کنید یا دوباره همان سؤال را بفرستید."
        )
    if "all_backends_failed" in raw or "unhealthy" in raw:
        return "سرویس مدل الان در دسترس نیست. لطفاً چند دقیقه بعد دوباره تلاش کنید."
    return "خطا در ارتباط با مدل. لطفاً دوباره تلاش کنید."


def _append_user_repair_once(
    messages: List[Dict[str, Any]],
    content: str,
) -> None:
    """Append a repair user turn without stacking duplicate repair prompts."""
    marker = (content or "").strip()[:48]
    if marker and messages:
        last = messages[-1]
        if last.get("role") == "user" and marker in _message_text_content(last):
            return
    messages.append({"role": "user", "content": content})


def _history_tool_digest_for_llm(
    chat_history: Optional[List[Dict[str, Any]]],
    *,
    limit: int = 4,
) -> str:
    if not chat_history:
        return ""
    rows: List[str] = []
    for row in reversed(chat_history):
        if not isinstance(row, dict):
            continue
        role = row.get("role")
        if role == "tool":
            body = _message_text_content(row).strip()
            if body:
                rows.append(body)
        elif role == "assistant" and row.get("tool_calls"):
            continue
        if len(rows) >= limit:
            break
    return "\n\n---\n\n".join(reversed(rows)).strip()


def _recent_tool_digest_from_messages(
    messages: Optional[List[Dict[str, Any]]],
    *,
    limit: int = 3,
) -> str:
    """Tool outputs already in the in-flight message stack (this turn only)."""
    if not messages:
        return ""
    turn_start = 0
    for idx in range(len(messages) - 1, -1, -1):
        row = messages[idx]
        if isinstance(row, dict) and row.get("role") == "user" and not row.get("_synthetic"):
            turn_start = idx + 1
            break
    rows: List[str] = []
    for row in reversed(messages[turn_start:]):
        if not isinstance(row, dict) or row.get("role") != "tool":
            continue
        body = _message_text_content(row).strip()
        if body and not body.startswith("[LOOP_BLOCKED]"):
            rows.append(body)
        if len(rows) >= limit:
            break
    return "\n\n---\n\n".join(reversed(rows)).strip()


def _synthesize_from_tool_evidence(
    *,
    accumulated_tool_runs: Optional[List[Dict[str, Any]]] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Best-effort user-facing answer from *this turn's* tool output when the LLM returns blank.

    Earlier turns' tool output is never replayed as an answer: it answers a different question.
    """
    del chat_history
    return (
        _deterministic_answer_from_tool_runs(accumulated_tool_runs or [])
        or _recent_tool_digest_from_messages(messages)
        or ""
    ).strip()


def _emit_preloop_tool_events(
    run: Dict[str, Any],
    *,
    step: int = 0,
) -> List[Dict[str, Any]]:
    """Yield-shaped SSE events for orchestrator-executed pre-loop tools."""
    tool_id = str(run.get("id") or "preloop_tool")
    tool_name = str(run.get("tool_id") or "unknown")
    tool_args = run.get("args") if isinstance(run.get("args"), dict) else {}
    ok = bool(run.get("ok"))
    summary_text = str(run.get("summary") or run.get("text_for_llm") or "")
    return [
        {
            "type": "tool_call_start",
            "step": step,
            "index": 0,
            "tool_id": tool_id,
            "tool_name": tool_name,
        },
        {
            "type": "tool_start",
            "step": step,
            "tool_id": tool_id,
            "tool_name": tool_name,
            "arguments": tool_args,
            "args": tool_args,
            "query": _display_query_from_args(tool_args),
        },
        _tool_result_sse_event(
            step=step,
            tool_id=tool_id,
            tool_name=tool_name,
            tool_args=tool_args,
            ok=ok,
            summary_text=summary_text,
            error=None if ok else str((run.get("result") or {}).get("error") or "tool_failed"),
        ),
    ]


def _tool_parameter_names(tools: Optional[List[Dict[str, Any]]]) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else None
        if not isinstance(fn, dict):
            continue
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        props = params.get("properties") if isinstance(params, dict) else {}
        if not isinstance(props, dict):
            continue
        for key in props.keys():
            if isinstance(key, str) and key.strip():
                names.add(key.strip())
    return names


def _unwrap_json_fence(text: str) -> str:
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return s


def _looks_like_orphan_tool_args_blob(
    text: str,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Detect a malformed tool attempt like ``{\"filename\":\"x.txt\"}``.

    Some model turns leak only an argument object or a fenced JSON parameter
    block instead of emitting a real native tool call. That blob should not be
    treated as a user-facing final answer.
    """
    s = _unwrap_json_fence(text)
    if not s or not (s.startswith("{") or s.startswith("[")):
        return False
    try:
        obj = json.loads(s)
    except Exception:
        return False

    disqualify = {
        "tool_calls",
        "calls",
        "functions",
        "function",
        "name",
        "tool_name",
        "tool",
        "result",
        "answer",
        "final_answer",
        "summary",
        "message",
        "content",
    }
    tool_arg_names = _tool_parameter_names(tools)
    if not tool_arg_names:
        return False

    def _dict_is_args(d: Dict[str, Any]) -> bool:
        keys = {str(k).strip() for k in d.keys() if str(k).strip()}
        if not keys or keys & disqualify:
            return False
        return keys.issubset(tool_arg_names)

    if isinstance(obj, dict):
        return _dict_is_args(obj)
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        return all(_dict_is_args(x) for x in obj)
    return False


_PLUGIN_SYSTEM_SUFFIXES: Dict[str, str] = {}  # deprecated — use plugin.json llm_host_prompt


def _dominant_script(text: str) -> str:
    """Unicode script name (``LATIN``, ``ARABIC``, ``CYRILLIC``, …) of most letters in ``text``."""
    counts: Dict[str, int] = {}
    for ch in str(text or ""):
        if ch.isalpha():
            try:
                script = unicodedata.name(ch).split(" ", 1)[0]
            except ValueError:
                continue
            counts[script] = counts.get(script, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def _render_script_template(template: Any, user_message: str) -> str:
    template = str(template or "").strip()
    script = _dominant_script(user_message)
    if not template or not script:
        return ""
    return template.replace("{script}", script.capitalize())


def _reply_language_hint(user_message: str) -> str:
    """Per-turn reply-language anchor (``settings.NATIVE_REPLY_LANGUAGE_HINT``; empty disables)."""
    return _render_script_template(settings.NATIVE_REPLY_LANGUAGE_HINT, user_message)


def _with_reply_language_note(content: Any, user_message: str) -> Any:
    """Model copy of the user turn + ``settings.NATIVE_REPLY_LANGUAGE_USER_NOTE`` (empty disables)."""
    note = _render_script_template(settings.NATIVE_REPLY_LANGUAGE_USER_NOTE, user_message)
    if not note:
        return content
    if isinstance(content, str):
        return f"{content}\n\n{note}"
    if isinstance(content, list):
        return [*content, {"type": "text", "text": note}]
    return content


def build_native_system_prompt(
    tools: Optional[List[Dict[str, Any]]] = None,
    env_details: Optional[Dict[str, Any]] = None,
    *,
    workspace_runtime: Optional[Dict[str, Any]] = None,
    multimodal_enabled: bool = True,
    extra_rules: Optional[str] = None,
    active_plugin: Optional[str] = None,
    rag_context: Optional[List[Dict[str, Any]]] = None,
    rag_prompt_block: Optional[str] = None,
    memory_prompt_block: Optional[str] = None,
    reply_language_hint: Optional[str] = None,
    host_blocks: Optional[Dict[str, str]] = None,
) -> str:
    """Build the slim system prompt used by the native-tools engine.

    Tool names, descriptions, and JSON schemas are delivered only via the OpenAI
    ``tools`` request parameter for this turn (``tools`` arg is accepted for API
    symmetry but not echoed into the prompt).
    """
    _ = tools  # catalog lives in vLLM ``tools=`` only
    sections: List[str] = [_NATIVE_HEADER, _NATIVE_TOOL_USE_RULES]
    if multimodal_enabled:
        sections.append(_NATIVE_MULTIMODAL_NOTE)
    if workspace_runtime:
        sections.append(_NATIVE_RUNTIME_NOTE)
    sections.append(_NATIVE_RULES_BLOCK)
    if extra_rules:
        sections.append(extra_rules.strip())
    try:
        from ..plugin_host_prompt import get_llm_host_prompt

        plugin_suffix = get_llm_host_prompt(active_plugin)
    except Exception:
        plugin_suffix = ""
    if plugin_suffix:
        sections.append(plugin_suffix)

    rag_block = (rag_prompt_block or "").strip()
    if not rag_block and rag_context:
        from ..rag import format_rag_prompt_block
        from ..config import settings as _settings

        rag_block = format_rag_prompt_block(
            rag_context,
            max_chars_per_hit=int(_settings.NATIVE_RAG_MAX_CHARS_PER_HIT or 1200),
        )
    if rag_block:
        sections.append(rag_block)
    if (memory_prompt_block or "").strip():
        sections.append(memory_prompt_block.strip())
    for _name, text in sorted((host_blocks or {}).items()):
        if (text or "").strip():
            sections.append(text.strip())

    if env_details:
        sections.append(_format_env_details(env_details))
    runtime_block = _format_workspace_runtime_block(workspace_runtime)
    if runtime_block:
        sections.append(runtime_block)
    if (reply_language_hint or "").strip():
        sections.append(reply_language_hint.strip())
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_result_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort unwrap of nested tool-runner envelopes."""
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    inner = data.get("result") if isinstance(data.get("result"), dict) else data
    payload = inner.get("data") if isinstance(inner.get("data"), dict) else inner
    return payload if isinstance(payload, dict) else {}


def _is_ambiguous_attach_result(result: Dict[str, Any]) -> bool:
    if not result.get("ok"):
        return False
    payload = _tool_result_payload(result)
    if payload.get("status") == "ambiguous":
        return True
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    inner = data.get("result") if isinstance(data.get("result"), dict) else data
    return isinstance(inner, dict) and inner.get("status") == "ambiguous"


def _ambiguous_attach_user_answer(result: Dict[str, Any], fallback: str) -> str:
    """Compact user-facing list (no raw JSON / long storage URIs)."""
    payload = _tool_result_payload(result)
    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    channels = result.get("channels") if isinstance(result.get("channels"), dict) else {}
    ui = channels.get("ui") if isinstance(channels.get("ui"), dict) else {}
    if not candidates and isinstance(ui.get("candidates"), list):
        candidates = ui["candidates"]
    if not candidates:
        cleaned = (fallback or "").strip()
        if cleaned.startswith("{") and '"status"' in cleaned:
            return (
                "Multiple files match this name. Nothing was attached yet. "
                "Please pick a more specific file (full path, case, or artifact id)."
            )
        return cleaned or (
            "Multiple files match. Nothing was attached yet — pick one more specifically."
        )

    lines = [
        f"Multiple files match ({len(candidates)}). Nothing was attached yet.",
        "Reply with the number or exact filename you want attached:",
        "",
    ]
    show = candidates[:25]
    for i, c in enumerate(show, start=1):
        if not isinstance(c, dict):
            continue
        name = str(c.get("filename") or c.get("name") or "?").strip() or "?"
        cid = str(c.get("case_id") or "").strip()
        ftype = str(c.get("media_type") or "").strip()
        extra = []
        if cid:
            extra.append(f"case `{cid}`")
        if ftype:
            extra.append(ftype)
        suffix = f" — {', '.join(extra)}" if extra else ""
        lines.append(f"{i}. `{name}`{suffix}")
    if len(candidates) > len(show):
        lines.append(f"… and {len(candidates) - len(show)} more")
    return "\n".join(lines)


def _deterministic_answer_from_tool_runs(tool_runs: List[Dict[str, Any]]) -> Optional[str]:
    """Return a user-facing answer synthesized from tool output when the LLM blanks."""
    if not tool_runs:
        return None
    for tr in reversed(tool_runs):
        text = str(tr.get("text_for_llm") or tr.get("summary") or "").strip()
        if text and text not in {"{}", "[]", "null", "None"}:
            return text
    return _fallback_final_answer_from_tool_runs(tool_runs)


def _dedup_session_read_in_history(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop repeated session_read tool rows across the reconstructed history."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for row in messages:
        if not isinstance(row, dict):
            continue
        if row.get("role") == "tool" and (row.get("name") or "") in _SESSION_READ_TOOL_NAMES:
            key = (
                row.get("name"),
                row.get("tool_call_id") or "",
                str(row.get("content") or "")[:600],
            )
            if key in seen:
                continue
            seen.add(key)
        out.append(row)
    return out


def _normalize_history_for_native(
    chat_history: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Pass-through normaliser for history rows shaped by ``chat_history``.

    The chat-history layer already returns assistant rows with native
    ``tool_calls`` and matching ``role="tool"`` rows (see
    ``_assistant_turn_blocks_for_stream_llm``). We only have to:

    * coerce ``tool_calls.function.arguments`` to a JSON string (vLLM wants
      strings, not dicts);
    * forward ``attachments`` on user rows verbatim so the vLLM client can
      attach images/videos from earlier turns when needed.
    """
    if not chat_history:
        return []
    out: List[Dict[str, Any]] = []
    for row in chat_history:
        if not isinstance(row, dict):
            continue
        role = row.get("role")
        if role == "user":
            d: Dict[str, Any] = {
                "role": "user",
                "content": row.get("content", "") or "",
            }
            att = row.get("attachments")
            if att:
                d["attachments"] = att
            out.append(d)
            continue
        if role == "assistant":
            ai: Dict[str, Any] = {
                "role": "assistant",
                "content": row.get("content", "") or "",
            }
            tcs = row.get("tool_calls") or []
            if tcs:
                norm: List[Dict[str, Any]] = []
                for i, tc in enumerate(tcs):
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else None
                    if fn:
                        name = fn.get("name") or ""
                        args = fn.get("arguments")
                    else:
                        name = (tc.get("name") or "").strip()
                        args = tc.get("arguments") if "arguments" in tc else tc.get("args")
                    if isinstance(args, str):
                        try:
                            args_dict = json.loads(args) if args.strip() else {}
                        except Exception:
                            args_dict = {}
                    else:
                        args_dict = args if isinstance(args, dict) else {}
                    if name:
                        args_dict = _prepare_tool_args(name, args_dict)
                    try:
                        args_str = json.dumps(args_dict or {}, ensure_ascii=False)
                    except Exception:
                        args_str = "{}"
                    norm.append(
                        {
                            "id": str(tc.get("id") or f"call_{i}"),
                            "type": "function",
                            "function": {"name": name, "arguments": args_str},
                        }
                    )
                if norm:
                    ai["tool_calls"] = norm
            out.append(ai)
            continue
        if role == "tool":
            tname = str(row.get("name") or "tool")
            content = str(row.get("content") or "")
            if tname == "session_attach" and session_attach_stale_error_text(content):
                content = SESSION_ATTACH_HISTORY_GUIDANCE
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": str(row.get("tool_call_id") or ""),
                    "name": tname,
                    "content": content,
                }
            )
            continue
    return _dedup_repeated_tool_messages(_dedup_session_read_in_history(out))


def _dedup_repeated_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop consecutive tool rows with identical content (stops empty-session echo loops)."""
    if not messages:
        return messages
    out: List[Dict[str, Any]] = []
    prev_key: Optional[tuple] = None
    for row in messages:
        if isinstance(row, dict) and row.get("role") == "tool":
            key = (
                row.get("name"),
                str(row.get("content") or "")[:500],
            )
            if key == prev_key:
                continue
            prev_key = key
        else:
            prev_key = None
        out.append(row)
    if len(out) < len(messages):
        log.info(
            "history tool dedup: %s -> %s messages",
            len(messages),
            len(out),
        )
    return out


def _sanitize_attach_terminal_answer(
    final_answer: Optional[str],
    tool_runs: List[Dict[str, Any]],
) -> Optional[str]:
    """Replace verbatim session_attach not-found dead-ends with actionable guidance."""
    text = (final_answer or "").strip()
    if not text or not session_attach_stale_error_text(text):
        return final_answer
    alt = _deterministic_answer_from_tool_runs(tool_runs)
    if alt and not session_attach_stale_error_text(alt):
        return alt
    return SESSION_ATTACH_HISTORY_GUIDANCE


async def _model_accepts_images(llm_client: Any) -> bool:
    """False only when the bound model's catalog entry declares it cannot take images."""
    from ..orchestrators.host_adapters.model_catalog import model_input_modalities

    try:
        model_id = await llm_client._resolve_model()
    except Exception:
        return True
    modalities = await model_input_modalities(model_id)
    return modalities is None or bool(modalities & {"image", "vision"})


def _refresh_native_workspace(
    *,
    tools_manifest: List[Dict[str, Any]],
    file_registry: List[Dict[str, Any]],
    file_context: Optional[Dict[str, Any]],
    session_id: str,
    pinned_facts: Optional[Dict[str, Any]],
    history_summary: Optional[str],
    workspace_runtime: Optional[Dict[str, Any]],
    tool_executor,
    discourse_focus: Optional[Dict[str, Any]] = None,
    user_message: str = "",
    active_plugin: Optional[str] = None,
    rag_context: Optional[List[Dict[str, Any]]] = None,
    rag_prompt_block: Optional[str] = None,
    multimodal_enabled: bool = True,
    memory_prompt_block: Optional[str] = None,
    reply_language_hint: Optional[str] = None,
    host_blocks: Optional[Dict[str, str]] = None,
) -> tuple:
    """Recompute selected tools + OpenAI schema + env details after a mid-turn
    file attach (e.g. after a successful ``session_attach``).

    Returns ``(selected_tools, openai_tools, env_details, system_prompt)``.
    """
    selected_tools = select_llm_tools_for_turn(
        tools_manifest,
        user_message=user_message or "",
        session_intent=None,
        current_file_context=file_context,
        session_file_registry=file_registry,
        discourse_focus=discourse_focus,
        active_plugin=active_plugin,
    )
    openai_tools = tool_executor.get_openai_tools_format(selected_tools)
    # ``_build_environment_details`` is an async helper in cline_engine.
    # We can't `await` here (the caller is sync); however the function is
    # cheap and side-effect-free, so we use ``asyncio.get_event_loop().run_until_complete``
    # via a tiny sync wrapper. Cleaner: replicate the body directly — it is
    # already lightweight in cline_engine.
    env_details = _build_env_details_sync(
        session_id=session_id,
        file_registry=file_registry,
        pinned_facts=pinned_facts,
        history_summary=history_summary,
    )
    system_prompt = build_native_system_prompt(
        tools=openai_tools,
        env_details=env_details,
        workspace_runtime=workspace_runtime,
        multimodal_enabled=multimodal_enabled,
        active_plugin=active_plugin,
        rag_context=rag_context,
        rag_prompt_block=rag_prompt_block,
        memory_prompt_block=memory_prompt_block,
        reply_language_hint=reply_language_hint,
        host_blocks=host_blocks,
    )
    return selected_tools, openai_tools, env_details, system_prompt


def _build_env_details_sync(
    *,
    session_id: str,
    file_registry: Optional[List[Dict[str, Any]]],
    pinned_facts: Optional[Dict[str, Any]],
    history_summary: Optional[str],
) -> Dict[str, Any]:
    """Synchronous replica of ``cline_engine._build_environment_details``.

    The original is declared ``async`` purely for API symmetry; its body is
    pure data formatting. Duplicating it here lets the workspace refresh path
    stay synchronous and avoids a needless event-loop ping.
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


def _coerce_tool_args(raw: Any) -> Dict[str, Any]:
    """Best-effort conversion of a streaming-collected ``arguments`` payload
    into a Python dict.

    The vLLM client returns ``tool_calls[i]["arguments"]`` already JSON-decoded
    (it calls ``json.loads`` inside ``chat_stream_with_tools``), but defensive
    coercion is cheap and lets us survive a model emitting an empty string
    or a stray markdown fence.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _prepare_tool_args(tool_name: str, raw_args: Any) -> Dict[str, Any]:
    from ..tools.tool_arg_repair import repair_tool_arguments

    return repair_tool_arguments(tool_name, _coerce_tool_args(raw_args))


# ---------------------------------------------------------------------------
# Main streaming entrypoint
# ---------------------------------------------------------------------------


async def stream_agent_native(
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
    workspace_runtime: Optional[Dict[str, Any]] = None,
    discourse_focus: Optional[Dict[str, Any]] = None,
    task_checklist_state: Optional[Dict[str, Any]] = None,
    rule_policies: Optional[Dict[str, Any]] = None,
    active_plugin: Optional[str] = None,
    host_page_prompt: Optional[str] = None,
    rag_context: Optional[List[Dict[str, Any]]] = None,
    rag_prompt_block: str = "",
    memory_prompt_block: str = "",
    orchestrator_config: Optional[Dict[str, Any]] = None,
    self_compact: Optional[Dict[str, Any]] = None,
    **_kwargs: Any,
) -> AsyncGenerator[Dict[str, Any], None]:
    """SSE-shaped generator (mirrors ``stream_agent_cline``).

    Yields the same event shapes as the Cline engine so the frontend does
    not have to branch on the engine flavour. Engine label on
    ``start``/``done`` is ``native_tools``.

    ``orchestrator_config`` is forwarded from hub chat (settings_ui). When it
    contains a non-empty ``tool_ids`` list, that allowlist replaces the default
    turn selection (still intersected with RBAC-filtered ``tools_manifest``).
    Extra kwargs are ignored for forward-compat with ``main.py`` / RLM callers.
    """
    host_blocks = {
        key[: -len("_prompt_block")]: value.strip()
        for key, value in _kwargs.items()
        if key.endswith("_prompt_block") and isinstance(value, str) and value.strip()
    }
    del _kwargs  # accept hub kwargs without failing (e.g. future fields)
    trace_id = str(uuid.uuid4())
    effective_max_steps = int(max_steps if max_steps is not None else settings.MAX_STEPS)
    t_engine_start = _time.monotonic()
    step_timings: List[Dict[str, Any]] = []
    rag_context = list(rag_context or [])
    rag_prompt_block = (rag_prompt_block or "").strip()
    memory_prompt_block = (memory_prompt_block or "").strip()
    reply_language_hint = _reply_language_hint(user_message)
    host_page_prompt = (host_page_prompt or "").strip()
    user_cfg = (
        dict(orchestrator_config)
        if isinstance(orchestrator_config, dict)
        else {}
    )

    yield {
        "type": "start",
        "trace_id": trace_id,
        "session_id": session_id,
        "engine": "native_tools",
        "timestamp": _dt.datetime.utcnow().isoformat(),
    }

    llm_client = get_gateway_llm_client()
    tool_executor = get_tool_executor()
    accepts_images = await _model_accepts_images(llm_client)

    from ..discourse_focus import DiscourseFocus, update_focus_from_tool_run
    from ..context.compaction import maybe_compact_messages
    from ..context.self_compact import SelfCompactController

    try:
        self_compactor = SelfCompactController.for_turn(
            user_cfg, llm_client=llm_client, carried=self_compact
        )
    except Exception as exc:  # noqa: BLE001 — bad override disables it for this turn only
        log.warning("self_compact disabled for this turn: %s", exc)
        self_compactor = None
    if self_compactor is not None and settings.SELF_COMPACT_SUMMARY_EXCLUDE_RECALLED:
        self_compactor.exclude_from_summary(memory_prompt_block, rag_prompt_block, *host_blocks.values())
    if self_compactor is not None and isinstance(self_compact, dict):
        # The trajectory after the last checkpoint replaces the recent-window
        # history; the checkpoint summary supersedes the lossy history_summary.
        chat_history = list(self_compact.get("messages") or [])
        history_summary = ""
    from ..context.transcript_search import build_transcript_search_hint
    from ..task_checklist import build_task_checklist_block, update_checklist_from_tool_run
    from ..repetition import render_anti_loop_block

    current_discourse_focus = DiscourseFocus.from_dict(discourse_focus)
    task_checklist_state: Dict[str, Any] = dict(task_checklist_state or {})

    if current_discourse_focus.disambiguation_candidates:
        yield {
            "type": "file_disambiguation",
            "session_id": session_id,
            "candidates": current_discourse_focus.disambiguation_candidates,
            "message": (
                "Multiple session files match «this/این». "
                "Ask the user to pick one or use the file picker."
            ),
        }

    effective_user_message = user_message
    confirmed_artifact_id = _pending_confirmed_artifact_id(user_message, chat_history)
    direct_attach_artifact_id = _explicit_or_confirmed_attach_artifact_id(user_message, chat_history)
    if confirmed_artifact_id:
        effective_user_message = (
            f"{user_message}\n\n"
            "[Confirmed prior selection]\n"
            f"The user is confirming the previously proposed artifact_id `{confirmed_artifact_id}`.\n"
            "If you still need to import that file into the session, call `session_attach` with this exact `artifact_id` now."
        )

    # ------------------------------------------------------------------
    # File context detection (fresh upload or auto-resolved attachment)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Tools manifest + OpenAI schema
    # ------------------------------------------------------------------
    try:
        tools_manifest = await tool_executor.get_tools_manifest()
    except Exception as exc:
        log.error("Failed to fetch tools manifest: %s", exc, exc_info=True)
        tools_manifest = []

    # Filter the manifest so the model only sees tools the user may use.
    # Execution enforcement is authoritative in tool-runner (authorize_tool_execute).
    tools_manifest = await resolve_allowed_tools(tools_manifest, user_id)

    # Mutable copy — grows when ``session_attach`` links DB files mid-turn.
    workspace_registry: List[Dict[str, Any]] = list(file_registry or [])
    from ..session_intent import (
        extract_graph_queries,
        infer_session_intent,
        meaningful_graph_queries,
    )

    session_intent = infer_session_intent(
        user_message,
        file_registry=workspace_registry,
        discourse_focus=current_discourse_focus.to_dict(),
    )
    preattached_run: Optional[Dict[str, Any]] = None
    preread_run: Optional[Dict[str, Any]] = None
    if direct_attach_artifact_id:
        preattach_result = await tool_executor.execute(
            "session_attach",
            {"artifact_id": direct_attach_artifact_id},
            session_id=session_id,
            user_id=user_id,
        )
        if preattach_result.get("ok"):
            attach_file = _extract_attach_file_from_result(preattach_result)
            if attach_file:
                reg_row = _registry_entry_from_attach(attach_file)
                _append_registry_entry(workspace_registry, reg_row)
                if _is_visual_registry_entry(reg_row):
                    exact_context = _detect_file_context_from_registry([reg_row])
                    if exact_context:
                        active_file_context = exact_context
            effective_user_message = (
                f"{user_message}\n\n"
                f"[Resolved before this turn]\nThe selected artifact_id `{direct_attach_artifact_id}` "
                "has already been attached to the current session workspace."
            )
            preattached_run = {
                "id": "preloop_session_attach",
                "tool_id": "session_attach",
                "args": {"artifact_id": direct_attach_artifact_id},
                "ok": True,
                "result": preattach_result,
                "text_for_llm": _extract_llm_summary(preattach_result),
                "summary": _cap_llm_text(_extract_llm_summary(preattach_result)),
                "step": 0,
            }

    preloop_graph_run: Optional[Dict[str, Any]] = None
    graph_queries: List[str] = []
    if session_intent == "graph":
        graph_queries = extract_graph_queries(
            user_message,
            chat_history=chat_history,
            discourse_focus=current_discourse_focus.to_dict(),
        )
        if not meaningful_graph_queries(graph_queries):
            graph_queries = []
        if not graph_queries:
            graph_queries = [user_message.strip()[:80] or "topic"]
            if not meaningful_graph_queries(graph_queries):
                graph_queries = []
    if session_intent == "graph" and graph_queries:
        graph_args = {
            "queries": graph_queries,
            "question": user_message,
        }
        graph_result = await tool_executor.execute(
            "graph_search",
            graph_args,
            session_id=session_id,
            user_id=user_id,
        )
        graph_ok = bool(graph_result.get("ok"))
        graph_summary = _extract_llm_summary(graph_result)
        preloop_graph_run = {
            "id": f"preloop_graph_search_{uuid.uuid4().hex[:12]}",
            "tool_id": "graph_search",
            "args": graph_args,
            "ok": graph_ok,
            "result": graph_result,
            "text_for_llm": graph_summary,
            "summary": _cap_llm_text((graph_summary or "")),
            "step": 0,
        }
        effective_user_message = (
            f"{user_message}\n\n"
            "[Graph search — pre-executed]\n"
            f"graph_search already ran with queries={graph_queries!r}. "
            "Use its result below to answer."
        )
        log.info(
            "preloop graph_search: ok=%s queries=%s",
            graph_ok,
            graph_queries,
        )

    preloop_files_run: Optional[Dict[str, Any]] = None
    if (
        session_intent not in ("graph", "attach")
        and user_wants_file_read(user_message)
        and not user_wants_file_discovery(user_message)
        and not _primary_registry_file_for_read(workspace_registry)
    ):
        files_result = await tool_executor.execute(
            "session_files",
            {"session_id": session_id, "limit": 100, "offset": 0},
            session_id=session_id,
            user_id=user_id,
        )
        files_ok = bool(files_result.get("ok"))
        files_payload = _tool_result_payload(files_result)
        files_list = files_payload.get("files") if isinstance(files_payload.get("files"), list) else []
        for f in files_list:
            if not isinstance(f, dict):
                continue
            fid = f.get("id")
            workspace_registry = _append_registry_entry(
                workspace_registry,
                {
                    "artifact_id": fid,
                    "id": fid,
                    "name": f.get("name") or f.get("filename") or "unnamed",
                    "path": f.get("path") or f.get("storage_path"),
                    "mime_type": f.get("mime_type") or "",
                    "type": f.get("type") or "file",
                },
            )
        summary_text = _extract_llm_summary(files_result)
        preloop_files_run = {
            "id": f"preloop_session_files_{uuid.uuid4().hex[:12]}",
            "tool_id": "session_files",
            "args": {"session_id": session_id, "limit": 100, "offset": 0},
            "ok": files_ok,
            "result": files_result,
            "text_for_llm": summary_text,
            "summary": _cap_llm_text((summary_text or "")),
            "step": 0,
        }
        if not files_list:
            effective_user_message = (
                f"{user_message}\n\n"
                "[Session workspace]\n"
                "This session has **no uploaded files** (session_files returned empty).\n"
                "Do NOT invent filenames (e.g. tet.txt). Tell the user no file is available "
                "to read, or ask them to upload/attach a file first."
            )
        elif len(files_list) == 1:
            only = files_list[0]
            effective_user_message = (
                f"{user_message}\n\n"
                "[Session workspace]\n"
                f"Exactly one file in session: `{only.get('name')}` id=`{only.get('id')}`.\n"
                "Use session_read_file with this file_id (or filename) — do not guess other names."
            )
        else:
            names = [
                f"`{f.get('name')}` id=`{f.get('id')}`"
                for f in files_list[:8]
                if isinstance(f, dict)
            ]
            effective_user_message = (
                f"{user_message}\n\n"
                "[Session workspace]\n"
                f"{len(files_list)} files in session. Pick the correct file_id from session_files output:\n"
                + "\n".join(f"- {n}" for n in names)
            )
        log.info(
            "preloop session_files: ok=%s count=%d registry_now=%d",
            files_ok,
            len(files_list),
            len(workspace_registry or []),
        )

    read_target = _primary_registry_file_for_read(workspace_registry)
    if should_preloop_session_read(user_message, chat_history) and read_target and not preread_run:
        explicit: Dict[str, Any] = {}
        artifact_id = read_target.get("artifact_id") or read_target.get("id") or read_target.get("file_id")
        if artifact_id:
            explicit["file_id"] = str(artifact_id)
        else:
            name = read_target.get("name") or read_target.get("filename")
            if name:
                explicit["filename"] = str(name)
        read_args = resolve_read_file_args(
            user_message=user_message,
            chat_history=chat_history,
            file_registry=workspace_registry,
            discourse_focus=(
                current_discourse_focus.to_dict()
                if not current_discourse_focus.is_empty()
                else None
            ),
            explicit_args=explicit,
        )
        if read_args:
            read_result = await tool_executor.execute(
                "session_read_file",
                read_args,
                session_id=session_id,
                user_id=user_id,
                discourse_focus=current_discourse_focus.to_dict(),
                file_registry=workspace_registry,
            )
            target_label = read_args.get("filename") or read_args.get("file_id") or "file"
            line_range = {}
            if read_args.get("start_line") is not None:
                line_range = {
                    "start_line": read_args["start_line"],
                    "end_line": read_args.get("end_line"),
                }
            range_hint = ""
            if line_range.get("start_line") is not None:
                range_hint = (
                    f" (lines {line_range['start_line']}-{line_range.get('end_line', '?')})"
                )
            if read_result.get("ok"):
                effective_user_message = (
                    f"{user_message}\n\n"
                    f"[Resolved before this turn]\n"
                    f"`session_read_file` already returned `{target_label}`{range_hint}. "
                    "Answer from the tool result message below. "
                    "Call `session_read_file` again only for a different line range or file."
                )
                preread_run = {
                    "id": f"preloop_session_read_file_{uuid.uuid4().hex[:12]}",
                    "tool_id": "session_read_file",
                    "args": read_args,
                    "ok": True,
                    "result": read_result,
                    "text_for_llm": _extract_llm_summary(read_result),
                    "summary": _cap_llm_text(_extract_llm_summary(read_result)),
                    "step": 0,
                }
                log.info(
                    "preloop_read_ok tool=session_read_file file=%s range=%s",
                    target_label,
                    line_range or "default",
                )
            else:
                log.warning(
                    "preloop session_read_file failed: %s args=%s",
                    read_result.get("error"),
                    read_args,
                )

    preloop_case_attach_run: Optional[Dict[str, Any]] = None
    if user_wants_case_lookup(user_message):
        attach_hint = extract_attach_filename_hint(user_message)
        if attach_hint:
            attach_args = {"filename": attach_hint}
            attach_result = await tool_executor.execute(
                "session_attach",
                attach_args,
                session_id=session_id,
                user_id=user_id,
            )
            attach_result = await tool_executor.ensure_session_attach_recovered(
                attach_result,
                tool_args={**attach_args, "session_id": session_id},
                session_id=session_id,
                user_id=user_id,
            )
            attach_summary = _extract_llm_summary(attach_result)
            preloop_case_attach_run = {
                "id": f"preloop_session_attach_{uuid.uuid4().hex[:12]}",
                "tool_id": "session_attach",
                "args": attach_args,
                "ok": bool(attach_result.get("ok")),
                "result": attach_result,
                "text_for_llm": attach_summary,
                "summary": _cap_llm_text((attach_summary or "")),
                "step": 0,
            }
            log.info(
                "preloop session_attach (case lookup): ok=%s hint=%s",
                attach_result.get("ok"),
                attach_hint,
            )

    env_details = await _build_environment_details(
        session_id=session_id,
        file_registry=workspace_registry,
        pinned_facts=pinned_facts,
        history_summary=history_summary,
    )

    override_ids = [
        str(x).strip()
        for x in (user_cfg.get("tool_ids") or [])
        if str(x).strip()
    ] if isinstance(user_cfg.get("tool_ids"), list) else []
    if override_ids:
        by_id = {
            str(t.get("id") or t.get("name") or "").strip(): t
            for t in (tools_manifest or [])
            if str(t.get("id") or t.get("name") or "").strip()
        }
        # Preserve client order; drop ids not in RBAC-filtered manifest or not on this page.
        selected_tools = [
            by_id[tid]
            for tid in override_ids
            if tid in by_id and page_allows_tool(by_id[tid], active_plugin)
        ]
        if not selected_tools:
            log.warning(
                "orchestrator_config.tool_ids=%s matched none of allowed tools; "
                "falling back to select_llm_tools_for_turn",
                override_ids,
            )
            selected_tools = select_llm_tools_for_turn(
                tools_manifest,
                user_message=user_message,
                session_intent=session_intent,
                current_file_context=active_file_context,
                session_file_registry=workspace_registry,
                discourse_focus=current_discourse_focus.to_dict(),
                active_plugin=active_plugin,
            )
        else:
            log.info(
                "orchestrator_config.tool_ids override: %s → %d tools",
                override_ids,
                len(selected_tools),
            )
    else:
        selected_tools = select_llm_tools_for_turn(
            tools_manifest,
            user_message=user_message,
            session_intent=session_intent,
            current_file_context=active_file_context,
            session_file_registry=workspace_registry,
            discourse_focus=current_discourse_focus.to_dict(),
            active_plugin=active_plugin,
        )
    openai_tools = await _openai_tools_for_selected(tool_executor, selected_tools)
    log.info("session_intent=%s tools_selected=%d", session_intent, len(selected_tools))
    log.info(
        "📦 Native tools selected for LLM: %s/%s %s",
        len(selected_tools),
        len(tools_manifest or []),
        [t.get("id") or t.get("name") for t in selected_tools],
    )
    log.debug("📦 Native tool selection detail: %s", selection_debug_summary(selected_tools))

    transcript_hint = build_transcript_search_hint(
        session_id=session_id,
        query=user_message,
        chat_history=chat_history,
    )
    _prompt_extras: List[str] = []
    if host_page_prompt:
        _prompt_extras.append(host_page_prompt)
    if len(chat_history or []) > 6 and transcript_hint:
        _prompt_extras.append(transcript_hint)
    _checklist = build_task_checklist_block(
        user_message=user_message,
        pinned_facts=pinned_facts,
        discourse_focus=current_discourse_focus.to_dict(),
        checklist_state=task_checklist_state,
    )
    if _checklist:
        _prompt_extras.append(_checklist)
    _anti_loop = render_anti_loop_block(conversation_state)
    if _anti_loop:
        _prompt_extras.append(_anti_loop)
    checklist_and_rules = "\n\n".join(x for x in _prompt_extras if x)
    if self_compactor is not None and settings.SELF_COMPACT_SUMMARY_EXCLUDE_RECALLED:
        _pinned_env = ((env_details or {}).get("extras") or {}).get("pinned_facts")
        self_compactor.exclude_from_summary(
            _checklist, f"- pinned_facts: {_pinned_env}" if _pinned_env else ""
        )

    system_prompt = build_native_system_prompt(
        tools=openai_tools,
        env_details=env_details,
        workspace_runtime=workspace_runtime,
        multimodal_enabled=accepts_images,
        extra_rules=checklist_and_rules or None,
        active_plugin=active_plugin,
        rag_context=rag_context,
        rag_prompt_block=rag_prompt_block,
        memory_prompt_block=memory_prompt_block,
        reply_language_hint=reply_language_hint,
        host_blocks=host_blocks,
    )

    # ------------------------------------------------------------------
    # Build the message stack
    # ------------------------------------------------------------------
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(_normalize_history_for_native(chat_history))

    # Current user message — possibly multimodal.
    user_content: Any = effective_user_message
    injected_multimodal_keys: set[str] = set()
    if active_file_context and accepts_images:
        try:
            user_content = await build_multimodal_content(
                effective_user_message,
                active_file_context,
                auth_context={"user_id": user_id},
            )
            if isinstance(user_content, list):
                injected_multimodal_keys = _visual_file_context_keys(active_file_context)
                yield {"type": "vision_attached", "parts": len(user_content)}
        except Exception as exc:
            log.warning("build_multimodal_content failed: %s", exc)
            user_content = effective_user_message
    if accepts_images and _visual_attachment_missing(active_file_context, injected_multimodal_keys):
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
    messages.append({"role": "user", "content": _with_reply_language_note(user_content, user_message)})
    current_user_msg = messages[-1]
    if self_compactor is not None:
        messages = self_compactor.attach_summary(messages)

    if preattached_run:
        _inject_preloop_tool_into_messages(messages, preattached_run)
    if preloop_graph_run:
        _inject_preloop_tool_into_messages(messages, preloop_graph_run)
    if preloop_files_run:
        _inject_preloop_tool_into_messages(messages, preloop_files_run)
    if preread_run:
        _inject_preloop_tool_into_messages(messages, preread_run)
    if preloop_case_attach_run:
        _inject_preloop_tool_into_messages(messages, preloop_case_attach_run)

    # ------------------------------------------------------------------
    # Loop state
    # ------------------------------------------------------------------
    accumulated_tool_runs: List[Dict[str, Any]] = []
    evidence_ledger: List[Dict[str, Any]] = []
    final_answer: Optional[str] = None
    final_assistant_text: str = ""
    failed_call_counts: Dict[str, int] = {}
    last_failure_summary: Optional[str] = None
    empty_answer_retry_count = 0
    repair_round_no_thinking = False
    malformed_tool_blob_retry_count = 0
    loop_blocked_repair_count = 0
    force_text_reply_retry_count = 0
    consecutive_mistake_count = 0
    force_text_only_next_turn = False
    runtime_first_requested = bool((workspace_runtime or {}).get("prefer_runtime_first"))
    if preattached_run:
        accumulated_tool_runs.append(preattached_run)
        for ev in _emit_preloop_tool_events(preattached_run):
            yield ev
    if preloop_graph_run:
        accumulated_tool_runs.append(preloop_graph_run)
        for ev in _emit_preloop_tool_events(preloop_graph_run):
            yield ev
    if preloop_files_run:
        accumulated_tool_runs.append(preloop_files_run)
        for ev in _emit_preloop_tool_events(preloop_files_run):
            yield ev
    if preread_run:
        accumulated_tool_runs.append(preread_run)
        for ev in _emit_preloop_tool_events(preread_run):
            yield ev
    if preloop_case_attach_run:
        accumulated_tool_runs.append(preloop_case_attach_run)
        for ev in _emit_preloop_tool_events(preloop_case_attach_run):
            yield ev

    # Ground evidence.
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

    if workspace_runtime:
        runtime_selected = workspace_runtime.get("selected_files") or []
        runtime_snippets = workspace_runtime.get("text_snippets") or []
        runtime_rec = _evidence_record(
            step=0,
            phase="attach",
            kind="workspace_runtime",
            summary=(
                "Prepared passive workspace runtime with "
                f"{len(runtime_selected)} resolved file target(s) and "
                f"{len(runtime_snippets)} inline text snippet(s)."
            ),
            files=[
                str(item.get("name") or item.get("path") or "")
                for item in runtime_selected[:3]
                if isinstance(item, dict)
            ],
            meta={"runtime_context": runtime_first_requested},
        )
        evidence_ledger.append(runtime_rec)
        yield {"type": "evidence_step", **runtime_rec}
    if confirmed_artifact_id:
        confirm_rec = _evidence_record(
            step=0,
            phase="attach",
            kind="confirmed_artifact_selection",
            summary=f"Resolved affirmative follow-up to artifact_id `{confirmed_artifact_id}` from recent assistant context.",
            files=[confirmed_artifact_id],
        )
        evidence_ledger.append(confirm_rec)
        yield {"type": "evidence_step", **confirm_rec}
    if preattached_run:
        preattach_rec = _evidence_record(
            step=0,
            phase="attach",
            kind="preloop_session_attach",
            summary=(
                f"Executed `session_attach` before the LLM round using explicit artifact_id "
                f"`{direct_attach_artifact_id}` from the current turn."
            ),
            files=[direct_attach_artifact_id] if direct_attach_artifact_id else [],
        )
        evidence_ledger.append(preattach_rec)
        yield {"type": "evidence_step", **preattach_rec}
    if preread_run:
        preread_rec = _evidence_record(
            step=0,
            phase="attach",
            kind="preloop_session_read_file",
            summary="Executed `session_read_file` before the LLM round because the user asked to read a session file.",
            files=[
                str(preread_run.get("args", {}).get("file_id") or preread_run.get("args", {}).get("filename") or "")
            ],
        )
        evidence_ledger.append(preread_rec)
        yield {"type": "evidence_step", **preread_rec}

    # ------------------------------------------------------------------
    # ReAct loop — but each LLM round may decide N tools at once.
    # ------------------------------------------------------------------
    from ..guardrails import MistakeBudget

    tool_loop_guard = ToolLoopGuard(block_at=2)
    mistake_budget = MistakeBudget(max_mistakes=2)
    conversation_summary_checkpoint: Optional[str] = None
    session_workspace_confirmed_empty = False
    _SESSION_FILES_EMPTY_BLOCK_MSG = (
        "[WORKSPACE_EMPTY] session_files already confirmed this chat workspace has zero files. "
        "Do NOT call session_files again. For files in storage/database use session_attach "
        "(filename / case_id / path). Tell the user the chat workspace is empty if they "
        "expected an uploaded file."
    )
    step = 0
    while step < effective_max_steps:
        step += 1
        step_started_at = _time.monotonic()
        step_llm_ms = 0.0
        step_tool_ms = 0.0
        runtime_first_this_step = False  # telemetry only; tools stay enabled (Continue-style)

        plan_rec = _evidence_record(
            step=step,
            phase="plan",
            kind="llm_turn",
            summary=f"Starting LLM reasoning round {step}.",
            meta={
                "tools_so_far": len(accumulated_tool_runs),
                "max_steps": effective_max_steps,
                "tools_available": len(openai_tools or []),
                "runtime_context": runtime_first_requested,
            },
        )
        evidence_ledger.append(plan_rec)
        yield {"type": "evidence_step", **plan_rec}

        collected_text = ""
        answer_emitted = False
        collected_tool_calls: List[Dict[str, Any]] = []
        seen_tool_names: Dict[int, str] = {}
        forcing_text_only_this_step = force_text_only_next_turn
        turn_tools = None if forcing_text_only_this_step else openai_tools
        is_last_step = step >= effective_max_steps
        turn_tool_choice = (
            "none"
            if forcing_text_only_this_step or not openai_tools or is_last_step
            else "auto"
        )
        force_text_only_next_turn = False

        # Context hygiene before each LLM round.
        if self_compactor is not None:
            messages, sc_events = await self_compactor.before_round(
                messages, tools=turn_tools, anchor=current_user_msg, step=step
            )
            for sc_ev in sc_events:
                yield sc_ev
            yield {
                "type": "context_usage",
                "step": step,
                **self_compactor.usage(messages, turn_tools),
            }
        else:
            tools_json_len = len(str(turn_tools or openai_tools or []))
            messages = compile_messages_continue_style(
                messages,
                char_budget=int(settings.STREAM_CONTEXT_CHAR_BUDGET),
                tools_extra_chars=tools_json_len // 2,
            )
            messages, compact_summary, ctx_telem = await maybe_compact_messages(
                messages,
                llm_client=llm_client,
                openai_tools=turn_tools or openai_tools,
                active_focus=current_discourse_focus.to_dict(),
                threshold_pct=80.0,
            )
            if compact_summary:
                conversation_summary_checkpoint = compact_summary
                yield {
                    "type": "context_compacted",
                    "step": step,
                    "summary_chars": len(compact_summary),
                }
            yield {
                "type": "context_usage",
                "step": step,
                "context_usage_pct": ctx_telem.get("context_usage_pct"),
                "context_chars": ctx_telem.get("chars"),
                "context_budget": ctx_telem.get("budget"),
            }

        yield {"type": "llm_start", "step": step}
        t_llm_start = _time.monotonic()
        # When the model dumps `{"tool_calls":[...]}` as bare content (no native
        # channel + no fence), we still see it streamed as `answer` chunks. We
        # buffer them and only flush to the UI once we are SURE the content
        # isn't tool-call JSON — otherwise the user sees a raw JSON blob
        # streaming in instead of prose.
        pending_answer_buffer = ""
        answer_flushed = False  # True once we decide the content is real prose
        llm_turn_aborted = False
        overflow_error: Optional[str] = None
        no_thinking_this_round, repair_round_no_thinking = repair_round_no_thinking, False

        try:
            async for ev in llm_client.chat_stream_with_tools(
                messages=messages,
                tools=turn_tools,
                tool_choice=turn_tool_choice,
                max_tokens=(
                    self_compactor.completion_budget(messages, turn_tools)
                    if self_compactor is not None else None
                ),
                **(
                    {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
                    if no_thinking_this_round else {}
                ),
            ):
                etype = ev.get("type")
                if etype == "tool_call_start":
                    idx = ev.get("index", 0)
                    tname = ev.get("tool_name") or ""
                    seen_tool_names[idx] = tname
                    turn_offered = _offered_tool_names(openai_tools)
                    if turn_offered and tname not in turn_offered:
                        continue
                    yield {
                        "type": "tool_call_start",
                        "step": step,
                        "index": idx,
                        "tool_id": ev.get("tool_id") or f"call_{step}_{idx}",
                        "tool_name": tname,
                    }
                elif etype == "tool_call_delta":
                    # Streamed arg deltas — keep the UI alive but do not
                    # forward by default (frontend just shows "calling X").
                    continue
                elif etype == "tool_calls":
                    # Final aggregated list, ready to execute.
                    collected_tool_calls = list(ev.get("tool_calls") or [])
                elif etype == "answer":
                    chunk = ev.get("content", "") or ""
                    if not chunk:
                        continue
                    collected_text += chunk
                    # Lazily decide whether to forward. If the running text
                    # looks like the model is dumping a tool_calls JSON blob
                    # (very common for Qwen3.5 when the chat template doesn't
                    # surface `<tool_call>` wrappers), we keep it buffered and
                    # never emit it — the recovery layer will pull the calls
                    # out, and the user shouldn't see the JSON.
                    if answer_flushed:
                        yield {"type": "answer", "content": chunk}
                        continue
                    pending_answer_buffer += chunk
                    stripped = pending_answer_buffer.lstrip()
                    # Heuristic: if the buffer (after whitespace) starts with
                    # `{` or `[` and we already see a tool-shaped key, keep
                    # buffering silently. Otherwise the content is real prose
                    # and we can start flushing — including everything buffered
                    # so far.
                    if stripped.startswith("{") or stripped.startswith("["):
                        # Looks like JSON. Keep buffering — we'll decide at
                        # stream end whether tool_calls were recovered.
                        continue
                    # Real prose — flush and switch to passthrough mode.
                    if not answer_emitted:
                        yield {"type": "answer_start", "step": step}
                        answer_emitted = True
                    yield {"type": "answer", "content": pending_answer_buffer}
                    pending_answer_buffer = ""
                    answer_flushed = True
                elif etype in ("thinking_start", "thinking", "thinking_end"):
                    # Reasoning is private scratchpad — never forward.
                    continue
                elif etype == "repetition_aborted":
                    # Legacy event — chat streams no longer abort on repetition.
                    # Keep accumulated answer text for the user.
                    continue
                elif etype == "error":
                    raw_msg = ev.get("content") or "llm_error"
                    if (
                        self_compactor is not None
                        and not answer_emitted
                        and self_compactor.is_overflow(raw_msg)
                        and self_compactor.can_retry()
                    ):
                        overflow_error = str(raw_msg)
                        break
                    if raw_msg == "chat_model_unavailable":
                        log.error(
                            "Chat model unavailable for users — configure "
                            "service.orchestrator.chat via super-admin LLM routes"
                        )
                    msg = _user_facing_llm_error(raw_msg)
                    final_answer = msg
                    llm_turn_aborted = True
                    log.warning("Native LLM stream error: %s", raw_msg)
                    yield {"type": "error", "message": msg, "step": step}
                    if not answer_emitted:
                        yield {"type": "answer_start", "step": step}
                        answer_emitted = True
                    yield {"type": "answer", "content": msg}
                    break
                # "done" / unknown → ignore; await stream exit.
        except Exception as exc:
            if (
                self_compactor is not None
                and not answer_emitted
                and self_compactor.is_overflow(str(exc))
                and self_compactor.can_retry()
            ):
                overflow_error = str(exc)
            else:
                log.error("Native LLM streaming failed step=%d: %s", step, exc, exc_info=True)
                msg = _user_facing_llm_error(str(exc))
                final_answer = msg
                llm_turn_aborted = True
                yield {"type": "error", "message": msg, "step": step}
                if not answer_emitted:
                    yield {"type": "answer_start", "step": step}
                    answer_emitted = True
                yield {"type": "answer", "content": msg}

        if overflow_error is not None:
            log.warning("Native LLM context overflow step=%d — compacting and retrying", step)
            messages, sc_ev = await self_compactor.recover_overflow(
                messages,
                tools=turn_tools,
                anchor=current_user_msg,
                step=step,
                error_text=overflow_error,
            )
            yield sc_ev
            force_text_only_next_turn = forcing_text_only_this_step
            step -= 1
            continue

        step_llm_ms = (_time.monotonic() - t_llm_start) * 1000.0

        if llm_turn_aborted:
            step_total_ms = (_time.monotonic() - step_started_at) * 1000.0
            step_timings.append(
                {
                    "step": step,
                    "llm_ms": round(step_llm_ms, 1),
                    "tool_ms": 0.0,
                    "total_ms": round(step_total_ms, 1),
                    "tools_executed": 0,
                    "repair": "llm_error_abort",
                }
            )
            break

        # Decide the fate of any buffered "looks-like-JSON" answer chunks.
        if pending_answer_buffer:
            if collected_tool_calls:
                # The recovery layer pulled tool_calls out of the buffered
                # JSON — suppress the raw text so the UI doesn't see the blob.
                collected_text = collected_text[: len(collected_text) - len(pending_answer_buffer)]
                pending_answer_buffer = ""
            elif _looks_like_orphan_tool_args_blob(pending_answer_buffer, openai_tools):
                # Keep malformed argument-only JSON hidden from the UI. The
                # no-tool branch below will convert it into a repair retry.
                pass
            else:
                # No tools recovered — the buffered text really is the answer
                # (maybe a JSON literal the user asked for). Flush it now.
                if not answer_emitted:
                    yield {"type": "answer_start", "step": step}
                    answer_emitted = True
                yield {"type": "answer", "content": pending_answer_buffer}
                pending_answer_buffer = ""
                answer_flushed = True

        yield {
            "type": "llm_done",
            "step": step,
            "llm_ms": round(step_llm_ms, 1),
            "tool_calls": len(collected_tool_calls),
        }

        # ----------------------------------------------------------
        # Branch: no tool_calls → final answer (or recovery)
        # ----------------------------------------------------------
        if not collected_tool_calls:
            cleaned = (collected_text or "").strip()
            if (
                cleaned
                and _looks_like_orphan_tool_args_blob(cleaned, openai_tools)
                and malformed_tool_blob_retry_count < 1
                and step < effective_max_steps
            ):
                malformed_tool_blob_retry_count += 1
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "[Malformed Tool Intent]\n"
                            "Your previous assistant turn output only a JSON arguments object without a tool name. "
                            "That is NOT a valid response.\n\n"
                            "If you need a tool, emit a proper native tool call using a name from the API tools list for this turn.\n"
                            "If you do not need a tool, answer the user directly in normal prose.\n"
                            "Do NOT output a standalone JSON object, a fenced ```json block of arguments, or placeholders."
                        ),
                    }
                )
                yield {
                    "type": "warning",
                    "step": step,
                    "message": "Model emitted JSON arguments without a tool call; retrying once with repair instructions.",
                }
                step_total_ms = (_time.monotonic() - step_started_at) * 1000.0
                step_timings.append(
                    {
                        "step": step,
                        "llm_ms": round(step_llm_ms, 1),
                        "tool_ms": 0.0,
                        "total_ms": round(step_total_ms, 1),
                        "tools_executed": 0,
                        "repair": "orphan_tool_args_blob",
                    }
                )
                continue
            if cleaned:
                final_answer = cleaned
                final_assistant_text = collected_text
                messages.append({"role": "assistant", "content": collected_text})
                step_total_ms = (_time.monotonic() - step_started_at) * 1000.0
                step_timings.append(
                    {
                        "step": step,
                        "llm_ms": round(step_llm_ms, 1),
                        "tool_ms": 0.0,
                        "total_ms": round(step_total_ms, 1),
                        "tools_executed": 0,
                    }
                )
                break

            # No tools AND no text — ask the model to repair first; raw tool output is the last resort.
            repair_retries = max(0, int(settings.NATIVE_EMPTY_ANSWER_REPAIR_RETRIES or 0))
            can_repair = (
                empty_answer_retry_count < repair_retries
                and step < effective_max_steps
                and not mistake_budget.exhausted()
            )
            synthesized_early = "" if can_repair else _synthesize_from_tool_evidence(
                accumulated_tool_runs=accumulated_tool_runs,
                messages=messages,
                chat_history=chat_history,
            )
            if synthesized_early:
                final_answer = synthesized_early
                final_assistant_text = synthesized_early
                if not answer_emitted:
                    yield {"type": "answer_start", "step": step}
                yield {"type": "answer", "content": synthesized_early}
                step_total_ms = (_time.monotonic() - step_started_at) * 1000.0
                step_timings.append(
                    {
                        "step": step,
                        "llm_ms": round(step_llm_ms, 1),
                        "tool_ms": round(step_tool_ms, 1),
                        "total_ms": round(step_total_ms, 1),
                        "tools_executed": 0,
                        "repair": "deterministic_tool_synthesis_early",
                    }
                )
                break

            # No tools AND no text — repair once, then synthesize from tool output.
            if mistake_budget.exhausted():
                honest = _synthesize_from_tool_evidence(
                    accumulated_tool_runs=accumulated_tool_runs,
                    messages=messages,
                    chat_history=chat_history,
                )
                if honest:
                    final_answer = honest
                    final_assistant_text = final_answer
                    if not answer_emitted:
                        yield {"type": "answer_start", "step": step}
                    yield {"type": "answer", "content": final_answer}
                    break

            if can_repair:
                empty_answer_retry_count += 1
                repair_round_no_thinking = bool(settings.NATIVE_EMPTY_ANSWER_REPAIR_NO_THINKING)
                mistake_budget.record_mistake()
                digest = _tool_runs_digest_for_llm(accumulated_tool_runs) if accumulated_tool_runs else ""
                latest_user = " ".join(str(user_message or "").split())
                quote_max = int(settings.NATIVE_REPAIR_QUOTE_MAX_CHARS or 0)
                if quote_max > 0 and len(latest_user) > quote_max:
                    latest_user = latest_user[: quote_max - 1].rstrip() + "…"
                messages.append(
                    {
                        "role": "user",
                        "_synthetic": True,
                        "content": (
                            "[No Final Answer Produced]\n"
                            "The previous assistant turn produced no visible user-facing answer.\n\n"
                            "You MUST repair this now.\n"
                            "- If you already have enough information, answer the user directly, in the same "
                            "language as the user's latest message quoted below.\n"
                            "- If the user asked for search, files, attach, read, or database lookup, emit the proper native tool call(s) now.\n"
                            "- Answer only the latest message; earlier tool output answered earlier questions.\n"
                            "- Do not output JSON, code fences, or placeholders.\n"
                            "- Do not output an empty message.\n"
                            f"\nUser's latest message: «{latest_user}»\n"
                            + (
                                "\nThis turn's tool results digest:\n" + digest
                                if digest
                                else "\nNo tools were run in this turn; answer from the conversation above."
                            )
                        ),
                    }
                )
                yield {
                    "type": "warning",
                    "step": step,
                    "message": "Model returned no visible answer; retrying once with repair instructions.",
                }
                step_total_ms = (_time.monotonic() - step_started_at) * 1000.0
                step_timings.append(
                    {
                        "step": step,
                        "llm_ms": round(step_llm_ms, 1),
                        "tool_ms": round(step_tool_ms, 1),
                        "total_ms": round(step_total_ms, 1),
                        "tools_executed": 0,
                    }
                )
                continue

            synthesized = _synthesize_from_tool_evidence(
                accumulated_tool_runs=accumulated_tool_runs,
                messages=messages,
                chat_history=chat_history,
            )
            if synthesized:
                final_answer = synthesized
                final_assistant_text = synthesized
                if not answer_emitted:
                    yield {"type": "answer_start", "step": step}
                yield {"type": "answer", "content": synthesized}
                step_total_ms = (_time.monotonic() - step_started_at) * 1000.0
                step_timings.append(
                    {
                        "step": step,
                        "llm_ms": round(step_llm_ms, 1),
                        "tool_ms": round(step_tool_ms, 1),
                        "total_ms": round(step_total_ms, 1),
                        "tools_executed": 0,
                        "repair": "deterministic_tool_synthesis",
                    }
                )
                break

            if (
                not accumulated_tool_runs
                and force_text_reply_retry_count < 1
                and step < effective_max_steps
                and not mistake_budget.exhausted()
            ):
                force_text_reply_retry_count += 1
                mistake_budget.record_mistake()
                force_text_only_next_turn = True
                synthesized = _deterministic_answer_from_tool_runs(accumulated_tool_runs)
                if synthesized:
                    final_answer = synthesized
                    final_assistant_text = synthesized
                    if not answer_emitted:
                        yield {"type": "answer_start", "step": step}
                    yield {"type": "answer", "content": synthesized}
                    break
                messages.append(
                    {
                        "role": "user",
                        "_synthetic": True,
                        "content": (
                            "Answer the user directly from conversation context and tool results already shown. "
                            "If evidence is incomplete, state what is missing — do not invent content."
                        ),
                    }
                )
                yield {
                    "type": "warning",
                    "step": step,
                    "message": "Model returned empty answer; one text-only synthesis retry.",
                }
                step_total_ms = (_time.monotonic() - step_started_at) * 1000.0
                step_timings.append(
                    {
                        "step": step,
                        "llm_ms": round(step_llm_ms, 1),
                        "tool_ms": round(step_tool_ms, 1),
                        "total_ms": round(step_total_ms, 1),
                        "tools_executed": 0,
                        "repair": "text_synthesis_retry",
                    }
                )
                continue

            final_answer = (
                _synthesize_from_tool_evidence(
                    accumulated_tool_runs=accumulated_tool_runs,
                    messages=messages,
                    chat_history=chat_history,
                )
                or _fallback_final_answer_from_tool_runs(accumulated_tool_runs)
            )
            final_assistant_text = final_answer or collected_text
            if final_answer and not answer_emitted:
                yield {"type": "answer_start", "step": step}
                yield {"type": "answer", "content": final_answer}
            break

        # ----------------------------------------------------------
        # Branch: tool_calls present → execute in parallel
        # ----------------------------------------------------------
        # First, build the OpenAI assistant turn that triggered the calls
        # so the next round has a properly paired assistant+tool sequence.
        oa_tool_calls: List[Dict[str, Any]] = []
        prepared_calls: List[Dict[str, Any]] = []
        for idx, tc in enumerate(collected_tool_calls):
            tname = (tc.get("name") or "").strip()
            if not tname:
                continue
            tid = tc.get("id") or f"call_{step}_{idx}"
            args_obj = _prepare_tool_args(tname, tc.get("arguments"))
            try:
                args_str = json.dumps(args_obj, ensure_ascii=False)
            except Exception:
                args_str = "{}"
            oa_tool_calls.append(
                {
                    "id": tid,
                    "type": "function",
                    "function": {"name": tname, "arguments": args_str},
                }
            )
            prepared_calls.append({"id": tid, "name": tname, "args": args_obj})

        if not prepared_calls:
            # All tool_calls were malformed — treat as no-tools fallback.
            cleaned = (collected_text or "").strip()
            if cleaned:
                final_answer = cleaned
                final_assistant_text = collected_text
                messages.append({"role": "assistant", "content": collected_text})
                break
            final_answer = _fallback_final_answer_from_tool_runs(accumulated_tool_runs)
            final_assistant_text = collected_text
            break

        messages.append(
            {
                "role": "assistant",
                "content": _strip_looping_assistant_content(
                    collected_text or "",
                    has_tool_calls=bool(oa_tool_calls),
                ),
                "tool_calls": oa_tool_calls,
            }
        )

        # Decide active_file_context BEFORE execution — if any single call
        # references a specific session file, pin that context for the round.
        for pc in prepared_calls:
            selected_registry_entry = _match_registry_entry_from_tool_args(
                pc["args"], workspace_registry
            )
            if selected_registry_entry:
                if _is_visual_registry_entry(selected_registry_entry):
                    exact_context = _detect_file_context_from_registry([selected_registry_entry])
                    if exact_context:
                        active_file_context = exact_context
                else:
                    active_file_context = None
                break  # Only pin once — the first explicit reference wins.

        # Emit per-call evidence + UI start events, and filter out repeat-blocked calls.
        runnable_calls: List[Dict[str, Any]] = []
        loop_blocked_this_step = 0
        offered_names = _offered_tool_names(openai_tools)
        for pc in prepared_calls:
            tname = pc["name"]
            targs = pc["args"]
            tid = pc["id"]
            if offered_names and tname not in offered_names:
                blocked_body = _cap_llm_text(_unknown_tool_message(tname, offered_names))
                accumulated_tool_runs.append(
                    {
                        "id": tid,
                        "tool_id": tname,
                        "args": targs,
                        "ok": False,
                        "result": {"ok": False, "error": "unknown_tool"},
                        "text_for_llm": blocked_body,
                        "summary": _cap_llm_text(blocked_body),
                        "step": step,
                        "skipped": True,
                    }
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tid, "name": tname, "content": blocked_body}
                )
                continue
            call_key = _canonical_tool_call_key(tname, targs)
            if failed_call_counts.get(call_key, 0) >= 2:
                blocked_summary = (
                    f"[RETRY_BLOCKED] The exact same tool call `{tname}` with the same arguments "
                    "has already failed twice in this conversation turn. Treat this as no usable "
                    "result for that query; do not call it again. Explain the failure to the user "
                    "or try a meaningfully different tool/filename/file_id."
                )
                last_failure_summary = blocked_summary
                blocked_body = _cap_llm_text(blocked_summary)
                accumulated_tool_runs.append(
                    {
                        "id": tid,
                        "tool_id": tname,
                        "args": targs,
                        "ok": False,
                        "result": {
                            "ok": False,
                            "error": "repeat_failed_tool_call_blocked",
                            "tool_id": tname,
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
                    tool_id=tid,
                    tool_name=tname,
                    tool_args=targs,
                    ok=False,
                    summary_text=blocked_summary,
                    error="repeat_failed_tool_call_blocked",
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tid,
                        "name": tname,
                        "content": blocked_body,
                    }
                )
                continue

            act_rec = _evidence_record(
                step=step,
                phase="act",
                kind="tool_call",
                summary=f"Calling tool `{tname}`.",
                tool_name=tname,
                args=targs,
                files=_evidence_files_from_tool_args(targs),
            )
            evidence_ledger.append(act_rec)
            yield {"type": "evidence_step", **act_rec}

            yield {
                "type": "tool_start",
                "step": step,
                "tool_id": tid,
                "tool_name": tname,
                "arguments": targs,
                "args": targs,
                "query": _display_query_from_args(targs),
            }
            from ..tools.tool_arg_repair import validate_tool_arguments

            targs = _prepare_tool_args(tname, targs)
            pc["args"] = targs
            validation_err = validate_tool_arguments(tname, targs)
            if validation_err:
                blocked_body = _cap_llm_text(validation_err)
                accumulated_tool_runs.append(
                    {
                        "id": tid,
                        "tool_id": tname,
                        "args": targs,
                        "ok": False,
                        "result": {"ok": False, "error": "invalid_arguments"},
                        "text_for_llm": blocked_body,
                        "summary": _cap_llm_text(blocked_body),
                        "step": step,
                        "skipped": True,
                    }
                )
                yield _tool_result_sse_event(
                    step=step,
                    tool_id=tid,
                    tool_name=tname,
                    tool_args=targs,
                    ok=False,
                    summary_text=blocked_body,
                    error="invalid_arguments",
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tid, "name": tname, "content": blocked_body}
                )
                continue

            if session_workspace_confirmed_empty and tname == "session_files":
                blocked_body = _cap_llm_text(_SESSION_FILES_EMPTY_BLOCK_MSG)
                accumulated_tool_runs.append(
                    {
                        "id": tid,
                        "tool_id": tname,
                        "args": targs,
                        "ok": False,
                        "result": {"ok": False, "error": "workspace_empty"},
                        "text_for_llm": blocked_body,
                        "summary": _cap_llm_text(blocked_body),
                        "step": step,
                        "skipped": True,
                    }
                )
                yield _tool_result_sse_event(
                    step=step,
                    tool_id=tid,
                    tool_name=tname,
                    tool_args=targs,
                    ok=False,
                    summary_text=blocked_body,
                    error="workspace_empty",
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tid, "name": tname, "content": blocked_body}
                )
                continue

            loop_status = tool_loop_guard.record(tname, targs)
            if loop_status == "block":
                blocked_body = _cap_llm_text(tool_loop_guard.blocked_message(tname, targs))
                loop_blocked_this_step += 1
                accumulated_tool_runs.append(
                    {
                        "id": tid,
                        "tool_id": tname,
                        "args": targs,
                        "ok": False,
                        "result": {"ok": False, "error": "loop_blocked"},
                        "text_for_llm": blocked_body,
                        "summary": _cap_llm_text(blocked_body),
                        "step": step,
                        "skipped": True,
                    }
                )
                yield _tool_result_sse_event(
                    step=step,
                    tool_id=tid,
                    tool_name=tname,
                    tool_args=targs,
                    ok=False,
                    summary_text=blocked_body,
                    error="loop_blocked",
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tid, "name": tname, "content": blocked_body}
                )
                continue
            runnable_calls.append(pc)

        if loop_blocked_this_step > 0:
            loop_blocked_repair_count += 1
            _append_user_repair_once(
                messages,
                tool_loop_guard.repair_user_message(effective_user_message),
            )
            force_text_only_next_turn = True
            yield {
                "type": "warning",
                "step": step,
                "message": (
                    f"Blocked {loop_blocked_this_step} repeated tool call(s); "
                    "forcing text-only next turn with repair instructions."
                ),
            }
            if loop_blocked_repair_count >= 2 or step >= effective_max_steps - 1:
                synthesized = _synthesize_from_tool_evidence(
                    accumulated_tool_runs=accumulated_tool_runs,
                    messages=messages,
                    chat_history=chat_history,
                )
                if synthesized:
                    final_answer = synthesized
                    final_assistant_text = synthesized
                    if not answer_emitted:
                        yield {"type": "answer_start", "step": step}
                    yield {"type": "answer", "content": synthesized}
                    step_total_ms = (_time.monotonic() - step_started_at) * 1000.0
                    step_timings.append(
                        {
                            "step": step,
                            "llm_ms": round(step_llm_ms, 1),
                            "tool_ms": round(step_tool_ms, 1),
                            "total_ms": round(step_total_ms, 1),
                            "tools_executed": len(runnable_calls),
                            "repair": "loop_blocked_synthesis",
                        }
                    )
                    break

        # ===== Parallel execution =====
        t_tool_start = _time.monotonic()
        if runnable_calls:
            exec_tasks = [
                tool_executor.execute(
                    pc["name"],
                    pc["args"],
                    file_context=active_file_context,
                    session_id=session_id,
                    user_id=user_id,
                    discourse_focus=current_discourse_focus.to_dict(),
                    file_registry=workspace_registry,
                    user_message=effective_user_message,
                )
                for pc in runnable_calls
            ]
            raw_results = await asyncio.gather(*exec_tasks, return_exceptions=True)
        else:
            raw_results = []
        step_tool_ms += (_time.monotonic() - t_tool_start) * 1000.0

        # Process results in declared order.
        any_session_attach = False
        ambiguous_attach_answer: Optional[str] = None
        for pc, raw in zip(runnable_calls, raw_results):
            tname = pc["name"]
            targs = pc["args"]
            tid = pc["id"]
            call_key = _canonical_tool_call_key(tname, targs)

            if isinstance(raw, Exception):
                log.error("Tool %s execution exception: %s", tname, raw, exc_info=raw)
                result = {
                    "ok": False,
                    "error": f"execution_failed: {raw}",
                    "tool_id": tname,
                }
            else:
                result = raw  # type: ignore[assignment]

            if tname == "session_attach":
                result = await tool_executor.ensure_session_attach_recovered(
                    result,
                    tool_args=targs,
                    session_id=session_id,
                    user_id=user_id,
                )

            ok = bool(result.get("ok"))
            if ok and tname == "session_files":
                payload = _tool_result_payload(result)
                total_files = payload.get("total")
                if total_files is None and isinstance(payload.get("files"), list):
                    total_files = len(payload["files"])
                if (total_files or 0) == 0:
                    off = int(targs.get("offset") or 0)
                    if off <= 0:
                        session_workspace_confirmed_empty = True
            summary_text = _normalized_tool_summary(
                tname, ok, _extract_llm_summary(result)
            )
            if ok:
                failed_call_counts.pop(call_key, None)
                current_discourse_focus = update_focus_from_tool_run(
                    current_discourse_focus,
                    tool_name=tname,
                    tool_args=targs,
                    ok=True,
                    turn=step,
                )
                task_checklist_state = update_checklist_from_tool_run(
                    task_checklist_state,
                    tool_name=tname,
                    ok=True,
                    tool_args=targs,
                )
            else:
                failed_call_counts[call_key] = failed_call_counts.get(call_key, 0) + 1
                last_failure_summary = summary_text

            llm_body = _cap_llm_text(summary_text)
            accumulated_tool_runs.append(
                {
                    "id": tid,
                    "tool_id": tname,
                    "args": targs,
                    "ok": ok,
                    "result": result,
                    "text_for_llm": llm_body,
                    "summary": _cap_llm_text(llm_body),
                    "step": step,
                }
            )
            yield _tool_result_sse_event(
                step=step,
                tool_id=tid,
                tool_name=tname,
                tool_args=targs,
                ok=ok,
                summary_text=summary_text,
            )

            observe_rec = _evidence_record(
                step=step,
                phase="observe",
                kind="tool_result",
                summary=summary_text,
                ok=ok,
                tool_name=tname,
                args=targs,
                files=_evidence_files_from_tool_args(targs),
            )
            evidence_ledger.append(observe_rec)
            yield {"type": "evidence_step", **observe_rec}

            # Append matching role="tool" message so the next LLM round sees it.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tid,
                    "name": tname,
                    "content": llm_body,
                }
            )

            if ok and tname == "session_attach":
                if _is_ambiguous_attach_result(result):
                    ambiguous_attach_answer = _ambiguous_attach_user_answer(
                        result, llm_body
                    )
                attached_file = _extract_attach_file_from_result(result)
                if attached_file:
                    any_session_attach = True
                    attached_entry = _registry_entry_from_attach(attached_file)
                    workspace_registry = _append_registry_entry(
                        workspace_registry,
                        attached_entry,
                    )
                    attached_context = _detect_file_context_from_registry([attached_entry])
                    if attached_context:
                        active_file_context = (
                            attached_context
                            if _is_visual_registry_entry(attached_entry)
                            else None
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
                            f"Linked file `{attached_file.get('name') or attached_file.get('path') or attached_file.get('id')}` "
                            "into the session workspace."
                        ),
                        files=_evidence_files_from_context(attached_file)
                        or [str(attached_file.get('id') or attached_file.get('artifact_id') or '')],
                        meta={"registry_size": len(workspace_registry)},
                    )
                    evidence_ledger.append(attach_rec)
                    yield {"type": "evidence_step", **attach_rec}

        # If anything attached, refresh tools + system prompt for the next round.
        if any_session_attach:
            if isinstance(workspace_runtime, dict):
                refreshed_runtime = dict(workspace_runtime)
                refreshed_runtime["available_files_count"] = len(workspace_registry or [])
                refreshed_runtime["file_index"] = [
                    {
                        "artifact_id": f.get("artifact_id") or f.get("id") or f.get("file_id"),
                        "name": f.get("name") or f.get("filename") or f.get("original_filename"),
                        "type": f.get("type") or f.get("mime_type"),
                        "mime_type": f.get("mime_type") or "",
                        "ordinal": f.get("ordinal"),
                        "turn": f.get("turn"),
                    }
                    for f in (workspace_registry or [])
                    if isinstance(f, dict)
                ][:40]
                workspace_runtime = refreshed_runtime
            (
                selected_tools,
                openai_tools,
                env_details,
                system_prompt,
            ) = _refresh_native_workspace(
                tools_manifest=tools_manifest,
                file_registry=workspace_registry,
                file_context=active_file_context,
                session_id=session_id,
                pinned_facts=pinned_facts,
                history_summary=history_summary,
                workspace_runtime=workspace_runtime,
                tool_executor=tool_executor,
                discourse_focus=current_discourse_focus.to_dict(),
                user_message=effective_user_message,
                active_plugin=active_plugin,
                rag_context=rag_context,
                rag_prompt_block=rag_prompt_block,
                multimodal_enabled=accepts_images,
                memory_prompt_block=memory_prompt_block,
                reply_language_hint=reply_language_hint,
                host_blocks=host_blocks,
            )
            # Replace the leading system message in-place so the new round
            # sees the refreshed file registry and tool catalogue.
            if messages and messages[0].get("role") == "system":
                messages[0] = {"role": "system", "content": system_prompt}
            else:
                messages.insert(0, {"role": "system", "content": system_prompt})
            log.info(
                "📎 session_attach refreshed workspace: %s files, %s LLM tools",
                len(workspace_registry),
                len(selected_tools),
            )

        # Multimodal injection for the NEXT round: if active visual context
        # exists but its bytes have not been embedded yet, append a synthetic
        # user message after the tool messages with the image_url parts so
        # the model can actually see the picture in the follow-up turn.
        current_context_keys = _visual_file_context_keys(active_file_context)
        if (
            accepts_images
            and active_file_context
            and current_context_keys
            and not current_context_keys.issubset(injected_multimodal_keys)
        ):
            notice_text = (
                "Tool results above are textual metadata. The visual file "
                "below is attached so you can inspect it directly."
            )
            try:
                mm_parts = await build_multimodal_content(
                    notice_text,
                    active_file_context,
                    auth_context={"user_id": user_id},
                )
                if isinstance(mm_parts, list):
                    messages.append({"role": "user", "content": mm_parts})
                    injected_multimodal_keys |= current_context_keys
                    yield {"type": "vision_attached", "parts": len(mm_parts)}
                    attach_rec = _evidence_record(
                        step=step,
                        phase="attach",
                        kind="vision_attached",
                        summary=(
                            f"Embedded {len(mm_parts)} multimodal part(s) for the next model turn."
                        ),
                        files=_evidence_files_from_context(active_file_context),
                    )
                    evidence_ledger.append(attach_rec)
                    yield {"type": "evidence_step", **attach_rec}
            except Exception as exc:
                log.warning("build_multimodal_content for tool result failed: %s", exc)

        if accepts_images and _visual_attachment_missing(active_file_context, injected_multimodal_keys):
            log.warning(
                "Visual file context present but no multimodal bytes were attached after tool result step=%s.",
                step,
            )
            yield {
                "type": "warning",
                "step": step,
                "message": "A visual file is selected, but its bytes were not attached to the model for the next turn.",
            }

        if ambiguous_attach_answer:
            final_answer = ambiguous_attach_answer
            final_assistant_text = ambiguous_attach_answer
            yield {"type": "answer_start", "step": step}
            yield {"type": "answer", "content": ambiguous_attach_answer}
            step_total_ms = (_time.monotonic() - step_started_at) * 1000.0
            step_timings.append(
                {
                    "step": step,
                    "llm_ms": round(step_llm_ms, 1),
                    "tool_ms": round(step_tool_ms, 1),
                    "total_ms": round(step_total_ms, 1),
                    "tools_executed": len(runnable_calls),
                    "repair": "ambiguous_session_attach_short_circuit",
                }
            )
            break

        step_total_ms = (_time.monotonic() - step_started_at) * 1000.0
        step_timings.append(
            {
                "step": step,
                "llm_ms": round(step_llm_ms, 1),
                "tool_ms": round(step_tool_ms, 1),
                "total_ms": round(step_total_ms, 1),
                "tools_executed": len(runnable_calls),
            }
        )
        # Loop continues for the next LLM round.

    # ------------------------------------------------------------------
    # Wrap-up
    # ------------------------------------------------------------------
    if final_answer is None:
        if last_failure_summary:
            not_found = session_attach_stale_error_text(last_failure_summary)
            recovered_guidance = (
                _deterministic_answer_from_tool_runs(accumulated_tool_runs)
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
                or _synthesize_from_tool_evidence(
                    accumulated_tool_runs=accumulated_tool_runs,
                    messages=messages,
                    chat_history=chat_history,
                )
                or _fallback_final_answer_from_tool_runs(accumulated_tool_runs)
            )
    final_answer = _sanitize_attach_terminal_answer(final_answer, accumulated_tool_runs)
    if accepts_images and _visual_attachment_missing(active_file_context, injected_multimodal_keys):
        log.warning(
            "Overriding final answer because visual media was not attached to the model turn."
        )
        final_answer = _final_answer_for_missing_visual_attachment(
            active_file_context,
            accumulated_tool_runs,
        )
        final_assistant_text = final_answer

    def _strip_channels(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Slim tool_runs for SSE clients; keep channels.ui for host page effects."""
        stripped = []
        for tr in runs:
            tr = dict(tr)
            tr.pop("output", None)
            try:
                result = tr.get("result")
                if not isinstance(result, dict):
                    stripped.append(tr)
                    continue
                result = dict(result)

                def _keep(channels: dict) -> dict:
                    kept: Dict[str, Any] = {}
                    for key in ("llm", "ui", "neo4j", "metadata"):
                        if key in channels:
                            kept[key] = channels.get(key)
                    # Prefer ui; keep a slim rawdata only when it carries map_action
                    raw = channels.get("rawdata")
                    if "ui" not in kept and isinstance(raw, dict) and raw.get("map_action"):
                        kept["ui"] = raw
                    return kept

                if isinstance(result.get("channels"), dict):
                    result["channels"] = _keep(result["channels"])
                    if "ui" not in result["channels"] and isinstance(result.get("data"), dict):
                        if result["data"].get("map_action"):
                            result["channels"]["ui"] = result["data"]
                    tr["result"] = result
                else:
                    data = result.get("data") if isinstance(result.get("data"), dict) else {}
                    inner = data.get("result") if isinstance(data.get("result"), dict) else {}
                    channels = inner.get("channels") if isinstance(inner.get("channels"), dict) else None
                    if isinstance(channels, dict) and channels:
                        inner = dict(inner)
                        data = dict(data)
                        kept = _keep(channels)
                        if "ui" not in kept and isinstance(inner.get("data"), dict) and inner["data"].get("map_action"):
                            kept["ui"] = inner["data"]
                        inner["channels"] = kept
                        tool_data = inner.get("data")
                        if isinstance(tool_data, dict) and isinstance(tool_data.get("resolutions"), list):
                            tool_data = dict(tool_data)
                            tool_data["resolutions"] = [
                                _slim_resolution_for_export(r) for r in tool_data["resolutions"]
                            ]
                            inner["data"] = tool_data
                        data["result"] = inner
                        result["data"] = data
                        tr["result"] = result
            except Exception:
                pass
            stripped.append(tr)
        return stripped

    conclude_rec = _evidence_record(
        step=max((r.get("step", 0) for r in evidence_ledger), default=0),
        phase="conclude",
        kind="final_answer",
        summary=(final_answer or "")[:600],
        meta={
            "tool_runs": len(accumulated_tool_runs),
            "tool_failures": sum(1 for tr in accumulated_tool_runs if not tr.get("ok")),
            "visual_attached": sorted(injected_multimodal_keys) if injected_multimodal_keys else [],
            "parallel_max_per_round": max(
                (s.get("tools_executed", 0) for s in step_timings), default=0
            ),
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
        "tools_total_executed": sum(s.get("tools_executed", 0) for s in step_timings),
        "per_step": step_timings,
    }
    log.info(
        "⏱️ native_tools_engine done: total=%.1fms rounds=%d llm=%.1fms tools=%.1fms parallel_total=%d",
        perf_summary["engine_total_ms"],
        perf_summary["rounds"],
        perf_summary["llm_total_ms"],
        perf_summary["tool_total_ms"],
        perf_summary["tools_total_executed"],
    )

    yield {
        "type": "done",
        "trace_id": trace_id,
        "final_answer": final_answer,
        "thinking": None,
        "tool_runs": _strip_channels(accumulated_tool_runs),
        "evidence_ledger": evidence_ledger,
        "evidence_summary": evidence_summary,
        "perf_summary": perf_summary,
        "engine": "native_tools",
        "assistant_raw": final_assistant_text,
        "discourse_focus": current_discourse_focus.to_dict(),
        "conversation_summary": conversation_summary_checkpoint,
        "task_checklist": task_checklist_state,
        "self_compact": self_compactor.done_payload() if self_compactor is not None else None,
        "timestamp": _dt.datetime.utcnow().isoformat(),
    }
