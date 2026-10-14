"""RLM session: kernel + children registry + inbox + harness + turn loop helpers."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import logging
import os
import re
import tempfile
import threading
import time
import traceback
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

# Which RLM session is inside an in-flight LLM call (shared chat_fn paths).
ACTIVE_RLM_SESSION_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "active_rlm_session_id", default=""
)

from src.orchestrators.rlm.schema_utils import (
    agent_model_is_explicit,
    clamp_int,
    coerce_bool,
    normalize_run_lifetime,
    normalize_tool_id_list,
)
from src.orchestrators.rlm.runtime.context_budget import (
    CHARS_PER_TOKEN,
    SAFETY_MARGIN_TOKENS,
    apply_context_budget,
    coerce_context_length,
    coerce_positive_int,
    estimate_messages_tokens,
    looks_like_context_overflow,
    looks_like_invalid_image_error,
    looks_like_request_too_large,
    media_parts_per_turn,
    parse_context_overflow_error,
)
from src.orchestrators.rlm.runtime.kernel_env import (
    KERNEL_ENV_PROBE_CODE,
    format_kernel_env_notice,
    module_not_found_hint,
    parse_kernel_env_probe,
)
from src.orchestrators.rlm.runtime.autonomous import (
    DEFAULT_AUTONOMOUS_LIMITS,
    EMPTY_ASSISTANT_NUDGE_AFTER,
    add_autonomous_usage,
    autonomous_limit_reason,
    build_goal_continuation_message,
    create_autonomous_runtime_state,
    gates_configured,
    goal_budget_exhausted,
    next_autonomous_continuation,
    note_autonomous_tool_progress,
    refresh_autonomous_quality_gates,
)
from src.orchestrators.rlm.runtime.harness import HarnessStore
from src.orchestrators.rlm.runtime.jobs import SessionJobs
from src.orchestrators.rlm.runtime.kernel import RlmKernel
from src.orchestrators.rlm.runtime.persist import (
    persist_session_files,
    slim_file_envelopes,
    slim_messages,
)
from src.orchestrators.rlm.runtime.prompts import (
    CHILD_HEALTHY_IDLE_SEC,
    CHILD_WAIT_EXTENSION_SEC,
    EMPTY_IPYTHON_REPEAT_NUDGE,
    EMPTY_IPYTHON_REPEAT_THRESHOLD,
    MALFORMED_TOOL_ARGS_NUDGE,
    MAX_CHILDREN_SOFT_WAIT_ROUNDS,
    MAX_MALFORMED_TOOL_ARGS_NUDGES,
    IPYTHON_HOWTO,
    IPYTHON_TOOL_SCHEMA,
    CELL_TOOL_NAME,
    MAX_CHILDREN_WAIT_EXTENSIONS,
    POST_COMPACT_GUARD_DISTINCT_OK,
    POST_COMPACT_GUARD_MAX_TURNS,
    POST_COMPACT_LOOP_NUDGE,
    POST_COMPACT_LOOP_THRESHOLD,
    REFINE_GUIDANCE,
    RLM_SYSTEM_APPENDIX,
    TRUNCATED_TOOL_ARGS_RESULT,
    build_child_agent_doctrine,
    build_completion_doctrine,
    build_fanin_continuation_message,
    build_recursion_capabilities,
    build_rlm_session_meta,
    build_subagent_guidance,
    format_attached_surface,
    format_tool_surface,
    reject_non_ipython_tool,
    replace_tool_surface,
)
from src.orchestrators._shared.host_context import replace_host_context, split_host_context
from src.orchestrators._shared.tool_surface import call_hint, surface_namespace, tool_alias

# Terminal log retention: keep full ipython/result text for meta.json, session
# envelope, and live progress polls (UI history must not show 240-char stubs).
LOG_LINE_CHARS = 65536
# Warn the model this many turns before rlm_max_turns runs out so it switches
# from exploring to writing the deliverable (Prime has no such signal — runs
# there just die on budget; here the run should still ship something).
TURN_BUDGET_WARN_TURNS = int(os.environ.get("RLM_TURN_BUDGET_WARN_TURNS") or 3)
# meta.json/transcript writes are debounced and run off the event loop; these
# event kinds flush synchronously so terminal state is durable immediately.
PERSIST_DEBOUNCE_SEC = max(1.0, float(os.environ.get("RLM_PERSIST_DEBOUNCE_SEC") or 1.0))
PERSIST_FLUSH_KINDS = frozenset(
    {"run_end", "run_error", "context_overflow_incomplete", "shutdown"}
)
# Set inside the background persist worker so any session touched from there
# (parent or child) routes _emit back to the owning event loop.
_persist_worker = threading.local()


def _persist_files_in_worker(session: "RlmSession", loop: asyncio.AbstractEventLoop) -> None:
    _persist_worker.loop = loop
    try:
        persist_session_files(session)
    finally:
        _persist_worker.loop = None


def build_turn_budget_notice(remaining: int, *, synthesis_turns: int) -> str:
    n = max(1, int(remaining))
    synth = max(0, int(synthesis_turns))
    tail = (
        f" After that only {synth} short synthesis turn(s) remain."
        if synth
        else " There are no extra synthesis turns after that."
    )
    return (
        f"[turn budget] {n} model turn(s) left in this run.{tail} "
        "Stop exploring now. In your next ipython call write the final deliverable "
        "file (e.g. the requested report) with everything you already know, then reply "
        "with a concise summary of what was produced and where it is saved. "
        "Partial-but-saved beats complete-but-unwritten."
    )


def build_turn_budget_exhausted_note(synthesis_turns: int) -> str:
    synth = max(1, int(synthesis_turns))
    return (
        f"\n\n[turn budget] The main turn budget is exhausted; you have at most {synth} "
        "synthesis turn(s). Do not open new files or run exploratory code. Write the "
        "deliverable file from the data already in your kernel variables and finish "
        "with a short summary of what was saved."
    )
LOG_LINES_KEEP = 10000

logger = logging.getLogger(__name__)

ChatFn = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
# body + optional on_delta(kwargs) → assembled chat.completion dict
ChatStreamFn = Callable[..., Awaitable[Dict[str, Any]]]
ToolFn = Callable[[str, Dict[str, Any]], Awaitable[Any]]
PoolSearchFn = Callable[[str, str], Awaitable[Dict[str, Any]]]
PoolResolveFn = Callable[[str], Awaitable[Dict[str, Any]]]
PoolListFn = Callable[[str], Awaitable[Dict[str, Any]]]
PoolReadFn = Callable[..., Awaitable[Dict[str, Any]]]
ExtractAssistantFn = Callable[[Dict[str, Any]], Dict[str, Any]]

_THINK_PAIR_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)
_THINK_TAG_RE = re.compile(r"</?think>", re.I)
# Gemma 4 thinking: <|channel>thought\n ... <channel|>
_CHANNEL_THOUGHT_RE = re.compile(
    r"<\|channel>\s*thought\s*\n?(.*?)(?:<channel\|>|$)",
    re.S | re.I,
)
_CHANNEL_MARK_RE = re.compile(r"<\|channel>\s*thought\s*\n?|<channel\|>", re.I)


def _split_incomplete_special_suffix(text: str) -> tuple[str, str]:
    """Hold back a trailing partial ``<|…`` token so it is not flashed to the UI."""
    s = str(text or "")
    lt = s.rfind("<")
    if lt < 0:
        return s, ""
    tail = s[lt:]
    if ">" in tail:
        return s, ""
    # Unclosed angle-bracket sequence at the end (e.g. "<|chan", "<|channel").
    if tail.startswith("<"):
        return s[:lt], tail
    return s, ""


def _reasoning_from_message(msg: Dict[str, Any]) -> str:
    reasoning = (
        msg.get("reasoning_content")
        or msg.get("reasoning")
        or msg.get("reasoning_text")
        or msg.get("thinking")
        or msg.get("thought")
        or ""
    )
    if isinstance(reasoning, dict):
        return str(reasoning.get("content") or reasoning.get("text") or "").strip()
    return str(reasoning or "").strip()


def split_channel_markup(text: str) -> tuple[str, str]:
    """Split Gemma4 ``<|channel>thought … <channel|>`` from visible answer.

    While the closing ``<channel|>`` is still missing (mid-stream), everything
    after the thought opener is treated as reasoning. Incomplete openers
    (``<|channel>`` without ``thought`` yet) are held back — never shown as
    assistant text.
    """
    raw = str(text or "")
    if not raw:
        return "", ""
    s, held = _split_incomplete_special_suffix(raw)

    if "<|channel>" not in s and "<channel|>" not in s:
        # Still accumulating an opener like "<|chan" — show nothing yet.
        if held and not s.strip():
            return "", ""
        return s, ""

    lower = s.lower()
    open_at = lower.find("<|channel>")
    if open_at >= 0:
        after = s[open_at + len("<|channel>") :]
        after_l = after.lstrip().lower()
        if not after_l.startswith("thought") and "<channel|>" not in lower:
            # Opener present but "thought" keyword not complete — hide it.
            return s[:open_at], ""

    parts = list(_CHANNEL_THOUGHT_RE.finditer(s))
    if not parts:
        cleaned = _CHANNEL_MARK_RE.sub("", s)
        return cleaned, ""
    reasoning_chunks = [m.group(1) for m in parts if str(m.group(1) or "").strip()]
    visible = _CHANNEL_THOUGHT_RE.sub("", s)
    visible = _CHANNEL_MARK_RE.sub("", visible)
    reasoning = "\n".join(c.strip() for c in reasoning_chunks if str(c).strip())
    return visible, reasoning


def split_assistant_visible(msg: Dict[str, Any]) -> tuple[str, str]:
    """Return (visible content, reasoning). Thinking is never the answer.

    Qwen/vLLM may put CoT in ``reasoning_content``, wrap it in ``<think>``,
    leak an orphan ``</think>``, or (Gemma 4) use ``<|channel>thought``.
    The visible answer is only the post-think text.
    """
    if not isinstance(msg, dict):
        return "", ""
    content = msg.get("content")
    if content is None:
        content_s = ""
    elif isinstance(content, str):
        content_s = content
    else:
        content_s = str(content)
    reasoning = _reasoning_from_message(msg)

    channel_vis, channel_reason = split_channel_markup(content_s)
    if channel_reason or channel_vis != content_s:
        content_s = channel_vis
        if channel_reason and not reasoning:
            reasoning = channel_reason
        elif channel_reason and channel_reason not in reasoning:
            reasoning = (reasoning + "\n" + channel_reason).strip()

    chunks = [m.strip() for m in _THINK_PAIR_RE.findall(content_s) if str(m).strip()]
    if chunks:
        content_s = _THINK_PAIR_RE.sub("", content_s)
        if not reasoning:
            reasoning = "\n".join(chunks)
    if _THINK_TAG_RE.search(content_s):
        orphan = _THINK_TAG_RE.sub("", content_s).strip()
        if not reasoning and orphan:
            reasoning = orphan
        content_s = ""
    content_s = content_s.strip()
    reasoning = reasoning.strip()
    if content_s and reasoning and content_s == reasoning:
        content_s = ""
    return content_s, reasoning


_INCOMPLETE_INTENT_RE = re.compile(
    r"(?is)^\s*("
    r"let me\b|i(?:'m| am) (?:going to|gonna)\b|i(?:'ll| will)\b|"
    r"i need to\b|i should\b|trying to\b|try(?:ing)? a different|"
    r"wait[,.]|hold on\b|"
    r"بگذار|صبر کن"
    r")"
)
_INCOMPLETE_COURTESY_RE = re.compile(r"(?is)^\s*let me know\b")
_INCOMPLETE_ACTION_RE = re.compile(
    r"(?i)\b(await|ipython|search|function|tool|subagent|pool|dir\(|help\()\b"
)
_INCOMPLETE_TRAILING_RE = re.compile(r"[:：…]\s*$")
_AWAITING_CHILDREN_RE = re.compile(
    r"(?is)\b("
    r"awaiting|waiting for|initiated|deployed a (?:sub[- ]?agent|child)|"
    r"have spawned|i(?:'ve| have) deployed|still running|"
    r"once (?:the )?(?:data|results?|reply|message) (?:is|are) received|"
    r"منتظر|راه‌اندازی کردم|اسپان"
    r")\b"
)
_FUTURE_INTENT_RE = re.compile(
    r"(?is)\b("
    r"i(?:'ll| will)\b|before delegating|next i need|let's continue|"
    r"spawn(?:ing)?|subagent|sub-agent|execute in ipython|"
    r"now i(?:'ll| will)|going to sample|plan the work|sample some images"
    r")\b"
)
# Mid-turn planning phrases anywhere in prose (not only line-start).
_ONGOING_WORK_RE = re.compile(
    r"(?is)\b("
    r"i need to\b|"
    r"let me look(?: at| for)?\b|"
    r"let me (?:analyze|extract|check|save|write|create|read|open|search)\b|"
    r"i(?:'ll| will) (?:analyze|extract|look|create|write|save|check|sample)\b|"
    r"create the comprehensive report\b"
    r")\b"
)
_DELIVERABLE_HINT_RE = re.compile(
    r"(?is)\b("
    r"(?:saved|written|wrote)\s+(?:to|the\b)|"
    r"report[_\w-]*\.md\b|"
    r"deliverable complete|task complete"
    r")\b"
)
_CONTEXT_CHECKPOINT_SECTION_RE = re.compile(
    r"(?im)^##\s*(Goal|Progress|Next Steps|Constraints\s*&\s*Preferences|Key Decisions|Critical Context)\s*$"
)


class ContextOverflowExhausted(RuntimeError):
    """Context window overflow persisted after compact/retry."""


def looks_like_context_checkpoint(text: str) -> bool:
    """True when visible text is an LLM compaction/checkpoint summary, not a deliverable."""
    s = str(text or "").strip()
    if not s:
        return False
    if s.startswith("[context checkpoint]"):
        return True
    sections = _CONTEXT_CHECKPOINT_SECTION_RE.findall(s)
    if not sections:
        return False
    names = {str(x).strip().lower() for x in sections}
    if "goal" in names and ("progress" in names or "next steps" in names):
        return True
    if "progress" in names and "next steps" in names:
        return True
    return False


def looks_like_incomplete_assistant(text: str) -> bool:
    """True when the model narrated the next step instead of answering or calling ipython."""
    s = str(text or "").strip()
    if not s or len(s) > 800:
        return False
    if _INCOMPLETE_COURTESY_RE.match(s):
        return False
    # "Let me …" is almost always plan narration, not a deliverable answer.
    if re.match(r"(?is)^\s*let me\b", s):
        return True
    if _ONGOING_WORK_RE.search(s):
        return True
    trailing = bool(_INCOMPLETE_TRAILING_RE.search(s))
    intent = bool(_INCOMPLETE_INTENT_RE.match(s))
    mentions_action = bool(_INCOMPLETE_ACTION_RE.search(s))
    if trailing and (intent or mentions_action or len(s) < 240):
        return True
    if intent and mentions_action and len(s) < 400:
        return True
    # Other plan phrases ("I'll …", "I need to …") without ipython in the same turn.
    if intent and len(s) < 400:
        return True
    return False


def looks_like_awaiting_children(text: str) -> bool:
    """True when the model reported spawn/status instead of a finished user deliverable.

    No length cap — long \"I have initiated…\" status reports must not finalize the run.
    """
    s = str(text or "").strip()
    if not s:
        return False
    return bool(_AWAITING_CHILDREN_RE.search(s))


def looks_like_terminal_answer(text: str) -> bool:
    """True when this turn's prose looks like a finished user-facing answer.

    Short status asides (\"Network works…\") must not count as terminal evidence
    or autonomous will stop mid-task. Incomplete / awaiting-children narration
    also does not count.
    """
    s = str(text or "").strip()
    if not s:
        return False
    if looks_like_incomplete_assistant(s) or looks_like_awaiting_children(s):
        return False
    if looks_like_context_checkpoint(s):
        return False
    if _FUTURE_INTENT_RE.search(s):
        return False
    # Progress notes and environment checks are typically short.
    if len(s) < 160:
        return False
    return True


def looks_like_malformed_tool_args_error(text: str) -> bool:
    """Provider rejected chat because history has truncated tool-call JSON arguments."""
    t = str(text or "").lower()
    if "unterminated string" in t:
        return True
    if "400" in t and ("badrequest" in t or "bad request" in t) and (
        "json" in t or "tool" in t or "arguments" in t or "unterminated" in t
    ):
        return True
    return False


def parse_tool_call_arguments(raw_args: Any) -> tuple[Optional[Dict[str, Any]], bool]:
    """Return ``(args_dict_or_none, ok)``. ``ok`` is False when JSON is truncated/invalid."""
    if isinstance(raw_args, dict):
        return raw_args, True
    if raw_args is None:
        return {}, True
    if not isinstance(raw_args, str):
        return None, False
    text = raw_args.strip()
    if not text:
        return {}, True
    try:
        parsed = json.loads(text)
    except Exception:
        return None, False
    if isinstance(parsed, dict):
        return parsed, True
    return None, False


def heal_invalid_tool_call_arguments(messages: List[Dict[str, Any]]) -> int:
    """Rewrite invalid ``tool_calls[].function.arguments`` to valid JSON so providers accept history.

    Returns the number of tool calls healed. Does not remove turns — the model is told
    via nudge / tool result to re-issue a complete call.
    """
    healed = 0
    empty_args = json.dumps({"code": ""}, ensure_ascii=False)
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tcs = msg.get("tool_calls")
        if not isinstance(tcs, list):
            continue
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else None
            if not isinstance(fn, dict):
                continue
            raw = fn.get("arguments")
            _, ok = parse_tool_call_arguments(raw)
            if ok:
                continue
            fn["arguments"] = empty_args
            healed += 1
    return healed


def compact_rlm_text(text: str, max_length: int = 160) -> str:
    """Prime ``compactRlmText``: single-line preview for terminal notices."""
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= max_length:
        return compact
    return compact[: max(0, max_length - 3)].rstrip() + "..."


def format_completed_without_reply_notice(
    *,
    session_name: str,
    child_id: str,
    last_assistant_text: str = "",
    produced_files: Optional[List[Any]] = None,
) -> str:
    """Prime ``createRlmChildTerminalNoticeMessage`` content for completed_without_reply."""
    name = str(session_name or "").strip() or child_id
    cid = str(child_id or "").strip() or "?"
    lines = [f"RLM child {name} ({cid}) completed without sending a reply."]
    preview = compact_rlm_text(last_assistant_text) if str(last_assistant_text or "").strip() else ""
    if preview:
        lines.append(f"Last assistant text: {preview}")
    paths: List[str] = []
    for item in produced_files or []:
        if isinstance(item, str) and item.strip():
            paths.append(item.strip().replace("\\", "/"))
            continue
        if not isinstance(item, dict):
            continue
        rel = str(item.get("relpath") or item.get("path") or "").strip().replace("\\", "/")
        if not rel:
            continue
        size = item.get("size_bytes")
        if size is not None:
            try:
                paths.append(f"{rel} ({int(size)} bytes)")
                continue
            except (TypeError, ValueError):
                pass
        paths.append(rel)
    if paths:
        lines.append(
            "Files created/updated in the shared workspace during this child "
            "(read these even though no agent_message was sent):"
        )
        lines.extend(f"- {p}" for p in paths[:40])
    else:
        lines.append(
            "No new/updated deliverable files were detected in the shared workspace."
        )
    return "\n".join(lines)


def format_child_failure_notice(
    *,
    session_name: str,
    child_id: str,
    error: str,
    status: str = "error",
    last_content: str = "",
    last_reasoning: str = "",
    log_tail: Optional[List[str]] = None,
) -> str:
    """Detailed parent inbox notice when a child fails before finishing."""
    name = str(session_name or "").strip() or child_id
    cid = str(child_id or "").strip() or "?"
    err = str(error or "unknown error").strip() or "unknown error"
    lines = [
        f"RLM child {name} ({cid}) failed before finishing.",
        f"status: {str(status or 'error').strip() or 'error'}",
        f"error: {err}",
    ]
    content = str(last_content or "").strip()
    if content:
        clipped = content if len(content) <= 4000 else content[:4000] + "\n…[truncated]"
        lines.append(f"last_content:\n{clipped}")
    reasoning = str(last_reasoning or "").strip()
    if reasoning:
        clipped_r = (
            reasoning if len(reasoning) <= 4000 else reasoning[:4000] + "\n…[truncated]"
        )
        lines.append(f"last_reasoning:\n{clipped_r}")
    logs = [str(x).strip() for x in (log_tail or []) if str(x).strip()]
    if logs:
        lines.append("log_tail:\n" + "\n".join(logs[-20:]))
    lines.append(
        "Reassign this child's work or synthesize from the surviving children/files."
    )
    return "\n".join(lines)


def format_child_cancelled_notice(
    *,
    session_name: str,
    child_id: str,
    reason: str = "",
    last_content: str = "",
    last_reasoning: str = "",
    log_tail: Optional[List[str]] = None,
) -> str:
    """Parent inbox notice when a child is cancelled before finishing."""
    name = str(session_name or "").strip() or child_id
    cid = str(child_id or "").strip() or "?"
    base = f"RLM child {name} ({cid}) was cancelled before finishing"
    reason_s = str(reason or "").strip()
    if reason_s:
        base = f"{base}: {reason_s}"
    lines = [base]
    content = str(last_content or "").strip()
    if content:
        clipped = content if len(content) <= 2000 else content[:2000] + "\n…[truncated]"
        lines.append(f"last_content:\n{clipped}")
    reasoning = str(last_reasoning or "").strip()
    if reasoning:
        clipped_r = (
            reasoning if len(reasoning) <= 2000 else reasoning[:2000] + "\n…[truncated]"
        )
        lines.append(f"last_reasoning:\n{clipped_r}")
    logs = [str(x).strip() for x in (log_tail or []) if str(x).strip()]
    if logs:
        lines.append("log_tail:\n" + "\n".join(logs[-12:]))
    return "\n".join(lines)


def content_has_user_query_text(content: Any) -> bool:
    """True when message content has non-empty text (chat templates may require it)."""
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str) and part.strip():
                return True
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if text is None and part.get("type") == "text":
                text = part.get("content")
            if isinstance(text, str) and text.strip():
                return True
    return False


def messages_have_user_query(messages: Optional[List[Any]]) -> bool:
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        if str(msg.get("role") or "") != "user":
            continue
        if content_has_user_query_text(msg.get("content")):
            return True
    return False


def model_preserves_all_thinking(model: str) -> bool:
    """Fallback when catalog does not declare ``preserve_thinking``.

    Prefer catalog / cfg / gen_params. This name heuristic is only the last
    resort: Qwen-style templates need full history CoT (Prime #3325); Gemma 4
    docs default ``preserve_thinking=false``.
    """
    mid = str(model or "").strip().lower()
    if not mid:
        return False
    if "gemma" in mid:
        return False
    if "qwen" in mid:
        return True
    return False


def resolve_preserve_thinking(
    *,
    model: str = "",
    cfg: Optional[Mapping[str, Any]] = None,
    gen_params: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Wire-time preserve_thinking with catalog/cfg precedence over name heuristic."""
    ctk = None
    if isinstance(gen_params, Mapping):
        ctk = gen_params.get("chat_template_kwargs")
        if isinstance(ctk, Mapping) and "preserve_thinking" in ctk and ctk.get("preserve_thinking") is not None:
            return coerce_bool(ctk.get("preserve_thinking"), default=False)
    if isinstance(cfg, Mapping):
        if "preserve_thinking" in cfg and cfg.get("preserve_thinking") is not None:
            return coerce_bool(cfg.get("preserve_thinking"), default=False)
        ctk = cfg.get("chat_template_kwargs")
        if isinstance(ctk, Mapping) and "preserve_thinking" in ctk and ctk.get("preserve_thinking") is not None:
            return coerce_bool(ctk.get("preserve_thinking"), default=False)
        hist = str(cfg.get("reasoning_history_policy") or "").strip().lower()
        if hist == "preserve_all":
            return True
        if hist in ("strip", "tool_chain_only"):
            return False
    return model_preserves_all_thinking(model)


def resolve_reasoning_replay_max(
    *,
    model: str = "",
    cfg: Optional[Mapping[str, Any]] = None,
    gen_params: Optional[Mapping[str, Any]] = None,
    preserve_all: Optional[bool] = None,
) -> int:
    """How many recent assistant CoTs to keep on the outbound wire."""
    if isinstance(cfg, Mapping) and cfg.get("reasoning_replay_max") is not None:
        try:
            return max(0, min(64, int(cfg.get("reasoning_replay_max"))))
        except (TypeError, ValueError):
            pass
    if isinstance(cfg, Mapping):
        hist = str(cfg.get("reasoning_history_policy") or "").strip().lower()
        if hist == "strip":
            return 0
        if hist == "tool_chain_only":
            return 2
        if hist == "preserve_all":
            return 8
    if preserve_all is None:
        preserve_all = resolve_preserve_thinking(
            model=model, cfg=cfg, gen_params=gen_params
        )
    return 8 if preserve_all else 2


def normalize_ipython_code(code: Any) -> str:
    """Canonical form for exact-repeat detection (CRLF→LF, strip edges)."""
    return str(code or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def is_bare_empty_ipython_result(text: Any) -> bool:
    """True only for the silent-cell placeholder, not errors or real output."""
    return str(text or "").strip() == "(empty)"


def note_empty_ipython_repeat(
    *,
    code: Any,
    result_text: Any,
    last_code: Optional[str],
    streak: int,
    threshold: int = EMPTY_IPYTHON_REPEAT_THRESHOLD,
) -> tuple[str, Optional[str], int, bool]:
    """Track identical silent ipython cells and optionally append a nudge.

    Returns ``(content_for_model, next_last_code, next_streak, nudged)``.
    Non-empty results clear the streak. A nudge fires at ``threshold`` consecutive
    identical empties, then resets so the model must repeat twice again.
    """
    text = str(result_text or "")
    if not is_bare_empty_ipython_result(text):
        return text, None, 0, False
    norm = normalize_ipython_code(code)
    if not norm:
        # Empty code cell — still empty output, but don't treat as a sticky loop key.
        return text, None, 0, False
    if last_code is not None and norm == last_code:
        next_streak = max(1, int(streak or 0)) + 1
    else:
        next_streak = 1
    if next_streak >= max(2, int(threshold or 2)):
        nudged = f"{text}\n\n{EMPTY_IPYTHON_REPEAT_NUDGE}"
        return nudged, None, 0, True
    return text, norm, next_streak, False


def outbound_messages_sanitize_for_wire(
    messages: Optional[List[Any]],
) -> List[Dict[str, Any]]:
    """Coerce message fields so strict OpenAI-compat validators accept the body.

    Some upstreams (e.g. Sepidour) reject ``content: null`` with Pydantic
    ``string_type`` errors even when ``tool_calls`` are present. OpenAI itself
    allows null content for tool-call turns; we still send ``\"\"`` on the wire.
    """
    out: List[Dict[str, Any]] = []
    for raw in messages or []:
        if not isinstance(raw, dict):
            continue
        msg = dict(raw)
        role = str(msg.get("role") or "").strip() or "user"
        msg["role"] = role
        if "content" in msg and msg.get("content") is None:
            msg["content"] = ""
        elif "content" not in msg:
            # Tool results / assistant tool-calls must still have a content key
            # for backends that union-match on ChatCompletion*MessageParam.
            if role in ("assistant", "user", "system", "developer", "tool"):
                msg["content"] = ""
        out.append(msg)
    return out


def outbound_messages_trim_reasoning(
    messages: Optional[List[Any]],
    *,
    preserve_all: bool = False,
    max_recent_with_reasoning: int = 2,
) -> List[Dict[str, Any]]:
    """Request-only: keep session history intact, shrink CoT on the wire.

    - Session / UI still store full ``reasoning_content``.
    - Outbound copies drop old reasoning so context does not grow without bound.
    - Keep reasoning on the last ``max_recent_with_reasoning`` assistant turns
      (active tool-call chain continuity). With ``preserve_all`` (Qwen), still
      cap to a higher default so compaction remains the long-term safety net.
    """
    msgs = [dict(m) for m in (messages or []) if isinstance(m, dict)]
    keep = int(max_recent_with_reasoning or 0)
    if preserve_all:
        keep = max(keep, 8)
    if keep <= 0:
        for msg in msgs:
            if str(msg.get("role") or "") != "assistant":
                continue
            msg.pop("reasoning_content", None)
            msg.pop("reasoning", None)
            msg.pop("reasoning_text", None)
        return msgs
    assistant_idxs = [
        i for i, m in enumerate(msgs) if str(m.get("role") or "") == "assistant"
    ]
    keep_idxs = set(assistant_idxs[-keep:]) if assistant_idxs else set()
    for i, msg in enumerate(msgs):
        if str(msg.get("role") or "") != "assistant":
            continue
        if i in keep_idxs:
            continue
        msg.pop("reasoning_content", None)
        msg.pop("reasoning", None)
        msg.pop("reasoning_text", None)
    return msgs


def outbound_messages_with_user_query(
    messages: Optional[List[Any]],
    *,
    fallback_text: str = "",
) -> List[Dict[str, Any]]:
    """Return a request-only copy that satisfies chat templates needing a user turn.

    Agents may legitimately run with only a system prompt (no input wire). Some
    backends (e.g. Qwen) still 400 with ``No user query found in messages`` on
    system-only payloads — mirror the system/fallback text as a user turn on the
    *outgoing* body only; never mutate the session transcript.
    """
    msgs = [m for m in (messages or []) if isinstance(m, dict)]
    if messages_have_user_query(msgs):
        return msgs
    text = ""
    for msg in reversed(msgs):
        if str(msg.get("role") or "") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            text = content.strip()
            break
        if content_has_user_query_text(content):
            # Multimodal system is rare; flatten text parts.
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, str) and part.strip():
                        parts.append(part.strip())
                    elif isinstance(part, dict):
                        t = part.get("text")
                        if isinstance(t, str) and t.strip():
                            parts.append(t.strip())
                text = "\n".join(parts)
            break
    if not text:
        text = str(fallback_text or "").strip()
    if not text:
        return msgs
    if len(text) > 8000:
        text = text[:8000]
    return list(msgs) + [{"role": "user", "content": text}]


def _tool_result_preview(value: Any, limit: int) -> str:
    """Short text for a tool result: its own summary field when it has one."""
    if isinstance(value, dict):
        for key in ("text_for_llm", "summary", "message", "error"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                value = text
                break
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)
    text = " ".join(text.split())
    return text[:limit] if limit > 0 else text


def format_rlm_log_line(kind: str, fields: Optional[Dict[str, Any]] = None) -> str:
    """Human log line for the live RLM terminal (deploy-panel style)."""
    data = fields if isinstance(fields, dict) else {}
    name = str(kind or "event")
    if name == "tool_start":
        args = _tool_result_preview(data.get("args") or {}, 0)
        return f"[tool] {data.get('tool_id') or '?'} {args}".rstrip()
    if name == "tool_result":
        status = "ok" if data.get("ok") else "error"
        detail = data.get("summary") if data.get("ok") else data.get("error") or data.get("summary")
        return f"[tool_result] {data.get('tool_id') or '?'} {status}: {detail or ''}".rstrip()
    if name == "thought":
        tk = str(data.get("thought_kind") or "thought")
        return f"[{tk}] {data.get('preview') or ''}".rstrip()
    if name == "run_error":
        text = str(
            data.get("detail") or data.get("error") or data.get("message") or ""
        ).strip()
        return f"[error] {text[:LOG_LINE_CHARS]}" if text else "[error] unknown"
    if name == "run_end":
        status = str(data.get("status") or "")
        reason = str(data.get("autonomous_reason") or "").strip()
        if status != "completed" and reason:
            return f"[end] {status} ({reason})"
        return f"[end] {status}".rstrip()
    if name == "run_start":
        return "[start] RLM run"
    if name == "kernel_ready":
        return "[kernel] ready"
    if name == "kernel_queue":
        return f"[waiting] {data.get('preview') or 'kernel queue'}".rstrip()
    if name == "ipython":
        return f"[ipython] {data.get('code_preview') or ''}".rstrip()
    if name == "ipython_result":
        return f"[result] {data.get('preview') or ''}".rstrip()
    if name == "empty_ipython_repeat_nudge":
        return (
            f"[nudge] identical empty ipython ×{data.get('streak') or '?'} — "
            "ask model to change approach"
        )
    if name == "inbox_inject":
        who = data.get("from_name") or data.get("from_role") or ""
        return f"[inbox] {who}".rstrip()
    if name == "chat_stream_fallback":
        return f"[stream-fallback] {str(data.get('error') or '')[:LOG_LINE_CHARS]}".rstrip()
    if name == "llm_reconnect":
        preview = str(data.get("preview") or "").strip()
        if preview:
            return f"[reconnecting] {preview[:LOG_LINE_CHARS]}"
        n = data.get("attempt") or "?"
        total = data.get("max_attempts") or "?"
        wait = data.get("wait_sec")
        if wait is not None:
            return f"[reconnecting] model gateway 503 — retry {n}/{total} in {wait}s"
        return f"[reconnecting] model gateway 503 — retry {n}/{total}"
    if name == "context_budget_retry":
        err = str(data.get("error") or "").strip()
        if looks_like_request_too_large(err):
            return "[overflow] request too large — compacting"
        return "[overflow] context window — compacting"
    if name == "look_overflow_split":
        n = data.get("deferred") or 0
        return f"[look] 413 — sending 1 file now, {n} queued for later"
    if name == "look_overflow_recompress":
        n = data.get("recompressed") or 0
        return f"[look] 413 — recompressed {n} image(s)"
    if name == "look_media_stripped":
        n = data.get("stripped") or 0
        return f"[look] stripped {n} media part(s) after 413/overflow"
    if name == "stream_wait":
        return f"[stream] {data.get('preview') or 'waiting'}".rstrip()
    if name == "llm_call":
        flags = []
        if data.get("thinking"):
            flags.append("thinking")
        if data.get("stream"):
            flags.append("stream")
        flag = (" " + " ".join(flags)) if flags else ""
        return f"[chat] {data.get('model') or 'model'}{flag}".rstrip()
    if name == "assistant_turn":
        flags = []
        if data.get("has_reasoning"):
            flags.append("thinking")
        if data.get("has_tools"):
            flags.append("tools")
        preview = str(data.get("preview") or "").strip()
        if not flags and not preview:
            return "[llm] empty"
        head = " ".join(flags) if flags else "reply"
        if preview:
            if len(preview) > 160:
                preview = preview[:157].rstrip() + "…"
            return f"[llm] {head}: {preview}"
        return f"[llm] {head}"
    if name == "empty_reply_nudge":
        return f"[nudge] empty reply — re-asking model (#{data.get('n') or '?'})"
    if name == "empty_assistant_exhausted":
        # Legacy event name; product no longer hard-stops on blank turns.
        return (
            f"[empty] model returned blank ×{data.get('n') or '?'} — "
            "continuing with stronger nudge"
        )
    if name == "empty_assistant_nudge":
        return (
            f"[empty] blank ×{data.get('n') or '?'} — nudging model to act"
        )
    if name == "answer_retry_nudge":
        reason = str(data.get("reason") or "retry")
        return f"[nudge] synthesize answer ({reason})"
    if name == "autonomous_continuation":
        return (
            f"[continue] autonomous "
            f"({data.get('reason') or 'missing_terminal_evidence'} "
            f"#{data.get('n') or '?'})"
        )
    if name == "goal_continuation":
        return f"[continue] goal (#{data.get('n') or '?'})"
    if name == "context_budget":
        bits = [
            f"[budget] input≈{data.get('est_input')}",
            f"max_tokens={data.get('max_tokens')}",
            f"dropped={data.get('dropped')}",
        ]
        if data.get("media_stripped"):
            bits.append(f"media_stripped={data.get('media_stripped')}")
        return " ".join(bits)
    if name == "context_budget_retry":
        return f"[budget-retry] {str(data.get('error') or '')[:LOG_LINE_CHARS]}"
    if name == "context_overflow_incomplete":
        why = data.get("reason") or data.get("error") or "context_overflow"
        return f"[incomplete] context overflow — {str(why)[:LOG_LINE_CHARS]}"
    if name == "turn_budget_notice":
        return (
            f"[budget] {data.get('remaining')} turn(s) left of {data.get('budget')} "
            "— asked model to write the deliverable now"
        )
    if name == "turn_budget_exhausted":
        tail = " mid tool-work" if data.get("mid_work") else ""
        return f"[budget] rlm_max_turns={data.get('budget')} exhausted{tail}"
    if name == "kernel_env":
        return (
            f"[kernel] python {data.get('python') or '?'}; "
            f"packages: {data.get('available') or 'none'}; "
            f"missing: {data.get('missing') or 0}"
        )
    if name == "kernel_env_hint":
        return f"[env] {str(data.get('preview') or '')[:LOG_LINE_CHARS]}"
    if name == "autonomous_limit_exhausted":
        return f"[incomplete] autonomous stopped — {data.get('reason') or 'missing_terminal_evidence'}"
    if name == "look_inject":
        deferred = int(data.get("deferred") or 0)
        count = int(data.get("count") or 0)
        names = str(data.get("filenames") or data.get("preview") or "").strip()
        base = f"[look] ({count}) {names}".rstrip() if count else f"[look] {names}".rstrip()
        return base + (f" (+{deferred} queued for next turn)" if deferred else "")
    if name == "look_wire_blind":
        return (
            f"[look] wire blind (session_media={data.get('session_media')} "
            f"wire_media={data.get('wire_media')}) — restoring pixels"
        )
    if name == "look_wire_restored":
        return f"[look] wire restored media={data.get('media')}"
    if name == "look_strip_before_compact":
        n = int(data.get("count") or 0)
        names = str(data.get("filenames") or "").strip()
        if names:
            return f"[look] strip before compact ({n}): {names}"
        return f"[look] strip before compact ({n})"
    if name == "malformed_tool_args_nudge":
        n = data.get("n") or "?"
        healed = data.get("healed")
        return (
            f"[nudge] truncated tool-call JSON — asking model to retry "
            f"(#{n}, healed={healed})"
        )
    if name == "child_admitted":
        return f"[child] {data.get('name') or data.get('rlm_child_id') or ''}".rstrip()
    if name == "compact":
        mode = data.get("mode") or "llm"
        why = data.get("reason") or ""
        kept = data.get("kept")
        before = data.get("tokens_before")
        bits = [f"[compact] {mode}"]
        if why:
            bits.append(str(why))
        if before is not None:
            bits.append(f"tokens_before={before}")
        if kept is not None:
            bits.append(f"kept={kept}")
        return " ".join(bits)
    if name == "compact_error":
        return f"[compact-error] {str(data.get('error') or '')[:LOG_LINE_CHARS]}"
    if name == "shutdown":
        return "[shutdown]"
    if name == "file":
        return f"[file] {data.get('filename') or data.get('preview') or ''}".rstrip()
    if name == "incomplete_nudge":
        return "[nudge] narrated next step — asking for ipython or a real answer"
    extra = data.get("error") or data.get("preview") or data.get("status") or ""
    return f"[{name}] {extra}".rstrip()


def format_exception_for_log(exc: BaseException, *, limit: int = 1600) -> str:
    """Always include exception type; str(exc) is often empty (AssertionError, Exception())."""
    name = type(exc).__name__
    msg = str(exc).strip()
    parts = [f"{name}: {msg}" if msg else name]
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        cmsg = str(cause).strip()
        cname = type(cause).__name__
        parts.append(f"caused by {cname}: {cmsg}" if cmsg else f"caused by {cname}")
    if len(msg) < 48:
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        tail = "".join(tb[-5:]).strip()
        if tail:
            parts.append(tail)
    return "\n".join(parts)[: max(80, int(limit))]


class RlmSession:
    """One RLM agent invocation (parent or child)."""

    def __init__(
        self,
        *,
        cfg: Dict[str, Any],
        user_content: Any,
        depth: int = 0,
        session_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        flow_id: str = "",
        user_id: str = "",
        node_id: str = "",
        artifact_dir: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        chat_fn: ChatFn,
        extract_assistant_fn: ExtractAssistantFn,
        gen_params: Dict[str, Any],
        tool_runner: ToolFn,
        pool_search: Optional[PoolSearchFn] = None,
        pool_resolve: Optional[PoolResolveFn] = None,
        pool_list: Optional[PoolListFn] = None,
        pool_read: Optional[PoolReadFn] = None,
        pool_defs: Optional[List[Dict[str, Any]]] = None,
        spawn_child: Optional[Callable[..., Awaitable["RlmSession"]]] = None,
        model_finder: Optional[Callable[[str, int], Awaitable[List[Dict[str, Any]]]]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        chat_stream_fn: Optional[ChatStreamFn] = None,
    ) -> None:
        self.cfg = dict(cfg or {})
        self.user_content = user_content
        self.depth = int(depth)
        self.session_id = session_id or f"rlm_{uuid.uuid4().hex[:16]}"
        self.parent_session_id = parent_session_id
        self.run_id = run_id
        self.flow_id = flow_id
        self.user_id = user_id
        self.node_id = node_id
        from src.orchestrators.rlm.runtime.resource_budget import get_run_budget

        self._resource_budget = get_run_budget(run_id or self.session_id, self.cfg)
        self._kernel_slot_acquired = False
        self.artifact_dir = artifact_dir or tempfile.mkdtemp(prefix=f"rlm_{self.session_id}_")
        os.makedirs(self.artifact_dir, exist_ok=True)
        os.makedirs(os.path.join(self.artifact_dir, "harness"), exist_ok=True)
        os.makedirs(os.path.join(self.artifact_dir, "children"), exist_ok=True)
        # Shared filesystem root for the IPython kernel (cwd + jail). Root sessions
        # use artifact_dir; children inherit the parent's workspace so relative
        # paths the parent wrote (``work/batches/4.json``) resolve correctly.
        # Session-private state (transcript, kernel snapshot, harness) stays in
        # artifact_dir even when that is a subdirectory of workspace_dir.
        ws = str(workspace_dir or "").strip() or self.artifact_dir
        self.workspace_dir = os.path.realpath(ws) if os.path.isdir(ws) else ws
        os.makedirs(self.workspace_dir, exist_ok=True)

        self.chat_fn = chat_fn
        self.chat_stream_fn = chat_stream_fn
        self.extract_assistant_fn = extract_assistant_fn
        self._stream_pub_at = 0.0
        self._stream_reason_logged_len = 0
        self._stream_content_buf = ""
        self._chat_turn_request = ""
        self._stream_content_logged_len = 0
        self._stream_tool_calls: List[Dict[str, Any]] = []
        self._stream_tool_logged_len = 0
        self.gen_params = dict(gen_params or {})
        self.context_length = coerce_context_length(self.cfg, self.gen_params)
        self.max_output_tokens = coerce_positive_int(
            self.cfg.get("max_output_tokens"),
            self.gen_params.get("max_output_tokens"),
        )
        self.tool_runner = tool_runner
        self.pool_search = pool_search
        self.pool_resolve = pool_resolve
        self.pool_list = pool_list
        self.pool_read = pool_read
        self.pool_defs = list(pool_defs or [])
        self.spawn_child = spawn_child
        self.model_finder = model_finder
        self.on_event = on_event

        self.max_steps = clamp_int(self.cfg.get("max_steps"), default=5, min_v=1, max_v=20)
        # 0 = unlimited turn loop (product); positive = hard cap (tests / opt-in).
        try:
            self.rlm_max_turns = max(0, int(self.cfg.get("rlm_max_turns") or 0))
        except (TypeError, ValueError):
            self.rlm_max_turns = 0
        self.rlm_synthesis_turns = clamp_int(
            self.cfg.get("rlm_synthesis_turns"), default=5, min_v=0, max_v=20
        )
        self.max_depth = clamp_int(
            self.cfg.get("max_subagent_depth"), default=1, min_v=0, max_v=4
        )
        self.kernel_timeout = clamp_int(
            self.cfg.get("rlm_kernel_timeout_sec"), default=0, min_v=0, max_v=3600
        )
        self.tool_ids = normalize_tool_id_list(self.cfg.get("tool_ids"))
        self.tool_specs: List[Dict[str, Any]] = [
            s for s in (self.cfg.get("tool_specs") or []) if isinstance(s, dict)
        ]
        self.tool_runs: List[Dict[str, Any]] = []
        sub = normalize_tool_id_list(self.cfg.get("subagent_tool_ids"))
        self.subagent_tool_ids = sub if sub else list(self.tool_ids)
        self.model = str(self.cfg.get("model") or "").strip()
        # Catalog logical_id for this agent node (same as cfg.model in RLM).
        self.model_logical_id = self.model
        self.run_lifetime = normalize_run_lifetime(self.cfg.get("run_lifetime"))
        self.heartbeat_interval_sec = clamp_int(
            self.cfg.get("heartbeat_interval_sec"), default=300, min_v=30, max_v=86400
        )
        self.heartbeat_prompt = str(self.cfg.get("heartbeat_prompt") or "")
        self.refine_enabled = coerce_bool(self.cfg.get("refine_enabled"), default=True)
        self.daemon_enabled = coerce_bool(self.cfg.get("daemon_enabled"), default=True)
        self.refine_auto_enabled = coerce_bool(
            self.cfg.get("refine_auto_enabled"), default=True
        )
        self.refine_turn_interval = clamp_int(
            self.cfg.get("refine_turn_interval"), default=25, min_v=1, max_v=500
        )
        self.refine_cooldown_sec = clamp_int(
            self.cfg.get("refine_cooldown_sec"), default=1200, min_v=0, max_v=86400
        )
        self.refine_on_compact = coerce_bool(
            self.cfg.get("refine_on_compact"), default=True
        )
        self._apply_refine_env_overrides()
        self._assistant_turns_since_refine = 0
        self._last_refine_review_at = 0.0
        self._compact_refine_pending = False
        self._refine_in_progress = False

        # Durable harness: one JSON per (user, flow, node_id, model_logical_id).
        self.harness = self._open_node_model_harness()

        self.jobs = SessionJobs()
        if self.run_lifetime == "heartbeat":
            self.jobs.heartbeat_create(
                interval_sec=self.heartbeat_interval_sec,
                prompt=self.heartbeat_prompt,
                name="node_heartbeat",
            )
        # Unattended one_shot defaults to Prime-style autonomous continuation.
        # chat / heartbeat stay turn- or tick-driven (no autonomous keep-alive).
        auto_default = self.run_lifetime == "one_shot"
        auto_cfg = self.cfg.get("rlm_autonomous")
        if isinstance(auto_cfg, dict):
            auto_enabled = coerce_bool(auto_cfg.get("enabled"), default=auto_default)
            auto_limits = dict(auto_cfg)
        else:
            auto_enabled = coerce_bool(auto_cfg, default=auto_default)
            # Product default: all caps 0 (= unlimited). Explicit positive values
            # in cfg remain for unit tests that construct sessions directly.
            max_cont = self.cfg.get("rlm_autonomous_max_continuations")
            max_cont_total = self.cfg.get("rlm_autonomous_max_continuations_total")
            if max_cont_total is None:
                max_cont_total = max_cont if max_cont is not None else 0
            auto_limits = {
                "enabled": auto_enabled,
                "max_continuations": 0 if max_cont is None else max_cont,
                "max_continuations_total": (
                    0 if max_cont_total is None else max_cont_total
                ),
                "max_turns": self.cfg.get("rlm_autonomous_max_turns")
                if self.cfg.get("rlm_autonomous_max_turns") is not None
                else 0,
                "max_tokens": self.cfg.get("rlm_autonomous_max_tokens")
                if self.cfg.get("rlm_autonomous_max_tokens") is not None
                else 0,
                "timeout_ms": self.cfg.get("rlm_autonomous_timeout_ms"),
                "continuation_prompt": self.cfg.get("rlm_autonomous_continuation_prompt"),
                "gates": self.cfg.get("rlm_autonomous_gates"),
            }
        gates_cfg = auto_limits.get("gates")
        if gates_cfg is None and self.cfg.get("rlm_autonomous_gates") is not None:
            auto_limits["gates"] = self.cfg.get("rlm_autonomous_gates")
        self.jobs.autonomous = create_autonomous_runtime_state(
            auto_limits,
            default_enabled=auto_enabled,
            default_max_turns=0,
        )
        # Force enabled flag from resolved default when cfg omitted nested enabled.
        self.jobs.autonomous.enabled = auto_enabled
        if auto_enabled and self.jobs.autonomous.started_at is None:
            self.jobs.autonomous.started_at = time.time()

        self.status = "idle"
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.last_content = ""
        self.last_reasoning = ""
        # Live stream may grow the open ``[reasoning]`` line; a new LLM turn
        # appends a fresh line so prior thoughts stay in chronological order.
        self._reasoning_open = False
        # Identical silent ipython loop detector (exact code + bare "(empty)").
        self._empty_ipython_last_code: Optional[str] = None
        self._empty_ipython_streak: int = 0
        # After compact: catch identical ipython/look loops before the model spins.
        self._post_compact_guard: bool = False
        self._post_compact_last_key: Optional[str] = None
        self._post_compact_streak: int = 0
        self._post_compact_distinct_ok: int = 0
        self._post_compact_turns_left: int = 0
        self._last_compact_next_steps: str = ""
        # Consecutive *truly blank* assistant turns (no text, no tools, no reasoning).
        # Thinking-only turns use the existing thinking_only continuation and must
        # not trip this hard stop.
        self._empty_assistant_streak: int = 0
        self._empty_assistant_exhausted_emitted: bool = False
        # True once the session has entered "running" at least once — used so
        # wait_until_paused does not treat virgin idle as "already paused".
        self._was_running = False
        self._suppress_autonomous_continuation = False
        self._autonomous_stop_reason: Optional[str] = None
        self._last_quiescent_turn_content = ""
        self._malformed_tool_args_nudges = 0
        self._context_budget_overflow = False
        self._context_overflow_exhausted = False
        # Prime semantics: terminal evidence is the *final* turn. Any tool call
        # after a stamped answer makes that answer stale (work resumed).
        self._deliverable_stale = False
        # After spawn / child terminal: parent must fan-in before a deliverable counts.
        self._fanin_pending = False
        # Parent ended a turn while waiting on children (status/awaiting text).
        # Host waits silently and buffers child/rlm inbox until children settle —
        # without disabling real parent tool-work on earlier/later turns.
        self._awaiting_children_quiet = False
        self._turn_budget_exhausted = False
        self._turn_budget_notice_sent = False
        self.kernel_env: Optional[Dict[str, Any]] = None
        self.autonomous_strict = coerce_bool(
            self.cfg.get("rlm_autonomous_strict"), default=False
        )
        # Data Pool / gates imply artifact mode even when strict toggle is off.
        if self.pool_defs or (
            self.jobs.autonomous is not None and gates_configured(self.jobs.autonomous)
        ):
            self.autonomous_strict = True
        self._flow_released = False
        self._completion_reviews_used = 0
        self._completion_review_passed = False
        self._completion_review_budget_emitted = False
        self._last_completion_review_fp = ""
        self._last_completion_review_reason = ""
        self._completion_review_unchanged_streak = 0
        self.acceptance_contract: Optional[Dict[str, Any]] = None
        self.acceptance_probe_results: List[Dict[str, Any]] = []
        self.last_worker_claim: Optional[Dict[str, Any]] = None
        self._acceptance_contract_building = False
        self._children_wait_elapsed = 0.0
        self._children_wait_extensions = 0
        self._parent_reply_count = 0
        self._replied_to_parent_since_task = False
        self.error: Optional[str] = None
        self.messages: List[Dict[str, Any]] = []
        self.thoughts: List[Dict[str, Any]] = []
        self.inbox: List[Dict[str, Any]] = []
        self._inbox_llm_cursor = 0
        self._inbox_py_cursor = 0
        self.children: Dict[str, Dict[str, Any]] = {}
        self._child_sessions: Dict[str, RlmSession] = {}
        self._child_tasks: Dict[str, asyncio.Task] = {}
        self.events: List[Dict[str, Any]] = []
        self.log_lines: List[str] = []
        self.deliverable_artifacts: Dict[str, str] = {}
        self.deliverable_fingerprints: Dict[str, Tuple[int, int]] = {}
        self._emitted_deliverable_names: set[str] = set()
        self._cached_files: List[Dict[str, Any]] = []
        self.kernel: Optional[RlmKernel] = None
        self._hit_index: Dict[str, List[Dict[str, Any]]] = {}
        self._pending_looks: List[Dict[str, Any]] = []
        # Prime-like: idle | attempted — at most one compact+retry per overflow.
        self._overflow_recovery: str = "idle"
        self._look_media_invalid_recovery: bool = False
        self._looks_injected_this_turn = False
        self._closed = False
        self._turns_lock = asyncio.Lock()
        self.snapshot_path = os.path.join(self.artifact_dir, "kernel-state.dill")
        self._transcript_written = 0
        self._persist_dirty = False
        self._persist_timer: Optional[asyncio.TimerHandle] = None
        self._persist_task: Optional["asyncio.Task[None]"] = None
        self._restored_from_disk = False
        self.session_fingerprint = self.compute_session_fingerprint(self.cfg)
        self._ensure_transcript_header()

    def _apply_refine_env_overrides(self) -> None:
        from src.orchestrators.rlm.runtime.auto_refine import apply_env_overrides

        apply_env_overrides(self)

    @staticmethod
    def compute_session_fingerprint(cfg: Optional[Dict[str, Any]]) -> str:
        import hashlib

        c = cfg if isinstance(cfg, dict) else {}
        payload = {
            "model": str(c.get("model") or "").strip(),
            "system_prompt": str(c.get("system_prompt") or ""),
            "tool_ids": sorted(normalize_tool_id_list(c.get("tool_ids"))),
            "run_lifetime": normalize_run_lifetime(c.get("run_lifetime")),
            "execution_mode": str(c.get("execution_mode") or "rlm"),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _ensure_transcript_header(self) -> None:
        try:
            from src.orchestrators.rlm.runtime.transcript import (
                write_session_header,
            )

            write_session_header(
                self.artifact_dir,
                session_id=self.session_id,
                run_id=str(self.run_id or ""),
                flow_id=self.flow_id,
                node_id=self.node_id,
                user_id=self.user_id,
                rlm_depth=self.depth,
            )
        except Exception:
            pass

    def sync_transcript(self) -> None:
        """Append new in-memory messages (skip system) onto transcript.full.jsonl."""
        try:
            from src.orchestrators.rlm.runtime.transcript import append_message

            while self._transcript_written < len(self.messages):
                msg = self.messages[self._transcript_written]
                self._transcript_written += 1
                if not isinstance(msg, dict):
                    continue
                if str(msg.get("role") or "") == "system":
                    continue
                append_message(self.artifact_dir, msg)
        except Exception:
            pass

    def hydrate_messages_from_disk(self, *, base_system: str = "") -> bool:
        """Rebuild ``self.messages`` from transcript.full.jsonl. Returns True if loaded."""
        try:
            from src.orchestrators.rlm.runtime.transcript import (
                FULL_TRANSCRIPT_NAME,
                hydrate_messages_from_disk,
            )

            path = os.path.join(self.artifact_dir, FULL_TRANSCRIPT_NAME)
            if not os.path.isfile(path) or os.path.getsize(path) < 8:
                return False
            system = self.build_system_prompt(base_system)
            msgs = hydrate_messages_from_disk(self.artifact_dir, system_prompt=system)
            if not msgs:
                return False
            # Header-only transcript is not a real resume (avoid false "resumed").
            if not any(
                isinstance(m, dict) and str(m.get("role") or "") != "system" for m in msgs
            ):
                return False
            self.messages = msgs
            # Everything currently in memory is already durable on disk.
            self._transcript_written = len(self.messages)
            self._restored_from_disk = True
            return True
        except Exception:
            return False

    def inject_ipython_state_notice(
        self, *, restored: bool, names: Optional[List[str]] = None
    ) -> None:
        """Hidden user message so the model knows which kernel vars survived."""
        names = [n for n in (names or []) if isinstance(n, str) and n.strip()]
        if restored:
            custom_type = "ipython_state_restored"
            if names:
                body = (
                    "<ipython_state_restored>\n"
                    "Your IPython kernel was restored from a previous turn. "
                    f"These names are still defined: {', '.join(names[:80])}.\n"
                    "</ipython_state_restored>"
                )
            else:
                body = (
                    "<ipython_state_restored>\n"
                    "Kernel snapshot restore ran but no user variables were revived. "
                    "Re-create any variables you need.\n"
                    "</ipython_state_restored>"
                )
        else:
            custom_type = "ipython_state"
            body = (
                "<ipython_state>\n"
                "Your IPython kernel persisted through compaction; all variables, "
                "imports, and helpers you defined remain available. "
                + (
                    f"These names are still defined: {', '.join(names[:80])}."
                    if names
                    else "Probe with dir() if unsure."
                )
                + "\n</ipython_state>"
            )
        self.messages.append({"role": "user", "content": body})
        # Persist as custom_message (not a plain message entry); skip sync for this row.
        self._transcript_written = len(self.messages)
        try:
            from src.orchestrators.rlm.runtime.transcript import (
                append_custom_message,
            )

            append_custom_message(
                self.artifact_dir,
                custom_type=custom_type,
                content=body,
                display=False,
            )
        except Exception:
            pass

    async def list_kernel_names(self) -> List[str]:
        if not self.kernel or not self.kernel.alive:
            return []
        try:
            return await self.kernel.list_namespace_names()
        except Exception:
            return []

    async def flush_kernel_snapshot(self) -> bool:
        if not self.kernel or not self.kernel.alive or not self.snapshot_path:
            return False
        try:
            return bool(await self.kernel.dump_snapshot(self.snapshot_path))
        except Exception:
            return False

    def harness_path(self) -> str:
        """Absolute path of this session's node+model harness JSON."""
        from src.orchestrators.rlm.runtime.daemon import RlmDaemon

        return RlmDaemon.instance().harness_path_for(
            flow_id=self.flow_id,
            user_id=self.user_id,
            node_id=self.node_id,
            model_logical_id=self.model_logical_id or self.model,
        )

    def _open_node_model_harness(self) -> HarnessStore:
        try:
            return HarnessStore.shared(self.harness_path())
        except Exception:
            # Fallback: ephemeral store (still works for the run; not durable).
            return HarnessStore(path=None)

    def harness_store_for(self, scope: str = "") -> HarnessStore:
        """Compatibility shim — all writes go to the node+model store."""
        _ = scope
        return self.harness

    def overlay_harness_text(self) -> str:
        body = self.harness.overlay_text() if self.harness is not None else ""
        return body.strip()

    def merged_refinement_history(self, *, limit: int = 20) -> List[Dict[str, Any]]:
        if self.harness is None:
            return []
        return self.harness.list_refinements(limit=limit)

    def _touch(self) -> None:
        self.updated_at = time.time()

    def _persist_local_now(self) -> None:
        try:
            self.sync_transcript()
        except Exception:
            pass
        try:
            persist_session_files(self)
        except Exception:
            pass

    def _cancel_persist_timer(self) -> None:
        timer = self._persist_timer
        self._persist_timer = None
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass

    def persist_local(self, *, flush: bool = False) -> None:
        """Debounced off-loop persist; synchronous when flushing or without a loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if flush or loop is None or loop.is_closed() or self._closed:
            self._cancel_persist_timer()
            self._persist_dirty = False
            self._persist_local_now()
            return
        self._persist_dirty = True
        if self._persist_timer is not None:
            return
        if self._persist_task is not None and not self._persist_task.done():
            return
        try:
            self._persist_timer = loop.call_later(PERSIST_DEBOUNCE_SEC, self._fire_persist)
        except Exception:
            self._persist_dirty = False
            self._persist_local_now()

    def _fire_persist(self) -> None:
        self._persist_timer = None
        if not self._persist_dirty:
            return
        self._persist_dirty = False
        try:
            self.sync_transcript()
        except Exception:
            pass

        async def _bg() -> None:
            try:
                await asyncio.to_thread(
                    _persist_files_in_worker, self, asyncio.get_running_loop()
                )
            except Exception:
                pass
            finally:
                self._persist_task = None
                if self._persist_dirty:
                    self.persist_local()

        try:
            self._persist_task = asyncio.get_running_loop().create_task(_bg())
        except Exception:
            self._persist_task = None
            self._persist_local_now()

    def _emit(self, kind: str, **fields: Any) -> None:
        persist = bool(fields.pop("persist", True))
        worker_loop = getattr(_persist_worker, "loop", None)
        if worker_loop is not None and not worker_loop.is_closed():
            # Background persist scans may discover [file] events — hop back so
            # on_event / asyncio queues are only touched from the loop thread.
            try:
                worker_loop.call_soon_threadsafe(
                    functools.partial(self._emit, kind, persist=persist, **fields)
                )
                return
            except RuntimeError:
                pass
        ev = {"ts": time.time(), "kind": kind, "session_id": self.session_id, **fields}
        self.events.append(ev)
        if len(self.events) > 500:
            self.events = self.events[-500:]
        line = format_rlm_log_line(kind, fields)
        if line:
            self.log_lines.append(line[:LOG_LINE_CHARS])
            if len(self.log_lines) > LOG_LINES_KEEP:
                self.log_lines = self.log_lines[-LOG_LINES_KEEP:]
            try:
                from src.orchestrators.rlm.runtime.persist import (
                    append_terminal_log,
                )

                append_terminal_log(
                    self.artifact_dir,
                    line[:LOG_LINE_CHARS],
                    ts=float(ev.get("ts") or time.time()),
                )
            except Exception:
                pass
        self._touch()
        # Notify UI BEFORE disk I/O — persist_local can stall the event loop for
        # seconds and made the terminal look frozen right after [chat].
        if self.on_event:
            try:
                # Always attach a live snapshot so progress polls don't wait on
                # a later stream_progress heartbeat.
                if "rlm" not in ev:
                    ev["rlm"] = self.live_progress_snapshot(
                        focus_session_id=self._ui_focus_session_id()
                    )
                self.on_event(ev)
            except Exception:
                pass
        if persist:
            self.persist_local(flush=kind in PERSIST_FLUSH_KINDS)

    def _remember_thought(self, kind: str, text: str) -> None:
        clipped = str(text or "").strip()
        if not clipped:
            return
        if kind in ("reasoning", "thinking"):
            self.last_reasoning = clipped
        limit = 4000 if kind in ("reasoning", "thinking") else 2000
        rec = {"ts": time.time(), "kind": kind, "text": clipped[:limit]}
        self.thoughts.append(rec)
        if len(self.thoughts) > 80:
            self.thoughts = self.thoughts[-80:]
        preview_n = 4000 if kind in ("reasoning", "thinking") else 1200
        self._emit("thought", thought_kind=kind, preview=clipped[:preview_n])

    async def host_request(self, request_type: str, payload: Dict[str, Any]) -> Any:
        from src.orchestrators.rlm.runtime import handlers as h

        return await h.dispatch(self, request_type, payload or {})

    def _tool_setting(self, key: str) -> int:
        """Tool-surface sizing: session cfg, else the RLM manifest default."""
        from src.orchestrators._shared.manifest_config import manifest_defaults, package_manifest

        fallback = manifest_defaults(package_manifest(__file__)).get(key, 0)
        try:
            return int(self.cfg.get(key, fallback) or 0)
        except (TypeError, ValueError):
            return int(fallback or 0)

    @property
    def tool_namespace(self) -> str:
        """Kernel name of the tools object: session cfg, else this engine's tools_binding.surface."""
        from src.orchestrators._shared.manifest_config import package_manifest

        declared = self.cfg.get("tool_namespace")
        if not declared:
            binding = getattr(package_manifest(__file__), "tools_binding", None) or {}
            declared = (binding.get("surface") or {}).get("namespace") if isinstance(binding, dict) else None
        return surface_namespace({"namespace": declared})

    def tool_surface_options(self) -> Dict[str, Any]:
        return {
            "namespace": self.tool_namespace,
            "description_chars": self._tool_setting("rlm_tool_description_chars"),
            "max_params": self._tool_setting("rlm_tool_signature_max_params"),
            "usage_hint": self._tool_text_setting("rlm_tool_usage_hint"),
        }

    def _tool_text_setting(self, key: str) -> str:
        """Tool-surface text: session cfg (empty string disables), else the RLM manifest default."""
        from src.orchestrators._shared.manifest_config import manifest_defaults, package_manifest

        if key in self.cfg and self.cfg.get(key) is not None:
            return str(self.cfg.get(key))
        return str(manifest_defaults(package_manifest(__file__)).get(key) or "")

    async def refresh_tool_surface(self, tool_ids: Any, tool_specs: Any) -> None:
        """Adopt this turn's host-selected tools: allowlist, kernel binding, prompt."""
        ids = normalize_tool_id_list(tool_ids)
        specs = [s for s in (tool_specs or []) if isinstance(s, dict)]
        changed = ids != self.tool_ids or specs != self.tool_specs
        self.tool_ids, self.tool_specs = ids, specs
        self.cfg["tool_specs"] = specs
        if not normalize_tool_id_list(self.cfg.get("subagent_tool_ids")):
            self.subagent_tool_ids = list(ids)
        if not changed:
            return
        if self.kernel is not None:
            try:
                await self.kernel.set_tools(ids, namespace=self.tool_namespace)
            except Exception:
                logger.warning("rlm kernel tool rebind failed", exc_info=True)
        head = self.messages[0] if self.messages else None
        if isinstance(head, dict) and head.get("role") == "system" and isinstance(head.get("content"), str):
            section = format_tool_surface(specs, ids, **self.tool_surface_options())
            self.messages[0] = {**head, "content": replace_tool_surface(head["content"], section)}
        self._emit("tools_refreshed", count=len(ids))

    def refresh_host_context(self, blocks: Dict[str, str]) -> None:
        """Swap this turn's host prompt blocks into the system message (never accumulates)."""
        head = self.messages[0] if self.messages else None
        if isinstance(head, dict) and head.get("role") == "system" and isinstance(head.get("content"), str):
            updated = replace_host_context(head["content"], blocks)
            if updated != head["content"]:
                self.messages[0] = {**head, "content": updated}
                self._emit("host_context_refreshed", blocks=sorted(blocks))

    async def call_project_tool(self, tool_id: str, args: Any) -> Any:
        """Run one allowed project tool through the host port and record it."""
        tid = str(tool_id or "").strip()
        if tid not in set(self.tool_ids):
            raise PermissionError(f"tool_not_allowed:{tid}")
        if self.tool_runner is None:
            raise RuntimeError("tool_runner_unavailable")
        clean = {str(k): v for k, v in args.items() if not str(k).startswith("_")} if isinstance(args, dict) else {}
        run: Dict[str, Any] = {
            "id": f"rlm_tool_{uuid.uuid4().hex[:12]}",
            "tool_id": tid,
            "args": clean,
            "step": len(self.tool_runs) + 1,
        }
        self._emit("tool_start", run_id=run["id"], tool_id=tid, args=clean, step=run["step"])
        started = time.monotonic()
        try:
            result = await self.tool_runner(tid, clean)
        except Exception as exc:  # noqa: BLE001 — the model receives the failure as data
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "tool_id": tid}
        ok = not (isinstance(result, dict) and (result.get("ok") is False or bool(result.get("error"))))
        run.update(ok=ok, result=result, execution_time=round(time.monotonic() - started, 3))
        self.tool_runs.append(run)
        limit = self._tool_setting("rlm_tool_result_preview_chars")
        self._emit(
            "tool_result",
            run_id=run["id"],
            tool_id=tid,
            args=clean,
            ok=ok,
            step=run["step"],
            execution_time=run["execution_time"],
            summary=_tool_result_preview(result, limit),
            error=None if ok else _tool_result_preview((result or {}).get("error"), limit),
        )
        return result

    async def _non_ipython_tool_result(self, name: str, args: Dict[str, Any]) -> str:
        """A native call to an allowed project tool runs it; anything else is refused."""
        raw = str(name or "").strip()
        prefix = f"{self.tool_namespace}."
        bare = raw[len(prefix):] if raw.startswith(prefix) else raw
        by_alias = {tool_alias(t): t for t in self.tool_ids}
        tid = bare if bare in set(self.tool_ids) else by_alias.get(bare, "")
        if not tid and bare == self.tool_namespace and isinstance(args, dict):
            # ``tools(<key>=<tool id>, **kwargs)``: the namespace called as a dispatcher.
            for key, value in list(args.items()):
                if isinstance(value, str) and (value in set(self.tool_ids) or value in by_alias):
                    tid = value if value in set(self.tool_ids) else by_alias[value]
                    args = {k: v for k, v in args.items() if k != key}
                    break
        if tid:
            name = tid
            result = await self.call_project_tool(name, args)
            body = self._maybe_offload_tool_content(
                json.dumps(result, ensure_ascii=False, default=str), kind="json"
            )
            return f"{body}\n\n[rlm] In {CELL_TOOL_NAME} this tool is {call_hint(name, self.tool_namespace)}."
        return reject_non_ipython_tool(
            name, pool_defs=self.pool_defs, tool_ids=self.tool_ids, namespace=self.tool_namespace
        )

    async def start_kernel(self) -> None:
        if self.kernel and self.kernel.alive:
            return
        from src.orchestrators.rlm.runtime.resource_budget import kernel_governor

        if not self._kernel_slot_acquired:

            async def _kernel_queue_waiting(status: Dict[str, Any]) -> None:
                queue = int(status.get("waiting") or 0)
                active = int(status.get("active") or 0)
                cpu = float(status.get("load_ratio") or 0.0)
                mem = float(status.get("mem_used_ratio") or 0.0)
                avail = int(status.get("available_mb") or 0)
                self._upsert_log_line(
                    "kernel_queue",
                    (
                        f"waiting for kernel slot — queue={queue} active={active} "
                        f"cpu={cpu:.0%} mem={mem:.0%} free={avail}MB"
                    ),
                )
                self._publish_stream_progress(force=True)

            await kernel_governor().acquire(
                self.user_id,
                on_waiting=_kernel_queue_waiting,
                cancel_check=lambda: bool(
                    self._closed or self.status in ("cancelled", "error")
                ),
            )
            self._kernel_slot_acquired = True
            self._upsert_log_line("kernel_queue", "kernel slot acquired — starting")
            self._publish_stream_progress(force=True)
        try:
            self.kernel = RlmKernel(
                host_handler=self.host_request,
                timeout_sec=float(self.kernel_timeout),
                cwd=self.workspace_dir,
                snapshot_path=self.snapshot_path,
            )
            meta = await self.kernel.bootstrap(
                pools=self.pool_defs, tools=self.tool_ids, tool_namespace=self.tool_namespace
            )
        except BaseException:
            self.kernel = None
            if self._kernel_slot_acquired:
                self._kernel_slot_acquired = False
                await kernel_governor().release(self.user_id)
            raise
        self._emit("kernel_ready")
        await self._probe_kernel_environment()
        if isinstance(meta, dict) and meta.get("restored"):
            names = list(meta.get("names") or [])
            self.inject_ipython_state_notice(restored=True, names=names)
            try:
                from src.orchestrators.rlm.runtime.transcript import (
                    append_kernel_state,
                )

                append_kernel_state(
                    self.artifact_dir,
                    op="restore",
                    restored=names,
                )
            except Exception:
                pass

    async def _probe_kernel_environment(self) -> None:
        """Learn which optional packages exist so the model never burns turns probing."""
        if self.kernel is None or self.kernel_env is not None:
            return
        try:
            result = await asyncio.wait_for(
                self.kernel.execute(KERNEL_ENV_PROBE_CODE, timeout_sec=15.0),
                timeout=20.0,
            )
        except Exception:
            logger.debug("rlm kernel env probe failed", exc_info=True)
            return
        env = parse_kernel_env_probe(str((result or {}).get("stdout") or ""))
        if env is None:
            return
        self.kernel_env = env
        self._emit(
            "kernel_env",
            python=env.get("python"),
            available=",".join(env.get("available") or []),
            missing=len(env.get("missing") or []),
        )

    def build_system_prompt(self, base_system: str) -> str:
        from src.orchestrators.rlm.runtime.transcript import (
            full_transcript_path,
        )

        parts = [RLM_SYSTEM_APPENDIX.strip()]
        meta = build_rlm_session_meta(
            cwd=self.workspace_dir,
            depth=self.depth,
            messages_path=full_transcript_path(self.artifact_dir),
            max_depth=self.max_depth,
        )
        if meta:
            parts.append(meta)
        parts.append(build_recursion_capabilities())
        howto = IPYTHON_HOWTO.strip()
        if howto:
            parts.append(howto)
        doctrine_done = build_completion_doctrine(
            artifact_required=self.artifact_required()
        )
        if doctrine_done:
            parts.append(doctrine_done)
        parts.append(build_subagent_guidance())
        env_notice = format_kernel_env_notice(self.kernel_env)
        if env_notice:
            parts.append(env_notice)
        refine = REFINE_GUIDANCE.strip()
        if refine:
            parts.append(refine)
        doctrine = build_child_agent_doctrine(
            depth=self.depth,
            parent_name=str(
                getattr(self, "_parent_name", None)
                or self.cfg.get("_rlm_parent_name")
                or ""
            ),
            child_name=str(self.cfg.get("_rlm_child_name") or ""),
        )
        if doctrine:
            parts.append("## Child agent doctrine\n" + doctrine)
        base, host_context = split_host_context(base_system or "")
        if base:
            parts.append("## Base system prompt (immutable)\n" + base)
        overlay = self.overlay_harness_text()
        if overlay:
            parts.append("## Continual Harness (supplemental)\n" + overlay)
        inventory = format_attached_surface(
            self.pool_defs,
            self.tool_ids,
            cwd=self.workspace_dir,
            input_modalities=self.cfg.get("input_modalities"),
            tool_specs=self.tool_specs,
            tool_surface_options=self.tool_surface_options(),
        )
        if inventory:
            parts.append(inventory)
        # Per-turn host context last, where refresh_host_context keeps it.
        if host_context:
            parts.append(host_context)
        return "\n\n".join(parts)

    def thinking_enabled(self) -> bool:
        """Prime-style: thinking is a request flag, not a silent always-on.

        Precedence: gen_params / cfg.enable_thinking → catalog wire default
        stamped onto cfg → True (RLM prefers thinking when unknown).
        """
        ctk = self.gen_params.get("chat_template_kwargs")
        if isinstance(ctk, dict) and "enable_thinking" in ctk:
            return coerce_bool(ctk.get("enable_thinking"), default=True)
        if "enable_thinking" in (self.cfg or {}):
            return coerce_bool(self.cfg.get("enable_thinking"), default=True)
        if "enable_thinking_default" in (self.cfg or {}):
            return coerce_bool(self.cfg.get("enable_thinking_default"), default=True)
        return True

    def _apply_thinking_kwargs(self, body: Dict[str, Any]) -> bool:
        thinking_on = self.thinking_enabled()
        ctk = body.get("chat_template_kwargs")
        ctk = dict(ctk) if isinstance(ctk, dict) else {}
        ctk["enable_thinking"] = thinking_on
        if thinking_on:
            # Catalog/cfg/gen_params win; name heuristic only when unset.
            if "preserve_thinking" not in ctk or ctk.get("preserve_thinking") is None:
                ctk["preserve_thinking"] = resolve_preserve_thinking(
                    model=self.model,
                    cfg=self.cfg if isinstance(self.cfg, dict) else None,
                    gen_params=self.gen_params if isinstance(self.gen_params, dict) else None,
                )
            else:
                ctk["preserve_thinking"] = coerce_bool(
                    ctk.get("preserve_thinking"), default=False
                )
        else:
            ctk.pop("preserve_thinking", None)
        body["chat_template_kwargs"] = ctk
        # Optional intensity (e.g. reasoning_effort) from catalog when unset.
        inten_param = str((self.cfg or {}).get("reasoning_intensity_param") or "").strip()
        if inten_param and inten_param not in body and thinking_on:
            inten_default = (self.cfg or {}).get(inten_param)
            if inten_default is None:
                inten_default = (self.cfg or {}).get("reasoning_intensity_default")
            if inten_default is not None:
                body[inten_param] = inten_default
        return thinking_on

    def _assistant_history_message(
        self, msg: Dict[str, Any], *, content: str, reasoning: str = ""
    ) -> Dict[str, Any]:
        """Store visible content + tool calls; keep CoT in session for UI/replay.

        Full ``reasoning_content`` stays on the session transcript. Outbound
        requests trim old CoT via ``outbound_messages_trim_reasoning`` so Gemma
        does not accumulate every prior plan into the context window.
        """
        stored: Dict[str, Any] = {"role": "assistant", "content": content or None}
        tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else None
        if tool_calls:
            stored["tool_calls"] = tool_calls
            if not content:
                stored["content"] = None
        elif stored["content"] is None:
            stored["content"] = ""
        reason = str(reasoning or "").strip()
        if not reason and isinstance(msg, dict):
            reason = str(
                msg.get("reasoning_content")
                or msg.get("reasoning")
                or msg.get("reasoning_text")
                or ""
            ).strip()
        if reason and self.thinking_enabled():
            stored["reasoning_content"] = reason
        return stored

    def _sync_child_registry_status(self) -> None:
        """Heal stale ``running`` rows when the child task/session already finished."""
        for cid, rec in list(self.children.items()):
            if not isinstance(rec, dict):
                continue
            if str(rec.get("status") or "") != "running":
                continue
            task = self._child_tasks.get(cid)
            child = self._child_sessions.get(cid)
            if task is not None and not task.done():
                continue
            if child is not None and not getattr(child, "_closed", False):
                cst = str(getattr(child, "status", "") or "")
                if cst in ("running", "idle"):
                    continue
                if cst in ("completed", "error", "cancelled", "deleted"):
                    rec["status"] = cst
                    if cst == "error" and getattr(child, "error", None):
                        rec["error"] = str(child.error)
                    continue
            # Only heal when we have a finished task (real spawn). Status-only
            # rows without a task stay ``running`` so wait gates keep working.
            if task is not None and task.done():
                if task.cancelled():
                    rec["status"] = "error"
                    rec["error"] = "cancelled"
                    continue
                exc = task.exception()
                if exc is not None:
                    rec["status"] = "error"
                    rec["error"] = str(exc)
                else:
                    rec["status"] = "completed"

    def _has_running_children(self) -> bool:
        self._sync_child_registry_status()
        for cid, rec in self.children.items():
            if str(rec.get("status") or "") == "running":
                return True
            task = self._child_tasks.get(cid)
            if task is not None and not task.done():
                return True
            child = self._child_sessions.get(cid)
            if (
                child is not None
                and not getattr(child, "_closed", False)
                and str(getattr(child, "status", "") or "") in ("running", "idle")
            ):
                return True
        return False

    def _fanin_buffer_active(self) -> bool:
        """True while one_shot/chat parent should buffer mid-wait child/rlm inbox."""
        if self.run_lifetime not in ("one_shot", "chat"):
            return False
        if not self._has_running_children():
            return False
        return bool(
            getattr(self, "_fanin_pending", False)
            or getattr(self, "_awaiting_children_quiet", False)
        )

    def _inbox_item_deferred_while_children(self, item: Dict[str, Any]) -> bool:
        """Defer child/runtime notices until children settle (not user interrupts)."""
        if not self._fanin_buffer_active():
            return False
        if not isinstance(item, dict):
            return False
        role = str(item.get("from_role") or "").strip().lower()
        kind = str(item.get("kind") or "").strip().lower()
        if role in ("child", "rlm"):
            return True
        if kind in (
            "failed",
            "cancelled",
            "completed_without_reply",
            "child_error",
            "child_failed",
        ):
            return True
        return False

    def _inbox_llm_pending(self) -> bool:
        """True when there is inbox ready for an LLM turn (skips buffered fan-in)."""
        for item in self.inbox[self._inbox_llm_cursor :]:
            if not self._inbox_item_deferred_while_children(item):
                return True
        return False

    def _mark_awaiting_children_quiet(self, *, reason: str = "") -> None:
        if getattr(self, "_awaiting_children_quiet", False):
            return
        self._awaiting_children_quiet = True
        self._mark_fanin_pending(reason=reason or "awaiting_children_quiet")
        self._upsert_log_line(
            "waiting",
            "children running — parent idle until fan-in (inbox buffered)",
        )
        self._emit("awaiting_children_quiet", reason=reason or "awaiting")
        self._publish_stream_progress(force=True)

    def _clear_awaiting_children_quiet(self) -> None:
        if not getattr(self, "_awaiting_children_quiet", False):
            return
        self._awaiting_children_quiet = False
        self._upsert_log_line("waiting", "children settled — resuming parent")
        self._emit("awaiting_children_resumed")
        self._publish_stream_progress(force=True)

    def children_wait_budget_sec(self) -> float:
        """Absolute wait budget while children run (Prime: stay alive for fan-in)."""
        turns = int(self.rlm_max_turns or 0)
        if turns <= 0:
            turns = 120  # soft estimate for wait only; loop itself is unlimited
        turns = max(1, turns)
        return float(max(300, min(3600, turns * 60)))

    def _child_wait_slice_sec(self) -> float:
        """One polling slice; never the whole budget."""
        if float(self.kernel_timeout) > 0:
            return float(max(1.0, min(30.0, float(self.kernel_timeout))))
        return 15.0

    def _assess_running_children(self) -> List[Dict[str, Any]]:
        """Peek at live children: recent activity ⇒ healthy; silent ⇒ stale."""
        self._sync_child_registry_status()
        now = time.time()
        rows: List[Dict[str, Any]] = []
        for cid, rec in self.children.items():
            if not isinstance(rec, dict):
                continue
            task = self._child_tasks.get(cid)
            child = self._child_sessions.get(cid)
            st = str(rec.get("status") or "")
            still = st == "running" or (
                task is not None and not task.done()
            ) or (
                child is not None
                and not getattr(child, "_closed", False)
                and str(getattr(child, "status", "") or "") in ("running", "idle")
            )
            if not still:
                continue
            updated = 0.0
            logs: List[str] = []
            if child is not None:
                try:
                    updated = float(getattr(child, "updated_at", 0) or 0)
                except (TypeError, ValueError):
                    updated = 0.0
                logs = [str(x) for x in (getattr(child, "log_lines", []) or [])[-8:]]
            if not updated:
                try:
                    updated = float(rec.get("created_at") or 0)
                except (TypeError, ValueError):
                    updated = 0.0
            idle_sec = (now - updated) if updated > 0 else 9999.0
            # Recent tool/chat lines count as alive even if updated_at is slightly stale.
            recent_kinds = ("[chat]", "[ipython]", "[result]", "[llm]", "[stream]", "[wait]")
            has_live_log = any(
                any(str(ln).startswith(k) for k in recent_kinds) for ln in logs[-3:]
            )
            healthy = idle_sec <= float(CHILD_HEALTHY_IDLE_SEC) or (
                has_live_log and idle_sec <= float(CHILD_HEALTHY_IDLE_SEC) * 2
            )
            rows.append(
                {
                    "rlm_child_id": cid,
                    "name": rec.get("name") or cid,
                    "status": st or "running",
                    "idle_sec": round(idle_sec, 1),
                    "healthy": bool(healthy),
                    "log_tail": logs,
                }
            )
        return rows

    def _format_children_health_log(
        self, assessments: List[Dict[str, Any]], *, reason: str
    ) -> str:
        """Host-side log summary only — not injected into the model transcript."""
        parts = [f"children_wait_budget reason={reason}"]
        if not assessments:
            parts.append("(no running children found)")
            return "\n".join(parts)
        for row in assessments:
            flag = "healthy" if row.get("healthy") else "STALE/stuck?"
            parts.append(
                f"- {row.get('name')} ({row.get('rlm_child_id')}): "
                f"{flag}, idle={row.get('idle_sec')}s, status={row.get('status')}"
            )
            for ln in row.get("log_tail") or []:
                parts.append(f"    {ln}")
        return "\n".join(parts)

    async def _handle_children_wait_budget(self) -> str:
        """When the absolute wait budget is spent: extend or cancel stale kids.

        Host-side only — do not inject health-check user messages into the transcript.
        """
        assessments = self._assess_running_children()
        if not assessments:
            return "clear"
        healthy = [a for a in assessments if a.get("healthy")]
        stale = [a for a in assessments if not a.get("healthy")]
        extensions = int(getattr(self, "_children_wait_extensions", 0) or 0)

        if healthy and extensions < int(MAX_CHILDREN_WAIT_EXTENSIONS):
            self._children_wait_extensions = extensions + 1
            # Grant another slice of wall-clock wait without resetting history.
            self._children_wait_elapsed = max(
                0.0,
                float(self._children_wait_elapsed) - float(CHILD_WAIT_EXTENSION_SEC),
            )
            detail = self._format_children_health_log(
                assessments,
                reason=(
                    f"wait_extended#{self._children_wait_extensions} "
                    f"(+{int(CHILD_WAIT_EXTENSION_SEC)}s; healthy children still active)"
                ),
            )
            self._emit(
                "children_wait_extended",
                extension=self._children_wait_extensions,
                healthy=len(healthy),
                stale=len(stale),
                detail=detail[:2000],
            )
            return "extended"

        detail = self._format_children_health_log(
            assessments,
            reason=(
                "intervene_required" if not healthy else "extensions_exhausted"
            ),
        )
        self._emit(
            "children_wait_intervene",
            healthy=len(healthy),
            stale=len(stale),
            extensions=extensions,
            detail=detail[:2000],
        )

        # Cancel only clearly stale children so healthy ones can finish.
        for row in stale:
            cid = str(row.get("rlm_child_id") or "")
            if not cid:
                continue
            try:
                await self.delete_subagent(cid)
            except Exception:
                logger.warning(
                    "rlm cancel stale child failed id=%s", cid, exc_info=True
                )
            self._emit(
                "child_cancelled_stale",
                rlm_child_id=cid,
                name=row.get("name"),
                idle_sec=row.get("idle_sec"),
            )

        # Small grace window so parent can act / healthy children can finish.
        self._children_wait_elapsed = max(
            0.0, float(self._children_wait_elapsed) - 180.0
        )
        return "intervened"

    async def _ensure_acceptance_contract(self) -> None:
        """Build once per one_shot run when artifact mode needs a checklist."""
        if self.run_lifetime != "one_shot":
            return
        auto = self.jobs.autonomous
        if auto is None or not auto.enabled:
            return
        if getattr(self, "acceptance_contract", None) is not None:
            return
        from src.orchestrators.rlm.runtime.acceptance_contract import (
            MODE_ARTIFACT,
            MODE_CHAT,
            build_acceptance_contract,
            fallback_acceptance_contract,
            infer_run_mode,
        )

        artifact = self.artifact_required()
        mode = infer_run_mode(
            str(self.user_content or ""),
            pools=self.pool_defs,
            gates=bool(auto is not None and gates_configured(auto)),
            strict=bool(getattr(self, "autonomous_strict", False) or artifact),
        )
        if not artifact and mode == MODE_CHAT:
            self.acceptance_contract = {
                "mode": MODE_CHAT,
                "items": [],
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": "chat_skip",
                "goal_preview": str(self.user_content or "")[:500],
            }
            self._emit("acceptance_contract", mode=MODE_CHAT, items=0, source="chat_skip")
            return
        self._acceptance_contract_building = True
        try:
            contract = await build_acceptance_contract(
                self,
                chat_fn=self.chat_fn,
                extract_assistant_fn=self.extract_assistant_fn,
                user_goal=str(self.user_content or ""),
                mode=MODE_ARTIFACT,
            )
        except Exception:
            logger.warning("acceptance contract build failed", exc_info=True)
            contract = fallback_acceptance_contract(
                str(self.user_content or ""), mode=MODE_ARTIFACT
            )
            contract["source"] = "fallback_exception"
        finally:
            self._acceptance_contract_building = False
        self.acceptance_contract = contract
        items = contract.get("items") if isinstance(contract, dict) else []
        self._emit(
            "acceptance_contract",
            mode=str(contract.get("mode") or MODE_ARTIFACT),
            items=len(items or []),
            source=str(contract.get("source") or ""),
        )

    async def _take_continuation_messages(
        self,
        *,
        turn_content: Optional[str] = None,
        turn_reasoning: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Prime-style goal / autonomous continuation at a quiescent turn boundary.

        Goal wins over autonomous. Terminal evidence is judged from *this turn's*
        content only — never from a stale mid-run ``last_content`` status line.
        """
        if self._suppress_autonomous_continuation:
            return []

        # Fan-in before goal/autonomous: a draft answer while/after children is not final.
        if (
            self.run_lifetime in ("one_shot", "chat")
            and getattr(self, "_fanin_pending", False)
            and not self._has_running_children()
        ):
            self._fanin_pending = False
            self._deliverable_stale = True
            msg = build_fanin_continuation_message()
            auto = self.jobs.autonomous
            if auto is not None and auto.enabled:
                auto.continuations_used += 1
                auto.continuations_total += 1
                auto.turns_used += 1
                self._emit(
                    "autonomous_continuation",
                    reason="children_fanin",
                    n=int(auto.continuations_used),
                    thinking_only=False,
                )
            else:
                self._emit("fanin_continuation", reason="children_fanin")
            return [msg]

        # Active goal: continue until goal.complete() or token / soft turn budget.
        if self.jobs.goal_is_active():
            goal = self.jobs.goal or {}
            if goal_budget_exhausted(goal):
                self.jobs.goal_mark_budget_limited()
                self._emit("goal_budget_limited", goal_id=goal.get("id"))
                return []
            # Soft cap only when rlm_max_turns is explicitly positive (tests).
            turns_cap = int(self.rlm_max_turns or 0)
            if turns_cap > 0 and int(goal.get("continuations_used") or 0) >= turns_cap:
                self._emit(
                    "goal_continuation_limit",
                    goal_id=goal.get("id"),
                    n=int(goal.get("continuations_used") or 0),
                )
                return []
            self.jobs.goal_mark_continuation()
            msg = build_goal_continuation_message(self.jobs.goal or {})
            self._emit(
                "goal_continuation",
                goal_id=(self.jobs.goal or {}).get("id"),
                n=int((self.jobs.goal or {}).get("continuations_used") or 0),
            )
            return [msg]

        auto = self.jobs.autonomous
        if auto is None or not auto.enabled:
            return []
        # Heartbeat / pause paths should not push autonomous keep-alive.
        if self.run_lifetime != "one_shot":
            return []

        if turn_content is None:
            evidence = ""
        else:
            evidence = str(turn_content or "")
        self._last_quiescent_turn_content = evidence
        reasoning = str(
            turn_reasoning
            if turn_reasoning is not None
            else (self.last_reasoning or "")
        ).strip()
        thinking_only = not evidence.strip() and bool(reasoning)
        empty_streak = int(getattr(self, "_empty_assistant_streak", 0) or 0)
        gate_result = await asyncio.to_thread(
            refresh_autonomous_quality_gates, auto, self.workspace_dir
        )
        if gate_result == "passed":
            self._emit("autonomous_gate_passed")
            # Gate pass is necessary host evidence, not sufficient to seal.
            # Stub/first-chunk files often exist only to satisfy `test -f …`;
            # keep going through probes + completion review.
        elif gate_result in ("failed", "retry_exhausted") and auto.last_gate_failure:
            self._emit(
                "autonomous_gate_failed",
                command=auto.last_gate_failure.command,
                attempt=auto.last_gate_failure.attempt,
                result=gate_result,
            )
        has_file = self._has_workspace_deliverable_file()
        artifact = self.artifact_required()
        if artifact:
            # Prose alone is not host evidence; workspace file or gate pass is.
            # Quality of that file is the AI completion-reviewer's job — not a
            # host regex. Gate pass alone must not seal (see review path below).
            has_terminal = bool(has_file or gate_result == "passed")
        else:
            has_terminal = self._content_counts_as_deliverable(evidence)
        nxt = next_autonomous_continuation(
            auto,
            has_user_facing_answer=has_terminal,
            gate_result=gate_result,
            thinking_only=thinking_only,
            empty_streak=empty_streak,
            artifact_required=artifact,
        )
        if nxt:
            if empty_streak >= int(EMPTY_ASSISTANT_NUDGE_AFTER) and not has_terminal:
                self._emit("empty_assistant_nudge", n=empty_streak)
            self._emit(
                "autonomous_continuation",
                reason=nxt.get("_rlm_continuation_reason"),
                n=int(auto.continuations_used),
                thinking_only=thinking_only,
            )
            return [nxt]

        if gate_result == "retry_exhausted" or (
            gate_result is None and autonomous_limit_reason(auto)
        ):
            # With product unlimited caps this is rare; keep going via missing
            # evidence path rather than sealing incomplete.
            if autonomous_limit_reason(auto) and not has_terminal:
                # Force another missing-evidence nudge even if caps were set in tests.
                pass
            self._autonomous_stop_reason = (
                autonomous_limit_reason(auto)
                or gate_result
                or "missing_terminal_evidence"
            )
            # Only stop the loop when a positive cap was explicitly configured
            # (unit tests). Product caps are 0 → never stop here.
            if autonomous_limit_reason(auto):
                return []
            # Unlimited: fall through should not happen; inject deliverable nudge.
            from src.orchestrators.rlm.runtime.autonomous import (
                EMPTY_ASSISTANT_CONTINUATION_PROMPT,
                MISSING_DELIVERABLE_CONTINUATION_PROMPT,
            )

            auto.continuations_used += 1
            auto.continuations_total += 1
            auto.turns_used += 1
            prompt = (
                MISSING_DELIVERABLE_CONTINUATION_PROMPT
                if artifact
                else EMPTY_ASSISTANT_CONTINUATION_PROMPT
            )
            return [
                {
                    "role": "user",
                    "content": prompt,
                    "_rlm_continuation": "autonomous",
                    "_rlm_continuation_reason": "missing_terminal_evidence",
                }
            ]

        # Host checklist probes after gates — blocking fail before judge.
        if has_terminal and artifact:
            from src.orchestrators.rlm.runtime.acceptance_contract import (
                blocking_probe_failures,
                build_probe_failed_continuation,
                run_checklist_probes,
            )

            if getattr(self, "acceptance_contract", None) is None:
                await self._ensure_acceptance_contract()
            probes = await asyncio.to_thread(
                run_checklist_probes,
                getattr(self, "acceptance_contract", None),
                str(self.workspace_dir or ""),
            )
            self.acceptance_probe_results = list(probes or [])
            fails = blocking_probe_failures(
                getattr(self, "acceptance_contract", None), probes
            )
            if fails:
                self._emit(
                    "acceptance_probe",
                    passed=False,
                    failed=len(fails),
                    detail=[str(f.get("id") or "") for f in fails[:12]],
                )
                self._autonomous_stop_reason = "acceptance_probe_failed"
                # Always continue with probe feedback unless an explicit positive
                # cap is exhausted (unit tests). Product caps are unlimited (0).
                if autonomous_limit_reason(auto):
                    return []
                auto.continuations_used += 1
                auto.continuations_total += 1
                auto.turns_used += 1
                if autonomous_limit_reason(auto):
                    return []
                return [build_probe_failed_continuation(fails)]
            self._emit(
                "acceptance_probe",
                passed=True,
                failed=0,
                n=len(probes or []),
            )

        # About to stop with host evidence — run completion review first.
        if has_terminal:
            review_msg = await self._maybe_completion_review_continuation(
                evidence=evidence,
                gate_result=gate_result,
                has_file=has_file,
                artifact_required=artifact,
            )
            if review_msg:
                return [review_msg]
            # Helper sets _completion_review_passed on accept/skip only.
            if not getattr(self, "_completion_review_passed", False):
                self._autonomous_stop_reason = (
                    self._autonomous_stop_reason or "completion_review_rejected"
                )
                # Product: never seal incomplete here — keep nudging if unlimited.
                if not autonomous_limit_reason(auto):
                    from src.orchestrators.rlm.runtime.completion_review import (
                        build_completion_review_rejection_message,
                    )

                    auto.continuations_total += 1
                    auto.turns_used += 1
                    return [
                        build_completion_review_rejection_message(
                            {
                                "complete": False,
                                "reason": str(
                                    self._autonomous_stop_reason
                                    or "completion_review_rejected"
                                ),
                                "missing": [
                                    "Continue until the host acceptance criteria are met."
                                ],
                            }
                        )
                    ]
            return []

        self._autonomous_stop_reason = (
            self._autonomous_stop_reason or "missing_terminal_evidence"
        )
        if not autonomous_limit_reason(auto):
            from src.orchestrators.rlm.runtime.autonomous import (
                EMPTY_ASSISTANT_CONTINUATION_PROMPT,
                MISSING_DELIVERABLE_CONTINUATION_PROMPT,
            )

            auto.continuations_used += 1
            auto.continuations_total += 1
            auto.turns_used += 1
            return [
                {
                    "role": "user",
                    "content": (
                        MISSING_DELIVERABLE_CONTINUATION_PROMPT
                        if artifact
                        else EMPTY_ASSISTANT_CONTINUATION_PROMPT
                    ),
                    "_rlm_continuation": "autonomous",
                    "_rlm_continuation_reason": "missing_terminal_evidence",
                }
            ]
        return []

    async def _maybe_completion_review_continuation(
        self,
        *,
        evidence: str,
        gate_result: Optional[str],
        has_file: bool,
        artifact_required: bool,
        force_fresh: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Gate entry to completed: reject → continuation; accept → None.

        ``force_fresh`` skips the unchanged short-circuit (used at finalize /
        turn-budget end so a finished deliverable gets one real LLM review).
        Order: probes (already refreshed) → claim → host stub/narration → judge.
        """
        from src.orchestrators.rlm.runtime.acceptance_contract import (
            VERDICT_ACCEPT,
            VERDICT_NEED_CLAIM,
            acceptance_evidence_extra,
            all_blocking_probes_green,
            blocking_probe_failures,
            build_claim_contradiction_continuation,
            build_claim_required_continuation,
            build_probe_failed_continuation,
            claim_contradicts_probes,
            evaluate_host_verdict,
            format_checklist_for_prompt,
            format_claim_for_prompt,
            format_probes_for_prompt,
            open_semantic_blocking_items,
            parse_worker_claim,
            run_checklist_probes,
        )
        from src.orchestrators.rlm.runtime.completion_review import (
            MAX_COMPLETION_REVIEWS_PER_RUN,
            build_completion_review_rejection_message,
            build_unchanged_deliverable_continuation,
            completion_review_evidence_fingerprint,
            load_deliverable_excerpts_for_review,
            review_completion,
            should_skip_completion_review,
        )

        host_ok = bool(has_file or gate_result == "passed") or (
            not artifact_required and self._content_counts_as_deliverable(evidence)
        )
        # Gate pass is necessary evidence, not sufficient — always run the
        # completion review (except short-chat skip below). Skipping on
        # gate_passed caused premature [end] completed after the first stub
        # file satisfied `test -f …`.
        if should_skip_completion_review(
            artifact_required=artifact_required,
            evidence_text=evidence,
            host_evidence_ok=host_ok,
        ):
            self._completion_review_passed = True
            self._emit("completion_review_skipped", reason="short_chat")
            return None

        used = int(getattr(self, "_completion_reviews_used", 0) or 0)
        review_cap = MAX_COMPLETION_REVIEWS_PER_RUN + (1 if force_fresh else 0)
        if used >= review_cap:
            # Do not seal incomplete — keep nudging; reset window so reviews resume.
            self._autonomous_stop_reason = "completion_review_budget"
            self._completion_review_passed = False
            if not getattr(self, "_completion_review_budget_emitted", False):
                self._completion_review_budget_emitted = True
                self._emit(
                    "completion_review_budget",
                    n=used,
                    max=review_cap,
                )
            self._completion_reviews_used = max(0, review_cap - 2)
            auto = self.jobs.autonomous
            if auto is not None and auto.enabled:
                auto.continuations_total += 1
                auto.turns_used += 1
            return build_completion_review_rejection_message(
                {
                    "complete": False,
                    "reason": (
                        "Host review budget rolled; continue improving the deliverable."
                    ),
                    "missing": [
                        "Expand the deliverable until the user task is fully done."
                    ],
                }
            )

        if artifact_required and getattr(self, "acceptance_contract", None) is None:
            await self._ensure_acceptance_contract()

        # Refresh probes at review time (workspace may have changed).
        if artifact_required:
            probes = await asyncio.to_thread(
                run_checklist_probes,
                getattr(self, "acceptance_contract", None),
                str(self.workspace_dir or ""),
            )
            self.acceptance_probe_results = list(probes or [])
            fails = blocking_probe_failures(
                getattr(self, "acceptance_contract", None), probes
            )
            if fails:
                self._emit(
                    "acceptance_probe",
                    passed=False,
                    failed=len(fails),
                    detail=[str(f.get("id") or "") for f in fails[:12]],
                )
                self._completion_review_passed = False
                self._autonomous_stop_reason = "acceptance_probe_failed"
                auto = self.jobs.autonomous
                if auto is not None and auto.enabled:
                    if autonomous_limit_reason(auto):
                        return None
                    auto.continuations_used += 1
                    auto.continuations_total += 1
                    auto.turns_used += 1
                    if autonomous_limit_reason(auto):
                        return None
                return build_probe_failed_continuation(fails)

        claim = parse_worker_claim(evidence or "")
        if claim is None and artifact_required:
            # Also scan recent assistant messages for a trailing claim.
            for msg in reversed(list(self.messages or [])[-8:]):
                if not isinstance(msg, dict) or msg.get("role") != "assistant":
                    continue
                claim = parse_worker_claim(str(msg.get("content") or ""))
                if claim is not None:
                    break
        self.last_worker_claim = claim
        if artifact_required and claim is not None:
            self._emit(
                "acceptance_claim",
                done=bool(claim.get("done")),
                artifacts=len(claim.get("artifacts") or []),
            )

        if artifact_required and claim is None:
            self._completion_review_passed = False
            self._autonomous_stop_reason = "acceptance_claim_missing"
            auto = self.jobs.autonomous
            if auto is not None and auto.enabled:
                if autonomous_limit_reason(auto) and not force_fresh:
                    return None
                auto.continuations_used += 1
                auto.continuations_total += 1
                auto.turns_used += 1
                if autonomous_limit_reason(auto) and not force_fresh:
                    return None
            return build_claim_required_continuation(
                getattr(self, "acceptance_contract", None)
            )

        if artifact_required and claim is not None:
            bad = claim_contradicts_probes(
                claim, getattr(self, "acceptance_probe_results", []) or []
            )
            if bad:
                self._completion_review_passed = False
                self._autonomous_stop_reason = "acceptance_claim_contradiction"
                auto = self.jobs.autonomous
                if auto is not None and auto.enabled:
                    if autonomous_limit_reason(auto) and not force_fresh:
                        return None
                    auto.continuations_used += 1
                    auto.continuations_total += 1
                    auto.turns_used += 1
                    if autonomous_limit_reason(auto) and not force_fresh:
                        return None
                return build_claim_contradiction_continuation(bad)

        file_names: List[str] = []
        file_details: List[str] = []
        file_records: List[Dict[str, Any]] = []
        deliverable_contents = "(none)"
        try:
            from src.orchestrators.rlm.runtime.persist import (
                list_kernel_deliverable_files,
            )

            raw_recs = [
                rec
                for rec in (list_kernel_deliverable_files(self.workspace_dir) or [])
                if isinstance(rec, dict)
            ]
            excerpt_pack = load_deliverable_excerpts_for_review(
                raw_recs,
                workspace_dir=str(self.workspace_dir or ""),
            )
            file_records = list(excerpt_pack.get("records") or [])
            # Preserve mtime from listing for fingerprint stability.
            mtime_by_name = {
                str(r.get("relpath") or r.get("filename") or ""): r.get("mtime_ns")
                for r in raw_recs
            }
            for rec in file_records:
                name = str(rec.get("relpath") or "").strip()
                if name and "mtime_ns" not in rec:
                    rec["mtime_ns"] = mtime_by_name.get(name)
            deliverable_contents = str(
                excerpt_pack.get("contents_block") or "(none)"
            )
            for rec in file_records:
                name = str(rec.get("relpath") or "").strip()
                if not name:
                    continue
                file_names.append(name)
                size = rec.get("size_bytes")
                size_s = f"{size}B" if size is not None else "?"
                trunc = "truncated" if rec.get("truncated") else "full"
                preview = str(rec.get("preview") or "")[:240]
                detail = f"{name} ({size_s}, {trunc})"
                if preview:
                    detail += f" preview={preview!r}"
                file_details.append(detail)
            if not file_names:
                for rec in raw_recs:
                    name = str(rec.get("relpath") or rec.get("filename") or "").strip()
                    if name:
                        file_names.append(name)
                        file_details.append(name)
                        file_records.append(
                            {
                                "relpath": name,
                                "size_bytes": rec.get("size_bytes"),
                                "mtime_ns": rec.get("mtime_ns"),
                                "preview": "",
                            }
                        )
        except Exception:
            file_names = []
            file_details = []
            file_records = []
            deliverable_contents = "(none)"

        accept_extra = acceptance_evidence_extra(
            contract=getattr(self, "acceptance_contract", None),
            probes=getattr(self, "acceptance_probe_results", []) or [],
            claim=claim,
        )
        evidence_fp = completion_review_evidence_fingerprint(
            evidence=evidence,
            gate_result=gate_result,
            file_records=file_records,
            extra=accept_extra,
        )
        # Same files as the last reject: nudge without burning review budget.
        # At finalize / turn-budget end, force a real review instead of appending
        # a nudge that will never execute (that caused false incomplete).
        force_now = bool(force_fresh) or bool(
            getattr(self, "_turn_budget_exhausted", False)
        )
        if (
            not force_now
            and evidence_fp
            and evidence_fp == getattr(self, "_last_completion_review_fp", "")
            and not getattr(self, "_completion_review_passed", False)
        ):
            streak = int(getattr(self, "_completion_review_unchanged_streak", 0) or 0) + 1
            self._completion_review_unchanged_streak = streak
            self._emit(
                "completion_review_unchanged",
                n=used,
                streak=streak,
                fingerprint=evidence_fp,
            )
            auto = self.jobs.autonomous
            limit = autonomous_limit_reason(auto) if auto is not None else None
            if limit:
                # Last chance: fall through to a fresh LLM review before stopping.
                force_now = True
                self._emit(
                    "completion_review_forced",
                    reason="autonomous_limit",
                    limit=str(limit),
                )
            else:
                if auto is not None and auto.enabled:
                    auto.continuations_total += 1
                    auto.turns_used += 1
                return build_unchanged_deliverable_continuation(
                    str(getattr(self, "_last_completion_review_reason", "") or "")
                )

        if force_now and evidence_fp == getattr(self, "_last_completion_review_fp", ""):
            self._emit(
                "completion_review_forced",
                reason="finalize" if force_fresh else "turn_budget_or_limit",
                fingerprint=evidence_fp,
            )

        self._completion_review_unchanged_streak = 0

        # Host hard reject: unfinished narration only.
        # Deliverable quality (stub vs complete) is the AI reviewer's job with
        # file excerpts — never regex-reject stubs here.
        host_reject_reason = ""
        if (not force_now) and evidence and (
            looks_like_incomplete_assistant(evidence)
            or looks_like_awaiting_children(evidence)
            or _FUTURE_INTENT_RE.search(evidence)
        ):
            host_reject_reason = "incomplete_narration"

        if host_reject_reason:
            self._completion_reviews_used = used + 1
            review = {
                "complete": False,
                "reason": (
                    "Assistant still indicates unfinished work; a passed gate or "
                    "partial draft is not enough."
                ),
                "missing": [
                    "Continue expanding the deliverable until the user task is fully done."
                ],
            }
            self._emit(
                "completion_review",
                complete=False,
                reason=str(review["reason"])[:240],
                missing=list(review["missing"]),
                n=self._completion_reviews_used,
                host_override=host_reject_reason,
            )
            self._completion_review_passed = False
            self._last_completion_review_fp = evidence_fp
            self._last_completion_review_reason = str(review["reason"])[:240]
            auto = self.jobs.autonomous
            if auto is not None and auto.enabled:
                auto.continuations_total += 1
                auto.turns_used += 1
            return build_completion_review_rejection_message(review)

        # Artifact mode always asks the AI judge. probes_and_claim skip is only
        # for non-artifact (chat) runs — otherwise stub files sealed without LLM.
        semantic_open = open_semantic_blocking_items(
            getattr(self, "acceptance_contract", None),
            getattr(self, "acceptance_probe_results", []) or [],
        )
        if (
            not artifact_required
            and not semantic_open
            and all_blocking_probes_green(
                getattr(self, "acceptance_contract", None),
                getattr(self, "acceptance_probe_results", []) or [],
            )
        ):
            verdict = evaluate_host_verdict(
                contract=getattr(self, "acceptance_contract", None),
                gate_result=gate_result,
                probes=getattr(self, "acceptance_probe_results", []) or [],
                claim=claim,
                judge={"complete": True, "reason": "probes_and_claim"},
                artifact_required=False,
                gates_configured=gates_configured(self.jobs.autonomous)
                if self.jobs.autonomous is not None
                else False,
            )
            if verdict.get("verdict") == VERDICT_ACCEPT:
                self._completion_reviews_used = used + 1
                self._completion_review_passed = True
                self._last_completion_review_fp = evidence_fp
                self._last_completion_review_reason = ""
                self._emit(
                    "completion_review",
                    complete=True,
                    reason="probes_and_claim",
                    missing=[],
                    n=self._completion_reviews_used,
                    host_override="probes_and_claim",
                )
                return None

        self._completion_reviews_used = used + 1
        checklist_txt = format_checklist_for_prompt(
            getattr(self, "acceptance_contract", None),
            getattr(self, "acceptance_probe_results", []) or [],
        )
        probes_txt = format_probes_for_prompt(
            getattr(self, "acceptance_probe_results", []) or []
        )
        claim_txt = format_claim_for_prompt(claim)
        # Always show file bodies to the AI judge when present.
        include_excerpts = bool(file_names) or bool(semantic_open) or not artifact_required
        try:
            gate_commands: List[str] = []
            auto_for_gates = self.jobs.autonomous
            if auto_for_gates is not None:
                gate_commands = [
                    str(c).strip()
                    for c in (auto_for_gates.gates.get("commands") or [])
                    if str(c).strip()
                ]
            review = await review_completion(
                self,
                chat_fn=self.chat_fn,
                extract_assistant_fn=self.extract_assistant_fn,
                user_goal=str(self.user_content or ""),
                evidence_text=evidence or self._visible_session_output(),
                deliverable_files=file_names,
                deliverable_details=file_details or file_names,
                deliverable_contents=deliverable_contents,
                artifact_required=artifact_required,
                gate_result=gate_result,
                gate_commands=gate_commands,
                acceptance_checklist=checklist_txt,
                worker_claim=claim_txt,
                probe_results=probes_txt,
                include_excerpts=include_excerpts,
            )
        except Exception as exc:
            logger.warning("completion review failed", exc_info=True)
            review = {
                "complete": False,
                "reason": f"completion_review_error: {exc}",
                "missing": ["Retry finishing after fixing the last step."],
            }

        # Host veto even if judge says complete.
        verdict = evaluate_host_verdict(
            contract=getattr(self, "acceptance_contract", None),
            gate_result=gate_result,
            probes=getattr(self, "acceptance_probe_results", []) or [],
            claim=claim,
            judge=review,
            artifact_required=artifact_required,
            gates_configured=gates_configured(self.jobs.autonomous)
            if self.jobs.autonomous is not None
            else False,
        )
        if verdict.get("verdict") == VERDICT_NEED_CLAIM:
            self._completion_review_passed = False
            self._autonomous_stop_reason = "acceptance_claim_missing"
            auto = self.jobs.autonomous
            if auto is not None and auto.enabled:
                auto.continuations_total += 1
                auto.turns_used += 1
            return build_claim_required_continuation(
                getattr(self, "acceptance_contract", None)
            )
        if verdict.get("verdict") != VERDICT_ACCEPT:
            review = {
                "complete": False,
                "reason": str(verdict.get("reason") or review.get("reason") or ""),
                "missing": list(verdict.get("missing") or review.get("missing") or [])[
                    :8
                ],
            }

        self._emit(
            "completion_review",
            complete=bool(review.get("complete"))
            and verdict.get("verdict") == VERDICT_ACCEPT,
            reason=str(review.get("reason") or "")[:240],
            missing=list(review.get("missing") or [])[:8],
            n=self._completion_reviews_used,
            host_verdict=str(verdict.get("verdict") or ""),
        )
        if verdict.get("verdict") == VERDICT_ACCEPT and host_ok:
            self._completion_review_passed = True
            self._last_completion_review_fp = evidence_fp
            self._last_completion_review_reason = ""
            return None

        self._completion_review_passed = False
        self._last_completion_review_fp = evidence_fp
        self._last_completion_review_reason = str(review.get("reason") or "")[:240]
        auto = self.jobs.autonomous
        if auto is not None and auto.enabled:
            auto.continuations_total += 1
            auto.turns_used += 1
        return build_completion_review_rejection_message(review)

    async def _ensure_completion_review_before_finalize(self) -> None:
        """Final gate: always run a fresh review before incomplete/completed."""
        if self.run_lifetime != "one_shot":
            return
        auto = self.jobs.autonomous
        if auto is None or not auto.enabled:
            self._completion_review_passed = True
            return
        if getattr(self, "_completion_review_passed", False):
            return
        if not self._has_autonomous_deliverable():
            return
        evidence = str(
            getattr(self, "_last_quiescent_turn_content", "")
            or self.last_content
            or self._visible_session_output()
            or ""
        )
        reasoning = str(getattr(self, "last_reasoning", "") or "").strip()
        if reasoning:
            evidence = (
                f"{evidence}\n\n[assistant_reasoning]\n{reasoning[:6000]}"
            ).strip()
        gate_result = None
        if gates_configured(auto):
            gate_result = auto.gates_last_result
        msg = await self._maybe_completion_review_continuation(
            evidence=evidence,
            gate_result=gate_result if gate_result in ("passed", "failed", "retry_exhausted") else None,
            has_file=self._has_workspace_deliverable_file(),
            artifact_required=self.artifact_required(),
            force_fresh=True,
        )
        if msg:
            # Rejected at the finish line after a fresh review — incomplete.
            self._completion_review_passed = False
            self.messages.append(msg)
            self._autonomous_stop_reason = (
                self._autonomous_stop_reason or "completion_review_rejected"
            )
        elif not getattr(self, "_completion_review_passed", False):
            self._autonomous_stop_reason = (
                self._autonomous_stop_reason or "completion_review_rejected"
            )
        # accept path sets _completion_review_passed inside helper

    def artifact_required(self) -> bool:
        """True when host completion needs a workspace file or passing gates."""
        if getattr(self, "autonomous_strict", False):
            return True
        if list(getattr(self, "pool_defs", None) or []):
            return True
        auto = getattr(self, "jobs", None)
        auto = getattr(auto, "autonomous", None) if auto is not None else None
        if auto is not None and gates_configured(auto):
            return True
        return False

    def mark_flow_released(self) -> None:
        """FlowRunner finished awaiting this one_shot node — block idle nudges."""
        self._flow_released = True

    def clear_flow_released(self) -> None:
        self._flow_released = False

    def _mark_fanin_pending(self, *, reason: str = "") -> None:
        """Parent must synthesize after children; prior drafts are not final."""
        self._fanin_pending = True
        self._deliverable_stale = True
        if reason:
            self._emit("fanin_pending", reason=reason)

    def _content_counts_as_deliverable(self, text: str) -> bool:
        if getattr(self, "_fanin_pending", False):
            return False
        if getattr(self, "children", None) is not None and self._has_running_children():
            return False
        s = str(text or "").strip()
        if not s:
            return False
        if looks_like_incomplete_assistant(s) or looks_like_awaiting_children(s):
            return False
        if looks_like_context_checkpoint(s):
            return False
        if _FUTURE_INTENT_RE.search(s):
            return False
        if looks_like_terminal_answer(s):
            return True
        # Short replies ("ok", "391", "Porto") are valid answers to short asks.
        if not re.search(r"\w", s):
            return False
        # Short environment/status asides are not deliverables.
        if len(s) < 160 and re.search(
            r"(?i)\b(network works|blocked/empty|blocked|unavailable)\b", s
        ):
            return False
        if _DELIVERABLE_HINT_RE.search(s):
            return True
        # Prime semantics: any remaining user-facing prose is an answer. Everything
        # that is *not* an answer (narration, status asides, awaiting-children,
        # context checkpoints, future intent) was rejected above.
        return True

    def _stamp_assistant_output(self, content: str, last: str) -> str:
        """Update last_content when visible text looks like a deliverable."""
        blocked = getattr(self, "_fanin_pending", False) or (
            getattr(self, "children", None) is not None and self._has_running_children()
        )
        if blocked:
            # Keep turn return value for logs, but never seal a final answer yet.
            if str(content or "").strip() and not looks_like_incomplete_assistant(content):
                return content
            return last
        if self._content_counts_as_deliverable(content):
            self.last_content = content
            self._deliverable_stale = False
            return content
        if str(content or "").strip() and not looks_like_incomplete_assistant(content):
            return content
        return last

    def _has_autonomous_deliverable(self) -> bool:
        if getattr(self, "_fanin_pending", False):
            return False
        if getattr(self, "children", None) is not None and self._has_running_children():
            return False
        auto = self.jobs.autonomous
        text = str(self.last_content or "").strip()
        if auto is not None and gates_configured(auto):
            # Gate pass = host evidence exists; AI review decides if it's enough.
            return auto.gates_last_result == "passed"
        # An answer stamped before the model went back to tool calls is not the
        # final turn — the run resumed work and never re-concluded.
        if getattr(self, "_deliverable_stale", False):
            return False
        if self.artifact_required():
            return self._has_workspace_deliverable_file()
        return self._content_counts_as_deliverable(text)

    def _has_workspace_deliverable_file(self) -> bool:
        try:
            from src.orchestrators.rlm.runtime.persist import (
                list_kernel_deliverable_files,
            )

            return bool(list_kernel_deliverable_files(self.workspace_dir))
        except Exception:
            return False

    def _visible_session_output(self) -> str:
        """Best-effort final visible assistant text (not only stamped last_content)."""
        for candidate in (
            str(self.last_content or "").strip(),
            str(getattr(self, "_last_quiescent_turn_content", "") or "").strip(),
        ):
            if candidate:
                return candidate
        for msg in reversed(self.messages):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            content = str(msg.get("content") or "").strip()
            if content:
                return content
        return ""

    def _finalize_one_shot_status(self) -> None:
        """Mark completed vs incomplete.

        Gate pass alone is not enough. Completion requires the AI review path
        to set ``_completion_review_passed``. Soft incomplete ≠ infra error.
        """
        if self.status in ("error", "cancelled", "incomplete"):
            return
        auto = self.jobs.autonomous
        if (
            auto is not None
            and auto.enabled
            and bool(getattr(self, "_completion_review_passed", False))
        ):
            self.status = "completed"
            self._autonomous_stop_reason = None
            self.error = None
            return
        if getattr(self, "_context_overflow_exhausted", False):
            # Still recoverable in principle; mark incomplete not infra error.
            self.status = "incomplete"
            self._autonomous_stop_reason = (
                self._autonomous_stop_reason or "context_overflow"
            )
            self.error = None
            self._emit("context_overflow_incomplete", reason="retry_exhausted")
            return
        if getattr(self, "_context_budget_overflow", False) and looks_like_context_checkpoint(
            self._visible_session_output()
        ):
            self.status = "incomplete"
            self._autonomous_stop_reason = (
                self._autonomous_stop_reason or "context_overflow_checkpoint"
            )
            self.error = None
            self._emit("context_overflow_incomplete", reason="checkpoint_only")
            return
        if auto is None or not auto.enabled or self.run_lifetime != "one_shot":
            self.status = "completed"
            return
        has_deliverable = self._has_autonomous_deliverable()
        review_ok = bool(getattr(self, "_completion_review_passed", False))
        if has_deliverable and review_ok:
            self.status = "completed"
            return
        if has_deliverable and not review_ok:
            limit = (
                autonomous_limit_reason(auto)
                or self._autonomous_stop_reason
                or "completion_review_rejected"
            )
            self.status = "incomplete"
            self._autonomous_stop_reason = str(limit)
            # Do not stamp self.error — incomplete ≠ infra failure.
            self.error = None
            self._emit("autonomous_limit_exhausted", reason=str(limit))
            return
        limit = autonomous_limit_reason(auto) or self._autonomous_stop_reason
        if getattr(self, "_turn_budget_exhausted", False):
            limit = limit or "turn_budget_exhausted"
        elif self.artifact_required():
            limit = limit or "missing_terminal_evidence"
        streak_cap = int(auto.limits.get("max_continuations") or 0)
        streak_exhausted = streak_cap > 0 and auto.continuations_used >= streak_cap
        if limit or streak_exhausted or self.artifact_required():
            reason = str(limit or "missing_terminal_evidence")
            self.status = "incomplete"
            self._autonomous_stop_reason = reason
            self.error = None
            self._emit("autonomous_limit_exhausted", reason=reason)
            return
        self.status = "incomplete"
        self._autonomous_stop_reason = (
            self._autonomous_stop_reason or "missing_terminal_evidence"
        )
        self.error = None
        self._emit(
            "autonomous_limit_exhausted", reason="missing_terminal_evidence"
        )
    def _note_turn_usage(self, data: Any) -> None:
        """Best-effort token accounting for autonomous / goal budgets."""
        usage = {}
        if isinstance(data, dict):
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        inp = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        cache_w = int(usage.get("cache_creation_input_tokens") or usage.get("cache_write") or 0)
        total = max(0, inp + out + cache_w)
        auto = self.jobs.autonomous
        if auto is not None:
            add_autonomous_usage(
                auto,
                input_tokens=inp,
                output_tokens=out,
                cache_write_tokens=cache_w,
            )
        if total:
            self.jobs.goal_add_usage(tokens=total)

    async def _wait_for_children_or_inbox(self, timeout: float) -> bool:
        """Wait for child tasks. Returns True only for *ready* (non-buffered) inbox.

        Mid-wait child/rlm messages stay buffered until children settle so the
        parent does not burn LLM turns on partial fan-in.
        """
        deadline = time.time() + max(0.05, float(timeout))
        while time.time() < deadline:
            if self._inbox_llm_pending():
                return True
            if not self._has_running_children():
                # Children settled — any buffered inbox becomes ready now.
                return self._inbox_llm_cursor < len(self.inbox)
            tasks = [t for t in self._child_tasks.values() if not t.done()]
            if not tasks:
                await asyncio.sleep(0.05)
                if not self._has_running_children():
                    return self._inbox_llm_cursor < len(self.inbox)
                continue
            try:
                await asyncio.wait(
                    tasks, timeout=0.2, return_when=asyncio.FIRST_COMPLETED
                )
            except Exception:
                await asyncio.sleep(0.2)
        return self._inbox_llm_pending()

    async def _silent_wait_for_children_fanin(self) -> bool:
        """Host-side wait while quiet-awaiting children. No LLM calls.

        Returns True if parent should resume (children settled and/or ready inbox).
        """
        if not self._has_running_children():
            self._clear_awaiting_children_quiet()
            return True
        wait_budget = self.children_wait_budget_sec()
        while (
            self._has_running_children()
            and not getattr(self, "_closed", False)
            and self.status not in ("cancelled", "error")
        ):
            remaining = wait_budget - self._children_wait_elapsed
            if remaining <= 0:
                outcome = await self._handle_children_wait_budget()
                if outcome == "intervened":
                    break
                wait_budget = self.children_wait_budget_sec()
                continue
            slice_sec = min(self._child_wait_slice_sec(), remaining)
            woke = await self._wait_for_children_or_inbox(slice_sec)
            self._children_wait_elapsed += slice_sec
            if woke:
                # Ready (non-buffered) inbox — e.g. attach interrupt path.
                break
            if not self._has_running_children():
                break
        if not self._has_running_children():
            self._clear_awaiting_children_quiet()
        return (not self._has_running_children()) or self._inbox_llm_pending()

    async def _drain_children_before_finish(self) -> bool:
        """Block finish while children run. True if still pending after wait budget."""
        if self.run_lifetime != "one_shot":
            return self._has_running_children()
        wait_budget = self.children_wait_budget_sec()
        while self._has_running_children() or self._inbox_llm_pending() or (
            not self._has_running_children()
            and self._inbox_llm_cursor < len(self.inbox)
        ):
            if not self._has_running_children():
                self._clear_awaiting_children_quiet()
            if self._inbox_llm_pending() or (
                not self._has_running_children()
                and self._inbox_llm_cursor < len(self.inbox)
            ):
                await self._inject_inbox()
                # Inbox needs another LLM turn — caller must not mark completed.
                return True
            if not self._has_running_children():
                break
            remaining = wait_budget - self._children_wait_elapsed
            if remaining <= 0:
                outcome = await self._handle_children_wait_budget()
                if outcome == "intervened":
                    return self._has_running_children() or self._inbox_llm_pending()
                wait_budget = self.children_wait_budget_sec()
                if self._children_wait_elapsed >= wait_budget and self._has_running_children():
                    return True
                continue
            slice_sec = min(self._child_wait_slice_sec(), remaining)
            await self._wait_for_children_or_inbox(slice_sec)
            self._children_wait_elapsed += slice_sec
        return self._has_running_children() or self._inbox_llm_pending()

    async def run_turns(self, *, base_system: str = "") -> str:
        if not agent_model_is_explicit(self.model):
            raise ValueError("rlm_model_required")
        # Do not proactive-compact here — ``_run_turn_loop`` compacts at 90% of
        # context_length between turns (structured checkpoint + recent tail).
        async with self._turns_lock:
            out = await self._run_turn_loop(
                base_system=base_system or self._cfg_base_system(), terminal="complete"
            )
            try:
                await self.flush_kernel_snapshot()
            except Exception:
                pass
            await self._maybe_auto_refine_checkpoint()
            return out

    async def continue_with_message(self, text: Any) -> str:
        """Attach / chat-turn path: push a user message and run additional RLM turns."""
        self.clear_flow_released()
        self._completion_review_passed = False
        if text is not None and text != "":
            self.messages.append({"role": "user", "content": text})
        if self.messages and (self.messages[-1] or {}).get("role") == "user":
            # A new user turn: the previous turn's visible answer must not be
            # reported as this turn's when it ends without prose.
            self.last_content = ""
            self._last_quiescent_turn_content = ""
        self._chat_turn_request = self._latest_user_text()
        if not self.messages:
            system = self.build_system_prompt(self._cfg_base_system())
            if system:
                self.messages.append({"role": "system", "content": system})
        async with self._turns_lock:
            self._emit("attach_continue")
            out = await self._run_turn_loop(base_system="", terminal="pause")
            try:
                await self.flush_kernel_snapshot()
            except Exception:
                pass
            await self._maybe_auto_refine_checkpoint()
            return out

    def _reset_failure_flags_for_resume(self) -> None:
        """Clear terminal failure stamps so a failed session can run again."""
        self.clear_flow_released()
        self._closed = False
        self.error = None
        self.status = "running"
        self._was_running = True
        self._autonomous_stop_reason = None
        self._completion_review_passed = False
        self._context_overflow_exhausted = False
        self._context_budget_overflow = False
        self._turn_budget_exhausted = False
        self._turn_budget_notice_sent = False
        self._empty_assistant_streak = 0
        self._completion_reviews_used = 0
        self._completion_review_budget_emitted = False
        self._last_completion_review_fp = ""
        self._last_completion_review_reason = ""
        auto = self.jobs.autonomous
        if auto is not None and auto.enabled:
            # Allow a fresh streak; keep lifetime totals for observability.
            auto.continuations_used = 0
            auto.turns_used = 0

    async def continue_after_failure(self, text: Any = None) -> str:
        """Resume a failed/incomplete one_shot from current transcript until goal."""
        self._reset_failure_flags_for_resume()
        prompt = str(text or "").strip() or (
            "Continue from where you left off. The previous run stopped without "
            "finishing. Keep working until the host observes terminal success "
            "(gates / deliverable / acceptance). Do not restart the whole task "
            "from scratch unless the workspace evidence is missing."
        )
        self.messages.append({"role": "user", "content": prompt})
        async with self._turns_lock:
            self._emit("attach_continue", mode="continue")
            out = await self._run_turn_loop(base_system="", terminal="complete")
            try:
                await self.flush_kernel_snapshot()
            except Exception:
                pass
            await self._maybe_auto_refine_checkpoint()
            return out

    async def retry_from_scratch(self, *, already_wiped: bool = False) -> str:
        """Cold-restart this session: wipe prior files/logs/children, same goal.

        When ``already_wiped`` is True the caller (rlm_attach cold path) has
        already deleted Storage/disk deliverables and children so the first
        progress publish is empty — skip the destructive prelude.
        """
        self._reset_failure_flags_for_resume()
        if not already_wiped:
            # Tear down children first so their files/artifacts are included in wipe.
            for cid in list(self._child_sessions.keys()):
                try:
                    await self.delete_subagent(cid)
                except Exception:
                    logger.warning(
                        "retry wipe child delete failed cid=%s", cid, exc_info=True
                    )
            try:
                await self.stop_kernel(dump_snapshot=False)
            except Exception:
                logger.debug("retry stop_kernel failed", exc_info=True)
            try:
                from src.orchestrators.rlm.runtime.persist import (
                    wipe_session_for_retry,
                )

                await wipe_session_for_retry(self)
            except Exception:
                logger.warning("retry wipe_session_for_retry failed", exc_info=True)
        auto = self.jobs.autonomous
        if auto is not None:
            auto.gate_attempts = {}
            auto.last_gate_failure = None
            auto.last_gate_snapshot = None
            auto.gates_last_result = None
        self.messages = []
        self.last_content = ""
        self.last_reasoning = ""
        self._last_quiescent_turn_content = ""
        self._deliverable_stale = False
        self._emitted_deliverable_names = set()
        self.events = []
        self.log_lines = []
        self.thoughts = []
        self.children = {}
        self._child_sessions = {}
        self._child_tasks = {}
        self._hit_index = {}
        self._pending_looks = []
        self._overflow_recovery = "idle"
        self._look_media_invalid_recovery = False
        self._transcript_written = 0
        self._restored_from_disk = False
        self.error = None
        self.status = "running"
        try:
            self._ensure_transcript_header()
        except Exception:
            pass
        self._emit("attach_continue", mode="retry")
        return await self.run_turns(base_system=self._cfg_base_system())

    def _cfg_base_system(self) -> str:
        cfg = self.cfg or {}
        return str(cfg.get("system_prompt") or cfg.get("system") or "")

    async def _run_turn_loop(self, *, base_system: str, terminal: str) -> str:
        await self.start_kernel()
        self.status = "running"
        self._was_running = True
        self.error = None
        if terminal == "complete":
            self._emit("run_start")
            try:
                await self._ensure_acceptance_contract()
            except Exception:
                logger.warning("acceptance contract ensure failed", exc_info=True)

        if not self.messages:
            system = self.build_system_prompt(base_system)
            if system:
                self.messages.append({"role": "system", "content": system})
            if self.user_content is not None and self.user_content != "":
                self.messages.append({"role": "user", "content": self.user_content})
                self._chat_turn_request = self._latest_user_text()

        last = self.last_content
        # 0 / negative = unlimited product loop (stop only on success, cancel, error).
        unlimited_turns = int(self.rlm_max_turns or 0) <= 0
        base_turns = 10**9 if unlimited_turns else max(1, int(self.rlm_max_turns))
        synth_extra = (
            0
            if unlimited_turns
            else max(0, int(getattr(self, "rlm_synthesis_turns", None) or 5))
        )
        hard_turn_cap = base_turns if unlimited_turns else base_turns + synth_extra
        budget = base_turns
        step = 0
        self._children_wait_elapsed = 0.0
        try:
            while unlimited_turns or step < budget:
                # Quiet fan-in: parent already said it is awaiting children — do not
                # burn LLM turns; buffer child inbox until children settle.
                if (
                    self.run_lifetime in ("one_shot", "chat")
                    and getattr(self, "_awaiting_children_quiet", False)
                    and self._has_running_children()
                    and not self._inbox_llm_pending()
                    and not getattr(self, "_closed", False)
                ):
                    await self._silent_wait_for_children_fanin()
                    if getattr(self, "_closed", False) or self.status in (
                        "cancelled",
                        "error",
                    ):
                        break
                    # Flush buffered notices once, then one fan-in continuation turn.
                    await self._inject_inbox()
                    if not self._has_running_children():
                        cont = await self._take_continuation_messages(
                            turn_content="",
                            turn_reasoning=self.last_reasoning,
                        )
                        if cont:
                            self.messages.extend(cont)
                            # Product unlimited: never exhaust. Finite caps may
                            # grow only up to rlm_max_turns + synthesis_turns.
                            if unlimited_turns or step < hard_turn_cap:
                                budget = max(budget, step + 1)
                                if not unlimited_turns:
                                    budget = min(budget, hard_turn_cap)
                                # Fall through into a normal LLM turn below.
                            else:
                                # Cap hit; exit to post-drain / finalize.
                                break
                        else:
                            continue
                    else:
                        continue

                step += 1
                await self._inject_inbox()
                # Compact *before* injecting looks. Compaction rewrites history and
                # drops multimodal parts; looks must land on the wire after that.
                # maybe_compact also salvages any prior look pixels into pending.
                try:
                    await self.maybe_compact(
                        force=False, reason="proactive_threshold"
                    )
                except Exception as exc:
                    self._emit("compact_error", error=str(exc)[:LOG_LINE_CHARS])
                self._post_compact_turn_tick()
                self._inject_looks()
                self._maybe_inject_turn_budget_notice(
                    step=step, budget=budget, terminal=terminal
                )
                body: Dict[str, Any] = {
                    "messages": self.messages,
                    "stream": True,
                    **self.gen_params,
                    "model": self.model,
                    "tools": [IPYTHON_TOOL_SCHEMA],
                    "tool_choice": "auto",
                }
                thinking_on = self._apply_thinking_kwargs(body)
                # Prepare before announcing so the gap after [chat] is only network TTFT.
                self.prepare_chat_body(body)
                self._assert_look_pixels_on_wire(body)
                # Skip disk persist here — it blocked the event loop for seconds and
                # made the terminal freeze on "[chat] … stream" with no [stream] line.
                self._emit(
                    "llm_call",
                    model=self.model,
                    step=step - 1,
                    thinking=thinking_on,
                    stream=True,
                    persist=False,
                )
                # Publish TTFT wait *before* awaiting HTTP so polls see motion immediately.
                self._upsert_log_line("stream", "waiting for first token…")
                self._publish_stream_progress(force=True)
                await asyncio.sleep(0)
                # Fresh reasoning buffer for this turn's stream.
                if thinking_on:
                    self.last_reasoning = ""
                    self._reasoning_open = False
                # _chat_with_budget re-prepares (cheap) and handles overflow retry.
                data = await self._chat_with_budget(body)
                self._note_turn_usage(data)
                msg = self.extract_assistant_fn(data)
                content, reasoning = split_assistant_visible(msg)
                last = content
                if thinking_on and reasoning:
                    self._set_live_reasoning(reasoning, final=True)
                elif thinking_on and self.last_reasoning:
                    self._set_live_reasoning(self.last_reasoning, final=True)
                if content:
                    self._set_live_assistant(content, final=True)
                last = self._stamp_assistant_output(content, last)
                self.messages.append(
                    self._assistant_history_message(
                        msg, content=content, reasoning=reasoning
                    )
                )
                self._emit(
                    "assistant_turn",
                    step=step - 1,
                    has_tools=bool(msg.get("tool_calls")),
                    has_reasoning=bool(thinking_on and reasoning),
                    thinking=thinking_on,
                    preview=str(content or "")[:200],
                )
                self._assistant_turns_since_refine += 1

                tool_calls = msg.get("tool_calls") or []
                if tool_calls or str(content or "").strip():
                    self._empty_assistant_streak = 0
                elif str(reasoning or "").strip():
                    # Thinking-only: keep going via thinking_only continuation;
                    # do not treat as a blank spin.
                    self._empty_assistant_streak = 0
                else:
                    self._empty_assistant_streak = (
                        int(getattr(self, "_empty_assistant_streak", 0) or 0) + 1
                    )
                if tool_calls:
                    auto = self.jobs.autonomous
                    if auto is not None:
                        note_autonomous_tool_progress(auto)
                    self._deliverable_stale = True
                    for tc in tool_calls:
                        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                        name = str(fn.get("name") or "")
                        raw_args = fn.get("arguments") or "{}"
                        args, args_ok = parse_tool_call_arguments(raw_args)
                        if not args_ok:
                            # Stream truncated mid-arguments — heal history so the
                            # next provider call is not rejected, tell the model via
                            # tool result, and continue (do not kill the run).
                            heal_invalid_tool_call_arguments(self.messages)
                            content = TRUNCATED_TOOL_ARGS_RESULT
                            self._clear_empty_ipython_repeat()
                            self._remember_thought(
                                "ipython",
                                content,
                            )
                            self.messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.get("id") if isinstance(tc, dict) else None,
                                    "name": name or "ipython",
                                    "content": content,
                                }
                            )
                            self._emit(
                                "malformed_tool_args_nudge",
                                n=0,
                                healed=1,
                                via="tool_result",
                                error="truncated_tool_call_arguments",
                            )
                            continue
                        if not isinstance(args, dict):
                            args = {}
                        if name != "ipython":
                            content = await self._non_ipython_tool_result(name, args)
                            self._clear_empty_ipython_repeat()
                        else:
                            code = str(args.get("code") or "")
                            content = await self._run_ipython(code)
                            content = self._prepare_tool_result_for_messages(code, content)
                        self._remember_thought(
                            "ipython",
                            (str(args.get("code") or "")[:240] + "\n" + str(content)[:1200]).strip(),
                        )
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.get("id"),
                                "name": name or "ipython",
                                "content": content,
                            }
                        )
                    continue

                # Children still running: wait/drain host-side without burning LLM turns.
                # Mid-wait child inbox is buffered; only ready inbox wakes the parent.
                if self._has_running_children():
                    self._mark_fanin_pending(reason="children_running")
                    if (
                        looks_like_awaiting_children(content)
                        or not str(content or "").strip()
                    ):
                        self._mark_awaiting_children_quiet(
                            reason="awaiting_text"
                            if looks_like_awaiting_children(content)
                            else "ended_turn_with_children"
                        )
                    if self.run_lifetime in ("one_shot", "chat"):
                        while (
                            self._has_running_children()
                            and not getattr(self, "_closed", False)
                        ):
                            wait_budget = self.children_wait_budget_sec()
                            remaining = wait_budget - self._children_wait_elapsed
                            if remaining <= 0:
                                outcome = await self._handle_children_wait_budget()
                                if outcome == "intervened":
                                    break
                                continue
                            slice_sec = min(self._child_wait_slice_sec(), remaining)
                            await self._wait_for_children_or_inbox(slice_sec)
                            self._children_wait_elapsed += slice_sec
                            if self._inbox_llm_pending():
                                break
                        if self._inbox_llm_pending():
                            if not self._has_running_children():
                                self._clear_awaiting_children_quiet()
                                await self._inject_inbox()
                            continue
                        if self._has_running_children():
                            # Intervened / still stuck — leave to post-loop drain.
                            break
                        # Children finished during wait — flush buffered inbox once.
                        self._clear_awaiting_children_quiet()
                        await self._inject_inbox()
                        # Fall through to continuation (single fan-in turn).
                    else:
                        # heartbeat / long-lived: pause rather than fake-complete.
                        break

                # No tool calls: one_shot/chat still waits if ready inbox arrived mid-turn.
                if self.run_lifetime in ("one_shot", "chat") and self._inbox_llm_pending():
                    continue

                # Quiescent boundary: Prime-style goal / autonomous continuation.
                # While children still run, do not push autonomous keep-alives.
                if (
                    terminal == "complete"
                    and self.run_lifetime == "one_shot"
                    and not self._has_running_children()
                ):
                    cont = await self._take_continuation_messages(
                        turn_content=content,
                        turn_reasoning=reasoning,
                    )
                    if cont:
                        self.messages.extend(cont)
                        if unlimited_turns or step < hard_turn_cap:
                            budget = max(budget, step + 1)
                            if not unlimited_turns:
                                budget = min(budget, hard_turn_cap)
                            continue
                        break
                break
            else:
                # Loop ended because rlm_max_turns ran out, not because the model
                # concluded. If the last turn was still calling tools, whatever
                # last_content holds is a stale mid-run message, not an answer.
                if terminal == "complete" and self.run_lifetime == "one_shot":
                    self._turn_budget_exhausted = True
                    self._emit(
                        "turn_budget_exhausted",
                        step=step,
                        budget=budget,
                        mid_work=bool(self._deliverable_stale),
                    )

            if self.status == "error":
                pass
            elif (
                self.run_lifetime in ("heartbeat", "chat")
                or terminal == "pause"
            ):
                # Hub chat: model often runs ipython then ghosts with empty
                # prose. AF one_shot would nudge; chat used to seal idle and
                # the hub only painted «کار ناتمام ماند». Force synthesis first.
                if (
                    self.run_lifetime == "chat" or terminal == "pause"
                ) and not getattr(self, "_closed", False):
                    try:
                        forced = await self._maybe_force_chat_answer_synthesis(
                            step=step
                        )
                        if str(forced or "").strip():
                            last = forced
                    except Exception:
                        logger.warning(
                            "rlm hub chat answer synthesis failed session=%s",
                            self.session_id,
                            exc_info=True,
                        )
                self.status = "idle"
            else:
                # one_shot: never soft-exit empty while children may still finish.
                soft_rounds = 0
                while (
                    self._has_running_children() or self._inbox_llm_pending()
                ) and not getattr(self, "_closed", False):
                    if self._inbox_llm_pending():
                        await self._inject_inbox()
                        break
                    self._mark_fanin_pending(reason="children_drain")
                    stuck = await self._drain_children_before_finish()
                    if not stuck and not self._has_running_children():
                        break
                    if stuck and self._has_running_children():
                        soft_rounds += 1
                        outcome = await self._handle_children_wait_budget()
                        self._emit(
                            "run_paused_children",
                            detail="children_still_running_after_wait",
                            outcome=outcome,
                            soft_round=soft_rounds,
                        )
                        if not self._has_running_children():
                            break
                        if soft_rounds >= int(MAX_CHILDREN_SOFT_WAIT_ROUNDS):
                            for cid in list(self.children.keys()):
                                rec = self.children.get(cid) or {}
                                if str(rec.get("status") or "") not in (
                                    "running",
                                    "idle",
                                ):
                                    continue
                                try:
                                    await self.delete_subagent(cid)
                                except Exception:
                                    logger.warning(
                                        "rlm force-cancel child failed id=%s",
                                        cid,
                                        exc_info=True,
                                    )
                                rec2 = self.children.get(cid)
                                if isinstance(rec2, dict):
                                    rec2["status"] = "cancelled"
                                task = self._child_tasks.get(cid)
                                if task is not None and not task.done():
                                    task.cancel()
                                child = self._child_sessions.get(cid)
                                if child is not None and not getattr(
                                    child, "_closed", False
                                ):
                                    child.status = "cancelled"
                            self._emit(
                                "children_force_cancelled",
                                soft_rounds=soft_rounds,
                            )
                            self._mark_fanin_pending(reason="children_force_cancelled")
                            break
                        if outcome == "intervened":
                            self._children_wait_extensions = max(
                                0,
                                int(MAX_CHILDREN_WAIT_EXTENSIONS) - 2,
                            )
                            self._children_wait_elapsed = 0.0
                        await self._wait_for_children_or_inbox(
                            min(
                                float(CHILD_WAIT_EXTENSION_SEC),
                                max(30.0, self._child_wait_slice_sec() * 4),
                            )
                        )
                        continue
                    break

                if self._inbox_llm_pending():
                    try:
                        await self._inject_inbox()
                    except Exception:
                        pass

                # After child drain / inbox: one more continuation + turn pass if needed.
                post_passes = 0
                max_post = 8
                auto = self.jobs.autonomous
                if auto is not None:
                    streak_cap = int(auto.limits.get("max_continuations") or 0)
                    # Unlimited (0): generous local post-drain passes; lifetime
                    # continuation budget is enforced elsewhere.
                    max_post = 64 if streak_cap <= 0 else max(3, streak_cap + 2)
                if not self._has_autonomous_deliverable() or getattr(
                    self, "_fanin_pending", False
                ):
                    while (
                        self.status not in ("error", "cancelled")
                        and not getattr(self, "_closed", False)
                        and post_passes < max_post
                    ):
                        cont = await self._take_continuation_messages(
                            turn_content="",
                            turn_reasoning=self.last_reasoning,
                        )
                        if not cont:
                            break
                        post_passes += 1
                        if self._turn_budget_exhausted and post_passes == 1:
                            head = cont[0]
                            head["content"] = str(
                                head.get("content") or ""
                            ) + build_turn_budget_exhausted_note(
                                int(self.rlm_synthesis_turns or 5)
                            )
                        self.messages.extend(cont)
                        synth_budget = max(
                            step + 1,
                            step + max(1, int(self.rlm_synthesis_turns or 5)),
                        )
                        more = await self._run_turn_loop_answer_retry(
                            base_system=base_system,
                            terminal=terminal,
                            start_step=step,
                            budget=synth_budget,
                        )
                        step = synth_budget
                        if str(more or "").strip():
                            last = more
                        elif str(self.last_content or "").strip():
                            last = self.last_content
                        if self._has_autonomous_deliverable():
                            break
                await self._ensure_completion_review_before_finalize()
                self._finalize_one_shot_status()
        except ContextOverflowExhausted as exc:
            # Context overflow is recoverable while the model is up — not infra error.
            self.status = "incomplete"
            self._autonomous_stop_reason = "context_overflow"
            text = format_exception_for_log(exc)
            # Keep a short diagnostic for hub chat / attach UI (was cleared →
            # empty final_answer → "پاسخی تولید نشد").
            self.error = text[:800]
            notice = (
                "Context window is full and automatic compaction could not free "
                "enough space for another model turn. Summarize or start a fresh "
                "chat, then continue."
            )
            if not str(self.last_content or "").strip():
                self.last_content = notice
            if not str(last or "").strip():
                last = notice
            logger.warning(
                "rlm context_overflow session=%s\n%s", self.session_id, text
            )
            self._emit(
                "context_overflow_incomplete",
                detail=text,
                error=text,
                notice=notice,
            )
        except Exception as exc:
            # Model / llm-proxy unavailable or unexpected crash.
            # Do not re-raise — status/error are stamped for the runner/daemon.
            text = format_exception_for_log(exc)
            self.error = text
            # AF soft-incomplete pattern for hub: gateway outages mid-work are
            # resumable (session memory kept). Stamp a user-visible notice so
            # chat finalize never collapses to «پاسخی تولید نشد.»
            outage = False
            try:
                from src.orchestrators.host_adapters.hub_final_answer import (
                    friendly_llm_outage_notice,
                    is_llm_gateway_outage,
                )

                outage = is_llm_gateway_outage(text)
                notice = friendly_llm_outage_notice(text)
            except Exception:
                outage = any(
                    n in text.lower()
                    for n in (
                        "all_backends_failed",
                        "chat_model_unavailable",
                        "catalog_unavailable",
                    )
                )
                notice = (
                    "مدل در دسترس نیست. وضعیت جلسه حفظ شده؛ بعداً ادامه دهید."
                    if outage
                    else f"RLM با خطا متوقف شد: {text[:500]}"
                )
            mid_work = bool(
                len(self.messages or []) > 2
                or self.log_lines
                or str(self.last_content or "").strip()
                or str(self.last_reasoning or "").strip()
            )
            if outage and mid_work:
                self.status = "incomplete"
                self._autonomous_stop_reason = (
                    self._autonomous_stop_reason or "llm_gateway_outage"
                )
            else:
                self.status = "error"
            if not str(self.last_content or "").strip():
                self.last_content = notice
            if not str(last or "").strip():
                last = notice
            logger.warning(
                "rlm run_error session=%s status=%s\n%s",
                self.session_id,
                self.status,
                text,
            )
            self._emit(
                "run_error",
                detail=text,
                error=text,
                notice=notice,
                status=self.status,
            )
        finally:
            self._touch()
            try:
                self._seal_children_for_terminal()
            except Exception:
                pass
            self.persist_local()
            if terminal == "complete":
                auto = self.jobs.autonomous
                self._emit(
                    "run_end",
                    status=self.status,
                    autonomous_reason=self._autonomous_stop_reason,
                    autonomous=auto.status_dict() if auto is not None else None,
                )
        return last

    def _maybe_inject_turn_budget_notice(
        self, *, step: int, budget: int, terminal: str
    ) -> None:
        """Once, shortly before rlm_max_turns runs out: tell the model to ship."""
        if self._turn_budget_notice_sent:
            return
        if terminal != "complete" or self.run_lifetime != "one_shot":
            return
        auto = self.jobs.autonomous
        if auto is None or not auto.enabled:
            return
        remaining = int(budget) - int(step) + 1
        warn_at = max(1, min(int(TURN_BUDGET_WARN_TURNS), max(1, int(budget) // 3)))
        if remaining > warn_at:
            return
        self._turn_budget_notice_sent = True
        self.messages.append(
            {
                "role": "user",
                "content": build_turn_budget_notice(
                    remaining, synthesis_turns=int(self.rlm_synthesis_turns or 0)
                ),
            }
        )
        self._emit("turn_budget_notice", remaining=remaining, budget=int(budget))

    async def _maybe_force_chat_answer_synthesis(self, *, step: int) -> str:
        """Hub chat: after tools/mid-work with no stamped answer, force a reply.

        ``run_lifetime=chat`` skips autonomous empty-assistant nudges. Without
        this, the loop seals ``idle`` with empty ``last_content`` while the
        terminal still shows ipython results — hub then invents an incomplete
        notice instead of a real answer.
        """
        if getattr(self, "_closed", False) or self.status in ("error", "cancelled"):
            return ""
        if self._has_running_children():
            return ""

        def _has_real_answer() -> bool:
            text = str(self.last_content or "").strip()
            if not text:
                return False
            if getattr(self, "_deliverable_stale", False):
                return False
            return bool(self._content_counts_as_deliverable(text))

        if _has_real_answer():
            return str(self.last_content or "")

        mid_work = bool(getattr(self, "_deliverable_stale", False))
        recent = list(self.messages or [])[-6:]
        tools_recent = any(
            isinstance(m, dict) and str(m.get("role") or "") == "tool" for m in recent
        )
        # Also force when the model ended blank (no tools this turn) with no stamp.
        blank_end = not str(self.last_content or "").strip()
        if not mid_work and not tools_recent and not blank_end:
            return ""

        from src.orchestrators.rlm.runtime.autonomous import (
            EMPTY_ASSISTANT_CONTINUATION_PROMPT,
        )

        chat_synthesis = (
            "[hub chat] You already used tools / ipython (or returned no visible "
            "answer). The user is waiting on a final reply — do NOT stop empty.\n"
            "Write a clear user-facing answer now in the user's language. "
            "Summarize findings even if searches were empty. Prefer prose over "
            "more tool calls unless a single short ipython call is required.\n"
            "Do not reply with an empty message."
        )
        max_nudges = 3
        cur_step = int(step or 0)
        for n in range(1, max_nudges + 1):
            if _has_real_answer() or getattr(self, "_closed", False):
                break
            prompt = chat_synthesis if n == 1 else EMPTY_ASSISTANT_CONTINUATION_PROMPT
            self.messages.append({"role": "user", "content": prompt})
            self._emit(
                "answer_retry_nudge",
                n=n,
                max=max_nudges,
                reason="hub_chat_empty_after_tools"
                if (mid_work or tools_recent)
                else "hub_chat_empty_reply",
            )
            self._upsert_log_line(
                "stream",
                f"forcing user-facing answer {n}/{max_nudges}…",
            )
            self._publish_stream_progress(force=True)
            # Allow a couple of provider calls (tool then prose) per nudge.
            budget = cur_step + 3
            more = await self._run_turn_loop_answer_retry(
                base_system="",
                terminal="pause",
                start_step=cur_step,
                budget=budget,
            )
            cur_step = budget
            if str(more or "").strip():
                return str(more).strip()
            if _has_real_answer():
                return str(self.last_content or "").strip()
        return str(self.last_content or "").strip()

    async def _run_turn_loop_answer_retry(
        self,
        *,
        base_system: str,
        terminal: str,
        start_step: int,
        budget: int,
    ) -> str:
        """Extra LLM turns after drain / continuation (lock already held). No nudge stack."""
        _ = (base_system, terminal)
        last = str(self.last_content or "")
        step = int(start_step or 0)
        max_steps = max(step + 1, int(budget or step + 1))
        while step < max_steps:
            step += 1
            if getattr(self, "_closed", False) or self.status in ("error", "cancelled"):
                break
            await self._inject_inbox()
            self._inject_looks()
            body: Dict[str, Any] = {
                "messages": self.messages,
                "stream": True,
                **self.gen_params,
                "model": self.model,
                "tools": [IPYTHON_TOOL_SCHEMA],
                "tool_choice": "auto",
            }
            thinking_on = self._apply_thinking_kwargs(body)
            self.prepare_chat_body(body)
            self._assert_look_pixels_on_wire(body)
            self._emit(
                "llm_call",
                model=self.model,
                step=step - 1,
                thinking=thinking_on,
                stream=True,
                persist=False,
                retry=True,
            )
            self._upsert_log_line("stream", "waiting for first token…")
            self._publish_stream_progress(force=True)
            await asyncio.sleep(0)
            if thinking_on:
                self.last_reasoning = ""
                self._reasoning_open = False
            data = await self._chat_with_budget(body)
            self._note_turn_usage(data)
            msg = self.extract_assistant_fn(data)
            content, reasoning = split_assistant_visible(msg)
            last = content
            if thinking_on and reasoning:
                self._set_live_reasoning(reasoning, final=True)
            elif thinking_on and self.last_reasoning:
                self._set_live_reasoning(self.last_reasoning, final=True)
            if content:
                self._set_live_assistant(content, final=True)
            last = self._stamp_assistant_output(content, last)
            self.messages.append(
                self._assistant_history_message(
                    msg, content=content, reasoning=reasoning
                )
            )
            self._emit(
                "assistant_turn",
                step=step - 1,
                has_tools=bool(msg.get("tool_calls")),
                preview=str(content or "")[:200],
                retry=True,
            )
            self._assistant_turns_since_refine += 1
            tool_calls = msg.get("tool_calls") or []
            if tool_calls or str(content or "").strip():
                self._empty_assistant_streak = 0
            elif str(reasoning or "").strip():
                self._empty_assistant_streak = 0
            else:
                self._empty_assistant_streak = (
                    int(getattr(self, "_empty_assistant_streak", 0) or 0) + 1
                )
            if tool_calls:
                auto = self.jobs.autonomous
                if auto is not None:
                    note_autonomous_tool_progress(auto)
                self._deliverable_stale = True
                for tc in tool_calls:
                    fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                    name = str(fn.get("name") or "")
                    raw_args = fn.get("arguments") or "{}"
                    args, args_ok = parse_tool_call_arguments(raw_args)
                    if not args_ok:
                        heal_invalid_tool_call_arguments(self.messages)
                        content = TRUNCATED_TOOL_ARGS_RESULT
                        self._clear_empty_ipython_repeat()
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.get("id") if isinstance(tc, dict) else None,
                                "name": name or "ipython",
                                "content": content,
                            }
                        )
                        self._emit(
                            "malformed_tool_args_nudge",
                            n=0,
                            healed=1,
                            via="tool_result",
                            error="truncated_tool_call_arguments",
                            retry=True,
                        )
                        continue
                    if not isinstance(args, dict):
                        args = {}
                    if name != "ipython":
                        content = await self._non_ipython_tool_result(name, args)
                        self._clear_empty_ipython_repeat()
                    else:
                        code = str(args.get("code") or "")
                        content = await self._run_ipython(code)
                        content = self._prepare_tool_result_for_messages(code, content)
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id"),
                            "name": name or "ipython",
                            "content": content,
                        }
                    )
                continue
            if self._has_running_children():
                while self._has_running_children() and not getattr(self, "_closed", False):
                    await self._wait_for_children_or_inbox(
                        min(60.0, float(CHILD_WAIT_EXTENSION_SEC))
                    )
                    if self._inbox_llm_pending():
                        break
                if self._inbox_llm_pending():
                    await self._inject_inbox()
                    continue
                if self._has_running_children():
                    break
            # Quiescent: stop this retry pass (outer loop may inject another continuation).
            break
        return str(self.last_content or last or "")

    def prepare_chat_body(
        self,
        body: Dict[str, Any],
        *,
        chars_per_token: int = CHARS_PER_TOKEN,
        sync_messages: bool = False,
    ) -> Dict[str, Any]:
        """Hard-cap messages + max_tokens to the deployed model window.

        Trimmed copies stay on the outgoing request only. The canonical
        transcript is not overwritten — otherwise a budget retry can replace
        the original user prompt with ``…[truncated]`` and the model gives up.
        """
        # Request-only: system-prompt-only agents are valid; some chat templates
        # still require a user turn on the wire (do not mutate self.messages).
        fallback = (
            str(self.heartbeat_prompt or "").strip()
            or (
                self.user_content
                if isinstance(self.user_content, str)
                else ""
            )
        )
        body["messages"] = outbound_messages_with_user_query(
            list(body.get("messages") or self.messages or []),
            fallback_text=str(fallback or ""),
        )
        preserve_all = resolve_preserve_thinking(
            model=self.model,
            cfg=self.cfg if isinstance(self.cfg, dict) else None,
            gen_params=self.gen_params if isinstance(self.gen_params, dict) else None,
        )
        replay_max = resolve_reasoning_replay_max(
            model=self.model,
            cfg=self.cfg if isinstance(self.cfg, dict) else None,
            gen_params=self.gen_params if isinstance(self.gen_params, dict) else None,
            preserve_all=preserve_all,
        )
        # Wire-only CoT trim — never mutate self.messages.
        body["messages"] = outbound_messages_trim_reasoning(
            body["messages"],
            preserve_all=preserve_all,
            max_recent_with_reasoning=replay_max,
        )
        body["messages"] = outbound_messages_sanitize_for_wire(body["messages"])
        ctx = self.context_length
        if not ctx:
            return body
        extra = 0
        try:
            extra = len(json.dumps(body.get("tools") or [], ensure_ascii=False))
        except Exception:
            extra = 400
        # Do not mutate ``self.messages`` with proactive compact here — that
        # summary wipe is reserved for 413 overflow recovery. Wire trim only.
        stats = apply_context_budget(
            body,
            ctx,
            extra_chars=extra,
            max_output_tokens=self.max_output_tokens,
            chars_per_token=chars_per_token,
        )
        fitted = body.get("messages")
        if sync_messages and isinstance(fitted, list):
            self.messages = list(fitted)
        if stats.get("applied") and (
            stats.get("dropped")
            or stats.get("truncated")
            or stats.get("media_stripped")
        ):
            self._emit(
                "context_budget",
                context_length=ctx,
                est_input=stats.get("est_input"),
                max_tokens=stats.get("max_tokens"),
                dropped=stats.get("dropped"),
                truncated=stats.get("truncated"),
                media_stripped=stats.get("media_stripped"),
            )
        return body

    def _upsert_log_line(self, tag: str, text: str) -> None:
        """Create or replace the trailing ``[tag] …`` line in the live log."""
        display = str(text or "").strip()
        if not display:
            return
        line = f"[{tag}] {display[:LOG_LINE_CHARS]}"
        prefix = f"[{tag}]"
        # Prefer replacing the most recent matching tag so mid-stream growth
        # does not flood the terminal with one line per token.
        for i in range(len(self.log_lines) - 1, -1, -1):
            if str(self.log_lines[i]).startswith(prefix):
                self.log_lines[i] = line
                break
        else:
            self.log_lines.append(line)
            if len(self.log_lines) > LOG_LINES_KEEP:
                self.log_lines = self.log_lines[-LOG_LINES_KEEP:]
        self._touch()

    def _drop_junk_assistant_markup_lines(self) -> None:
        """Remove flash artifacts like ``[assistant] <|channel>``."""
        cleaned: List[str] = []
        for line in self.log_lines:
            s = str(line)
            if s.startswith("[assistant]"):
                body = s[len("[assistant]") :].strip()
                if not body or body.startswith("<|") or body.startswith("<channel"):
                    continue
            cleaned.append(s)
        if len(cleaned) != len(self.log_lines):
            self.log_lines = cleaned
            self._touch()

    def _set_live_reasoning(self, text: str, *, final: bool = False) -> None:
        """Grow the current turn's ``[reasoning]`` line, then seal it.

        Previous turns keep their own ``[reasoning]`` rows in log order (after
        the ``[chat]``/tools that produced them). Upserting *any* last matching
        tag would overwrite turn 1 when turn 2 starts, and the UI would show
        only the latest thought stuck near the top.
        """
        raw = str(text or "")
        clipped = raw.strip() if final else raw
        self.last_reasoning = clipped if final else raw
        display = (clipped if final else raw).strip()
        if display:
            self._drop_junk_assistant_markup_lines()
            line = f"[reasoning] {display[:LOG_LINE_CHARS]}"
            last = str(self.log_lines[-1]) if self.log_lines else ""
            if self._reasoning_open and last.startswith("[reasoning]"):
                self.log_lines[-1] = line
            else:
                self.log_lines.append(line)
                if len(self.log_lines) > LOG_LINES_KEEP:
                    self.log_lines = self.log_lines[-LOG_LINES_KEEP:]
                self._reasoning_open = True
            self._touch()
        elif final:
            self._touch()
        if final:
            self._reasoning_open = False
            if clipped:
                rec = {"ts": time.time(), "kind": "reasoning", "text": clipped[:4000]}
                self.thoughts.append(rec)
                if len(self.thoughts) > 80:
                    self.thoughts = self.thoughts[-80:]

    def _set_live_assistant(self, text: str, *, final: bool = False) -> None:
        """Update a rolling ``[assistant]`` log line (visible answer only)."""
        raw = str(text or "")
        clipped = raw.strip() if final else raw
        display = (clipped if final else raw).strip()
        if not display:
            return
        if display.startswith("<|") or display.startswith("<channel"):
            return
        self._upsert_log_line("assistant", display)
        if final and clipped:
            rec = {"ts": time.time(), "kind": "assistant", "text": clipped[:2000]}
            self.thoughts.append(rec)
            if len(self.thoughts) > 80:
                self.thoughts = self.thoughts[-80:]

    def _set_live_ipython_preview(self, code: str) -> None:
        preview = str(code or "").strip()
        if not preview:
            self._upsert_log_line("ipython", "…")
            return
        self._upsert_log_line("ipython", preview[:LOG_LINE_CHARS])

    def live_progress_snapshot(
        self, *, focus_session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """RLM snapshot for mid-stream progress polls.

        When ``focus_session_id`` is set (including empty string), child rows
        only include full ``log_lines`` for that session; others stay light so
        SSE payloads stay small while many subagents run.
        """
        # During an active chat stream, expose the live visible buffer so hub
        # SSE can emit answer deltas (``last_content`` is only stamped at turn end).
        live_answer = str(self.last_content or "")
        if self._stream_content_buf or getattr(self, "_stream_first_token", None) is not None:
            try:
                vis, _thought = split_assistant_visible(
                    {"role": "assistant", "content": self._stream_content_buf or ""}
                )
                live_answer = vis
            except Exception:
                live_answer = str(self._stream_content_buf or "") or live_answer
        snap: Dict[str, Any] = {
            "session_id": self.session_id,
            "status": self.status,
            "error": self.error,
            "node_id": self.node_id or "",
            "parent_session_id": self.parent_session_id,
            "last_content": live_answer,
            "last_reasoning": (self.last_reasoning or "")[:8000],
            "enable_thinking": bool(self.thinking_enabled()),
            "log_lines": list(self.log_lines[-LOG_LINES_KEEP:]),
            "thoughts_tail": list(self.thoughts[-20:]),
            "live": bool(
                not self._closed
                and self.status == "running"
                and not getattr(self, "_shutdown_requested", False)
            ),
            "updated_at": self.updated_at,
        }
        # NODE LOGS chips bind to rlm.files — keep mid-run stamped deliverables
        # visible (Explorer already has them via Storage persist).
        files = slim_file_envelopes(getattr(self, "_cached_files", None) or [])
        if not files:
            mapping = getattr(self, "deliverable_artifacts", None)
            if isinstance(mapping, dict):
                files = [
                    {
                        "filename": str(name),
                        "artifact_id": str(artifact_id),
                        "kind": "file",
                    }
                    for name, artifact_id in mapping.items()
                    if name and artifact_id
                ]
        if files:
            snap["files"] = files
        # Parent polls must surface child status/logs without a separate window.
        if not self.parent_session_id and self.children:
            snap["children"] = self.list_subagents(
                focus_session_id=focus_session_id
            )
        return snap

    def light_progress_snapshot(
        self, *, focus_session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Badge/status row without bulky log buffers (non-focused terminals).

        When ``focus_session_id`` names a child, that child's full ``log_lines``
        are still included so selecting a subagent terminal is not empty.
        """
        focus = str(focus_session_id or "").strip()
        return {
            "session_id": self.session_id,
            "status": self.status,
            "error": self.error,
            "node_id": self.node_id or "",
            "parent_session_id": self.parent_session_id,
            "last_content": (self.last_content or "")[:160],
            "live": bool(
                not self._closed
                and self.status == "running"
                and not getattr(self, "_shutdown_requested", False)
            ),
            "updated_at": self.updated_at,
            "model": self.model,
            "log_lines": [],
            "thoughts_tail": [],
            "children": (
                self.list_subagents(focus_session_id=focus)
                if (not self.parent_session_id and self.children)
                else []
            ),
        }

    def _ui_focus_session_id(self) -> str:
        """Which terminal the UI is watching (full logs); empty = badges only for kids."""
        try:
            def ui_view_session_get(*_a, **_k):  # host optional; agent_flow UI not required
                return None

            focus = ui_view_session_get(self.run_id)
            if focus:
                return focus
        except Exception:
            pass
        return str(self.session_id or "").strip()

    def _publish_stream_progress(self, *, force: bool = False) -> None:
        now = time.time()
        reason_len = len(self.last_reasoning or "")
        # Cursor must match the visible string hub_sse slices (not raw markup).
        try:
            vis_buf, _ = split_assistant_visible(
                {"role": "assistant", "content": self._stream_content_buf or ""}
            )
            content_len = len(vis_buf)
        except Exception:
            content_len = len(self._stream_content_buf or "")
        tool_len = sum(
            len(str(((tc.get("function") or {}) if isinstance(tc, dict) else {}).get("arguments") or ""))
            for tc in (self._stream_tool_calls or [])
        )
        grew = (
            (reason_len - int(self._stream_reason_logged_len or 0))
            + (content_len - int(self._stream_content_logged_len or 0))
            + (tool_len - int(self._stream_tool_logged_len or 0))
        )
        if not force and not (
            now - float(self._stream_pub_at or 0) >= 0.2 or grew >= 24
        ):
            return
        self._stream_pub_at = now
        self._stream_reason_logged_len = reason_len
        self._stream_content_logged_len = content_len
        self._stream_tool_logged_len = tool_len
        if not self.on_event:
            return
        try:
            focus = self._ui_focus_session_id()
            self.on_event(
                {
                    "ts": now,
                    "kind": "stream_progress",
                    "session_id": self.session_id,
                    "reasoning_len": reason_len,
                    "content_len": content_len,
                    "rlm": self.live_progress_snapshot(focus_session_id=focus),
                }
            )
        except Exception:
            pass

    def _on_chat_stream_delta(self, delta: Dict[str, Any]) -> None:
        """Live UI updates while an RLM chat stream is in flight.

        Gemma 4 often streams thinking inside ``content`` as
        ``<|channel>thought … <channel|>`` rather than ``reasoning_content``.
        Tool-call argument streams (ipython code) are also mirrored live —
        otherwise the UI freezes between end-of-reasoning and tool execution.
        """
        from src.orchestrators.rlm.runtime.chat_stream import merge_tool_call_delta

        reasoning_piece = str(
            delta.get("reasoning") or delta.get("reasoning_content") or ""
        )
        content_piece = str(delta.get("content") or "")
        tool_calls = delta.get("tool_calls")

        if reasoning_piece:
            self._set_live_reasoning(str(self.last_reasoning or "") + reasoning_piece)

        if content_piece:
            self._stream_content_buf = str(self._stream_content_buf or "") + content_piece
            vis, thought = split_assistant_visible(
                {"role": "assistant", "content": self._stream_content_buf}
            )
            if thought and len(thought) >= len(str(self.last_reasoning or "").strip()):
                self._set_live_reasoning(thought)
            if vis.strip():
                self._set_live_assistant(vis)

        if tool_calls:
            merge_tool_call_delta(self._stream_tool_calls, tool_calls)
            for tc in self._stream_tool_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                name = str(fn.get("name") or "")
                args = str(fn.get("arguments") or "")
                if name == "ipython" or (not name and args):
                    # Parse code field when possible; otherwise show raw args growth.
                    code = args
                    try:
                        parsed = json.loads(args) if args.strip().startswith("{") else None
                        if isinstance(parsed, dict) and "code" in parsed:
                            code = str(parsed.get("code") or "")
                    except Exception:
                        pass
                    self._set_live_ipython_preview(code or args)

        self._publish_stream_progress()

    def _stream_failure_is_fallbackable(self, exc: BaseException) -> bool:
        text = str(exc or "").lower()
        cause = getattr(exc, "__cause__", None)
        cause_text = str(cause or "").lower()
        blob = f"{text} {type(exc).__name__.lower()} {cause_text}"
        if self._stream_tool_calls:
            return False
        # Context overflow needs compact/recover — non-stream retry just 400s again.
        try:
            if looks_like_context_overflow(str(exc)):
                return False
        except Exception:
            pass
        # Host chat ports already reconnect on gateway outages (AF parity).
        # Falling back to non-stream would burn a second full retry budget.
        if "all_backends_failed" in text or "chat_model_unavailable" in text:
            return False
        # Transport drops mid-thinking: allow non-stream fallback when we have
        # no assistant content yet (reasoning alone is OK to discard/retry).
        transport = any(
            n in blob
            for n in (
                "readerror",
                "remoteprotocolerror",
                "connecterror",
                "networkerror",
                "llm stream failed",
                "server disconnected",
                "incomplete read",
            )
        )
        if transport and not str(self._stream_content_buf or "").strip():
            return True
        if self._stream_content_buf or str(self.last_reasoning or "").strip():
            return False
        return (
            "400" in text
            or "stream" in text and "not" in text
            or "bad request" in text
        )

    def _clear_stream_wait_line(self) -> None:
        before = len(self.log_lines)
        self.log_lines = [ln for ln in self.log_lines if not str(ln).startswith("[stream]")]
        if len(self.log_lines) != before:
            self._touch()

    def _mark_stream_token_seen(self) -> None:
        """First SSE payload arrived — stop TTFT wait indicator."""
        if getattr(self, "_stream_first_token", None) is not None:
            try:
                self._stream_first_token.set()
            except Exception:
                pass
        self._clear_stream_wait_line()

    async def _ttft_heartbeat(
        self, first_token: "asyncio.Event", *, t0: Optional[float] = None
    ) -> None:
        """Keep the live log moving while the model is still silent (TTFT)."""
        started = float(t0) if t0 is not None else time.time()
        try:
            while not first_token.is_set():
                try:
                    await asyncio.wait_for(first_token.wait(), timeout=1.0)
                    break
                except asyncio.TimeoutError:
                    elapsed = time.time() - started
                    self._upsert_log_line(
                        "stream", f"waiting for first token… {elapsed:.0f}s"
                    )
                    self._publish_stream_progress(force=True)
        except asyncio.CancelledError:
            return

    async def _invoke_chat(self, body: Dict[str, Any]) -> Dict[str, Any]:
        token = ACTIVE_RLM_SESSION_ID.set(str(self.session_id or ""))
        await self._resource_budget.reserve_llm()
        try:
            return await self._invoke_chat_unbudgeted(body)
        finally:
            await self._resource_budget.release_llm()
            ACTIVE_RLM_SESSION_ID.reset(token)

    async def _invoke_chat_unbudgeted(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Prefer SSE streaming when ``chat_stream_fn`` is wired (live reasoning).

        Catalog ``max_output_tokens`` still caps each provider call; when the
        model hits ``finish_reason=length`` mid-answer we stitch more chunks
        so chatbot / AF replies can grow well past one window.
        """
        from src.orchestrators.rlm.runtime.chat_stream import (
            invoke_with_length_continuation,
            max_output_continuations,
            max_tool_arg_retries,
        )

        cfg = self.cfg if isinstance(self.cfg, dict) else None
        cont_limit = max_output_continuations(cfg)
        tool_retry_limit = max_tool_arg_retries(cfg)

        def _prepare_cont(req: Dict[str, Any]) -> Dict[str, Any]:
            # Re-fit growing stitch history into the context window.
            try:
                return self.prepare_chat_body(dict(req))
            except Exception:
                return req

        def _on_continue(n: int, _partial: Dict[str, Any]) -> None:
            self._emit("output_length_continue", n=n, max=cont_limit)
            self._upsert_log_line(
                "stream",
                f"output length cap — continuing chunk {n}/{cont_limit}…",
            )

        def _on_tool_retry(n: int, _partial: Dict[str, Any]) -> None:
            # Fresh stream buffers so the re-emitted tool call does not append
            # onto truncated argument deltas from the failed completion.
            self._stream_tool_calls = []
            self._stream_tool_logged_len = 0
            self._stream_content_buf = ""
            self._emit(
                "malformed_tool_args_nudge",
                n=n,
                healed=1,
                via="length_invoke_retry",
                error="truncated_tool_call_arguments",
            )
            self._upsert_log_line(
                "stream",
                f"truncated tool JSON — retry {n}/{tool_retry_limit}…",
            )

        # Internal calls (compaction, reviews, refine) send stream=False: they
        # must not go through the live buffer that hub chat shows as the answer.
        if self.chat_stream_fn is not None and body.get("stream") is not False:
            req = dict(body)
            req["stream"] = True
            self._stream_pub_at = 0.0
            self._stream_reason_logged_len = 0
            self._stream_content_logged_len = 0
            self._stream_content_buf = ""
            self._stream_tool_calls = []
            self._stream_tool_logged_len = 0
            first_token = asyncio.Event()
            self._stream_first_token = first_token
            # Sync wait line + progress *before* any await on HTTP so the UI is
            # never stuck on a lone "[chat] … stream" line during connect/TTFT.
            ttft_t0 = time.time()
            self._upsert_log_line("stream", "waiting for first token…")
            # Tell hub SSE to reset answer cursors (seeded from prior last_content)
            # so this turn's stream / length-continuation chunks are not skipped.
            try:
                if self.on_event:
                    self.on_event(
                        {
                            "ts": ttft_t0,
                            "kind": "stream_begin",
                            "session_id": self.session_id,
                        }
                    )
            except Exception:
                pass
            self._publish_stream_progress(force=True)
            await asyncio.sleep(0)
            hb = asyncio.create_task(
                self._ttft_heartbeat(first_token, t0=ttft_t0)
            )
            await asyncio.sleep(0)

            def on_delta(delta: Dict[str, Any]) -> None:
                if not first_token.is_set():
                    self._mark_stream_token_seen()
                self._on_chat_stream_delta(delta)

            async def _stream_once(
                one_body: Dict[str, Any], on_delta=None
            ) -> Dict[str, Any]:
                one = dict(one_body)
                one["stream"] = True
                try:
                    return await self.chat_stream_fn(one, on_delta=on_delta)
                except Exception as exc:
                    if not self._stream_failure_is_fallbackable(exc):
                        raise
                    self._clear_stream_wait_line()
                    self._emit(
                        "chat_stream_fallback",
                        error=str(exc)[:LOG_LINE_CHARS],
                    )
                    fb = dict(one_body)
                    fb["stream"] = False
                    return await self.chat_fn(fb)

            try:
                return await invoke_with_length_continuation(
                    _stream_once,
                    req,
                    on_delta=on_delta,
                    max_continuations=cont_limit,
                    max_tool_retries=tool_retry_limit,
                    prepare_body=_prepare_cont,
                    on_continue=_on_continue,
                    on_tool_retry=_on_tool_retry,
                )
            finally:
                first_token.set()
                self._clear_stream_wait_line()
                if not hb.done():
                    hb.cancel()
                # Never let a cancelled heartbeat poison the chat return path.
                await asyncio.gather(hb, return_exceptions=True)
                self._stream_first_token = None
                # Deferred disk write from llm_call (persist=False).
                try:
                    self.persist_local()
                except Exception:
                    pass
        req = dict(body)
        req["stream"] = False

        async def _chat_once(
            one_body: Dict[str, Any], on_delta=None
        ) -> Dict[str, Any]:
            del on_delta
            one = dict(one_body)
            one["stream"] = False
            return await self.chat_fn(one)

        return await invoke_with_length_continuation(
            _chat_once,
            req,
            max_continuations=cont_limit,
            max_tool_retries=tool_retry_limit,
            prepare_body=_prepare_cont,
            on_continue=_on_continue,
            on_tool_retry=_on_tool_retry,
        )

    async def _retry_after_malformed_tool_args(
        self, body: Dict[str, Any], error_text: str
    ) -> Dict[str, Any]:
        """Heal truncated tool-call JSON in history, tell the model, retry (twice)."""
        last_error = str(error_text or "")
        while self._malformed_tool_args_nudges < MAX_MALFORMED_TOOL_ARGS_NUDGES:
            self._malformed_tool_args_nudges += 1
            healed = heal_invalid_tool_call_arguments(self.messages)
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{MALFORMED_TOOL_ARGS_NUDGE} "
                        f"(attempt {self._malformed_tool_args_nudges}/"
                        f"{MAX_MALFORMED_TOOL_ARGS_NUDGES})"
                    ),
                }
            )
            body["messages"] = list(self.messages)
            self.prepare_chat_body(body)
            self._emit(
                "malformed_tool_args_nudge",
                n=self._malformed_tool_args_nudges,
                healed=healed,
                error=last_error[:400],
            )
            try:
                return await self._invoke_chat(body)
            except Exception as exc:
                last_error = str(exc)
                if looks_like_malformed_tool_args_error(last_error):
                    continue
                raise
        raise RuntimeError(last_error)

    async def _chat_with_budget(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Budget-trim then chat (used by compact/refine and overflow retry)."""
        self.prepare_chat_body(body)
        try:
            out = await self._invoke_chat(body)
            self._overflow_recovery = "idle"
            return out
        except Exception as exc:
            text = str(exc)
            if looks_like_malformed_tool_args_error(text):
                return await self._retry_after_malformed_tool_args(body, text)
            if looks_like_invalid_image_error(text):
                return await self._retry_after_invalid_look_media(body, text)
            if not (
                parse_context_overflow_error(text) or looks_like_context_overflow(text)
            ):
                raise
            if getattr(self, "_overflow_recovery", "idle") != "idle":
                self._context_overflow_exhausted = True
                self._autonomous_stop_reason = "context_overflow"
                detail = (
                    "Context/request overflow after compact retry: "
                    f"{text[:800]}"
                )
                self.error = detail
                raise ContextOverflowExhausted(detail) from exc
            self._overflow_recovery = "attempted"
            self._context_budget_overflow = True
            parsed = parse_context_overflow_error(text)
            ctx = (
                coerce_positive_int((parsed or {}).get("context_length"))
                or self.context_length
                or 8192
            )
            self.context_length = ctx
            ctk = body.get("chat_template_kwargs")
            if isinstance(ctk, dict) and ctk.get("enable_thinking"):
                ctk["enable_thinking"] = False

            # 413 ladder + compact live in ``_overflow_recover_chat``.
            recovered = await self._overflow_recover_chat(body, text)
            self._overflow_recovery = "idle"
            return recovered

    async def _retry_after_invalid_look_media(
        self, body: Dict[str, Any], error_text: str
    ) -> Dict[str, Any]:
        """Strip corrupt look images from history and retry once (not overflow)."""
        if getattr(self, "_look_media_invalid_recovery", False):
            raise RuntimeError(
                "look_media_invalid_after_strip:"
                f"{str(error_text or '')[:600]}"
            )
        self._look_media_invalid_recovery = True
        try:
            stripped = self._strip_prior_look_media(keep_last=False)
            # Also drop any still-queued looks so we do not re-inject the same bytes.
            pending_n = len(self._pending_looks or [])
            self._pending_looks = []
            self._emit(
                "look_media_invalid",
                stripped=stripped,
                pending_cleared=pending_n,
                error=str(error_text or "")[:400],
            )
            note = (
                "[system] One or more rlm.look images were rejected by the model "
                "(corrupt or truncated image file). Native media was removed from "
                "chat history; files remain on disk. Prefer a different file or "
                "re-export the image before looking again.\n"
                f"Detail: {str(error_text or '')[:400]}"
            )
            self.messages.append({"role": "user", "content": note})
            body = dict(body)
            body["messages"] = list(self.messages)
            self.prepare_chat_body(body)
            try:
                return await self._invoke_chat(body)
            except Exception as exc2:
                text2 = str(exc2)
                if looks_like_invalid_image_error(text2):
                    # Last resort: strip again and fail clearly (should be rare).
                    self._strip_prior_look_media(keep_last=False)
                    raise RuntimeError(
                        "look_media_invalid_after_strip:"
                        f"{text2[:600]}"
                    ) from exc2
                raise
        finally:
            self._look_media_invalid_recovery = False

    async def _overflow_recover_chat(
        self, body: Dict[str, Any], error_text: str
    ) -> Dict[str, Any]:
        """Mitigate token overflow / 413 then retry once (or ladder for look 413)."""
        last_error = str(error_text or "")

        if looks_like_request_too_large(last_error):
            # Stage 1: if multiple media in the latest look, keep one and queue rest.
            from src.orchestrators.rlm.runtime.native_media import (
                split_latest_look_to_one,
            )

            rebuilt, deferred, changed = split_latest_look_to_one(self.messages)
            if changed:
                self.messages = rebuilt
                if deferred:
                    self._pending_looks = list(deferred) + list(self._pending_looks)
                self._emit("look_overflow_split", deferred=len(deferred))
                body = dict(body)
                body["messages"] = list(self.messages)
                self.prepare_chat_body(body, chars_per_token=1)
                try:
                    return await self._invoke_chat(body)
                except Exception as exc:
                    last_error = str(exc)
                    if not (
                        parse_context_overflow_error(last_error)
                        or looks_like_context_overflow(last_error)
                    ):
                        raise

            # Stage 2: recompress remaining images harder (~512KiB encoded).
            if looks_like_request_too_large(last_error):
                from src.orchestrators.rlm.runtime.native_media import (
                    recompress_look_media_in_messages,
                )

                rebuilt, n = recompress_look_media_in_messages(
                    self.messages, max_encoded_bytes=512 * 1024
                )
                if n:
                    self.messages = rebuilt
                    self._emit("look_overflow_recompress", recompressed=n)
                    body = dict(body)
                    body["messages"] = list(self.messages)
                    self.prepare_chat_body(body, chars_per_token=1)
                    try:
                        return await self._invoke_chat(body)
                    except Exception as exc:
                        last_error = str(exc)
                        if not (
                            parse_context_overflow_error(last_error)
                            or looks_like_context_overflow(last_error)
                        ):
                            raise

            # Stage 3: strip all look media (files stay on disk).
            if looks_like_request_too_large(last_error):
                self._strip_prior_look_media(keep_last=False)

        # Semantic compact only for HTTP 413 / request-too-large — not for routine
        # token pressure (that path uses wire trim via prepare_chat_body).
        if looks_like_request_too_large(last_error):
            try:
                await self.maybe_compact(force=True)
            except Exception:
                self.compact_messages(keep_last=4)
        body = dict(body)
        body["messages"] = list(self.messages)
        self.prepare_chat_body(body, chars_per_token=1)
        self._emit(
            "context_budget_retry",
            context_length=self.context_length,
            error=last_error[:400],
        )
        try:
            return await self._invoke_chat(body)
        except Exception as exc2:
            retry_text = str(exc2)
            if not (
                parse_context_overflow_error(retry_text)
                or looks_like_context_overflow(retry_text)
            ):
                raise
            self._context_overflow_exhausted = True
            self._autonomous_stop_reason = "context_overflow"
            detail = (
                "Context window overflow after compact retry: "
                f"{retry_text[:800]}"
            )
            self.error = detail
            raise ContextOverflowExhausted(detail) from exc2

    def _strip_prior_look_media(self, *, keep_last: bool = False) -> int:
        from src.orchestrators.rlm.runtime.native_media import (
            strip_look_media_from_messages,
        )

        rebuilt, stripped = strip_look_media_from_messages(
            self.messages, keep_last_media_message=keep_last
        )
        if stripped:
            self.messages = rebuilt
            self._emit("look_media_stripped", stripped=stripped)
            self._touch()
        return int(stripped)

    async def wait_until_paused(self, timeout: float = 180.0) -> str:
        """Block until the session pauses or finishes.

        Fresh sessions start as ``idle``; that must *not* count as paused until
        the session has actually entered ``running`` at least once.
        """
        deadline = time.time() + max(0.1, float(timeout))
        while time.time() < deadline:
            st = self.status
            if st in ("completed", "error", "cancelled"):
                return self.last_content
            if st == "idle" and self._was_running:
                return self.last_content
            await asyncio.sleep(0.05)
        return self.last_content

    def _apply_empty_ipython_repeat_guard(self, code: str, content: str) -> str:
        """If the same silent cell repeats, append a change-approach nudge."""
        text, last, streak, nudged = note_empty_ipython_repeat(
            code=code,
            result_text=content,
            last_code=self._empty_ipython_last_code,
            streak=self._empty_ipython_streak,
        )
        self._empty_ipython_last_code = last
        self._empty_ipython_streak = streak
        if nudged:
            self._emit(
                "empty_ipython_repeat_nudge",
                streak=EMPTY_IPYTHON_REPEAT_THRESHOLD,
                code_preview=normalize_ipython_code(code)[:240],
            )
        return text

    def _clear_empty_ipython_repeat(self) -> None:
        self._empty_ipython_last_code = None
        self._empty_ipython_streak = 0

    def _arm_post_compact_guard(self) -> None:
        self._post_compact_guard = True
        self._post_compact_last_key = None
        self._post_compact_streak = 0
        self._post_compact_distinct_ok = 0
        self._post_compact_turns_left = int(POST_COMPACT_GUARD_MAX_TURNS)

    def _clear_post_compact_guard(self) -> None:
        self._post_compact_guard = False
        self._post_compact_last_key = None
        self._post_compact_streak = 0
        self._post_compact_distinct_ok = 0
        self._post_compact_turns_left = 0

    def _post_compact_turn_tick(self) -> None:
        if not self._post_compact_guard:
            return
        left = int(self._post_compact_turns_left or 0) - 1
        self._post_compact_turns_left = left
        if left <= 0:
            self._clear_post_compact_guard()

    def _post_compact_loop_key(self, code: str) -> str:
        from src.orchestrators.rlm.runtime.compact_resume import extract_look_path

        look = extract_look_path(code)
        if look:
            return f"look:{look.casefold()}"
        return f"code:{normalize_ipython_code(code)}"

    def _apply_post_compact_loop_guard(self, code: str, content: str) -> str:
        """After compact, nudge when the same cell/look path repeats."""
        if not self._post_compact_guard:
            return content
        key = self._post_compact_loop_key(code)
        if not key or key in ("code:", "look:"):
            return content
        text = str(content or "")
        if self._post_compact_last_key is not None and key == self._post_compact_last_key:
            self._post_compact_streak = max(1, int(self._post_compact_streak or 0)) + 1
        else:
            self._post_compact_streak = 1
            self._post_compact_last_key = key
            # Distinct successful tool result advances the clearance counter.
            if text.strip() and text.strip() != "(empty)" and "[offloaded" not in text[:40]:
                self._post_compact_distinct_ok = int(self._post_compact_distinct_ok or 0) + 1
                if self._post_compact_distinct_ok >= int(POST_COMPACT_GUARD_DISTINCT_OK):
                    self._clear_post_compact_guard()
                    return text
        if self._post_compact_streak >= int(POST_COMPACT_LOOP_THRESHOLD):
            nudged = f"{text}\n\n{POST_COMPACT_LOOP_NUDGE}"
            if self._last_compact_next_steps:
                nudged += f"\nNext Steps reminder:\n{self._last_compact_next_steps[:600]}"
            self._emit(
                "post_compact_loop_nudge",
                key=key[:240],
                streak=self._post_compact_streak,
            )
            self._post_compact_streak = 0
            self._post_compact_last_key = None
            return nudged
        return text

    def _maybe_offload_tool_content(self, content: str, *, kind: str = "ipython") -> str:
        from src.orchestrators.rlm.runtime.tool_offload import (
            offload_large_tool_content,
        )

        stub, path = offload_large_tool_content(
            str(self.artifact_dir or ""),
            content,
            kind=kind,
        )
        if path:
            self._emit(
                "tool_offload",
                path=path,
                chars=len(str(content or "")),
                stub_chars=len(stub),
            )
        return stub

    def _prepare_tool_result_for_messages(self, code: str, content: str) -> str:
        """Offload bulky bodies, then apply post-compact + empty-repeat guards."""
        text = self._maybe_offload_tool_content(content, kind="ipython")
        text = self._apply_post_compact_loop_guard(code, text)
        return self._apply_empty_ipython_repeat_guard(code, text)

    def _persist_compact_why_not(self, summary: str) -> None:
        from src.orchestrators.rlm.runtime.compact_resume import (
            why_not_memory_content,
        )

        body = why_not_memory_content(summary)
        if not body.strip():
            return
        try:
            self.harness.upsert(
                kind="memory",
                content=body,
                entry_id="why_not",
                title="why_not",
                meta={"source": "compaction"},
            )
        except Exception as exc:
            self._emit("compact_why_not_error", error=str(exc)[:240])

    def _latest_user_text(self) -> str:
        for m in reversed(self.messages or []):
            if not isinstance(m, dict) or m.get("role") != "user":
                continue
            content = m.get("content")
            if isinstance(content, list):
                return "\n".join(
                    str(p.get("text") or "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ).strip()
            return str(content or "").strip()
        return ""

    def _inject_compact_continue_nudge(self, summary: str) -> None:
        from src.orchestrators.rlm.runtime.compact_resume import (
            build_continue_nudge,
            parse_checkpoint_sections,
        )

        sections = parse_checkpoint_sections(summary)
        self._last_compact_next_steps = str(sections.get("next steps") or "").strip()
        request = ""
        if str(getattr(self, "run_lifetime", "") or "") == "chat":
            request = str(getattr(self, "_chat_turn_request", "") or "")
        body = build_continue_nudge(summary, current_request=request)
        self.messages.append({"role": "user", "content": body})
        self._transcript_written = len(self.messages)
        try:
            from src.orchestrators.rlm.runtime.transcript import (
                append_custom_message,
            )

            append_custom_message(
                self.artifact_dir,
                custom_type="compaction_resume",
                content=body,
                display=False,
            )
        except Exception:
            pass

    async def _run_ipython(self, code: str) -> str:
        assert self.kernel is not None
        self._set_live_ipython_preview(code or "")
        self._emit("ipython", code_preview=(code or "")[:LOG_LINE_CHARS])
        started = time.time()
        exec_task = asyncio.create_task(
            self.kernel.execute(code, timeout_sec=float(self.kernel_timeout))
        )
        try:
            while not exec_task.done():
                done, _pending = await asyncio.wait({exec_task}, timeout=1.0)
                if done:
                    break
                elapsed = time.time() - started
                self._upsert_log_line("wait", f"ipython running {elapsed:.0f}s")
                self._publish_stream_progress(force=True)
            result = exec_task.result()
        except Exception:
            if not exec_task.done():
                exec_task.cancel()
                try:
                    await exec_task
                except Exception:
                    pass
            raise
        # Clear the wait line once execution finishes.
        self.log_lines = [ln for ln in self.log_lines if not str(ln).startswith("[wait]")]
        self._touch()
        parts: List[str] = []
        if result.get("stdout"):
            parts.append(str(result["stdout"]))
        if result.get("stderr"):
            parts.append("stderr:\n" + str(result["stderr"]))
        if result.get("result") is not None:
            parts.append("result: " + str(result["result"]))
        if result.get("error"):
            parts.append("error: " + str(result["error"]))
        text = "\n".join(parts).strip() or "(empty)"
        # Match Prime Agent's default IPython output cap (64 KiB).
        if len(text) > LOG_LINE_CHARS:
            text = text[: LOG_LINE_CHARS - 36] + "\n…[truncated]"
        hint = module_not_found_hint(text)
        if hint:
            text = f"{text}\n{hint}"
            self._emit("kernel_env_hint", preview=hint[:400])
        self._emit("ipython_result", preview=text[:LOG_LINE_CHARS])
        return text

    async def _inject_inbox(self) -> None:
        if self._inbox_llm_cursor >= len(self.inbox):
            return
        # Inject ready items in order; stop at the first deferred fan-in buffer
        # entry so mid-wait child/rlm notices stay queued until children settle.
        while self._inbox_llm_cursor < len(self.inbox):
            item = self.inbox[self._inbox_llm_cursor]
            if self._inbox_item_deferred_while_children(item):
                break
            self._inbox_llm_cursor += 1
            role = str(item.get("from_role") or "child")
            name = str(item.get("from_name") or "")
            text = str(item.get("text") or "")
            kind = str(item.get("kind") or "")
            if kind == "completed_without_reply" or role == "rlm" or kind in (
                "failed",
                "cancelled",
                "child_error",
                "child_failed",
            ):
                # Runtime notices: content is already a full labeled sentence.
                self.messages.append({"role": "user", "content": text})
            else:
                header = f"[agent_message from {role}:{name or '?'}]"
                self.messages.append({"role": "user", "content": f"{header}\n{text}"})
            self._emit(
                "inbox_inject",
                from_role=role,
                from_name=name,
                inbox_kind=kind or None,
            )

    def queue_look(self, *, filename: str, kind: str, part: Dict[str, Any]) -> None:
        from src.orchestrators.rlm.runtime.native_media import part_encoded_bytes

        enc = part_encoded_bytes(part)
        self._pending_looks.append(
            {
                "filename": str(filename or "").strip() or "file",
                "kind": str(kind or "").strip(),
                "part": part,
                "encoded_bytes": enc,
            }
        )
        self._touch()

    def _strip_look_media_before_compact(self) -> int:
        """Drop already-shown look pixels from history before compaction.

        Compaction omits multimodal parts (``[image_url omitted]``). We must
        remove base64 from ``messages`` first so the summarizer / hydrate path
        does not keep multi-MB payloads.

        Important: do **not** push those pixels back onto ``_pending_looks``.
        They were already injected on a prior turn; re-queueing them prepends
        the same filenames every compact and blocks the never-shown queue
        (``+32 → +30`` while always showing the same two names).
        """
        from src.orchestrators.rlm.runtime.native_media import (
            requeue_looks_from_messages,
        )

        rebuilt, pending = requeue_looks_from_messages(self.messages)
        if not pending:
            return 0
        self.messages = rebuilt
        self._emit(
            "look_strip_before_compact",
            count=len(pending),
            filenames=",".join(
                str(p.get("filename") or "") for p in pending[:8] if isinstance(p, dict)
            ),
        )
        return len(pending)

    def _inject_looks(self) -> None:
        if not self._pending_looks:
            self._looks_injected_this_turn = False
            return
        # Hard kind caps (≤2 images / ≤1 video) plus soft context-window and
        # encoded-byte budgets so remote gateways do not 413 on multi-base64 bodies.
        from src.orchestrators.rlm.runtime.native_media import (
            look_turn_limit_hint,
            select_look_batch_for_turn,
        )

        batch, deferred = select_look_batch_for_turn(
            self._pending_looks,
            context_length=self.context_length,
            total_budget=self._look_batch_limit(),
        )
        self._pending_looks = deferred
        names = [str(item.get("filename") or "file") for item in batch]
        kinds = [str(item.get("kind") or "") for item in batch]
        header = "[rlm.look] " + ", ".join(
            f"{n}" + (f" ({k})" if k else "") for n, k in zip(names, kinds)
        )
        if deferred:
            header += (
                f"\n({len(deferred)} more queued file(s) will be shown on your next "
                "turn automatically — do not call rlm.look for them again; note what "
                f"you see here first, e.g. save findings to a variable or file. "
                f"Limit: {look_turn_limit_hint()}.)"
            )
        parts: List[Any] = [{"type": "text", "text": header}]
        for item in batch:
            part = item.get("part")
            if isinstance(part, dict):
                parts.append(part)
        self.messages.append({"role": "user", "content": parts})
        self._looks_injected_this_turn = bool(batch)
        self._emit(
            "look_inject",
            filenames=",".join(names),
            count=len(batch),
            deferred=len(deferred),
        )

    def _assert_look_pixels_on_wire(self, body: Dict[str, Any]) -> None:
        """Detect budget/sanitize wiping active look media before the model sees it.

        When this fires, the model only got a ``[rlm.look]`` text stub and will
        keep calling look forever. Prefer recovering by re-copying session
        messages (still multimodal) onto the body over sending a blind prompt.
        """
        if not getattr(self, "_looks_injected_this_turn", False):
            return
        from src.orchestrators.rlm.runtime.context_budget import _is_media_part

        def _media_count(messages: Any) -> int:
            n = 0
            for msg in messages or []:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                n += sum(1 for p in content if _is_media_part(p))
            return n

        wire_n = _media_count(body.get("messages"))
        sess_n = _media_count(self.messages)
        if wire_n > 0:
            return
        self._emit(
            "look_wire_blind",
            session_media=sess_n,
            wire_media=wire_n,
        )
        if sess_n > 0:
            # Budget trim wiped pixels on the request copy — restore from session.
            body["messages"] = list(self.messages)
            # Re-run light sanitize without re-trimming media away.
            body["messages"] = outbound_messages_sanitize_for_wire(body["messages"])
            if _media_count(body.get("messages")) > 0:
                self._emit("look_wire_restored", media=_media_count(body.get("messages")))
                # Soften completion room rather than drop vision.
                try:
                    ctx = int(self.context_length or 0)
                    if ctx > 0:
                        body["max_tokens"] = max(
                            64, min(int(body.get("max_tokens") or 256), ctx // 8)
                        )
                except Exception:
                    pass

    def _look_batch_limit(self) -> int:
        override = coerce_positive_int(os.environ.get("RLM_LOOK_MAX_PER_TURN"))
        if override:
            return max(1, int(override))
        return media_parts_per_turn(self.context_length)

    def drain_python_inbox(self) -> List[Dict[str, Any]]:
        batch = self.inbox[self._inbox_py_cursor :]
        self._inbox_py_cursor = len(self.inbox)
        return list(batch)

    def enqueue_message(
        self,
        *,
        text: str,
        from_role: str,
        from_name: str = "",
        to_role: str = "parent",
        kind: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        item: Dict[str, Any] = {
            "text": str(text or ""),
            "from_role": from_role,
            "from_name": from_name,
            "to_role": to_role,
            "ts": time.time(),
        }
        if kind:
            item["kind"] = str(kind)
        if isinstance(details, dict) and details:
            item["details"] = dict(details)
        self.inbox.append(item)
        self._touch()
        self._emit(
            "inbox_enqueue",
            from_role=from_role,
            from_name=from_name,
            inbox_kind=str(kind or "") or None,
        )

    def compact_messages(self, *, keep_last: int = 12) -> Dict[str, Any]:
        """Mechanical transcript compact. Kernel namespace is left intact.

        Prefer :meth:`maybe_compact` (LLM summary) when a model is available.
        """
        self._strip_look_media_before_compact()
        keep = max(4, int(keep_last or 12))
        msgs = list(self.messages)
        if len(msgs) <= keep + 2:
            return {"ok": True, "compacted": False, "kept": len(msgs)}
        system = [m for m in msgs if isinstance(m, dict) and m.get("role") == "system"][:1]
        first_user = next(
            (m for m in msgs if isinstance(m, dict) and m.get("role") == "user"),
            None,
        )
        tail = msgs[-keep:]
        dropped = msgs[len(system) : max(len(system), len(msgs) - keep)]
        summary_bits: List[str] = []
        for m in dropped:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "")
            text = str(m.get("content") or "").strip().replace("\n", " ")
            if not text:
                if m.get("tool_calls"):
                    summary_bits.append(f"{role}: [tool_calls]")
                continue
            summary_bits.append(f"{role}: {text[:240]}")
        summary = {
            "role": "user",
            "content": (
                "[compacted prior turns — kernel state is unchanged]\n"
                + "\n".join(summary_bits[:40])
            )[:4000],
        }
        rebuilt: List[Dict[str, Any]] = list(system)
        if first_user and first_user not in tail and first_user not in rebuilt:
            rebuilt.append(first_user)
        rebuilt.append(summary)
        for m in tail:
            if m not in rebuilt:
                rebuilt.append(m)
        self.messages = rebuilt
        self._emit("compact", kept=len(self.messages), dropped=len(dropped), mode="mechanical")
        self.persist_local()
        return {"ok": True, "compacted": True, "kept": len(self.messages), "dropped": len(dropped)}

    async def maybe_compact(
        self,
        *,
        force: bool = False,
        custom_instructions: str = "",
        reason: str = "",
    ) -> Dict[str, Any]:
        """LLM compaction at ~90% of ``context_length`` (or ``force`` / 413).

        Preserves a structured checkpoint + recent verbatim tail + IPython
        kernel state. Tool/result pairs are never cut mid-pair.
        """
        from src.orchestrators.rlm.runtime.compaction import (
            compact_usage_ratio,
            generate_summary,
            mechanical_fallback_summary,
            prepare_compaction,
            settings_for_window,
            should_compact,
        )
        from src.orchestrators.rlm.runtime.context_budget import (
            estimate_messages_tokens,
        )
        from src.orchestrators.rlm.runtime.transcript import (
            append_compaction,
            load_entries,
        )

        ctx = int(self.context_length or 0)
        if not ctx:
            return {"ok": True, "compacted": False, "reason": "no_context_length"}

        settings = settings_for_window(ctx)
        try:
            self.sync_transcript()
        except Exception:
            pass

        tokens = estimate_messages_tokens(list(self.messages))
        if not force and not should_compact(tokens, ctx, settings):
            return {
                "ok": True,
                "compacted": False,
                "tokens": tokens,
                "window": ctx,
                "reserve": settings.reserve_tokens,
                "trigger": settings.trigger_tokens,
                "usage_ratio": compact_usage_ratio(tokens, ctx),
            }

        # About to rewrite history — strip already-shown look pixels (do not
        # re-queue; that would block the pending queue on the same filenames).
        self._strip_look_media_before_compact()

        entries = load_entries(self.artifact_dir)
        prep = prepare_compaction(entries, settings)
        if not prep:
            # Nothing durable to summarize — mechanical fallback on memory.
            return self.compact_messages(keep_last=12)

        to_sum = list(prep.get("messages_to_summarize") or [])
        previous = prep.get("previous_summary")
        summary = ""
        mode = "llm"
        from src.orchestrators.rlm.runtime.compact_resume import (
            resolve_compact_model,
        )

        compact_model = resolve_compact_model(self.cfg, self.model)

        async def _compact_chat(body: Dict[str, Any]) -> Dict[str, Any]:
            req = dict(body or {})
            req["model"] = compact_model or self.model
            # Summarizer should not spend budget on CoT.
            ctk = dict(req.get("chat_template_kwargs") or {})
            ctk["enable_thinking"] = False
            ctk.pop("preserve_thinking", None)
            req["chat_template_kwargs"] = ctk
            return await self._invoke_chat(req)

        try:
            if self.chat_fn and agent_model_is_explicit(compact_model or self.model):
                summary = await generate_summary(
                    chat_fn=_compact_chat,
                    model=compact_model or self.model,
                    messages_to_summarize=to_sum,
                    reserve_tokens=settings.reserve_tokens,
                    previous_summary=previous if isinstance(previous, str) else None,
                    custom_instructions=custom_instructions,
                    extract_assistant_fn=self.extract_assistant_fn,
                    context_length=self.context_length,
                    max_output_tokens=self.max_output_tokens,
                )
            if prep.get("is_split_turn") and prep.get("turn_prefix_messages"):
                prefix_sum = await generate_summary(
                    chat_fn=_compact_chat,
                    model=compact_model or self.model,
                    messages_to_summarize=list(prep["turn_prefix_messages"]),
                    reserve_tokens=max(1024, settings.reserve_tokens // 2),
                    extract_assistant_fn=self.extract_assistant_fn,
                    turn_prefix=True,
                    context_length=self.context_length,
                    max_output_tokens=self.max_output_tokens,
                )
                if prefix_sum:
                    summary = (summary + "\n\n## Turn prefix\n" + prefix_sum).strip()
        except Exception as exc:
            self._emit("compact_error", error=str(exc)[:400])
            summary = ""
            mode = "mechanical"

        if not summary:
            summary = mechanical_fallback_summary(to_sum)
            mode = "mechanical"

        append_compaction(
            self.artifact_dir,
            summary=summary,
            first_kept_entry_id=str(prep["first_kept_entry_id"]),
            tokens_before=int(prep.get("tokens_before") or tokens),
            details={"mode": mode, "compact_model": compact_model},
        )
        hydrated = self.hydrate_messages_from_disk(
            base_system=str(self.cfg.get("system_prompt") or "")
        )
        if not hydrated:
            # Still apply summary in-memory if disk hydrate failed.
            system = [m for m in self.messages if isinstance(m, dict) and m.get("role") == "system"][:1]
            keep_from = list(self.messages)[-12:]
            self.messages = system + [
                {"role": "user", "content": f"[context checkpoint]\n{summary}"}
            ] + [m for m in keep_from if m not in system]

        names: List[str] = []
        try:
            names = await self.list_kernel_names()
        except Exception:
            names = []
        self.inject_ipython_state_notice(restored=False, names=names)
        self._persist_compact_why_not(summary)
        self._inject_compact_continue_nudge(summary)
        self._arm_post_compact_guard()
        why = str(reason or ("force" if force else "threshold")).strip()
        self._emit(
            "compact",
            kept=len(self.messages),
            tokens_before=prep.get("tokens_before"),
            mode=mode,
            reason=why,
            trigger=settings.trigger_tokens,
            window=ctx,
            compact_model=compact_model,
            usage_ratio=compact_usage_ratio(
                int(prep.get("tokens_before") or tokens), ctx
            ),
        )
        self.persist_local()
        self._compact_refine_pending = True
        refine_out: Any = None
        if self.refine_enabled and self.refine_on_compact:
            try:
                from src.orchestrators.rlm.runtime.auto_refine import (
                    maybe_auto_refine,
                )

                refine_out = await maybe_auto_refine(self, reason="compact")
            except Exception as exc:
                refine_out = f"skipped:{exc}"
        out = {
            "ok": True,
            "compacted": True,
            "mode": mode,
            "kept": len(self.messages),
            "tokens_before": prep.get("tokens_before"),
            "first_kept_entry_id": prep.get("first_kept_entry_id"),
            "reason": why,
            "window": ctx,
            "compact_model": compact_model,
        }
        if refine_out is not None:
            out["refine"] = refine_out
        return out

    async def _maybe_auto_refine_checkpoint(self) -> None:
        if self.status == "error":
            return
        try:
            from src.orchestrators.rlm.runtime.auto_refine import maybe_auto_refine

            if self._compact_refine_pending:
                await maybe_auto_refine(self, reason="compact")
            await maybe_auto_refine(self, reason="turn_interval")
        except Exception:
            logger.warning("rlm auto-refine checkpoint failed", exc_info=True)

    def register_child(self, handle: Dict[str, Any], child: "RlmSession", task: asyncio.Task) -> None:
        cid = str(handle.get("rlm_child_id") or "")
        workspace = str(
            getattr(self, "workspace_dir", None) or self.artifact_dir or ""
        ).strip()
        baseline: Dict[str, Any] = {}
        try:
            from src.orchestrators.rlm.runtime.persist import (
                snapshot_workspace_files,
            )

            baseline = snapshot_workspace_files(workspace)
        except Exception:
            baseline = {}
        self.children[cid] = {
            **handle,
            "status": "running",
            "created_at": time.time(),
            "_workspace_files_at_start": baseline,
        }
        self._child_sessions[cid] = child
        self._child_tasks[cid] = task
        self._mark_fanin_pending(reason="child_admitted")
        # Prime: compare parent-reply count before vs after the child run.
        reply_baseline = int(getattr(child, "_parent_reply_count", 0) or 0)

        def _done(t: asyncio.Task, child_id: str = cid, baseline: int = reply_baseline) -> None:
            rec = self.children.get(child_id)
            if not rec:
                return
            child_sess = self._child_sessions.get(child_id)
            name = str(rec.get("name") or child_id)
            if t.cancelled():
                rec["status"] = "cancelled"
                rec["error"] = "cancelled"
                self._mark_fanin_pending(reason="child_cancelled")
                self._notify_child_terminal_notice(
                    child_id,
                    rec,
                    kind="cancelled",
                    reason="cancelled",
                )
            else:
                exc = t.exception() if not t.cancelled() else None
                if exc:
                    rec["status"] = "error"
                    rec["error"] = str(exc)
                    self._mark_fanin_pending(reason="child_failed")
                    self._notify_child_terminal_notice(
                        child_id,
                        rec,
                        kind="failed",
                        error=str(exc),
                    )
                else:
                    rec["status"] = "completed"
                    self._mark_fanin_pending(reason="child_completed")
            self._touch()
            self._emit("child_done", rlm_child_id=child_id, status=rec["status"])
            if str(rec.get("status") or "") == "completed":
                replies = (
                    int(getattr(child_sess, "_parent_reply_count", 0) or 0)
                    if child_sess
                    else baseline
                )
                if replies == baseline and not rec.get("_without_reply_notified"):
                    self._notify_child_completed_without_reply(child_id, rec, child_sess)

        task.add_done_callback(_done)

    def _child_terminal_snapshot(
        self, child_id: str, child: Optional["RlmSession"]
    ) -> Dict[str, Any]:
        """Collect rich failure/cancel context for the parent inbox."""
        last_content = ""
        last_reasoning = ""
        log_tail: List[str] = []
        status = ""
        if child is not None:
            last_content = str(getattr(child, "last_content", "") or "")
            last_reasoning = str(getattr(child, "last_reasoning", "") or "")
            log_tail = [str(x) for x in (getattr(child, "log_lines", []) or [])[-20:]]
            status = str(getattr(child, "status", "") or "")
            if not last_content:
                # Fall back to recent assistant thought if last_content empty.
                for th in reversed(list(getattr(child, "thoughts", []) or [])[-10:]):
                    if not isinstance(th, dict):
                        continue
                    if str(th.get("kind") or "") in ("assistant", "reasoning", "thinking"):
                        text = str(th.get("text") or "").strip()
                        if text:
                            if str(th.get("kind")) == "assistant" and not last_content:
                                last_content = text
                            elif not last_reasoning:
                                last_reasoning = text
            err = str(getattr(child, "error", "") or "")
        else:
            err = ""
        rec = self.children.get(child_id) if isinstance(self.children, dict) else None
        if isinstance(rec, dict):
            status = status or str(rec.get("status") or "")
            err = err or str(rec.get("error") or "")
        return {
            "last_content": last_content,
            "last_reasoning": last_reasoning,
            "log_tail": log_tail,
            "status": status,
            "error": err,
        }

    def _notify_child_terminal_notice(
        self,
        child_id: str,
        rec: Dict[str, Any],
        *,
        kind: str,
        error: str = "",
        reason: str = "",
    ) -> None:
        if rec.get("_terminal_notice_notified"):
            return
        rec["_terminal_notice_notified"] = True
        name = str(rec.get("name") or child_id)
        child_sess = self._child_sessions.get(child_id)
        snap = self._child_terminal_snapshot(child_id, child_sess)
        if kind == "failed":
            text = format_child_failure_notice(
                session_name=name,
                child_id=child_id,
                error=error or str(snap.get("error") or rec.get("error") or "unknown error"),
                status=str(snap.get("status") or rec.get("status") or "error"),
                last_content=str(snap.get("last_content") or ""),
                last_reasoning=str(snap.get("last_reasoning") or ""),
                log_tail=list(snap.get("log_tail") or []),
            )
            notice_kind = "failed"
            details: Dict[str, Any] = {
                "kind": notice_kind,
                "childId": child_id,
                "sessionName": name,
                "error": str(error or snap.get("error") or rec.get("error") or ""),
                "status": str(snap.get("status") or rec.get("status") or "error"),
            }
            if snap.get("last_content"):
                details["lastContentPreview"] = compact_rlm_text(
                    str(snap["last_content"]), max_length=400
                )
            if snap.get("last_reasoning"):
                details["lastReasoningPreview"] = compact_rlm_text(
                    str(snap["last_reasoning"]), max_length=400
                )
            if snap.get("log_tail"):
                details["logTail"] = list(snap["log_tail"])[-12:]
        else:
            text = format_child_cancelled_notice(
                session_name=name,
                child_id=child_id,
                reason=reason or str(rec.get("error") or ""),
                last_content=str(snap.get("last_content") or ""),
                last_reasoning=str(snap.get("last_reasoning") or ""),
                log_tail=list(snap.get("log_tail") or []),
            )
            notice_kind = "cancelled"
            details = {
                "kind": notice_kind,
                "childId": child_id,
                "sessionName": name,
            }
            if reason:
                details["reason"] = reason
            if snap.get("last_content"):
                details["lastContentPreview"] = compact_rlm_text(
                    str(snap["last_content"]), max_length=400
                )
        self._suppress_autonomous_continuation = True
        try:
            self.enqueue_message(
                text=text,
                from_role="rlm",
                from_name="runtime",
                to_role="parent",
                kind=notice_kind,
                details=details,
            )
        finally:
            self._suppress_autonomous_continuation = False
        self._emit(
            "child_terminal_notice",
            notice_kind=notice_kind,
            rlm_child_id=child_id,
            name=name,
        )

    def _child_produced_files(
        self, child_id: str, child: Optional["RlmSession"] = None
    ) -> List[Dict[str, Any]]:
        """Shared-workspace files created/updated while this child ran.

        Always diff the parent's shared workspace (the cwd children inherit), not
        the child's private ``artifact_dir`` under ``children/``.
        """
        _ = child
        rec = self.children.get(child_id) if isinstance(self.children, dict) else None
        baseline = None
        if isinstance(rec, dict):
            baseline = rec.get("_workspace_files_at_start")
        workspace = str(
            getattr(self, "workspace_dir", None) or self.artifact_dir or ""
        ).strip()
        try:
            from src.orchestrators.rlm.runtime.persist import (
                diff_workspace_produced_files,
            )

            return diff_workspace_produced_files(workspace, baseline)
        except Exception:
            return []

    def _notify_child_completed_without_reply(
        self,
        child_id: str,
        rec: Dict[str, Any],
        child: Optional["RlmSession"],
    ) -> None:
        """Prime ``completed_without_reply`` terminal notice → parent inbox."""
        if rec.get("_without_reply_notified"):
            return
        rec["_without_reply_notified"] = True
        name = str(rec.get("name") or child_id)
        preview = ""
        if child is not None:
            preview = str(getattr(child, "last_content", "") or "")
        produced = self._child_produced_files(child_id, child)
        text = format_completed_without_reply_notice(
            session_name=name,
            child_id=child_id,
            last_assistant_text=preview,
            produced_files=produced,
        )
        details: Dict[str, Any] = {
            "kind": "completed_without_reply",
            "childId": child_id,
            "sessionName": name,
        }
        if preview.strip():
            details["lastAssistantTextPreview"] = compact_rlm_text(preview)
        if produced:
            details["producedFiles"] = [
                {
                    "relpath": str(p.get("relpath") or ""),
                    "size_bytes": int(p.get("size_bytes") or 0),
                    "new": bool(p.get("new")),
                }
                for p in produced
                if str(p.get("relpath") or "").strip()
            ]
        self.enqueue_message(
            text=text,
            from_role="rlm",
            from_name="runtime",
            to_role="parent",
            kind="completed_without_reply",
            details=details,
        )
        self._emit(
            "child_terminal_notice",
            notice_kind="completed_without_reply",
            rlm_child_id=child_id,
            name=name,
            produced_files=[str(p.get("relpath") or "") for p in produced][:20],
        )

    def list_subagents(
        self, *, focus_session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return in-memory child registry rows without filesystem discovery.

        ``focus_session_id``:
        * ``None`` — full detail for every child (persist / kernel / tests).
        * ``str`` (may be empty) — full logs only for the matching child; others
          get a light badge row so UI payloads stay cheap with many subagents.
        """
        self._sync_child_registry_status()
        out: List[Dict[str, Any]] = []
        parent_nid = str(self.node_id or "").strip()
        focus_mode = focus_session_id is not None
        focus = str(focus_session_id or "").strip() if focus_mode else ""
        for cid, rec in self.children.items():
            if not isinstance(rec, dict):
                continue
            child = self._child_sessions.get(cid)
            status = str(rec.get("status") or "running")
            row: Dict[str, Any] = {
                "rlm_child_id": cid,
                "session_id": cid,
                "name": rec.get("name"),
                "session_dir": rec.get("session_dir"),
                "model": rec.get("model"),
                "status": status,
                "error": rec.get("error"),
                "parent_session_id": self.session_id,
                "node_id": (
                    str(getattr(child, "node_id", "") or "").strip()
                    or (
                        f"{parent_nid}::child::{cid}"
                        if parent_nid
                        else f"child::{cid}"
                    )
                ),
                "active_session_id": cid if status == "running" else None,
                "live": False,
            }
            want_full = (not focus_mode) or (focus == cid)
            if child is not None:
                cst = str(getattr(child, "status", "") or "")
                if cst == "running":
                    row["status"] = "running"
                elif cst in ("completed", "error", "cancelled", "deleted"):
                    # Prefer the finished child session over a stale registry row.
                    if status == "running" or not status:
                        row["status"] = cst
                    elif cst == "error":
                        row["status"] = "error"
                if getattr(child, "model", None):
                    row["model"] = child.model
                row["last_content"] = getattr(child, "last_content", None) or ""
                row["error"] = getattr(child, "error", None) or row.get("error")
                row["updated_at"] = getattr(child, "updated_at", None)
                row["created_at"] = getattr(child, "created_at", None) or rec.get(
                    "created_at"
                )
                row["live"] = bool(
                    not getattr(child, "_closed", False) and cst == "running"
                )
                if want_full:
                    row["log_lines"] = list(getattr(child, "log_lines", []) or [])[
                        -LOG_LINES_KEEP:
                    ]
                    # Memory may have been compacted; durable terminal.jsonl is source of truth.
                    if len(row["log_lines"]) < 2:
                        try:
                            from src.orchestrators.rlm.runtime.persist import (
                                load_terminal_log,
                            )

                            durable = load_terminal_log(
                                str(getattr(child, "artifact_dir", "") or ""),
                                limit=LOG_LINES_KEEP,
                            )
                            if durable:
                                row["log_lines"] = durable
                        except Exception:
                            pass
                    row["last_reasoning"] = (
                        getattr(child, "last_reasoning", None) or ""
                    )[:8000]
                    row["thoughts_tail"] = list(
                        getattr(child, "thoughts", []) or []
                    )[-30:]
                    row["files"] = slim_file_envelopes(
                        getattr(child, "_cached_files", []) or []
                    )
                    if not row["files"]:
                        known = getattr(child, "deliverable_artifacts", None)
                        if isinstance(known, dict):
                            row["files"] = [
                                {
                                    "filename": str(name),
                                    "artifact_id": str(artifact_id),
                                    "kind": "file",
                                }
                                for name, artifact_id in known.items()
                                if name and artifact_id
                            ]
                else:
                    row["last_content"] = str(row.get("last_content") or "")[:160]
                    row["log_lines"] = []
                    row["last_reasoning"] = ""
                    row["thoughts_tail"] = []
                    row["files"] = []
            else:
                # Restored registries may already carry a compact persisted row.
                if not rec.get("_meta_hydrated"):
                    rec["_meta_hydrated"] = True
                    art = str(rec.get("session_dir") or "").strip()
                    meta_path = os.path.join(art, "meta.json") if art else ""
                    if meta_path and os.path.isfile(meta_path):
                        try:
                            with open(meta_path, encoding="utf-8") as fh:
                                meta = json.load(fh)
                        except Exception:
                            meta = None
                        if isinstance(meta, dict):
                            try:
                                from src.orchestrators.rlm.runtime.persist import (
                                    hydrate_log_lines,
                                )

                                meta = hydrate_log_lines(
                                    {
                                        **meta,
                                        "session_dir": art,
                                        "artifact_dir": art,
                                    }
                                )
                            except Exception:
                                pass
                            for key in (
                                "status",
                                "model",
                                "error",
                                "log_lines",
                                "last_content",
                                "last_reasoning",
                                "thoughts_tail",
                                "updated_at",
                                "created_at",
                                "node_id",
                                "files",
                            ):
                                if meta.get(key) is not None:
                                    rec[key] = meta.get(key)
                    elif art:
                        try:
                            from src.orchestrators.rlm.runtime.persist import (
                                load_terminal_log,
                            )

                            durable = load_terminal_log(art, limit=LOG_LINES_KEEP)
                            if durable:
                                rec["log_lines"] = durable
                        except Exception:
                            pass
                if want_full:
                    row["files"] = slim_file_envelopes(rec.get("files") or [])
                    for key in (
                        "log_lines",
                        "last_content",
                        "last_reasoning",
                        "thoughts_tail",
                        "updated_at",
                        "created_at",
                    ):
                        if rec.get(key) is not None and row.get(key) in (None, "", []):
                            row[key] = rec.get(key)
                else:
                    row["files"] = []
                    row["log_lines"] = []
                    row["last_reasoning"] = ""
                    row["thoughts_tail"] = []
                    if rec.get("last_content") is not None:
                        row["last_content"] = str(rec.get("last_content") or "")[:160]
                    for key in ("updated_at", "created_at"):
                        if rec.get(key) is not None and row.get(key) in (None, "", []):
                            row[key] = rec.get(key)
            # Finished / cancelled parents must not keep spinning child badges
            # from a stale registry ``running`` row (common after soft-wait pause).
            parent_st = str(self.status or "")
            if (
                parent_st in ("completed", "error", "cancelled")
                and str(row.get("status") or "") == "running"
                and not row.get("live")
            ):
                row["status"] = (
                    "error" if parent_st == "error" else "completed"
                )
                row["live"] = False
            out.append(row)
        return out

    def _seal_children_for_terminal(self) -> None:
        """Flip stale ``running`` child rows when this session is finished."""
        parent_st = str(self.status or "")
        if parent_st not in ("completed", "error", "cancelled"):
            return
        self._sync_child_registry_status()
        for cid, rec in list(self.children.items()):
            if not isinstance(rec, dict):
                continue
            if str(rec.get("status") or "") != "running":
                continue
            task = self._child_tasks.get(cid)
            child = self._child_sessions.get(cid)
            still_live = (task is not None and not task.done()) or (
                child is not None
                and not getattr(child, "_closed", False)
                and str(getattr(child, "status", "") or "") == "running"
            )
            if still_live:
                continue
            rec["status"] = "error" if parent_st == "error" else "completed"

    def sibling_map(self) -> Dict[str, "RlmSession"]:
        parent = getattr(self, "_parent_ref", None)
        root = parent if parent is not None else self
        out: Dict[str, RlmSession] = {}
        for cid, child in root._child_sessions.items():
            rec = root.children.get(cid) or {}
            name = str(rec.get("name") or cid)
            out[name] = child
            out[cid] = child
        return out

    async def delete_subagent(self, target: str) -> Dict[str, Any]:
        sel = str(target or "").strip()
        match_id = None
        for cid, rec in self.children.items():
            if cid == sel or str(rec.get("name") or "") == sel:
                match_id = cid
                break
        if not match_id:
            return {"outcome": "not_found", "target": sel}
        task = self._child_tasks.get(match_id)
        child = self._child_sessions.get(match_id)
        if task and not task.done():
            task.cancel()
        if child:
            await child.shutdown()
        rec = self.children.get(match_id) or {}
        rec["status"] = "deleted"
        self._emit("child_deleted", rlm_child_id=match_id)
        return {"outcome": "deleted", "subagent": rec}

    def save_kernel_snapshot(self) -> Optional[str]:
        if self.kernel and self.kernel.alive and self.snapshot_path:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.kernel.dump_snapshot(self.snapshot_path))
            except Exception:
                pass
            return self.snapshot_path
        marker = os.path.join(self.artifact_dir, "kernel-snapshot.marker")
        try:
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write(self.session_id)
            return marker
        except Exception:
            return None

    async def stop_kernel(self, *, dump_snapshot: bool = True) -> None:
        """Release a kernel slot while retaining resumable session metadata."""
        kernel = self.kernel
        self.kernel = None
        try:
            if kernel:
                await kernel.shutdown(dump_snapshot=dump_snapshot)
        finally:
            if self._kernel_slot_acquired:
                from src.orchestrators.rlm.runtime.resource_budget import (
                    kernel_governor,
                )

                self._kernel_slot_acquired = False
                await kernel_governor().release(self.user_id)

    def compact_completed_child(self) -> None:
        """Drop large in-memory history after the durable transcript is written."""
        self.sync_transcript()
        self.messages = self.messages[-8:]
        self.thoughts = self.thoughts[-12:]
        self.events = self.events[-80:]
        self.log_lines = self.log_lines[-120:]
        self._stream_content_buf = self._stream_content_buf[-8192:]
        self._stream_tool_calls = self._stream_tool_calls[-16:]

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._persist_timer is not None:
            self._persist_timer.cancel()
            self._persist_timer = None
        persist_task, self._persist_task = self._persist_task, None
        if persist_task is not None and not persist_task.done():
            persist_task.cancel()
            try:
                await persist_task
            except (asyncio.CancelledError, Exception):
                pass
        for cid in list(self._child_tasks.keys()):
            try:
                await self.delete_subagent(cid)
            except Exception:
                pass
        await self.stop_kernel(dump_snapshot=True)
        self.save_kernel_snapshot()
        self.status = "cancelled" if self.status == "running" else self.status
        self._emit("shutdown")
        self._touch()
        try:
            from src.orchestrators.rlm.runtime.daemon import RlmDaemon

            daemon = RlmDaemon._instance
            if daemon is not None:
                for cid in list(self._child_sessions.keys()):
                    daemon._pop(cid)
                daemon._pop(self.session_id)
        except Exception:
            pass

    def public_status(self) -> Dict[str, Any]:
        """Build a cheap, side-effect-free control-plane status snapshot."""
        files = slim_file_envelopes(self._cached_files)
        if not files:
            files = [
                {
                    "filename": str(name),
                    "artifact_id": str(artifact_id),
                    "kind": "file",
                }
                for name, artifact_id in self.deliverable_artifacts.items()
                if name and artifact_id
            ]
        return {
            "session_id": self.session_id,
            "parent_session_id": self.parent_session_id,
            "run_id": self.run_id,
            "flow_id": self.flow_id,
            "node_id": self.node_id,
            "status": self.status,
            "model": self.model,
            "depth": self.depth,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_content": self.last_content or "",
            "last_reasoning": (self.last_reasoning or "")[:8000],
            "enable_thinking": self.thinking_enabled(),
            "thoughts_tail": list(self.thoughts[-30:]),
            "log_lines": list(self.log_lines[-LOG_LINES_KEEP:]),
            "error": self.error,
            "user_id": self.user_id,
            "children": self.list_subagents(focus_session_id=None),
            "events": self.events[-50:],
            "messages_tail": slim_messages(self.messages, limit=40, content_chars=800),
            "live": bool(
                not self._closed
                and self.status == "running"
                and not getattr(self, "_shutdown_requested", False)
            ),
            "artifact_dir": self.artifact_dir,
            "workspace_dir": self.workspace_dir,
            "daemon_enabled": self.daemon_enabled,
            "harness_path": self.harness_path() if self.refine_enabled else None,
            "model_logical_id": self.model_logical_id,
            "refine_enabled": self.refine_enabled,
            "refine_auto_enabled": self.refine_auto_enabled,
            "run_lifetime": self.run_lifetime,
            "heartbeat": self.jobs.heartbeat_list(),
            "goal": self.jobs.goal_get(),
            "autonomous": (
                self.jobs.autonomous.status_dict()
                if self.jobs.autonomous is not None
                else None
            ),
            "context_length": self.context_length,
            "max_output_tokens": self.max_output_tokens,
            "files": files,
        }
