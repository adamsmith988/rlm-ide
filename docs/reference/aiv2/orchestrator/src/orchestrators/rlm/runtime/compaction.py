"""Prime-like semantic compaction for RLM transcripts.

Industry practice (Claude Code ~80–83%, Continue/Cline auto-compact before
exhaustion, Anthropic server compact): trigger *before* the hard window fills
so the summarizer and the next turn still have headroom. Defaults to **90% of
``context_length``** (override with ``AIV2_RLM_COMPACT_TRIGGER_RATIO``).

After compaction we keep a verbatim recent tail plus a structured checkpoint
so the model can continue. Tool/result pairs are never cut mid-pair.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.orchestrators.rlm.runtime.context_budget import (
    estimate_messages_tokens,
)
from src.orchestrators.rlm.runtime.prompts import (
    KERNEL_PERSIST_SUMMARY_NOTE,
    SUMMARIZATION_SYSTEM_PROMPT,
    TURN_PREFIX_SUMMARIZATION_PROMPT,
    build_summarization_prompt,
)
from src.orchestrators.rlm.runtime.transcript import (
    _entry_to_message,
    build_session_context,
)

# 90% trigger → 10% reserve for completion + summarizer output.
DEFAULT_TRIGGER_RATIO = 0.90
DEFAULT_RESERVE_RATIO = 1.0 - DEFAULT_TRIGGER_RATIO  # 0.10
# Verbatim tail after compact — leave room to grow again before next compact.
DEFAULT_KEEP_RECENT_RATIO = 0.35


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


@dataclass
class CompactionSettings:
    enabled: bool = True
    reserve_tokens: int = 8192
    keep_recent_tokens: int = 10240
    trigger_tokens: int = 0  # compact when usage >= this (0 → derive from reserve)


@dataclass
class CutPointResult:
    first_kept_entry_index: int
    turn_start_index: int
    is_split_turn: bool


def settings_for_window(
    context_window: int,
    *,
    enabled: bool = True,
    trigger_ratio: Optional[float] = None,
    keep_recent_ratio: Optional[float] = None,
    reserve_ratio: Optional[float] = None,
) -> CompactionSettings:
    """Derive thresholds from the deployed context window.

    ``trigger_ratio`` (default 0.90): compact when estimated input tokens reach
    this fraction of the window. ``keep_recent_ratio`` (default 0.35): how much
    of the window to keep as a verbatim recent tail after the checkpoint.
    """
    window = max(1024, int(context_window or 0))
    tr = (
        float(trigger_ratio)
        if trigger_ratio is not None
        else _env_float("AIV2_RLM_COMPACT_TRIGGER_RATIO", DEFAULT_TRIGGER_RATIO)
    )
    tr = min(0.97, max(0.55, tr))
    kr = (
        float(keep_recent_ratio)
        if keep_recent_ratio is not None
        else _env_float(
            "AIV2_RLM_COMPACT_KEEP_RECENT_RATIO", DEFAULT_KEEP_RECENT_RATIO
        )
    )
    kr = min(0.70, max(0.15, kr))
    if reserve_ratio is not None:
        rr = min(0.45, max(0.03, float(reserve_ratio)))
        tr = 1.0 - rr
    reserve = max(1024, int(window * (1.0 - tr)))
    keep_recent = max(2048, int(window * kr))
    trigger = max(1024, int(window * tr))
    trigger = min(trigger, max(1024, window - reserve))
    return CompactionSettings(
        enabled=bool(enabled),
        reserve_tokens=reserve,
        keep_recent_tokens=keep_recent,
        trigger_tokens=trigger,
    )


def should_compact(
    context_tokens: int,
    context_window: int,
    settings: CompactionSettings,
) -> bool:
    if not settings.enabled:
        return False
    window = int(context_window or 0)
    if window <= 0:
        return False
    trigger = int(settings.trigger_tokens or 0)
    if trigger <= 0:
        trigger = max(1, window - int(settings.reserve_tokens))
    return int(context_tokens) >= trigger


def compact_usage_ratio(context_tokens: int, context_window: int) -> float:
    window = int(context_window or 0)
    if window <= 0:
        return 0.0
    return max(0.0, min(1.0, float(context_tokens) / float(window)))


def estimate_message_tokens(message: Any) -> int:
    if isinstance(message, dict):
        return estimate_messages_tokens([message])
    return 0


def _message_role(entry: Dict[str, Any]) -> str:
    if entry.get("type") == "message":
        msg = entry.get("message")
        if isinstance(msg, dict):
            return str(msg.get("role") or "")
    if entry.get("type") in ("custom_message", "branch_summary"):
        return "user"
    return ""


def find_valid_cut_points(
    entries: Sequence[Dict[str, Any]],
    start_index: int,
    end_index: int,
) -> List[int]:
    """Indices safe to cut at. Never cut on ``role:tool`` (must follow tool_calls)."""
    cut_points: List[int] = []
    for i in range(start_index, end_index):
        entry = entries[i]
        et = str(entry.get("type") or "")
        if et == "message":
            role = _message_role(entry)
            if role in ("user", "assistant", "system"):
                cut_points.append(i)
            # role == "tool" → skip
        elif et in ("custom_message", "branch_summary"):
            cut_points.append(i)
    return cut_points


def find_turn_start_index(
    entries: Sequence[Dict[str, Any]],
    entry_index: int,
    start_index: int,
) -> int:
    for i in range(entry_index, start_index - 1, -1):
        et = str(entries[i].get("type") or "")
        if et in ("custom_message", "branch_summary"):
            return i
        if et == "message" and _message_role(entries[i]) == "user":
            return i
    return -1


def find_cut_point(
    entries: Sequence[Dict[str, Any]],
    start_index: int,
    end_index: int,
    keep_recent_tokens: int,
) -> CutPointResult:
    cut_points = find_valid_cut_points(entries, start_index, end_index)
    if not cut_points:
        return CutPointResult(start_index, -1, False)

    accumulated = 0
    cut_index = cut_points[0]
    for i in range(end_index - 1, start_index - 1, -1):
        entry = entries[i]
        if str(entry.get("type") or "") != "message":
            continue
        msg = entry.get("message")
        accumulated += estimate_message_tokens(msg)
        if accumulated >= int(keep_recent_tokens):
            for cp in cut_points:
                if cp >= i:
                    cut_index = cp
                    break
            break

    while cut_index > start_index:
        prev = entries[cut_index - 1]
        et = str(prev.get("type") or "")
        if et == "compaction":
            break
        if et == "message":
            break
        cut_index -= 1

    cut_entry = entries[cut_index]
    is_user = str(cut_entry.get("type") or "") == "message" and _message_role(cut_entry) == "user"
    turn_start = -1 if is_user else find_turn_start_index(entries, cut_index, start_index)
    return CutPointResult(
        first_kept_entry_index=cut_index,
        turn_start_index=turn_start,
        is_split_turn=(not is_user and turn_start != -1),
    )


def _head_tail_truncate(text: str, *, max_chars: int = 8000) -> str:
    """Prefer head+tail so trailing errors survive summarization."""
    s = str(text or "")
    limit = max(1024, int(max_chars or 8000))
    if len(s) <= limit:
        return s
    head = max(512, int(limit * 0.55))
    tail = max(256, limit - head - 48)
    omitted = len(s) - head - tail
    return f"{s[:head]}\n…[{omitted} chars omitted]…\n{s[-tail:]}"


def _assistant_reasoning_snippet(msg: Mapping[str, Any], *, max_chars: int = 1500) -> str:
    reasoning = (
        msg.get("reasoning_content")
        or msg.get("reasoning")
        or msg.get("reasoning_text")
        or msg.get("thinking")
        or ""
    )
    if isinstance(reasoning, dict):
        reasoning = reasoning.get("content") or reasoning.get("text") or ""
    text = str(reasoning or "").strip()
    if not text:
        return ""
    if len(text) > max_chars:
        text = text[: max_chars - 16] + "\n…[truncated]"
    return text


def serialize_conversation(messages: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    assistant_idxs = [
        i
        for i, m in enumerate(messages)
        if isinstance(m, dict) and str(m.get("role") or "") == "assistant"
    ]
    # Include CoT only for the last few assistant turns to bound serialize size.
    reasoning_keep = set(assistant_idxs[-6:])
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "unknown")
        content = msg.get("content")
        if isinstance(content, list):
            bits = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    bits.append(str(block.get("text") or ""))
                elif isinstance(block, dict):
                    # Never dump multi-MB data-URLs into the summarizer prompt.
                    kind = str(block.get("type") or "")
                    if kind in ("image_url", "video_url", "audio_url", "input_image"):
                        bits.append(f"[{kind} omitted]")
                    else:
                        bits.append(json.dumps(block, ensure_ascii=False)[:400])
            text = "\n".join(bits)
        else:
            text = str(content or "")
        if msg.get("tool_calls"):
            try:
                text = (text + "\n" + json.dumps(msg["tool_calls"], ensure_ascii=False)).strip()
            except Exception:
                text = (text + "\n[tool_calls]").strip()
        if role == "assistant" and i in reasoning_keep:
            cot = _assistant_reasoning_snippet(msg)
            if cot:
                text = (text + "\n[reasoning]\n" + cot).strip()
        if role == "tool" and msg.get("tool_call_id"):
            text = f"[tool_call_id={msg.get('tool_call_id')}]\n{text}"
        # Tool results: head+tail. Other roles: same helper with 8k budget.
        parts.append(f"[{role}]\n{_head_tail_truncate(text, max_chars=8000)}")
    return "\n\n".join(parts)


def prepare_compaction(
    path_entries: Sequence[Dict[str, Any]],
    settings: CompactionSettings,
) -> Optional[Dict[str, Any]]:
    rows = [e for e in path_entries if isinstance(e, dict) and e.get("type") != "session"]
    if not rows:
        return None
    if str(rows[-1].get("type") or "") == "compaction":
        return None

    prev_compaction_index = -1
    for i in range(len(rows) - 1, -1, -1):
        if str(rows[i].get("type") or "") == "compaction":
            prev_compaction_index = i
            break

    previous_summary: Optional[str] = None
    boundary_start = 0
    if prev_compaction_index >= 0:
        prev = rows[prev_compaction_index]
        previous_summary = str(prev.get("summary") or "") or None
        first_kept = str(prev.get("first_kept_entry_id") or "").strip()
        found = -1
        if first_kept:
            for i, e in enumerate(rows):
                if str(e.get("id") or "") == first_kept:
                    found = i
                    break
        boundary_start = found if found >= 0 else prev_compaction_index + 1

    boundary_end = len(rows)
    ctx = build_session_context(rows)
    tokens_before = estimate_messages_tokens(list(ctx.get("messages") or []))

    cut = find_cut_point(
        rows, boundary_start, boundary_end, settings.keep_recent_tokens
    )
    first_kept_entry = rows[cut.first_kept_entry_index]
    first_kept_id = str(first_kept_entry.get("id") or "").strip()
    if not first_kept_id:
        return None

    history_end = (
        cut.turn_start_index if cut.is_split_turn else cut.first_kept_entry_index
    )

    def msgs_from(start: int, end: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for i in range(start, end):
            et = str(rows[i].get("type") or "")
            if et == "compaction":
                continue
            m = _entry_to_message(rows[i])
            if m:
                m.pop("_compaction", None)
                out.append(m)
        return out

    messages_to_summarize = msgs_from(boundary_start, history_end)
    turn_prefix: List[Dict[str, Any]] = []
    if cut.is_split_turn:
        turn_prefix = msgs_from(cut.turn_start_index, cut.first_kept_entry_index)

    if not messages_to_summarize and not turn_prefix and not previous_summary:
        return None

    return {
        "first_kept_entry_id": first_kept_id,
        "messages_to_summarize": messages_to_summarize,
        "turn_prefix_messages": turn_prefix,
        "is_split_turn": cut.is_split_turn,
        "tokens_before": tokens_before,
        "previous_summary": previous_summary,
        "settings": settings,
        "rows": rows,
    }


def build_summary_user_prompt(
    *,
    messages_to_summarize: Sequence[Dict[str, Any]],
    previous_summary: Optional[str] = None,
    custom_instructions: str = "",
    turn_prefix: bool = False,
) -> str:
    conversation = serialize_conversation(messages_to_summarize)
    if turn_prefix:
        base = TURN_PREFIX_SUMMARIZATION_PROMPT
        prompt = f"<conversation>\n{conversation}\n</conversation>\n\n{base}"
        return f"{prompt}\n\n{KERNEL_PERSIST_SUMMARY_NOTE}"
    body = f"<conversation>\n{conversation}\n</conversation>\n\n"
    if previous_summary:
        body += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    body += build_summarization_prompt(
        custom_instructions=custom_instructions,
        previous_summary=previous_summary,
    )
    return body


async def generate_summary(
    *,
    chat_fn: Any,
    model: str,
    messages_to_summarize: Sequence[Dict[str, Any]],
    reserve_tokens: int,
    previous_summary: Optional[str] = None,
    custom_instructions: str = "",
    extract_assistant_fn: Any = None,
    turn_prefix: bool = False,
    context_length: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
) -> str:
    prompt = build_summary_user_prompt(
        messages_to_summarize=messages_to_summarize,
        previous_summary=previous_summary,
        custom_instructions=custom_instructions,
        turn_prefix=turn_prefix,
    )
    max_tokens = max(256, int(0.8 * max(1, int(reserve_tokens))))
    body: Dict[str, Any] = {
        "model": model,
        "stream": False,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    from src.orchestrators.rlm.runtime.context_budget import (
        apply_context_budget,
        coerce_positive_int,
    )

    ctx = coerce_positive_int(context_length) or coerce_positive_int(
        max(1024, int(reserve_tokens) * 2)
    )
    if ctx:
        apply_context_budget(
            body,
            ctx,
            max_output_tokens=coerce_positive_int(max_output_tokens) or max_tokens,
        )
    data = await chat_fn(body)
    if extract_assistant_fn:
        msg = extract_assistant_fn(data)
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            return "\n".join(
                str(b.get("text") or "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
        return str(content or "").strip()
    try:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        return str(msg.get("content") or "").strip()
    except Exception:
        return ""


def mechanical_fallback_summary(messages: Sequence[Dict[str, Any]], *, limit: int = 40) -> str:
    bits: List[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        text = str(m.get("content") or "").strip().replace("\n", " ")
        if not text and m.get("tool_calls"):
            bits.append(f"{role}: [tool_calls]")
            continue
        if text:
            bits.append(f"{role}: {text[:240]}")
    return (
        "## Goal\n(unknown — mechanical fallback)\n\n## Progress\n### Done\n"
        + "\n".join(f"- {b}" for b in bits[:limit])
        + "\n\n## Next Steps\n1. Continue from the kept recent turns.\n"
    )
