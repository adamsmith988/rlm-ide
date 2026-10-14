"""Hard-cap chat bodies to the model's deployed context window.

GPU Nodes & Deploy writes vLLM ``--max-model-len`` as ``max_model_len``.
The llm-proxy catalog publishes that as ``context_length``. RLM must never
send ``estimated_input + max_tokens >= context_length`` (vLLM 400 / proxy 503).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Conservative: multilingual (Persian) is denser than English BPE.
# Prefer over-estimating so we never send a prompt vLLM will 400.
CHARS_PER_TOKEN = 2
# Chat template + special tokens + tokenizer mismatch slack.
SAFETY_MARGIN_TOKENS = 160
MIN_COMPLETION_TOKENS = 256
DEFAULT_COMPLETION_TOKENS = 1024
MSG_OVERHEAD_TOKENS = 8
# Floor for one native image/audio part (VLM visual tokens). Real base64 data-URL
# length is used when larger so multi-MB looks are not under-counted on the wire.
MEDIA_PART_TOKENS = 1600
MEDIA_PART_CHARS = MEDIA_PART_TOKENS * CHARS_PER_TOKEN
_MEDIA_PART_TYPES = frozenset(
    {
        "image_url",
        "image",
        "input_image",
        "audio_url",
        "video_url",
        "input_audio",
    }
)


def media_parts_per_turn(context_length: Optional[int]) -> int:
    """How many native media parts fit comfortably in one turn for this window."""
    ctx = int(context_length or 0)
    if ctx <= 0:
        return 4
    # Never let one look batch claim more than ~25% of the window.
    return max(1, min(12, (ctx // 4) // MEDIA_PART_TOKENS))


def _media_url_chars(part: Dict[str, Any]) -> int:
    """ASCII length of a data-URL (or similar) inside a multimodal content part."""
    for key in ("image_url", "video_url", "audio_url"):
        blob = part.get(key)
        if isinstance(blob, dict):
            url = str(blob.get("url") or "")
            if url:
                return len(url)
        elif isinstance(blob, str) and blob:
            return len(blob)
    data = part.get("data")
    if isinstance(data, str) and data:
        return len(data)
    return 0


def _is_media_part(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    kind = str(part.get("type") or "").lower()
    if kind in _MEDIA_PART_TYPES:
        return True
    return any(
        isinstance(part.get(k), (dict, str)) and part.get(k)
        for k in ("image_url", "video_url", "audio_url")
    )


def _media_part_chars(part: Dict[str, Any]) -> int:
    """Char cost of one media part for *token* budgeting.

    Native image/video tokens are roughly ``MEDIA_PART_TOKENS``, not one token
    per base64 character. Treating the full data-URL length as text made a
    single ``rlm.look`` estimate as hundreds of thousands of tokens, so the
    last-resort trim dropped active look pixels and left only a ``[rlm.look]``
    text header — the model stayed blind and re-looked forever.

    Gateway body size is enforced separately via look turn encoded-byte caps
    and ``_strip_oldest_media_by_bytes``.
    """
    return MEDIA_PART_CHARS


def _total_media_url_chars(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if _is_media_part(part):
                total += _media_url_chars(part)
    return total


# Soft wire cap for cumulative look data-URLs still in history (beyond the
# active look). Older looks are text-stubbed first so active pixels stay.
MAX_HISTORY_MEDIA_CHARS = int(
    os.environ.get("AIV2_RLM_HISTORY_MEDIA_CHARS", str(4 * 1024 * 1024))
)

def _truncate_content_preserving_media(content: Any, max_chars: int) -> Any:
    """Shrink text parts only — never drop image/audio/video parts."""
    if not isinstance(content, list):
        return _truncate_content(content, max_chars)
    media = [p for p in content if _is_media_part(p)]
    texts = [p for p in content if isinstance(p, dict) and not _is_media_part(p)]
    media_cost = sum(_content_chars(p) for p in media)
    text_budget = max(0, int(max_chars) - media_cost)
    trimmed_texts = _truncate_content(texts, text_budget) if texts else []
    if not isinstance(trimmed_texts, list):
        trimmed_texts = (
            [{"type": "text", "text": str(trimmed_texts)}] if trimmed_texts else []
        )
    # Preserve original order: text headers first, then media (look shape).
    return list(trimmed_texts) + list(media)
_CTX_PATTERNS = (
    re.compile(
        r"maximum context length is\s+(\d+).*?requested\s+(\d+)\s+tokens",
        re.I | re.S,
    ),
    # Sepidour / OpenAI-compat: "'max_tokens' is too large: N. This model's
    # maximum context length is 30000 tokens and your request has 30999 input tokens"
    re.compile(
        r"maximum context length is\s+(\d+)\s+tokens\s+and\s+your\s+request\s+has\s+(\d+)\s+input\s+tokens",
        re.I | re.S,
    ),
    re.compile(
        r"passed\s+(\d+)\s+input tokens.*?context length is only\s+(\d+)",
        re.I | re.S,
    ),
    re.compile(
        r"decoder prompt \(length\s+(\d+)\).*?maximum model length of\s+(\d+)",
        re.I | re.S,
    ),
    re.compile(
        r"max(?:imum)? context(?: length)?(?: is)?\s+(\d+).*?"
        r"(?:prompt|input).*?(\d+)\s*(?:input\s+)?tokens",
        re.I | re.S,
    ),
)


def coerce_positive_int(*values: Any) -> Optional[int]:
    for value in values:
        if value in (None, "", False):
            continue
        try:
            n = int(value)
        except (TypeError, ValueError):
            continue
        if n > 0:
            return n
    return None


def coerce_context_length(*sources: Any) -> Optional[int]:
    """Read context window from catalog meta, cfg, or a raw int."""
    keys = ("context_length", "max_model_len", "max_seq_len", "n_ctx")
    for src in sources:
        if src is None:
            continue
        if isinstance(src, bool):
            continue
        if isinstance(src, (int, float)):
            n = coerce_positive_int(src)
            if n:
                return n
            continue
        if isinstance(src, str):
            n = coerce_positive_int(src.strip())
            if n:
                return n
            continue
        if not isinstance(src, dict):
            continue
        n = coerce_positive_int(*(src.get(k) for k in keys))
        if n:
            return n
        rs = src.get("run_schema")
        if isinstance(rs, dict):
            n = coerce_positive_int(*(rs.get(k) for k in keys))
            if n:
                return n
        caps = src.get("capabilities")
        if isinstance(caps, dict):
            n = coerce_positive_int(*(caps.get(k) for k in keys))
            if n:
                return n
    return None


def _content_chars(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(_content_chars(p) for p in content)
    if isinstance(content, dict):
        if _is_media_part(content):
            return _media_part_chars(content)
        if "text" in content:
            return len(str(content.get("text") or ""))
        try:
            return len(json.dumps(content, ensure_ascii=False))
        except Exception:
            return len(str(content))
    return len(str(content))


def _truncate_content(content: Any, max_chars: int) -> Any:
    max_chars = max(0, int(max_chars))
    if isinstance(content, str):
        if len(content) <= max_chars:
            return content
        keep = max(0, max_chars - 18)
        return content[:keep] + "\n…[truncated]"
    if isinstance(content, list):
        if max_chars <= 0:
            return []
        out: List[Any] = []
        used = 0
        for part in content:
            remain = max_chars - used
            if remain <= 0:
                break
            if isinstance(part, dict) and "text" in part:
                text = str(part.get("text") or "")
                if len(text) <= remain:
                    out.append(part)
                    used += len(text)
                else:
                    clipped = dict(part)
                    clipped["text"] = _truncate_content(text, remain)
                    out.append(clipped)
                    used = max_chars
                    break
            else:
                ch = _content_chars(part)
                if ch <= remain:
                    out.append(part)
                    used += ch
        return out
    if isinstance(content, dict) and "text" in content:
        clipped = dict(content)
        clipped["text"] = _truncate_content(str(content.get("text") or ""), max_chars)
        return clipped
    try:
        blob = json.dumps(content, ensure_ascii=False)
    except Exception:
        blob = str(content)
    return _truncate_content(blob, max_chars)


def estimate_text_tokens(text: Any, *, chars_per_token: int = CHARS_PER_TOKEN) -> int:
    cpt = max(1, int(chars_per_token))
    return _content_chars(text) // cpt


def estimate_messages_tokens(
    messages: List[Dict[str, Any]],
    *,
    extra_chars: int = 0,
    chars_per_token: int = CHARS_PER_TOKEN,
) -> int:
    cpt = max(1, int(chars_per_token))
    total = max(0, int(extra_chars)) // cpt
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        total += MSG_OVERHEAD_TOKENS
        total += estimate_text_tokens(msg.get("content"), chars_per_token=cpt)
        # Prior CoT is part of the rendered prompt when preserve_thinking is on.
        total += estimate_text_tokens(
            msg.get("reasoning_content") or msg.get("reasoning") or msg.get("reasoning_text"),
            chars_per_token=cpt,
        )
        if msg.get("tool_calls"):
            try:
                total += len(json.dumps(msg.get("tool_calls"), ensure_ascii=False)) // cpt
            except Exception:
                total += 80
        name = msg.get("name")
        if name:
            total += len(str(name)) // cpt
    return total + 32


def budget_completion_tokens(
    *,
    est_input: int,
    context_length: int,
    requested: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    min_completion: int = MIN_COMPLETION_TOKENS,
) -> int:
    ctx = max(1, int(context_length))
    room = ctx - int(est_input) - SAFETY_MARGIN_TOKENS
    if room < 1:
        return 1
    cap = room
    if max_output_tokens:
        cap = min(cap, int(max_output_tokens))
    want = int(requested) if requested and int(requested) > 0 else min(
        DEFAULT_COMPLETION_TOKENS, cap
    )
    out = min(want, cap)
    if room >= min_completion:
        out = max(min_completion, out) if want >= min_completion else out
        out = min(out, cap)
    return max(1, out)


def _drop_oldest_nonessential(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep first system, first user, and the tail; drop one oldest middle msg."""
    if len(messages) <= 3:
        return messages
    system_idx = next(
        (i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "system"),
        None,
    )
    protected = {0, len(messages) - 1, len(messages) - 2}
    if system_idx is not None:
        protected.add(system_idx)
    first_user = next(
        (i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"),
        None,
    )
    if first_user is not None:
        protected.add(first_user)
    for i in range(len(messages)):
        if i not in protected:
            return messages[:i] + messages[i + 1 :]
    return messages[:-1]


def _protected_trim_indexes(messages: List[Dict[str, Any]]) -> set:
    protected = {0, len(messages) - 1, len(messages) - 2} if messages else set()
    system_idx = next(
        (i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "system"),
        None,
    )
    if system_idx is not None:
        protected.add(system_idx)
    first_user = next(
        (i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"),
        None,
    )
    if first_user is not None:
        protected.add(first_user)
    return protected


def _latest_media_message_index(messages: List[Dict[str, Any]]) -> Optional[int]:
    """Index of the newest multimodal message (active look), even if not last turn."""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list) and any(_is_media_part(p) for p in content):
            return i
    return None


def _media_strip_note(header: str = "") -> str:
    """Wire-only note after dropping old look pixels — must not invite re-look loops."""
    note = (
        "[rlm.look] older image pixels were dropped from this prompt to fit "
        "the window (files remain on disk). Do NOT re-look the same paths — "
        "write notes from what you already observed, or look only at new files."
    )
    head = str(header or "").strip()
    if head:
        return head[:500] + "\n" + note
    return note


def _strip_oldest_media_message(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], bool]:
    """Replace media in the oldest non-active multimodal message with a text note.

    Keeps conversation text; drops heavy base64 so wire trim can succeed without
    sending multi-MB payloads that remote VLMs report as truncated images.

    The newest media-bearing message is preserved even when later assistant/tool
    turns follow it — otherwise the model loses the images mid-look and re-looks
    in a loop.
    """
    if not messages:
        return messages, False
    protected = _protected_trim_indexes(messages)
    latest_media = _latest_media_message_index(messages)
    if latest_media is not None:
        protected.add(latest_media)
    # Prefer stripping older history; never strip the newest message or active look.
    for i in range(len(messages)):
        if i in protected or i == len(messages) - 1:
            continue
        msg = messages[i]
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        if not any(_is_media_part(p) for p in content):
            continue
        texts: List[str] = []
        media_n = 0
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" or (
                "text" in part and not _is_media_part(part)
            ):
                t = str(part.get("text") or "").strip()
                if t:
                    texts.append(t)
            elif _is_media_part(part):
                media_n += 1
        if media_n <= 0:
            continue
        new_msg = dict(msg)
        new_msg["content"] = _media_strip_note(texts[0] if texts else "")
        out = list(messages)
        out[i] = new_msg
        return out, True
    return messages, False


def trim_messages_to_budget(
    messages: List[Dict[str, Any]],
    *,
    max_input_tokens: int,
    extra_chars: int = 0,
    chars_per_token: int = CHARS_PER_TOKEN,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return messages whose estimated tokens fit ``max_input_tokens``."""
    msgs = [m for m in (messages or []) if isinstance(m, dict)]
    dropped = 0
    truncated = False
    media_stripped = 0
    max_in = max(32, int(max_input_tokens))

    # First: drop heavy base64 from older turns (keep text). Under-counting media
    # used to leave multi-MB data-URLs in the wire body after message drops.
    while estimate_messages_tokens(
        msgs, extra_chars=extra_chars, chars_per_token=chars_per_token
    ) > max_in or _total_media_url_chars(msgs) > MAX_HISTORY_MEDIA_CHARS:
        msgs, changed = _strip_oldest_media_message(msgs)
        if not changed:
            break
        media_stripped += 1
        truncated = True

    while len(msgs) > 3 and estimate_messages_tokens(
        msgs, extra_chars=extra_chars, chars_per_token=chars_per_token
    ) > max_in:
        before = len(msgs)
        msgs = _drop_oldest_nonessential(msgs)
        if len(msgs) == before:
            break
        dropped += before - len(msgs)

    est = estimate_messages_tokens(
        msgs, extra_chars=extra_chars, chars_per_token=chars_per_token
    )
    if est <= max_in:
        return msgs, {
            "dropped": dropped,
            "truncated": truncated,
            "media_stripped": media_stripped,
            "est_input": est,
        }

    # Shrink contents from oldest non-system / non-original-user messages until it fits.
    # Never mid-slice the active look: dropping its pixels mid-turn causes re-look loops.
    cpt = max(1, int(chars_per_token))
    first_user_idx = next(
        (i for i, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") == "user"),
        None,
    )
    latest_media = _latest_media_message_index(msgs)
    for i, msg in enumerate(msgs):
        if est <= max_in:
            break
        if msg.get("role") == "system" and i == 0:
            continue
        if first_user_idx is not None and i == first_user_idx:
            continue
        if latest_media is not None and i == latest_media:
            continue
        overflow_tokens = est - max_in
        overflow_chars = overflow_tokens * cpt + 32
        current_chars = _content_chars(msg.get("content"))
        if current_chars <= 40:
            if msg.get("tool_calls"):
                msg = dict(msg)
                msg.pop("tool_calls", None)
                msgs[i] = msg
                truncated = True
            continue
        floor = 400 if msg.get("role") == "system" else 24
        new_chars = max(floor, current_chars - overflow_chars)
        msg = dict(msg)
        msg["content"] = _truncate_content(msg.get("content"), new_chars)
        if msg.get("tool_calls") and new_chars < current_chars // 2:
            msg.pop("tool_calls", None)
        msgs[i] = msg
        truncated = True
        est = estimate_messages_tokens(
            msgs, extra_chars=extra_chars, chars_per_token=chars_per_token
        )

    # Original user prompt: shrink only after tool/assistant history is gone.
    if est > max_in and first_user_idx is not None:
        msg = msgs[first_user_idx]
        overflow_chars = (est - max_in) * cpt + 32
        current_chars = _content_chars(msg.get("content"))
        if current_chars > 80:
            msg = dict(msg)
            msg["content"] = _truncate_content(
                msg.get("content"), max(1200, current_chars - overflow_chars)
            )
            msgs[first_user_idx] = msg
            truncated = True
            est = estimate_messages_tokens(
                msgs, extra_chars=extra_chars, chars_per_token=chars_per_token
            )

    # System prompt can itself exceed the window — shrink it last, keep a prefix.
    if est > max_in:
        for i, msg in enumerate(msgs):
            if est <= max_in:
                break
            if msg.get("role") != "system":
                continue
            overflow_chars = (est - max_in) * cpt + 32
            current_chars = _content_chars(msg.get("content"))
            if current_chars <= 400:
                continue
            msg = dict(msg)
            msg["content"] = _truncate_content(
                msg.get("content"), max(400, current_chars - overflow_chars)
            )
            msgs[i] = msg
            truncated = True
            est = estimate_messages_tokens(
                msgs, extra_chars=extra_chars, chars_per_token=chars_per_token
            )

    # Last resort: shrink the last message too.
    # Never drop active look media parts — text-only ``[rlm.look]`` headers make
    # the VLM blind and trigger infinite re-look loops. Prefer text-only shrink;
    # if still oversized, leave media and let the 413 / split ladder recover.
    if est > max_in and msgs:
        overflow_chars = (est - max_in) * cpt + 32
        last = dict(msgs[-1])
        content = last.get("content")
        latest_media = _latest_media_message_index(msgs)
        target_chars = max(24, _content_chars(content) - overflow_chars)
        if (
            latest_media is not None
            and latest_media == len(msgs) - 1
            and isinstance(content, list)
            and any(_is_media_part(p) for p in content)
        ):
            last["content"] = _truncate_content_preserving_media(content, target_chars)
        else:
            last["content"] = _truncate_content(content, target_chars)
            last.pop("tool_calls", None)
        msgs[-1] = last
        truncated = True
        est = estimate_messages_tokens(
            msgs, extra_chars=extra_chars, chars_per_token=chars_per_token
        )

    return msgs, {
        "dropped": dropped,
        "truncated": truncated,
        "media_stripped": media_stripped,
        "est_input": est,
    }


def apply_context_budget(
    body: Dict[str, Any],
    context_length: Optional[int],
    *,
    extra_chars: int = 0,
    max_output_tokens: Optional[int] = None,
    chars_per_token: int = CHARS_PER_TOKEN,
) -> Dict[str, Any]:
    """Mutate ``body`` so input + max_tokens stays under the deployed window."""
    ctx = coerce_positive_int(context_length)
    stats: Dict[str, Any] = {
        "context_length": ctx,
        "applied": False,
        "dropped": 0,
        "truncated": False,
        "media_stripped": 0,
        "est_input": 0,
        "max_tokens": None,
    }
    if not ctx:
        return stats

    tools = body.get("tools")
    extra = int(extra_chars)
    if tools and extra <= 0:
        try:
            extra = len(json.dumps(tools, ensure_ascii=False))
        except Exception:
            extra = 400
    extra += 240  # chat template / tool_choice
    ctk = body.get("chat_template_kwargs")
    if isinstance(ctk, dict) and ctk.get("enable_thinking"):
        extra += 1600  # thinking chat-template reserve

    messages = list(body.get("messages") or [])
    requested = coerce_positive_int(body.get("max_tokens"))
    max_out = coerce_positive_int(max_output_tokens)

    # Reserve at least a small completion slice so the model can answer.
    min_comp = MIN_COMPLETION_TOKENS if ctx > MIN_COMPLETION_TOKENS + SAFETY_MARGIN_TOKENS + 64 else 1
    input_cap = ctx - SAFETY_MARGIN_TOKENS - min_comp
    fitted, trim = trim_messages_to_budget(
        messages,
        max_input_tokens=input_cap,
        extra_chars=extra,
        chars_per_token=chars_per_token,
    )
    body["messages"] = fitted
    est = int(trim.get("est_input") or estimate_messages_tokens(
        fitted, extra_chars=extra, chars_per_token=chars_per_token
    ))
    mt = budget_completion_tokens(
        est_input=est,
        context_length=ctx,
        requested=requested,
        max_output_tokens=max_out,
        min_completion=min_comp,
    )
    # Final hard inequality: est + max_tokens + safety < ctx
    while est + mt + SAFETY_MARGIN_TOKENS >= ctx and mt > 1:
        mt -= 1
    if est + mt + SAFETY_MARGIN_TOKENS >= ctx:
        # Input still too large even with max_tokens=1 — shrink again.
        fitted, trim = trim_messages_to_budget(
            fitted,
            max_input_tokens=max(32, ctx - SAFETY_MARGIN_TOKENS - 1),
            extra_chars=extra,
            chars_per_token=max(1, int(chars_per_token) - 1) if chars_per_token > 1 else 1,
        )
        body["messages"] = fitted
        est = int(trim.get("est_input") or 0)
        mt = max(1, ctx - est - SAFETY_MARGIN_TOKENS)

    body["max_tokens"] = int(mt)
    stats.update(
        {
            "applied": True,
            "dropped": int(trim.get("dropped") or 0),
            "truncated": bool(trim.get("truncated")),
            "media_stripped": int(trim.get("media_stripped") or 0),
            "est_input": est,
            "max_tokens": int(mt),
        }
    )
    return stats


def parse_context_overflow_error(text: str) -> Optional[Dict[str, int]]:
    """Extract (context_length, input_tokens) from a vLLM / proxy error body."""
    raw = str(text or "")
    if not raw:
        return None
    lowered = raw.lower()
    if not any(
        s in lowered
        for s in (
            "context length",
            "maximum context",
            "max_model_len",
            "maximum model length",
            "decoder prompt",
        )
    ):
        return None
    m = _CTX_PATTERNS[0].search(raw)
    if m:
        return {"context_length": int(m.group(1)), "requested": int(m.group(2))}
    m = _CTX_PATTERNS[1].search(raw)
    if m:
        return {"context_length": int(m.group(1)), "input_tokens": int(m.group(2))}
    m = _CTX_PATTERNS[2].search(raw)
    if m:
        return {"context_length": int(m.group(2)), "input_tokens": int(m.group(1))}
    m = _CTX_PATTERNS[3].search(raw)
    if m:
        return {"context_length": int(m.group(2)), "input_tokens": int(m.group(1))}
    m = _CTX_PATTERNS[4].search(raw)
    if m:
        return {"context_length": int(m.group(1)), "input_tokens": int(m.group(2))}
    m2 = re.search(r"context length is only\s+(\d+)", raw, re.I)
    if m2:
        return {"context_length": int(m2.group(1))}
    m3 = re.search(
        r"maximum context length is\s+(\d+)",
        raw,
        re.I,
    )
    if m3:
        return {"context_length": int(m3.group(1))}
    return None


# Byte-size / gateway body overflows (Prime-aligned). Not token-window math —
# remote edges (e.g. nginx) reject large JSON before the model sees it.
_REQUEST_TOO_LARGE_NEEDLES = (
    "request_too_large",
    "entity too large",
    "request entity too large",
    "request exceeds the maximum size",
    "payload too large",
    "content too large",
)

# Explicit 413 status surfaces — avoid matching random ports like :41356.
_REQUEST_TOO_LARGE_STATUS = re.compile(
    r"(?:llm\s+(?:stream\s+)?error|http(?:/\d\.\d)?|status(?:\s+code)?)\s*[:=]?\s*413"
    r"|\b413\s+request\s+entity"
    r"|['\"]status['\"]\s*:\s*413",
    re.I,
)

_OVERFLOW_NEEDLES = (
    "prompt is too long",
    "exceeds the context window",
    "input token count",
    "maximum prompt length",
    "reduce the length of the messages",
    "maximum context length",
    "exceeds the available context size",
    "greater than the context length",
    "context window exceeds limit",
    "exceeded model token limit",
    "too large for model with",
    "model_context_window_exceeded",
    "context length exceeded",
    "context_length_exceeded",
    "too many tokens",
    "token limit exceeded",
    "max_model_len",
    "maximum model length",
    "decoder prompt",
    "max_tokens' is too large",
    "max_tokens is too large",
    "'max_tokens' is too large",
)

_NON_OVERFLOW_NEEDLES = (
    "rate limit",
    "too many requests",
    "throttling",
    # Schema / payload validation — not a context-window problem.
    "string_type",
    "validation error",
    "validation errors",
    "input should be a valid string",
    "iterable_type",
    "chatcompletiondevelopermessageparam",
    "chatcompletionassistantmessageparam",
)

_INVALID_IMAGE_NEEDLES = (
    "failed to load image",
    "image file is truncated",
    "cannot identify image",
    "truncated jpeg",
    "truncated png",
    "broken data stream",
    "image decode",
    "invalid image",
    "corrupt image",
    "look_image_decode_failed",
)


def looks_like_message_schema_error(text: str) -> bool:
    """True when upstream 400'd on message shape (e.g. content=null), not tokens."""
    low = str(text or "").lower()
    if not low:
        return False
    if "validation error" in low or "string_type" in low:
        return True
    if "input should be a valid string" in low and "content" in low:
        return True
    if "content" in low and "null" in low and ("400" in low or "bad request" in low):
        return True
    return False


def looks_like_invalid_image_error(text: str) -> bool:
    """True when the VLM/backend rejected a corrupt or truncated image payload."""
    low = str(text or "").lower()
    if not low:
        return False
    return any(n in low for n in _INVALID_IMAGE_NEEDLES)


def looks_like_request_too_large(text: str) -> bool:
    """True when upstream rejected the HTTP body (413 / entity too large)."""
    raw = str(text or "")
    low = raw.lower()
    if not low:
        return False
    if any(n in low for n in _NON_OVERFLOW_NEEDLES):
        return False
    if any(n in low for n in _REQUEST_TOO_LARGE_NEEDLES):
        return True
    if _REQUEST_TOO_LARGE_STATUS.search(raw):
        return True
    return False


def looks_like_context_overflow(text: str) -> bool:
    """True for token-window overflow *or* HTTP request-too-large (413).

    Corrupt/truncated image 400s and message-schema 400s (content=null) are
    *not* overflow — those must not enter compact/ContextOverflowExhausted.
    """
    if looks_like_invalid_image_error(text):
        return False
    if looks_like_message_schema_error(text):
        return False
    if parse_context_overflow_error(text):
        return True
    if looks_like_request_too_large(text):
        return True
    low = str(text or "").lower()
    if not low:
        return False
    if any(n in low for n in _NON_OVERFLOW_NEEDLES):
        return False
    if any(n in low for n in _OVERFLOW_NEEDLES):
        return True
    # Bare "400 bad request" without overflow needles is ambiguous — do not
    # treat as overflow (message schema / tool-args / etc.).
    return "400" in low and (
        "maximum context" in low
        or "context length" in low
        or "too many tokens" in low
        or "context_length" in low
    )
