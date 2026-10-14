"""SSE chat stream helpers for RLM (assemble OpenAI chunks → completion)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Union

from src.orchestrators.rlm.runtime.context_budget import MIN_COMPLETION_TOKENS

OnDelta = Callable[[Dict[str, Any]], Union[None, Awaitable[None]]]

# Catalog ``max_output_tokens`` caps each completion; stitch until stop so chat
# answers can grow far beyond one window (same idea as planning synthesis).
LENGTH_FINISH_REASONS = frozenset({"length", "max_tokens"})
DEFAULT_MAX_OUTPUT_CONTINUATIONS = 64
# How many trailing characters of the cut-off reply we quote so the model
# can latch onto the exact resume point (instead of regenerating the intro).
LENGTH_CONTINUE_ANCHOR_CHARS = 700

LENGTH_CONTINUE_USER = (
    "Your previous reply was cut off by the output length limit. "
    "Continue from the exact stopping point; output only the missing "
    "remainder (user-visible answer). Do not repeat text already shown. "
    "If you were still thinking, finish briefly then write the full answer."
)


def build_length_continue_user(
    partial_content: str,
    *,
    reasoning: str = "",
    anchor_chars: int = LENGTH_CONTINUE_ANCHOR_CHARS,
) -> str:
    """User nudge that quotes the end of what was already produced."""
    text = str(partial_content or "").rstrip()
    reason = str(reasoning or "").strip()
    if not text.strip():
        if reason:
            tail = reason[-max(120, int(anchor_chars)) :]
            return (
                "Your previous reply hit the output length limit during thinking; "
                "no user-visible answer was emitted yet.\n"
                "Write the user-visible answer now. Do not restart a long thought "
                "trace. Here is the END of your prior thinking for context only:\n\n"
                "<<<THINKING_ENDS_HERE>>>\n"
                f"{tail}\n"
                "<<<START_USER_VISIBLE_ANSWER_NOW>>>\n"
            )
        return LENGTH_CONTINUE_USER

    n = max(120, int(anchor_chars or LENGTH_CONTINUE_ANCHOR_CHARS))
    tail = text[-n:]
    omitted = max(0, len(text) - len(tail))
    head_note = f"(…{omitted} characters omitted above…)\n" if omitted else ""
    return (
        "Your previous reply was cut off by the output length limit.\n"
        "The block below is the END of what was ALREADY delivered to the user. "
        "Resume EXACTLY after the last character of that block. "
        "Output ONLY the missing continuation — no title restart, no repeated "
        "introduction, no paraphrase of earlier paragraphs.\n\n"
        "<<<ALREADY_WRITTEN_ENDS_HERE>>>\n"
        f"{head_note}{tail}\n"
        "<<<CONTINUE_AFTER_THIS>>>\n"
    )

# AF parity: truncated tool-call JSON → heal + tool-result notice + re-ask
# (see prompts.TRUNCATED_TOOL_ARGS_RESULT / MAX_MALFORMED_TOOL_ARGS_NUDGES).
DEFAULT_MAX_TOOL_ARG_RETRIES = 2
TRUNCATED_TOOL_ARGS_TOOL_RESULT = (
    "[rlm] ipython tool call arguments were truncated or invalid JSON "
    "(Unterminated string / incomplete stream). Code was not executed. "
    "Re-send a complete ipython tool call with valid JSON arguments."
)

ChatInvokeFn = Callable[..., Awaitable[Dict[str, Any]]]


def _tool_args_parse_ok(raw_args: Any) -> bool:
    if isinstance(raw_args, dict):
        return True
    if raw_args is None:
        return True
    if not isinstance(raw_args, str):
        return False
    text = raw_args.strip()
    if not text:
        return True
    try:
        parsed = json.loads(text)
    except Exception:
        return False
    return isinstance(parsed, dict)


def heal_completion_tool_arguments(msg: Dict[str, Any]) -> int:
    """Rewrite invalid tool ``arguments`` to ``{\"code\":\"\"}`` (wire-safe)."""
    healed = 0
    empty_args = json.dumps({"code": ""}, ensure_ascii=False)
    tcs = msg.get("tool_calls")
    if not isinstance(tcs, list):
        return 0
    for tc in tcs:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else None
        if not isinstance(fn, dict):
            continue
        if _tool_args_parse_ok(fn.get("arguments")):
            continue
        fn["arguments"] = empty_args
        healed += 1
    return healed


def completion_tool_args_invalid(data: Dict[str, Any] | None) -> bool:
    msg = completion_message(data)
    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        if not _tool_args_parse_ok(fn.get("arguments")):
            return True
    return False


def should_retry_truncated_tool_calls(data: Dict[str, Any] | None) -> bool:
    """AF pattern: incomplete tool JSON must not be executed — re-ask the model."""
    return completion_has_tool_calls(data) and completion_tool_args_invalid(data)


def max_tool_arg_retries(cfg: Optional[Dict[str, Any]] = None) -> int:
    if isinstance(cfg, dict):
        raw = cfg.get("rlm_max_tool_arg_retries")
        if raw is not None:
            try:
                return max(0, int(raw))
            except (TypeError, ValueError):
                pass
    env = str(os.environ.get("AIV2_RLM_MAX_TOOL_ARG_RETRIES") or "").strip()
    if env:
        try:
            return max(0, int(env))
        except (TypeError, ValueError):
            pass
    return DEFAULT_MAX_TOOL_ARG_RETRIES


def tool_truncation_retry_body(
    body: Dict[str, Any],
    partial: Dict[str, Any],
) -> Dict[str, Any]:
    """AF turn-loop shape: healed assistant + ``TRUNCATED_TOOL_ARGS`` tool results."""
    import copy

    req = dict(body or {})
    msgs = list(req.get("messages") or [])
    asst = copy.deepcopy(completion_message(partial))
    if not asst:
        asst = {"role": "assistant", "content": None, "tool_calls": []}
    asst["role"] = "assistant"
    heal_completion_tool_arguments(asst)
    msgs.append(asst)
    for tc in asst.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "name": str(fn.get("name") or "ipython"),
                "content": TRUNCATED_TOOL_ARGS_TOOL_RESULT,
            }
        )
    req["messages"] = msgs
    # Keep tools so the model can re-emit a complete ipython call.
    return req


def max_output_continuations(cfg: Optional[Dict[str, Any]] = None) -> int:
    """How many length-stitch rounds after the first completion (0 = disable)."""
    if isinstance(cfg, dict):
        raw = cfg.get("rlm_max_output_continuations")
        if raw is not None:
            try:
                return max(0, int(raw))
            except (TypeError, ValueError):
                pass
    env = str(os.environ.get("AIV2_RLM_MAX_OUTPUT_CONTINUATIONS") or "").strip()
    if env:
        try:
            return max(0, int(env))
        except (TypeError, ValueError):
            pass
    return DEFAULT_MAX_OUTPUT_CONTINUATIONS


def completion_choice(data: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return {}
    return choices[0]


def completion_message(data: Dict[str, Any] | None) -> Dict[str, Any]:
    choice = completion_choice(data)
    msg = choice.get("message")
    return msg if isinstance(msg, dict) else {}


def completion_finish_reason(data: Dict[str, Any] | None) -> str:
    choice = completion_choice(data)
    return str(choice.get("finish_reason") or "").strip().lower()


def completion_has_tool_calls(data: Dict[str, Any] | None) -> bool:
    msg = completion_message(data)
    tcs = msg.get("tool_calls") or []
    return isinstance(tcs, list) and len(tcs) > 0


def _completion_reasoning_text(msg: Mapping[str, Any] | None) -> str:
    if not isinstance(msg, Mapping):
        return ""
    return str(
        msg.get("reasoning_content")
        or msg.get("reasoning")
        or msg.get("reasoning_text")
        or ""
    ).strip()


def _looks_like_cut_channel(content: str) -> bool:
    """Unclosed Gemma/Qwen thinking markup — visible answer never closed."""
    s = str(content or "")
    if not s:
        return False
    lower = s.lower()
    if "<|channel>" in lower and "<channel|>" not in lower:
        return True
    if "<think>" in lower and "</think>" not in lower:
        return True
    return False


def _usage_hit_max_tokens(
    data: Dict[str, Any] | None, max_tokens: Optional[int]
) -> bool:
    if not max_tokens or int(max_tokens) <= 0 or not isinstance(data, dict):
        return False
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    raw = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("completion_tokens_total")
    )
    try:
        used = int(raw)
    except (TypeError, ValueError):
        return False
    return used >= max(1, int(0.95 * int(max_tokens)))


def should_continue_on_length(
    data: Dict[str, Any] | None,
    *,
    max_tokens: Optional[int] = None,
) -> bool:
    """True when the model hit the per-call output cap mid-answer (not tools).

    Continuations are for truncated streams only — not for a finished long
    answer that already closed with natural ``stop`` (avoids repeating a
    closing line like «لیست به پایان رسید» across dozens of stitch rounds).
    """
    if completion_has_tool_calls(data):
        # Tool JSON recovery is ``should_retry_truncated_tool_calls`` (not prose stitch).
        return False
    msg = completion_message(data)
    content = str(msg.get("content") or "").strip()
    reasoning = _completion_reasoning_text(msg)
    if not content and not reasoning:
        return False
    fr = completion_finish_reason(data)
    if fr in LENGTH_FINISH_REASONS:
        return True
    raw_content = str(msg.get("content") or "")
    if _looks_like_cut_channel(raw_content):
        return True
    # Proxies may strip finish_reason or lie with stop while usage is at cap.
    # Only continue when prose still looks truncated — a closed sentence means
    # the model finished on purpose.
    if _usage_hit_max_tokens(data, max_tokens):
        if fr in ("",) or fr is None:
            return True
        if fr == "stop" and _looks_truncated_prose(content):
            return True
    return False


_SENTENCE_END_RE = re.compile(
    r'(?:[.!?…؟۔]|[.!?]["\'»\)\]])\s*$',
    re.UNICODE,
)


def _looks_truncated_prose(content: str) -> bool:
    """Heuristic: complete closing sentences should not trigger length stitch."""
    text = str(content or "").rstrip()
    if not text:
        return False
    if _looks_like_cut_channel(text):
        return True
    if len(text) < 40:
        return not bool(_SENTENCE_END_RE.search(text))
    if _SENTENCE_END_RE.search(text):
        return False
    return True


def _merge_assistant_text(prev: str, nxt: str) -> str:
    a = str(prev or "")
    b = str(nxt or "")
    if not b:
        return a
    if not a:
        return b
    # Suffix/prefix overlap (model resumes mid-sentence and repeats a few chars).
    max_overlap = min(240, len(a), len(b))
    for n in range(max_overlap, 0, -1):
        if a.endswith(b[:n]):
            return a + b[n:]
    # Model restarted from the introduction: longest common prefix with ``a``.
    if len(a) >= 40 and len(b) >= 40:
        limit = min(len(a), len(b))
        i = 0
        while i < limit and a[i] == b[i]:
            i += 1
        if i >= 40:
            # Prefer breaking on a newline so we do not splice mid-word.
            cut = b.rfind("\n", 0, i)
            if cut >= 40:
                i = cut + 1
            return a + b[i:]
    if a.endswith((" ", "\n")) or b.startswith((" ", "\n")):
        return a + b
    return a + b


def stitch_length_completions(
    base: Dict[str, Any],
    cont: Dict[str, Any],
) -> Dict[str, Any]:
    """Append continuation content/reasoning onto ``base``; keep last finish_reason."""
    out = dict(base or {})
    base_msg = dict(completion_message(base))
    cont_msg = completion_message(cont)
    merged = dict(base_msg)
    merged["role"] = "assistant"
    merged["content"] = _merge_assistant_text(
        str(base_msg.get("content") or ""),
        str(cont_msg.get("content") or ""),
    )
    for key in ("reasoning_content", "reasoning"):
        prev = str(base_msg.get(key) or "")
        nxt = str(cont_msg.get(key) or "")
        if prev or nxt:
            merged[key] = _merge_assistant_text(prev, nxt)
    choice = dict(completion_choice(base))
    choice["message"] = merged
    choice["finish_reason"] = completion_finish_reason(cont) or choice.get(
        "finish_reason"
    )
    out["choices"] = [choice]
    return out


def length_continuation_body(
    body: Dict[str, Any],
    partial: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the next request: history + partial assistant + anchored continue nudge."""
    req = dict(body or {})
    msgs = list(req.get("messages") or [])
    msg = completion_message(partial)
    content = str(msg.get("content") or "")
    reasoning = _completion_reasoning_text(msg)
    if not content.strip() and reasoning:
        # Thinking burned the token budget — stub assistant turn, then ask for answer.
        content = (
            "(Output length limit hit during thinking; no user-visible answer "
            "was emitted yet.)"
        )
    assistant: Dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    for key in ("reasoning_content", "reasoning"):
        if msg.get(key):
            assistant[key] = msg.get(key)
    msgs.append(assistant)
    msgs.append(
        {
            "role": "user",
            "content": build_length_continue_user(
                str(msg.get("content") or ""),
                reasoning=reasoning,
            ),
        }
    )
    req["messages"] = msgs
    # Continuations are prose-only; drop tools so the model keeps writing.
    req.pop("tools", None)
    req.pop("tool_choice", None)
    return req


async def invoke_with_length_continuation(
    invoke_once: ChatInvokeFn,
    body: Dict[str, Any],
    *,
    on_delta: Optional[OnDelta] = None,
    max_continuations: Optional[int] = None,
    max_tool_retries: Optional[int] = None,
    prepare_body: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    on_continue: Optional[Callable[[int, Dict[str, Any]], Any]] = None,
    on_tool_retry: Optional[Callable[[int, Dict[str, Any]], Any]] = None,
) -> Dict[str, Any]:
    """Call ``invoke_once``, recover truncated tool JSON, then stitch length cuts.

    1. **Tool truncation (AF):** heal + tool-result notice + re-invoke (replace
       the bad completion; do not execute truncated args).
    2. **Prose length:** stitch ``finish_reason=length`` chunks until stop.
    """
    limit = (
        DEFAULT_MAX_OUTPUT_CONTINUATIONS
        if max_continuations is None
        else max(0, int(max_continuations))
    )
    tool_limit = (
        DEFAULT_MAX_TOOL_ARG_RETRIES
        if max_tool_retries is None
        else max(0, int(max_tool_retries))
    )
    req = dict(body or {})
    if prepare_body is not None:
        req = prepare_body(req) or req

    data = await invoke_once(req, on_delta=on_delta)
    if not isinstance(data, dict):
        return data  # type: ignore[return-value]

    # --- AF: truncated tool-call arguments → re-ask (replace completion) ---
    tool_n = 0
    while tool_n < tool_limit and should_retry_truncated_tool_calls(data):
        tool_n += 1
        if on_tool_retry is not None:
            maybe = on_tool_retry(tool_n, data)
            if hasattr(maybe, "__await__"):
                await maybe  # type: ignore[misc]
        retry_body = tool_truncation_retry_body(body, data)
        if prepare_body is not None:
            retry_body = prepare_body(retry_body) or retry_body
        nxt = await invoke_once(retry_body, on_delta=on_delta)
        if not isinstance(nxt, dict):
            break
        data = nxt

    # --- Prose: stitch across output-length caps ---
    requested_max = None
    try:
        requested_max = int(
            req.get("max_tokens")
            or req.get("max_completion_tokens")
            or (body or {}).get("max_tokens")
            or 0
        ) or None
    except (TypeError, ValueError):
        requested_max = None
    n = 0
    while n < limit and should_continue_on_length(data, max_tokens=requested_max):
        n += 1
        if on_continue is not None:
            maybe = on_continue(n, data)
            if hasattr(maybe, "__await__"):
                await maybe  # type: ignore[misc]
        cont_body = length_continuation_body(body, data)
        if prepare_body is not None:
            cont_body = prepare_body(cont_body) or cont_body
            room = cont_body.get("max_tokens") or cont_body.get("max_completion_tokens")
            if isinstance(room, int) and room < MIN_COMPLETION_TOKENS:
                # The stitched history fills the window: let the turn loop compact.
                break
        cont = await invoke_once(cont_body, on_delta=on_delta)
        if not isinstance(cont, dict):
            break
        cont_msg = completion_message(cont)
        if not (
            str(cont_msg.get("content") or "").strip()
            or _completion_reasoning_text(cont_msg)
        ):
            break
        data = stitch_length_completions(data, cont)
        # Refresh cap from the prepared continuation body when available.
        try:
            requested_max = int(
                cont_body.get("max_tokens")
                or cont_body.get("max_completion_tokens")
                or requested_max
                or 0
            ) or requested_max
        except (TypeError, ValueError):
            pass
    return data


def _delta_reasoning(delta: Dict[str, Any]) -> str:
    return str(
        delta.get("reasoning_content")
        or delta.get("reasoning")
        or delta.get("reasoning_text")
        or ""
    )


def merge_tool_call_delta(acc: List[Dict[str, Any]], pieces: Any) -> None:
    """Merge OpenAI streaming ``tool_calls`` deltas into ``acc`` by index."""
    if not isinstance(pieces, list):
        return
    for tc in pieces:
        if not isinstance(tc, dict):
            continue
        try:
            idx = int(tc.get("index") if tc.get("index") is not None else 0)
        except (TypeError, ValueError):
            idx = 0
        while len(acc) <= idx:
            acc.append(
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            )
        slot = acc[idx]
        if tc.get("id"):
            slot["id"] = str(tc["id"])
        if tc.get("type"):
            slot["type"] = str(tc["type"])
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        if fn.get("name"):
            # Name usually arrives once; prefer non-empty overwrite.
            slot["function"]["name"] = str(fn["name"])
        if "arguments" in fn:
            slot["function"]["arguments"] = str(slot["function"].get("arguments") or "") + str(
                fn.get("arguments") or ""
            )


def apply_chat_delta(
    state: Dict[str, Any],
    delta: Dict[str, Any],
    *,
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Mutate assembly ``state`` with one chat.completion.chunk delta.

    Returns a normalized delta dict suitable for live UI callbacks
    (``content`` / ``reasoning`` / ``tool_calls``).
    """
    content = str(delta.get("content") or "")
    reasoning = _delta_reasoning(delta)
    tool_calls = delta.get("tool_calls")

    if content:
        state["content"] = str(state.get("content") or "") + content
    if reasoning:
        state["reasoning"] = str(state.get("reasoning") or "") + reasoning
    if tool_calls:
        merge_tool_call_delta(state.setdefault("tool_calls", []), tool_calls)
    if finish_reason:
        state["finish_reason"] = finish_reason

    out: Dict[str, Any] = {}
    if content:
        out["content"] = content
    if reasoning:
        out["reasoning"] = reasoning
    if tool_calls:
        out["tool_calls"] = tool_calls
    return out


def completion_from_stream_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Build a non-stream chat.completion payload from assembled stream state."""
    msg: Dict[str, Any] = {"role": "assistant", "content": state.get("content") or ""}
    reasoning = str(state.get("reasoning") or "").strip()
    if reasoning:
        msg["reasoning_content"] = reasoning
        msg["reasoning"] = reasoning
    tool_calls = state.get("tool_calls") or []
    if tool_calls:
        cleaned: List[Dict[str, Any]] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            cleaned.append(
                {
                    "id": str(tc.get("id") or ""),
                    "type": str(tc.get("type") or "function"),
                    "function": {
                        "name": str(fn.get("name") or ""),
                        "arguments": str(fn.get("arguments") or ""),
                    },
                }
            )
        if cleaned:
            msg["tool_calls"] = cleaned
            if not msg.get("content"):
                msg["content"] = None
    # Do NOT invent ``stop`` when the provider omitted finish_reason — that
    # disables length-continuation stitching. Only default tool_calls when tools
    # are present.
    fr = state.get("finish_reason")
    if fr:
        finish: Optional[str] = str(fr)
    elif tool_calls:
        finish = "tool_calls"
    else:
        finish = None
    out: Dict[str, Any] = {
        "choices": [
            {
                "index": 0,
                "message": msg,
                "finish_reason": finish,
            }
        ]
    }
    usage = state.get("usage")
    if isinstance(usage, dict) and usage:
        out["usage"] = usage
    return out


async def maybe_await_delta(on_delta: Optional[OnDelta], delta: Dict[str, Any]) -> None:
    if not on_delta or not delta:
        return
    result = on_delta(delta)
    if hasattr(result, "__await__"):
        await result  # type: ignore[misc]


async def consume_openai_sse_lines(
    lines,
    *,
    on_delta: Optional[OnDelta] = None,
) -> Dict[str, Any]:
    """Consume ``data: …`` SSE lines (async or sync iterable) → completion dict."""
    state: Dict[str, Any] = {"content": "", "reasoning": "", "tool_calls": []}
    async for raw in _aiter(lines):
        line = str(raw or "").strip()
        if not line:
            continue
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            payload = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("error"):
            err = payload["error"]
            detail = err if isinstance(err, str) else json.dumps(err, ensure_ascii=False)
            raise RuntimeError(f"LLM stream error: {detail[:2000]}")
        choice = (payload.get("choices") or [{}])[0]
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        finish = choice.get("finish_reason")
        norm = apply_chat_delta(state, delta or {}, finish_reason=str(finish) if finish else None)
        await maybe_await_delta(on_delta, norm)
    return completion_from_stream_state(state)


async def _aiter(lines):
    if hasattr(lines, "__aiter__"):
        async for item in lines:
            yield item
        return
    for item in lines:
        yield item
