"""
vLLM Client - کلاینت OpenAI-compatible برای vLLM
"""
import asyncio
import json
import logging
import re
import uuid
from json import JSONDecoder
from typing import Any, Dict, List, Optional, AsyncGenerator, Set

import httpx
from openai import AsyncOpenAI, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings, get_llm_config
from ..debug_ndjson import debug_ndjson

log = logging.getLogger(__name__)

# Repetition guard: abort generation and let orchestrator ask the model to repair.
_STREAM_REP_MIN_CHUNK = 48
_STREAM_REP_MIN_COUNT = 4

REPETITION_ABORT_REPAIR_HINT = (
    "[Repetitive output aborted]\n"
    "Your previous generation was cut off because it repeated the same sentences or requests "
    "without making progress.\n\n"
    "Do NOT repeat that text. Either:\n"
    "- emit a proper native tool_call if you need data, or\n"
    "- give a short, new answer in the user's language using evidence you already have.\n"
    "If you lack data, say what is missing and which tool you will call next."
)


def _stream_should_abort_repetition(text: str) -> bool:
    """True when the model is looping the same paragraph in ``content``."""
    t = (text or "").strip()
    if len(t) < _STREAM_REP_MIN_CHUNK * _STREAM_REP_MIN_COUNT:
        return False
    parts = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
    if len(parts) >= _STREAM_REP_MIN_COUNT:
        tail = parts[-1]
        if len(tail) >= _STREAM_REP_MIN_CHUNK:
            recent = parts[-_STREAM_REP_MIN_COUNT:]
            if all(p == tail for p in recent):
                return True
    # Same medium-sized substring repeated many times (no paragraph breaks).
    window = t[-2500:]
    if len(window) >= _STREAM_REP_MIN_CHUNK * 2:
        chunk = window[-_STREAM_REP_MIN_CHUNK:]
        if window.count(chunk) >= _STREAM_REP_MIN_COUNT + 2:
            return True
    return False


def _has_repeated_sentence(text: str, *, min_len: int = 36, min_count: int = 3) -> bool:
    """Detect the same sentence/line repeated many times (common in Persian planning loops)."""
    t = (text or "").strip()
    if len(t) < min_len * min_count:
        return False
    parts = [p.strip() for p in re.split(r"[\n]+", t) if len(p.strip()) >= min_len]
    if not parts:
        return False
    from collections import Counter

    counts = Counter(parts)
    return max(counts.values()) >= min_count


def _strip_looping_assistant_content(content: str, *, has_tool_calls: bool) -> str:
    """Drop internal planning monologue when the model should only emit tool_calls."""
    c = (content or "").strip()
    if not c or not has_tool_calls:
        return c
    if _stream_should_abort_repetition(c) or _has_repeated_sentence(c):
        return ""
    return c


def _resolve_max_tokens(
    override: Optional[int],
    client_default: Optional[int],
) -> Optional[int]:
    """Return max_tokens for the API, or None to omit (no orchestrator output cap)."""
    if override is not None:
        return override if override > 0 else None
    if client_default is not None and client_default > 0:
        return client_default
    return None


def _params_with_max_tokens(
    params: Dict[str, Any],
    override: Optional[int],
    client_default: Optional[int],
) -> Dict[str, Any]:
    effective = _resolve_max_tokens(override, client_default)
    if effective is not None:
        params["max_tokens"] = effective
    return params


def _extract_tool_calls_from_choice_message(message: Any) -> List[Dict[str, Any]]:
    """
    Some OpenAI-compatible servers (or proxies) omit ``delta.tool_calls`` and only
    attach the completed ``tool_calls`` list on ``choice.message`` — often on the
    final stream chunk. Without this, the orchestrator never reaches ``execute_tools``.
    """
    raw = getattr(message, "tool_calls", None)
    if not raw:
        return []
    out: List[Dict[str, Any]] = []
    for i, tc in enumerate(raw):
        fn = getattr(tc, "function", None)
        name = ""
        raw_args: Any = ""
        if fn is not None:
            name = getattr(fn, "name", None) or ""
            raw_args = getattr(fn, "arguments", None)
        args: Dict[str, Any] = {}
        if isinstance(raw_args, str) and raw_args.strip():
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        tid = getattr(tc, "id", None) or f"call_{i}"
        if name:
            out.append({"id": tid, "name": name, "arguments": args})
    return out


def _try_parse_qwen_xml_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    Fallback when ``delta.tool_calls`` is empty: Qwen3.x with vLLM ``qwen3_coder`` may still leave a
    completed ``<tool_call>...</tool_call>`` block inside ``content`` (often JSON with ``name`` /
    ``arguments``). vLLM normally rewrites this to OpenAI deltas; we parse only if that step missed.
    """
    import re

    if not text or "<tool_call>" not in text:
        return []
    out: List[Dict[str, Any]] = []
    for m in re.finditer(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", text):
        raw = (m.group(1) or "").strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool_name")
        if not name or not isinstance(name, str):
            continue
        args = obj.get("arguments")
        if args is None:
            args = {k: v for k, v in obj.items() if k not in ("name", "tool_name")}
        if not isinstance(args, dict):
            args = {}
        out.append({"id": f"xml_{len(out)}", "name": name, "arguments": args})
    return out


def _openai_tool_names_from_schema(tools: Optional[List[Dict[str, Any]]]) -> Optional[Set[str]]:
    if not tools:
        return None
    names: Set[str] = set()
    for t in tools:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict):
            continue
        n = fn.get("name")
        if isinstance(n, str) and n.strip():
            names.add(n.strip())
    return names or None


def _try_parse_json_fence_tool_calls(
    text: str,
    allowed_tool_names: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Fallback when native ``delta.tool_calls`` is empty: some models write a single
    OpenAI-shaped object (``name`` + ``arguments``) inside a markdown `` ```json `` fence
    instead of using the wire tool-call channel — same symptom as mis-tuned templates.
    """
    if not text or "```" not in text:
        return []
    out: List[Dict[str, Any]] = []
    dec = JSONDecoder()
    for m in re.finditer(r"```(?:json)?\s*", text, re.IGNORECASE):
        sub = text[m.end() :]
        # Some models emit a bare ``[`` as the top level (array of calls), not
        # an object. Try both starts and pick whichever decoded earlier.
        starts = [i for i in (sub.find("{"), sub.find("[")) if i >= 0]
        if not starts:
            continue
        br = min(starts)
        try:
            obj, end = dec.raw_decode(sub, br)
        except json.JSONDecodeError:
            continue
        tail = sub[end:].lstrip()
        if not tail.startswith("```"):
            continue

        # Unwrap common wrapper shapes the model invents:
        #   {"tool_calls":[ ... ]}        → take inner array
        #   {"calls":[ ... ]}             → take inner array
        #   {"function":{name, args}}     → treat as one OpenAI-shaped call
        #   [ {...}, {...} ]              → already correct
        #   {name, arguments}             → already correct
        items: List[Any]
        if isinstance(obj, list):
            items = obj
        elif isinstance(obj, dict):
            inner = obj.get("tool_calls") or obj.get("calls")
            if isinstance(inner, list):
                items = inner
            elif isinstance(obj.get("function"), dict):
                items = [obj["function"]]
            else:
                items = [obj]
        else:
            continue

        for o in items:
            if not isinstance(o, dict):
                continue
            # OpenAI tool_call shape sometimes nests inside ``function``.
            fn = o.get("function") if isinstance(o.get("function"), dict) else None
            name_src = fn or o
            name = name_src.get("name") or name_src.get("tool_name") or name_src.get("tool")
            if not isinstance(name, str) or not name.strip():
                continue
            name = name.strip()
            if allowed_tool_names is not None and name not in allowed_tool_names:
                continue
            args_src = fn or o
            args = (
                args_src.get("arguments")
                if "arguments" in args_src
                else args_src.get("parameters")
            )
            # OpenAI streams ``arguments`` as a JSON string — accept both.
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if args is None:
                args = {
                    k: v
                    for k, v in args_src.items()
                    if k
                    not in ("name", "tool_name", "tool", "id", "type", "function", "parameters")
                }
            if not isinstance(args, dict):
                args = {}
            tid = o.get("id") if isinstance(o.get("id"), str) and o.get("id") else f"fence_{uuid.uuid4().hex[:12]}"
            out.append({"id": tid, "name": name, "arguments": args})
    return out


def _try_parse_bare_json_tool_calls(
    text: str,
    allowed_tool_names: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Recover tool calls when the model emits **bare** JSON in ``content``
    (no markdown fence) — the most common failure mode for Qwen3.5 / Qwen2.5
    when vLLM's chat template doesn't surface the proper ``<tool_call>``
    wrapper for the ``--tool-call-parser`` to catch.

    Detected shapes (anywhere in the content, not just at the start):

    * ``{"tool_calls":[{...}, {...}]}``               — OpenAI wrapper
    * ``{"calls":[ ... ]}`` / ``{"functions":[ ... ]}`` — common aliases
    * ``{"name":"…","arguments":{…}}``                — single call
    * ``{"function":{"name":"…","arguments":"…"}}``   — OpenAI inner shape
    * A series of the above separated by whitespace or commas

    Behaviour mirrors :func:`_try_parse_json_fence_tool_calls` so the rest of
    the orchestrator sees the same ``{"id","name","arguments"}`` triples.
    """
    if not text:
        return []
    s = text.strip()
    if not s:
        return []
    if "{" not in s and "[" not in s:
        return []

    dec = JSONDecoder()
    out: List[Dict[str, Any]] = []

    def _consume_object(obj: Any) -> None:
        if isinstance(obj, list):
            items: List[Any] = obj
        elif isinstance(obj, dict):
            inner = (
                obj.get("tool_calls")
                or obj.get("calls")
                or obj.get("functions")
                or obj.get("function_calls")
            )
            if isinstance(inner, list):
                items = inner
            elif isinstance(obj.get("function"), dict):
                items = [obj["function"]]
            else:
                items = [obj]
        else:
            return

        for o in items:
            if not isinstance(o, dict):
                continue
            fn = o.get("function") if isinstance(o.get("function"), dict) else None
            name_src = fn or o
            name = (
                name_src.get("name")
                or name_src.get("tool_name")
                or name_src.get("tool")
            )
            if not isinstance(name, str) or not name.strip():
                continue
            name = name.strip()
            if allowed_tool_names is not None and name not in allowed_tool_names:
                continue
            args_src = fn or o
            args = (
                args_src.get("arguments")
                if "arguments" in args_src
                else args_src.get("parameters")
            )
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {}
            if args is None:
                args = {
                    k: v
                    for k, v in args_src.items()
                    if k
                    not in ("name", "tool_name", "tool", "id", "type", "function", "parameters")
                }
            if not isinstance(args, dict):
                args = {}
            tid = (
                o.get("id")
                if isinstance(o.get("id"), str) and o.get("id")
                else f"json_{len(out)}"
            )
            out.append({"id": tid, "name": name, "arguments": args})

    # Walk the text and try to decode JSON objects/arrays at any position.
    # We try every `{` and `[` cursor; ``raw_decode`` jumps past the decoded
    # span so we never re-parse the same region twice.
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch not in "{[":
            i += 1
            continue
        try:
            obj, end = dec.raw_decode(s, i)
        except json.JSONDecodeError:
            i += 1
            continue
        _consume_object(obj)
        i = end if end > i else i + 1

    # De-duplicate by (name, JSON-args, id) to keep things stable when a model
    # repeats the same intent multiple times in one breath.
    seen: Set[str] = set()
    dedup: List[Dict[str, Any]] = []
    for tc in out:
        try:
            key = f"{tc['name']}::{json.dumps(tc.get('arguments') or {}, sort_keys=True)}"
        except Exception:
            key = f"{tc['name']}::?"
        if key in seen:
            continue
        seen.add(key)
        dedup.append(tc)
    return dedup


def _kwargs_from_python_call_body(body: str) -> Dict[str, Any]:
    """Parse ``key="..."`` / ``key='...'`` inside a fake ``tool(arg=...)`` line."""
    out: Dict[str, Any] = {}
    if not body or not body.strip():
        return out
    for mm in re.finditer(r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"', body):
        out[mm.group(1)] = mm.group(2).replace('\\"', '"')
    for mm in re.finditer(r"(\w+)\s*=\s*'((?:[^'\\]|\\.)*)'", body):
        if mm.group(1) not in out:
            out[mm.group(1)] = mm.group(2).replace("\\'", "'")
    return out


def _args_from_call_body(body: str) -> Optional[Dict[str, Any]]:
    """
    Parse the body of a fake ``tool(...)`` call.

    Accepts:
    - ``key="..."`` / ``key='...'`` (Python kwargs)
    - ``{"queries": [...], ...}`` (single JSON object as positional arg)
    - ``"value"`` only — returned as ``{"value": "value"}`` is NOT done here;
      caller decides how to map positional values.

    Returns ``None`` when the body is unparseable, so the caller can skip it.
    """
    s = (body or "").strip()
    if not s:
        return {}
    # Single JSON object positional arg: tool({"key": "val"})
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    # Python kwargs
    kw = _kwargs_from_python_call_body(s)
    if kw:
        return kw
    return None


_CALL_RE = re.compile(
    r"\b([a-zA-Z_][\w.]{1,80})\s*\(\s*(\{[\s\S]*?\}|[^()]*?)\s*\)",
)


def _try_parse_tool_call_markdown_fence(
    text: str,
    allowed_tool_names: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Fallback recovery for tool calls that the model wrote as prose instead of
    using native ``delta.tool_calls``. Catches THREE shapes:

    1. ``\u0060\u0060\u0060tool_call`` fence (legacy Cline-ish)
    2. Any fence (e.g. ``\u0060\u0060\u0060`` with no lang, or ``\u0060\u0060\u0060python``)
       whose body contains ``tool_name(...)`` where ``tool_name`` is in
       ``allowed_tool_names``.
    3. Outside any fence: a bare ``tool_name(...)`` call on its own line where
       ``tool_name`` is in ``allowed_tool_names``.

    All three require a known tool name to avoid false positives on natural
    text.
    """
    if not text:
        return []

    out: List[Dict[str, Any]] = []
    seen_calls: Set[str] = set()  # de-dupe identical name+body matches

    def _consider_call(name: str, body: str) -> None:
        name = name.strip()
        if not name:
            return
        if allowed_tool_names is not None and name not in allowed_tool_names:
            return
        args = _args_from_call_body(body)
        if args is None:
            return
        sig = f"{name}::{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
        if sig in seen_calls:
            return
        seen_calls.add(sig)
        out.append({"id": f"callrec_{len(out)}", "name": name, "arguments": args})

    # Collect fence ranges so step #3 (bare-text) can skip them.
    fence_ranges: List[tuple[int, int]] = []
    for fm in re.finditer(r"```([a-zA-Z_]*)\s*([\s\S]*?)\s*```", text):
        fence_ranges.append((fm.start(), fm.end()))
        body = fm.group(2) or ""
        # Search every call-like substring inside the fence body.
        for cm in _CALL_RE.finditer(body):
            _consider_call(cm.group(1), cm.group(2))

    # Outside fences: accept lines that look like a tool-call (or a bullet
    # entry of one). This catches the "fake tool log" pattern where the
    # model writes a bullet list of past invocations, e.g.
    #     - session_files({"session_id": "..."}) → در حال اجرا
    # We strip common bullet markers and tolerate a trailing `→ ...`
    # arrow that we ignore (the args are what matters).
    def _in_fence(idx: int) -> bool:
        return any(s <= idx < e for s, e in fence_ranges)

    bullet_call_re = re.compile(
        r"^([a-zA-Z_][\w.]{1,80})\s*\(\s*([\s\S]*?)\s*\)\s*(?:[\u2192\u279C\->]+[^\n]*)?\s*$"
    )
    for line_m in re.finditer(r"[^\n]+", text):
        if _in_fence(line_m.start()):
            continue
        line = line_m.group(0).strip("`*-+\u2022\u25E6 \t")
        if "(" not in line or ")" not in line:
            continue
        cm = bullet_call_re.match(line)
        if not cm:
            continue
        _consider_call(cm.group(1), cm.group(2))

    return out


def _coerce_yamlish_scalar(val_raw: str) -> Any:
    v = (val_raw or "").strip()
    if not v:
        return ""
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1].replace('\\"', '"').replace("\\'", "'")
    vl = v.lower()
    if vl in ("true", "false"):
        return vl == "true"
    if vl in ("null", "none", "~"):
        return None
    if re.fullmatch(r"-?\d+", v):
        try:
            return int(v)
        except ValueError:
            return v
    if re.fullmatch(r"-?\d+\.\d+", v):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def _try_parse_yaml_tool_calls_leak(
    text: str,
    allowed_tool_names: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Fallback when the model prints pseudo-YAML instead of calling tools, e.g.::

        tool_calls:
        - tool_name: plugin_graph_explorer_graph_search
          arguments:
            query: \"جعفر\"
            node_type: \"person\"
    """
    if not text or "tool_name:" not in text.lower():
        return []
    lines = text.replace("\r\n", "\n").split("\n")
    n = len(lines)
    out: List[Dict[str, Any]] = []

    def parse_args_block(start_idx: int) -> tuple[Dict[str, Any], int]:
        args: Dict[str, Any] = {}
        i = start_idx
        if i >= n or not re.match(r"^\s*arguments:\s*$", lines[i], re.I):
            return args, i
        i += 1
        base_indent: Optional[int] = None
        while i < n:
            raw = lines[i]
            if not raw.strip():
                i += 1
                continue
            km = re.match(r"^(\s+)(\w+):\s*(.*)$", raw)
            if not km:
                break
            k_ind = len(km.group(1))
            if base_indent is None:
                base_indent = k_ind
            elif k_ind < base_indent:
                break
            key = km.group(2)
            if key.lower() == "arguments":
                i += 1
                continue
            val_raw = km.group(3)
            args[key] = _coerce_yamlish_scalar(val_raw)
            i += 1
        return args, i

    i = 0
    while i < n:
        if re.match(r"^\s*tool_calls:\s*$", lines[i], re.I):
            i += 1
            while i < n:
                m_tool = re.match(r"^\s*-\s*tool_name:\s*(\S+)\s*$", lines[i], re.I)
                if not m_tool:
                    if not lines[i].strip():
                        i += 1
                        continue
                    stripped = lines[i].lstrip()
                    if stripped.startswith("- ") or (
                        stripped and not stripped.startswith("#") and not lines[i].startswith((" ", "\t"))
                    ):
                        break
                    i += 1
                    continue
                name = m_tool.group(1).strip()
                i += 1
                args, i = parse_args_block(i)
                if allowed_tool_names is not None and name not in allowed_tool_names:
                    continue
                out.append({"id": f"yaml_leak_{len(out)}", "name": name, "arguments": args})
            break
        i += 1

    if not out:
        for j in range(n):
            m_tool = re.match(r"^\s*-\s*tool_name:\s*(\S+)\s*$", lines[j], re.I)
            if not m_tool:
                continue
            name = m_tool.group(1).strip()
            args, _ = parse_args_block(j + 1)
            if allowed_tool_names is not None and name not in allowed_tool_names:
                continue
            out.append({"id": f"yaml_leak_{len(out)}", "name": name, "arguments": args})
            break

    return out


def _should_retry_chat_without_tools(exc: BaseException) -> bool:
    """
    Only strip ``tools`` and retry when the server clearly rejects tool calling,
    not on generic 400s (e.g. bad tool schema) — those must surface to the caller.
    """
    msg = str(exc).lower()
    needles = (
        "does not support tools",
        "tools are not supported",
        "tool calling is not supported",
        "unknown parameter: tools",
        "unexpected keyword argument 'tools'",
        "invalid value for 'tool_choice'",
    )
    if any(n in msg for n in needles):
        return True
    # OpenAI SDK APIStatusError / HTTPStatusError
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 501:
        return True
    return False


def _normalize_messages_for_vllm(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Coalesce consecutive ``system`` messages into a single one.

    Qwen3.5's ``chat_template.jinja`` enforces ``loop.first`` for any system
    message and otherwise raises ``"System message must be at the beginning."``
    The orchestrator legitimately builds multiple system blocks (main prompt,
    file registry, agent prompt, file context, inline text, etc.) — we merge
    them here so callers don't have to think about the underlying model's
    template constraints.

    Behaviour:
      * The leading run of ``system`` messages is merged into a single message
        whose content is joined with a blank line.
      * Any later ``system`` messages are folded into the leading one too —
        otherwise the template would reject them mid-conversation.
      * Non-string content (e.g. multimodal content-parts) is preserved as-is
        but only the first such item keeps its structure; subsequent system
        messages with structured content are stringified to keep the merge
        safe.
    """
    if not messages:
        return messages

    def _to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        chunks.append(str(part.get("text", "")))
                    else:
                        # Non-text content part — drop in system context.
                        continue
                else:
                    chunks.append(str(part))
            return "\n".join(c for c in chunks if c)
        return str(content) if content is not None else ""

    system_chunks: List[str] = []
    rest: List[Dict[str, Any]] = []
    seen_non_system = False

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            text = _to_text(msg.get("content"))
            if text:
                system_chunks.append(text)
            # Drop the system entry — we'll re-emit a single merged one.
        else:
            # ``_``-prefixed keys are host-private markers, never sent upstream.
            rest.append({k: v for k, v in msg.items() if not str(k).startswith("_")})
            seen_non_system = True

    if not system_chunks:
        return list(rest)

    merged_system = {"role": "system", "content": "\n\n".join(system_chunks)}
    if len(messages) != len(rest) + 1:
        log.debug(
            "Merged %d system messages into one (had %d total messages)",
            len(messages) - len(rest),
            len(messages),
        )
    return [merged_system, *rest]


def _merge_vllm_extra_body(extra_body: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge vLLM ``extra_body`` with default ``chat_template_kwargs`` for Qwen3.

    ``enable_thinking`` defaults to ``settings.LLM_ENABLE_THINKING`` so vLLM can
    return chain-of-thought in the ``reasoning`` field when using
    ``--reasoning-parser qwen3``. Callers may set ``chat_template_kwargs`` in
    ``extra_body`` to override per request (e.g. summarizers pass
    ``enable_thinking: false``).

    With ``--tool-call-parser qwen3_coder``, tool calls stay structured while
    thinking is streamed separately; the old Hermes-only workaround (force-disable
    thinking whenever ``tools`` is set) is no longer applied.
    """
    merged = dict(extra_body or {})
    ctk = dict(merged.get("chat_template_kwargs") or {})
    if "enable_thinking" not in ctk:
        ctk["enable_thinking"] = settings.LLM_ENABLE_THINKING
    merged["chat_template_kwargs"] = ctk
    return merged


def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        value = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


class GatewayLLMClient:
    """
    کلاینت async برای ارتباط با vLLM از طریق OpenAI-compatible API.
    - پشتیبانی از streaming
    - retry خودکار
    - مدیریت timeout
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        timeout: int = 120,
        max_tokens: Optional[int] = None,
        purpose: str = "chat",
        service: str = "orchestrator",
    ):
        config = get_llm_config()
        self.base_url = base_url or config["base_url"]
        self.api_key = api_key or config["api_key"]
        # Optional pin (tests); production uses binding_client on each call.
        self._static_model = (model or "").strip()
        self.model = self._static_model
        self.purpose = purpose
        self.service = service
        self.temperature = temperature
        self.timeout = timeout
        cfg_mt = config.get("max_tokens")
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else (int(cfg_mt) if cfg_mt is not None and int(cfg_mt) > 0 else None)
        )

        # OpenAI async client
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=int(config["max_retries"]),
        )

        # Lazy import — avoid circular dep with binding_client.
        from .binding_client import get_binding_client
        self._binding = get_binding_client(self.service)

        log.info(
            "GatewayLLMClient initialized: service=%s purpose=%s static_model=%s base_url=%s",
            self.service, self.purpose, self._static_model, self.base_url,
        )

    async def create_completion(self, **params: Any) -> Any:
        """``chat.completions.create`` that waits out gateway rate limits (429).

        The SDK's own short retries run first; after that we back off
        (``Retry-After`` when sent, else exponential) within a total budget.
        """
        waited = 0.0
        attempt = 0
        while True:
            try:
                return await self._client.chat.completions.create(**params)
            except RateLimitError as exc:
                delay = _retry_after_seconds(exc)
                if delay is None:
                    delay = min(
                        float(settings.LLM_RATE_LIMIT_MAX_BACKOFF_SEC),
                        float(settings.LLM_RATE_LIMIT_BACKOFF_SEC) * (2 ** attempt),
                    )
                if waited + delay > float(settings.LLM_RATE_LIMIT_MAX_WAIT_SEC):
                    raise
                log.warning(
                    "LLM gateway rate limited — waiting %.1fs (attempt %d, waited %.1fs)",
                    delay, attempt + 1, waited,
                )
                await asyncio.sleep(delay)
                waited += delay
                attempt += 1

    async def _resolve_model(self) -> str:
        """Look up the currently bound model from llm-gateway (cached)."""
        try:
            self.model = await self._binding.get_model(self.purpose)
        except Exception as exc:  # noqa: BLE001 — never block a chat on binding lookup
            log.debug("binding lookup failed: %s", exc)
            self.model = self._static_model
        if not (self.model or "").strip() and self._static_model:
            self.model = self._static_model
        return self.model
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        فراخوانی chat completion با پشتیبانی از tool calling.
        
        Args:
            messages: لیست پیام‌ها
            temperature: دمای تولید
            max_tokens: حداکثر توکن‌های خروجی
            tools: لیست ابزارهای در دسترس (OpenAI format)
            tool_choice: نحوه انتخاب ابزار (auto, none, required)
            extra_body: پارامترهای اضافی vLLM (مثل chat_template_kwargs)
        
        Returns:
            Dict with 'ok', 'content', 'tool_calls', 'raw'
        """
        try:
            # Refresh logical model binding before each call so admin-side
            # rebinds take effect within the binding-client TTL (~30s).
            current_model = await self._resolve_model()
            if not (current_model or "").strip():
                return {
                    "ok": False,
                    "error": "chat_model_unavailable",
                    "content": "",
                    "tool_calls": [],
                    "raw": None,
                }
            # Qwen3.5's chat template only allows a *single* system message at
            # the very beginning. We coalesce here so callers can compose
            # prompts naturally.
            messages = _normalize_messages_for_vllm(messages)

            params = _params_with_max_tokens(
                {
                    "model": current_model,
                    "messages": messages,
                    "temperature": temperature if temperature is not None else self.temperature,
                },
                max_tokens,
                self.max_tokens,
            )

            params["extra_body"] = _merge_vllm_extra_body(extra_body)
            
            native_tool_calling = False
            
            if tools:
                params["tools"] = tools
                if tool_choice:
                    params["tool_choice"] = tool_choice
            
            log.debug(f"📡 Calling vLLM: model={self.model}")
            
            try:
                completion = await self.create_completion(**params)
                # اگه tools بود و موفق شد، native tool calling بود
                if tools:
                    native_tool_calling = True
            except Exception as e:
                # فقط وقتی سرور صریحاً tool calling را رد کرد، بدون tools تکرار کن
                if tools and _should_retry_chat_without_tools(e):
                    log.warning(
                        "Tool calling rejected by upstream — retrying without tools: %s",
                        e,
                    )
                    params.pop("tools", None)
                    params.pop("tool_choice", None)
                    completion = await self.create_completion(**params)
                    native_tool_calling = False  # صریحاً False کن
                else:
                    raise
            
            # استخراج محتوا و tool_calls
            choice = completion.choices[0]
            message = choice.message
            content = message.content or ""
            tool_calls = []
            
            # ✨ Qwen3 reasoning/thinking (thinking mode + --reasoning-parser qwen3)
            reasoning_content = ""
            # vLLM exposes assistant reasoning on ``message.reasoning`` (OpenAI compat extension)
            if hasattr(message, 'reasoning') and message.reasoning:
                reasoning_content = message.reasoning
            elif hasattr(message, 'reasoning_content') and message.reasoning_content:
                reasoning_content = message.reasoning_content
            
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {}
                    })

            if not tool_calls and content:
                allowed = _openai_tool_names_from_schema(tools)
                recovered = _try_parse_qwen_xml_tool_calls(content)
                if not recovered:
                    recovered = _try_parse_json_fence_tool_calls(content, allowed)
                if not recovered:
                    recovered = _try_parse_tool_call_markdown_fence(content, allowed)
                if not recovered:
                    recovered = _try_parse_yaml_tool_calls_leak(content, allowed)
                if recovered:
                    tool_calls = recovered
                    log.info(
                        "Recovered %d tool call(s) from non-stream message content (xml / json / tool_call fence / yaml leak)",
                        len(tool_calls),
                    )
            
            # استخراج token usage
            tokens_prompt = None
            tokens_completion = None
            if hasattr(completion, 'usage') and completion.usage:
                tokens_prompt = getattr(completion.usage, 'prompt_tokens', None)
                tokens_completion = getattr(completion.usage, 'completion_tokens', None)
            
            log.debug(f"✅ vLLM response: {len(content)} chars, {len(tool_calls)} tool_calls, native={native_tool_calling}, tokens={tokens_prompt}/{tokens_completion}")
            
            return {
                "ok": True,
                "content": content,
                "tool_calls": tool_calls,
                "reasoning_content": reasoning_content,  # ✨ اضافه شد برای Qwen3
                "finish_reason": choice.finish_reason,
                "raw": completion,
                "native_tool_calling": native_tool_calling,
                "tokens_prompt": tokens_prompt,
                "tokens_completion": tokens_completion,
            }
            
        except Exception as exc:
            log.error(f"🔴 vLLM call failed: {exc}", exc_info=True)
            return {
                "ok": False,
                "error": f"vllm_error: {type(exc).__name__}: {str(exc)}",
                "content": "",
                "tool_calls": [],
                "raw": None,
            }
    
    async def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming chat completion با پشتیبانی از vLLM reasoning و <think> tags.
        
        Yields dicts with:
        - type: "thinking" | "thinking_start" | "thinking_end" | "answer" | "answer_start" | "done" | "error"
        - content: محتوا
        
        ✨ پشتیبانی از دو حالت:
        1. vLLM native reasoning (delta.reasoning) — ``--reasoning-parser qwen3`` / deepseek_r1
        2. <think> tags fallback — وقتی پارسر reasoning رو خاموش باشد
        """
        import re
        
        try:
            current_model = await self._resolve_model()
            if not (current_model or "").strip():
                yield {"type": "error", "content": "chat_model_unavailable"}
                return
            messages = _normalize_messages_for_vllm(messages)
            params = _params_with_max_tokens(
                {
                    "model": current_model,
                    "messages": messages,
                    "temperature": temperature if temperature is not None else self.temperature,
                    "stream": True,
                },
                max_tokens,
                self.max_tokens,
            )
            params["extra_body"] = _merge_vllm_extra_body(extra_body)
            stream = await self.create_completion(**params)
            
            buffer = ""
            in_thinking = False
            thinking_started = False
            thinking_ended = False
            in_tool_call = False
            answer_started = False
            last_finish_reason = None

            async for chunk in stream:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                fr = getattr(choice, "finish_reason", None)
                if fr:
                    last_finish_reason = fr

                delta = choice.delta
                
                # ✨ vLLM native reasoning (--reasoning-parser qwen3 / deepseek_r1 / …)
                if hasattr(delta, 'reasoning') and delta.reasoning:
                    if not thinking_started:
                        thinking_started = True
                        in_thinking = True
                        yield {"type": "thinking_start"}
                    yield {"type": "thinking", "content": delta.reasoning}
                    continue
                
                # ✨ Content handling
                if delta.content:
                    content = delta.content
                    
                    # وقتی reasoning بود و الان content شروع شد، یعنی thinking تموم شد
                    if thinking_started and in_thinking:
                        in_thinking = False
                        thinking_ended = True
                        yield {"type": "thinking_end"}
                        yield {"type": "answer_start"}
                        answer_started = True
                    
                    buffer += content
                    
                    # ✨ فیلتر کردن <tool_call> tags
                    if "<tool_call>" in buffer:
                        in_tool_call = True
                        idx = buffer.find("<tool_call>")
                        pre_tool = buffer[:idx]
                        buffer = buffer[idx:]
                        if pre_tool.strip() and (thinking_ended or answer_started):
                            yield {"type": "answer", "content": pre_tool}
                    
                    if in_tool_call:
                        if "</tool_call>" in buffer:
                            idx = buffer.find("</tool_call>")
                            buffer = buffer[idx + 13:]
                            in_tool_call = False
                        continue
                    
                    # ✨ Fallback: تشخیص <think> tags (برای مدل‌هایی که reasoning parser ندارن)
                    if not thinking_started and "<think>" in buffer:
                        thinking_started = True
                        in_thinking = True
                        idx = buffer.find("<think>")
                        buffer = buffer[idx + 7:]
                        yield {"type": "thinking_start"}
                        if buffer.strip():
                            yield {"type": "thinking", "content": buffer}
                        buffer = ""
                        continue
                    
                    # ✨ Fallback: تشخیص پایان </think> tags
                    if in_thinking and "</think>" in buffer:
                        thinking_ended = True
                        in_thinking = False
                        # ارسال باقیمانده thinking (قبل از </think>)
                        idx = buffer.find("</think>")
                        remaining_thinking = buffer[:idx]
                        if remaining_thinking.strip():
                            yield {"type": "thinking", "content": remaining_thinking}
                        # آنچه بعد از </think> هست
                        buffer = buffer[idx + 8:].lstrip()
                        yield {"type": "thinking_end"}
                        yield {"type": "answer_start"}
                        # ارسال شروع پاسخ اگه چیزی مونده
                        if buffer.strip():
                            yield {"type": "answer", "content": buffer}
                        buffer = ""
                        continue
                    
                    # ✨ اگه در thinking mode هستیم (با <think> tags)
                    if in_thinking:
                        yield {"type": "thinking", "content": content}
                        buffer = ""
                    # اگه thinking تموم شده یا answer شروع شده
                    elif thinking_ended or answer_started:
                        if not answer_started:
                            yield {"type": "answer_start"}
                            answer_started = True
                        yield {"type": "answer", "content": content}
                        buffer = ""
                    # اگه هنوز شروع نشده، صبر کن ببینیم <think> یا reasoning میاد یا نه
                    elif not thinking_started and len(buffer) > 50 and "<think>" not in buffer:
                        # بدون thinking شروع شده (مثلاً reasoning parser غیرفعاله)
                        yield {"type": "answer_start"}
                        answer_started = True
                        yield {"type": "answer", "content": buffer}
                        buffer = ""
                        thinking_ended = True
            
            # ✨ Handle end of stream
            # اگه reasoning بود ولی content نیومد، thinking_end بزن
            if thinking_started and in_thinking:
                yield {"type": "thinking_end"}
                yield {"type": "answer_start"}
            
            # اگه چیزی توی buffer مونده
            if buffer.strip():
                # فیلتر کردن <tool_call> tags که نباید به کاربر نشون داده بشه
                import re
                buffer = re.sub(r'<tool_call>.*?</tool_call>', '', buffer, flags=re.DOTALL)
                buffer = re.sub(r'<tool_call>.*', '', buffer, flags=re.DOTALL)  # برای حالتی که tag بسته نشده
                
                if buffer.strip():
                    if in_thinking:
                        yield {"type": "thinking", "content": buffer}
                    elif not thinking_ended:
                        # شاید هیچ tag ای نبود
                        yield {"type": "answer_start"}
                        yield {"type": "answer", "content": buffer}
                    else:
                        yield {"type": "answer", "content": buffer}
            
            done_ev: Dict[str, Any] = {"type": "done"}
            if last_finish_reason:
                done_ev["finish_reason"] = last_finish_reason
            yield done_ev
                    
        except Exception as exc:
            log.error(f"🔴 vLLM stream failed: {exc}")
            yield {"type": "error", "content": str(exc)}
    
    async def chat_stream_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming chat completion with tool calling support.
        
        Yields:
        - type: "thinking" | "answer" | "tool_calls" | "done" | "error"
        - content: محتوا
        - tool_calls: لیست tool calls (فقط برای type=tool_calls)
        """
        import re
        
        try:
            current_model = await self._resolve_model()
            if not (current_model or "").strip():
                yield {"type": "error", "content": "chat_model_unavailable"}
                return
            messages = _normalize_messages_for_vllm(messages)
            extra_body = kwargs.pop("extra_body", None)
            params = _params_with_max_tokens(
                {
                    "model": current_model,
                    "messages": messages,
                    "temperature": temperature if temperature is not None else self.temperature,
                    "stream": True,
                    "extra_body": _merge_vllm_extra_body(extra_body),
                },
                max_tokens,
                self.max_tokens,
            )
            
            if tools:
                params["tools"] = tools
                params["tool_choice"] = tool_choice

            # #region agent log
            try:
                _roles = [m.get("role") for m in messages if isinstance(m, dict)]
                _last_user = next(
                    (
                        (m.get("content") or "")[:180]
                        for m in reversed(messages)
                        if isinstance(m, dict)
                        and m.get("role") == "user"
                        and isinstance(m.get("content"), str)
                    ),
                    "",
                )
                debug_ndjson(
                    "H9+H10",
                    "gateway_llm_client.py:chat_stream_with_tools:pre_request",
                    "pre_request_shape",
                    {
                        "n_messages": len(messages),
                        "n_system_messages": sum(1 for r in _roles if r == "system"),
                        "first_roles": _roles[:8],
                        "last_roles": _roles[-8:],
                        "n_tools_param": len(tools or []),
                        "tool_choice": tool_choice,
                        "last_user_head": _last_user,
                    },
                )
                # #region agent log — H12: prove tool_call serialization
                _tc_dump: List[Dict[str, Any]] = []
                _tool_dump: List[Dict[str, Any]] = []
                for _idx_m, _mm in enumerate(messages):
                    if not isinstance(_mm, dict):
                        continue
                    _r = _mm.get("role")
                    if _r == "assistant":
                        _tcs = _mm.get("tool_calls") or []
                        for _tc in _tcs:
                            _fn = (_tc.get("function") if isinstance(_tc, dict) else None) or {}
                            _args_str = _fn.get("arguments")
                            _tc_dump.append(
                                {
                                    "msg_idx": _idx_m,
                                    "tc_id": _tc.get("id") if isinstance(_tc, dict) else None,
                                    "name": _fn.get("name"),
                                    "args_len": len(_args_str) if isinstance(_args_str, str) else -1,
                                    "args_head": (_args_str or "")[:120] if isinstance(_args_str, str) else str(_args_str)[:120],
                                    "args_is_empty_obj": _args_str in ("{}", "", None),
                                }
                            )
                    elif _r == "tool":
                        _c = _mm.get("content") or ""
                        _tool_dump.append(
                            {
                                "msg_idx": _idx_m,
                                "tool_call_id": _mm.get("tool_call_id"),
                                "content_len": len(str(_c)),
                                "content_head": str(_c)[:120],
                            }
                        )
                debug_ndjson(
                    "H12",
                    "gateway_llm_client.py:chat_stream_with_tools:tool_wire_shape",
                    "tool_wire_shape",
                    {
                        "n_assistant_tool_calls": len(_tc_dump),
                        "n_tool_messages": len(_tool_dump),
                        "assistant_tool_calls": _tc_dump[:8],
                        "tool_messages": _tool_dump[:8],
                    },
                )
                # #endregion
            except Exception:
                pass
            # #endregion
            
            stream = await self.create_completion(**params)
            
            buffer = ""
            in_thinking = False
            thinking_started = False
            thinking_ended = False
            collected_tool_calls = []
            tool_call_map = {}  # id -> {"name": ..., "arguments": ...}
            # #region agent log
            _chunks_tool = 0
            _chunks_content = 0
            _chunks_reasoning = 0
            _last_finish: Optional[str] = None
            # #endregion
            latest_message_tool_calls: List[Dict[str, Any]] = []
            raw_stream_text: str = ""

            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice0 = chunk.choices[0]
                fr = getattr(choice0, "finish_reason", None)
                if fr:
                    _last_finish = fr  # type: ignore[misc]

                msg = getattr(choice0, "message", None)
                if msg:
                    recovered = _extract_tool_calls_from_choice_message(msg)
                    if recovered:
                        latest_message_tool_calls = recovered

                delta = getattr(choice0, "delta", None)
                if delta is None:
                    continue

                # ✨ vLLM --reasoning-parser qwen3: reasoning streams before content/tool_calls
                if hasattr(delta, "reasoning") and delta.reasoning:
                    _chunks_reasoning += 1  # type: ignore[misc]
                    if not thinking_started:
                        thinking_started = True
                        in_thinking = True
                        yield {"type": "thinking_start"}
                    yield {"type": "thinking", "content": delta.reasoning}
                    continue

                # ✨ Handle tool_calls delta — stream deltas as they arrive
                if delta.tool_calls:
                    _chunks_tool += 1  # type: ignore[misc]
                    if in_thinking:
                        in_thinking = False
                        thinking_ended = True
                        yield {"type": "thinking_end"}
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        is_new = idx not in tool_call_map
                        if is_new:
                            tool_call_map[idx] = {
                                "id": tc_delta.id or f"call_{idx}",
                                "name": "",
                                "arguments": ""
                            }
                        if tc_delta.id:
                            tool_call_map[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                old_name = tool_call_map[idx]["name"]
                                tool_call_map[idx]["name"] += tc_delta.function.name
                                # Emit tool_call_start when we first see the tool name
                                if not old_name and tool_call_map[idx]["name"]:
                                    yield {
                                        "type": "tool_call_start",
                                        "index": idx,
                                        "tool_id": tool_call_map[idx]["id"],
                                        "tool_name": tool_call_map[idx]["name"],
                                    }
                            if tc_delta.function.arguments:
                                tool_call_map[idx]["arguments"] += tc_delta.function.arguments
                                # Emit incremental argument delta
                                yield {
                                    "type": "tool_call_delta",
                                    "index": idx,
                                    "tool_id": tool_call_map[idx]["id"],
                                    "tool_name": tool_call_map[idx]["name"],
                                    "arguments_delta": tc_delta.function.arguments,
                                }
                
                # ✨ Handle content delta
                if delta.content:
                    _chunks_content += 1  # type: ignore[misc]
                    if in_thinking:
                        in_thinking = False
                        thinking_ended = True
                        yield {"type": "thinking_end"}
                    content = delta.content
                    raw_stream_text += content
                    buffer += content
                    # #region agent log
                    if _chunks_content <= 3:
                        debug_ndjson(
                            "H9",
                            "gateway_llm_client.py:chat_stream_with_tools:chunk",
                            "raw_content_chunk",
                            {
                                "chunk_index": _chunks_content,
                                "len": len(content),
                                "repr": repr(content[:300]),
                            },
                        )
                    # #endregion
                    
                    # تشخیص شروع <think>
                    if not thinking_started and "<think>" in buffer:
                        thinking_started = True
                        in_thinking = True
                        idx = buffer.find("<think>")
                        buffer = buffer[idx + 7:]
                        yield {"type": "thinking_start"}
                        if buffer.strip():
                            yield {"type": "thinking", "content": buffer}
                        buffer = ""
                        continue
                    
                    # تشخیص پایان </think>
                    if in_thinking and "</think>" in buffer:
                        thinking_ended = True
                        in_thinking = False
                        idx = buffer.find("</think>")
                        remaining_thinking = buffer[:idx]
                        if remaining_thinking.strip():
                            yield {"type": "thinking", "content": remaining_thinking}
                        buffer = buffer[idx + 8:].lstrip()
                        yield {"type": "thinking_end"}
                        if buffer.strip():
                            yield {"type": "answer", "content": buffer}
                        buffer = ""
                        continue
                    
                    # اگه در thinking هستیم - ارسال فوری
                    if in_thinking:
                        yield {"type": "thinking", "content": content}
                        buffer = ""
                    elif thinking_ended:
                        yield {"type": "answer", "content": content}
                        buffer = ""
                    elif not thinking_started and len(buffer) > 50:
                        # بدون thinking
                        yield {"type": "answer", "content": buffer}
                        buffer = ""
                        thinking_ended = True
            
            if thinking_started and in_thinking:
                yield {"type": "thinking_end"}

            # Recover tool calls delivered only on ``choice.message`` (no deltas),
            # or embedded as ``<tool_call>`` / markdown `` ```json `` (model/template drift).
            #
            # IMPORTANT: when the caller did not pass ``tools``, the model
            # MUST NOT be able to "summon" a tool by writing pseudo-JSON in
            # its answer. We use ``tools is None`` (no schema given) as the
            # signal to disable text-based recovery entirely — this is the
            # right behaviour for the "forced final synthesis" path.
            tool_recover_source: Optional[str] = None
            combined_text = raw_stream_text + buffer
            recovery_enabled = bool(tools)

            if not tool_call_map and latest_message_tool_calls:
                for idx, tc in enumerate(latest_message_tool_calls):
                    tool_call_map[idx] = {
                        "id": tc.get("id") or f"call_{idx}",
                        "name": tc.get("name") or "",
                        "arguments": json.dumps(tc.get("arguments") or {}, ensure_ascii=False),
                    }
                tool_recover_source = "choice_message"
                log.info(
                    "Recovered %d tool call(s) from stream message.tool_calls (delta path was empty)",
                    len(tool_call_map),
                )
            if recovery_enabled and not tool_call_map:
                xml_calls = _try_parse_qwen_xml_tool_calls(combined_text)
                if xml_calls:
                    for idx, tc in enumerate(xml_calls):
                        tool_call_map[idx] = {
                            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                            "name": tc.get("name") or "",
                            "arguments": json.dumps(tc.get("arguments") or {}, ensure_ascii=False),
                        }
                    tool_recover_source = "xml_tool_call"
                    log.info(
                        "Recovered %d tool call(s) from <tool_call> XML in stream content",
                        len(tool_call_map),
                    )
            if recovery_enabled and not tool_call_map:
                allowed_names = _openai_tool_names_from_schema(tools)
                fence_calls = _try_parse_json_fence_tool_calls(combined_text, allowed_names)
                if fence_calls:
                    for idx, tc in enumerate(fence_calls):
                        tool_call_map[idx] = {
                            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                            "name": tc.get("name") or "",
                            "arguments": json.dumps(tc.get("arguments") or {}, ensure_ascii=False),
                        }
                    tool_recover_source = "json_fence"
                    log.info(
                        "Recovered %d tool call(s) from ```json markdown fences in stream content",
                        len(tool_call_map),
                    )
            if recovery_enabled and not tool_call_map:
                allowed_names = _openai_tool_names_from_schema(tools)
                md_tc = _try_parse_tool_call_markdown_fence(combined_text, allowed_names)
                if md_tc:
                    for idx, tc in enumerate(md_tc):
                        tool_call_map[idx] = {
                            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                            "name": tc.get("name") or "",
                            "arguments": json.dumps(tc.get("arguments") or {}, ensure_ascii=False),
                        }
                    tool_recover_source = "tool_call_md_fence"
                    log.info(
                        "Recovered %d tool call(s) from ```tool_call markdown fences",
                        len(tool_call_map),
                    )
            if recovery_enabled and not tool_call_map:
                allowed_names = _openai_tool_names_from_schema(tools)
                bare = _try_parse_bare_json_tool_calls(combined_text, allowed_names)
                if bare:
                    for idx, tc in enumerate(bare):
                        tool_call_map[idx] = {
                            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                            "name": tc.get("name") or "",
                            "arguments": json.dumps(tc.get("arguments") or {}, ensure_ascii=False),
                        }
                    tool_recover_source = "bare_json_in_content"
                    log.info(
                        "Recovered %d tool call(s) from bare JSON in stream content",
                        len(tool_call_map),
                    )
            if recovery_enabled and not tool_call_map:
                allowed_names = _openai_tool_names_from_schema(tools)
                yaml_calls = _try_parse_yaml_tool_calls_leak(combined_text, allowed_names)
                if yaml_calls:
                    for idx, tc in enumerate(yaml_calls):
                        tool_call_map[idx] = {
                            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                            "name": tc.get("name") or "",
                            "arguments": json.dumps(tc.get("arguments") or {}, ensure_ascii=False),
                        }
                    tool_recover_source = "yaml_tool_calls_leak"
                    log.info(
                        "Recovered %d tool call(s) from pseudo-YAML tool_calls: blocks",
                        len(tool_call_map),
                    )
            
            # #region agent log
            debug_ndjson(
                "H1",
                "gateway_llm_client.py:chat_stream_with_tools",
                "stream_aggregate",
                {
                    "n_tools_param": len(tools or []),
                    "tool_choice": tool_choice,
                    "chunks_reasoning": _chunks_reasoning,
                    "chunks_with_tool_calls": _chunks_tool,
                    "chunks_with_content": _chunks_content,
                    "tool_call_map_size": len(tool_call_map),
                    "tool_names_partial": [tool_call_map[k].get("name") for k in sorted(tool_call_map.keys())],
                    "last_finish_reason": _last_finish,
                    "message_tool_calls_recovered": len(latest_message_tool_calls),
                    "tool_recover_source": tool_recover_source,
                    "expected_upstream_tool_parser": settings.LLM_TOOL_CALL_PARSER,
                    "raw_stream_text_len": len(raw_stream_text),
                    "raw_stream_text_head": raw_stream_text[:500],
                    "buffer_len": len(buffer),
                    "buffer_head": buffer[:500],
                },
            )
            # #endregion

            # Parse tool_calls
            if tool_call_map:
                for idx in sorted(tool_call_map.keys()):
                    tc = tool_call_map[idx]
                    try:
                        args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    collected_tool_calls.append({
                        "id": tc["id"],
                        "name": tc["name"],
                        "arguments": args
                    })
                
                # ✨ Emit tool_calls event
                yield {"type": "tool_calls", "tool_calls": collected_tool_calls}
            
            # Always flush remaining buffer — do not abort chat streams for
            # repetition; user Stop / natural finish_reason ends generation.
            if buffer.strip():
                buffer = re.sub(r'<tool_call>.*?</tool_call>', '', buffer, flags=re.DOTALL)
                buffer = re.sub(r'<tool_call>.*', '', buffer, flags=re.DOTALL)
                if buffer.strip():
                    if in_thinking:
                        yield {"type": "thinking", "content": buffer}
                    else:
                        yield {"type": "answer", "content": buffer}
            
            yield {"type": "done"}
                    
        except Exception as exc:
            log.error(f"🔴 vLLM stream with tools failed: {exc}")
            # #region agent log
            debug_ndjson(
                "H3",
                "gateway_llm_client.py:chat_stream_with_tools",
                "stream_exception",
                {"error_type": type(exc).__name__, "error_msg": str(exc)[:500]},
            )
            # #endregion
            yield {"type": "error", "content": str(exc)}
    
    async def health_check(self) -> bool:
        """بررسی سلامت سرویس vLLM / llm-gateway.

        Avoid ``GET /v1/models`` — that fans out to every backend and can hang.
        Prefer a cheap authenticated catalog/health probe instead.
        """
        import httpx

        probe_timeout_s = float(__import__("os").getenv("ORCH_LLM_HEALTH_TIMEOUT_S", "3"))
        root = (self.base_url or "").rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        key = (self.api_key or "").strip()
        headers = {}
        if key and key.lower() not in ("dummy-key", "not-needed", "none"):
            headers["Authorization"] = f"Bearer {key}"
            headers["X-Internal-Key"] = key
        try:
            async with httpx.AsyncClient(timeout=probe_timeout_s) as client:
                # Catalog is the chat/search source of truth and is auth-gated.
                resp = await client.get(f"{root}/catalog", headers=headers or None)
                if resp.status_code == 200:
                    body = resp.json() or {}
                    data = body.get("data") if isinstance(body, dict) else body
                    return isinstance(data, list) and len(data) > 0
                resp = await client.get(f"{root}/health", headers=None)
                return resp.status_code == 200
        except Exception:
            return False


# Global client instance
_gateway_llm_client: Optional[GatewayLLMClient] = None


def get_gateway_llm_client() -> GatewayLLMClient:
    """Get or create the global vLLM client."""
    global _gateway_llm_client
    if _gateway_llm_client is None:
        _gateway_llm_client = GatewayLLMClient()
    return _gateway_llm_client


async def call_llm(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for LLM calls."""
    client = get_gateway_llm_client()
    return await client.chat(messages, tools=tools, **kwargs)
