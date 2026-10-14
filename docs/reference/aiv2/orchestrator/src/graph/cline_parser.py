"""
Streaming XML parser for the Cline-style tool-calling protocol.

The orchestrator feeds raw LLM-content chunks into ``ClineXmlStreamParser.feed``
as they arrive from vLLM. The parser emits a strict, monotone tape of events:

* ``text_delta``   — prose to forward to the UI as ``answer`` chunks.
* ``thinking_delta`` — content inside ``<thinking>...</thinking>``, surfaced
  to the UI as reasoning (not part of the chat history sent back to the model).
* ``tool_call``    — fully-buffered tool invocation: name + parsed argument
  dict + the original raw XML text (so we can persist it in chat history).
* ``attempt_completion`` — terminal event carrying the ``<result>`` body.
* ``parse_error``  — malformed input (unknown tag, missing close, etc.).
  We surface the literal text to the UI so the user is never silently dropped.

Design rules (copied from Cline's parser):
1. **Greedy, streaming**: every ``feed(chunk)`` may yield zero or more events;
   the caller does NOT have to wait for ``flush``. This keeps the UI live.
2. **Single-tool atomicity**: after ``tool_call`` (or ``attempt_completion``)
   is emitted, the parser switches to ``done`` state and silently swallows
   any trailing tokens. The model is supposed to stop generating; if it does
   not, we ignore the noise rather than executing a second tool.
3. **No regex**: we walk the buffer with explicit ``find`` calls. Streamed
   XML can split a tag across chunks ("<gra" + "ph_search>"), so we only
   commit text when we are sure no further ``<`` is pending.
4. **Tag whitelist**: the parser is constructed with the set of legal tool
   names; an opening tag for an unknown name is treated as text. This is
   what prevents the model from accidentally invoking a tool that does not
   exist in this turn's catalog.
"""
from __future__ import annotations

import json as _json
import re as _re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple


# Tag names follow the same identifier rules as Python attribute names plus
# a dot/period (to support names like ``image_meta`` and
# ``plugin_graph_explorer_graph_search``).
_TAG_NAME_RE = _re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]*")


@dataclass
class ParserEvent:
    """One unit of work emitted by the parser."""

    kind: str  # "text_delta" | "thinking_delta" | "tool_call" | "attempt_completion" | "parse_error"
    text: str = ""  # used by text_delta / thinking_delta / parse_error
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw_xml: str = ""  # original text of a tool_call (for chat-history persistence)
    result: str = ""  # body of <result> inside an attempt_completion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_param_body(body: str, declared_type: str) -> Any:
    """Convert the inner text of a single ``<param>...</param>`` into a value.

    The model is told to pass JSON for arrays/objects and verbatim text
    otherwise. We try to honor that, but we are resilient to either side
    of the contract being wrong:

    * ``"[\"a\", \"b\"]"`` → list (preferred).
    * ``"a, b, c"`` for an ``array``-typed param → split on comma.
    * Trim wrapping whitespace and a single set of CDATA brackets just in
      case the model emits ``<param><![CDATA[...]]></param>``.
    """
    text = body.strip()
    if text.startswith("<![CDATA[") and text.endswith("]]>"):
        text = text[len("<![CDATA[") : -len("]]>")]
        text = text.strip()

    if declared_type in ("array", "list"):
        if text.startswith("[") and text.endswith("]"):
            try:
                v = _json.loads(text)
                if isinstance(v, list):
                    return v
            except Exception:
                pass
        # Fallback: comma-separated, single-quoted, etc.
        if "," in text:
            return [p.strip().strip('"').strip("'") for p in text.split(",") if p.strip()]
        return [text] if text else []

    if declared_type in ("object", "dict", "json"):
        if text.startswith("{") and text.endswith("}"):
            try:
                v = _json.loads(text)
                if isinstance(v, dict):
                    return v
            except Exception:
                pass
        return text

    if declared_type == "integer":
        try:
            return int(text)
        except Exception:
            return text
    if declared_type == "number":
        try:
            return float(text)
        except Exception:
            return text
    if declared_type == "boolean":
        return text.lower() in ("true", "1", "yes", "y")

    # Default: opportunistic JSON for things that look like JSON literals,
    # otherwise the verbatim string.
    if (text.startswith("{") and text.endswith("}")) or (
        text.startswith("[") and text.endswith("]")
    ):
        try:
            return _json.loads(text)
        except Exception:
            return text
    return text


def _extract_params(
    inner_xml: str,
    param_types: Dict[str, str],
) -> Tuple[Dict[str, Any], List[str]]:
    """Pull every top-level ``<param>...</param>`` out of an XML body.

    The parser is intentionally naive — it walks for ``<name>`` then matches
    the same ``</name>`` later. This is what Cline does, and it works as
    long as parameter bodies do not themselves contain nested XML that
    happens to share a name. Returns ``(arguments_dict, errors)``.
    """
    args: Dict[str, Any] = {}
    errors: List[str] = []
    pos = 0
    while pos < len(inner_xml):
        lt = inner_xml.find("<", pos)
        if lt < 0:
            break
        # Skip XML comments / CDATA at this top level — they should not appear
        # but the model might output them.
        if inner_xml.startswith("<!--", lt):
            close = inner_xml.find("-->", lt)
            if close < 0:
                break
            pos = close + 3
            continue
        gt = inner_xml.find(">", lt + 1)
        if gt < 0:
            break
        tag_inner = inner_xml[lt + 1 : gt].strip()
        if tag_inner.startswith("/"):
            # stray closing tag — ignore and advance
            pos = gt + 1
            continue
        # Self-closing tag <param/> → empty string value.
        self_closing = tag_inner.endswith("/")
        if self_closing:
            tag_inner = tag_inner[:-1].strip()
        name_match = _TAG_NAME_RE.match(tag_inner.split()[0] if tag_inner else "")
        if not name_match:
            pos = gt + 1
            continue
        pname = name_match.group(0)
        if self_closing:
            args[pname] = ""
            pos = gt + 1
            continue
        close_tag = f"</{pname}>"
        close_pos = inner_xml.find(close_tag, gt + 1)
        if close_pos < 0:
            errors.append(f"unclosed_param:{pname}")
            break
        body = inner_xml[gt + 1 : close_pos]
        declared = param_types.get(pname, "string")
        try:
            args[pname] = _parse_param_body(body, declared)
        except Exception as exc:
            errors.append(f"param_parse_failed:{pname}:{exc}")
        pos = close_pos + len(close_tag)
    return args, errors


# ---------------------------------------------------------------------------
# Parser core
# ---------------------------------------------------------------------------


class ClineXmlStreamParser:
    """Incremental parser for Cline-style assistant streams.

    Usage::

        parser = ClineXmlStreamParser(
            tool_names={"graph_search", "session_files"},
            param_types_by_tool={"graph_search": {"queries": "array"}},
        )
        async for chunk in llm_stream:
            for ev in parser.feed(chunk):
                ...
        for ev in parser.flush():
            ...
    """

    # --- internal state machine ----------------------------------------
    _MODE_TEXT = "text"
    _MODE_THINKING = "thinking"
    _MODE_TOOL = "tool"
    _MODE_UNKNOWN_TAG = "unknown_tag"
    _MODE_DONE = "done"

    def __init__(
        self,
        tool_names: Iterable[str],
        param_types_by_tool: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        self._tool_names: Set[str] = {t for t in (tool_names or [])}
        self._param_types: Dict[str, Dict[str, str]] = dict(param_types_by_tool or {})

        self._buffer: str = ""
        self._mode: str = self._MODE_TEXT
        # In tool/thinking modes we hold inner XML until the close tag arrives.
        self._open_tag: str = ""  # tool name we're currently inside
        self._inner_buffer: str = ""

        # Diagnostics
        self.tool_events_emitted: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, chunk: str) -> List[ParserEvent]:
        """Push a streamed chunk and return any events ready to surface."""
        if not chunk:
            return []
        if self._mode == self._MODE_DONE:
            # Silently swallow trailing garbage so a runaway model can't
            # double-execute. (Same as Cline.)
            return []
        self._buffer += chunk
        out: List[ParserEvent] = []
        self._drain(out)
        return out

    def flush(self) -> List[ParserEvent]:
        """Finalize: emit whatever text or partial structure remains.

        Called once when the LLM stream ends. If we are still inside a
        ``<tool>`` block without a closer, we surface a ``parse_error``
        rather than silently dropping data.
        """
        if self._mode == self._MODE_DONE:
            return []
        out: List[ParserEvent] = []
        # One last drain to consume anything we deferred waiting for a
        # potential opening tag.
        self._drain(out, eof=True)

        if self._mode == self._MODE_TOOL:
            out.append(
                ParserEvent(
                    kind="parse_error",
                    text=f"unclosed_tool:{self._open_tag}",
                )
            )
            self._mode = self._MODE_DONE
        elif self._mode == self._MODE_THINKING:
            # Unclosed thinking is harmless — flush as thinking text.
            if self._inner_buffer:
                out.append(ParserEvent(kind="thinking_delta", text=self._inner_buffer))
                self._inner_buffer = ""
            self._mode = self._MODE_DONE
        else:
            # Flush any remaining text bytes verbatim.
            if self._buffer:
                out.append(ParserEvent(kind="text_delta", text=self._buffer))
                self._buffer = ""
            self._mode = self._MODE_DONE
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _drain(self, out: List[ParserEvent], *, eof: bool = False) -> None:
        """Walk through ``self._buffer`` once; called by feed/flush."""
        # The loop body must always either make progress (advance the
        # buffer / inner buffer) or break out of the loop, otherwise we
        # could spin forever on a malformed input.
        while True:
            if self._mode == self._MODE_TEXT:
                if not self._consume_text(out, eof):
                    return
            elif self._mode == self._MODE_THINKING:
                if not self._consume_thinking(out, eof):
                    return
            elif self._mode == self._MODE_TOOL:
                if not self._consume_tool(out, eof):
                    return
            elif self._mode == self._MODE_UNKNOWN_TAG:
                if not self._consume_unknown_tag(out, eof):
                    return
            else:
                return

    def _consume_text(self, out: List[ParserEvent], eof: bool) -> bool:
        """Inside plain prose mode. Look for the next ``<`` opener."""
        buf = self._buffer
        if not buf:
            return False

        lt = buf.find("<")
        if lt < 0:
            # No tags at all — safe to flush as user-visible text.
            out.append(ParserEvent(kind="text_delta", text=buf))
            self._buffer = ""
            return False

        # Emit everything before the ``<`` immediately.
        if lt > 0:
            out.append(ParserEvent(kind="text_delta", text=buf[:lt]))
            self._buffer = buf[lt:]
            buf = self._buffer

        # We are now at a ``<``. Check whether we have enough characters to
        # identify the tag.
        gt = buf.find(">")
        if gt < 0:
            if eof:
                # Truncated tag at EOF — emit as text so the user sees it.
                out.append(ParserEvent(kind="text_delta", text=buf))
                self._buffer = ""
                return False
            return False  # wait for more input

        tag_inner = buf[1:gt].strip()

        # Reject obvious non-tags (e.g. "<3", "<= ...") — emit ``<`` as text
        # and advance by one character.
        if tag_inner.startswith("/"):
            # stray closing tag in text mode — drop it as text
            out.append(ParserEvent(kind="text_delta", text=buf[: gt + 1]))
            self._buffer = buf[gt + 1 :]
            return True
        name_match = _TAG_NAME_RE.match(tag_inner.split()[0] if tag_inner else "")
        if not name_match:
            out.append(ParserEvent(kind="text_delta", text=buf[:1]))
            self._buffer = buf[1:]
            return True
        tag_name = name_match.group(0)

        # Recognized opener?
        if tag_name == "thinking":
            self._mode = self._MODE_THINKING
            self._open_tag = "thinking"
            self._inner_buffer = ""
            self._buffer = buf[gt + 1 :]
            return True
        if tag_name in self._tool_names:
            self._mode = self._MODE_TOOL
            self._open_tag = tag_name
            self._inner_buffer = ""
            self._buffer = buf[gt + 1 :]
            return True

        # Unknown XML-looking tag. Do not leak raw tool-shaped XML like
        # <vision_info>...</vision_info> to the user. Consume the block and
        # ask the model to retry using one of the whitelisted tools.
        self._mode = self._MODE_UNKNOWN_TAG
        self._open_tag = tag_name
        self._inner_buffer = ""
        self._buffer = buf[gt + 1 :]
        return True

    def _consume_thinking(self, out: List[ParserEvent], eof: bool) -> bool:
        buf = self._buffer
        close = "</thinking>"
        idx = buf.find(close)
        if idx < 0:
            # No close yet: stream what we have so far. To avoid leaking a
            # partial ``</thinking`` half-token, hold back the last 11 bytes
            # (``len('</thinking>') = 11``).
            keep = len(close)
            if eof:
                if buf:
                    out.append(ParserEvent(kind="thinking_delta", text=buf))
                    self._buffer = ""
                self._mode = self._MODE_TEXT  # treat EOF inside thinking as benign
                return False
            if len(buf) > keep:
                head = buf[:-keep]
                out.append(ParserEvent(kind="thinking_delta", text=head))
                self._buffer = buf[-keep:]
            return False

        # Got the closer.
        if idx > 0:
            out.append(ParserEvent(kind="thinking_delta", text=buf[:idx]))
        self._buffer = buf[idx + len(close) :]
        self._mode = self._MODE_TEXT
        self._open_tag = ""
        return True

    def _consume_tool(self, out: List[ParserEvent], eof: bool) -> bool:
        buf = self._buffer
        name = self._open_tag
        close = f"</{name}>"
        idx = buf.find(close)
        if idx < 0:
            # Stream the buffer into ``inner_buffer`` but keep the last
            # ``len(close)-1`` bytes back in ``_buffer`` — otherwise a
            # close tag split across two chunks (e.g. ``</gra`` + ``ph_search>``)
            # would never be detected.
            keep = max(0, len(close) - 1)
            if eof:
                self._inner_buffer += buf
                self._buffer = ""
                return False
            if len(buf) > keep:
                self._inner_buffer += buf[:-keep] if keep else buf
                self._buffer = buf[-keep:] if keep else ""
            return False

        self._inner_buffer += buf[:idx]
        self._buffer = buf[idx + len(close) :]

        inner = self._inner_buffer
        self._inner_buffer = ""
        raw_xml = f"<{name}>{inner}</{name}>"

        if name == "attempt_completion":
            # Result body lives inside a nested <result>...</result>.
            result_text, _errors = _extract_result_body(inner)
            out.append(
                ParserEvent(
                    kind="attempt_completion",
                    raw_xml=raw_xml,
                    result=result_text,
                )
            )
            self._mode = self._MODE_DONE
            self._open_tag = ""
            return False

        param_types = self._param_types.get(name, {})
        arguments, errors = _extract_params(inner, param_types)
        if errors:
            out.append(ParserEvent(kind="parse_error", text=";".join(errors)))
            self._mode = self._MODE_DONE
            self._open_tag = ""
            return False
        out.append(
            ParserEvent(
                kind="tool_call",
                tool_name=name,
                arguments=arguments,
                raw_xml=raw_xml,
            )
        )
        self.tool_events_emitted += 1
        self._mode = self._MODE_DONE  # one tool per assistant turn
        self._open_tag = ""
        return False

    def _consume_unknown_tag(self, out: List[ParserEvent], eof: bool) -> bool:
        """Consume an unknown XML block and emit a parse error, not text."""
        buf = self._buffer
        name = self._open_tag
        close = f"</{name}>"
        idx = buf.find(close)
        if idx < 0:
            keep = max(0, len(close) - 1)
            if eof:
                self._buffer = ""
                out.append(ParserEvent(kind="parse_error", text=f"unknown_or_unclosed_tool:{name}"))
                self._mode = self._MODE_DONE
                self._open_tag = ""
                return False
            if len(buf) > keep:
                self._inner_buffer += buf[:-keep] if keep else buf
                self._buffer = buf[-keep:] if keep else ""
            return False

        self._buffer = buf[idx + len(close) :]
        out.append(ParserEvent(kind="parse_error", text=f"unknown_tool:{name}"))
        self._mode = self._MODE_DONE
        self._open_tag = ""
        self._inner_buffer = ""
        return False


def _extract_result_body(inner_xml: str) -> Tuple[str, List[str]]:
    """Return the verbatim text inside the first ``<result>...</result>``."""
    open_tag = "<result>"
    close_tag = "</result>"
    open_pos = inner_xml.find(open_tag)
    if open_pos < 0:
        # No <result> wrapper — accept the whole body as the result text.
        return inner_xml.strip(), ["missing_result_tag"]
    close_pos = inner_xml.find(close_tag, open_pos + len(open_tag))
    if close_pos < 0:
        return inner_xml[open_pos + len(open_tag) :].strip(), ["unclosed_result_tag"]
    return inner_xml[open_pos + len(open_tag) : close_pos].strip(), []


# ---------------------------------------------------------------------------
# Helpers for building the parser from a tool manifest
# ---------------------------------------------------------------------------


def build_param_types_index(tools: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """Pre-compute ``{tool_name: {param: declared_type}}`` for the parser.

    The orchestrator already has the manifest; this just flattens it.
    Pulled out of the parser ctor so the same dict can be reused across
    many requests without re-parsing.
    """
    index: Dict[str, Dict[str, str]] = {}
    for tm in tools or []:
        if not isinstance(tm, dict):
            continue
        name = tm.get("id") or tm.get("name")
        if not name:
            continue
        args_raw = tm.get("args") or tm.get("arguments")
        if not isinstance(args_raw, dict):
            schema = tm.get("args_schema")
            props = schema.get("properties") if isinstance(schema, dict) else None
            if not isinstance(props, dict):
                params = tm.get("parameters")
                props = params.get("properties") if isinstance(params, dict) else None
            args_raw = props if isinstance(props, dict) else {}
        if not isinstance(args_raw, dict):
            continue
        per_tool: Dict[str, str] = {}
        for pname, pspec in args_raw.items():
            t = "string"
            if isinstance(pspec, str):
                low = pspec.lower()
                if low.startswith("list[") or low.startswith("array"):
                    t = "array"
                elif "int" in low and "string" not in low.split("(")[0]:
                    t = "integer"
                elif "float" in low or "number" in low:
                    t = "number"
                elif "bool" in low:
                    t = "boolean"
                elif "object" in low or "dict" in low or "json" in low:
                    t = "object"
            elif isinstance(pspec, dict):
                t = str(pspec.get("type") or "string")
            per_tool[str(pname)] = t
        index[str(name)] = per_tool
    # attempt_completion: result is a free-form string.
    index.setdefault("attempt_completion", {"result": "string"})
    return index
