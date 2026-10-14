"""
Real-Time LangGraph Streaming Agent
استریم واقعی با LangGraph - توکن به توکن

این فایل از امکانات پیشرفته LangGraph استفاده می‌کنه:
- StateGraph با TypedDict State
- Conditional Edges برای routing
- Cycles (برگشت به LLM بعد از اجرای tools)
- Custom streaming با get_stream_writer()

الگوریتم کلی:
1. Initialize: بارگذاری artifacts از DB (از طریق storage service)
2. Build Context: ساخت پیام‌های سیستم با file_context
3. Call LLM: فراخوانی LLM با tools (از طریق llm-proxy service)
4. Execute Tools: اجرای ابزارها (از طریق tool-runner service)
5. Loop: برگشت به LLM تا پاسخ نهایی

تمام ارتباطات از طریق سرویس‌ها انجام می‌شه:
- llm-proxy:8005 برای LLM
- tool-runner:8002 برای ابزارها
- storage:8004 برای دیتابیس
"""
import logging
import re
import uuid
import json
import asyncio
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, AsyncGenerator, Literal, Annotated
from typing_extensions import TypedDict, NotRequired
import operator

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    HumanMessage, 
    AIMessage, 
    SystemMessage, 
    ToolMessage,
    AnyMessage
)

from ..config import settings
from ..llm.gateway_llm_client import get_gateway_llm_client
from ..tools.executor import get_tool_executor
from .nodes import render_tool_result_for_llm
from ..prompts.qwen3_prompts import build_qwen3_system_messages, build_qwen3_tool_result_message
from ..utils.text_inline import build_text_files_inline_block
from ..utils.file_context import detect_file_context
from ..utils.multimodal import build_multimodal_content
from ..utils.message_compile import compile_messages_continue_style
from ..rules.inject import inject_rules_into_openai_messages
from ..rules.loader import get_rule_directory_roots
from ..rules.types import RuleMetadata
from ..debug_ndjson import debug_ndjson

log = logging.getLogger(__name__)


# #region agent log
def _dbg32fb75(location: str, message: str, data: dict, hypothesis_id: str = "") -> None:
    """Emit a single NDJSON debug line to stdout (collected via kubectl logs)."""
    try:
        import json as _json
        import time as _time
        import sys as _sys
        _entry = {
            "sessionId": "32fb75",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(_time.time() * 1000),
        }
        print("ORCH_DBG_32fb75 " + _json.dumps(_entry, ensure_ascii=False, default=str), flush=True)
    except Exception:
        pass
# #endregion


# ==================== State Definition ====================

class AgentState(TypedDict):
    """
    State for the LangGraph agent.
    
    Uses Annotated with add_messages reducer for proper message handling.
    """
    # Messages with proper reducer for appending
    messages: Annotated[List[AnyMessage], add_messages]
    
    # Session info
    session_id: str
    user_id: str
    trace_id: str
    
    # File context (detected from artifacts)
    artifacts: List[Dict[str, Any]]
    file_context: Optional[Dict[str, Any]]
    
    # Execution tracking
    current_step: int
    max_steps: int
    
    # Tool tracking
    tool_runs: Annotated[List[Dict[str, Any]], operator.add]
    pending_tool_calls: List[Dict[str, Any]]
    
    # Thinking and answer
    thinking: Optional[str]
    final_answer: Optional[str]
    
    # Status for routing
    status: str
    error: Optional[str]
    
    # Whether artifacts are fresh uploads (True) or auto-loaded from session (False)
    _has_fresh_artifacts: bool

    # Conversation intelligence
    _conversation_state: Optional[Any]
    _history_summary: str
    _pinned_facts: Optional[Dict[str, Any]]

    # File intelligence
    _file_registry: List[Dict[str, Any]]

    # API ``context[\"rule_policies\"]`` — per-rule ``on`` / ``off``
    _rule_policies: NotRequired[Optional[Dict[str, Any]]]


# ==================== Helper Functions ====================


_TOOL_FENCE_SIGNALS = (
    '"tool_calls"',
    "'tool_calls'",
    "tool_calls:",
    '"tool_name"',
    "'tool_name'",
    "tool_name:",
    '"name":',  # OpenAI-style {"name": "x", "arguments": {...}}
    '"arguments"',
)
_TOOL_FENCE_PYCALL_RE = re.compile(
    r"\b[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+\s*\(",
)

# Header phrases the orchestrator (or the model imitating earlier turns)
# emits before a fake "tool log" bullet list. Treating these as the start
# of a suppress region also kills the bullets that follow.
_FAKE_TOOL_LOG_HEADERS = (
    "(خلاصهٔ خروجی ابزار",
    "[Previous tool outputs from this assistant turn",
    "خلاصهٔ خروجی ابزار در همین پاسخ",
)

# A bullet line like:
#   - session_files({"session_id": "..."}) → در حال اجرا
#   * graph_search({"queries": [...]}) -> ok
#   • plugin.x.y(...) → خطا: ...
_FAKE_TOOL_BULLET_RE = re.compile(
    r"^\s*[-*+\u2022\u25E6]?\s*([a-zA-Z_][\w.]{1,80})\s*\(.*\)\s*(?:[\u2192\u279C\->]+[^\n]*)?\s*$"
)


class _ToolFenceContentFilter:
    """
    Cline-style content filter for ``answer`` chunks during streaming.

    The local model often interleaves a short user-facing sentence with a
    fenced JSON/python block that holds a leaked tool call (which our
    ``chat_stream_with_tools`` recovery converts into a real
    ``tool_calls`` payload). If we forward that fence to the UI, the
    user sees raw JSON instead of the tool's actual result.

    Behaviour:

    - Outside a fence, text streams through immediately (with a small
      tail held back so we never split ```` ``` ```` across pushes).
    - When ```` ``` ```` appears, we buffer the fence body until the
      closing ```` ``` ```` arrives.
    - When the fence closes:
        * If the body looks like a tool call (mentions ``tool_calls``,
          ``tool_name``, an OpenAI-shaped ``"name"/"arguments"`` pair,
          a known tool name from ``tool_name_hints``, or a Python-ish
          dotted call) → DROP it.
        * Otherwise → re-emit it intact (preserves real code examples).
    - At end of stream, an unclosed fence is always dropped.

    This is what makes "live update prose + leaked tool fence" feel
    natural in the UI: only the prose is shown, the fence is silently
    handed to the recovery + tool executor.
    """

    __slots__ = (
        "_mode",
        "_fence_buf",
        "_hold_tail",
        "_tool_hints",
        "_line_buf",
        "_suppress_log",
    )

    def __init__(self, tool_name_hints: Optional[Iterable[str]] = None) -> None:
        self._mode = "text"  # text | fence
        self._fence_buf = ""
        self._hold_tail = ""  # last 1-2 backticks held to avoid splitting ```
        self._tool_hints: List[str] = list(tool_name_hints or [])
        # Per-line tentative buffer in text mode. We don't release a line
        # until we see its trailing ``\n`` (or end of stream), so we can
        # inspect it and drop "fake tool log" lines before they hit UI.
        self._line_buf = ""
        # When True, we're inside a suppressed legacy "(tool output summary ...)"
        # block. Empty lines and bullet/call lines following the header
        # are dropped until we see real prose.
        self._suppress_log = False

    def _looks_like_tool_call(self, body: str) -> bool:
        if not body:
            return False
        low = body.lower()
        for sig in _TOOL_FENCE_SIGNALS:
            if sig in low:
                return True
        for n in self._tool_hints:
            if n and n in body:
                return True
        if _TOOL_FENCE_PYCALL_RE.search(body):
            return True
        return False

    def _is_fake_tool_log_header(self, line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        for h in _FAKE_TOOL_LOG_HEADERS:
            if h in s:
                return True
        return False

    def _is_fake_tool_log_bullet(self, line: str) -> bool:
        s = line.rstrip()
        if not s.strip():
            return False
        m = _FAKE_TOOL_BULLET_RE.match(s)
        if not m:
            return False
        name = m.group(1)
        if name in self._tool_hints:
            return True
        # Dotted call like ``plugin.x.y(...)`` is a strong signal even
        # without a hint match.
        if "." in name and not name.startswith("."):
            return True
        # ``name(...) → in-progress / ok / error`` arrow strongly implies
        # a fake tool log even when the name isn't in our manifest hints
        # (e.g. an old or renamed tool the model still remembers).
        if "\u2192" in s or "->" in s or "\u279C" in s:
            return True
        return False

    def _emit_line(self, line: str, out_parts: List[str]) -> None:
        """Decide whether a completed text-mode line is safe to forward."""
        decision = "emit"
        reason = ""
        # Header opens a suppress region.
        if self._is_fake_tool_log_header(line):
            self._suppress_log = True
            decision = "drop"
            reason = "header"
            _dbg32fb75(  # #region agent log
                "streaming.py:_emit_line",
                "filter_decision",
                {"decision": decision, "reason": reason, "line_head": line[:160], "suppress_now": True},
                hypothesis_id="H4",
            )  # #endregion
            return
        # While in a suppress region, drop blank lines + tool bullets;
        # break out when we see normal prose.
        if self._suppress_log:
            if not line.strip():
                _dbg32fb75(  # #region agent log
                    "streaming.py:_emit_line",
                    "filter_decision",
                    {"decision": "drop", "reason": "blank_in_suppress", "line_head": ""},
                    hypothesis_id="H4",
                )  # #endregion
                return
            if self._is_fake_tool_log_bullet(line):
                _dbg32fb75(  # #region agent log
                    "streaming.py:_emit_line",
                    "filter_decision",
                    {"decision": "drop", "reason": "bullet_in_suppress", "line_head": line[:160]},
                    hypothesis_id="H4",
                )  # #endregion
                return
            self._suppress_log = False
            # fall through and emit this real prose line
        if self._is_fake_tool_log_bullet(line):
            _dbg32fb75(  # #region agent log
                "streaming.py:_emit_line",
                "filter_decision",
                {"decision": "drop", "reason": "bare_bullet", "line_head": line[:160]},
                hypothesis_id="H4",
            )  # #endregion
            return
        # Only log emits of suspicious shape to limit noise: short lines
        # containing parens are most interesting for leak diagnostics.
        if "(" in line and ")" in line and len(line) < 300:
            _dbg32fb75(  # #region agent log
                "streaming.py:_emit_line",
                "filter_decision",
                {"decision": "emit", "reason": "ok", "line_head": line[:200]},
                hypothesis_id="H4",
            )  # #endregion
        out_parts.append(line)

    def push(self, chunk: str) -> str:
        if not chunk:
            return ""
        text = self._hold_tail + chunk
        self._hold_tail = ""
        out_parts: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            if self._mode == "text":
                idx = text.find("```", i)
                if idx < 0:
                    # No fence in the rest of this chunk. We still want
                    # to process the text line-by-line.
                    segment = text[i:]
                    # Hold partial fence opener (up to 2 backticks) at
                    # end so a fence split across pushes is detected.
                    tail_keep = 0
                    if segment.endswith("``"):
                        tail_keep = 2
                    elif segment.endswith("`"):
                        tail_keep = 1
                    if tail_keep:
                        self._hold_tail = segment[-tail_keep:]
                        segment = segment[:-tail_keep]
                    # Append to per-line buffer and emit any completed
                    # lines (those followed by ``\n``).
                    self._line_buf += segment
                    while "\n" in self._line_buf:
                        nl = self._line_buf.index("\n")
                        line = self._line_buf[:nl]
                        self._line_buf = self._line_buf[nl + 1:]
                        self._emit_line(line + "\n", out_parts)
                    break
                # Fence found. Flush any text up to it (line by line).
                pre = text[i:idx]
                self._line_buf += pre
                while "\n" in self._line_buf:
                    nl = self._line_buf.index("\n")
                    line = self._line_buf[:nl]
                    self._line_buf = self._line_buf[nl + 1:]
                    self._emit_line(line + "\n", out_parts)
                # Anything still in _line_buf is the trailing partial
                # line before the fence — emit it now (it's a complete
                # boundary), then enter fence mode.
                if self._line_buf:
                    self._emit_line(self._line_buf, out_parts)
                    self._line_buf = ""
                i = idx + 3
                self._mode = "fence"
                self._fence_buf = ""
            else:  # fence
                close = text.find("```", i)
                if close < 0:
                    self._fence_buf += text[i:]
                    break
                self._fence_buf += text[i:close]
                if self._looks_like_tool_call(self._fence_buf):
                    pass  # silently drop a leaked tool-call fence
                else:
                    out_parts.append("```" + self._fence_buf + "```")
                self._fence_buf = ""
                self._mode = "text"
                i = close + 3
        return "".join(out_parts)

    def flush(self) -> str:
        """End of stream. Drop unclosed fence; emit any pending text line."""
        out_parts: List[str] = []
        if self._mode == "text":
            # If the last line never received \n, treat it as a complete
            # line for emit-decision purposes.
            if self._line_buf:
                self._emit_line(self._line_buf, out_parts)
                self._line_buf = ""
            if self._hold_tail:
                out_parts.append(self._hold_tail)
        self._hold_tail = ""
        self._fence_buf = ""
        self._line_buf = ""
        self._mode = "text"
        self._suppress_log = False
        return "".join(out_parts)


def messages_to_dict(messages: List[AnyMessage]) -> List[Dict[str, Any]]:
    """Convert LangChain messages to dict format for vLLM."""
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            _ai_content = msg.content or ""
            # Qwen3 chat template can produce a malformed prompt (and vLLM
            # then closes the stream with "peer closed connection without
            # sending complete message body") when an assistant message has
            # ``tool_calls`` set but only whitespace in ``content``. The
            # OpenAI tool-call spec already allows ``content=null`` in that
            # case — normalise empty/whitespace prose to an empty string.
            if hasattr(msg, "tool_calls") and msg.tool_calls and isinstance(_ai_content, str):
                if not _ai_content.strip():
                    _ai_content = ""
            msg_dict = {"role": "assistant", "content": _ai_content}
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                serialized_calls = []
                for i, tc in enumerate(msg.tool_calls):
                    # Tool-call dicts come from two sources:
                    #   * native vLLM ``delta.tool_calls`` (key ``arguments``)
                    #   * text-recovery in ``gateway_llm_client`` (also ``arguments``)
                    #   * LangChain's own canonical shape (key ``args``)
                    # Previously we only read ``args``, so anything routed
                    # through recovery serialised as ``arguments="{}"`` and
                    # Qwen3 silently stopped on the next turn. Accept both.
                    raw_args = tc.get("args")
                    if raw_args is None:
                        raw_args = tc.get("arguments")
                    if isinstance(raw_args, dict):
                        args_str = json.dumps(raw_args, ensure_ascii=False)
                    elif isinstance(raw_args, str):
                        args_str = raw_args
                    else:
                        args_str = "{}"
                    serialized_calls.append(
                        {
                            "id": tc.get("id", f"call_{i}"),
                            "type": "function",
                            "function": {
                                "name": tc.get("name"),
                                "arguments": args_str,
                            },
                        }
                    )
                msg_dict["tool_calls"] = serialized_calls
            result.append(msg_dict)
        elif isinstance(msg, ToolMessage):
            result.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content
            })
        elif isinstance(msg, dict):
            result.append(msg)
    return result


def has_tool_results(messages: List[AnyMessage]) -> bool:
    """Check if messages contain any tool results."""
    for msg in messages:
        if isinstance(msg, ToolMessage):
            return True
        if isinstance(msg, dict) and msg.get("role") == "tool":
            return True
    return False


def drop_orphan_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep OpenAI/Qwen message order valid.

    ``role=tool`` messages are only legal immediately after an assistant
    message with matching ``tool_calls``. Historical rows can contain orphan
    tool summaries (for example after old leak-stripping removed the
    assistant's fake call), and Qwen/vLLM can respond with a blank completion
    when such invalid sequences reach the chat template.
    """
    cleaned: List[Dict[str, Any]] = []
    pending_tool_ids: set[str] = set()
    dropped: List[Dict[str, Any]] = []

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role == "assistant":
            cleaned.append(msg)
            pending_tool_ids = set()
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                tid = tc.get("id")
                if tid:
                    pending_tool_ids.add(str(tid))
            continue

        if role == "tool":
            tid = str(msg.get("tool_call_id") or "")
            if tid and tid in pending_tool_ids:
                cleaned.append(msg)
                pending_tool_ids.discard(tid)
            else:
                dropped.append(
                    {
                        "idx": idx,
                        "tool_call_id": tid,
                        "content_head": str(msg.get("content") or "")[:80],
                    }
                )
            continue

        # Any non-tool role closes the current assistant->tool chain.
        pending_tool_ids = set()
        cleaned.append(msg)

    if dropped:
        _dbg32fb75(
            "streaming.py:drop_orphan_tool_messages",
            "dropped_orphan_tools",
            {
                "dropped_count": len(dropped),
                "dropped": dropped[:10],
                "in_len": len(messages),
                "out_len": len(cleaned),
            },
            hypothesis_id="H11",
        )

    return cleaned


# ==================== Node Implementations ====================

class AgentNodes:
    """
    کلاس برای نگهداری node های LangGraph.
    
    هر node یک تابع است که state رو می‌گیره و update برمی‌گردونه.
    از deps برای دسترسی به services استفاده می‌کنه.
    """
    
    def __init__(self, deps: Dict[str, Any]):
        self.llm_client = deps["llm_client"]
        self.tool_executor = deps["tool_executor"]
        self.stream_writer = deps.get("stream_writer")
    
    def _write(self, event: Dict[str, Any]):
        """Write event to stream if writer available."""
        if self.stream_writer:
            self.stream_writer(event)
    
    async def _llm_check_file_relevance(self, user_message: str, chat_history_messages: list) -> bool:
        """
        Ask LLM whether the current user message is related to a previously uploaded file.
        
        This prevents the agent from using file tools when the user changes topic
        mid-conversation, while still supporting implicit references like
        "What is the exact file size?" or "Yes, do all of these items."
        
        Uses a lightweight LLM call with max_tokens=5, temperature=0.
        
        Returns:
            True if the message is about the file, False otherwise.
        """
        # Build a minimal context: last 2 exchanges + current message
        recent_context = ""
        recent_msgs = chat_history_messages[-4:] if chat_history_messages else []
        for msg in recent_msgs:
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content[:200]}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content[:200]}\n"
        
        prompt_messages = [
            {
                "role": "system",
                "content": "You are a classifier. Answer only YES or NO. /no_think"
            },
            {
                "role": "user",
                "content": f"""Recent conversation:
{recent_context}
Current user message: "{user_message}"

Is this message asking about, referring to, or requesting operations on a previously uploaded file/image/audio/video?
Answer YES or NO only."""
            }
        ]
        
        try:
            result = await self.llm_client.chat(
                messages=prompt_messages,
                max_tokens=5,
                temperature=0.0,
            )
            
            if result.get("ok"):
                answer = (result.get("content") or "").strip().upper()
                is_relevant = "YES" in answer
                log.info(f"🔍 File relevance check: '{user_message[:50]}' → {answer} (relevant={is_relevant})")
                return is_relevant
            else:
                log.warning(f"⚠️ File relevance check failed: {result.get('error')}")
                return True  # در صورت خطا، فرض بر مرتبط بودن
        except Exception as e:
            log.warning(f"⚠️ File relevance check error: {e}")
            return True  # در صورت خطا، فرض بر مرتبط بودن
    
    async def initialize(self, state: AgentState) -> Dict[str, Any]:
        """
        Initialize node: Detect file context and build system messages.

        Uses ConversationState (when available) to decide file relevance
        without an extra LLM call.  Falls back to _llm_check_file_relevance
        when no conversation_state is present.
        """
        log.info(f"🚀 Initialize: session={state['session_id']}")
        
        artifacts = state.get("artifacts", [])
        conv_state = state.get("_conversation_state")
        history_summary = state.get("_history_summary", "")
        pinned_facts = state.get("_pinned_facts")
        
        # Decide whether to keep auto-loaded artifacts
        if artifacts and not state.get("_has_fresh_artifacts"):
            should_clear = False

            if conv_state and conv_state.topic_changed:
                # ConversationState already detected a topic switch — no LLM call needed
                should_clear = True
                log.info("🔄 ConversationState detected topic change — clearing file context")
            elif not conv_state:
                # Fallback: use LLM relevance check
                user_message = ""
                all_messages = state.get("messages", [])
                for msg in reversed(all_messages):
                    if isinstance(msg, HumanMessage):
                        user_message = msg.content
                        break
                    elif isinstance(msg, dict) and msg.get("role") == "user":
                        user_message = msg.get("content", "")
                        break
                if user_message:
                    is_relevant = await self._llm_check_file_relevance(user_message, all_messages)
                    if not is_relevant:
                        should_clear = True
                        log.info("🔄 LLM relevance check: topic changed — clearing file context")

            if should_clear:
                artifacts = []
        
        # Detect file context from artifacts
        file_context = detect_file_context(artifacts)
        
        if file_context:
            log.info(f"📁 File context: {file_context.get('type')} - {file_context.get('path')}")
            self._write({"type": "file_detected", "file_context": file_context})
        
        # Get tools manifest from tool-runner service
        tools_manifest = await self.tool_executor.get_tools_manifest()

        # Pre-load small text/code/markdown files into the system prompt so the
        # LLM can reason about their *actual content* without making a tool
        # call. This mirrors what `multimodal.py` does for images/videos.
        inline_block, inlined_metas = await build_text_files_inline_block(file_context)
        if inline_block:
            self._write({
                "type": "text_inlined",
                "files": [m.get("name") for m in inlined_metas],
                "total_chars": sum(int(m.get("chars", 0)) for m in inlined_metas),
            })

        # Annotate file_context + file_registry so downstream prompts know
        # which files are already inlined.
        if file_context and inlined_metas:
            inlined_names = {m["name"] for m in inlined_metas if m.get("name")}
            for f in (file_context.get("all_files") or []):
                if f.get("name") in inlined_names:
                    f["text_content_inlined"] = True
            if file_context.get("name") in inlined_names:
                file_context["text_content_inlined"] = True

        file_registry = state.get("_file_registry")
        if file_registry and inlined_metas:
            inlined_names = {m["name"] for m in inlined_metas if m.get("name")}
            for entry in file_registry:
                if entry.get("name") in inlined_names:
                    entry["inlined"] = True

        # Build system messages with conversation + file intelligence
        system_messages = build_qwen3_system_messages(
            file_context=file_context,
            memory_context=[],
            agent_type="general",
            step_number=1,
            conversation_state=conv_state,
            history_summary=history_summary,
            file_registry=file_registry,
            pinned_facts=pinned_facts,
            chat_session_id=state.get("session_id"),
        )

        if inline_block:
            # Qwen3.5's chat template (`loop.first` check in chat_template.jinja)
            # rejects more than one `system` message. Merge the inline-text
            # block into the first system message instead of appending a
            # second one.
            if system_messages:
                first = system_messages[0]
                existing = first.get("content") or ""
                first["content"] = (
                    f"{existing}\n\n{inline_block}" if existing else inline_block
                )
            else:
                system_messages = [{"role": "system", "content": inline_block}]

        # Markdown session rules (merged into first system message — Qwen single-system constraint)
        rules_meta: List[RuleMetadata] = []
        user_message_for_rules = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                user_message_for_rules = msg.content or ""
                break
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_message_for_rules = msg.get("content") or ""
                break
        if settings.RULES_ENABLED and get_rule_directory_roots():
            system_messages, rules_meta = inject_rules_into_openai_messages(
                system_messages,
                user_message=user_message_for_rules,
                file_context=file_context,
                file_registry=file_registry,
                rule_policies=state.get("_rule_policies"),
            )
            if rules_meta:
                log.info("📜 Markdown rules applied: %s", [m.name for m in rules_meta])

        # Qwen/vLLM chat templates are happiest with one leading system block.
        # Keep the modular builders above, then collapse their outputs into
        # a single system message right before conversion. This also mirrors
        # Cline's style: one large system context, then the conversation.
        if len(system_messages) > 1:
            merged_system_content = "\n\n".join(
                str(m.get("content") or "").strip()
                for m in system_messages
                if m.get("role") == "system" and str(m.get("content") or "").strip()
            )
            non_system_system_messages = [
                m for m in system_messages if m.get("role") != "system"
            ]
            # #region agent log
            _dbg32fb75(
                "streaming.py:initialize:merge_systems",
                "merged_system_messages",
                {
                    "before_count": len(system_messages),
                    "after_count": 1 + len(non_system_system_messages),
                    "merged_chars": len(merged_system_content),
                },
                hypothesis_id="H10",
            )
            # #endregion
            system_messages = (
                [{"role": "system", "content": merged_system_content}]
                + non_system_system_messages
            )

        # Convert to LangChain messages
        messages = []
        for msg in system_messages:
            if msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))
        
        # Add existing user messages
        for msg in state.get("messages", []):
            if isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
                messages.append(msg)
            elif isinstance(msg, dict):
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg.get("content", "")))

        # ✨ Vision multimodal: when the active turn references an
        # image/video, attach it directly to the *latest* user message
        # so the vision-capable LLM sees the pixels natively, instead of
        # being forced to call ``vision_info`` and read a text summary.
        # This is what makes "explain this picture" feel instant on a
        # multimodal model — exactly the same pattern planning_nodes
        # already uses for its executor path.
        # #region agent log
        _dbg32fb75(
            "streaming.py:initialize:pre_mm",
            "pre_multimodal_attach",
            {
                "session_id": str(state.get("session_id"))[:12],
                "has_file_context": bool(file_context),
                "file_context_type": (file_context or {}).get("type"),
                "file_context_keys": list((file_context or {}).keys()) if isinstance(file_context, dict) else [],
                "artifacts_count": len(state.get("artifacts") or []),
                "messages_len": len(messages),
                "last_msg_classes": [type(m).__name__ for m in messages[-5:]],
            },
            hypothesis_id="H2",
        )
        # #endregion
        if file_context and (file_context.get("type") or "").lower() in ("image", "video"):
            last_human_idx: Optional[int] = None
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], HumanMessage):
                    last_human_idx = i
                    break
            if last_human_idx is not None:
                last_hm = messages[last_human_idx]
                base_text = last_hm.content if isinstance(last_hm.content, str) else ""
                try:
                    mm_content = await build_multimodal_content(
                        text=base_text,
                        file_context=file_context,
                        auth_context={
                            "user_id": state.get("user_id"),
                            "group_id": state.get("group_id"),
                            "case_id": state.get("case_id"),
                        },
                    )
                    # #region agent log
                    _dbg32fb75(
                        "streaming.py:initialize:mm_result",
                        "multimodal_build_result",
                        {
                            "mm_content_type": type(mm_content).__name__,
                            "mm_is_list": isinstance(mm_content, list),
                            "mm_len": len(mm_content) if isinstance(mm_content, (list, str)) else None,
                            "mm_part_types": [
                                (p.get("type") if isinstance(p, dict) else type(p).__name__)
                                for p in (mm_content if isinstance(mm_content, list) else [])
                            ][:6],
                        },
                        hypothesis_id="H2",
                    )
                    # #endregion
                    if isinstance(mm_content, list) and len(mm_content) > 1:
                        messages[last_human_idx] = HumanMessage(content=mm_content)
                        n_media = sum(
                            1 for p in mm_content
                            if isinstance(p, dict) and p.get("type") in ("image_url", "video_url")
                        )
                        log.info(
                            "👁️ Attached %d media part(s) to user message for vision LLM",
                            n_media,
                        )
                        self._write({
                            "type": "vision_attached",
                            "media_count": n_media,
                            "file_type": file_context.get("type"),
                        })
                except Exception as exc:
                    log.warning(
                        "⚠️ Multimodal attach failed (%s) — falling back to tool-based vision",
                        exc,
                    )
                    # #region agent log
                    _dbg32fb75(
                        "streaming.py:initialize:mm_error",
                        "multimodal_build_failed",
                        {"error": str(exc)[:300]},
                        hypothesis_id="H2",
                    )
                    # #endregion

        self._write({
            "type": "init_complete",
            "tools_count": len(tools_manifest),
            "rules_applied": [m.name for m in rules_meta] if rules_meta else [],
        })
        
        return {
            "messages": messages,
            "file_context": file_context,
            "status": "initialized",
            "current_step": 0
        }
    
    async def call_llm(self, state: AgentState) -> Dict[str, Any]:
        """
        Call LLM node with tool detection.
        
        Strategy (Tools First, Then Stream):
        1. اگر file_context داریم و tool results نداریم → اول یک call بدون streaming برای detect tools
        2. اگر tool_calls پیدا شد → برگردون برای execution
        3. اگر نه → streaming پاسخ نهایی
        """
        step = (state.get("current_step") or 0) + 1
        max_steps = state.get("max_steps") or settings.MAX_STEPS
        
        log.info(f"📡 LLM Call - Step {step}/{max_steps}")
        self._write({"type": "llm_start", "step": step})
        
        # Get file context
        file_context = state.get("file_context")
        
        # Get tools based on file_context and channel
        if file_context:
            # Convert to dict if needed
            file_context_dict = None
            if hasattr(file_context, "__dict__"):
                file_context_dict = {
                    "type": getattr(file_context, "type", ""),
                    "mime_type": getattr(file_context, "mime_type", ""),
                    "name": getattr(file_context, "name", ""),
                    "path": getattr(file_context, "path", "")
                }
            elif isinstance(file_context, dict):
                file_context_dict = file_context
            
            openai_tools = await self.tool_executor.get_tools_for_file_context(
                file_context=file_context_dict,
                channel="llm"
            )
            log.info(f"📁 File-based tools: {len(openai_tools)} for type={file_context_dict.get('type') if file_context_dict else 'unknown'}")
        else:
            # بدون فایل - فقط ابزارهای بدون نیاز به فایل (text_search, enrich_entity)
            openai_tools = await self.tool_executor.get_tools_for_file_context(
                file_context=None,
                channel="llm"
            )
            log.info(f"📦 No-file tools: {len(openai_tools)}")

            # Conversation may still be ABOUT a file uploaded earlier in
            # the session (no new attachment in this turn). In that case
            # the model needs ``vision_info``/``image_meta``/``file.*``
            # to keep working, otherwise it leaks fake tool fences for
            # tools it cannot actually call. Heuristic: scan the recent
            # message tail for media/file references in either language.
            history_blob = ""
            for _m in (state.get("messages") or [])[-12:]:
                _c = getattr(_m, "content", "") if hasattr(_m, "content") else _m.get("content") if isinstance(_m, dict) else ""
                if isinstance(_c, list):
                    _c = " ".join(
                        str(p.get("text", ""))
                        for p in _c
                        if isinstance(p, dict)
                    )
                history_blob += " " + str(_c or "")[:400]
            _hb = history_blob.lower()
            media_keywords = (
                "عکس", "تصویر", "فایل", "متادیتا", "ویدیو", "ویدئو", "فیلم",
                "صدا", "صوت", "image", "photo", "picture", "video", "audio",
                "file", "metadata",
            )
            if any(k in history_blob or k in _hb for k in media_keywords):
                # Probe both image and document tool sets; merge by name.
                seen_names = {
                    (t.get("function") or {}).get("name")
                    for t in (openai_tools or [])
                    if (t.get("function") or {}).get("name")
                }
                for _probe_type in ("image", "document"):
                    extra_tools = await self.tool_executor.get_tools_for_file_context(
                        file_context={"type": _probe_type},
                        channel="llm",
                    )
                    for t in extra_tools or []:
                        nm = (t.get("function") or {}).get("name")
                        if nm and nm not in seen_names:
                            openai_tools.append(t)
                            seen_names.add(nm)
                log.info(
                    f"📦 Session-aware file tools merged: total now {len(openai_tools)}"
                )
                # #region agent log
                _dbg32fb75(
                    "streaming.py:call_llm:session_aware_tools",
                    "session_aware_tools_merged",
                    {
                        "n_total": len(openai_tools),
                        "hint_kw_present": True,
                        "head_blob": history_blob[:200],
                    },
                    hypothesis_id="H13",
                )
                # #endregion
        
        # Convert messages
        messages = messages_to_dict(state.get("messages", []))
        # LangGraph's ``add_messages`` reducer appends new entries to the
        # existing state list, so even though ``initialize`` builds messages
        # in the canonical ``[system..., user, assistant, ...]`` shape, the
        # final state ends up as
        # ``[original_user, system..., user, assistant, ...]`` — leading
        # system block is broken, ``compile_messages_continue_style`` then
        # cannot recognise the system prefix, and the model goes silent.
        # Pull all system messages to the front (de-duplicated), preserving
        # the original relative order of every other message.
        _systems = [m for m in messages if m.get("role") == "system"]
        _non_systems = [m for m in messages if m.get("role") != "system"]
        _seen_sys: set = set()
        _unique_systems: List[Dict[str, Any]] = []
        for _s in _systems:
            _key = _s.get("content") or ""
            if _key in _seen_sys:
                continue
            _seen_sys.add(_key)
            _unique_systems.append(_s)
        _reordered = _unique_systems + _non_systems
        # #region agent log
        if _reordered != messages:
            _dbg32fb75(
                "streaming.py:call_llm:reorder",
                "messages_reordered",
                {
                    "step": step,
                    "in_len": len(messages),
                    "out_len": len(_reordered),
                    "n_systems": len(_unique_systems),
                    "n_duplicate_systems_dropped": len(_systems) - len(_unique_systems),
                    "first_roles_in": [m.get("role") for m in messages[:6]],
                    "first_roles_out": [m.get("role") for m in _reordered[:6]],
                    "last_roles_in": [m.get("role") for m in messages[-6:]],
                    "last_roles_out": [m.get("role") for m in _reordered[-6:]],
                },
                hypothesis_id="H7",
            )
        # #endregion
        messages = drop_orphan_tool_messages(_reordered)
        log.info(f"📤 Sending {len(messages)} messages, {len(openai_tools) if openai_tools else 0} tools")
        # #region agent log
        _tn = []
        for _t in openai_tools or []:
            _fn = (_t.get("function") or {}).get("name")
            if _fn:
                _tn.append(_fn)
        debug_ndjson(
            "H2",
            "streaming.py:call_llm",
            "openai_tools_built",
            {
                "n_tools": len(openai_tools or []),
                "tool_names": _tn[:40],
                "has_file_context": bool(file_context),
                "file_type": (file_context or {}).get("type") if isinstance(file_context, dict) else None,
            },
        )
        # #endregion
        
        # Check conditions.
        #
        # ``state["messages"]`` includes structured tool messages rebuilt from
        # previous chat turns. On the first LLM call of a *new* user turn,
        # those historical ToolMessages are context, not fresh tool results.
        # Treating them as current results puts the model in continuation mode
        # before any tool has executed in this turn.
        raw_has_tool_results = has_tool_results(state.get("messages", []))
        has_results = bool(step > 1 and raw_has_tool_results)
        
        log.info(f"🔍 Tool check: file_context={bool(file_context)}, has_tool_results={has_results}")
        # #region agent log
        _dbg32fb75(
            "streaming.py:call_llm:has_results",
            "effective_has_results",
            {
                "step": step,
                "raw_has_tool_results": raw_has_tool_results,
                "effective_has_results": has_results,
                "messages_len": len(state.get("messages", []) or []),
            },
            hypothesis_id="H3",
        )
        # #endregion

        # Continue-style prune: keep trailing tool chain, drop old turns (under char budget).
        if getattr(settings, "STREAM_ENABLE_CONTINUE_STYLE_PRUNE", True):
            try:
                tools_extra = 0
                if openai_tools:
                    raw = json.dumps(openai_tools, ensure_ascii=False)
                    tools_extra = min(len(raw), 80_000)
                messages = compile_messages_continue_style(
                    messages,
                    char_budget=int(
                        getattr(settings, "STREAM_CONTEXT_CHAR_BUDGET", 120_000)
                    ),
                    tools_extra_chars=tools_extra,
                )
                messages = drop_orphan_tool_messages(messages)
            except Exception as _pe:
                log.warning("message_compile skipped: %s", _pe)
        
        # ✨ Post-tool reflection prompt — Cline-style iterative decision step.
        # Pushes the model to (a) use what the tool returned, (b) keep collecting
        # if data is still missing, and (c) never fall back to fake/markdown
        # tool blocks in plain text.
        if has_results and step > 1:
            remaining = max(0, (state.get("max_steps") or settings.MAX_STEPS) - step)
            reflection_msg = {
                "role": "system",
                "content": (
                    f"## Step {step} — evaluate tool results\n\n"
                    "You just received tool outputs (see the `tool` messages above).\n\n"
                    "Decide ONE of the following — pick the strongest option:\n\n"
                    "1. **Answer now** — if the tool data is sufficient, write the FINAL reply to the user "
                    "in their language. Quote concrete evidence (names, ids, counts) from the tool output.\n"
                    "2. **Drill deeper** — if a tool returned a list, read the most relevant item "
                    "(e.g. after `session_files`, call `session_read` / `vision_info` on the best `file_id`). "
                    "After `graph_search`, refine `queries` or call related search tools.\n"
                    "3. **Retry differently** — if a tool failed (`ok=false` / `error`), fix the arguments "
                    "(e.g. `queries=['...']` not `query='...'`, real UUID `session_id`, real `file_id`) and call again. "
                    "Do not ask the user to fix what a tool can answer.\n\n"
                    f"Budget: you have {remaining} more tool round(s) before the loop ends. Do not stop early "
                    "if information is still missing, and do not call more tools if you can already answer well.\n\n"
                    "**Native tool_calls only** — never write fake tool blocks (no ```json / ```tool_call / "
                    "<tool_call> / pseudo-YAML). Either emit real `tool_calls` or write the final answer."
                ),
            }
            messages.append(reflection_msg)
        
        # === Strategy: ALWAYS use chat_stream_with_tools (Cline-style) ==========
        #
        # Cline's recursivelyMakeClineRequests calls attemptApiRequest with the
        # full tool schema on EVERY iteration. We do the same: even after we
        # already have tool results, the next round still offers tools so the
        # model can either:
        #   (a) call another tool (e.g. fall back to session_files after a
        #       graph_search failed), or
        #   (b) write the final answer.
        #
        # On the LAST allowed step we set tool_choice="none" so the model is
        # forced to produce a textual answer rather than yet another tool call
        # we cannot execute.
        is_last_step = step >= max_steps

        if is_last_step:
            tool_choice = "none"
        elif (
            file_context
            and getattr(settings, "STREAM_TOOL_CHOICE_REQUIRED_WITH_FILE", False)
            and not has_results
        ):
            tool_choice = "required"
        else:
            tool_choice = "auto"

        log.info(
            "🔍 LLM call (unified) — file_context=%s, tools=%s, has_results=%s, "
            "step=%s/%s, tool_choice=%s",
            bool(file_context),
            len(openai_tools) if openai_tools else 0,
            has_results,
            step,
            max_steps,
            tool_choice,
        )

        collected_thinking = ""
        collected_content = ""        # raw model text (for debug/history)
        displayed_content = ""        # text actually shown to the user
        tool_calls: List[Dict[str, Any]] = []
        stream_failed = False
        answer_started = False  # for real-time text streaming gate

        tool_name_hints: List[str] = []
        for _t in (openai_tools or []):
            fn = _t.get("function") if isinstance(_t, dict) else None
            if isinstance(fn, dict):
                nm = fn.get("name")
                if isinstance(nm, str) and nm.strip():
                    tool_name_hints.append(nm.strip())
        content_filter = _ToolFenceContentFilter(tool_name_hints=tool_name_hints)

        self._write({
            "type": "step_start",
            "step": step,
            "step_type": "planning" if not has_results else "continuation",
            "description": (
                f"📋 Step {step}: Analyzing request and planning"
                if not has_results
                else f"🧭 Step {step}: Continuing with tool results"
            ),
        })

        try:
            stream_kwargs: Dict[str, Any] = {"messages": messages}
            if openai_tools and tool_choice != "none":
                stream_kwargs["tools"] = openai_tools
                stream_kwargs["tool_choice"] = tool_choice

            # #region agent log
            def _msg_shape(m):
                if isinstance(m, dict):
                    role = m.get("role")
                    c = m.get("content")
                elif isinstance(m, (HumanMessage, AIMessage, SystemMessage, ToolMessage)):
                    role = type(m).__name__
                    c = m.content
                else:
                    return {"role": type(m).__name__, "ctype": "?", "clen": 0}
                if isinstance(c, list):
                    ctype = "list:" + ",".join(
                        p.get("type", "?") if isinstance(p, dict) else type(p).__name__
                        for p in c[:5]
                    )
                    clen = sum(
                        len(p.get("text", "")) if isinstance(p, dict) and p.get("type") == "text" else 0
                        for p in c
                    )
                    text_part = next(
                        (p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"),
                        "",
                    )
                    head = text_part[:160]
                else:
                    ctype = "str"
                    clen = len(c or "")
                    head = (c or "")[:160]
                return {"role": role, "ctype": ctype, "clen": clen, "head": head}

            shapes = [_msg_shape(m) for m in messages]
            legacy_hits = sum(
                1 for s in shapes
                if "خلاصهٔ خروجی ابزار" in (s.get("head") or "")
                or "Previous tool outputs" in (s.get("head") or "")
            )
            _dbg32fb75(
                "streaming.py:call_llm:pre_send",
                "pre_llm_send",
                {
                    "step": step,
                    "n_messages": len(messages),
                    "tool_choice": tool_choice,
                    "n_tools": len(openai_tools or []),
                    "first_5_shapes": shapes[:5],
                    "last_5_shapes": shapes[-5:],
                    "legacy_marker_hits_in_history": legacy_hits,
                    "tool_name_hints": tool_name_hints[:20],
                },
                hypothesis_id="H1+H2",
            )
            # #endregion

            async for event in self.llm_client.chat_stream_with_tools(**stream_kwargs):
                etype = event.get("type")

                if etype == "thinking_start":
                    self._write({"type": "thinking_start", "step": step})
                elif etype == "thinking":
                    chunk = event.get("content", "")
                    collected_thinking += chunk
                    self._write({"type": "thinking", "content": chunk})
                elif etype == "thinking_end":
                    self._write({"type": "thinking_end", "step": step})
                elif etype == "tool_call_start":
                    self._write({
                        "type": "tool_call_start",
                        "step": step,
                        "index": event.get("index", 0),
                        "tool_id": event.get("tool_id"),
                        "tool_name": event.get("tool_name"),
                    })
                elif etype == "tool_call_delta":
                    self._write({
                        "type": "tool_call_delta",
                        "step": step,
                        "index": event.get("index", 0),
                        "tool_id": event.get("tool_id"),
                        "tool_name": event.get("tool_name"),
                        "arguments_delta": event.get("arguments_delta", ""),
                    })
                elif etype == "answer":
                    chunk_content = event.get("content", "") or ""
                    if chunk_content:
                        collected_content += chunk_content
                        # Cline-style filter: if the model's answer starts
                        # with a code fence (which is almost always a
                        # leaked tool call that our recovery will catch
                        # and convert into a real tool_calls payload),
                        # don't show the raw JSON/code to the user.
                        safe = content_filter.push(chunk_content)
                        if safe:
                            if not answer_started:
                                self._write({"type": "answer_start", "step": step})
                                answer_started = True
                            self._write({"type": "answer", "content": safe})
                            displayed_content += safe
                elif etype == "tool_calls":
                    tool_calls = event.get("tool_calls", [])
                elif etype == "error":
                    log.warning(f"⚠️ chat_stream_with_tools error: {event.get('content')} — falling back")
                    stream_failed = True
                    break

        except Exception as exc:
            log.warning(f"⚠️ chat_stream_with_tools exception: {exc} — falling back to non-streaming")
            stream_failed = True

        # Flush any text the filter was still holding (real prose that
        # happens to follow a closed fence, or pure prose that never
        # tripped the suppress branch).
        tail_safe = content_filter.flush()
        if tail_safe:
            if not answer_started:
                self._write({"type": "answer_start", "step": step})
                answer_started = True
            self._write({"type": "answer", "content": tail_safe})
            displayed_content += tail_safe

        # #region agent log
        debug_ndjson(
            "H1",
            "streaming.py:call_llm",
            "after_chat_stream_with_tools",
            {
                "stream_failed": stream_failed,
                "n_tool_calls": len(tool_calls or []),
                "tool_call_names": [tc.get("name") for tc in (tool_calls or [])],
                "collected_content_len": len(collected_content or ""),
                "collected_content_preview": (collected_content or "")[:400],
                "displayed_content_len": len(displayed_content or ""),
                "displayed_content_preview": (displayed_content or "")[:400],
                "filter_mode_final": getattr(content_filter, "_mode", "?"),
                "is_last_step": is_last_step,
                "tool_choice_used": tool_choice,
            },
        )
        # #endregion

        # ---- Fallback: non-streaming if streaming failed ----
        if stream_failed:
            try:
                fb_kwargs: Dict[str, Any] = {"messages": messages}
                if openai_tools and tool_choice != "none":
                    fb_kwargs["tools"] = openai_tools
                    fb_kwargs["tool_choice"] = tool_choice
                result = await self.llm_client.chat(**fb_kwargs)
                if result.get("ok"):
                    tool_calls = result.get("tool_calls", []) or tool_calls
                    fb_content = result.get("content", "") or ""
                    if fb_content and not collected_content:
                        collected_content = fb_content
                        # Run the same fence filter on the buffered text
                        # so we don't ship a JSON tool-call to the UI here
                        # either.
                        fb_filter = _ToolFenceContentFilter(tool_name_hints=tool_name_hints)
                        safe_fb = fb_filter.push(fb_content) + fb_filter.flush()
                        if safe_fb:
                            if not answer_started:
                                self._write({"type": "answer_start", "step": step})
                                answer_started = True
                            self._write({"type": "answer", "content": safe_fb})
                            displayed_content += safe_fb
                    fb_thinking = result.get("reasoning_content", "") or ""
                    if fb_thinking and not collected_thinking:
                        self._write({"type": "thinking_start", "step": step})
                        self._write({"type": "thinking", "content": fb_thinking})
                        self._write({"type": "thinking_end", "step": step})
                        collected_thinking = fb_thinking
                else:
                    log.error(f"🔴 LLM fallback call failed: {result.get('error')}")
            except Exception as exc_fb:
                log.error(f"🔴 LLM fallback exception: {exc_fb}", exc_info=True)

        # ---- Route on result ----
        if tool_calls and not is_last_step:
            log.info(f"✅ {len(tool_calls)} tool_call(s) requested at step {step}")
            # If the model leaked the tool call as a JSON fence in
            # ``collected_content``, the user-visible text is empty and
            # ``displayed_content`` is what we should keep on the AIMessage,
            # not the raw fence — otherwise chat history will store the JSON.
            #
            # Normalise pure-whitespace prose to empty so the next round's
            # Qwen3 template does not see ``content="\n\n"`` next to
            # ``tool_calls`` (vLLM closes the stream on that shape).
            _ai_text = displayed_content or ""
            if isinstance(_ai_text, str) and not _ai_text.strip():
                _ai_text = ""
            ai_msg = AIMessage(content=_ai_text)
            ai_msg.tool_calls = tool_calls

            tool_names = [tc.get("name", "unknown") for tc in tool_calls]
            self._write({
                "type": "tool_plan",
                "step": step,
                "tools": tool_names,
                "description": f"🔧 Execution plan: {', '.join(tool_names)}",
            })
            self._write({"type": "llm_done", "step": step})

            return {
                "messages": [ai_msg],
                "pending_tool_calls": tool_calls,
                "thinking": collected_thinking,
                "current_step": step,
                "status": "tools_pending",
            }

        # No tool calls (or last step) — finalize with whatever content we got.
        #
        # We force a final-synthesis pass in TWO scenarios where the user
        # would otherwise see nothing useful:
        #
        #   A) ``has_results`` and the stream went silent (or leaked only
        #      a fenced tool call that the filter suppressed). Classic
        #      "stuck after failed tools" state.
        #   B) Even at step 1 (no prior tool results), the model produced
        #      ONLY a leaked tool call that no recovery accepted — usually
        #      because the model hallucinated a tool name that isn't in
        #      this turn's manifest. Without this branch the user would
        #      see "..." and the conversation would just end.
        leak_with_no_recovered_call = (
            not displayed_content
            and not tool_calls
            and bool(collected_content)
        )
        # Same root cause, but the model also produced a small amount of
        # user-visible prose ("Let me call tool X...") before/after
        # the fence we filtered out. ``displayed_content > 0`` previously
        # disabled synthesis, leaving the user with a half-message and no
        # tool execution. Detect this when the filtered-out volume is large
        # relative to what was shown.
        fence_leak_partial = (
            not tool_calls
            and bool(collected_content)
            and len(displayed_content) > 0
            and len(displayed_content) < int(len(collected_content) * 0.7)
            and (len(collected_content) - len(displayed_content)) >= 60
        )
        # Extract whatever tool name(s) the model wrote in the fence so the
        # synthesis prompt can name them explicitly. This is best-effort,
        # not a structural parse — we just want a hint for the diagnostic.
        requested_tool_names: List[str] = []
        for _m in re.finditer(
            r'"(?:name|tool_name|tool)"\s*:\s*"([A-Za-z_][\w.\-]{0,80})"',
            collected_content or "",
        ):
            n = _m.group(1)
            if n and n not in requested_tool_names:
                requested_tool_names.append(n)
        if leak_with_no_recovered_call:
            log.info(
                "🪤 Detected fenced tool-call leak with no recoverable call "
                "(likely hallucinated tool name). Forcing synthesis."
            )
        if fence_leak_partial:
            log.info(
                "🪤 Partial fence leak: shown=%d of collected=%d chars, requested_tools=%s",
                len(displayed_content),
                len(collected_content),
                requested_tool_names,
            )
        # #region agent log
        _dbg32fb75(
            "streaming.py:call_llm:stream_done",
            "post_stream_state",
            {
                "step": step,
                "displayed_len": len(displayed_content),
                "collected_len": len(collected_content),
                "displayed_head": displayed_content[:300],
                "collected_head": collected_content[:300],
                "has_results": bool(has_results),
                "tool_calls_count": len(tool_calls or []),
                "leak_with_no_recovered_call": bool(leak_with_no_recovered_call),
                "answer_started": bool(answer_started),
            },
            hypothesis_id="H4",
        )
        # #endregion
        needs_synthesis = (
            (not displayed_content and (has_results or leak_with_no_recovered_call))
            or fence_leak_partial
        )
        if needs_synthesis:
            log.info(
                "🛟 Forcing synthesis (has_results=%s leak_only=%s partial=%s)",
                has_results,
                leak_with_no_recovered_call,
                fence_leak_partial,
            )
            avail_tools_line = (
                ", ".join(f"`{n}`" for n in tool_name_hints[:30])
                if tool_name_hints
                else "(none in this turn)"
            )
            extra_diag = ""
            if leak_with_no_recovered_call or fence_leak_partial:
                req_line = (
                    ", ".join(f"`{n}`" for n in requested_tool_names[:6])
                    if requested_tool_names
                    else "(unknown)"
                )
                extra_diag = (
                    "\n\n**Why you are here:** in the previous attempt you wrote a "
                    "tool-call inside a code fence. The tool name(s) you wrote "
                    f"({req_line}) are NOT in THIS turn's tool list, or the format "
                    "was wrong. The user did NOT see that fence. The exact tools you "
                    f"may reference now are: {avail_tools_line}. If the user's "
                    "question genuinely needs a tool that is not available, say so "
                    "briefly and offer the closest alternative you can compute "
                    "yourself or with the available tools."
                )
            try:
                synth_messages = list(messages) + [
                    {
                        "role": "system",
                        "content": (
                            "## Final synthesis required\n\n"
                            "Your previous tool round produced no new text. Now you MUST "
                            "write the final answer for the user in their language using "
                            "ONLY the information already gathered in this conversation.\n\n"
                            "- Summarise what each tool returned (success or failure).\n"
                            "- If tool failures repeated, explicitly say what you tried and what is "
                            "still missing.\n"
                            "- Do NOT call any more tools.\n"
                            "- Do NOT print code fences or fake tool blocks.\n"
                            "- If you genuinely have no data, apologise briefly and ask one "
                            "focused clarifying question."
                            + extra_diag
                        ),
                    }
                ]
                synth_started = False
                synth_filter = _ToolFenceContentFilter(tool_name_hints=tool_name_hints)
                # IMPORTANT: pass tools=None — that disables text-based
                # recovery in chat_stream_with_tools so the model can't
                # "summon" another tool by writing fake JSON here.
                async for event in self.llm_client.chat_stream_with_tools(
                    messages=synth_messages,
                ):
                    etype = event.get("type")
                    if etype == "thinking":
                        chunk = event.get("content", "") or ""
                        if chunk:
                            collected_thinking += chunk
                            self._write({"type": "thinking", "content": chunk})
                    elif etype == "answer":
                        chunk_content = event.get("content", "") or ""
                        if not chunk_content:
                            continue
                        collected_content += chunk_content
                        safe = synth_filter.push(chunk_content)
                        if safe:
                            if not (answer_started or synth_started):
                                self._write({"type": "answer_start", "step": step})
                                answer_started = True
                                synth_started = True
                            self._write({"type": "answer", "content": safe})
                            displayed_content += safe
                tail_safe = synth_filter.flush()
                if tail_safe:
                    if not (answer_started or synth_started):
                        self._write({"type": "answer_start", "step": step})
                        answer_started = True
                        synth_started = True
                    self._write({"type": "answer", "content": tail_safe})
                    displayed_content += tail_safe
            except Exception as exc_synth:
                log.warning(f"⚠️ Forced synthesis failed: {exc_synth}")

        if not displayed_content:
            placeholder = (
                "I could not gather enough usable information in this attempt. "
                "Please provide a bit more detail or rephrase the request."
                if has_results
                else (
                    "I could not complete a response for this request. "
                    "Please try rephrasing or adding a bit more detail."
                )
            )
            if not answer_started:
                self._write({"type": "answer_start", "step": step})
                answer_started = True
            self._write({"type": "answer", "content": placeholder})
            collected_content = placeholder
            displayed_content = placeholder

        self._write({"type": "llm_done", "step": step})
        # Persist the user-visible answer only. ``collected_content`` may
        # contain a raw JSON tool-fence that the user never saw.
        ai_msg = AIMessage(content=displayed_content)
        return {
            "messages": [ai_msg],
            "pending_tool_calls": [],
            "thinking": collected_thinking or state.get("thinking"),
            "final_answer": displayed_content,
            "current_step": step,
            "status": "complete",
        }
    
    async def execute_tools(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute pending tools node.
        
        این node:
        1. tool calls رو از state می‌گیره
        2. هر tool رو از طریق tool-runner service اجرا می‌کنه
        3. نتایج رو به صورت ToolMessage برمی‌گردونه
        """
        tool_calls = state.get("pending_tool_calls", [])
        # #region agent log
        debug_ndjson(
            "H4",
            "streaming.py:execute_tools",
            "execute_tools_entry",
            {
                "n_pending": len(tool_calls or []),
                "sample": [
                    {"name": c.get("name"), "id": c.get("id"), "args_type": type(c.get("arguments")).__name__}
                    for c in (tool_calls or [])[:8]
                ],
            },
        )
        # #endregion

        if not tool_calls:
            return {"pending_tool_calls": [], "status": "no_tools"}
        
        step = state.get("current_step", 0)
        log.info(f"🔧 Executing {len(tool_calls)} tool(s) at step {step}")
        
        # ✨ Event: Start tool execution phase
        tool_names = [tc.get("name", "unknown") for tc in tool_calls]
        self._write({
            "type": "step_start",
            "step": step,
            "step_type": "tool_execution",
            "description": f"🔧 Step {step}: Executing tools ({', '.join(tool_names)})"
        })
        
        file_context = state.get("file_context")
        session_id = state.get("session_id")
        user_id = state.get("user_id")
        group_id = state.get("group_id")
        
        tool_cap = max(4_000, int(getattr(settings, "STREAM_TOOL_RESULT_MAX_CHARS", 24_000)))

        async def _execute_one(i: int, call: Dict[str, Any]):
            tool_name = call.get("name", "unknown")
            tool_id = call.get("id", f"call_{step}_{i}")
            arguments = call.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {}

            self._write({
                "type": "tool_start",
                "tool_id": tool_id,
                "tool_name": tool_name,
                "tool_index": i + 1,
                "total_tools": len(tool_calls),
                "args": arguments,
                "description": f"⚙️ Running {tool_name} ({i+1}/{len(tool_calls)})...",
            })

            started_at = datetime.utcnow()
            try:
                result = await self.tool_executor.execute(
                    tool_name=tool_name,
                    arguments=arguments,
                    file_context=file_context,
                    session_id=session_id,
                    user_id=user_id,
                    group_id=group_id,
                )
                execution_time = (datetime.utcnow() - started_at).total_seconds()
                self._write({
                    "type": "tool_result",
                    "tool_id": tool_id,
                    "tool_name": tool_name,
                    "ok": result.get("ok", False),
                    "execution_time": execution_time,
                    "description": f"{'✅' if result.get('ok') else '❌'} {tool_name} - {execution_time:.1f}s",
                })
                if result.get("ok"):
                    content = render_tool_result_for_llm(result)
                    if not content:
                        # Tool ran successfully but exposes no LLM-visible
                        # text — typical for index-only tools like
                        # ``vision_info`` (its job is to populate
                        # metadata/embeddings; the model can already see
                        # the image/video pixels attached to the user
                        # message). Send a short, distinctive marker so
                        # the model knows the call resolved without
                        # producing extra text to read.
                        content = (
                            f"[ok] {tool_name} executed; this tool does not "
                            "return text to the chat — use the media already "
                            "attached to the user message, or the data you "
                            "already gathered, to answer."
                        )
                else:
                    content = render_tool_result_for_llm(result) or f"Error: {result.get('error')}"
                tm = ToolMessage(
                    content=content[:tool_cap],
                    tool_call_id=tool_id,
                    name=tool_name,
                )
                run = {
                    "id": tool_id,
                    "tool_id": tool_name,
                    "args": arguments,
                    "result": result,
                    "step": step,
                    "execution_time": execution_time,
                }
                log.info(f"   {'✅' if result.get('ok') else '❌'} {tool_name}: {execution_time:.2f}s")
                return tm, run
            except Exception as e:
                log.error(f"Tool {tool_name} error: {e}")
                execution_time = (datetime.utcnow() - started_at).total_seconds()
                self._write({
                    "type": "tool_error",
                    "tool_id": tool_id,
                    "tool_name": tool_name,
                    "error": str(e),
                    "execution_time": execution_time,
                    "description": f"❌ Error in {tool_name}: {str(e)[:50]}",
                })
                tm = ToolMessage(
                    content=(f"Error executing {tool_name}: {str(e)}")[:tool_cap],
                    tool_call_id=tool_id,
                    name=tool_name,
                )
                run = {
                    "id": tool_id,
                    "tool_id": tool_name,
                    "args": arguments,
                    "result": {"ok": False, "error": str(e)},
                    "step": step,
                    "execution_time": execution_time,
                }
                return tm, run

        pairs = await asyncio.gather(
            *(_execute_one(i, c) for i, c in enumerate(tool_calls))
        )
        tool_messages = [p[0] for p in pairs]
        new_tool_runs = [p[1] for p in pairs]
        
        # ✨ Event: بازتاب و ارزیابی نتایج
        success_count = len([r for r in new_tool_runs if r.get("result", {}).get("ok")])
        self._write({
            "type": "reflection",
            "step": step,
            "tools_count": len(tool_calls),
            "success_count": success_count,
            "description": f"🔍 Reflection: {success_count}/{len(tool_calls)} tools succeeded"
        })
        
        # Reflection prompt without <think> tags (Qwen3-optimized)
        reflection_prompt = """## Evaluate Tool Results
- Tools executed: {tool_count}
- Successful tools: {success_count}

Based on these results, either provide the best answer now or call another tool only if it is genuinely necessary.
""".format(
            tool_count=len(tool_calls),
            success_count=success_count
        )
        
        # Emit reflection event
        self._write({
            "type": "reflection",
            "content": f"Evaluating results from {len(tool_calls)} tools...",
            "tools_executed": [t.get("tool_id") for t in new_tool_runs]
        })
        
        return {
            "messages": tool_messages,
            "tool_runs": new_tool_runs,
            "pending_tool_calls": [],
            "status": "tools_complete"
        }
    
    async def finalize(self, state: AgentState) -> Dict[str, Any]:
        """
        Finalize node: Extract final answer.
        """
        messages = state.get("messages", [])
        
        # Find the last assistant message with content
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = msg.content
                if content and not getattr(msg, 'tool_calls', None):
                    self._write({"type": "final_answer", "content": content})
                    return {
                        "final_answer": content,
                        "status": "completed"
                    }
        
        return {
            "final_answer": "No answer was produced.",
            "status": "completed"
        }


# ==================== Routing Functions ====================

def route_after_llm(state: AgentState) -> Literal["execute_tools", "finalize"]:
    """
    Route after LLM call.
    
    اگر tool_calls داریم → execute_tools
    وگرنه → finalize
    """
    if state.get("error"):
        return "finalize"
    
    max_steps = state.get("max_steps") or settings.MAX_STEPS
    current_step = state.get("current_step") or 0
    
    # Check for pending tools
    if state.get("pending_tool_calls"):
        if current_step >= max_steps:
            log.info(f"⚠️ Max steps reached ({current_step}/{max_steps}), finalizing")
            return "finalize"
        return "execute_tools"
    
    return "finalize"


def route_after_tools(state: AgentState) -> Literal["call_llm", "finalize"]:
    """
    Route after tool execution.
    
    این یک cycle ایجاد می‌کنه: tools → llm → tools → llm → ...
    """
    max_steps = state.get("max_steps") or settings.MAX_STEPS
    current_step = state.get("current_step") or 0
    
    if current_step >= max_steps:
        log.info(f"⚠️ Max steps reached ({current_step}/{max_steps}), finalizing")
        return "finalize"
    
    # After tools, always go back to LLM for next response
    return "call_llm"


# ==================== Build Graph ====================

def build_agent_graph(deps: Dict[str, Any]) -> StateGraph:
    """
    Build the LangGraph agent.
    
    Graph structure:
    
    START → initialize → call_llm → [routing]
                            ↓              ↓
                     execute_tools    finalize → END
                            ↓
                       call_llm (cycle)
    """
    nodes = AgentNodes(deps)
    
    # Build graph with proper state
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("initialize", nodes.initialize)
    workflow.add_node("call_llm", nodes.call_llm)
    workflow.add_node("execute_tools", nodes.execute_tools)
    workflow.add_node("finalize", nodes.finalize)
    
    # Entry point
    workflow.add_edge(START, "initialize")
    
    # initialize → call_llm
    workflow.add_edge("initialize", "call_llm")
    
    # call_llm → conditional routing
    workflow.add_conditional_edges(
        "call_llm",
        route_after_llm,
        {
            "execute_tools": "execute_tools",
            "finalize": "finalize"
        }
    )
    
    # execute_tools → conditional routing (creates cycle back to call_llm)
    workflow.add_conditional_edges(
        "execute_tools",
        route_after_tools,
        {
            "call_llm": "call_llm",  # This creates the cycle!
            "finalize": "finalize"
        }
    )
    
    # finalize → END
    workflow.add_edge("finalize", END)
    
    return workflow.compile()


# ==================== Public Streaming API ====================

async def stream_agent(
    user_message: str,
    session_id: str,
    artifacts: List[Dict[str, Any]] = None,
    user_id: str = "default",
    max_steps: Optional[int] = None,
    chat_history: List[Dict[str, str]] = None,
    has_fresh_artifacts: bool = False,
    conversation_state=None,
    history_summary: str = "",
    file_registry: list = None,
    pinned_facts: Optional[Dict[str, Any]] = None,
    rule_policies: Optional[Dict[str, Any]] = None,
    workspace_runtime: Optional[Dict[str, Any]] = None,
    discourse_focus: Optional[Dict[str, Any]] = None,
    task_checklist_state: Optional[Dict[str, Any]] = None,
    active_plugin: Optional[str] = None,
    host_page_prompt: Optional[str] = None,
    rag_context: Optional[List[Dict[str, Any]]] = None,
    rag_prompt_block: str = "",
    memory_prompt_block: str = "",
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream agent execution with real-time events.
    """
    if settings.NATIVE_TOOLS_ENGINE_ENABLED:
        from .native_tools_engine import stream_agent_native

        async for ev in stream_agent_native(
            user_message=user_message,
            session_id=session_id,
            artifacts=artifacts,
            user_id=user_id,
            max_steps=max_steps,
            chat_history=chat_history,
            has_fresh_artifacts=has_fresh_artifacts,
            conversation_state=conversation_state,
            history_summary=history_summary,
            file_registry=file_registry,
            pinned_facts=pinned_facts,
            workspace_runtime=workspace_runtime,
            discourse_focus=discourse_focus,
            task_checklist_state=task_checklist_state,
            rule_policies=rule_policies,
            active_plugin=active_plugin,
            host_page_prompt=host_page_prompt,
            rag_context=rag_context,
            rag_prompt_block=rag_prompt_block,
            memory_prompt_block=memory_prompt_block,
        ):
            yield ev
        return

    # Cline XML chat engine archived — see orchestrators/_archived/cline_xml.
    # Fall through to legacy LangGraph body below only when native is disabled.

    trace_id = str(uuid.uuid4())
    effective_max_steps = int(
        max_steps if max_steps is not None else settings.MAX_STEPS
    )
    
    # Build initial messages.
    #
    # Native OpenAI/Cline-style tool continuity:
    # chat_history may contain rows shaped like
    #   {role: "assistant", content: text, tool_calls: [{id,type,function:{name,arguments}}]}
    #   {role: "tool", tool_call_id, name, content: summary}
    # We mirror those into LangChain ``AIMessage(... tool_calls=...)`` and
    # ``ToolMessage(... tool_call_id=...)`` so vLLM's chat template can render
    # the past turn as ``<tool_call>`` + ``<tool_response>`` blocks —
    # exactly the shape Qwen3-coder was trained on. This stops the model
    # from imitating a text bullet-list of past tool calls.
    messages = []
    if chat_history:
        for msg in chat_history:
            role = msg.get("role")
            if role == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))
                continue
            if role == "assistant":
                ai_content = msg.get("content", "") or ""
                raw_tcs = msg.get("tool_calls") or []
                ai = AIMessage(content=ai_content)
                if raw_tcs:
                    lc_tcs = []
                    for i, tc in enumerate(raw_tcs):
                        fn = tc.get("function") if isinstance(tc, dict) else None
                        name = ""
                        args_obj: Any = {}
                        if isinstance(fn, dict):
                            name = fn.get("name") or ""
                            raw_args = fn.get("arguments")
                            if isinstance(raw_args, str):
                                try:
                                    args_obj = json.loads(raw_args) if raw_args else {}
                                except Exception:
                                    args_obj = {}
                            elif isinstance(raw_args, dict):
                                args_obj = raw_args
                        else:
                            name = (tc.get("name") or "").strip()
                            ra = tc.get("arguments") if "arguments" in tc else tc.get("args")
                            if isinstance(ra, str):
                                try:
                                    args_obj = json.loads(ra) if ra else {}
                                except Exception:
                                    args_obj = {}
                            elif isinstance(ra, dict):
                                args_obj = ra
                        lc_tcs.append(
                            {
                                "name": name,
                                "args": args_obj or {},
                                "id": str(tc.get("id") or f"call_{i}"),
                            }
                        )
                    ai.tool_calls = lc_tcs
                messages.append(ai)
                continue
            if role == "tool":
                messages.append(
                    ToolMessage(
                        content=msg.get("content", "") or "",
                        tool_call_id=str(msg.get("tool_call_id") or ""),
                        name=msg.get("name") or "tool",
                    )
                )
                continue
    messages.append(HumanMessage(content=user_message))
    
    # Initial state
    initial_state: AgentState = {
        "messages": messages,
        "session_id": session_id,
        "user_id": user_id,
        "trace_id": trace_id,
        "artifacts": artifacts or [],
        "file_context": None,
        "current_step": 0,
        "max_steps": effective_max_steps,
        "tool_runs": [],
        "pending_tool_calls": [],
        "thinking": None,
        "final_answer": None,
        "status": "starting",
        "error": None,
        "_has_fresh_artifacts": has_fresh_artifacts,
        "_conversation_state": conversation_state,
        "_history_summary": history_summary,
        "_pinned_facts": pinned_facts,
        "_file_registry": file_registry or [],
        "_rule_policies": rule_policies or {},
    }
    
    # Collected events to emit
    events_queue = asyncio.Queue()
    
    def stream_writer(event: Dict[str, Any]):
        """Custom stream writer that queues events."""
        events_queue.put_nowait(event)
    
    # Get dependencies
    deps = {
        "llm_client": get_gateway_llm_client(),
        "tool_executor": get_tool_executor(),
        "stream_writer": stream_writer
    }
    
    # Build graph
    graph = build_agent_graph(deps)
    
    # Start event
    yield {
        "type": "start",
        "trace_id": trace_id,
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        # Run graph in background task
        final_state = None
        
        async def run_graph():
            nonlocal final_state
            # Accumulate tool_runs separately because final_state is overwritten
            # each time with just the latest node's output (not accumulated state).
            # The 'finalize' node (last node) never returns tool_runs, so
            # final_state.get("tool_runs") would always be [].
            accumulated_tool_runs = []
            async for chunk in graph.astream(
                initial_state,
                stream_mode=["custom", "updates"]
            ):
                if isinstance(chunk, tuple):
                    mode, data = chunk
                    if mode == "custom":
                        events_queue.put_nowait(data)
                    elif mode == "updates":
                        for node_name, node_output in data.items():
                            if node_output:
                                final_state = node_output
                                # Accumulate tool_runs from execute_tools node
                                if node_output.get("tool_runs"):
                                    accumulated_tool_runs.extend(node_output["tool_runs"])
                                events_queue.put_nowait({
                                    "type": "state_update",
                                    "node": node_name,
                                    "status": node_output.get("status"),
                                    "timestamp": datetime.utcnow().isoformat()
                                })
                else:
                    for node_name, node_output in chunk.items():
                        if node_output:
                            final_state = node_output
                            if node_output.get("tool_runs"):
                                accumulated_tool_runs.extend(node_output.get("tool_runs", []))
            
            # Signal completion — pass accumulated tool_runs explicitly
            events_queue.put_nowait({
                "__done__": True,
                "final_state": final_state,
                "accumulated_tool_runs": accumulated_tool_runs
            })
        
        # Start graph execution
        graph_task = asyncio.create_task(run_graph())
        
        # Yield events as they come
        accumulated_tool_runs = []
        while True:
            try:
                event = await asyncio.wait_for(events_queue.get(), timeout=120.0)
                
                if isinstance(event, dict) and event.get("__done__"):
                    final_state = event.get("final_state")
                    accumulated_tool_runs = event.get("accumulated_tool_runs", [])
                    break
                
                yield event
                
            except asyncio.TimeoutError:
                log.warning("Event queue timeout")
                break
        
        # Wait for graph to complete
        await graph_task
        
        # Done event — use accumulated_tool_runs (not final_state.tool_runs which is
        # always empty because 'finalize' node never includes tool_runs in its output)
        # Keep channels.ui (host page effects e.g. map_action); strip embed / oversized rawdata.
        def _strip_channels(tool_runs: list) -> list:
            stripped = []
            for tr in tool_runs:
                tr = dict(tr)
                try:
                    result = tr.get("result", {})
                    if not isinstance(result, dict):
                        stripped.append(tr)
                        continue

                    def _keep_ui_channels(channels: dict) -> dict:
                        kept: dict = {}
                        if "llm" in channels:
                            kept["llm"] = channels.get("llm")
                        if "ui" in channels:
                            kept["ui"] = channels.get("ui")
                        if "metadata" in channels and isinstance(channels.get("metadata"), dict):
                            kept["metadata"] = channels.get("metadata")
                        return kept

                    # Shape A: result.channels
                    if isinstance(result.get("channels"), dict):
                        result = dict(result)
                        result["channels"] = _keep_ui_channels(result["channels"])
                        # Promote map_action on data into ui if missing
                        if "ui" not in result["channels"] and isinstance(result.get("data"), dict):
                            if result["data"].get("map_action"):
                                result["channels"]["ui"] = result["data"]
                        tr["result"] = result
                    else:
                        # Shape B: result.data.result.channels (double-wrapped)
                        data = result.get("data", {}) if isinstance(result.get("data"), dict) else {}
                        inner = data.get("result", {}) if isinstance(data.get("result"), dict) else {}
                        channels = inner.get("channels", {}) if isinstance(inner, dict) else {}
                        if isinstance(channels, dict) and channels:
                            inner = dict(inner)
                            data = dict(data)
                            result = dict(result)
                            kept = _keep_ui_channels(channels)
                            if "ui" not in kept and isinstance(data.get("data"), dict) and data["data"].get("map_action"):
                                kept["ui"] = data["data"]
                            elif "ui" not in kept and isinstance(inner.get("data"), dict) and inner["data"].get("map_action"):
                                kept["ui"] = inner["data"]
                            inner["channels"] = kept
                            data["result"] = inner
                            result["data"] = data
                            tr["result"] = result
                except Exception:
                    pass
                stripped.append(tr)
            return stripped

        yield {
            "type": "done",
            "trace_id": trace_id,
            "final_answer": final_state.get("final_answer") if final_state else None,
            "thinking": final_state.get("thinking") if final_state else None,
            "tool_runs": _strip_channels(accumulated_tool_runs),
            "tokens_prompt": final_state.get("tokens_prompt") if final_state else None,
            "tokens_completion": final_state.get("tokens_completion") if final_state else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        log.error(f"🔴 Stream agent error: {e}", exc_info=True)
        yield {
            "type": "error",
            "message": str(e),
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat()
        }


# ==================== Non-Streaming API ====================

async def run_agent(
    user_message: str,
    session_id: str,
    artifacts: List[Dict[str, Any]] = None,
    user_id: str = "default",
    max_steps: Optional[int] = None,
    chat_history: List[Dict[str, str]] = None,
    has_fresh_artifacts: bool = False,
    conversation_state=None,
    history_summary: str = "",
    file_registry: list = None,
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
) -> Dict[str, Any]:
    """
    Run agent without streaming (for non-streaming endpoint).
    Wraps ``stream_agent`` with the same conversation context as SSE ``/chat/stream``.
    """
    result = {
        "answer": "",
        "thinking": None,
        "tool_runs": [],
        "trace_id": str(uuid.uuid4()),
        "error": None
    }
    
    # Pick the active streaming engine. Mirrors chat_engine_resolve: native
    # tools is the only non-RLM hub chat engine still in service.
    from ..config import settings as _settings
    if _settings.NATIVE_TOOLS_ENGINE_ENABLED:
        from .native_tools_engine import stream_agent_native as _active_stream
    else:
        _active_stream = stream_agent

    async for event in _active_stream(
        user_message=user_message,
        session_id=session_id,
        artifacts=artifacts,
        user_id=user_id,
        max_steps=max_steps,
        chat_history=chat_history,
        has_fresh_artifacts=has_fresh_artifacts,
        conversation_state=conversation_state,
        history_summary=history_summary,
        file_registry=file_registry,
        pinned_facts=pinned_facts,
        workspace_runtime=workspace_runtime,
        discourse_focus=discourse_focus,
        task_checklist_state=task_checklist_state,
        rule_policies=rule_policies,
        active_plugin=active_plugin,
        host_page_prompt=host_page_prompt,
        rag_context=rag_context,
        rag_prompt_block=rag_prompt_block,
        memory_prompt_block=memory_prompt_block,
    ):
        event_type = event.get("type")
        
        if event_type == "done":
            result["answer"] = event.get("final_answer", "")
            result["thinking"] = event.get("thinking")
            result["tool_runs"] = event.get("tool_runs", [])
            result["trace_id"] = event.get("trace_id", result["trace_id"])
            result["tokens_prompt"] = event.get("tokens_prompt")
            result["tokens_completion"] = event.get("tokens_completion")
        elif event_type == "error":
            result["error"] = event.get("message")
            result["answer"] = f"Error: {event.get('message')}"
    
    return result


async def stream_agent_for_full_endpoint(
    user_message: str,
    session_id: str,
    artifacts: List[Dict[str, Any]] = None,
    user_id: str = "default",
    max_steps: Optional[int] = None,
    chat_history: List[Dict[str, str]] = None,
    has_fresh_artifacts: bool = False,
    conversation_state=None,
    history_summary: str = "",
    file_registry: list = None,
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
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Adapt ``stream_agent`` events to the shape expected by ``/chat/stream/full``
    (TOKEN chunks + final_state + planning-compatible tool events).
    This is the closest match to Continue's single ReAct streaming loop.
    """
    async for ev in stream_agent(
        user_message=user_message,
        session_id=session_id,
        artifacts=artifacts,
        user_id=user_id,
        max_steps=max_steps,
        chat_history=chat_history,
        has_fresh_artifacts=has_fresh_artifacts,
        conversation_state=conversation_state,
        history_summary=history_summary,
        file_registry=file_registry,
        pinned_facts=pinned_facts,
        workspace_runtime=workspace_runtime,
        discourse_focus=discourse_focus,
        task_checklist_state=task_checklist_state,
        rule_policies=rule_policies,
        active_plugin=active_plugin,
        host_page_prompt=host_page_prompt,
        rag_context=rag_context,
        rag_prompt_block=rag_prompt_block,
        memory_prompt_block=memory_prompt_block,
    ):
        t = ev.get("type", "")
        if t == "start":
            continue
        if t == "answer":
            chunk = ev.get("content", "")
            if chunk:
                yield {"type": "token", "content": chunk}
            continue
        if t in ("thinking", "thinking_start", "thinking_end"):
            yield ev
            continue
        if t == "tool_start":
            yield {
                "type": "tool_start",
                "tool_id": ev.get("tool_id"),
                "tool_name": ev.get("tool_name"),
                "args": ev.get("args", {}),
                "step": ev.get("step", 0),
            }
            continue
        if t == "tool_result":
            yield {
                "type": "tool_result",
                "tool_id": ev.get("tool_id"),
                "tool_name": ev.get("tool_name"),
                "ok": ev.get("ok", False),
                "execution_time": ev.get("execution_time", 0),
            }
            continue
        if t == "tool_error":
            yield {
                "type": "tool_result",
                "tool_id": ev.get("tool_id"),
                "tool_name": ev.get("tool_name"),
                "ok": False,
                "error": ev.get("error"),
                "execution_time": ev.get("execution_time", 0),
            }
            continue
        if t == "error":
            yield {
                "type": "error",
                "message": ev.get("message", ""),
                "error": ev.get("message", ""),
            }
            continue
        if t == "done":
            yield {
                "type": "final_state",
                "state": {
                    "final_answer": ev.get("final_answer") or "",
                    "tool_runs": ev.get("tool_runs", []),
                    "thinking": ev.get("thinking"),
                    "step_count": 0,
                    "tokens_prompt": ev.get("tokens_prompt"),
                    "tokens_completion": ev.get("tokens_completion"),
                },
            }
            yield ev
            continue
        if t in (
            "tool_call_start",
            "tool_call_delta",
            "tool_plan",
            "step_start",
            "answer_start",
            "llm_done",
            "llm_start",
        ):
            yield ev


__all__ = [
    "stream_agent",
    "stream_agent_for_full_endpoint",
    "run_agent",
    "AgentState",
    "build_agent_graph",
]
