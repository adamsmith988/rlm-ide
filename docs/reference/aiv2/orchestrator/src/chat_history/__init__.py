"""
Chat History Management via Storage Service
مدیریت تاریخچه چت از طریق سرویس Storage

این ماژول:
- ذخیره و بازیابی پیام‌ها از طریق Storage Service (نه مستقیم به DB)
- مدیریت session های چت
- پشتیبانی از tool calls و thinking
- خلاصه‌سازی هوشمند تاریخچه با schema ثابت + cache
"""
import asyncio
import hashlib
import json
import logging
import mimetypes
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import httpx

from ..config import settings

log = logging.getLogger(__name__)


# ==================== Structured Summary Cache ====================
#
# Re-summarising the same older block on every turn is expensive AND
# unreliable on quantized models. We cache by `(session_id, hash(older))` so
# turn N+1 — whose older-block is almost identical to turn N's — reuses the
# previous good summary instead of racing the LLM under a tight timeout.
#
# Per-process cache. Good enough for single-replica setups; scales linearly
# with sessions because we only store summaries, not raw history.

_SUMMARY_CACHE: Dict[str, Tuple[float, str]] = {}


def _older_block_hash(session_id: str, older: List[Dict[str, Any]]) -> str:
    h = hashlib.sha256()
    h.update(session_id.encode("utf-8", errors="ignore"))
    for m in older:
        role = (m.get("role") or "").encode("utf-8", errors="ignore")
        content = (m.get("content") or "").encode("utf-8", errors="ignore")
        h.update(b"|"); h.update(role); h.update(b":"); h.update(content)
    return h.hexdigest()


def _cache_get(key: str) -> Optional[str]:
    if not getattr(settings, "HISTORY_SUMMARY_CACHE_ENABLED", True):
        return None
    item = _SUMMARY_CACHE.get(key)
    if not item:
        return None
    ts, value = item
    ttl = getattr(settings, "HISTORY_SUMMARY_CACHE_TTL", 300)
    if time.monotonic() - ts > ttl:
        _SUMMARY_CACHE.pop(key, None)
        return None
    return value


def _cache_put(key: str, value: str) -> None:
    if not getattr(settings, "HISTORY_SUMMARY_CACHE_ENABLED", True):
        return
    if not value:
        return
    _SUMMARY_CACHE[key] = (time.monotonic(), value)


_LEGACY_TOOL_CONTEXT_RE = (
    # Old English wrapper, optionally preceded by ``---``.
    re.compile(
        r"(?ms)(?:\n?---\s*\n)?\s*\[Previous tool outputs from this assistant turn[^\]]*\]\s*\n(?:^\s*-\s[^\n]+\s*\n?)*"
    ),
    # Persian header (current), optionally preceded by ``---``.
    re.compile(
        r"(?ms)(?:\n?---\s*\n)?\s*\(?\s*خلاصهٔ خروجی ابزار در همین پاسخ[^\n]*\)?\s*\n(?:^\s*[-*+\u2022]?\s*[^\n]+\s*\n?)*"
    ),
)

# Standalone leaked tool-call lines (no surrounding header), e.g.
# ``- session_files({"session_id": "..."}) → در حال اجرا`` or
# ``    graph_search({...}) -> ok``. We strip them so legacy rows do not
# keep training the model on a fake-bullet pattern.
_LEGACY_TOOL_BULLET_RE = re.compile(
    r"(?m)^\s*[-*+\u2022\u25E6]?\s*[a-zA-Z_][\w.]{1,80}\s*\([\s\S]{0,400}?\)\s*[\u2192\u279C\->]+[^\n]*\n?"
)

# Fenced tool-call leaks saved in older assistant.content, e.g.
# ```tool_call
# graph_search(queries=["جعفر"])
# ```
# These are especially harmful in future turns because Qwen sees the
# assistant "using" tools as plain text instead of native tool_calls.
_LEGACY_TOOL_FENCE_RE = re.compile(
    r"(?ims)```(?:tool_call|json|python)?\s*(?:\{[\s\S]{0,2000}?tool_calls[\s\S]{0,2000}?|[a-zA-Z_][\w.]{0,120}\s*\([\s\S]{0,2000}?\))\s*```"
)


# #region agent log
def _dbg32fb75_ch(location: str, message: str, data: dict, hypothesis_id: str = "") -> None:
    """Emit a single NDJSON debug line to stdout (collected via kubectl logs)."""
    try:
        import json as _json
        import time as _time
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


def _strip_prior_tool_context_markers(text: str) -> str:
    """
    Remove "tool continuity" residue that older orchestrator versions
    appended to assistant content. The model would learn this pattern
    and start emitting it as plain prose ("- name(args) → در حال اجرا")
    instead of issuing native tool_calls — so we scrub it from both
    re-injection AND from any UI echo.

    Strips three shapes:

    1. ``--- \\n [Previous tool outputs from this assistant turn ...]``
       block, plus the bullet list that follows.
    2. ``--- \\n (خلاصهٔ خروجی ابزار در همین پاسخ ...)`` Persian variant,
       plus its bullets. The leading ``---`` is optional now since the
       model often omits it when imitating.
    3. Bare ``- name(args) → ...`` bullet lines anywhere in the text
       (e.g. when the model wrote the bullets without the header).
    """
    if not text:
        return ""
    t = text
    matched_any = False
    for rx in _LEGACY_TOOL_CONTEXT_RE:
        new_t, nsub = rx.subn("\n", t)
        if nsub:
            matched_any = True
        t = new_t
    new_t, n_fence = _LEGACY_TOOL_FENCE_RE.subn("", t)
    if n_fence:
        matched_any = True
    t = new_t
    new_t, n_bullet = _LEGACY_TOOL_BULLET_RE.subn("", t)
    if n_bullet:
        matched_any = True
    t = new_t
    t = re.sub(r"\n{3,}", "\n\n", t)
    # #region agent log
    if matched_any or any(
        h in text for h in (
            "خلاصهٔ خروجی ابزار",
            "Previous tool outputs",
            "session_files(",
        )
    ):
        _dbg32fb75_ch(
            "chat_history.__init__:_strip",
            "stripper_seen",
            {
                "matched_any": matched_any,
                "n_fence_subs": n_fence,
                "n_bullet_subs": n_bullet,
                "in_len": len(text),
                "out_len": len(t),
                "in_head": text[:240],
                "out_head": t[:240],
            },
            hypothesis_id="H1",
        )
    # #endregion
    return t.strip()


def extract_agent_state_from_messages(
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Load discourse_focus, conversation_summary, task_checklist from latest assistant metadata."""
    out: Dict[str, Any] = {}
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        meta = msg.get("metadata")
        if not isinstance(meta, dict):
            continue
        if "discourse_focus" not in out and meta.get("discourse_focus"):
            out["discourse_focus"] = meta["discourse_focus"]
        if "conversation_summary" not in out and meta.get("conversation_summary"):
            out["conversation_summary"] = meta["conversation_summary"]
        if "task_checklist" not in out and meta.get("task_checklist"):
            out["task_checklist"] = meta["task_checklist"]
        if len(out) >= 3:
            break
    return out


def _drop_superseded_session_read_errors(
    paired: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Remove failed session_read rows when a later successful read exists in the same turn."""
    read_idxs = [
        i
        for i, p in enumerate(paired)
        if p.get("name") in {
            "session_read_file",
            "session_read",
            "plugin_session_read",
        }
    ]
    if len(read_idxs) <= 1:
        return paired
    last_ok: Optional[int] = None
    for i in read_idxs:
        content = str(paired[i].get("content") or "")
        if paired[i].get("ok") and not content.startswith("[ERROR]"):
            last_ok = i
    if last_ok is None:
        return paired
    drop = {
        i
        for i in read_idxs
        if i != last_ok and str(paired[i].get("content") or "").startswith("[ERROR]")
    }
    if not drop:
        return paired
    return [p for i, p in enumerate(paired) if i not in drop]


def _dedup_session_read_tool_blocks(
    blocks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop duplicate session_read tool rows within one assistant turn."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("role") != "tool":
            out.append(block)
            continue
        name = (block.get("name") or "").strip()
        if name not in {
            "session_read_file",
            "session_read",
            "plugin_session_read",
        }:
            out.append(block)
            continue
        content = str(block.get("content") or "")[:600]
        key = (name, block.get("tool_call_id") or "", content)
        if key in seen:
            continue
        seen.add(key)
        out.append(block)
    return out


def _llm_text_from_orchestrator_tool_result(tr: Dict[str, Any], max_chars: int = 1200) -> str:
    """
    Human-readable tool text from a ``tool_runs`` row as produced by
    ``streaming.execute_tools`` (executor envelope: ``result.ok`` / ``result.data.result``).
    Graph tools expose prose under ``channels.llm`` as ``{text, summary}`` not a bare string.
    """
    for key in ("summary", "output", "text_for_llm", "content"):
        v = tr.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()[:max_chars]

    r = tr.get("result")
    if not isinstance(r, dict):
        return ""

    for key in ("summary", "output", "text_for_llm", "content"):
        v = r.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()[:max_chars]

    data = r.get("data")
    if not isinstance(data, dict):
        err = r.get("error")
        return str(err)[:max_chars] if err else ""
    inner = data.get("result")
    if not isinstance(inner, dict):
        # Some failure envelopes put channels directly under data.
        ch0 = data.get("channels")
        if isinstance(ch0, dict):
            llm0 = ch0.get("llm")
            if isinstance(llm0, str) and llm0.strip():
                return llm0.strip()[:max_chars]
            if isinstance(llm0, dict):
                s0 = (llm0.get("text") or llm0.get("summary") or "").strip()
                if s0:
                    return s0[:max_chars]
        err = r.get("error") or data.get("error")
        return str(err)[:max_chars] if err else ""
    ch = inner.get("channels")
    if not isinstance(ch, dict):
        ch = {}
    llm = ch.get("llm")
    if isinstance(llm, str):
        s = llm.strip()
    elif isinstance(llm, dict):
        s = (llm.get("text") or llm.get("summary") or "").strip()
    else:
        s = ""
    if s:
        return s[:max_chars]
    # Fallbacks when channels.llm is empty but structured data exists (e.g. some paths).
    data_block = inner.get("data")
    if isinstance(data_block, dict):
        tot = data_block.get("total")
        nfiles = data_block.get("files")
        if isinstance(tot, int) and isinstance(nfiles, list):
            return f"workspace files: total={tot} page_rows={len(nfiles)}"[:max_chars]
        if isinstance(tot, int):
            return f"tool output: total={tot}"[:max_chars]
    return ""


# ==================== Structured Summary Prompt ====================
#
# The previous prompt asked for "3-6 bullet points" — which the LLM
# interpreted very loosely, often dropping the goal/constraints. The
# replacement enforces a *fixed schema* with the same fields the
# pinned-facts module produces, so even if pinned-facts is disabled we
# still recover Goal / Constraints / Decisions reliably.

_SUMMARY_SYSTEM_PROMPT = (
    "You are a deterministic conversation summariser. Produce a strict, "
    "compact summary preserving exactly these sections: Goal, Constraints, "
    "Decisions made, Open questions, Files referenced. Do NOT invent. Do "
    "NOT include greetings or acknowledgements. /no_think"
)

_SUMMARY_USER_TEMPLATE = (
    "Summarise the conversation below using EXACTLY this Markdown skeleton "
    "(omit a section only when truly empty):\n\n"
    "## Goal\n<one sentence>\n\n"
    "## Constraints\n- ...\n\n"
    "## Decisions made\n- ...\n\n"
    "## Open questions\n- ...\n\n"
    "## Files referenced\n- ...\n\n"
    "Keep each bullet under 25 words. Prefer the user's own wording for "
    "constraints. Skip greetings.\n\n"
    "Conversation:\n{conversation}"
)


class ChatHistoryClient:
    """
    کلاینت مدیریت تاریخچه چت از طریق Storage Service.
    
    این کلاس هیچ اتصال مستقیمی به دیتابیس ندارد و همه عملیات
    از طریق HTTP به Storage Service ارسال می‌شود.
    """
    
    _instance: Optional["ChatHistoryClient"] = None
    
    def __init__(self):
        self.storage_url = settings.STORAGE_URL
        self.timeout = 10.0

    def _storage_headers(
        self,
        auth_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Headers for internal calls from orchestrator to storage."""
        headers = {
            "X-Internal-Key": settings.INTERNAL_API_KEY,
            "X-Service-Name": settings.SERVICE_NAME,
        }
        ctx = auth_context or {}
        if ctx.get("user_id"):
            headers["X-User-Id"] = str(ctx["user_id"])
        if ctx.get("group_id"):
            headers["X-Group-Id"] = str(ctx["group_id"])
        if ctx.get("case_id"):
            headers["X-Case-Id"] = str(ctx["case_id"])
        return headers
    
    @classmethod
    def get_instance(cls) -> "ChatHistoryClient":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            log.info(f"✅ ChatHistoryClient initialized (via Storage: {cls._instance.storage_url})")
        return cls._instance
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict] = None,
        params: Optional[Dict] = None,
        auth_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Make HTTP request to Storage Service."""
        url = f"{self.storage_url}{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    json=json,
                    params=params,
                    headers=self._storage_headers(auth_context),
                )
                
                if response.status_code >= 400:
                    log.warning(f"Storage request failed: {method} {endpoint} -> {response.status_code}")
                    return None
                
                return response.json()
        except Exception as e:
            log.error(f"Storage request error: {method} {endpoint} -> {e}")
            return None
    
    # ==================== Message Management ====================
    
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        thinking: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
        tool_results: Optional[List[Dict]] = None,
        tokens_prompt: Optional[int] = None,
        tokens_completion: Optional[int] = None,
        attachments: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
        reasoning_segments: Optional[List[Dict]] = None,
        user_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        افزودن پیام جدید از طریق Storage Service.
        
        Args:
            session_id: شناسه session
            role: نقش (user, assistant, system, tool)
            content: محتوای پیام
            thinking: تفکر مدل (متن کامل joined)
            tool_calls: ابزارهایی که LLM فراخوانی کرده [{name, arguments}]
            tool_results: خروجی اجرای ابزارها [{name, ok, data, error}]
            tokens_prompt: توکن‌های پرامپت (ورودی LLM)
            tokens_completion: توکن‌های تکمیل (خروجی LLM)
            attachments: فایل‌های ضمیمه (artifacts)
            metadata: اطلاعات اضافی
            reasoning_segments: قطعات تفکر برای حفظ interleaving
            trace_id: شناسه trace برای جزئیات اجرا
        """
        # اگر reasoning_segments داده شده، به metadata اضافه کن
        if reasoning_segments and metadata is None:
            metadata = {}
        if reasoning_segments:
            metadata["reasoning_segments"] = reasoning_segments
        
        result = await self._request(
            method="POST",
            endpoint="/chat/messages",
            json={
                "session_id": session_id,
                "role": role,
                "content": content,
                "thinking": thinking,
                "tool_calls": tool_calls,
                "tool_results": tool_results,
                "tokens_prompt": tokens_prompt,
                "tokens_completion": tokens_completion,
                "attachments": attachments,
                "metadata": metadata,
                "user_id": user_id,
                "trace_id": trace_id
            }
        )
        
        if result:
            log.debug(f"💾 Saved message to session {session_id} via Storage")
        
        return result
    
    # Note: save_tool_run has been deprecated - tool executions are saved 
    # via TraceClient to agent_tool_executions table
    
    async def get_messages(
        self,
        session_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        دریافت پیام‌های یک session از Storage Service.
        
        Args:
            session_id: شناسه session
            limit: حداکثر تعداد پیام‌ها
            
        Returns:
            لیست پیام‌ها به ترتیب زمان (قدیمی به جدید)
        """
        # Storage pages oldest first; walk the pages and keep the newest
        # ``limit`` rows so long sessions return their latest turns.
        page = max(1, int(settings.STORAGE_MESSAGES_PAGE_SIZE))
        result: Optional[List[Dict[str, Any]]] = None
        offset = 0
        while True:
            batch = await self._request(
                method="GET",
                endpoint=f"/chat/sessions/{session_id}/messages",
                params={"limit": page, "offset": offset},
            )
            if not isinstance(batch, list):
                break
            result = ((result or []) + batch)[-max(1, int(limit)):]
            if len(batch) < page:
                break
            offset += page

        if result is None:
            return []

        # Storage's GET /chat/sessions/{id}/messages returns rows ordered
        # ASC (oldest → newest). Every downstream caller in this module
        # assumes that exact order — see ``get_messages_for_llm_v2`` (slices
        # the *tail* as "recent"), ``get_session_attachments`` (does its
        # own ``reversed(...)`` to get newest-first), and
        # ``get_session_file_registry`` (assigns ordinals oldest → newest).
        #
        # An earlier version of the orchestrator did ``list(reversed(...))``
        # here based on a stale assumption that storage returned DESC.
        # That single line was the root cause of the admin/admin123
        # drift bug: it flipped the list, so "recent" turned into
        # "oldest" and the LLM kept seeing the *original* topic as its
        # most-recent context — guaranteeing it would re-generate the
        # original answer regardless of what the user asked next.
        if not isinstance(result, list):
            return []
        return result
    
    async def get_messages_for_llm(
        self,
        session_id: str,
        limit: int = 20,
        recent_full_count: int = 4  # تعداد پیام‌های اخیر که کامل ارسال می‌شن
    ) -> List[Dict[str, Any]]:
        """
        دریافت پیام‌ها در فرمت مناسب برای LLM با فشرده‌سازی هوشمند.
        
        ✨ استراتژی:
        - پیام‌های اخیر (recent_full_count عدد آخر): کامل ارسال می‌شن
        - پیام‌های قدیمی‌تر: فقط خلاصه می‌شن (برای جلوگیری از گیجی LLM)
        
        Returns:
            لیست پیام‌ها با ساختار {"role": "...", "content": "...", "attachments": [...]}
        """
        messages = await self.get_messages(session_id, limit)
        
        if not messages:
            return []
        
        llm_messages = []
        total_msgs = len(messages)
        
        for idx, msg in enumerate(messages):
            if msg.get("role") not in ("user", "assistant"):
                continue
            
            # ✨ محاسبه فاصله از آخرین پیام
            distance_from_end = total_msgs - idx - 1
            
            content = msg.get("content", "")
            
            # ✨ پیام‌های قدیمی رو فشرده کن (نه پیام‌های اخیر)
            if distance_from_end >= recent_full_count:
                # فشرده‌سازی: فقط 150 کاراکتر اول
                if len(content) > 150:
                    content = content[:150] + "..."
            
            msg_dict = {
                "role": msg["role"],
                "content": content
            }
            
            # اضافه کردن attachments (فقط برای پیام‌های اخیر)
            if distance_from_end < recent_full_count:
                attachments = msg.get("attachments")
                if attachments:
                    msg_dict["attachments"] = attachments
            
            llm_messages.append(msg_dict)
        
        return llm_messages
    
    # ==================== Smart History Summarization ====================

    async def get_messages_for_llm_v2(
        self,
        session_id: str,
        limit: int = 20,
        recent_full_count: Optional[int] = None,
        llm_client=None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Hybrid history: structured-LLM-summarised older messages + verbatim
        recent messages.

        Args:
            session_id: storage session id
            limit: total messages to pull
            recent_full_count: minimum number of recent user/assistant
                messages to keep verbatim. Defaults to
                ``settings.HISTORY_RECENT_FULL_COUNT``.
            llm_client: optional LLM client used for summarisation.

        Returns:
            (recent_messages, history_summary)
        """
        if recent_full_count is None:
            recent_full_count = getattr(settings, "HISTORY_RECENT_FULL_COUNT", 6)

        messages = await self.get_messages(session_id, limit)
        if not messages:
            return [], ""

        chat_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]
        total = len(chat_msgs)

        if total <= recent_full_count:
            recent = []
            for msg in chat_msgs:
                d = {"role": msg["role"], "content": msg.get("content", "")}
                att = msg.get("attachments")
                if att:
                    d["attachments"] = att
                recent.append(d)
            return recent, ""

        older = chat_msgs[: total - recent_full_count]
        recent_raw = chat_msgs[total - recent_full_count :]

        recent = []
        for msg in recent_raw:
            d = {"role": msg["role"], "content": msg.get("content", "")}
            att = msg.get("attachments")
            if att:
                d["attachments"] = att
            recent.append(d)

        # ---- Cache lookup ------------------------------------------------- #
        cache_key = _older_block_hash(session_id, older)
        cached = _cache_get(cache_key)
        if cached:
            log.debug(
                f"📚 Reusing cached summary for session={session_id} "
                f"(older={len(older)} msgs)"
            )
            return recent, cached

        # ---- Generate summary (LLM with retry → rule-based fallback) ----- #
        summary = ""
        if settings.HISTORY_SUMMARY_ENABLED:
            summary = await self._summarize_messages(older, llm_client)

        if not summary:
            summary = self._rule_based_summary(older)

        if summary:
            _cache_put(cache_key, summary)

        return recent, summary

    @staticmethod
    def _assistant_turn_blocks_for_stream_llm(
        msg: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Convert a stored ``assistant`` row (with its ``tool_calls`` and
        ``tool_results`` columns) into native OpenAI-shaped messages — the
        same pattern Cline / OpenAI / vLLM / Qwen all use:

            {role: "assistant", content: text, tool_calls: [...]}
            {role: "tool", tool_call_id: id, name: name, content: summary}
            {role: "tool", tool_call_id: id2, name: ..., content: summary2}

        Why this matters:

        - Older versions of this method *embedded* a bullet list like
          ``- session_files({...}) → خلاصه`` into ``assistant.content``.
          The model picked that pattern up from prior turns and started
          IMITATING it instead of emitting real tool_calls — answering
          with literal text ``session_files({...}) → در حال اجرا`` and
          ending the turn.
        - Returning structured messages instead lets vLLM's chat template
          (Qwen3-coder) render the past tools as ``<tool_call>`` and
          ``<tool_response>`` blocks, exactly how the model was trained.

        Always returns at least one element (the assistant message).
        """
        clean_text = _strip_prior_tool_context_markers(msg.get("content") or "")
        raw_tool_calls = msg.get("tool_calls") or []
        raw_tool_results = msg.get("tool_results") or []

        if not raw_tool_calls and not raw_tool_results:
            return [{"role": "assistant", "content": clean_text}]

        # Build assistant message with OpenAI-shaped tool_calls.
        oa_tool_calls: List[Dict[str, Any]] = []
        for i, tc in enumerate(raw_tool_calls):
            tc = tc or {}
            args = tc.get("arguments") if "arguments" in tc else tc.get("args")
            if isinstance(args, dict):
                args = {k: v for k, v in args.items() if k != "session_id"}
            if isinstance(args, str):
                args_str = args
            else:
                try:
                    args_str = json.dumps(args or {}, ensure_ascii=False)
                except Exception:
                    args_str = "{}"
            oa_tool_calls.append(
                {
                    "id": str(tc.get("id") or f"call_{i}"),
                    "type": "function",
                    "function": {
                        "name": (tc.get("name") or "").strip(),
                        "arguments": args_str,
                    },
                }
            )

        blocks: List[Dict[str, Any]] = [
            {
                "role": "assistant",
                "content": clean_text,
                "tool_calls": oa_tool_calls,
            }
        ]

        # Pair each tool_call with the matching tool_result (by id, then
        # by index), so the chat template renders them as
        # ``<tool_response>`` blocks right after the assistant.
        results_by_id: Dict[str, Dict[str, Any]] = {}
        results_in_order: List[Dict[str, Any]] = []
        for tr in raw_tool_results:
            if not isinstance(tr, dict):
                continue
            tid = tr.get("id")
            if tid:
                results_by_id[str(tid)] = tr
            results_in_order.append(tr)

        paired: List[Dict[str, Any]] = []
        for i, oa_tc in enumerate(oa_tool_calls):
            tid = oa_tc["id"]
            tr = results_by_id.get(tid)
            if tr is None and i < len(results_in_order):
                tr = results_in_order[i]
            tr = tr or {}
            tname = oa_tc["function"]["name"] or (tr.get("name") or "tool")
            summary = (tr.get("summary") or "").strip()
            err = tr.get("error")
            ok = tr.get("ok", bool(summary) and not err)
            if summary:
                content = summary
            elif err:
                content = f"error: {err}"
            elif ok:
                content = "ok"
            else:
                content = "no result recorded"
            paired.append(
                {
                    "role": "tool",
                    "tool_call_id": tid,
                    "name": tname,
                    "content": content[:1200],
                    "ok": ok,
                }
            )

        paired = _drop_superseded_session_read_errors(paired)
        for p in paired:
            blocks.append(
                {
                    "role": p["role"],
                    "tool_call_id": p["tool_call_id"],
                    "name": p["name"],
                    "content": p["content"],
                }
            )

        return _dedup_session_read_tool_blocks(blocks)

    def _recent_dicts_for_stream(
        self, chat_msgs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build recent message dicts for /chat/stream with native tool continuity."""
        recent: List[Dict[str, Any]] = []
        for msg in chat_msgs:
            role = msg.get("role")
            if role == "user":
                d: Dict[str, Any] = {
                    "role": "user",
                    "content": msg.get("content", ""),
                }
                att = msg.get("attachments")
                if att:
                    d["attachments"] = att
                recent.append(d)
            elif role == "assistant":
                recent.extend(self._assistant_turn_blocks_for_stream_llm(msg))
        return recent

    async def get_messages_for_llm_stream(
        self,
        session_id: str,
        limit: int = 20,
        recent_full_count: Optional[int] = None,
        llm_client=None,
        *,
        prefetched_messages: Optional[List[Dict[str, Any]]] = None,
        lazy_summary: bool = False,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Same hybrid strategy as get_messages_for_llm_v2, but assistant rows
        include summarized tool outputs so the streaming LangGraph sees what
        tools returned in prior turns.

        Performance knobs (used by the workspace preflight):

        * ``prefetched_messages`` — when the caller has already fetched the
          raw ``/messages`` array, pass it here to skip the HTTP round-trip.
        * ``lazy_summary`` — when True, on a cache miss we return the cheap
          rule-based fallback immediately AND fire the LLM summary in a
          detached background task (its result is cached for the next turn).
          This makes the preflight never block on a ~1s LLM round trip.
        """
        if recent_full_count is None:
            recent_full_count = getattr(settings, "HISTORY_RECENT_FULL_COUNT", 6)

        messages = (
            prefetched_messages
            if prefetched_messages is not None
            else await self.get_messages(session_id, limit)
        )
        if not messages:
            return [], ""

        chat_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]
        total = len(chat_msgs)

        if total <= recent_full_count:
            return self._recent_dicts_for_stream(chat_msgs), ""

        older = chat_msgs[: total - recent_full_count]
        recent_raw = chat_msgs[total - recent_full_count :]

        recent = self._recent_dicts_for_stream(recent_raw)

        cache_key = _older_block_hash(session_id, older)
        cached = _cache_get(cache_key)
        if cached:
            log.debug(
                f"📚 [stream] Reusing cached summary for session={session_id} "
                f"older={len(older)}"
            )
            return recent, cached

        if lazy_summary and settings.HISTORY_SUMMARY_ENABLED and llm_client is not None:
            # Continue-style: never block the user-visible turn on a summary
            # LLM call. Return a deterministic rule-based fallback now, and
            # warm the cache for the next turn in the background.
            fallback = self._rule_based_summary(older) or ""
            try:
                asyncio.create_task(
                    self._background_summarize_and_cache(
                        cache_key=cache_key,
                        session_id=session_id,
                        older=older,
                        llm_client=llm_client,
                    )
                )
                log.debug(
                    "📚 [stream] Scheduled background summary for session=%s older=%d (using rule-based now)",
                    session_id,
                    len(older),
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not schedule background summary: %s", exc)
            return recent, fallback

        summary = ""
        if settings.HISTORY_SUMMARY_ENABLED:
            summary = await self._summarize_messages(older, llm_client)

        if not summary:
            summary = self._rule_based_summary(older)

        checkpoint = extract_agent_state_from_messages(messages).get("conversation_summary")
        if isinstance(checkpoint, str) and checkpoint.strip():
            if not summary or len(checkpoint) > len(summary):
                summary = checkpoint.strip()

        if summary:
            _cache_put(cache_key, summary)

        return recent, summary

    async def _background_summarize_and_cache(
        self,
        *,
        cache_key: str,
        session_id: str,
        older: List[Dict[str, Any]],
        llm_client: Any,
    ) -> None:
        """Run the LLM summary off the request hot path and cache the result.

        Errors are swallowed — the rule-based summary already shipped to the
        user, so a missed enrichment is non-fatal. The cache key is content-
        addressed (hash of ``older``), so the next turn that sees the same
        prefix automatically picks this up.
        """
        try:
            summary = await self._summarize_messages(older, llm_client)
            if not summary:
                summary = self._rule_based_summary(older)
            if summary:
                _cache_put(cache_key, summary)
                log.debug(
                    "📚 [stream] Background summary cached for session=%s older=%d",
                    session_id,
                    len(older),
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("Background history summary failed: %s", exc)

    async def _summarize_messages(
        self,
        messages: List[Dict[str, Any]],
        llm_client=None,
    ) -> str:
        """
        Produce a *structured* summary of older messages using the LLM, with
        a generous-but-bounded timeout and one retry. On total failure
        returns "" so the caller can fall back to the rule-based path.
        """
        if not llm_client or not messages:
            return ""

        lines: List[str] = []
        for m in messages:
            role = "User" if m.get("role") == "user" else "Assistant"
            content = (m.get("content") or "")[:400]
            atts = m.get("attachments") or []
            att_str = ""
            if isinstance(atts, list) and atts:
                names = [a.get("name", a.get("path", "?")) for a in atts if isinstance(a, dict)]
                if names:
                    att_str = f"  [files: {', '.join(names[:3])}]"
            lines.append(f"{role}: {content}{att_str}")
        conversation_text = "\n".join(lines)

        prompt_messages = [
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _SUMMARY_USER_TEMPLATE.format(conversation=conversation_text),
            },
        ]

        timeout = getattr(settings, "HISTORY_SUMMARY_LLM_TIMEOUT", 8.0)
        retries = max(0, int(getattr(settings, "HISTORY_SUMMARY_LLM_RETRIES", 1)))

        last_err: Optional[str] = None
        for attempt in range(retries + 1):
            try:
                result = await asyncio.wait_for(
                    llm_client.chat(
                        messages=prompt_messages,
                        max_tokens=500,
                        temperature=0.0,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    ),
                    timeout=timeout,
                )
                if result and result.get("ok"):
                    content = (result.get("content") or "").strip()
                    if content:
                        log.debug(
                            f"📝 LLM summary OK on attempt {attempt + 1} "
                            f"({len(messages)} older msgs → {len(content)} chars)"
                        )
                        return content
                last_err = (result or {}).get("error") or "empty content"
            except asyncio.TimeoutError:
                last_err = f"timeout after {timeout}s"
                log.warning(
                    f"⏱️ LLM summary timed out (attempt {attempt + 1}/{retries + 1})"
                )
            except Exception as e:
                last_err = str(e)
                log.warning(
                    f"⚠️ LLM summary error on attempt {attempt + 1}/{retries + 1}: {e}"
                )

        log.warning(f"LLM summary giving up — using rule-based fallback ({last_err})")
        return ""

    @staticmethod
    def _rule_based_summary(messages: List[Dict[str, Any]]) -> str:
        """
        Structured deterministic summary used when the LLM path fails.

        Produces the same Markdown sections the LLM is asked to produce, so
        downstream consumers (and the model itself) see a consistent format
        regardless of which path generated the summary.
        """
        _skip_pattern = re.compile(
            r"^(سلام|درود|hello|hi|hey|ممنون|مرسی|thanks|thank you|"
            r"بله|آره|ok|okay|yes|no|نه)\s*[.!؟?]*$",
            re.IGNORECASE | re.UNICODE,
        )
        _constraint_re = re.compile(
            r"\b(must|should|need to|have to|require[ds]?|only|don't|do not|"
            r"avoid|never|always|prefer)\b|(باید|نباید|حتما|حتماً|فقط|تنها|"
            r"بدون|ترجیحاً|ترجیحا)",
            re.IGNORECASE | re.UNICODE,
        )
        _decision_re = re.compile(
            r"\b(I will|I'll|I'm going to|let's|we'll|decision is|chose|chosen|"
            r"decided)\b|(تصمیم گرفتم|انتخاب کردم|انجام می‌دهم|قرار شد|"
            r"ادامه می‌دهیم)",
            re.IGNORECASE | re.UNICODE,
        )

        goal: str = ""
        constraints: List[str] = []
        decisions: List[str] = []
        files: List[str] = []
        questions: List[str] = []

        def _seen_add(bucket: List[str], item: str, cap: int) -> None:
            key = re.sub(r"\s+", " ", item).strip().lower()
            if not key or key in {re.sub(r"\s+", " ", x).strip().lower() for x in bucket}:
                return
            if len(bucket) >= cap:
                return
            bucket.append(item.strip())

        for msg in messages:
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            role = msg.get("role", "")
            if _skip_pattern.match(content):
                continue

            if role == "user":
                if not goal:
                    first = content.splitlines()[0]
                    goal = re.split(r"[\.\!\?\؟]", first, maxsplit=1)[0][:240].strip()
                for line in re.split(r"[\n\.؟\?\!]+", content):
                    line = line.strip()
                    if 4 <= len(line) <= 220 and _constraint_re.search(line):
                        _seen_add(constraints, line, 6)

            elif role == "assistant":
                for line in re.split(r"[\n\.؟\?\!]+", content):
                    line = line.strip()
                    if 6 <= len(line) <= 220 and _decision_re.search(line):
                        _seen_add(decisions, line, 6)
                # Last assistant question — likely an open question if user
                # never replied. We just keep the most recent pending ones.
                for line in content.splitlines():
                    line = line.strip()
                    if 6 <= len(line) <= 200 and line.endswith(("?", "؟")):
                        _seen_add(questions, line, 4)

            att = msg.get("attachments")
            if att and isinstance(att, list):
                for a in att:
                    if isinstance(a, dict):
                        nm = a.get("name") or a.get("path") or ""
                        if nm:
                            _seen_add(files, str(nm), 8)

        sections: List[str] = []
        if goal:
            sections.append(f"## Goal\n{goal}")
        if constraints:
            sections.append("## Constraints\n" + "\n".join(f"- {c}" for c in constraints))
        if decisions:
            sections.append("## Decisions made\n" + "\n".join(f"- {d}" for d in decisions))
        if questions:
            sections.append("## Open questions\n" + "\n".join(f"- {q}" for q in questions))
        if files:
            sections.append("## Files referenced\n" + "\n".join(f"- {f}" for f in files))

        return "\n\n".join(sections)

    async def get_session_attachments(
        self,
        session_id: str,
        limit: int = 20,
        auth_context: Optional[Dict[str, Any]] = None,
        *,
        prefetched_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        دریافت تمام فایل‌های ضمیمه شده در یک session.
        
        این برای زمانی استفاده می‌شه که کاربر در مورد فایل قبلی سوال می‌پرسه.
        
        Returns:
            لیست artifacts از پیام‌های قبلی (جدیدترین اول)
        """
        messages = (
            prefetched_messages
            if prefetched_messages is not None
            else await self.get_messages(session_id, limit)
        )

        all_attachments = []
        # پیام‌ها از قدیم به جدید مرتب شده‌اند، reverse می‌کنیم تا جدیدترین اول باشه
        for msg in reversed(messages):
            attachments = msg.get("attachments")
            if attachments:
                # اگر string هست، parse کن (برای سازگاری با storage قدیمی)
                if isinstance(attachments, str):
                    try:
                        import json
                        attachments = json.loads(attachments)
                        log.debug(f"📎 Parsed string attachments: {len(attachments)} items")
                    except:
                        log.warning(f"⚠️ Failed to parse attachments string: {attachments[:50]}...")
                        continue
                
                # Validate که attachments لیست از dict باشه
                if isinstance(attachments, list):
                    for att in attachments:
                        if isinstance(att, dict) and att.get("path"):
                            all_attachments.append(att)
                        elif isinstance(att, str):
                            log.warning(f"⚠️ Skipping string attachment item: {att[:50]}...")
                        else:
                            log.warning(f"⚠️ Skipping invalid attachment: {type(att)}")
        
        log.info(f"📎 Found {len(all_attachments)} valid attachments in session {session_id}")
        return all_attachments
    
    async def get_session_file_registry(
        self,
        session_id: str,
        limit: int = 20,
        auth_context: Optional[Dict[str, Any]] = None,
        *,
        prefetched_messages: Optional[List[Dict[str, Any]]] = None,
        prefetched_storage_artifacts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build a chronological registry of every file uploaded in a session,
        enriched with accumulated knowledge from previous tool analyses.

        Returns a list ordered oldest-first.  Each entry contains:
            ordinal  – 1-based upload index across the whole session
            turn     – which user turn it was uploaded in
            name / path / mime_type / type  – from the stored attachment
            knowledge – list of {tool, summary} from previous tool runs

        When ``prefetched_messages`` / ``prefetched_storage_artifacts`` are
        supplied the corresponding HTTP fetches are skipped — used by the
        session-workspace preflight to share a single ``/messages`` and a
        single ``/artifacts`` round-trip across all three derivation paths.
        """
        messages = (
            prefetched_messages
            if prefetched_messages is not None
            else await self.get_messages(session_id, limit)
        )

        # Phase 1: build basic registry from user-message attachments
        registry: List[Dict[str, Any]] = []
        turn_idx = 0
        file_ordinal = 0

        for msg in messages:                         # oldest → newest
            if msg.get("role") != "user":
                continue
            turn_idx += 1

            attachments = msg.get("attachments")
            if not attachments:
                continue

            if isinstance(attachments, str):
                try:
                    import json as _json
                    attachments = _json.loads(attachments)
                except Exception:
                    continue

            if not isinstance(attachments, list):
                continue

            for att in attachments:
                if not isinstance(att, dict):
                    continue
                path = (
                    att.get("path")
                    or att.get("storage_path")
                    or att.get("file_path")
                )
                aid = att.get("id") or att.get("artifact_id")
                if not path and not aid:
                    continue
                name = (
                    att.get("name")
                    or att.get("filename")
                    or att.get("original_filename")
                    or "unknown"
                )
                file_ordinal += 1
                mime = att.get("mime_type") or att.get("content_type") or ""
                if not mime and path:
                    mime, _ = mimetypes.guess_type(str(path))
                    mime = mime or ""
                if not mime and name:
                    mime, _ = mimetypes.guess_type(str(name))
                    mime = mime or ""
                ftype = self._mime_to_category(mime)
                if ftype == "unknown" and name:
                    guessed, _ = mimetypes.guess_type(str(name))
                    if guessed:
                        ftype = self._mime_to_category(guessed)
                registry.append({
                    "ordinal": file_ordinal,
                    "turn": turn_idx,
                    "name": name,
                    "path": path or "",
                    "mime_type": mime,
                    "type": ftype,
                    "artifact_id": aid,
                    "knowledge": [],
                    "source": "message_attachment",
                })

        # Phase 2: extract knowledge from tool results in assistant messages
        file_knowledge: Dict[str, List[Dict[str, str]]] = {}
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls") or []
            tool_results = msg.get("tool_results") or []
            for tc, tr in zip(tool_calls, tool_results):
                args = tc.get("arguments") or {}
                path = args.get("path") or args.get("file_path") or ""
                summary = (tr.get("summary") or "").strip()
                tool_name = tc.get("name") or tr.get("name") or ""
                if path and summary and tr.get("ok"):
                    file_knowledge.setdefault(path, []).append({
                        "tool": tool_name,
                        "summary": summary[:300],
                    })

        # Phase 3: merge knowledge into registry entries
        for entry in registry:
            entry_path = entry.get("path", "")
            entry_name = entry.get("name", "")
            for known_path, analyses in file_knowledge.items():
                if (known_path == entry_path
                        or known_path.endswith("/" + entry_name)
                        or entry_path.endswith("/" + known_path.split("/")[-1])):
                    seen_tools = {k["tool"] for k in entry["knowledge"]}
                    for a in analyses:
                        if a["tool"] not in seen_tools:
                            entry["knowledge"].append(a)
                            seen_tools.add(a["tool"])
                    break

        analyzed = sum(1 for e in registry if e["knowledge"])
        log.debug(
            f"📋 File registry: {len(registry)} files across {turn_idx} turns, "
            f"{analyzed} with accumulated knowledge"
        )

        # Phase 4: merge storage session artifacts (uploads + session_attach links)
        try:
            if prefetched_storage_artifacts is not None:
                storage_arts = prefetched_storage_artifacts
            else:
                storage_arts = await self._request(
                    "GET",
                    f"/chat/sessions/{session_id}/artifacts",
                    params={"limit": 500, "offset": 0},
                    auth_context=auth_context,
                )
            if isinstance(storage_arts, list):
                seen_ids = {
                    str(e.get("artifact_id") or e.get("id") or "")
                    for e in registry
                }
                seen_paths = {str(e.get("path") or "") for e in registry if e.get("path")}
                for art in storage_arts:
                    if not isinstance(art, dict):
                        continue
                    aid = str(art.get("id") or "")
                    spath = str(art.get("storage_path") or art.get("path") or "")
                    if (aid and aid in seen_ids) or (spath and spath in seen_paths):
                        continue
                    filename = (
                        art.get("filename")
                        or art.get("name")
                        or art.get("original_filename")
                        or "unnamed"
                    )
                    mime = art.get("mime_type") or ""
                    if not mime:
                        guessed, _ = mimetypes.guess_type(str(filename))
                        mime = guessed or ""
                    media_type = str(art.get("media_type") or "").strip().lower()
                    category = media_type or self._mime_to_category(mime)
                    if category == "unknown" and filename:
                        guessed, _ = mimetypes.guess_type(str(filename))
                        if guessed:
                            category = self._mime_to_category(guessed)
                    file_ordinal += 1
                    registry.append({
                        "ordinal": file_ordinal,
                        "turn": turn_idx or 0,
                        "name": filename,
                        "path": spath,
                        "mime_type": mime,
                        "type": category,
                        "artifact_id": aid or None,
                        "knowledge": [],
                        "source": "storage_session",
                    })
                    if aid:
                        seen_ids.add(aid)
                    if spath:
                        seen_paths.add(spath)
        except Exception as exc:
            log.warning("Failed to merge storage session artifacts into registry: %s", exc)

        return registry

    @staticmethod
    def _mime_to_category(mime: str) -> str:
        if not mime:
            return "unknown"
        m = mime.lower()
        if m.startswith("image/"):
            return "image"
        if m.startswith("audio/"):
            return "audio"
        if m.startswith("video/"):
            return "video"
        if m.startswith("text/") or "pdf" in m or "document" in m or "word" in m:
            return "document"
        return "unknown"

    async def delete_session_messages(self, session_id: str) -> bool:
        """حذف تمام پیام‌های یک session."""
        result = await self._request(
            method="DELETE",
            endpoint=f"/chat/sessions/{session_id}/messages"
        )
        return result is not None


# ==================== Global Instance ====================

_chat_history_client: Optional[ChatHistoryClient] = None


def get_chat_history_client() -> ChatHistoryClient:
    """Get global chat history client instance."""
    global _chat_history_client
    if _chat_history_client is None:
        _chat_history_client = ChatHistoryClient.get_instance()
    return _chat_history_client


async def save_user_message(
    session_id: str,
    user_id: str,
    user_message: str,
    attachments: Optional[List[Dict]] = None,
) -> None:
    """Persist the user turn as soon as the stream starts (before the engine runs).

    Refresh / SSE abort mid-turn must not erase the user's message from the session.
    Pair with ``save_chat_turn(..., skip_user=True)`` for the assistant reply.
    """
    if not settings.CHAT_HISTORY_ENABLED:
        return
    text = (user_message or "").strip()
    if not text and not attachments:
        return
    try:
        client = get_chat_history_client()
        await client.add_message(
            session_id=session_id,
            role="user",
            content=user_message or "",
            attachments=attachments,
            user_id=user_id,
        )
        log.debug(
            "💾 Saved user message early session=%s attachments=%s",
            session_id,
            len(attachments or []),
        )
    except Exception as e:
        log.warning("Failed to save early user message: %s", e)


async def save_chat_turn(
    session_id: str,
    user_id: str,
    user_message: str,
    assistant_response: str,
    thinking: Optional[str] = None,
    tool_runs: Optional[List[Dict]] = None,
    attachments: Optional[List[Dict]] = None,
    metadata: Optional[Dict] = None,
    tokens_prompt: Optional[int] = None,
    tokens_completion: Optional[int] = None,
    trace_id: Optional[str] = None,
    *,
    skip_user: bool = False,
):
    """
    ذخیره یک turn کامل چت (پیام کاربر + پاسخ) از طریق Storage Service.
    
    Args:
        tool_runs: نتایج اجرای ابزارها از state [{id, tool_id, args, result, step}]
        tokens_prompt: مجموع توکن‌های ورودی LLM در این turn
        tokens_completion: مجموع توکن‌های خروجی LLM در این turn
        attachments: فایل‌های ضمیمه شده به پیام کاربر (artifacts)
        trace_id: شناسه trace برای لینک به جزئیات اجرا
        skip_user: If True, only save the assistant row (user already saved early).
    """
    if not settings.CHAT_HISTORY_ENABLED:
        return
    
    # Extract trace_id from metadata if not provided directly
    if trace_id is None and metadata:
        trace_id = metadata.get("trace_id")
    
    try:
        client = get_chat_history_client()

        response_text = (assistant_response or "").strip()
        if not response_text and tool_runs:
            parts: List[str] = []
            for tr in tool_runs:
                snippet = _llm_text_from_orchestrator_tool_result(tr, 4000)
                if snippet:
                    parts.append(snippet)
            if parts:
                response_text = "\n\n".join(parts)[:8000]
        
        # ذخیره پیام کاربر (با attachments) — skipped when already saved at stream start
        if not skip_user:
            await client.add_message(
                session_id=session_id,
                role="user",
                content=user_message,
                attachments=attachments,
                user_id=user_id
            )
        
        # تبدیل tool_runs به دو لیست جداگانه برای DB:
        # tool_calls = ابزارهایی که LLM انتخاب کرده (name + arguments)
        # tool_results = خلاصه خروجی هر ابزار (بدون data کامل - آن در agent_tool_executions است)
        db_tool_calls = None
        db_tool_results = None
        if tool_runs:
            db_tool_calls = [
                {
                    "id": tr.get("id", ""),
                    "name": tr.get("tool_id", ""),
                    "arguments": tr.get("args", {}),
                    "step": tr.get("step", 0),
                }
                for tr in tool_runs
            ]
            # Note: Full tool result data is stored in agent_tool_executions table
            # Here we only store summary for chat UI display (no duplication)
            db_tool_results = [
                {
                    "id": tr.get("id", ""),
                    "name": tr.get("tool_id", ""),
                    "ok": tr.get("result", {}).get("ok", False),
                    "error": tr.get("result", {}).get("error"),
                    "execution_time": tr.get("execution_time"),
                    "step": tr.get("step", 0),
                    "summary": _llm_text_from_orchestrator_tool_result(tr, 4000)
                    or None,
                }
                for tr in tool_runs
            ]
        
        # ذخیره پاسخ assistant
        await client.add_message(
            session_id=session_id,
            role="assistant",
            content=response_text or assistant_response or "",
            thinking=thinking,
            tool_calls=db_tool_calls,
            tool_results=db_tool_results,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            metadata=metadata,
            user_id=user_id,
            trace_id=trace_id
        )
        
        log.debug(f"💾 Saved chat turn to session {session_id} via Storage Service (attachments: {len(attachments or [])})")
        
    except Exception as e:
        log.warning(f"Failed to save chat turn: {e}")


async def get_chat_history(
    session_id: str,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    دریافت تاریخچه چت برای LLM از Storage Service.
    
    Returns:
        لیست پیام‌ها با ساختار {"role": "...", "content": "...", "attachments": [...]}
    """
    if not settings.CHAT_HISTORY_ENABLED:
        return []
    
    try:
        client = get_chat_history_client()
        return await client.get_messages_for_llm(
            session_id=session_id,
            limit=limit or settings.CHAT_HISTORY_LIMIT
        )
    except Exception as e:
        log.warning(f"Failed to get chat history: {e}")
        return []


async def get_chat_history_for_stream(
    session_id: str,
    llm_client=None,
    limit: Optional[int] = None,
    *,
    prefetched_messages: Optional[List[Dict[str, Any]]] = None,
    lazy_summary: bool = False,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    History for ``/chat/stream`` (stream_agent): recent turns + summary,
    with tool-result summaries embedded in assistant text.

    ``prefetched_messages`` and ``lazy_summary`` are passthroughs for the
    workspace preflight; see ``ChatHistoryClient.get_messages_for_llm_stream``.
    """
    if not settings.CHAT_HISTORY_ENABLED:
        return [], ""

    try:
        client = get_chat_history_client()
        return await client.get_messages_for_llm_stream(
            session_id=session_id,
            limit=limit or settings.CHAT_HISTORY_LIMIT,
            llm_client=llm_client,
            prefetched_messages=prefetched_messages,
            lazy_summary=lazy_summary,
        )
    except Exception as e:
        log.warning(f"Failed to get stream chat history: {e}")
        return [], ""


async def get_chat_history_v2(
    session_id: str,
    llm_client=None,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Get chat history with smart summarization (v2).

    Returns:
        (recent_messages, history_summary)
    """
    if not settings.CHAT_HISTORY_ENABLED:
        return [], ""

    try:
        client = get_chat_history_client()
        return await client.get_messages_for_llm_v2(
            session_id=session_id,
            limit=limit or settings.CHAT_HISTORY_LIMIT,
            llm_client=llm_client,
        )
    except Exception as e:
        log.warning(f"Failed to get chat history v2: {e}")
        return [], ""


async def get_session_attachments(
    session_id: str,
    limit: Optional[int] = None,
    auth_context: Optional[Dict[str, Any]] = None,
    *,
    prefetched_messages: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    دریافت تمام فایل‌های ضمیمه شده در یک session.
    
    این تابع برای استفاده در main.py وقتی که کاربر در مورد
    فایل قبلی سوال می‌پرسه (بدون آپلود جدید).
    
    Returns:
        لیست artifacts از پیام‌های قبلی (جدیدترین اول)
    """
    if not settings.CHAT_HISTORY_ENABLED:
        return []
    
    try:
        client = get_chat_history_client()
        return await client.get_session_attachments(
            session_id=session_id,
            limit=limit or settings.CHAT_HISTORY_LIMIT,
            auth_context=auth_context,
            prefetched_messages=prefetched_messages,
        )
    except Exception as e:
        log.warning(f"Failed to get session attachments: {e}")
        return []


async def get_session_file_registry(
    session_id: str,
    limit: Optional[int] = None,
    auth_context: Optional[Dict[str, Any]] = None,
    *,
    prefetched_messages: Optional[List[Dict[str, Any]]] = None,
    prefetched_storage_artifacts: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Build a chronological registry of all files in a session.

    Returns list of dicts with ordinal, turn, name, path, type, etc.
    Ordered oldest-first so ordinal #1 is the first file ever uploaded.
    """
    if not settings.CHAT_HISTORY_ENABLED:
        return []

    try:
        client = get_chat_history_client()
        return await client.get_session_file_registry(
            session_id=session_id,
            limit=limit or settings.CHAT_HISTORY_LIMIT,
            auth_context=auth_context,
            prefetched_messages=prefetched_messages,
            prefetched_storage_artifacts=prefetched_storage_artifacts,
        )
    except Exception as e:
        log.warning(f"Failed to build file registry: {e}")
        return []


__all__ = [
    "ChatHistoryClient",
    "extract_agent_state_from_messages",
    "get_chat_history_client",
    "save_chat_turn",
    "save_user_message",
    "get_chat_history",
    "get_chat_history_v2",
    "get_chat_history_for_stream",
    "get_session_attachments",
    "get_session_file_registry",
]
