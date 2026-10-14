"""
Orchestrator Service - FastAPI Application
سرویس ارکستراتور پیشرفته با vLLM و LangGraph
ویژگی‌ها: Streaming, Tool-Calling, Error Recovery, Suggestions
"""
import logging
import os
import uuid
import time
import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel
from contextlib import asynccontextmanager
from datetime import datetime

import anyio
import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

from .config import settings
from .schemas import (
    ChatRequest, ChatResponse, ToolRunInfo, FileContext,
    HealthResponse, ToolsListResponse, ToolInfo,
    DirectToolRequest, DirectToolResponse, ErrorResponse,
    StreamChunk, StreamEventType, ToolStatus,
    SuggestionItem, WarningItem
)
from .graph import (
    run_agent_graph,
    run_agent_graph_stream,
    AgentState,
    run_planning_agent,
    run_planning_agent_stream,
    run_agent as run_react_agent,
)
from .tracing import get_trace_client, TraceStatus
from .llm.gateway_llm_client import get_gateway_llm_client
from .tools.executor import get_tool_executor
from .memory import get_memory_client
from .utils.streaming import format_sse_event, StreamEventTypes
from .utils.file_context import (
    set_tools_by_category_cache,
    set_file_based_tools_cache
)
from .session_workspace import SessionWorkspaceState, build_session_workspace

# ==================== Logging Setup ====================

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer() if settings.DEBUG else structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

log = structlog.get_logger(__name__)
logging.basicConfig(level=logging.DEBUG if settings.DEBUG else logging.INFO)

# Background completion of tool-runner cache warmup if startup budget is exceeded.
_tool_warmup_bg_task: Optional[asyncio.Task] = None


# ==================== Lifespan ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup و Shutdown."""
    global _tool_warmup_bg_task
    log.info("🚀 Starting Orchestrator Service...")
    from .config import assert_secure_secrets
    assert_secure_secrets()
    # #region agent log
    from .debug_ndjson import debug_ndjson

    debug_ndjson(
        "BOOT",
        "main.py:lifespan",
        "orchestrator_startup",
        {
            "service": settings.SERVICE_NAME,
            "version": settings.SERVICE_VERSION,
            "hint": "If this line is missing in kubectl logs, this image is not the instrumented build.",
        },
    )
    # Also stdout + structlog: some clusters only attach stdout / JSON logs to kubectl logs.
    try:
        print(
            "ORCH_DEBUG_NDJSON "
            + json.dumps(
                {
                    "sessionId": "32fb75",
                    "hypothesisId": "BOOT",
                    "message": "orchestrator_startup_stdout",
                    "data": {"version": settings.SERVICE_VERSION},
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    except Exception:
        pass
    log.warning(
        f"ORCH_DEBUG_NDJSON orchestrator_startup service={settings.SERVICE_NAME} version={settings.SERVICE_VERSION}"
    )
    # #endregion

    # Health check برای LLM
    llm_client = get_gateway_llm_client()
    llm_ok = await llm_client.health_check()
    if llm_ok:
        log.info("✅ LLM connection established")
    else:
        log.warning("⚠️ LLM not available yet")
    
    async def _warm_tool_runner_caches() -> None:
        """Sequential tool-runner fetches; each HTTP op may wait up to executor client timeout."""
        tool_executor = get_tool_executor()
        manifest = await tool_executor.get_tools_manifest()
        log.info(f"✅ Tool Runner connected: {len(manifest)} tools available")

        from .prompts.qwen3_prompts import set_agent_prompts_cache

        agent_prompts = await tool_executor.get_all_agent_prompts()
        if agent_prompts:
            set_agent_prompts_cache(agent_prompts)
            log.info(f"✅ Agent prompts loaded: {list(agent_prompts.keys())}")

            from .prompts.system import set_agent_prompts_cache as set_system_agent_cache

            set_system_agent_cache(agent_prompts)

        category_map = await tool_executor.get_category_map()
        if category_map:
            set_tools_by_category_cache(category_map)

        file_based_tools = await tool_executor.get_file_based_tools()
        if file_based_tools:
            set_file_based_tools_cache(file_based_tools)

    warmup_s = float(os.getenv("ORCH_STARTUP_TOOL_WARMUP_S", "45"))
    try:
        await asyncio.wait_for(_warm_tool_runner_caches(), timeout=warmup_s)
    except asyncio.TimeoutError:
        log.warning(
            "⚠️ Tool-runner warmup exceeded %ss — serving /health now; finishing warmup in background",
            warmup_s,
        )
        # #region agent log
        from .debug_ndjson import debug_ndjson

        debug_ndjson(
            "H-startup",
            "main.py:lifespan",
            "tool_warmup_timeout",
            {"warmup_s": warmup_s, "hypothesis": "lifespan_blocked_readiness"},
        )
        # #endregion
        if _tool_warmup_bg_task is None or _tool_warmup_bg_task.done():
            _tool_warmup_bg_task = asyncio.create_task(_warm_tool_runner_caches())
    except Exception as e:
        log.warning(f"⚠️ Tool Runner not available: {e}")

    yield

    if _tool_warmup_bg_task and not _tool_warmup_bg_task.done():
        _tool_warmup_bg_task.cancel()
        try:
            await _tool_warmup_bg_task
        except asyncio.CancelledError:
            pass
    _tool_warmup_bg_task = None

    log.info("🛑 Shutting down Orchestrator Service...")


# ==================== App Creation ====================

app = FastAPI(
    title="AI Orchestrator Service",
    description="سرویس ارکستراتور هوش مصنوعی با vLLM و LangGraph",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Drop-in orchestrator / harness discovery (additive; does not alter /chat*).
from .orchestrators.routes import router as orchestrators_router

app.include_router(orchestrators_router)


# ==================== Health Check ====================

@app.post("/internal/cache/bust")
async def internal_cache_bust(request: Request):
    """Clear cached per-user RBAC permissions (internal: api-gateway after edits).

    Lets an admin RBAC change reflect in chat immediately instead of waiting for
    the per-user TTL. Requires the internal service key.
    """
    key = request.headers.get("X-Internal-Key") or request.headers.get("x-internal-key")
    if key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Internal API key required")

    from .auth.permissions_client import bust_cache

    user_id = request.query_params.get("user_id")
    cleared = bust_cache(user_id)
    return {"cleared": cleared, "scope": user_id or "all"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """بررسی سلامت سرویس."""
    from .debug_ndjson import debug_ndjson

    t_wall = time.monotonic()
    llm_client = get_gateway_llm_client()
    tool_executor = get_tool_executor()

    async def _llm_part():
        t0 = time.monotonic()
        try:
            ok = await llm_client.health_check()
        except Exception:
            ok = False
        return ok, (time.monotonic() - t0) * 1000

    async def _tool_part():
        t0 = time.monotonic()
        ok = False
        try:
            await asyncio.wait_for(tool_executor.get_tools_manifest(), timeout=5.0)
            ok = True
        except Exception:
            ok = False
        return ok, (time.monotonic() - t0) * 1000

    async def _storage_part():
        t0 = time.monotonic()
        storage_ok = False
        try:
            base = (settings.STORAGE_URL or "").rstrip("/")
            if base:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(f"{base}/health")
                    storage_ok = resp.status_code < 500
        except Exception:
            storage_ok = False
        return storage_ok, (time.monotonic() - t0) * 1000

    parts = await asyncio.gather(_llm_part(), _tool_part(), _storage_part(), return_exceptions=True)
    llm_ready, llm_ms = False, 0.0
    tool_runner_ready, tool_ms = False, 0.0
    storage_ready, storage_ms = False, 0.0
    for i, res in enumerate(parts):
        if isinstance(res, BaseException):
            continue
        ok, ms = res
        if i == 0:
            llm_ready, llm_ms = ok, ms
        elif i == 1:
            tool_runner_ready, tool_ms = ok, ms
        else:
            storage_ready, storage_ms = ok, ms

    memory_ready = settings.MEMORY_ENABLED

    status = (
        "healthy"
        if (llm_ready and tool_runner_ready and storage_ready)
        else "degraded"
    )

    # #region agent log
    wall_ms = (time.monotonic() - t_wall) * 1000
    debug_ndjson(
        "H-probe",
        "main.py:health_check",
        "health_timings",
        {
            "wall_ms": round(wall_ms, 1),
            "llm_ms": round(llm_ms, 1),
            "tool_ms": round(tool_ms, 1),
            "storage_ms": round(storage_ms, 1),
            "llm_ready": llm_ready,
            "tool_runner_ready": tool_runner_ready,
            "storage_ready": storage_ready,
            "status": status,
        },
    )
    # #endregion

    return HealthResponse(
        status=status,
        llm_ready=llm_ready,
        tool_runner_ready=tool_runner_ready,
        storage_ready=storage_ready,
        memory_ready=memory_ready
    )


# ==================== Chat Endpoints ====================

@app.get(
    "/chat/sessions/{session_id}/rlm/live",
    summary="Hub RLM live snapshot (logs + status) for chat reconnect",
)
async def chat_rlm_live(session_id: str, finalize: bool = True):
    """Agent-Flow-style progress poll for hub RLM chat.

    When the user leaves mid-turn, SSE dies but the durable kernel may keep
    running. The FE polls this endpoint to restore logs / partial answer, and
    optionally finalizes an assistant chat row when the engine is terminal.
    """
    from .orchestrators.host_adapters.session_binding import hub_rlm_live_snapshot

    sid = str(session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id required")

    payload = hub_rlm_live_snapshot(sid)
    status = str(payload.get("status") or "").strip().lower()
    terminal = status in (
        "completed",
        "complete",
        "done",
        "error",
        "cancelled",
        "canceled",
        "failed",
        "paused",
        "idle",
    ) and not bool(payload.get("live"))

    finalized = False
    if finalize and terminal and settings.CHAT_HISTORY_ENABLED:
        try:
            finalized = await _finalize_rlm_chat_from_live(sid, payload)
            payload["finalized"] = finalized
        except Exception as exc:
            log.warning(
                "rlm live finalize failed session=%s: %s", sid, exc, exc_info=True
            )
            payload["finalize_error"] = str(exc)[:300]

    payload["terminal"] = terminal
    return payload


@app.get(
    "/chat/sessions/{session_id}/case-bindings",
    summary="List durable @case bindings for a chat session",
)
async def chat_list_case_bindings(session_id: str, request: Request):
    """Return Bound Cases UI rows (handles + counts; no file dump)."""
    from .orchestrators.host_adapters.case_mentions import list_case_bindings_for_session

    sid = str(session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id required")
    user_id = str(
        request.query_params.get("user_id")
        or request.headers.get("x-user-id")
        or ""
    ).strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    return await list_case_bindings_for_session(session_id=sid, user_id=user_id)


@app.delete(
    "/chat/sessions/{session_id}/case-bindings/{case_id}",
    summary="Unbind one durable @case from a chat session",
)
async def chat_unbind_case_binding(
    session_id: str, case_id: str, request: Request
):
    """Remove one case binding and rebuild the case pool inventory."""
    from .orchestrators.host_adapters.case_mentions import unbind_case_for_session

    sid = str(session_id or "").strip()
    cid = str(case_id or "").strip()
    if not sid or not cid:
        raise HTTPException(status_code=400, detail="session_id and case_id required")
    user_id = str(
        request.query_params.get("user_id")
        or request.headers.get("x-user-id")
        or ""
    ).strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    return await unbind_case_for_session(
        session_id=sid, user_id=user_id, case_id=cid
    )


async def _finalize_rlm_chat_from_live(
    session_id: str, live_payload: Dict[str, Any]
) -> bool:
    """If chat history ends on a user turn, append assistant from live snapshot."""
    from .chat_history import get_chat_history_client, save_chat_turn

    client = get_chat_history_client()
    rows = await client.get_messages(session_id, limit=20)
    if not isinstance(rows, list) or not rows:
        return False
    # Storage may return newest-first or oldest-first — find last by created_at if present.
    ordered = list(rows)
    try:
        if all(isinstance(r, dict) and r.get("created_at") for r in ordered):
            ordered = sorted(ordered, key=lambda r: str(r.get("created_at") or ""))
    except Exception:
        pass
    last = ordered[-1] if ordered else None
    if not isinstance(last, dict):
        return False
    if str(last.get("role") or "").strip().lower() != "user":
        # Already have an assistant (or other) after the user — don't duplicate.
        # But if last assistant is a partial placeholder, upgrade it when we have
        # a richer live answer.
        if str(last.get("role") or "").strip().lower() == "assistant":
            content = str(last.get("content") or "").strip()
            live_answer = str(live_payload.get("last_content") or "").strip()
            live_logs = live_payload.get("log_lines") or []
            log_blob = "\n".join(str(x) for x in live_logs if str(x).strip())
            richer = live_answer and (
                content in ("", "_(turn interrupted before a final answer)_")
                or (len(live_answer) > len(content) + 40)
            )
            if richer:
                # Patch via new assistant row is messy; skip upgrade for now —
                # FE can show live overlay. Only auto-append when missing.
                return False
        return False

    answer = str(live_payload.get("last_content") or "").strip()
    logs = live_payload.get("log_lines") if isinstance(live_payload.get("log_lines"), list) else []
    thinking = "\n".join(str(x) for x in logs if str(x).strip())
    reasoning = str(live_payload.get("last_reasoning") or "").strip()
    if reasoning and reasoning not in thinking:
        thinking = (thinking + "\n" + reasoning).strip() if thinking else reasoning
    err = live_payload.get("error")
    from .orchestrators.host_adapters.hub_final_answer import (
        ensure_nonempty_hub_answer,
        is_placeholder_answer,
    )

    if is_placeholder_answer(answer):
        answer = ensure_nonempty_hub_answer(
            answer,
            thinking=thinking,
            error=str(err or ""),
            status=str(live_payload.get("status") or ""),
        )

    user_id = str(last.get("user_id") or "").strip() or "default"
    await save_chat_turn(
        session_id=session_id,
        user_id=user_id,
        user_message=str(last.get("content") or ""),
        assistant_response=answer,
        thinking=thinking or None,
        metadata={
            "source": "rlm_live_finalize",
            "engine_session_id": live_payload.get("engine_session_id"),
            "status": live_payload.get("status"),
            **({"error": str(err)[:500]} if err else {}),
        },
        skip_user=True,
    )
    log.info(
        "rlm live finalize saved assistant session=%s status=%s answer_len=%s log_lines=%s",
        session_id,
        live_payload.get("status"),
        len(answer),
        len(logs),
    )
    return True


@app.post(
    "/chat/sessions/{session_id}/agent-turn",
    summary="Record an Agent Flow result into a chat session",
)
async def chat_agent_turn(session_id: str, request: Request):
    """
    Persist one Agent Mode turn into normal chat history.

    Does not run Agent Flow. Frontend executes the flow, then posts text +
    result artifact ids here so later ``/chat/stream`` turns can discuss them.
    """
    from .agent_turn import AgentTurnRequest, record_agent_turn

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid_json") from None
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="invalid_body")

    try:
        payload = AgentTurnRequest.model_validate(body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = await record_agent_turn(session_id=session_id, request=payload)
    except LookupError:
        raise HTTPException(status_code=404, detail="session_not_found") from None
    except PermissionError:
        raise HTTPException(status_code=403, detail="session_access_denied") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.error("agent_turn failed", error=str(exc), session_id=session_id)
        raise HTTPException(status_code=502, detail="agent_turn_persist_failed") from exc

    return result.model_dump()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    پردازش پیام چت.
    
    این endpoint پیام کاربر را دریافت کرده و با استفاده از
    LangGraph agent پردازش می‌کند.
    """
    start_time = time.time()
    log.info(
        "📨 Chat request received",
        session_id=request.session_id,
        message_len=len(request.message),
        artifacts_count=len(request.artifacts)
    )
    
    try:
        # ==================== Unified Preflight (parallel) ====================
        # All session context is assembled in a single, parallelised pass.
        # See ``session_workspace.build_session_workspace`` for the layout.
        llm_client = get_gateway_llm_client()
        workspace = await build_session_workspace(
            request,
            llm_client=llm_client,
            enable_memory=True,
        )
        await workspace.apply_long_input(getattr(request, "orchestrator_config", None))
        artifacts = workspace.artifacts
        has_fresh_artifacts = workspace.has_fresh_artifacts
        chat_history = workspace.chat_history

        max_steps = request.max_steps or settings.MAX_STEPS

        # ==================== Run Agent Graph ====================
        # Default: Continue-like ReAct (`run_react_agent` → `stream_agent`).
        # Set CHAT_USE_PLANNING_AGENT=true for Plan-Execute-Critic on this endpoint.
        if settings.CHAT_USE_PLANNING_AGENT and settings.PLANNING_ENABLED:
            result = await run_planning_agent(
                max_steps=max_steps,
                max_replans=settings.MAX_REPLANS,
                **workspace.to_planning_kwargs(),
            )
        elif settings.CHAT_USE_PLANNING_AGENT:
            result = await run_agent_graph(
                max_steps=max_steps,
                **workspace.to_legacy_graph_kwargs(),
            )
        else:
            result = await run_react_agent(
                max_steps=max_steps,
                **workspace.to_engine_kwargs(),
            )

        # ==================== Reactive Repetition Detection ====================
        # If the new answer is suspiciously similar to a recent assistant turn
        # we surface a non-blocking warning and a `repetition_signal` field.
        try:
            if (
                settings.REPETITION_DETECTION_ENABLED
                and isinstance(result, dict)
                and result.get("answer")
                and chat_history
            ):
                from .repetition import detect_repetition

                signal = detect_repetition(
                    new_assistant_text=result.get("answer", ""),
                    chat_history=chat_history,
                )
                if signal.is_repetition:
                    log.warning(
                        f"♻️ Repetition detected: similarity={signal.similarity:.2f} "
                        f"compared_to_turn={signal.compared_to_turn}"
                    )
                # Always attach so callers can inspect, even when not flagged.
                result["repetition_signal"] = signal.to_dict()
        except Exception as e:
            log.debug(f"Repetition detection skipped: {e}")
        
        # ==================== Save to Memory ====================
        # ذخیره در حافظه با session isolation
        if request.use_memory and settings.MEMORY_ENABLED and settings.MEMORY_AUTO_SAVE:
            try:
                from .memory import get_advanced_memory_client, MemoryCategory
                
                memory_client = get_advanced_memory_client()
                memory_client.save_conversation_background(
                    user_id=request.user_id,
                    session_id=request.session_id,
                    user_message=request.message,
                    assistant_response=result.get("answer", ""),
                    category=MemoryCategory.CONTEXT,
                    is_long_term=False,  # فقط در این session
                    metadata={
                        "tools_used": [t.get("tool_id") for t in result.get("tool_runs", [])],
                        "trace_id": result.get("trace_id")
                    }
                )
            except Exception as e:
                log.warning(f"Memory save failed: {e}")
        
        # ==================== Save Chat History ====================
        # ذخیره در PostgreSQL
        log.info(f"🔍 DEBUG: CHAT_HISTORY_ENABLED={settings.CHAT_HISTORY_ENABLED}")
        if settings.CHAT_HISTORY_ENABLED:
            try:
                from .chat_history import save_chat_turn
                # فقط artifacts جدید را ذخیره کن (نه فایل‌های قبلی که از session گرفتیم)
                new_artifacts = [a.model_dump() for a in request.artifacts] if request.artifacts else None
                log.info(f"🔍 DEBUG: Calling save_chat_turn for session {request.session_id}")
                await save_chat_turn(
                    session_id=request.session_id,
                    user_id=request.user_id,
                    user_message=request.message,
                    assistant_response=result.get("answer", ""),
                    thinking=result.get("thinking"),
                    tool_runs=result.get("tool_runs", []),
                    attachments=new_artifacts,
                    metadata={"trace_id": result.get("trace_id")},
                    tokens_prompt=result.get("tokens_prompt"),
                    tokens_completion=result.get("tokens_completion"),
                )
                log.info(f"✅ DEBUG: save_chat_turn completed for session {request.session_id}")
            except Exception as e:
                log.warning(f"Chat history save failed: {e}", exc_info=True)
        
        elapsed = time.time() - start_time
        log.info(
            "✅ Chat response generated",
            session_id=request.session_id,
            tool_runs=len(result.get("tool_runs", [])),
            elapsed=f"{elapsed:.2f}s"
        )
        
        # تبدیل tool_runs به ToolRunInfo
        tool_runs = [
            ToolRunInfo(
                tool_id=tr.get("tool_id", ""),
                args=tr.get("args", {}),
                result=tr.get("result", {}),
                step=tr.get("step", 0)
            )
            for tr in result.get("tool_runs", [])
        ]
        
        # تبدیل file_context به FileContext (اگه موجود باشه)
        file_context = None
        if result.get("file_context"):
            fc = result["file_context"]
            try:
                file_context = FileContext(
                    type=fc.get("type", "unknown"),
                    name=fc.get("name", "unknown"),
                    path=fc.get("path", ""),
                    mime_type=fc.get("mime_type"),
                    suggested_tools=[]
                )
            except Exception as fc_err:
                log.warning(f"Failed to convert file_context: {fc_err}")
        
        # تبدیل suggestions به SuggestionItem
        suggestions = []
        for s in result.get("suggestions", []):
            if isinstance(s, str):
                suggestions.append(SuggestionItem(text=s, action_type="query"))
            elif isinstance(s, dict):
                suggestions.append(SuggestionItem(
                    text=s.get("text", str(s)),
                    action_type=s.get("action_type", "query"),
                    data=s.get("data")
                ))
        
        # تبدیل warnings به WarningItem
        warnings = []
        for w in result.get("warnings", []):
            if isinstance(w, str):
                warnings.append(WarningItem(message=w, level="warning"))
            elif isinstance(w, dict):
                warnings.append(WarningItem(
                    message=w.get("message", str(w)),
                    level=w.get("level", "warning"),
                    code=w.get("code")
                ))
        
        return ChatResponse(
            answer=result.get("answer", ""),
            tool_runs=tool_runs,
            trace_id=result.get("trace_id", str(uuid.uuid4())),
            steps=result.get("steps", 0),
            file_context=file_context,
            thinking=result.get("thinking"),
            suggestions=suggestions,
            warnings=warnings
        )
        
    except Exception as exc:
        log.error("🔴 Chat processing failed", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    استریم واقعی پاسخ چت با LangGraph.
    
    این endpoint از streaming واقعی استفاده می‌کنه - توکن به توکن.
    
    رویدادها (SSE):
    - start: شروع پردازش
    - file_detected: فایل شناسایی شد
    - thinking_start: شروع فرآیند فکر
    - thinking: توکن‌های thinking (real-time)
    - thinking_end: پایان thinking
    - tool_start: شروع اجرای ابزار
    - tool_result: نتیجه ابزار
    - answer_start: شروع پاسخ
    - answer: توکن‌های پاسخ (real-time)
    - done: اتمام با نتیجه نهایی
    - error: خطا
    """
    start_time = time.time()
    
    log.info(
        "📨 Real-time streaming chat request",
        session_id=request.session_id,
        message_len=len(request.message),
        artifacts_count=len(request.artifacts)
    )
    
    async def generate_stream():
        """Generator برای SSE events با streaming واقعی."""
        from .orchestrators.chat_engine_resolve import load_stream_agent

        stream_agent = load_stream_agent(
            request.orchestrator_id,
            getattr(request, "context", None),
        )

        try:
            # ==================== Unified Preflight (parallel) ====================
            llm_client = get_gateway_llm_client()
            workspace = await build_session_workspace(
                request,
                llm_client=llm_client,
                enable_memory=False,
            )
            artifacts = workspace.artifacts
            original_artifacts = list(workspace.fresh_artifacts)

            # ==================== Real-time Streaming ====================
            collected_answer = ""
            collected_thinking = ""
            trace_id = None
            tool_runs = []
            collected_tokens_prompt = None
            collected_tokens_completion = None
            done_discourse_focus = None
            done_conversation_summary = None
            done_task_checklist = None

            from .orchestrators.host_adapters.model_catalog import (
                prepare_orchestrator_config,
            )

            orch_cfg = await prepare_orchestrator_config(
                getattr(request, "orchestrator_config", None)
            )
            long_input = await workspace.apply_long_input(orch_cfg)
            if long_input:
                yield f"data: {json.dumps({'type': 'status', 'stage': 'long_input', 'long_input': long_input}, ensure_ascii=False)}\n\n"
            compacted = await workspace.apply_session_compaction(orch_cfg, llm=llm_client)
            if compacted:
                yield f"data: {json.dumps(compacted, ensure_ascii=False)}\n\n"
            done_self_compact = None
            served_by = None
            async for event in stream_agent(
                max_steps=request.max_steps or settings.MAX_STEPS,
                orchestrator_config=orch_cfg,
                **workspace.to_hub_engine_kwargs(),
            ):  # includes discourse_focus, workspace_runtime, task_checklist_state
                event_type = event.get("type")
                
                # Format and send SSE event
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                
                # Track important data
                if event_type == "start":
                    trace_id = event.get("trace_id")
                    served_by = served_by or event.get("engine")
                elif event_type == "thinking":
                    collected_thinking += event.get("content", "")
                elif event_type == "answer":
                    collected_answer += event.get("content", "")
                elif event_type == "tool_result":
                    tool_runs.append({
                        "tool_id": event.get("tool_name"),
                        "ok": event.get("ok", False)
                    })
                elif event_type == "done":
                    # Use final answer from done event if available
                    if event.get("final_answer"):
                        collected_answer = event.get("final_answer")
                    if event.get("thinking"):
                        collected_thinking = event.get("thinking")
                    if event.get("tool_runs"):
                        tool_runs = event.get("tool_runs")
                    collected_tokens_prompt = event.get("tokens_prompt")
                    collected_tokens_completion = event.get("tokens_completion")
                    if event.get("discourse_focus"):
                        done_discourse_focus = event.get("discourse_focus")
                    if event.get("conversation_summary"):
                        done_conversation_summary = event.get("conversation_summary")
                    if event.get("task_checklist"):
                        done_task_checklist = event.get("task_checklist")
                    if event.get("self_compact"):
                        done_self_compact = event.get("self_compact")
            
            # ==================== Save Chat History ====================
            if settings.CHAT_HISTORY_ENABLED and collected_answer:
                try:
                    from .chat_history import save_chat_turn
                    # فقط artifacts اصلی (جدید) را ذخیره کن، نه فایل‌های قبلی
                    log.info(f"💾 Saving chat turn: session={request.session_id}, attachments={len(original_artifacts)} items")
                    turn_metadata = {"trace_id": trace_id}
                    if done_discourse_focus:
                        turn_metadata["discourse_focus"] = done_discourse_focus
                    if done_conversation_summary:
                        turn_metadata["conversation_summary"] = done_conversation_summary
                    if done_task_checklist:
                        turn_metadata["task_checklist"] = done_task_checklist
                    turn_self_compact = workspace.turn_self_compact(done_self_compact)
                    if turn_self_compact:
                        turn_metadata["self_compact"] = turn_self_compact
                    if served_by:
                        turn_metadata["engine"] = served_by
                    if workspace.long_input:
                        turn_metadata["long_input"] = workspace.long_input
                    await save_chat_turn(
                        session_id=request.session_id,
                        user_id=request.user_id,
                        user_message=request.message,
                        assistant_response=collected_answer,
                        thinking=collected_thinking,
                        tool_runs=tool_runs,
                        attachments=original_artifacts if original_artifacts else None,
                        metadata=turn_metadata,
                        tokens_prompt=collected_tokens_prompt,
                        tokens_completion=collected_tokens_completion,
                    )
                    log.info(f"✅ Chat turn saved to session {request.session_id}")
                except Exception as e:
                    log.warning(f"Chat history save failed: {e}")
            
            elapsed = time.time() - start_time
            log.info(
                "✅ Streaming complete",
                session_id=request.session_id,
                trace_id=trace_id,
                elapsed=f"{elapsed:.2f}s"
            )
            
        except Exception as exc:
            log.error(f"🔴 Streaming failed: {exc}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/chat/stream/full")
async def chat_stream_full(request: ChatRequest):
    """
    استریم کامل با پشتیبانی از ابزار و events متعدد.
    """
    trace_id = str(uuid.uuid4())
    start_time = time.time()
    
    log.info(
        "📨 Full streaming chat request received",
        session_id=request.session_id,
        trace_id=trace_id
    )
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        """Generator برای SSE events."""
        user_saved_early = False
        session_checkpoint = None
        turn_persisted = False
        collected_answer: list = []
        collected_thinking_parts: list = []
        done_final_answer = None
        done_tool_runs: list = []
        done_thinking = None
        done_error = None
        done_rlm_status = None
        done_tokens_prompt = None
        done_tokens_completion = None
        done_discourse_focus = None
        done_conversation_summary = None
        done_task_checklist = None
        original_artifacts: list = []
        stream_error_message: Optional[str] = None
        stream_generator = None
        turn_completed = False

        try:
            # ارسال رویداد شروع
            yield format_sse_event(StreamEventTypes.STATUS, {
                "message": "درحال پردازش...",
                "trace_id": trace_id
            })
            
            # ==================== Unified Preflight (parallel) ====================
            llm_client = get_gateway_llm_client()
            workspace = await build_session_workspace(
                request,
                llm_client=llm_client,
                enable_memory=True,
            )
            artifacts = workspace.artifacts
            original_artifacts = list(workspace.fresh_artifacts)
            chat_history = workspace.chat_history
            history_summary = workspace.history_summary
            conversation_state = workspace.conversation_state
            file_registry = workspace.file_registry
            pinned_facts = workspace.pinned_facts
            has_fresh_artifacts = workspace.has_fresh_artifacts
            from .tools.side_effects import begin_tool_turn

            begin_tool_turn(trace_id, chat_history, request.message,
                            request_context=workspace.request_context,
                            summary=history_summary)
            if not workspace.memory_available and request.use_memory and settings.MEMORY_ENABLED:
                yield format_sse_event(StreamEventTypes.WARNING, {
                    "message": "حافظه موقتاً در دسترس نیست",
                    "level": "info"
                })

            # Persist user message ASAP so refresh / SSE abort does not wipe the turn.
            if settings.CHAT_HISTORY_ENABLED:
                try:
                    from .chat_history import save_user_message

                    await save_user_message(
                        session_id=request.session_id,
                        user_id=request.user_id,
                        user_message=request.message,
                        attachments=original_artifacts if original_artifacts else None,
                    )
                    user_saved_early = True
                except Exception as e:
                    log.warning(
                        "Early user message save failed: session=%s err=%s",
                        request.session_id,
                        e,
                        exc_info=True,
                    )
            
            # اجرای streaming graph
            final_state = None
            collected_answer = []
            done_final_answer = None
            done_tool_runs = []
            done_thinking = None
            done_error = None
            done_rlm_status = None
            done_tokens_prompt = None
            done_tokens_completion = None
            done_discourse_focus = None
            done_conversation_summary = None
            done_task_checklist = None
            done_self_compact = None

            from .orchestrators.host_adapters.model_catalog import (
                prepare_orchestrator_config,
            )

            orch_cfg = await prepare_orchestrator_config(
                getattr(request, "orchestrator_config", None)
            )
            long_input = await workspace.apply_long_input(orch_cfg)
            if long_input:
                yield format_sse_event(StreamEventTypes.STATUS, {
                    "stage": "long_input",
                    "long_input": long_input,
                    "trace_id": trace_id,
                })
            compacted = await workspace.apply_session_compaction(orch_cfg, llm=llm_client)
            session_checkpoint = workspace.session_compaction
            if compacted:
                yield format_sse_event("context_compacted", {
                    k: v for k, v in compacted.items() if k != "type"
                })
            served_by = None

            # Default: Continue-like single ReAct loop (`stream_agent`). Set
            # STREAM_CHAT_FULL_USE_PLANNING_AGENT=true for Plan-Execute-Critic.
            if (
                settings.STREAM_CHAT_FULL_USE_PLANNING_AGENT
                and settings.PLANNING_ENABLED
            ):
                stream_generator = run_planning_agent_stream(
                    max_steps=request.max_steps or settings.MAX_STEPS,
                    max_replans=settings.MAX_REPLANS,
                    trace_id=trace_id,
                    **workspace.to_planning_kwargs(),
                )
            elif settings.STREAM_CHAT_FULL_USE_PLANNING_AGENT:
                stream_generator = run_agent_graph_stream(
                    max_steps=request.max_steps or settings.MAX_STEPS,
                    **workspace.to_legacy_graph_kwargs(),
                )
            else:
                from .orchestrators.chat_engine_resolve import load_stream_agent_full

                _stream_full = load_stream_agent_full(
                    request.orchestrator_id,
                    getattr(request, "context", None),
                )
                stream_generator = _stream_full(
                    max_steps=request.max_steps or settings.MAX_STEPS,
                    orchestrator_config=orch_cfg,
                    **workspace.to_hub_engine_kwargs(),
                )
            
            async for event in stream_generator:
                event_type = event.get("type", "")
                
                if event_type == "start":
                    if event.get("engine") and not served_by:
                        served_by = event.get("engine")
                        yield format_sse_event(StreamEventTypes.STATUS, {
                            "stage": "engine",
                            "engine": served_by,
                            "trace_id": trace_id,
                        })

                elif event_type == "token":
                    # توکن جدید
                    content = event.get("content", "")
                    collected_answer.append(content)
                    yield format_sse_event(StreamEventTypes.TOKEN, {"content": content})
                
                elif event_type == "thinking":
                    # Always accumulate for persistence / reconnect (AF-style logs).
                    content = event.get("content", "")
                    if content:
                        collected_thinking_parts.append(str(content))
                    if request.show_thinking:
                        yield format_sse_event(StreamEventTypes.THINKING, {
                            "content": content
                        })

                elif event_type == "thinking_start":
                    if request.show_thinking:
                        yield format_sse_event(StreamEventTypes.THINKING_START, {})

                elif event_type == "thinking_end":
                    if request.show_thinking:
                        yield format_sse_event(StreamEventTypes.THINKING_END, {})

                # LLM tool-call streaming (Continue/Cline-style) — must not be dropped
                elif event_type == "tool_call_start":
                    tool_args = event.get("arguments")
                    if tool_args is None:
                        tool_args = event.get("args", {})
                    tool_query = event.get("query")
                    yield format_sse_event(StreamEventTypes.TOOL_CALL_START, {
                        "step": event.get("step", 0),
                        "index": event.get("index", 0),
                        "tool_id": event.get("tool_id"),
                        "tool_name": event.get("tool_name"),
                        "arguments": tool_args,
                        "args": tool_args,
                        "query": tool_query,
                    })
                elif event_type == "tool_call_delta":
                    yield format_sse_event(StreamEventTypes.TOOL_CALL_DELTA, {
                        "step": event.get("step", 0),
                        "index": event.get("index", 0),
                        "tool_id": event.get("tool_id"),
                        "tool_name": event.get("tool_name"),
                        "arguments_delta": event.get("arguments_delta", ""),
                    })
                elif event_type == "tool_plan":
                    yield format_sse_event(StreamEventTypes.TOOL_PLAN, {
                        "step": event.get("step", 0),
                        "tools": event.get("tools", []),
                        "description": event.get("description", ""),
                    })
                elif event_type == "answer_start":
                    yield format_sse_event(StreamEventTypes.ANSWER_START, {
                        "step": event.get("step", 0),
                    })
                elif event_type == "step_start":
                    yield format_sse_event("step_start", {
                        "step": event.get("step", 0),
                        "step_type": event.get("step_type", ""),
                        "description": event.get("description", ""),
                    })
                elif event_type == "llm_start":
                    yield format_sse_event("llm_start", {
                        "step": event.get("step", 0),
                    })
                elif event_type == "llm_done":
                    yield format_sse_event("llm_done", {
                        "step": event.get("step", 0),
                    })
                
                elif event_type == "tool_start":
                    # شروع ابزار
                    tool_args = event.get("args")
                    if tool_args is None:
                        tool_args = event.get("arguments", {})
                    tool_query = event.get("query")
                    yield format_sse_event(StreamEventTypes.TOOL_START, {
                        "tool_id": event.get("tool_id"),
                        "tool_name": event.get("tool_name"),
                        "args": tool_args,
                        "arguments": tool_args,
                        "query": tool_query,
                        "step": event.get("step", 0)
                    })
                
                elif event_type == "tool_progress":
                    # پیشرفت ابزار
                    yield format_sse_event(StreamEventTypes.TOOL_PROGRESS, {
                        "tool_id": event.get("tool_id"),
                        "progress": event.get("progress", 0),
                        "message": event.get("message", "")
                    })
                
                elif event_type == "tool_result":
                    # نتیجه ابزار — یک متن canonical (text_for_llm) برای جلوگیری از تکرار در SSE
                    tool_args = event.get("args")
                    if tool_args is None:
                        tool_args = event.get("arguments", {})
                    llm_text = (
                        event.get("text_for_llm")
                        or event.get("summary")
                        or event.get("output")
                        or event.get("content")
                        or ""
                    )
                    if isinstance(llm_text, str) and len(llm_text) > 500:
                        summary_preview = llm_text[:500]
                    else:
                        summary_preview = llm_text
                    yield format_sse_event(StreamEventTypes.TOOL_RESULT, {
                        "tool_id": event.get("tool_id"),
                        "tool_name": event.get("tool_name"),
                        "args": tool_args,
                        "arguments": tool_args,
                        "query": event.get("query"),
                        "ok": event.get("ok", False),
                        "text_for_llm": llm_text,
                        "summary": summary_preview,
                        "error": event.get("error"),
                        "execution_time": event.get("execution_time", 0),
                    })
                
                elif event_type == "suggestion":
                    # پیشنهاد
                    if request.include_suggestions:
                        yield format_sse_event(StreamEventTypes.SUGGESTION, {
                            "text": event.get("text", ""),
                            "action_type": event.get("action_type", "query")
                        })
                
                elif event_type == "warning":
                    # هشدار
                    yield format_sse_event(StreamEventTypes.WARNING, {
                        "message": event.get("message", ""),
                        "level": event.get("level", "warning")
                    })

                elif event_type == "context_usage":
                    yield format_sse_event("context_usage", {
                        "step": event.get("step", 0),
                        "context_usage_pct": event.get("context_usage_pct"),
                        "context_chars": event.get("context_chars"),
                        "context_budget": event.get("context_budget"),
                        "context_tokens": event.get("context_tokens"),
                        "context_window": event.get("context_window"),
                    })

                elif event_type == "context_compacted":
                    yield format_sse_event("context_compacted", {
                        k: v for k, v in event.items() if k != "type"
                    })

                elif event_type == "context_probe":
                    yield format_sse_event("context_probe", {
                        k: v for k, v in event.items() if k != "type"
                    })

                elif event_type == "file_disambiguation":
                    yield format_sse_event("file_disambiguation", {
                        "session_id": event.get("session_id"),
                        "candidates": event.get("candidates") or [],
                        "message": event.get("message", ""),
                    })
                
                elif event_type == "error":
                    # خطا
                    yield format_sse_event(StreamEventTypes.ERROR, {
                        "message": event.get("message") or event.get("error", ""),
                        "code": event.get("code", "UNKNOWN")
                    })
                
                # ==================== Planning Agent Events ====================
                elif event_type == "node":
                    # نام گره در حال اجرا (برای Planning Agent)
                    yield format_sse_event("node", {
                        "node": event.get("node", ""),
                        "timestamp": event.get("timestamp", "")
                    })
                
                elif event_type == "progress":
                    # پیشرفت کلی (Planning Agent)
                    yield format_sse_event("progress", {
                        "message": event.get("message", ""),
                        "task_id": event.get("task_id"),
                        "progress": event.get("progress"),
                        "status": event.get("status", "")
                    })
                
                elif event_type == "plan":
                    # برنامه اجرایی (Planning Agent)
                    yield format_sse_event("plan", {
                        "plan": event.get("plan", {}),
                        "timestamp": event.get("timestamp", "")
                    })
                
                elif event_type == "evaluation":
                    # ارزیابی Critic (Planning Agent)
                    yield format_sse_event("evaluation", {
                        "evaluation": event.get("evaluation", {}),
                        "overall_confidence": event.get("overall_confidence", 0)
                    })
                
                elif event_type == "answer":
                    # Cline XML engine emits answer deltas. Collect every
                    # chunk for persistence; older engines may later send a
                    # full final_answer in "done", which overrides this below.
                    content = event.get("content", "")
                    if content:
                        try:
                            from .orchestrators.host_adapters.hub_final_answer import (
                                is_placeholder_answer,
                            )

                            if not is_placeholder_answer(content):
                                collected_answer.append(content)
                        except Exception:
                            collected_answer.append(content)
                    yield format_sse_event("answer", {
                        "content": content,
                        "synthesis_reasoning": event.get("synthesis_reasoning", "")
                    })

                elif event_type == "answer_complete":
                    # پایان استریم ترکیب‌کننده — بدون تکرار بدنهٔ پاسخ
                    yield format_sse_event("answer_complete", {
                        "synthesis_streamed_tokens": event.get("synthesis_streamed_tokens", False),
                        "synthesis_reasoning": event.get("synthesis_reasoning", ""),
                        "trace_id": event.get("trace_id"),
                    })
                
                elif event_type == "evidence_step":
                    # Staged-reasoning trail (Continue-style evidence loop).
                    yield format_sse_event("evidence_step", {
                        "step": event.get("step", 0),
                        "phase": event.get("phase", "observe"),
                        "kind": event.get("kind"),
                        "summary": event.get("summary", ""),
                        "ok": event.get("ok"),
                        "tool_name": event.get("tool_name"),
                        "files": event.get("files") or [],
                        "args": event.get("args"),
                        "meta": event.get("meta"),
                        "timestamp": event.get("timestamp"),
                    })

                elif event_type == "vision_attached":
                    yield format_sse_event("vision_attached", {
                        "parts": event.get("parts", 0),
                    })

                elif event_type == "session_file_attached":
                    yield format_sse_event("session_file_attached", {
                        "file": event.get("file") or {},
                        "registry_size": event.get("registry_size", 0),
                    })

                elif event_type == "done":
                    fa = event.get("final_answer") or event.get("answer")
                    if fa and str(fa).strip():
                        done_final_answer = str(fa)
                    if event.get("tool_runs"):
                        done_tool_runs = event.get("tool_runs") or []
                    if event.get("thinking"):
                        done_thinking = event.get("thinking")
                    if event.get("error"):
                        done_error = str(event.get("error") or "")[:2000]
                    ext = event.get("extensions") if isinstance(event.get("extensions"), dict) else {}
                    st = event.get("status") or (ext.get("rlm_status") if ext else None)
                    if st:
                        done_rlm_status = str(st)
                    if not done_error and ext.get("rlm_error"):
                        done_error = str(ext.get("rlm_error") or "")[:2000]
                    done_tokens_prompt = event.get("tokens_prompt")
                    done_tokens_completion = event.get("tokens_completion")
                    if event.get("discourse_focus"):
                        done_discourse_focus = event.get("discourse_focus")
                    if event.get("conversation_summary"):
                        done_conversation_summary = event.get("conversation_summary")
                    if event.get("task_checklist"):
                        done_task_checklist = event.get("task_checklist")
                    if event.get("self_compact"):
                        done_self_compact = event.get("self_compact")
                    # Forward staged evidence aggregate so UIs can render a
                    # "what happened" timeline without re-parsing tool_runs.
                    if event.get("evidence_ledger") or event.get("evidence_summary"):
                        yield format_sse_event("evidence_summary", {
                            "evidence_ledger": event.get("evidence_ledger") or [],
                            "evidence_summary": event.get("evidence_summary") or {},
                            "trace_id": event.get("trace_id"),
                        })

                elif event_type == "final_state":
                    # حالت نهایی
                    final_state = event.get("state", {})
            
            # ساخت پاسخ نهایی
            elapsed = time.time() - start_time
            streamed_text = "".join(collected_answer).strip()
            from .orchestrators.host_adapters.hub_final_answer import (
                ensure_nonempty_hub_answer,
                is_placeholder_answer,
            )

            if final_state:
                candidate = (final_state.get("final_answer") or "").strip()
                # Prefer the longer of streamed vs final so short synthesis never shrinks a long reply.
                if candidate and streamed_text:
                    if is_placeholder_answer(streamed_text) and not is_placeholder_answer(
                        candidate
                    ):
                        answer = candidate
                    elif is_placeholder_answer(candidate) and not is_placeholder_answer(
                        streamed_text
                    ):
                        answer = streamed_text
                    else:
                        answer = (
                            candidate
                            if len(candidate) > len(streamed_text)
                            else streamed_text
                        )
                else:
                    answer = candidate or streamed_text
                tool_runs = [
                    {
                        "id": tr.get("id", ""),
                        "tool_id": tr.get("tool_id", ""),
                        "args": tr.get("args", {}),
                        "result": tr.get("result", {}),
                        "step": tr.get("step", 0),
                        "execution_time": tr.get("execution_time", 0)
                    }
                    for tr in final_state.get("tool_runs", [])
                ]
                suggestions = final_state.get("suggestions", [])
                warnings = final_state.get("warnings", [])
                thinking = final_state.get("thinking", None)
                file_context = final_state.get("file_context")
                steps = final_state.get("step_count", 0)
                tokens_prompt_fs = final_state.get("tokens_prompt")
                tokens_completion_fs = final_state.get("tokens_completion")
            else:
                candidate = (done_final_answer or "").strip()
                if candidate and streamed_text:
                    # Hub RLM: FE appends answer deltas; placeholder+notice concat
                    # must not beat a clean done.final_answer.
                    if is_placeholder_answer(streamed_text) and not is_placeholder_answer(
                        candidate
                    ):
                        answer = candidate
                    elif (
                        "see process log" in streamed_text.lower()
                        or "پاسخی تولید نشد" in streamed_text
                    ) and candidate and not is_placeholder_answer(candidate):
                        answer = candidate
                    elif is_placeholder_answer(candidate) and not is_placeholder_answer(
                        streamed_text
                    ):
                        answer = streamed_text
                    else:
                        answer = (
                            candidate
                            if len(candidate) > len(streamed_text)
                            else streamed_text
                        )
                else:
                    answer = candidate or streamed_text
                tool_runs = done_tool_runs
                suggestions = []
                warnings = []
                thinking = done_thinking
                file_context = None
                steps = 0
                tokens_prompt_fs = done_tokens_prompt
                tokens_completion_fs = done_tokens_completion

            streamed_thinking = "".join(collected_thinking_parts).strip()
            if streamed_thinking and (
                not thinking or len(streamed_thinking) > len(str(thinking or ""))
            ):
                thinking = streamed_thinking

            answer = ensure_nonempty_hub_answer(
                answer,
                thinking=str(
                    thinking
                    or done_thinking
                    or "".join(collected_thinking_parts)
                    or ""
                ),
                error=str(done_error or ""),
                status=str(done_rlm_status or ""),
            )
            # If we already had a longer *real* stream, keep it
            if (
                streamed_text
                and not is_placeholder_answer(streamed_text)
                and "see process log" not in streamed_text.lower()
                and len(streamed_text) > len(answer)
                and not is_placeholder_answer(answer)
            ):
                answer = streamed_text

            # Persist before FINAL so a client disconnect / tab refresh right after the last
            # visible token does not skip storage (save used to run only after yielding FINAL).
            if request.use_memory and settings.MEMORY_ENABLED and settings.MEMORY_AUTO_SAVE:
                try:
                    from .memory import get_advanced_memory_client, MemoryCategory

                    memory_client = get_advanced_memory_client()
                    memory_client.save_conversation_background(
                        user_id=request.user_id,
                        session_id=request.session_id,
                        user_message=request.message,
                        assistant_response=answer,
                        category=MemoryCategory.CONTEXT,
                        is_long_term=False,
                        metadata={
                            "tools_used": [t.get("tool_id") for t in tool_runs],
                            "trace_id": trace_id
                        }
                    )
                except Exception as e:
                    log.warning(f"Memory save failed: {e}")

            if settings.CHAT_HISTORY_ENABLED and answer:
                try:
                    from .chat_history import save_chat_turn

                    turn_metadata = {"trace_id": trace_id}
                    if done_discourse_focus:
                        turn_metadata["discourse_focus"] = done_discourse_focus
                    if done_conversation_summary:
                        turn_metadata["conversation_summary"] = done_conversation_summary
                    if done_task_checklist:
                        turn_metadata["task_checklist"] = done_task_checklist
                    turn_self_compact = workspace.turn_self_compact(done_self_compact)
                    if turn_self_compact:
                        turn_metadata["self_compact"] = turn_self_compact
                    if served_by:
                        turn_metadata["engine"] = served_by
                    if workspace.long_input:
                        turn_metadata["long_input"] = workspace.long_input
                    await save_chat_turn(
                        session_id=request.session_id,
                        user_id=request.user_id,
                        user_message=request.message,
                        assistant_response=answer,
                        thinking=thinking,
                        tool_runs=tool_runs,
                        attachments=None if user_saved_early else (
                            original_artifacts if original_artifacts else None
                        ),
                        metadata=turn_metadata,
                        tokens_prompt=tokens_prompt_fs,
                        tokens_completion=tokens_completion_fs,
                        skip_user=user_saved_early,
                    )
                    turn_persisted = True
                except Exception as e:
                    log.warning(
                        f"Chat history save failed: session={request.session_id} err={e}",
                        exc_info=True,
                    )

            yield format_sse_event(StreamEventTypes.FINAL, {
                "answer": answer,
                "tool_runs": tool_runs,
                "suggestions": suggestions,
                "warnings": warnings,
                "thinking": thinking,
                "file_context": file_context,
                "trace_id": trace_id,
                "steps": steps,
                "processing_time": elapsed
            })
            
            # رویداد اتمام
            yield format_sse_event(StreamEventTypes.DONE, {
                "trace_id": trace_id,
                "elapsed": elapsed
            })
            turn_completed = True
            
            log.info(
                "✅ Streaming complete",
                session_id=request.session_id,
                trace_id=trace_id,
                steps=steps,
                elapsed=f"{elapsed:.2f}s"
            )
            
        except Exception as exc:
            stream_error_message = str(exc)
            log.error(f"Streaming error: {exc}", exc_info=True)
            yield format_sse_event(StreamEventTypes.ERROR, {
                "message": str(exc),
                "code": "STREAM_ERROR"
            })
            yield format_sse_event(StreamEventTypes.DONE, {
                "trace_id": trace_id,
                "error": True
            })
            turn_completed = True
        finally:
          # A stop cancels this generator; shield the cleanup so its awaits are not
          # cancelled too (bounded by CHAT_STOP_CLEANUP_TIMEOUT_S).
          with anyio.CancelScope(shield=True) as cleanup_scope:
            cleanup_scope.deadline = anyio.current_time() + float(settings.CHAT_STOP_CLEANUP_TIMEOUT_S or 15.0)
            if not turn_completed and stream_generator is not None:
                try:
                    await stream_generator.aclose()
                except BaseException as e:  # noqa: BLE001 — engines may raise while closing
                    log.warning("Engine generator close failed: session=%s err=%r", request.session_id, e)
            # Best-effort persist when the client aborts mid-stream (refresh/navigate)
            # or the outer handler fails before the success-path save.
            if (
                settings.CHAT_HISTORY_ENABLED
                and not turn_persisted
                and (user_saved_early or (request.message or "").strip())
            ):
                try:
                    from .chat_history import save_chat_turn

                    streamed_partial = "".join(
                        str(x) for x in (collected_answer or []) if x is not None
                    ).strip()
                    thinking_partial = (
                        (done_thinking or "").strip()
                        or "".join(
                            str(x)
                            for x in (collected_thinking_parts or [])
                            if x is not None
                        ).strip()
                    )
                    assistant_text = (
                        (done_final_answer or "").strip()
                        or streamed_partial
                        or (stream_error_message or "").strip()
                        or "_(turn interrupted before a final answer)_"
                    )
                    await save_chat_turn(
                        session_id=request.session_id,
                        user_id=request.user_id,
                        user_message=request.message,
                        assistant_response=assistant_text,
                        thinking=thinking_partial or None,
                        tool_runs=done_tool_runs or None,
                        attachments=None if user_saved_early else (
                            original_artifacts if original_artifacts else None
                        ),
                        metadata={
                            "trace_id": trace_id,
                            "partial": True,
                            **({"error": stream_error_message} if stream_error_message else {}),
                            **({"self_compact": session_checkpoint} if session_checkpoint else {}),
                        },
                        tokens_prompt=done_tokens_prompt,
                        tokens_completion=done_tokens_completion,
                        skip_user=user_saved_early,
                    )
                    turn_persisted = True
                    log.info(
                        "Chat history best-effort save after incomplete stream session=%s",
                        request.session_id,
                    )
                except Exception as e:
                    log.warning(
                        "Chat history finally-save failed: session=%s err=%s",
                        request.session_id,
                        e,
                        exc_info=True,
                    )
            if not turn_completed and settings.CHAT_STOP_RELEASE_ENGINE_STATE:
                from .orchestrators.chat_engine_resolve import release_chat_session

                released = await release_chat_session(
                    request.session_id, user_id=request.user_id, reason="stopped"
                )
                log.info(
                    "chat turn stopped early",
                    session_id=request.session_id,
                    trace_id=trace_id,
                    partial_saved=turn_persisted,
                    released=released,
                )
    
    from .chat_turns import chat_turns, follow

    previous_turn = chat_turns.get(request.session_id)

    async def run_turn() -> AsyncGenerator[str, None]:
        if previous_turn is not None and previous_turn.running:
            yield format_sse_event(StreamEventTypes.STATUS, {
                "stage": "waiting_previous_turn",
                "previous_trace_id": previous_turn.trace_id,
                "trace_id": trace_id,
            })
            if not await chat_turns.wait_for(previous_turn):
                yield format_sse_event(StreamEventTypes.ERROR, {
                    "message": "previous turn of this chat is still running",
                    "code": "TURN_BUSY",
                })
                yield format_sse_event(StreamEventTypes.DONE, {"trace_id": trace_id, "error": True})
                return
        async for event in generate_stream():
            yield event

    turn = chat_turns.start(request.session_id, request.user_id or "", trace_id, run_turn())
    return StreamingResponse(
        follow(turn),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # برای nginx
            "X-Trace-ID": trace_id
        }
    )


@app.get("/chat/sessions/{session_id}/turn")
async def chat_turn_status(session_id: str):
    from .chat_turns import chat_turns

    turn = chat_turns.get(session_id)
    return turn.info() if turn else {"session_id": session_id, "running": False}


@app.get("/chat/sessions/{session_id}/turn/stream")
async def chat_turn_attach(session_id: str, start: int = 0):
    """Reattach to a turn (e.g. after a page change): replay buffered events, then follow."""
    from .chat_turns import chat_turns, follow

    turn = chat_turns.get(session_id)
    if not turn:
        raise HTTPException(status_code=404, detail="no active turn for this chat")
    return StreamingResponse(
        follow(turn, start),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Trace-ID": turn.trace_id},
    )


class ChatStopRequest(BaseModel):
    user_id: Optional[str] = None


@app.post("/chat/sessions/{session_id}/stop")
async def chat_turn_stop(session_id: str, body: Optional[ChatStopRequest] = None):
    """Explicit stop (the user's Stop button): cancel the running turn and release engine state."""
    from .chat_turns import chat_turns

    try:
        info = await chat_turns.stop(session_id, user_id=(body.user_id if body else "") or "")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    if info is None or not info.get("stopped"):
        info = dict(info or {"session_id": session_id, "running": False, "stopped": False})
        if settings.CHAT_STOP_RELEASE_ENGINE_STATE:
            from .orchestrators.chat_engine_resolve import release_chat_session

            info["released"] = await release_chat_session(
                session_id, user_id=(body.user_id if body else "") or "", reason="stopped"
            )
    return info


# ==================== Tool Endpoints ====================

@app.get("/tools", response_model=ToolsListResponse)
async def list_tools():
    """لیست ابزارهای موجود."""
    try:
        tool_executor = get_tool_executor()
        manifest = await tool_executor.get_tools_manifest()
        
        tools = [
            ToolInfo(
                id=t.get("id", ""),
                name=t.get("name", t.get("id", "")),
                description=t.get("description", ""),
                args=t.get("args", {})
            )
            for t in manifest
        ]
        
        return ToolsListResponse(tools=tools, count=len(tools))
        
    except Exception as exc:
        log.error(f"Failed to list tools: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/tools/run", response_model=DirectToolResponse)
async def run_tool_direct(request: DirectToolRequest):
    """اجرای مستقیم ابزار (بدون orchestrator)."""
    start_time = time.time()
    
    try:
        tool_executor = get_tool_executor()
        result = await tool_executor.execute(
            tool_name=request.tool_id,
            arguments=request.args,
            session_id=request.session_id,
            user_id=request.args.get("user_id") if isinstance(request.args, dict) else None,
            group_id=request.args.get("group_id") if isinstance(request.args, dict) else None,
        )
        
        elapsed = time.time() - start_time
        
        return DirectToolResponse(
            ok=result.get("ok", False),
            data=result.get("data"),
            error=result.get("error"),
            tool_id=request.tool_id,
            execution_time=elapsed
        )
        
    except Exception as exc:
        elapsed = time.time() - start_time
        return DirectToolResponse(
            ok=False,
            error=str(exc),
            tool_id=request.tool_id,
            execution_time=elapsed
        )


# ==================== Memory Management Endpoints ====================

@app.get("/memory/session/{session_id}")
async def get_session_memories(
    session_id: str,
    user_id: str,
    limit: int = 50
):
    """
    دریافت حافظه‌های یک session خاص.
    """
    if not settings.MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory service disabled")
    
    try:
        from .memory import get_advanced_memory_client
        
        memory_client = get_advanced_memory_client()
        memories = await memory_client.get_all_session_memories(
            user_id=user_id,
            session_id=session_id,
            limit=limit
        )
        
        return {
            "session_id": session_id,
            "user_id": user_id,
            "count": len(memories),
            "memories": memories
        }
    except Exception as e:
        log.error(f"Get session memories failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/user/{user_id}")
async def get_user_memories(
    user_id: str,
    limit: int = 100
):
    """
    دریافت همه حافظه‌های یک کاربر.
    """
    if not settings.MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory service disabled")
    
    try:
        from .memory import get_advanced_memory_client
        
        memory_client = get_advanced_memory_client()
        memories = await memory_client.get_all_user_memories(
            user_id=user_id,
            limit=limit
        )
        
        return {
            "user_id": user_id,
            "count": len(memories),
            "memories": memories
        }
    except Exception as e:
        log.error(f"Get user memories failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/search")
async def search_memories(
    query: str,
    user_id: str,
    session_id: str = None,
    scope: str = "combined",
    limit: int = 10
):
    """
    جستجو در حافظه‌ها.
    
    scope: session, user, combined
    """
    if not settings.MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory service disabled")
    
    try:
        from .memory import get_advanced_memory_client, MemoryScope
        
        memory_client = get_advanced_memory_client()
        
        scope_map = {
            "session": MemoryScope.SESSION,
            "user": MemoryScope.USER,
            "combined": MemoryScope.COMBINED
        }
        
        memories = await memory_client.get_relevant_memories(
            query=query,
            user_id=user_id,
            session_id=session_id or "default",
            scope=scope_map.get(scope, MemoryScope.COMBINED),
            limit=limit
        )
        
        return {
            "query": query,
            "scope": scope,
            "session_memories": len(memories.get("session", [])),
            "user_memories": len(memories.get("user", [])),
            "memories": memories
        }
    except Exception as e:
        log.error(f"Memory search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/add")
async def add_memory(
    content: str,
    user_id: str,
    session_id: str,
    category: str = "context",
    is_long_term: bool = False
):
    """
    افزودن حافظه جدید.
    """
    if not settings.MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory service disabled")
    
    try:
        from .memory import get_advanced_memory_client
        
        memory_client = get_advanced_memory_client()
        success = await memory_client.add_memory(
            content=content,
            user_id=user_id,
            session_id=session_id,
            category=category,
            is_long_term=is_long_term
        )
        
        return {"success": success, "message": "Memory added" if success else "Failed to add memory"}
    except Exception as e:
        log.error(f"Add memory failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/save-preference")
async def save_user_preference(
    user_id: str,
    session_id: str,
    preference: str
):
    """
    ذخیره ترجیح کاربر (بلندمدت).
    """
    if not settings.MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory service disabled")
    
    try:
        from .memory import get_advanced_memory_client
        
        memory_client = get_advanced_memory_client()
        success = await memory_client.save_preference(
            user_id=user_id,
            session_id=session_id,
            preference=preference
        )
        
        return {"success": success, "is_long_term": True}
    except Exception as e:
        log.error(f"Save preference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/session/{session_id}")
async def clear_session_memories(
    session_id: str,
    user_id: str
):
    """
    پاک کردن حافظه‌های یک session.
    """
    if not settings.MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory service disabled")
    
    try:
        from .memory import get_advanced_memory_client
        
        memory_client = get_advanced_memory_client()
        success = await memory_client.clear_session_memories(
            user_id=user_id,
            session_id=session_id
        )
        
        return {"success": success, "message": f"Session {session_id} memories cleared"}
    except Exception as e:
        log.error(f"Clear session memories failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/user/{user_id}")
async def clear_user_memories(user_id: str):
    """
    پاک کردن همه حافظه‌های کاربر.
    """
    if not settings.MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory service disabled")
    
    try:
        from .memory import get_advanced_memory_client
        
        memory_client = get_advanced_memory_client()
        success = await memory_client.clear_user_memories(user_id=user_id)
        
        return {"success": success, "message": f"All memories for user {user_id} cleared"}
    except Exception as e:
        log.error(f"Clear user memories failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/status")
async def memory_status():
    """Memory backend readiness (Qdrant reachability + embed binding)."""
    from .memory import get_advanced_memory_client

    return await get_advanced_memory_client().status()


@app.put("/memory/item/{memory_id}")
async def update_memory_item(memory_id: str, user_id: str, content: str):
    if not settings.MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory service disabled")
    from .memory import get_advanced_memory_client

    success = await get_advanced_memory_client().update_memory(memory_id, content, user_id=user_id)
    if not success:
        raise HTTPException(status_code=404, detail="memory_not_found")
    return {"success": True, "id": memory_id}


@app.delete("/memory/item/{memory_id}")
async def delete_memory_item(memory_id: str, user_id: str):
    if not settings.MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory service disabled")
    from .memory import get_advanced_memory_client

    success = await get_advanced_memory_client().delete_memory(memory_id, user_id=user_id)
    return {"success": success, "id": memory_id}


@app.get("/memory/item/{memory_id}/history")
async def memory_item_history(memory_id: str, user_id: str):
    if not settings.MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Memory service disabled")
    from .memory import get_advanced_memory_client

    history = await get_advanced_memory_client().get_memory_history(memory_id, user_id=user_id)
    return {"id": memory_id, "history": history}


# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            code=f"HTTP_{exc.status_code}"
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.DEBUG else None,
            code="INTERNAL_ERROR"
        ).model_dump()
    )


# ==================== Root ====================

@app.get("/")
async def root():
    """صفحه اصلی."""
    return {
        "service": "AI Orchestrator",
        "version": "2.0.0",
        "status": "running",
        "features": {
            "memory": {
                "enabled": settings.MEMORY_ENABLED,
                "session_isolation": True,
                "auto_save": settings.MEMORY_AUTO_SAVE,
            }
        },
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "chat_stream": "/chat/stream",
            "chat_stream_full": "/chat/stream/full",
            "chat_agent_turn": "POST /chat/sessions/{session_id}/agent-turn",
            "tools": "/tools",
            "run_tool": "/tools/run",
            "memory": {
                "search": "POST /memory/search",
                "add": "POST /memory/add",
                "session": "GET /memory/session/{session_id}",
                "user": "GET /memory/user/{user_id}",
                "save_preference": "POST /memory/save-preference",
                "clear_session": "DELETE /memory/session/{session_id}",
                "clear_user": "DELETE /memory/user/{user_id}",
                "status": "GET /memory/status",
                "update_item": "PUT /memory/item/{memory_id}",
                "delete_item": "DELETE /memory/item/{memory_id}",
                "item_history": "GET /memory/item/{memory_id}/history"
            }
        }
    }


# ==================== Development Mode ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="debug"
    )
