"""
Advanced LangGraph Agent with All UX Features
گراف عامل پیشرفته با تمام ویژگی‌های UX
"""
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, AsyncGenerator
from dataclasses import dataclass
import asyncio

from langgraph.graph import StateGraph, END

from .state import AgentState, create_initial_state
from .nodes import (
    initialize_node,
    call_llm_node,
    execute_tools_node,
    reflection_node,
    force_final_node,
    generate_suggestions_node
)
from ..config import settings
from ..llm.gateway_llm_client import get_gateway_llm_client
from ..tools.executor import get_tool_executor
from ..prompts.system import build_system_messages
from ..utils.file_context import detect_file_context

log = logging.getLogger(__name__)


# ==================== Dependencies ====================

@dataclass
class GraphDependencies:
    """وابستگی‌های گراف."""
    llm_client: Any
    tool_executor: Any
    build_system_messages: Any
    detect_file_context: Any


def get_dependencies() -> Dict[str, Any]:
    """دریافت وابستگی‌های گراف."""
    return {
        "llm_client": get_gateway_llm_client(),
        "tool_executor": get_tool_executor(),
        "build_system_messages": build_system_messages,
        "detect_file_context": detect_file_context,
    }


# ==================== Routing Functions ====================

def should_continue(state: AgentState) -> Literal["execute_tools", "force_final", "suggestions", "end"]:
    """
    تصمیم‌گیری برای ادامه یا پایان.
    
    Logic:
    1. اگر خطا داریم -> end
    2. اگر پاسخ نهایی داریم -> suggestions (برای UX بهتر)
    3. اگر tool_calls داریم و هنوز step مانده -> execute_tools
    4. اگر max_steps رسیدیم -> force_final
    5. اگر پاسخ بدون tool_calls داریم -> suggestions
    """
    
    # خطا
    if state.get("error"):
        log.warning(f"Ending due to error: {state['error']}")
        return "end"
    
    # پاسخ نهایی موجود
    if state.get("final_answer"):
        return "suggestions"
    
    # ابزار در انتظار
    if state.get("pending_tool_calls"):
        # بررسی حداکثر گام
        if state.get("current_step", 0) >= state.get("max_steps", 5):
            return "force_final"
        return "execute_tools"
    
    # بررسی پاسخ assistant
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1] if isinstance(messages[-1], dict) else {}
        
        # اگر tool_calls نداشته باشد و محتوا داشته باشد = پاسخ نهایی
        if last_msg.get("role") == "assistant" and last_msg.get("content"):
            if not last_msg.get("tool_calls"):
                # استخراج final_answer
                return "suggestions"
    
    # حداکثر گام
    if state.get("current_step", 0) >= state.get("max_steps", 5):
        return "force_final"
    
    return "end"


def after_reflection(state: AgentState) -> Literal["call_llm", "force_final"]:
    """Route after reflection/self-evaluation node."""
    if state.get("current_step", 0) >= state.get("max_steps", 5):
        return "force_final"
    return "call_llm"


def after_suggestions(state: AgentState) -> Literal["end"]:
    """بعد از تولید پیشنهادات."""
    return "end"


# ==================== Node Wrappers ====================

# برای inject کردن dependencies به nodes

async def wrapped_initialize(state: AgentState) -> Dict[str, Any]:
    deps = get_dependencies()
    return await initialize_node(state, deps)


async def wrapped_call_llm(state: AgentState) -> Dict[str, Any]:
    deps = get_dependencies()
    return await call_llm_node(state, deps)


async def wrapped_execute_tools(state: AgentState) -> Dict[str, Any]:
    deps = get_dependencies()
    return await execute_tools_node(state, deps)


async def wrapped_reflection(state: AgentState) -> Dict[str, Any]:
    deps = get_dependencies()
    return await reflection_node(state, deps)


async def wrapped_force_final(state: AgentState) -> Dict[str, Any]:
    deps = get_dependencies()
    return await force_final_node(state, deps)


async def wrapped_suggestions(state: AgentState) -> Dict[str, Any]:
    deps = get_dependencies()
    return await generate_suggestions_node(state, deps)


async def extract_final_answer(state: AgentState) -> Dict[str, Any]:
    """استخراج پاسخ نهایی از messages."""
    from .nodes import extract_thinking
    
    if state.get("final_answer"):
        return {}
    
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content", "")
            if content and not msg.get("tool_calls"):
                # استخراج thinking اگه هنوز توی content هست
                thinking, clean_content = extract_thinking(content)
                result = {
                    "final_answer": clean_content,
                    "current_status": "completed",
                    "completed_at": datetime.utcnow().isoformat()
                }
                if thinking and not state.get("thinking"):
                    result["thinking"] = thinking
                return result
    
    return {
        "final_answer": "پاسخی دریافت نشد.",
        "current_status": "completed",
        "completed_at": datetime.utcnow().isoformat()
    }


# ==================== Graph Builder ====================

def build_agent_graph() -> StateGraph:
    """
    ساخت گراف عامل پیشرفته.
    
    Nodes:
    - initialize: آماده‌سازی اولیه
    - call_llm: فراخوانی LLM با tool calling
    - execute_tools: اجرای موازی ابزارها
    - force_final: پاسخ اجباری در پایان
    - extract_answer: استخراج پاسخ نهایی
    - suggestions: تولید پیشنهادات
    
    Flow:
    initialize -> call_llm -> (execute_tools -> call_llm)* -> extract_answer -> suggestions -> end
    """
    
    workflow = StateGraph(AgentState)
    
    # اضافه کردن گره‌ها
    workflow.add_node("initialize", wrapped_initialize)
    workflow.add_node("call_llm", wrapped_call_llm)
    workflow.add_node("execute_tools", wrapped_execute_tools)
    workflow.add_node("reflection", wrapped_reflection)
    workflow.add_node("force_final", wrapped_force_final)
    workflow.add_node("extract_answer", extract_final_answer)
    workflow.add_node("suggestions", wrapped_suggestions)
    
    # نقطه ورود
    workflow.set_entry_point("initialize")
    
    # یال‌ها
    workflow.add_edge("initialize", "call_llm")
    
    workflow.add_conditional_edges(
        "call_llm",
        should_continue,
        {
            "execute_tools": "execute_tools",
            "force_final": "force_final",
            "suggestions": "extract_answer",
            "end": END
        }
    )
    
    # execute_tools → reflection (self-evaluation)
    if settings.REFLECTION_ENABLED:
        workflow.add_edge("execute_tools", "reflection")
        workflow.add_conditional_edges(
            "reflection",
            after_reflection,
            {
                "call_llm": "call_llm",
                "force_final": "force_final"
            }
        )
    else:
        # Without reflection: direct tools → call_llm loop
        workflow.add_conditional_edges(
            "execute_tools",
            after_reflection,
            {
                "call_llm": "call_llm",
                "force_final": "force_final"
            }
        )
    
    workflow.add_edge("force_final", "suggestions")
    workflow.add_edge("extract_answer", "suggestions")
    workflow.add_edge("suggestions", END)
    
    return workflow.compile()


# ==================== Compiled Graph ====================

agent_graph = build_agent_graph()


# ==================== Public API ====================

async def run_agent_graph(
    user_message: str,
    session_id: str,
    artifacts: List[Dict[str, Any]] = None,
    memory_context: List[Dict[str, Any]] = None,
    user_id: str = "default",
    max_steps: int = None,
    chat_history: List[Dict[str, str]] = None  # ✨ تاریخچه چت
) -> Dict[str, Any]:
    """
    اجرای گراف عامل (بدون streaming).
    
    Args:
        user_message: پیام کاربر
        session_id: شناسه سشن
        artifacts: فایل‌های ضمیمه
        memory_context: context حافظه
        user_id: شناسه کاربر
        max_steps: حداکثر گام‌ها
        chat_history: تاریخچه پیام‌های قبلی
    
    Returns:
        Dict with 'answer', 'tool_runs', 'trace_id', 'suggestions', etc.
    """
    
    initial_state = create_initial_state(
        user_message=user_message,
        session_id=session_id,
        user_id=user_id,
        artifacts=artifacts,
        memory_context=memory_context,
        max_steps=max_steps or settings.MAX_STEPS,
        is_streaming=False,
        chat_history=chat_history  # ✨ ارسال تاریخچه
    )
    
    try:
        log.info(f"🎯 Running agent graph for session: {session_id}")
        
        result = await agent_graph.ainvoke(initial_state)
        
        # استخراج پاسخ نهایی
        answer = result.get("final_answer")
        if not answer:
            answer = "متأسفانه پاسخی دریافت نشد."
        
        # محاسبه زمان اجرا
        started_at = result.get("started_at")
        completed_at = result.get("completed_at") or datetime.utcnow().isoformat()
        
        return {
            "answer": answer,
            "thinking": result.get("thinking"),
            "tool_runs": result.get("tool_runs", []),
            "trace_id": result.get("trace_id"),
            "steps": result.get("current_step", 0),
            "file_context": result.get("file_context"),
            "suggestions": result.get("suggestions", []),
            "warnings": result.get("warnings", []),
            "started_at": started_at,
            "completed_at": completed_at,
        }
        
    except Exception as exc:
        log.error(f"🔴 Agent graph failed: {exc}", exc_info=True)
        return {
            "answer": f"خطا در پردازش: {str(exc)}",
            "tool_runs": [],
            "trace_id": str(uuid.uuid4()),
            "error": str(exc)
        }


async def run_agent_graph_stream(
    user_message: str,
    session_id: str,
    artifacts: List[Dict[str, Any]] = None,
    memory_context: List[Dict[str, Any]] = None,
    user_id: str = "default",
    max_steps: int = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    اجرای گراف عامل با streaming.
    
    این generator events را yield می‌کند:
    - init: شروع
    - thinking_start: شروع تفکر
    - thinking: محتوای تفکر
    - tool_selection: انتخاب ابزارها
    - tool_start: شروع اجرای ابزار
    - tool_result: نتیجه ابزار
    - tools_complete: پایان اجرای ابزارها
    - answer_start: شروع پاسخ
    - answer_chunk: بخشی از پاسخ
    - suggestions: پیشنهادات
    - done: پایان
    - error: خطا
    """
    
    initial_state = create_initial_state(
        user_message=user_message,
        session_id=session_id,
        user_id=user_id,
        artifacts=artifacts,
        memory_context=memory_context,
        max_steps=max_steps or settings.MAX_STEPS,
        is_streaming=True
    )
    
    trace_id = initial_state["trace_id"]
    
    try:
        log.info(f"🎯 Running agent graph (streaming) for session: {session_id}")
        
        # اجرای گراف با stream
        async for step_output in agent_graph.astream(initial_state):
            # هر step_output یک dict با نام node و output است
            for node_name, node_output in step_output.items():
                if node_output is None:
                    continue
                
                # yield کردن events از state
                stream_events = node_output.get("stream_events", [])
                for event in stream_events:
                    event["trace_id"] = trace_id
                    yield event
                
                # اگر پاسخ نهایی داریم
                if node_output.get("final_answer"):
                    yield {
                        "type": "answer",
                        "content": node_output["final_answer"],
                        "trace_id": trace_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                # اگر پیشنهادات داریم
                if node_output.get("suggestions"):
                    yield {
                        "type": "suggestions",
                        "suggestions": node_output["suggestions"],
                        "trace_id": trace_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                
                # اگر خطا داریم
                if node_output.get("error"):
                    yield {
                        "type": "error",
                        "error": node_output["error"],
                        "trace_id": trace_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
        
        # Final done event
        yield {
            "type": "done",
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as exc:
        log.error(f"🔴 Agent graph stream failed: {exc}", exc_info=True)
        yield {
            "type": "error",
            "error": str(exc),
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat()
        }


# ==================== Compatibility Exports ====================

__all__ = [
    "run_agent_graph",
    "run_agent_graph_stream",
    "agent_graph",
    "AgentState",
    "create_initial_state"
]
