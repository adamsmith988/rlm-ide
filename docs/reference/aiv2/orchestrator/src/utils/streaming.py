"""
Streaming Utilities
ابزارهای streaming برای ارکستراتور
"""
import json
from typing import Any, Dict, AsyncGenerator
from datetime import datetime


def format_sse_event(event_type: str, data: Dict[str, Any] = None) -> str:
    """
    فرمت کردن event به صورت SSE.
    
    Args:
        event_type: نوع event
        data: داده‌های event
    
    Returns:
        رشته SSE formatted
    """
    event = {
        "type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        **(data or {})
    }
    data_json = json.dumps(event, ensure_ascii=False, default=str)
    return f"event: {event_type}\ndata: {data_json}\n\n"


async def stream_agent_response(
    agent_stream: AsyncGenerator[Dict[str, Any], None]
) -> AsyncGenerator[str, None]:
    """
    تبدیل stream عامل به SSE stream.
    
    Args:
        agent_stream: generator از run_agent_graph_stream
    
    Yields:
        رشته‌های SSE formatted
    """
    try:
        async for event in agent_stream:
            event_type = event.get("type", "message")
            yield format_sse_event(event_type, event)
    except Exception as exc:
        yield format_sse_event("error", {"error": str(exc)})
    finally:
        yield format_sse_event("done", {})


class StreamEventTypes:
    """انواع events برای streaming."""
    
    # Initialization
    INIT = "init"
    STATUS = "status"
    
    # Thinking
    THINKING_START = "thinking_start"
    THINKING = "thinking"
    THINKING_END = "thinking_end"
    
    # Token streaming
    TOKEN = "token"
    
    # Tool Selection
    TOOL_SELECTION = "tool_selection"
    
    # Tool Calling (streaming deltas — Continue-style)
    TOOL_CALL_START = "tool_call_start"    # LLM started generating a tool call
    TOOL_CALL_DELTA = "tool_call_delta"    # Incremental tool-argument tokens
    TOOL_PLAN = "tool_plan"                # Final tool plan (all calls decided)

    # Tool Execution
    TOOLS_START = "tools_start"
    TOOL_START = "tool_start"
    TOOL_PROGRESS = "tool_progress"
    TOOL_RESULT = "tool_result"
    TOOLS_COMPLETE = "tools_complete"
    
    # Answer
    ANSWER_START = "answer_start"
    ANSWER_CHUNK = "answer_chunk"
    ANSWER = "answer"
    FINAL = "final"
    
    # UX
    SUGGESTIONS = "suggestions"
    SUGGESTION = "suggestion"
    WARNING = "warning"
    
    # Progress
    PROGRESS = "progress"
    
    # Final
    DONE = "done"
    ERROR = "error"


def create_stream_event(
    event_type: str,
    trace_id: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    ساخت event برای streaming.
    
    Args:
        event_type: نوع event
        trace_id: شناسه ردیابی
        **kwargs: داده‌های اضافی
    
    Returns:
        dict event
    """
    event = {
        "type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    }
    if trace_id:
        event["trace_id"] = trace_id
    return event


# ==================== Streaming Response Builders ====================

def build_init_event(trace_id: str, session_id: str, file_context: Dict = None) -> Dict:
    """Event شروع."""
    return create_stream_event(
        StreamEventTypes.INIT,
        trace_id=trace_id,
        session_id=session_id,
        file_context=file_context
    )


def build_thinking_event(trace_id: str, step: int, content: str = None) -> Dict:
    """Event تفکر."""
    return create_stream_event(
        StreamEventTypes.THINKING,
        trace_id=trace_id,
        step=step,
        content=content
    )


def build_tool_selection_event(trace_id: str, step: int, tools: list) -> Dict:
    """Event انتخاب ابزارها."""
    return create_stream_event(
        StreamEventTypes.TOOL_SELECTION,
        trace_id=trace_id,
        step=step,
        tools=tools,
        count=len(tools)
    )


def build_tool_start_event(
    trace_id: str,
    tool_id: str,
    tool_name: str,
    args: Dict = None
) -> Dict:
    """Event شروع ابزار."""
    return create_stream_event(
        StreamEventTypes.TOOL_START,
        trace_id=trace_id,
        tool_id=tool_id,
        tool_name=tool_name,
        args=args or {}
    )


def build_tool_result_event(
    trace_id: str,
    tool_id: str,
    tool_name: str,
    ok: bool,
    execution_time: float,
    summary: str = None
) -> Dict:
    """Event نتیجه ابزار."""
    return create_stream_event(
        StreamEventTypes.TOOL_RESULT,
        trace_id=trace_id,
        tool_id=tool_id,
        tool_name=tool_name,
        ok=ok,
        execution_time=execution_time,
        summary=summary
    )


def build_answer_event(trace_id: str, content: str) -> Dict:
    """Event پاسخ نهایی."""
    return create_stream_event(
        StreamEventTypes.ANSWER,
        trace_id=trace_id,
        content=content
    )


def build_suggestions_event(trace_id: str, suggestions: list) -> Dict:
    """Event پیشنهادات."""
    return create_stream_event(
        StreamEventTypes.SUGGESTIONS,
        trace_id=trace_id,
        suggestions=suggestions
    )


def build_error_event(trace_id: str, error: str, recoverable: bool = False) -> Dict:
    """Event خطا."""
    return create_stream_event(
        StreamEventTypes.ERROR,
        trace_id=trace_id,
        error=error,
        recoverable=recoverable
    )


def build_done_event(trace_id: str, total_steps: int = 0, total_time: float = 0) -> Dict:
    """Event پایان."""
    return create_stream_event(
        StreamEventTypes.DONE,
        trace_id=trace_id,
        total_steps=total_steps,
        total_time=total_time
    )
