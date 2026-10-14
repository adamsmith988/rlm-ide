"""
Advanced Agent State Management
مدیریت حالت پیشرفته عامل با تمام ویژگی‌های UX
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, TypedDict, Annotated
from datetime import datetime
from enum import Enum
import operator
import uuid


# Custom reducer for messages - replaces with new list
def merge_messages(left: List[Dict], right: List[Dict]) -> List[Dict]:
    """Replace message list with new one (not concatenate)."""
    # اگه right داده شده، جایگزین کن
    if right is not None:
        return right
    return left or []


class ToolStatus(str, Enum):
    """وضعیت اجرای ابزار."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    """وضعیت گام."""
    THINKING = "thinking"
    SELECTING_TOOLS = "selecting_tools"
    EXECUTING_TOOLS = "executing_tools"
    ANALYZING_RESULTS = "analyzing_results"
    GENERATING_RESPONSE = "generating_response"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ToolCall:
    """اطلاعات کامل فراخوانی ابزار."""
    id: str
    name: str
    arguments: Dict[str, Any]
    status: ToolStatus = ToolStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    
    @property
    def execution_time(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class StepInfo:
    """اطلاعات یک گام."""
    step_number: int
    status: StepStatus
    thinking: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    llm_content: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class AgentState(TypedDict):
    """
    وضعیت پیشرفته عامل با معماری Multi-Agent.
    
    ویژگی‌ها:
    - Multi-Agent Architecture: انتخاب agent تخصصی بر اساس نوع فایل
    - Context Awareness: حفظ تاریخچه و ترجیحات
    - Progressive Disclosure: اطلاعات پیشرفت هر گام
    - Error Recovery: مدیریت خطا و بازیابی
    - Multi-step Planning: برنامه‌ریزی چند مرحله‌ای
    """
    
    # =============== Messages ===============
    # پیام‌ها با استفاده از merge_messages ترکیب می‌شوند
    messages: Annotated[List[Dict[str, Any]], merge_messages]
    
    # =============== Session & User ===============
    session_id: str
    user_id: str
    trace_id: str
    
    # =============== Multi-Agent ===============
    # نوع agent انتخاب شده (vision, text, audio, file, analysis, general)
    agent_type: Optional[str]
    
    # =============== Context Awareness ===============
    # حفظ context فایل و artifacts
    artifacts: List[Dict[str, Any]]
    file_context: Optional[Dict[str, Any]]
    
    # حافظه و ترجیحات کاربر
    memory_context: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    
    # =============== Step Management ===============
    current_step: int
    max_steps: int
    steps_history: List[Dict[str, Any]]  # تاریخچه کامل گام‌ها
    current_status: str  # وضعیت فعلی برای UI
    
    # =============== Tool Execution ===============
    # ابزارهای در انتظار و نتایج
    pending_tool_calls: List[Dict[str, Any]]
    tool_runs: List[Dict[str, Any]]
    
    # ابزارهای فعال (بر اساس context فایل و agent type)
    active_tools: List[str]
    
    # =============== Planning ===============
    # برنامه‌ریزی چند مرحله‌ای
    execution_plan: Optional[List[Dict[str, Any]]]
    plan_step: int
    
    # =============== Final Output ===============
    final_answer: Optional[str]
    thinking: Optional[str]  # تفکر مدل
    
    # =============== Error Handling ===============
    error: Optional[str]
    error_recovery_attempts: int
    last_error: Optional[Dict[str, Any]]
    
    # =============== UX Features ===============
    # پیشنهادات برای کاربر
    suggestions: List[str]
    
    # هشدارها
    warnings: List[str]
    
    # Preview قبل از اجرای عملیات حساس
    pending_confirmation: Optional[Dict[str, Any]]
    
    # =============== Streaming ===============
    # برای streaming events
    stream_events: List[Dict[str, Any]]
    is_streaming: bool
    
    # =============== Performance ===============
    started_at: Optional[str]
    completed_at: Optional[str]
    
    # =============== Token Usage ===============
    # جمع توکن‌های مصرف شده در همه LLM call ها
    tokens_prompt: Optional[int]
    tokens_completion: Optional[int]
    
    # =============== Reflection & Self-Evaluation ===============
    confidence_score: float
    needs_more_info: bool
    tools_used_names: List[str]


def create_initial_state(
    user_message: str,
    session_id: str,
    user_id: str = "default",
    artifacts: List[Dict[str, Any]] = None,
    memory_context: List[Dict[str, Any]] = None,
    max_steps: int = 5,
    is_streaming: bool = False,
    chat_history: List[Dict[str, str]] = None  # ✨ تاریخچه چت
) -> AgentState:
    """Create initial state with all defaults."""
    from ..utils.file_context import detect_file_context, detect_file_context_from_message
    from ..prompts.qwen3_prompts import CHAT_HISTORY_START, CHAT_HISTORY_END
    
    trace_id = str(uuid.uuid4())
    
    # Build messages list with history
    # If chat_history exists, add history first then new message
    messages = []
    if chat_history and len(chat_history) > 0:
        # ✨ Add marker for history start (English for better LLM understanding)
        messages.append({
            "role": "system",
            "content": CHAT_HISTORY_START
        })
        messages.extend(chat_history)
        # ✨ Add marker for history end
        messages.append({
            "role": "system",
            "content": CHAT_HISTORY_END
        })
    
    # ✨ New user message (this is the current request)
    messages.append({"role": "user", "content": user_message})
    
    # تشخیص file_context
    # اول از artifacts، اگه نبود از پیام کاربر
    file_context = None
    if artifacts:
        file_context = detect_file_context(artifacts)
    if not file_context:
        file_context = detect_file_context_from_message(user_message)
    
    return AgentState(
        # Messages - شامل تاریخچه + پیام جدید
        messages=messages,
        
        # Session & User
        session_id=session_id,
        user_id=user_id,
        trace_id=trace_id,
        
        # Multi-Agent - در initialize_node تنظیم می‌شود
        agent_type=None,
        
        # Context
        artifacts=artifacts or [],
        file_context=file_context,  # ✨ حالا از پیام هم detect میشه
        memory_context=memory_context or [],
        user_preferences={},
        
        # Step Management
        current_step=0,
        max_steps=max_steps,
        steps_history=[],
        current_status="initializing",
        
        # Tool Execution
        pending_tool_calls=[],
        tool_runs=[],
        active_tools=[],
        
        # Planning
        execution_plan=None,
        plan_step=0,
        
        # Final Output
        final_answer=None,
        thinking=None,
        
        # Error Handling
        error=None,
        error_recovery_attempts=0,
        last_error=None,
        
        # UX
        suggestions=[],
        warnings=[],
        pending_confirmation=None,
        
        # Streaming
        stream_events=[],
        is_streaming=is_streaming,
        
        # Performance
        started_at=datetime.utcnow().isoformat(),
        completed_at=None,
        
        # Token Usage
        tokens_prompt=None,
        tokens_completion=None,
        
        # Reflection & Self-Evaluation
        confidence_score=0.0,
        needs_more_info=True,
        tools_used_names=[],
    )


def add_step_to_history(state: AgentState, step_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """افزودن گام به تاریخچه."""
    history = state.get("steps_history", [])
    return history + [step_info]


def add_stream_event(state: AgentState, event: Dict[str, Any]) -> List[Dict[str, Any]]:
    """افزودن event به لیست streaming."""
    events = state.get("stream_events", [])
    return events + [event]
