"""
Pydantic Schemas for Orchestrator API
اسکیمای پیشرفته درخواست و پاسخ با تمام ویژگی‌های UX
"""
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# ==================== Enums ====================

class StreamEventType(str, Enum):
    """انواع رویداد استریم."""
    THINKING = "thinking"           # نمایش فرآیند فکر کردن
    TOKEN = "token"                 # توکن جدید
    TOOL_START = "tool_start"       # شروع اجرای ابزار
    TOOL_PROGRESS = "tool_progress" # پیشرفت ابزار
    TOOL_RESULT = "tool_result"     # نتیجه ابزار
    SUGGESTION = "suggestion"       # پیشنهاد
    WARNING = "warning"             # هشدار
    ERROR = "error"                 # خطا
    FINAL = "final"                 # پاسخ نهایی
    STATUS = "status"               # وضعیت
    DONE = "done"                   # اتمام


class ToolStatus(str, Enum):
    """وضعیت اجرای ابزار."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ==================== Artifacts ====================

class ArtifactInput(BaseModel):
    """Artifact ورودی."""
    id: Optional[str] = None
    path: str
    name: Optional[str] = None
    mime_type: Optional[str] = None


class FileContext(BaseModel):
    """Context فایل با اطلاعات کامل."""
    type: str = Field(..., description="نوع فایل: image, audio, video, document")
    name: str = Field(..., description="نام فایل")
    path: str = Field(..., description="مسیر فایل")
    mime_type: Optional[str] = Field(default=None, description="MIME type")
    size: Optional[int] = Field(default=None, description="سایز به بایت")
    suggested_tools: List[str] = Field(default_factory=list, description="ابزارهای پیشنهادی")


# ==================== Chat Request/Response ====================

class ChatRequest(BaseModel):
    """درخواست چت پیشرفته."""
    message: str = Field(..., description="پیام کاربر")
    session_id: str = Field(..., description="شناسه سشن")
    user_id: str = Field(default="default", description="شناسه کاربر")
    artifacts: List[ArtifactInput] = Field(default_factory=list, description="فایل‌های ضمیمه")
    max_steps: Optional[int] = Field(default=None, description="حداکثر گام‌ها")
    use_memory: bool = Field(default=True, description="استفاده از حافظه")
    use_rag: bool = Field(
        default=True,
        description="Native corpus RAG قبل از پاسخ مدل (ANN سراسری + rerank)",
    )
    # ویژگی‌های پیشرفته
    streaming: bool = Field(default=True, description="فعال بودن استریم")
    show_thinking: bool = Field(default=True, description="نمایش فرآیند فکر")
    include_suggestions: bool = Field(default=True, description="شامل پیشنهادات")
    context: Optional[Dict[str, Any]] = Field(default=None, description="context اضافی")
    # Optional LLM / engine overrides (gateway may forward these)
    model: Optional[str] = Field(default=None, description="مدل LLM (اختیاری)")
    orchestrator_id: Optional[str] = Field(
        default=None,
        description="شناسه ارکستراتور drop-in (native_tools, rlm, …)",
    )
    orchestrator_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="تنظیمات اختیاری ارکستراتور (از settings_ui؛ موتورهای فعلی نادیده می‌گیرند مگر host آن‌ها را بخواند)",
    )


class ToolRunInfo(BaseModel):
    """اطلاعات کامل اجرای ابزار."""
    tool_id: str
    args: Dict[str, Any] = Field(default_factory=dict)
    result: Dict[str, Any] = Field(default_factory=dict)
    step: int = 0
    status: ToolStatus = ToolStatus.SUCCESS
    execution_time: float = Field(default=0.0, description="زمان اجرا به ثانیه")
    error: Optional[str] = Field(default=None, description="پیام خطا")
    started_at: Optional[datetime] = Field(default=None, description="زمان شروع")
    completed_at: Optional[datetime] = Field(default=None, description="زمان پایان")


class SuggestionItem(BaseModel):
    """پیشنهاد به کاربر."""
    text: str = Field(..., description="متن پیشنهاد")
    action_type: Literal["query", "tool", "info"] = Field(default="query")
    data: Optional[Dict[str, Any]] = Field(default=None, description="داده اضافی")


class WarningItem(BaseModel):
    """هشدار به کاربر."""
    message: str = Field(..., description="متن هشدار")
    level: Literal["info", "warning", "error"] = Field(default="warning")
    code: Optional[str] = Field(default=None, description="کد هشدار")


class ChatResponse(BaseModel):
    """پاسخ چت پیشرفته با تمام ویژگی‌ها."""
    answer: str = Field(..., description="پاسخ نهایی")
    tool_runs: List[ToolRunInfo] = Field(default_factory=list, description="ابزارهای اجرا شده")
    trace_id: str = Field(..., description="شناسه ردیابی")
    steps: int = Field(default=0, description="تعداد گام‌ها")
    file_context: Optional[FileContext] = Field(default=None, description="context فایل")
    # ویژگی‌های پیشرفته UX
    thinking: Optional[str] = Field(default=None, description="فرآیند فکر کردن")
    suggestions: List[SuggestionItem] = Field(default_factory=list, description="پیشنهادات")
    warnings: List[WarningItem] = Field(default_factory=list, description="هشدارها")
    # متادیتا
    model: Optional[str] = Field(default=None, description="مدل استفاده شده")
    total_tokens: Optional[int] = Field(default=None, description="تعداد توکن‌ها")
    processing_time: Optional[float] = Field(default=None, description="زمان پردازش")
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ==================== Streaming ====================

class StreamChunk(BaseModel):
    """چانک استریم پیشرفته."""
    type: StreamEventType = Field(..., description="نوع رویداد")
    content: Optional[str] = Field(default=None, description="محتوا")
    # برای ابزارها
    tool_id: Optional[str] = Field(default=None, description="شناسه ابزار")
    tool_name: Optional[str] = Field(default=None, description="نام ابزار")
    tool_args: Optional[Dict[str, Any]] = Field(default=None, description="آرگومان‌ها")
    tool_result: Optional[Dict[str, Any]] = Field(default=None, description="نتیجه")
    tool_status: Optional[ToolStatus] = Field(default=None, description="وضعیت")
    # پیشرفت
    progress: Optional[float] = Field(default=None, ge=0, le=100, description="درصد پیشرفت")
    step: Optional[int] = Field(default=None, description="شماره گام")
    total_steps: Optional[int] = Field(default=None, description="کل گام‌ها")
    # متادیتا
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StreamStartEvent(BaseModel):
    """رویداد شروع استریم."""
    trace_id: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StreamEndEvent(BaseModel):
    """رویداد پایان استریم."""
    trace_id: str
    response: ChatResponse
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ==================== Health Check ====================

class HealthResponse(BaseModel):
    """پاسخ health check."""
    status: str
    llm_ready: bool
    tool_runner_ready: bool
    storage_ready: bool = False
    memory_ready: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ==================== Tool Management ====================

class ToolInfo(BaseModel):
    """اطلاعات ابزار."""
    id: str
    name: str
    description: str
    args: Dict[str, Any] = Field(default_factory=dict)


class ToolsListResponse(BaseModel):
    """لیست ابزارها."""
    tools: List[ToolInfo]
    count: int


class DirectToolRequest(BaseModel):
    """درخواست اجرای مستقیم ابزار."""
    tool_id: str
    args: Dict[str, Any] = Field(default_factory=dict)
    session_id: str


class DirectToolResponse(BaseModel):
    """پاسخ اجرای مستقیم ابزار."""
    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tool_id: str
    execution_time: float = 0.0


# ==================== Error Response ====================

class ErrorResponse(BaseModel):
    """پاسخ خطا."""
    error: str
    detail: Optional[str] = None
    code: str = "INTERNAL_ERROR"
