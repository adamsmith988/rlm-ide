"""
Agent Trace Client
کلاینت برای ذخیره‌سازی و بازیابی trace های اجرای agent

این ماژول trace های اجرا را از طریق Storage Service ذخیره می‌کند.
همچنین امکان ذخیره‌سازی async با queue برای عدم تأخیر در پاسخ‌دهی را دارد.
"""
import logging
import asyncio
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
import httpx

from ..config import settings

log = logging.getLogger(__name__)


# ==================== Enums ====================

class TraceStatus(str, Enum):
    STARTED = "started"
    PLANNING = "planning"
    EXECUTING = "executing"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"


class StepStatus(str, Enum):
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ToolExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


# ==================== Data Classes ====================

@dataclass
class TraceStep:
    """یک مرحله از اجرای agent."""
    trace_id: str
    step_number: int
    node_name: str
    task_id: Optional[str] = None
    task_description: Optional[str] = None
    status: str = "started"
    input_data: Optional[Dict] = None
    output_data: Optional[Dict] = None
    plan_generated: Optional[Dict] = None
    planning_reasoning: Optional[str] = None
    tool_calls_requested: Optional[List[Dict]] = None
    evaluation: Optional[Dict] = None
    confidence_score: Optional[float] = None
    needs_retry: bool = False
    needs_replan: bool = False
    synthesis_content: Optional[str] = None
    llm_messages_sent: Optional[List[Dict]] = None
    llm_response: Optional[Dict] = None
    llm_tokens_prompt: Optional[int] = None
    llm_tokens_completion: Optional[int] = None
    error: Optional[str] = None
    error_details: Optional[Dict] = None
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    duration_ms: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ToolExecution:
    """جزئیات اجرای یک ابزار."""
    trace_id: str
    tool_name: str
    arguments: Dict
    step_id: Optional[str] = None
    tool_id: Optional[str] = None
    task_id: Optional[str] = None
    status: str = "pending"
    result: Optional[Dict] = None
    result_summary: Optional[str] = None
    error: Optional[str] = None
    result_sent_to_llm: Optional[Dict] = None
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    duration_ms: Optional[int] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class StreamEvent:
    """یک event برای streaming."""
    trace_id: str
    event_type: str
    event_data: Dict
    sequence_number: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ==================== Trace Client ====================

class TraceClient:
    """
    کلاینت مدیریت trace های agent.
    
    این کلاس trace ها را از طریق Storage Service ذخیره می‌کند.
    همچنین یک queue محلی برای batch writing دارد.
    """
    
    _instance: Optional["TraceClient"] = None
    
    def __init__(self):
        self.storage_url = settings.STORAGE_URL
        self.timeout = 5.0
        self._event_sequence: Dict[str, int] = {}  # trace_id -> last sequence
        self._write_queue: asyncio.Queue = None
        self._writer_task: asyncio.Task = None
        self._enabled = True  # می‌توان غیرفعال کرد
        # Storage is an internal service and requires the service credential.
        self._headers = {
            "X-Internal-Key": settings.INTERNAL_API_KEY,
            "X-Service-Name": settings.SERVICE_NAME,
        }
    
    @classmethod
    def get_instance(cls) -> "TraceClient":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            log.info(f"✅ TraceClient initialized (via Storage: {cls._instance.storage_url})")
        return cls._instance
    
    async def _ensure_writer(self):
        """اطمینان از وجود writer task."""
        if self._write_queue is None:
            self._write_queue = asyncio.Queue()
        
        if self._writer_task is None or self._writer_task.done():
            self._writer_task = asyncio.create_task(self._background_writer())
    
    async def _background_writer(self):
        """نوشتن batch به storage در background."""
        while True:
            try:
                items = []
                # جمع‌آوری تمام items موجود در queue
                while not self._write_queue.empty() and len(items) < 50:
                    item = await asyncio.wait_for(self._write_queue.get(), timeout=1.0)
                    items.append(item)
                
                if items:
                    await self._batch_write(items)
                else:
                    await asyncio.sleep(0.1)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log.error(f"Background writer error: {e}")
                await asyncio.sleep(1.0)
    
    @staticmethod
    def _json_safe(obj):
        """تبدیل object به JSON-serializable با fallback به str."""
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            if isinstance(obj, dict):
                return {k: TraceClient._json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [TraceClient._json_safe(i) for i in obj]
            else:
                return str(obj)
    
    async def _batch_write(self, items: List[Dict]):
        """نوشتن batch به storage."""
        try:
            # اطمینان از JSON-serializable بودن همه items
            safe_items = TraceClient._json_safe(items)
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.storage_url}/traces/batch",
                    json={"items": safe_items},
                    headers=self._headers,
                )
                if response.status_code >= 400:
                    log.warning(f"Batch write failed: {response.status_code}")
        except Exception as e:
            log.error(f"Batch write error: {e}")
    
    async def flush(self):
        """Flush همه item های pending در queue به storage بفرست."""
        if self._write_queue is None or self._write_queue.empty():
            return
        try:
            items = []
            while not self._write_queue.empty():
                try:
                    items.append(self._write_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if items:
                log.debug(f"Flushing {len(items)} queued trace items")
                await self._batch_write(items)
        except Exception as e:
            log.error(f"Trace flush error: {e}")
    
    async def _queue_write(self, item_type: str, data: Dict):
        """افزودن به queue برای نوشتن غیرهمزمان."""
        if not self._enabled:
            return
        
        await self._ensure_writer()
        await self._write_queue.put({"type": item_type, "data": data})
    
    async def _direct_write(self, endpoint: str, data: Dict) -> Optional[Dict]:
        """نوشتن مستقیم (برای موارد مهم)."""
        if not self._enabled:
            return None
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.storage_url}{endpoint}",
                    json=data,
                    headers=self._headers,
                )
                if response.status_code >= 400:
                    log.warning(f"Direct write failed: {endpoint} -> {response.status_code}")
                    return None
                return response.json()
        except Exception as e:
            log.error(f"Direct write error: {endpoint} -> {e}")
            return None
    
    # ==================== Trace Management ====================
    
    async def create_trace(
        self,
        trace_id: str,
        session_id: str,
        user_message: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[Dict]:
        """ایجاد trace جدید."""
        data = {
            "trace_id": trace_id,
            "session_id": session_id,
            "user_id": user_id,
            "user_message": user_message,
            "status": TraceStatus.STARTED.value,
            "metadata": metadata or {}
        }
        
        # نوشتن مستقیم برای trace اصلی
        result = await self._direct_write("/traces", data)
        
        if result:
            self._event_sequence[trace_id] = 0
            log.debug(f"📝 Created trace: {trace_id}")
        
        return result
    
    async def update_trace(
        self,
        trace_id: str,
        status: Optional[str] = None,
        complexity: Optional[str] = None,
        execution_plan: Optional[Dict] = None,
        final_answer: Optional[str] = None,
        synthesis_reasoning: Optional[str] = None,
        overall_confidence: Optional[float] = None,
        total_steps: Optional[int] = None,
        total_tool_calls: Optional[int] = None,
        replan_count: Optional[int] = None,
        error: Optional[str] = None,
        error_details: Optional[Dict] = None,
        completed_at: Optional[str] = None,
        total_duration_ms: Optional[int] = None
    ):
        """به‌روزرسانی trace."""
        data = {
            "trace_id": trace_id,
            "status": status,
            "complexity": complexity,
            "execution_plan": execution_plan,
            "final_answer": final_answer,
            "synthesis_reasoning": synthesis_reasoning,
            "overall_confidence": overall_confidence,
            "total_steps": total_steps,
            "total_tool_calls": total_tool_calls,
            "replan_count": replan_count,
            "error": error,
            "error_details": error_details,
            "completed_at": completed_at,
            "total_duration_ms": total_duration_ms
        }
        
        # حذف None ها
        data = {k: v for k, v in data.items() if v is not None}
        
        # نوشتن مستقیم برای اطمینان از ثبت قبل از بازگشت پاسخ
        await self._batch_write([{"type": "trace_update", "data": data}])
    
    # ==================== Step Management ====================
    
    async def add_step(self, step: TraceStep) -> Optional[str]:
        """افزودن مرحله جدید."""
        await self._batch_write([{"type": "step", "data": step.to_dict()}])
        return step.trace_id
    
    async def update_step(
        self,
        trace_id: str,
        step_number: int,
        status: Optional[str] = None,
        output_data: Optional[Dict] = None,
        error: Optional[str] = None,
        completed_at: Optional[str] = None,
        duration_ms: Optional[int] = None,
        **kwargs
    ):
        """به‌روزرسانی مرحله."""
        data = {
            "trace_id": trace_id,
            "step_number": step_number,
            "status": status,
            "output_data": output_data,
            "error": error,
            "completed_at": completed_at,
            "duration_ms": duration_ms,
            **kwargs
        }
        data = {k: v for k, v in data.items() if v is not None}
        
        # نوشتن مستقیم برای اطمینان از ثبت وضعیت صحیح step
        await self._batch_write([{"type": "step_update", "data": data}])
    
    # ==================== Tool Execution ====================
    
    async def add_tool_execution(self, execution: ToolExecution) -> Optional[str]:
        """افزودن اجرای ابزار."""
        await self._batch_write([{"type": "tool_execution", "data": execution.to_dict()}])
        return execution.tool_id
    
    async def update_tool_execution(
        self,
        trace_id: str,
        tool_id: str,
        status: Optional[str] = None,
        result: Optional[Dict] = None,
        result_summary: Optional[str] = None,
        result_sent_to_llm: Optional[Dict] = None,
        error: Optional[str] = None,
        completed_at: Optional[str] = None,
        duration_ms: Optional[int] = None
    ):
        """به‌روزرسانی اجرای ابزار."""
        data = {
            "trace_id": trace_id,
            "tool_id": tool_id,
            "status": status,
            "result": result,
            "result_summary": result_summary,
            "result_sent_to_llm": result_sent_to_llm,
            "error": error,
            "completed_at": completed_at,
            "duration_ms": duration_ms
        }
        data = {k: v for k, v in data.items() if v is not None}
        
        await self._batch_write([{"type": "tool_execution_update", "data": data}])
    
    # ==================== Stream Events ====================
    
    async def add_stream_event(
        self,
        trace_id: str,
        event_type: str,
        event_data: Dict
    ) -> StreamEvent:
        """افزودن event برای streaming."""
        seq = self._event_sequence.get(trace_id, 0) + 1
        self._event_sequence[trace_id] = seq
        
        event = StreamEvent(
            trace_id=trace_id,
            event_type=event_type,
            event_data=event_data,
            sequence_number=seq
        )
        
        await self._queue_write("stream_event", event.to_dict())
        return event
    
    # ==================== Retrieval ====================
    
    async def get_trace(self, trace_id: str) -> Optional[Dict]:
        """دریافت یک trace."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.storage_url}/traces/{trace_id}", headers=self._headers)
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            log.error(f"Get trace error: {e}")
        return None
    
    async def get_trace_steps(self, trace_id: str) -> List[Dict]:
        """دریافت مراحل یک trace."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.storage_url}/traces/{trace_id}/steps", headers=self._headers)
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            log.error(f"Get trace steps error: {e}")
        return []
    
    async def get_trace_tools(self, trace_id: str) -> List[Dict]:
        """دریافت ابزارهای اجرا شده در یک trace."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.storage_url}/traces/{trace_id}/tools", headers=self._headers)
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            log.error(f"Get trace tools error: {e}")
        return []
    
    async def get_stream_events(
        self,
        trace_id: str,
        after_sequence: int = 0
    ) -> List[Dict]:
        """دریافت events برای replay."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.storage_url}/traces/{trace_id}/events",
                    params={"after": after_sequence},
                    headers=self._headers,
                )
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            log.error(f"Get stream events error: {e}")
        return []


# ==================== Singleton Access ====================

def get_trace_client() -> TraceClient:
    """دریافت instance کلاینت trace."""
    return TraceClient.get_instance()


# ==================== Exports ====================

__all__ = [
    "TraceClient",
    "get_trace_client",
    "TraceStep",
    "ToolExecution",
    "StreamEvent",
    "TraceStatus",
    "StepStatus",
    "ToolExecutionStatus",
]
