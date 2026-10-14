"""
Planning-Based Agent State Management
معماری Plan-Execute-Revise برای تفکر چند مرحله‌ای
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, TypedDict, Annotated
from datetime import datetime
from enum import Enum
import uuid

from typing_extensions import NotRequired


# ==================== Enums ====================

class TaskStatus(str, Enum):
    """وضعیت یک sub-task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class PlanStatus(str, Enum):
    """وضعیت کلی plan."""
    PLANNING = "planning"
    EXECUTING = "executing"
    REVISING = "revising"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentRole(str, Enum):
    """نقش‌های مختلف agent."""
    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"


# ==================== Data Classes ====================

@dataclass
class SubTask:
    """یک وظیفه فرعی در plan."""
    id: str
    description: str
    tools_needed: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)  # IDs of dependent tasks
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 2
    reasoning: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "tools_needed": self.tools_needed,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "retries": self.retries,
            "reasoning": self.reasoning,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubTask":
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            description=data.get("description", ""),
            tools_needed=data.get("tools_needed", []),
            depends_on=data.get("depends_on", []),
            status=TaskStatus(data.get("status", "pending")),
            result=data.get("result"),
            error=data.get("error"),
            retries=data.get("retries", 0),
            reasoning=data.get("reasoning", ""),
        )


@dataclass
class ExecutionPlan:
    """برنامه اجرایی کامل."""
    id: str
    original_query: str
    tasks: List[SubTask]
    reasoning: str
    complexity: str  # simple, moderate, complex
    estimated_steps: int
    version: int = 1  # برای re-planning
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "original_query": self.original_query,
            "tasks": [t.to_dict() for t in self.tasks],
            "reasoning": self.reasoning,
            "complexity": self.complexity,
            "estimated_steps": self.estimated_steps,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionPlan":
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            original_query=data.get("original_query", ""),
            tasks=[SubTask.from_dict(t) for t in data.get("tasks", [])],
            reasoning=data.get("reasoning", ""),
            complexity=data.get("complexity", "moderate"),
            estimated_steps=data.get("estimated_steps", 3),
            version=data.get("version", 1),
        )
    
    def get_next_task(self) -> Optional[SubTask]:
        """دریافت task بعدی که ready است."""
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                # بررسی dependencies
                deps_satisfied = all(
                    any(t.id == dep and t.status == TaskStatus.COMPLETED 
                        for t in self.tasks)
                    for dep in task.depends_on
                )
                if deps_satisfied:
                    return task
        return None
    
    def get_completed_tasks(self) -> List[SubTask]:
        """دریافت لیست task‌های تکمیل شده."""
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]
    
    def get_failed_tasks(self) -> List[SubTask]:
        """دریافت لیست task‌های شکست خورده."""
        return [t for t in self.tasks if t.status == TaskStatus.FAILED]
    
    def is_complete(self) -> bool:
        """آیا همه task‌ها تکمیل شده‌اند؟"""
        return all(t.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED] 
                   for t in self.tasks)
    
    def progress_percentage(self) -> float:
        """درصد پیشرفت."""
        if not self.tasks:
            return 0.0
        completed = len([t for t in self.tasks 
                        if t.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]])
        return (completed / len(self.tasks)) * 100


@dataclass
class CriticEvaluation:
    """ارزیابی Critic از نتیجه."""
    task_id: str
    is_successful: bool
    confidence_score: float  # 0.0 to 1.0
    issues_found: List[str]
    suggestions: List[str]
    needs_retry: bool
    needs_replan: bool
    replan_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "is_successful": self.is_successful,
            "confidence_score": self.confidence_score,
            "issues_found": self.issues_found,
            "suggestions": self.suggestions,
            "needs_retry": self.needs_retry,
            "needs_replan": self.needs_replan,
            "replan_reason": self.replan_reason,
        }


# ==================== Custom Reducers ====================

def merge_messages(left: List[Dict], right: List[Dict]) -> List[Dict]:
    """Replace message list with new one."""
    if right is not None:
        return right
    return left or []


def merge_plan(left: Optional[Dict], right: Optional[Dict]) -> Optional[Dict]:
    """Merge execution plan updates."""
    if right is not None:
        return right
    return left


# ==================== Planning Agent State ====================

class PlanningAgentState(TypedDict):
    """
    وضعیت پیشرفته عامل با معماری Plan-Execute-Revise.
    
    این state برای multi-step reasoning با قابلیت re-planning طراحی شده.
    """
    
    # =============== Messages ===============
    messages: Annotated[List[Dict[str, Any]], merge_messages]
    
    # =============== Session & User ===============
    session_id: str
    user_id: str
    trace_id: str
    
    # =============== Planning ===============
    # برنامه اجرایی
    execution_plan: Optional[Dict[str, Any]]  # ExecutionPlan.to_dict()
    plan_status: str  # PlanStatus value
    current_task_id: Optional[str]
    plan_version: int
    
    # =============== Task Execution ===============
    task_results: Dict[str, Any]  # task_id -> result
    current_task_tools: List[Dict[str, Any]]  # ابزارهای در حال اجرا برای task فعلی
    
    # =============== Critic Evaluations ===============
    evaluations: List[Dict[str, Any]]  # CriticEvaluation.to_dict()
    overall_confidence: float
    
    # =============== Re-Planning ===============
    replan_count: int
    max_replans: int
    replan_history: List[Dict[str, Any]]  # تاریخچه re-plans
    
    # =============== Multi-Agent ===============
    agent_type: Optional[str]
    current_agent_role: str  # AgentRole value
    
    # =============== Complexity Assessment ===============
    complexity: str   # "simple" | "single_tool" | "multi_step"
    needs_planning: bool  # آیا نیاز به multi-step planning داریم؟
    
    # =============== Context ===============
    artifacts: List[Dict[str, Any]]
    file_context: Optional[Dict[str, Any]]
    memory_context: List[Dict[str, Any]]
    
    # =============== Tool Execution ===============
    pending_tool_calls: List[Dict[str, Any]]
    tool_runs: List[Dict[str, Any]]
    active_tools: List[str]
    
    # =============== Final Output ===============
    final_answer: Optional[str]
    synthesis_reasoning: Optional[str]
    thinking: Optional[str]
    
    # =============== UX Features ===============
    suggestions: List[str]
    warnings: List[str]
    progress_updates: List[Dict[str, Any]]  # برای نمایش پیشرفت به کاربر
    
    # =============== Error Handling ===============
    error: Optional[str]
    error_recovery_attempts: int
    
    # =============== Streaming ===============
    stream_events: List[Dict[str, Any]]
    is_streaming: bool
    
    # =============== Performance ===============
    started_at: Optional[str]
    completed_at: Optional[str]
    
    # =============== Step Management ===============
    current_step: int
    max_steps: int
    steps_history: List[Dict[str, Any]]

    # =============== Conversation Intelligence ===============
    conversation_state: Optional[Any]
    history_summary: str
    pinned_facts: Optional[Dict[str, Any]]
    repetition_signal: Optional[Dict[str, Any]]

    # =============== File Intelligence ===============
    file_registry: List[Dict[str, Any]]

    # API ``context[\"rule_policies\"]`` for markdown session rules
    _rule_policies: NotRequired[Optional[Dict[str, Any]]]


def create_planning_state(
    user_message: str,
    session_id: str,
    user_id: str = "default",
    artifacts: List[Dict[str, Any]] = None,
    memory_context: List[Dict[str, Any]] = None,
    max_steps: int = 10,
    max_replans: int = 3,
    is_streaming: bool = False,
    chat_history: List[Dict[str, str]] = None,
    trace_id: str = None,
    conversation_state=None,
    history_summary: str = "",
    file_registry: List[Dict[str, Any]] = None,
    pinned_facts: Optional[Dict[str, Any]] = None,
    rule_policies: Optional[Dict[str, Any]] = None,
) -> PlanningAgentState:
    """ایجاد state اولیه برای Planning Agent."""
    from ..utils.file_context import detect_file_context, detect_file_context_from_message
    
    # Use provided trace_id or generate new one
    if trace_id is None:
        trace_id = str(uuid.uuid4())
    
    # Build messages
    messages = []
    if chat_history and len(chat_history) > 0:
        messages.append({
            "role": "system",
            "content": "=== CHAT HISTORY START ==="
        })
        messages.extend(chat_history)
        messages.append({
            "role": "system",
            "content": "=== CHAT HISTORY END ==="
        })
    
    messages.append({"role": "user", "content": user_message})
    
    # Detect file context
    file_context = None
    if artifacts:
        file_context = detect_file_context(artifacts)
    if not file_context:
        file_context = detect_file_context_from_message(user_message)
    
    return PlanningAgentState(
        # Messages
        messages=messages,
        
        # Session
        session_id=session_id,
        user_id=user_id,
        trace_id=trace_id,
        
        # Planning
        execution_plan=None,
        plan_status=PlanStatus.PLANNING.value,
        current_task_id=None,
        plan_version=0,
        
        # Task Execution
        task_results={},
        current_task_tools=[],
        
        # Critic
        evaluations=[],
        overall_confidence=0.0,
        
        # Re-Planning
        replan_count=0,
        max_replans=max_replans,
        replan_history=[],
        
        # Multi-Agent
        agent_type=None,
        current_agent_role=AgentRole.PLANNER.value,
        
        # Complexity Assessment (initial defaults, set by assess_complexity_node)
        complexity="simple",
        needs_planning=False,
        
        # Context
        artifacts=artifacts or [],
        file_context=file_context,
        memory_context=memory_context or [],
        
        # Tools
        pending_tool_calls=[],
        tool_runs=[],
        active_tools=[],
        
        # Output
        final_answer=None,
        synthesis_reasoning=None,
        thinking=None,
        
        # UX
        suggestions=[],
        warnings=[],
        progress_updates=[],
        
        # Error
        error=None,
        error_recovery_attempts=0,
        
        # Streaming
        stream_events=[],
        is_streaming=is_streaming,
        
        # Performance
        started_at=datetime.utcnow().isoformat(),
        completed_at=None,
        
        # Steps
        current_step=0,
        max_steps=max_steps,
        steps_history=[],

        # Conversation Intelligence
        conversation_state=conversation_state,
        history_summary=history_summary,
        pinned_facts=pinned_facts,
        repetition_signal=None,

        # File Intelligence
        file_registry=file_registry or [],

        # Markdown rules (same as streaming ReAct)
        _rule_policies=rule_policies or {},
    )


# ==================== Helper Functions ====================

def update_task_status(
    plan_dict: Dict[str, Any],
    task_id: str,
    status: TaskStatus,
    result: Any = None,
    error: str = None
) -> Dict[str, Any]:
    """به‌روزرسانی وضعیت یک task در plan."""
    plan = ExecutionPlan.from_dict(plan_dict)
    
    for task in plan.tasks:
        if task.id == task_id:
            task.status = status
            if result:
                task.result = result
            if error:
                task.error = error
            if status == TaskStatus.IN_PROGRESS:
                task.started_at = datetime.utcnow()
            elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                task.completed_at = datetime.utcnow()
            break
    
    plan.updated_at = datetime.utcnow()
    return plan.to_dict()


def add_progress_update(
    state: PlanningAgentState,
    message: str,
    progress: float = None,
    task_id: str = None
) -> List[Dict[str, Any]]:
    """افزودن یک پیام پیشرفت."""
    updates = state.get("progress_updates", [])
    updates.append({
        "timestamp": datetime.utcnow().isoformat(),
        "message": message,
        "progress": progress,
        "task_id": task_id
    })
    return updates
