"""
LangGraph Agent Implementation
پیاده‌سازی گراف عامل پیشرفته با LangGraph و vLLM

تمام ویژگی‌های UX:
- Context Awareness: حفظ تاریخچه و ترجیحات کاربر
- Streaming: پشتیبانی کامل از streaming events (توکن به توکن)
- Progressive Disclosure: نمایش پیشرفت هر گام
- Error Recovery: مدیریت خطا و retry خودکار
- Proactive Assistance: پیشنهادات هوشمند
- Multi-step Planning: برنامه‌ریزی چند مرحله‌ای
- Parallel Tool Execution: اجرای موازی ابزارها
"""

from .agent import (
    run_agent_graph,
    run_agent_graph_stream,
    agent_graph,
    build_agent_graph,
)
from .state import (
    AgentState,
    create_initial_state,
    ToolStatus,
    StepStatus,
)

# New Planning Agent (Plan-Execute-Revise architecture)
from .planning_agent import (
    run_planning_agent,
    run_planning_agent_stream,
    planning_agent_graph,
    PlanningAgentState,
    create_planning_state,
)
from .nodes import (
    initialize_node,
    call_llm_node,
    execute_tools_node,
    force_final_node,
    generate_suggestions_node,
)

# New real-time streaming
from .streaming import (
    stream_agent,
    stream_agent_for_full_endpoint,
    run_agent,
)

__all__ = [
    # Main API (Legacy ReAct)
    "run_agent_graph",
    "run_agent_graph_stream",
    "agent_graph",
    "build_agent_graph",
    # Planning Agent (Plan-Execute-Revise)
    "run_planning_agent",
    "run_planning_agent_stream",
    "planning_agent_graph",
    "PlanningAgentState",
    "create_planning_state",
    # Real-time Streaming
    "stream_agent",
    "stream_agent_for_full_endpoint",
    "run_agent",
    # State
    "AgentState",
    "create_initial_state",
    "ToolStatus",
    "StepStatus",
    # Nodes (for testing)
    "initialize_node",
    "call_llm_node",
    "execute_tools_node",
    "force_final_node",
    "generate_suggestions_node",
]
