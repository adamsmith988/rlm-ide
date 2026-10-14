"""
Planning-Based Agent Graph
گراف عامل با معماری Plan-Execute-Revise
"""
import asyncio
import logging
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, AsyncGenerator
from dataclasses import dataclass

from langgraph.graph import StateGraph, END

from .planning_state import (
    PlanningAgentState, create_planning_state,
    PlanStatus, AgentRole, ExecutionPlan
)
from .planning_nodes import (
    assess_complexity_node,
    planner_node,
    executor_node,
    critic_node,
    replanner_node,
    synthesizer_node,
    simple_response_node,
)
from ..tracing import get_trace_client, TraceStatus
from ..config import settings
from ..llm.gateway_llm_client import get_gateway_llm_client
from ..tools.executor import get_tool_executor
from ..prompts.system import build_system_messages

log = logging.getLogger(__name__)

# During `run_planning_agent_stream`, nodes call this writer so tokens escape
# the graph before the current node returns (same pattern as `stream_agent`).
_planning_stream_writer_ctx: ContextVar[Optional[Callable[[Dict[str, Any]], None]]] = ContextVar(
    "_planning_stream_writer_ctx",
    default=None,
)


def _planning_graph_config(max_steps: int) -> Dict[str, Any]:
    """
    LangGraph's default recursion_limit (25) is too low for assess → planner →
    many (executor + critic) cycles → synthesizer. Multi-file plans hit the
    limit and the graph stops before synthesis.
    """
    ms = max(int(max_steps or 0), settings.MAX_STEPS or 10)
    return {"recursion_limit": max(120, ms * 10 + 50)}


# ==================== Dependencies ====================

def get_dependencies() -> Dict[str, Any]:
    """Return the planning graph dependencies."""
    deps: Dict[str, Any] = {
        "llm_client": get_gateway_llm_client(),
        "tool_executor": get_tool_executor(),
        "build_system_messages": build_system_messages,
    }
    sw = _planning_stream_writer_ctx.get()
    if sw is not None:
        deps["stream_writer"] = sw
    return deps


# ==================== Routing Functions ====================

def route_after_assessment(state: PlanningAgentState) -> Literal["planner", "simple_response"]:
    """
    روتینگ بعد از ارزیابی پیچیدگی.
    """
    needs_planning = state.get("needs_planning", False)
    complexity = state.get("complexity", "simple")
    
    if needs_planning or complexity in ("multi_step", "single_tool"):
        log.info(f"📋 Routing to: PLANNER ({complexity})")
        return "planner"
    else:
        log.info("💬 Routing to: SIMPLE_RESPONSE")
        return "simple_response"


def route_after_planner(state: PlanningAgentState) -> Literal["executor", "synthesizer"]:
    """
    روتینگ بعد از Planning.
    On failure, still route to synthesizer so the user gets a visible final message
    (partial context or error explanation) instead of an empty stream.
    """
    if state.get("error"):
        log.warning("❌ Plan failed — routing to SYNTHESIZER for user-visible wrap-up")
        return "synthesizer"
    
    plan_dict = state.get("execution_plan")
    if not plan_dict:
        log.warning("⚠️ No plan created — routing to SYNTHESIZER")
        return "synthesizer"
    
    log.info("⚙️ Routing to: EXECUTOR")
    return "executor"


def route_after_executor(state: PlanningAgentState) -> Literal["critic", "synthesizer", "end"]:
    """
    روتینگ بعد از اجرای task.
    """
    if state.get("error"):
        log.warning("❌ Execution error — routing to SYNTHESIZER for wrap-up")
        return "synthesizer"
    
    plan_status = state.get("plan_status", "")
    
    if plan_status == PlanStatus.SYNTHESIZING.value:
        log.info("📝 Routing to: SYNTHESIZER (all tasks done)")
        return "synthesizer"
    
    log.info("🔍 Routing to: CRITIC")
    return "critic"


def route_after_critic(state: PlanningAgentState) -> Literal["executor", "replanner", "synthesizer", "end"]:
    """
    روتینگ بعد از ارزیابی Critic.
    
    تصمیمات:
    - ادامه اجرا (executor)
    - بازبینی برنامه (replanner)
    - ترکیب نتایج (synthesizer)
    - پایان (end)
    """
    if state.get("error"):
        log.warning("❌ Critic saw error state — routing to SYNTHESIZER")
        return "synthesizer"
    
    plan_status = state.get("plan_status", "")
    current_role = state.get("current_agent_role", "")
    
    # Check plan status
    if plan_status == PlanStatus.SYNTHESIZING.value:
        log.info("📝 Routing to: SYNTHESIZER")
        return "synthesizer"
    
    if plan_status == PlanStatus.REVISING.value:
        log.info("🔄 Routing to: REPLANNER")
        return "replanner"
    
    # Check agent role
    if current_role == AgentRole.SYNTHESIZER.value:
        log.info("📝 Routing to: SYNTHESIZER")
        return "synthesizer"
    
    if current_role == AgentRole.PLANNER.value:
        # Critic requested re-planning
        log.info("🔄 Routing to: REPLANNER")
        return "replanner"
    
    # Check if plan is complete
    plan_dict = state.get("execution_plan")
    if plan_dict:
        plan = ExecutionPlan.from_dict(plan_dict)
        if plan.is_complete():
            log.info("📝 Routing to: SYNTHESIZER (plan complete)")
            return "synthesizer"
    
    # Check max steps
    if state.get("current_step", 0) >= state.get("max_steps", 10):
        log.warning("⚠️ Max steps reached, routing to SYNTHESIZER")
        return "synthesizer"
    
    # Default: continue execution
    log.info("⚙️ Routing to: EXECUTOR (continue)")
    return "executor"


def route_after_replanner(state: PlanningAgentState) -> Literal["executor", "synthesizer", "end"]:
    """
    روتینگ بعد از Re-planning.
    """
    if state.get("error"):
        log.warning("❌ Replanner error state — routing to SYNTHESIZER")
        return "synthesizer"
    
    plan_dict = state.get("execution_plan")
    if not plan_dict:
        log.warning("⚠️ Re-planning failed, going to synthesizer")
        return "synthesizer"
    
    log.info("⚙️ Routing to: EXECUTOR (new plan)")
    return "executor"


def route_after_synthesizer(state: PlanningAgentState) -> Literal["end"]:
    """
    بعد از Synthesizer همیشه end.
    """
    return "end"


def route_after_simple_response(state: PlanningAgentState) -> Literal["planner", "end"]:
    """
    Route after simple response — escalate to planner if tool calls were
    detected in the LLM output (safety net for misrouted requests).
    """
    if state.get("needs_planning") and not state.get("final_answer"):
        log.info("🔄 Simple response escalating to PLANNER (tool call detected)")
        return "planner"
    return "end"


# ==================== Node Wrappers ====================

async def wrapped_assess_complexity(state: PlanningAgentState) -> Dict[str, Any]:
    deps = get_dependencies()
    return await assess_complexity_node(state, deps)


async def wrapped_planner(state: PlanningAgentState) -> Dict[str, Any]:
    deps = get_dependencies()
    return await planner_node(state, deps)


async def wrapped_executor(state: PlanningAgentState) -> Dict[str, Any]:
    deps = get_dependencies()
    return await executor_node(state, deps)


async def wrapped_critic(state: PlanningAgentState) -> Dict[str, Any]:
    deps = get_dependencies()
    return await critic_node(state, deps)


async def wrapped_replanner(state: PlanningAgentState) -> Dict[str, Any]:
    deps = get_dependencies()
    return await replanner_node(state, deps)


async def wrapped_synthesizer(state: PlanningAgentState) -> Dict[str, Any]:
    deps = get_dependencies()
    return await synthesizer_node(state, deps)


async def wrapped_simple_response(state: PlanningAgentState) -> Dict[str, Any]:
    deps = get_dependencies()
    return await simple_response_node(state, deps)


# ==================== Graph Builder ====================

def build_planning_agent_graph() -> StateGraph:
    """
    ساخت گراف عامل با معماری Plan-Execute-Revise.
    
    Flow:
    
    START → assess_complexity ──┬── (simple) ──→ simple_response ──┬── END
                                │                                  └── (escalate) → planner
                                │
                                └── (complex) ─→ planner → executor ←──────┐
                                                              ↓            │
                                                           critic ─────────┤
                                                              ↓            │
                                                    ┌─────────┴────────┐   │
                                                    ↓                  ↓   │
                                              replanner → executor ────┘   │
                                                              ↓            │
                                                         synthesizer → END │
                                                              ↑            │
                                                              └────────────┘
    """
    
    workflow = StateGraph(PlanningAgentState)
    
    # ==================== Add Nodes ====================
    workflow.add_node("assess_complexity", wrapped_assess_complexity)
    workflow.add_node("planner", wrapped_planner)
    workflow.add_node("executor", wrapped_executor)
    workflow.add_node("critic", wrapped_critic)
    workflow.add_node("replanner", wrapped_replanner)
    workflow.add_node("synthesizer", wrapped_synthesizer)
    workflow.add_node("simple_response", wrapped_simple_response)
    
    # ==================== Entry Point ====================
    workflow.set_entry_point("assess_complexity")
    
    # ==================== Edges ====================
    
    # After complexity assessment
    workflow.add_conditional_edges(
        "assess_complexity",
        route_after_assessment,
        {
            "planner": "planner",
            "simple_response": "simple_response"
        }
    )
    
    # After planning
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "executor": "executor",
            "synthesizer": "synthesizer",
        }
    )
    
    # After execution
    workflow.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "critic": "critic",
            "synthesizer": "synthesizer",
            "end": END
        }
    )
    
    # After critic evaluation
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "executor": "executor",
            "replanner": "replanner",
            "synthesizer": "synthesizer",
            "end": END
        }
    )
    
    # After re-planning
    workflow.add_conditional_edges(
        "replanner",
        route_after_replanner,
        {
            "executor": "executor",
            "synthesizer": "synthesizer",
            "end": END
        }
    )
    
    # After synthesizer
    workflow.add_edge("synthesizer", END)
    
    # After simple response (may escalate to planner if tool call leaked)
    workflow.add_conditional_edges(
        "simple_response",
        route_after_simple_response,
        {
            "planner": "planner",
            "end": END
        }
    )
    
    return workflow.compile()


# ==================== Compiled Graph ====================

planning_agent_graph = build_planning_agent_graph()


# ==================== Public API ====================

async def run_planning_agent(
    user_message: str,
    session_id: str,
    artifacts: List[Dict[str, Any]] = None,
    memory_context: List[Dict[str, Any]] = None,
    user_id: str = "default",
    max_steps: int = None,
    max_replans: int = None,
    chat_history: List[Dict[str, str]] = None,
    conversation_state=None,
    history_summary: str = "",
    file_registry: List[Dict[str, Any]] = None,
    pinned_facts: Optional[Dict[str, Any]] = None,
    rule_policies: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    اجرای گراف عامل Planning-Based.
    """
    
    initial_state = create_planning_state(
        user_message=user_message,
        session_id=session_id,
        user_id=user_id,
        artifacts=artifacts,
        memory_context=memory_context,
        max_steps=max_steps or settings.MAX_STEPS or 10,
        max_replans=max_replans or 3,
        is_streaming=False,
        chat_history=chat_history,
        conversation_state=conversation_state,
        history_summary=history_summary,
        file_registry=file_registry,
        pinned_facts=pinned_facts,
        rule_policies=rule_policies,
    )
    
    try:
        log.info(f"🎯 Running planning agent for session: {session_id}")
        
        # Create trace in database
        tracer = get_trace_client()
        trace_id = initial_state["trace_id"]
        await tracer.create_trace(
            trace_id=trace_id,
            session_id=session_id,
            user_message=user_message,
            user_id=user_id,
            metadata={
                "artifacts_count": len(artifacts or []),
                "has_memory": len(memory_context or []) > 0,
                "has_history": len(chat_history or []) > 0,
                "max_steps": max_steps or settings.MAX_STEPS or 10
            }
        )
        
        result = await planning_agent_graph.ainvoke(
            initial_state,
            config=_planning_graph_config(initial_state.get("max_steps")),
        )
        
        # Mark trace as completed
        try:
            tracer = get_trace_client()
            tool_runs_list = result.get("tool_runs", [])
            plan_dict = result.get("execution_plan")
            await tracer.update_trace(
                trace_id=trace_id,
                status=TraceStatus.COMPLETED.value,
                complexity=plan_dict.get("complexity") if plan_dict else "simple",
                execution_plan=plan_dict,
                final_answer=result.get("final_answer"),
                overall_confidence=result.get("overall_confidence"),
                total_steps=result.get("current_step", 0),
                total_tool_calls=len(tool_runs_list),
                replan_count=result.get("replan_count", 0),
            )
            # Flush all pending step/tool updates before returning
            await tracer.flush()
        except Exception as _te:
            log.warning(f"Failed to finalize trace: {_te}")
        
        answer = result.get("final_answer") or ""
        if not answer.strip():
            answer = "No answer was received."
        
        # Get plan info
        plan_dict = result.get("execution_plan")
        plan_info = None
        if plan_dict:
            plan = ExecutionPlan.from_dict(plan_dict)
            plan_info = {
                "id": plan.id,
                "complexity": plan.complexity,
                "total_tasks": len(plan.tasks),
                "completed_tasks": len(plan.get_completed_tasks()),
                "failed_tasks": len(plan.get_failed_tasks()),
                "version": plan.version,
                "reasoning": plan.reasoning,
            }
        
        return {
            "answer": answer,
            "thinking": result.get("thinking"),
            "synthesis_reasoning": result.get("synthesis_reasoning"),
            "tool_runs": result.get("tool_runs", []),
            "trace_id": result.get("trace_id"),
            "plan": plan_info,
            "steps": result.get("current_step", 0),
            "replan_count": result.get("replan_count", 0),
            "overall_confidence": result.get("overall_confidence", 0),
            "file_context": result.get("file_context"),
            "suggestions": result.get("suggestions", []),
            "warnings": result.get("warnings", []),
            "started_at": result.get("started_at"),
            "completed_at": result.get("completed_at"),
        }
        
    except Exception as exc:
        log.error(f"🔴 Planning agent failed: {exc}", exc_info=True)
        try:
            tracer = get_trace_client()
            await tracer.update_trace(
                trace_id=initial_state.get("trace_id", ""),
                status=TraceStatus.FAILED.value,
                error=str(exc),
            )
            await tracer.flush()
        except Exception:
            pass
        return {
            "answer": f"Processing error: {str(exc)}",
            "tool_runs": [],
            "trace_id": str(uuid.uuid4()),
            "error": str(exc)
        }


async def run_planning_agent_stream(
    user_message: str,
    session_id: str,
    artifacts: List[Dict[str, Any]] = None,
    memory_context: List[Dict[str, Any]] = None,
    user_id: str = "default",
    max_steps: int = None,
    max_replans: int = None,
    trace_id: str = None,
    chat_history: List[Dict[str, str]] = None,
    conversation_state=None,
    history_summary: str = "",
    file_registry: List[Dict[str, Any]] = None,
    pinned_facts: Optional[Dict[str, Any]] = None,
    rule_policies: Optional[Dict[str, Any]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    اجرای گراف عامل با streaming.
    """
    
    initial_state = create_planning_state(
        user_message=user_message,
        session_id=session_id,
        user_id=user_id,
        artifacts=artifacts,
        memory_context=memory_context,
        max_steps=max_steps or settings.MAX_STEPS or 10,
        max_replans=max_replans or 3,
        is_streaming=True,
        trace_id=trace_id,
        chat_history=chat_history,
        conversation_state=conversation_state,
        history_summary=history_summary,
        file_registry=file_registry,
        pinned_facts=pinned_facts,
        rule_policies=rule_policies,
    )
    
    trace_id = initial_state["trace_id"]
    
    try:
        log.info(f"🎯 Running planning agent (streaming) for session: {session_id}")
        
        # Create trace in database
        tracer = get_trace_client()
        await tracer.create_trace(
            trace_id=trace_id,
            session_id=session_id,
            user_message=user_message,
            user_id=user_id,
            metadata={
                "artifacts_count": len(artifacts or []),
                "has_memory": len(memory_context or []) > 0,
                "max_steps": max_steps or settings.MAX_STEPS or 10,
                "streaming": True
            }
        )
        
        answer_emitted_nonempty = False
        last_final_answer = None
        terminal_answer_for_done = None

        events_queue: asyncio.Queue = asyncio.Queue()
        stream_finished = object()

        def queue_writer(ev: Dict[str, Any]) -> None:
            """Drain synthesizer/executor LLM chunks while the LangGraph node runs."""
            events_queue.put_nowait(ev)

        ctx_reset = _planning_stream_writer_ctx.set(queue_writer)

        async def consume_graph_stream() -> None:
            nonlocal answer_emitted_nonempty, last_final_answer, terminal_answer_for_done
            try:
                async for step_output in planning_agent_graph.astream(
                    initial_state,
                    config=_planning_graph_config(initial_state.get("max_steps")),
                ):
                    try:
                        node_items = list(step_output.items())
                    except AttributeError:
                        log.debug(
                            "Skipping non-dict stream chunk: %s",
                            type(step_output).__name__,
                        )
                        continue
                    for node_name, node_output in node_items:
                        if node_output is None:
                            continue
                        if not isinstance(node_output, dict):
                            continue

                        events_queue.put_nowait(
                            {
                                "type": "node",
                                "node": node_name,
                                "trace_id": trace_id,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )

                        stream_events = node_output.get("stream_events", [])
                        for event in stream_events:
                            event["trace_id"] = trace_id
                            events_queue.put_nowait(event)

                        if node_output.get("execution_plan") and node_name == "planner":
                            events_queue.put_nowait(
                                {
                                    "type": "plan",
                                    "plan": node_output["execution_plan"],
                                    "trace_id": trace_id,
                                    "timestamp": datetime.utcnow().isoformat(),
                                }
                            )

                        if node_output.get("evaluations") and node_name == "critic":
                            evaluations = node_output["evaluations"]
                            if evaluations:
                                events_queue.put_nowait(
                                    {
                                        "type": "evaluation",
                                        "evaluation": evaluations[-1],
                                        "overall_confidence": node_output.get(
                                            "overall_confidence", 0
                                        ),
                                        "trace_id": trace_id,
                                        "timestamp": datetime.utcnow().isoformat(),
                                    }
                                )

                        if "final_answer" in node_output and node_output["final_answer"] is not None:
                            fa = node_output["final_answer"]
                            last_final_answer = fa
                            text = str(fa).strip() if fa is not None else ""
                            streamed = bool(
                                node_output.get("synthesis_streamed_tokens")
                            )
                            if text:
                                answer_emitted_nonempty = True
                                terminal_answer_for_done = text
                                if not streamed:
                                    events_queue.put_nowait(
                                        {
                                            "type": "answer",
                                            "content": fa,
                                            "synthesis_reasoning": node_output.get(
                                                "synthesis_reasoning"
                                            ),
                                            "trace_id": trace_id,
                                            "timestamp": datetime.utcnow().isoformat(),
                                        }
                                    )
                                else:
                                    events_queue.put_nowait(
                                        {
                                            "type": "answer_complete",
                                            "synthesis_streamed_tokens": True,
                                            "synthesis_reasoning": node_output.get(
                                                "synthesis_reasoning"
                                            ),
                                            "trace_id": trace_id,
                                            "timestamp": datetime.utcnow().isoformat(),
                                        }
                                    )

                        if node_output.get("error"):
                            events_queue.put_nowait(
                                {
                                    "type": "error",
                                    "error": node_output["error"],
                                    "trace_id": trace_id,
                                    "timestamp": datetime.utcnow().isoformat(),
                                }
                            )
            except Exception as exc:
                log.error("🔴 Planning graph stream failed: %s", exc, exc_info=True)
                events_queue.put_nowait(
                    {
                        "type": "error",
                        "error": str(exc),
                        "trace_id": trace_id,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            finally:
                events_queue.put_nowait(stream_finished)

        graph_task = asyncio.create_task(consume_graph_stream())
        try:
            while True:
                item = await events_queue.get()
                if item is stream_finished:
                    break
                yield item
        finally:
            await graph_task
            _planning_stream_writer_ctx.reset(ctx_reset)

        if not answer_emitted_nonempty:
            fallback = (
                str(last_final_answer).strip()
                if last_final_answer is not None and str(last_final_answer).strip()
                else "No final answer was produced. Please ask again or shorten the request."
            )
            terminal_answer_for_done = fallback
            log.warning("⚠️ Stream ended without a non-empty final_answer — emitting fallback")
            yield {
                "type": "answer",
                "content": fallback,
                "trace_id": trace_id,
                "timestamp": datetime.utcnow().isoformat(),
            }

        # Final done event (include final text for clients that only read `done`)
        yield {
            "type": "done",
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat(),
            "final_answer": terminal_answer_for_done,
        }
        
    except Exception as exc:
        log.error(f"🔴 Planning agent stream failed: {exc}", exc_info=True)
        yield {
            "type": "error",
            "error": str(exc),
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat()
        }


# ==================== Exports ====================

__all__ = [
    "run_planning_agent",
    "run_planning_agent_stream",
    "planning_agent_graph",
    "PlanningAgentState",
    "create_planning_state"
]
