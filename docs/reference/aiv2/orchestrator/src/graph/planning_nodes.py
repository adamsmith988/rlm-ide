"""
Planning Agent Nodes
گره‌های معماری Plan-Execute-Revise
"""
import logging
import asyncio
import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .planning_state import (
    PlanningAgentState, ExecutionPlan, SubTask, CriticEvaluation,
    TaskStatus, PlanStatus, AgentRole,
    update_task_status, add_progress_update
)
from ..prompts.planning_prompts import (
    PLANNER_SYSTEM_PROMPT, PLANNER_USER_TEMPLATE,
    EXECUTOR_SYSTEM_PROMPT, EXECUTOR_USER_TEMPLATE,
    CRITIC_SYSTEM_PROMPT, CRITIC_USER_TEMPLATE,
    REPLANNER_SYSTEM_PROMPT, REPLANNER_USER_TEMPLATE,
    SYNTHESIZER_SYSTEM_PROMPT, SYNTHESIZER_USER_TEMPLATE,
    PLANNING_DECISION_PROMPT, SIMPLE_RESPONSE_PROMPT,
    get_progress_message
)
from ..config import settings
from ..tracing import (
    TraceClient, get_trace_client, TraceStep, ToolExecution,
    TraceStatus, StepStatus, ToolExecutionStatus
)
from ..utils.multimodal import build_multimodal_content
from ..utils.text_inline import build_text_files_inline_block
from .nodes import render_tool_result_for_llm

log = logging.getLogger(__name__)


def _synthesis_max_tokens(plan: ExecutionPlan, all_results: str) -> int:
    """Scale completion budget with task count and aggregated result size."""
    base = settings.LLM_SYNTHESIS_MAX_TOKENS
    n = len(plan.tasks)
    ar_len = len(all_results or "")
    scaled = base + min(n * 256, 4096) + min(ar_len // 24, 6144)
    return min(int(scaled), 32768)


# ==================== Utility Functions ====================

def extract_json_from_response(content: str) -> Optional[Dict]:
    """استخراج JSON از پاسخ LLM."""
    if not content:
        return None
    
    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in markdown code blocks
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'\{[\s\S]*\}'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                return json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                continue
    
    return None


def detect_language(text: str) -> str:
    """تشخیص زبان متن (fa یا en)."""
    # Count Persian characters
    persian_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    total_chars = len(re.findall(r'\w', text))
    
    if total_chars == 0:
        return "en"
    
    return "fa" if persian_chars / total_chars > 0.3 else "en"


_TOOL_CALL_LEAK_PATTERN = re.compile(
    r'"tool_name"\s*:\s*"|"tool_args"\s*:\s*\{|"function"\s*:\s*"|"name"\s*:\s*"[a-z_]+\.\w+"',
    re.IGNORECASE,
)


_KNOWN_TOOL_NAME_RE = re.compile(
    # Either snake_case with underscore (e.g. graph_search, session_files)
    # or dotted module path (e.g. plugin_graph_explorer_graph_search).
    # We deliberately require either an underscore or a dot to avoid matching
    # natural-language verbs like ``add(2,2)``.
    r"\b(?:[a-zA-Z][\w]*\.[a-zA-Z][\w.]*|[a-zA-Z][a-z0-9]*(?:_[a-z0-9]+)+)\s*\(\s*[\s\S]*?\)",
)


def _response_contains_tool_call(text: str) -> bool:
    """Detect if the LLM leaked a tool call as plain text instead of using function calling."""
    if not text:
        return False
    low = text.lower()
    # Qwen3 + vLLM qwen3_coder native wire uses these sentinels when tools appear in content.
    if "<tool_call>" in text or "<function=" in text:
        return True
    if "```tool_call" in low:
        return True
    if re.search(r"^\s*tool_calls:\s*$", text, re.MULTILINE | re.I):
        return True
    if re.search(r"^\s*-\s*tool_name:\s*\S", text, re.MULTILINE | re.I):
        return True
    if _TOOL_CALL_LEAK_PATTERN.search(text):
        return True
    # Bare ``snake_case(...)`` or ``plugin.xxx.yyy(...)`` — only inside a code
    # fence to avoid catching the LLM merely mentioning a tool in prose.
    if "```" in text and _KNOWN_TOOL_NAME_RE.search(text):
        return True
    return False


def format_tools_list(tools: List[Dict]) -> str:
    """فرمت کردن لیست ابزارها برای prompt."""
    lines = []
    for tool in tools[:30]:  # Max 30 tools
        name = tool.get("function", {}).get("name") or tool.get("name", "")
        desc = tool.get("function", {}).get("description") or tool.get("description", "")
        if name:
            lines.append(f"- **{name}**: {desc[:100]}")
    return "\n".join(lines)


def _format_one_task_result_for_planner(result: Any) -> str:
    """
    برای planner/synthesizer: متن قابل‌خواندن برای LLM، نه JSON خام با elementId.
    اگر task ابزار زده، خروجی رندرشدهٔ هر ابزار (channels.llm) را می‌آورد.
    """
    if not isinstance(result, dict):
        return str(result)[:8000]

    tool_results = result.get("tool_results")
    if isinstance(tool_results, list) and tool_results:
        parts: List[str] = []
        for i, tr in enumerate(tool_results, 1):
            if isinstance(tr, dict) and tr.get("ok"):
                parts.append(f"### Tool output {i}\n{render_tool_result_for_llm(tr)}")
            elif isinstance(tr, dict):
                parts.append(f"### Tool output {i}\nError: {tr.get('error', 'unknown')}")
            else:
                parts.append(f"### Tool output {i}\n{str(tr)[:2000]}")
        body = "\n\n".join(parts)
        tout = result.get("task_output")
        if isinstance(tout, dict) and tout:
            body += f"\n\n**Task structured output**:\n{json.dumps(tout, ensure_ascii=False)[:1500]}"
        return body[:12000]

    task_out = result.get("task_output")
    if isinstance(task_out, dict) and task_out:
        return json.dumps(task_out, ensure_ascii=False)[:4000]
    lc = result.get("llm_content")
    if isinstance(lc, str) and lc.strip():
        return lc.strip()[:8000]
    return json.dumps(result, ensure_ascii=False)[:2000]


def format_task_results(results: Dict[str, Any], tasks: List[SubTask]) -> str:
    """فرمت کردن نتایج task‌ها."""
    lines = []
    for task in tasks:
        if task.id in results:
            result = results[task.id]
            lines.append(f"### Task {task.id}: {task.description[:50]}")
            lines.append(f"**Status**: {task.status.value}")
            lines.append("**Result**:")
            lines.append(_format_one_task_result_for_planner(result))
            lines.append("")
    return "\n".join(lines) if lines else "No results yet."


def _planning_messages_with_rules(
    state: PlanningAgentState,
    messages: List[Dict[str, Any]],
    *,
    user_message: str,
) -> List[Dict[str, Any]]:
    """Merge applicable markdown session rules into the first system message."""
    from ..rules.inject import inject_rules_into_openai_messages

    merged, _ = inject_rules_into_openai_messages(
        messages,
        user_message=user_message,
        file_context=state.get("file_context"),
        file_registry=state.get("file_registry"),
        rule_policies=state.get("_rule_policies"),
    )
    return merged


# ==================== Complexity Assessment Node ====================

async def assess_complexity_node(state: PlanningAgentState, deps: Dict[str, Any]) -> Dict[str, Any]:
    """
    ارزیابی پیچیدگی درخواست و تصمیم‌گیری درباره نیاز به planning.
    
    Returns:
        - needs_planning: True if multi-step planning needed
        - complexity: simple, single_tool, multi_step
    """
    log.info("🔍 Assessing request complexity...")
    
    trace_id = state.get("trace_id", "")
    step_number = state.get("current_step", 0) + 1
    started_at = datetime.utcnow()
    
    # Create trace step
    tracer = get_trace_client()
    step = TraceStep(
        trace_id=trace_id,
        step_number=step_number,
        node_name="assess_complexity",
        status=StepStatus.RUNNING.value,
        started_at=started_at.isoformat()
    )
    await tracer.add_step(step)
    
    # Log stream event
    await tracer.add_stream_event(
        trace_id=trace_id,
        event_type="node_start",
        event_data={"node": "assess_complexity", "step": step_number}
    )
    
    messages = state.get("messages", [])
    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break
    
    # Use ConversationState if available for smarter detection
    conv_state = state.get("conversation_state")

    # Quick heuristics — enhanced with conversation state
    simple_patterns = [
        r'^(سلام|hello|hi|hey|درود)',  # Greetings
        r'^(ممنون|thanks|thank you|مرسی)',  # Thanks
        r'^(چطوری|how are you)',  # Casual
    ]
    
    for pattern in simple_patterns:
        if re.match(pattern, user_query.strip(), re.IGNORECASE):
            log.info("📊 Complexity: SIMPLE (greeting/casual)")
            
            duration = int((datetime.utcnow() - started_at).total_seconds() * 1000)
            await tracer.update_step(
                trace_id=trace_id,
                step_number=step_number,
                status=StepStatus.COMPLETED.value,
                output_data={"complexity": "simple", "reason": "greeting pattern"},
                completed_at=datetime.utcnow().isoformat(),
                duration_ms=duration
            )
            
            return {
                "complexity": "simple",
                "needs_planning": False,
                "current_agent_role": AgentRole.EXECUTOR.value,
                "current_step": step_number,
            }

    # If conversation state detects a topic change away from file context,
    # treat as simple/single_tool to avoid dragging old file context in
    if conv_state and conv_state.topic_changed and conv_state.active_topic == "general_chat":
        log.info("📊 Complexity: SIMPLE (topic changed to general chat)")
        duration = int((datetime.utcnow() - started_at).total_seconds() * 1000)
        await tracer.update_step(
            trace_id=trace_id,
            step_number=step_number,
            status=StepStatus.COMPLETED.value,
            output_data={"complexity": "simple", "reason": "topic_change_to_general"},
            completed_at=datetime.utcnow().isoformat(),
            duration_ms=duration
        )
        return {
            "complexity": "simple",
            "needs_planning": False,
            "current_agent_role": AgentRole.EXECUTOR.value,
            "current_step": step_number,
        }
    
    # Check for file context — non-visual files always need tools
    file_context = state.get("file_context")
    if file_context:
        file_type = file_context.get("type", "")
        is_visual = file_type in ("image", "video")
        if not is_visual:
            log.info("📊 Complexity: MULTI_STEP (non-visual file context)")
            duration = int((datetime.utcnow() - started_at).total_seconds() * 1000)
            await tracer.update_step(
                trace_id=trace_id,
                step_number=step_number,
                status=StepStatus.COMPLETED.value,
                output_data={"complexity": "multi_step", "reason": "non-visual file context"},
                completed_at=datetime.utcnow().isoformat(),
                duration_ms=duration
            )
            return {
                "complexity": "multi_step",
                "needs_planning": True,
                "current_agent_role": AgentRole.PLANNER.value,
                "current_step": step_number,
            }
        # Visual files: let LLM decide (some need tools like metadata/OCR,
        # others just need visual description which LLM can do natively)
    
    # Use LLM for complex assessment
    llm_client = deps["llm_client"]
    
    file_hint = ""
    if file_context:
        file_hint = f"\nAttached file: {file_context.get('name', 'unknown')} (type: {file_context.get('type', 'unknown')}, mime: {file_context.get('mime_type', 'unknown')})"
    
    llm_messages = [
        {"role": "system", "content": "You analyze query complexity. Respond with JSON only."},
        {"role": "user", "content": PLANNING_DECISION_PROMPT.format(query=user_query, file_hint=file_hint)}
    ]
    
    try:
        result = await llm_client.chat(llm_messages)
        
        if result.get("ok"):
            data = extract_json_from_response(result.get("content", ""))
            if data:
                complexity = data.get("complexity", "single_tool")
                log.info(f"📊 Complexity: {complexity.upper()}")
                
                duration = int((datetime.utcnow() - started_at).total_seconds() * 1000)
                await tracer.update_step(
                    trace_id=trace_id,
                    step_number=step_number,
                    status=StepStatus.COMPLETED.value,
                    output_data={"complexity": complexity, "llm_assessment": data},
                    llm_response={"content": result.get("content", ""), "ok": True,
                                  "tokens_prompt": result.get("tokens_prompt"),
                                  "tokens_completion": result.get("tokens_completion")},
                    completed_at=datetime.utcnow().isoformat(),
                    duration_ms=duration
                )
                
                return {
                    "complexity": complexity,
                    "needs_planning": complexity == "multi_step",
                    "current_agent_role": AgentRole.PLANNER.value if complexity == "multi_step" else AgentRole.EXECUTOR.value,
                    "current_step": step_number,
                }
    except Exception as e:
        log.warning(f"⚠️ Complexity assessment failed: {e}")
        await tracer.update_step(
            trace_id=trace_id,
            step_number=step_number,
            status=StepStatus.FAILED.value,
            error=str(e),
            completed_at=datetime.utcnow().isoformat()
        )
    
    # Default to single_tool
    return {
        "complexity": "single_tool",
        "needs_planning": False,
        "current_agent_role": AgentRole.EXECUTOR.value,
        "current_step": step_number,
    }


# ==================== Planner Node ====================

async def planner_node(state: PlanningAgentState, deps: Dict[str, Any]) -> Dict[str, Any]:
    """
    گره برنامه‌ریزی - تجزیه درخواست به sub-tasks.
    """
    log.info("📋 Planner node: Creating execution plan...")
    
    trace_id = state.get("trace_id", "")
    step_number = state.get("current_step", 0) + 1
    started_at = datetime.utcnow()
    
    # Create trace step
    tracer = get_trace_client()
    step = TraceStep(
        trace_id=trace_id,
        step_number=step_number,
        node_name="planner",
        status=StepStatus.RUNNING.value,
        started_at=started_at.isoformat()
    )
    await tracer.add_step(step)
    
    await tracer.add_stream_event(
        trace_id=trace_id,
        event_type="node_start",
        event_data={"node": "planner", "step": step_number}
    )
    
    # Get user query
    user_query = ""
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break
    
    language = detect_language(user_query)
    
    # Progress update
    events = list(state.get("stream_events", []))
    if state.get("is_streaming"):
        events.append({
            "type": "progress",
            "message": get_progress_message("planning", language),
            "timestamp": datetime.utcnow().isoformat()
        })
    
    # Build context with file workspace (knowledge-enriched registry)
    context_parts = []
    file_registry = state.get("file_registry") or []

    if file_registry:
        context_parts.append(f"Session workspace ({len(file_registry)} files, chronological):")
        for f in file_registry:
            knowledge = f.get("knowledge") or []
            if knowledge:
                known_summary = "; ".join(
                    f'{k["tool"]}: "{k["summary"][:80]}"' for k in knowledge
                )
                context_parts.append(
                    f"  #{f['ordinal']} {f.get('name','?')} — {f.get('type','?')} "
                    f"(turn {f.get('turn','?')}) [KNOWN: {known_summary}]"
                )
            else:
                ftype = f.get("type", "unknown")
                suggested = {
                    "image": "vision_info, file_identify",
                    "audio": "audio_transcribe, file_identify",
                    "video": "vision_info, file_identify",
                    "document": "text.read, session_read, embed_text, file_identify",
                    "text": "text.read, session_read, embed_text, file_identify",
                }.get(ftype, "file_identify")
                context_parts.append(
                    f"  #{f['ordinal']} {f.get('name','?')} — {ftype} "
                    f"(turn {f.get('turn','?')}) [NOT YET ANALYZED — suggest: {suggested}]"
                )
    elif state.get("file_context"):
        fc = state["file_context"]
        context_parts.append(f"File: {fc.get('name', 'unknown')} (type: {fc.get('type', 'unknown')})")
    if state.get("memory_context"):
        context_parts.append(f"Has {len(state['memory_context'])} memory items")
    
    context = "\n".join(context_parts) if context_parts else "No special context"
    
    # Get tools
    tool_executor = deps["tool_executor"]
    openai_tools = await tool_executor.get_tools_for_file_context(
        file_context=state.get("file_context"),
        channel="llm"
    )
    tools_list = format_tools_list(openai_tools)
    
    # Call LLM for planning
    llm_client = deps["llm_client"]

    # Inject pinned facts into the planner's system prompt so multi-turn
    # plans never lose sight of the original goal / constraints.
    from ..pinned_facts import render_pinned_facts_block
    from ..repetition import render_anti_loop_block

    planner_system = PLANNER_SYSTEM_PROMPT
    pinned_block = render_pinned_facts_block(state.get("pinned_facts"))
    if pinned_block:
        planner_system = pinned_block + "\n\n" + planner_system
    history_summary = state.get("history_summary", "") or ""
    if history_summary:
        planner_system = (
            planner_system
            + "\n\n## Earlier Conversation Summary\n"
            + history_summary
        )
    anti_loop = render_anti_loop_block(state.get("conversation_state"))
    if anti_loop:
        planner_system = planner_system + "\n\n" + anti_loop

    messages = [
        {"role": "system", "content": planner_system},
        {"role": "user", "content": PLANNER_USER_TEMPLATE.format(
            user_query=user_query,
            context=context,
            tools_list=tools_list
        )}
    ]
    messages = _planning_messages_with_rules(state, messages, user_message=user_query)

    try:
        result = await llm_client.chat(messages=messages)
        
        if not result.get("ok"):
            log.error(f"❌ Planner LLM call failed: {result.get('error')}")
            
            duration = int((datetime.utcnow() - started_at).total_seconds() * 1000)
            await tracer.update_step(
                trace_id=trace_id,
                step_number=step_number,
                status=StepStatus.FAILED.value,
                error=result.get("error"),
                llm_messages_sent=messages,
                completed_at=datetime.utcnow().isoformat(),
                duration_ms=duration
            )
            
            return {
                "error": f"Planning failed: {result.get('error')}",
                "plan_status": PlanStatus.FAILED.value,
                "current_step": step_number,
            }
        
        plan_data = extract_json_from_response(result.get("content", ""))
        
        if not plan_data or "tasks" not in plan_data:
            log.warning("⚠️ Invalid plan response, creating fallback plan")
            # Fallback: single task
            plan_data = {
                "reasoning": "Fallback plan - single task execution",
                "complexity": "simple",
                "tasks": [{
                    "id": "task_1",
                    "description": user_query,
                    "tools_needed": [],
                    "depends_on": [],
                    "reasoning": "Direct execution"
                }],
                "estimated_steps": 1
            }
        
        # Create ExecutionPlan
        plan = ExecutionPlan(
            id=f"plan_{state['trace_id'][:8]}",
            original_query=user_query,
            tasks=[SubTask.from_dict(t) for t in plan_data.get("tasks", [])],
            reasoning=plan_data.get("reasoning", ""),
            complexity=plan_data.get("complexity", "moderate"),
            estimated_steps=plan_data.get("estimated_steps", len(plan_data.get("tasks", [])))
        )
        
        log.info(f"✅ Plan created: {len(plan.tasks)} tasks, complexity={plan.complexity}")
        
        # Update trace step with plan
        duration = int((datetime.utcnow() - started_at).total_seconds() * 1000)
        await tracer.update_step(
            trace_id=trace_id,
            step_number=step_number,
            status=StepStatus.COMPLETED.value,
            plan_generated=plan.to_dict(),
            planning_reasoning=plan.reasoning,
            llm_messages_sent=messages,
            llm_response={"content": result.get("content"), "ok": True},
            completed_at=datetime.utcnow().isoformat(),
            duration_ms=duration
        )
        
        # Update trace with plan
        await tracer.update_trace(
            trace_id=trace_id,
            status=TraceStatus.EXECUTING.value,
            execution_plan=plan.to_dict()
        )
        
        # Progress update
        if state.get("is_streaming"):
            events.append({
                "type": "progress",
                "message": get_progress_message("planning_done", language, num_tasks=len(plan.tasks)),
                "plan": plan.to_dict(),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            await tracer.add_stream_event(
                trace_id=trace_id,
                event_type="plan_created",
                event_data={"plan": plan.to_dict(), "tasks": len(plan.tasks)}
            )
        
        return {
            "execution_plan": plan.to_dict(),
            "plan_status": PlanStatus.EXECUTING.value,
            "plan_version": state.get("plan_version", 0) + 1,
            "current_agent_role": AgentRole.EXECUTOR.value,
            "stream_events": events,
            "current_step": step_number,
        }
        
    except Exception as e:
        log.error(f"❌ Planner node error: {e}", exc_info=True)
        
        await tracer.update_step(
            trace_id=trace_id,
            step_number=step_number,
            status=StepStatus.FAILED.value,
            error=str(e),
            completed_at=datetime.utcnow().isoformat()
        )
        
        return {
            "error": f"Planning error: {str(e)}",
            "plan_status": PlanStatus.FAILED.value,
            "current_step": step_number,
        }


# ==================== Executor Node ====================

async def executor_node(state: PlanningAgentState, deps: Dict[str, Any]) -> Dict[str, Any]:
    """
    گره اجرا - اجرای task فعلی.
    """
    plan_dict = state.get("execution_plan")
    if not plan_dict:
        log.error("❌ No execution plan found")
        return {"error": "No execution plan available"}
    
    plan = ExecutionPlan.from_dict(plan_dict)
    
    # Get next task
    current_task = plan.get_next_task()
    if not current_task:
        log.info("✅ All tasks completed or blocked")
        return {
            "plan_status": PlanStatus.SYNTHESIZING.value,
            "current_agent_role": AgentRole.SYNTHESIZER.value,
        }
    
    trace_id = state.get("trace_id", "")
    step_number = state.get("current_step", 0) + 1
    started_at = datetime.utcnow()
    
    # Create trace step
    tracer = get_trace_client()
    step = TraceStep(
        trace_id=trace_id,
        step_number=step_number,
        node_name="executor",
        task_id=current_task.id,
        task_description=current_task.description,
        status=StepStatus.RUNNING.value,
        started_at=started_at.isoformat()
    )
    await tracer.add_step(step)
    
    log.info(f"⚙️ Executing task: {current_task.id} - {current_task.description[:50]}...")
    
    await tracer.add_stream_event(
        trace_id=trace_id,
        event_type="task_start",
        event_data={
            "node": "executor",
            "step": step_number,
            "task_id": current_task.id,
            "task_description": current_task.description
        }
    )
    
    # Update task status
    current_task.status = TaskStatus.IN_PROGRESS
    current_task.started_at = datetime.utcnow()
    
    # Language detection
    language = detect_language(plan.original_query)
    
    # Progress event
    events = list(state.get("stream_events", []))
    if state.get("is_streaming"):
        events.append({
            "type": "progress",
            "message": get_progress_message("executing_task", language, task_desc=current_task.description[:50]),
            "task_id": current_task.id,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    # Build previous results context
    task_results = state.get("task_results", {})
    previous_results = format_task_results(task_results, plan.get_completed_tasks())
    
    # Build file context with accumulated knowledge from session workspace
    context_parts = []
    file_registry = state.get("file_registry") or []
    if file_registry:
        context_parts.append(f"Session workspace ({len(file_registry)} files):")
        for f in file_registry:
            line = f"  #{f['ordinal']} {f.get('name','?')} → {f.get('path','?')}"
            if f.get("artifact_id"):
                line += f" (artifact_id: {f['artifact_id']})"
            knowledge = f.get("knowledge") or []
            if knowledge:
                known = "; ".join(f'{k["tool"]}: "{k["summary"][:60]}"' for k in knowledge)
                line += f" [KNOWN: {known}]"
            else:
                ftype = f.get("type", "unknown")
                suggested = {
                    "image": "vision_info, file_identify",
                    "audio": "audio_transcribe, file_identify",
                    "video": "vision_info, file_identify",
                    "document": "text.read, session_read, embed_text, file_identify",
                    "text": "text.read, session_read, embed_text, file_identify",
                }.get(ftype, "file_identify")
                line += f" [NOT YET ANALYZED — suggest: {suggested}]"
            context_parts.append(line)

    if state.get("file_context"):
        fc = state["file_context"]
        context_parts.append(f"\nPrimary file for this task: {fc.get('name', 'unknown')} → {fc.get('path', 'unknown')}")
        if fc.get("artifact_id"):
            context_parts.append(f"Primary artifact_id: {fc['artifact_id']}")
    context = "\n".join(context_parts) if context_parts else "No file context"
    
    # Get tools for this task (only if tools_needed is non-empty)
    tool_executor = deps["tool_executor"]
    openai_tools = None
    
    if current_task.tools_needed:
        all_tools = await tool_executor.get_tools_for_file_context(
            file_context=state.get("file_context"),
            channel="llm"
        )
        needed_names = set(current_task.tools_needed)
        filtered_tools = [t for t in all_tools 
                         if t.get("function", {}).get("name") in needed_names]
        openai_tools = filtered_tools if filtered_tools else all_tools
        log.info(f"🔧 Task needs tools: {current_task.tools_needed}, providing {len(openai_tools)} tools")
    else:
        log.info(f"💡 Task needs no tools — LLM will respond directly")
    
    llm_client = deps["llm_client"]
    
    executor_text = EXECUTOR_USER_TEMPLATE.format(
        task_id=current_task.id,
        task_description=current_task.description,
        tools_needed=", ".join(current_task.tools_needed) if current_task.tools_needed else "None — respond directly using your knowledge",
        previous_results=previous_results,
        context=context
    )
    # Always attach visuals when file_context is image/video (up to vLLM limits
    # inside build_multimodal_content). Executor needs both vision and tools.
    user_content = await build_multimodal_content(
        text=executor_text,
        file_context=state.get("file_context"),
        auth_context={
            "user_id": state.get("user_id"),
            "group_id": state.get("group_id"),
            "case_id": state.get("case_id"),
        },
    )

    # Pre-inline small text/code/markdown files so executor can answer textual
    # questions without first calling text.read / session_read.
    inline_block, _ = await build_text_files_inline_block(state.get("file_context"))

    # Qwen3.5 chat template forbids multiple system messages (`loop.first`
    # check in chat_template.jinja). Merge the text-inline block into the
    # single system message instead of appending a second one.
    system_content = EXECUTOR_SYSTEM_PROMPT
    if inline_block:
        system_content = f"{system_content}\n\n{inline_block}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    messages = _planning_messages_with_rules(
        state, messages, user_message=plan.original_query
    )

    # Long multi-task plans: executor needs headroom for tool JSON + commentary
    if settings.LLM_MAX_TOKENS is None:
        exec_max = None
    else:
        exec_max = min(
            settings.LLM_MAX_TOKENS + min(len(plan.tasks) * 128, 3072),
            8192,
        )
    
    try:
        result = await llm_client.chat(
            messages=messages,
            tools=openai_tools,
            tool_choice="auto" if openai_tools else None,
            max_tokens=exec_max,
        )
        
        if not result.get("ok"):
            log.error(f"❌ Executor LLM call failed: {result.get('error')}")
            current_task.status = TaskStatus.FAILED
            current_task.error = result.get("error", "LLM call failed")
            current_task.retries += 1
            
            duration = int((datetime.utcnow() - started_at).total_seconds() * 1000)
            await tracer.update_step(
                trace_id=trace_id,
                step_number=step_number,
                status=StepStatus.FAILED.value,
                error=result.get("error"),
                llm_messages_sent=messages,
                completed_at=datetime.utcnow().isoformat(),
                duration_ms=duration
            )
            
            return {
                "execution_plan": plan.to_dict(),
                "current_task_id": current_task.id,
                "current_agent_role": AgentRole.CRITIC.value,
                "stream_events": events,
                "current_step": step_number,
            }
        
        content = result.get("content", "")
        tool_calls = result.get("tool_calls", [])
        _reasoning = (result.get("reasoning_content") or "").strip()
        if _reasoning and state.get("is_streaming"):
            events.append(
                {
                    "type": "thinking",
                    "step": step_number,
                    "content": _reasoning,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
        
        # Execute tool calls if any
        new_tool_runs = list(state.get("tool_runs", []))
        tool_results_for_task = []
        
        if tool_calls:
            log.info(f"🔧 Executing {len(tool_calls)} tool calls...")
            
            # Record tool calls requested
            await tracer.update_step(
                trace_id=trace_id,
                step_number=step_number,
                tool_calls_requested=[{"name": tc.get("name"), "arguments": tc.get("arguments")} for tc in tool_calls]
            )
            
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})
                tool_id = str(uuid.uuid4())[:12]
                tool_started = datetime.utcnow()
                
                # Parse arguments if string
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}
                
                # Add file_path if needed
                file_path = None
                if state.get("file_context") and "file_path" not in tool_args:
                    file_path = state["file_context"].get("path")
                    tool_args["file_path"] = file_path
                
                # Create tool execution record
                tool_exec = ToolExecution(
                    trace_id=trace_id,
                    tool_name=tool_name,
                    arguments=tool_args,
                    tool_id=tool_id,
                    task_id=current_task.id,
                    status=ToolExecutionStatus.RUNNING.value,
                    file_path=file_path,
                    started_at=tool_started.isoformat()
                )
                await tracer.add_tool_execution(tool_exec)
                
                # Execute tool
                tool_result = await tool_executor.execute(
                    tool_name,
                    tool_args,
                    file_context=state.get("file_context"),
                    session_id=state.get("session_id"),
                    user_id=state.get("user_id"),
                    group_id=state.get("group_id"),
                )
                
                tool_completed = datetime.utcnow()
                tool_duration = int((tool_completed - tool_started).total_seconds() * 1000)
                
                # متنی که واقعاً به مدل می‌رسد (channels.llm) — نه کل payload خام
                preview = (
                    render_tool_result_for_llm(tool_result)
                    if tool_result.get("ok")
                    else None
                )
                result_sent_to_llm = {
                    "tool": tool_name,
                    "ok": tool_result.get("ok", False),
                    # متن انسانی برای LLM؛ result همان برای سازگاری با trace/UI قدیمی
                    "text_for_llm": preview,
                    "result": preview,
                    "error": tool_result.get("error") if not tool_result.get("ok") else None,
                }
                
                # Update tool execution with result
                await tracer.update_tool_execution(
                    trace_id=trace_id,
                    tool_id=tool_id,
                    status=ToolExecutionStatus.COMPLETED.value if tool_result.get("ok") else ToolExecutionStatus.FAILED.value,
                    result=tool_result,
                    result_summary=(
                        render_tool_result_for_llm(tool_result)[:500]
                        if tool_result.get("ok")
                        else None
                    ),
                    result_sent_to_llm=result_sent_to_llm,
                    error=tool_result.get("error") if not tool_result.get("ok") else None,
                    completed_at=tool_completed.isoformat(),
                    duration_ms=tool_duration
                )
                
                tool_run_info = {
                    "tool_name": tool_name,
                    "tool_id": tool_id,
                    "arguments": tool_args,
                    "result": tool_result,
                    "result_sent_to_llm": result_sent_to_llm,
                    "task_id": current_task.id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                new_tool_runs.append(tool_run_info)
                tool_results_for_task.append(tool_result)
                
                log.info(f"  ✓ {tool_name}: {'ok' if tool_result.get('ok') else 'failed'}")
                
                # Stream event for tool execution
                await tracer.add_stream_event(
                    trace_id=trace_id,
                    event_type="tool_executed",
                    event_data={
                        "tool_name": tool_name,
                        "tool_id": tool_id,
                        "ok": tool_result.get("ok", False),
                        "duration_ms": tool_duration
                    }
                )
        
        # Determine task status
        task_output = extract_json_from_response(content)
        
        if tool_results_for_task:
            # Check if all tools succeeded
            all_ok = all(r.get("ok") for r in tool_results_for_task)
            current_task.status = TaskStatus.COMPLETED if all_ok else TaskStatus.FAILED
            current_task.result = {
                "tool_results": tool_results_for_task,
                "llm_content": content,
                "task_output": task_output
            }
            if not all_ok:
                current_task.error = "One or more tool calls failed"
        else:
            # No tools - use LLM output
            current_task.status = TaskStatus.COMPLETED
            current_task.result = {
                "llm_content": content,
                "task_output": task_output
            }
        
        current_task.completed_at = datetime.utcnow()
        
        # Update task results
        new_task_results = dict(task_results)
        new_task_results[current_task.id] = current_task.result
        
        # Update trace step
        duration = int((datetime.utcnow() - started_at).total_seconds() * 1000)
        await tracer.update_step(
            trace_id=trace_id,
            step_number=step_number,
            status=StepStatus.COMPLETED.value if current_task.status == TaskStatus.COMPLETED else StepStatus.FAILED.value,
            output_data=current_task.result,
            llm_messages_sent=messages,
            llm_response={"content": content, "tool_calls": len(tool_calls)},
            error=current_task.error if current_task.status == TaskStatus.FAILED else None,
            completed_at=datetime.utcnow().isoformat(),
            duration_ms=duration
        )
        
        # Progress event
        task_num = next((i+1 for i, t in enumerate(plan.tasks) if t.id == current_task.id), "?")
        if state.get("is_streaming"):
            msg_key = "task_completed" if current_task.status == TaskStatus.COMPLETED else "task_failed"
            events.append({
                "type": "progress",
                "message": get_progress_message(msg_key, language, task_num=task_num),
                "task_id": current_task.id,
                "status": current_task.status.value,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        log.info(f"✅ Task {current_task.id} completed with status: {current_task.status.value}")
        
        return {
            "execution_plan": plan.to_dict(),
            "current_task_id": current_task.id,
            "task_results": new_task_results,
            "tool_runs": new_tool_runs,
            "current_agent_role": AgentRole.CRITIC.value,
            "current_step": step_number,
            "stream_events": events,
        }
        
    except Exception as e:
        log.error(f"❌ Executor node error: {e}", exc_info=True)
        current_task.status = TaskStatus.FAILED
        current_task.error = str(e)
        
        await tracer.update_step(
            trace_id=trace_id,
            step_number=step_number,
            status=StepStatus.FAILED.value,
            error=str(e),
            completed_at=datetime.utcnow().isoformat()
        )
        
        return {
            "execution_plan": plan.to_dict(),
            "current_task_id": current_task.id,
            "current_agent_role": AgentRole.CRITIC.value,
            "current_step": step_number,
        }


# ==================== Critic Node ====================

async def critic_node(state: PlanningAgentState, deps: Dict[str, Any]) -> Dict[str, Any]:
    """
    گره ارزیابی - بررسی نتیجه task و تصمیم‌گیری.
    """
    plan_dict = state.get("execution_plan")
    if not plan_dict:
        return {"error": "No plan to evaluate"}
    
    plan = ExecutionPlan.from_dict(plan_dict)
    current_task_id = state.get("current_task_id")
    
    # Find current task
    current_task = next((t for t in plan.tasks if t.id == current_task_id), None)
    if not current_task:
        log.warning("⚠️ Current task not found for evaluation")
        return {
            "current_agent_role": AgentRole.EXECUTOR.value,
        }
    
    trace_id = state.get("trace_id", "")
    step_number = state.get("current_step", 0) + 1
    started_at = datetime.utcnow()
    
    # Create trace step
    tracer = get_trace_client()
    step = TraceStep(
        trace_id=trace_id,
        step_number=step_number,
        node_name="critic",
        task_id=current_task_id,
        status=StepStatus.RUNNING.value,
        started_at=started_at.isoformat()
    )
    await tracer.add_step(step)
    
    log.info(f"🔍 Critic evaluating task: {current_task.id}")
    
    await tracer.add_stream_event(
        trace_id=trace_id,
        event_type="node_start",
        event_data={"node": "critic", "step": step_number, "task_id": current_task_id}
    )
    
    # Get task result
    task_results = state.get("task_results", {})
    task_result = task_results.get(current_task_id, {})
    
    # Build evaluation context
    llm_client = deps["llm_client"]
    
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": CRITIC_USER_TEMPLATE.format(
            task_id=current_task.id,
            task_description=current_task.description,
            expected_outcome=current_task.reasoning or "Complete the task successfully",
            execution_result=json.dumps(task_result, ensure_ascii=False, default=str)[:2000],
            original_query=plan.original_query,
            plan_progress=f"{len(plan.get_completed_tasks())}/{len(plan.tasks)} tasks completed"
        )}
    ]
    messages = _planning_messages_with_rules(
        state, messages, user_message=plan.original_query
    )

    try:
        result = await llm_client.chat(messages=messages)
        
        if not result.get("ok"):
            log.warning(f"⚠️ Critic LLM call failed, defaulting to continue")
            
            duration = int((datetime.utcnow() - started_at).total_seconds() * 1000)
            await tracer.update_step(
                trace_id=trace_id,
                step_number=step_number,
                status=StepStatus.COMPLETED.value,
                output_data={"default": True, "reason": "LLM failed"},
                completed_at=datetime.utcnow().isoformat(),
                duration_ms=duration
            )
            
            # Default: continue execution
            return {
                "current_agent_role": AgentRole.EXECUTOR.value,
                "current_step": step_number,
            }
        
        eval_data = extract_json_from_response(result.get("content", ""))
        
        if not eval_data:
            log.warning("⚠️ Invalid critic response, defaulting to continue")
            
            duration = int((datetime.utcnow() - started_at).total_seconds() * 1000)
            await tracer.update_step(
                trace_id=trace_id,
                step_number=step_number,
                status=StepStatus.COMPLETED.value,
                output_data={"default": True, "reason": "Invalid response"},
                llm_messages_sent=messages,
                llm_response={"content": result.get("content")},
                completed_at=datetime.utcnow().isoformat(),
                duration_ms=duration
            )
            
            return {
                "current_agent_role": AgentRole.EXECUTOR.value,
                "current_step": step_number,
            }
        
        evaluation = CriticEvaluation(
            task_id=current_task_id,
            is_successful=eval_data.get("is_successful", True),
            confidence_score=eval_data.get("confidence_score", 0.7),
            issues_found=eval_data.get("issues_found", []),
            suggestions=eval_data.get("suggestions", []),
            needs_retry=eval_data.get("needs_retry", False),
            needs_replan=eval_data.get("needs_replan", False),
            replan_reason=eval_data.get("replan_reason"),
        )
        
        log.info(f"📊 Evaluation: success={evaluation.is_successful}, "
                 f"confidence={evaluation.confidence_score:.0%}, "
                 f"retry={evaluation.needs_retry}, replan={evaluation.needs_replan}")
        
        # Update evaluations
        evaluations = list(state.get("evaluations", []))
        evaluations.append(evaluation.to_dict())
        
        # Calculate overall confidence
        all_scores = [e.get("confidence_score", 0.5) for e in evaluations]
        overall_confidence = sum(all_scores) / len(all_scores) if all_scores else 0.5
        
        # Update trace step with evaluation
        duration = int((datetime.utcnow() - started_at).total_seconds() * 1000)
        await tracer.update_step(
            trace_id=trace_id,
            step_number=step_number,
            status=StepStatus.COMPLETED.value,
            evaluation=evaluation.to_dict(),
            confidence_score=evaluation.confidence_score,
            needs_retry=evaluation.needs_retry,
            needs_replan=evaluation.needs_replan,
            llm_messages_sent=messages,
            llm_response={"content": result.get("content")},
            completed_at=datetime.utcnow().isoformat(),
            duration_ms=duration
        )
        
        await tracer.add_stream_event(
            trace_id=trace_id,
            event_type="evaluation",
            event_data={
                "task_id": current_task_id,
                "is_successful": evaluation.is_successful,
                "confidence": evaluation.confidence_score,
                "needs_retry": evaluation.needs_retry,
                "needs_replan": evaluation.needs_replan
            }
        )
        
        # Decision logic
        # If task succeeded and there are more tasks to run, never replan
        has_remaining_tasks = bool(plan.get_next_task())
        if evaluation.needs_replan and evaluation.is_successful and has_remaining_tasks:
            log.info(f"✅ Task {current_task.id} succeeded — skipping replan, {len(plan.tasks) - len(plan.get_completed_tasks())} tasks remain")
            evaluation.needs_replan = False

        if evaluation.needs_replan:
            # Check if we can replan
            if state.get("replan_count", 0) < state.get("max_replans", 3):
                log.info("🔄 Critic requests re-planning")
                return {
                    "evaluations": evaluations,
                    "overall_confidence": overall_confidence,
                    "current_agent_role": AgentRole.PLANNER.value,  # Go to re-planner
                    "plan_status": PlanStatus.REVISING.value,
                    "replan_count": state.get("replan_count", 0) + 1,
                    "current_step": step_number,
                }
            else:
                log.warning("⚠️ Max replans reached, continuing anyway")
        
        if evaluation.needs_retry and current_task.retries < current_task.max_retries:
            log.info(f"🔄 Retrying task {current_task.id} (attempt {current_task.retries + 1})")
            # Reset task for retry
            current_task.status = TaskStatus.PENDING
            current_task.retries += 1
            return {
                "execution_plan": plan.to_dict(),
                "evaluations": evaluations,
                "overall_confidence": overall_confidence,
                "current_agent_role": AgentRole.EXECUTOR.value,
                "current_step": step_number,
            }
        
        # Check if plan is complete
        if plan.is_complete():
            log.info("✅ All tasks completed, moving to synthesis")
            return {
                "evaluations": evaluations,
                "overall_confidence": overall_confidence,
                "plan_status": PlanStatus.SYNTHESIZING.value,
                "current_agent_role": AgentRole.SYNTHESIZER.value,
                "current_step": step_number,
            }
        
        # Continue to next task
        return {
            "evaluations": evaluations,
            "overall_confidence": overall_confidence,
            "current_agent_role": AgentRole.EXECUTOR.value,
            "current_step": step_number,
        }
        
    except Exception as e:
        log.error(f"❌ Critic node error: {e}", exc_info=True)
        
        await tracer.update_step(
            trace_id=trace_id,
            step_number=step_number,
            status=StepStatus.FAILED.value,
            error=str(e),
            completed_at=datetime.utcnow().isoformat()
        )
        
        # Default: continue execution
        return {
            "current_agent_role": AgentRole.EXECUTOR.value,
            "current_step": step_number,
        }


# ==================== Re-Planner Node ====================

async def replanner_node(state: PlanningAgentState, deps: Dict[str, Any]) -> Dict[str, Any]:
    """
    گره بازبینی برنامه - ایجاد plan جدید بر اساس بازخوردها.
    """
    log.info("🔄 Re-planner: Creating revised plan...")
    
    trace_id = state.get("trace_id", "")
    step_number = state.get("current_step", 0) + 1
    started_at = datetime.utcnow()
    
    # Create trace step
    tracer = get_trace_client()
    step = TraceStep(
        trace_id=trace_id,
        step_number=step_number,
        node_name="replanner",
        status=StepStatus.RUNNING.value,
        started_at=started_at.isoformat()
    )
    await tracer.add_step(step)
    
    await tracer.add_stream_event(
        trace_id=trace_id,
        event_type="node_start",
        event_data={"node": "replanner", "step": step_number}
    )
    
    plan_dict = state.get("execution_plan")
    if not plan_dict:
        await tracer.update_step(
            trace_id=trace_id,
            step_number=step_number,
            status=StepStatus.FAILED.value,
            error="No plan to revise",
            completed_at=datetime.utcnow().isoformat()
        )
        return {"error": "No plan to revise", "current_step": step_number}
    
    plan = ExecutionPlan.from_dict(plan_dict)
    
    # Get completed and failed tasks
    completed_tasks = format_task_results(
        state.get("task_results", {}),
        plan.get_completed_tasks()
    )
    failed_tasks = "\n".join([
        f"- {t.id}: {t.description} (Error: {t.error})"
        for t in plan.get_failed_tasks()
    ]) or "None"
    
    # Get critic feedback
    evaluations = state.get("evaluations", [])
    last_eval = evaluations[-1] if evaluations else {}
    critic_feedback = json.dumps(last_eval, ensure_ascii=False) if last_eval else "No feedback"
    replan_reason = last_eval.get("replan_reason", "General re-planning needed")
    
    # Get tools
    tool_executor = deps["tool_executor"]
    openai_tools = await tool_executor.get_tools_for_file_context(
        file_context=state.get("file_context"),
        channel="llm"
    )
    
    # Call LLM for re-planning
    llm_client = deps["llm_client"]
    
    messages = [
        {"role": "system", "content": REPLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": REPLANNER_USER_TEMPLATE.format(
            original_query=plan.original_query,
            version=plan.version,
            previous_plan=json.dumps(plan.to_dict(), ensure_ascii=False)[:2000],
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            critic_feedback=critic_feedback,
            replan_reason=replan_reason
        )}
    ]
    messages = _planning_messages_with_rules(
        state, messages, user_message=plan.original_query
    )

    try:
        result = await llm_client.chat(messages=messages)
        
        if not result.get("ok"):
            log.error(f"❌ Re-planner failed: {result.get('error')}")
            
            duration = int((datetime.utcnow() - started_at).total_seconds() * 1000)
            await tracer.update_step(
                trace_id=trace_id,
                step_number=step_number,
                status=StepStatus.FAILED.value,
                error=result.get("error"),
                llm_messages_sent=messages,
                completed_at=datetime.utcnow().isoformat(),
                duration_ms=duration
            )
            
            # Continue with existing plan
            return {
                "plan_status": PlanStatus.EXECUTING.value,
                "current_agent_role": AgentRole.EXECUTOR.value,
                "current_step": step_number,
            }
        
        replan_data = extract_json_from_response(result.get("content", ""))
        
        if not replan_data or "new_plan" not in replan_data:
            log.warning("⚠️ Invalid replan response, continuing with existing plan")
            
            duration = int((datetime.utcnow() - started_at).total_seconds() * 1000)
            await tracer.update_step(
                trace_id=trace_id,
                step_number=step_number,
                status=StepStatus.COMPLETED.value,
                output_data={"continued": True, "reason": "Invalid response"},
                llm_messages_sent=messages,
                llm_response={"content": result.get("content")},
                completed_at=datetime.utcnow().isoformat(),
                duration_ms=duration
            )
            
            return {
                "plan_status": PlanStatus.EXECUTING.value,
                "current_agent_role": AgentRole.EXECUTOR.value,
                "current_step": step_number,
            }
        
        new_plan_data = replan_data.get("new_plan", {})
        
        # Create new ExecutionPlan
        new_plan = ExecutionPlan(
            id=f"plan_{state['trace_id'][:8]}_v{plan.version + 1}",
            original_query=plan.original_query,
            tasks=[SubTask.from_dict(t) for t in new_plan_data.get("tasks", [])],
            reasoning=replan_data.get("replan_reasoning", ""),
            complexity=new_plan_data.get("complexity", plan.complexity),
            estimated_steps=new_plan_data.get("estimated_steps", len(new_plan_data.get("tasks", []))),
            version=plan.version + 1
        )
        
        # Preserve results from specified tasks
        preserved_ids = set(replan_data.get("preserved_results", []))
        # Results are already in state.task_results, just keep them
        
        # Update replan history
        replan_history = list(state.get("replan_history", []))
        replan_history.append({
            "version": plan.version,
            "reason": replan_reason,
            "lessons": replan_data.get("lessons_learned", []),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Update trace step
        duration = int((datetime.utcnow() - started_at).total_seconds() * 1000)
        await tracer.update_step(
            trace_id=trace_id,
            step_number=step_number,
            status=StepStatus.COMPLETED.value,
            plan_generated=new_plan.to_dict(),
            planning_reasoning=new_plan.reasoning,
            llm_messages_sent=messages,
            llm_response={"content": result.get("content")},
            completed_at=datetime.utcnow().isoformat(),
            duration_ms=duration
        )
        
        # Update trace with new plan
        await tracer.update_trace(
            trace_id=trace_id,
            execution_plan=new_plan.to_dict(),
            replan_count=state.get("replan_count", 0)
        )
        
        # Language for events
        language = detect_language(plan.original_query)
        events = list(state.get("stream_events", []))
        if state.get("is_streaming"):
            events.append({
                "type": "progress",
                "message": get_progress_message("replanning", language),
                "new_plan": new_plan.to_dict(),
                "timestamp": datetime.utcnow().isoformat()
            })
        
        await tracer.add_stream_event(
            trace_id=trace_id,
            event_type="plan_revised",
            event_data={
                "version": new_plan.version,
                "tasks": len(new_plan.tasks),
                "reason": replan_reason
            }
        )
        
        log.info(f"✅ New plan created: version {new_plan.version}, {len(new_plan.tasks)} tasks")
        
        return {
            "execution_plan": new_plan.to_dict(),
            "plan_version": new_plan.version,
            "plan_status": PlanStatus.EXECUTING.value,
            "current_agent_role": AgentRole.EXECUTOR.value,
            "replan_history": replan_history,
            "stream_events": events,
            "current_step": step_number,
        }
        
    except Exception as e:
        log.error(f"❌ Re-planner error: {e}", exc_info=True)
        
        await tracer.update_step(
            trace_id=trace_id,
            step_number=step_number,
            status=StepStatus.FAILED.value,
            error=str(e),
            completed_at=datetime.utcnow().isoformat()
        )
        
        return {
            "plan_status": PlanStatus.EXECUTING.value,
            "current_agent_role": AgentRole.EXECUTOR.value,
            "current_step": step_number,
        }


# ==================== Synthesizer Node ====================

async def synthesizer_node(state: PlanningAgentState, deps: Dict[str, Any]) -> Dict[str, Any]:
    """
    گره ترکیب - ایجاد پاسخ نهایی از تمام نتایج.
    """
    log.info("📝 Synthesizer: Creating final response...")
    
    trace_id = state.get("trace_id", "")
    step_number = state.get("current_step", 0) + 1
    started_at = datetime.utcnow()
    
    # Create trace step
    tracer = get_trace_client()
    step = TraceStep(
        trace_id=trace_id,
        step_number=step_number,
        node_name="synthesizer",
        status=StepStatus.RUNNING.value,
        started_at=started_at.isoformat()
    )
    await tracer.add_step(step)
    
    await tracer.add_stream_event(
        trace_id=trace_id,
        event_type="node_start",
        event_data={"node": "synthesizer", "step": step_number}
    )
    
    plan_dict = state.get("execution_plan")
    if not plan_dict:
        err = state.get("error")
        msg = (
            f"اجرای برنامه کامل نشد. جزئیات: {err}"
            if err
            else "متأسفانه پاسخی یافت نشد."
        )
        await tracer.update_step(
            trace_id=trace_id,
            step_number=step_number,
            status=StepStatus.COMPLETED.value,
            output_data={"no_plan": True, "error": err},
            completed_at=datetime.utcnow().isoformat()
        )
        return {
            "final_answer": msg,
            "plan_status": PlanStatus.COMPLETED.value,
            "current_step": step_number,
        }
    
    plan = ExecutionPlan.from_dict(plan_dict)
    
    # Language detection  
    language = detect_language(plan.original_query)
    
    # Progress event
    events = list(state.get("stream_events", []))
    if state.get("is_streaming"):
        events.append({
            "type": "progress",
            "message": get_progress_message("synthesizing", language),
            "timestamp": datetime.utcnow().isoformat()
        })
    
    # Gather all results
    task_results = state.get("task_results", {})
    all_results = format_task_results(task_results, plan.tasks)
    
    # Plan summary
    plan_summary = f"""
Plan: {plan.reasoning}
Complexity: {plan.complexity}
Total Tasks: {len(plan.tasks)}
Completed: {len(plan.get_completed_tasks())}
Failed: {len(plan.get_failed_tasks())}
Version: {plan.version}
"""
    
    overall_confidence = state.get("overall_confidence", 0.7) * 100
    
    # Call LLM for synthesis
    llm_client = deps["llm_client"]
    
    err_note = ""
    if state.get("error"):
        err_note = f"\n\n[System: an error occurred during execution — explain honestly and use any partial results below. Error: {state.get('error')}]\n"

    # Inject pinned facts + history summary so the final answer stays
    # consistent with what was already established earlier in the chat.
    from ..pinned_facts import render_pinned_facts_block
    from ..repetition import render_anti_loop_block

    synth_system = SYNTHESIZER_SYSTEM_PROMPT
    pinned_block = render_pinned_facts_block(state.get("pinned_facts"))
    if pinned_block:
        synth_system = pinned_block + "\n\n" + synth_system
    history_summary = state.get("history_summary", "") or ""
    if history_summary:
        synth_system = (
            synth_system
            + "\n\n## Earlier Conversation Summary\n"
            + history_summary
        )
    anti_loop = render_anti_loop_block(state.get("conversation_state"))
    if anti_loop:
        synth_system = synth_system + "\n\n" + anti_loop

    messages = [
        {"role": "system", "content": synth_system},
        {"role": "user", "content": err_note + SYNTHESIZER_USER_TEMPLATE.format(
            original_query=plan.original_query,
            plan_summary=plan_summary,
            all_results=all_results,
            confidence=f"{overall_confidence:.0f}"
        )}
    ]
    messages = _planning_messages_with_rules(
        state, messages, user_message=plan.original_query
    )

    synth_max = _synthesis_max_tokens(plan, all_results)
    synth_extra = {"chat_template_kwargs": {"enable_thinking": False}}
    
    try:
        stream_writer = deps.get("stream_writer")

        if state.get("is_streaming") and stream_writer:
            collected_chunks: List[str] = []
            stream_failed = False
            stream_finish_reason: Optional[str] = None
            try:
                async for ev in llm_client.chat_stream(
                    messages=messages,
                    max_tokens=synth_max,
                    extra_body=synth_extra,
                ):
                    et = ev.get("type")
                    if et == "thinking_start":
                        stream_writer(
                            {
                                "type": "thinking_start",
                                "trace_id": trace_id,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                    elif et == "thinking":
                        stream_writer(
                            {
                                "type": "thinking",
                                "content": ev.get("content", ""),
                                "trace_id": trace_id,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                    elif et == "thinking_end":
                        stream_writer(
                            {
                                "type": "thinking_end",
                                "trace_id": trace_id,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                    elif et == "answer":
                        chunk = ev.get("content") or ""
                        if chunk:
                            collected_chunks.append(chunk)
                            stream_writer(
                                {
                                    "type": "token",
                                    "content": chunk,
                                    "trace_id": trace_id,
                                    "timestamp": datetime.utcnow().isoformat(),
                                }
                            )
                    elif et == "error":
                        stream_failed = True
                        log.warning(
                            "⚠️ Synthesizer stream chunk error: %s", ev.get("content")
                        )
                        break
                    elif et == "done":
                        stream_finish_reason = ev.get("finish_reason")
                        break
            except Exception as exc:
                stream_failed = True
                log.warning(
                    "⚠️ Synthesizer streaming failed, falling back to chat(): %s",
                    exc,
                )

            final_answer = "".join(collected_chunks).strip()

            if (
                not stream_failed
                and final_answer
                and stream_finish_reason == "length"
            ):
                log.warning(
                    "⚠️ Synthesis stream stopped at max_tokens — continuation via chat()"
                )
                cont = await llm_client.chat(
                    messages=messages
                    + [{"role": "assistant", "content": final_answer}]
                    + [
                        {
                            "role": "user",
                            "content": (
                                "Your previous reply was cut off by the output length limit. "
                                "Continue from the exact stopping point; output only the "
                                "missing remainder. Do not repeat text already shown."
                            ),
                        }
                    ],
                    max_tokens=min(synth_max * 2, 32768),
                    extra_body=synth_extra,
                )
                if cont.get("ok") and (cont.get("content") or "").strip():
                    tail = (cont.get("content") or "").strip()
                    final_answer = (final_answer.rstrip() + "\n\n" + tail).strip()

            if not stream_failed and final_answer:
                duration = int((datetime.utcnow() - started_at).total_seconds() * 1000)
                await tracer.update_step(
                    trace_id=trace_id,
                    step_number=step_number,
                    status=StepStatus.COMPLETED.value,
                    synthesis_content=final_answer[:1000],
                    llm_messages_sent=messages,
                    llm_response={"content": final_answer[:500]},
                    completed_at=datetime.utcnow().isoformat(),
                    duration_ms=duration,
                )
                total_duration = int(
                    (
                        datetime.utcnow()
                        - datetime.fromisoformat(
                            state.get("started_at", datetime.utcnow().isoformat()).replace(
                                "Z", ""
                            )
                        )
                    ).total_seconds()
                    * 1000
                )
                total_tool_calls = len(state.get("tool_runs", []))
                await tracer.update_trace(
                    trace_id=trace_id,
                    status=TraceStatus.COMPLETED.value,
                    final_answer=final_answer[:2000],
                    synthesis_reasoning=f"Synthesized from {len(plan.get_completed_tasks())} completed tasks",
                    overall_confidence=state.get("overall_confidence", 0.7),
                    total_steps=step_number,
                    total_tool_calls=total_tool_calls,
                    completed_at=datetime.utcnow().isoformat(),
                    total_duration_ms=total_duration,
                )
                if state.get("is_streaming"):
                    events.append(
                        {
                            "type": "progress",
                            "message": get_progress_message("completed", language),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                await tracer.add_stream_event(
                    trace_id=trace_id,
                    event_type="completed",
                    event_data={
                        "total_steps": step_number,
                        "total_tool_calls": total_tool_calls,
                        "confidence": state.get("overall_confidence", 0.7),
                    },
                )
                log.info(f"✅ Synthesis complete (streamed): {len(final_answer)} chars")
                return {
                    "final_answer": final_answer,
                    "synthesis_reasoning": f"Synthesized from {len(plan.get_completed_tasks())} completed tasks",
                    "synthesis_streamed_tokens": True,
                    "plan_status": PlanStatus.COMPLETED.value,
                    "completed_at": datetime.utcnow().isoformat(),
                    "stream_events": events,
                    "current_step": step_number,
                }

        result = await llm_client.chat(
            messages=messages,
            max_tokens=synth_max,
            extra_body=synth_extra,
        )
        
        if not result.get("ok"):
            log.error(f"❌ Synthesizer LLM failed: {result.get('error')}")
            # Fallback: concatenate results
            fallback_answer = f"نتایج:\n\n{all_results}"
            
            duration = int((datetime.utcnow() - started_at).total_seconds() * 1000)
            await tracer.update_step(
                trace_id=trace_id,
                step_number=step_number,
                status=StepStatus.COMPLETED.value,
                synthesis_content=fallback_answer[:500],
                error="LLM failed, using fallback",
                llm_messages_sent=messages,
                completed_at=datetime.utcnow().isoformat(),
                duration_ms=duration
            )
            
            return {
                "final_answer": fallback_answer,
                "plan_status": PlanStatus.COMPLETED.value,
                "completed_at": datetime.utcnow().isoformat(),
                "stream_events": events,
                "current_step": step_number,
            }
        
        final_answer = (result.get("content") or "").strip()
        if not final_answer and result.get("reasoning_content"):
            final_answer = str(result.get("reasoning_content", "")).strip()
            log.warning("⚠️ Synthesizer: empty content, using reasoning_content fallback")
        if not final_answer and all_results.strip():
            final_answer = (
                "بر اساس نتایج ابزارها:\n\n" + all_results[:8000]
            ).strip()
            log.warning("⚠️ Synthesizer: empty LLM output, using task results fallback")
        if not final_answer:
            final_answer = "متأسفانه نتوانستم پاسخ متنی نهایی تولید کنم؛ لطفاً دوباره بپرسید یا جزئیات بیشتری بدهید."
            log.error("❌ Synthesizer: no content, reasoning, or task results to show")
        
        # Update trace step
        duration = int((datetime.utcnow() - started_at).total_seconds() * 1000)
        await tracer.update_step(
            trace_id=trace_id,
            step_number=step_number,
            status=StepStatus.COMPLETED.value,
            synthesis_content=final_answer[:1000],
            llm_messages_sent=messages,
            llm_response={"content": final_answer[:500]},
            completed_at=datetime.utcnow().isoformat(),
            duration_ms=duration
        )
        
        # Update trace as completed
        total_duration = int((datetime.utcnow() - datetime.fromisoformat(state.get("started_at", datetime.utcnow().isoformat()).replace("Z", ""))).total_seconds() * 1000)
        total_tool_calls = len(state.get("tool_runs", []))
        
        await tracer.update_trace(
            trace_id=trace_id,
            status=TraceStatus.COMPLETED.value,
            final_answer=final_answer[:2000],
            synthesis_reasoning=f"Synthesized from {len(plan.get_completed_tasks())} completed tasks",
            overall_confidence=state.get("overall_confidence", 0.7),
            total_steps=step_number,
            total_tool_calls=total_tool_calls,
            completed_at=datetime.utcnow().isoformat(),
            total_duration_ms=total_duration
        )
        
        # Progress event
        if state.get("is_streaming"):
            events.append({
                "type": "progress",
                "message": get_progress_message("completed", language),
                "timestamp": datetime.utcnow().isoformat()
            })
        
        await tracer.add_stream_event(
            trace_id=trace_id,
            event_type="completed",
            event_data={
                "total_steps": step_number,
                "total_tool_calls": total_tool_calls,
                "confidence": state.get("overall_confidence", 0.7)
            }
        )
        
        log.info(f"✅ Synthesis complete: {len(final_answer)} chars")
        
        return {
            "final_answer": final_answer,
            "synthesis_reasoning": f"Synthesized from {len(plan.get_completed_tasks())} completed tasks",
            "plan_status": PlanStatus.COMPLETED.value,
            "completed_at": datetime.utcnow().isoformat(),
            "stream_events": events,
            "current_step": step_number,
        }
        
    except Exception as e:
        log.error(f"❌ Synthesizer error: {e}", exc_info=True)
        
        await tracer.update_step(
            trace_id=trace_id,
            step_number=step_number,
            status=StepStatus.FAILED.value,
            error=str(e),
            completed_at=datetime.utcnow().isoformat()
        )
        
        await tracer.update_trace(
            trace_id=trace_id,
            status=TraceStatus.FAILED.value,
            error=str(e)
        )
        
        return {
            "final_answer": f"خطا در ترکیب نتایج: {str(e)}",
            "plan_status": PlanStatus.FAILED.value,
            "error": str(e),
            "current_step": step_number,
        }


# ==================== Simple Response Node ====================

async def simple_response_node(state: PlanningAgentState, deps: Dict[str, Any]) -> Dict[str, Any]:
    """
    گره پاسخ ساده - برای درخواست‌های ساده بدون نیاز به planning.
    """
    log.info("💬 Simple response node...")
    
    trace_id = state.get("trace_id", "")
    step_number = state.get("current_step", 0) + 1
    started_at = datetime.utcnow()
    
    # Create trace step
    tracer = get_trace_client()
    step = TraceStep(
        trace_id=trace_id,
        step_number=step_number,
        node_name="simple_response",
        status=StepStatus.RUNNING.value,
        started_at=started_at.isoformat()
    )
    await tracer.add_step(step)
    
    # Get user query
    user_query = ""
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break
    
    llm_client = deps["llm_client"]
    
    from ..prompts.system import BASE_SYSTEM_PROMPT
    from ..prompts.qwen3_prompts import build_conversation_meta_block, build_file_registry_block
    from ..pinned_facts import render_pinned_facts_block
    from ..repetition import render_anti_loop_block

    user_content = await build_multimodal_content(
        text=user_query,
        file_context=state.get("file_context"),
        auth_context={
            "user_id": state.get("user_id"),
            "group_id": state.get("group_id"),
            "case_id": state.get("case_id"),
        },
    )
    
    system_prompt = BASE_SYSTEM_PROMPT + "\n\n" + SIMPLE_RESPONSE_PROMPT.format(query=user_query)

    # Inject pinned facts FIRST so the LLM always sees the anchor before any
    # potentially-stale summary content.
    pinned_block = render_pinned_facts_block(state.get("pinned_facts"))
    if pinned_block:
        system_prompt += "\n\n" + pinned_block

    # Inject conversation metadata
    conv_state = state.get("conversation_state")
    meta_block = build_conversation_meta_block(conv_state)
    if meta_block:
        system_prompt += "\n\n" + meta_block

    # Inject history summary
    history_summary = state.get("history_summary", "")
    if history_summary:
        system_prompt += "\n\n## Earlier Conversation Summary\n" + history_summary

    # Inject file registry
    registry_block = build_file_registry_block(
        state.get("file_registry"),
    )
    if registry_block:
        system_prompt += "\n\n" + registry_block

    # Anti-loop guard for long conversations
    anti_loop = render_anti_loop_block(conv_state)
    if anti_loop:
        system_prompt += "\n\n" + anti_loop

    # Inline small text/code/markdown files so the simple-response path can
    # answer textual questions directly without tool calls. Qwen3.5's chat
    # template only allows a single system message at position 0, so we merge
    # the block into the existing system prompt instead of appending it.
    inline_block, _ = await build_text_files_inline_block(state.get("file_context"))
    if inline_block:
        system_prompt = f"{system_prompt}\n\n{inline_block}"

    messages = [{"role": "system", "content": system_prompt}]
    
    # Include chat history (skip system markers, keep user/assistant turns)
    all_msgs = state.get("messages", [])
    for msg in all_msgs[:-1]:
        role = msg.get("role", "")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": msg.get("content", "")})
    
    messages.append({"role": "user", "content": user_content})
    messages = _planning_messages_with_rules(state, messages, user_message=user_query)

    try:
        result = await llm_client.chat(
            messages=messages,
        )
        
        duration = int((datetime.utcnow() - started_at).total_seconds() * 1000)
        
        if result.get("ok"):
            final_answer = result.get("content", "")
            if not final_answer and result.get("reasoning_content"):
                final_answer = result["reasoning_content"]

            # Safety: if LLM leaked a tool call as text, escalate to planning
            if _response_contains_tool_call(final_answer):
                log.warning("⚠️ Simple response contains tool call JSON — escalating to planning")
                await tracer.update_step(
                    trace_id=trace_id,
                    step_number=step_number,
                    status=StepStatus.COMPLETED.value,
                    output_data={"escalated": True, "reason": "tool_call_in_simple_response"},
                    completed_at=datetime.utcnow().isoformat(),
                    duration_ms=duration
                )
                return {
                    "complexity": "single_tool",
                    "needs_planning": True,
                    "current_agent_role": AgentRole.PLANNER.value,
                    "current_step": step_number,
                    "plan_status": PlanStatus.IN_PROGRESS.value,
                }

            await tracer.update_step(
                trace_id=trace_id,
                step_number=step_number,
                status=StepStatus.COMPLETED.value,
                synthesis_content=final_answer[:500],
                llm_messages_sent=messages,
                llm_response={"content": final_answer[:500]},
                completed_at=datetime.utcnow().isoformat(),
                duration_ms=duration
            )
            
            await tracer.update_trace(
                trace_id=trace_id,
                status=TraceStatus.COMPLETED.value,
                complexity="simple",
                final_answer=final_answer[:2000],
                total_steps=step_number,
                total_tool_calls=0,
                completed_at=datetime.utcnow().isoformat(),
                total_duration_ms=duration
            )
            
            return {
                "final_answer": final_answer,
                "plan_status": PlanStatus.COMPLETED.value,
                "completed_at": datetime.utcnow().isoformat(),
                "current_step": step_number,
            }
        else:
            await tracer.update_step(
                trace_id=trace_id,
                step_number=step_number,
                status=StepStatus.FAILED.value,
                error=result.get("error"),
                llm_messages_sent=messages,
                completed_at=datetime.utcnow().isoformat(),
                duration_ms=duration
            )
            
            await tracer.update_trace(
                trace_id=trace_id,
                status=TraceStatus.FAILED.value,
                error=result.get("error")
            )
            
            return {
                "error": result.get("error", "LLM call failed"),
                "plan_status": PlanStatus.FAILED.value,
                "current_step": step_number,
            }
            
    except Exception as e:
        log.error(f"❌ Simple response error: {e}")
        
        await tracer.update_step(
            trace_id=trace_id,
            step_number=step_number,
            status=StepStatus.FAILED.value,
            error=str(e),
            completed_at=datetime.utcnow().isoformat()
        )
        
        await tracer.update_trace(
            trace_id=trace_id,
            status=TraceStatus.FAILED.value,
            error=str(e)
        )
        
        return {
            "error": str(e),
            "plan_status": PlanStatus.FAILED.value,
            "current_step": step_number,
        }


# ==================== Exports ====================

__all__ = [
    "assess_complexity_node",
    "planner_node",
    "executor_node",
    "critic_node",
    "replanner_node",
    "synthesizer_node",
    "simple_response_node",
]
