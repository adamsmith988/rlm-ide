"""
Advanced Agent Graph Nodes
گره‌های پیشرفته گراف عامل با تمام ویژگی‌های UX
"""
import logging
import asyncio
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from .state import AgentState, add_step_to_history, add_stream_event
from ..chat_history import get_chat_history_client
from ..utils.text_inline import build_text_files_inline_block

log = logging.getLogger(__name__)


# ==================== JSON Tool Call Parser ====================

def parse_json_tool_calls(content: str) -> List[Dict[str, Any]]:
    """
    پارس کردن tool calls از خروجی JSON مدل.
    وقتی native tool calling کار نمی‌کنه، مدل رو به خروجی JSON هدایت می‌کنیم.
    
    فرمت‌های پشتیبانی شده:
    - {"tool_calls": [{"name": "...", "arguments": {...}}]}
    - {"type": "tool_calls", "calls": [{"name": "...", "arguments": {...}}]}
    - خروجی مستقیم JSON در متن
    """
    if not content:
        return []
    
    try:
        # تلاش برای پارس مستقیم JSON
        data = json.loads(content)
        
        # فرمت 1: {"tool_calls": [...]}
        if isinstance(data, dict) and "tool_calls" in data:
            calls = data["tool_calls"]
            if isinstance(calls, list):
                return [
                    {
                        "id": f"parsed_{i}",
                        "name": c.get("name", ""),
                        "arguments": c.get("arguments", {})
                    }
                    for i, c in enumerate(calls) if c.get("name")
                ]
        
        # فرمت 2: {"type": "tool_calls", "calls": [...]}
        if isinstance(data, dict) and data.get("type") == "tool_calls":
            calls = data.get("calls", [])
            if isinstance(calls, list):
                return [
                    {
                        "id": f"parsed_{i}",
                        "name": c.get("name", ""),
                        "arguments": c.get("arguments", {})
                    }
                    for i, c in enumerate(calls) if c.get("name")
                ]
        
    except json.JSONDecodeError:
        pass
    
    # تلاش برای استخراج JSON از متن
    patterns = [
        r'\{"tool_calls":\s*\[(.*?)\]\}',
        r'\{"type":\s*"tool_calls",\s*"calls":\s*\[(.*?)\]\}',
        r'```json\s*(\{.*?\})\s*```',
        r'(\{[^{}]*"name"[^{}]*"arguments"[^{}]*\})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                json_str = match.group(0)
                # پاکسازی markdown code blocks
                json_str = re.sub(r'```json\s*|\s*```', '', json_str)
                data = json.loads(json_str)
                
                if isinstance(data, dict):
                    calls = data.get("tool_calls") or data.get("calls") or []
                    if isinstance(calls, list):
                        return [
                            {
                                "id": f"parsed_{i}",
                                "name": c.get("name", ""),
                                "arguments": c.get("arguments", {})
                            }
                            for i, c in enumerate(calls) if c.get("name")
                        ]
            except:
                continue
    
    return []


# ==================== Initialize Node ====================

async def initialize_node(state: AgentState, deps: Dict[str, Any]) -> Dict[str, Any]:
    """
    گره آماده‌سازی اولیه با معماری Multi-Agent.
    
    ویژگی‌ها:
    - تشخیص خودکار file context
    - انتخاب agent مناسب بر اساس نوع فایل
    - بارگذاری ابزارهای تخصصی agent
    - دریافت ترجیحات کاربر
    - آماده‌سازی حافظه
    """
    log.info(f"🚀 Initializing agent for session: {state['session_id']}")
    
    # تشخیص context فایل
    # اول از state (که ممکنه از پیام extract شده باشه)، بعد از artifacts
    file_context = state.get("file_context")
    if not file_context:
        file_context = deps["detect_file_context"](state.get("artifacts", []))
    
    if file_context:
        log.info(f"📁 File context: {file_context.get('type')} - {file_context.get('path')}")
    
    # دریافت Multi-Agent manifest
    tool_executor = deps["tool_executor"]
    
    # تلاش برای گرفتن multi-agent manifest
    try:
        multi_agent_manifest = await tool_executor.get_multi_agent_manifest()
        tools_manifest = multi_agent_manifest.get("all_tools", [])
    except Exception as e:
        log.warning(f"⚠️ Failed to get multi-agent manifest, falling back to simple: {e}")
        multi_agent_manifest = None
        tools_manifest = await tool_executor.get_tools_manifest()
    
    # تعیین نوع agent بر اساس file context
    agent_type = tool_executor.determine_agent_type_from_context(file_context)
    log.info(f"🤖 Selected agent type: {agent_type}")
    
    # دریافت ابزارهای agent تخصصی + general
    agent_tools = []
    if multi_agent_manifest:
        agents = multi_agent_manifest.get("agents", {})
        
        # ابزارهای agent تخصصی
        if agent_type in agents:
            agent_tools.extend(agents[agent_type].get("tools", []))
            log.info(f"   {agent_type} tools: {len(agents[agent_type].get('tools', []))}")
        
        # ابزارهای general همیشه در دسترس هستند
        if agent_type != "general" and "general" in agents:
            general_tools = agents["general"].get("tools", [])
            agent_tools.extend(general_tools)
            log.info(f"   general tools: {len(general_tools)}")
    
    # فیلتر ابزارهای فعال (برای OpenAI tools)
    active_tools = get_active_tools(
        agent_tools if agent_tools else tools_manifest, 
        file_context
    )
    
    # Pre-load small text/code/markdown files into the system prompt so the
    # LLM can reason about their actual content without making a tool call.
    inline_block, inlined_metas = await build_text_files_inline_block(file_context)
    if file_context and inlined_metas:
        inlined_names = {m["name"] for m in inlined_metas if m.get("name")}
        for f in (file_context.get("all_files") or []):
            if f.get("name") in inlined_names:
                f["text_content_inlined"] = True
        if file_context.get("name") in inlined_names:
            file_context["text_content_inlined"] = True

    # ساخت پیام‌های سیستم با Multi-Agent
    if multi_agent_manifest:
        from ..prompts.system import build_multi_agent_system_messages
        system_messages = build_multi_agent_system_messages(
            file_context=file_context,
            memory_context=state.get("memory_context", []),
            multi_agent_manifest=multi_agent_manifest
        )
    else:
        system_messages = deps["build_system_messages"](
            file_context=file_context,
            memory_context=state.get("memory_context", []),
            tools_manifest=tools_manifest
        )

    if inline_block:
        # Qwen3.5's chat template (`loop.first` check) forbids more than one
        # `system` message. Merge the inline-text block into the *first*
        # system message instead of appending a second one. Fallback: prepend
        # if the list is somehow empty.
        if system_messages:
            first = system_messages[0]
            existing = first.get("content") or ""
            first["content"] = f"{existing}\n\n{inline_block}" if existing else inline_block
        else:
            system_messages = [{"role": "system", "content": inline_block}]

    
    # ذخیره event برای streaming
    events = []
    if state.get("is_streaming"):
        events.append({
            "type": "init",
            "trace_id": state["trace_id"],
            "session_id": state["session_id"],
            "file_context": file_context,
            "agent_type": agent_type,
            "active_tools": active_tools,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    # ثبت در تاریخچه
    step_info = {
        "step": 0,
        "action": "initialize",
        "agent_type": agent_type,
        "file_context": file_context,
        "active_tools_count": len(active_tools),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return {
        "file_context": file_context,
        "agent_type": agent_type,  # ذخیره نوع agent در state
        "messages": system_messages + state["messages"],
        "active_tools": active_tools,
        "current_status": "ready",
        "current_step": 0,
        "steps_history": add_step_to_history(state, step_info),
        "stream_events": events if events else state.get("stream_events", [])
    }


# ==================== Planning Node (Optional) ====================

async def planning_node(state: AgentState, deps: Dict[str, Any]) -> Dict[str, Any]:
    """
    گره برنامه‌ریزی (اختیاری).
    
    برای کارهای پیچیده، ابتدا یک برنامه اجرا می‌سازد.
    این می‌تواند به LLM کمک کند تا بهتر تصمیم بگیرد.
    """
    # فعلاً skip - برای future implementation
    return {}


# ==================== Call LLM Node ====================

async def call_llm_node(state: AgentState, deps: Dict[str, Any]) -> Dict[str, Any]:
    """
    گره فراخوانی LLM با tool calling.
    
    ویژگی‌ها:
    - انتخاب خودکار ابزار
    - استخراج thinking
    - مدیریت خطا و retry
    - پشتیبانی از streaming
    """
    step = state.get("current_step", 0) + 1
    log.info(f"📡 LLM call - Step {step}/{state.get('max_steps', 5)}")
    
    # ارسال event شروع thinking
    events = list(state.get("stream_events", []))
    if state.get("is_streaming"):
        events.append({
            "type": "thinking_start",
            "step": step,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    try:
        llm_client = deps["llm_client"]
        tool_executor = deps["tool_executor"]
        
        # دریافت ابزارها بر اساس file_context
        # اگه file_context داریم، ابزارها بر اساس نوع فایل فیلتر میشن
        # اگه فایل نداریم، فقط ابزارهای بدون نیاز به فایل
        file_context = state.get("file_context")
        
        if file_context:
            # فیلتر بر اساس category (از نوع فایل) و channel=llm
            file_context_dict = None
            if hasattr(file_context, "__dict__"):
                file_context_dict = {
                    "type": getattr(file_context, "type", ""),
                    "mime_type": getattr(file_context, "mime_type", ""),
                    "name": getattr(file_context, "name", ""),
                    "path": getattr(file_context, "path", "")
                }
            elif isinstance(file_context, dict):
                file_context_dict = file_context
            
            openai_tools = await tool_executor.get_tools_for_file_context(
                file_context=file_context_dict,
                channel="llm"
            )
            log.info(f"📁 File-based tools: {len(openai_tools)} tools for type={file_context_dict.get('type') if file_context_dict else 'unknown'}")
        else:
            # بدون فایل - فقط ابزارهای بدون نیاز به فایل (text_search, enrich_entity)
            openai_tools = await tool_executor.get_tools_for_file_context(
                file_context=None,
                channel="llm"
            )
            log.info(f"📦 No-file tools: {len(openai_tools)} tools")
        
        # فیلتر ابزارهای فعال (اگر محدود شده باشند)
        active_tool_names = set(state.get("active_tools", []))
        if active_tool_names:
            openai_tools = [t for t in openai_tools if t.get("function", {}).get("name") in active_tool_names]
        
        # Debug: log messages being sent
        log.info(f"📤 Sending {len(state['messages'])} messages to LLM")
        for i, msg in enumerate(state["messages"]):
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))[:100]
            log.debug(f"  Message[{i}]: role={role}, content={content}...")
        log.info(f"📤 Tools count: {len(openai_tools) if openai_tools else 0}")
        log.info(f"📤 Active tools: {list(active_tool_names)[:5]}...")
        
        # Inject reflection guidance if available from previous reflection
        messages_for_llm = list(state["messages"])
        if state.get("confidence_score", 0) > 0 and step > 1:
            if state.get("needs_more_info"):
                messages_for_llm.append({
                    "role": "system",
                    "content": f"[SELF-EVALUATION from previous step: Confidence {state['confidence_score']:.0%}. "
                               f"Information is incomplete. Use additional tools to gather more data before answering.]"
                })
            elif state.get("confidence_score", 0) >= 0.7:
                messages_for_llm.append({
                    "role": "system",
                    "content": f"[SELF-EVALUATION from previous step: Confidence {state['confidence_score']:.0%}. "
                               f"You have sufficient information. Provide your final comprehensive answer now.]"
                })
        
        # فراخوانی LLM با tools
        try:
            result = await llm_client.chat(
                messages=messages_for_llm,
                tools=openai_tools if openai_tools else None,
                tool_choice="auto" if openai_tools else None
            )
        except Exception as tool_error:
            # اگه خطای tool calling بود، بدون tools امتحان کن
            log.warning(f"⚠️ Native tool calling failed: {tool_error}")
            result = await llm_client.chat(
                messages=messages_for_llm,
                tools=None,
                tool_choice=None
            )
            # صریحاً native_tool_calling رو False کن
            result["native_tool_calling"] = False
        
        # خواندن flag از result (می‌تونه از vllm_client برگردونده بشه)
        native_tool_calling = result.get("native_tool_calling", True)
        
        if not result.get("ok"):
            error_msg = result.get("error", "unknown_llm_error")
            log.error(f"🔴 LLM call failed: {error_msg}")
            
            # بررسی امکان retry
            retry_count = state.get("error_recovery_attempts", 0)
            if retry_count < 2:
                return {
                    "error_recovery_attempts": retry_count + 1,
                    "last_error": {"step": step, "error": error_msg, "type": "llm_error"},
                    "current_status": "retrying",
                    "stream_events": events
                }
            
            return {
                "error": error_msg,
                "current_status": "error",
                "stream_events": events
            }
        
        content = result.get("content", "")
        tool_calls = result.get("tool_calls", [])
        
        # ==================== JSON Tool Call Fallback ====================
        # اگه native tool calling کار نکرد و tool_calls خالیه،
        # تلاش کن از محتوای پاسخ JSON tool calls رو استخراج کنی
        if not tool_calls and not native_tool_calling and openai_tools:
            log.info("🔄 Trying JSON-based tool call parsing...")
            json_tool_calls = parse_json_tool_calls(content)
            
            if json_tool_calls:
                log.info(f"✅ Parsed {len(json_tool_calls)} tool calls from JSON output")
                # اعتبارسنجی ابزارهای پارس شده
                valid_tool_names = {t.get("function", {}).get("name") for t in openai_tools if t.get("function")}
                validated_calls = []
                for tc in json_tool_calls:
                    if tc.get("name") in valid_tool_names:
                        validated_calls.append(tc)
                    else:
                        log.warning(f"⚠️ Skipping invalid tool: {tc.get('name')}")
                
                if validated_calls:
                    tool_calls = validated_calls
                    # محتوای JSON رو از پاسخ نهایی حذف کن
                    content = re.sub(r'```json\s*\{.*?\}\s*```', '', content, flags=re.DOTALL)
                    content = re.sub(r'\{"tool_calls":\s*\[.*?\]\}', '', content, flags=re.DOTALL)
                    content = content.strip()
        
        # استخراج thinking از محتوا + فیلد reasoning جداگانهٔ vLLM (--reasoning-parser qwen3)
        thinking, clean_content = extract_thinking(content)
        reasoning_extra = (result.get("reasoning_content") or "").strip()
        if reasoning_extra:
            thinking = f"{thinking}\n\n{reasoning_extra}".strip() if thinking else reasoning_extra
        
        log.info(f"✅ LLM response: {len(content)} chars, {len(tool_calls)} tool_calls, native={native_tool_calling}")
        
        # اضافه کردن پاسخ assistant به messages (محتوای تمیز بدون thinking)
        new_message = {"role": "assistant", "content": clean_content}
        if tool_calls:
            new_message["tool_calls"] = [
                {
                    "id": tc.get("id", f"call_{step}_{i}"),
                    "type": "function",
                    "function": {
                        "name": tc.get("name"),
                        "arguments": json.dumps(tc.get("arguments", {})) if isinstance(tc.get("arguments"), dict) else tc.get("arguments", "{}")
                    }
                }
                for i, tc in enumerate(tool_calls)
            ]
        
        # ذخیره در تاریخچه
        step_info = {
            "step": step,
            "action": "llm_call",
            "thinking": thinking,
            "tool_calls_count": len(tool_calls),
            "content_length": len(content),
            "native_tool_calling": native_tool_calling,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # events برای streaming
        if state.get("is_streaming"):
            if thinking:
                events.append({
                    "type": "thinking",
                    "step": step,
                    "content": thinking,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            if tool_calls:
                events.append({
                    "type": "tool_selection",
                    "step": step,
                    "tools": [tc.get("name") for tc in tool_calls],
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        # انباشت توکن‌های مصرفی در state
        prev_prompt = state.get("tokens_prompt") or 0
        prev_completion = state.get("tokens_completion") or 0
        new_tokens_prompt = (prev_prompt + result.get("tokens_prompt", 0)) if result.get("tokens_prompt") else (prev_prompt or None)
        new_tokens_completion = (prev_completion + result.get("tokens_completion", 0)) if result.get("tokens_completion") else (prev_completion or None)

        return {
            "messages": state.get("messages", []) + [new_message],  # اضافه کردن به لیست موجود
            "pending_tool_calls": tool_calls,
            "current_step": step,
            "thinking": thinking,
            "current_status": "tools_selected" if tool_calls else "generating_response",
            "steps_history": add_step_to_history(state, step_info),
            "stream_events": events,
            "error_recovery_attempts": 0,  # Reset on success
            "tokens_prompt": new_tokens_prompt,
            "tokens_completion": new_tokens_completion,
        }
        
    except Exception as exc:
        log.error(f"🔴 LLM call exception: {exc}", exc_info=True)
        
        retry_count = state.get("error_recovery_attempts", 0)
        if retry_count < 2:
            return {
                "error_recovery_attempts": retry_count + 1,
                "last_error": {"step": step, "error": str(exc), "type": "exception"},
                "current_status": "retrying",
                "stream_events": events
            }
        
        return {
            "error": str(exc),
            "current_status": "error",
            "current_step": step,
            "stream_events": events
        }


# ==================== Execute Tools Node ====================

async def execute_tools_node(state: AgentState, deps: Dict[str, Any]) -> Dict[str, Any]:
    """
    گره اجرای ابزارها.
    
    ویژگی‌ها:
    - اجرای موازی ابزارها
    - Progress tracking
    - Error handling per tool
    - Streaming updates
    """
    tool_calls = state.get("pending_tool_calls", [])
    
    if not tool_calls:
        return {"pending_tool_calls": []}
    
    step = state.get("current_step", 0)
    log.info(f"🔧 Executing {len(tool_calls)} tool(s) at step {step}")
    
    tool_executor = deps["tool_executor"]
    
    events = list(state.get("stream_events", []))
    
    # ارسال event شروع اجرای ابزارها
    if state.get("is_streaming"):
        events.append({
            "type": "tools_start",
            "step": step,
            "tools": [tc.get("name") for tc in tool_calls],
            "count": len(tool_calls),
            "timestamp": datetime.utcnow().isoformat()
        })
    
    # اجرای موازی با tracking
    results = []
    tool_messages = []
    new_tool_runs = []
    warnings = list(state.get("warnings", []))
    
    # اجرای هر ابزار
    async def execute_single(call: Dict[str, Any], index: int) -> Dict[str, Any]:
        tool_id = call.get("id", f"tool_{step}_{index}")
        tool_name = call.get("name", "unknown")
        arguments = call.get("arguments", {})
        
        started_at = datetime.utcnow()
        
        # ارسال event شروع ابزار
        if state.get("is_streaming"):
            events.append({
                "type": "tool_start",
                "tool_id": tool_id,
                "tool_name": tool_name,
                "args": arguments,
                "timestamp": started_at.isoformat()
            })
        
        try:
            result = await tool_executor.execute(
                tool_name=tool_name,
                arguments=arguments,
                file_context=state.get("file_context"),
                session_id=state.get("session_id"),
                user_id=state.get("user_id"),
                group_id=state.get("group_id"),
            )
            
            completed_at = datetime.utcnow()
            execution_time = (completed_at - started_at).total_seconds()
            
            # ارسال event نتیجه ابزار
            if state.get("is_streaming"):
                events.append({
                    "type": "tool_result",
                    "tool_id": tool_id,
                    "tool_name": tool_name,
                    "ok": result.get("ok", False),
                    "execution_time": execution_time,
                    "timestamp": completed_at.isoformat()
                })
            
            return {
                "id": tool_id,
                "name": tool_name,
                "arguments": arguments,
                "result": result,
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "execution_time": execution_time
            }
            
        except Exception as exc:
            log.error(f"Tool {tool_name} exception: {exc}")
            completed_at = datetime.utcnow()
            
            return {
                "id": tool_id,
                "name": tool_name,
                "arguments": arguments,
                "result": {"ok": False, "error": str(exc)},
                "started_at": started_at.isoformat(),
                "completed_at": completed_at.isoformat(),
                "execution_time": (completed_at - started_at).total_seconds()
            }
    
    # اجرای موازی
    tasks = [execute_single(call, i) for i, call in enumerate(tool_calls)]
    execution_results = await asyncio.gather(*tasks)
    
    # پردازش نتایج
    for exec_result in execution_results:
        tool_id = exec_result["id"]
        tool_name = exec_result["name"]
        result = exec_result["result"]
        
        # اضافه کردن به tool_runs
        new_tool_runs.append({
            "id": tool_id,
            "tool_id": tool_name,
            "args": exec_result["arguments"],
            "result": result,
            "step": step,
            "execution_time": exec_result["execution_time"]
        })
        
        # ساخت پیام tool response برای LLM (کل envelope تا channels.llm استخراج شود)
        content = render_tool_result_for_llm(result)
        if not content:
            if result.get("ok"):
                content = ""
            else:
                content = f"Error: {result.get('error', 'unknown error')}"
                warnings.append(f"ابزار {tool_name} با خطا مواجه شد: {result.get('error')}")
        elif not result.get("ok"):
            warnings.append(f"ابزار {tool_name} با خطا مواجه شد: {result.get('error')}")
        
        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool_id,
            "name": tool_name,
            "content": content[:8000]  # محدود کردن طول
        })
        
        log.info(f"   {'✅' if result.get('ok') else '❌'} {tool_name}: {exec_result['execution_time']:.2f}s")
    
    # Note: Tool executions are saved via TraceClient to agent_tool_executions table
    # The duplicate tool_runs table has been deprecated
    
    # ذخیره در تاریخچه
    step_info = {
        "step": step,
        "action": "execute_tools",
        "tools_count": len(tool_calls),
        "successful": sum(1 for r in execution_results if r["result"].get("ok")),
        "failed": sum(1 for r in execution_results if not r["result"].get("ok")),
        "total_time": sum(r["execution_time"] for r in execution_results),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if state.get("is_streaming"):
        events.append({
            "type": "tools_complete",
            "step": step,
            "successful": step_info["successful"],
            "failed": step_info["failed"],
            "timestamp": datetime.utcnow().isoformat()
        })
    
    return {
        "messages": state.get("messages", []) + tool_messages,  # اضافه کردن به لیست موجود
        "tool_runs": state.get("tool_runs", []) + new_tool_runs,
        "pending_tool_calls": [],
        "current_status": "analyzing_results",
        "warnings": warnings,
        "tools_used_names": list(state.get("tools_used_names", [])) + [tc.get("name", "unknown") for tc in tool_calls],
        "steps_history": add_step_to_history(state, step_info),
        "stream_events": events
    }


# ==================== Reflection / Self-Evaluation Node ====================

REFLECTION_PROMPT_TEMPLATE = """You are evaluating whether an AI agent has gathered enough information to answer a user's question.

## User's Original Question:
{question}

## Tools Used So Far:
{tools_list}

## Tool Results Summary:
{results_summary}

## Current Step: {iteration}

## Task:
Assess the quality and completeness of gathered information.

Score each dimension from 1 to 10:
1. COMPLETENESS: Does the information fully address all aspects of the question?
2. ACCURACY: Are the tool results reliable, relevant, and non-contradictory?
3. CONFIDENCE: How confident would you be giving a final answer right now?

Respond ONLY with valid JSON (no markdown fences, no extra text):
{{"completeness": <1-10>, "accuracy": <1-10>, "confidence": <1-10>, "needs_more_tools": <true|false>, "reasoning": "<one sentence explanation>", "suggested_next_tool": "<tool name or null>"}}"""


def _parse_reflection_json(content: str) -> Dict[str, Any]:
    """Parse JSON from reflection LLM response."""
    if not content:
        return {"confidence": 5, "needs_more_tools": True, "reasoning": "No response"}
    
    # Try direct parse (strip markdown fences if present)
    clean = re.sub(r'```(?:json)?\s*|\s*```', '', content).strip()
    try:
        data = json.loads(clean)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Try to find JSON object in text
    match = re.search(r'\{[^{}]*"confidence"[^{}]*\}', content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
    
    # Fallback: extract numbers from text
    conf_match = re.search(r'"?confidence"?\s*[:=]\s*(\d+)', content)
    needs_match = re.search(r'"?needs_more(?:_tools)?"?\s*[:=]\s*(true|false)', content, re.IGNORECASE)
    
    return {
        "confidence": int(conf_match.group(1)) if conf_match else 5,
        "needs_more_tools": needs_match.group(1).lower() == "true" if needs_match else True,
        "reasoning": "Parsed from unstructured text"
    }


async def reflection_node(state: AgentState, deps: Dict[str, Any]) -> Dict[str, Any]:
    """
    Self-evaluation node that runs after tool execution.
    
    Calls LLM with a compact reflection prompt to evaluate:
    - Information completeness
    - Confidence level
    - Whether more tools are needed
    
    Updates state with confidence_score and needs_more_info.
    """
    log.info("🔍 Running self-evaluation (reflection)...")
    
    # Extract user's original question from messages
    user_question = "Unknown"
    for msg in state.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            user_question = msg.get("content", "Unknown")
            break
    
    # Tools used so far
    tools_used = state.get("tools_used_names", [])
    tools_list = ", ".join(tools_used) if tools_used else "None"
    
    # Summarize recent tool results (last 5)
    tool_runs = state.get("tool_runs", [])
    results_parts = []
    for run in tool_runs[-5:]:
        name = run.get("tool_id", run.get("name", "unknown"))
        ok = run.get("result", {}).get("ok", False)
        # Include a brief snippet of the result
        result_data = run.get("result", {})
        snippet = ""
        if ok:
            text = render_tool_result_for_llm(result_data)
            if text:
                snippet = f" — {text[:150]}"
        results_parts.append(f"- {name}: {'✅ success' if ok else '❌ failed'}{snippet}")
    
    results_summary = "\n".join(results_parts) if results_parts else "No results yet"
    iteration = state.get("current_step", 0)
    
    # Build reflection prompt
    prompt = REFLECTION_PROMPT_TEMPLATE.format(
        question=user_question[:500],
        tools_list=tools_list,
        results_summary=results_summary[:2000],
        iteration=iteration
    )
    
    # Streaming events
    events = list(state.get("stream_events", []))
    if state.get("is_streaming"):
        events.append({
            "type": "reflection_start",
            "step": iteration,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    try:
        llm_client = deps["llm_client"]
        result = await llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            tool_choice=None
        )
        
        if result.get("ok"):
            content = result.get("content", "")
            # Strip thinking tags if present
            _, clean_content = extract_thinking(content)
            reflection_data = _parse_reflection_json(clean_content)
            
            completeness = reflection_data.get("completeness", 5)
            accuracy = reflection_data.get("accuracy", 5)
            confidence_raw = reflection_data.get("confidence", 5)
            # Average of all three dimensions, normalized to 0-1
            confidence = (completeness + accuracy + confidence_raw) / 30.0
            needs_more = reflection_data.get("needs_more_tools", True)
            reasoning = reflection_data.get("reasoning", "")
            
            log.info(
                f"🔍 Reflection result: confidence={confidence:.0%} "
                f"(C={completeness}/A={accuracy}/Conf={confidence_raw}), "
                f"needs_more={needs_more}, reason={reasoning}"
            )
            
            if state.get("is_streaming"):
                events.append({
                    "type": "reflection_result",
                    "step": iteration,
                    "confidence": round(confidence, 2),
                    "needs_more": needs_more,
                    "reasoning": reasoning,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            return {
                "confidence_score": confidence,
                "needs_more_info": needs_more,
                "current_status": "reflected",
                "stream_events": events,
            }
        else:
            log.warning(f"⚠️ Reflection LLM call failed: {result.get('error')}")
            # Don't block the flow on reflection failure
            return {
                "confidence_score": 0.5,
                "needs_more_info": False,
                "current_status": "reflected",
                "stream_events": events,
            }
    
    except Exception as exc:
        log.error(f"🔴 Reflection exception: {exc}", exc_info=True)
        return {
            "confidence_score": 0.5,
            "needs_more_info": False,
            "current_status": "reflected",
            "stream_events": events,
        }


# ==================== Force Final Node ====================

async def force_final_node(state: AgentState, deps: Dict[str, Any]) -> Dict[str, Any]:
    """
    گره پاسخ نهایی اجباری.
    
    وقتی max_steps تمام شد، بدون ابزار پاسخ می‌دهد.
    """
    log.warning("⚠️ Max steps reached, forcing final answer")
    
    llm_client = deps["llm_client"]
    
    events = list(state.get("stream_events", []))
    if state.get("is_streaming"):
        events.append({
            "type": "forcing_final",
            "reason": "max_steps_reached",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    # اضافه کردن پرامپت اجباری
    forced_messages = state["messages"] + [
        {
            "role": "system", 
            "content": """حداکثر تعداد گام‌ها به پایان رسید.
دیگر نمی‌توانید از ابزار استفاده کنید.
بر اساس اطلاعاتی که تا الان جمع‌آوری کردید، پاسخ نهایی خود را به فارسی ارائه دهید.
اگر اطلاعات کافی ندارید، این را به کاربر بگویید."""
        }
    ]
    
    try:
        result = await llm_client.chat(
            messages=forced_messages,
            tools=None
        )
        
        if result.get("ok"):
            content = result.get("content", "")
            thinking, clean_content = extract_thinking(content)
            
            answer = clean_content if clean_content else content
            if not answer.strip():
                answer = "متأسفانه نتوانستم پاسخ مناسبی تولید کنم. لطفاً سوال خود را دوباره مطرح کنید."
            
            return {
                "final_answer": answer,
                "thinking": thinking or state.get("thinking"),
                "current_status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "stream_events": events
            }
        else:
            return {
                "final_answer": "خطا در تولید پاسخ نهایی. لطفاً دوباره تلاش کنید.",
                "error": result.get("error"),
                "current_status": "error",
                "stream_events": events
            }
            
    except Exception as exc:
        log.error(f"🔴 Force final exception: {exc}")
        return {
            "final_answer": f"خطا: {str(exc)}",
            "error": str(exc),
            "current_status": "error",
            "stream_events": events
        }


# ==================== Generate Suggestions Node ====================

async def generate_suggestions_node(state: AgentState, deps: Dict[str, Any]) -> Dict[str, Any]:
    """
    گره تولید پیشنهادات.
    
    بعد از پاسخ نهایی، پیشنهاداتی برای کاربر تولید می‌کند.
    """
    # اگر خطا داریم، پیشنهاد نمی‌دهیم
    if state.get("error"):
        return {}
    
    suggestions = []
    
    # بر اساس context فایل
    file_context = state.get("file_context")
    if file_context:
        file_type = file_context.get("type", "").lower()
        
        if file_type == "image":
            suggestions = [
                "آیا می‌خواهید متن موجود در تصویر را استخراج کنم؟",
                "آیا می‌خواهید چهره‌های موجود را شناسایی کنم؟",
                "آیا اطلاعات بیشتری درباره این تصویر می‌خواهید؟"
            ]
        elif file_type == "audio":
            suggestions = [
                "آیا می‌خواهید این فایل صوتی را به متن تبدیل کنم؟",
                "آیا می‌خواهید گویندگان را شناسایی کنم؟"
            ]
    
    # اگر ابزار زیادی استفاده شد
    if len(state.get("tool_runs", [])) > 3:
        suggestions.append("آیا نیاز به تحلیل عمیق‌تری دارید؟")
    
    # فقط 3 پیشنهاد نگه دار
    suggestions = suggestions[:3]
    
    return {
        "suggestions": suggestions
    }


# ==================== Helper Functions ====================

def get_active_tools(tools_manifest: List[Dict], file_context: Optional[Dict]) -> List[str]:
    """
    فیلتر ابزارهای فعال بر اساس context.
    
    اگر فایلی ضمیمه نشده: همه ابزارها فعال هستند تا LLM تصمیم بگیره
    اگر فایل ضمیمه شده: ابزارهای مرتبط با نوع فایل + ابزارهای عمومی
    """
    if not tools_manifest:
        return []
    
    all_tools = [t.get("name") for t in tools_manifest if t.get("name")]
    
    # اگر file_context نداریم - همه ابزارها فعال
    # LLM خودش تصمیم می‌گیره که کدوم رو استفاده کنه
    if not file_context:
        log.debug(f"No file context - all {len(all_tools)} tools active")
        return all_tools
    
    file_type = file_context.get("type", "").lower()
    
    # ابزارهای مرتبط با نوع فایل
    type_prefixes = {
        "image": ("image.", "face.", "ocr.", "file.", "embed."),
        "audio": ("audio.", "file.", "embed."),
        "video": ("video.", "audio.", "file.", "image.", "embed."),
        "text": ("text.", "file.", "embed."),
        "file": ("file.", "image.", "audio.", "text.", "embed."),  # ناشناخته
    }
    
    allowed_prefixes = type_prefixes.get(file_type, ("file.", "image.", "audio.", "text.", "embed."))
    
    # ابزارهای عمومی همیشه فعال
    general_tools = ("text.", "enrich", "geo", "analysis.")
    
    active = [
        t for t in all_tools 
        if any(t.startswith(p) for p in allowed_prefixes) or any(t.startswith(g) for g in general_tools)
    ]
    
    log.debug(f"File type '{file_type}' - {len(active)}/{len(all_tools)} tools active")
    return active


def extract_thinking(text: str) -> tuple:
    """استخراج thinking از پاسخ LLM."""
    import re
    
    if not text:
        return None, text
    
    # الگوهای مختلف thinking
    patterns = [
        r"<think>([\s\S]*?)</think>",
        r"<thinking>([\s\S]*?)</thinking>",
        r"<thoughts>([\s\S]*?)</thoughts>",
        r"<analysis>([\s\S]*?)</analysis>",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            thinking = match.group(1).strip()
            clean = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
            return thinking, clean
    
    return None, text


def _normalize_tool_runner_payload(data: Any) -> Any:
    """
    یکسان‌سازی لایه‌های پاسخ:
    - خروجی ToolExecutor: { ok, data: <بدنه HTTP tool-runner> }
    - بدنه tool-runner: { ok, tool_id, result: <خروجی واقعی ابزار با channels> }
    خروجی: دیکشنری ابزار (معمولاً شامل channels.llm).
    """
    if not isinstance(data, dict):
        return data

    http_body = data.get("data")
    if isinstance(http_body, dict):
        inner = http_body.get("result")
        if isinstance(inner, dict) and (
            "channels" in inner or inner.get("ok") is not None or inner.get("error") is not None
        ):
            return inner
        if "channels" in http_body:
            return http_body

    inner = data.get("result")
    if isinstance(inner, dict) and (
        "channels" in inner or inner.get("ok") is not None or inner.get("error") is not None
    ):
        return inner

    return data


def render_tool_result_for_llm(data: Any) -> str:
    """
    فقط محتوای کانال `channels.llm` ابزار برای مدل زبانی.

    بدون fallback به metadata، neo4j، rawdata، data آماری، یا فیلدهای دیگر.
    """
    if isinstance(data, str):
        return data.strip()

    if not isinstance(data, dict):
        return ""

    payload = _normalize_tool_runner_payload(data)

    if not isinstance(payload, dict):
        return ""

    ch = payload.get("channels") if isinstance(payload.get("channels"), dict) else {}
    llm_channel = ch.get("llm") if ch else None

    if llm_channel is None:
        return ""

    if isinstance(llm_channel, str):
        return llm_channel.strip()

    if isinstance(llm_channel, dict):
        text = llm_channel.get("text") or llm_channel.get("summary") or llm_channel.get("content")
        if text is not None and str(text).strip():
            return str(text).strip()
        return ""

    return str(llm_channel).strip()
