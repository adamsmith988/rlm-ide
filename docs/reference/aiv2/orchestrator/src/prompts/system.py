"""
Advanced System Prompts - Multi-Agent Architecture
Full English prompts for better LLM comprehension.
"""

# ==================== Base System Prompt ====================

BASE_SYSTEM_PROMPT = """You are an advanced AI assistant with access to powerful file analysis and search tools.

## Core Rules

1. **Language Matching**: ALWAYS respond in the SAME language as the user's message.
   - Persian/Farsi input → respond in Persian
   - English input → respond in English
   - Match the user's language exactly

2. **File Analysis** (when FILE_CONTEXT is present):
   - Call relevant tools IMMEDIATELY without asking permission
   - Use the appropriate analysis tools for the file type
   - Do NOT ask "should I analyze this?" — just do it

3. **No File Context**:
   - If user mentions "this image/file" but NO FILE_CONTEXT → ask them to upload

4. **General Conversation**:
   - For greetings and general questions, respond directly without tools

5. **Parallel Execution**: You CAN call multiple tools simultaneously

## Self-Evaluation Protocol

After receiving tool results, always assess:
- **Completeness**: Do I have all the information needed to answer?
- **Accuracy**: Are the results reliable and relevant?
- **Gaps**: Is there anything missing that another tool could provide?

If your confidence is low, use additional tools before responding.
If you have sufficient information, provide a comprehensive final answer.

## Response Quality Standards
- Provide structured, well-organized responses (headers, lists, etc.)
- Be thorough but concise
- If information is incomplete or uncertain, say so honestly
- Synthesize information from multiple tools into a coherent answer
"""

# ==================== Multi-Agent System Prompts ====================
# ⚠️ DYNAMIC ONLY - All agent prompts come from tool-runner API
# No hardcoded tool names - cache is filled at startup from /tools/agent-prompts

# ==================== File Context Prompts ====================

IMAGE_CONTEXT_PROMPT = """## FILE_CONTEXT: Image Ready for Analysis
- **Filename**: {file_name}
- **Path**: {file_path}

**Action Required**: Analyze this image using the available image analysis tools.
Do not ask for permission — execute tools immediately.
"""

AUDIO_CONTEXT_PROMPT = """## FILE_CONTEXT: Audio Ready for Analysis
- **Filename**: {file_name}
- **Path**: {file_path}

**Action Required**: Process this audio file using available audio tools.
Do not ask for permission — execute tools immediately.
"""

VIDEO_CONTEXT_PROMPT = """## FILE_CONTEXT: Video
- **Filename**: {file_name}
- **Path**: {file_path}

A video file has been attached. Extract basic information first.
"""

GENERIC_FILE_CONTEXT_PROMPT = """## FILE_CONTEXT: File Attached
- **Filename**: {file_name}
- **Path**: {file_path}

A file has been attached. Run appropriate analysis tools.
"""

# ==================== Special Prompts ====================

FORCED_FINAL_PROMPT = """## ⚠️ MAX STEPS REACHED

Maximum number of steps has been reached.
**You can no longer use any tools.**

Based on all information gathered so far:
1. Provide a summary of your findings
2. Give your final answer in the user's language
3. If information is insufficient, honestly tell the user

**Write your final answer now:**
"""

RETRY_PROMPT = """## 🔄 RETRY

The previous call encountered an error. Please try again:
1. Use a different tool
2. Try with different parameters
3. Or provide your final answer
"""

NO_FILE_CONTEXT_PROMPT = """## FILE_CONTEXT: None

No file is currently attached.

If the user refers to "this image/file/photo":
- Ask them to upload the file or provide the file path
- Do NOT call any file-dependent tools

For general questions, respond using your knowledge.
"""

# ==================== Memory Context ====================

MEMORY_CONTEXT_TEMPLATE = """## Conversation History

Summary of previous conversations with this user:
{memory_summary}

Use this context to personalize your responses.
"""

# ==================== Functions ====================

# ==================== Dynamic Agent Prompts ====================
# ⚠️ NO FALLBACK - All prompts must come from tool-runner
# Cache is filled at startup from /tools/agent-prompts endpoint

_agent_prompts_cache: dict = {}


def set_agent_prompts_cache(prompts: dict):
    """Set cached agent prompts from tool-runner."""
    global _agent_prompts_cache
    _agent_prompts_cache = prompts


def get_agent_prompt(agent_type: str) -> str:
    """
    دریافت پرامپت تخصصی برای یک agent - فقط از cache داینامیک.
    
    ⚠️ بدون fallback - اگه cache خالی باشه، generic می‌برگردونه.
    """
    # vision → image mapping
    if agent_type == "vision":
        lookup_type = "image"
    else:
        lookup_type = agent_type
    
    # فقط از cache - بدون hardcoded fallback
    if _agent_prompts_cache and lookup_type in _agent_prompts_cache:
        return _agent_prompts_cache[lookup_type].get("prompt", "")
    
    # Generic prompt بدون نام ابزار خاص
    return f"""## Role: {lookup_type.title()} Specialist

You are a specialist in {lookup_type} processing.
Use the available tools shown in your tool list to complete the task.
Execute appropriate tools based on the user's request.
"""


def determine_agent_type_from_file(file_context: dict) -> str:
    """
    تعیین نوع agent بر اساس نوع فایل.
    
    Args:
        file_context: اطلاعات فایل
    
    Returns:
        نوع agent مناسب
    """
    if not file_context:
        return "general"
    
    file_type = file_context.get("type", "").lower()
    mime_type = file_context.get("mime_type", "").lower()
    
    # Image types
    if file_type == "image" or mime_type.startswith("image/"):
        return "vision"
    
    # Audio types
    if file_type == "audio" or mime_type.startswith("audio/"):
        return "audio"
    
    # Video types - use vision for frame analysis
    if file_type == "video" or mime_type.startswith("video/"):
        return "vision"
    
    # Text types
    if file_type in ["text", "document"] or mime_type.startswith("text/"):
        return "text"
    
    # File types
    if file_type == "file":
        return "file"
    
    return "general"


def get_file_context_prompt(file_type: str, file_name: str, file_path: str) -> str:
    """دریافت پرامپت مناسب برای نوع فایل."""
    if file_type == "image":
        template = IMAGE_CONTEXT_PROMPT
    elif file_type == "audio":
        template = AUDIO_CONTEXT_PROMPT
    elif file_type == "video":
        template = VIDEO_CONTEXT_PROMPT
    else:
        template = GENERIC_FILE_CONTEXT_PROMPT
    
    return template.format(
        file_type=file_type,
        file_name=file_name,
        file_path=file_path
    )


def build_system_messages(
    file_context: dict = None,
    memory_context: list = None,
    tools_manifest: list = None,
    agent_type: str = None,
    agent_tools: list = None
) -> list:
    """
    ساخت پیام‌های سیستم با معماری Multi-Agent.
    
    Args:
        file_context: اطلاعات فایل ضمیمه
        memory_context: تاریخچه مکالمات
        tools_manifest: لیست همه ابزارها
        agent_type: نوع agent (vision, text, audio, file, analysis, general)
        agent_tools: لیست ابزارهای خاص این agent
    
    Returns:
        لیست پیام‌های سیستم
    """
    messages = []
    
    # 1. پرامپت پایه
    messages.append({"role": "system", "content": BASE_SYSTEM_PROMPT})
    
    # 2. تعیین agent type از file_context اگر مشخص نشده
    if not agent_type and file_context:
        agent_type = determine_agent_type_from_file(file_context)
    
    # 3. اضافه کردن پرامپت تخصصی agent
    if agent_type and agent_type != "general":
        agent_prompt = get_agent_prompt(agent_type)
        messages.append({"role": "system", "content": agent_prompt})
    
    # 4. اضافه کردن context فایل
    if file_context:
        file_prompt = get_file_context_prompt(
            file_type=file_context.get("type", "unknown"),
            file_name=file_context.get("name", "unknown"),
            file_path=file_context.get("path", "N/A")
        )
        messages.append({"role": "system", "content": file_prompt})
    else:
        messages.append({"role": "system", "content": NO_FILE_CONTEXT_PROMPT})
    
    # 5. اضافه کردن خلاصه حافظه
    if memory_context and len(memory_context) > 0:
        memory_items = []
        for m in memory_context[:5]:
            content = m.get('content', '')
            if content:
                memory_items.append(f"- {content[:150]}...")
        
        if memory_items:
            memory_summary = "\n".join(memory_items)
            memory_prompt = MEMORY_CONTEXT_TEMPLATE.format(memory_summary=memory_summary)
            messages.append({"role": "system", "content": memory_prompt})
    
    # 6. اضافه کردن لیست ابزارها (ترجیحاً ابزارهای agent، وگرنه همه)
    tools_to_show = agent_tools if agent_tools else tools_manifest
    
    if tools_to_show:
        tools_info = []
        for t in tools_to_show[:20]:  # حداکثر 20 ابزار
            name = t.get("name", "")
            desc = t.get("description", "")[:100]
            if name:
                tools_info.append(f"- **{name}**: {desc}")
        
        if tools_info:
            tools_list = "\n".join(tools_info)
            agent_info = f" (Agent: {agent_type})" if agent_type else ""
            more_text = f"\n\n... و {len(tools_to_show) - 20} ابزار دیگر" if len(tools_to_show) > 20 else ""
            
            messages.append({
                "role": "system",
                "content": f"""## AVAILABLE_TOOLS{agent_info}

ابزارهای در دسترس شما:
{tools_list}{more_text}

**یادآوری**: برای استفاده از ابزار، از OpenAI function calling یا JSON format استفاده کنید.
برای تحلیل فایل، حتماً از ابزارهای مرتبط استفاده کنید."""
            })
    
    return messages


def build_multi_agent_system_messages(
    file_context: dict = None,
    memory_context: list = None,
    multi_agent_manifest: dict = None
) -> list:
    """
    ساخت پیام‌های سیستم با اطلاعات کامل Multi-Agent.
    
    این تابع agent مناسب را انتخاب کرده و ابزارهای آن را در پرامپت قرار می‌دهد.
    همچنین ابزارهای general را هم اضافه می‌کند.
    
    Args:
        file_context: اطلاعات فایل ضمیمه
        memory_context: تاریخچه مکالمات
        multi_agent_manifest: manifest دسته‌بندی شده از tool-runner
    
    Returns:
        لیست پیام‌های سیستم
    """
    # تعیین agent type
    agent_type = determine_agent_type_from_file(file_context) if file_context else "general"
    
    # جمع‌آوری ابزارها
    agent_tools = []
    
    if multi_agent_manifest:
        agents = multi_agent_manifest.get("agents", {})
        
        # ابزارهای agent تخصصی
        if agent_type in agents:
            agent_tools.extend(agents[agent_type].get("tools", []))
        
        # ابزارهای general همیشه در دسترس هستند
        if agent_type != "general" and "general" in agents:
            agent_tools.extend(agents["general"].get("tools", []))
    
    return build_system_messages(
        file_context=file_context,
        memory_context=memory_context,
        tools_manifest=multi_agent_manifest.get("all_tools", []) if multi_agent_manifest else None,
        agent_type=agent_type,
        agent_tools=agent_tools if agent_tools else None
    )


def build_tool_result_message(tool_name: str, result: dict) -> str:
    """
    ساخت پیام نتیجه ابزار برای نمایش به کاربر.
    
    Args:
        tool_name: نام ابزار
        result: نتیجه ابزار
    
    Returns:
        پیام formatted
    """
    if result.get("ok"):
        return f"✅ ابزار {tool_name} با موفقیت اجرا شد."
    else:
        error = result.get("error", "unknown error")
        return f"❌ ابزار {tool_name} با خطا مواجه شد: {error}"

