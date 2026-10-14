"""
Qwen3-Optimized Prompts (Non-ReAct)
English prompts optimized for Qwen3 with Native Tool Calling

Based on Qwen documentation:
- No ReAct (model has built-in thinking mode)
- No manual <think> tags
- Trust native OpenAI ``tool_calls`` from the API (upstream vLLM: ``--tool-call-parser qwen3_coder``)
- Model does its own reasoning
"""

# ==================== Core System Prompt (Qwen3-Optimized) ====================

QWEN3_SYSTEM_PROMPT = """You are AIv2 — an intelligent multi-step assistant with native function-calling.
Your job is to actually accomplish the user's request, not just narrate intentions.

## ⚠️ GOLDEN RULES (read first, never break)

1. **Latest message is the priority.** Use chat history only for context — never repeat past answers.
2. **Match the user's language exactly.** Persian → Persian, English → English. If they mix, follow the dominant language of their latest message.
3. **Iterative work, one step at a time.** This is Cline-style ReAct: emit ONE round of tool_calls, wait for results, then decide the next step using those results.
4. **Never fake tool use.** The ONLY valid way to call a tool is the native OpenAI `tool_calls` payload — wired up in your function-calling channel. You must NOT, under any circumstances, write any of the following in your visible reply:
   - `tool_name(arg=value)` as prose or in any code fence (e.g. ```` ``` ```` , ```` ```python ```` , ```` ```text ```` ).
   - `plugin.x.y({"queries": [...]})` style calls inside any fence.
   - ```` ```json ```` blocks containing `{"tool_calls": [...]}` or `{"name": "...", "arguments": {...}}`.
   - ```` ```tool_call ```` / ```` ```yaml ```` / `<tool_call>` / `<function=...>` blocks.
   - Pseudo-YAML like `tool_calls:\\n- tool_name: ...`.
   - **A bullet list of past tool calls.** Lines like `- session_files({"session_id": "..."}) → در حال اجرا` or `* graph_search({...}) → ok` are FORBIDDEN. These come from a system-internal continuity block and must NEVER appear in your own reply.
   - **Headers like `(خلاصهٔ خروجی ابزار در همین پاسخ ...)`** or `[Previous tool outputs from this assistant turn …]` — those are internal markers, never write them.
   - Narrate "I'm calling …" / "let me search …" without an actual native tool_call in the same turn.

   If you need data → emit a real native `tool_call` and stop talking; the system will run it and feed you the result in the next turn. If you can answer → write the final answer with NO tool blocks at all and NO bullet list of past tools.
5. **Do not stop short.** Collect enough information before answering. If a tool fails or returns empty, try different arguments or another tool from the API `tools` list before giving up.

## Tools (single source)

Tool definitions for this turn are supplied **only** via the API `tools` parameter (OpenAI function calling). Read each function's `description` and `parameters` there before calling. This prompt does not duplicate tool catalogs.

## Session workspace

The user's chat session is your project root (Cline-style CWD). Uploaded files and runtime blocks below describe what is already in the session — not which tools exist. Choose tools from the API `tools` list.

## 👁️ Native vision (image + video)

**You are a vision-capable model.** When the user attaches an image or video to a turn, the orchestrator embeds the actual pixels in the `user` message as `image_url` / `video_url` parts. You can SEE them directly — describe, OCR, count, compare, answer spatial questions, anything. No tool call required.

- Do NOT call a vision-indexing tool just to "view" or describe what's in a picture when pixels are already in the user message.
- For media in the session but not attached to this turn, use an appropriate tool from the API `tools` list (read its description first).

## 🧭 Decision Loop (run this every turn before responding)

1. **Restate the user's latest ask** privately (one sentence).
2. **What data do I already have?** Check earlier tool results in this conversation.
3. **What's missing?** If nothing is missing → produce the final answer. If something is missing → pick the *best* tool(s) and emit native `tool_calls`.
4. **Parallel when independent.** You may emit multiple `tool_calls` in one turn when the calls don't depend on each other.
5. **One round per turn.** After your tool_calls leave the model, you'll get their results in the next system turn — then loop.
6. **Stop when you can answer well.** Don't keep calling tools just to be safe; don't stop early if you genuinely lack info.

## 🚫 Anti-patterns (never do these)

- Don't ask the user clarifying questions for things a tool can answer (e.g. "what files are uploaded?" — use the appropriate tool from the API `tools` list).
- Don't echo or summarize earlier tool blocks back into your final answer ("Previous tool outputs from this assistant turn …" is internal — never repeat).
- Don't paste raw JSON tool envelopes back to the user.
- Don't promise to do something next turn — do it this turn if a tool can.

## 🪶 Response Style

- Be direct and technical. Skip filler like "Sure!", "Great!".
- Format with concise headers / bullet lists when there is structure.
- Quote evidence from tool outputs (numbers, file names, ids) — never invent.
- If a tool errored, say so briefly and either retry intelligently or explain what input is needed.
- The final reply is the only thing the user sees from this turn — make it complete.
"""

# ==================== File Context Prompts ====================

QWEN3_IMAGE_CONTEXT = """## Attached File: Image

**Filename:** {file_name}
**Path:** {file_path}

An image has been uploaded. Run appropriate image analysis tools to analyze it.
"""

# ==================== Multi-File Context ====================

QWEN3_MULTI_FILE_CONTEXT = """## Session Files ({file_count} files)

You have access to the following files:

{file_list}

**How to use these files:**
- When calling tools, use the `filename` argument with the exact filename shown above.
- The system resolves paths automatically.
- Pick tools from the API `tools` list (read each `description` / `parameters`).

If the user asks about a specific file, use that filename.
If the user says "all files" or "همه فایل‌ها", process each one using appropriate tools from the API `tools` list.
"""

# ==================== Session Workspace ====================
# Rich context block giving the LLM full awareness of session files
# and accumulated knowledge from previous tool analyses.

SESSION_WORKSPACE_BLOCK = """## Session Workspace
Files uploaded in this session ({total_files} total, ordered by upload time):

{workspace_entries}

### File Reference Rules
- #1 = **first** file uploaded, #{latest_ordinal} = **latest/last** file uploaded
- "this file" / "این فایل" = latest file (#{latest_ordinal})
- "first image" / "اولین عکس" = #1 (or first of that type)
- "all files" / "همه فایل‌ها" = every file listed above
- When no specific file is mentioned, default to **#{latest_ordinal}** (latest)

### How to Work with Files
- [ANALYZED] entries above are accumulated knowledge — use them directly in your answers
- [INLINED] entries already have text embedded ("## File Content") — answer from that content, do NOT re-read
- [NOT YET ANALYZED] files need a tool call (choose from the API `tools` list) before you describe their contents
- Use `file_id` / filename from listings when calling session-scoped tools
"""

MULTI_FILE_ANALYSIS_HINT = """## Multi-File Analysis Required
The user wants to analyze **{file_count} files** together.
Strategy: process each file **individually** first, then produce a **combined summary**.
Call the appropriate tool once per file, then write a unified analysis covering all results.
"""

QWEN3_SINGLE_FILE_CONTEXT = """## Attached File

**Filename:** {file_name}
**Type:** {file_type}

A {file_type} file has been uploaded. When calling tools, use `filename="{file_name}"`.
"""

# Specialised context block for text / document files. The LLM is told explicitly
# how to access the file's actual content (the gap that previously made text files
# essentially invisible to the model).
QWEN3_TEXT_FILE_CONTEXT = """## Attached File: {file_type}

**Filename:** {file_name}
**Type:** {file_type}{inline_note}

This is a **textual file** (code / markdown / json / csv / log / plain text).

**To read its content:** use a text/session read tool from the API `tools` list (read `description` / `parameters`).
For large files, use line-range arguments if the tool supports them.

If the user asks "what does this file contain", read it with an appropriate tool first, then answer.
"""

QWEN3_UNKNOWN_FILE_CONTEXT = """## Attached File (Unknown Type)

**Filename:** {file_name}

⚠️ This file has no extension. Use a file-identify tool from the API `tools` list first to detect its type.
"""

QWEN3_AUDIO_CONTEXT = """## Attached File: Audio

**Filename:** {file_name}
**Path:** {file_path}

An audio file has been uploaded. Use audio processing tools to transcribe or analyze it.
"""

QWEN3_VIDEO_CONTEXT = """## Attached File: Video

**Filename:** {file_name}
**Path:** {file_path}

A video has been uploaded. Extract basic information first.
"""

QWEN3_GENERIC_FILE_CONTEXT = """## Attached File: {file_type}

**Filename:** {file_name}
**Path:** {file_path}

A file has been uploaded. Run appropriate tools to analyze it.
"""

QWEN3_NO_FILE_CONTEXT = """## File status: no file attached *in this message*

The user did not attach a new file in this turn — **that does not mean the session is empty**.

### Session workspace (project folder / CWD)
If you need files already in this chat or storage, use tools from the API `tools` list (read each `description`).
Pick `file_id` / filename from session listings when tools require them.

Do **not** describe file contents from imagination: either read with a tool or ask a clarifying question.
"""

# ==================== Tool Result Prompts ====================

QWEN3_TOOL_SUCCESS_PROMPT = """Tool {tool_name} result received.

Based on this result:
- If you have enough info to answer the user, provide the final response
- If you need more info, call another tool
"""

QWEN3_TOOL_ERROR_PROMPT = """Tool {tool_name} failed with error: {error}

Options:
1. Try an alternative tool
2. Retry with different parameters
3. Explain the issue to the user
"""

# ==================== Agent-Specific Prompts (DYNAMIC) ====================
# ⚠️ DYNAMIC ONLY - All prompts come from tool-runner API
# No hardcoded fallbacks - cache is filled at startup from /tools/agent-prompts

# ==================== Chat History Markers ====================

CHAT_HISTORY_START = """📜 **CONVERSATION HISTORY** (for context only - focus on the LATEST message):"""

CHAT_HISTORY_END = """📜 **END OF HISTORY** - Now respond to the user's NEW request above:"""

# ==================== Conversation Metadata Block ====================

CONVERSATION_META_BLOCK = """## Conversation State
- Turn: {turn_number} | Phase: {phase}
{greeting_line}
- Active topic: {active_topic}
{topic_change_line}
{file_line}

RULES based on conversation state:
{rules}
"""

# ==================== Dynamic Caches ====================

# Cache برای agent prompts (از tool-runner)
_agent_prompts_cache: dict = {}


def set_dynamic_tools_info(tools_info: str):
    """Legacy no-op: tool catalogs are not injected into chat prompts anymore."""
    del tools_info


def get_dynamic_tools_info() -> str:
    """Legacy no-op: returns empty string (tools SSOT is API ``tools`` only)."""
    return ""


def set_agent_prompts_cache(prompts: dict):
    """
    Set cached agent prompts from tool-runner.
    
    Args:
        prompts: {"image": {"role": "...", "prompt": "..."}, ...}
    """
    global _agent_prompts_cache
    _agent_prompts_cache = prompts


def get_agent_prompt(agent_type: str) -> str:
    """
    Get agent prompt - فقط از cache داینامیک.
    
    ⚠️ بدون fallback - اگه cache خالی باشه، generic می‌برگردونه.
    
    Args:
        agent_type: نوع agent (image, audio, text, general, ...)
    
    Returns:
        Prompt متنی
    """
    # vision → image mapping
    if agent_type == "vision":
        agent_type = "image"
    
    # فقط از cache - بدون hardcoded fallback
    if _agent_prompts_cache and agent_type in _agent_prompts_cache:
        return _agent_prompts_cache[agent_type].get("prompt", "")
    
    # Generic prompt بدون نام ابزار خاص
    return f"""## Role: {agent_type.title()} Specialist

You are a specialist in {agent_type} processing.
Use the available tools shown via function calling to complete the task.
"""


# ==================== Builder Functions ====================


def build_conversation_meta_block(conv_state) -> str:
    """
    Build the conversation metadata system message from a ConversationState.

    Returns empty string when conv_state is None (first turn / disabled).
    """
    if conv_state is None:
        return ""

    # Greeting line
    if conv_state.greeting_done:
        greeting_line = "- Greeting: already exchanged — do NOT greet again"
    else:
        greeting_line = "- Greeting: not yet exchanged"

    # Topic change line
    if conv_state.topic_changed:
        topic_change_line = (
            f"- Topic change detected: user switched from previous topic to '{conv_state.active_topic}' "
            "— do NOT reference previous topics unless asked"
        )
    else:
        topic_change_line = "- Topic change: none (continuing current topic)"

    # File context line
    if conv_state.file_discussed:
        turns_ago = conv_state.turn_number - conv_state.file_last_mentioned_turn
        if turns_ago == 0:
            file_line = "- File context: actively being discussed this turn"
        elif turns_ago <= 2:
            file_line = f"- File context: discussed {turns_ago} turn(s) ago — only reference if user asks"
        else:
            file_line = f"- File context: discussed {turns_ago} turns ago — do NOT reference unless user explicitly asks about it"
    else:
        file_line = "- File context: no file discussed in this session"

    # Build rules dynamically based on state
    rules = []
    if conv_state.greeting_done:
        rules.append("- Do NOT repeat greetings — the conversation is already underway")
    if conv_state.topic_changed:
        rules.append("- User changed topic — respond ONLY to the new topic")
        rules.append("- Do NOT bring up previous topics (files, images, etc.) unless explicitly asked")
    if conv_state.file_discussed and conv_state.turn_number - conv_state.file_last_mentioned_turn > 2:
        rules.append("- Do NOT mention or reference previously uploaded files unless the user asks about them")
    rules.append("- Focus ONLY on the user's current message")

    return CONVERSATION_META_BLOCK.format(
        turn_number=conv_state.turn_number,
        phase=conv_state.conversation_phase,
        greeting_line=greeting_line,
        active_topic=conv_state.active_topic,
        topic_change_line=topic_change_line,
        file_line=file_line,
        rules="\n".join(rules),
    )


def _format_workspace_entry(f: dict) -> str:
    """Format a single file entry for the session workspace prompt."""
    ftype = f.get("type", "unknown")
    header = f"#{f['ordinal']} `{f['name']}` -- {ftype} (turn {f['turn']})"

    extra_lines: list = []
    if f.get("inlined"):
        extra_lines.append(
            "   [INLINED] full text already embedded under \"## File Content\" — answer from there"
        )
    elif f.get("text_extracted"):
        chars = f.get("text_chars")
        chars_str = f" (~{chars} chars stored in metadata.text_extract)" if chars else ""
        extra_lines.append(
            f"   [TEXT_EXTRACTED]{chars_str} — use a read/search tool from the API `tools` list to view content"
        )

    knowledge = f.get("knowledge") or []
    if knowledge:
        for k in knowledge:
            summary = k.get("summary", "").replace("\n", " ").strip()
            if len(summary) > 200:
                summary = summary[:197] + "..."
            extra_lines.append(f'   [ANALYZED] {k["tool"]}: "{summary}"')
    elif not f.get("inlined") and not f.get("text_extracted"):
        extra_lines.append("   [NOT YET ANALYZED] — call an appropriate tool from the API `tools` list")

    return header + "\n" + "\n".join(extra_lines) if extra_lines else header


def build_file_registry_block(
    file_registry: list = None,
) -> str:
    """Build the session workspace block with accumulated file knowledge."""
    if not file_registry:
        return ""

    entries = "\n\n".join(_format_workspace_entry(f) for f in file_registry)

    return SESSION_WORKSPACE_BLOCK.format(
        total_files=len(file_registry),
        workspace_entries=entries,
        latest_ordinal=file_registry[-1]["ordinal"],
    )


def format_chat_session_runtime_block(session_id: str | None) -> str:
    """Stable UUID for session-scoped tools — models otherwise invent placeholders like ``current``."""
    sid = (session_id or "").strip()
    if not sid:
        return ""
    return (
        "## Runtime: this chat session (ground truth)\n"
        f"- **`chat_session_id`**: `{sid}`\n"
        "- For session-scoped tools that accept **`session_id`**, pass exactly this UUID (copy verbatim).\n"
        "- Use `file_id` values from session file listings when a tool's parameters require them."
    )


def build_qwen3_system_messages(
    file_context: dict = None,
    memory_context: list = None,
    agent_type: str = "general",
    step_number: int = 1,
    conversation_state=None,
    history_summary: str = "",
    file_registry: list = None,
    pinned_facts=None,
    chat_session_id: str | None = None,
) -> list:
    """
    Build optimized system messages for Qwen3.
    
    No ReAct, no <think> tags, trusting native model reasoning.
    
    Args:
        file_context: Attached file info
        memory_context: Conversation history
        agent_type: Agent type (vision, audio, text, general)
        step_number: Current step number
        conversation_state: ConversationState from conversation_state.py
        history_summary: Summarized older conversation history
        file_registry: Chronological file list from get_session_file_registry
        pinned_facts: PinnedFacts (or dict) to anchor goal/constraints/etc.
        chat_session_id: Storage chat session UUID for session_* / vision_info(session_id=…)
    """
    # Local import to avoid a circular import with `pinned_facts.py` (which
    # imports settings; settings has no edge into prompts but keeping the
    # import lazy is safer for future refactors).
    from ..pinned_facts import render_pinned_facts_block
    from ..repetition import render_anti_loop_block

    # Tool catalog is only in the API ``tools`` parameter — not duplicated here.
    system_prompt = QWEN3_SYSTEM_PROMPT
    runtime = format_chat_session_runtime_block(chat_session_id)
    if runtime:
        system_prompt = f"{system_prompt}\n\n{runtime}"
    messages = [{"role": "system", "content": system_prompt}]
    
    # Pinned conversation facts — placed BEFORE history summary so the LLM
    # always sees the unchanging anchor first, regardless of how aggressive
    # the rolling-history compaction is.
    pinned_block = render_pinned_facts_block(pinned_facts)
    if pinned_block:
        messages.append({"role": "system", "content": pinned_block})

    # Conversation metadata (turn count / phase / topic-change signals)
    meta_block = build_conversation_meta_block(conversation_state)
    if meta_block:
        messages.append({"role": "system", "content": meta_block})

    # History summary (from smart summarizer)
    if history_summary:
        messages.append({"role": "system", "content": f"## Earlier Conversation Summary\n{history_summary}"})

    # Session file registry (chronological listing of all session files)
    registry_block = build_file_registry_block(file_registry)
    if registry_block:
        messages.append({"role": "system", "content": registry_block})

    # Anti-loop guard — only injected once the conversation is long enough
    # that drift / repetition becomes likely.
    anti_loop = render_anti_loop_block(conversation_state)
    if anti_loop:
        messages.append({"role": "system", "content": anti_loop})

    # Agent-specific prompt
    agent_prompt = get_agent_prompt(agent_type)
    if agent_prompt:
        messages.append({"role": "system", "content": agent_prompt})
    
    # File context
    if file_context:
        file_type = file_context.get("type", "unknown")
        file_name = file_context.get("name", "unknown")
        is_multi_file = file_context.get("is_multi_file", False)
        all_files = file_context.get("all_files", [])
        
        # ==========================================
        # Multi-file context: show filenames only
        # ==========================================
        if is_multi_file and len(all_files) > 1:
            # Build simple file list (filename + type only)
            file_list_lines = []
            for i, f in enumerate(all_files, 1):
                emoji = {"image": "🖼️", "audio": "🎵", "video": "🎬", "document": "📄", "unknown": "❓"}.get(f.get("type", "unknown"), "📎")
                
                # Check for auto-detected extension
                if f.get("extension_detected"):
                    original_name = f.get("original_name_no_ext", "")
                    note = f" ✅ (auto-detected from '{original_name}')"
                elif f.get("extension_unknown"):
                    note = " ⚠️ (unknown type - use file_identify)"
                elif not f.get("has_extension", True):
                    note = " ⚠️ (no extension - use file_identify first)"
                else:
                    note = ""
                
                file_list_lines.append(
                    f"{i}. {emoji} `{f.get('name')}` — {f.get('type')}{note}"
                )
            
            file_list = "\n".join(file_list_lines)
            context_prompt = QWEN3_MULTI_FILE_CONTEXT.format(
                file_count=len(all_files),
                file_list=file_list
            )
        # ==========================================
        # Single file with unknown type
        # ==========================================
        elif file_type == "unknown" or not file_context.get("has_extension", True):
            context_prompt = QWEN3_UNKNOWN_FILE_CONTEXT.format(
                file_name=file_name
            )
        # ==========================================
        # Single file with known type
        # ==========================================
        else:
            # Special-case textual files (document / text). They need an explicit
            # nudge that *content reading* is possible via text.read / session_read,
            # otherwise the LLM tends to either refuse or pick irrelevant tools.
            is_textual = file_type in ("document", "text")
            if not is_textual:
                mt = (file_context.get("mime_type") or "").lower()
                if mt.startswith("text/") or mt in (
                    "application/json", "application/xml",
                    "application/x-yaml", "application/yaml",
                ):
                    is_textual = True

            inline_note = ""
            if file_context.get("text_content_inlined"):
                inline_note = (
                    "\n\n> Note: the file's text content is already embedded earlier "
                    "in this prompt under \"## File Content\". You may answer directly "
                    "without calling `text.read` again."
                )

            if is_textual:
                context_prompt = QWEN3_TEXT_FILE_CONTEXT.format(
                    file_name=file_name,
                    file_type=file_type,
                    inline_note=inline_note,
                )
            elif file_context.get("extension_detected"):
                original_name = file_context.get("original_name_no_ext", "")
                context_prompt = QWEN3_SINGLE_FILE_CONTEXT.format(
                    file_name=file_name,
                    file_type=file_type
                ) + f"\n\nNote: This file was uploaded without extension. Auto-detected as {file_type} from '{original_name}'."
            else:
                context_prompt = QWEN3_SINGLE_FILE_CONTEXT.format(
                    file_name=file_name,
                    file_type=file_type
                )
        
        messages.append({"role": "system", "content": context_prompt})
    else:
        messages.append({"role": "system", "content": QWEN3_NO_FILE_CONTEXT})
    
    # Memory context (if exists)
    if memory_context:
        memory_text = "\n".join([
            f"- {m.get('content', '')[:100]}" 
            for m in memory_context[:5]
        ])
        messages.append({
            "role": "system",
            "content": f"## Previous conversation context:\n{memory_text}"
        })
    
    # Step indicator (only for subsequent steps)
    if step_number > 1:
        messages.append({
            "role": "system",
            "content": f"## Step {step_number}: Tool results received. Based on results, provide appropriate response or call another tool."
        })
    
    return messages


def build_qwen3_tool_result_message(
    tool_name: str,
    tool_result: dict,
    is_successful: bool = True
) -> str:
    """
    Build message after tool execution (no ReAct).
    """
    if is_successful:
        return QWEN3_TOOL_SUCCESS_PROMPT.format(tool_name=tool_name)
    else:
        error = tool_result.get("error", "Unknown error")
        return QWEN3_TOOL_ERROR_PROMPT.format(tool_name=tool_name, error=error)


def get_qwen3_final_instruction() -> str:
    """Instruction for final response."""
    return "Based on all gathered information, provide the final response in the user's language."


# ==================== Exports ====================

__all__ = [
    "QWEN3_SYSTEM_PROMPT",
    "QWEN3_IMAGE_CONTEXT",
    "QWEN3_AUDIO_CONTEXT",
    "QWEN3_VIDEO_CONTEXT",
    "QWEN3_GENERIC_FILE_CONTEXT",
    "QWEN3_TEXT_FILE_CONTEXT",
    "QWEN3_NO_FILE_CONTEXT",
    "QWEN3_TOOL_SUCCESS_PROMPT",
    "QWEN3_TOOL_ERROR_PROMPT",
    "QWEN3_VISION_AGENT",
    "QWEN3_AUDIO_AGENT",
    "QWEN3_TEXT_AGENT",
    "QWEN3_GENERAL_AGENT",
    "CHAT_HISTORY_START",
    "CHAT_HISTORY_END",
    "CONVERSATION_META_BLOCK",
    "SESSION_WORKSPACE_BLOCK",
    "MULTI_FILE_ANALYSIS_HINT",
    "build_conversation_meta_block",
    "build_file_registry_block",
    "build_qwen3_system_messages",
    "build_qwen3_tool_result_message",
    "get_qwen3_final_instruction",
]
