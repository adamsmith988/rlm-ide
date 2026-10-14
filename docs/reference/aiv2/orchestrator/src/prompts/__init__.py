"""Prompts Module."""
from .system import (
    BASE_SYSTEM_PROMPT,
    IMAGE_CONTEXT_PROMPT,
    AUDIO_CONTEXT_PROMPT,
    VIDEO_CONTEXT_PROMPT,
    GENERIC_FILE_CONTEXT_PROMPT,
    NO_FILE_CONTEXT_PROMPT,
    FORCED_FINAL_PROMPT,
    RETRY_PROMPT,
    MEMORY_CONTEXT_TEMPLATE,
    get_file_context_prompt,
    build_system_messages,
    build_tool_result_message,
)

__all__ = [
    "BASE_SYSTEM_PROMPT",
    "IMAGE_CONTEXT_PROMPT",
    "AUDIO_CONTEXT_PROMPT",
    "VIDEO_CONTEXT_PROMPT",
    "GENERIC_FILE_CONTEXT_PROMPT",
    "NO_FILE_CONTEXT_PROMPT",
    "FORCED_FINAL_PROMPT",
    "RETRY_PROMPT",
    "MEMORY_CONTEXT_TEMPLATE",
    "get_file_context_prompt",
    "build_system_messages",
    "build_tool_result_message",
]
