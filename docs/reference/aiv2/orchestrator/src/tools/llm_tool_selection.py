"""
LLM-facing tool selection for the Cline XML engine.

The tool-runner can expose many operational tools. The chat LLM should only
see a small candidate set that can actually return useful text to the LLM and
is relevant to the current session/file context.

Rules implemented here:

1. A tool is eligible only if it has an ``llm`` output channel.
2. Session tools are always included (session = workspace).
3. Text-input tools with an LLM output are general tools and are always
   included — except plugin tools (see 6).
4. If the current user message has uploaded/attached files, include LLM-output
   tools whose input type matches those files.
5. Also include LLM-output tools matching any files already present in the
   session registry, so the model can act on existing workspace files.
6. Plugin tools (``plugin_id``, stamped by the tool-runner from plugin.json)
   are only exposed when the host page sends matching ``active_plugin``. Main
   chat (no active_plugin) never sees plugin tools. No per-plugin hardcodes.
7. Per-turn intent narrowing matches tools by their tool.json ``category`` /
   ``capabilities`` using the rules in ``tool_intents.json`` — never by tool id.

``input_types`` is the preferred explicit metadata in ``tool.json``.
Singular ``input_type`` remains a read-time fallback. For older manifests we
still infer from ``file_types``, ``category`` and argument names so production
stays backwards-compatible while manifests finish migrating.
"""
from __future__ import annotations

import mimetypes
from typing import Any, Dict, Iterable, List, Optional, Set

from . import tool_intents


SESSION_INPUT = "session"
TEXT_INPUT = "text"
FILE_INPUT = "file"
IMAGE_INPUT = "image"
VIDEO_INPUT = "video"
AUDIO_INPUT = "audio"
DOCUMENT_INPUT = "document"


def _tool_name(tool: Dict[str, Any]) -> str:
    return str(tool.get("id") or tool.get("name") or "")


def _has_llm_output(tool: Dict[str, Any]) -> bool:
    if tool.get("enabled") is False:
        return False
    if tool.get("llm_exposed") is False or tool.get("hidden_from_llm") is True:
        return False
    channels = tool.get("channels")
    if isinstance(channels, list) and "llm" in channels:
        return True
    output_channels = tool.get("output_channels")
    if isinstance(output_channels, dict) and "llm" in output_channels:
        return True
    returns_schema = tool.get("returns_schema")
    if isinstance(returns_schema, dict):
        props = returns_schema.get("properties") or {}
        ch = props.get("channels") if isinstance(props, dict) else None
        ch_props = ch.get("properties") if isinstance(ch, dict) else None
        if isinstance(ch_props, dict) and "llm" in ch_props:
            return True
    return False


def _args_dict(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer ``args_schema.properties``, then legacy ``args`` / ``parameters``."""
    schema = tool.get("args_schema")
    if isinstance(schema, dict):
        props = schema.get("properties")
        if isinstance(props, dict) and props:
            return props
    params = tool.get("parameters")
    if isinstance(params, dict):
        props = params.get("properties")
        if isinstance(props, dict) and props:
            return props
    args = tool.get("args") or tool.get("arguments")
    if isinstance(args, dict):
        return args
    return {}


def _mime_to_input_type(mime_or_type: str) -> str:
    value = (mime_or_type or "").lower()
    if value.startswith("image/") or value == "image":
        return IMAGE_INPUT
    if value.startswith("video/") or value == "video":
        return VIDEO_INPUT
    if value.startswith("audio/") or value == "audio":
        return AUDIO_INPUT
    if value.startswith("text/") or value in {"text", "code", "markdown", "csv", "json"}:
        return TEXT_INPUT
    if value in {"document", "pdf"}:
        return DOCUMENT_INPUT
    if value == "session":
        return SESSION_INPUT
    return FILE_INPUT


def _normalise_input_types(value: Any) -> Set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {_mime_to_input_type(v.strip()) for v in value.split(",") if v.strip()}
    if isinstance(value, list):
        out: Set[str] = set()
        for item in value:
            if isinstance(item, str):
                out.add(_mime_to_input_type(item))
        return out
    return set()


def infer_tool_input_types(tool: Dict[str, Any]) -> Set[str]:
    """Return the tool's input types, preferring explicit ``input_types``.

    Order: ``input_types`` → singular ``input_type`` / ``input_scope`` →
    infer from ``file_types`` / ``category`` / arg names.
    """
    # Prefer canonical list first (matches registry.get_input_types).
    explicit = _normalise_input_types(tool.get("input_types"))
    if not explicit:
        explicit = _normalise_input_types(tool.get("input_type"))
    if not explicit:
        explicit = _normalise_input_types(tool.get("input_scope"))
    if explicit:
        return explicit

    category = (tool.get("category") or "").lower()
    file_types = tool.get("file_types")
    inferred: Set[str] = set()

    if category == SESSION_INPUT:
        inferred.add(SESSION_INPUT)

    if isinstance(file_types, list) and file_types:
        for ft in file_types:
            if isinstance(ft, str):
                inferred.add(_mime_to_input_type(ft))

    args = _args_dict(tool)
    arg_names = {str(k).lower() for k in args.keys()}
    if {"text", "query", "queries", "question", "prompt"} & arg_names:
        inferred.add(TEXT_INPUT)
    if {"path", "file_path", "filename", "file_id"} & arg_names:
        inferred.add(FILE_INPUT)

    if category in {"graph", "search", "entity", "general"}:
        inferred.add(TEXT_INPUT)
    elif category in {"image", "video", "audio", "text", "file"}:
        inferred.add(_mime_to_input_type(category))

    return inferred or {TEXT_INPUT}


def _file_input_types(file_item: Dict[str, Any]) -> Set[str]:
    mime = file_item.get("mime_type") or file_item.get("content_type") or ""
    ftype = file_item.get("type") or ""
    path = file_item.get("path") or file_item.get("name") or file_item.get("filename") or ""
    if not mime and path:
        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
    out = {_mime_to_input_type(str(v)) for v in (mime, ftype) if v}
    # A concrete visual/audio/document file is also a generic file input.
    out.add(FILE_INPUT)
    return out


def _context_input_types(
    current_file_context: Optional[Dict[str, Any]],
    session_file_registry: Optional[List[Dict[str, Any]]],
    discourse_focus: Optional[Dict[str, Any]] = None,
) -> Set[str]:
    out: Set[str] = set()
    if current_file_context:
        files = current_file_context.get("all_files") or [current_file_context]
        for f in files:
            if isinstance(f, dict):
                out |= _file_input_types(f)
    for f in session_file_registry or []:
        if isinstance(f, dict):
            out |= _file_input_types(f)
    # Narrow tool exposure when discourse focus points at a specific file type.
    if isinstance(discourse_focus, dict):
        focus_ids = discourse_focus.get("active_artifact_ids") or []
        if focus_ids and session_file_registry:
            for entry in session_file_registry:
                aid = str(entry.get("artifact_id") or entry.get("id") or "")
                if aid and aid in focus_ids:
                    out |= _file_input_types(entry)
                    break
    return out


def select_llm_tools(
    tools: Iterable[Dict[str, Any]],
    *,
    current_file_context: Optional[Dict[str, Any]] = None,
    session_file_registry: Optional[List[Dict[str, Any]]] = None,
    discourse_focus: Optional[Dict[str, Any]] = None,
    active_plugin: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Select and order the tools that should be rendered to the LLM prompt."""
    context_types = _context_input_types(
        current_file_context, session_file_registry, discourse_focus
    )
    selected: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    def include(tool: Dict[str, Any]) -> None:
        name = _tool_name(tool)
        if name and name not in seen:
            selected.append(tool)
            seen.add(name)

    for tool in tools or []:
        if not isinstance(tool, dict) or not _has_llm_output(tool):
            continue
        input_types = infer_tool_input_types(tool)

        # Session tools are the Cline workspace primitives.
        if SESSION_INPUT in input_types or (tool.get("category") or "").lower() == "session":
            include(tool)
            continue

        # Text-input LLM tools are always general.
        if TEXT_INPUT in input_types:
            include(tool)
            continue

        # File-specific tools are exposed only when matching files exist in
        # the current turn or session registry.
        if context_types and input_types & context_types:
            include(tool)
            continue

    return _apply_plugin_scope(selected, active_plugin)


def _plugin_of(tool: Dict[str, Any]) -> str:
    """The plugin a tool belongs to — ``plugin_id`` as the tool-runner serves it."""
    return str(tool.get("plugin_id") or "").strip()


def _is_plugin_tool(tool: Dict[str, Any]) -> bool:
    return bool(_plugin_of(tool))


def _mention_unlocked(tool: Dict[str, Any], message: str) -> Optional[bool]:
    """``None`` when the tool is not mention-gated, else whether the user mentioned it."""
    words = tool_intents.mention_gate(tool)
    if words is None:
        return None
    return tool_intents.mentioned(message, words)


def select_llm_tools_for_turn(
    tools: Iterable[Dict[str, Any]],
    *,
    user_message: str = "",
    session_intent: Optional[str] = None,
    current_file_context: Optional[Dict[str, Any]] = None,
    session_file_registry: Optional[List[Dict[str, Any]]] = None,
    discourse_focus: Optional[Dict[str, Any]] = None,
    intent_filter_enabled: Optional[bool] = None,
    active_plugin: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Narrow tools by inferred session intent (read/search/attach/list/general)."""
    from ..config import settings
    from ..session_intent import infer_session_intent

    intent = session_intent or infer_session_intent(
        user_message,
        file_registry=session_file_registry,
        discourse_focus=discourse_focus,
    )
    use_filter = (
        settings.SESSION_INTENT_TOOL_FILTER
        if intent_filter_enabled is None
        else intent_filter_enabled
    )
    if not use_filter or intent == "general":
        return select_llm_tools(
            tools,
            current_file_context=current_file_context,
            session_file_registry=session_file_registry,
            discourse_focus=discourse_focus,
            active_plugin=active_plugin,
        )

    if intent == "graph":
        graph_rule = tool_intents.intents().get("graph_tools")
        eligible = [t for t in tools or [] if isinstance(t, dict) and _has_llm_output(t)]
        scoped = _apply_plugin_scope([t for t in eligible if tool_intents.matches(t, graph_rule)], active_plugin)
        if scoped:
            return scoped
        return _apply_plugin_scope(
            [t for t in eligible if _mention_unlocked(t, user_message)], active_plugin
        )

    session_caps = tool_intents.session_capabilities_for(intent)
    discover_rule = tool_intents.intents().get("discover_tools")

    context_types = _context_input_types(
        current_file_context, session_file_registry, discourse_focus
    )
    selected: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    registry = session_file_registry or []

    def include(tool: Dict[str, Any]) -> None:
        name = _tool_name(tool)
        if name and name not in seen:
            selected.append(tool)
            seen.add(name)

    for tool in tools or []:
        if not isinstance(tool, dict) or not _has_llm_output(tool):
            continue

        if (tool.get("category") or "").lower() == SESSION_INPUT or infer_tool_input_types(tool) & {SESSION_INPUT}:
            if session_caps is not None and not (session_caps & tool_intents.capabilities(tool)):
                continue
            if len(registry) >= tool_intents.session_min_files(intent, tool):
                include(tool)
            continue

        unlocked = _mention_unlocked(tool, user_message)
        if unlocked is not None:
            if unlocked:
                include(tool)
            continue

        if intent == "discover" and tool_intents.matches(tool, discover_rule):
            include(tool)
            continue

        if context_types and infer_tool_input_types(tool) & context_types:
            include(tool)

    return _apply_plugin_scope(selected, active_plugin)


def _apply_plugin_scope(
    tools: Iterable[Dict[str, Any]],
    plugin_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Gate plugin tools by host-page ``active_plugin``.

    - ``plugin_id`` set: that plugin's tools + core (non-plugin) tools.
    - ``plugin_id`` empty/None (main chat): core tools only; every plugin's tools dropped.
    Never falls back to the unfiltered list.
    """
    pid = (plugin_id or "").strip() or None
    out: List[Dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if _is_plugin_tool(tool):
            if page_allows_tool(tool, pid):
                out.append(tool)
            continue
        if _has_llm_output(tool):
            out.append(tool)
    return out


def page_allows_tool(tool: Dict[str, Any], active_plugin: Optional[str] = None) -> bool:
    """Core tools run on every page; a plugin's tools only on that plugin's page.

    Applies to explicit ``tool_ids`` lists too, so no engine config can pull
    another page's plugin tools into this chat.
    """
    if not isinstance(tool, dict):
        return False
    if not _is_plugin_tool(tool):
        return True
    pid = (active_plugin or "").strip()
    return bool(pid) and _plugin_of(tool) == pid


def selection_debug_summary(tools: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Small serialisable summary for logs/tests."""
    out = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        out.append(
            {
                "name": _tool_name(tool),
                "category": tool.get("category"),
                "input_types": sorted(infer_tool_input_types(tool)),
                "llm": _has_llm_output(tool),
            }
        )
    return out
