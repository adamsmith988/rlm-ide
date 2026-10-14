"""Compatibility shim — Cline XML chat engine is archived.

Shared helpers are still imported by ``native_tools_engine``. The selectable
chat orchestrator drop-in and ``stream_agent_cline`` chat path are archived
under ``graph/_archived/`` and ``orchestrators/_archived/cline_xml/``.

Note: ``from … import *`` does not re-export ``_private`` names, so helpers
used by native_tools are listed explicitly below.
"""

from __future__ import annotations

from ._archived.cline_engine import (  # noqa: F401
    stream_agent_cline,
    _append_registry_entry,
    _build_environment_details,
    _canonical_tool_call_key,
    _cap_llm_text,
    _detect_file_context_from_registry,
    _display_query_from_args,
    _evidence_files_from_context,
    _evidence_files_from_tool_args,
    _evidence_record,
    _extract_attach_file_from_result,
    _extract_llm_summary,
    _fallback_final_answer_from_tool_runs,
    _final_answer_for_missing_visual_attachment,
    _format_tool_result_message,
    _is_visual_registry_entry,
    _match_registry_entry_from_tool_args,
    _missing_visual_attachment_notice,
    _normalized_tool_summary,
    _registry_entry_from_attach,
    _slim_resolution_for_export,
    _summarize_evidence_for_done,
    _tool_result_sse_event,
    _tool_runs_digest_for_llm,
    _visual_attachment_missing,
    _visual_file_context_keys,
)

__all__ = [
    "stream_agent_cline",
    "_append_registry_entry",
    "_build_environment_details",
    "_canonical_tool_call_key",
    "_cap_llm_text",
    "_detect_file_context_from_registry",
    "_display_query_from_args",
    "_evidence_files_from_context",
    "_evidence_files_from_tool_args",
    "_evidence_record",
    "_extract_attach_file_from_result",
    "_extract_llm_summary",
    "_fallback_final_answer_from_tool_runs",
    "_final_answer_for_missing_visual_attachment",
    "_format_tool_result_message",
    "_is_visual_registry_entry",
    "_match_registry_entry_from_tool_args",
    "_missing_visual_attachment_notice",
    "_normalized_tool_summary",
    "_registry_entry_from_attach",
    "_slim_resolution_for_export",
    "_summarize_evidence_for_done",
    "_tool_result_sse_event",
    "_tool_runs_digest_for_llm",
    "_visual_attachment_missing",
    "_visual_file_context_keys",
]
