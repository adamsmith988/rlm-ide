"""
Repair and validate LLM tool call arguments before execution.

Models sometimes emit ``{"session_search_workspace": "[\\"token\\"]"}`` instead of
``{"query": "token"}``. This module normalizes those shapes so remapped tools
(e.g. session_read scope=session) receive required fields.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# Primary parameter when the model uses the tool name as the only JSON key.
_TOOL_NAME_AS_KEY_TARGET: Dict[str, str] = {
    "session_search_workspace": "query",
    "session_read_file": "query",
    "text_search": "text",
}

# Required fields after repair (minimal gate before execute).
_REQUIRED_AFTER_REPAIR: Dict[str, List[str]] = {
    "session_search_workspace": ["query"],
    "graph_search": ["queries"],
    "text_search": ["text"],
    "session_attach": [],  # any of filename/case_id/path/artifact_id checked in tool
    "session_read_file": [],  # file_id or filename — tool validates
}


def _unwrap_scalar(value: Any) -> Any:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return value
    if s.startswith("[") or s.startswith("{"):
        try:
            return json.loads(s)
        except Exception:
            return value
    return value


def repair_tool_arguments(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort fix for common malformed tool argument shapes."""
    out = dict(args or {})
    if not out:
        return out

    keys = [k for k in out.keys() if k]
    if len(keys) == 1 and keys[0] == tool_name:
        raw = _unwrap_scalar(out[tool_name])
        target = _TOOL_NAME_AS_KEY_TARGET.get(tool_name)
        if tool_name == "graph_search":
            if isinstance(raw, list):
                out = {"queries": [str(x) for x in raw if str(x).strip()]}
            elif raw is not None and str(raw).strip():
                out = {"queries": [str(raw).strip()]}
            else:
                out = {}
        elif target:
            if isinstance(raw, list) and raw:
                out = {target: str(raw[0]).strip()}
            elif raw is not None and str(raw).strip():
                out = {target: str(raw).strip()}
            else:
                out = {}
        elif isinstance(raw, dict):
            out = raw

    # graph_search: single string queries → list
    if tool_name == "graph_search":
        q = out.get("queries")
        if isinstance(q, str) and q.strip():
            out["queries"] = [q.strip()]
        elif q is None and out.get("query"):
            out["queries"] = [str(out.pop("query")).strip()]

    # session_search_workspace: alias keys
    if tool_name == "session_search_workspace" and not out.get("query"):
        for alt in ("q", "search", "keyword", "text"):
            if out.get(alt):
                out["query"] = str(out.pop(alt)).strip()
                break
    if tool_name == "session_search_workspace" and out.get("mode"):
        mode = str(out["mode"]).strip().lower()
        if mode in {"grep", "literal", "phrase", "exact"}:
            out["mode"] = "keyword"
        elif mode in {"meaning", "ann", "vector"}:
            out["mode"] = "semantic"

    return out


def validate_tool_arguments(tool_name: str, args: Dict[str, Any]) -> Optional[str]:
    """
    Return a user/LLM-facing error string if required args are still missing.
    None means OK to execute (tool may apply its own validation).
    """
    required = _REQUIRED_AFTER_REPAIR.get(tool_name)
    if required is None:
        return None
    missing = [k for k in required if not args.get(k)]
    if missing:
        hints = ", ".join(f"{k}=..." for k in missing)
        return (
            f"Invalid arguments for `{tool_name}`: missing {', '.join(missing)}. "
            f"Provide {hints}. See the tool description in the API schema."
        )
    if tool_name == "session_attach":
        locators = (
            "filename",
            "name",
            "case_id",
            "artifact_id",
            "path",
            "storage_path",
        )
        if not any(args.get(k) for k in locators):
            return (
                "Invalid arguments for `session_attach`: provide at least one of "
                "filename, case_id, artifact_id, or path."
            )
    if tool_name == "session_read_file":
        locators = ("file_id", "artifact_id", "filename", "name", "path")
        if not any(args.get(k) for k in locators):
            return (
                "Invalid arguments for `session_read_file`: provide file_id or "
                "filename from session_files / session_attach results."
            )
    return None


__all__ = ["repair_tool_arguments", "validate_tool_arguments"]
