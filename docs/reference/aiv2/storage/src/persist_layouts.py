"""Allowlisted plugin-output object-key templates (mirrors modules.tools.persist_profile)."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Tuple

ALLOWED_TEMPLATE_VARS = frozenset(
    {
        "user_id",
        "case_id",
        "session_id",
        "group_id",
        "plugin_id",
        "tool_id",
        "flow_id",
        "record_id",
        "run_id",
        "hex",
        "filename",
        "node_kind",
    }
)

ALLOWED_LAYOUT_PREFIXES = ("case_importer/", "cases/", "users/")

_SESSIONS_LAYOUT_TMPL = (
    "users/{user_id}/plugins/{plugin_id}/sessions/"
    "{record_id}/{run_id}/{hex}_{filename}"
)
DEFAULT_BLOB_LAYOUTS: Dict[str, Dict[str, str]] = {
    "default": {
        "when_case": "case_importer/{case_id}/plugins/{plugin_id}/runs/{run_id}/{hex}_{filename}",
        "when_user": "users/{user_id}/plugins/{plugin_id}/flows/{flow_id}/runs/{run_id}/{hex}_{filename}",
    },
    # Plugin session records — user-scoped, never case evidence (chat-style).
    "sessions": {
        "when_case": _SESSIONS_LAYOUT_TMPL,
        "when_user": _SESSIONS_LAYOUT_TMPL,
    },
}


def is_plugin_session_artifact(
    *,
    original_filename: str = "",
    storage_path: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> bool:
    """True for tool-runner plugin session blobs (exclude from case file pickers)."""
    meta = metadata if isinstance(metadata, Mapping) else {}
    name = str(original_filename or "").strip().rsplit("/", 1)[-1]
    if name == "session.json":
        return True
    kind = str(meta.get("kind") or "").strip()
    if kind == "plugin_session":
        return True
    layout = str(meta.get("layout") or "").strip().lower()
    if layout == "sessions":
        return True
    path = str(storage_path or "").replace("\\", "/")
    if "/plugins/" in path and "/sessions/" in path:
        return True
    return False

_VAR_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def validate_layout_template(template: str) -> Tuple[bool, str]:
    text = (template or "").strip().replace("\\", "/")
    if not text:
        return False, "empty_template"
    if text.startswith("/") or ".." in text.split("/"):
        return False, "unsafe_path"
    if not any(text.startswith(p) for p in ALLOWED_LAYOUT_PREFIXES):
        return False, "prefix_not_allowed"
    for var in _VAR_RE.findall(text):
        if var not in ALLOWED_TEMPLATE_VARS:
            return False, f"unknown_var:{var}"
    return True, ""


def render_layout_template(template: str, values: Mapping[str, Any]) -> str:
    ok, err = validate_layout_template(template)
    if not ok:
        raise ValueError(f"invalid_layout_template:{err}")

    def _repl(match: re.Match) -> str:
        key = match.group(1)
        raw = values.get(key)
        text = str(raw if raw is not None else "").strip()
        text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("._-")
        return text or "_"

    rendered = _VAR_RE.sub(_repl, template).replace("//", "/").strip("/")
    if ".." in rendered.split("/") or rendered.startswith("/"):
        raise ValueError("rendered_layout_unsafe")
    if not any(rendered.startswith(p) for p in ALLOWED_LAYOUT_PREFIXES):
        raise ValueError("rendered_layout_prefix")
    return rendered


def resolve_layout_template(
    *,
    layout_name: Optional[str],
    layout_template: Optional[str],
    has_case: bool,
) -> str:
    if layout_template and str(layout_template).strip():
        tmpl = str(layout_template).strip()
        ok, err = validate_layout_template(tmpl)
        if not ok:
            raise ValueError(f"layout_template:{err}")
        return tmpl
    name = (layout_name or "default").strip() or "default"
    layout = DEFAULT_BLOB_LAYOUTS.get(name) or DEFAULT_BLOB_LAYOUTS["default"]
    key = "when_case" if has_case else "when_user"
    return str(layout[key])
