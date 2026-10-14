"""Host catalog tools as seen from an engine's code kernel (``tools_binding.mode = kernel_callable``).

One source for the callable alias of a tool id, the namespace it is bound under
(``tools_binding.surface.namespace``) and the prompt section rendered from the
tools' canonical JSON Schemas. Stdlib only: the kernel worker imports it too.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

DEFAULT_NAMESPACE = "tools"
SURFACE_HEADING = "## Project tools"

_NON_IDENT_RE = re.compile(r"[^0-9A-Za-z_]")
_JSON_TYPE_NAMES = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}


def tool_alias(tool_id: Any) -> str:
    """Python attribute name a kernel uses for ``tool_id`` (``namespace.<alias>``)."""
    alias = _NON_IDENT_RE.sub("_", str(tool_id or "").strip())
    if alias and alias[0].isdigit():
        alias = f"t_{alias}"
    return alias


def surface_namespace(surface: Optional[Mapping[str, Any]], default: str = DEFAULT_NAMESPACE) -> str:
    """Kernel name the tools object is bound under, from ``tools_binding.surface``."""
    raw = str((surface or {}).get("namespace") or "").strip() if isinstance(surface, Mapping) else ""
    return raw if raw.isidentifier() else default


def surface_intro(namespace: str, cell_tool: str = "", usage_hint: str = "") -> str:
    """``usage_hint`` is engine-configured guidance; ``{namespace}`` in it becomes the tools object name."""
    where = f"inside a `{cell_tool}` cell " if cell_tool else ""
    hint = str(usage_hint or "").strip()
    hint = hint.replace("{namespace}", namespace) + " " if hint else ""
    return (
        f"Platform tools available in this chat, callable {where}as "
        f"`r = await {namespace}.<name>(arg=value, ...)`. Each returns a dict: check `r.get(\"ok\")`, "
        "then read the fields you need (print `list(r.keys())` if unsure). When one of them does "
        "what the user asked, call it instead of re-implementing that work in code, and base "
        "your answer on what it returned — never invent tool output. "
        f"{hint}"
        f"`await {namespace}.describe(\"<name>\")` returns a tool's full description and parameters; "
        f"`{namespace}.list()` lists the names."
    )


def call_hint(tool_id: str, namespace: str = DEFAULT_NAMESPACE) -> str:
    return f"`r = await {namespace}.{tool_alias(tool_id)}(...)`"


def _param_type(schema: Mapping[str, Any]) -> str:
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        shown = "|".join(repr(v) if isinstance(v, str) else str(v) for v in enum[:4])
        return shown + ("|…" if len(enum) > 4 else "")
    raw = schema.get("type")
    names = raw if isinstance(raw, list) else [raw]
    out = [_JSON_TYPE_NAMES.get(str(n), str(n)) for n in names if n and n != "null"]
    return "|".join(out) or "any"


def tool_signature(parameters: Any, max_params: int = 0) -> str:
    """``a: str, b?: int`` from a JSON Schema object; required first, ``…`` when cut."""
    if not isinstance(parameters, Mapping):
        return ""
    props = parameters.get("properties")
    if not isinstance(props, Mapping):
        return ""
    required = [str(r) for r in parameters.get("required") or [] if str(r) in props]
    order = required + [str(k) for k in props if str(k) not in required]
    shown = order[: max(0, int(max_params))] if max_params > 0 else order
    parts = []
    for name in shown:
        spec = props.get(name) if isinstance(props.get(name), Mapping) else {}
        mark = "" if name in required else "?"
        parts.append(f"{name}{mark}: {_param_type(spec)}")
    if len(order) > len(shown):
        parts.append("…")
    return ", ".join(parts)


def brief(text: Any, limit: int = 0) -> str:
    """First paragraph of a tool description, cut at a sentence end within ``limit``."""
    body = str(text or "").strip()
    for stop in ("\n\n", "\n#"):
        cut = body.find(stop)
        if cut > 0:
            body = body[:cut]
    body = " ".join(body.split())
    if limit <= 0 or len(body) <= limit:
        return body
    head = body[:limit]
    end = max(head.rfind(". "), head.rfind("; "))
    return head[: end + 1] if end >= limit // 3 else head.rstrip() + "…"


def format_tool_surface(
    tool_specs: Optional[Sequence[Mapping[str, Any]]],
    tool_ids: Optional[Sequence[str]],
    *,
    namespace: str = DEFAULT_NAMESPACE,
    cell_tool: str = "",
    description_chars: int = 0,
    max_params: int = 0,
    usage_hint: str = "",
) -> str:
    """Prompt section: one line per allowed tool, rendered from its schema."""
    ids = [str(t).strip() for t in (tool_ids or []) if str(t).strip()]
    if not ids:
        return ""
    by_name: Dict[str, Mapping[str, Any]] = {}
    for spec in tool_specs or []:
        fn = spec.get("function") if isinstance(spec, Mapping) else None
        if isinstance(fn, Mapping) and fn.get("name"):
            by_name[str(fn["name"])] = fn
    lines: List[str] = [SURFACE_HEADING, surface_intro(namespace, cell_tool, usage_hint)]
    for tid in ids:
        fn = by_name.get(tid, {})
        sig = tool_signature(fn.get("parameters"), max_params)
        desc = brief(fn.get("description"), description_chars)
        line = f"- `{namespace}.{tool_alias(tid)}({sig})`"
        lines.append(f"{line} — {desc}" if desc else line)
    return "\n".join(lines)


def replace_tool_surface(
    system_text: str, section: str, *, headings: Sequence[str] = (SURFACE_HEADING,)
) -> str:
    """Swap the tools section of an existing system prompt (append when absent)."""
    text = str(system_text or "")
    for heading in headings:
        start = text.find(heading + "\n")
        if start < 0:
            continue
        end = text.find("\n\n", start)
        text = text[:start] + (text[end + 2 :] if end >= 0 else "")
    text = text.rstrip()
    if section:
        text = f"{text}\n\n{section}" if text else section
    return text


__all__ = [
    "DEFAULT_NAMESPACE",
    "SURFACE_HEADING",
    "brief",
    "call_hint",
    "format_tool_surface",
    "replace_tool_surface",
    "surface_intro",
    "surface_namespace",
    "tool_alias",
    "tool_signature",
]
