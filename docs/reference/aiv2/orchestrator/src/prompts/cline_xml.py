"""
Cline-style XML tool-calling protocol for Qwen3 + AIv2.

Why this file exists
--------------------
Native OpenAI ``tool_calls`` over Qwen3.x + the local vLLM gateway has been
unreliable in production:

* The upstream chat template + ``--tool-call-parser qwen3_coder`` combination
  routinely emits the call inside ```` ```json ```` instead of ``delta.tool_calls``.
* When an assistant message reaches the next round with ``tool_calls`` set
  but only whitespace in ``content``, the gateway closes the stream with
  ``RemoteProtocolError: peer closed connection without sending complete message body``.
* The model sometimes invents keys (``tool_name`` vs ``name``, ``arguments``
  vs ``args``, lists vs objects), forcing an ever-growing pile of text-recovery
  parsers in ``gateway_llm_client.py``.

Cline (https://github.com/cline/cline) sidesteps the entire wire-format problem
by treating tool use as **plain text inside an XML envelope**:

    <tool_name>
    <param_a>value</param_a>
    <param_b>value</param_b>
    </tool_name>

The model never touches ``tools=`` on the OpenAI API; the orchestrator parses
XML out of the streamed content. There is exactly one wire protocol, and it
is the same wire protocol no matter which backend (Anthropic, OpenAI, vLLM,
Ollama) is in use.

This file implements that contract for AIv2:

* ``render_tool_xml_block(tool_meta)``: turn one ``tool.json`` dict into the
  XML "## tool_name" block that goes in the system prompt.
* ``build_cline_system_prompt(...)``: assemble the full system prompt
  (tool-use rules, tool catalog, multimodal note, rules, environment
  details, capabilities) the way ``Cline.ts`` does.
* ``ATTEMPT_COMPLETION_NAME`` / ``THINKING_TAG``: shared constants used by
  the streaming parser in ``graph/cline_parser.py``.

The accompanying parser is in ``graph/cline_parser.py``; the orchestrator
loop wiring lives in ``graph/streaming.py``. This file is pure rendering —
no I/O, no networking.
"""
from __future__ import annotations

import datetime as _dt
import re as _re
from typing import Any, Dict, Iterable, List, Optional


THINKING_TAG = "thinking"
ATTEMPT_COMPLETION_NAME = "attempt_completion"


# ---------------------------------------------------------------------------
# Argument-spec parsing
# ---------------------------------------------------------------------------
#
# ``tool.json`` declares parameters as a string of the form
# ``"queries": "list[string] (required) - description ..."``. We need three
# pieces from that: declared type (string|integer|number|boolean|array), the
# ``(required)`` marker, and the human description. The same data flows back
# out to the LLM as the Parameters bullet list.

_REQUIRED_RE = _re.compile(r"\(required\)", _re.IGNORECASE)
_DEFAULT_RE = _re.compile(r"\(\s*optional[^)]*default\s*=\s*([^)]+?)\)", _re.IGNORECASE)


def _parse_arg_spec(spec: Any) -> Dict[str, Any]:
    """Return ``{type, required, default, description}`` for one tool arg.

    Accepts either the legacy string format used by AIv2 tool.json files,
    or a pre-shaped dict (in case a tool already declares JSON-schema).
    Unknown shapes default to a permissive string parameter so we never
    drop a tool the model might need.
    """
    if isinstance(spec, dict):
        return {
            "type": str(spec.get("type") or "string"),
            "required": bool(spec.get("required")),
            "default": spec.get("default"),
            "description": str(spec.get("description") or ""),
        }
    if not isinstance(spec, str):
        return {"type": "string", "required": False, "default": None, "description": ""}

    s = spec.strip()
    low = s.lower()

    # Order matters: ``list[...]`` / ``array`` must be checked before
    # ``string`` because "list[string]" contains the substring "string".
    if low.startswith("list[") or low.startswith("array"):
        arg_type = "array"
    elif "int" in low and "string" not in low.split("(")[0]:
        arg_type = "integer"
    elif "float" in low or "number" in low:
        arg_type = "number"
    elif "bool" in low:
        arg_type = "boolean"
    elif "object" in low or "dict" in low or "json" in low:
        arg_type = "object"
    else:
        arg_type = "string"

    required = bool(_REQUIRED_RE.search(s))
    default_match = _DEFAULT_RE.search(s)
    default = default_match.group(1).strip() if default_match else None

    # Human description — anything after the first " - "/em-dash.
    description = ""
    for sep in (" - ", " — ", ": "):
        if sep in s:
            description = s.split(sep, 1)[1].strip()
            break

    return {
        "type": arg_type,
        "required": required,
        "default": default,
        "description": description,
    }


def _tool_args_with_required(tool_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Return arg specs from either legacy ``args`` or JSON-schema ``args_schema``.

    Plugin manifests commonly use ``args_schema``. If we ignore that shape the
    prompt renders a tool with no parameter docs, which encourages malformed
    calls such as ``<plugin_graph_explorer_graph_search></...>``.
    """
    args_raw = tool_meta.get("args") or tool_meta.get("arguments")
    if isinstance(args_raw, dict):
        return args_raw

    schema = tool_meta.get("args_schema")
    if not isinstance(schema, dict):
        return {}
    props = schema.get("properties")
    if not isinstance(props, dict):
        params = tool_meta.get("parameters")
        props = params.get("properties") if isinstance(params, dict) else None
        if not isinstance(props, dict):
            return {}
        required = set(params.get("required") or [])
        out: Dict[str, Any] = {}
        for name, spec in props.items():
            if isinstance(spec, dict):
                spec = dict(spec)
                spec["required"] = name in required or bool(spec.get("required"))
                out[name] = spec
            else:
                out[name] = spec
        return out
    required = set(schema.get("required") or [])
    out: Dict[str, Any] = {}
    for name, spec in props.items():
        if isinstance(spec, dict):
            spec = dict(spec)
            spec["required"] = name in required or bool(spec.get("required"))
            out[name] = spec
        else:
            out[name] = spec
    return out


def _render_param_line(name: str, parsed: Dict[str, Any]) -> str:
    """Render one ``- name: (required) description`` bullet for the catalog."""
    marker_parts: List[str] = []
    marker_parts.append("required" if parsed.get("required") else "optional")
    arg_type = parsed.get("type") or "string"
    if arg_type and arg_type != "string":
        marker_parts.append(arg_type)
    default = parsed.get("default")
    if default is not None and not parsed.get("required"):
        marker_parts.append(f"default={default}")
    marker = ", ".join(marker_parts)
    description = parsed.get("description") or ""
    if description:
        return f"- {name}: ({marker}) {description}"
    return f"- {name}: ({marker})"


def _example_value_for(
    parsed: Dict[str, Any],
    name: str,
    *,
    tool_name: str = "",
) -> str:
    """Pick a realistic example value for the Usage block of a tool."""
    arg_type = parsed.get("type")
    if arg_type == "array":
        return '["example"]'
    if arg_type == "integer":
        return "10"
    if arg_type == "number":
        return "0.5"
    if arg_type == "boolean":
        return "true"
    if arg_type == "object":
        return "{}"
    lname = name.lower()
    tname = (tool_name or "").lower()
    if "session" in lname and "id" in lname:
        return "Session ID here"
    if "file" in lname and "id" in lname:
        return "File ID here"
    if "artifact" in lname and "id" in lname:
        return "artifact-uuid-here"
    if "query" in lname or "question" in lname:
        return "Search text here"
    if lname in {"path", "storage_path", "file_path"}:
        return "/path/to/file.ext"
    if lname == "case_id":
        return "case-id-here"
    if lname in {"filename", "name"}:
        return "filename.ext"
    return "value here"


def _usage_params_for_tool(
    tool_meta: Dict[str, Any],
    parsed_params: List[tuple],
) -> List[tuple]:
    """Choose which parameters appear in the Usage XML example."""
    explicit = tool_meta.get("usage_params")
    if isinstance(explicit, list) and explicit:
        by_name = {n: p for n, p in parsed_params}
        return [(str(k), by_name[k]) for k in explicit if str(k) in by_name]

    required = [(n, p) for n, p in parsed_params if p.get("required")]
    if required:
        return required
    return parsed_params[:2]


def render_tool_xml_block(tool_meta: Dict[str, Any]) -> str:
    """Return the Markdown block that documents one tool in the prompt.

    Output shape matches Cline's prompts/system.ts exactly enough that
    Qwen3 generalises from its training distribution:

        ## graph_search
        Description: ...
        Parameters:
        - queries: (required, array) one or multiple topics
        - question: (optional) ...
        Usage:
        <graph_search>
        <queries>["example"]</queries>
        </graph_search>
    """
    # Use the executable id as the XML tag. Plugin manifests may have a
    # shorter display ``name`` (e.g. graph_explorer.graph_search), but
    # tool-runner executes by full id (plugin_graph_explorer_graph_search).
    name = tool_meta.get("id") or tool_meta.get("name") or "unknown_tool"
    description = (tool_meta.get("description") or "").strip()
    llm_hint = (tool_meta.get("llm_hint") or "").strip()

    # ``args`` from tool.json is a dict mapping param-name to spec. Plugin
    # manifests may instead use JSON-schema ``args_schema``.
    args_raw = _tool_args_with_required(tool_meta)

    parsed_params: List[tuple] = []  # (name, parsed_dict)
    for pname, pspec in args_raw.items():
        parsed_params.append((str(pname), _parse_arg_spec(pspec)))

    # Sort: required first, then declared order (we already iterate dict
    # in insertion order, so we just do a stable re-sort by required-ness).
    parsed_params.sort(key=lambda kv: 0 if kv[1].get("required") else 1)

    lines: List[str] = []
    lines.append(f"## {name}")
    if description:
        lines.append(f"Description: {description}")
    if llm_hint:
        lines.append(f"Hint: {llm_hint}")

    not_for = tool_meta.get("not_for")
    if isinstance(not_for, str) and not_for.strip():
        lines.append(f"Not for: {not_for.strip()}")
    elif isinstance(not_for, list):
        joined = "; ".join(str(x).strip() for x in not_for if str(x).strip())
        if joined:
            lines.append(f"Not for: {joined}")

    usage_example = tool_meta.get("usage_example")
    if parsed_params:
        lines.append("Parameters:")
        for pname, parsed in parsed_params:
            lines.append(_render_param_line(pname, parsed))
        lines.append("Usage:")
        lines.append(f"<{name}>")
        if isinstance(usage_example, dict) and usage_example:
            for pname, example_val in usage_example.items():
                lines.append(f"<{pname}>{example_val}</{pname}>")
        else:
            usage_params = _usage_params_for_tool(tool_meta, parsed_params)
            for pname, parsed in usage_params:
                lines.append(
                    f"<{pname}>{_example_value_for(parsed, pname, tool_name=name)}</{pname}>"
                )
        lines.append(f"</{name}>")
    else:
        lines.append("Usage:")
        lines.append(f"<{name}>")
        lines.append(f"</{name}>")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in pseudo-tool: attempt_completion
# ---------------------------------------------------------------------------
#
# Cline always ships ``attempt_completion`` as the only way for the model to
# signal that the task is done. The orchestrator does not "execute" it; it
# just closes the loop and surfaces ``<result>`` to the user.

ATTEMPT_COMPLETION_BLOCK = """## attempt_completion
Description: After you have finished the user's request — either by gathering
all required information, performing the work yourself, or determining that no
further tool calls would help — use this to deliver the final answer. Once you
send this block, the conversation turn ends and the user sees ``<result>``.
Parameters:
- result: (required) The final, complete answer to deliver to the user. It
  must stand on its own; do not ask a follow-up question here.
Usage:
<attempt_completion>
<result>
Your final answer for the user, in their language.
</result>
</attempt_completion>"""


# ---------------------------------------------------------------------------
# Top-level system prompt builder
# ---------------------------------------------------------------------------

_PROMPT_HEADER = """You are AIv2, a highly skilled assistant operating inside a session-scoped workspace. You can search a knowledge graph, read uploaded files, analyse images and audio, and write detailed answers in the user's language."""

_TOOL_USE_RULES = """====

TOOL USE

You have access to a set of tools that the orchestrator executes for you on each round of the conversation. **You may use exactly one tool per assistant message.** After you emit a tool call, stop generating; the orchestrator will run it and return the result in the next user message. Use that result to inform what you do next.

# Tool Use Formatting

Tool calls are XML blocks. The tag name is the tool name; each parameter is a nested tag whose body is the value. Example:

<some_tool>
<param_one>value one</param_one>
<param_two>value two</param_two>
</some_tool>

Rules:
- The tag name MUST match a tool listed below — never invent one.
- Parameter values are passed verbatim; for arrays/objects, pass JSON text (e.g. `<queries>["A","B"]</queries>`).
- For file tools, never invent absolute paths. If a tool asks for `path`, pass only the filename shown in the session files (for example `<path>photo.jpg</path>`). The orchestrator resolves that filename to the real session artifact path.
- For `session_read_file`, you may pass `<filename>exact-name.txt</filename>` directly when the name is already visible in SESSION ENVIRONMENT — no prior `session_files` needed.
- For search inside this chat's files, use `session_search_workspace` with `<query>…</query>` (default mode=semantic; use mode=keyword for literal phrase, not regex). Global/case meaning search → `text_search`.
- Before calling a tool, read its **Description**, **Hint**, and **Not for** lines in the catalog below — each tool documents when to use it and which other tool to prefer.
- For text files, use `start_line`/`end_line` to read a range, or `query` on `session_read_file` to search inside one file and receive compact snippets.
- Do NOT wrap the XML inside a code fence. No ```` ```xml ```` , no ```` ```json ```` .
- Do NOT add extra prose **after** the closing tag in the same message. The orchestrator stops reading at `</tool_name>`.
- If you have no tool to call this turn, finish the task with `<attempt_completion>` instead.

# Thinking

Before you commit to a tool, you may write a short internal thought:

<thinking>
What does the user actually want? Do I have enough info? Which tool advances the task most?
</thinking>

The user does not see `<thinking>` blocks. Keep them concise (1–4 lines)."""

_RULES_BLOCK = """====

RULES

- Match the user's language exactly. If they wrote Persian, answer in Persian; English in English; mixed → dominant language of the latest message.
- The user's latest message is the priority. Use chat history only as context — never repeat earlier answers.
- Iterate: emit one tool, wait for the result, decide what to do, then either call another tool or finish with `<attempt_completion>`.
- Do not stop short. If a tool fails or returns nothing useful, try a different argument, a different tool, or read more session files before giving up.
- Do not ask the user for information you can fetch with a tool yourself.
- Inside `<result>` quote concrete evidence from tool outputs (numbers, file ids, names); never invent data.
- If you genuinely cannot fulfill the request after honest effort, say so explicitly in `<result>` and state what is missing."""

_MULTIMODAL_NOTE = """====

VISION (native multimodal)

When the user attaches an image or video to a turn, the orchestrator embeds the actual pixels in your `user` message as `image_url` / `video_url` parts. You can SEE the media directly — describe, OCR/read visible text, count, compare, answer spatial questions. **Do not call a tool just to "view" what is in the picture or read text visible in it**; describe/OCR it from your own perception. `session_read_file` is only for text files or previously extracted OCR/transcripts/metadata, not for reading pixels from JPG/PNG/MP4. Tools like `vision_info` only refresh search/index metadata and do not give you a richer view of pixels."""


def _format_env_details(env_details: Dict[str, Any]) -> str:
    """Render the per-turn environment block (files, time, session id)."""
    lines: List[str] = ["====", "", "SESSION ENVIRONMENT", ""]

    sid = env_details.get("session_id") or env_details.get("chat_session_id")
    if sid:
        lines.append(f"# Session ID")
        lines.append(str(sid))
        lines.append("")

    files = env_details.get("session_files") or []
    if files:
        lines.append(f"# Session Files ({len(files)})")
        lines.append("Format: `file_id | name | type | size_bytes`")
        for f in files[:80]:
            fid = f.get("artifact_id") or f.get("id") or f.get("file_id") or "?"
            fname = f.get("name") or f.get("filename") or "?"
            ftype = f.get("type") or f.get("mime_type") or "?"
            fsize = f.get("size_bytes") or f.get("size") or "?"
            lines.append(f"- {fid} | {fname} | {ftype} | {fsize}")
        if len(files) > 80:
            lines.append(f"(+ {len(files) - 80} more — call `session_files` for the full list.)")
        lines.append("")
    else:
        lines.append("# Session Files")
        lines.append("(no files uploaded in this session)")
        lines.append("")

    now = env_details.get("current_time") or _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    lines.append("# Current Time")
    lines.append(str(now))
    lines.append("")

    extras = env_details.get("extras")
    if isinstance(extras, dict) and extras:
        lines.append("# Additional Context")
        for k, v in extras.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    return "\n".join(lines).rstrip()


def build_cline_system_prompt(
    tools: Iterable[Dict[str, Any]],
    env_details: Optional[Dict[str, Any]] = None,
    *,
    multimodal_enabled: bool = True,
    extra_rules: Optional[str] = None,
    rag_context: Optional[List[Dict[str, Any]]] = None,
    rag_prompt_block: Optional[str] = None,
) -> str:
    """Assemble the full Cline-style system prompt.

    Parameters
    ----------
    tools
        Iterable of ``tool.json``-shaped dicts (must at least include
        ``name`` and ``args``).
    env_details
        Per-turn environment dict — usually built once per request from
        the storage service. May be ``None`` on the very first call.
    multimodal_enabled
        Include the VISION section. Turn off only for text-only deployments.
    extra_rules
        Project-specific rules to append after the standard RULES block
        (e.g. case-tenant data isolation reminders).
    rag_context / rag_prompt_block
        Optional native corpus RAG hits (or a preformatted block) to inject.
    """
    sections: List[str] = [_PROMPT_HEADER, _TOOL_USE_RULES]

    tool_blocks: List[str] = []
    ordered_tools = _order_tools_for_prompt(list(tools or []))
    for tm in ordered_tools:
        if not isinstance(tm, dict):
            continue
        try:
            tool_blocks.append(render_tool_xml_block(tm))
        except Exception:
            # A malformed manifest entry should never poison the whole
            # prompt. Silently skip and continue.
            continue
    tool_blocks.append(ATTEMPT_COMPLETION_BLOCK)

    sections.append("# Tools\n\n" + "\n\n".join(tool_blocks))

    if multimodal_enabled:
        sections.append(_MULTIMODAL_NOTE)

    sections.append(_RULES_BLOCK)
    if extra_rules:
        sections.append(extra_rules.strip())

    rag_block = (rag_prompt_block or "").strip()
    if not rag_block and rag_context:
        from ..rag import format_rag_prompt_block
        from ..config import settings as _settings

        rag_block = format_rag_prompt_block(
            list(rag_context),
            max_chars_per_hit=int(_settings.NATIVE_RAG_MAX_CHARS_PER_HIT or 1200),
        )
    if rag_block:
        sections.append(rag_block)

    if env_details:
        sections.append(_format_env_details(env_details))

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Convenience for the parser side
# ---------------------------------------------------------------------------


def _order_tools_for_prompt(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stable order: optional ``prompt_order`` in tool.json, then name."""

    def sort_key(tm: Dict[str, Any]) -> tuple:
        order = tm.get("prompt_order")
        try:
            o = int(order) if order is not None else 9999
        except (TypeError, ValueError):
            o = 9999
        name = str(tm.get("id") or tm.get("name") or "")
        return (o, name)

    return sorted(
        [tm for tm in tools if isinstance(tm, dict)],
        key=sort_key,
    )


def known_tool_names(tools: Iterable[Dict[str, Any]]) -> List[str]:
    """Return the canonical names of every tool the model is allowed to emit.

    The parser uses this set to validate opening tags, so a typo by the
    model (``<vision_inf0>``) is rejected at parse time instead of after
    a round-trip to ``tool-runner``.
    """
    names: List[str] = []
    for tm in tools or []:
        if not isinstance(tm, dict):
            continue
        n = tm.get("id") or tm.get("name")
        if n:
            names.append(str(n))
    names.append(ATTEMPT_COMPLETION_NAME)
    return names
