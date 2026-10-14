"""Smoke test for the Cline XML prompt + streaming parser.

Run with::

    cd services/orchestrator && python -m src.prompts._smoke_cline_xml

Or via the existing test harness::

    pytest services/orchestrator/src/prompts/_smoke_cline_xml.py -q

The point is to catch obvious regressions in argument typing, multi-chunk
streaming, and the attempt_completion happy-path before shipping a new
image to k3s.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

# Load the two modules under test directly from disk to avoid pulling in
# the rest of the orchestrator package (langgraph, redis, etc.). This lets
# us validate the cline-xml protocol locally even without a full venv.
import importlib.util as _ilu

_HERE = Path(__file__).resolve()
_PROMPT_PATH = _HERE.parent / "cline_xml.py"
_PARSER_PATH = _HERE.parent.parent / "graph" / "cline_parser.py"


def _load(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cline_xml = _load("_cline_xml_smoke", _PROMPT_PATH)
cline_parser = _load("_cline_parser_smoke", _PARSER_PATH)

build_cline_system_prompt = cline_xml.build_cline_system_prompt
known_tool_names = cline_xml.known_tool_names
render_tool_xml_block = cline_xml.render_tool_xml_block
ClineXmlStreamParser = cline_parser.ClineXmlStreamParser
build_param_types_index = cline_parser.build_param_types_index


_FAKE_TOOLS = [
    {
        "name": "graph_search",
        "description": "Find graph relationships.",
        "args": {
            "queries": "list[string] (required) - one or multiple topics",
            "max_hops": "int (optional, default=6) - graph hops",
        },
        "llm_hint": "Always pass an array.",
    },
    {
        "name": "session_files",
        "description": "List uploaded files in the current session.",
        "args": {
            "session_id": "string (required) - the chat session UUID",
            "limit": "int (optional, default=200)",
        },
    },
    {
        "name": "image_meta",
        "description": "Extract image metadata.",
        "args": {
            "session_id": "string (required)",
            "file_id": "string (required) - artifact id from session_files",
        },
    },
]


def _check(name: str, cond: bool, detail: str = "") -> None:
    status = "✅" if cond else "❌"
    print(f"{status} {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        sys.exit(1)


def test_prompt_contains_every_tool() -> None:
    prompt = build_cline_system_prompt(
        _FAKE_TOOLS,
        env_details={"session_id": "abc", "session_files": [], "current_time": "2026-05-13"},
    )
    for tm in _FAKE_TOOLS:
        n = tm["name"]
        _check(f"prompt has '## {n}'", f"## {n}" in prompt)
        _check(f"prompt has <{n}> usage", f"<{n}>" in prompt and f"</{n}>" in prompt)
    _check("prompt has attempt_completion", "<attempt_completion>" in prompt)
    _check("prompt has VISION section", "VISION" in prompt)


def test_render_required_marker() -> None:
    block = render_tool_xml_block(_FAKE_TOOLS[0])
    _check("required marker rendered", "(required, array)" in block, block)
    _check("default rendered", "default=6" in block, block)


def test_parser_simple_tool_call() -> None:
    parser = ClineXmlStreamParser(
        tool_names=known_tool_names(_FAKE_TOOLS),
        param_types_by_tool=build_param_types_index(_FAKE_TOOLS),
    )
    stream = textwrap.dedent(
        """
        I'll search the graph.
        <graph_search>
        <queries>["Jafar","Michael"]</queries>
        <max_hops>4</max_hops>
        </graph_search>
        trailing noise that should be ignored
        """
    ).strip()
    events = []
    # Feed in 5-character chunks to exercise the cross-chunk paths.
    for i in range(0, len(stream), 5):
        events.extend(parser.feed(stream[i : i + 5]))
    events.extend(parser.flush())

    text_chunks = [e.text for e in events if e.kind == "text_delta"]
    tool_calls = [e for e in events if e.kind == "tool_call"]
    _check("got at least one text chunk", bool(text_chunks))
    _check("got exactly one tool_call", len(tool_calls) == 1, str(len(tool_calls)))
    tc = tool_calls[0]
    _check("tool name is graph_search", tc.tool_name == "graph_search", tc.tool_name)
    _check("queries parsed as list", isinstance(tc.arguments.get("queries"), list))
    _check(
        "queries values correct",
        tc.arguments.get("queries") == ["Jafar", "Michael"],
        str(tc.arguments.get("queries")),
    )
    _check(
        "max_hops parsed as int",
        tc.arguments.get("max_hops") == 4,
        str(tc.arguments.get("max_hops")),
    )


def test_parser_thinking_and_attempt_completion() -> None:
    parser = ClineXmlStreamParser(
        tool_names=known_tool_names(_FAKE_TOOLS),
        param_types_by_tool=build_param_types_index(_FAKE_TOOLS),
    )
    stream = (
        "<thinking>quick plan</thinking>"
        "<attempt_completion>"
        "<result>کارت تمومه — ۲ فایل دیدم.</result>"
        "</attempt_completion>"
    )
    events = []
    for i in range(0, len(stream), 3):
        events.extend(parser.feed(stream[i : i + 3]))
    events.extend(parser.flush())

    kinds = [e.kind for e in events]
    _check("thinking emitted", "thinking_delta" in kinds, str(kinds))
    completions = [e for e in events if e.kind == "attempt_completion"]
    _check("attempt_completion emitted", len(completions) == 1, str(len(completions)))
    _check(
        "result text correct",
        completions[0].result == "کارت تمومه — ۲ فایل دیدم.",
        completions[0].result,
    )


def test_parser_unknown_tag_becomes_parse_error() -> None:
    parser = ClineXmlStreamParser(
        tool_names=known_tool_names(_FAKE_TOOLS),
        param_types_by_tool=build_param_types_index(_FAKE_TOOLS),
    )
    stream = "before <not_a_tool>x</not_a_tool> after"
    events = []
    events.extend(parser.feed(stream))
    events.extend(parser.flush())
    joined_text = "".join(e.text for e in events if e.kind == "text_delta")
    _check(
        "unknown tag prefix surfaced as text",
        "before" in joined_text and "not_a_tool" not in joined_text,
        joined_text,
    )
    _check(
        "unknown tag became parse_error",
        any(e.kind == "parse_error" and "unknown_tool:not_a_tool" in e.text for e in events),
        str([(e.kind, e.text) for e in events]),
    )
    _check(
        "no tool_call from unknown tag",
        all(e.kind != "tool_call" for e in events),
    )


def test_parser_single_tool_per_turn() -> None:
    """Cline contract: after a tool_call we drop everything else."""
    parser = ClineXmlStreamParser(
        tool_names=known_tool_names(_FAKE_TOOLS),
        param_types_by_tool=build_param_types_index(_FAKE_TOOLS),
    )
    stream = (
        "<session_files><session_id>abc</session_id></session_files>"
        "<image_meta><session_id>abc</session_id><file_id>fid</file_id></image_meta>"
    )
    events = list(parser.feed(stream)) + list(parser.flush())
    tool_calls = [e for e in events if e.kind == "tool_call"]
    _check(
        "exactly one tool emitted despite two tags",
        len(tool_calls) == 1,
        str([t.tool_name for t in tool_calls]),
    )
    _check(
        "first tool was the one emitted",
        tool_calls[0].tool_name == "session_files",
        tool_calls[0].tool_name,
    )


if __name__ == "__main__":
    test_prompt_contains_every_tool()
    test_render_required_marker()
    test_parser_simple_tool_call()
    test_parser_thinking_and_attempt_completion()
    test_parser_unknown_tag_becomes_parse_error()
    test_parser_single_tool_per_turn()
    print("\n🎉 cline_xml + cline_parser smoke tests passed.")
