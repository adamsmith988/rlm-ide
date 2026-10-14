"""End-to-end smoke test for the Cline engine.

This stubs out vLLM and ToolExecutor so we can exercise the loop without
a running cluster — verifying:

* The first user turn produces a tool_call event.
* The synthetic ``[Tool Result]`` user message is appended.
* The second LLM call sees the tool result and emits ``attempt_completion``.
* The final ``done`` event has the right shape (final_answer, tool_runs, …).

Run with::

    cd services/orchestrator && python -m pytest src/graph/_smoke_cline_engine.py -q

or as a plain script::

    cd services/orchestrator && python src/graph/_smoke_cline_engine.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Bypass the package __init__'s heavyweight imports by loading the engine
# from disk under an isolated module name.
import importlib.util as _ilu

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))  # services/orchestrator


def _stub_module(name: str) -> object:
    """Create an empty stub module."""
    m = type(sys)(name)
    sys.modules[name] = m
    return m


# Stub out the things the engine imports that we do not want to actually run.
import types as _types  # noqa: E402

# Pre-create a 'src' package shim so the relative imports inside the engine
# resolve. We then plug in just the leaf modules the engine actually touches.
_src_pkg = _types.ModuleType("src")
_src_pkg.__path__ = [str(_HERE.parent.parent)]  # services/orchestrator/src
sys.modules["src"] = _src_pkg

# 1. config.settings
config_stub = _stub_module("src.config")
class _Settings:
    MAX_STEPS = 5
config_stub.settings = _Settings()

# 2. llm.gateway_llm_client.get_gateway_llm_client
llm_pkg = _stub_module("src.llm")
llm_pkg.__path__ = []  # mark as package
vllm_stub = _stub_module("src.llm.gateway_llm_client")

class _FakeLLM:
    """Drip-feeds a scripted set of answer chunks per call."""

    def __init__(self):
        self._scripts: list[list[dict]] = []
        self._call_idx = 0

    def queue(self, *scripts: list[dict]) -> None:
        self._scripts.extend(scripts)

    async def chat_stream(self, *, messages, **kwargs):
        script = self._scripts[self._call_idx]
        self._call_idx += 1
        for ev in script:
            yield ev

_fake_llm = _FakeLLM()
vllm_stub.get_gateway_llm_client = lambda: _fake_llm

# 3. tools.executor.get_tool_executor
tools_pkg = _stub_module("src.tools")
tools_pkg.__path__ = []
tools_exec_stub = _stub_module("src.tools.executor")

class _FakeToolExecutor:
    def __init__(self):
        self.last_calls: list[tuple[str, dict]] = []

    async def get_tools_manifest(self, force_refresh: bool = False):
        return [
            {
                "name": "graph_search",
                "description": "Find connections in graph.",
                "args": {"queries": "list[string] (required) - topics"},
                "channels": ["llm"],
                "category": "graph",
                "input_type": "text",
            },
            {
                "name": "session_files",
                "description": "List session files.",
                "args": {"session_id": "string (required)"},
                "channels": ["llm"],
                "category": "session",
                "input_type": "session",
            },
        ]

    async def execute(self, tool_name, args, file_context=None, session_id=None, **kwargs):
        self.last_calls.append((tool_name, args))
        if tool_name == "graph_search":
            return {
                "ok": True,
                "tool_id": tool_name,
                "data": {
                    "result": {
                        "channels": {"llm": "Jafar is connected to Acme Corp via 2 hops."}
                    }
                },
            }
        return {"ok": True, "tool_id": tool_name, "data": {"result": {"channels": {"llm": "(empty)"}}}}

_fake_executor = _FakeToolExecutor()
tools_exec_stub.get_tool_executor = lambda: _fake_executor

# Real tool selector used by the engine.
_load_selector_later = True

# 4. utils.file_context.detect_file_context (returns None — text-only run)
utils_pkg = _stub_module("src.utils")
utils_pkg.__path__ = []
fc_stub = _stub_module("src.utils.file_context")
fc_stub.detect_file_context = lambda artifacts: None

# 5. utils.multimodal.build_multimodal_content
mm_stub = _stub_module("src.utils.multimodal")
async def _build_mm(text, file_context, **kw):
    return text
mm_stub.build_multimodal_content = _build_mm

# 6. Real prompts.cline_xml — load from disk.
prompts_pkg = _stub_module("src.prompts")
prompts_pkg.__path__ = [str(_HERE.parent.parent / "prompts")]

def _load_real(name: str, path: Path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_load_real("src.prompts.cline_xml", _HERE.parent.parent / "prompts" / "cline_xml.py")
_load_real("src.tools.llm_tool_selection", _HERE.parent.parent / "tools" / "llm_tool_selection.py")

# 7. Real graph.cline_parser
graph_pkg = _stub_module("src.graph")
graph_pkg.__path__ = [str(_HERE.parent)]
_load_real("src.graph.cline_parser", _HERE.parent / "cline_parser.py")

# 8. Now load the engine under test.
cline_engine = _load_real("src.graph.cline_engine", _HERE.parent / "cline_engine.py")


# ---------------------------------------------------------------------------
# Scripted scenario
# ---------------------------------------------------------------------------


def _check(name: str, cond: bool, detail: str = "") -> None:
    status = "✅" if cond else "❌"
    print(f"{status} {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        sys.exit(1)


def _split(s: str, n: int = 8):
    """Helper to split a string into n-char streaming chunks (as answer events)."""
    return [{"type": "answer", "content": s[i : i + n]} for i in range(0, len(s), n)]


async def main():
    turn1 = _split(
        "<thinking>Plan: search the graph for Jafar.</thinking>"
        "<graph_search>\n"
        "<queries>[\"Jafar\"]</queries>\n"
        "</graph_search>"
    )
    turn2 = _split(
        "<attempt_completion>\n"
        "<result>جعفر با شرکت Acme در ۲ هاپ مرتبط است.</result>\n"
        "</attempt_completion>"
    )
    _fake_llm.queue(turn1, turn2)

    events = []
    async for ev in cline_engine.stream_agent_cline(
        user_message="جعفر رو در گراف بگرد",
        session_id="sess-abc",
    ):
        events.append(ev)

    kinds = [e.get("type") for e in events]
    print("\nevents:")
    for e in events:
        keys = {k: v for k, v in e.items() if k not in ("timestamp",)}
        print(f"  {keys}")

    _check("got start", "start" in kinds)
    _check("does not stream thinking", all(k != "thinking" for k in kinds), str(kinds))
    _check("got tool_call_start", any(k == "tool_call_start" for k in kinds))
    _check("got tool_start", any(k == "tool_start" for k in kinds))
    _check("got tool_result", any(k == "tool_result" for k in kinds))
    _check("got llm_done at least once", any(k == "llm_done" for k in kinds))
    _check("got answer at least once", any(k == "answer" for k in kinds))
    _check("got exactly one done", sum(1 for k in kinds if k == "done") == 1)

    done = next(e for e in events if e.get("type") == "done")
    _check(
        "final_answer correct",
        done.get("final_answer") == "جعفر با شرکت Acme در ۲ هاپ مرتبط است.",
        str(done.get("final_answer")),
    )
    _check("one tool_run recorded", len(done.get("tool_runs", [])) == 1, str(done.get("tool_runs")))
    tr = done["tool_runs"][0]
    _check("tool_run.tool_id is graph_search", tr.get("tool_id") == "graph_search", str(tr))
    _check("tool_run.args has queries", tr.get("args", {}).get("queries") == ["Jafar"], str(tr.get("args")))
    _check(
        "tool was executed exactly once",
        len(_fake_executor.last_calls) == 1,
        str(_fake_executor.last_calls),
    )

    print("\n🎉 cline_engine end-to-end smoke passed.")


if __name__ == "__main__":
    asyncio.run(main())
