"""RLM uses the project's tools: schema-driven prompt surface, per-turn refresh,
recorded/streamed tool runs, host-owned arguments, direct-call fallback."""

import asyncio
from typing import Any, Dict, List

import pytest

from src.config import settings
from src.orchestrators.host_adapters import tools_binding as tb
from src.orchestrators.host_adapters.hub_sse import HubEventBridge
from src.orchestrators.rlm.runtime.prompts import (
    TOOL_SURFACE_HEADING,
    format_tool_surface,
    replace_tool_surface,
)
from src.orchestrators.rlm.runtime.session import RlmSession


def spec(name: str, desc: str, props: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


SEARCH = spec(
    "session_search_workspace",
    "Search files already in THIS chat workspace only. Requires query.\n\n## When to use\n- never shown",
    {
        "session_id": {"type": "string"},
        "mode": {"type": "string", "enum": ["semantic", "keyword"]},
        "query": {"type": "string"},
        "top_k": {"type": "integer"},
    },
    ["session_id", "query"],
)
FILES = spec("session_files", "List artifacts in this chat.", {"limit": {"type": "integer"}}, [])


run = asyncio.run


# ---------------------------------------------------------------------------
# prompt surface
# ---------------------------------------------------------------------------

def test_surface_renders_signature_from_schema():
    owned = frozenset({"session_id", "user_id"})
    section = format_tool_surface([tb._without_host_args(SEARCH, owned)], ["session_search_workspace"])
    line = next(ln for ln in section.splitlines() if ln.startswith("- "))
    assert line.startswith("- `tools.session_search_workspace(query: str, mode?: 'semantic'|'keyword', top_k?: int)`")
    assert "session_id" not in line
    assert "Requires query." in line and "When to use" not in line


def test_surface_limits_params_and_description():
    section = format_tool_surface([SEARCH], ["session_search_workspace"], description_chars=40, max_params=2)
    line = section.splitlines()[-1]
    assert "(session_id: str, query: str, …)" in line
    assert line.endswith("only.") or line.endswith("…")
    assert len(line.split(" — ", 1)[1]) <= 41


def test_surface_lists_tools_without_schema_and_skips_empty():
    assert format_tool_surface([], []) == ""
    section = format_tool_surface([], ["custom_tool"])
    assert section.splitlines()[-1] == "- `tools.custom_tool()`"


def test_replace_tool_surface_swaps_appends_and_drops_legacy():
    base = "intro\n\n## Allowed tools\n`tools.a`\n\n## Attached data pools\nx"
    swapped = replace_tool_surface(base, format_tool_surface([FILES], ["session_files"]))
    assert "## Allowed tools" not in swapped and "## Attached data pools\nx" in swapped
    assert swapped.endswith("- `tools.session_files(limit?: int)` — List artifacts in this chat.")
    again = replace_tool_surface(swapped, format_tool_surface([SEARCH], ["session_search_workspace"]))
    assert again.count(TOOL_SURFACE_HEADING) == 1 and "session_files" not in again
    assert replace_tool_surface(again, "") == "intro\n\n## Attached data pools\nx"


# ---------------------------------------------------------------------------
# host adapter
# ---------------------------------------------------------------------------

class FakeExecutor:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    async def get_openai_tools_for_ids(self, ids):
        return [s for s in (SEARCH,) if s["function"]["name"] in ids]

    async def get_tools_manifest(self):
        return [{"id": "session_files", "description": "List artifacts in this chat.",
                 "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}}}]

    def get_openai_tools_format(self, rows):
        return [spec(r["id"], r["description"], r["parameters"]["properties"], []) for r in rows]

    async def execute(self, tool_id, args, **kw):
        self.calls.append({"tool_id": tool_id, "args": args, **kw})
        return {"ok": True, "echo": args}


def test_kernel_tool_specs_canonical_then_manifest_in_selection_order(monkeypatch):
    monkeypatch.setattr(tb, "host_owned_tool_args", lambda: frozenset({"session_id"}))
    specs = run(tb.kernel_tool_specs(FakeExecutor(), ["session_files", "session_search_workspace"]))
    assert [s["function"]["name"] for s in specs] == ["session_files", "session_search_workspace"]
    params = specs[1]["function"]["parameters"]
    assert "session_id" not in params["properties"] and params["required"] == ["query"]
    assert "session_id" in SEARCH["function"]["parameters"]["properties"]


def test_host_port_owns_configured_args(monkeypatch):
    monkeypatch.setattr(settings, "TOOL_HOST_OWNED_ARGS", "session_id, user_id")
    ex = FakeExecutor()
    port = tb.make_tools_host_port(ex, session_id="chat-1", user_id="u-1")
    run(port("session_files", {"session_id": "rlm_chat_x", "user_id": "evil", "limit": 3}))
    assert ex.calls == [{"tool_id": "session_files", "args": {"limit": 3},
                         "session_id": "chat-1", "user_id": "u-1", "user_message": ""}]


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------

def make_session(tmp_path, runner, tool_ids=("session_files",), specs=(FILES,)):
    events: List[Dict[str, Any]] = []

    async def chat_fn(body):
        return {}

    sess = RlmSession(
        cfg={"model": "m", "tool_ids": list(tool_ids), "tool_specs": list(specs), "run_lifetime": "chat"},
        user_content="hi",
        artifact_dir=str(tmp_path / "s"),
        chat_fn=chat_fn,
        extract_assistant_fn=lambda r: {},
        gen_params={},
        tool_runner=runner,
        on_event=events.append,
    )
    return sess, events


def test_system_prompt_carries_tool_surface(tmp_path):
    async def runner(tid, args):
        return {"ok": True}

    sess, _ = make_session(tmp_path, runner)
    prompt = sess.build_system_prompt("")
    assert TOOL_SURFACE_HEADING in prompt
    assert "- `tools.session_files(limit?: int)` — List artifacts in this chat." in prompt


def test_cold_run_uses_cfg_system_prompt_with_host_context(tmp_path, monkeypatch):
    from src.orchestrators._shared.host_context import replace_host_context

    async def runner(tid, args):
        return {"ok": True}

    sess, _ = make_session(tmp_path, runner)
    sess.cfg["system_prompt"] = replace_host_context("BASE", {"memory": "## Recalled memory\n- name is Lio"})
    seen: Dict[str, str] = {}

    async def fake_loop(*, base_system, terminal):
        seen["system"] = sess.build_system_prompt(base_system)
        return ""

    async def noop():
        return None

    monkeypatch.setattr(sess, "_run_turn_loop", fake_loop)
    monkeypatch.setattr(sess, "flush_kernel_snapshot", noop)
    monkeypatch.setattr(sess, "_maybe_auto_refine_checkpoint", noop)
    run(sess.run_turns())
    assert "BASE" in seen["system"]
    assert seen["system"].index("name is Lio") > seen["system"].index(TOOL_SURFACE_HEADING)


def test_call_project_tool_records_and_emits(tmp_path):
    async def runner(tid, args):
        return {"ok": True, "summary": "2 files", "files": ["a", "b"]}

    sess, events = make_session(tmp_path, runner)
    out = run(sess.call_project_tool("session_files", {"limit": 2, "_secret": 1}))
    assert out["files"] == ["a", "b"]
    run_rec = sess.tool_runs[-1]
    assert run_rec["tool_id"] == "session_files" and run_rec["args"] == {"limit": 2} and run_rec["ok"]
    kinds = [e["kind"] for e in events if e["kind"].startswith("tool_")]
    assert kinds == ["tool_start", "tool_result"]
    result_ev = [e for e in events if e["kind"] == "tool_result"][0]
    assert result_ev["summary"] == "2 files" and result_ev["execution_time"] >= 0


def test_tool_failures_come_back_as_data_and_disallowed_raise(tmp_path):
    async def runner(tid, args):
        raise TimeoutError("tool-runner timed out")

    sess, events = make_session(tmp_path, runner)
    out = run(sess.call_project_tool("session_files", {}))
    assert out["ok"] is False and "timed out" in out["error"]
    assert sess.tool_runs[-1]["ok"] is False
    with pytest.raises(PermissionError):
        run(sess.call_project_tool("image_generate", {}))


def test_refresh_tool_surface_updates_prompt_kernel_and_allowlist(tmp_path):
    async def runner(tid, args):
        return {"ok": True}

    class Kernel:
        tools = None
        namespace = None

        async def set_tools(self, ids, *, namespace=None):
            Kernel.tools = list(ids)
            Kernel.namespace = namespace

    sess, _ = make_session(tmp_path, runner)
    sess.messages = [{"role": "system", "content": sess.build_system_prompt("")}, {"role": "user", "content": "hi"}]
    sess.kernel = Kernel()
    run(sess.refresh_tool_surface(["session_search_workspace"], [SEARCH]))
    assert sess.tool_ids == ["session_search_workspace"] and Kernel.tools == ["session_search_workspace"]
    assert Kernel.namespace == sess.tool_namespace
    system = sess.messages[0]["content"]
    assert system.count(TOOL_SURFACE_HEADING) == 1
    assert "tools.session_search_workspace(" in system and "tools.session_files(" not in system
    assert sess.subagent_tool_ids == ["session_search_workspace"]


def test_namespace_called_as_dispatcher_runs_the_named_tool(tmp_path):
    seen = []

    async def runner(tid, args):
        seen.append((tid, args))
        return {"ok": True}

    sess, _ = make_session(tmp_path, runner)
    run(sess._non_ipython_tool_result(sess.tool_namespace, {"tool_name": "session_files", "limit": 2}))
    assert seen == [("session_files", {"limit": 2})]


def test_direct_call_of_allowed_tool_runs_it(tmp_path):
    async def runner(tid, args):
        return {"ok": True, "count": 4}

    sess, _ = make_session(tmp_path, runner)
    content = run(sess._non_ipython_tool_result("session_files", {"limit": 4}))
    assert '"count": 4' in content and "await tools.session_files(" in content
    refused = run(sess._non_ipython_tool_result("rm_rf", {}))
    assert "only the `ipython` tool" in refused


def test_new_chat_turn_does_not_inherit_previous_answer(tmp_path):
    async def runner(tid, args):
        return {"ok": True}

    async def noop(*a, **k):
        return None

    sess, _ = make_session(tmp_path, runner)
    sess.messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q1"},
                     {"role": "assistant", "content": "answer one"}]
    sess.last_content = "answer one"
    sess._last_quiescent_turn_content = "answer one"
    seen = {}

    async def loop(**kw):
        seen["last"] = sess.last_content
        seen["quiescent"] = sess._last_quiescent_turn_content
        return sess.last_content

    sess._run_turn_loop = loop
    sess.flush_kernel_snapshot = noop
    sess._maybe_auto_refine_checkpoint = noop
    assert run(sess.continue_with_message("q2")) == ""
    assert seen == {"last": "", "quiescent": ""}


def test_kernel_cell_shows_trailing_expression_value(tmp_path):
    from src.orchestrators.rlm.runtime.worker import _exec_cell

    class Shell:
        user_ns: Dict[str, Any] = {}

    shell = Shell()
    cell = lambda code: run(_exec_cell(shell, code, cwd=str(tmp_path)))
    out = cell("async def tool():\n    return {'ok': True, 'n': 2}\nawait tool()")
    assert out["result"] == "{'ok': True, 'n': 2}" and not out["error"]
    assert shell.user_ns["_"] == {"ok": True, "n": 2}
    assert cell("r = await tool()")["result"] is None and shell.user_ns["r"]["n"] == 2
    assert cell("await tool();")["result"] is None
    printed = cell("print('hi')")
    assert printed["stdout"] == "hi\n" and printed["result"] is None
    assert "SyntaxError" in cell("def (")["error"]


def test_namespace_follows_manifest_and_binding(tmp_path):
    from src.orchestrators._shared.manifest_config import package_manifest
    import src.orchestrators.rlm.runtime.session as rlm_session

    async def runner(tid, args):
        return {"ok": True}

    declared = tb.read_tools_binding(package_manifest(rlm_session.__file__)).namespace
    sess, _ = make_session(tmp_path, runner)
    assert sess.tool_namespace == declared
    assert f"`{declared}.session_files(" in sess.build_system_prompt("")

    sess.cfg["tool_namespace"] = "platform"
    assert "`platform.session_files(" in sess.build_system_prompt("")
    hint = run(sess._non_ipython_tool_result("session_files", {}))
    assert "await platform.session_files(" in hint
    assert "await platform.<id>(" in run(sess._non_ipython_tool_result("nope", {}))
    custom = tb.read_tools_binding({"tools_binding": {"mode": "kernel_callable", "surface": {"namespace": "kit"}}})
    assert custom.namespace == "kit"
    bad = tb.read_tools_binding({"tools_binding": {"mode": "kernel_callable", "surface": {"namespace": "not valid"}}})
    assert bad.namespace == bad.host_port


def test_prompt_aliases_resolve_in_kernel():
    import re as _re

    from src.orchestrators.rlm.runtime import worker

    class Host:
        async def request(self, *a, **k):
            return None

    ids = ["plugin-map.export", "3d_render", "session_files"]
    specs = [spec(t, "d", {}, []) for t in ids]
    surface = format_tool_surface(specs, ids, namespace="tools")
    tools_obj = worker._build_namespace(Host(), [], ids)["tools"]
    advertised = _re.findall(r"^- `tools\.([A-Za-z_][A-Za-z0-9_]*)\(", surface, _re.M)
    assert len(advertised) == len(ids)
    for alias, tid in zip(advertised, ids):
        assert getattr(tools_obj, alias).__name__ == alias
        assert tools_obj._resolve(alias) == tid


def test_hub_bridge_streams_tool_events():
    bridge = HubEventBridge()
    start = bridge.map_event({"kind": "tool_start", "tool_id": "session_files", "args": {"limit": 2}, "step": 1})
    done = bridge.map_event({"kind": "tool_result", "tool_id": "session_files", "args": {"limit": 2},
                             "ok": True, "summary": "2 files", "execution_time": 0.2, "step": 1})
    assert start[0]["type"] == "tool_start" and start[0]["tool_name"] == "session_files"
    assert done[0]["type"] == "tool_result" and done[0]["ok"] and done[0]["summary"] == "2 files"
