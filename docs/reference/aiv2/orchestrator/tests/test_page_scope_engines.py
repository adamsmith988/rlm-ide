"""Every chat engine honours the page the chat runs on (tools + page prompt), not just native_tools."""

from __future__ import annotations

import pytest

from src.orchestrators._shared import host_context as hc
from src.tools.llm_tool_selection import page_allows_tool, select_llm_tools

CORE = {"id": "web_search", "channels": ["llm"], "category": "search", "args": {"query": "string"}}
MAP_TOOL = {"id": "plugin_map_route", "plugin_id": "map", "channels": ["llm"], "args": {"query": "string"}}
GRAPH_TOOL = {"id": "plugin_graph_explorer_graph_search", "plugin_id": "graph_explorer", "channels": ["llm"], "args": {"queries": "array"}}
# The page gate reads plugin_id only — an id that merely looks like a plugin's is a core tool.
LOOKALIKE = {"id": "plugin_map_lookalike", "channels": ["llm"], "args": {"query": "string"}}
MANIFEST = [CORE, MAP_TOOL, GRAPH_TOOL]


def test_page_gate_core_everywhere_plugins_only_on_their_page():
    assert page_allows_tool(CORE, None) and page_allows_tool(CORE, "map")
    assert not page_allows_tool(MAP_TOOL, None)
    assert page_allows_tool(MAP_TOOL, "map")
    assert not page_allows_tool(MAP_TOOL, "graph_explorer")
    assert page_allows_tool(GRAPH_TOOL, "graph_explorer")
    assert not page_allows_tool(GRAPH_TOOL, "map")
    assert page_allows_tool(LOOKALIKE, None) and page_allows_tool(LOOKALIKE, "graph_explorer")


def test_native_selection_and_gate_agree():
    for page in (None, "map", "graph_explorer"):
        selected = {t["id"] for t in select_llm_tools(MANIFEST, active_plugin=page)}
        assert selected == {t["id"] for t in MANIFEST if page_allows_tool(t, page)}


class _Exec:
    async def get_tools_manifest(self):
        return list(MANIFEST)


async def _all_allowed(manifest, user_id, roles=None):
    return list(manifest)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "page, expected",
    [(None, ["web_search"]), ("map", ["plugin_map_route", "web_search"]), ("graph_explorer", ["web_search"])],
)
async def test_kernel_engine_tool_ids_override_stays_on_the_page(monkeypatch, page, expected):
    from src.orchestrators.host_adapters.tools_binding import ToolsBinding, resolve_tool_ids_for_run

    monkeypatch.setattr("src.auth.permissions_client.resolve_allowed_tools", _all_allowed)
    ids = await resolve_tool_ids_for_run(
        binding=ToolsBinding(mode="kernel_callable"),
        tool_executor=_Exec(),
        user_id="u1",
        orchestrator_config={"tool_ids": ["plugin_map_route", "web_search"]},
        active_plugin=page,
    )
    assert ids == expected


@pytest.mark.asyncio
async def test_rrsi_evolver_catalog_is_page_scoped(monkeypatch):
    from src.orchestrators.rrsi import chat_stream as rrsi_stream

    monkeypatch.setattr("src.auth.permissions_client.resolve_allowed_tools", _all_allowed)
    monkeypatch.setattr("src.tools.executor.get_tool_executor", lambda: _Exec())
    _rows, main_chat = await rrsi_stream._tool_catalog("u1")
    _rows, on_map = await rrsi_stream._tool_catalog("u1", "map")
    assert main_chat == {"web_search"}
    assert on_map == {"web_search", "plugin_map_route"}


def test_page_block_joins_plugin_prompt_and_page_context(monkeypatch):
    monkeypatch.setattr(
        "src.plugin_host_prompt.get_llm_host_prompt",
        lambda pid, **_k: f"PLUGIN PROMPT {pid}" if pid else "",
    )
    assert hc.page_prompt_block(None, None) == ""
    assert hc.page_prompt_block(None, "HOST PAGE CONTEXT\n- surface: map") == "HOST PAGE CONTEXT\n- surface: map"
    text = hc.page_prompt_block("map", "HOST PAGE CONTEXT\n- surface: map")
    assert text.index("PLUGIN PROMPT map") < text.index("HOST PAGE CONTEXT")


def test_page_block_refreshes_in_place_each_turn(monkeypatch):
    """Kernel engines swap host blocks per turn — a page change replaces, never stacks."""
    monkeypatch.setattr("src.plugin_host_prompt.get_llm_host_prompt", lambda pid, **_k: "")
    system = "BASE"
    for surface in ("map", "graph", "map"):
        blocks = hc.prompt_blocks({"page_prompt_block": hc.page_prompt_block(None, f"- surface: {surface}")}, "*")
        system = hc.replace_host_context(system, blocks)
    assert system.count("- surface:") == 1 and "- surface: map" in system
    # Settings can still turn the page block off.
    assert hc.prompt_blocks({"page_prompt_block": "x"}, "memory") == {}


@pytest.mark.asyncio
async def test_srlm_summary_step_sees_the_page(monkeypatch):
    """SRLM writes the final reply itself, so its select step must carry the page block."""
    from src.orchestrators.srlm import chat_stream as srlm_stream
    from src.orchestrators.srlm.runtime import controller as ctl

    monkeypatch.setattr("src.plugin_host_prompt.get_llm_host_prompt", lambda pid, **_k: f"PLUGIN {pid}" if pid else "")
    seen = {}

    async def fake_plan_select(self, user_message, context, samples):
        seen["context"] = context
        raise RuntimeError("stop here")

    monkeypatch.setattr(ctl.SelfReflectController, "plan_select", fake_plan_select)
    monkeypatch.setattr(srlm_stream, "gateway_llm", lambda: object())
    async for _ev in srlm_stream.stream_agent_srlm(
        "where am I?",
        "s1",
        orchestrator_config={"summary_strategy": "greedy", "peer_engine": ""},
        active_plugin="map",
        host_page_prompt="HOST PAGE CONTEXT\n- surface: map",
    ):
        pass
    system = "\n".join(m["content"] for m in seen["context"] if m.get("role") == "system")
    assert "PLUGIN map" in system and "- surface: map" in system
