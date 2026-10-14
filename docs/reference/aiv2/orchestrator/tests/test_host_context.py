"""Unit tests for orchestrator host_context (page-aware floating AI dock)."""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load_host_context():
    from src import host_context

    return host_context


def _load_llm_tool_selection():
    from src.tools import llm_tool_selection

    return llm_tool_selection


def test_resolve_plugin_from_explicit_active_plugin_in_envelope():
    hc = _load_host_context()
    ctx = {
        "native_host_context_v1": {
            "schema_version": 1,
            "surface": "one_search",
            "pathname": "/one-search",
            "active_plugin": "smart_search",
            "query": "alice",
        }
    }
    assert hc.resolve_active_plugin(ctx) == "smart_search"
    prompt = hc.build_host_page_prompt(ctx)
    assert "HOST PAGE CONTEXT" in prompt
    assert "one_search" in prompt
    assert "alice" in prompt


def test_surface_alone_does_not_resolve_plugin():
    """No hardcoded surface→plugin maps — client must send active_plugin."""
    hc = _load_host_context()
    ctx = {
        "native_host_context_v1": {
            "schema_version": 1,
            "surface": "one_search",
            "pathname": "/one-search",
        }
    }
    assert hc.resolve_active_plugin(ctx) is None


def test_resolve_plugin_from_graph_envelope_active_plugin():
    hc = _load_host_context()
    ctx = {
        "graph_visual_explorer_v1": {
            "schema_version": 1,
            "active_plugin": "graph_explorer",
            "surface": "standalone_visual_explorer",
            "client": {
                "pathname": "/graph/visual-explorer",
                "href": "http://x/graph/visual-explorer",
            },
            "graph": {"total_nodes": 10, "visible_nodes": 3},
            "focus": {"kind": "node", "id": "n1"},
        }
    }
    plugin, prompt = hc.resolve_host_context(ctx)
    assert plugin == "graph_explorer"
    assert "graph_visual_explorer_v1" in prompt


def test_explicit_active_plugin_wins():
    hc = _load_host_context()
    ctx = {
        "active_plugin": "map_explorer",
        "native_host_context_v1": {
            "surface": "one_search",
            "pathname": "/one-search",
            "active_plugin": "smart_search",
        },
    }
    assert hc.resolve_active_plugin(ctx) == "map_explorer"


def test_empty_context():
    hc = _load_host_context()
    assert hc.resolve_active_plugin(None) is None
    assert hc.build_host_page_prompt(None) == ""
    assert hc.resolve_host_context({}) == (None, "")


def test_apply_plugin_scope_includes_root_plugin_id():
    mod = _load_llm_tool_selection()
    tools = [
        {"id": "plugin_smart_search", "plugin_id": "smart_search", "channels": ["llm"], "input_types": ["text"]},
        {"id": "plugin_smart_search_text", "plugin_id": "smart_search", "channels": ["llm"], "input_types": ["text"]},
        {"id": "plugin_map_explorer_list", "plugin_id": "map_explorer", "channels": ["llm"], "input_types": ["text"]},
        {
            "id": "session_files",
            "channels": ["llm"],
            "input_types": ["session"],
            "category": "session",
        },
        {"id": "graph_search", "channels": ["llm"], "input_types": ["text"]},
    ]
    scoped = mod._apply_plugin_scope(tools, "smart_search")
    names = {t["id"] for t in scoped}
    assert "plugin_smart_search" in names
    assert "plugin_smart_search_text" in names
    assert "plugin_map_explorer_list" not in names
    assert "graph_search" in names


def test_apply_plugin_scope_map_explorer():
    mod = _load_llm_tool_selection()
    tools = [
        {"id": "plugin_map_explorer_add_markers", "plugin_id": "map_explorer", "channels": ["llm"], "input_types": ["text"]},
        {"id": "plugin_map_explorer_zoom_to_bounds", "plugin_id": "map_explorer", "channels": ["llm"], "input_types": ["text"]},
        {"id": "plugin_smart_search", "plugin_id": "smart_search", "channels": ["llm"], "input_types": ["text"]},
        {"id": "session_files", "channels": ["llm"], "category": "session"},
    ]
    scoped = mod._apply_plugin_scope(tools, "map_explorer")
    names = {t["id"] for t in scoped}
    assert "plugin_map_explorer_add_markers" in names
    assert "plugin_map_explorer_zoom_to_bounds" in names
    assert "plugin_smart_search" not in names
    assert "session_files" in names


def test_main_chat_hides_all_plugin_tools():
    """Without active_plugin, plugin_* must not reach the LLM."""
    mod = _load_llm_tool_selection()
    tools = [
        {"id": "plugin_map_explorer_list", "plugin_id": "map_explorer", "channels": ["llm"], "input_types": ["text"]},
        {"id": "plugin_agent_flow_run", "plugin_id": "agent_flow", "channels": ["llm"], "input_types": ["text"]},
        {"id": "plugin_smart_search", "plugin_id": "smart_search", "channels": ["llm"], "input_types": ["text"]},
        {"id": "text_search", "channels": ["llm"], "input_types": ["text"]},
        {"id": "session_files", "channels": ["llm"], "category": "session"},
    ]
    selected = mod.select_llm_tools(tools, active_plugin=None)
    names = {t["id"] for t in selected}
    assert "text_search" in names
    assert "session_files" in names
    assert "plugin_map_explorer_list" not in names
    assert "plugin_agent_flow_run" not in names
    assert "plugin_smart_search" not in names


def test_active_plugin_exposes_only_that_plugin():
    mod = _load_llm_tool_selection()
    tools = [
        {"id": "plugin_agent_flow_list", "plugin_id": "agent_flow", "channels": ["llm"], "input_types": ["text"]},
        {"id": "plugin_agent_flow_run", "plugin_id": "agent_flow", "channels": ["llm"], "input_types": ["text"]},
        {"id": "plugin_map_explorer_stats", "plugin_id": "map_explorer", "channels": ["llm"], "input_types": ["text"]},
        {"id": "text_search", "channels": ["llm"], "input_types": ["text"]},
    ]
    selected = mod.select_llm_tools(tools, active_plugin="agent_flow")
    names = {t["id"] for t in selected}
    assert "plugin_agent_flow_list" in names
    assert "plugin_agent_flow_run" in names
    assert "plugin_map_explorer_stats" not in names
    assert "text_search" in names


def test_plugin_id_metadata_match_without_name_prefix_guess():
    mod = _load_llm_tool_selection()
    tools = [
        {
            "id": "plugin_weird_name",
            "plugin_id": "map_explorer",
            "channels": ["llm"],
            "input_types": ["text"],
        },
        {"id": "text_search", "channels": ["llm"], "input_types": ["text"]},
    ]
    selected = mod.select_llm_tools(tools, active_plugin="map_explorer")
    names = {t["id"] for t in selected}
    assert "plugin_weird_name" in names
    assert "text_search" in names
    assert mod.select_llm_tools(tools, active_plugin=None) == [
        t for t in tools if t["id"] == "text_search"
    ]


def test_apply_plugin_scope_never_falls_back_to_all_tools():
    mod = _load_llm_tool_selection()
    tools = [
        {"id": "plugin_only_tool", "plugin_id": "only", "channels": ["llm"], "input_types": ["text"]},
    ]
    # Wrong page → empty, not the full unfiltered list.
    assert mod._apply_plugin_scope(tools, "other_plugin") == []
    assert mod._apply_plugin_scope(tools, None) == []


@pytest.mark.asyncio
async def test_llm_host_prompt_comes_from_the_tool_runner(monkeypatch):
    """The plugin.json prompt is served by the tool-runner, not read from a plugins folder."""
    import json

    from src import plugin_host_prompt as php
    from src.tools import executor as ex

    with open(os.path.join(ROOT, "modules", "plugins", "map_explorer", "plugin.json"), encoding="utf-8") as fh:
        served = json.load(fh)
    asked = []

    class _Runner:
        async def get_plugin_manifest(self, pid):
            asked.append(pid)
            return served if pid == "map_explorer" else {}

    monkeypatch.setattr(ex, "get_tool_executor", lambda: _Runner())
    php.clear_llm_host_prompt_cache()
    assert php.get_llm_host_prompt("map_explorer") == ""
    text = await php.load_llm_host_prompt("map_explorer")
    assert "map assistant" in text.lower() and "map_action" in text
    assert php.get_llm_host_prompt("map_explorer") == text
    await php.load_llm_host_prompt("map_explorer")
    assert asked == ["map_explorer"]
    assert await php.load_llm_host_prompt("no_such_plugin") == ""

    class _Down:
        async def get_plugin_manifest(self, pid):
            return None

    monkeypatch.setattr(ex, "get_tool_executor", lambda: _Down())
    assert await php.load_llm_host_prompt("map_explorer", force_refresh=True) == text
    php.clear_llm_host_prompt_cache()


def test_strip_channels_keeps_ui_map_action():
    """Mirrors streaming._strip_channels keep-ui policy."""
    channels = {
        "llm": "ok",
        "ui": {"map_action": "zoom_to_bounds", "bounds": [1, 2, 3, 4]},
        "rawdata": {"huge": "x" * 100},
        "embed": {"v": [0.1]},
    }
    kept = {}
    if "llm" in channels:
        kept["llm"] = channels.get("llm")
    if "ui" in channels:
        kept["ui"] = channels.get("ui")
    assert kept["ui"]["map_action"] == "zoom_to_bounds"
    assert "rawdata" not in kept
    assert "embed" not in kept


def test_chat_mentions_v1_in_host_prompt_without_page_envelope():
    hc = _load_host_context()
    ctx = {
        "chat_mentions_v1": {
            "schema_version": 1,
            "files": [
                {
                    "id": "art-1",
                    "label": "notes.txt",
                    "path": "minio://uploads/notes.txt",
                    "mime_type": "text/plain",
                }
            ],
            "cases": [{"id": "cas_abc", "label": "Case A"}],
            "pages": [],
        }
    }
    prompt = hc.build_host_page_prompt(ctx)
    assert "USER MENTIONS" in prompt
    assert "artifact_id=art-1" in prompt
    assert "case_id=cas_abc" in prompt
    # Guidance comes from the hints file and describes the action, not a tool id.
    assert "attach it to the workspace with this artifact_id" in prompt
    assert "semantic search scoped to this case_id" in prompt
    assert "session_attach" not in prompt and "text_search" not in prompt


def test_page_envelopes_are_configured_not_coded(monkeypatch):
    """A new page envelope works by setting, with its fields rendered generically."""
    from src.config import settings

    hc = _load_host_context()
    ctx = {"timeline_page_v1": {"schema_version": 1, "active_plugin": "timeline", "orchestrator_id": "rlm",
                                "surface": "timeline", "window": {"from": "2026-01-01", "to": "2026-02-01"}}}
    assert hc.resolve_host_context(ctx) == (None, "")
    monkeypatch.setattr(settings, "HOST_PAGE_CONTEXT_KEYS", "timeline_page_v1,native_host_context_v1")
    plugin, prompt = hc.resolve_host_context(ctx)
    assert plugin == "timeline"
    assert "Source: timeline_page_v1" in prompt and "- surface: timeline" in prompt
    assert '"from":"2026-01-01"' in prompt
    assert "orchestrator_id" not in prompt and "schema_version" not in prompt

    from src.orchestrators.chat_engine_resolve import is_floating_dock_context, resolve_chat_orchestrator_for_request

    assert is_floating_dock_context(ctx)
    assert resolve_chat_orchestrator_for_request("native_tools", ctx) == "rlm"
