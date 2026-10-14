"""Unit tests for chat orchestrator_id → engine resolve (registry-driven)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.orchestrators import chat_engine_resolve as cer
from src.orchestrators.contract import normalize_manifest, validate_manifest
from src.orchestrators.chat_engine_resolve import (
    chat_stream_orchestrator_ids,
    default_chat_orchestrator_id,
    is_floating_dock_context,
    resolve_chat_orchestrator_for_request,
    resolve_chat_orchestrator_id,
    stream_agent_for_id,
    wraps_engines,
    chat_stream_manifest,
)


def test_chat_engines_come_from_registry():
    ids = chat_stream_orchestrator_ids()
    assert {"native_tools", "rlm", "srlm", "rrsi"} <= ids


def test_every_chat_engine_entry_imports():
    missing = []
    for oid in chat_stream_orchestrator_ids():
        try:
            fn = stream_agent_for_id(oid)
        except ModuleNotFoundError as exc:
            if (exc.name or "").split(".")[0] == "src":
                raise
            missing.append(f"{oid}: {exc.name}")
            continue
        assert callable(fn), oid
    if missing:
        pytest.skip(f"third-party deps not installed here: {missing}")


def test_controllers_are_marked_as_wrapping_engines():
    assert wraps_engines(chat_stream_manifest("srlm"))
    assert wraps_engines(chat_stream_manifest("rrsi"))
    assert not wraps_engines(chat_stream_manifest("native_tools"))
    assert not wraps_engines(chat_stream_manifest("rlm"))


def test_resolve_known_ids():
    for oid in ("native_tools", "rlm", "srlm", "rrsi"):
        assert resolve_chat_orchestrator_id(oid) == oid


def test_archived_ids_fall_back_to_default():
    """cline_xml / langgraph_stream are archived — must not stay selectable."""
    out = resolve_chat_orchestrator_id("cline_xml")
    assert out == default_chat_orchestrator_id()
    assert out == "native_tools"
    assert resolve_chat_orchestrator_id("langgraph_stream") == "native_tools"


def test_unknown_falls_back():
    assert resolve_chat_orchestrator_id("does_not_exist") == "native_tools"


def test_empty_uses_default():
    assert resolve_chat_orchestrator_id(None) == "native_tools"
    assert resolve_chat_orchestrator_id("") == "native_tools"


def test_strict_lookup_raises_for_unknown():
    with pytest.raises(KeyError):
        stream_agent_for_id("does_not_exist")


def test_default_follows_settings(monkeypatch):
    monkeypatch.setattr(cer.settings, "DEFAULT_CHAT_ORCHESTRATOR", "rlm")
    assert default_chat_orchestrator_id() == "rlm"
    assert resolve_chat_orchestrator_id("does_not_exist") == "rlm"
    # A configured id that is not available falls back to a plain (non-controller) engine.
    monkeypatch.setattr(cer.settings, "DEFAULT_CHAT_ORCHESTRATOR", "gone")
    assert default_chat_orchestrator_id() in {"native_tools", "rlm"}


class _FakeRegistry:
    def __init__(self, entries):
        self._entries = {e.id: e for e in entries}

    def list_orchestrators(self, *, ready_only: bool = True, **_kw):
        return [e for e in self._entries.values() if e.ready or not ready_only]

    def get(self, oid):
        return self._entries.get(oid)


def _entry(oid: str, extensions: dict, ready: bool = True):
    real = Path(cer.__file__).resolve().parent
    raw = json.loads((real / "native_tools" / "manifest.json").read_text(encoding="utf-8"))
    raw["id"] = oid
    raw["extensions"] = extensions
    manifest = validate_manifest(normalize_manifest(raw))
    return SimpleNamespace(id=oid, ready=ready, manifest=manifest)


def test_new_drop_in_is_picked_up_without_code_changes(monkeypatch):
    entry = "src.orchestrators.chat_engine_resolve:is_floating_dock_context"
    reg = _FakeRegistry([
        _entry("native_tools", {"chat_stream": True, "stream_entry": entry}),
        _entry("brand_new_engine", {"chat_stream": True, "stream_entry": entry}),
        _entry("not_ready_engine", {"chat_stream": True, "stream_entry": "x:y"}, ready=False),
        _entry("no_entry_engine", {"chat_stream": True}),
    ])
    monkeypatch.setattr(cer, "get_registry", lambda: reg)
    ids = chat_stream_orchestrator_ids()
    assert ids == frozenset({"native_tools", "brand_new_engine"})
    assert resolve_chat_orchestrator_id("brand_new_engine") == "brand_new_engine"
    assert stream_agent_for_id("brand_new_engine") is is_floating_dock_context


def test_dock_context_forces_dock_engine_even_when_rlm_requested():
    assert is_floating_dock_context({"native_host_context_v1": {"page": "map"}})
    assert is_floating_dock_context({"graph_visual_explorer_v1": {"nodes": []}})
    assert not is_floating_dock_context({"chat_mentions_v1": {}})
    assert not is_floating_dock_context(None)

    for requested in ("rlm", "srlm", "rrsi"):
        assert resolve_chat_orchestrator_for_request(
            requested, {"native_host_context_v1": {"pathname": "/map"}}
        ) == "native_tools"
        assert resolve_chat_orchestrator_for_request(
            requested, {"graph_visual_explorer_v1": {}}
        ) == "native_tools"
        # Hub chat without dock envelope keeps the requested engine.
        assert resolve_chat_orchestrator_for_request(requested, None) == requested
        assert resolve_chat_orchestrator_for_request(
            requested, {"chat_mentions_v1": {"cases": []}}
        ) == requested


@pytest.mark.parametrize("engine", ["native_tools", "rlm", "srlm", "rrsi"])
def test_any_engine_can_serve_a_page_mini_chat(engine):
    """The page's own envelope picks the mini-chat engine; the hub's top-level id still does not."""
    for key in ("native_host_context_v1", "graph_visual_explorer_v1"):
        ctx = {key: {"surface": "map", "orchestrator_id": engine}}
        assert resolve_chat_orchestrator_for_request("rlm" if engine != "rlm" else "srlm", ctx) == engine


def test_mini_chat_engine_per_page_setting(monkeypatch):
    monkeypatch.setattr(cer.settings, "DOCK_CHAT_ORCHESTRATOR_BY_PAGE", "map=rlm, graph_explorer=srlm, bad=gone")
    page = lambda **env: {"native_host_context_v1": env}  # noqa: E731
    assert resolve_chat_orchestrator_for_request(None, page(surface="map")) == "rlm"
    assert resolve_chat_orchestrator_for_request(None, page(active_plugin="graph_explorer")) == "srlm"
    assert resolve_chat_orchestrator_for_request(None, {"active_plugin": "graph_explorer", "graph_visual_explorer_v1": {}}) == "srlm"
    # Envelope choice wins; unknown engines fall back to the dock default.
    assert resolve_chat_orchestrator_for_request(None, page(surface="map", orchestrator_id="rrsi")) == "rrsi"
    assert resolve_chat_orchestrator_for_request(None, page(surface="bad")) == "native_tools"
    assert resolve_chat_orchestrator_for_request(None, page(surface="map", orchestrator_id="gone")) == "rlm"
    assert resolve_chat_orchestrator_for_request(None, page(surface="other")) == "native_tools"
