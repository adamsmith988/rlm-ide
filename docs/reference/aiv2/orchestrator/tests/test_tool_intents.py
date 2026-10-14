"""Intent narrowing reads tool.json metadata only: renaming every tool changes nothing."""

from __future__ import annotations

import pytest

from src.session_intent import infer_session_intent
from src.tools import tool_intents
from src.tools.llm_tool_selection import select_llm_tools_for_turn

# Metadata as the tool-runner serves it (copied from the real tool.json files).
CATALOG = [
    {"id": "session_files", "category": "session", "capabilities": ["session_access", "file_listing"], "input_types": ["session"], "channels": ["llm"]},
    {"id": "session_read_file", "category": "session", "capabilities": ["session_access", "file_read", "text_extraction"], "input_types": ["session", "file", "text"], "channels": ["llm"]},
    {"id": "session_search_workspace", "category": "session", "capabilities": ["session_access", "text_search", "semantic_search"], "input_types": ["session", "text"], "channels": ["llm"]},
    {"id": "session_attach", "category": "session", "capabilities": ["session_access", "file_linking", "storage_lookup"], "input_types": ["session"], "channels": ["llm"]},
    {"id": "graph_search", "category": "graph", "capabilities": ["knowledge_graph", "semantic_search"], "input_types": ["text"], "channels": ["llm"]},
    {"id": "text_search", "category": "text", "capabilities": ["semantic_search", "text_retrieval"], "input_types": ["text"], "channels": ["llm"]},
    {"id": "face_search", "category": "face", "capabilities": ["face_search", "face_recognition"], "input_types": ["image", "file"], "channels": ["llm"]},
    {"id": "image_search", "category": "image", "capabilities": ["image_search", "similarity_search"], "input_types": ["image", "file"], "channels": ["llm"]},
    {"id": "plugin_map_explorer_inspect", "plugin_id": "map_explorer", "category": "map", "capabilities": ["geo"], "input_types": ["text"], "channels": ["llm"]},
    {"id": "plugin_map_explorer_semantic_search", "plugin_id": "map_explorer", "category": "map", "capabilities": ["geo", "search"], "input_types": ["text"], "channels": ["llm"]},
]

TURNS = [
    ("read", "بخون فایل گزارش رو", [{"artifact_id": "a1"}, {"artifact_id": "a2"}]),
    ("read", "read the report", [{"artifact_id": "a1"}]),
    ("search", "search for invoice numbers", [{"artifact_id": "a1"}]),
    ("attach", "attach the case file from storage", []),
    ("list", "which files do I have?", []),
    ("graph", "show the knowledge graph links for Arash", []),
    ("search", "search faces in my photos", [{"artifact_id": "a1"}]),
]


def _renamed(catalog):
    return [{**t, "id": f"zz_{i}_{t['id'][::-1]}"} for i, t in enumerate(catalog)]


def _select(catalog, message, intent, registry, page=None):
    return [
        t["id"]
        for t in select_llm_tools_for_turn(
            catalog,
            user_message=message,
            session_intent=intent,
            session_file_registry=registry,
            intent_filter_enabled=True,
            active_plugin=page,
        )
    ]


def test_vocabulary_comes_from_the_data_file():
    data = tool_intents.intents()
    assert data["markers"]["graph"] and data["session_tools"]["read"]
    assert infer_session_intent("which files do I have?") == "list"
    assert infer_session_intent("show the knowledge graph links") == "graph"


@pytest.mark.parametrize("intent, message, registry", TURNS)
@pytest.mark.parametrize("page", [None, "map_explorer"])
def test_renaming_every_tool_keeps_the_selection(intent, message, registry, page):
    renamed = _renamed(CATALOG)
    by_new = {r["id"]: o["id"] for o, r in zip(CATALOG, renamed)}
    original = _select(CATALOG, message, intent, registry, page)
    after = [by_new[i] for i in _select(renamed, message, intent, registry, page)]
    assert after == original


def test_intent_rules_by_metadata():
    two_files = [{"artifact_id": "a1"}, {"artifact_id": "a2"}]
    assert set(_select(CATALOG, "read it", "read", two_files)) >= {"session_read_file", "session_files"}
    assert "session_files" not in _select(CATALOG, "read it", "read", [{"artifact_id": "a1"}])
    assert _select(CATALOG, "attach from storage", "attach", []) == ["session_attach"]
    assert _select(CATALOG, "which files", "list", []) == ["session_files"]
    graph = _select(CATALOG, "knowledge graph", "graph", [])
    assert graph == ["graph_search"]
    # Mention-gated tools stay out of file intents unless the user names them.
    assert "face_search" not in _select(CATALOG, "search invoices", "search", [{"artifact_id": "a1", "mime_type": "image/png"}])
    assert "face_search" in _select(CATALOG, "search faces", "search", [{"artifact_id": "a1", "mime_type": "image/png"}])
    # Discover offers search-capable tools; the page's own plugin search tool only on its page.
    assert "text_search" in _select(CATALOG, "find a file about invoices", "discover", [])
    assert "plugin_map_explorer_semantic_search" not in _select(CATALOG, "find", "discover", [])
    assert "plugin_map_explorer_semantic_search" in _select(CATALOG, "find", "discover", [], "map_explorer")


def test_intents_file_is_configurable(monkeypatch, tmp_path):
    from src.config import settings

    custom = tool_intents.intents() | {"session_tools": {"list": ["storage_lookup"]}}
    path = tmp_path / "intents.json"
    import json

    path.write_text(json.dumps(custom), encoding="utf-8")
    monkeypatch.setattr(settings, "TOOL_INTENTS_FILE", str(path))
    tool_intents.intents.cache_clear()
    try:
        assert _select(CATALOG, "which files", "list", []) == ["session_attach"]
    finally:
        tool_intents.intents.cache_clear()
