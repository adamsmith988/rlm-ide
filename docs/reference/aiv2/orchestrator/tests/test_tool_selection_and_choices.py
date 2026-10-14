"""Generic file tools reach the LLM when files exist; choice arguments are checked against the schema."""

import asyncio
from typing import Any, Dict, List

import pytest

from src.tools.executor import ToolExecutor
from src.tools.llm_tool_selection import select_llm_tools, select_llm_tools_for_turn


def row(tool_id: str, **extra: Any) -> Dict[str, Any]:
    return {"id": tool_id, "channels": ["llm"], **extra}


CATALOG = [
    row("file_identify", category="file", input_types=["file"], file_types=["*/*"]),
    row("image_meta", category="image", input_types=["image", "file"]),
    row("document_text", category="document", input_types=["document"]),
    row("audio_transcribe", category="audio", input_types=["audio"]),
]

PDF = {"name": "report.pdf", "mime_type": "application/pdf", "type": "document"}


def ids(tools: List[Dict[str, Any]]) -> List[str]:
    return [t["id"] for t in tools]


def test_generic_file_tool_selected_for_any_attached_file():
    chosen = ids(select_llm_tools(CATALOG, session_file_registry=[PDF]))
    assert "file_identify" in chosen
    assert "document_text" in chosen
    assert "audio_transcribe" not in chosen


def test_generic_file_tool_hidden_without_files():
    assert "file_identify" not in ids(select_llm_tools(CATALOG))


def test_turn_selection_keeps_generic_file_tool():
    chosen = ids(select_llm_tools_for_turn(
        CATALOG, user_message="what is the real type of report.pdf?", session_file_registry=[PDF],
    ))
    assert "file_identify" in chosen


def test_catalog_modalities():
    from src.orchestrators.host_adapters.model_catalog import input_modalities_from_catalog as mods

    assert mods({"input_modalities": ["text"]}) == {"text"}
    assert mods({"input_modalities": ["Text", "Image"]}) == {"text", "image"}
    assert mods({"capabilities": {"supports_vision": False}}) == {"text"}
    assert mods({"capabilities": {"supports_vision": True}}) == {"text", "image"}
    assert mods({"capabilities": {}}) is None and mods(None) is None


def test_native_engine_gates_images_on_model_capability(monkeypatch):
    pytest.importorskip("langgraph")
    from src.graph import native_tools_engine as nte
    from src.orchestrators.host_adapters import model_catalog

    class Client:
        async def _resolve_model(self):
            return "m"

    def gate(modalities):
        async def lookup(_model_id):
            return modalities

        monkeypatch.setattr(model_catalog, "model_input_modalities", lookup)
        return asyncio.run(nte._model_accepts_images(Client()))

    assert gate(frozenset({"text"})) is False
    assert gate(frozenset({"text", "image"})) is True
    assert gate(None) is True


class _SchemaExecutor(ToolExecutor):
    def __init__(self, properties: Dict[str, Any]):
        self._properties = properties

    async def get_openai_tools_for_ids(self, tool_ids, channel="llm"):
        return [{"type": "function", "function": {
            "name": tool_ids[0], "parameters": {"type": "object", "properties": self._properties}}}]


PROPS = {
    "media_type": {"type": "string", "description": "image|audio|video|document|unknown"},
    "mode": {"type": "string", "enum": ["semantic", "keyword"]},
    "query": {"type": "string", "description": "free text"},
}


def check(args: Dict[str, Any]):
    return asyncio.run(_SchemaExecutor(PROPS)._choice_argument_error("t", args))


def test_value_outside_declared_choices_is_reported():
    err = check({"media_type": "pdf"})
    assert err and "media_type='pdf'" in err and "image|audio|video|document|unknown" in err


def test_enum_violation_is_reported():
    assert "mode='fuzzy'" in check({"mode": "fuzzy"})


def test_choice_case_is_normalised_in_place():
    args = {"media_type": "Document", "mode": "KEYWORD", "query": "x|y"}
    assert check(args) is None
    assert args == {"media_type": "document", "mode": "keyword", "query": "x|y"}


def test_free_text_and_absent_choices_pass():
    assert check({"query": "a|b"}) is None
    assert check({}) is None
