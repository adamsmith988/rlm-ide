"""Native engine reply-language anchor: script detection, configurable template, prompt placement."""

from __future__ import annotations

import pytest

from src.config import settings

pytest.importorskip("langgraph")
from src.graph import native_tools_engine as ne  # noqa: E402


def test_dominant_script():
    assert ne._dominant_script("Is quarterly_report.pdf really a PDF?") == "LATIN"
    assert ne._dominant_script("آیا این فایل واقعاً PDF است؟") == "ARABIC"
    assert ne._dominant_script("Привет, это файл report.pdf") == "CYRILLIC"
    assert ne._dominant_script("12345 !?") == ""


def test_hint_is_configurable_and_last_in_prompt(monkeypatch):
    hint = ne._reply_language_hint("Which camera took photo.jpg?")
    assert "Latin" in hint
    prompt = ne.build_native_system_prompt(memory_prompt_block="## Recalled memory\n- x", reply_language_hint=hint)
    assert prompt.endswith(hint)
    monkeypatch.setattr(settings, "NATIVE_REPLY_LANGUAGE_HINT", "")
    assert ne._reply_language_hint("Which camera?") == ""
    monkeypatch.setattr(settings, "NATIVE_REPLY_LANGUAGE_HINT", "Reply in {script}.")
    assert ne._reply_language_hint("سلام") == "Reply in Arabic."


def test_user_note_appends_to_model_copy_and_is_configurable(monkeypatch):
    monkeypatch.setattr(settings, "NATIVE_REPLY_LANGUAGE_USER_NOTE", "[lang {script}]")
    assert ne._with_reply_language_note("What date?", "What date?") == "What date?\n\n[lang Latin]"
    parts = ne._with_reply_language_note([{"type": "text", "text": "سلام"}], "سلام")
    assert parts[-1] == {"type": "text", "text": "[lang Arabic]"} and len(parts) == 2
    monkeypatch.setattr(settings, "NATIVE_REPLY_LANGUAGE_USER_NOTE", "")
    assert ne._with_reply_language_note("What date?", "What date?") == "What date?"
