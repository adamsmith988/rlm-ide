"""RLM run_lifetime=chat: turn memory without heartbeat loop."""

from __future__ import annotations

from src.orchestrators.rlm.schema_utils import (
    is_resident_run_lifetime,
    is_turn_wait_run_lifetime,
    normalize_run_lifetime,
)


def test_normalize_run_lifetime_chat():
    assert normalize_run_lifetime("chat") == "chat"
    assert normalize_run_lifetime("multi-turn") == "chat"
    assert normalize_run_lifetime("conversational") == "chat"
    assert normalize_run_lifetime("one_shot") == "one_shot"
    assert normalize_run_lifetime("heartbeat") == "heartbeat"
    assert is_resident_run_lifetime("chat")
    assert is_resident_run_lifetime("heartbeat")
    assert not is_resident_run_lifetime("one_shot")
    assert is_turn_wait_run_lifetime("chat")
    assert is_turn_wait_run_lifetime("one_shot")
    assert not is_turn_wait_run_lifetime("heartbeat")


def test_rlm_manifest_exposes_chat_lifetime():
    import json
    from pathlib import Path

    raw = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "src/orchestrators/rlm/manifest.json"
        ).read_text(encoding="utf-8")
    )
    enum = raw["config_schema"]["properties"]["run_lifetime"]["enum"]
    assert "chat" in enum
    widgets = {w["name"]: w for w in raw.get("settings_ui") or []}
    lifetime = widgets["run_lifetime"]
    values = {o["value"] for o in lifetime["options"]}
    assert values == {"chat", "one_shot", "heartbeat"}
    assert lifetime["default"] == "chat"
