"""Unit tests for Storage persist layout helpers and ensure-chain models."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "services" / "storage" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from persist_layouts import (  # noqa: E402
    render_layout_template,
    resolve_layout_template,
    validate_layout_template,
)


def test_resolve_default_case_and_user_layouts():
    case_tmpl = resolve_layout_template(layout_name="default", layout_template=None, has_case=True)
    user_tmpl = resolve_layout_template(layout_name="default", layout_template=None, has_case=False)
    assert case_tmpl.startswith("case_importer/")
    assert user_tmpl.startswith("users/")


def test_custom_template_validated():
    with pytest.raises(ValueError):
        resolve_layout_template(
            layout_name=None,
            layout_template="tmp/{user_id}/x",
            has_case=False,
        )
    ok, _ = validate_layout_template(
        "users/{user_id}/plugins/{plugin_id}/sessions/{record_id}/{run_id}/{hex}_{filename}"
    )
    assert ok
    rendered = render_layout_template(
        "users/{user_id}/plugins/{plugin_id}/sessions/{record_id}/{run_id}/{hex}_{filename}",
        {
            "user_id": "u1",
            "plugin_id": "agent_flow",
            "record_id": "owner__flow-1",
            "run_id": "r1",
            "hex": "abcd",
            "filename": "session.json",
        },
    )
    assert rendered == "users/u1/plugins/agent_flow/sessions/owner__flow-1/r1/abcd_session.json"


def test_sessions_layout_is_caseless_platform_default():
    from persist_layouts import is_plugin_session_artifact

    case_tmpl = resolve_layout_template(layout_name="sessions", layout_template=None, has_case=True)
    user_tmpl = resolve_layout_template(layout_name="sessions", layout_template=None, has_case=False)
    assert case_tmpl == user_tmpl
    assert case_tmpl.startswith("users/")
    assert "/sessions/" in case_tmpl
    assert is_plugin_session_artifact(
        original_filename="session.json",
        storage_path="minio://bucket/users/u1/plugins/agent_flow/sessions/f/r/hex_session.json",
        metadata={"case_id": "c1", "kind": "plugin_session"},
    )
    assert is_plugin_session_artifact(
        original_filename="notes.txt",
        storage_path="users/u/plugins/x/sessions/a/b/hex_notes.txt",
        metadata={"layout": "sessions"},
    )
    assert not is_plugin_session_artifact(
        original_filename="MB.txt",
        storage_path="case_importer/c1/plugins/agent_flow/runs/r/hex_MB.txt",
        metadata={"case_id": "c1"},
    )


def test_users_layout_template_wins_over_has_case_flag():
    """Explicit users/ template must be kept even when has_case would prefer case_importer."""
    tmpl = (
        "users/{user_id}/plugins/{plugin_id}/flows/{flow_id}/runs/{run_id}/{hex}_{filename}"
    )
    resolved = resolve_layout_template(
        layout_name="default",
        layout_template=tmpl,
        has_case=True,
    )
    assert resolved == tmpl
    assert "case_importer" not in resolved
