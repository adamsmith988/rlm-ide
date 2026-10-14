"""Unit tests for tool_sessions helpers and strip_session_blob_intents."""

from __future__ import annotations

import asyncio
import datetime as dt
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "services" / "storage" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_strip_session_blob_intents_keeps_deliverables():
    from modules.tools.tool_sessions import strip_session_blob_intents

    out = strip_session_blob_intents(
        [
            {"op": "new_blob", "filename": "session.json", "layout": "sessions"},
            {"op": "new_blob", "filename": "report.html", "layout": "default"},
        ]
    )
    assert [x["filename"] for x in out] == ["report.html"]


def test_upsert_tool_session_posts_storage():
    from modules.tools.tool_sessions import upsert_tool_session

    captured = {}

    class _Resp:
        status_code = 200
        content = b'{"id":"s1","session_id":"s1"}'

        def json(self):
            return {"id": "s1", "session_id": "s1"}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return _Resp()

    async def _run():
        with patch("modules.tools.tool_sessions.httpx.AsyncClient", _Client):
            row = await upsert_tool_session(
                envelope={
                    "ok": True,
                    "status": "success",
                    "flow_id": "f1",
                    "run_id": "r1",
                    "output_preview": "hi",
                },
                auth_context={"user_id": "u1"},
                plugin_id="agent_flow",
                tool_id="plugin_agent_flow_run",
                internal_api_key="k",
            )
        assert row["session_id"] == "s1"
        assert captured["url"].endswith("/tool-sessions")
        assert captured["json"]["plugin_id"] == "agent_flow"
        assert captured["json"]["record_id"] == "f1"
        assert captured["json"]["run_id"] == "r1"
        assert captured["headers"].get("X-Internal-Key") == "k"
        assert captured["headers"].get("X-Service-Name") == "tool-runner"

    asyncio.run(_run())


def test_tool_session_repository_row_shape():
    import importlib.util

    path = SRC / "repositories" / "tool_sessions.py"
    spec = importlib.util.spec_from_file_location("tool_sessions_repo", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    ToolSessionRepository = mod.ToolSessionRepository

    class FakeRow:
        def __init__(self, data):
            self._data = data

        def __getitem__(self, key):
            return self._data[key]

    repo = ToolSessionRepository(postgres=None)
    row = FakeRow(
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "user_id": "22222222-2222-2222-2222-222222222222",
            "plugin_id": "agent_flow",
            "tool_id": "plugin_agent_flow_run",
            "record_id": "f1",
            "run_id": "r1",
            "status": "success",
            "ok": True,
            "error": None,
            "output_preview": "hi",
            "payload": {"files": [{"filename": "a.html"}], "flow_id": "f1"},
            "created_at": dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
            "updated_at": dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
        }
    )
    out = repo._row_to_dict(row)
    assert out["session_id"] == "11111111-1111-1111-1111-111111111111"
    assert out["files"][0]["filename"] == "a.html"
    assert out["flow_id"] == "f1"

    # List path omits payload column entirely — must not explode or invent files.
    list_row = FakeRow(
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "user_id": "22222222-2222-2222-2222-222222222222",
            "plugin_id": "agent_flow",
            "tool_id": "plugin_agent_flow_run",
            "record_id": "f1",
            "run_id": "r1",
            "status": "success",
            "ok": True,
            "error": None,
            "output_preview": "hi",
            "created_at": dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
            "updated_at": dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
        }
    )
    listed = repo._row_to_dict(list_row, include_payload=False)
    assert listed["files"] == []
    assert "payload" not in listed
    assert listed["flow_id"] == "f1"
    assert listed["run_id"] == "r1"
