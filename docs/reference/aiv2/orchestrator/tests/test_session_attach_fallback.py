import os
import sys
from typing import Any, Dict, List

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.executor import ToolExecutor


@pytest.mark.asyncio
async def test_session_attach_not_found_triggers_bounded_fallback() -> None:
    ex = ToolExecutor("http://tool-runner:8002")
    calls: List[str] = []

    async def fake_execute(*, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        calls.append(tool_name)
        if tool_name == "session_files":
            return {
                "ok": True,
                "data": {
                    "result": {
                        "data": {
                            "files": [
                                {"name": "jaafar_notes.txt"},
                                {"name": "report.pdf"},
                            ]
                        }
                    }
                },
            }
        if tool_name == "session_search_workspace":
            return {"ok": True, "data": {"result": {"matches": []}}}
        return {"ok": False, "error": "unexpected"}

    ex.execute = fake_execute  # type: ignore[assignment]
    recovered = await ex._recover_after_tool_failure(
        tool_name="session_attach",
        failed_result={"ok": False, "error": "No accessible file found for: path=/data/uploads/جعفر"},
        normalized_args={"path": "/data/uploads/جعفر"},
        session_id="s1",
        user_id="u1",
        group_id=None,
    )
    assert recovered is not None
    assert recovered["ok"] is True
    data = recovered["data"]
    assert data["recovered"] is True
    assert data["recovery_strategy"] == "session_attach_not_found_fallback_v1"
    assert "session_attach could not find an accessible storage file." in data["channels"]["llm"]
    assert calls == ["session_files", "session_search_workspace"]


@pytest.mark.asyncio
async def test_artifact_not_found_envelope_triggers_fallback() -> None:
    ex = ToolExecutor("http://tool-runner:8002")
    calls: List[str] = []

    async def fake_execute(*, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        calls.append(tool_name)
        if tool_name == "session_files":
            return {"ok": True, "data": {"result": {"data": {"files": []}}}}
        return {"ok": False, "error": "unexpected"}

    ex.execute = fake_execute  # type: ignore[assignment]
    recovered = await ex._recover_after_tool_failure(
        tool_name="session_attach",
        failed_result={
            "ok": False,
            "error": "artifact_not_found",
            "data": {
                "ok": False,
                "tool_id": "session_attach",
                "error": "artifact_not_found",
                "result": {
                    "ok": False,
                    "error": "artifact_not_found",
                    "channels": {
                        "llm": (
                            "No accessible file found for: path=/data/uploads/جعفر. "
                            "Try a different filename, full path, or case_id."
                        ),
                    },
                },
            },
        },
        normalized_args={"path": "/data/uploads/جعفر"},
        session_id="s1",
        user_id="u1",
        group_id=None,
    )
    assert recovered is not None
    assert recovered["ok"] is True
    assert calls == ["session_files"]


@pytest.mark.asyncio
async def test_non_matching_error_does_not_trigger_fallback() -> None:
    ex = ToolExecutor("http://tool-runner:8002")
    recovered = await ex._recover_after_tool_failure(
        tool_name="session_attach",
        failed_result={"ok": False, "error": "other_error"},
        normalized_args={"path": "/data/uploads/جعفر"},
        session_id="s1",
        user_id="u1",
        group_id=None,
    )
    assert recovered is None

