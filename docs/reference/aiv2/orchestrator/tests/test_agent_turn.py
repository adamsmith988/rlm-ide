"""Tests for orchestrator agent_turn (chat handoff of Agent Flow results)."""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict, List
from unittest.mock import patch

ORCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ORCH_ROOT not in sys.path:
    sys.path.insert(0, ORCH_ROOT)


def _load_mod():
    class _Settings:
        STORAGE_URL = "http://storage.test"
        INTERNAL_API_KEY = "test-key"
        SERVICE_NAME = "orchestrator"

    import src.config as config

    real_settings = config.settings
    config.settings = _Settings()  # type: ignore[assignment]

    class _Client:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        async def add_message(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "id": f"msg-{len(self.calls)}",
                "role": kwargs["role"],
                "content": kwargs["content"],
                "attachments": kwargs.get("attachments"),
                "metadata": kwargs.get("metadata"),
            }

    client = _Client()

    import src.chat_history as chat_history

    chat_history.get_chat_history_client = lambda: client  # type: ignore

    # Fresh import of agent_turn after stubs.
    sys.modules.pop("src.agent_turn", None)
    import src.agent_turn as agent_turn

    # agent_turn bound the stub at import; later tests need the real settings.
    config.settings = real_settings  # type: ignore[assignment]
    agent_turn._test_chat_client = client  # type: ignore[attr-defined]
    return agent_turn


def test_record_agent_turn_attaches_and_saves():
    mod = _load_mod()
    req = mod.AgentTurnRequest(
        user_id="user-1",
        user_content="run the report",
        assistant_content="Here is the report.",
        user_attachments=[
            mod.AgentTurnAttachment(
                id="up-1", name="in.pdf", path="up-1", mime_type="application/pdf"
            )
        ],
        result_artifacts=[
            mod.AgentTurnArtifact(
                artifact_id="art-out-1",
                filename="report.docx",
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                size_bytes=100,
            )
        ],
        agent=mod.AgentTurnRef(
            flow_id="flow-abc",
            flow_name="Report Agent",
            run_id="run-xyz",
            flow_session_id="afs-1",
        ),
    )

    async def fake_storage(method, endpoint, **kwargs):
        if method == "GET" and endpoint.endswith("/chat/sessions/sess-1"):
            return {"id": "sess-1", "user_id": "user-1"}
        if method == "POST" and endpoint.endswith("/artifacts/attach"):
            return {
                "status": "attached",
                "file": {
                    "id": "art-out-1",
                    "artifact_id": "art-out-1",
                    "filename": "report.docx",
                    "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "size_bytes": 100,
                },
            }
        raise AssertionError(f"unexpected {method} {endpoint}")

    async def run():
        with patch.object(mod, "_storage_request", side_effect=fake_storage):
            return await mod.record_agent_turn(session_id="sess-1", request=req)

    result = asyncio.run(run())
    assert result.user_message["role"] == "user"
    assert result.assistant_message["role"] == "assistant"
    assert result.assistant_message["metadata"]["source"] == "agent_flow"
    assert result.assistant_message["metadata"]["flow_id"] == "flow-abc"
    assert result.assistant_message["metadata"]["run_id"] == "run-xyz"
    assert len(result.attached_artifacts) == 1
    assert result.attached_artifacts[0]["id"] == "art-out-1"
    client = mod._test_chat_client
    assert len(client.calls) == 2
    assert client.calls[1]["metadata"]["flow_name"] == "Report Agent"


def test_record_agent_turn_rejects_foreign_session():
    mod = _load_mod()
    req = mod.AgentTurnRequest(
        user_id="user-1",
        user_content="hi",
        assistant_content="ok",
        agent=mod.AgentTurnRef(flow_id="f1", run_id="r1"),
    )

    async def fake_storage(method, endpoint, **kwargs):
        return {"id": "sess-1", "user_id": "other-user"}

    async def run():
        with patch.object(mod, "_storage_request", side_effect=fake_storage):
            await mod.record_agent_turn(session_id="sess-1", request=req)

    try:
        asyncio.run(run())
        raise AssertionError("expected PermissionError")
    except PermissionError:
        pass


def test_record_agent_turn_session_not_found():
    mod = _load_mod()
    req = mod.AgentTurnRequest(
        user_id="user-1",
        user_content="hi",
        assistant_content="ok",
        agent=mod.AgentTurnRef(flow_id="f1", run_id="r1"),
    )

    async def fake_storage(method, endpoint, **kwargs):
        return None

    async def run():
        with patch.object(mod, "_storage_request", side_effect=fake_storage):
            await mod.record_agent_turn(session_id="missing", request=req)

    try:
        asyncio.run(run())
        raise AssertionError("expected LookupError")
    except LookupError:
        pass
