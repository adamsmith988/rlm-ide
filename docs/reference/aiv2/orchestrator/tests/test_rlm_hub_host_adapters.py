"""Unit tests for hub RLM host adapters (additive; no AF / native_tools)."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest

from src.orchestrators.host_adapters.chat_session import inventory_from_session_files
from src.orchestrators.host_adapters.hub_sse import HubEventBridge
from src.orchestrators.host_pools import (
    empty_inventory,
    lookup_inventory_row,
    make_pool_host_closures,
    materialize_pool_files,
)


def test_inventory_from_session_files_merges_registry_and_artifacts():
    inv = inventory_from_session_files(
        file_registry=[
            {"artifact_id": "a1", "name": "doc.txt", "path": "/tmp/doc.txt"},
            {"artifact_id": "a1", "name": "dup.txt"},  # dedupe by aid
        ],
        artifacts=[{"id": "a2", "name": "fresh.pdf", "mime_type": "application/pdf"}],
    )
    assert inv["count"] == 2
    aids = {f.get("artifact_id") for f in inv["files"]}
    assert aids == {"a1", "a2"}


@pytest.mark.asyncio
async def test_materialize_and_pool_closures(tmp_path):
    cwd = str(tmp_path)
    inv = empty_inventory()
    local = tmp_path / "src.bin"
    local.write_bytes(b"hello-pool")
    inv["files"] = [
        {
            "filename": "src.bin",
            "path": str(local),
            "artifact_id": None,
        }
    ]
    inv["count"] = 1
    mat = await materialize_pool_files(cwd, inv, download=None)
    assert mat["materialized"] == 1
    assert os.path.isfile(os.path.join(cwd, "pool", "files", "src.bin"))
    assert os.path.isfile(os.path.join(cwd, "pool", "INDEX.md"))

    pool_defs = [
        {
            "name": "session",
            "pool_key": "session",
            "inventory": inv,
        }
    ]
    closures = make_pool_host_closures(
        artifact_dir=cwd, pool_defs=pool_defs, union_inventory=inv
    )
    listed = await closures["pool_list"]("session")
    assert listed["count"] == 1
    resolved = await closures["pool_resolve"]("src.bin", pool_key="session")
    assert resolved.get("ok") is not False
    assert os.path.isfile(str(resolved.get("path") or ""))
    row = lookup_inventory_row(inv, "src.bin")
    assert row is not None


@pytest.mark.asyncio
async def test_materialize_skips_existing_and_reuses_cache(tmp_path, monkeypatch):
    """Same session must not re-download; second cwd reuses shared cache."""
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("AIV2_ARTIFACT_CACHE", str(cache_root))

    downloads: List[str] = []

    async def fake_download(aid: str, filename_hint: str) -> bytes:
        downloads.append(aid)
        return b"blob-" + aid.encode("utf-8")

    cwd1 = str(tmp_path / "sess1")
    inv1 = empty_inventory()
    inv1["files"] = [
        {"filename": "case.bin", "artifact_id": "aid-42", "path": None},
    ]
    inv1["count"] = 1
    mat1 = await materialize_pool_files(cwd1, inv1, download=fake_download)
    assert mat1["downloaded"] == 1
    assert mat1["materialized"] == 1
    assert downloads == ["aid-42"]
    dest1 = os.path.join(cwd1, "pool", "files", "case.bin")
    assert os.path.isfile(dest1)

    # Fresh inventory (no local path) — same session dir → session_hit, no download.
    inv1b = empty_inventory()
    inv1b["files"] = [
        {"filename": "case.bin", "artifact_id": "aid-42", "path": None},
    ]
    inv1b["count"] = 1
    mat1b = await materialize_pool_files(cwd1, inv1b, download=fake_download)
    assert mat1b["skipped"] == 1
    assert mat1b["downloaded"] == 0
    assert downloads == ["aid-42"]

    # New session → hardlink/copy from shared cache, still no download.
    cwd2 = str(tmp_path / "sess2")
    inv2 = empty_inventory()
    inv2["files"] = [
        {"filename": "case.bin", "artifact_id": "aid-42", "path": None},
    ]
    inv2["count"] = 1
    mat2 = await materialize_pool_files(cwd2, inv2, download=fake_download)
    assert mat2["cache_hits"] == 1
    assert mat2["downloaded"] == 0
    assert downloads == ["aid-42"]
    dest2 = os.path.join(cwd2, "pool", "files", "case.bin")
    assert open(dest2, "rb").read() == b"blob-aid-42"


@pytest.mark.asyncio
async def test_pool_resolve_lazy_binds_from_cache(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("AIV2_ARTIFACT_CACHE", str(cache_root))

    async def fake_download(aid: str, filename_hint: str) -> bytes:
        return b"lazy-" + aid.encode("utf-8")

    cwd = str(tmp_path / "sess")
    inv = empty_inventory()
    inv["files"] = [
        {"filename": "lazy.bin", "artifact_id": "aid-lazy", "path": None},
    ]
    # INDEX only — no eager materialize of this row's dest (simulate wipe).
    from src.orchestrators.host_pools import write_pool_index

    write_pool_index(cwd, inv)
    dest = os.path.join(cwd, "pool", "files", "lazy.bin")
    assert not os.path.isfile(dest)

    closures = make_pool_host_closures(
        artifact_dir=cwd,
        pool_defs=[{"name": "session", "pool_key": "session", "inventory": inv}],
        union_inventory=inv,
        download=fake_download,
    )
    resolved = await closures["pool_resolve"]("lazy.bin", pool_key="session")
    assert resolved.get("ok") is True
    assert os.path.isfile(str(resolved.get("path") or ""))
    assert open(resolved["path"], "rb").read() == b"lazy-aid-lazy"


def test_hub_event_bridge_skips_prior_logs_on_continue():
    class _Live:
        log_lines = ["[kernel] ready", "[start] RLM run"]
        last_content = "prev answer"
        last_reasoning = "prev think"

    bridge = HubEventBridge()
    bridge.seed_from_live_session(_Live())
    out = bridge.map_event(
        {
            "kind": "stream_progress",
            "reasoning_len": len("prev think"),
            "content_len": len("prev answer"),
            "rlm": {
                "last_reasoning": "prev think",
                "last_content": "prev answer",
                "log_lines": list(_Live.log_lines),
            },
        }
    )
    assert out == []
    # New LLM stream resets answer cursors; prior terminal lines stay skipped.
    assert bridge.map_event({"kind": "stream_begin"}) == []
    out2 = bridge.map_event(
        {
            "kind": "stream_progress",
            "reasoning_len": 0,
            "content_len": 5,
            "rlm": {
                "last_reasoning": "",
                "last_content": "hello",
                "log_lines": list(_Live.log_lines) + ["[ipython] x=1"],
            },
        }
    )
    texts = [e.get("content") for e in out2 if e.get("type") == "thinking"]
    answers = [e.get("content") for e in out2 if e.get("type") == "answer"]
    assert any("ipython" in str(t) for t in texts)
    assert not any("kernel] ready" in str(t) for t in texts)
    assert "hello" in "".join(str(a) for a in answers)


def test_hub_event_bridge_content_len_zero_does_not_replay_prior_answer():
    """content_len=0 must not fall back to len(prev last_content) after stream_begin."""

    class _Live:
        log_lines = ["[kernel] ready"]
        last_content = "prev answer that must not reappear"
        last_reasoning = ""

    bridge = HubEventBridge()
    bridge.seed_from_live_session(_Live())
    assert bridge.map_event({"kind": "stream_begin"}) == []
    # Heartbeat / empty buffer: content_len=0 while rlm.last_content still holds prior.
    out = bridge.map_event(
        {
            "kind": "stream_progress",
            "reasoning_len": 0,
            "content_len": 0,
            "rlm": {
                "last_reasoning": "",
                "last_content": _Live.last_content,
                "log_lines": list(_Live.log_lines),
            },
        }
    )
    answers = [e.get("content") for e in out if e.get("type") == "answer"]
    assert answers == []
    assert bridge.collected_answer == ""
    # Real new tokens arrive.
    out2 = bridge.map_event(
        {
            "kind": "stream_progress",
            "reasoning_len": 0,
            "content_len": 4,
            "rlm": {
                "last_reasoning": "",
                "last_content": "next",
                "log_lines": list(_Live.log_lines),
            },
        }
    )
    joined = "".join(str(e.get("content")) for e in out2 if e.get("type") == "answer")
    assert joined == "next"
    assert "prev answer" not in bridge.collected_answer



def test_hub_event_bridge_stream_begin_keeps_one_thinking_block():
    """stream_begin must not open a second thinking card mid-turn."""
    bridge = HubEventBridge()
    out1 = bridge.map_event(
        {
            "kind": "stream_progress",
            "reasoning_len": 0,
            "content_len": 0,
            "rlm": {"log_lines": ["[kernel] ready"]},
        }
    )
    assert [e["type"] for e in out1] == ["thinking_start", "thinking"]
    assert bridge._thinking_open is True
    assert bridge.collected_thinking.startswith("[kernel] ready")

    assert bridge.map_event({"kind": "stream_begin"}) == []
    assert bridge._thinking_open is True
    assert "[kernel] ready" in bridge.collected_thinking

    out2 = bridge.map_event(
        {
            "kind": "stream_progress",
            "reasoning_len": 0,
            "content_len": 0,
            "rlm": {"log_lines": ["[kernel] ready", "[chat] turn 1"]},
        }
    )
    types2 = [e["type"] for e in out2]
    assert "thinking_start" not in types2
    assert types2 == ["thinking"]
    assert "[chat] turn 1" in "".join(e["content"] for e in out2)


def test_hub_event_bridge_llm_reconnect_drains_log_once():
    bridge = HubEventBridge()
    logs = ["[kernel] ready", "[reconnecting] model gateway 503 — retry 2/12 in 10s"]
    out = bridge.map_event(
        {
            "kind": "llm_reconnect",
            "preview": "model gateway 503 — reconnecting, retry 2/12 in 10s",
            "rlm": {"log_lines": list(logs)},
        }
    )
    thinking = [e["content"] for e in out if e.get("type") == "thinking"]
    assert len(thinking) == 2
    assert "[reconnecting]" in thinking[1]
    # Later stream_progress must not replay the same lines.
    out2 = bridge.map_event(
        {
            "kind": "stream_progress",
            "rlm": {"log_lines": list(logs)},
        }
    )
    assert out2 == []


def test_hub_event_bridge_stream_progress():
    bridge = HubEventBridge()
    ev1 = {
        "kind": "stream_progress",
        "reasoning_len": 5,
        "content_len": 0,
        "rlm": {"last_reasoning": "hello", "last_content": "", "log_lines": []},
    }
    out1 = bridge.map_event(ev1)
    types1 = [e["type"] for e in out1]
    assert "thinking_start" in types1
    assert "thinking" in types1
    assert bridge.collected_thinking == "hello"

    ev2 = {
        "kind": "stream_progress",
        "reasoning_len": 5,
        "content_len": 3,
        "rlm": {"last_reasoning": "hello", "last_content": "bye", "log_lines": ["[x]"]},
    }
    out2 = bridge.map_event(ev2)
    types2 = [e["type"] for e in out2]
    assert "thinking_end" in types2 or "answer" in types2
    assert any(e["type"] == "answer" for e in out2) or any(
        e["type"] == "thinking" for e in out2
    )
    # Never emit both answer and token for the same piece (UI doubles text).
    assert "token" not in types2
    answer_chunks = [e["content"] for e in out2 if e.get("type") == "answer"]
    assert "".join(answer_chunks) == "bye" or not answer_chunks


def test_manifest_settings_ui_normalize():
    from src.orchestrators.contract import normalize_manifest
    import json
    from pathlib import Path

    raw = json.loads(
        (Path(__file__).resolve().parents[1] / "src/orchestrators/rlm/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    m = normalize_manifest(raw)
    assert m.id == "rlm"
    assert isinstance(m.settings_ui, list) and len(m.settings_ui) > 0
    d = m.to_dict()
    assert "settings_ui" in d and len(d["settings_ui"]) > 0


def test_attached_sse_events_shapes():
    from src.orchestrators.host_adapters.deliverables import attached_sse_events

    events = attached_sse_events(
        [
            {"id": "a1", "name": "out.txt", "path": "a1"},
            {"id": "a2", "name": "plot.png", "path": "a2"},
        ],
        base_registry_size=3,
    )
    assert len(events) == 2
    assert events[0]["type"] == "session_file_attached"
    assert events[0]["file"]["id"] == "a1"
    assert events[0]["registry_size"] == 4
    assert events[1]["registry_size"] == 5


@pytest.mark.asyncio
async def test_sync_rlm_deliverables_to_chat_upload_and_attach(monkeypatch, tmp_path):
    from src.orchestrators.host_adapters import deliverables as mod
    import src.agent_turn as at
    import src.orchestrators.rlm.runtime.persist as persist_mod

    class FakeSess:
        def __init__(self) -> None:
            self.session_id = "rlm_x"
            self.artifact_dir = str(tmp_path)
            self.deliverable_artifacts: Dict[str, Any] = {}
            self._child_sessions: Dict[str, Any] = {}

    sess = FakeSess()

    intent = {
        "op": "new_blob",
        "filename": "report.md",
        "local_path": str(tmp_path / "report.md"),
        "mime_type": "text/markdown",
        "metadata": {"relpath": "report.md"},
    }

    async def fake_intents(_s: Any) -> List[Dict[str, Any]]:
        return [dict(intent)]

    remembered: List[Any] = []

    def fake_remember(s: Any, entries: List[Dict[str, Any]]) -> None:
        remembered.append((s, entries))

    def fake_persist_files(_s: Any) -> None:
        return None

    async def fake_upload(
        intents: List[Dict[str, Any]],
        *,
        user_id: str,
        chat_session_id: str,
        run_id: str,
    ) -> List[Dict[str, Any]]:
        assert user_id == "user-1"
        assert chat_session_id == "chat-1"
        assert intents and intents[0]["filename"] == "report.md"
        return [
            {
                "op": "new_blob",
                "artifact_id": "art-99",
                "filename": "report.md",
                "mime_type": "text/markdown",
            }
        ]

    class _Art:
        def __init__(self, **kw: Any) -> None:
            self.artifact_id = kw.get("artifact_id")
            self.filename = kw.get("filename")
            self.mime_type = kw.get("mime_type")

    async def fake_attach(*, session_id: str, user_id: str, artifact: Any) -> Dict[str, Any]:
        return {
            "id": artifact.artifact_id,
            "path": artifact.artifact_id,
            "name": artifact.filename or artifact.artifact_id,
            "mime_type": artifact.mime_type,
        }

    monkeypatch.setattr(persist_mod, "persist_intents_for_session_async", fake_intents)
    monkeypatch.setattr(persist_mod, "persist_session_files", fake_persist_files)
    monkeypatch.setattr(persist_mod, "remember_session_deliverables", fake_remember)
    monkeypatch.setattr(mod, "_upload_intents", fake_upload)
    monkeypatch.setattr(at, "AgentTurnArtifact", _Art)
    monkeypatch.setattr(at, "attach_result_artifact", fake_attach)

    attached = await mod.sync_rlm_deliverables_to_chat(
        orch_session=sess,
        chat_session_id="chat-1",
        user_id="user-1",
        run_id="chat-1",
    )
    assert len(attached) == 1
    assert attached[0]["id"] == "art-99"
    assert attached[0]["name"] == "report.md"
    assert remembered and remembered[0][1][0]["artifact_id"] == "art-99"


def test_read_tools_binding_from_rlm_manifest():
    from pathlib import Path
    import json

    from src.orchestrators.contract import normalize_manifest
    from src.orchestrators.host_adapters.tools_binding import read_tools_binding

    raw = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "src/orchestrators/rlm/manifest.json"
        ).read_text(encoding="utf-8")
    )
    m = normalize_manifest(raw)
    binding = read_tools_binding(m)
    assert binding is not None
    assert binding.mode == "kernel_callable"
    assert binding.config_key == "tool_ids"
    assert binding.is_kernel_callable


def test_read_session_binding_durable_stable_id():
    from pathlib import Path
    import json

    from src.orchestrators.contract import normalize_manifest
    from src.orchestrators.host_adapters.session_binding import (
        engine_session_id_for,
        read_session_binding,
    )

    raw = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "src/orchestrators/rlm/manifest.json"
        ).read_text(encoding="utf-8")
    )
    m = normalize_manifest(raw)
    binding = read_session_binding(m)
    assert binding is not None
    assert binding.is_durable
    assert binding.next_op == "continue"
    a = engine_session_id_for(binding, chat_session_id="chat-abc", orchestrator_id="rlm")
    b = engine_session_id_for(binding, chat_session_id="chat-abc", orchestrator_id="rlm")
    assert a == b
    assert a.startswith("rlm_chat_")
    assert "chat-abc" in a


@pytest.mark.asyncio
async def test_resolve_tool_ids_override_wins(monkeypatch):
    from src.orchestrators.host_adapters.tools_binding import (
        ToolsBinding,
        resolve_tool_ids_for_run,
    )

    binding = ToolsBinding(
        mode="kernel_callable",
        config_key="tool_ids",
        selection="host_provided",
    )

    class FakeExec:
        async def get_tools_manifest(self):
            return [
                {"id": "session_files", "channels": ["llm"]},
                {"id": "session_read_file", "channels": ["llm"]},
                {"id": "other_tool", "channels": ["llm"]},
            ]

    async def fake_allowed(manifest, user_id, roles=None):
        return list(manifest)

    def fake_select(tools, **kwargs):
        # Would select everything — override must win instead.
        return list(tools)

    monkeypatch.setattr(
        "src.auth.permissions_client.resolve_allowed_tools", fake_allowed
    )
    monkeypatch.setattr(
        "src.tools.llm_tool_selection.select_llm_tools_for_turn", fake_select
    )

    ids = await resolve_tool_ids_for_run(
        binding=binding,
        tool_executor=FakeExec(),
        user_id="u1",
        user_message="list files",
        orchestrator_config={"tool_ids": ["session_files", "missing_tool"]},
    )
    assert ids == ["session_files"]


@pytest.mark.asyncio
async def test_resolve_tool_ids_auto_select_when_no_override(monkeypatch):
    from src.orchestrators.host_adapters.tools_binding import (
        ToolsBinding,
        resolve_tool_ids_for_run,
    )

    binding = ToolsBinding(
        mode="kernel_callable",
        config_key="tool_ids",
        selection="host_provided",
    )

    class FakeExec:
        async def get_tools_manifest(self):
            return [
                {"id": "session_files", "channels": ["llm"]},
                {"id": "session_read_file", "channels": ["llm"]},
            ]

    async def fake_allowed(manifest, user_id, roles=None):
        return list(manifest)

    def fake_select(tools, **kwargs):
        assert kwargs.get("active_plugin") == "smart_search"
        return [t for t in tools if t["id"] == "session_files"]

    monkeypatch.setattr(
        "src.auth.permissions_client.resolve_allowed_tools", fake_allowed
    )
    monkeypatch.setattr(
        "src.tools.llm_tool_selection.select_llm_tools_for_turn", fake_select
    )

    ids = await resolve_tool_ids_for_run(
        binding=binding,
        tool_executor=FakeExec(),
        user_id="u1",
        user_message="what files?",
        orchestrator_config={},
        active_plugin="smart_search",
    )
    assert ids == ["session_files"]


@pytest.mark.asyncio
async def test_make_tools_host_port_calls_executor():
    from src.orchestrators.host_adapters.tools_binding import make_tools_host_port

    calls: List[Any] = []

    class FakeExec:
        async def execute(self, tool_id, args, **kwargs):
            calls.append((tool_id, args, kwargs))
            return {"ok": True}

    fn = make_tools_host_port(
        FakeExec(), session_id="s1", user_id="u1", user_message="hi"
    )
    out = await fn("session_files", {"limit": 5})
    assert out == {"ok": True}
    assert calls[0][0] == "session_files"
    assert calls[0][1] == {"limit": 5}
    assert calls[0][2]["session_id"] == "s1"
