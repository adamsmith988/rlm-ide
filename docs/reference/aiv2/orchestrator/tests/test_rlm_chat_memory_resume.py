"""Hub RLM multi-turn memory: disk hydrate + chat_history seed (AF ladder)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.orchestrators.host_adapters.rlm_chat_memory import (
    hub_history_prior_turns,
    messages_have_memory,
    resume_or_seed_live_memory,
    seed_session_from_hub_history,
)


class _FakeLive:
    def __init__(self, artifact_dir: str, messages: Optional[List[Dict[str, Any]]] = None):
        self.artifact_dir = artifact_dir
        self.messages = list(messages or [])
        self.user_content: Any = "stale"
        self._transcript_written = 0
        self._hydrate = False
        self._hydrate_msgs: List[Dict[str, Any]] = []

    def build_system_prompt(self, base: str) -> str:
        return str(base or "SYS")

    def hydrate_messages_from_disk(self, *, base_system: str = "") -> bool:
        if not self._hydrate:
            return False
        system = self.build_system_prompt(base_system)
        self.messages = [{"role": "system", "content": system}] + list(self._hydrate_msgs)
        return messages_have_memory(self.messages)


def test_messages_have_memory():
    assert not messages_have_memory([])
    assert not messages_have_memory([{"role": "system", "content": "s"}])
    assert messages_have_memory([{"role": "user", "content": "hi"}])


def test_hub_history_drops_duplicate_current_user():
    hist = [
        {"role": "user", "content": "make file"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "continue"},
    ]
    prior = hub_history_prior_turns(hist, current_user="continue")
    assert prior == [
        {"role": "user", "content": "make file"},
        {"role": "assistant", "content": "done"},
    ]


def test_seed_from_hub_history(tmp_path):
    live = _FakeLive(str(tmp_path))
    ok = seed_session_from_hub_history(
        live,
        chat_history=[
            {"role": "user", "content": "write report.md"},
            {"role": "assistant", "content": "created"},
        ],
        user_message="finish it",
        base_system="SYS",
    )
    assert ok
    assert live.user_content is None
    roles = [m["role"] for m in live.messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert live.messages[-1]["content"] == "finish it"


def test_resume_warm_keeps_messages(tmp_path):
    live = _FakeLive(
        str(tmp_path),
        messages=[
            {"role": "system", "content": "s"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "ans"},
        ],
    )
    mode, user_on = resume_or_seed_live_memory(
        live,
        user_message="next",
        base_system="SYS",
        chat_history=None,
    )
    assert mode == "warm"
    assert user_on is False
    assert len(live.messages) == 3


def test_resume_from_disk_appends_user(tmp_path):
    live = _FakeLive(str(tmp_path))
    live._hydrate = True
    live._hydrate_msgs = [
        {"role": "user", "content": "build x.py"},
        {"role": "assistant", "content": "working"},
    ]
    mode, user_on = resume_or_seed_live_memory(
        live,
        user_message="keep going",
        base_system="SYS",
        chat_history=[{"role": "user", "content": "ignored when disk works"}],
    )
    assert mode == "resumed"
    assert user_on is True
    assert live.user_content is None
    assert live.messages[-1] == {"role": "user", "content": "keep going"}
    assert any(m.get("content") == "build x.py" for m in live.messages)


def test_resume_falls_back_to_chat_history(tmp_path):
    live = _FakeLive(str(tmp_path))
    mode, user_on = resume_or_seed_live_memory(
        live,
        user_message="again",
        base_system="SYS",
        chat_history=[
            {"role": "user", "content": "goal A"},
            {"role": "assistant", "content": "progress"},
            {"role": "user", "content": "again"},
        ],
    )
    assert mode == "seeded"
    assert user_on is True
    assert any(m.get("content") == "goal A" for m in live.messages)
    assert live.messages[-1]["content"] == "again"


def test_hydrate_messages_from_disk_roundtrip(tmp_path):
    from src.orchestrators.rlm.runtime.transcript import (
        FULL_TRANSCRIPT_NAME,
        append_message,
        hydrate_messages_from_disk,
    )

    art = tmp_path / "rlm"
    art.mkdir()
    append_message(str(art), {"role": "user", "content": "create notes.txt"})
    append_message(str(art), {"role": "assistant", "content": "writing notes"})
    assert (art / FULL_TRANSCRIPT_NAME).is_file()
    msgs = hydrate_messages_from_disk(str(art), system_prompt="SYS")
    assert msgs[0]["role"] == "system"
    assert any(m.get("content") == "create notes.txt" for m in msgs)
