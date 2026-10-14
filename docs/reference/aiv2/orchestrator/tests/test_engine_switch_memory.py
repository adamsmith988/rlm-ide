"""Switching engines mid-chat: RLM's own transcript follows the shared hub conversation;
zero-shot runs carry no conversation memory."""

from __future__ import annotations

from typing import Any, Dict, List

from src.orchestrators.host_adapters import rlm_chat_memory as mem


class Live:
    def __init__(self, artifact_dir: str, messages: List[Dict[str, Any]] | None = None) -> None:
        self.artifact_dir = artifact_dir
        self.messages = list(messages or [])
        self.user_content = None

    def build_system_prompt(self, base: str) -> str:
        return base


def hub(*pairs: tuple) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for q, a in pairs:
        rows += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return rows


def test_rlm_catches_up_with_turns_other_engines_served(tmp_path):
    live = Live(str(tmp_path), [{"role": "system", "content": "sys"}, *hub(("my code is 7731", "Noted: 7731."))])
    mem.record_served_turn(live, "my code is 7731", "Noted: 7731.")
    conv = {"summary": "", "messages": hub(("my code is 7731", "Noted: 7731."), ("I live in Graz", "Graz, got it."))}
    mode, user_on = mem.resume_or_seed_live_memory(
        live, user_message="where do I live?", base_system="sys", chat_history=None, self_compact=conv
    )
    assert (mode, user_on) == ("warm", False)
    assert live.messages[-2:] == hub(("I live in Graz", "Graz, got it."))


def test_rlm_in_sync_appends_nothing(tmp_path):
    live = Live(str(tmp_path), [{"role": "system", "content": "s"}, *hub(("a", "b"))])
    mem.record_served_turn(live, "a", "b")
    mem.resume_or_seed_live_memory(live, user_message="c", base_system="s", chat_history=None,
                                   self_compact={"summary": "", "messages": hub(("a", "b"))})
    assert len(live.messages) == 3


def test_rlm_reseeds_from_the_checkpoint_when_its_turns_were_compacted_away(tmp_path):
    live = Live(str(tmp_path), [{"role": "system", "content": "s"}, *hub(("old q", "old a"))])
    mem.record_served_turn(live, "old q", "old a")
    conv = {"summary": "- user code: 7731", "messages": hub(("later q", "later a"))}
    mode, user_on = mem.resume_or_seed_live_memory(live, user_message="now", base_system="s",
                                                   chat_history=None, self_compact=conv)
    assert (mode, user_on) == ("reseeded", True)
    assert live.messages[0] == {"role": "system", "content": "s"}
    assert "7731" in live.messages[1]["content"] and live.messages[1]["role"] == "system"
    assert live.messages[2:] == hub(("later q", "later a")) + [{"role": "user", "content": "now"}]


def test_rlm_seeds_a_new_session_from_the_hub_conversation_not_the_window(tmp_path):
    live = Live(str(tmp_path))
    conv = {"summary": "- name: Ada", "messages": hub(("q", "a"))}
    window = hub(("unrelated", "window row"))
    mode, user_on = mem.resume_or_seed_live_memory(live, user_message="hi", base_system="s",
                                                   chat_history=window, self_compact=conv)
    assert (mode, user_on) == ("seeded", True)
    text = " ".join(str(m["content"]) for m in live.messages)
    assert "Ada" in text and "window row" not in text


def test_served_turn_record_survives_a_restart(tmp_path):
    a = Live(str(tmp_path))
    mem.record_served_turn(a, "q", "Answer   text")
    b = Live(str(tmp_path))
    served = mem.served_turns(b)
    assert mem.turn_key("answer text") in served["answers"]
    assert mem.last_served_index(hub(("q", "Answer text"), ("x", "y")), served) == 1


def test_untracked_live_session_is_left_alone(tmp_path):
    live = Live(str(tmp_path), [{"role": "system", "content": "s"}, *hub(("a", "b"))])
    mem.resume_or_seed_live_memory(live, user_message="c", base_system="s", chat_history=None,
                                   self_compact={"summary": "", "messages": hub(("x", "y"))})
    assert len(live.messages) == 3
