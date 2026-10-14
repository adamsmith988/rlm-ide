"""Long input: parallel part readers, task checklist, digest, overrides, degradation."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List

from src.context.long_input import LongInputReader, LongInputSettings, load_prompts
from src.orchestrators._shared.llm_roles import LLMReply

NEEDLE = "The vault code is ZEBRA-7731."


def run(coro):
    return asyncio.run(coro)


class FakeLLM:
    """Checklist calls list tasks; worker calls echo needles and indented code found in their part."""

    def __init__(self, *, fail_parts: tuple = (), fail_all: bool = False, notes_pad: int = 0) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.active = 0
        self.peak = 0
        self.fail_parts = fail_parts
        self.fail_all = fail_all
        self.notes_pad = notes_pad

    async def __call__(self, messages, **kw) -> LLMReply:
        self.calls.append({"messages": messages, **kw})
        system, user = messages[0]["content"], messages[1]["content"]
        if system == load_prompts()["checklist"]:
            return LLMReply(text=json.dumps({"tasks": ["What is the vault code?", "Reply in one line."]}))
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        part = int(re.search(r"Part (\d+) of", user).group(1))
        if self.fail_all or part in self.fail_parts:
            return LLMReply(text="", ok=False, error="boom")
        notes = [f"part {part} read"]
        if "ZEBRA-7731" in user:
            notes.append(NEEDLE)
        if "    return x * 2" in user:
            notes.append("code: def f(x):\n    return x * 2")
        tasks = ["Also list the colours."] if "list the colours" in user else []
        return LLMReply(text=json.dumps({"tasks": tasks, "notes": "\n".join(notes) + "." * self.notes_pad}))


def reader(llm: FakeLLM, **over: Any) -> LongInputReader:
    cfg = {"context_length": 4_000, "long_input_chars_per_token": 1.0, "long_input_max_parallel": 2}
    cfg.update({f"long_input_{k}": v for k, v in over.items()})
    r = LongInputReader.for_turn(cfg, llm=llm)
    assert r is not None
    return r


def long_message(chars: int = 6_000) -> str:
    filler = "Lorem ipsum dolor sit amet. " * (chars // 28)
    mid = len(filler) // 2
    return (
        "Please read this report and tell me the vault code, in one line.\n\n"
        + filler[:mid]
        + f"\n{NEEDLE}\nAlso, please list the colours.\n"
        + "def f(x):\n    return x * 2\n"
        + filler[mid:]
        + "\nThanks."
    )


def test_settings_follow_config_and_overrides():
    base = LongInputSettings.resolve(None)
    assert base.enabled is True and base.max_parallel >= 1
    over = LongInputSettings.resolve({"long_input_max_parallel": "7", "long_input_enabled": "false"})
    assert over.max_parallel == 7 and over.enabled is False
    assert LongInputReader.for_turn({"long_input_enabled": False}) is None


def test_small_message_is_left_alone():
    llm = FakeLLM()
    r = reader(llm)
    assert not r.needed("short question")
    assert run(r.read("short question")) is None
    assert llm.calls == []


def test_trigger_is_relative_to_window_or_absolute():
    r = reader(FakeLLM(), trigger_ratio=0.5)
    assert r.trigger_tokens == 2_000
    assert reader(FakeLLM(), trigger_tokens=123).trigger_tokens == 123
    assert reader(FakeLLM()).window_tokens == 4_000


def test_middle_fact_and_tasks_survive_in_digest():
    llm = FakeLLM()
    r = reader(llm, chunk_ratio=0.3, keep_head_chars=200, keep_tail_chars=50)
    msg = long_message()
    res = run(r.read(msg))
    assert res is not None and res.parts >= 4 and res.failed_parts == 0
    assert NEEDLE in res.message
    assert "def f(x):\n    return x * 2" in res.message
    assert NEEDLE not in msg[:200] and NEEDLE not in msg[-50:]
    assert msg[:200] in res.message and msg[-50:] in res.message
    assert res.tasks[:2] == ["What is the vault code?", "Reply in one line."]
    assert "Also list the colours." in res.tasks
    assert 1 < llm.peak <= 2
    meta = res.meta()
    assert meta["original_chars"] == len(msg) and meta["parts"] == res.parts


def test_workers_get_raw_text_and_checklist():
    llm = FakeLLM()
    r = reader(llm, chunk_ratio=0.3)
    run(r.read(long_message()))
    workers = [c for c in llm.calls if c["messages"][0]["content"] == load_prompts()["worker"]]
    assert workers and all("What is the vault code?" in c["messages"][1]["content"] for c in workers)
    assert any("def f(x):\n    return x * 2" in c["messages"][1]["content"] for c in workers)


def test_failed_part_keeps_verbatim_excerpt():
    llm = FakeLLM(fail_parts=(1,))
    r = reader(llm, chunk_ratio=0.3)
    msg = long_message()
    res = run(r.read(msg))
    assert res is not None and res.failed_parts == 1
    assert "Lorem ipsum" in res.message


def test_all_workers_failing_keeps_original():
    r = reader(FakeLLM(fail_all=True), chunk_ratio=0.3)
    assert run(r.read(long_message())) is None


def test_oversized_notes_trigger_reduce_pass_and_fit_budget():
    llm = FakeLLM(notes_pad=1_500)
    r = reader(llm, chunk_ratio=0.3, digest_max_ratio=0.5, max_reduce_passes=1)
    res = run(r.read(long_message()))
    assert res is not None and res.reduce_passes == 1
    notes_part = res.message.split("## Notes on the full message, in order", 1)[1]
    assert len(notes_part) < 0.5 * 4_000 + 2_000


def test_prompts_dir_is_configurable(tmp_path):
    (tmp_path / "digest.md").write_text("CUSTOM {parts} {tasks} {head} {notes} {tail} {chars}", encoding="utf-8")
    res = run(reader(FakeLLM(), chunk_ratio=0.3, prompts_dir=str(tmp_path)).read(long_message()))
    assert res is not None and res.message.startswith("CUSTOM ")


def test_workspace_swaps_message_and_keeps_original_on_failure():
    from src.session_workspace import SessionWorkspaceState

    msg = long_message()
    ws = SessionWorkspaceState(session_id="s", user_id="u", user_message=msg)
    cfg = {"context_length": 4_000, "long_input_chars_per_token": 1.0, "long_input_chunk_ratio": 0.3}
    meta = run(ws.apply_long_input(cfg, llm=FakeLLM()))
    assert meta and meta["parts"] >= 4 and ws.long_input == meta
    assert NEEDLE in ws.user_message and ws.user_message != msg
    assert ws.to_engine_kwargs()["user_message"] == ws.user_message

    ws2 = SessionWorkspaceState(session_id="s", user_id="u", user_message=msg)
    assert run(ws2.apply_long_input(cfg, llm=FakeLLM(fail_all=True))) is None
    assert ws2.user_message == msg and ws2.long_input is None

    ws3 = SessionWorkspaceState(session_id="s", user_id="u", user_message=msg)
    assert run(ws3.apply_long_input({**cfg, "long_input_enabled": False}, llm=FakeLLM())) is None
    assert ws3.user_message == msg
