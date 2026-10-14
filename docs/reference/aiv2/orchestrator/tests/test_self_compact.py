"""SelfCompact: rubric parsing, gates, compaction, overflow recovery, persistence."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Any, Dict, List, Optional

import pytest

from src.context import self_compact as sc
from src.context.self_compact import (
    COMPRESS,
    CONTINUE,
    SelfCompactController,
    SelfCompactSettings,
    load_prompts,
    parse_verdict,
    split_trajectory,
)

COMPRESS_REPLY = (
    'C1: Y -- "The codename is BLUEHERON."\n'
    'C2: Y -- 1. codename "BLUEHERON" | 2. sky "Rayleigh scattering" | 3. done "answered"\n'
    'C3: Y -- answered the sky question\n'
    "N1: N -- fewer than 4 attempts"
)
CONTINUE_REPLY = (
    'C1: N -- "Let me now check the file"\n'
    "C2: N -- list of values\n"
    "C3: Y -- new fact\n"
    "N1: N -- fewer than 4 attempts"
)


def run(coro):
    return asyncio.run(coro)


class FakeLLM:
    """Scripted ``chat`` replies; records every request."""

    def __init__(self, replies: List[Any], max_tokens: Optional[int] = None) -> None:
        self.replies = list(replies)
        self.calls: List[Dict[str, Any]] = []
        self.max_tokens = max_tokens

    async def chat(self, messages, **kwargs):
        self.calls.append({"messages": list(messages), **kwargs})
        reply = self.replies.pop(0) if self.replies else ""
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, dict):
            return reply
        return {"ok": True, "content": reply, "tokens_prompt": None}


def controller(llm: FakeLLM, *, ctx: int = 10_000, carried: Optional[Dict[str, Any]] = None, **over: Any):
    cfg = {"context_length": ctx, "self_compact_summary_chunk_ratio": 0}
    cfg.update({f"self_compact_{k}": v for k, v in over.items()})
    ctl = SelfCompactController.for_turn(cfg, llm_client=llm, carried=carried)
    assert ctl is not None
    return ctl


def convo(n_turns: int, chars: int = 200) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": "You are helpful."}]
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"Note {i}: code K{i:02d}. " + "x" * chars})
        msgs.append({"role": "assistant", "content": f"Noted K{i:02d}."})
    return msgs


# ---------------------------------------------------------------------------
# Settings and prompts
# ---------------------------------------------------------------------------


def test_settings_follow_config_and_overrides():
    base = SelfCompactSettings.resolve(None)
    assert base.backstop_ratio == sc.settings.SELF_COMPACT_BACKSTOP_RATIO
    st = SelfCompactSettings.resolve(
        {"self_compact_backstop_ratio": "0.5", "self_compact_enabled": "false", "self_compact_rubric": "math"}
    )
    assert st.backstop_ratio == 0.5 and st.enabled is False and st.rubric == "math"
    assert SelfCompactSettings.resolve({"self_compact_probe_period": ""}).probe_period == base.probe_period


@pytest.mark.parametrize("bad", ["../etc", "nope", "Chat"])
def test_unknown_rubric_rejected(bad):
    with pytest.raises(ValueError):
        SelfCompactSettings.resolve({"self_compact_rubric": bad})


def test_disabled_returns_none():
    assert SelfCompactController.for_turn({"self_compact_enabled": False}, llm_client=FakeLLM([])) is None


@pytest.mark.parametrize("name", ["chat", "search", "math"])
def test_every_prompt_set_loads(name):
    p = load_prompts(name)
    assert p.rubric and p.summarizer and p.continue_prompt and p.labels and p.fire_any
    assert "{summary}" in p.summary_section


# ---------------------------------------------------------------------------
# Verdict parsing (paper fire rules)
# ---------------------------------------------------------------------------


def test_search_rule_fires_only_when_all_four_hold():
    p = load_prompts("chat")
    assert parse_verdict(COMPRESS_REPLY, p).decision == COMPRESS
    assert parse_verdict(CONTINUE_REPLY, p).decision == CONTINUE
    stuck = COMPRESS_REPLY.replace("N1: N -- fewer than 4 attempts", "N1: Y -- try a different tool")
    assert parse_verdict(stuck, p).decision == CONTINUE


def test_yes_without_evidence_counts_as_no():
    p = load_prompts("chat")
    reply = COMPRESS_REPLY.replace('C1: Y -- "The codename is BLUEHERON."', "C1: Y")
    v = parse_verdict(reply, p)
    assert v.answers["C1"] is False and v.decision == CONTINUE


def test_missing_lines_default_to_no():
    v = parse_verdict("I think we should compress.", load_prompts("chat"))
    assert v.decision == CONTINUE and not any(v.answers.values())


def test_parser_tolerates_markdown_and_think_blocks():
    reply = "<think>C1: N -- hidden</think>\n" + "\n".join(
        f"**{line.split(':', 1)[0]}**:{line.split(':', 1)[1]}" for line in COMPRESS_REPLY.splitlines()
    )
    assert parse_verdict(reply, load_prompts("chat")).decision == COMPRESS


def test_math_rule_is_answer_or_stuck_with_next():
    p = load_prompts("math")
    answer = 'Q1: Y -- "\\boxed{42}"\nQ2: N -- new bound\nQ3: N -- NONE'
    stuck_next = "Q1: N -- unknown x\nQ2: Y -- rounds 3,4 no new fact\nQ3: Y -- Substitute x=2."
    stuck_only = "Q1: N -- unknown x\nQ2: Y -- rounds 3,4 no new fact\nQ3: N -- NONE"
    assert parse_verdict(answer, p).decision == COMPRESS
    assert parse_verdict(stuck_next, p).decision == COMPRESS
    assert parse_verdict(stuck_only, p).decision == CONTINUE


# ---------------------------------------------------------------------------
# Gates, probe and compaction
# ---------------------------------------------------------------------------


def test_no_probe_before_round_and_token_gates():
    llm = FakeLLM([])
    ctl = controller(llm, carried={"messages": [], "rounds": 1})
    msgs = convo(1)
    out, events = run(ctl.before_round(msgs, tools=None, anchor=msgs[-2], step=1))
    assert out is msgs and events == [] and llm.calls == []


def test_probe_continue_leaves_trajectory_untouched():
    llm = FakeLLM([CONTINUE_REPLY])
    msgs = convo(6, chars=1_000)
    msgs.append({"role": "user", "content": "next question"})
    ctl = controller(llm, ctx=10_000, carried={"messages": msgs[1:-1], "rounds": 6})
    out, events = run(ctl.before_round(msgs, tools=None, anchor=msgs[-1], step=1))
    assert out is msgs
    assert [e["type"] for e in events] == ["context_probe"]
    assert events[0]["decision"] == CONTINUE
    probe_msgs = llm.calls[0]["messages"]
    # The unanswered turn message is left out so the model judges instead of answering it.
    assert probe_msgs[:-1] == msgs[:-1] and probe_msgs[-1]["content"] == load_prompts("chat").rubric
    assert llm.calls[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    # PERIOD gate: the next round does not probe again.
    out, events = run(ctl.before_round(msgs, tools=None, anchor=msgs[-1], step=2))
    assert events == [] and len(llm.calls) == 1


def test_rubric_compress_rebuilds_from_summary():
    summary = "Facts:\nK00..K05 codes noted."
    llm = FakeLLM([COMPRESS_REPLY, summary])
    msgs = convo(6, chars=1_000)
    question = {"role": "user", "content": "What was code 0?"}
    msgs.append(question)
    ctl = controller(llm, carried={"messages": msgs[1:-1], "rounds": 6}, probe_min_ratio=0.0)
    out, events = run(ctl.before_round(msgs, tools=None, anchor=question, step=1))
    assert [e["type"] for e in events] == ["context_probe", "context_compacted"]
    assert events[1]["trigger"] == "rubric" and events[1]["verdict"]["decision"] == COMPRESS
    assert out[0] == msgs[0]
    assert out[1] is question
    assert out[2] == {"role": "assistant", "content": summary}
    assert out[3]["role"] == "user" and out[3]["content"] == load_prompts("chat").continue_prompt
    assert len(out) == 4
    summarizer_req = llm.calls[1]["messages"]
    assert summarizer_req[-1]["content"] == load_prompts("chat").summarizer
    payload = ctl.done_payload()
    assert payload["summary"] == summary and payload["compactions"] == 1
    # CAP gate: a second rubric compaction in the same turn is not probed.
    out2, ev2 = run(ctl.before_round(out + [{"role": "assistant", "content": "a"}] * 3, tools=None, anchor=question, step=5))
    assert all(e["type"] != "context_probe" for e in ev2)


def test_large_trajectory_is_summarized_in_chained_chunks():
    replies = ["Facts: K00, K01.", "Facts: K00, K01, K02, K03.", "Facts: K00..K05."]
    llm = FakeLLM([COMPRESS_REPLY, *replies])
    msgs = convo(6, chars=3_000)
    question = {"role": "user", "content": "What was code 0?"}
    msgs.append(question)
    turn = controller(FakeLLM([])).est.tokens(msgs[1:3])
    ctl = controller(llm, ctx=40_000, carried={"messages": msgs[1:-1], "rounds": 6},
                     probe_min_ratio=0.0, summary_chunk_ratio=2.5 * turn / 40_000, summary_parallel=1)
    out, events = run(ctl.before_round(msgs, tools=None, anchor=question, step=1))
    assert events[-1]["type"] == "context_compacted"
    summarizer_calls = llm.calls[1:]
    assert len(summarizer_calls) == 3
    for i, call in enumerate(summarizer_calls):
        view = call["messages"]
        assert view[0] == msgs[0] and view[-1]["content"] == load_prompts("chat").summarizer
        assert all(m is not question for m in view)
        users = [m["content"] for m in view if m["role"] == "user"][:-1]
        assert users and all(u.startswith(f"Note {2 * i + j}:") for j, u in enumerate(users))
        if i:
            section = [m for m in view if m["role"] == "system" and replies[i - 1] in m["content"]]
            assert len(section) == 1
    assert out[2] == {"role": "assistant", "content": replies[-1]}


def test_large_trajectory_is_summarized_in_parallel_chunks_then_merged():
    llm = FakeLLM([COMPRESS_REPLY, "Facts: K00, K01.", "Facts: K02, K03.", "Facts: K04, K05.", "Facts: K00..K05."])
    msgs = convo(6, chars=3_000)
    question = {"role": "user", "content": "What was code 0?"}
    msgs.append(question)
    turn = controller(FakeLLM([])).est.tokens(msgs[1:3])
    ctl = controller(llm, ctx=40_000, carried={"messages": msgs[1:-1], "rounds": 6},
                     probe_min_ratio=0.0, summary_chunk_ratio=2.5 * turn / 40_000, summary_parallel=3)
    out, events = run(ctl.before_round(msgs, tools=None, anchor=question, step=1))
    assert events[-1]["type"] == "context_compacted"
    chunk_calls, merge_call = llm.calls[1:4], llm.calls[4]
    starts = sorted(
        [m["content"] for m in call["messages"] if m["role"] == "user"][0][:7] for call in chunk_calls
    )
    assert starts == ["Note 0:", "Note 2:", "Note 4:"]
    merge_view = merge_call["messages"]
    assert merge_view[-1]["content"] == load_prompts("chat").summarizer
    sections = [m["content"] for m in merge_view if m["role"] == "system"][1:]
    assert len(sections) == 3 and all(any(p in s for p in ("K00, K01", "K02, K03", "K04, K05")) for s in sections)
    assert not any(m["role"] == "user" and m["content"].startswith("Note") for m in merge_view)
    assert out[2] == {"role": "assistant", "content": "Facts: K00..K05."}


def test_failed_chunk_falls_back_without_partial_summary():
    llm = FakeLLM([COMPRESS_REPLY, "Facts: K00, K01.", {"ok": False, "error": "boom"}])
    msgs = convo(6, chars=3_000)
    question = {"role": "user", "content": "q"}
    msgs.append(question)
    ctl = controller(llm, ctx=40_000, carried={"messages": msgs[1:-1], "rounds": 6},
                     probe_min_ratio=0.0, summary_chunk_ratio=0.05)
    run(ctl.before_round(msgs, tools=None, anchor=question, step=1))
    assert ctl.done_payload().get("summary") in (None, "")


def test_backstop_compacts_without_probe():
    llm = FakeLLM(["summary of everything"])
    msgs = convo(4, chars=3_000)
    q = {"role": "user", "content": "q"}
    msgs.append(q)
    ctl = controller(llm, ctx=10_000, carried={"messages": msgs[1:-1], "rounds": 4})
    assert ctl.est.tokens(msgs) >= 0.30 * 10_000
    out, events = run(ctl.before_round(msgs, tools=None, anchor=q, step=1))
    assert [e["type"] for e in events] == ["context_compacted"]
    assert events[0]["trigger"] == "backstop"
    assert len(llm.calls) == 1 and out[1] is q
    assert events[0]["tokens_after"] < events[0]["tokens_before"]


def test_summarizer_does_not_see_recalled_memory_block():
    llm = FakeLLM(["summary of everything"])
    memory = "## Recalled memory\n- (user) The oven is set to 215. (assistant) noted"
    msgs = convo(4, chars=3_000)
    msgs[0] = {"role": "system", "content": f"You are helpful.\n\n{memory}\n\nBe brief."}
    q = {"role": "user", "content": "q"}
    msgs.append(q)
    ctl = controller(llm, ctx=10_000, carried={"messages": msgs[1:-1], "rounds": 4})
    ctl.exclude_from_summary(memory, "")
    run(ctl.before_round(msgs, tools=None, anchor=q, step=1))
    system = llm.calls[0]["messages"][0]["content"]
    assert "215" not in system and system == "You are helpful.\n\nBe brief."
    assert "215" in msgs[0]["content"]


def test_summarizer_always_sees_the_carried_summary():
    llm = FakeLLM(["summary of everything"])
    msgs = convo(4, chars=3_000)
    q = {"role": "user", "content": "q"}
    msgs.append(q)
    carried = "Facts and values:\n- Tracking code: TRK-1234-AB"
    ctl = controller(llm, ctx=10_000, carried={"messages": msgs[1:-1], "rounds": 4, "summary": carried})
    run(ctl.before_round(msgs, tools=None, anchor=q, step=1))
    seen = " ".join(str(m.get("content")) for m in llm.calls[0]["messages"] if m.get("role") == "system")
    assert "TRK-1234-AB" in seen


def test_dropped_value_lines_only_reports_lost_values():
    from src.context.self_compact import dropped_value_lines

    prev = "Facts:\n- Tracking code: TRK-1234-AB\n- Budget: 50 EUR\n- The user likes tea"
    assert dropped_value_lines(prev, "Facts:\n- Budget: 50 EUR") == ["- Tracking code: TRK-1234-AB"]
    assert dropped_value_lines(prev, "TRK-1234-AB and 50") == []


def test_summary_that_drops_a_carried_value_gets_it_back():
    llm = FakeLLM(["Facts and values:\n- None"])
    msgs = convo(4, chars=3_000)
    q = {"role": "user", "content": "q"}
    msgs.append(q)
    carried = "Facts and values:\n- Tracking code: TRK-1234-AB"
    ctl = controller(llm, ctx=10_000, carried={"messages": msgs[1:-1], "rounds": 4, "summary": carried})
    run(ctl.before_round(msgs, tools=None, anchor=q, step=1))
    assert "TRK-1234-AB" in ctl.summary and ctl.summary.startswith("Facts and values:\n- None")


def test_keep_carried_values_can_be_disabled(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "SELF_COMPACT_KEEP_CARRIED_VALUES", False)
    llm = FakeLLM(["Facts and values:\n- None"])
    msgs = convo(4, chars=3_000)
    q = {"role": "user", "content": "q"}
    msgs.append(q)
    ctl = controller(llm, ctx=10_000, carried={"messages": msgs[1:-1], "rounds": 4,
                                               "summary": "- Tracking code: TRK-1234-AB"})
    run(ctl.before_round(msgs, tools=None, anchor=q, step=1))
    assert "TRK-1234-AB" not in ctl.summary


def test_summarizer_drops_a_system_prompt_that_crowds_the_window():
    llm = FakeLLM(["summary of everything"])
    msgs = convo(4, chars=3_000)
    msgs[0] = {"role": "system", "content": "RULES " + "r" * 16_000}
    msgs[1] = {"role": "user", "content": "The tracking code is TRK-1234-AB. " + msgs[1]["content"]}
    q = {"role": "user", "content": "q"}
    msgs.append(q)
    ctl = controller(llm, ctx=10_000, carried={"messages": msgs[1:-1], "rounds": 4})
    run(ctl.before_round(msgs, tools=None, anchor=q, step=1))
    seen = " ".join(str(m.get("content")) for m in llm.calls[0]["messages"])
    assert "RULES" not in seen and "TRK-1234-AB" in seen


def test_backstop_skips_when_only_system_prompt_and_turn_message_are_large():
    llm = FakeLLM(["summary"])
    q = {"role": "user", "content": "note " + "n" * 6_000}
    msgs = [{"role": "system", "content": "sys " + "s" * 3_000}, q]
    ctl = controller(llm, ctx=10_000)
    _, events = run(ctl.before_round(msgs, tools=None, anchor=q, step=1))
    assert llm.calls == [] and not events
    msgs = msgs[:1] + convo(2, chars=3_000)[1:] + [q]
    _, events = run(ctl.before_round(msgs, tools=None, anchor=q, step=2))
    assert events and events[0]["trigger"] == "backstop"


def test_completion_budget_stays_inside_the_configured_window():
    ctl = controller(FakeLLM([]), ctx=8_000)
    small = [{"role": "user", "content": "hi"}]
    big = [{"role": "user", "content": "z" * 18_000}]
    room = int(8_000 * ctl.cfg.hard_ratio)
    assert ctl.completion_budget(small, None) == room - ctl.est.tokens(small)
    assert ctl.completion_budget(big, None) == max(ctl.output_reserve, room - ctl.est.tokens(big))
    ctl.cfg = dataclasses.replace(ctl.cfg, cap_completion=False)
    assert ctl.completion_budget(small, None) is None


def test_window_guard_trims_single_huge_message_without_summarizer():
    llm = FakeLLM([])
    huge = {"role": "user", "content": "start " + "y" * 60_000 + " end"}
    msgs = [{"role": "system", "content": "sys"}, huge]
    ctl = controller(llm, ctx=8_000)
    out, events = run(ctl.before_round(msgs, tools=None, anchor=huge, step=1))
    assert llm.calls == []
    assert events[0]["trigger"] == "window_guard"
    assert events[0]["fallback"] == "nothing_to_summarize"
    assert ctl.est.tokens(out) <= ctl.hard_limit
    assert out[-1]["content"].startswith("start ") and out[-1]["content"].endswith(" end")


def test_summarizer_failure_falls_back_to_trimming():
    llm = FakeLLM([{"ok": False, "error": "boom"}])
    msgs = convo(5, chars=8_000)
    q = {"role": "user", "content": "keep me"}
    msgs.append(q)
    ctl = controller(llm, ctx=10_000, carried={"messages": msgs[1:-1], "rounds": 5})
    out, events = run(ctl.before_round(msgs, tools=None, anchor=q, step=1))
    assert events[0]["fallback"] == "summarizer_failed"
    assert ctl.est.tokens(out) <= ctl.hard_limit
    assert any(m.get("content") == "keep me" for m in out)
    assert "summary" not in ctl.done_payload()


def test_summarizer_view_is_fit_to_window():
    llm = FakeLLM(["short summary"])
    msgs = convo(3, chars=40_000)
    q = {"role": "user", "content": "q"}
    msgs.append(q)
    ctl = controller(llm, ctx=10_000, carried={"messages": msgs[1:-1], "rounds": 3})
    run(ctl.before_round(msgs, tools=None, anchor=q, step=1))
    view = llm.calls[0]["messages"]
    budget = int(10_000 * ctl.cfg.hard_ratio) - ctl.cfg.summary_max_tokens
    assert ctl.est.tokens(view) <= budget
    assert view[-1]["content"] == load_prompts("chat").summarizer
    assert not any(m.get("content") == "q" for m in view)


def test_mid_turn_summarizer_keeps_the_turn_message_for_context():
    q = {"role": "user", "content": "find the file"}
    msgs = convo(2) + [
        q,
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result " + "r" * 40_000},
    ]
    llm = FakeLLM(["summary"])
    ctl = controller(llm, ctx=10_000, carried={"messages": [], "rounds": 2})
    out, _ = run(ctl.before_round(msgs, tools=None, anchor=q, step=2))
    assert any(m is q for m in llm.calls[0]["messages"])
    assert out[1] is q and out[2]["content"] == "summary"


def test_echoed_reply_is_not_accepted_as_summary():
    tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
    msgs = convo(4, chars=3_000) + [{"role": "user", "content": "q"}]
    echo = "Noted K03."
    # Echo with tools → retried without tools → real summary accepted.
    llm = FakeLLM([echo, "K00..K03 codes"])
    ctl = controller(llm, carried={"messages": msgs[1:-1], "rounds": 4})
    out, ev = run(ctl.before_round(msgs, tools=tools, anchor=msgs[-1], step=1))
    assert ev[0]["summary_chars"] == len("K00..K03 codes")
    assert "tools" in llm.calls[0] and "tools" not in llm.calls[1]
    # Echo twice (the second one stitched from several prior replies) → trajectory kept.
    llm2 = FakeLLM([echo, "Noted K02.\n\nNoted K03."])
    ctl2 = controller(llm2, carried={"messages": msgs[1:-1], "rounds": 4})
    out2, ev2 = run(ctl2.before_round(msgs, tools=tools, anchor=msgs[-1], step=1))
    assert ev2[0]["fallback"] == "summarizer_failed" and out2 is msgs
    assert "summary" not in ctl2.done_payload()


def test_token_scale_persists_across_turns():
    llm = FakeLLM([{"ok": True, "content": CONTINUE_REPLY, "tokens_prompt": 600}])
    msgs = convo(6, chars=1_000) + [{"role": "user", "content": "q"}]
    ctl = controller(llm, ctx=40_000, carried={"messages": msgs[1:-1], "rounds": 6}, probe_min_ratio=0.0)
    run(ctl.before_round(msgs, tools=None, anchor=msgs[-1], step=1))
    scale = ctl.done_payload()["token_scale"]
    assert scale < 1.0
    nxt = controller(FakeLLM([]), carried={"messages": [], "rounds": 0, "state": ctl.done_payload()})
    assert nxt.est.scale == pytest.approx(scale)


def test_tools_are_sent_with_tool_choice_none_for_prefix_reuse():
    tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
    llm = FakeLLM([CONTINUE_REPLY])
    msgs = convo(6, chars=1_000)
    msgs.append({"role": "user", "content": "q"})
    ctl = controller(llm, carried={"messages": msgs[1:-1], "rounds": 6})
    run(ctl.before_round(msgs, tools=tools, anchor=msgs[-1], step=1))
    assert llm.calls[0]["tools"] == tools and llm.calls[0]["tool_choice"] == "none"


def test_probe_calibrates_token_estimate():
    llm = FakeLLM([{"ok": True, "content": CONTINUE_REPLY, "tokens_prompt": 9_000}])
    msgs = convo(6, chars=500)
    msgs.append({"role": "user", "content": "q"})
    ctl = controller(llm, ctx=40_000, carried={"messages": msgs[1:-1], "rounds": 6}, probe_min_ratio=0.0)
    run(ctl.before_round(msgs, tools=None, anchor=msgs[-1], step=1))
    assert ctl.est.scale > 1.0


# ---------------------------------------------------------------------------
# Provider overflow
# ---------------------------------------------------------------------------


def test_overflow_error_learns_window_and_compacts():
    err = (
        "Error code: 400 - This model's maximum context length is 6000 tokens. "
        "However, your request has 7200 input tokens. Please reduce the length of the messages."
    )
    llm = FakeLLM(["compact summary"])
    msgs = convo(5, chars=2_000)
    q = {"role": "user", "content": "q"}
    msgs.append(q)
    ctl = controller(llm, ctx=32_000, carried={"messages": msgs[1:-1], "rounds": 5})
    assert ctl.is_overflow(err) and ctl.can_retry()
    out, ev = run(ctl.recover_overflow(msgs, tools=None, anchor=q, step=2, error_text=err))
    assert ev["trigger"] == "provider_overflow"
    assert ctl.context_tokens == 6000
    assert out[1] is q and out[2]["content"] == "compact summary"
    assert ctl.est.tokens(out) <= ctl.hard_limit


VLLM_SUMMARIZER_OVERFLOW = (
    "vllm_error: InternalServerError: Error code: 503 - {'detail': 'all_backends_failed: "
    "openai_compat:local-vllm:RuntimeError:HTTP 400 from http://vllm-chat:8000/v1/chat/completions: "
    "{\"error\":{\"message\":\"This model's maximum context length is 32768 tokens. However, you requested "
    "2048 output tokens and your prompt contains at least 30721 input tokens, for a total of at least "
    "32769 tokens. Please reduce the length of the input prompt or the number of requested output tokens. "
    "(parameter=input_tokens, value=30721)\""
)


def test_summarizer_overflow_teaches_real_size_then_refits():
    # Estimator told to under-count 6x (as with an unusual tokenizer).
    llm = FakeLLM([{"ok": False, "error": VLLM_SUMMARIZER_OVERFLOW}, "real summary"])
    big = {"role": "tool", "tool_call_id": "c1", "content": "z" * 220_000}
    q = {"role": "user", "content": "q"}
    msgs = [{"role": "system", "content": "sys"}, q,
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
            big]
    ctl = controller(llm, ctx=32_768, carried={"messages": [], "rounds": 3},
                     chars_per_token=20, probe_min_ratio=0.9)
    out, events = run(ctl.before_round(msgs, tools=None, anchor=q, step=2))
    assert ctl.est.scale > 2.0
    assert len(llm.calls) == 2
    assert ctl.est.tokens(llm.calls[1]["messages"]) <= int(32_768 * ctl.cfg.hard_ratio) - ctl.cfg.summary_max_tokens
    assert out[1] is q and out[2]["content"] == "real summary"
    assert ctl.est.tokens(out) <= ctl.hard_limit


def test_lower_bound_count_backs_off_past_it():
    ctl = controller(FakeLLM([]), ctx=32_768)
    view = [{"role": "user", "content": "w" * 30_000}]
    exact = ("This model's maximum context length is 32768 tokens. "
             "However, your request has 20000 input tokens. Please reduce the length of the messages.")
    assert ctl.learn_from_error(exact, view, None)
    assert ctl.est.tokens(view) == pytest.approx(20_000, rel=0.01)
    ctl.est.scale = 1.0
    assert ctl.learn_from_error(VLLM_SUMMARIZER_OVERFLOW, view, None, max_output=2048)
    assert ctl.est.tokens(view) > 30_721
    assert ctl.est.scale <= ctl.cfg.max_token_scale


def test_probe_overflow_forces_compaction_in_the_same_round():
    llm = FakeLLM([{"ok": False, "error": VLLM_SUMMARIZER_OVERFLOW}, "real summary"])
    msgs = convo(6, chars=17_000) + [{"role": "user", "content": "q"}]
    ctl = controller(llm, ctx=32_768, carried={"messages": msgs[1:-1], "rounds": 6}, chars_per_token=20)
    assert ctl.cfg.probe_min_ratio * 32_768 <= ctl.est.tokens(msgs) < ctl.cfg.backstop_ratio * 32_768
    out, events = run(ctl.before_round(msgs, tools=None, anchor=msgs[-1], step=1))
    assert ctl.est.scale > 1.0
    assert [e["type"] for e in events] == ["context_probe", "context_compacted"]
    assert events[1]["trigger"] in ("window_guard", "backstop")
    assert ctl.est.tokens(out) <= ctl.hard_limit


def test_overflow_retries_are_bounded():
    ctl = controller(FakeLLM(["s"] * 5), max_overflow_retries=1)
    msgs = convo(2) + [{"role": "user", "content": "q"}]
    run(ctl.recover_overflow(msgs, tools=None, anchor=msgs[-1], step=1, error_text="maximum context length is 9000"))
    assert not ctl.can_retry()


def test_non_overflow_errors_are_not_recovered():
    ctl = controller(FakeLLM([]))
    assert not ctl.is_overflow("connection refused")
    assert not ctl.is_overflow("chat_model_unavailable")


# ---------------------------------------------------------------------------
# Window resolution
# ---------------------------------------------------------------------------


def test_window_from_catalog_and_reserve_clamp():
    ctl = SelfCompactController.for_turn(
        {"context_length": 32768, "max_output_tokens": 32768}, llm_client=FakeLLM([])
    )
    assert ctl.context_tokens == 32768
    assert ctl.output_reserve == int(32768 * ctl.cfg.max_reserve_ratio)


def test_window_fallback_without_catalog():
    ctl = SelfCompactController.for_turn({}, llm_client=FakeLLM([], max_tokens=1000))
    assert ctl.context_tokens == int(sc.settings.STREAM_CONTEXT_CHAR_BUDGET / ctl.cfg.chars_per_token)
    assert ctl.output_reserve == 1000


# ---------------------------------------------------------------------------
# Persistence round trip
# ---------------------------------------------------------------------------


def _row(role: str, content: str, meta: Optional[Dict[str, Any]] = None, as_text: bool = False):
    row: Dict[str, Any] = {"role": role, "content": content}
    if meta is not None:
        row["metadata"] = json.dumps(meta) if as_text else meta
    return row


def test_split_without_checkpoint_keeps_everything():
    rows = [_row("user", "u1"), _row("assistant", "a1"), _row("user", "u2"), _row("assistant", "a2")]
    t = split_trajectory(rows)
    assert t["summary"] == "" and t["messages"] == rows and t["rounds"] == 2 and t["state"] == {}


def test_split_resumes_from_compacting_turn():
    rows = [
        _row("user", "u1"), _row("assistant", "a1"),
        _row("user", "u2"), _row("assistant", "a2", {"self_compact": {"summary": "S1", "rounds_since_probe": 0, "compactions": 1}}, as_text=True),
        _row("user", "u3"), _row("assistant", "a3", {"self_compact": {"rounds_since_probe": 1, "compactions": 1}}),
    ]
    t = split_trajectory(rows)
    assert t["summary"] == "S1"
    assert [m["content"] for m in t["messages"]] == ["u2", "a2", "u3", "a3"]
    assert t["state"]["rounds_since_probe"] == 1
    assert t["rounds"] == 2


def test_carried_state_feeds_controller_and_summary_section():
    carried = {"summary": "S1", "messages": [], "rounds": 2, "state": {"rounds_since_probe": 1, "compactions": 3}}
    ctl = controller(FakeLLM([]), carried=carried)
    assert ctl.rounds_since_probe == 2 and ctl.compactions_total == 3
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    out = ctl.attach_summary(msgs)
    assert out[0]["content"] == "sys" and out[1]["role"] == "system" and "S1" in out[1]["content"]
    assert out[2]["content"] == "q"
    assert "summary" not in ctl.done_payload()


def test_new_compaction_drops_old_summary_section():
    carried = {"summary": "OLD", "messages": [], "rounds": 6, "state": {"rounds_since_probe": 5}}
    llm = FakeLLM([COMPRESS_REPLY, "NEW"])
    ctl = controller(llm, carried=carried, probe_min_ratio=0.0)
    msgs = ctl.attach_summary(convo(6, chars=200) + [{"role": "user", "content": "q"}])
    out, _ = run(ctl.before_round(msgs, tools=None, anchor=msgs[-1], step=1))
    assert not any("OLD" in str(m.get("content")) for m in out)
    assert out[-2]["content"] == "NEW"
    # The summarizer saw the old summary so it can fold it in.
    assert any("OLD" in str(m.get("content")) for m in llm.calls[1]["messages"])


# ---------------------------------------------------------------------------
# Plain-completion engines (SRLM)
# ---------------------------------------------------------------------------


def test_prepare_history_compacts_turns_beyond_the_cap_instead_of_dropping():
    llm = FakeLLM(["codes K00..K07 recorded"])
    turns = convo(8)[1:]
    ctl = controller(llm, ctx=100_000, carried={"messages": turns, "rounds": 8})
    context, events = run(ctl.prepare_history(turns, "what was K00?", max_turns=6))
    assert [e["trigger"] for e in events] == ["history_turns"]
    assert "Note 0" in json.dumps(llm.calls[0]["messages"])
    assert len(context) == 1 and context[0]["role"] == "system"
    assert "codes K00..K07 recorded" in context[0]["content"]
    assert ctl.done_payload()["summary"] == "codes K00..K07 recorded"


def test_prepare_history_keeps_turns_and_carried_summary_under_the_cap():
    llm = FakeLLM([])
    turns = convo(2)[1:]
    ctl = controller(llm, ctx=100_000, carried={"summary": "older: K99", "messages": turns, "rounds": 2})
    context, events = run(ctl.prepare_history(turns, "next", max_turns=6))
    assert events == [] and llm.calls == []
    assert "older: K99" in context[0]["content"]
    assert [m["content"] for m in context[1:]] == [m["content"] for m in turns]
    assert all(m["content"] != "next" for m in context)


def test_prepare_history_keeps_turns_when_the_summarizer_fails():
    llm = FakeLLM([RuntimeError("down")])
    turns = convo(8)[1:]
    ctl = controller(llm, ctx=100_000, carried={"messages": turns, "rounds": 8})
    context, events = run(ctl.prepare_history(turns, "q", max_turns=6))
    assert events[0]["fallback"] == "summarizer_failed"
    assert len(context) == len(turns)
