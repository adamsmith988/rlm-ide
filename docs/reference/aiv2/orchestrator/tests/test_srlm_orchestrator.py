"""SRLM (SelfReflect) controller, chat stream and contract adapter."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from src.orchestrators._shared.engine_turn import EngineTurn
from src.orchestrators._shared.llm_roles import LLMReply
from src.orchestrators._shared.manifest_config import package_manifest, resolve_config
from src.orchestrators.contract import RunEnvelope
from src.orchestrators.host import HostPorts
from src.orchestrators.registry import get_registry
from src.orchestrators.srlm import chat_stream as srlm_stream
from src.orchestrators.srlm.adapter import get_engine
from src.orchestrators.srlm.runtime import strategies as S
from src.orchestrators.srlm.runtime.controller import (
    SelfReflectController,
    SrlmSettings,
    answer_distribution,
    Sample,
)

MANIFEST = get_registry().get("srlm").manifest


def run(coro):
    return asyncio.run(coro)


def settings(**over: Any) -> SrlmSettings:
    return SrlmSettings.from_config(resolve_config(MANIFEST, over))


class FakeLLM:
    def __init__(self, samples: List[str], summary: str = "Most likely Paris; could also be Lyon.") -> None:
        self.samples = list(samples)
        self.summary = summary
        self.calls: List[Dict[str, Any]] = []
        self._i = 0

    async def __call__(self, messages, *, temperature=None, max_tokens=None, enable_thinking=None, sampling=None):
        self.calls.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                           "enable_thinking": enable_thinking, "sampling": sampling})
        last = messages[-1]["content"]
        if last == "Q?":
            text = self.samples[self._i % len(self.samples)]
            self._i += 1
            return LLMReply(text=text, tokens_prompt=3, tokens_completion=2)
        return LLMReply(text=self.summary, tokens_prompt=10, tokens_completion=5)


def test_manifest_defaults_follow_the_paper():
    st = settings()
    assert st.summary_strategy == "sample_and_summarize"
    assert st.sample_temperature == 1.0
    assert st.summary_temperature == 0.0
    assert st.sample_max_tokens == 4096 and st.summary_max_tokens == 4096
    assert st.enable_thinking_for_summaries is False
    assert st.thinking_sampling == {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
    assert st.peer_engine == ""


def test_user_config_overrides_and_coerces():
    st = settings(num_samples="5", sample_temperature="0.7", show_samples="false")
    assert st.num_samples == 5 and st.sample_temperature == 0.7 and st.show_samples is False
    with pytest.raises(ValueError):
        settings(summary_strategy="nope")


def test_sample_and_summarize_end_to_end():
    llm = FakeLLM(["Paris", "Paris", "Lyon"])
    progress: List[Dict[str, Any]] = []
    ctrl = SelfReflectController(settings(num_samples=3), llm, on_progress=progress.append)
    res = run(ctrl.run("Q?"))
    sample_calls = [c for c in llm.calls if c["messages"][-1]["content"] == "Q?"]
    summary_calls = [c for c in llm.calls if c["messages"][-1]["content"] != "Q?"]
    assert len(sample_calls) == 3 and len(summary_calls) == 1
    assert all(c["temperature"] == 1.0 and c["enable_thinking"] is False for c in sample_calls)
    assert summary_calls[0]["temperature"] == 0.0
    prompt = summary_calls[0]["messages"][-1]["content"]
    assert "Below, you are given 3 individual answers to the question 'Q?'" in prompt
    assert "x_1 = 'Paris'\nx_2 = 'Paris'\nx_3 = 'Lyon'" in prompt
    assert res.text == "Most likely Paris; could also be Lyon."
    assert res.distribution["n_ok"] == 3 and res.distribution["n_distinct"] == 2
    assert res.distribution["top_share"] == pytest.approx(2 / 3, abs=1e-3)
    assert res.tokens == 3 * 5 + 15
    assert [p["completed"] for p in progress if "sample" in p] == [1, 2, 3]


def test_greedy_and_basic_do_not_sample():
    llm = FakeLLM(["x"])
    res = run(SelfReflectController(settings(summary_strategy="greedy"), llm).run("Q?"))
    assert len(llm.calls) == 1 and res.samples == []
    llm2 = FakeLLM(["x"], summary="A or B")
    res2 = run(SelfReflectController(settings(summary_strategy="basic"), llm2).run("Q?"))
    assert len(llm2.calls) == 1
    assert llm2.calls[0]["messages"][-1]["content"].startswith("Please respond to the following question 'Q?'.")
    assert res2.text == "A or B"


def test_cot_keeps_only_the_summary():
    llm = FakeLLM(["x"], summary="Reasoning: could be A or B.\nSummary: Most likely A, could also be B.")
    res = run(SelfReflectController(settings(summary_strategy="cot"), llm).run("Q?"))
    assert res.text == "Most likely A, could also be B."


def test_twostep_cot_feeds_the_analysis_back():
    llm = FakeLLM(["x"], summary="analysis-or-summary")
    res = run(SelfReflectController(settings(summary_strategy="twostep_cot"), llm).run("Q?"))
    assert len(llm.calls) == 2
    second = llm.calls[1]["messages"]
    assert second[-2] == {"role": "assistant", "content": "analysis-or-summary"}
    assert second[-1]["content"].startswith("Now distill these insights into a summary.")
    assert res.analysis == "analysis-or-summary"


def test_concatenated_answers_needs_no_summary_call():
    llm = FakeLLM(["A", "B"])
    res = run(SelfReflectController(settings(summary_strategy="concatenated_answers", num_samples=2), llm).run("Q?"))
    assert len(llm.calls) == 2
    assert res.text == "x_1 = 'A'\nx_2 = 'B'"


def test_thinking_sampling_applies_only_to_eligible_strategies():
    llm = FakeLLM(["x"], summary="<think>hmm</think>\n\nA or B")
    st = settings(summary_strategy="basic", enable_thinking_for_summaries=True)
    res = run(SelfReflectController(st, llm).run("Q?"))
    call = llm.calls[0]
    assert call["enable_thinking"] is True and call["temperature"] == 0.6
    assert call["sampling"] == {"top_p": 0.95, "top_k": 20, "min_p": 0.0}
    assert res.text == "A or B"
    llm2 = FakeLLM(["x"], summary="Summary: ok")
    run(SelfReflectController(settings(summary_strategy="cot", enable_thinking_for_summaries=True), llm2).run("Q?"))
    assert llm2.calls[0]["enable_thinking"] is False and llm2.calls[0]["temperature"] == 0.0


def test_failed_samples_are_missing_and_all_failed_is_an_error():
    class Flaky(FakeLLM):
        async def __call__(self, messages, **kw):
            if messages[-1]["content"] == "Q?":
                self.calls.append({"messages": messages, **kw})
                return LLMReply(text="", ok=False, error="boom")
            return await super().__call__(messages, **kw)

    with pytest.raises(RuntimeError):
        run(SelfReflectController(settings(num_samples=2), Flaky(["x"])).run("Q?"))
    dist = answer_distribution([Sample(0, "A"), Sample(1, error="x"), Sample(2, "a.")])
    assert dist["n_ok"] == 2 and dist["n_failed"] == 1 and dist["n_distinct"] == 1


def test_history_context_is_bounded():
    llm = FakeLLM(["A"])
    hist = [{"role": "user", "content": f"u{i}"} if i % 2 == 0 else {"role": "assistant", "content": f"a{i}"}
            for i in range(10)] + [{"role": "tool", "content": "ignored"}]
    run(SelfReflectController(settings(num_samples=1, history_turns=2), llm).run("Q?", hist))
    msgs = llm.calls[0]["messages"]
    assert [m["content"] for m in msgs[:-1]] == ["u6", "a7", "u8", "a9"]


def _collect(agen) -> List[Dict[str, Any]]:
    async def go():
        return [ev async for ev in agen]
    return asyncio.run(go())


def test_chat_stream_direct_sampling(monkeypatch):
    llm = FakeLLM(["Paris", "Lyon"])
    monkeypatch.setattr(srlm_stream, "gateway_llm", lambda: llm)

    async def fake_stream(messages, **kw):
        llm.calls.append({"messages": messages, **kw})
        for piece in ("Most likely ", "Paris."):
            yield {"type": "answer", "content": piece}

    monkeypatch.setattr("src.orchestrators._shared.hub_stream.gateway_llm_stream", fake_stream)
    events = _collect(srlm_stream.stream_agent_srlm(
        user_message="Q?", session_id="s1", orchestrator_config={"num_samples": 2}, chat_history=[]))
    types = [e["type"] for e in events]
    assert types[0] == "start" and events[0]["engine"] == "srlm"
    assert "progress" in types and "thinking" in types
    assert "".join(e["content"] for e in events if e["type"] == "answer") == "Most likely Paris."
    done = events[-1]
    assert done["type"] == "done" and done["final_answer"] == "Most likely Paris."
    assert done["extensions"]["srlm"]["distribution"]["n_ok"] == 2
    assert done["extensions"]["srlm"]["strategy"] == "sample_and_summarize"


def test_chat_stream_direct_sampling_gets_host_context(monkeypatch):
    async def fake_stream(messages, **kw):
        llm.calls.append({"messages": messages, **kw})
        yield {"type": "answer", "content": "ok"}

    monkeypatch.setattr("src.orchestrators._shared.hub_stream.gateway_llm_stream", fake_stream)
    hist = [{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"}]
    block = "## Recalled memory\n- The user's name is Dara."

    llm = FakeLLM(["A"])
    monkeypatch.setattr(srlm_stream, "gateway_llm", lambda: llm)
    _collect(srlm_stream.stream_agent_srlm(
        user_message="Q?", session_id="s1", orchestrator_config={"num_samples": 1},
        chat_history=hist, memory_prompt_block=block, rag_prompt_block=""))
    for call in llm.calls:
        head = call["messages"][0]
        assert head["role"] == "system" and block in head["content"]
        assert sum(1 for m in call["messages"] if m["role"] == "system") == 1

    llm = FakeLLM(["A"])
    monkeypatch.setattr(srlm_stream, "gateway_llm", lambda: llm)
    _collect(srlm_stream.stream_agent_srlm(
        user_message="Q?", session_id="s1", orchestrator_config={"num_samples": 1, "host_context_blocks": ""},
        chat_history=hist, memory_prompt_block=block))
    assert all(m["role"] != "system" for call in llm.calls for m in call["messages"])

    class Compactor:
        async def prepare_history(self, messages, user_message, max_turns):
            return [{"role": "system", "content": "SUMMARY"}], []

        def done_payload(self):
            return {}

    llm = FakeLLM(["A"])
    monkeypatch.setattr(srlm_stream, "gateway_llm", lambda: llm)
    monkeypatch.setattr(srlm_stream, "_self_compactor", lambda *a: Compactor())
    _collect(srlm_stream.stream_agent_srlm(
        user_message="Q?", session_id="s1", orchestrator_config={"num_samples": 1},
        chat_history=hist, memory_prompt_block=block, self_compact={"messages": []}))
    head = llm.calls[0]["messages"][0]
    assert head["content"].startswith("SUMMARY") and block in head["content"]
    assert sum(1 for m in llm.calls[0]["messages"] if m["role"] == "system") == 1


def test_chat_stream_peer_engine(monkeypatch):
    seen: List[Dict[str, Any]] = []

    async def fake_turn(engine_id, engine_kwargs, *, orchestrator_config=None, max_steps=None,
                        session_suffix="", timeout_sec=None, on_event=None):
        seen.append({"engine": engine_id, "kwargs": engine_kwargs, "cfg": orchestrator_config,
                     "suffix": session_suffix, "steps": max_steps})
        return EngineTurn(engine_id=engine_id, text=f"answer {len(seen)}", tokens_prompt=1, tokens_completion=1)

    llm = FakeLLM(["unused"], summary="summary")
    monkeypatch.setattr(srlm_stream, "run_engine_turn", fake_turn)
    monkeypatch.setattr(srlm_stream, "gateway_llm", lambda: llm)

    async def fake_stream(messages, **kw):
        yield {"type": "answer", "content": "summary"}

    monkeypatch.setattr("src.orchestrators._shared.hub_stream.gateway_llm_stream", fake_stream)
    cfg = {"num_samples": 3, "peer_engine": "rlm", "peer_config": {"temperature": 0.9},
           "model_stamp": "m1", "tool_ids": ["t1"]}
    events = _collect(srlm_stream.stream_agent_srlm(
        user_message="Q?", session_id="s1", max_steps=7, orchestrator_config=cfg, user_id="u"))
    assert len(seen) == 3
    assert {s["engine"] for s in seen} == {"rlm"}
    assert len({s["suffix"] for s in seen}) == 3
    for s in seen:
        assert s["cfg"] == {"model_stamp": "m1", "tool_ids": ["t1"], "temperature": 0.9}
        assert s["kwargs"]["user_message"] == "Q?" and s["steps"] == 7
    assert events[-1]["final_answer"] == "summary"
    assert events[-1]["extensions"]["srlm"]["peer_engine"] == "rlm"


@pytest.mark.parametrize("peer", ["srlm", "rrsi", "does_not_exist"])
def test_chat_stream_rejects_invalid_peers(peer):
    events = _collect(srlm_stream.stream_agent_srlm(
        user_message="Q?", session_id="s1", orchestrator_config={"peer_engine": peer}))
    assert events[-1]["type"] == "done" and events[-1]["status"] == "error"
    assert any(e["type"] == "error" for e in events)


def test_contract_adapter_runs_over_host_chat():
    calls: List[Dict[str, Any]] = []

    async def chat(body):
        calls.append(body)
        last = body["messages"][-1]["content"]
        text = "Paris" if last == "Q?" else "Most likely Paris."
        return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    events: List[Dict[str, Any]] = []
    engine = get_engine()
    assert engine.manifest.id == "srlm"
    env = RunEnvelope(api_version=1, orchestrator_id="srlm", request_id="r1",
                      input={"question": "Q?"}, config={"num_samples": 2})

    async def go():
        session = await engine.create_session(env, HostPorts(chat=chat, events=events.append))
        searched = await session.search()
        assert searched.output["distribution"]["n_ok"] == 2
        result = await session.select()
        status = await session.status()
        return result, status

    result, status = asyncio.run(go())
    assert result.status == "completed" and result.output["text"] == "Most likely Paris."
    assert status["status"] == "completed"
    assert calls[0]["temperature"] == 1.0 and calls[-1]["temperature"] == 0.0
    assert any(e.get("phase") == "search" for e in events)
    assert package_manifest(srlm_stream.__file__).id == "srlm"


def test_overflow_recovery_helpers():
    from src.orchestrators.srlm import chat_stream as cs
    from src.orchestrators.srlm.runtime.controller import Sample

    class FakeCompactor:
        new_summary = False
        retries_left = True

        def can_retry(self):
            return self.retries_left

        def is_overflow(self, text):
            return "maximum context length" in text

    comp = FakeCompactor()
    err = "HTTP 400: maximum context length is 32768 tokens"
    failed = [Sample(index=0, text="", error=err), Sample(index=1, text="", error="timeout")]
    assert cs._overflow_error(failed, comp) == err
    assert cs._overflow_error(failed + [Sample(index=2, text="ok")], comp) == ""
    assert cs._overflow_error([Sample(index=0, text="", error="timeout")], comp) == ""
    assert cs._overflow_error(failed, None) == ""
    comp.retries_left = False
    assert cs._overflow_error(failed, comp) == ""

    anchor = {"role": "user", "content": "q"}
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}, dict(anchor)]
    assert cs._compacted_context(comp, msgs, anchor) == msgs[:2]
    assert cs._with_host_text([{"role": "system", "content": "S"}], "H")[0]["content"] == "S\n\nH"
    assert cs._with_host_text([], "H") == [{"role": "system", "content": "H"}]
