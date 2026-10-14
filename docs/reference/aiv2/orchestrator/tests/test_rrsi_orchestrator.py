"""RRSI controller (Algorithms 1+2 over a chat evolve set), chat stream and adapter."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import pytest

from src.orchestrators._shared.engine_turn import EngineTurn
from src.orchestrators._shared.llm_roles import LLMReply
from src.orchestrators._shared.manifest_config import resolve_config
from src.orchestrators.contract import RunEnvelope
from src.orchestrators.host import HostPorts
from src.orchestrators.registry import get_registry
from src.orchestrators.rrsi import chat_stream as rrsi_stream
from src.orchestrators.rrsi.adapter import get_engine
from src.orchestrators.rrsi.core.config import RRSIConfig
from src.orchestrators.rrsi.core.evaluate import aggregate, TaskResult
from src.orchestrators.rrsi.runtime import harness as H
from src.orchestrators.rrsi.runtime import roles as R
from src.orchestrators.rrsi.runtime.controller import RrsiController, RrsiSettings, build_evolve_set
from src.orchestrators.rrsi.runtime.domain import Task, Trial, default_domain
from src.orchestrators.rrsi.runtime.state import RrsiState

MANIFEST = get_registry().get("rrsi").manifest
DOMAIN = default_domain()
GOOD_INSTRUCTION = "Begin every answer with a STRUCTURED one-line summary, then details."


def cfg(**over: Any) -> Dict[str, Any]:
    return resolve_config(MANIFEST, over)


class ScriptedLLM:
    """Role LLM: dispatches on the role prompt, records every call."""

    def __init__(self, proposal: Dict[str, Any], critic: str = "accept") -> None:
        self.proposal = proposal
        self.critic = critic
        self.calls: List[str] = []

    async def __call__(self, messages, *, temperature=None, max_tokens=None, enable_thinking=None, sampling=None):
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        user = messages[-1]["content"]
        if "Write a grading rubric" in user:
            self.calls.append("rubric")
            body = {"criteria": [{"id": "c1", "text": "starts with a structured summary"},
                                 {"id": "c2", "text": "answers the request"}]}
        elif "Grade the ASSISTANT ANSWER" in user:
            self.calls.append("grade")
            answer = user.split("=== ASSISTANT ANSWER ===", 1)[1]
            body = {"verdicts": [{"id": "c1", "pass": "STRUCTURED" in answer, "why": "-"},
                                 {"id": "c2", "pass": True, "why": "-"}]}
        elif "batch analyst" in system:
            self.calls.append("analyst")
            body = {"action": "report",
                    "failure_modes": [{"mode": "missing_summary", "n_tasks": 1, "affected_tasks": [],
                                       "description": "answers lack an upfront summary"}],
                    "capability_gaps": [], "success_habits": []}
        elif "harness engineer agent" in system:
            self.calls.append("proposer")
            body = self.proposal
        elif "strict reviewer" in system:
            self.calls.append("critic")
            body = {"verdict": self.critic, "reasons": ["scripted"], "risk_notes": []}
        else:
            self.calls.append("other")
            return LLMReply(text="reviewed STRUCTURED answer")
        return LLMReply(text=json.dumps(body), tokens_prompt=10, tokens_completion=10)


def proposal(changes: List[Dict[str, Any]], component: str = "prompt") -> Dict[str, Any]:
    return {"action": "done", "summary": "summary-first answers", "edits": [{
        "id": "C1", "component": component, "changes": changes,
        "hypothesis": "an upfront summary satisfies the summary habit",
        "targets_mode": "missing_summary", "why_not_lower_lever": "prose is the lever",
        "trigger_condition": "every answer", "predicted_affected": ["q0"],
        "retroactive_check": "corrective/preservative/transfer", "regression_risk": "none"}]}


async def policy(harness: H.Harness, task: Task, index: int) -> Trial:
    text = str(harness.get("instructions") or "")
    answer = "STRUCTURED: done" if "STRUCTURED" in text else "plain answer"
    return Trial(task_id=task.id, index=index, answer=answer, tokens=100.0)


def controller(tmp_path, llm, policy_fn=policy, **over) -> RrsiController:
    c = cfg(**over)
    return RrsiController(RrsiSettings.from_config(c), RRSIConfig.from_mapping(c), DOMAIN, llm, policy_fn,
                          RrsiState(tmp_path), allowed_engine_keys={"temperature"}, known_tool_ids={"web_search"})


# ------------------------------------------------------------------ harness --
def test_harness_changes_diff_and_component_tagging():
    h0 = H.default_harness(DOMAIN)
    h1, problems = H.apply_changes(DOMAIN, h0, [
        {"field": "instructions", "op": "set", "value": "Be precise."},
        {"field": "memory", "op": "append", "value": "Prefer tables for comparisons."},
    ])
    assert not problems
    diff = H.render_diff(h0, h1)
    assert '+ instructions = "Be precise."' in diff and "+ memory = " in diff
    from src.orchestrators.rrsi.core.components import normalize
    only_mem, _ = H.apply_changes(DOMAIN, h0, [{"field": "memory", "op": "append", "value": "x"}])
    assert normalize("prompt", H.render_diff(h0, only_mem), DOMAIN.component_signals) == "memory"
    assert normalize("memory", H.render_diff(h0, only_mem), DOMAIN.component_signals) == "memory"
    h2, _ = H.apply_changes(DOMAIN, h1, [{"field": "memory", "op": "remove", "value": "Prefer tables for comparisons."}])
    assert h2["memory"] == []
    _, bad = H.apply_changes(DOMAIN, h0, [{"field": "nope", "value": 1}, {"field": "instructions", "op": "append", "value": "x"}])
    assert len(bad) == 2


def test_harness_validation_enforces_frozen_and_declared_keys():
    h = H.default_harness(DOMAIN)
    h["engine_config"] = {"model": "x", "temperature": 0.1, "made_up": 1}
    h["tool_ids"] = ["web_search", "ghost_tool"]
    problems = H.validate(DOMAIN, h, allowed_engine_keys={"temperature", "model"},
                          frozen_engine_keys={"model"}, known_tool_ids={"web_search"})
    text = " ".join(problems)
    assert "frozen" in text and "made_up" in text and "ghost_tool" in text
    h2 = H.default_harness(DOMAIN)
    h2["max_steps"] = 999
    assert any("max_steps" in p for p in H.validate(DOMAIN, h2, allowed_engine_keys=set(),
                                                     frozen_engine_keys=set(), known_tool_ids=set()))


def test_prompt_block_and_history_trim():
    h = H.default_harness(DOMAIN)
    h["instructions"] = "Be precise."
    h["skills"] = [{"name": "compare", "when": "two options", "procedure": "table"}]
    block = H.prompt_block(DOMAIN, h)
    assert block.startswith(DOMAIN.schema["prompt_block"]["header"])
    assert "Be precise." in block and "compare" in block
    assert H.prompt_block(DOMAIN, H.default_harness(DOMAIN)) == ""
    hist = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}]
    assert H.trim_history(hist, 1) == hist[2:]
    assert H.trim_history(hist, None) == hist and H.trim_history(hist, 0) == []


def test_critic_precheck_catches_leakage_and_grader_references():
    request = "please compare the quarterly revenue of the northern and southern branches in detail"
    added = json.dumps("Always compare the quarterly revenue of the northern and southern branches first")
    hits = R.precheck(DOMAIN, "+ instructions = x", added, [request], [], 8)
    assert any("overlap" in h for h in hits)
    assert any("grader" in h for h in R.precheck(DOMAIN, "", json.dumps("satisfy the rubric"), [], [], 8))
    assert R.precheck(DOMAIN, "", json.dumps("State assumptions explicitly."), [request], [], 8) == []


def test_evolve_set_is_current_plus_bounded_replay():
    hist = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "a"},
            {"role": "user", "content": "second"}, {"role": "assistant", "content": "b"},
            {"role": "tool", "content": "x"}]
    tasks, cur = build_evolve_set("third", hist, 1)
    assert list(tasks)[-1] == cur and tasks[cur].current and tasks[cur].request == "third"
    assert [t.request for t in tasks.values()] == ["second", "third"]
    assert tasks[cur].context[-1]["content"] == "b"
    tasks0, _ = build_evolve_set("third", hist, 0)
    assert len(tasks0) == 1


def test_evolve_set_skips_oversized_earlier_requests_and_judge_clips():
    from src.orchestrators.rrsi.runtime.judge import clip_text

    big = "x" * 500
    hist = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "a"},
            {"role": "user", "content": big}, {"role": "assistant", "content": "b"}]
    tasks, _ = build_evolve_set("third", hist, 2, max_request_chars=100)
    assert [t.request for t in tasks.values()] == ["first", "third"]
    tasks, _ = build_evolve_set("third", hist, 2)
    assert [t.request for t in tasks.values()] == ["first", big, "third"]
    clipped = clip_text("a" * 50 + "b" * 50, 20)
    assert clipped.startswith("a" * 10) and clipped.endswith("b" * 10) and len(clipped) < 40
    assert clip_text("short", 0) == "short"


# --------------------------------------------------------------- full round --
def test_round_accepts_a_measured_improvement(tmp_path):
    llm = ScriptedLLM(proposal([{"field": "instructions", "op": "set", "value": GOOD_INSTRUCTION}]))
    ctl = controller(tmp_path, llm, m=1, k=1, replay_turns=0)
    tasks, cur = build_evolve_set("Explain the plan.", [], 0)
    out = asyncio.run(ctl.turn(tasks, cur))
    r = out.rounds[0]
    assert r.incumbent.ev.S == pytest.approx(0.5) and r.next_incumbent.ev.S == pytest.approx(1.0)
    assert r.accepted == "A" and out.version == 1
    assert out.harness["instructions"] == GOOD_INSTRUCTION
    assert out.answer is not None and out.answer.reward == pytest.approx(1.0)
    assert "STRUCTURED" in out.answer.answer
    recs = ctl.state.history.records()
    assert recs[0]["outcome"] == "BASELINE"
    acc = [x for x in recs if x.get("outcome") == "ACCEPTED"]
    assert acc and acc[0]["component"] == "prompt" and acc[0]["delta_S"] == pytest.approx(0.5)
    fr = ctl.state.frontier()
    assert fr["t"] == 1 and fr["incumbent"]["version"] == 1
    assert fr["trajectory"][-1]["G"] == pytest.approx(0.5)
    assert (tmp_path / "r0" / "decisions.json").is_file()
    assert (tmp_path / "harness" / "v1.json").is_file()
    assert llm.calls.count("rubric") == 1  # rubric written once per task, reused for the candidate
    assert "analyst" in llm.calls and "critic" in llm.calls


def test_no_gain_keeps_the_incumbent(tmp_path):
    llm = ScriptedLLM(proposal([{"field": "instructions", "op": "set", "value": "Be kind."}]))
    ctl = controller(tmp_path, llm, m=1)
    tasks, cur = build_evolve_set("Explain the plan.", [], 0)
    out = asyncio.run(ctl.turn(tasks, cur))
    assert out.rounds[0].accepted is None and out.version == 0
    rec = [x for x in ctl.state.history.records() if x.get("edit_id")]
    assert rec[0]["outcome"] == "REJECTED" and rec[0]["accepted"] is False


def test_leaky_candidate_is_dropped_by_the_critic(tmp_path):
    request = "summarize the meeting notes about the new office relocation budget and the timeline"
    leaky = "When asked, summarize the meeting notes about the new office relocation budget and the timeline."
    llm = ScriptedLLM(proposal([{"field": "instructions", "op": "set", "value": leaky}]))
    ctl = controller(tmp_path, llm, m=1, repair_rounds=1)
    tasks, cur = build_evolve_set(request, [], 0)
    out = asyncio.run(ctl.turn(tasks, cur))
    assert out.rounds[0].accepted is None
    rec = [x for x in ctl.state.history.records() if x.get("edit_id")]
    assert rec[0]["outcome"] == "critic_reject" and rec[0]["delta_S"] is None
    assert "critic" not in llm.calls  # deterministic precheck rejected before the LLM critic


def test_llm_critic_reject_and_repair_rounds(tmp_path):
    llm = ScriptedLLM(proposal([{"field": "instructions", "op": "set", "value": GOOD_INSTRUCTION}]), critic="reject")
    ctl = controller(tmp_path, llm, m=1, repair_rounds=2)
    tasks, cur = build_evolve_set("Explain the plan.", [], 0)
    asyncio.run(ctl.turn(tasks, cur))
    assert llm.calls.count("critic") == 3 and llm.calls.count("proposer") == 3


def test_over_budget_or_mislabelled_proposals_are_bounced(tmp_path):
    two = proposal([{"field": "instructions", "op": "set", "value": GOOD_INSTRUCTION}])
    two["edits"].append(dict(two["edits"][0], id="C2", component="memory",
                             changes=[{"field": "instructions", "op": "set", "value": "x"}]))
    llm = ScriptedLLM(two)
    ctl = controller(tmp_path, llm, m=1, proposer_max_turns=2, T=20)
    tasks, cur = build_evolve_set("Explain the plan.", [], 0)
    asyncio.run(ctl.turn(tasks, cur))
    rec = [x for x in ctl.state.history.records() if x.get("edit_id") or x.get("outcome") == "no_proposal"]
    assert rec[0]["outcome"] == "no_proposal"
    assert llm.calls.count("proposer") == 2


def test_guard_blocks_candidates_that_fail_more_trials(tmp_path):
    llm = ScriptedLLM(proposal([{"field": "instructions", "op": "set", "value": GOOD_INSTRUCTION}]))

    async def flaky_policy(harness, task, index):
        if "STRUCTURED" in str(harness.get("instructions") or ""):
            return Trial(task_id=task.id, index=index, error="crash")
        return await policy(harness, task, index)

    ctl = controller(tmp_path, llm, flaky_policy, m=1, k=1, invalid_missing_frac=1.0)
    tasks, cur = build_evolve_set("Explain the plan.", [], 0)
    out = asyncio.run(ctl.turn(tasks, cur))
    assert out.rounds[0].accepted is None


def test_subagent_review_and_retry_are_bounded(tmp_path):
    llm = ScriptedLLM(proposal([]))
    attempts = {"n": 0}

    async def once_failing(harness, task, index):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return Trial(task_id=task.id, index=index, error="transient")
        return Trial(task_id=task.id, index=index, answer="draft", tokens=10.0)

    ctl = controller(tmp_path, llm, once_failing)
    h = H.default_harness(DOMAIN)
    h["retry_on_failure"] = True
    h["subagent_review"] = {"enabled": True, "instructions": "check facts"}
    trial = asyncio.run(ctl.run_trial(h, Task(id="q0", request="r"), 0))
    assert attempts["n"] == 2 and trial.reviewed and trial.answer == "reviewed STRUCTURED answer"


def test_horizon_T_switches_to_serving(tmp_path):
    llm = ScriptedLLM(proposal([{"field": "instructions", "op": "set", "value": GOOD_INSTRUCTION}]))
    ctl = controller(tmp_path, llm, m=1, T=1)
    tasks, cur = build_evolve_set("Explain the plan.", [], 0)
    asyncio.run(ctl.turn(tasks, cur))
    ctl2 = controller(tmp_path, llm, m=1, T=1)
    out = asyncio.run(ctl2.turn(*build_evolve_set("Next.", [], 0)))
    assert out.served_only and out.version == 1 and out.rounds == []


def test_domain_guard_unit():
    inc = aggregate("i", 1, {"a": TaskResult(rewards=[1.0])}, {"failed_trials": 0})
    cand = aggregate("c", 1, {"a": TaskResult(rewards=[1.0])}, {"failed_trials": 1})
    assert DOMAIN.guards(inc, cand) and not DOMAIN.guards(cand, inc)


# -------------------------------------------------------------- chat stream --
def _collect(agen) -> List[Dict[str, Any]]:
    async def go():
        return [ev async for ev in agen]
    return asyncio.run(go())


def _patch_stream(monkeypatch, tmp_path, llm, seen):
    async def fake_turn(engine_id, engine_kwargs, *, orchestrator_config=None, max_steps=None,
                        session_suffix="", timeout_sec=None, on_event=None):
        seen.append({"engine": engine_id, "kwargs": engine_kwargs, "cfg": orchestrator_config,
                     "suffix": session_suffix, "steps": max_steps})
        prompt = str(engine_kwargs.get("host_page_prompt") or "")
        text = "STRUCTURED: ok" if "STRUCTURED" in prompt else "plain"
        return EngineTurn(engine_id=engine_id, text=text, tokens_prompt=50, tokens_completion=50)

    async def no_catalog(user_id, active_plugin=None):
        return [{"id": "web_search", "description": "search"}], {"web_search"}

    monkeypatch.setattr(rrsi_stream, "run_engine_turn", fake_turn)
    monkeypatch.setattr(rrsi_stream, "gateway_llm", lambda: llm)
    monkeypatch.setattr(rrsi_stream, "_tool_catalog", no_catalog)
    monkeypatch.setattr(rrsi_stream, "session_state_dir", lambda *a: str(tmp_path / "state"))


def test_chat_stream_evolves_and_answers(monkeypatch, tmp_path):
    llm = ScriptedLLM(proposal([{"field": "instructions", "op": "set", "value": GOOD_INSTRUCTION}]))
    seen: List[Dict[str, Any]] = []
    _patch_stream(monkeypatch, tmp_path, llm, seen)
    events = _collect(rrsi_stream.stream_agent_rrsi(
        user_message="Explain the plan.", session_id="s1", max_steps=5, user_id="u1",
        orchestrator_config={"m": 1, "replay_turns": 0, "model_stamp": "m1"},
        chat_history=[], host_page_prompt="PAGE"))
    assert events[0]["type"] == "start" and events[0]["engine"] == "rrsi"
    done = events[-1]
    assert done["type"] == "done" and "STRUCTURED" in done["final_answer"]
    ext = done["extensions"]["rrsi"]
    assert ext["policy_engine"] == "native_tools" and ext["harness_version"] == 1
    assert ext["rounds"][0]["accepted"] == "A" and ext["answer_reward"] == pytest.approx(1.0)
    assert any(e["type"] == "progress" for e in events)
    # policy runs: incumbent + candidate, with page prompt preserved and harness block appended
    assert len(seen) == 2 and {s["engine"] for s in seen} == {"native_tools"}
    assert seen[1]["kwargs"]["host_page_prompt"].startswith("PAGE")
    assert seen[0]["cfg"] == {"model_stamp": "m1"} and seen[0]["steps"] == 5


def test_chat_stream_serves_live_when_no_round_is_due(monkeypatch, tmp_path):
    llm = ScriptedLLM(proposal([]))
    seen: List[Dict[str, Any]] = []
    _patch_stream(monkeypatch, tmp_path, llm, seen)
    received: Dict[str, Any] = {}

    async def fake_engine(max_steps=None, orchestrator_config=None, **kwargs):
        received.update(kwargs, cfg=orchestrator_config)
        yield {"type": "start", "engine": "native_tools"}
        yield {"type": "answer_start"}
        yield {"type": "answer", "content": "live answer"}
        yield {"type": "done", "final_answer": "live answer", "extensions": {"x": 1}}

    monkeypatch.setattr(rrsi_stream, "stream_agent_for_id", lambda oid: fake_engine)
    events = _collect(rrsi_stream.stream_agent_rrsi(
        user_message="Hi", session_id="s2", orchestrator_config={"rounds_per_turn": 0}, chat_history=[]))
    assert [e["type"] for e in events].count("start") == 1
    done = events[-1]
    assert done["final_answer"] == "live answer" and done["engine"] == "rrsi"
    assert done["extensions"]["x"] == 1 and done["extensions"]["rrsi"]["served_only"] is True
    assert seen == [] and llm.calls == []


def test_chat_stream_rejects_controller_policy(monkeypatch, tmp_path):
    events = _collect(rrsi_stream.stream_agent_rrsi(
        user_message="Hi", session_id="s3", orchestrator_config={"policy_engine": "srlm"}))
    assert events[-1]["status"] == "error"


# ----------------------------------------------------------------- adapter --
def test_contract_adapter_round_over_host_chat(tmp_path):
    llm = ScriptedLLM(proposal([{"field": "instructions", "op": "set", "value": GOOD_INSTRUCTION}]))

    async def chat(body):
        msgs = body["messages"]
        if msgs[-1]["content"] == "Explain the plan." and "Write a grading rubric" not in msgs[-1]["content"]:
            sys_txt = msgs[0]["content"] if msgs[0]["role"] == "system" else ""
            text = "STRUCTURED: ok" if "STRUCTURED" in sys_txt else "plain"
            return {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 5, "completion_tokens": 5}}
        reply = await llm(msgs)
        return {"choices": [{"message": {"content": reply.text}}]}

    engine = get_engine()
    env = RunEnvelope(api_version=1, orchestrator_id="rrsi", request_id="r1",
                      input={"question": "Explain the plan."},
                      config={"m": 1, "replay_turns": 0, "state_dir": str(tmp_path)})

    async def go():
        session = await engine.create_session(env, HostPorts(chat=chat))
        res = await session.run()
        return res, await session.status()

    res, status = asyncio.run(go())
    assert res.status == "completed" and "STRUCTURED" in res.output["text"]
    assert res.output["harness_version"] == 1 and status["round"] == 1
