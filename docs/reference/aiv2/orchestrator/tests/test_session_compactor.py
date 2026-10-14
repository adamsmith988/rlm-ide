"""Session compaction: the host sizes the history from the catalog window for every engine."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from src.config import settings
from src.context import session_compactor as scm
from src.context.self_compact import SelfCompactController, SelfCompactSettings, split_trajectory


def run(coro):
    return asyncio.run(coro)


class FakeLLM:
    def __init__(self, replies: Optional[List[str]] = None) -> None:
        self.replies = list(replies or [])
        self.calls: List[Dict[str, Any]] = []

    async def chat(self, messages, **kwargs):
        self.calls.append({"messages": list(messages), **kwargs})
        return {"ok": True, "content": self.replies.pop(0) if self.replies else "## Facts & Values\n- x", "tokens_prompt": None}


def expand(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stand-in for the chat-history expansion: an assistant row with tool calls
    becomes the assistant message plus one tool message per call."""
    out: List[Dict[str, Any]] = []
    for r in rows:
        if r["role"] == "assistant" and r.get("tool_calls"):
            out.append({"role": "assistant", "content": r["content"], "tool_calls": r["tool_calls"]})
            out.extend({"role": "tool", "tool_call_id": c["id"], "content": c["result"]} for c in r["tool_calls"])
        else:
            out.append({"role": r["role"], "content": r["content"]})
    return out


def turns(n: int, size: int = 400, *, tool_at: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        rows.append({"id": f"u{i}", "role": "user", "content": f"question {i} " + "q" * size})
        reply: Dict[str, Any] = {"id": f"a{i}", "role": "assistant", "content": f"answer {i} " + "a" * size}
        if tool_at == i:
            reply["tool_calls"] = [{"id": f"call{i}", "result": "tool result " + "t" * size}]
        rows.append(reply)
    return rows


CFG = {"context_length": 8000, "max_output_tokens": 1000}


def test_budget_comes_from_the_catalog_window_and_output_limit():
    st = scm.SessionCompactSettings.resolve(CFG)
    b = scm.budget_for(CFG, st)
    assert b.window == 8000 and b.output_reserve == 1000
    assert b.history == int((8000 - 1000) * st.history_share)
    assert b.trigger == int(b.history * st.trigger_ratio)
    assert b.keep == int(b.history * st.keep_recent_ratio)
    huge_out = scm.budget_for({"context_length": 8000, "max_output_tokens": 8000}, st)
    assert huge_out.output_reserve == int(8000 * st.max_output_ratio)


def test_settings_override_per_chat():
    st = scm.SessionCompactSettings.resolve({"session_compact_trigger_ratio": "0.5", "session_compact_enabled": "false"})
    assert st.trigger_ratio == 0.5 and st.enabled is False


def test_plan_cut_keeps_newest_whole_turns():
    assert scm.plan_cut([100, 100, 100, 100], keep=250) == 2
    assert scm.plan_cut([100, 100, 100], keep=10_000) == 1
    assert scm.plan_cut([100, 900], keep=500) == 2
    assert scm.plan_cut([100, 900], keep=500, newest_limit=1000) == 1
    assert scm.plan_cut([100, 900], keep=500, newest_limit=800) == 2
    assert scm.plan_cut([900], keep=500, newest_limit=1000) == 1
    assert scm.turn_spans([{"role": "assistant"}, {"role": "user"}, {"role": "assistant"}]) == [(0, 1), (1, 3)]


def test_under_budget_nothing_happens():
    llm = FakeLLM()
    res = run(scm.compact_session({"summary": "", "state": {}}, turns(2, 50), user_message="hi",
                                  orchestrator_config=CFG, llm_client=llm, expand=expand))
    assert res is None and not llm.calls


def test_over_budget_folds_old_turns_and_keeps_tail_and_tool_pairs():
    rows = turns(12, 400, tool_at=10)
    llm = FakeLLM(["## Facts & Values\n- folded"] * 8)
    res = run(scm.compact_session({"summary": "", "state": {}}, rows, user_message="next?",
                                  orchestrator_config=CFG, llm_client=llm, expand=expand))
    assert res is not None and res.summary.endswith("folded")
    assert all("question 11" not in str(m.get("content")) for c in llm.calls for m in c["messages"])
    assert res.kept_rows and res.kept_rows[0]["role"] == "user"
    assert res.kept_from_id == res.kept_rows[0]["id"]
    assert res.kept_turns == sum(1 for r in res.kept_rows if r["role"] == "user")
    assert res.tokens_after < res.tokens_before
    kept_ids = [r["id"] for r in res.kept_rows]
    assert "a10" in kept_ids and "u10" in kept_ids
    assert llm.calls[0]["messages"][-1]["content"] == scm._summarizer_text(str(scm._BUNDLED_SUMMARIZER))


def test_previous_checkpoint_is_updated_not_restarted():
    llm = FakeLLM()
    run(scm.compact_session({"summary": "codename BLUEHERON", "state": {"compactions": 2}}, turns(12, 400),
                            user_message="next?", orchestrator_config=CFG, llm_client=llm, expand=expand))
    view = llm.calls[0]["messages"]
    assert any(m["role"] == "system" and "BLUEHERON" in m["content"] for m in view)


def test_rows_trigger_when_the_fetch_window_fills():
    llm = FakeLLM()
    res = run(scm.compact_session({"summary": "", "state": {}}, turns(10, 5), user_message="x",
                                  orchestrator_config={"context_length": 1_000_000}, llm_client=llm,
                                  expand=expand, rows_limit=20))
    assert res is not None and res.trigger == "rows"


def test_checkpoint_round_trips_through_storage():
    rows = turns(12, 400)
    res = run(scm.compact_session({"summary": "", "state": {}}, rows, user_message="next?",
                                  orchestrator_config=CFG, llm_client=FakeLLM(["CHECKPOINT"] * 8), expand=expand))
    stored = rows + [
        {"id": "u12", "role": "user", "content": "next?"},
        {"id": "a12", "role": "assistant", "content": "ok", "metadata": json.dumps({"self_compact": res.payload()})},
    ]
    traj = split_trajectory(stored)
    assert traj["summary"] == res.summary == "CHECKPOINT"
    assert [r["id"] for r in traj["messages"]] == [r["id"] for r in res.kept_rows] + ["u12", "a12"]
    no_ids = [{k: v for k, v in r.items() if k != "id"} for r in stored]
    assert len(split_trajectory(no_ids)["messages"]) == len(traj["messages"])


def test_engine_summary_in_the_same_turn_wins():
    session = {"summary": "S", "kept_turns": 2, "kept_from_id": "u3", "compacted_by": "session", "compactions": 1,
               "events": [{"trigger": "window"}]}
    assert scm.merge_turn_payload(session, None)["kept_turns"] == 2
    merged = scm.merge_turn_payload(session, {"rubric": "chat", "compactions": 1})
    assert merged["summary"] == "S" and merged["kept_turns"] == 2 and merged["rubric"] == "chat"
    engine_won = scm.merge_turn_payload(session, {"summary": "E", "compactions": 2, "events": [{"trigger": "rubric"}]})
    assert engine_won["summary"] == "E" and "kept_turns" not in engine_won
    assert [e["trigger"] for e in engine_won["events"]] == ["window", "rubric"]
    assert scm.merge_turn_payload(None, {"summary": "E"}) == {"summary": "E"}


def _ctrl(llm: FakeLLM, carried: Dict[str, Any]) -> SelfCompactController:
    return SelfCompactController(
        SelfCompactSettings.resolve({"self_compact_summary_chunk_ratio": 0}),
        llm_client=llm, context_tokens=10_000, output_reserve=500, carried=carried,
    )


def test_engine_leaves_the_turn_start_to_the_host():
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(6):
        msgs += [{"role": "user", "content": "u" * 2000}, {"role": "assistant", "content": "a" * 2000}]
    anchor = {"role": "user", "content": "now"}
    msgs.append(anchor)
    llm = FakeLLM()
    owned = _ctrl(llm, {"owner": scm.OWNER, "rounds": 9, "messages": msgs})
    out, events = run(owned.before_round(msgs, tools=None, anchor=anchor, step=1))
    assert out is msgs and not events and not llm.calls
    free = _ctrl(llm, {"rounds": 9, "messages": msgs})
    _, events = run(free.before_round(msgs, tools=None, anchor=anchor, step=1))
    assert events and llm.calls


def test_engine_still_guards_the_hard_window_when_the_host_owns_history():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "x" * 60_000},
            {"role": "assistant", "content": "y"}, {"role": "user", "content": "now"}]
    llm = FakeLLM(["short"])
    owned = _ctrl(llm, {"owner": scm.OWNER})
    _, events = run(owned.before_round(msgs, tools=None, anchor=msgs[-1], step=1))
    assert events and events[0]["trigger"] == "window_guard"


def test_prepare_history_turn_cap_is_the_hosts_call():
    history = []
    for i in range(8):
        history += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    llm = FakeLLM()
    owned = _ctrl(llm, {"owner": scm.OWNER, "messages": history})
    ctx, events = run(owned.prepare_history(history, "now", max_turns=2))
    assert not events and not llm.calls and len(ctx) == len(history)


def test_engine_compaction_writes_the_owners_checkpoint(tmp_path):
    prompt = tmp_path / "checkpoint.md"
    prompt.write_text("WRITE THE CHECKPOINT", encoding="utf-8")
    llm = FakeLLM()
    owned = _ctrl(llm, {"owner": scm.OWNER, "owner_summarizer": str(prompt)})
    assert owned.prompts.summarizer == "WRITE THE CHECKPOINT"
    free = _ctrl(llm, {"owner_summarizer": str(prompt)})
    assert free.prompts.summarizer != "WRITE THE CHECKPOINT"
    default = scm.summarizer_path(scm.SessionCompactSettings.resolve(None))
    assert scm.checkpoint_prompt(default).startswith("You are writing a context checkpoint")


def test_session_compaction_settings_are_configurable():
    for name in ("ENABLED", "HISTORY_SHARE", "TRIGGER_RATIO", "KEEP_RECENT_RATIO", "MAX_OUTPUT_RATIO",
                 "MAX_ROWS_RATIO", "SUMMARIZER_FILE"):
        assert hasattr(settings, f"SESSION_COMPACT_{name}")
