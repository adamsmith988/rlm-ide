import asyncio
from types import SimpleNamespace

import pytest

from src.config import settings
from src.tools import side_effects as se

ACTION = {"id": "book", "description": "Book a flight", "side_effects": True}
READ = {"id": "weather", "description": "Weather"}


def _llm(verdict: str, calls: list):
    async def call(messages, **kw):
        calls.append(messages[0]["content"])
        await asyncio.sleep(0.05)
        return SimpleNamespace(ok=True, text=verdict)
    return call


def _runner(counter: list, result=None):
    async def run():
        counter.append(1)
        await asyncio.sleep(0.05)
        return result or {"ok": True, "booking": "BK-1"}
    return run


async def _in_turn(coro_fn, history=(), message="book it", summary=""):
    async def body():
        se.begin_tool_turn("t1", list(history), message, summary=summary)
        return await coro_fn()
    return await asyncio.create_task(body())


@pytest.mark.asyncio
async def test_read_only_tools_run_without_check():
    ran, asked = [], []
    out = await _in_turn(lambda: se.guarded("weather", {"city": "Rome"}, READ, "x", _runner(ran),
                                            llm=_llm("NO", asked)))
    assert out["ok"] and ran == [1] and asked == []


@pytest.mark.asyncio
async def test_action_held_when_check_says_no():
    ran, asked = [], []
    out = await _in_turn(lambda: se.guarded("book", {"date": "2026-10-12"}, ACTION, "x", _runner(ran),
                                            llm=_llm("NO\nThe user asked to wait.", asked)))
    assert ran == [] and out["error"] == "action_not_confirmed"
    assert "The user asked to wait" in out["message"]


@pytest.mark.asyncio
async def test_verdict_line_wins_over_reasoning_lines():
    ran, asked = [], []
    reply = ("date = 2026-10-20: user said Tuesday, October 20\ntime = 20:30: not given, NO time was confirmed\n"
             "request: 'go ahead and book it'\nVERDICT: YES")
    out = await _in_turn(lambda: se.guarded("book", {"date": "2026-10-20"}, ACTION, "x", _runner(ran),
                                            llm=_llm(reply, asked)))
    assert out["ok"] and ran == [1]
    held = await _in_turn(lambda: se.guarded("book", {"a": 2}, ACTION, "x", _runner(ran),
                                             llm=_llm("a = 2: not given\nVERDICT: NO - a was never given\nextra", asked)))
    assert held["error"] == "action_not_confirmed" and "a was never given" in held["message"]


@pytest.mark.asyncio
async def test_cut_off_reply_without_verdict_follows_on_error(monkeypatch):
    ran, asked = [], []
    monkeypatch.setattr(settings, "TOOL_ACTION_CHECK_ON_ERROR", "deny")
    out = await _in_turn(lambda: se.guarded("book", {"a": 1}, ACTION, "x", _runner(ran),
                                            llm=_llm("a = 1: user said YES to one", asked)))
    assert out["error"] == "action_not_confirmed" and ran == []


@pytest.mark.asyncio
async def test_identical_calls_in_one_turn_run_once():
    ran, asked = [], []

    async def both():
        call = lambda: se.guarded("book", {"origin": "Rome ", "passengers": "2"}, ACTION, "x",
                                  _runner(ran), llm=_llm("YES", asked))
        other = lambda: se.guarded("book", {"origin": "rome", "passengers": 2}, ACTION, "x",
                                   _runner(ran), llm=_llm("YES", asked))
        return await asyncio.gather(asyncio.create_task(call()), asyncio.create_task(other()))

    a, b = await _in_turn(both)
    assert ran == [1] and len(asked) == 1 and a == b


@pytest.mark.asyncio
async def test_waiter_runs_itself_when_first_call_is_cancelled():
    ran, asked = [], []

    async def scenario():
        first = asyncio.create_task(se.guarded("book", {"a": 1}, ACTION, "x", _runner(ran), llm=_llm("YES", asked)))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(se.guarded("book", {"a": 1}, ACTION, "x", _runner(ran), llm=_llm("YES", asked)))
        await asyncio.sleep(0.01)
        first.cancel()
        return await second

    out = await _in_turn(scenario)
    assert out["ok"] and ran == [1]


@pytest.mark.asyncio
async def test_rehearsal_simulates_actions_but_runs_reads():
    ran, asked = [], []

    async def scenario():
        with se.rehearsal():
            act = await se.guarded("book", {"a": 1}, ACTION, "x", _runner(ran), llm=_llm("YES", asked))
            read = await se.guarded("weather", {}, READ, "x", _runner(ran), llm=_llm("YES", asked))
        return act, read

    act, read = await _in_turn(scenario)
    assert act["simulated"] and act["message"] == settings.TOOL_SIDE_EFFECTS_REHEARSAL_MESSAGE
    assert read["ok"] and ran == [1] and asked == []


@pytest.mark.asyncio
async def test_check_sees_conversation_summary_and_turn_results():
    ran, asked = [], []

    async def scenario():
        await se.guarded("lookup", {"q": "x"}, READ, "x", _runner(ran, {"ok": True, "order_id": "ORD-55555"}))
        return await se.guarded("book", {"order_id": "ORD-55555"}, ACTION, "x", _runner(ran),
                                llm=_llm("YES", asked))

    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    await _in_turn(scenario, history=history, message="cancel my order", summary="User lives in Lyon.")
    prompt = asked[0]
    assert "User lives in Lyon." in prompt and "cancel my order" in prompt
    assert "ORD-55555" in prompt and "Book a flight" in prompt


@pytest.mark.asyncio
async def test_check_failure_follows_on_error_setting(monkeypatch):
    ran = []

    async def broken(messages, **kw):
        return SimpleNamespace(ok=False, text="", error="down")

    monkeypatch.setattr(settings, "TOOL_ACTION_CHECK_ON_ERROR", "deny")
    out = await _in_turn(lambda: se.guarded("book", {"a": 1}, ACTION, "x", _runner(ran), llm=broken))
    assert out["error"] == "action_not_confirmed" and ran == []

    monkeypatch.setattr(settings, "TOOL_ACTION_CHECK_ON_ERROR", "allow")
    out = await _in_turn(lambda: se.guarded("book", {"a": 1}, ACTION, "x", _runner(ran), llm=broken))
    assert out["ok"] and ran == [1]


@pytest.mark.asyncio
async def test_policy_can_be_disabled(monkeypatch):
    ran, asked = [], []
    monkeypatch.setattr(settings, "TOOL_SIDE_EFFECTS_POLICY", False)
    out = await _in_turn(lambda: se.guarded("book", {"a": 1}, ACTION, "x", _runner(ran), llm=_llm("NO", asked)))
    assert out["ok"] and asked == []


@pytest.mark.asyncio
async def test_use_turn_binds_a_turn_for_older_tasks():
    seen = []

    async def worker(q):
        while True:
            fn = await q.get()
            seen.append(await fn())
            q.task_done()

    q: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(worker(q))  # created outside any turn

    async def scenario():
        turn = se.current_turn()

        async def call():
            with se.use_turn(turn):
                return se.current_turn() is turn
        q.put_nowait(call)
        await q.join()

    await _in_turn(scenario)
    task.cancel()
    assert seen == [True]
