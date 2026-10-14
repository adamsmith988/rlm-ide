import asyncio

import pytest

from src.chat_turns import ChatTurnRegistry, follow
from src.config import settings


def _source(n, log, delay=0.02):
    async def gen():
        try:
            for i in range(n):
                await asyncio.sleep(delay)
                yield f"e{i}"
            log.append("completed")
        finally:
            log.append("cleanup")

    return gen()


@pytest.mark.asyncio
async def test_disconnect_does_not_stop_the_turn(monkeypatch):
    monkeypatch.setattr(settings, "CHAT_TURN_CONTINUE_ON_DISCONNECT", True)
    reg, log = ChatTurnRegistry(), []
    turn = reg.start("s1", "u1", "t1", _source(5, log))
    sub = follow(turn)
    assert await sub.__anext__() == "e0"
    await sub.aclose()
    await turn.task
    assert log == ["completed", "cleanup"]
    assert [e async for e in follow(turn)] == [f"e{i}" for i in range(5)]


@pytest.mark.asyncio
async def test_disconnect_can_be_configured_to_stop(monkeypatch):
    monkeypatch.setattr(settings, "CHAT_TURN_CONTINUE_ON_DISCONNECT", False)
    reg, log = ChatTurnRegistry(), []
    turn = reg.start("s1", "u1", "t1", _source(50, log))
    sub = follow(turn)
    await sub.__anext__()
    await sub.aclose()
    with pytest.raises(asyncio.CancelledError):
        await turn.task
    assert "completed" not in log and "cleanup" in log


@pytest.mark.asyncio
async def test_stop_cancels_and_runs_cleanup():
    reg, log = ChatTurnRegistry(), []
    turn = reg.start("s1", "u1", "t1", _source(50, log))
    await asyncio.sleep(0.05)
    info = await reg.stop("s1", user_id="u1")
    assert info["stopped"] is True and info["running"] is False
    assert log == ["cleanup"]


@pytest.mark.asyncio
async def test_stop_refuses_other_users():
    reg = ChatTurnRegistry()
    reg.start("s1", "u1", "t1", _source(50, []))
    with pytest.raises(PermissionError):
        await reg.stop("s1", user_id="u2")
    await reg.stop("s1")


@pytest.mark.asyncio
async def test_reattach_replays_from_offset():
    reg = ChatTurnRegistry()
    turn = reg.start("s1", "u1", "t1", _source(4, []))
    await turn.task
    assert [e async for e in follow(reg.get("s1"), 2)] == ["e2", "e3"]


@pytest.mark.asyncio
async def test_new_turn_waits_for_running_turn():
    reg, log = ChatTurnRegistry(), []
    first = reg.start("s1", "u1", "t1", _source(5, log))
    assert await reg.wait_for(first) is True
    assert log == ["completed", "cleanup"]


@pytest.mark.asyncio
async def test_wait_previous_honours_limit(monkeypatch):
    monkeypatch.setattr(settings, "CHAT_TURN_WAIT_PREVIOUS_S", 0.05)
    reg = ChatTurnRegistry()
    first = reg.start("s1", "u1", "t1", _source(50, []))
    assert await reg.wait_for(first) is False
    await reg.stop("s1")
