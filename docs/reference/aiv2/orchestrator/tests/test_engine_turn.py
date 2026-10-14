from __future__ import annotations

import asyncio

from src.orchestrators._shared import engine_turn


def test_inner_turn_keeps_the_engine_context_checkpoint(monkeypatch):
    checkpoint = {"summary": "code TRK-1", "messages": 3}

    def fake_stream_for(_engine_id):
        async def stream(**_kw):
            yield {"type": "answer", "content": "ok"}
            yield {"type": "done", "final_answer": "ok", "self_compact": checkpoint}

        return stream

    monkeypatch.setattr(engine_turn, "stream_agent_for_id", fake_stream_for)
    monkeypatch.setattr(engine_turn, "_isolated_session_id", lambda e, s, x: s)
    turn = asyncio.run(engine_turn.run_engine_turn("native_tools", {"session_id": "s"}))
    assert turn.ok and turn.self_compact == checkpoint
