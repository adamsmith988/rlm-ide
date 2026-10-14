from __future__ import annotations

from pathlib import Path

from src.orchestrators._chat_stream_base import ChatStreamEngine


def get_engine() -> ChatStreamEngine:
    return ChatStreamEngine(Path(__file__).resolve().parent)


Engine = ChatStreamEngine
create_engine = get_engine
