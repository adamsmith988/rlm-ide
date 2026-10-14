"""Chat turns that outlive their HTTP connection.

A turn runs as a background task that buffers its SSE events. Clients subscribe to
the buffer, so leaving or reloading the page only ends the subscription; the turn
finishes and persists normally. Only an explicit stop cancels it, which lets the
turn's own cleanup save the partial answer and release engine state.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator, AsyncIterator, Dict, List, Optional

from .config import settings

log = logging.getLogger(__name__)


class ChatTurn:
    def __init__(self, session_id: str, user_id: str, trace_id: str) -> None:
        self.session_id = session_id
        self.user_id = user_id
        self.trace_id = trace_id
        self.started = time.time()
        self.finished: Optional[float] = None
        self.stopped = False
        self.task: Optional[asyncio.Task] = None
        self._events: List[str] = []
        self._dropped = 0
        self._changed = asyncio.Condition()

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def stop_nowait(self) -> None:
        if self.running and self.task is not None:
            self.stopped = True
            self.task.cancel()

    def info(self) -> Dict[str, object]:
        return {
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "running": self.running,
            "stopped": self.stopped,
            "started": self.started,
            "finished": self.finished,
            "events": self._dropped + len(self._events),
        }

    async def _publish(self, event: str) -> None:
        async with self._changed:
            self._events.append(event)
            limit = int(settings.CHAT_TURN_REPLAY_MAX_EVENTS or 0)
            if limit > 0 and len(self._events) > limit:
                cut = len(self._events) - limit
                del self._events[:cut]
                self._dropped += cut
            self._changed.notify_all()

    async def _pump(self, source: AsyncGenerator[str, None]) -> None:
        try:
            async for event in source:
                await self._publish(event)
        finally:
            # A stop can land between events; closing the source runs its own cleanup.
            try:
                await source.aclose()
            except BaseException as exc:  # noqa: BLE001
                log.warning("chat turn source close failed session=%s err=%r", self.session_id, exc)
            self.finished = time.time()
            async with self._changed:
                self._changed.notify_all()

    async def subscribe(self, start: int = 0) -> AsyncIterator[str]:
        """Replay buffered events from ``start`` and follow until the turn ends."""
        index = max(0, int(start))
        while True:
            async with self._changed:
                while index >= self._dropped + len(self._events) and self.running:
                    await self._changed.wait()
                index = max(index, self._dropped)
                batch = self._events[index - self._dropped:]
                done = not self.running
            for event in batch:
                yield event
            index += len(batch)
            if done and index >= self._dropped + len(self._events):
                return


async def follow(turn: ChatTurn, start: int = 0) -> AsyncIterator[str]:
    """HTTP side of a turn: a disconnect ends only this subscription unless configured otherwise."""
    finished = False
    try:
        async for event in turn.subscribe(start):
            yield event
        finished = True
    finally:
        if not finished and not settings.CHAT_TURN_CONTINUE_ON_DISCONNECT:
            turn.stop_nowait()


class ChatTurnRegistry:
    def __init__(self) -> None:
        self._turns: Dict[str, ChatTurn] = {}

    def get(self, session_id: str) -> Optional[ChatTurn]:
        self._evict()
        return self._turns.get(str(session_id or ""))

    def _evict(self) -> None:
        keep = float(settings.CHAT_TURN_RETAIN_S or 0)
        now = time.time()
        for sid, turn in list(self._turns.items()):
            if not turn.running and turn.finished and now - turn.finished > keep:
                self._turns.pop(sid, None)

    async def wait_for(self, prev: Optional[ChatTurn]) -> bool:
        """Wait for an earlier still-running turn; False when the wait timed out."""
        if not prev or not prev.running or prev.task is None:
            return True
        limit = float(settings.CHAT_TURN_WAIT_PREVIOUS_S or 0)
        try:
            await asyncio.wait_for(asyncio.shield(prev.task), timeout=limit if limit > 0 else None)
        except asyncio.TimeoutError:
            return False
        except BaseException:  # noqa: BLE001 — the previous turn's own failure is not ours
            if not prev.task.done():
                raise
        return True

    def start(self, session_id: str, user_id: str, trace_id: str,
              source: AsyncGenerator[str, None]) -> ChatTurn:
        turn = ChatTurn(str(session_id or ""), str(user_id or ""), trace_id)
        turn.task = asyncio.create_task(turn._pump(source), name=f"chat-turn:{session_id}")
        if turn.session_id:
            self._turns[turn.session_id] = turn
        return turn

    async def stop(self, session_id: str, *, user_id: str = "") -> Optional[Dict[str, object]]:
        turn = self.get(session_id)
        if not turn:
            return None
        if user_id and turn.user_id and user_id != turn.user_id:
            raise PermissionError("turn belongs to another user")
        if turn.running and turn.task is not None:
            turn.stop_nowait()
            try:
                await asyncio.wait_for(asyncio.shield(turn.task),
                                       timeout=float(settings.CHAT_STOP_CLEANUP_TIMEOUT_S or 15.0) + 5)
            except BaseException:  # noqa: BLE001 — cancellation is the expected outcome
                pass
            log.info("chat turn stopped session=%s trace=%s", turn.session_id, turn.trace_id)
        return turn.info()


chat_turns = ChatTurnRegistry()
