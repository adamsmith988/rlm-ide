"""Low-overhead event-loop lag watchdog for worker saturation diagnosis."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LoopLagSnapshot:
    current_seconds: float = 0.0
    max_seconds: float = 0.0
    samples: int = 0
    over_threshold: int = 0
    updated_at: float = 0.0


class EventLoopMonitor:
    def __init__(self, *, interval: float = 0.5, warn_threshold: float = 0.2) -> None:
        self.interval = max(0.05, interval)
        self.warn_threshold = max(0.05, warn_threshold)
        self.snapshot = LoopLagSnapshot()
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="event-loop-monitor")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        target = time.monotonic() + self.interval
        while True:
            await asyncio.sleep(max(0.0, target - time.monotonic()))
            now = time.monotonic()
            lag = max(0.0, now - target)
            snap = self.snapshot
            snap.current_seconds = lag
            snap.max_seconds = max(snap.max_seconds, lag)
            snap.samples += 1
            snap.updated_at = time.time()
            if lag >= self.warn_threshold:
                snap.over_threshold += 1
                logger.warning("event loop lag %.3fs exceeds %.3fs", lag, self.warn_threshold)
            target = now + self.interval


_MONITOR = EventLoopMonitor()


def get_event_loop_monitor() -> EventLoopMonitor:
    return _MONITOR
