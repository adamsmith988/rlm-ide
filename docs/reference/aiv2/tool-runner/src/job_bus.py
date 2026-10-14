"""Redis visibility and remote-cancellation bus shared by all workloads."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional

try:
    import redis.asyncio as aioredis  # type: ignore
except Exception:  # pragma: no cover
    aioredis = None  # type: ignore

logger = logging.getLogger(__name__)

KEY_PREFIX = "tool_runner:jobs:state:"
CONTROL_CHANNEL = "tool_runner:jobs:control"
MAX_SNAPSHOT_BYTES = 768 * 1024
TERMINAL = frozenset({"completed", "failed", "cancelled", "dead"})


class JobBus:
    def __init__(
        self,
        redis_url: Optional[str],
        *,
        cancel_callback: Optional[Callable[[str], Awaitable[bool]]] = None,
        snapshot_ttl_seconds: int = 3600,
        client_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.redis_url = (redis_url or "").strip() or None
        self.cancel_callback = cancel_callback
        self.snapshot_ttl_seconds = max(60, int(snapshot_ttl_seconds))
        self.client_factory = client_factory
        self.client: Any = None
        self._listener: Optional["asyncio.Task[Any]"] = None

    @property
    def enabled(self) -> bool:
        return bool(self.redis_url and (self.client_factory or aioredis))

    def _new_client(self) -> Any:
        if self.client_factory:
            return self.client_factory()
        return aioredis.from_url(
            self.redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=5,
            health_check_interval=30,
        )

    async def start(self) -> None:
        if not self.enabled or self.client is not None:
            return
        self.client = self._new_client()
        await self.client.ping()
        self._listener = asyncio.create_task(
            self._listen_controls(), name="tool-job-control-listener"
        )

    async def stop(self) -> None:
        if self._listener:
            self._listener.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener
            self._listener = None
        if self.client is not None:
            close = getattr(self.client, "aclose", None) or getattr(
                self.client, "close", None
            )
            if close:
                value = close()
                if hasattr(value, "__await__"):
                    await value
            self.client = None

    async def publish(self, snapshot: Dict[str, Any]) -> None:
        if self.client is None:
            return
        job_id = str(snapshot.get("job_id") or "").strip()
        if not job_id:
            return
        encoded = json.dumps(snapshot, ensure_ascii=False, default=str)
        if len(encoded.encode("utf-8")) > MAX_SNAPSHOT_BYTES:
            slim = dict(snapshot)
            slim["progress"] = {"truncated": True}
            encoded = json.dumps(slim, ensure_ascii=False, default=str)
        key = f"{KEY_PREFIX}{job_id}"
        try:
            pipe = self.client.pipeline(transaction=False)
            pipe.hset(
                key,
                mapping={
                    "snapshot": encoded,
                    "status": str(snapshot.get("status") or ""),
                    "tenant_id": str(snapshot.get("tenant_id") or ""),
                    "updated_at": str(time.time()),
                },
            )
            pipe.expire(key, self.snapshot_ttl_seconds)
            pipe.publish(key, encoded)
            await pipe.execute()
        except Exception as exc:
            logger.warning("job bus publish failed for %s: %s", job_id, exc)

    async def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        if self.client is None:
            return None
        raw = await self.client.hget(f"{KEY_PREFIX}{job_id}", "snapshot")
        if not raw:
            return None
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else None
        except (TypeError, ValueError):
            return None

    async def request_cancel(self, job_id: str) -> bool:
        if self.client is None:
            return False
        payload = json.dumps({"type": "cancel", "job_id": job_id})
        key = f"{KEY_PREFIX}{job_id}"
        pipe = self.client.pipeline(transaction=False)
        pipe.hset(key, mapping={"cancel_requested": "1"})
        pipe.expire(key, self.snapshot_ttl_seconds)
        pipe.publish(CONTROL_CHANNEL, payload)
        await pipe.execute()
        return True

    async def events(self, job_id: str) -> AsyncIterator[Dict[str, Any]]:
        current = await self.get(job_id)
        if current:
            yield current
            if current.get("status") in TERMINAL:
                return
        if self.client is None:
            return
        pubsub = self.client.pubsub()
        await pubsub.subscribe(f"{KEY_PREFIX}{job_id}")
        try:
            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=15
                )
                if not message:
                    yield {"type": "heartbeat", "job_id": job_id}
                    continue
                try:
                    snapshot = json.loads(message.get("data") or "{}")
                except (TypeError, ValueError):
                    continue
                if not isinstance(snapshot, dict):
                    continue
                yield snapshot
                if snapshot.get("status") in TERMINAL:
                    return
        finally:
            with contextlib.suppress(Exception):
                await pubsub.unsubscribe(f"{KEY_PREFIX}{job_id}")
                await pubsub.close()

    async def _listen_controls(self) -> None:
        if self.client is None:
            return
        pubsub = self.client.pubsub()
        await pubsub.subscribe(CONTROL_CHANNEL)
        try:
            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=5
                )
                if not message:
                    await asyncio.sleep(0.05)
                    continue
                try:
                    payload = json.loads(message.get("data") or "{}")
                    job_id = str(payload.get("job_id") or "").strip()
                except (TypeError, ValueError, AttributeError):
                    continue
                if job_id and self.cancel_callback:
                    with contextlib.suppress(Exception):
                        await self.cancel_callback(job_id)
        finally:
            with contextlib.suppress(Exception):
                await pubsub.unsubscribe(CONTROL_CHANNEL)
                await pubsub.close()


def redis_url_from_env() -> Optional[str]:
    from .config import settings

    value = str(settings.TOOL_JOB_REDIS_URL or "").strip()
    return value or None
