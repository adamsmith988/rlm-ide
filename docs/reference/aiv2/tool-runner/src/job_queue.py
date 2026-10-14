"""Crash-safe Redis Streams dispatch and ownership for generic tool jobs."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional, Protocol, Sequence, Tuple

logger = logging.getLogger(__name__)

KEY_PREFIX = "tool_runner:jobs:"
ACTIVE_STATUSES = frozenset({"queued", "claimed", "running"})
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "dead"})
_UNSAFE_KEY = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_key_part(value: Any) -> str:
    return _UNSAFE_KEY.sub("_", str(value or "default").strip()).strip("._") or "default"

_ENQUEUE_LUA = """
local status = redis.call('HGET', KEYS[1], 'status')
if status == 'queued' or status == 'claimed' or status == 'running' then
  return {0, redis.call('HGET', KEYS[1], 'stream_id') or ''}
end
local id = redis.call('XADD', KEYS[2], '*', 'job', ARGV[1], 'job_id', ARGV[2])
redis.call('HSET', KEYS[1],
  'job_id', ARGV[2], 'tool_id', ARGV[3], 'owner_namespace', ARGV[4],
  'tenant_id', ARGV[5], 'record_id', ARGV[6], 'status', 'queued',
  'queue_key', KEYS[2], 'enqueued_at', ARGV[7], 'enqueued_by', ARGV[8],
  'owner', ARGV[9], 'stream_id', id)
redis.call('EXPIRE', KEYS[1], ARGV[10])
return {1, id}
"""


class Role(str, Enum):
    ALL = "all"
    API = "api"
    WORKER = "worker"

    @property
    def executes_locally(self) -> bool:
        return self in (Role.ALL, Role.WORKER)

    @property
    def enqueues(self) -> bool:
        return self is Role.API

    @property
    def consumes_queue(self) -> bool:
        return self in (Role.ALL, Role.WORKER)


def parse_role(raw: Any) -> Role:
    value = str(raw or "").strip().lower()
    try:
        return Role(value) if value else Role.ALL
    except ValueError:
        logger.warning("TOOL_RUNNER_ROLE=%r unknown; using all", raw)
        return Role.ALL


def get_role() -> Role:
    from .config import settings
    return parse_role(settings.TOOL_RUNNER_ROLE)


def pod_identity(explicit: Optional[str] = None) -> str:
    for candidate in (explicit, os.environ.get("POD_IP"), os.environ.get("HOSTNAME")):
        value = str(candidate or "").strip()
        if value:
            return value
    return socket.gethostname() or "tool-runner"


class RedisLike(Protocol):
    async def ping(self) -> Any: ...
    async def set(self, key: str, value: str, ex: Optional[int] = None) -> Any: ...
    async def get(self, key: str) -> Optional[str]: ...
    async def expire(self, key: str, seconds: int) -> Any: ...
    async def delete(self, *keys: str) -> Any: ...
    async def hset(self, key: str, mapping: Dict[str, str]) -> Any: ...
    async def hgetall(self, key: str) -> Dict[str, str]: ...
    async def xadd(self, key: str, fields: Dict[str, str], maxlen: Optional[int] = None, approximate: bool = True) -> str: ...
    async def xgroup_create(self, key: str, groupname: str, id: str = "0", mkstream: bool = False) -> Any: ...
    async def xreadgroup(self, groupname: str, consumername: str, streams: Dict[str, str], count: Optional[int] = None, block: Optional[int] = None) -> Any: ...
    async def xack(self, key: str, group: str, *ids: str) -> int: ...
    async def xdel(self, key: str, *ids: str) -> int: ...
    async def xautoclaim(self, key: str, group: str, consumer: str, min_idle_time: int, start_id: str, count: int = 1) -> Any: ...
    async def xlen(self, key: str) -> int: ...
    async def xinfo_groups(self, key: str) -> Any: ...
    async def aclose(self) -> Any: ...


class QueueLedger(Protocol):
    async def on_enqueued(self, job: "Job", message_id: str) -> Optional[bool]: ...
    async def on_claimed(self, job: "Job", message_id: str, attempts: int) -> Optional[bool]: ...
    async def on_heartbeat(self, job_id: str) -> None: ...
    async def on_cancelled(self, job_id: str, error: str) -> None: ...
    async def on_terminal(self, job_id: str, status: str, fields: Dict[str, Any]) -> None: ...


@dataclass
class TerminalOutcome:
    status: str
    fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Job:
    job_id: str
    tool_id: str
    owner_namespace: str
    tenant_id: str
    args: Dict[str, Any]
    record_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    resource_class: str = "default"
    resource_weight: int = 1
    validate_args: bool = True
    enqueued_at: float = field(default_factory=time.time)
    enqueued_by: str = ""
    prefer_owner: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("job_id", "tool_id", "owner_namespace", "tenant_id"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ValueError(f"job envelope missing {name}")
            setattr(self, name, value)
        self.resource_class = str(self.resource_class or "default").strip() or "default"
        self.resource_weight = max(1, int(self.resource_weight or 1))

    def to_json(self) -> str:
        return json.dumps({
            "job_id": self.job_id,
            "tool_id": self.tool_id,
            "owner_namespace": self.owner_namespace,
            "tenant_id": self.tenant_id,
            "record_id": self.record_id,
            "metadata": self.metadata,
            "resource_class": self.resource_class,
            "resource_weight": self.resource_weight,
            "args": self.args,
            "validate_args": self.validate_args,
            "enqueued_at": self.enqueued_at,
            "enqueued_by": self.enqueued_by,
            "prefer_owner": self.prefer_owner,
        }, ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, raw: str) -> "Job":
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("job envelope must be an object")
        return cls(
            job_id=value.get("job_id"),
            tool_id=value.get("tool_id"),
            owner_namespace=value.get("owner_namespace"),
            tenant_id=value.get("tenant_id"),
            record_id=str(value["record_id"]) if value.get("record_id") is not None else None,
            metadata=dict(value.get("metadata") or {}),
            resource_class=value.get("resource_class") or "default",
            resource_weight=value.get("resource_weight") or 1,
            args=dict(value.get("args") or {}),
            validate_args=bool(value.get("validate_args", True)),
            enqueued_at=float(value.get("enqueued_at") or time.time()),
            enqueued_by=str(value.get("enqueued_by") or ""),
            prefer_owner=str(value["prefer_owner"]) if value.get("prefer_owner") else None,
        )


def _redis_from_url(url: str) -> Optional[RedisLike]:
    try:
        import redis.asyncio as redis
    except Exception:
        logger.warning("redis package unavailable; job queue disabled")
        return None
    return redis.from_url(url, decode_responses=True, socket_connect_timeout=5, socket_timeout=None)


class JobQueue:
    def __init__(
        self,
        client: RedisLike,
        *,
        pod: str,
        queue_key: str = f"{KEY_PREFIX}queue",
        group: str = "tool-runner",
        max_attempts: int = 5,
        claim_idle_ms: int = 60000,
        reclaim_interval_sec: float = 10,
        stream_maxlen: int = 100000,
        ledger: Optional[QueueLedger] = None,
        owner_ttl_sec: int = 90,
        owner_linger_sec: int = 900,
        record_ttl_sec: int = 86400,
        supported_resource_classes: Sequence[str] = ("default",),
        worker_capabilities: Sequence[str] = (),
    ) -> None:
        if not str(queue_key).startswith(KEY_PREFIX):
            raise ValueError(f"queue key must start with {KEY_PREFIX}")
        self._client = client
        self.pod = str(pod or "").strip() or pod_identity()
        self.queue_key = queue_key
        self.group = group
        self.dlq_key = f"{KEY_PREFIX}dlq"
        self.max_attempts = max(1, int(max_attempts))
        self.claim_idle_ms = max(1, int(claim_idle_ms))
        self.reclaim_interval_sec = max(.05, float(reclaim_interval_sec))
        self.stream_maxlen = max(100, int(stream_maxlen))
        self._ledger = ledger
        self.owner_ttl_sec = max(5, int(owner_ttl_sec))
        self.owner_linger_sec = max(0, int(owner_linger_sec))
        self.record_ttl_sec = max(60, int(record_ttl_sec))
        self.supported_resource_classes = frozenset(
            _safe_key_part(value)
            for value in supported_resource_classes
            if str(value or "").strip()
        ) or frozenset({"default"})
        self.worker_capabilities = frozenset(
            str(value or "").strip()
            for value in worker_capabilities
            if str(value or "").strip()
        )
        self._groups: set[str] = set()
        self._deliveries: Dict[str, Tuple[str, str]] = {}
        self._last_reclaim = 0.0
        self._connected = False

    def pod_queue_key(self, pod: Optional[str] = None) -> str:
        return f"{KEY_PREFIX}queue:owner:{str(pod or self.pod)}"

    def resource_queue_key(self, resource_class: str) -> str:
        return f"{self.queue_key}:{_safe_key_part(resource_class)}"

    @staticmethod
    def job_key(job_id: str) -> str:
        return f"{KEY_PREFIX}record:{job_id}"

    @staticmethod
    def owner_key(job_id: str) -> str:
        return f"{KEY_PREFIX}owner:{job_id}"

    @staticmethod
    def owner_record_key(owner_namespace: str, record_id: str) -> str:
        return f"{KEY_PREFIX}owner-record:{owner_namespace}:{record_id}"

    async def connect(self) -> None:
        await self._client.ping()
        await self._ensure_group(self.pod_queue_key())
        for resource_class in self.supported_resource_classes:
            await self._ensure_group(self.resource_queue_key(resource_class))
        self._connected = True

    async def _ensure_group(self, key: str) -> None:
        if key in self._groups:
            return
        try:
            await self._client.xgroup_create(key, self.group, id="0", mkstream=True)
        except Exception as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        self._groups.add(key)

    @property
    def connected(self) -> bool:
        return self._connected

    def mark_disconnected(self) -> None:
        self._connected = False

    async def close(self) -> None:
        self._connected = False
        try:
            await self._client.aclose()
        except Exception:
            pass

    async def enqueue(self, job: Job) -> int:
        key = (
            self.pod_queue_key(job.prefer_owner)
            if job.prefer_owner
            else self.resource_queue_key(job.resource_class)
        )
        if self._ledger is not None and await self._ledger.on_enqueued(job, "") is False:
            return max(1, await self._stream_depth(key))
        await self._ensure_group(key)
        evaluate = getattr(self._client, "eval", None)
        if callable(evaluate):
            await evaluate(
                _ENQUEUE_LUA, 2, self.job_key(job.job_id), key, job.to_json(), job.job_id,
                job.tool_id, job.owner_namespace, job.tenant_id, job.record_id or "",
                job.enqueued_at, job.enqueued_by, job.prefer_owner or "", self.record_ttl_sec,
            )
        else:
            existing = await self.job_get(job.job_id)
            if existing and existing.get("status") in ACTIVE_STATUSES:
                return max(1, await self._stream_depth(key))
            message_id = str(await self._client.xadd(
                key, {"job": job.to_json(), "job_id": job.job_id},
                maxlen=self.stream_maxlen, approximate=True,
            ))
            await self._client.hset(self.job_key(job.job_id), mapping={
                "job_id": job.job_id, "tool_id": job.tool_id,
                "owner_namespace": job.owner_namespace, "tenant_id": job.tenant_id,
                "record_id": job.record_id or "", "status": "queued", "queue_key": key,
                "enqueued_at": str(job.enqueued_at), "enqueued_by": job.enqueued_by,
                "owner": job.prefer_owner or "", "stream_id": message_id,
            })
            await self._client.expire(self.job_key(job.job_id), self.record_ttl_sec)
        return max(1, await self._stream_depth(key))

    async def dequeue(self, timeout: float) -> Optional[Job]:
        keys = (
            self.pod_queue_key(),
            *(
                self.resource_queue_key(resource_class)
                for resource_class in sorted(self.supported_resource_classes)
            ),
        )
        for key in keys:
            await self._ensure_group(key)
        try:
            message = await self._reclaim_one(keys)
            if message is None:
                reply = await self._client.xreadgroup(
                    self.group, self.pod, streams={key: ">" for key in keys}, count=1,
                    block=max(1, int(max(.01, timeout) * 1000)),
                )
                message = self._parse_reply(reply)
        except asyncio.TimeoutError:
            return None
        except Exception as exc:
            if "timeout" in type(exc).__name__.lower() or "timeout" in str(exc).lower():
                return None
            raise
        if message is None:
            return None
        key, message_id, fields = message
        raw = str(fields.get("job") or "")
        try:
            job = Job.from_json(raw)
        except Exception as exc:
            await self._dead_letter(key, message_id, raw, "", 0, f"malformed envelope: {exc}")
            return None
        required_capability = str(
            job.metadata.get("required_capability") or ""
        ).strip()
        if (
            _safe_key_part(job.resource_class) not in self.supported_resource_classes
            or required_capability
            and required_capability not in self.worker_capabilities
        ):
            await self._dead_letter(
                key,
                message_id,
                raw,
                job.job_id,
                0,
                "worker_capability_mismatch",
            )
            return None
        record = await self.job_get(job.job_id) or {}
        status, recorded_id = str(record.get("status") or ""), str(record.get("stream_id") or "")
        if recorded_id == message_id and status in TERMINAL_STATUSES:
            await self._discard(key, message_id)
            return None
        if status in {"claimed", "running"} and await self.owner_get(job.job_id):
            return None
        if recorded_id and recorded_id != message_id and status in ACTIVE_STATUSES | TERMINAL_STATUSES:
            await self._discard(key, message_id)
            return None
        attempts = int(record.get("attempts") or 0) + 1
        if attempts > self.max_attempts:
            await self._dead_letter(key, message_id, raw, job.job_id, attempts, "max attempts exceeded")
            return None
        if self._ledger is not None:
            try:
                claimed = await self._ledger.on_claimed(job, message_id, attempts)
            except RuntimeError as exc:
                if str(exc).startswith("job_ledger_claim_busy:"):
                    return None
                raise
            if claimed is False:
                await self._discard(key, message_id)
                return None
        self._deliveries[job.job_id] = (key, message_id)
        await self.job_status_set(job.job_id, "claimed", owner=self.pod, attempts=attempts, stream_id=message_id)
        return job

    @staticmethod
    def _parse_reply(reply: Any) -> Optional[Tuple[str, str, Dict[str, str]]]:
        if not reply or not reply[0][1]:
            return None
        key, messages = reply[0]
        message_id, fields = messages[0]
        return str(key), str(message_id), dict(fields)

    async def _reclaim_one(self, keys: Sequence[str]) -> Optional[Tuple[str, str, Dict[str, str]]]:
        now = time.monotonic()
        if now - self._last_reclaim < self.reclaim_interval_sec:
            return None
        self._last_reclaim = now
        for key in keys:
            reply = await self._client.xautoclaim(key, self.group, self.pod, self.claim_idle_ms, "0-0", count=1)
            messages = reply[1] if reply and len(reply) > 1 else []
            if messages:
                message_id, fields = messages[0]
                return key, str(message_id), dict(fields)
        return None

    async def ack_terminal(self, job_id: str, status: str, **fields: Any) -> bool:
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"non-terminal status cannot be acknowledged: {status}")
        # Ordering is intentional: PostgreSQL commit, Redis terminal record, ACK, delete.
        if self._ledger is not None:
            await self._ledger.on_terminal(job_id, status, dict(fields))
        await self.job_status_set(job_id, status, **fields)
        delivery = self._deliveries.pop(job_id, None)
        if delivery is None:
            return False
        await self._discard(*delivery)
        return True

    async def cancel_queued(self, job_id: str, *, error: str = "cancelled") -> bool:
        record = await self.job_get(job_id)
        if not record or str(record.get("status") or "") != "queued":
            return False
        if self._ledger is not None:
            await self._ledger.on_cancelled(job_id, error)
        await self.job_status_set(job_id, "cancelled", error=error)
        queue_key = str(record.get("queue_key") or "")
        stream_id = str(record.get("stream_id") or "")
        if queue_key and stream_id:
            await self._discard(queue_key, stream_id)
        return True

    async def _discard(self, key: str, message_id: str) -> None:
        await self._client.xack(key, self.group, message_id)
        await self._client.xdel(key, message_id)

    async def _dead_letter(
        self, source: str, message_id: str, raw: str, job_id: str, attempts: int, reason: str
    ) -> None:
        await self._client.xadd(self.dlq_key, {
            "job": raw, "job_id": job_id, "source_stream": source, "source_id": message_id,
            "attempts": str(attempts), "reason": reason, "failed_at": str(time.time()),
        }, maxlen=self.stream_maxlen, approximate=True)
        if job_id:
            if self._ledger is not None:
                await self._ledger.on_terminal(job_id, "dead", {"attempts": attempts, "error": reason})
            await self.job_status_set(job_id, "dead", attempts=attempts, error=reason)
        await self._discard(source, message_id)

    async def _stream_depth(self, key: str) -> int:
        try:
            for info in await self._client.xinfo_groups(key) or []:
                if str(info.get("name") or "") == self.group:
                    return int(info.get("pending") or 0) + int(info.get("lag") or 0)
        except Exception:
            pass
        return int(await self._client.xlen(key) or 0)

    async def queue_depth(self) -> Dict[str, Any]:
        resources = {
            resource_class: await self._stream_depth(
                self.resource_queue_key(resource_class)
            )
            for resource_class in sorted(self.supported_resource_classes)
        }
        return {
            "shared": sum(resources.values()),
            "pod": await self._stream_depth(self.pod_queue_key()),
            "resources": resources,
        }

    async def job_get(self, job_id: str) -> Optional[Dict[str, str]]:
        value = await self._client.hgetall(self.job_key(job_id))
        return dict(value) if value else None

    async def job_status_set(self, job_id: str, status: str, **fields: Any) -> None:
        mapping = {"status": status, "updated_at": str(time.time())}
        mapping.update({str(k): "" if v is None else str(v) for k, v in fields.items()})
        await self._client.hset(self.job_key(job_id), mapping=mapping)
        await self._client.expire(self.job_key(job_id), self.record_ttl_sec)

    async def job_is_active(self, job_id: str) -> bool:
        record = await self.job_get(job_id)
        if not record:
            return False
        if record.get("status") == "queued":
            return True
        return record.get("status") in {"claimed", "running"} and bool(await self.owner_get(job_id))

    async def owner_set(
        self, job_id: str, *, ttl: Optional[int] = None,
        owner_namespace: Optional[str] = None, record_ids: Iterable[str] = (),
    ) -> None:
        seconds = int(ttl if ttl is not None else self.owner_ttl_sec)
        await self._client.set(self.owner_key(job_id), self.pod, ex=seconds)
        if owner_namespace:
            for record_id in record_ids:
                value = str(record_id or "").strip()
                if value:
                    await self._client.set(self.owner_record_key(owner_namespace, value), self.pod, ex=seconds)

    async def owner_get(self, job_id: str) -> Optional[str]:
        value = str(await self._client.get(self.owner_key(job_id)) or "").strip()
        return value or None

    async def owner_get_for_record(self, owner_namespace: str, record_id: str) -> Optional[str]:
        value = str(await self._client.get(self.owner_record_key(owner_namespace, record_id)) or "").strip()
        return value or None

    async def owner_release(
        self, job_id: str, *, owner_namespace: Optional[str] = None, record_ids: Iterable[str] = (),
    ) -> None:
        keys = [self.owner_key(job_id)]
        if owner_namespace:
            keys.extend(self.owner_record_key(owner_namespace, str(r)) for r in record_ids if str(r or "").strip())
        if self.owner_linger_sec <= 0:
            await self._client.delete(*keys)
        else:
            for key in keys:
                await self._client.set(key, self.pod, ex=self.owner_linger_sec)


TerminalSource = Callable[[str], Awaitable[TerminalOutcome] | TerminalOutcome]
RecordSource = Callable[
    [str], Iterable[str] | Awaitable[Iterable[str]]
]


async def owner_heartbeat(
    queue: JobQueue,
    job: Job,
    *,
    alive: Callable[[], bool],
    terminal_source: TerminalSource,
    interval_sec: float,
    record_source: RecordSource = lambda _job_id: (),
) -> None:
    seen: list[str] = []
    try:
        while True:
            records_value = record_source(job.job_id)
            if hasattr(records_value, "__await__"):
                records_value = await records_value  # type: ignore[assignment]
            records = list(records_value)
            seen.extend(value for value in records if value not in seen)
            try:
                target = get_queue() or queue
                await target.owner_set(
                    job.job_id, owner_namespace=job.owner_namespace, record_ids=records,
                )
                if target._ledger is not None:
                    await target._ledger.on_heartbeat(job.job_id)
            except Exception:
                logger.warning("job owner heartbeat failed job_id=%s", job.job_id, exc_info=True)
            if not alive():
                break
            await asyncio.sleep(max(1.0, float(interval_sec)))
    finally:
        target = get_queue() or queue
        outcome = terminal_source(job.job_id)
        if hasattr(outcome, "__await__"):
            outcome = await outcome  # type: ignore[assignment]
        if not isinstance(outcome, TerminalOutcome):
            raise TypeError("terminal_source must return TerminalOutcome")
        await target.ack_terminal(job.job_id, outcome.status, **outcome.fields)
        await target.owner_release(job.job_id, owner_namespace=job.owner_namespace, record_ids=seen)


Executor = Callable[[Job], Awaitable[Any]]


class JobWorker:
    def __init__(
        self, queue: JobQueue, *, execute: Executor,
        has_capacity: Callable[[], bool] = lambda: True,
        draining: Callable[[], bool] = lambda: False,
        poll_sec: int = 5, busy_backoff_sec: float = 1,
    ) -> None:
        self._queue, self._execute = queue, execute
        self._has_capacity, self._draining = has_capacity, draining
        self.poll_sec, self.busy_backoff_sec = max(1, int(poll_sec)), max(.05, float(busy_backoff_sec))
        self._task: Optional[asyncio.Task[None]] = None
        self._stop = asyncio.Event()
        self.consumed = self.failed = 0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> asyncio.Task[None]:
        if not self.running:
            self._stop.clear()
            self._task = asyncio.create_task(self._loop(), name="job-queue-worker")
        return self._task  # type: ignore[return-value]

    async def stop(self, timeout: float = 10) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            except Exception:
                pass
        self._task = None

    async def _loop(self) -> None:
        while not self._stop.is_set() and not self._draining():
            try:
                if not self._has_capacity():
                    await asyncio.sleep(self.busy_backoff_sec)
                    continue
                job = await self._queue.dequeue(self.poll_sec)
                if job is None:
                    continue
                self.consumed += 1
                await self._execute(job)
                await self._queue.job_status_set(job.job_id, "running", owner=self._queue.pod, tool_id=job.tool_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.failed += 1
                logger.exception("queued job failed to start")
                if "job" in locals() and job is not None:
                    try:
                        await self._queue.ack_terminal(job.job_id, "failed", error=str(exc))
                    except Exception:
                        pass


_QUEUE: Optional[JobQueue] = None
_WORKER: Optional[JobWorker] = None
_SUPERVISOR: Optional["QueueSupervisor"] = None


def build_queue_from_settings(client: Optional[RedisLike] = None) -> Optional[JobQueue]:
    from .config import settings
    resource_classes = tuple(
        value.strip()
        for value in str(settings.TOOL_JOB_WORKER_RESOURCE_CLASSES or "").split(",")
        if value.strip()
    )
    worker_capabilities = tuple(
        value.strip()
        for value in str(settings.TOOL_JOB_WORKER_CAPABILITIES or "").split(",")
        if value.strip()
    )
    if client is None:
        if not str(settings.TOOL_JOB_REDIS_URL or "").strip():
            return None
        client = _redis_from_url(str(settings.TOOL_JOB_REDIS_URL))
        if client is None:
            return None
    ledger_hook = None
    from .job_ledger import QueueLedgerAdapter, get_job_ledger
    ledger = get_job_ledger()
    if ledger.ready:
        ledger_hook = QueueLedgerAdapter(ledger)
    elif settings.TOOL_JOB_LEDGER_REQUIRED:
        raise RuntimeError(ledger.error or "job_ledger_not_ready")
    return JobQueue(
        client, pod=pod_identity(settings.TOOL_JOB_POD_IDENTITY),
        queue_key=settings.TOOL_JOB_QUEUE_KEY, group=settings.TOOL_JOB_QUEUE_GROUP,
        max_attempts=settings.TOOL_JOB_QUEUE_MAX_ATTEMPTS,
        claim_idle_ms=settings.TOOL_JOB_QUEUE_CLAIM_IDLE_MS,
        reclaim_interval_sec=settings.TOOL_JOB_QUEUE_RECLAIM_INTERVAL_SEC,
        stream_maxlen=settings.TOOL_JOB_QUEUE_STREAM_MAXLEN, ledger=ledger_hook,
        owner_ttl_sec=settings.TOOL_JOB_OWNER_TTL_SEC,
        owner_linger_sec=settings.TOOL_JOB_OWNER_LINGER_SEC,
        record_ttl_sec=settings.TOOL_JOB_RECORD_TTL_SEC,
        supported_resource_classes=resource_classes,
        worker_capabilities=worker_capabilities,
    )


async def init_queue(client: Optional[RedisLike] = None) -> Optional[JobQueue]:
    global _QUEUE
    queue = build_queue_from_settings(client)
    if queue is None:
        _QUEUE = None
        return None
    try:
        await queue.connect()
    except Exception:
        logger.warning("job queue unavailable", exc_info=True)
        _QUEUE = None
        return None
    _QUEUE = queue
    return queue


def get_queue() -> Optional[JobQueue]:
    return _QUEUE


def set_queue_for_tests(queue: Optional[JobQueue]) -> None:
    global _QUEUE
    _QUEUE = queue


class QueueSupervisor:
    """Reconnect Redis, replay the durable outbox and supervise consumption."""

    def __init__(
        self,
        execute: Executor,
        *,
        client_factory: Optional[Callable[[], Optional[RedisLike]]] = None,
        has_capacity: Callable[[], bool] = lambda: True,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self.execute = execute
        self.client_factory = client_factory
        self.has_capacity = has_capacity
        self.sleep = sleep
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task[None]] = None
        self.last_error: Optional[str] = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> asyncio.Task[None]:
        if not self.running:
            self._stop.clear()
            self._task = asyncio.create_task(self._loop(), name="job-queue-supervisor")
        return self._task  # type: ignore[return-value]

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _loop(self) -> None:
        global _QUEUE, _WORKER
        from .config import settings
        from .job_admission import get_admission
        from .job_ledger import get_job_ledger

        initial = max(.01, float(settings.TOOL_JOB_REDIS_RECONNECT_INITIAL_SEC))
        maximum = max(initial, float(settings.TOOL_JOB_REDIS_RECONNECT_MAX_SEC))
        jitter = max(0.0, float(settings.TOOL_JOB_REDIS_RECONNECT_JITTER))
        ping_sec = max(.05, float(settings.TOOL_JOB_REDIS_SUPERVISOR_PING_SEC))
        delay = initial
        while not self._stop.is_set():
            try:
                ledger = get_job_ledger()
                if settings.TOOL_JOB_LEDGER_REQUIRED and not ledger.ready:
                    await ledger.start()
                    if not ledger.ready:
                        raise RuntimeError(ledger.error or "job_ledger_not_ready")
                client = self.client_factory() if self.client_factory else None
                queue = build_queue_from_settings(client)
                if queue is None:
                    return
                await queue.connect()
                if queue._ledger is not None:
                    for payload in await ledger.dispatchable():
                        try:
                            await queue.enqueue(Job.from_json(json.dumps(payload)))
                        except Exception:
                            logger.warning("job outbox dispatch failed", exc_info=True)
                            break
                _QUEUE = queue
                self.last_error = None
                delay = initial
                if get_role().consumes_queue:
                    _WORKER = JobWorker(
                        queue, execute=self.execute, has_capacity=self.has_capacity,
                        draining=lambda: get_admission().draining,
                        poll_sec=settings.TOOL_JOB_WORKER_POLL_SEC,
                        busy_backoff_sec=settings.TOOL_JOB_WORKER_BUSY_BACKOFF_SEC,
                    )
                    _WORKER.start()
                while not self._stop.is_set() and queue.connected:
                    await self.sleep(ping_sec)
                    await queue._client.ping()
                    if get_role().consumes_queue and (_WORKER is None or not _WORKER.running):
                        raise ConnectionError("job queue consumer stopped")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = str(exc)
                logger.warning("job queue disconnected; retrying: %s", exc)
            if _WORKER is not None:
                await _WORKER.stop()
                _WORKER = None
            old, _QUEUE = _QUEUE, None
            if old is not None:
                await old.close()
            if self._stop.is_set():
                break
            await self.sleep(max(0.0, delay * (1 + random.uniform(-jitter, jitter))))
            delay = min(maximum, delay * 2)


async def start_queue_supervisor(
    execute: Executor,
    *,
    client_factory: Optional[Callable[[], Optional[RedisLike]]] = None,
    has_capacity: Callable[[], bool] = lambda: True,
) -> Optional[QueueSupervisor]:
    global _SUPERVISOR
    from .config import settings
    if not str(settings.TOOL_JOB_REDIS_URL or "").strip() and client_factory is None:
        return None
    if _SUPERVISOR is not None:
        await _SUPERVISOR.stop()
    _SUPERVISOR = QueueSupervisor(
        execute, client_factory=client_factory, has_capacity=has_capacity,
    )
    _SUPERVISOR.start()
    return _SUPERVISOR


def get_supervisor() -> Optional[QueueSupervisor]:
    return _SUPERVISOR


async def start_worker(execute: Executor, has_capacity: Callable[[], bool] = lambda: True) -> Optional[JobWorker]:
    global _WORKER
    queue = get_queue()
    if queue is None or not get_role().consumes_queue:
        return None
    from .config import settings
    from .job_admission import get_admission
    _WORKER = JobWorker(
        queue, execute=execute, has_capacity=has_capacity,
        draining=lambda: get_admission().draining,
        poll_sec=settings.TOOL_JOB_WORKER_POLL_SEC,
        busy_backoff_sec=settings.TOOL_JOB_WORKER_BUSY_BACKOFF_SEC,
    )
    _WORKER.start()
    return _WORKER


def get_worker() -> Optional[JobWorker]:
    return _WORKER


def start_owner_heartbeat(
    job: Job, *, alive: Callable[[], bool], terminal_source: TerminalSource,
    record_source: RecordSource = lambda _job_id: (),
) -> Optional[asyncio.Task[None]]:
    queue = get_queue()
    if queue is None:
        return None
    from .config import settings
    return asyncio.create_task(owner_heartbeat(
        queue, job, alive=alive, terminal_source=terminal_source,
        record_source=record_source, interval_sec=settings.TOOL_JOB_OWNER_HEARTBEAT_SEC,
    ), name=f"job-owner:{job.job_id}")


def queue_readiness() -> Dict[str, Any]:
    from .config import settings
    role = get_role()
    required = bool(settings.TOOL_JOB_REDIS_URL) or role in (Role.API, Role.WORKER)
    producer = bool(_QUEUE and _QUEUE.connected)
    consumer = bool(_WORKER and _WORKER.running and producer)
    return {
        "ready": not required or (producer and (not role.consumes_queue or consumer)),
        "required": required, "producer": producer, "consumer": consumer,
        "error": _SUPERVISOR.last_error if _SUPERVISOR is not None else None,
    }


async def shutdown_queue() -> None:
    global _WORKER, _SUPERVISOR
    supervisor, _SUPERVISOR = _SUPERVISOR, None
    if supervisor is not None:
        await supervisor.stop()
    if _WORKER is not None:
        await _WORKER.stop()
        _WORKER = None


async def close_queue() -> None:
    global _QUEUE
    queue, _QUEUE = _QUEUE, None
    if queue is not None:
        await queue.close()
