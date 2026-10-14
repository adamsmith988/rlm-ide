"""Weighted, tenant-fair admission with atomic Redis resource leases."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Optional

logger = logging.getLogger(__name__)
Publisher = Callable[..., None]


class AdmissionDraining(RuntimeError):
    pass


@dataclass
class AdmissionTicket:
    job_id: str
    owner_namespace: str
    tenant_id: str
    resource_class: str
    resource_weight: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    queued: bool = False
    position: int = 0
    enqueued_at: float = field(default_factory=time.time)
    admitted: bool = False
    _holds_cluster: bool = field(default=False, repr=False)
    _renew_task: Optional[asyncio.Task[Any]] = field(default=None, repr=False)
    _ready: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


_ACQUIRE_LUA = """
local global = KEYS[1]
local resource = KEYS[2]
local tenant = KEYS[3]
local tenants_wait = KEYS[4]
local tenant_wait = KEYS[5]
local leases = KEYS[6]
local now = tonumber(ARGV[1])
local expires = tonumber(ARGV[2])
local job = ARGV[3]
local weight = tonumber(ARGV[4])
local global_max = tonumber(ARGV[5])
local tenant_max = tonumber(ARGV[6])
local resource_max = tonumber(ARGV[7])
local tenant_id = ARGV[8]

local expired = redis.call('ZRANGEBYSCORE', leases, '-inf', now)
for _, member in ipairs(expired) do
  local encoded = redis.call('HGET', leases .. ':weights', member)
  if encoded then
    local w, resource_key, tenant_key = string.match(encoded, '([^|]+)|([^|]+)|([^|]+)')
    redis.call('DECRBY', global, tonumber(w))
    redis.call('DECRBY', resource_key, tonumber(w))
    redis.call('DECRBY', tenant_key, tonumber(w))
    redis.call('HDEL', leases .. ':weights', member)
  end
  redis.call('ZREM', leases, member)
end
if redis.call('ZSCORE', leases, job) then
  redis.call('ZADD', leases, expires, job)
  return 1
end
redis.call('ZADD', tenant_wait, 'NX', now, job)
redis.call('ZADD', tenants_wait, 'NX', now, tenant_id)
local next_tenant = redis.call('ZRANGE', tenants_wait, 0, 0)[1]
local next_job = redis.call('ZRANGE', tenant_wait, 0, 0)[1]
if next_tenant ~= tenant_id or next_job ~= job then return 0 end
if global_max > 0 and tonumber(redis.call('GET', global) or '0') + weight > global_max then return 0 end
if resource_max > 0 and tonumber(redis.call('GET', resource) or '0') + weight > resource_max then return 0 end
if tenant_max > 0 and tonumber(redis.call('GET', tenant) or '0') + weight > tenant_max then return 0 end
redis.call('INCRBY', global, weight)
redis.call('INCRBY', resource, weight)
redis.call('INCRBY', tenant, weight)
redis.call('ZADD', leases, expires, job)
redis.call('HSET', leases .. ':weights', job, weight .. '|' .. resource .. '|' .. tenant)
redis.call('ZREM', tenant_wait, job)
if redis.call('ZCARD', tenant_wait) == 0 then
  redis.call('ZREM', tenants_wait, tenant_id)
else
  redis.call('ZADD', tenants_wait, now, tenant_id)
end
redis.call('PEXPIRE', global, math.max(1000, (expires-now)*2))
redis.call('PEXPIRE', resource, math.max(1000, (expires-now)*2))
redis.call('PEXPIRE', tenant, math.max(1000, (expires-now)*2))
return 1
"""

_RELEASE_LUA = """
if redis.call('ZREM', KEYS[1], ARGV[1]) == 1 then
  local encoded = redis.call('HGET', KEYS[1] .. ':weights', ARGV[1])
  if encoded then
    local w, resource_key, tenant_key = string.match(encoded, '([^|]+)|([^|]+)|([^|]+)')
    redis.call('DECRBY', KEYS[2], tonumber(w))
    redis.call('DECRBY', resource_key, tonumber(w))
    redis.call('DECRBY', tenant_key, tonumber(w))
    redis.call('HDEL', KEYS[1] .. ':weights', ARGV[1])
  end
end
redis.call('ZREM', KEYS[6], ARGV[1])
if redis.call('ZCARD', KEYS[6]) == 0 then redis.call('ZREM', KEYS[5], ARGV[2]) end
return 1
"""


def _safe(value: str) -> str:
    return str(value or "").replace(":", "_")[:160]


class RedisAdmissionLease:
    def __init__(
        self, *, max_weight: int, max_weight_per_tenant: int,
        resource_limits: Optional[Dict[str, int]] = None,
    ) -> None:
        from .config import settings
        self.max_weight = max_weight
        self.max_weight_per_tenant = max_weight_per_tenant
        self.resource_limits = dict(resource_limits or {})
        self.ttl_ms = max(15000, int(settings.TOOL_JOB_ADMISSION_LEASE_MS))
        self.redis_url = str(settings.TOOL_JOB_REDIS_URL or "").strip()
        self.enabled = bool(self.redis_url and settings.TOOL_JOB_CLUSTER_ADMISSION_ENABLED)
        self._client: Any = None

    async def _redis(self) -> Any:
        if self._client is None:
            import redis.asyncio as redis
            self._client = redis.from_url(
                self.redis_url, decode_responses=True, socket_connect_timeout=2, socket_timeout=3,
            )
        return self._client

    @staticmethod
    def _keys(ticket: AdmissionTicket) -> tuple[str, ...]:
        tenant, resource = _safe(ticket.tenant_id), _safe(ticket.resource_class)
        prefix = "tool_runner:jobs:admission"
        return (
            f"{prefix}:weight",
            f"{prefix}:resource:{resource}:weight",
            f"{prefix}:tenant:{tenant}:weight",
            f"{prefix}:waiting-tenants",
            f"{prefix}:tenant:{tenant}:waiting",
            f"{prefix}:leases",
        )

    async def try_acquire(self, ticket: AdmissionTicket) -> bool:
        if not self.enabled:
            return True
        now = int(time.time() * 1000)
        keys = self._keys(ticket)
        result = await (await self._redis()).eval(
            _ACQUIRE_LUA, 6, *keys, now, now + self.ttl_ms, ticket.job_id,
            ticket.resource_weight, self.max_weight, self.max_weight_per_tenant,
            int(self.resource_limits.get(ticket.resource_class, self.max_weight)),
            _safe(ticket.tenant_id),
        )
        return bool(int(result or 0))

    async def release(self, ticket: AdmissionTicket) -> None:
        if not self.enabled:
            return
        keys = self._keys(ticket)
        try:
            await (await self._redis()).eval(
                _RELEASE_LUA, 6, keys[5], keys[0], keys[1], keys[2], keys[3], keys[4],
                ticket.job_id, _safe(ticket.tenant_id),
            )
        except Exception:
            logger.warning("job admission release failed job_id=%s", ticket.job_id, exc_info=True)
        ticket._holds_cluster = False

    async def close(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.aclose()
            self._client = None


class JobAdmission:
    def __init__(
        self, *, max_weight: int = 8, max_weight_per_tenant: int = 2,
        resource_limits: Optional[Dict[str, int]] = None,
        publish: Optional[Publisher] = None,
    ) -> None:
        self.max_weight = int(max_weight)
        self.max_weight_per_tenant = int(max_weight_per_tenant)
        self.resource_limits = dict(resource_limits or {})
        self._publish = publish
        self._running: Dict[str, AdmissionTicket] = {}
        self._waiting: Dict[str, Deque[AdmissionTicket]] = {}
        self._tenant_order: Deque[str] = deque()
        self._used_weight = 0
        self._tenant_weight: Dict[str, int] = {}
        self._resource_weight: Dict[str, int] = {}
        self._draining = False
        self._lock = asyncio.Lock()
        self._cluster = RedisAdmissionLease(
            max_weight=self.max_weight,
            max_weight_per_tenant=self.max_weight_per_tenant,
            resource_limits=self.resource_limits,
        )

    @property
    def draining(self) -> bool:
        return self._draining

    @property
    def max_concurrent(self) -> int:
        """Capacity contract for callers; expressed in weight units."""
        return self.max_weight

    def running_count(self) -> int:
        return len(self._running)

    def queued_count(self) -> int:
        return sum(len(queue) for queue in self._waiting.values())

    def running_job_ids(self) -> list[str]:
        return list(self._running)

    def queued_job_ids(self) -> list[str]:
        return [ticket.job_id for tenant in self._tenant_order for ticket in self._waiting.get(tenant, ())]

    def _fits(self, ticket: AdmissionTicket) -> bool:
        weight = ticket.resource_weight
        if self.max_weight > 0 and self._used_weight + weight > self.max_weight:
            return False
        tenant_used = self._tenant_weight.get(ticket.tenant_id, 0)
        if self.max_weight_per_tenant > 0 and tenant_used + weight > self.max_weight_per_tenant:
            return False
        limit = int(self.resource_limits.get(ticket.resource_class, self.max_weight))
        return limit <= 0 or self._resource_weight.get(ticket.resource_class, 0) + weight <= limit

    def has_capacity(self, resource_class: str = "default", weight: int = 1, tenant_id: str = "") -> bool:
        probe = AdmissionTicket("", "", tenant_id or "__probe__", resource_class, max(1, weight))
        return not self._draining and self._fits(probe)

    def has_any_capacity(self) -> bool:
        """Conservative pre-dequeue gate; exact class/tenant checks occur on submit."""
        return not self._draining and (self.max_weight <= 0 or self._used_weight < self.max_weight)

    def update_resource_limits(
        self, limits: Dict[str, int], *, merge: bool = True
    ) -> Dict[str, int]:
        """Live update from llm-proxy /capacity (bootstrap env remains fallback)."""
        parsed: Dict[str, int] = {}
        for key, value in (limits or {}).items():
            try:
                parsed[str(key)] = max(1, int(value))
            except (TypeError, ValueError):
                continue
        if merge:
            self.resource_limits.update(parsed)
        else:
            self.resource_limits = dict(parsed)
        self._cluster.resource_limits = dict(self.resource_limits)
        return dict(self.resource_limits)

    def _reserve_local(self, ticket: AdmissionTicket) -> None:
        self._used_weight += ticket.resource_weight
        self._tenant_weight[ticket.tenant_id] = self._tenant_weight.get(ticket.tenant_id, 0) + ticket.resource_weight
        self._resource_weight[ticket.resource_class] = self._resource_weight.get(ticket.resource_class, 0) + ticket.resource_weight

    def _release_local(self, ticket: AdmissionTicket) -> None:
        if not ticket.admitted:
            return
        self._used_weight -= ticket.resource_weight
        for mapping, key in ((self._tenant_weight, ticket.tenant_id), (self._resource_weight, ticket.resource_class)):
            value = mapping.get(key, 0) - ticket.resource_weight
            if value > 0:
                mapping[key] = value
            else:
                mapping.pop(key, None)
        self._running.pop(ticket.job_id, None)
        ticket.admitted = False

    def _enqueue(self, ticket: AdmissionTicket) -> None:
        queue = self._waiting.setdefault(ticket.tenant_id, deque())
        if not queue:
            self._tenant_order.append(ticket.tenant_id)
        queue.append(ticket)
        ticket.queued = True
        ticket.position = self.queued_count()
        self._publish_status(ticket)

    def _publish_status(self, ticket: AdmissionTicket) -> None:
        if self._publish is not None:
            try:
                self._publish(
                    ticket.job_id, status="queued", queue_position=ticket.position,
                    tenant_id=ticket.tenant_id, owner_namespace=ticket.owner_namespace,
                    metadata=ticket.metadata,
                )
            except Exception:
                logger.debug("job admission publisher failed", exc_info=True)

    async def submit(
        self, job_id: str, *, owner_namespace: str, tenant_id: str,
        resource_class: str = "default", resource_weight: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AdmissionTicket:
        if self._draining:
            raise AdmissionDraining("tool_runner_draining")
        ticket = AdmissionTicket(
            str(job_id), str(owner_namespace), str(tenant_id), str(resource_class or "default"),
            max(1, int(resource_weight)), dict(metadata or {}),
        )
        async with self._lock:
            if not self._tenant_order and self._fits(ticket):
                try:
                    acquired = await self._cluster.try_acquire(ticket)
                except Exception:
                    acquired = False
                    logger.warning("cluster admission unavailable", exc_info=True)
                if acquired:
                    self._reserve_local(ticket)
                    ticket.admitted = True
                    ticket._holds_cluster = self._cluster.enabled
                    self._running[ticket.job_id] = ticket
                    self._start_renewal(ticket)
                    return ticket
            self._enqueue(ticket)
        return ticket

    async def wait(self, ticket: AdmissionTicket) -> None:
        if ticket.admitted:
            return
        try:
            while not ticket.admitted:
                await self._schedule()
                if ticket.admitted:
                    return
                try:
                    await asyncio.wait_for(ticket._ready.wait(), timeout=.25)
                except asyncio.TimeoutError:
                    continue
                ticket._ready.clear()
        except BaseException:
            await self.cancel_queued(ticket.job_id)
            raise

    async def _schedule(self) -> None:
        async with self._lock:
            rounds = len(self._tenant_order)
            for _ in range(rounds):
                tenant = self._tenant_order.popleft()
                queue = self._waiting.get(tenant)
                if not queue:
                    continue
                ticket = queue[0]
                if self._fits(ticket):
                    try:
                        acquired = await self._cluster.try_acquire(ticket)
                    except Exception:
                        acquired = False
                    if acquired:
                        queue.popleft()
                        self._reserve_local(ticket)
                        ticket.queued, ticket.position, ticket.admitted = False, 0, True
                        ticket._holds_cluster = self._cluster.enabled
                        self._running[ticket.job_id] = ticket
                        self._start_renewal(ticket)
                        ticket._ready.set()
                if queue:
                    self._tenant_order.append(tenant)
                else:
                    self._waiting.pop(tenant, None)

    async def _renew(self, ticket: AdmissionTicket) -> None:
        try:
            while ticket.admitted and ticket._holds_cluster:
                await asyncio.sleep(max(5, self._cluster.ttl_ms / 3000))
                if not await self._cluster.try_acquire(ticket):
                    ticket._holds_cluster = False
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("job admission renewal failed job_id=%s", ticket.job_id, exc_info=True)

    def _start_renewal(self, ticket: AdmissionTicket) -> None:
        if ticket._holds_cluster:
            ticket._renew_task = asyncio.create_task(self._renew(ticket))

    async def release(self, ticket: Optional[AdmissionTicket]) -> None:
        if ticket is None:
            return
        if ticket._renew_task is not None:
            ticket._renew_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ticket._renew_task
        await self._cluster.release(ticket)
        async with self._lock:
            self._release_local(ticket)
            for queue in self._waiting.values():
                for waiting in queue:
                    waiting._ready.set()

    async def cancel_queued(self, job_id: str) -> bool:
        async with self._lock:
            for tenant, queue in list(self._waiting.items()):
                for ticket in list(queue):
                    if ticket.job_id == job_id:
                        queue.remove(ticket)
                        if not queue:
                            self._waiting.pop(tenant, None)
                            with contextlib.suppress(ValueError):
                                self._tenant_order.remove(tenant)
                        await self._cluster.release(ticket)
                        return True
        return False

    def begin_drain(self) -> list[str]:
        self._draining = True
        return self.queued_job_ids()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "max_weight": self.max_weight,
            "max_weight_per_tenant": self.max_weight_per_tenant,
            "used_weight": self._used_weight,
            "running": self.running_count(),
            "queued": self.queued_count(),
            "queued_job_ids": self.queued_job_ids(),
            "resource_weight": dict(self._resource_weight),
            "resource_limits": dict(self.resource_limits),
            "tenant_weight": dict(self._tenant_weight),
            "draining": self.draining,
            "cluster_enabled": self._cluster.enabled,
        }


_ADMISSION: Optional[JobAdmission] = None


def get_admission() -> JobAdmission:
    global _ADMISSION
    if _ADMISSION is None:
        from .config import settings
        try:
            resource_limits = json.loads(settings.TOOL_JOB_RESOURCE_LIMITS or "{}")
            if not isinstance(resource_limits, dict):
                resource_limits = {}
            resource_limits = {
                str(key): int(value) for key, value in resource_limits.items()
            }
        except (TypeError, ValueError):
            logger.warning("Invalid TOOL_JOB_RESOURCE_LIMITS; using global capacity")
            resource_limits = {}
        _ADMISSION = JobAdmission(
            max_weight=settings.TOOL_JOB_MAX_WEIGHT,
            max_weight_per_tenant=settings.TOOL_JOB_MAX_WEIGHT_PER_TENANT,
            resource_limits=resource_limits,
        )
    return _ADMISSION


def configure_admission(
    *, max_weight: int, max_weight_per_tenant: int,
    resource_limits: Optional[Dict[str, int]] = None, publish: Optional[Publisher] = None,
) -> JobAdmission:
    global _ADMISSION
    _ADMISSION = JobAdmission(
        max_weight=max_weight, max_weight_per_tenant=max_weight_per_tenant,
        resource_limits=resource_limits, publish=publish,
    )
    return _ADMISSION


def reset_admission_for_tests() -> None:
    global _ADMISSION
    _ADMISSION = None
