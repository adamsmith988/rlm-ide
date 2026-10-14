"""Generic graceful drain driven entirely by runtime protocol callbacks."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Dict, Mapping, Optional, Protocol

from .job_admission import JobAdmission, get_admission

logger = logging.getLogger(__name__)


class JobRegistry(Protocol):
    """Generic workload registry lifecycle callback."""

    async def shutdown(self) -> None: ...


class JobRuntime(Protocol):
    """Runtime integration required by the generic drain.

    Implementations own task registration, cancellation semantics and any
    workload-specific session/resource shutdown.
    """

    def active_jobs(self) -> Mapping[str, "asyncio.Task[Any]"]: ...
    async def cancel_job(self, job_id: str) -> bool: ...
    async def checkpoint(self, job_id: str) -> bool: ...


_RUNTIME: Optional[JobRuntime] = None
_REGISTRY: Optional[JobRegistry] = None


def configure_job_runtime(runtime: JobRuntime, registry: JobRegistry) -> None:
    global _RUNTIME, _REGISTRY
    _RUNTIME, _REGISTRY = runtime, registry


def clear_job_runtime() -> None:
    global _RUNTIME, _REGISTRY
    _RUNTIME = _REGISTRY = None


def _live(runtime: JobRuntime) -> Dict[str, "asyncio.Task[Any]"]:
    return {
        str(job_id): task
        for job_id, task in dict(runtime.active_jobs()).items()
        if task is not None and not task.done()
    }


async def _wait(tasks: list["asyncio.Task[Any]"], timeout: float) -> int:
    if not tasks:
        return 0
    if timeout <= 0:
        return sum(not task.done() for task in tasks)
    _done, pending = await asyncio.wait(tasks, timeout=timeout)
    return len(pending)


async def drain_jobs(
    *,
    wait_timeout: float,
    checkpoint_timeout: float = 60,
    cancel_timeout: float = 20,
    runtime: Optional[JobRuntime] = None,
    registry: Optional[JobRegistry] = None,
    admission: Optional[JobAdmission] = None,
) -> Dict[str, Any]:
    """Gate, wait, checkpoint and cancel generic jobs without owner imports."""
    runtime = runtime or _RUNTIME
    registry = registry or _REGISTRY
    admission = admission or get_admission()
    if runtime is None or registry is None:
        raise RuntimeError("job runtime and registry callbacks must be configured")

    started = time.monotonic()
    report: Dict[str, Any] = {
        "queued_cancelled": 0,
        "running_at_start": 0,
        "finished_in_wait": 0,
        "force_cancelled": 0,
        "still_pending": 0,
        "checkpoint": {},
    }

    queued = admission.begin_drain()
    for job_id in queued:
        try:
            if await runtime.cancel_job(job_id):
                report["queued_cancelled"] += 1
            await admission.cancel_queued(job_id)
        except Exception:
            logger.warning("drain failed to cancel queued job_id=%s", job_id, exc_info=True)

    running = _live(runtime)
    report["running_at_start"] = len(running)
    pending = await _wait(list(running.values()), wait_timeout)
    report["finished_in_wait"] = len(running) - pending

    leftovers = _live(runtime)
    checkpointed = 0
    checkpoint_failures = 0
    deadline = time.monotonic() + max(0, checkpoint_timeout)
    for job_id in leftovers:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            if await asyncio.wait_for(runtime.checkpoint(job_id), timeout=remaining):
                checkpointed += 1
        except Exception:
            checkpoint_failures += 1
            logger.warning("job drain checkpoint failed job_id=%s", job_id, exc_info=True)
    registry_shutdown = False
    try:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            await asyncio.wait_for(registry.shutdown(), timeout=remaining)
            registry_shutdown = True
    except Exception:
        checkpoint_failures += 1
        logger.warning("job registry shutdown failed", exc_info=True)
    report["checkpoint"] = {
        "jobs": checkpointed,
        "failed": checkpoint_failures,
        "timed_out": time.monotonic() >= deadline,
        "registry_shutdown": registry_shutdown,
    }

    for job_id, task in leftovers.items():
        try:
            cancelled = await runtime.cancel_job(job_id)
        except Exception:
            cancelled = False
            logger.warning("job drain cancellation failed job_id=%s", job_id, exc_info=True)
        if not cancelled:
            task.cancel()
        report["force_cancelled"] += 1
    report["still_pending"] = await _wait(list(leftovers.values()), cancel_timeout)
    report["elapsed_sec"] = round(time.monotonic() - started, 1)
    return report
