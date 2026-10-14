"""Generic lifecycle registry for detached tool jobs.

The queue transports jobs and the ledger owns durable state.  This module owns
only resources currently executing in this process: asyncio tasks, subprocess
groups, cancellation, checkpoint hooks and bounded live status/events.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Protocol, Sequence


CheckpointFn = Callable[[], Awaitable[None]]
CancelFn = Callable[[], Awaitable[Any]]
PublishFn = Callable[[Dict[str, Any]], Awaitable[None]]


class JobHandle(Protocol):
    @property
    def done(self) -> bool: ...

    async def wait(self) -> Any: ...

    async def cancel(self, *, grace_seconds: float = 10.0) -> None: ...

    async def checkpoint(self) -> None: ...


@dataclass
class JobSnapshot:
    job_id: str
    tool_id: str
    owner_namespace: str
    tenant_id: Optional[str]
    record_id: Optional[str]
    resource_class: str
    status: str = "queued"
    queue_position: int = 0
    progress: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    def public(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "tool_id": self.tool_id,
            "owner_namespace": self.owner_namespace,
            "tenant_id": self.tenant_id,
            "record_id": self.record_id,
            "resource_class": self.resource_class,
            "status": self.status,
            "queue_position": self.queue_position,
            "progress": dict(self.progress),
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class AsyncTaskHandle:
    def __init__(
        self,
        task: "asyncio.Task[Any]",
        *,
        checkpoint_fn: Optional[CheckpointFn] = None,
        cancel_fn: Optional[CancelFn] = None,
    ) -> None:
        self.task = task
        self._checkpoint_fn = checkpoint_fn
        self._cancel_fn = cancel_fn

    @property
    def done(self) -> bool:
        return self.task.done()

    async def wait(self) -> Any:
        return await asyncio.shield(self.task)

    async def cancel(self, *, grace_seconds: float = 10.0) -> None:
        if self.task.done():
            return
        if self._cancel_fn is not None:
            with contextlib.suppress(Exception):
                await self._cancel_fn()
            try:
                await asyncio.wait_for(
                    asyncio.shield(self.task), timeout=max(0.1, grace_seconds)
                )
                return
            except asyncio.TimeoutError:
                pass
        self.task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(self.task, timeout=max(0.1, grace_seconds))

    async def checkpoint(self) -> None:
        if self._checkpoint_fn is not None:
            await self._checkpoint_fn()


class ProcessGroupHandle:
    """Managed argv execution; shell interpolation is intentionally unsupported."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        *,
        checkpoint_fn: Optional[CheckpointFn] = None,
        output_limit_bytes: int = 2 * 1024 * 1024,
        timeout_seconds: float = 0,
    ) -> None:
        self.process = process
        self._checkpoint_fn = checkpoint_fn
        self.output_limit_bytes = max(4096, int(output_limit_bytes))
        self.timeout_seconds = max(0.0, float(timeout_seconds))
        self._output_truncated = False
        self.stdout = bytearray()
        self.stderr = bytearray()
        self._captured_bytes = 0
        self._drainers = [
            asyncio.create_task(self._drain(process.stdout, self.stdout)),
            asyncio.create_task(self._drain(process.stderr, self.stderr)),
        ]

    @classmethod
    async def spawn(
        cls,
        argv: Sequence[str],
        *,
        cwd: str,
        env: Mapping[str, str],
        workspace_root: Optional[str] = None,
        env_allowlist: Sequence[str] = (
            "PATH",
            "PYTHONPATH",
            "LANG",
            "LC_ALL",
            "HOME",
            "TMPDIR",
            "TZ",
        ),
        checkpoint_fn: Optional[CheckpointFn] = None,
        output_limit_bytes: int = 2 * 1024 * 1024,
        timeout_seconds: float = 0,
    ) -> "ProcessGroupHandle":
        if not argv or any(not isinstance(arg, str) or "\x00" in arg for arg in argv):
            raise ValueError("invalid subprocess argv")
        real_cwd = os.path.realpath(cwd)
        if not os.path.isdir(real_cwd):
            raise ValueError("subprocess cwd does not exist")
        root = os.path.realpath(workspace_root or real_cwd)
        if os.path.commonpath((root, real_cwd)) != root:
            raise ValueError("subprocess cwd escapes workspace")
        allowed = frozenset(str(name) for name in env_allowlist)
        safe_env = {
            str(key): str(value)
            for key, value in env.items()
            if str(key) in allowed
        }
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=real_cwd,
            env=safe_env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        return cls(
            process,
            checkpoint_fn=checkpoint_fn,
            output_limit_bytes=output_limit_bytes,
            timeout_seconds=timeout_seconds,
        )

    async def _drain(
        self, stream: Optional[asyncio.StreamReader], target: bytearray
    ) -> None:
        if stream is None:
            return
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                return
            remaining = self.output_limit_bytes - self._captured_bytes
            if remaining > 0:
                captured = chunk[:remaining]
                target.extend(captured)
                self._captured_bytes += len(captured)
            if len(chunk) > remaining:
                self._output_truncated = True

    @property
    def done(self) -> bool:
        return self.process.returncode is not None

    async def wait(self) -> Dict[str, Any]:
        try:
            if self.timeout_seconds > 0:
                code = await asyncio.wait_for(
                    self.process.wait(), timeout=self.timeout_seconds
                )
            else:
                code = await self.process.wait()
        except asyncio.TimeoutError:
            await self.cancel(grace_seconds=2.0)
            raise TimeoutError("subprocess execution timed out")
        await asyncio.gather(*self._drainers, return_exceptions=True)
        return {
            "returncode": code,
            "stdout": bytes(self.stdout).decode("utf-8", errors="replace"),
            "stderr": bytes(self.stderr).decode("utf-8", errors="replace"),
            "output_truncated": self._output_truncated,
        }

    async def checkpoint(self) -> None:
        if self._checkpoint_fn is not None:
            await self._checkpoint_fn()

    async def cancel(self, *, grace_seconds: float = 10.0) -> None:
        if self.done:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            self.process.terminate()
        try:
            await asyncio.wait_for(
                self.process.wait(), timeout=max(0.1, float(grace_seconds))
            )
        except asyncio.TimeoutError:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                self.process.kill()
            await self.process.wait()
        await asyncio.gather(*self._drainers, return_exceptions=True)


class JobRuntime:
    TERMINAL = frozenset({"completed", "failed", "cancelled", "dead"})

    def __init__(self, *, history_limit: int = 5000) -> None:
        self.history_limit = max(100, int(history_limit))
        self._snapshots: "OrderedDict[str, JobSnapshot]" = OrderedDict()
        self._handles: Dict[str, JobHandle] = {}
        self._subscribers: Dict[str, List["asyncio.Queue[Dict[str, Any]]"]] = {}
        self.publisher: Optional[PublishFn] = None

    def create(
        self,
        *,
        job_id: str,
        tool_id: str,
        owner_namespace: str,
        tenant_id: Optional[str],
        record_id: Optional[str],
        resource_class: str,
        status: str = "queued",
        queue_position: int = 0,
    ) -> JobSnapshot:
        snapshot = JobSnapshot(
            job_id=job_id,
            tool_id=tool_id,
            owner_namespace=owner_namespace,
            tenant_id=tenant_id,
            record_id=record_id,
            resource_class=resource_class,
            status=status,
            queue_position=queue_position,
        )
        self._snapshots[job_id] = snapshot
        self._trim()
        self._publish(snapshot)
        return snapshot

    def attach(
        self, job_id: str, handle: JobHandle, *, mark_running: bool = True
    ) -> None:
        if job_id not in self._snapshots:
            raise KeyError(job_id)
        self._handles[job_id] = handle
        if mark_running:
            self.update(
                job_id, status="running", queue_position=0, started_at=time.time()
            )

    def update(self, job_id: str, **fields: Any) -> Optional[JobSnapshot]:
        snapshot = self._snapshots.get(job_id)
        if snapshot is None:
            return None
        for key, value in fields.items():
            if hasattr(snapshot, key):
                setattr(snapshot, key, value)
        snapshot.updated_at = time.time()
        if snapshot.status in self.TERMINAL and snapshot.finished_at is None:
            snapshot.finished_at = snapshot.updated_at
        self._snapshots.move_to_end(job_id)
        self._publish(snapshot)
        return snapshot

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        snapshot = self._snapshots.get(str(job_id or ""))
        return snapshot.public() if snapshot is not None else None

    def progress(self, job_id: str, **fields: Any) -> Optional[JobSnapshot]:
        snapshot = self._snapshots.get(str(job_id or ""))
        if snapshot is None or snapshot.status in self.TERMINAL:
            return snapshot
        merged = dict(snapshot.progress)
        merged.update(fields)
        return self.update(job_id, progress=merged)

    def alive(self, job_id: str) -> bool:
        handle = self._handles.get(str(job_id or ""))
        return bool(handle is not None and not handle.done)

    def active_jobs(self) -> Mapping[str, "asyncio.Task[Any]"]:
        """Expose active asyncio tasks for generic drain orchestration."""
        return {
            job_id: handle.task
            for job_id, handle in self._handles.items()
            if isinstance(handle, AsyncTaskHandle) and not handle.done
        }

    async def cancel_job(self, job_id: str) -> bool:
        return await self.cancel(job_id)

    def terminal_outcome(self, job_id: str) -> tuple[str, Dict[str, Any]]:
        snapshot = self._snapshots.get(str(job_id or ""))
        if snapshot is None or snapshot.status not in self.TERMINAL:
            return "failed", {"error": "runtime_terminal_result_missing"}
        return snapshot.status, {
            "result": snapshot.result,
            "error": snapshot.error,
            "progress": dict(snapshot.progress),
        }

    async def cancel(self, job_id: str, *, grace_seconds: float = 10.0) -> bool:
        handle = self._handles.get(str(job_id or ""))
        if handle is None:
            return False
        await handle.cancel(grace_seconds=grace_seconds)
        self._handles.pop(job_id, None)
        self.update(job_id, status="cancelled", error="cancelled")
        return True

    async def checkpoint(self, job_id: str) -> bool:
        handle = self._handles.get(str(job_id or ""))
        if handle is None:
            return False
        await handle.checkpoint()
        return True

    async def drain(
        self,
        *,
        wait_seconds: float,
        checkpoint_seconds: float,
        cancel_seconds: float,
    ) -> Dict[str, int]:
        handles = list(self._handles.items())
        if handles and wait_seconds > 0:
            _, pending = await asyncio.wait(
                [asyncio.create_task(handle.wait()) for _, handle in handles],
                timeout=wait_seconds,
            )
            for task in pending:
                task.cancel()
        checkpointed = 0
        for job_id, handle in list(self._handles.items()):
            if handle.done:
                continue
            try:
                await asyncio.wait_for(
                    handle.checkpoint(), timeout=max(0.1, checkpoint_seconds)
                )
                checkpointed += 1
            except Exception:
                pass
        cancelled = 0
        for job_id, handle in list(self._handles.items()):
            if handle.done:
                self._handles.pop(job_id, None)
                continue
            await handle.cancel(grace_seconds=cancel_seconds)
            self._handles.pop(job_id, None)
            self.update(job_id, status="cancelled", error="process_shutdown")
            cancelled += 1
        return {
            "running_at_start": len(handles),
            "checkpointed": checkpointed,
            "cancelled": cancelled,
        }

    def subscribe(self, job_id: str) -> "asyncio.Queue[Dict[str, Any]]":
        queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(maxsize=64)
        self._subscribers.setdefault(job_id, []).append(queue)
        current = self.get(job_id)
        if current is not None:
            queue.put_nowait(current)
        return queue

    def unsubscribe(
        self, job_id: str, queue: "asyncio.Queue[Dict[str, Any]]"
    ) -> None:
        rows = self._subscribers.get(job_id)
        if not rows:
            return
        with contextlib.suppress(ValueError):
            rows.remove(queue)
        if not rows:
            self._subscribers.pop(job_id, None)

    def _publish(self, snapshot: JobSnapshot) -> None:
        payload = snapshot.public()
        for queue in list(self._subscribers.get(snapshot.job_id, [])):
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(payload)
        if self.publisher is not None:
            with contextlib.suppress(RuntimeError):
                asyncio.get_running_loop().create_task(self.publisher(payload))

    def _trim(self) -> None:
        while len(self._snapshots) > self.history_limit:
            job_id, snapshot = next(iter(self._snapshots.items()))
            if snapshot.status not in self.TERMINAL:
                break
            self._snapshots.pop(job_id, None)
            self._handles.pop(job_id, None)


_RUNTIME = JobRuntime()


def get_job_runtime() -> JobRuntime:
    return _RUNTIME
