from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict

import pytest

SERVICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from src.job_admission import JobAdmission  # noqa: E402
from src.job_drain import drain_jobs  # noqa: E402
from src.job_queue import Job, JobQueue, TerminalOutcome, owner_heartbeat  # noqa: E402
from tests.fake_redis import FakeRedis  # noqa: E402


def run(coro):
    return asyncio.run(coro)


def fake_job(job_id: str, tenant: str = "tenant-a", resource: str = "cpu", weight: int = 1) -> Job:
    return Job(
        job_id=job_id,
        tool_id="fake_tool",
        owner_namespace="tests",
        tenant_id=tenant,
        record_id=f"record-{job_id}",
        metadata={"fixture": True},
        resource_class=resource,
        resource_weight=weight,
        args={"value": job_id},
    )


def test_job_envelope_is_strictly_generic() -> None:
    job = fake_job("job-1", weight=3)
    decoded = Job.from_json(job.to_json())
    assert decoded.job_id == "job-1"
    assert decoded.owner_namespace == "tests"
    assert decoded.tenant_id == "tenant-a"
    assert decoded.record_id == "record-job-1"
    assert decoded.metadata == {"fixture": True}
    assert decoded.resource_weight == 3
    with pytest.raises(ValueError, match="job_id"):
        Job.from_json('{"run_id":"legacy"}')


def test_stream_terminal_commit_precedes_ack() -> None:
    events: list[str] = []

    class Ledger:
        async def on_enqueued(self, job, message_id):
            return True

        async def on_claimed(self, job, message_id, attempts):
            return True

        async def on_terminal(self, job_id, status, fields):
            events.append("ledger")
            raise ConnectionError("database unavailable")

    async def scenario() -> None:
        redis = FakeRedis()
        queue = JobQueue(
            redis, pod="worker", ledger=Ledger(), supported_resource_classes=("cpu",)
        )
        await queue.enqueue(fake_job("commit-order"))
        await queue.dequeue(.1)
        with pytest.raises(ConnectionError):
            await queue.ack_terminal("commit-order", "completed")
        stream = queue.resource_queue_key("cpu")
        assert redis.pending[(stream, queue.group)]
        assert events == ["ledger"]

    run(scenario())


def test_reclaim_and_dead_letter_use_generic_keys() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        first = JobQueue(
            redis,
            pod="one",
            max_attempts=1,
            claim_idle_ms=1,
            reclaim_interval_sec=.01,
            supported_resource_classes=("cpu",),
        )
        second = JobQueue(
            redis,
            pod="two",
            max_attempts=1,
            claim_idle_ms=1,
            reclaim_interval_sec=.01,
            supported_resource_classes=("cpu",),
        )
        await first.enqueue(fake_job("poison"))
        assert (await first.dequeue(.1)).job_id == "poison"
        await asyncio.sleep(.02)
        assert await second.dequeue(.1) is None
        assert (await second.job_get("poison"))["status"] == "dead"
        assert second.dlq_key == "tool_runner:jobs:dlq"
        assert redis.streams[second.dlq_key][0][1]["job_id"] == "poison"

    run(scenario())


def test_resource_class_streams_prevent_incompatible_claims() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        producer = JobQueue(redis, pod="api")
        cpu = JobQueue(
            redis, pod="cpu-worker", supported_resource_classes=("cpu",)
        )
        gpu = JobQueue(
            redis,
            pod="gpu-worker",
            supported_resource_classes=("gpu",),
            worker_capabilities=("gpu",),
        )
        await producer.enqueue(fake_job("gpu-job", resource="gpu"))
        assert await cpu.dequeue(0.01) is None
        claimed = await gpu.dequeue(0.1)
        assert claimed is not None and claimed.job_id == "gpu-job"
        assert claimed.resource_class == "gpu"

    run(scenario())


def test_queued_cancel_commits_ledger_before_stream_removal() -> None:
    events: list[str] = []

    class Ledger:
        async def on_enqueued(self, job, message_id):
            return True

        async def on_cancelled(self, job_id, error):
            events.append(f"ledger:{job_id}:{error}")

    async def scenario() -> None:
        redis = FakeRedis()
        queue = JobQueue(
            redis,
            pod="api",
            ledger=Ledger(),
            supported_resource_classes=("cpu",),
        )
        await queue.enqueue(fake_job("cancel-me"))
        stream = queue.resource_queue_key("cpu")
        assert redis.streams[stream]
        assert await queue.cancel_queued("cancel-me", error="requested")
        assert events == ["ledger:cancel-me:requested"]
        assert redis.streams[stream] == []
        assert (await queue.job_get("cancel-me"))["status"] == "cancelled"

    run(scenario())


def test_owner_heartbeat_uses_generic_terminal_callback() -> None:
    async def scenario() -> None:
        redis = FakeRedis()
        queue = JobQueue(redis, pod="worker", owner_linger_sec=60)
        job = fake_job("heartbeat")
        await queue.job_status_set(job.job_id, "running")
        alive = {"value": True}

        def terminal(job_id: str) -> TerminalOutcome:
            assert job_id == job.job_id
            return TerminalOutcome("completed", {"result": "ok"})

        task = asyncio.create_task(owner_heartbeat(
            queue, job, alive=lambda: alive["value"], terminal_source=terminal,
            interval_sec=1, record_source=lambda _job_id: [job.record_id],
        ))
        await asyncio.sleep(.02)
        alive["value"] = False
        await asyncio.wait_for(task, 2)
        assert (await queue.job_get(job.job_id))["status"] == "completed"
        assert await queue.owner_get_for_record("tests", job.record_id) == "worker"

    run(scenario())


class LocalLease:
    enabled = False
    ttl_ms = 60_000

    async def try_acquire(self, ticket):
        return True

    async def release(self, ticket):
        ticket._holds_cluster = False


def test_admission_is_weighted_resource_aware_and_tenant_fair() -> None:
    async def scenario() -> None:
        admission = JobAdmission(
            max_weight=4, max_weight_per_tenant=4, resource_limits={"gpu": 3, "cpu": 4}
        )
        admission._cluster = LocalLease()
        running = await admission.submit(
            "a1", owner_namespace="tests", tenant_id="a",
            resource_class="gpu", resource_weight=3,
        )
        a2 = await admission.submit(
            "a2", owner_namespace="tests", tenant_id="a",
            resource_class="cpu", resource_weight=2,
        )
        b1 = await admission.submit(
            "b1", owner_namespace="tests", tenant_id="b",
            resource_class="cpu", resource_weight=1,
        )
        assert running.admitted and a2.queued and b1.queued
        await admission.release(running)
        b_wait = asyncio.create_task(admission.wait(b1))
        a_wait = asyncio.create_task(admission.wait(a2))
        await asyncio.wait_for(b_wait, 1)
        assert b1.admitted
        await asyncio.wait_for(a_wait, 1)
        assert a2.admitted
        assert admission.snapshot()["used_weight"] == 3
        await admission.release(b1)
        await admission.release(a2)

    run(scenario())


def test_drain_uses_runtime_and_registry_protocols() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.tasks: Dict[str, asyncio.Task[Any]] = {}
            self.cancelled: list[str] = []

        def active_jobs(self):
            return self.tasks

        async def cancel_job(self, job_id):
            self.cancelled.append(job_id)
            task = self.tasks.get(job_id)
            if task:
                task.cancel()
            return True

        async def checkpoint(self, job_id):
            return job_id in self.tasks

    class Registry:
        def __init__(self) -> None:
            self.shutdown_called = False

        async def shutdown(self):
            self.shutdown_called = True

    async def scenario() -> None:
        runtime, registry = Runtime(), Registry()
        admission = JobAdmission(max_weight=1, max_weight_per_tenant=1)
        admission._cluster = LocalLease()
        running = await admission.submit("j1", owner_namespace="tests", tenant_id="a")
        queued = await admission.submit("j2", owner_namespace="tests", tenant_id="b")
        runtime.tasks["j1"] = asyncio.create_task(asyncio.sleep(60))
        report = await drain_jobs(
            wait_timeout=0, checkpoint_timeout=1, cancel_timeout=1,
            runtime=runtime, registry=registry, admission=admission,
        )
        assert queued.queued and running.admitted
        assert report["queued_cancelled"] == 1
        assert report["checkpoint"]["jobs"] == 1
        assert registry.shutdown_called
        assert report["force_cancelled"] == 1
        assert runtime.cancelled == ["j2", "j1"]

    run(scenario())
