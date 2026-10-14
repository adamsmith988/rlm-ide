from __future__ import annotations

import asyncio
import os
import sys

SERVICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from src.job_ledger import JobClaim, QueueLedgerAdapter  # noqa: E402
from src.job_queue import Job  # noqa: E402


class FakeLedger:
    ready = True

    def __init__(self, *, finish_ok: bool = True) -> None:
        self.calls = []
        self.finish_ok = finish_ok
        self.state = "queued"

    async def enqueue(self, **kwargs):
        self.calls.append(("enqueue", kwargs))
        return {"job_id": kwargs["job_id"], "state": self.state}

    async def claim(self, job_id, *, owner, lease_seconds):
        self.calls.append(("claim", job_id, owner, lease_seconds))
        self.state = "running"
        return JobClaim(job_id, 7, 1, 5, {"job_id": job_id})

    async def renew(self, job_id, *, owner, fencing_token, lease_seconds):
        self.calls.append(("renew", job_id, fencing_token))
        return fencing_token == 7

    async def finish(self, job_id, *, fencing_token, state, result, error):
        self.calls.append(("finish", job_id, fencing_token, state, result, error))
        return self.finish_ok and fencing_token == 7

    async def cancel_queued(self, job_id, *, error):
        self.calls.append(("cancel", job_id, error))
        self.state = "cancelled"
        return True

    async def get(self, job_id):
        return {"job_id": job_id, "state": self.state}


def _job(job_id: str) -> Job:
    return Job(
        job_id=job_id,
        tool_id="plugin_fake_run",
        owner_namespace="fake",
        tenant_id="tenant-a",
        record_id="record-a",
        metadata={"idempotency_key": f"idem:{job_id}"},
        args={"job_id": job_id},
    )


def test_ledger_adapter_preserves_generic_identity_and_fencing() -> None:
    async def scenario() -> None:
        ledger = FakeLedger()
        adapter = QueueLedgerAdapter(ledger)
        job = _job("job-1")
        assert await adapter.on_enqueued(job, "")
        enqueue = ledger.calls[0][1]
        assert enqueue["owner_namespace"] == "fake"
        assert enqueue["tenant_id"] == "tenant-a"
        assert enqueue["record_id"] == "record-a"
        assert await adapter.on_claimed(job, "1-0", 1)
        assert job.metadata["fencing_token"] == 7
        await adapter.on_heartbeat(job.job_id)
        await adapter.on_terminal(
            job.job_id, "completed", {"result": {"ok": True}}
        )
        assert any(
            call[:4] == ("finish", "job-1", 7, "completed")
            for call in ledger.calls
        )

    asyncio.run(scenario())


def test_stale_fencing_prevents_terminal_commit() -> None:
    async def scenario() -> None:
        ledger = FakeLedger(finish_ok=False)
        adapter = QueueLedgerAdapter(ledger)
        job = _job("job-stale")
        await adapter.on_claimed(job, "1-0", 1)
        try:
            await adapter.on_terminal(job.job_id, "failed", {"error": "boom"})
        except RuntimeError as exc:
            assert "stale_fencing_token" in str(exc)
        else:
            raise AssertionError("stale fencing token was accepted")

    asyncio.run(scenario())


def test_queued_cancel_is_durable() -> None:
    async def scenario() -> None:
        ledger = FakeLedger()
        adapter = QueueLedgerAdapter(ledger)
        await adapter.on_cancelled("job-cancel", "requested")
        assert ("cancel", "job-cancel", "requested") in ledger.calls

    asyncio.run(scenario())
