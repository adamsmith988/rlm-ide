"""PostgreSQL source of truth and fencing for generic tool jobs."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_jobs (
    job_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    envelope_version INTEGER NOT NULL DEFAULT 1,
    tool_id TEXT NOT NULL,
    owner_namespace TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    record_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    payload JSONB NOT NULL,
    state TEXT NOT NULL CHECK (
      state IN ('queued','running','retry','completed','failed','cancelled','dead')
    ),
    attempt INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 5,
    lease_owner TEXT,
    lease_until TIMESTAMPTZ,
    fencing_token BIGINT NOT NULL DEFAULT 0,
    checkpoint JSONB,
    terminal_result JSONB,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS tool_jobs_recovery_idx
ON tool_jobs (state, lease_until)
WHERE state IN ('queued','running','retry');
"""

TERMINAL_STATES = frozenset({"completed", "failed", "cancelled", "dead"})


@dataclass(frozen=True)
class JobClaim:
    job_id: str
    fencing_token: int
    attempt: int
    max_attempts: int
    payload: Dict[str, Any]


class JobLedger:
    def __init__(self, dsn: Optional[str] = None) -> None:
        self.dsn = str(
            dsn or os.environ.get("TOOL_JOB_LEDGER_DSN") or os.environ.get("DATABASE_URL") or ""
        ).strip()
        self.pool: Any = None
        self.error: Optional[str] = None

    @property
    def configured(self) -> bool:
        return bool(self.dsn)

    @property
    def ready(self) -> bool:
        return self.pool is not None and self.error is None

    async def start(self) -> bool:
        if self.ready:
            return True
        if not self.configured:
            self.error = "job_ledger_dsn_not_configured"
            return False
        try:
            import asyncpg

            self.pool = await asyncpg.create_pool(
                self.dsn,
                min_size=1,
                max_size=max(2, int(os.environ.get("TOOL_JOB_LEDGER_POOL_SIZE", "6"))),
                command_timeout=10,
            )
            async with self.pool.acquire() as conn:
                await conn.execute(SCHEMA)
            self.error = None
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}:{exc}"
            logger.error("job ledger startup failed: %s", self.error)
            if self.pool is not None:
                await self.pool.close()
                self.pool = None
            return False

    async def close(self) -> None:
        pool, self.pool = self.pool, None
        if pool is not None:
            await pool.close()

    def _require(self) -> Any:
        if self.pool is None:
            raise RuntimeError(self.error or "job_ledger_not_ready")
        return self.pool

    async def enqueue(
        self,
        *,
        job_id: str,
        idempotency_key: str,
        tool_id: str,
        owner_namespace: str,
        tenant_id: str,
        record_id: Optional[str],
        metadata: Dict[str, Any],
        payload: Dict[str, Any],
        max_attempts: int,
        envelope_version: int = 1,
    ) -> Dict[str, Any]:
        row = await self._require().fetchrow(
            """
            INSERT INTO tool_jobs (
              job_id,idempotency_key,envelope_version,tool_id,owner_namespace,
              tenant_id,record_id,metadata,payload,state,max_attempts
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9::jsonb,'queued',$10)
            ON CONFLICT (idempotency_key) DO UPDATE SET updated_at=now()
            RETURNING job_id,state,attempt,max_attempts,fencing_token
            """,
            job_id, idempotency_key, envelope_version, tool_id, owner_namespace,
            tenant_id, record_id, json.dumps(metadata), json.dumps(payload),
            max(1, int(max_attempts)),
        )
        return dict(row)

    async def claim(
        self, job_id: str, *, owner: str, lease_seconds: int
    ) -> Optional[JobClaim]:
        pool = self._require()
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow("SELECT * FROM tool_jobs WHERE job_id=$1 FOR UPDATE", job_id)
                if row is None or str(row["state"]) in TERMINAL_STATES:
                    return None
                if (
                    str(row["state"]) == "running"
                    and row["lease_until"] is not None
                    and await conn.fetchval("SELECT $1::timestamptz > now()", row["lease_until"])
                ):
                    return None
                attempt = int(row["attempt"] or 0) + 1
                maximum = int(row["max_attempts"] or 1)
                if attempt > maximum:
                    await conn.execute(
                        """UPDATE tool_jobs SET state='dead',attempt=$2,
                           error='max_attempts_exhausted',completed_at=now(),updated_at=now()
                           WHERE job_id=$1""",
                        job_id, attempt,
                    )
                    return None
                claimed = await conn.fetchrow(
                    """UPDATE tool_jobs SET state='running',attempt=$2,lease_owner=$3,
                       lease_until=now()+($4::bigint * interval '1 second'),
                       fencing_token=fencing_token+1,updated_at=now()
                       WHERE job_id=$1
                       RETURNING job_id,fencing_token,attempt,max_attempts,payload""",
                    job_id, attempt, owner, max(5, int(lease_seconds)),
                )
                payload = claimed["payload"]
                if isinstance(payload, str):
                    payload = json.loads(payload)
                return JobClaim(
                    str(claimed["job_id"]), int(claimed["fencing_token"]),
                    int(claimed["attempt"]), int(claimed["max_attempts"]), dict(payload or {}),
                )

    async def renew(
        self, job_id: str, *, owner: str, fencing_token: int, lease_seconds: int
    ) -> bool:
        status = await self._require().execute(
            """UPDATE tool_jobs SET
               lease_until=now()+($4::bigint * interval '1 second'),updated_at=now()
               WHERE job_id=$1 AND lease_owner=$2 AND fencing_token=$3 AND state='running'""",
            job_id, owner, int(fencing_token), max(5, int(lease_seconds)),
        )
        return status == "UPDATE 1"

    async def checkpoint(
        self, job_id: str, *, fencing_token: int, checkpoint: Dict[str, Any]
    ) -> bool:
        status = await self._require().execute(
            """UPDATE tool_jobs SET checkpoint=$3::jsonb,updated_at=now()
               WHERE job_id=$1 AND fencing_token=$2 AND state='running'""",
            job_id, int(fencing_token), json.dumps(checkpoint),
        )
        return status == "UPDATE 1"

    async def finish(
        self,
        job_id: str,
        *,
        fencing_token: int,
        state: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> bool:
        if state not in TERMINAL_STATES:
            raise ValueError(f"invalid terminal job state: {state}")
        status = await self._require().execute(
            """UPDATE tool_jobs SET state=$3,terminal_result=$4::jsonb,error=$5,
               lease_owner=NULL,lease_until=NULL,completed_at=now(),updated_at=now()
               WHERE job_id=$1 AND fencing_token=$2 AND state='running'""",
            job_id, int(fencing_token), state,
            json.dumps(result) if result is not None else None, error,
        )
        return status == "UPDATE 1"

    async def cancel_queued(self, job_id: str, *, error: str = "cancelled") -> bool:
        status = await self._require().execute(
            """UPDATE tool_jobs SET state='cancelled',error=$2,
               lease_owner=NULL,lease_until=NULL,completed_at=now(),updated_at=now()
               WHERE job_id=$1 AND state IN ('queued','retry')""",
            job_id,
            error,
        )
        return status == "UPDATE 1"

    async def dead_letter(
        self, job_id: str, *, error: str, result: Optional[Dict[str, Any]] = None
    ) -> bool:
        status = await self._require().execute(
            """UPDATE tool_jobs SET state='dead',terminal_result=$2::jsonb,error=$3,
               fencing_token=fencing_token+1,lease_owner=NULL,lease_until=NULL,
               completed_at=now(),updated_at=now()
               WHERE job_id=$1 AND state NOT IN ('completed','failed','cancelled','dead')""",
            job_id, json.dumps(result) if result is not None else None, error,
        )
        return status == "UPDATE 1"

    async def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        row = await self._require().fetchrow("SELECT * FROM tool_jobs WHERE job_id=$1", job_id)
        return dict(row) if row is not None else None

    async def dispatchable(self, *, limit: int = 500) -> list[Dict[str, Any]]:
        rows = await self._require().fetch(
            """SELECT payload FROM tool_jobs WHERE state IN ('queued','retry')
               ORDER BY created_at LIMIT $1""",
            max(1, min(int(limit), 5000)),
        )
        result: list[Dict[str, Any]] = []
        for row in rows:
            payload = row["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            if isinstance(payload, dict):
                result.append(dict(payload))
        return result


_LEDGER = JobLedger()
_CLAIMS: Dict[str, JobClaim] = {}


def get_job_ledger() -> JobLedger:
    return _LEDGER


class QueueLedgerAdapter:
    """Queue hook; durable terminal commit always occurs before Stream ACK."""

    def __init__(self, ledger: Optional[JobLedger] = None) -> None:
        self.ledger = ledger or get_job_ledger()
        self._claims = _CLAIMS
        self.owner = str(os.environ.get("POD_NAME") or os.environ.get("HOSTNAME") or "worker")

    async def on_enqueued(self, job: Any, message_id: str) -> Optional[bool]:
        row = await self.ledger.enqueue(
            job_id=job.job_id,
            idempotency_key=str(job.metadata.get("idempotency_key") or job.job_id),
            tool_id=job.tool_id,
            owner_namespace=job.owner_namespace,
            tenant_id=job.tenant_id,
            record_id=job.record_id,
            metadata=dict(job.metadata),
            payload=json.loads(job.to_json()),
            max_attempts=max(1, int(os.environ.get("TOOL_JOB_QUEUE_MAX_ATTEMPTS", "5"))),
            envelope_version=1,
        )
        return str(row.get("state") or "") in ("queued", "retry")

    async def on_claimed(self, job: Any, message_id: str, attempts: int) -> Optional[bool]:
        claim = await self.ledger.claim(
            job.job_id, owner=self.owner,
            lease_seconds=max(15, int(os.environ.get("TOOL_JOB_LEDGER_LEASE_SEC", "90"))),
        )
        if claim is None:
            row = await self.ledger.get(job.job_id)
            if row and str(row.get("state") or "") in TERMINAL_STATES:
                return False
            raise RuntimeError(f"job_ledger_claim_busy:{job.job_id}")
        self._claims[job.job_id] = claim
        job.metadata["fencing_token"] = claim.fencing_token
        return True

    async def on_heartbeat(self, job_id: str) -> None:
        claim = self._claims.get(job_id)
        if claim is None:
            raise RuntimeError(f"job_ledger_missing_claim:{job_id}")
        ok = await self.ledger.renew(
            job_id, owner=self.owner, fencing_token=claim.fencing_token,
            lease_seconds=max(15, int(os.environ.get("TOOL_JOB_LEDGER_LEASE_SEC", "90"))),
        )
        if not ok:
            raise RuntimeError(f"job_ledger_stale_fencing_token:{job_id}")

    async def on_cancelled(self, job_id: str, error: str) -> None:
        if not await self.ledger.cancel_queued(job_id, error=error):
            row = await self.ledger.get(job_id)
            if not row or str(row.get("state") or "") != "cancelled":
                raise RuntimeError(f"job_ledger_cancel_conflict:{job_id}")

    async def on_terminal(self, job_id: str, status: str, fields: Dict[str, Any]) -> None:
        claim = self._claims.get(job_id)
        if status == "dead" and claim is None:
            await self.ledger.dead_letter(job_id, error=str(fields.get("error") or "dead_letter"), result=fields)
            return
        if claim is None:
            raise RuntimeError(f"job_ledger_missing_claim:{job_id}")
        mapped = {"completed": "completed", "failed": "failed", "cancelled": "cancelled", "dead": "dead"}[status]
        ok = await self.ledger.finish(
            job_id, fencing_token=claim.fencing_token, state=mapped, result=fields,
            error=str(fields.get("error") or "") or None,
        )
        if not ok:
            raise RuntimeError(f"job_ledger_stale_fencing_token:{job_id}")
        self._claims.pop(job_id, None)
