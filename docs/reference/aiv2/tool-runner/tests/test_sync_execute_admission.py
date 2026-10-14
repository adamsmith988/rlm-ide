"""Admission backpressure for sync execute path (tool-agnostic resource classes).

Mirrors ``_execute_with_admission`` without importing FastAPI route modules.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List

SERVICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from src.job_admission import JobAdmission  # noqa: E402
from src.tool_resource import resolve_tool_resource_from_adapter  # noqa: E402
from tests.test_job_infrastructure import LocalLease  # noqa: E402


def run(coro):
    return asyncio.run(coro)


async def _execute_with_admission(
    *,
    admission: JobAdmission,
    tool_id: str,
    adapter: Any,
    tenant_id: str,
    run_body: Callable[[], Awaitable[Any]],
) -> Any:
    resource_class, resource_weight = resolve_tool_resource_from_adapter(adapter)
    job_id = f"sync-{tool_id}-{uuid.uuid4().hex[:8]}"
    ticket = await admission.submit(
        job_id,
        owner_namespace="tool-runner",
        tenant_id=str(tenant_id or "anonymous"),
        resource_class=resource_class,
        resource_weight=resource_weight,
        metadata={"tool_id": tool_id, "mode": "sync_execute"},
    )
    try:
        await admission.wait(ticket)
        return await run_body()
    finally:
        await admission.release(ticket)


def test_llm_class_queues_when_capacity_full() -> None:
    admission = JobAdmission(
        max_weight=2,
        max_weight_per_tenant=2,
        resource_limits={"llm": 1, "cpu": 2},
    )
    admission._cluster = LocalLease()

    adapter = SimpleNamespace(
        backend="vllm",
        vram_mb=0,
        capabilities=["vision"],
        meta={},
    )
    started: List[str] = []
    gate = asyncio.Event()

    async def hold_slot(label: str) -> Dict[str, Any]:
        started.append(f"start:{label}")
        await gate.wait()
        started.append(f"end:{label}")
        return {"ok": True, "label": label}

    async def scenario() -> None:
        first = asyncio.create_task(
            _execute_with_admission(
                admission=admission,
                tool_id="any_llm_tool",
                adapter=adapter,
                tenant_id="t1",
                run_body=lambda: hold_slot("a"),
            )
        )
        await asyncio.sleep(0.05)
        assert admission.running_count() == 1

        second = asyncio.create_task(
            _execute_with_admission(
                admission=admission,
                tool_id="another_llm_tool",
                adapter=adapter,
                tenant_id="t1",
                run_body=lambda: hold_slot("b"),
            )
        )
        await asyncio.sleep(0.1)
        assert "start:a" in started
        assert "start:b" not in started
        assert admission.queued_count() >= 1

        gate.set()
        out_a, out_b = await asyncio.wait_for(asyncio.gather(first, second), timeout=2)
        assert out_a["label"] == "a"
        assert out_b["label"] == "b"
        assert started == ["start:a", "end:a", "start:b", "end:b"]
        assert admission.running_count() == 0

    run(scenario())


def test_cpu_class_admits_while_llm_slot_held() -> None:
    admission = JobAdmission(
        max_weight=4,
        max_weight_per_tenant=4,
        resource_limits={"llm": 1, "cpu": 2},
    )
    admission._cluster = LocalLease()

    llm_adapter = SimpleNamespace(backend="vllm", vram_mb=0, capabilities=["chat"], meta={})
    cpu_adapter = SimpleNamespace(backend="local", vram_mb=0, capabilities=["file"], meta={})
    gate = asyncio.Event()

    async def hold_llm() -> Dict[str, Any]:
        await gate.wait()
        return {"ok": True}

    async def cpu_body() -> Dict[str, Any]:
        return {"ok": True, "kind": "cpu"}

    async def scenario() -> None:
        hold = asyncio.create_task(
            _execute_with_admission(
                admission=admission,
                tool_id="llmish",
                adapter=llm_adapter,
                tenant_id="t1",
                run_body=hold_llm,
            )
        )
        await asyncio.sleep(0.05)

        cpu_out = await asyncio.wait_for(
            _execute_with_admission(
                admission=admission,
                tool_id="cpuish",
                adapter=cpu_adapter,
                tenant_id="t1",
                run_body=cpu_body,
            ),
            timeout=1,
        )
        assert cpu_out["kind"] == "cpu"
        gate.set()
        await hold

    run(scenario())
