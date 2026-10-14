"""Uniform status, events and cancellation API for every detached workload."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..authz import authorize_tool_execute
from ..job_queue import get_queue
from ..job_runtime import get_job_runtime

router = APIRouter(tags=["Jobs"])


async def _ledger_snapshot(request: Request, job_id: str) -> Optional[Dict[str, Any]]:
    ledger = getattr(request.app.state, "job_ledger", None)
    if ledger is None:
        return None
    for method_name in ("get_job", "get", "status"):
        method = getattr(ledger, method_name, None)
        if method is None:
            continue
        value = method(job_id)
        if hasattr(value, "__await__"):
            value = await value
        if value is None:
            continue
        if isinstance(value, dict):
            value = dict(value)
            if "status" not in value and value.get("state"):
                value["status"] = value["state"]
            if "result" not in value and value.get("terminal_result") is not None:
                value["result"] = value["terminal_result"]
            return value
        public = getattr(value, "public", None)
        if callable(public):
            value = public()
        elif hasattr(value, "__dict__"):
            value = dict(value.__dict__)
        if isinstance(value, dict):
            return value
    return None


async def _snapshot(request: Request, job_id: str) -> Optional[Dict[str, Any]]:
    value = get_job_runtime().get(job_id)
    if value is None:
        bus = getattr(request.app.state, "job_bus", None)
        if bus is not None:
            value = await bus.get(job_id)
    if value is None:
        queue = get_queue()
        if queue is not None:
            value = await queue.job_get(job_id)
    if value is None:
        value = await _ledger_snapshot(request, job_id)
    if value is None:
        return None
    if not value.get("status") and value.get("state"):
        value = dict(value)
        value["status"] = value.get("state")
    tool_id = str(value.get("tool_id") or "")
    if tool_id:
        from ..workloads import workload_registry

        progress = await workload_registry.status(tool_id, job_id)
        if progress:
            value = dict(value)
            value["progress"] = progress
    return value


async def _authorize(request: Request, snapshot: Dict[str, Any]) -> Any:
    tool_id = str(snapshot.get("tool_id") or "").strip()
    if not tool_id:
        raise HTTPException(status_code=404, detail="job_not_found")
    authz = await authorize_tool_execute(request, tool_id)
    tenant_id = str(snapshot.get("tenant_id") or "").strip()
    if tenant_id and authz.user_id and tenant_id != str(authz.user_id):
        raise HTTPException(status_code=403, detail="job_access_denied")
    return authz


@router.get("/jobs/{job_id}")
async def job_status(job_id: str, request: Request) -> Dict[str, Any]:
    snapshot = await _snapshot(request, job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="job_not_found")
    await _authorize(request, snapshot)
    return snapshot


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request) -> Dict[str, Any]:
    snapshot = await _snapshot(request, job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="job_not_found")
    await _authorize(request, snapshot)
    status = str(snapshot.get("status") or "")
    if status in get_job_runtime().TERMINAL:
        return {"job_id": job_id, "status": status, "accepted": False}
    cancelled_local = await get_job_runtime().cancel(job_id)
    accepted = cancelled_local
    bus = getattr(request.app.state, "job_bus", None)
    if bus is not None:
        accepted = bool(await bus.request_cancel(job_id)) or accepted
    queue = get_queue()
    if queue is not None and status == "queued":
        accepted = bool(
            await queue.cancel_queued(job_id, error="cancelled_by_user")
        ) or accepted
    return {"job_id": job_id, "status": "cancelling", "accepted": accepted}


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request) -> StreamingResponse:
    snapshot = await _snapshot(request, job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="job_not_found")
    authz = await _authorize(request, snapshot)
    runtime = get_job_runtime()
    bus = getattr(request.app.state, "job_bus", None)
    tool_id = str(snapshot.get("tool_id") or "").strip()

    async def frames() -> AsyncIterator[str]:
        if tool_id:
            from ..workloads import workload_registry

            workload_events = workload_registry.event_stream(
                tool_id,
                job_id,
                {"user_id": str(authz.user_id or "")},
            )
            if workload_events is not None:
                async for value in workload_events:
                    if isinstance(value, bytes):
                        yield value.decode("utf-8", errors="replace")
                    elif isinstance(value, str):
                        yield value
                    elif isinstance(value, dict):
                        yield _frame(str(value.get("type") or "progress"), value)
                return
        if runtime.get(job_id) is not None:
            queue = runtime.subscribe(job_id)
            try:
                while True:
                    try:
                        value = await asyncio.wait_for(queue.get(), timeout=15)
                        event = (
                            "terminal"
                            if value.get("status") in runtime.TERMINAL
                            else "progress"
                        )
                        yield _frame(event, value)
                        if event == "terminal":
                            return
                    except asyncio.TimeoutError:
                        refreshed = await _snapshot(request, job_id)
                        yield _frame(
                            "progress" if refreshed else "heartbeat",
                            refreshed or {"job_id": job_id},
                        )
            finally:
                runtime.unsubscribe(job_id, queue)
        elif bus is not None:
            async for value in bus.events(job_id):
                if value.get("type") == "heartbeat":
                    value = await _snapshot(request, job_id) or value
                event = str(value.get("type") or "progress")
                if value.get("status") in runtime.TERMINAL:
                    event = "terminal"
                yield _frame(event, value)
        else:
            yield _frame("progress", snapshot)

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _frame(event: str, payload: Dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {body}\n\n"
