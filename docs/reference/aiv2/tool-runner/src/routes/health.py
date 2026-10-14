"""
Health Check Endpoints
بررسی سلامت سرویس
"""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from typing import Dict, Any
import logging

from ..authz import require_internal_key
from ..config import settings
from ..utils import get_registry

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/healthz", include_in_schema=False)
async def healthz() -> Dict[str, str]:
    """Process liveness only; deliberately performs no dependency I/O.

    Deliberately touches nothing (no registry, no locks, no I/O) so a probe
    cannot be starved by tool execution or registry reload under load.
    """
    return {"status": "ok"}


@router.get("/readyz", include_in_schema=False)
async def readyz() -> JSONResponse:
    """Role-aware readiness for routing traffic and scheduling workers."""
    from ..job_queue import queue_readiness
    from ..job_ledger import get_job_ledger

    registry = get_registry()
    registry_ready = bool(registry.is_ready)
    queue = queue_readiness()
    ledger = get_job_ledger()
    ledger_ready = ledger.ready if settings.TOOL_JOB_LEDGER_REQUIRED else True
    ready = registry_ready and bool(queue["ready"]) and ledger_ready
    body = {
        "status": "ready" if ready else "not_ready",
        "registry": registry_ready,
        "queue": queue,
        "ledger": {
            "required": settings.TOOL_JOB_LEDGER_REQUIRED,
            "configured": ledger.configured,
            "ready": ledger.ready,
            "error": ledger.error,
        },
    }
    return JSONResponse(status_code=200 if ready else 503, content=body)


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check.
    Returns service status and tool count.
    """
    registry = get_registry()
    tools_count = len(registry.list_tools()) if registry.is_ready else 0
    
    return {
        "status": "healthy" if registry.is_ready else "initializing",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "tools_count": tools_count,
        "ready": registry.is_ready,
        **_job_role_info(),
        **_job_bus_info(),
    }


def _job_bus_info() -> Dict[str, Any]:
    """Shared status/event/cancel bus summary without dependency I/O."""
    try:
        from ..job_bus import redis_url_from_env
        return {"job_bus": {"configured": bool(redis_url_from_env())}}
    except Exception:
        return {}


def _job_role_info() -> Dict[str, Any]:
    """Role and generic queue wiring summary."""
    try:
        from ..job_queue import get_queue, get_role, get_worker

        queue = get_queue()
        worker = get_worker()
        return {
            "role": get_role().value,
            "job_queue": {
                "enabled": queue is not None,
                "pod": queue.pod if queue is not None else None,
                "queue_key": queue.queue_key if queue is not None else None,
                "consumer": bool(worker is not None and worker.running),
                "consumed": worker.consumed if worker is not None else 0,
            },
        }
    except Exception:  # pragma: no cover - health must never fail
        return {"role": "all", "job_queue": {"enabled": False}}


def _workload_ids() -> list:
    try:
        from ..workloads import workload_registry
        return sorted(spec.tool_id for spec in workload_registry.specs())
    except Exception:  # pragma: no cover - health must never fail
        return []


async def _job_health() -> Dict[str, Any]:
    """Role, admission snapshot and Redis queue depth (one Redis round-trip)."""
    from ..job_admission import get_admission
    from ..job_queue import get_queue
    from ..event_loop_monitor import get_event_loop_monitor

    out: Dict[str, Any] = {**_job_role_info(), "admission": get_admission().snapshot()}
    lag = get_event_loop_monitor().snapshot
    out["event_loop"] = {
        "lag_seconds": lag.current_seconds,
        "max_lag_seconds": lag.max_seconds,
        "samples": lag.samples,
        "over_threshold": lag.over_threshold,
        "updated_at": lag.updated_at,
    }
    queue = get_queue()
    if queue is not None:
        try:
            out["job_queue"]["depth"] = await queue.queue_depth()
        except Exception as exc:
            out["job_queue"]["error"] = str(exc)
    return out


@router.get("/health/jobs")
async def job_health() -> Dict[str, Any]:
    """Generic workload pool health and registered capabilities."""
    out = await _job_health()
    out["workload_ids"] = _workload_ids()
    return out


@router.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def prometheus_metrics() -> PlainTextResponse:
    """Prometheus exposition for the worker control plane (no prompt content)."""
    health = await _job_health()
    queue = dict(health.get("job_queue") or {})
    admission = dict(health.get("admission") or {})
    lag = dict(health.get("event_loop") or {})
    depth_raw = queue.get("depth")
    depth = (
        int(depth_raw.get("shared") or 0) + int(depth_raw.get("pod") or 0)
        if isinstance(depth_raw, dict)
        else int(depth_raw or 0)
    )
    lines = [
        "# HELP tool_runner_up Process liveness.",
        "# TYPE tool_runner_up gauge",
        "tool_runner_up 1",
        "# HELP tool_job_queue_depth Jobs waiting in the dispatch queue.",
        "# TYPE tool_job_queue_depth gauge",
        f"tool_job_queue_depth {depth}",
        "# HELP tool_job_queue_consumer_active Whether this pod consumes jobs.",
        "# TYPE tool_job_queue_consumer_active gauge",
        f"tool_job_queue_consumer_active {1 if queue.get('consumer') else 0}",
        "# HELP tool_job_admission_running Jobs admitted on this pod.",
        "# TYPE tool_job_admission_running gauge",
        f"tool_job_admission_running {int(admission.get('running') or 0)}",
        "# HELP tool_job_admission_queued Jobs waiting for admission on this pod.",
        "# TYPE tool_job_admission_queued gauge",
        f"tool_job_admission_queued {int(admission.get('queued') or 0)}",
        "# HELP python_event_loop_lag_seconds Latest event-loop scheduling lag.",
        "# TYPE python_event_loop_lag_seconds gauge",
        f"python_event_loop_lag_seconds {float(lag.get('lag_seconds') or 0.0):.6f}",
        "# HELP python_event_loop_lag_max_seconds Maximum observed event-loop lag.",
        "# TYPE python_event_loop_lag_max_seconds gauge",
        f"python_event_loop_lag_max_seconds {float(lag.get('max_lag_seconds') or 0.0):.6f}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n")


@router.get("/health/detailed")
async def detailed_health() -> Dict[str, Any]:
    """
    Detailed health check with tool information.
    """
    registry = get_registry()
    
    if not registry.is_ready:
        return {
            "status": "not_ready",
            "service": settings.SERVICE_NAME,
            "version": settings.SERVICE_VERSION,
            "tools_count": 0,
            "categories": {},
            "message": "Registry not initialized",
        }
    
    tools = registry.list_tools()
    
    # Group by category
    categories: Dict[str, int] = {}
    for tool in tools.values():
        cat = tool.category or "general"
        categories[cat] = categories.get(cat, 0) + 1
    
    return {
        "status": "healthy",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "tools_count": len(tools),
        "categories": categories,
        "tools_path": settings.TOOLS_MODULE_PATH,
    }


@router.post("/health/cleanup")
async def cleanup_resources(
    _service: str = Depends(require_internal_key),
) -> Dict[str, Any]:
    """
    Trigger resource cleanup (free ML models, etc).
    """
    registry = get_registry()
    
    if not registry.is_ready:
        return {"ok": False, "message": "Registry not ready"}
    
    cleaned = registry.cleanup_resources()
    
    return {
        "ok": True,
        "cleaned": cleaned,
        "message": "Resources cleaned" if cleaned else "No resources to clean",
    }


@router.post("/health/reload")
async def reload_tools(
    _service: str = Depends(require_internal_key),
) -> Dict[str, Any]:
    """
    Reload all tools from disk.
    Useful after adding/modifying tools.
    """
    registry = get_registry()
    
    if not registry.is_ready:
        return {"ok": False, "message": "Registry not ready"}
    
    success = registry.reload()
    tools = registry.list_tools() if success else {}
    tools_count = len(tools)

    catalog_upserted = 0
    if success and tools:
        try:
            from ..catalog_sync import sync_tool_catalog

            catalog_upserted = await sync_tool_catalog(tools)
        except Exception as exc:
            logger.warning("Tool catalog sync after reload failed: %s", exc)

    return {
        "ok": success,
        "tools_count": tools_count,
        "catalog_upserted": catalog_upserted,
        "message": f"Reloaded {tools_count} tools" if success else "Reload failed",
    }
