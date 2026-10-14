"""Poll llm-proxy /capacity and apply published JobAdmission resource_limits."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

from .config import settings
from .job_admission import get_admission

logger = logging.getLogger(__name__)

_poll_task: Optional[asyncio.Task[Any]] = None


def _capacity_url() -> str:
    base = (settings.LLM_PROXY_URL or "").rstrip("/")
    return f"{base}/capacity"


def _headers() -> Dict[str, str]:
    key = (settings.INTERNAL_API_KEY or "").strip()
    return {"X-Internal-Key": key} if key else {}


def apply_capacity_payload(payload: Dict[str, Any]) -> Dict[str, int]:
    """Merge proxy ``resource_limits`` into JobAdmission; keep unlisted classes."""
    raw = payload.get("resource_limits") if isinstance(payload, dict) else None
    if not isinstance(raw, dict) or not raw:
        return {}
    parsed: Dict[str, int] = {}
    for key, value in raw.items():
        try:
            parsed[str(key)] = max(1, int(value))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return {}
    admission = get_admission()
    admission.update_resource_limits(parsed, merge=True)
    return dict(admission.resource_limits)


async def poll_capacity_once(*, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    url = _capacity_url()
    if not url.startswith("http"):
        return None
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, headers=_headers())
        resp.raise_for_status()
        data = resp.json()
    if isinstance(data, dict):
        apply_capacity_payload(data)
        return data
    return None


async def _poll_loop(interval_sec: float) -> None:
    while True:
        try:
            await poll_capacity_once()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("capacity poll failed: %s", exc)
        await asyncio.sleep(interval_sec)


async def start_capacity_poller() -> Optional[asyncio.Task[Any]]:
    global _poll_task
    if not bool(getattr(settings, "CAPACITY_POLL_ENABLED", True)):
        logger.info("Capacity poller disabled")
        return None
    interval = float(getattr(settings, "CAPACITY_POLL_INTERVAL_SEC", 10.0) or 10.0)
    if _poll_task is not None and not _poll_task.done():
        return _poll_task
    # Prime once so cold start does not wait a full interval.
    try:
        await poll_capacity_once()
    except Exception as exc:
        logger.info("Initial capacity poll deferred: %s", exc)
    _poll_task = asyncio.create_task(_poll_loop(max(2.0, interval)), name="capacity-poller")
    logger.info("📡 Capacity poller started (interval=%.1fs)", interval)
    return _poll_task


async def stop_capacity_poller() -> None:
    global _poll_task
    task = _poll_task
    _poll_task = None
    if task is not None:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
