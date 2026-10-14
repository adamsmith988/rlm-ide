"""
Internal Routes — اندپوینت‌های داخلی (service-to-service)

These endpoints are called by other services (e.g. llm-proxy)
and are NOT exposed through the API Gateway.

POST /internal/cleanup/{tool_id} → حذف مدل از حافظه
POST /internal/sessions/checkpoint → mid-run tool_sessions upsert (opt-in plugins)
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..authz import require_internal_key
from ..config import settings
from ..utils.registry_wrapper import get_registry
from .internal_llm import router as internal_llm_router

router = APIRouter(prefix="/internal", tags=["internal"])
router.include_router(internal_llm_router)
logger = logging.getLogger(__name__)


class SessionCheckpointBody(BaseModel):
    """Standard plugin_session envelope (+ optional force/throttle controls)."""

    envelope: Dict[str, Any] = Field(default_factory=dict)
    plugin_id: Optional[str] = None
    tool_id: Optional[str] = None
    record_id: Optional[str] = None
    run_id: Optional[str] = None
    user_id: Optional[str] = None
    force: bool = False
    min_interval_sec: Optional[float] = None


@router.post("/cleanup/{tool_id}")
async def cleanup_tool_model(
    tool_id: str,
    _service: str = Depends(require_internal_key),
) -> Dict[str, Any]:
    """
    Clean up a tool's model from memory.

    Called by llm-proxy GPU Manager when evicting idle or
    manually selected models. This endpoint performs the actual
    model unloading from this tool-runner instance.
    """
    try:
        registry = get_registry()
        if not registry.is_ready:
            raise HTTPException(503, "Registry not ready")

        success = registry.cleanup_tool(tool_id)
        if success:
            logger.info(f"🧹 Remote cleanup completed: {tool_id}")
            return {"ok": True, "cleaned": tool_id}
        else:
            logger.debug(f"No model to clean for {tool_id}")
            return {"ok": True, "cleaned": None, "message": f"No model loaded for {tool_id}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cleanup failed for {tool_id}: {e}")
        raise HTTPException(500, f"Cleanup failed: {e}")


@router.post("/sessions/checkpoint")
async def checkpoint_session(
    body: SessionCheckpointBody,
    _service: str = Depends(require_internal_key),
) -> Dict[str, Any]:
    """Upsert a mid-run (or terminal) plugin_session into Storage tool_sessions.

    Opt-in long-running plugins call this so incomplete runs survive tool-runner
    restarts. Throttled unless ``force`` or status is terminal.
    """
    from modules.tools.session_checkpoint import checkpoint_tool_session

    env = dict(body.envelope or {})
    uid = str(body.user_id or env.get("user_id") or "").strip()
    if not uid:
        raise HTTPException(400, "user_id required")
    auth = {"user_id": uid}
    try:
        result = await checkpoint_tool_session(
            envelope=env,
            auth_context=auth,
            plugin_id=body.plugin_id,
            tool_id=body.tool_id,
            record_id=body.record_id,
            run_id=body.run_id,
            force=bool(body.force),
            min_interval_sec=body.min_interval_sec,
            internal_api_key=settings.INTERNAL_API_KEY,
        )
    except Exception as exc:
        logger.error("session checkpoint failed: %s", exc)
        raise HTTPException(500, f"checkpoint_failed: {exc}") from exc
    return result
