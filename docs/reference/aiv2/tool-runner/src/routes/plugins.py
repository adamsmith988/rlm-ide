"""
Plugin manifests (plugin.json) — same discovery pipeline as tools, extra metadata for UI/gateway.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from ..authz import require_internal_key
from ..utils import get_registry

router = APIRouter(prefix="/plugins", tags=["Plugins"])


@router.get("")
async def list_plugins(
    include_tools: bool = False,
    _service: str = Depends(require_internal_key),
) -> Dict[str, Any]:
    """List all plugin.json bundles under PLUGINS_MODULE_PATH.

    When `include_tools=true`, embeds full tool.json contents per plugin so the
    API Gateway can build OpenAPI/Swagger entries for each plugin tool.
    """
    reg = get_registry()
    plugins: List[Dict[str, Any]] = (
        reg.list_plugins(include_tools=include_tools) if reg.is_ready else []
    )
    return {"plugins": plugins, "count": len(plugins)}


@router.get("/swagger-index")
async def list_plugins_swagger_index(
    _service: str = Depends(require_internal_key),
) -> Dict[str, Any]:
    """Compact per-plugin tool list with `args_schema`/`returns_schema` for Swagger injection."""
    reg = get_registry()
    plugins = reg.list_plugins_swagger() if reg.is_ready else []
    return {"plugins": plugins, "count": len(plugins)}


@router.get("/{plugin_id}")
async def get_plugin(
    plugin_id: str,
    include_tools: bool = False,
    _service: str = Depends(require_internal_key),
) -> Dict[str, Any]:
    """Single plugin manifest by id."""
    reg = get_registry()
    if not reg.is_ready:
        raise HTTPException(status_code=503, detail="registry_not_ready")
    for p in reg.list_plugins(include_tools=include_tools):
        if str(p.get("id")) == plugin_id:
            return p
    raise HTTPException(status_code=404, detail="plugin_not_found")
