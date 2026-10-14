"""HTTP routes for generic plugin per-user preferences.

All routes are internal-only (guarded by the global storage internal-service
middleware). The per-user GET/PUT routes additionally require ``X-User-Id`` so
a user can only read/write their own row.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import BaseModel

from .plugin_user_prefs import PluginUserPrefsService

router = APIRouter(tags=["plugin-prefs"])


def _svc(request: Request) -> PluginUserPrefsService:
    return PluginUserPrefsService(request.app.state.storage)


def _require_user_id(request: Request) -> str:
    uid = request.headers.get("X-User-Id")
    if not uid or not str(uid).strip():
        raise HTTPException(status_code=400, detail="X-User-Id required")
    return str(uid).strip()


class NamespaceRegisterBody(BaseModel):
    plugin_id: str
    namespace: str
    schema_version: int = 1
    metadata: Optional[Dict[str, Any]] = None


class PrefsPutBody(BaseModel):
    prefs: Dict[str, Any]
    schema_version: Optional[int] = None


@router.post("/plugin-prefs/namespaces/register")
async def plugin_prefs_register(request: Request, body: NamespaceRegisterBody):
    try:
        return await _svc(request).register_namespace(
            plugin_id=body.plugin_id,
            namespace=body.namespace,
            schema_version=body.schema_version,
            metadata=body.metadata,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/plugin-prefs/{namespace}")
async def plugin_prefs_get(request: Request, namespace: str):
    user_id = _require_user_id(request)
    try:
        return await _svc(request).get_prefs(user_id, namespace)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError:
        raise HTTPException(status_code=404, detail="namespace_not_registered")


@router.put("/plugin-prefs/{namespace}")
async def plugin_prefs_put(
    request: Request,
    namespace: str,
    body: PrefsPutBody,
):
    user_id = _require_user_id(request)
    try:
        return await _svc(request).put_prefs(
            user_id, namespace, body.prefs, body.schema_version
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError:
        raise HTTPException(status_code=404, detail="namespace_not_registered")


def register_plugin_prefs_routes(app) -> None:
    app.include_router(router)
