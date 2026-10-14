"""HTTP discovery for drop-in orchestrators (does not alter /chat*)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from .registry import get_registry

router = APIRouter(tags=["orchestrators"])


def _public_manifest(reg) -> Dict[str, Any]:
    m = reg.manifest.to_dict()
    return {
        **m,
        "ready": reg.ready,
        "load_error": reg.load_error,
    }


@router.get("/orchestrators")
async def list_orchestrators(
    capability: Optional[str] = Query(
        None,
        description="Comma-separated required capabilities",
    ),
    operation: Optional[str] = Query(
        None,
        description="Comma-separated required operations",
    ),
    kind: Optional[str] = Query(None, description="Exact kind filter"),
) -> Dict[str, Any]:
    registry = get_registry()
    items = registry.list_orchestrators(
        capability=capability,
        operation=operation,
        kind=kind,
        ready_only=False,
    )
    return {
        "count": len(items),
        "orchestrators": [_public_manifest(r) for r in items],
    }


@router.get("/orchestrators/{orchestrator_id}")
async def get_orchestrator(orchestrator_id: str) -> Dict[str, Any]:
    registry = get_registry()
    reg = registry.get(orchestrator_id)
    if reg is None:
        raise HTTPException(status_code=404, detail=f"orchestrator not found: {orchestrator_id}")
    missing = registry.missing_dependencies(orchestrator_id)
    return {
        **_public_manifest(reg),
        "missing_dependencies": missing,
    }
