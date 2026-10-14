"""Sync discovered tools into the auth-service permission catalog.

On startup the tool-runner knows every tool that exists. It pushes a
``tools:execute:{tool_id}`` permission for each into auth-service so the admin
Permissions Matrix can offer per-tool access without any code change. This is
best-effort: failures are logged and never block tool-runner startup.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from .config.settings import settings

logger = logging.getLogger(__name__)


def build_catalog_items(tools: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build catalog upsert items from the registry's tool map."""
    items: List[Dict[str, Any]] = []
    for tool_id, info in tools.items():
        category = getattr(info, "category", None) or "tools"
        description = getattr(info, "description", "") or ""
        items.append(
            {
                "permission_id": f"tools:execute:{tool_id}",
                "resource_type": "tool",
                "resource_id": tool_id,
                "label": f"Execute: {tool_id}",
                "category": "tools",
                "metadata": {
                    "tool_category": category,
                    "description": description[:200],
                },
            }
        )
    return items


async def sync_tool_catalog(tools: Dict[str, Any]) -> int:
    """Push tool permissions to auth-service. Returns count or 0 on failure."""
    items = build_catalog_items(tools)
    if not items:
        return 0

    url = f"{settings.AUTH_SERVICE_URL.rstrip('/')}/rbac/catalog/upsert"
    headers = {
        "X-Internal-Key": settings.INTERNAL_API_KEY,
        "X-Service-Name": settings.SERVICE_NAME,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.TOOL_RUNNER_RBAC_TIMEOUT_SEC) as client:
            resp = await client.post(
                url, headers=headers, json={"source": "tool-runner", "items": items}
            )
        resp.raise_for_status()
        count = resp.json().get("upserted", 0)
        logger.info("✅ Synced %d tool permissions to auth-service catalog", count)
        return count
    except Exception as exc:
        logger.warning("Tool catalog sync skipped (auth-service unreachable): %s", exc)
        return 0
