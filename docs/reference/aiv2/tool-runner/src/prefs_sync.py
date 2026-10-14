"""Register plugin per-user preference namespaces with Storage at startup.

On startup the tool-runner scans every ``plugin.json`` for a ``user_prefs``
declaration and registers each namespace with Storage. Storage creates the
backing table on demand (``user_prefs_{namespace}``). This is the same dynamic,
zero-migration pattern used by ``catalog_sync`` for tool permissions: adding a
new plugin preference never requires a hand-written init-script.

Best-effort: failures are logged and never block tool-runner startup.
"""

from __future__ import annotations

import logging
from typing import List

import httpx

from .config.settings import settings

logger = logging.getLogger(__name__)


# Platform UI preferences (locale/appearance/nav/uiFlags) — not plugin-scoped.
_PLATFORM_USER_PREFS = {
    "plugin_id": "platform",
    "namespace": "platform.user_preferences",
    "schema_version": 1,
    "metadata": {
        "description": "Unified UI preferences (locale, appearance, nav pins, uiFlags)",
    },
}

# Phase C: sensitive browser work product (drafts / topology / report docs).
_PLATFORM_CLIENT_WORK = {
    "plugin_id": "platform",
    "namespace": "platform.client_work",
    "schema_version": 1,
    "metadata": {
        "description": "Per-user client work product (drafts, topology layouts, report docs)",
    },
}


async def sync_user_pref_namespaces(plugins_root: str) -> int:
    """Discover and register plugin + platform preference namespaces. Returns count synced."""
    namespaces: List[dict] = []
    try:
        from modules.plugins.discovery import build_user_pref_namespaces

        namespaces = build_user_pref_namespaces(plugins_root)
    except Exception as exc:  # pragma: no cover - import guard
        logger.warning("modules.plugins.discovery not importable: %s", exc)

    namespaces = list(namespaces) + [_PLATFORM_USER_PREFS, _PLATFORM_CLIENT_WORK]

    url = f"{settings.STORAGE_URL.rstrip('/')}/plugin-prefs/namespaces/register"
    headers = {
        "X-Internal-Key": settings.INTERNAL_API_KEY,
        "X-Service-Name": settings.SERVICE_NAME,
    }
    synced = 0
    try:
        async with httpx.AsyncClient(timeout=settings.TOOL_RUNNER_RBAC_TIMEOUT_SEC) as client:
            for ns in namespaces:
                try:
                    resp = await client.post(url, headers=headers, json=ns)
                    resp.raise_for_status()
                    synced += 1
                except Exception as exc:
                    logger.warning(
                        "Failed to register pref namespace %s: %s",
                        ns.get("namespace"),
                        exc,
                    )
    except Exception as exc:
        logger.warning("Plugin prefs sync skipped (storage unreachable): %s", exc)
        return synced

    if synced:
        logger.info("✅ Registered %d preference namespace(s) with storage", synced)
    return synced
