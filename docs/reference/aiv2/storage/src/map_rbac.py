"""Map layer visibility — auth-service RBAC is the source of truth.

Role permissions and user permission overrides (auth-service) decide which
``map_layer:{uuid}`` rows a caller may read. Storage does not maintain a
parallel grant table for layers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Set

import httpx

from .config import settings

logger = logging.getLogger(__name__)

MAP_LAYER_PREFIX = "map_layer:"
MAP_LAYERS_IMPORT_SHARED = "map_layers:import_shared"
MAP_LAYERS_REGISTER_API = "map_layers:register_api"


@dataclass(frozen=True)
class MapLayerRbac:
    """Resolved layer access for one user."""

    user_id: str
    is_super_admin: bool = False
    granted_layer_ids: frozenset[str] = frozenset()
    denied_layer_ids: frozenset[str] = frozenset()
    granted_permissions: frozenset[str] = frozenset()
    unavailable: bool = False

    def can_import_shared(self) -> bool:
        """Register layers from MAP_LAYERS_ROOT (shared server paths)."""
        if self.is_super_admin:
            return True
        if self.unavailable:
            return False
        return MAP_LAYERS_IMPORT_SHARED in self.granted_permissions

    def can_register_api(self) -> bool:
        """Register vector_api layers that trigger server-side outbound HTTP."""
        if self.is_super_admin:
            return True
        if self.unavailable:
            return False
        return MAP_LAYERS_REGISTER_API in self.granted_permissions

    def can_read_layer(self, layer_id: str, *, owner_user_id: Optional[str] = None) -> bool:
        if self.unavailable:
            return owner_user_id is not None and str(owner_user_id) == str(self.user_id)
        if self.is_super_admin:
            return True
        lid = str(layer_id)
        if owner_user_id is not None and str(owner_user_id) == str(self.user_id):
            return True
        if lid in self.denied_layer_ids:
            return False
        if lid in self.granted_layer_ids:
            return True
        return False

    def can_mutate_layer(self, layer_id: str, *, owner_user_id: Optional[str] = None) -> bool:
        """Write/delete — owner or super-admin only (grants are read-only)."""
        if self.unavailable:
            return owner_user_id is not None and str(owner_user_id) == str(self.user_id)
        if self.is_super_admin:
            return True
        return owner_user_id is not None and str(owner_user_id) == str(self.user_id)


def layer_ids_from_permissions(perms: Any) -> Set[str]:
    ids: Set[str] = set()
    for raw in perms or []:
        p = str(raw).strip()
        if not p.startswith(MAP_LAYER_PREFIX):
            continue
        lid = p[len(MAP_LAYER_PREFIX) :].strip()
        if lid:
            ids.add(lid)
    return ids


async def resolve_map_layer_rbac(
    user_id: str,
    user_roles: Optional[str] = None,
) -> MapLayerRbac:
    """Load effective permissions from auth-service (roles + overrides)."""
    base = (getattr(settings, "AUTH_SERVICE_URL", None) or "").rstrip("/")
    if not base or not user_id:
        return MapLayerRbac(user_id=user_id, unavailable=True)

    url = f"{base}/permissions/internal/effective/{user_id}"
    headers = {
        "X-Internal-Key": settings.INTERNAL_API_KEY,
        "X-Service-Name": "storage",
    }
    if user_roles and str(user_roles).strip():
        headers["X-User-Roles"] = str(user_roles).strip()

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, headers=headers)
        if resp.status_code >= 400:
            logger.warning(
                "map_rbac effective permissions HTTP %s for user %s",
                resp.status_code,
                user_id,
            )
            return MapLayerRbac(user_id=user_id, unavailable=True)
        body = resp.json()
    except Exception as exc:
        logger.warning("map_rbac auth-service unreachable for %s: %s", user_id, exc)
        return MapLayerRbac(user_id=user_id, unavailable=True)

    granted = set(body.get("permissions") or [])
    denied = set(body.get("denied_permissions") or [])
    granted_ids = layer_ids_from_permissions(granted)
    denied_ids = layer_ids_from_permissions(denied)
    granted_ids -= denied_ids

    return MapLayerRbac(
        user_id=user_id,
        is_super_admin=bool(body.get("is_super_admin")),
        granted_layer_ids=frozenset(granted_ids),
        denied_layer_ids=frozenset(denied_ids),
        granted_permissions=frozenset(str(p) for p in granted),
    )
