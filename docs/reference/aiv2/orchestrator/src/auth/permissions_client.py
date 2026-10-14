"""Per-user RBAC permission resolution for the orchestrator.

The chat path reaches tool-runner over the trusted internal hop, so the
end-user's bearer token is not available here. To still honour the admin's
RBAC matrix and per-user overrides, we ask auth-service to resolve a user's
effective permissions (role permissions + group RBAC + overrides) and cache
the result briefly.

This lets the orchestrator:
  * gate the whole tool surface on ``chat:use_tools`` (chat-only roles get a
    plain LLM with no tools), and
  * filter individual tools by ``tools:execute`` / ``tools:execute:{tool_id}``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import httpx

from ..config import settings

log = logging.getLogger(__name__)

CHAT_USE_TOOLS = settings.CHAT_USE_TOOLS_PERMISSION
TOOLS_EXECUTE = settings.TOOLS_EXECUTE_PERMISSION


def permission_matches(permission: str, granted: Set[str]) -> bool:
    """Exact match with trailing-``*`` wildcard support (mirrors auth-service)."""
    if permission in granted:
        return True
    for candidate in granted:
        if candidate.endswith("*") and permission.startswith(candidate[:-1]):
            return True
    return False


@dataclass
class UserPermissions:
    """Resolved permission view for a single user."""

    user_id: str
    permissions: Set[str] = field(default_factory=set)
    denied: Set[str] = field(default_factory=set)
    is_super_admin: bool = False
    resolved: bool = True

    def can_use_tools(self) -> bool:
        if self.is_super_admin:
            return True
        if permission_matches(CHAT_USE_TOOLS, self.denied):
            return False
        return permission_matches(CHAT_USE_TOOLS, self.permissions)

    def can_execute_tool(self, tool_id: str) -> bool:
        if self.is_super_admin:
            return True
        specific = f"{TOOLS_EXECUTE}:{tool_id}"
        if permission_matches(specific, self.denied) or TOOLS_EXECUTE in self.denied:
            return False
        return permission_matches(specific, self.permissions)


# user_id -> (UserPermissions, expires_at)
_cache: Dict[str, Tuple[UserPermissions, float]] = {}


def bust_cache(user_id: Optional[str] = None) -> int:
    """Clear cached permissions. Returns the number of entries removed.

    With ``user_id`` clears just that user; otherwise clears everything. Called
    when an admin changes RBAC so chat reflects the new policy immediately
    instead of waiting for the TTL to expire.
    """
    if user_id:
        existed = 1 if _cache.pop(user_id, None) is not None else 0
        return existed
    count = len(_cache)
    _cache.clear()
    return count


def _allow_all(user_id: str) -> UserPermissions:
    return UserPermissions(
        user_id=user_id,
        permissions={CHAT_USE_TOOLS, f"{TOOLS_EXECUTE}:*"},
        denied=set(),
        is_super_admin=True,
        resolved=False,
    )


async def get_user_permissions(
    user_id: Optional[str],
    roles: Optional[Iterable[str]] = None,
    *,
    force_refresh: bool = False,
) -> UserPermissions:
    """Resolve (and cache) a user's effective permissions via auth-service.

    On failure the result depends on ``TOOL_PERMISSION_FAIL_OPEN``: fail-open
    returns an allow-all view, fail-closed returns an empty permission set.
    """
    if not settings.TOOL_PERMISSION_ENFORCEMENT:
        return _allow_all(user_id or "unknown")

    if not user_id:
        # No identity to enforce against — treat like a service caller.
        return _allow_all("unknown")

    now = time.time()
    if not force_refresh:
        cached = _cache.get(user_id)
        if cached and cached[1] > now:
            return cached[0]

    headers = {
        "X-Internal-Key": settings.INTERNAL_API_KEY,
        "X-Service-Name": settings.SERVICE_NAME,
    }
    role_list = [r for r in (roles or []) if r]
    if role_list:
        headers["X-User-Roles"] = ",".join(role_list)

    url = (
        f"{settings.AUTH_SERVICE_URL.rstrip('/')}"
        f"/permissions/internal/effective/{user_id}"
    )
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        result = UserPermissions(
            user_id=user_id,
            permissions=set(data.get("permissions") or []),
            denied=set(data.get("denied_permissions") or []),
            is_super_admin=bool(data.get("is_super_admin")),
            resolved=True,
        )
    except Exception as exc:
        log.warning(
            "Permission resolve failed for user=%s (fail_open=%s): %s",
            user_id,
            settings.TOOL_PERMISSION_FAIL_OPEN,
            exc,
        )
        if settings.TOOL_PERMISSION_FAIL_OPEN:
            return _allow_all(user_id)
        return UserPermissions(user_id=user_id, permissions=set(), resolved=False)

    _cache[user_id] = (result, now + settings.TOOL_PERMISSION_CACHE_TTL)
    return result


async def resolve_allowed_tools(
    tools_manifest: List[dict],
    user_id: Optional[str],
    roles: Optional[Iterable[str]] = None,
) -> List[dict]:
    """Return the subset of the manifest the user may use in chat.

    Applies the ``chat:use_tools`` gate first (chat-only users get no tools)
    then per-tool ``tools:execute`` filtering. Enforcement is a no-op when
    ``TOOL_PERMISSION_ENFORCEMENT`` is disabled.
    """
    if not settings.TOOL_PERMISSION_ENFORCEMENT:
        return tools_manifest

    perms = await get_user_permissions(user_id, roles)
    if not perms.can_use_tools():
        log.info(
            "🔒 user=%s lacks %s — chat runs with no tools", user_id, CHAT_USE_TOOLS
        )
        return []
    filtered = filter_tools_by_permission(tools_manifest or [], perms)
    if len(filtered) != len(tools_manifest or []):
        log.info(
            "🔒 tool permission filter user=%s: %s/%s tools allowed",
            user_id,
            len(filtered),
            len(tools_manifest or []),
        )
    return filtered


def filter_tools_by_permission(
    tools: List[dict],
    perms: UserPermissions,
) -> List[dict]:
    """Drop tools the user cannot execute.

    Tool dicts use ``name`` (orchestrator manifest) or ``id`` as the tool id.
    """
    if perms.is_super_admin:
        return tools
    allowed: List[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_id = tool.get("name") or tool.get("id") or ""
        if tool_id and perms.can_execute_tool(tool_id):
            allowed.append(tool)
    return allowed
