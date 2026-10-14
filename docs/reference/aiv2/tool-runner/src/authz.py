"""Central authorization for tool-runner routes."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import httpx
from fastapi import HTTPException, Request, status

from .config import settings

logger = logging.getLogger(__name__)

TOOLS_SECTION = settings.TOOLS_SECTION_ID
PERMISSION_TOOLS_VIEW = settings.TOOLS_VIEW_PERMISSION
PERMISSION_TOOLS_EXECUTE = settings.TOOLS_EXECUTE_PERMISSION


# ---------------------------------------------------------------------------
# Shared HTTP client (connection reuse to auth-service)
# ---------------------------------------------------------------------------

_http_client: Optional[httpx.AsyncClient] = None


def _get_http_client() -> httpx.AsyncClient:
    """Lazily build one module-level ``httpx.AsyncClient`` for auth-service calls.

    Per-request clients tear down the TCP/keep-alive pool on every authz check;
    a shared client lets hot paths (tool execute) reuse connections.
    """
    global _http_client
    client = _http_client
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=settings.TOOL_RUNNER_RBAC_TIMEOUT_SEC)
        _http_client = client
    return client


async def aclose_http_client() -> None:
    """Close the shared client (call from app shutdown)."""
    global _http_client
    client = _http_client
    _http_client = None
    if client is not None and not client.is_closed:
        await client.aclose()


# ---------------------------------------------------------------------------
# Short TTL cache for impersonated effective-permission lookups
# ---------------------------------------------------------------------------

# key -> (expires_at_monotonic, effective dict)
_effective_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _effective_cache_ttl() -> float:
    try:
        return float(settings.TOOL_RUNNER_RBAC_EFFECTIVE_CACHE_TTL_SEC or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _effective_cache_key(
    user_id: str, user_roles: Optional[str], group_id: Optional[str]
) -> str:
    return f"{user_id}|{user_roles or ''}|{group_id or ''}"


def _effective_cache_get(key: str) -> Optional[Dict[str, Any]]:
    entry = _effective_cache.get(key)
    if entry is None:
        return None
    expires_at, value = entry
    if time.monotonic() >= expires_at:
        _effective_cache.pop(key, None)
        return None
    return value


def _effective_cache_put(key: str, value: Dict[str, Any], ttl: float) -> None:
    now = time.monotonic()
    max_entries = int(settings.TOOL_RUNNER_RBAC_EFFECTIVE_CACHE_MAX_ENTRIES or 0)
    if max_entries > 0 and len(_effective_cache) >= max_entries:
        # Drop expired entries first; if still full, reset (bounded memory).
        for k in [k for k, (exp, _) in _effective_cache.items() if now >= exp]:
            _effective_cache.pop(k, None)
        if len(_effective_cache) >= max_entries:
            _effective_cache.clear()
    _effective_cache[key] = (now + ttl, value)


def reset_effective_cache() -> None:
    """Clear cached effective permissions (tests / admin RBAC changes)."""
    _effective_cache.clear()


def _permission_matches(permission: str, granted: set[str]) -> bool:
    """Exact match with trailing-``*`` wildcard support (mirrors auth-service)."""
    if permission in granted:
        return True
    for candidate in granted:
        if candidate.endswith("*") and permission.startswith(candidate[:-1]):
            return True
    return False


def _tool_execute_allowed(tool_id: str, granted: set[str], denied: set[str]) -> bool:
    """Fine-grained + wildcard + deny semantics for tool execution.

    Coarse ``tools:execute`` alone does not authorize a specific tool run; see
    auth-service ``tool_execute_allowed`` for the shared contract.
    """
    specific = f"{PERMISSION_TOOLS_EXECUTE}:{tool_id}"
    if _permission_matches(specific, denied) or PERMISSION_TOOLS_EXECUTE in denied:
        return False
    return _permission_matches(specific, granted)


@dataclass(frozen=True)
class AuthzContext:
    allowed: bool
    mode: str
    principal: str
    user_id: Optional[str] = None
    group_id: Optional[str] = None
    case_id: Optional[str] = None
    service_name: Optional[str] = None
    permission: Optional[str] = None
    user_roles: Optional[str] = None

    def as_tool_context(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "principal": self.principal,
            "user_id": self.user_id,
            "group_id": self.group_id,
            "case_id": self.case_id,
            "service_name": self.service_name,
            "permission": self.permission,
            "user_roles": self.user_roles,
        }


def _trusted_services() -> set[str]:
    return {
        item.strip()
        for item in (settings.TOOL_RUNNER_TRUSTED_SERVICES or "").split(",")
        if item.strip()
    }


def _header(request: Request, name: str) -> Optional[str]:
    value = request.headers.get(name)
    return value.strip() if value and value.strip() else None


def _bearer_token(request: Request) -> Optional[str]:
    authorization = _header(request, "authorization")
    if not authorization:
        return None
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header",
        )
    return authorization


async def require_internal_key(request: Request) -> str:
    """FastAPI dependency: require a valid internal service credential.

    Use on service-to-service routes (``/internal/*``, ``/plugins/*``, mutating
    ``/health/*``) that must never be reachable by untrusted clients. Returns the
    asserted service name on success.
    """
    internal_key = _header(request, "x-internal-key")
    service_name = _header(request, "x-service-name") or ""
    if not internal_key or internal_key != settings.INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Valid internal service credential required",
        )
    return service_name


def _internal_context(request: Request, permission: str) -> Optional[AuthzContext]:
    internal_key = _header(request, "x-internal-key")
    service_name = _header(request, "x-service-name")
    if not internal_key:
        return None
    if internal_key != settings.INTERNAL_API_KEY:
        logger.warning(
            "tool_authz_internal_failed reason=invalid_key service=%s permission=%s",
            service_name,
            permission,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Valid internal service credential required",
        )
    if service_name not in _trusted_services():
        logger.warning(
            "tool_authz_internal_failed reason=untrusted_service service=%s permission=%s",
            service_name,
            permission,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Service '{service_name or ''}' is not allowed to use tool-runner",
        )
    return AuthzContext(
        allowed=True,
        mode="internal",
        principal=f"service:{service_name}",
        user_id=_header(request, "x-user-id"),
        group_id=_header(request, "x-group-id"),
        case_id=_header(request, "x-case-id"),
        service_name=service_name,
        permission=permission,
        user_roles=_header(request, "x-user-roles"),
    )


async def _check_auth_service(
    *,
    authorization: str,
    permission: str,
    group_id: Optional[str],
) -> bool:
    url = f"{settings.AUTH_SERVICE_URL.rstrip('/')}/permissions/check-access"
    client = _get_http_client()
    response = await client.post(
        url,
        headers={"Authorization": authorization},
        json={
            "section": TOOLS_SECTION,
            "permission": permission,
            "group_id": group_id,
        },
    )
    if response.status_code == 401:
        logger.warning(
            "tool_authz_user_failed reason=invalid_token permission=%s group_id=%s",
            permission,
            group_id,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    response.raise_for_status()
    data = response.json()
    return bool(data.get("allowed"))


async def _fetch_internal_effective(
    user_id: str,
    user_roles: Optional[str],
    group_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve a user's effective permissions via auth-service (impersonation).

    Returns ``{"granted": set, "denied": set, "is_super_admin": bool}`` or
    ``None`` when auth-service is unreachable.

    Successful replies are cached for ``TOOL_RUNNER_RBAC_EFFECTIVE_CACHE_TTL_SEC``
    keyed by ``user_id|roles|group``. Failures are never cached, so a miss
    still surfaces the transport error to the caller (fail-closed by default).
    """
    ttl = _effective_cache_ttl()
    cache_key = _effective_cache_key(user_id, user_roles, group_id)
    if ttl > 0:
        cached = _effective_cache_get(cache_key)
        if cached is not None:
            return cached

    url = (
        f"{settings.AUTH_SERVICE_URL.rstrip('/')}"
        f"/permissions/internal/effective/{user_id}"
    )
    headers = {
        "X-Internal-Key": settings.INTERNAL_API_KEY,
        "X-Service-Name": settings.SERVICE_NAME,
    }
    if user_roles:
        headers["X-User-Roles"] = user_roles
    client = _get_http_client()
    response = await client.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    granted = set(data.get("permissions") or [])
    denied = set(data.get("denied_permissions") or [])

    # Deny-wins union with the relevant group's scoped permissions, when the
    # caller is acting within a group the user actually belongs to.
    if group_id:
        group_info = (data.get("groups") or {}).get(group_id)
        if group_info:
            granted |= set(group_info.get("permissions") or [])
            denied |= set(group_info.get("denied_permissions") or [])

    effective = {
        "granted": granted,
        "denied": denied,
        "is_super_admin": bool(data.get("is_super_admin")),
    }
    if ttl > 0:
        _effective_cache_put(cache_key, effective, ttl)
    return effective


async def _authorize_impersonated(
    request: Request,
    internal: AuthzContext,
    *,
    tool_id: Optional[str],
    permission: str,
) -> AuthzContext:
    """Enforce the on-behalf-of user's RBAC for a trusted internal caller.

    Internal callers with no ``X-User-Id`` (e.g. case-importer batch jobs) keep
    the legacy bypass. Otherwise the user's effective permissions decide.
    """
    if not settings.TOOL_RUNNER_RBAC_IMPERSONATION or not internal.user_id:
        return internal

    user_roles = _header(request, "x-user-roles")
    try:
        eff = await _fetch_internal_effective(
            internal.user_id, user_roles, group_id=internal.group_id
        )
    except Exception as exc:
        logger.error(
            "tool_authz_impersonation_failed user=%s permission=%s error=%s",
            internal.user_id,
            permission,
            exc,
        )
        if settings.TOOL_RUNNER_RBAC_FAIL_OPEN:
            return internal
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authorization service unavailable",
        ) from exc

    if eff is None:
        if settings.TOOL_RUNNER_RBAC_FAIL_OPEN:
            return internal
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authorization service unavailable",
        )

    if eff["is_super_admin"]:
        return internal

    if tool_id is not None:
        allowed = _tool_execute_allowed(tool_id, eff["granted"], eff["denied"])
    else:
        # Honour deny overrides on the non-tool path too (deny wins over grant).
        allowed = (
            not _permission_matches(permission, eff["denied"])
            and _permission_matches(permission, eff["granted"])
        )
    if not allowed:
        logger.warning(
            "PERMISSION_DENIED permission=%s user_id=%s resource_id=%s principal=%s",
            permission,
            internal.user_id,
            tool_id,
            internal.principal,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission '{permission}' required",
        )
    return internal


async def _authorize_user(
    request: Request,
    *,
    permission: str,
    fallback_permission: Optional[str] = None,
) -> AuthzContext:
    authorization = _bearer_token(request)
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    group_id = _header(request, "x-group-id")
    case_id = _header(request, "x-case-id")
    user_id = _header(request, "x-user-id")
    user_roles = _header(request, "x-user-roles")
    checked_permission = permission

    try:
        allowed = await _check_auth_service(
            authorization=authorization,
            permission=permission,
            group_id=group_id,
        )
        if not allowed and fallback_permission and fallback_permission != permission:
            checked_permission = fallback_permission
            allowed = await _check_auth_service(
                authorization=authorization,
                permission=fallback_permission,
                group_id=group_id,
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "tool_authz_transport_failed permission=%s group_id=%s error=%s",
            permission,
            group_id,
            exc,
        )
        if settings.TOOL_RUNNER_RBAC_FAIL_OPEN:
            logger.warning("RBAC check failed open for %s: %s", permission, exc)
            return AuthzContext(
                allowed=True,
                mode="fail-open",
                principal=user_id or "unknown-user",
                user_id=user_id,
                group_id=group_id,
                case_id=case_id,
                permission=permission,
                user_roles=user_roles,
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authorization service unavailable",
        ) from exc

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission '{permission}' required",
        )

    return AuthzContext(
        allowed=True,
        mode="user",
        principal=user_id or "user-token",
        user_id=user_id,
        group_id=group_id,
        case_id=case_id,
        permission=checked_permission,
        user_roles=user_roles,
    )


async def authorize_tool_view(request: Request) -> AuthzContext:
    if not settings.TOOL_RUNNER_RBAC_ENABLED:
        return AuthzContext(True, "disabled", "rbac-disabled", permission=PERMISSION_TOOLS_VIEW)

    internal = _internal_context(request, PERMISSION_TOOLS_VIEW)
    if internal:
        return await _authorize_impersonated(
            request, internal, tool_id=None, permission=PERMISSION_TOOLS_VIEW
        )

    return await _authorize_user(request, permission=PERMISSION_TOOLS_VIEW)


def execute_permission(tool_id: str) -> str:
    return f"{PERMISSION_TOOLS_EXECUTE}:{tool_id}"


async def authorize_tool_execute(request: Request, tool_id: str) -> AuthzContext:
    if not settings.TOOL_RUNNER_RBAC_ENABLED:
        return AuthzContext(True, "disabled", "rbac-disabled", permission=PERMISSION_TOOLS_EXECUTE)

    permission = (
        execute_permission(tool_id)
        if settings.TOOL_RUNNER_RBAC_FINE_GRAINED
        else PERMISSION_TOOLS_EXECUTE
    )

    internal = _internal_context(request, permission)
    if internal:
        return await _authorize_impersonated(
            request, internal, tool_id=tool_id, permission=permission
        )

    started = time.perf_counter()
    ctx = await _authorize_user(
        request,
        permission=permission,
        fallback_permission=PERMISSION_TOOLS_EXECUTE,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    logger.info(
        "tool authz allowed: tool=%s principal=%s permission=%s group=%s latency_ms=%s",
        tool_id,
        ctx.principal,
        ctx.permission,
        ctx.group_id,
        elapsed_ms,
    )
    return ctx


async def filter_executable_tool_ids(
    request: Request,
    tool_ids: Iterable[str],
) -> set[str]:
    """Return tool IDs the caller can execute; internal/coarse execute sees all."""
    ids = set(tool_ids)
    if not ids or not settings.TOOL_RUNNER_RBAC_ENABLED:
        return ids

    await authorize_tool_view(request)
    internal = _internal_context(request, PERMISSION_TOOLS_EXECUTE)
    if internal:
        # Impersonated internal call → filter by the user's effective RBAC.
        if settings.TOOL_RUNNER_RBAC_IMPERSONATION and internal.user_id:
            user_roles = _header(request, "x-user-roles")
            try:
                eff = await _fetch_internal_effective(internal.user_id, user_roles)
            except Exception as exc:
                if settings.TOOL_RUNNER_RBAC_FAIL_OPEN:
                    logger.warning("RBAC manifest impersonation failed open: %s", exc)
                    return ids
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Authorization service unavailable",
                ) from exc
            if eff is None or eff["is_super_admin"]:
                return ids
            return {
                tid
                for tid in ids
                if _tool_execute_allowed(tid, eff["granted"], eff["denied"])
            }
        return ids

    authorization = _bearer_token(request)
    group_id = _header(request, "x-group-id")
    try:
        if await _check_auth_service(
            authorization=authorization or "",
            permission=PERMISSION_TOOLS_EXECUTE,
            group_id=group_id,
        ):
            return ids

        allowed: set[str] = set()
        for tool_id in sorted(ids):
            if await _check_auth_service(
                authorization=authorization or "",
                permission=execute_permission(tool_id),
                group_id=group_id,
            ):
                allowed.add(tool_id)
        return allowed
    except HTTPException:
        raise
    except Exception as exc:
        if settings.TOOL_RUNNER_RBAC_FAIL_OPEN:
            logger.warning("RBAC manifest filtering failed open: %s", exc)
            return ids
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authorization service unavailable",
        ) from exc
