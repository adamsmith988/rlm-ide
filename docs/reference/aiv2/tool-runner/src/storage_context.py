"""Propagate tool auth context to Storage HTTP calls made inside tools."""

from __future__ import annotations

import contextvars
import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

try:
    import requests
except Exception:  # pragma: no cover - requests may not be installed
    requests = None  # type: ignore

from .config import settings

logger = logging.getLogger(__name__)

_storage_headers: contextvars.ContextVar[Dict[str, str]] = contextvars.ContextVar(
    "tool_runner_storage_headers",
    default={},
)
_patched = False


def set_storage_context(auth_context: Optional[Dict[str, Any]]):
    """Set per-tool Storage headers; returns token for reset."""
    headers = {
        "X-Internal-Key": settings.INTERNAL_API_KEY,
        "X-Service-Name": settings.SERVICE_NAME,
    }
    ctx = auth_context or {}
    user_id = ctx.get("user_id")
    group_id = ctx.get("group_id")
    case_id = ctx.get("case_id")
    if user_id:
        headers["X-User-Id"] = str(user_id)
    if group_id:
        headers["X-Group-Id"] = str(group_id)
    if case_id:
        headers["X-Case-Id"] = str(case_id)
    roles = ctx.get("user_roles")
    if roles:
        headers["X-User-Roles"] = str(roles)
    return _storage_headers.set(headers)


def reset_storage_context(token) -> None:
    _storage_headers.reset(token)


def _is_storage_url(url: Any) -> bool:
    try:
        parsed = urlparse(str(url))
        storage = urlparse(settings.STORAGE_URL)
        return parsed.hostname == storage.hostname and (parsed.port or 80) == (storage.port or 80)
    except Exception:
        return False


def _merge_headers(existing: Optional[Dict[str, str]]) -> Dict[str, str]:
    merged = dict(existing or {})
    for key, value in _storage_headers.get().items():
        merged.setdefault(key, value)
    return merged


def patch_storage_http_clients() -> None:
    """Patch httpx/requests once so legacy tools inherit Storage auth context."""
    global _patched
    if _patched:
        return
    _patched = True

    original_httpx_client_request = httpx.Client.request
    original_httpx_async_request = httpx.AsyncClient.request

    def sync_request(self, method, url, *args, **kwargs):
        if _is_storage_url(url):
            kwargs["headers"] = _merge_headers(kwargs.get("headers"))
        return original_httpx_client_request(self, method, url, *args, **kwargs)

    async def async_request(self, method, url, *args, **kwargs):
        if _is_storage_url(url):
            kwargs["headers"] = _merge_headers(kwargs.get("headers"))
        return await original_httpx_async_request(self, method, url, *args, **kwargs)

    httpx.Client.request = sync_request  # type: ignore[method-assign]
    httpx.AsyncClient.request = async_request  # type: ignore[method-assign]

    if requests is not None:
        original_requests_request = requests.sessions.Session.request

        def requests_request(self, method, url, *args, **kwargs):
            if _is_storage_url(url):
                kwargs["headers"] = _merge_headers(kwargs.get("headers"))
            return original_requests_request(self, method, url, *args, **kwargs)

        requests.sessions.Session.request = requests_request  # type: ignore[method-assign]

    logger.info("Storage HTTP clients patched for per-tool auth context")

