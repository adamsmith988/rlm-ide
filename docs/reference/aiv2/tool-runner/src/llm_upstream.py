"""
Upstream helpers for tool-runner → llm-proxy.

Queue vs retry (do not conflate):

  **Queue** — request is valid and accepted; caller waits for its turn.
  llm-proxy ConcurrencyLimiter holds the HTTP connection and admits the
  request when a slot opens. tool-runner participates by issuing *one* forward
  with a long read timeout (LLM_FORWARD_TIMEOUT_SEC) and keeping the
  connection open. No HTTP 503 retry loop — that would restart queue position.

  **Retry** — infrastructure blip before a response (connect refused, TCP
  reset, gateway not up yet). A *new* connection may succeed; cap attempts low.

Catalog/model validation uses an in-process OK cache so parallel subagents do
not hammer GET /catalog on every turn (that is not queue semantics either).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import httpx

from .config import settings

log = logging.getLogger(__name__)

_validation_ok_cache: Dict[Tuple[str, str, str], float] = {}


def _validation_cache_ttl() -> float:
    return float(getattr(settings, "LLM_VALIDATION_CACHE_TTL_SEC", 120.0) or 120.0)


def _infra_retry_attempts() -> int:
    return max(1, int(getattr(settings, "LLM_INFRA_RETRY_MAX_ATTEMPTS", 3) or 3))


def _infra_retry_delay_sec() -> float:
    return float(getattr(settings, "LLM_INFRA_RETRY_DELAY_SEC", 1.0) or 1.0)


def _validation_cache_key(
    model: str,
    purpose: str,
    *,
    tool_id: Optional[str],
) -> Tuple[str, str, str]:
    return (model.strip(), purpose.strip(), (tool_id or "").strip())


def mark_model_validated(
    model: str,
    purpose: str,
    *,
    tool_id: Optional[str] = None,
) -> None:
    key = _validation_cache_key(model, purpose, tool_id=tool_id)
    _validation_ok_cache[key] = time.monotonic() + _validation_cache_ttl()


def is_model_recently_validated(
    model: str,
    purpose: str,
    *,
    tool_id: Optional[str] = None,
) -> bool:
    key = _validation_cache_key(model, purpose, tool_id=tool_id)
    exp = _validation_ok_cache.get(key)
    if exp is None:
        return False
    if time.monotonic() >= exp:
        _validation_ok_cache.pop(key, None)
        return False
    return True


def reset_validation_cache_for_tests() -> None:
    _validation_ok_cache.clear()


def is_infra_connect_error(exc: BaseException) -> bool:
    """True only for failures before an HTTP response (worth a reconnect retry)."""
    return isinstance(exc, httpx.ConnectError)


def infra_retry_attempts() -> int:
    return _infra_retry_attempts()


def infra_retry_delay_sec() -> float:
    return _infra_retry_delay_sec()


async def validate_model_cached(
    validate_fn,
    model: str,
    purpose: str,
    *,
    tool_id: Optional[str] = None,
    plugin_id: Optional[str] = None,
    binding_constraints: Optional[Dict[str, Any]] = None,
    tool_profile: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Single validation pass; skip if recently OK for this model/purpose."""
    if is_model_recently_validated(model, purpose, tool_id=tool_id):
        return None
    err = await validate_fn(
        model,
        purpose,
        tool_id=tool_id,
        binding_constraints=binding_constraints,
        tool_profile=tool_profile,
    )
    if err is None:
        mark_model_validated(model, purpose, tool_id=tool_id)
    return err


async def upstream_post(
    client: httpx.AsyncClient,
    url: str,
    *,
    content: Optional[bytes] = None,
    headers: Optional[Mapping[str, str]] = None,
    data: Optional[Mapping[str, Any]] = None,
    files: Any = None,
    log_label: str = "upstream",
) -> httpx.Response:
    """One POST; long client timeout covers llm-proxy queue wait. Retry connect only."""
    hdrs = dict(headers or {})
    attempts = _infra_retry_attempts()
    delay = _infra_retry_delay_sec()
    last_exc: Optional[BaseException] = None

    for attempt in range(attempts):
        try:
            return await client.post(
                url,
                content=content,
                headers=hdrs or None,
                data=data,
                files=files,
            )
        except httpx.ConnectError as exc:
            last_exc = exc
            if attempt + 1 >= attempts:
                raise
            log.warning(
                "internal_llm %s: connect failed (%s); infra retry %d/%d in %.1fs",
                log_label,
                exc,
                attempt + 1,
                attempts,
                delay,
            )
            await asyncio.sleep(delay)

    assert last_exc is not None
    raise last_exc


async def upstream_get(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
    headers: Optional[Mapping[str, str]] = None,
    log_label: str = "upstream",
) -> httpx.Response:
    """One GET; retry only on connect failure (not on HTTP 503 body)."""
    hdrs = dict(headers or {})
    attempts = _infra_retry_attempts()
    delay = _infra_retry_delay_sec()
    last_exc: Optional[BaseException] = None

    for attempt in range(attempts):
        try:
            return await client.get(url, params=params, headers=hdrs or None)
        except httpx.ConnectError as exc:
            last_exc = exc
            if attempt + 1 >= attempts:
                raise
            log.warning(
                "internal_llm %s: GET connect failed; infra retry %d/%d",
                log_label,
                attempt + 1,
                attempts,
            )
            await asyncio.sleep(delay)

    assert last_exc is not None
    raise last_exc
