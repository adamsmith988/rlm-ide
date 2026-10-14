"""Direct llm-gateway target for the RLM streaming hot path (Phase 1).

Why
---
``RlmSession`` runs *inside* the tool-runner uvicorn process. Each streaming
turn used to POST to ``http://127.0.0.1:8002/internal/llm/chat/completions`` —
the same process serving UI/progress HTTP — which then re-opened a second
socket to llm-gateway and relayed the SSE bytes. Two hops, two connections,
one event loop doing both. This module lets the runner POST the SSE turn
straight to llm-gateway (``{LLM_PROXY_URL}/v1/chat/completions``).

What the tool-runner gateway did for us and how it is preserved
--------------------------------------------------------------
* **Binding resolve** (``model`` empty/``auto`` → logical_id): replicated with
  the same ``GET /binding/resolve?keys=…`` chain the gateway builds for
  ``X-AIv2-Tool-Id: plugin_agent_flow_run`` / ``X-AIv2-Plugin-Id: agent_flow``.
  Agent Flow does **not** forward ``X-AIv2-Route-Key`` — each Agent/Model node
  picks any catalog model; llm-gateway LB routes by that logical_id alone.
* **Modality validation** is not re-implemented: the RLM path is chat-only and
  llm-gateway rejects unknown / non-chat models on its own.

Anything that cannot be prepared directly (binding lookup failed, httpx
missing) returns ``None`` so the caller falls back to the loopback gateway and
keeps the exact same error surface (``503 binding_not_configured`` …).

Ordinary tools (``modules/tools/*``) and non-stream plugin calls are **not**
touched — they still bind through ``TOOL_LLM_GATEWAY_URL`` / ``tool_runner_base()``.
"""

from __future__ import annotations

import asyncio
import logging
import random
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore

_HTTP_PRECOMMIT_ERRORS = (
    (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
    if httpx is not None
    else ()
)

logger = logging.getLogger(__name__)

# Identity the plugin runner presents to tool-runner's gateway
# (see ``_shared.internal_llm_headers``); reused here to build the same chain.
RLM_TOOL_ID = "plugin_agent_flow_run"
RLM_PLUGIN_ID = "agent_flow"
CHAT_PURPOSE = "llm"

# Mirrors tool-runner ``internal_llm._AUTO_TOKENS``.
_AUTO_TOKENS = frozenset({"", "auto", "default"})

BINDING_RESOLVE_TIMEOUT_SEC = 10.0
HTTP_CONNECT_TIMEOUT_SEC = 10.0
HTTP_POOL_TIMEOUT_SEC = 10.0
HTTP_WRITE_TIMEOUT_SEC = 30.0
HTTP_MAX_CONNECTIONS = 100
HTTP_MAX_KEEPALIVE_CONNECTIONS = 20
HTTP_KEEPALIVE_EXPIRY_SEC = 30.0
HTTP_PRECOMMIT_ATTEMPTS = 3

_client = None
_client_lock: Optional[asyncio.Lock] = None
_client_idle: Optional[asyncio.Event] = None
_client_active = 0
_client_closing = False
_client_factory = None


def _lock() -> asyncio.Lock:
    global _client_lock
    if _client_lock is None:
        _client_lock = asyncio.Lock()
    return _client_lock


def _idle_event() -> asyncio.Event:
    global _client_idle
    if _client_idle is None:
        _client_idle = asyncio.Event()
        _client_idle.set()
    return _client_idle


def _timeout(read: float) -> "httpx.Timeout":
    return httpx.Timeout(
        connect=HTTP_CONNECT_TIMEOUT_SEC,
        read=read,
        write=HTTP_WRITE_TIMEOUT_SEC,
        pool=HTTP_POOL_TIMEOUT_SEC,
    )


def _new_client() -> "httpx.AsyncClient":
    return httpx.AsyncClient(
        timeout=_timeout(BINDING_RESOLVE_TIMEOUT_SEC),
        limits=httpx.Limits(
            max_connections=HTTP_MAX_CONNECTIONS,
            max_keepalive_connections=HTTP_MAX_KEEPALIVE_CONNECTIONS,
            keepalive_expiry=HTTP_KEEPALIVE_EXPIRY_SEC,
        ),
    )


@asynccontextmanager
async def shared_http_client():
    """Lease the process client; shutdown waits for all leases to finish."""
    global _client, _client_active, _client_factory
    while True:
        async with _lock():
            if not _client_closing:
                factory = httpx.AsyncClient
                if (
                    _client is not None
                    and _client_factory is not factory
                    and _client_active == 0
                ):
                    stale, _client = _client, None
                    close = getattr(stale, "aclose", None)
                    if close is not None and not getattr(stale, "is_closed", False):
                        await close()
                if _client is None or getattr(_client, "is_closed", False):
                    _client = _new_client()
                    _client_factory = factory
                _client_active += 1
                _idle_event().clear()
                client = _client
                break
        await asyncio.sleep(0)
    try:
        yield client
    finally:
        async with _lock():
            _client_active -= 1
            if _client_active == 0:
                _idle_event().set()


async def close_http_client() -> None:
    """Drain active calls, close the pool, and allow lazy recreation."""
    global _client, _client_closing, _client_factory
    async with _lock():
        _client_closing = True
        if _client_active == 0:
            _idle_event().set()
    await _idle_event().wait()
    async with _lock():
        client, _client = _client, None
        _client_factory = None
        _client_closing = False
    close = getattr(client, "aclose", None)
    if close is not None and not getattr(client, "is_closed", False):
        await close()


async def _get_with_retry(client: "httpx.AsyncClient", url: str, **kwargs: Any):
    for attempt in range(HTTP_PRECOMMIT_ATTEMPTS):
        try:
            return await client.get(url, **kwargs)
        except _HTTP_PRECOMMIT_ERRORS:
            if attempt + 1 >= HTTP_PRECOMMIT_ATTEMPTS:
                raise
            await asyncio.sleep(random.uniform(0.0, 0.1 * (2**attempt)))


def model_needs_binding(model: Any) -> bool:
    """True when the gateway would have rewritten ``model`` from a binding."""
    if model is None:
        return True
    return isinstance(model, str) and model.strip().lower() in _AUTO_TOKENS


def binding_chain(
    purpose: str = CHAT_PURPOSE,
    *,
    tool_id: Optional[str] = RLM_TOOL_ID,
    plugin_id: Optional[str] = RLM_PLUGIN_ID,
) -> List[str]:
    """Route keys for exactly ``purpose`` — same as tool-runner ``_build_binding_chain``."""
    suffix = str(purpose or "").strip()
    chain: List[str] = []
    if not suffix:
        return chain
    if tool_id:
        chain.append(f"tool.{tool_id}.{suffix}")
    if plugin_id:
        chain.append(f"plugin.{plugin_id}.{suffix}")
    return chain


def direct_chat_url(base: str) -> str:
    return f"{str(base or '').rstrip('/')}/v1/chat/completions"


def direct_binding_resolve_url(base: str) -> str:
    return f"{str(base or '').rstrip('/')}/binding/resolve"


def _internal_auth_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    try:
        from ....config import settings as _settings

        key = str(getattr(_settings, "INTERNAL_API_KEY", "") or "").strip()
        if key:
            headers["X-Internal-Key"] = key
    except Exception:
        pass
    return headers


def direct_chat_headers(
    *,
    route_key: Optional[str] = None,
    user_id: Optional[str] = None,
    case_id: Optional[Any] = None,
) -> Dict[str, str]:
    """Headers for llm-gateway ``/v1/chat/completions`` (SSE).

    Deliberately does **not** carry ``X-Tool-Runner-Internal-Secret`` — that
    secret authenticates callers *to tool-runner*; llm-gateway is a different
    service and must not receive it.
    """
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-AIv2-Tool-Id": RLM_TOOL_ID,
        "X-AIv2-Plugin-Id": RLM_PLUGIN_ID,
        **_internal_auth_headers(),
    }
    if route_key:
        headers["X-AIv2-Route-Key"] = str(route_key)
    if user_id:
        headers["X-User-Id"] = str(user_id)
    if case_id:
        headers["X-Case-Id"] = str(case_id)
    return headers


async def resolve_binding(
    base: str,
    *,
    chain: Optional[List[str]] = None,
    timeout: float = BINDING_RESOLVE_TIMEOUT_SEC,
) -> Optional[Tuple[str, Optional[str]]]:
    """``GET {base}/binding/resolve?keys=…`` → ``(model, route_key)`` or ``None``."""
    if httpx is None:
        return None
    keys = list(chain) if chain is not None else binding_chain()
    if not keys:
        return None
    url = direct_binding_resolve_url(base)
    try:
        async with shared_http_client() as client:
            resp = await _get_with_retry(
                client,
                url,
                params={"keys": ",".join(keys)},
                timeout=_timeout(timeout),
                headers=_internal_auth_headers(),
            )
    except Exception as exc:  # noqa: BLE001 — any failure → fall back to gateway
        logger.warning("rlm direct llm: binding resolve failed (%s): %s", keys, exc)
        return None
    if resp.status_code != 200:
        logger.info(
            "rlm direct llm: no binding matched (%s) status=%s — falling back to gateway",
            keys,
            resp.status_code,
        )
        return None
    try:
        data = resp.json() or {}
    except Exception:  # noqa: BLE001
        return None
    model = str((data.get("model") if isinstance(data, dict) else None) or "").strip()
    if not model:
        return None
    route_key = data.get("route_key") if isinstance(data, dict) else None
    return model, (str(route_key).strip() or None) if route_key else None


async def prepare_direct_chat(
    base: str,
    payload: Dict[str, Any],
    *,
    user_id: Optional[str] = None,
    case_id: Optional[Any] = None,
) -> Optional[Tuple[str, Dict[str, str], Dict[str, Any]]]:
    """Build ``(url, headers, payload)`` for a direct llm-gateway SSE call.

    Returns ``None`` when the call cannot be prepared (no base, httpx missing,
    binding lookup failed) so the caller can use the loopback gateway instead.
    ``payload`` is not mutated; a copy with ``model`` filled in is returned.
    """
    base = str(base or "").strip().rstrip("/")
    if not base or httpx is None:
        return None
    body = dict(payload or {})
    # Agent Flow never pins an authoritative route_key: users pick any catalog
    # model per node, and a missing route empties llm-proxy backends.
    if model_needs_binding(body.get("model")):
        resolved = await resolve_binding(base)
        if resolved is None:
            return None
        body["model"], _matched_route = resolved
        logger.info(
            "rlm direct llm: tool=%s plugin=%s purpose=%s → model=%s (auto fill, no route_key)",
            RLM_TOOL_ID,
            RLM_PLUGIN_ID,
            CHAT_PURPOSE,
            body["model"],
        )
    else:
        body["model"] = str(body.get("model")).strip()
    body.pop("route_key", None)
    headers = direct_chat_headers(route_key=None, user_id=user_id, case_id=case_id)
    return direct_chat_url(base), headers, body
