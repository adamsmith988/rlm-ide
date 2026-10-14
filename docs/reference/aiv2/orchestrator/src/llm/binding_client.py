"""
Service-to-model binding client.

Resolves the logical model name from ``llm-gateway`` / Storage ``llm_routes``
so super-admins can rebind via ``PUT /admin/services/{service}/{purpose}/binding``
without redeploying consumers.

Resolution order:
    1. ``GET /binding/resolve?keys=service.<svc>.<purpose>``
    2. Cached previous value (if non-stale)
    3. Empty string when ``STRICT_LLM_BINDINGS`` and no binding (callers fail fast)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import httpx

from ..config import settings

log = logging.getLogger(__name__)


@dataclass
class _CachedBinding:
    model: str
    fetched_at: float


_TTL_SEC = 30.0
_AUTO_TOKENS = {"", "auto", "default", None}


class ModelBindingClient:
    """Async, in-process model binding resolver.

    Each call returns the model name to send in the ``model`` field to ``llm-gateway``.
    """

    def __init__(
        self,
        service: str,
        *,
        gateway_url: Optional[str] = None,
        timeout: float = 3.0,
    ) -> None:
        self.service = service
        self.gateway_url = (gateway_url or settings.LLM_PROXY_URL).rstrip("/")
        self.strict = bool(settings.STRICT_LLM_BINDINGS)
        self.timeout = timeout
        self._cache: dict[str, _CachedBinding] = {}
        self._unbound_optional: set[str] = set()

    def _binding_chain(self, purpose: str) -> List[str]:
        p = (purpose or "chat").strip()
        return [f"service.{self.service}.{p}"]

    async def _resolve_from_gateway(self, purpose: str) -> Optional[str]:
        chain = self._binding_chain(purpose)
        url = f"{self.gateway_url}/binding/resolve"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                headers = {}
                key = (settings.INTERNAL_API_KEY or "").strip()
                if key:
                    headers["X-Internal-Key"] = key
                resp = await client.get(
                    url, params={"keys": ",".join(chain)}, headers=headers or None
                )
            if resp.status_code == 200:
                data = resp.json() or {}
                model = str(data.get("model") or "").strip()
                if model and model.lower() not in _AUTO_TOKENS:
                    log.debug(
                        "binding resolved service=%s purpose=%s → %s (via %s)",
                        self.service,
                        purpose,
                        model,
                        data.get("route_key"),
                    )
                    return model
            elif resp.status_code != 404:
                log.warning("binding resolve %s returned %s", chain, resp.status_code)
        except Exception as exc:
            log.warning("binding resolve %s failed: %s", chain, exc)
        return None

    async def get_model(self, purpose: str, *, force_refresh: bool = False, optional: bool = False) -> str:
        """Return the logical model bound to ``(self.service, purpose)``.

        ``optional`` purposes that are unbound are reported once at INFO instead of ERROR.
        """
        purpose = (purpose or "chat").strip()

        if not force_refresh:
            cached = self._cache.get(purpose)
            if cached and (time.time() - cached.fetched_at) < _TTL_SEC:
                return cached.model

        model = await self._resolve_from_gateway(purpose)
        if model:
            self._cache[purpose] = _CachedBinding(model=model, fetched_at=time.time())
            return model

        cached = self._cache.get(purpose)
        if cached:
            log.warning(
                "binding stale cache used for service.%s.%s → %s",
                self.service,
                purpose,
                cached.model,
            )
            return cached.model

        if optional:
            if purpose not in self._unbound_optional:
                self._unbound_optional.add(purpose)
                log.info("optional binding service.%s.%s is not set; feature skipped", self.service, purpose)
            return ""
        log.error(
            "No llm_routes binding for service.%s.%s (tried %s). "
            "Super-admin: PUT /admin/services/%s/%s/binding",
            self.service,
            purpose,
            self._binding_chain(purpose),
            self.service,
            purpose,
        )
        return ""


_clients: dict[str, ModelBindingClient] = {}


def get_binding_client(service: str = "orchestrator") -> ModelBindingClient:
    """Return a process-wide cached client for the given consumer service."""
    if service not in _clients:
        _clients[service] = ModelBindingClient(service)
    return _clients[service]
