"""Embeddings through the llm-gateway, model resolved from the orchestrator binding."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Tuple

import httpx

from ..config import llm_gateway_v1_url, settings


def _retryable_statuses() -> set:
    return {int(s) for s in str(settings.MEMORY_EMBED_RETRY_STATUSES or "").replace(" ", "").split(",") if s.isdigit()}


class EmbeddingUnavailable(RuntimeError):
    """No embed binding, or the bound backend failed."""


def _binding() -> Tuple[str, str]:
    return (
        (settings.MEMORY_BINDING_SERVICE or "orchestrator").strip(),
        (settings.MEMORY_EMBED_PURPOSE or "embed").strip(),
    )


def _headers() -> Dict[str, str]:
    service, purpose = _binding()
    key = (settings.INTERNAL_API_KEY or "").strip()
    headers = {"Authorization": f"Bearer {key}", "X-Internal-Key": key} if key else {"Authorization": "Bearer not-needed"}
    headers["X-AIv2-Route-Key"] = f"service.{service}.{purpose}"
    if (settings.MEMORY_EMBED_PRIORITY or "").strip():
        headers["X-AIv2-Priority"] = settings.MEMORY_EMBED_PRIORITY.strip()
    return headers


async def bound_embed_model() -> str:
    from ..llm.binding_client import get_binding_client

    service, purpose = _binding()
    model = (await get_binding_client(service).get_model(purpose) or "").strip()
    if not model:
        raise EmbeddingUnavailable(f"no model bound to service.{service}.{purpose}")
    return model


def _retry_wait(resp: httpx.Response, attempt: int) -> float:
    cap = float(settings.MEMORY_EMBED_RETRY_MAX_WAIT_S or 5.0)
    try:
        hinted = float(resp.headers.get("retry-after") or 0)
    except ValueError:
        hinted = 0.0
    return min(cap, hinted or float(settings.MEMORY_EMBED_RETRY_BASE_S) * (2 ** attempt))


async def post_embeddings(client: httpx.AsyncClient, url: str, payload: Dict, headers: Dict[str, str]) -> httpx.Response:
    """POST an embeddings request, waiting out gateway rate limits / transient 5xx."""
    retries = max(0, int(settings.MEMORY_EMBED_RETRIES or 0))
    for attempt in range(retries + 1):
        try:
            resp = await client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            if attempt >= retries:
                raise EmbeddingUnavailable(f"embedding request failed: {exc}") from exc
            await asyncio.sleep(min(float(settings.MEMORY_EMBED_RETRY_MAX_WAIT_S or 5.0), 0.5 * (2 ** attempt)))
            continue
        if resp.status_code in _retryable_statuses() and attempt < retries:
            await asyncio.sleep(_retry_wait(resp, attempt))
            continue
        if resp.status_code >= 400:
            raise EmbeddingUnavailable(f"embedding request failed: HTTP {resp.status_code}: {resp.text[:200]}")
        return resp
    raise EmbeddingUnavailable("embedding request failed: retries exhausted")


async def _post(client: httpx.AsyncClient, model: str, part: List[str]) -> httpx.Response:
    return await post_embeddings(
        client, f"{llm_gateway_v1_url()}/embeddings", {"model": model, "input": part}, _headers(),
    )


async def embed_texts(texts: List[str]) -> Tuple[str, List[List[float]]]:
    """Return ``(model, vectors)`` aligned with ``texts``."""
    model = await bound_embed_model()
    if not texts:
        return model, []
    batch = max(1, int(settings.MEMORY_EMBED_BATCH or 32))
    vectors: List[List[float]] = []
    async with httpx.AsyncClient(timeout=float(settings.MEMORY_TIMEOUT_S or 20.0)) as client:
        for i in range(0, len(texts), batch):
            part = texts[i : i + batch]
            resp = await _post(client, model, part)
            rows = sorted((resp.json() or {}).get("data") or [], key=lambda r: r.get("index", 0))
            if len(rows) != len(part):
                raise EmbeddingUnavailable(f"embedding count mismatch: {len(rows)} != {len(part)}")
            vectors.extend(list(r.get("embedding") or []) for r in rows)
    if any(not v for v in vectors):
        raise EmbeddingUnavailable("empty embedding vector")
    return model, vectors
