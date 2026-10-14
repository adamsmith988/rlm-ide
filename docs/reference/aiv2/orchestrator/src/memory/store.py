"""Thin async Qdrant REST store for chat memory.

REST (not qdrant-client) keeps us independent of client/server version skew.
One collection per (prefix, embed model, dimension) so switching the embed
binding never mixes vector spaces; management ops span every prefix match.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from ..config import settings

log = logging.getLogger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def match(key: str, value: Any) -> Dict[str, Any]:
    return {"key": key, "match": {"value": value}}


def build_filter(*, must: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> Dict[str, Any]:
    conds = list(must or [])
    if ids:
        conds.append({"has_id": list(ids)})
    return {"must": conds}


def _indexed_fields() -> List[str]:
    return [f.strip() for f in str(settings.MEMORY_INDEXED_FIELDS or "").split(",") if f.strip()]


def _optimizers_config() -> Dict[str, Any]:
    segments = int(settings.MEMORY_QDRANT_SEGMENTS or 0)
    return {"default_segment_number": segments} if segments > 0 else {}


class QdrantMemoryStore:
    def __init__(self, base_url: str, *, prefix: str, api_key: Optional[str] = None, timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.prefix = _UNSAFE.sub("_", prefix.strip() or "memory")
        self._headers = {"api-key": api_key} if api_key else {}
        self._timeout = timeout
        self._ready: set[str] = set()

    def collection_for(self, model: str, dim: int) -> str:
        slug = _UNSAFE.sub("_", (model or "embed").strip()).strip("_").lower() or "embed"
        return f"{self.prefix}__{slug}__{int(dim)}"

    async def _req(self, method: str, path: str, json: Any = None, *, ok404: bool = False) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.request(method, f"{self.base_url}{path}", json=json, headers=self._headers)
        if ok404 and resp.status_code == 404:
            return {}
        if resp.status_code >= 400:
            raise RuntimeError(f"qdrant {method} {path} -> {resp.status_code}: {resp.text[:300]}")
        return resp.json() if resp.content else {}

    async def collections(self) -> List[str]:
        data = await self._req("GET", "/collections")
        names = [c.get("name", "") for c in ((data.get("result") or {}).get("collections") or [])]
        return sorted(n for n in names if n.startswith(f"{self.prefix}__"))

    async def ensure(self, name: str, dim: int) -> None:
        if name in self._ready:
            return
        info = await self._req("GET", f"/collections/{name}", ok404=True)
        optimizers = _optimizers_config()
        if not info:
            body: Dict[str, Any] = {"vectors": {"size": int(dim), "distance": "Cosine"}}
            if optimizers:
                body["optimizers_config"] = optimizers
            await self._req("PUT", f"/collections/{name}", body)
            for key in _indexed_fields():
                await self._req(
                    "PUT", f"/collections/{name}/index?wait=true",
                    {"field_name": key, "field_schema": "keyword"},
                )
            log.info("memory: created Qdrant collection %s (dim=%d)", name, dim)
        elif optimizers:
            current = ((info.get("result") or {}).get("config") or {}).get("optimizer_config") or {}
            if any(current.get(k) != v for k, v in optimizers.items()):
                await self._req("PATCH", f"/collections/{name}", {"optimizers_config": optimizers})
                log.info("memory: updated Qdrant collection %s optimizers %s", name, optimizers)
        self._ready.add(name)

    async def upsert(self, name: str, points: List[Dict[str, Any]]) -> None:
        if points:
            await self._req("PUT", f"/collections/{name}/points?wait=true", {"points": points})

    async def search(
        self, name: str, vector: List[float], *, flt: Dict[str, Any], limit: int,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        body: Dict[str, Any] = {"vector": vector, "filter": flt, "limit": int(limit), "with_payload": True}
        if score_threshold is not None:
            body["score_threshold"] = float(score_threshold)
        data = await self._req("POST", f"/collections/{name}/points/search", body, ok404=True)
        return list(data.get("result") or [])

    async def scroll(self, name: str, *, flt: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        offset: Any = None
        while len(out) < limit:
            body: Dict[str, Any] = {
                "filter": flt, "limit": min(max(1, int(settings.MEMORY_SCROLL_PAGE)), limit - len(out)),
                "with_payload": True, "with_vector": False,
            }
            if offset is not None:
                body["offset"] = offset
            data = await self._req("POST", f"/collections/{name}/points/scroll", body, ok404=True)
            res = data.get("result") or {}
            out.extend(res.get("points") or [])
            offset = res.get("next_page_offset")
            if offset is None:
                break
        return out

    async def count(self, name: str, *, flt: Dict[str, Any]) -> int:
        data = await self._req("POST", f"/collections/{name}/points/count", {"filter": flt, "exact": True}, ok404=True)
        return int(((data.get("result") or {}).get("count")) or 0)

    async def delete(self, name: str, *, flt: Dict[str, Any]) -> None:
        await self._req("POST", f"/collections/{name}/points/delete?wait=true", {"filter": flt}, ok404=True)
