"""
Qdrant backend for vector embeddings.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..base import BaseBackend

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    QdrantClient = None
    Distance = None
    VectorParams = None
    PointStruct = None

log = logging.getLogger(__name__)


class QdrantBackend(BaseBackend):
    """
    Qdrant backend for:
    - Face embeddings
    - Voice embeddings
    - Document embeddings
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__()
        self.url = url
        self.api_key = api_key
        self.client = None

    async def connect(self) -> None:
        """Initialize Qdrant connection."""
        if self.url and QdrantClient and self.client is None:
            log.info("Connecting to Qdrant...")
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
            )
            self._connected = True
            log.info("✅ Qdrant connected")

    async def disconnect(self) -> None:
        """Close Qdrant connection."""
        if self.client is not None:
            self.client.close()
            self.client = None
            self._connected = False

    async def ensure_collection(
        self,
        name: str,
        dimension: int,
        distance: str = "Cosine",
    ) -> None:
        """Ensure collection exists."""
        if not self.client:
            return

        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]

        if name not in existing:
            dist_map = {
                "Cosine": Distance.COSINE,
                "Euclidean": Distance.EUCLID,
                "Dot": Distance.DOT,
            }
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=dist_map.get(distance, Distance.COSINE),
                ),
            )
            log.info(f"Created Qdrant collection: {name}")

    async def upsert_vectors(
        self,
        collection: str,
        vectors: List[Dict[str, Any]],
    ) -> int:
        """
        Upsert vectors into collection.
        
        Args:
            collection: Collection name
            vectors: List of dicts with 'id', 'vector', 'payload'
        
        Returns:
            Number of upserted vectors
        """
        if not self.client:
            return 0

        points = [
            PointStruct(
                id=v["id"],
                vector=v["vector"],
                payload=v.get("payload", {}),
            )
            for v in vectors
        ]

        self.client.upsert(
            collection_name=collection,
            points=points,
        )
        return len(points)

    async def search(
        self,
        collection: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            collection: Collection name
            query_vector: Query embedding
            limit: Max results
            score_threshold: Minimum score
            filter_dict: Qdrant filter conditions
        
        Returns:
            List of matches with id, score, payload
        """
        if not self.client:
            return []

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        qdrant_filter = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            qdrant_filter = Filter(must=conditions)

        # The cluster currently runs Qdrant 1.7.x. New qdrant-client versions
        # expose query_points(), but that calls /points/query which older
        # servers do not support. Prefer /points/search for compatibility.
        if hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
            )
        elif self.url:
            import httpx

            body: Dict[str, Any] = {
                "vector": query_vector,
                "limit": limit,
                "with_payload": True,
            }
            if score_threshold is not None:
                body["score_threshold"] = score_threshold
            if filter_dict:
                body["filter"] = {
                    "must": [
                        {"key": key, "match": {"value": value}}
                        for key, value in filter_dict.items()
                    ]
                }

            async with httpx.AsyncClient(timeout=60.0) as http:
                resp = await http.post(
                    f"{self.url.rstrip('/')}/collections/{collection}/points/search",
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()

            return [
                {
                    "id": r.get("id"),
                    "score": r.get("score"),
                    "payload": r.get("payload") or {},
                }
                for r in data.get("result", [])
            ]
        else:
            response = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            results = getattr(response, "points", response)

        return [
            {
                "id": r.id,
                "score": r.score,
                "payload": r.payload,
            }
            for r in results
        ]

    async def delete_vectors(
        self,
        collection: str,
        ids: List[str],
    ) -> int:
        """Delete vectors by IDs."""
        if not self.client:
            return 0

        self.client.delete(
            collection_name=collection,
            points_selector=ids,
        )
        return len(ids)

    async def scroll_artifact_all_collections(
        self,
        artifact_id: str,
        *,
        per_collection_limit: int = 100,
        skip_collections: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generic cross-collection discovery: scroll every collection and return
        points whose payload has `artifact_id == <aid>`.

        Returns:
            {
              "<collection>": {
                  "count": int,         # number of points returned (best-effort)
                  "truncated": bool,    # True if we hit the per_collection_limit
                  "points": [ {"id": ..., "payload": {...}}, ... ],
              },
              ...
            }
        """
        if not self.client or not artifact_id:
            return {}

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        skip = {s.lower() for s in (skip_collections or [])}
        per_collection_limit = max(1, min(int(per_collection_limit or 100), 1000))

        try:
            collections = self.client.get_collections().collections
        except Exception as exc:
            log.warning("scroll_artifact_all_collections: list failed: %s", exc)
            return {}

        flt = Filter(must=[
            FieldCondition(key="artifact_id", match=MatchValue(value=artifact_id))
        ])

        results: Dict[str, Dict[str, Any]] = {}
        for coll in collections:
            name = getattr(coll, "name", None)
            if not name or name.lower() in skip:
                continue
            try:
                points, _next = self.client.scroll(
                    collection_name=name,
                    scroll_filter=flt,
                    limit=per_collection_limit,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:
                log.debug(
                    "scroll_artifact_all_collections: %s skipped (%s)", name, exc
                )
                continue

            if not points:
                continue
            items = [
                {"id": getattr(p, "id", None), "payload": getattr(p, "payload", None) or {}}
                for p in points
            ]
            results[name] = {
                "count": len(items),
                "truncated": _next is not None,
                "points": items,
            }
        return results

    async def get_collection_info(self, collection: str) -> Optional[Dict[str, Any]]:
        """Get collection statistics."""
        if not self.client:
            return None

        try:
            info = self.client.get_collection(collection)
            return {
                "name": collection,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.name,
            }
        except Exception:
            return None

    async def count_by_payload_field(
        self, field: str, value: str
    ) -> Dict[str, int]:
        """Count points per collection where payload[field] equals value."""
        if not self.client or not value:
            return {}
        fname = (field or "").strip()
        if fname not in ("artifact_id", "case_id"):
            return {}

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        try:
            collections = self.client.get_collections().collections
        except Exception as exc:
            log.warning("count_by_payload_field list failed: %s", exc)
            return {}

        flt = Filter(must=[FieldCondition(key=fname, match=MatchValue(value=value))])
        counts: Dict[str, int] = {}
        for coll in collections:
            name = getattr(coll, "name", None)
            if not name:
                continue
            try:
                n = int(
                    self.client.count(
                        collection_name=name, count_filter=flt, exact=True
                    ).count
                    or 0
                )
            except Exception as exc:
                log.debug("count_by_payload_field %s skipped: %s", name, exc)
                continue
            if n > 0:
                counts[name] = n
        return counts

    async def purge_by_payload_field(self, field: str, value: str) -> Dict[str, int]:
        """Delete Qdrant points whose payload[field] equals value (all collections)."""
        if not self.client or not value:
            return {}
        fname = (field or "").strip()
        if fname not in ("artifact_id", "case_id"):
            return {}

        from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

        counts: Dict[str, int] = {}
        try:
            collections = self.client.get_collections().collections
        except Exception as exc:
            log.warning("purge_by_payload_field list failed: %s", exc)
            return {}

        flt = Filter(must=[FieldCondition(key=fname, match=MatchValue(value=value))])
        selector = FilterSelector(filter=flt)

        for coll in collections:
            name = getattr(coll, "name", None)
            if not name:
                continue
            try:
                before = self.client.count(collection_name=name, count_filter=flt).count
                if not before:
                    continue
                self.client.delete(collection_name=name, points_selector=selector)
                counts[name] = int(before)
            except Exception as exc:
                log.debug("purge_by_payload_field %s skipped: %s", name, exc)
        return counts
