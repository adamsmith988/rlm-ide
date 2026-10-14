"""
MongoDB backend for unstructured data and tool outputs.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from ..base import BaseBackend, now_iso

try:
    from motor.motor_asyncio import AsyncIOMotorClient
except ImportError:
    AsyncIOMotorClient = None

log = logging.getLogger(__name__)


class MongoBackend(BaseBackend):
    """
    MongoDB backend for:
    - Tool outputs (raw/unstructured)
    - Observations and insights
    - Review bundles
    - Session data
    """

    def __init__(self, mongo_url: Optional[str] = None, db_name: str = "localai"):
        super().__init__()
        self.mongo_url = mongo_url
        self.db_name = db_name
        self.client = None
        self.db = None

    async def connect(self) -> None:
        """Initialize MongoDB connection."""
        if self.mongo_url and AsyncIOMotorClient and self.client is None:
            log.info("Connecting to MongoDB...")
            self.client = AsyncIOMotorClient(self.mongo_url)
            self.db = self.client[self.db_name]
            self._connected = True
            log.info("✅ MongoDB connected")

    async def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self.client is not None:
            self.client.close()
            self.client = None
            self.db = None
            self._connected = False

    # ------------------------------------------------------------------ Tool Outputs
    async def append_tool_output(
        self,
        artifact_id: str,
        tool_id: str,
        output: Dict[str, Any],
    ) -> None:
        """Append tool output to artifact document."""
        if self.db is None:
            return

        await self.db.tool_outputs.update_one(
            {"artifact_id": artifact_id},
            {
                "$push": {
                    "outputs": {
                        "tool_id": tool_id,
                        "output": output,
                        "created_at": now_iso(),
                    }
                },
                "$setOnInsert": {"artifact_id": artifact_id},
            },
            upsert=True,
        )

    async def get_tool_outputs(self, artifact_id: str) -> List[Dict[str, Any]]:
        """Get all tool outputs for an artifact."""
        if self.db is None:
            return []

        doc = await self.db.tool_outputs.find_one({"artifact_id": artifact_id})
        return doc.get("outputs", []) if doc else []

    async def search_tool_outputs_text(
        self,
        query: str,
        *,
        tool_name: Optional[str] = None,
        channel: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Substring search over persisted vision/tool text (Mongo tool_outputs)."""
        if self.db is None:
            return []
        needle = (query or "").strip()
        if not needle:
            return []
        limit = max(1, min(int(limit or 50), 200))
        escaped = re.escape(needle)
        rx = {"$regex": escaped, "$options": "i"}
        clause: Dict[str, Any] = {
            "$or": [
                {"data.visual_description": rx},
                {"data.extracted_text": rx},
                {"data.description_text": rx},
                {"data.text": rx},
                {"data.caption": rx},
                {"data.items.text": rx},
            ]
        }
        if tool_name:
            clause["tool_name"] = str(tool_name)
        if channel:
            clause["channel"] = str(channel)
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        try:
            cursor = self.db.tool_outputs.find(clause).limit(limit * 4)
            async for doc in cursor:
                aid = str(doc.get("artifact_id") or "").strip()
                if not aid or aid in seen:
                    continue
                seen.add(aid)
                data = doc.get("data") if isinstance(doc.get("data"), dict) else {}
                blob = ""
                for key in (
                    "visual_description",
                    "extracted_text",
                    "description_text",
                    "text",
                    "caption",
                ):
                    val = data.get(key)
                    if isinstance(val, str) and val.strip():
                        blob = val
                        if needle.lower() in val.lower():
                            break
                if not blob:
                    for item in data.get("items") or []:
                        if isinstance(item, dict) and isinstance(item.get("text"), str):
                            blob = item["text"]
                            if needle.lower() in blob.lower():
                                break
                idx = blob.lower().find(needle.lower()) if blob else -1
                if idx >= 0:
                    start = max(0, idx - 24)
                    end = min(len(blob), idx + len(needle) + 140)
                    snippet = blob[start:end].strip()
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(blob):
                        snippet = snippet + "..."
                else:
                    snippet = (blob or "")[:180]
                out.append(
                    {
                        "artifact_id": aid,
                        "tool_name": doc.get("tool_name"),
                        "channel": doc.get("channel"),
                        "file_name": doc.get("file_name") or doc.get("filename"),
                        "file_path": doc.get("file_path"),
                        "snippet": snippet,
                        "match": "description",
                    }
                )
                if len(out) >= limit:
                    break
        except Exception as exc:
            log.warning("search_tool_outputs_text failed: %s", exc)
            return []
        return out

    # ------------------------------------------------------------------ Generic by-artifact discovery
    async def find_artifact_in_all_collections(
        self,
        artifact_id: str,
        *,
        per_collection_limit: int = 200,
        skip_collections: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generic cross-collection discovery: scan every collection and return any
        document that has `artifact_id == <aid>` (top-level field).

        Returns:
            {
              "<coll_name>": {
                  "count": int,
                  "returned": int,
                  "truncated": bool,
                  "docs": [ {...}, ... ],
              },
              ...
            }
        """
        if self.db is None or not artifact_id:
            return {}

        skip = {s.lower() for s in (skip_collections or [])}
        per_collection_limit = max(1, min(int(per_collection_limit or 200), 2000))

        try:
            coll_names = await self.db.list_collection_names()
        except Exception as exc:
            log.warning("find_artifact_in_all_collections: list failed: %s", exc)
            return {}

        results: Dict[str, Dict[str, Any]] = {}
        for name in coll_names:
            if not name or name.startswith("system."):
                continue
            if name.lower() in skip:
                continue

            try:
                count = await self.db[name].count_documents(
                    {"artifact_id": artifact_id}
                )
                if count <= 0:
                    continue
                cursor = self.db[name].find(
                    {"artifact_id": artifact_id}
                ).limit(per_collection_limit)
                docs: List[Dict[str, Any]] = []
                async for d in cursor:
                    item = dict(d)
                    if "_id" in item:
                        item["_id"] = str(item["_id"])
                    docs.append(item)
            except Exception as exc:
                log.debug(
                    "find_artifact_in_all_collections: %s skipped (%s)", name, exc
                )
                continue

            results[name] = {
                "count": count,
                "returned": len(docs),
                "truncated": count > len(docs),
                "docs": docs,
            }
        return results

    async def purge_by_field(self, field: str, value: str) -> Dict[str, int]:
        """Delete documents where top-level `field` equals value in every collection."""
        if self.db is None or not value:
            return {}
        fname = (field or "").strip()
        if fname not in ("artifact_id", "case_id"):
            return {}

        counts: Dict[str, int] = {}
        try:
            coll_names = await self.db.list_collection_names()
        except Exception as exc:
            log.warning("purge_by_field list failed: %s", exc)
            return {}

        for name in coll_names:
            if not name or name.startswith("system."):
                continue
            try:
                res = await self.db[name].delete_many({fname: value})
                deleted = int(getattr(res, "deleted_count", 0) or 0)
                if deleted > 0:
                    counts[name] = deleted
            except Exception as exc:
                log.debug("purge_by_field %s skipped: %s", name, exc)
        return counts

    # ------------------------------------------------------------------ Observations
    async def save_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Save an observation document."""
        if self.db is None:
            return observation

        observation["created_at"] = now_iso()
        result = await self.db.observations.insert_one(observation)
        observation["_id"] = str(result.inserted_id)
        return observation

    async def list_observations(
        self,
        person_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List observations, optionally filtered by person."""
        if self.db is None:
            return []

        query = {"person_id": person_id} if person_id else {}
        cursor = self.db.observations.find(query).sort("created_at", -1).limit(limit)
        
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results

    # ------------------------------------------------------------------ Insights
    async def save_insight(self, insight: Dict[str, Any]) -> Dict[str, Any]:
        """Save an insight document."""
        if self.db is None:
            return insight

        insight["created_at"] = now_iso()
        result = await self.db.insights.insert_one(insight)
        insight["_id"] = str(result.inserted_id)
        return insight

    async def list_insights(
        self,
        person_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List insights, optionally filtered by person."""
        if self.db is None:
            return []

        query = {"person_id": person_id} if person_id else {}
        cursor = self.db.insights.find(query).sort("created_at", -1).limit(limit)
        
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results

    # ------------------------------------------------------------------ Review Bundles
    async def create_review_bundle(
        self,
        bundle_id: str,
        session_id: str,
        trace_id: str,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Create a review bundle."""
        if self.db is None:
            return {"bundle_id": bundle_id}

        doc = {
            "bundle_id": bundle_id,
            "session_id": session_id,
            "trace_id": trace_id,
            "items": items,
            "status": "pending",
            "created_at": now_iso(),
        }
        await self.db.review_bundles.insert_one(doc)
        return {"bundle_id": bundle_id}

    async def get_review_bundle(self, bundle_id: str) -> Optional[Dict[str, Any]]:
        """Get review bundle by ID."""
        if self.db is None:
            return None

        doc = await self.db.review_bundles.find_one({"bundle_id": bundle_id})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    async def update_review_bundle(
        self,
        bundle_id: str,
        *,
        items: Optional[List[Dict[str, Any]]] = None,
        status: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update review bundle."""
        if self.db is None:
            return None

        updates = {"updated_at": now_iso()}
        if items is not None:
            updates["items"] = items
        if status is not None:
            updates["status"] = status

        await self.db.review_bundles.update_one(
            {"bundle_id": bundle_id},
            {"$set": updates},
        )
        return await self.get_review_bundle(bundle_id)

    # ------------------------------------------------------------------ Generic Results
    async def write_result(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Write a generic result document."""
        if self.db is None:
            return payload

        payload["created_at"] = now_iso()
        result = await self.db.results.insert_one(payload)
        payload["_id"] = str(result.inserted_id)
        return payload

    async def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Get result by ID."""
        if self.db is None:
            return None

        from bson import ObjectId
        try:
            doc = await self.db.results.find_one({"_id": ObjectId(result_id)})
            if doc:
                doc["_id"] = str(doc["_id"])
            return doc
        except Exception:
            return None
