"""
Neo4j backend — generic Cypher query interface.

This backend provides raw Cypher execution capabilities.
It does NOT define domain-specific schema or queries;
callers (case-importer, orchestrator, etc.) decide what to store.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..base import BaseBackend

try:
    from neo4j import AsyncGraphDatabase
except ImportError:
    AsyncGraphDatabase = None

log = logging.getLogger(__name__)


class Neo4jBackend(BaseBackend):
    """
    Neo4j backend — generic graph database operations.

    Provides:
    - run_query()       → read Cypher (MATCH, CALL, ...)
    - run_write()       → single write Cypher (CREATE, MERGE, DELETE, ...)
    - run_write_batch() → multiple writes in one atomic transaction
    - health_check()    → connectivity check
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        super().__init__()
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None

    # ------------------------------------------------------------------ lifecycle

    async def connect(self) -> None:
        """Initialize Neo4j async driver and verify connectivity."""
        if not self.uri or AsyncGraphDatabase is None:
            return
        if self.driver is not None:
            return

        log.info("Connecting to Neo4j at %s ...", self.uri)
        self.driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=(self.username or "neo4j", self.password or ""),
        )
        await self.driver.verify_connectivity()
        self._connected = True
        log.info("✅ Neo4j connected")

    async def disconnect(self) -> None:
        """Close Neo4j driver."""
        if self.driver is not None:
            await self.driver.close()
            self.driver = None
            self._connected = False

    # ------------------------------------------------------------------ read

    async def run_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a read Cypher query, return records as list of dicts.

        Example:
            await backend.run_query(
                "MATCH (n:Entity {case_id: $cid}) RETURN n.name AS name",
                {"cid": "case_123"},
            )
        """
        if not self.driver:
            return []

        async with self.driver.session() as session:
            result = await session.run(query, parameters or {})
            return await result.data()

    # ------------------------------------------------------------------ write

    async def run_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a write Cypher query.
        Returns both records (if RETURN clause exists) and mutation counters.

        Example:
            await backend.run_write(
                "MERGE (n:Entity {name: $name, case_id: $cid}) "
                "SET n.type = $type RETURN n",
                {"name": "ALICE", "cid": "case_123", "type": "person"},
            )
        """
        if not self.driver:
            return {"records": [], "counters": {}}

        async with self.driver.session() as session:
            result = await session.run(query, parameters or {})
            records = [record.data() async for record in result]
            summary = await result.consume()
            c = summary.counters
            return {
                "records": records,
                "counters": {
                    "nodes_created": c.nodes_created,
                    "nodes_deleted": c.nodes_deleted,
                    "relationships_created": c.relationships_created,
                    "relationships_deleted": c.relationships_deleted,
                    "properties_set": c.properties_set,
                    "labels_added": c.labels_added,
                    "labels_removed": c.labels_removed,
                },
            }

    # ------------------------------------------------------------------ batch

    async def run_write_batch(
        self,
        queries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Execute multiple write queries in a single atomic transaction.
        All succeed or all fail.

        Args:
            queries: [{"query": "MERGE ...", "parameters": {...}}, ...]

        Returns:
            {"count": N, "counters": {nodes_created: ...}}
        """
        if not self.driver:
            return {"count": 0, "counters": {}}

        totals = {
            "nodes_created": 0,
            "nodes_deleted": 0,
            "relationships_created": 0,
            "relationships_deleted": 0,
            "properties_set": 0,
            "labels_added": 0,
            "labels_removed": 0,
        }

        async with self.driver.session() as session:
            tx = await session.begin_transaction()
            try:
                for item in queries:
                    result = await tx.run(
                        item["query"], item.get("parameters", {})
                    )
                    summary = await result.consume()
                    c = summary.counters
                    for key in totals:
                        totals[key] += getattr(c, key, 0)
                await tx.commit()
            except Exception:
                await tx.rollback()
                raise

        return {"count": len(queries), "counters": totals}

    # ------------------------------------------------------------------ generic by-artifact discovery

    async def find_artifact_subgraph(
        self,
        artifact_id: str,
        *,
        max_relationships: int = 200,
    ) -> Dict[str, Any]:
        """
        Return all nodes whose `artifact_id` property equals the given id, plus
        any direct relationships (1-hop) to other nodes.

        Output:
            {
              "nodes":         [ { "labels": [...], "props": {...} }, ... ],
              "relationships": [ {
                   "from_labels": [...], "from_artifact_id": "...",
                   "rel_type": "...", "rel_props": {...},
                   "to_labels":   [...], "to_props": {...}
              }, ... ],
            }

        Notes:
          - Generic by design — no domain knowledge of node labels needed.
          - Limited via `max_relationships` to keep payloads bounded.
        """
        if not self.driver or not artifact_id:
            return {"nodes": [], "relationships": []}

        max_relationships = max(1, min(int(max_relationships or 200), 2000))

        nodes_q = (
            "MATCH (n {artifact_id: $aid}) "
            "RETURN labels(n) AS labels, properties(n) AS props"
        )
        rels_q = (
            "MATCH (n {artifact_id: $aid})-[r]-(m) "
            "RETURN labels(n) AS from_labels, "
            "       n.artifact_id AS from_artifact_id, "
            "       type(r) AS rel_type, "
            "       properties(r) AS rel_props, "
            "       labels(m) AS to_labels, "
            "       properties(m) AS to_props "
            "LIMIT $lim"
        )

        nodes: List[Dict[str, Any]] = []
        rels: List[Dict[str, Any]] = []
        try:
            async with self.driver.session() as session:
                node_res = await session.run(nodes_q, {"aid": artifact_id})
                async for rec in node_res:
                    nodes.append({
                        "labels": list(rec.get("labels") or []),
                        "props": dict(rec.get("props") or {}),
                    })

                rel_res = await session.run(
                    rels_q, {"aid": artifact_id, "lim": max_relationships}
                )
                async for rec in rel_res:
                    rels.append({
                        "from_labels": list(rec.get("from_labels") or []),
                        "from_artifact_id": rec.get("from_artifact_id"),
                        "rel_type": rec.get("rel_type"),
                        "rel_props": dict(rec.get("rel_props") or {}),
                        "to_labels": list(rec.get("to_labels") or []),
                        "to_props": dict(rec.get("to_props") or {}),
                    })
        except Exception as exc:
            log.debug("find_artifact_subgraph failed: %s", exc)
            return {"nodes": nodes, "relationships": rels, "error": str(exc)}

        return {"nodes": nodes, "relationships": rels}

    async def purge_by_property(self, prop: str, value: str) -> Dict[str, Any]:
        """DETACH DELETE all nodes where n[prop] == value."""
        if not self.driver or not value:
            return {"nodes_deleted": 0, "relationships_deleted": 0}
        pname = (prop or "").strip()
        if pname not in ("artifact_id", "case_id"):
            return {"error": "invalid_property"}

        query = f"MATCH (n) WHERE n.{pname} = $val DETACH DELETE n"
        result = await self.run_write(query, {"val": value})
        return result.get("counters") or {}

    # ------------------------------------------------------------------ health

    async def health_check(self) -> bool:
        """Check Neo4j connectivity."""
        if not self.driver:
            return False
        try:
            await self.driver.verify_connectivity()
            return True
        except Exception:
            return False
