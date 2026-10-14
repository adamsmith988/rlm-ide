"""Generic per-user plugin preferences with dynamic per-namespace tables.

Each plugin declares one or more preference namespaces in its ``plugin.json``
(``user_prefs``). tool-runner registers them at startup via
``POST /plugin-prefs/namespaces/register`` and Storage creates a dedicated
table ``user_prefs_{namespace}`` on demand. All data tables share the same
shape so the read/write path is uniform:

    user_id TEXT PRIMARY KEY,
    prefs JSONB NOT NULL DEFAULT '{}',
    schema_version INT NOT NULL DEFAULT 1,
    updated_at TIMESTAMPTZ DEFAULT now()
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

_NAMESPACE_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)*$")
_SANITIZE_RE = re.compile(r"[^a-z0-9_]+")

def _registry_sql_candidates() -> tuple[Path, ...]:
    here = Path(__file__).resolve()
    candidates: list[Path] = [Path("/app/init-scripts/13-plugin-user-prefs-registry.sql")]
    for depth in range(2, 6):
        try:
            candidates.append(
                here.parents[depth] / "init-scripts" / "13-plugin-user-prefs-registry.sql"
            )
        except IndexError:
            break
    return tuple(candidates)


def safe_table_name(namespace: str) -> str:
    """Map a dotted namespace to a safe physical table name.

    ``map_explorer.layer_stack`` -> ``user_prefs_map_explorer_layer_stack``
    """
    cleaned = _SANITIZE_RE.sub("_", (namespace or "").strip().lower()).strip("_")
    if not cleaned:
        raise ValueError("invalid_namespace")
    return f"user_prefs_{cleaned}"


def _normalize_namespace(namespace: str) -> str:
    ns = (namespace or "").strip().lower()
    if not ns or not _NAMESPACE_RE.match(ns):
        raise ValueError("invalid_namespace")
    return ns


class PluginUserPrefsService:
    def __init__(self, storage: Any) -> None:
        self.storage = storage

    def _pool(self):
        pg = getattr(self.storage, "postgres", None)
        if not pg or not pg.pool:
            raise RuntimeError("PostgreSQL not available")
        return pg.pool

    # Mirrors init-scripts/13-plugin-user-prefs-registry.sql (fallback when file not mounted).
    _REGISTRY_DDL_FALLBACK = """
        CREATE TABLE IF NOT EXISTS plugin_pref_namespaces (
            namespace      TEXT PRIMARY KEY,
            plugin_id      TEXT NOT NULL,
            table_name     TEXT NOT NULL UNIQUE,
            schema_version INT  NOT NULL DEFAULT 1,
            metadata       JSONB DEFAULT '{}'::jsonb,
            registered_at  TIMESTAMPTZ DEFAULT now(),
            updated_at     TIMESTAMPTZ DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_plugin_pref_namespaces_plugin
            ON plugin_pref_namespaces(plugin_id);
    """

    @staticmethod
    def _registry_sql() -> str:
        for candidate in _registry_sql_candidates():
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8")
        return PluginUserPrefsService._REGISTRY_DDL_FALLBACK

    async def ensure_schema(self) -> None:
        """Create the stable registry table (idempotent)."""
        sql = self._registry_sql()
        async with self._pool().acquire() as conn:
            await conn.execute(sql)

    async def register_namespace(
        self,
        *,
        plugin_id: str,
        namespace: str,
        schema_version: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ns = _normalize_namespace(namespace)
        table = safe_table_name(ns)
        pid = (plugin_id or "").strip()
        if not pid:
            raise ValueError("invalid_plugin_id")
        version = int(schema_version or 1)

        async with self._pool().acquire() as conn:
            # Per-namespace data table (dynamic, identical shape everywhere).
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    user_id TEXT PRIMARY KEY,
                    prefs JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    schema_version INT NOT NULL DEFAULT 1,
                    updated_at TIMESTAMPTZ DEFAULT now()
                )
                """
            )
            await conn.execute(
                """
                INSERT INTO plugin_pref_namespaces
                    (namespace, plugin_id, table_name, schema_version, metadata, updated_at)
                VALUES ($1, $2, $3, $4, $5::jsonb, now())
                ON CONFLICT (namespace) DO UPDATE SET
                    plugin_id = EXCLUDED.plugin_id,
                    schema_version = EXCLUDED.schema_version,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                """,
                ns,
                pid,
                table,
                version,
                json.dumps(metadata or {}, ensure_ascii=False),
            )
        return {
            "namespace": ns,
            "plugin_id": pid,
            "table_name": table,
            "schema_version": version,
        }

    async def _resolve_table(self, namespace: str) -> Optional[str]:
        ns = _normalize_namespace(namespace)
        async with self._pool().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT table_name FROM plugin_pref_namespaces WHERE namespace = $1",
                ns,
            )
        return str(row["table_name"]) if row else None

    async def get_prefs(self, user_id: str, namespace: str) -> Dict[str, Any]:
        if not user_id:
            raise ValueError("missing_user_id")
        table = await self._resolve_table(namespace)
        if not table:
            raise KeyError("namespace_not_registered")
        async with self._pool().acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT prefs, schema_version, updated_at FROM {table} WHERE user_id = $1",
                user_id,
            )
        if not row:
            return {"prefs": {}, "schema_version": None, "updated_at": None}
        prefs = row["prefs"]
        if isinstance(prefs, str):
            try:
                prefs = json.loads(prefs)
            except json.JSONDecodeError:
                prefs = {}
        return {
            "prefs": prefs or {},
            "schema_version": row["schema_version"],
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        }

    async def put_prefs(
        self,
        user_id: str,
        namespace: str,
        prefs: Dict[str, Any],
        schema_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not user_id:
            raise ValueError("missing_user_id")
        if not isinstance(prefs, dict):
            raise ValueError("prefs_must_be_object")
        table = await self._resolve_table(namespace)
        if not table:
            raise KeyError("namespace_not_registered")
        version = int(schema_version) if schema_version is not None else 1
        async with self._pool().acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {table} (user_id, prefs, schema_version, updated_at)
                VALUES ($1, $2::jsonb, $3, now())
                ON CONFLICT (user_id) DO UPDATE SET
                    prefs = EXCLUDED.prefs,
                    schema_version = EXCLUDED.schema_version,
                    updated_at = now()
                """,
                user_id,
                json.dumps(prefs, ensure_ascii=False),
                version,
            )
        return await self.get_prefs(user_id, namespace)
