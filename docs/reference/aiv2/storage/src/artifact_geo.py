"""Upsert artifact_geo rows from file_meta / image_meta tool_outputs."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .case_importer_support import tool_id_lookup_variants
from .map_geo import _extract_location

_GEO_TOOLS = set()
for _t in ("file_meta", "image_meta"):
    _GEO_TOOLS.update(tool_id_lookup_variants(_t))


async def ensure_artifact_geo_schema(pool) -> None:
    from pathlib import Path

    schema_path = Path(__file__).resolve().parents[3] / "init-scripts" / "08-artifact-geo-schema.sql"
    if not schema_path.is_file():
        return
    sql = schema_path.read_text(encoding="utf-8")
    async with pool.acquire() as conn:
        await conn.execute(sql)


async def upsert_artifact_geo_from_doc(doc: Dict[str, Any], storage: Any) -> bool:
    tool_name = str(doc.get("tool_name") or "")
    if tool_name not in _GEO_TOOLS:
        return False
    loc = _extract_location(doc.get("data"))
    if not loc:
        return False

    artifact_id = str(doc.get("artifact_id") or "")
    if not artifact_id:
        return False

    pg = getattr(storage, "postgres", None)
    if not pg or not pg.pool:
        return False

    art: Dict[str, Any] = {}
    try:
        art = await storage.artifact_get(artifact_id) or {}
    except Exception:
        pass

    async with pg.pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO artifact_geo
            (artifact_id, user_id, case_id, latitude, longitude, location_source,
             filename, mime_type, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, now())
            ON CONFLICT (artifact_id) DO UPDATE SET
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude,
                location_source = EXCLUDED.location_source,
                filename = EXCLUDED.filename,
                mime_type = EXCLUDED.mime_type,
                case_id = EXCLUDED.case_id,
                updated_at = now()
            """,
            artifact_id,
            art.get("user_id") or doc.get("user_id"),
            doc.get("case_id") or art.get("case_id"),
            float(loc["latitude"]),
            float(loc["longitude"]),
            loc.get("source"),
            art.get("filename") or art.get("original_filename"),
            art.get("mime_type"),
            art.get("created_at"),
        )
    return True


async def upsert_artifact_geo_batch(docs: List[Dict[str, Any]], storage: Any) -> int:
    count = 0
    for doc in docs:
        if await upsert_artifact_geo_from_doc(doc, storage):
            count += 1
    return count
