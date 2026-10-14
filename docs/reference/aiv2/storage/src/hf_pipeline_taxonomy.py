"""HF pipeline_tag taxonomy tables (versioned PIPELINE_DATA snapshot)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_FALLBACK_MODALITIES = (
    {"id": "text", "label": "Text", "label_fa": "متن"},
    {"id": "image", "label": "Image", "label_fa": "تصویر"},
    {"id": "audio", "label": "Audio", "label_fa": "صوت"},
    {"id": "video", "label": "Video", "label_fa": "ویدیو"},
    {"id": "embedding", "label": "Embedding", "label_fa": "بردار / امبدینگ"},
    {"id": "3d", "label": "3D", "label_fa": "سه‌بعدی"},
    {"id": "tabular", "label": "Tabular", "label_fa": "جدولی"},
    {"id": "time-series", "label": "Time series", "label_fa": "سری زمانی"},
    {"id": "mask", "label": "Mask / segmentation", "label_fa": "ماسک / سگمنت"},
    {"id": "depth", "label": "Depth", "label_fa": "عمق"},
)


def _modality_catalog_seed() -> List[Dict[str, Any]]:
    try:
        from aiv2_model_contract.llm_taxonomy import modality_catalog

        return modality_catalog()
    except Exception:
        return [dict(m) for m in _FALLBACK_MODALITIES]


def _sql_candidates() -> List[Path]:
    out: List[Path] = []
    here = Path(__file__).resolve()
    for up in (3, 2, 1):
        try:
            out.append(here.parents[up] / "init-scripts" / "14-hf-pipeline-taxonomy.sql")
        except IndexError:
            continue
    out.append(Path("/app/init-scripts/14-hf-pipeline-taxonomy.sql"))
    return out


def _drop_dispatch_sql_candidates() -> List[Path]:
    out: List[Path] = []
    here = Path(__file__).resolve()
    for up in (3, 2, 1):
        try:
            out.append(here.parents[up] / "init-scripts" / "19-drop-dispatch-key.sql")
        except IndexError:
            continue
    out.append(Path("/app/init-scripts/19-drop-dispatch-key.sql"))
    return out


async def ensure_hf_pipeline_taxonomy_schema(pool) -> None:
    path = next((p for p in _sql_candidates() if p.is_file()), None)
    if path is None:
        logger.warning("hf pipeline taxonomy SQL not found; skip ensure_schema")
        return
    sql = path.read_text(encoding="utf-8")
    drop_path = next((p for p in _drop_dispatch_sql_candidates() if p.is_file()), None)
    async with pool.acquire() as conn:
        await conn.execute(sql)
        if drop_path is not None:
            # Only the column-drop portion is safe for ensure_schema (skip model backfill
            # when llm_local_models may not exist yet during fresh init order).
            await conn.execute(
                """
                ALTER TABLE IF EXISTS hf_pipeline_tags
                  DROP COLUMN IF EXISTS suggested_dispatch_key;
                ALTER TABLE IF EXISTS hf_pipeline_tags
                  DROP COLUMN IF EXISTS suggested_task;
                """
            )
        else:
            await conn.execute(
                """
                ALTER TABLE IF EXISTS hf_pipeline_tags
                  DROP COLUMN IF EXISTS suggested_dispatch_key;
                ALTER TABLE IF EXISTS hf_pipeline_tags
                  DROP COLUMN IF EXISTS suggested_task;
                """
            )


async def hf_pipeline_tags_list(
    pool,
    *,
    modality: Optional[str] = None,
    needs_review: Optional[bool] = None,
    include_hidden: bool = False,
) -> List[Dict[str, Any]]:
    clauses = ["TRUE"]
    args: List[Any] = []
    if modality:
        args.append(modality)
        clauses.append(f"hf_modality = ${len(args)}")
    if needs_review is not None:
        args.append(needs_review)
        clauses.append(f"needs_review = ${len(args)}")
    if not include_hidden:
        clauses.append("hide_in_models = FALSE")
    where = " AND ".join(clauses)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, name, hf_modality, subtasks, hide_in_models, hide_in_datasets,
                   input_modalities, output_modalities, suggested_api,
                   api_protocol, endpoint_path, io_confidence, needs_review, io_sources,
                   source_version, synced_at, updated_at
            FROM hf_pipeline_tags
            WHERE {where}
            ORDER BY hf_modality, id
            """,
            *args,
        )
    return [_row_to_dict(r) for r in rows]


async def hf_pipeline_tag_get(pool, tag_id: str) -> Optional[Dict[str, Any]]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, name, hf_modality, subtasks, hide_in_models, hide_in_datasets,
                   input_modalities, output_modalities, suggested_api,
                   api_protocol, endpoint_path, io_confidence, needs_review, io_sources,
                   source_version, synced_at, updated_at
            FROM hf_pipeline_tags
            WHERE id = $1
            """,
            tag_id,
        )
    return _row_to_dict(row) if row else None


async def hf_taxonomy_meta_get(pool) -> Dict[str, Any]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT value, updated_at FROM hf_taxonomy_meta WHERE key = 'pipeline_data'"
        )
    if not row:
        return {}
    val = row["value"]
    if isinstance(val, str):
        val = json.loads(val)
    out = dict(val or {})
    out["updated_at"] = row["updated_at"].isoformat() if row["updated_at"] else None
    return out


async def io_modalities_upsert_catalog(pool, rows: List[Dict[str, Any]]) -> int:
    """Upsert I/O modality atoms (id/label/…)."""
    if not rows:
        return 0
    async with pool.acquire() as conn:
        for i, r in enumerate(rows):
            mid = str(r.get("id") or "").strip()
            if not mid:
                continue
            await conn.execute(
                """
                INSERT INTO io_modalities (id, label, label_fa, source, sort_order, updated_at)
                VALUES ($1, $2, $3, $4, $5, now())
                ON CONFLICT (id) DO UPDATE SET
                  label = EXCLUDED.label,
                  label_fa = COALESCE(EXCLUDED.label_fa, io_modalities.label_fa),
                  source = EXCLUDED.source,
                  sort_order = EXCLUDED.sort_order,
                  updated_at = now()
                """,
                mid,
                str(r.get("label") or mid),
                r.get("label_fa"),
                str(r.get("source") or "aiv2"),
                int(r.get("sort_order") if r.get("sort_order") is not None else (i + 1) * 10),
            )
    return len(rows)


async def io_modalities_list(pool) -> List[Dict[str, Any]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, label, label_fa, source, sort_order, updated_at
            FROM io_modalities
            ORDER BY sort_order, id
            """
        )
    out: List[Dict[str, Any]] = []
    for row in rows:
        d = dict(row)
        if d.get("updated_at") is not None:
            d["updated_at"] = d["updated_at"].isoformat()
        out.append(d)
    return out


async def io_modalities_ensure_seeded(pool) -> List[Dict[str, Any]]:
    """Ensure AIv2 catalog + distinct HF-tag I/O atoms exist, then return DB list."""
    seed: List[Dict[str, Any]] = []
    for i, m in enumerate(_modality_catalog_seed()):
        seed.append(
            {
                "id": m["id"],
                "label": m.get("label") or m["id"],
                "label_fa": m.get("label_fa"),
                "source": "aiv2",
                "sort_order": (i + 1) * 10,
            }
        )
    known = {s["id"] for s in seed}
    async with pool.acquire() as conn:
        derived = await conn.fetch(
            """
            SELECT DISTINCT lower(trim(m)) AS id
            FROM (
              SELECT jsonb_array_elements_text(input_modalities) AS m FROM hf_pipeline_tags
              UNION ALL
              SELECT jsonb_array_elements_text(output_modalities) AS m FROM hf_pipeline_tags
            ) t
            WHERE m IS NOT NULL AND length(trim(m)) > 0
            """
        )
    for row in derived:
        mid = str(row["id"] or "").strip()
        if mid and mid not in known:
            seed.append(
                {
                    "id": mid,
                    "label": mid.replace("-", " ").title(),
                    "label_fa": None,
                    "source": "hf_derived",
                    "sort_order": 500 + len(seed),
                }
            )
            known.add(mid)
    # Only upsert missing rows so operator-edited labels are preserved.
    existing = {r["id"] for r in await io_modalities_list(pool)}
    missing = [s for s in seed if s["id"] not in existing]
    if missing:
        await io_modalities_upsert_catalog(pool, missing)
    return await io_modalities_list(pool)


def _json_field(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def _row_to_dict(row) -> Dict[str, Any]:
    d = dict(row)
    for k in (
        "subtasks",
        "input_modalities",
        "output_modalities",
        "io_sources",
    ):
        if k in d:
            d[k] = _json_field(d[k])
    for ts in ("synced_at", "updated_at"):
        if d.get(ts) is not None:
            d[ts] = d[ts].isoformat()
    return d
