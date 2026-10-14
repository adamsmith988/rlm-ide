"""Map layer catalog, RBAC, detect/import helpers."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .sas_sqlite_tiles import (
    guess_image_media_type,
    infer_sas_sqlite_role,
    infer_sas_sqlite_tile_ext,
    looks_like_sas_sqlite_tree,
    sas_sqlite_tile_bytes,
    sas_sqlite_tile_response,
    sas_sqlite_zoom_range,
)

_LAYER_KINDS = {
    "artifact_gps",
    "raster_xyz",
    "raster_sas",
    "raster_sas_sqlite",
    "raster_pmtiles",
    "raster_mbtiles",
    "raster_cog",
    "raster_wms",
    "raster_wmts",
    "raster_remote",
    "vector_geojson",
    "vector_api",
    "vector_pmtiles",
    "vector_mbtiles",
    "vector_flatgeobuf",
    "vector_geopackage",
    "vector_shapefile",
    "vector_kml",
    "vector_gpx",
    "streetview_folder",
    "streetview_tiled",
    "streetview_equirect",
    "session_draw",
}

_JPEG_RE = re.compile(r"\.(jpe?g|png|webp)$", re.I)


def _parse_lat_lon(lat: Any, lon: Any) -> Optional[Tuple[float, float]]:
    try:
        la = float(lat) if lat is not None else None
        lo = float(lon) if lon is not None else None
    except (TypeError, ValueError):
        return None
    if la is None or lo is None:
        return None
    if not (-90.0 <= la <= 90.0 and -180.0 <= lo <= 180.0):
        return None
    return la, lo

def map_layers_root() -> Path:
    root = (os.environ.get("MAP_LAYERS_ROOT") or "/data/map-layers").strip()
    return Path(root)


def _safe_relative(path: str) -> Optional[str]:
    p = (path or "").strip().replace("\\", "/").lstrip("/")
    if not p or ".." in p.split("/"):
        return None
    return p


def _resolve_under_root(path: Optional[str]) -> Optional[Path]:
    """Resolve path under MAP_LAYERS_ROOT; follow symlinks to external mounts."""
    if not path or not str(path).strip():
        return None
    root = map_layers_root().resolve()
    raw = str(path).strip()
    p = Path(raw)
    if p.is_absolute():
        try:
            entry = p
            entry.relative_to(root)
            return entry.resolve() if entry.exists() or entry.is_symlink() else entry
        except ValueError:
            return None
    rel = _safe_relative(raw)
    if not rel:
        return None
    full = root / rel
    try:
        full.relative_to(root)
    except ValueError:
        return None
    if full.exists() or full.is_symlink():
        return full.resolve()
    return full


class MapLayersService:
    def __init__(self, storage: Any) -> None:
        self.storage = storage

    def _pool(self):
        pg = getattr(self.storage, "postgres", None)
        if not pg or not pg.pool:
            raise RuntimeError("PostgreSQL not available")
        return pg.pool

    async def ensure_schema(self) -> None:
        candidates = [
            Path(__file__).resolve().parents[3] / "init-scripts" / "07-map-layers-schema.sql",
            Path("/app/init-scripts/07-map-layers-schema.sql"),
        ]
        schema_path = next((p for p in candidates if p.is_file()), None)
        if schema_path is None:
            return
        sql = schema_path.read_text(encoding="utf-8")
        async with self._pool().acquire() as conn:
            await conn.execute(sql)

    async def layer_is_accessible(
        self,
        *,
        user_id: str,
        layer_id: str,
        action: str = "read",
        user_roles: Optional[str] = None,
    ) -> Dict[str, Any]:
        from .map_rbac import resolve_map_layer_rbac

        rbac = await resolve_map_layer_rbac(user_id, user_roles)
        lid = uuid.UUID(str(layer_id))

        async with self._pool().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, owner_user_id FROM map_layers WHERE id = $1",
                lid,
            )
        if not row:
            return {
                "accessible": False,
                "access": {"is_owner": False, "is_granted": False, "is_super_admin": rbac.is_super_admin},
            }

        owner = row["owner_user_id"]
        is_owner = str(owner) == str(user_id)
        is_granted = str(lid) in rbac.granted_layer_ids

        if action in ("write", "delete"):
            allowed = rbac.can_mutate_layer(str(lid), owner_user_id=owner)
        else:
            allowed = rbac.can_read_layer(str(lid), owner_user_id=owner)

        return {
            "accessible": allowed,
            "access": {
                "is_owner": is_owner,
                "is_granted": is_granted and not is_owner,
                "is_super_admin": rbac.is_super_admin,
            },
        }

    async def layer_accessible_list(
        self,
        user_id: str,
        *,
        user_roles: Optional[str] = None,
        layer_kind: Optional[str] = None,
        enabled_only: bool = True,
        limit: int = 200,
        offset: int = 0,
    ) -> Dict[str, Any]:
        from .map_rbac import resolve_map_layer_rbac

        rbac = await resolve_map_layer_rbac(user_id, user_roles)
        args: List[Any] = []
        where_parts: List[str] = []

        if not rbac.is_super_admin:
            granted: List[uuid.UUID] = []
            for lid in rbac.granted_layer_ids:
                try:
                    granted.append(uuid.UUID(str(lid)))
                except ValueError:
                    continue
            args.append(user_id)
            if granted:
                args.append(granted)
                where_parts.append("(ml.owner_user_id = $1 OR ml.id = ANY($2::uuid[]))")
            else:
                where_parts.append("ml.owner_user_id = $1")

        if enabled_only:
            where_parts.append("ml.enabled = TRUE")
        if layer_kind:
            args.append(layer_kind)
            where_parts.append(f"ml.layer_kind = ${len(args)}")

        where_sql = " AND ".join(where_parts) if where_parts else "TRUE"
        args.extend([limit, offset])
        lim_idx = len(args) - 1
        off_idx = len(args)

        async with self._pool().acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT ml.*
                FROM map_layers ml
                WHERE {where_sql}
                ORDER BY ml.sort_order, ml.name
                LIMIT ${lim_idx} OFFSET ${off_idx}
                """,
                *args,
            )
            total = await conn.fetchval(
                f"SELECT COUNT(*) FROM map_layers ml WHERE {where_sql}",
                *args[:-2],
            )

        items = [self._enrich_catalog_row(self._row_to_dict(r)) for r in rows]
        if not rbac.is_super_admin:
            items = [
                item
                for item in items
                if rbac.can_read_layer(
                    str(item.get("id")),
                    owner_user_id=item.get("owner_user_id"),
                )
            ]
        for item in items:
            item["data_available"] = self.layer_data_available(item)
        return {
            "items": items,
            "total_count": len(items) if not rbac.is_super_admin else int(total or 0),
            "limit": limit,
            "offset": offset,
        }

    def layer_data_available(self, layer: Dict[str, Any]) -> bool:
        """True when layer metadata points at readable on-disk (or remote) data."""
        if layer.get("enabled") is False:
            return False
        status = str(layer.get("import_status") or "ready").lower()
        if status in ("failed", "error", "missing"):
            return False

        kind = str(layer.get("layer_kind") or "")
        root = layer.get("storage_root")

        if kind in self._BASEMAP_KINDS:
            p = self.pmtiles_file_for_layer(layer)
            try:
                return bool(p and p.is_file() and p.stat().st_size > 0)
            except OSError:
                return False

        if kind == "vector_api":
            sc = layer.get("source_config")
            if isinstance(sc, str):
                try:
                    sc = json.loads(sc)
                except Exception:
                    sc = {}
            if not isinstance(sc, dict):
                return False
            http_cfg = sc.get("http") if isinstance(sc.get("http"), dict) else {}
            return bool(str(http_cfg.get("url") or layer.get("source_url") or "").strip())

        if layer.get("source_url") and not root:
            return True

        if not root:
            return False

        full = _resolve_under_root(str(root))
        if not full or not full.exists():
            return False

        if kind == "raster_sas_sqlite":
            return looks_like_sas_sqlite_tree(full)

        if kind in ("raster_sas", "raster_xyz"):
            if full.is_file():
                try:
                    return full.stat().st_size > 0
                except OSError:
                    return False
            if not full.is_dir():
                return False
            if looks_like_sas_sqlite_tree(full) or self._looks_like_sas(full):
                return True
            for child in full.iterdir():
                if child.is_dir() and child.name.isdigit():
                    return True
            return False

        if kind.startswith("streetview"):
            if not full.is_dir():
                return False
            for child in full.rglob("*.jpg"):
                if child.is_file():
                    return True
            for child in full.rglob("*.jpeg"):
                if child.is_file():
                    return True
            return False

        if full.is_file():
            try:
                return full.stat().st_size > 0
            except OSError:
                return False
        if full.is_dir():
            try:
                return any(full.iterdir())
            except OSError:
                return False
        return False

    def _row_to_dict(self, row: Any) -> Dict[str, Any]:
        from .map_layer_api import redact_source_config

        d = dict(row)
        for k in ("id",):
            if k in d and d[k] is not None:
                d[k] = str(d[k])
        for k in ("source_config", "maplibre", "style", "bounds"):
            if k in d and isinstance(d[k], str):
                try:
                    d[k] = json.loads(d[k])
                except Exception:
                    pass
        if str(d.get("layer_kind") or "") == "vector_api":
            d["source_config"] = redact_source_config(d.get("source_config"))
        return self._enrich_sas_metadata(d)

    async def get_layer_raw(self, layer_id: str) -> Optional[Dict[str, Any]]:
        """Load a layer row without redacting secrets (server-side fetch only)."""
        async with self._pool().acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM map_layers WHERE id = $1",
                uuid.UUID(str(layer_id)),
            )
        if not row:
            return None
        d = dict(row)
        for k in ("id",):
            if k in d and d[k] is not None:
                d[k] = str(d[k])
        for k in ("source_config", "maplibre", "style", "bounds"):
            if k in d and isinstance(d[k], str):
                try:
                    d[k] = json.loads(d[k])
                except Exception:
                    pass
        return d

    def _enrich_catalog_row(self, layer: Dict[str, Any]) -> Dict[str, Any]:
        """Lightweight street-view validation on list (no full detect_path walk)."""
        kind = str(layer.get("layer_kind") or "")
        root = layer.get("storage_root")
        if not kind.startswith("streetview") or not root:
            return layer
        full = map_layers_root() / str(root)
        if not full.exists():
            return {**layer, "data_available": False, "import_status": "missing"}
        # PMTiles / GeoJSON / tile pyramids are never street-view — fast glob only.
        try:
            if list(full.glob("*.pmtiles")) or self._try_detect_pmtiles_dir(full, str(root)):
                return {
                    **layer,
                    "data_available": False,
                    "import_status": "kind_mismatch",
                    "detected_kind": "vector_pmtiles",
                }
            geo_hit = None
            if full.is_file() and full.suffix.lower() in (".geojson", ".json"):
                geo_hit = "vector_geojson"
            elif full.is_dir() and (
                list(full.glob("*.geojson"))
                or self._try_detect_vector_geojson_dir(full, str(root))
            ):
                geo_hit = "vector_geojson"
            if geo_hit:
                return {
                    **layer,
                    "data_available": False,
                    "import_status": "kind_mismatch",
                    "detected_kind": geo_hit,
                }
        except OSError:
            return {**layer, "data_available": False, "import_status": "missing"}
        return {**layer, "data_available": self.layer_data_available(layer)}

    def _enrich_sas_metadata(self, layer: Dict[str, Any]) -> Dict[str, Any]:
        """Attach sas_role / tile_ext from tile sampling when missing in source_config."""
        kind = str(layer.get("layer_kind") or "")
        if kind not in ("raster_sas_sqlite", "raster_sas"):
            return layer

        root = layer.get("storage_root")
        if not root:
            return layer

        sc = layer.get("source_config")
        if not isinstance(sc, dict):
            sc = {}

        full = map_layers_root() / str(root)
        if kind != "raster_sas_sqlite" or not full.is_dir():
            return layer

        role = sc.get("sas_role")
        if not role:
            inferred = infer_sas_sqlite_role(full)
            if inferred:
                role = inferred
                sc = {**sc, "sas_role": role}

        if role and not sc.get("tile_ext"):
            sc = {**sc, "tile_ext": infer_sas_sqlite_tile_ext(full, role)}

        if sc != layer.get("source_config"):
            layer = {**layer, "source_config": sc}
        if role:
            layer["sas_role"] = role
        if sc.get("tile_ext"):
            layer["tile_ext"] = sc["tile_ext"]
        return layer

    def detect_path(self, path: str) -> Dict[str, Any]:
        """Sniff layer kind from filesystem path under MAP_LAYERS_ROOT."""
        root = map_layers_root()
        raw = (path or "").strip()
        if not raw:
            return {"ok": False, "error": "invalid_path"}

        p = Path(raw)
        if p.is_absolute():
            try:
                rel = p.relative_to(root)
            except ValueError:
                return {"ok": False, "error": "path_outside_root"}
            rel_str = rel.as_posix()
        else:
            rel_str = _safe_relative(raw)
            if not rel_str:
                return {"ok": False, "error": "invalid_path"}

        full = root / rel_str
        if not full.exists():
            return {"ok": False, "error": "path_not_found", "path": str(full)}

        if full.is_file():
            ext = full.suffix.lower()
            if ext == ".pmtiles":
                kind = "vector_pmtiles"
                lower = full.name.lower()
                if "satellite" in lower or "raster" in lower or "dem" in lower:
                    kind = "raster_pmtiles"
            elif ext == ".mbtiles":
                kind = "raster_mbtiles"
            elif ext in (".tif", ".tiff"):
                kind = "raster_cog"
            elif ext in (".geojson", ".json"):
                kind = "vector_geojson"
            elif ext == ".fgb":
                kind = "vector_flatgeobuf"
            elif ext == ".gpkg":
                kind = "vector_geopackage"
            elif ext == ".kml":
                kind = "vector_kml"
            elif ext == ".kmz":
                kind = "vector_kml"
            elif ext == ".gpx":
                kind = "vector_gpx"
            elif ext == ".shp":
                kind = "vector_shapefile"
            else:
                kind = "raster_remote"
            return self._enrich_detect(
                {"ok": True, "layer_kind": kind, "storage_root": rel_str, "source_type": "file"},
                full,
            )

        if full.is_dir():
            if looks_like_sas_sqlite_tree(full):
                return self._enrich_detect(
                    {
                        "ok": True,
                        "layer_kind": "raster_sas_sqlite",
                        "storage_root": rel_str,
                        "source_type": "file",
                    },
                    full,
                )
            nested = self.discover_sas_sqlite_layers_under(rel_str)
            if nested:
                layers_under: List[Dict[str, Any]] = []
                for sub in nested:
                    sub_full = map_layers_root() / str(sub["storage_root"])
                    layers_under.append(
                        self._enrich_detect(
                            {
                                "ok": True,
                                "layer_kind": "raster_sas_sqlite",
                                "storage_root": sub["storage_root"],
                                "source_type": "file",
                                "sas_role": sub.get("sas_role"),
                            },
                            sub_full,
                        )
                    )
                return {
                    "ok": True,
                    "layer_kind": "folder",
                    "storage_root": rel_str,
                    "source_type": "file",
                    "layers_under": layers_under,
                    "layer_count": len(layers_under),
                    "data_available": any(l.get("data_available") for l in layers_under),
                }
            if self._looks_like_xyz(full):
                return self._enrich_detect(
                    {
                        "ok": True,
                        "layer_kind": "raster_xyz",
                        "storage_root": rel_str,
                        "source_type": "file",
                    },
                    full,
                )
            if self._looks_like_sas(full):
                return self._enrich_detect(
                    {
                        "ok": True,
                        "layer_kind": "raster_sas",
                        "storage_root": rel_str,
                        "source_type": "file",
                    },
                    full,
                )
            pmtiles_hit = self._try_detect_pmtiles_dir(full, rel_str)
            if pmtiles_hit:
                return self._enrich_detect(pmtiles_hit, full)
            # GeoJSON (+ optional thumbnail sidecars) must win over streetview JPEG heuristic.
            geojson_hit = self._try_detect_vector_geojson_dir(full, rel_str)
            if geojson_hit:
                return self._enrich_detect(geojson_hit, full)
            try:
                visible = [c for c in full.iterdir() if not c.name.startswith(".")]
            except OSError:
                visible = []
            if len(visible) == 1:
                only = visible[0]
                try:
                    is_data_dir = only.is_dir() or (
                        only.is_symlink() and only.resolve().is_dir()
                    )
                except OSError:
                    is_data_dir = False
                if is_data_dir:
                    child_rel = f"{rel_str}/{only.name}" if rel_str else only.name
                    nested = self.detect_path(child_rel)
                    if nested.get("ok"):
                        if nested.get("layer_kind") != "folder":
                            nested["import_root"] = rel_str
                        return nested
            if self._looks_like_streetview_folder(full):
                return self._enrich_detect(
                    {
                        "ok": True,
                        "layer_kind": "streetview_folder",
                        "storage_root": rel_str,
                        "source_type": "file",
                        "jpeg_count": len(list(full.rglob("*.jp*g"))),
                    },
                    full,
                )
            if list(full.glob("*/*/*.*")):
                return self._enrich_detect(
                    {
                        "ok": True,
                        "layer_kind": "raster_xyz",
                        "storage_root": rel_str,
                        "source_type": "file",
                    },
                    full,
                )
            try:
                visible = [c for c in full.iterdir() if not c.name.startswith(".")]
            except OSError:
                visible = []
            if not visible:
                return {
                    "ok": False,
                    "error": "empty_directory",
                    "storage_root": rel_str,
                    "hint": "Directory exists but has no tile data; run scripts/setup_map_external.sh and restart storage",
                }
        return {"ok": False, "error": "unknown_format"}

    @staticmethod
    def _classify_pmtiles_kind(filename: str) -> str:
        lower = filename.lower()
        if any(x in lower for x in ("satellite", "sat", "raster", "dem", "terrain")):
            return "raster_pmtiles"
        return "vector_pmtiles"

    def _try_detect_pmtiles_dir(self, full: Path, rel_str: str) -> Optional[Dict[str, Any]]:
        """Detect *.pmtiles in the immediate directory (no deep scan)."""
        if not full.is_dir():
            return None
        files = sorted(full.glob("*.pmtiles"))
        if not files:
            return None
        primary = files[0]
        for candidate in files:
            name = candidate.name.lower()
            if "iran" in name or "middle-east" in name or "middle_east" in name:
                primary = candidate
                break
        kind = self._classify_pmtiles_kind(primary.name)
        return {
            "ok": True,
            "layer_kind": kind,
            "storage_root": rel_str,
            "source_type": "file",
            "pmtiles_file": primary.name,
        }

    def _enrich_detect(self, base: Dict[str, Any], full: Path) -> Dict[str, Any]:
        """Attach probe metadata (data_available, zoom range, warnings) to detect output."""
        kind = str(base.get("layer_kind") or "")
        warnings: List[str] = []
        min_zoom: Optional[int] = None
        max_zoom: Optional[int] = None
        tile_ext = "jpg"

        if kind == "raster_sas_sqlite" and full.is_dir():
            min_zoom, max_zoom = sas_sqlite_zoom_range(full)
            if min_zoom is None:
                warnings.append("sas_sqlite_zoom_not_inferred")
        elif kind in ("raster_xyz", "raster_sas") and full.is_dir():
            levels: List[int] = []
            for child in full.iterdir():
                if child.is_dir() and child.name.isdigit():
                    levels.append(int(child.name))
                elif child.is_dir() and child.name.startswith("z") and child.name[1:].isdigit():
                    levels.append(int(child.name[1:]))
            if levels:
                min_zoom, max_zoom = min(levels), max(levels)

        layer_stub = {
            "layer_kind": kind,
            "storage_root": base.get("storage_root"),
            "source_url": base.get("source_url"),
            "enabled": True,
            "import_status": "ready",
        }
        data_available = self.layer_data_available(layer_stub)
        sas_role = None
        if kind == "raster_sas_sqlite" and full.is_dir() and data_available:
            sas_role = infer_sas_sqlite_role(full)
            if sas_role:
                tile_ext = infer_sas_sqlite_tile_ext(full, sas_role)
            elif max_zoom is not None:
                try:
                    sample = sas_sqlite_tile_bytes(full, max_zoom, 0, 0)
                    if sample:
                        tile_ext = "png" if sample.startswith(b"\x89PNG") else "jpg"
                except OSError:
                    pass

        base["data_available"] = data_available
        if min_zoom is not None:
            base["min_zoom"] = min_zoom
        if max_zoom is not None:
            base["max_zoom"] = max_zoom
        base["tile_ext"] = tile_ext
        if sas_role:
            base["sas_role"] = sas_role
        if warnings:
            base["warnings"] = warnings
        return base

    def detect_url(self, url: str) -> Dict[str, Any]:
        """Sniff layer kind from a remote URL or tile template."""
        raw = (url or "").strip()
        if not raw:
            return {"ok": False, "error": "invalid_url"}
        lower = raw.lower()
        kind = "raster_remote"
        if ".pmtiles" in lower:
            kind = (
                "raster_pmtiles"
                if any(x in lower for x in ("sat", "raster", "dem", "tile"))
                else "vector_pmtiles"
            )
        elif ".mbtiles" in lower:
            kind = "raster_mbtiles"
        elif "tilejson" in lower or lower.endswith(".json"):
            kind = "raster_remote"
        elif "{z}" in raw and "{x}" in raw and "{y}" in raw:
            kind = "raster_xyz"
        elif "wmts" in lower or "service=wmts" in lower:
            kind = "raster_wmts"
        elif "wms" in lower or "service=wms" in lower:
            kind = "raster_wms"
        return {
            "ok": True,
            "layer_kind": kind,
            "source_url": raw,
            "source_type": "url",
            "data_available": True,
        }

    def _try_detect_vector_geojson_dir(self, full: Path, rel_str: str) -> Optional[Dict[str, Any]]:
        """Prefer an explicit *.geojson file over JPEG sidecar folders (thumbnails/)."""
        if not full.is_dir():
            return None
        preferred = ("points-preview.geojson", "points.geojson", "data.geojson", "layer.geojson")
        primary: Optional[Path] = None
        for name in preferred:
            cand = full / name
            if cand.is_file():
                primary = cand
                break
        if primary is None:
            files = sorted(p for p in full.glob("*.geojson") if p.is_file())
            if not files:
                return None
            primary = files[0]
        file_rel = f"{rel_str}/{primary.name}" if rel_str else primary.name
        return {
            "ok": True,
            "layer_kind": "vector_geojson",
            "storage_root": file_rel,
            "source_type": "file",
            "geojson_file": primary.name,
        }

    def load_vector_geojson_features(self, layer: Dict[str, Any]) -> Dict[str, Any]:
        """Read FeatureCollection for a catalog ``vector_geojson`` layer from disk."""
        from .map_layer_api import ApiLayerError, normalize_geojson

        root = str(layer.get("storage_root") or "").strip()
        if not root:
            raise ApiLayerError("missing_storage_root", "vector_geojson requires storage_root")
        full = _resolve_under_root(root)
        if not full or not full.is_file():
            raise ApiLayerError("geojson_file_missing", f"GeoJSON not found: {root}")
        try:
            with full.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ApiLayerError("geojson_read_failed", str(exc)) from exc
        return normalize_geojson(payload)

    def resolve_layer_asset_path(
        self, layer: Dict[str, Any], asset_path: str
    ) -> Optional[Path]:
        """Resolve a sidecar file under the layer's directory (or import_root).

        ``storage_root`` may be a file (``photo-tagger-gps/points-preview.geojson``)
        or a directory. Sidecars like ``thumbnails/<id>.jpg`` live next to the
        GeoJSON / under ``source_config.import_root``.
        """
        rel = _safe_relative(asset_path)
        if not rel:
            raise ValueError("invalid_asset_path")
        candidates: List[Path] = []
        sc = layer.get("source_config") if isinstance(layer.get("source_config"), dict) else {}
        import_root = str(sc.get("import_root") or "").strip()
        if import_root:
            base = _resolve_under_root(import_root)
            if base is not None:
                candidates.append(base)
        storage_root = str(layer.get("storage_root") or "").strip()
        if storage_root:
            resolved = _resolve_under_root(storage_root)
            if resolved is not None:
                candidates.append(resolved.parent if resolved.is_file() else resolved)
        root = map_layers_root().resolve()
        for base in candidates:
            try:
                full = (base / rel).resolve()
                full.relative_to(root)
            except (OSError, ValueError):
                continue
            if full.is_file():
                return full
        return None

    def _looks_like_streetview_folder(self, path: Path) -> bool:
        """True for panorama folders — not PMTiles dirs or tile pyramids with JPEG."""
        if list(path.glob("*.pmtiles")):
            return False
        # Explicit vector GeoJSON (+ thumbnail JPGs) is never street view.
        if list(path.glob("*.geojson")):
            return False
        if (path / "panorama_north.json").is_file():
            return True
        if looks_like_sas_sqlite_tree(path) or self._looks_like_xyz(path) or self._looks_like_sas(path):
            return False
        try:
            children = [c for c in path.iterdir() if not c.name.startswith(".")]
        except OSError:
            return False
        for child in children:
            if child.is_dir() and list(child.glob("*.pmtiles")):
                return False
            if child.is_dir() and (
                self._looks_like_xyz(child)
                or self._looks_like_sas(child)
                or looks_like_sas_sqlite_tree(child)
            ):
                return False
        jpgs: List[Path] = []
        for child in children:
            if child.is_file() and child.suffix.lower() in (".jpg", ".jpeg"):
                jpgs.append(child)
            elif child.is_dir():
                for jpg in child.rglob("*.jp*g"):
                    jpgs.append(jpg)
                    if len(jpgs) >= 20:
                        break
            if len(jpgs) >= 20:
                break
        if not jpgs:
            return False
        for jpg in jpgs[:5]:
            try:
                parts = jpg.relative_to(path).parts
            except ValueError:
                continue
            if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
                return False
        return True

    def _looks_like_sas(self, path: Path) -> bool:
        for child in path.iterdir():
            if child.is_dir() and child.name.startswith("z") and child.name[1:].isdigit():
                return True
        return False

    def _looks_like_xyz(self, path: Path) -> bool:
        """Detect a numeric z/x/y tile pyramid (z, x dirs are integers; tile is int.ext)."""
        try:
            for z in path.iterdir():
                if not (z.is_dir() and z.name.isdigit()):
                    continue
                for x in z.iterdir():
                    if not (x.is_dir() and x.name.isdigit()):
                        continue
                    for y in x.iterdir():
                        if y.is_file() and y.stem.isdigit():
                            return True
                    return False
                return False
        except OSError:
            return False
        return False

    def _file_mvt_hint(self, path: Path) -> bool:
        return "vector" in path.name.lower()

    async def find_layer_by_storage_root(
        self,
        owner_user_id: str,
        storage_root: str,
    ) -> Optional[Dict[str, Any]]:
        rel = _safe_relative(storage_root)
        if not rel:
            return None
        async with self._pool().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM map_layers
                WHERE owner_user_id = $1 AND storage_root = $2
                ORDER BY created_at DESC LIMIT 1
                """,
                owner_user_id,
                rel,
            )
        return self._row_to_dict(row) if row else None

    async def find_any_layer_by_storage_root(self, storage_root: str) -> Optional[Dict[str, Any]]:
        """Return newest catalog row for a storage_root (any owner)."""
        rel = _safe_relative(storage_root)
        if not rel:
            return None
        async with self._pool().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM map_layers
                WHERE storage_root = $1
                ORDER BY created_at DESC LIMIT 1
                """,
                rel,
            )
        return self._row_to_dict(row) if row else None

    async def find_or_create_layer(
        self,
        *,
        owner_user_id: str,
        name: str,
        layer_kind: str,
        storage_root: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if storage_root:
            existing = await self.find_any_layer_by_storage_root(storage_root)
            if existing:
                return existing
        if storage_root and layer_kind == "raster_sas_sqlite":
            full = map_layers_root() / str(storage_root)
            if full.is_dir():
                mn, mx = sas_sqlite_zoom_range(full)
                if kwargs.get("min_zoom") is None and mn is not None:
                    kwargs["min_zoom"] = mn
                if kwargs.get("max_zoom") is None and mx is not None:
                    kwargs["max_zoom"] = mx
                sc = kwargs.get("source_config")
                if not isinstance(sc, dict):
                    sc = {}
                if not sc.get("sas_role"):
                    role = infer_sas_sqlite_role(full)
                    if role:
                        sc = {**sc, "sas_role": role}
                if sc.get("sas_role") and not sc.get("tile_ext"):
                    sc = {**sc, "tile_ext": infer_sas_sqlite_tile_ext(full, sc["sas_role"])}
                kwargs["source_config"] = sc
        return await self.create_layer(
            owner_user_id=owner_user_id,
            name=name,
            layer_kind=layer_kind,
            storage_root=storage_root,
            **kwargs,
        )

    def discover_sas_sqlite_layers_under(self, base_rel: str) -> List[Dict[str, Any]]:
        """Find SAS sqlite caches under ``base_rel`` only (not sibling folders elsewhere)."""
        rel = _safe_relative(base_rel)
        if not rel:
            return []
        base = map_layers_root() / rel
        if not base.is_dir():
            return []

        candidates: List[Path] = []
        if looks_like_sas_sqlite_tree(base):
            candidates.append(base)
        else:
            for child in sorted(base.iterdir()):
                if not child.is_dir() or child.name.startswith("."):
                    continue
                if looks_like_sas_sqlite_tree(child):
                    candidates.append(child)
                    continue
                try:
                    for sub in sorted(child.iterdir()):
                        if sub.is_dir() and looks_like_sas_sqlite_tree(sub):
                            candidates.append(sub)
                except OSError:
                    continue

        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for path in candidates:
            sub_rel = path.relative_to(map_layers_root()).as_posix()
            if sub_rel in seen:
                continue
            seen.add(sub_rel)
            role = infer_sas_sqlite_role(path)
            mn, mx = sas_sqlite_zoom_range(path)
            stub = {
                "layer_kind": "raster_sas_sqlite",
                "storage_root": sub_rel,
                "enabled": True,
                "import_status": "ready",
            }
            out.append(
                {
                    "storage_root": sub_rel,
                    "layer_kind": "raster_sas_sqlite",
                    "sas_role": role,
                    "tile_ext": infer_sas_sqlite_tile_ext(path, role),
                    "min_zoom": mn,
                    "max_zoom": mx,
                    "data_available": self.layer_data_available(stub),
                }
            )
        return out

    async def create_layer(
        self,
        *,
        owner_user_id: str,
        name: str,
        layer_kind: str,
        source_type: str = "file",
        storage_root: Optional[str] = None,
        source_url: Optional[str] = None,
        source_config: Optional[Dict[str, Any]] = None,
        bounds: Optional[Any] = None,
        min_zoom: Optional[int] = None,
        max_zoom: Optional[int] = None,
    ) -> Dict[str, Any]:
        kind = (layer_kind or "").strip()
        if kind not in _LAYER_KINDS:
            raise ValueError(f"unsupported layer_kind: {kind}")
        if storage_root:
            rel = _safe_relative(storage_root)
            if not rel:
                raise ValueError("invalid storage_root")
            storage_root = rel

        cfg = source_config or {}
        if kind == "vector_api":
            from .map_layer_api import ApiLayerError, validate_vector_api_config

            try:
                cfg = validate_vector_api_config(cfg if isinstance(cfg, dict) else {})
            except ApiLayerError as exc:
                raise ValueError(f"{exc.code}: {exc.message}") from exc
            source_type = "api"
            if not source_url:
                source_url = str(cfg.get("http", {}).get("url") or "")

        async with self._pool().acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO map_layers (
                    owner_user_id, name, layer_kind, source_type,
                    storage_root, source_url, source_config, bounds,
                    min_zoom, max_zoom, import_status
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9, $10, 'ready')
                RETURNING *
                """,
                owner_user_id,
                name.strip(),
                kind,
                source_type,
                storage_root,
                source_url,
                json.dumps(cfg or {}),
                json.dumps(bounds) if bounds is not None else None,
                min_zoom,
                max_zoom,
            )
        layer = self._row_to_dict(row)
        if kind.startswith("streetview"):
            await self.scan_streetview_layer(str(layer["id"]))
        return layer

    async def update_layer(
        self,
        layer_id: str,
        *,
        user_id: str,
        patch: Dict[str, Any],
        user_roles: Optional[str] = None,
    ) -> Dict[str, Any]:
        access = await self.layer_is_accessible(
            user_id=user_id,
            layer_id=layer_id,
            action="write",
            user_roles=user_roles,
        )
        if not access.get("accessible"):
            raise PermissionError("not_authorized_for_layer")

        allowed = {"name", "enabled", "sort_order", "style", "source_config", "bounds", "min_zoom", "max_zoom"}
        sets = []
        args: List[Any] = []
        for k, v in patch.items():
            if k not in allowed:
                continue
            args.append(json.dumps(v) if k in ("style", "source_config", "bounds") and isinstance(v, (dict, list)) else v)
            sets.append(f"{k} = ${len(args)}")
        if not sets:
            raise ValueError("empty_patch")
        args.append(uuid.UUID(layer_id))
        async with self._pool().acquire() as conn:
            row = await conn.fetchrow(
                f"UPDATE map_layers SET {', '.join(sets)}, updated_at = now() WHERE id = ${len(args)} RETURNING *",
                *args,
            )
        if not row:
            raise KeyError("layer_not_found")
        return self._row_to_dict(row)

    async def delete_layer(
        self,
        layer_id: str,
        *,
        user_id: str,
        purge_data: bool = False,
        user_roles: Optional[str] = None,
    ) -> Dict[str, Any]:
        access = await self.layer_is_accessible(
            user_id=user_id,
            layer_id=layer_id,
            action="delete",
            user_roles=user_roles,
        )
        if not access.get("accessible"):
            raise PermissionError("not_authorized_for_layer")
        async with self._pool().acquire() as conn:
            row = await conn.fetchrow(
                "DELETE FROM map_layers WHERE id = $1 RETURNING *",
                uuid.UUID(layer_id),
            )
        if not row:
            raise KeyError("layer_not_found")
        layer = self._row_to_dict(row)
        if purge_data and layer.get("storage_root"):
            full = map_layers_root() / str(layer["storage_root"])
            try:
                resolved = full.resolve()
                resolved.relative_to(map_layers_root().resolve())
            except (ValueError, OSError):
                resolved = None
            if resolved:
                if resolved.is_file():
                    try:
                        resolved.unlink()
                    except OSError:
                        pass
                elif resolved.is_dir():
                    try:
                        shutil.rmtree(resolved)
                    except OSError:
                        pass
        return {"deleted": True, "layer": layer}

    async def scan_streetview_layer(self, layer_id: str) -> Dict[str, Any]:
        async with self._pool().acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM map_layers WHERE id = $1", uuid.UUID(layer_id))
        if not row:
            raise KeyError("layer_not_found")
        layer = self._row_to_dict(row)
        rel = layer.get("storage_root")
        if not rel:
            return {"panoramas": 0}
        folder = map_layers_root() / rel
        if not folder.is_dir():
            return {"panoramas": 0, "error": "folder_missing"}

        meta_file = folder / "panorama_north.json"
        meta_map: Dict[str, Any] = {}
        if meta_file.is_file():
            try:
                meta_map = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                meta_map = {}

        panos: List[Dict[str, Any]] = []

        def _walk_dir(base: Path) -> None:
            stack = [base]
            while stack:
                current = stack.pop()
                try:
                    with os.scandir(current) as it:
                        for entry in it:
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(Path(entry.path))
                                continue
                            if not entry.is_file(follow_symlinks=False):
                                continue
                            if not _JPEG_RE.search(entry.name):
                                continue
                            path = Path(entry.path)
                            rel_file = path.relative_to(folder).as_posix()
                            entry_meta = meta_map.get(path.name) or meta_map.get(rel_file) or {}
                            lat_raw = entry_meta.get("lat") or entry_meta.get("latitude")
                            lon_raw = (
                                entry_meta.get("lon")
                                or entry_meta.get("lng")
                                or entry_meta.get("longitude")
                            )
                            coords = _parse_lat_lon(lat_raw, lon_raw)
                            if coords is None:
                                continue
                            la, lo = coords
                            headings = entry_meta.get("headings") or entry_meta.get("heading")
                            if headings is not None and not isinstance(headings, list):
                                headings = [headings]
                            panos.append(
                                {
                                    "filename": path.name,
                                    "relative_path": rel_file,
                                    "latitude": la,
                                    "longitude": lo,
                                    "headings": headings or [],
                                    "metadata": entry_meta,
                                }
                            )
                except OSError:
                    continue

        _walk_dir(folder)

        async with self._pool().acquire() as conn:
            await conn.execute("DELETE FROM map_streetview_pano WHERE layer_id = $1", uuid.UUID(layer_id))
            batch: List[Dict[str, Any]] = []
            for p in panos:
                batch.append(p)
                if len(batch) >= 200:
                    for item in batch:
                        await conn.execute(
                            """
                            INSERT INTO map_streetview_pano
                            (layer_id, filename, latitude, longitude, headings, relative_path, metadata)
                            VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7::jsonb)
                            """,
                            uuid.UUID(layer_id),
                            item["filename"],
                            item["latitude"],
                            item["longitude"],
                            json.dumps(item["headings"]),
                            item["relative_path"],
                            json.dumps(item["metadata"]),
                        )
                    batch.clear()
            for item in batch:
                await conn.execute(
                    """
                    INSERT INTO map_streetview_pano
                    (layer_id, filename, latitude, longitude, headings, relative_path, metadata)
                    VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7::jsonb)
                    """,
                    uuid.UUID(layer_id),
                    item["filename"],
                    item["latitude"],
                    item["longitude"],
                    json.dumps(item["headings"]),
                    item["relative_path"],
                    json.dumps(item["metadata"]),
                )
        return {"panoramas": len(panos), "layer_id": layer_id}

    async def streetview_panoramas(
        self,
        layer_id: str,
        *,
        user_id: str,
        folder: Optional[str] = None,
        user_roles: Optional[str] = None,
    ) -> Dict[str, Any]:
        access = await self.layer_is_accessible(
            user_id=user_id,
            layer_id=layer_id,
            action="read",
            user_roles=user_roles,
        )
        if not access.get("accessible"):
            raise PermissionError("not_authorized_for_layer")
        async with self._pool().acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT filename, latitude, longitude, headings, relative_path, metadata
                FROM map_streetview_pano WHERE layer_id = $1 ORDER BY filename
                """,
                uuid.UUID(layer_id),
            )
        if not rows:
            await self.scan_streetview_layer(layer_id)
            async with self._pool().acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT filename, latitude, longitude, headings, relative_path, metadata
                    FROM map_streetview_pano WHERE layer_id = $1 ORDER BY filename
                    """,
                    uuid.UUID(layer_id),
                )
        panos = [p for p in (self._pano_row_to_dict(r) for r in rows) if p is not None]
        return {"panoramas": panos, "count": len(panos), "layer_id": layer_id}

    def _pano_row_to_dict(self, row: Any) -> Optional[Dict[str, Any]]:
        d = dict(row)
        for k in ("headings", "metadata"):
            if k in d and isinstance(d[k], str):
                try:
                    d[k] = json.loads(d[k])
                except Exception:
                    pass
        coords = _parse_lat_lon(d.get("latitude"), d.get("longitude"))
        if coords is None:
            return None
        la, lo = coords
        d["latitude"] = la
        d["longitude"] = lo
        d["lat"] = la
        d["lon"] = lo
        return d

    @staticmethod
    def _quick_sas_cache_hint(directory: Path) -> bool:
        """Fast SAS sqlite hint for browse listing — no recursive glob."""
        try:
            for child in directory.iterdir():
                if child.name.startswith(".") or not child.is_dir():
                    continue
                name = child.name
                if name.startswith("z") and name[1:].isdigit():
                    return True
        except OSError:
            return False
        return False

    def _browse_layers_root(self, path: Optional[str] = None) -> Dict[str, Any]:
        """List directories under MAP_LAYERS_ROOT (relative paths for catalog import).

        Logical paths stay under MAP_LAYERS_ROOT (e.g. ``sas-both``) even when the
        entry is a symlink to an external mount — ``resolve()`` is only used to
        open the directory for listing, not for ``relative_to`` checks.
        """
        root = map_layers_root()
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        root = root.resolve()
        raw = (path or "").strip().replace("\\", "/").strip("/")
        if raw:
            rel_path = _safe_relative(raw)
            if not rel_path:
                return {"error": "invalid_path", "path": path}
            entry = root / rel_path
            if not entry.exists() and not entry.is_symlink():
                return {"error": "invalid_path", "path": path}
            try:
                resolved = entry.resolve()
            except OSError:
                return {"error": "invalid_path", "path": path}
            if not resolved.is_dir():
                return {"error": "invalid_path", "path": path}
        else:
            rel_path = ""
            resolved = root

        parts = rel_path.split("/") if rel_path else []
        parent_rel = "/".join(parts[:-1]) if len(parts) > 1 else ""

        items: List[Dict[str, Any]] = []
        for child in sorted(resolved.iterdir()):
            if child.name.startswith("."):
                continue
            if not child.is_dir():
                continue
            has_sqlite = self._quick_sas_cache_hint(child)
            items.append(
                {
                    "name": child.name,
                    "isDir": True,
                    "is_dir": True,
                    "sqlitedb_count": 1 if has_sqlite else 0,
                }
            )

        return {
            "path": rel_path,
            "absolute_path": str(resolved),
            "parent": parent_rel,
            "root": str(root),
            "items": items,
            "entries": items,
        }

    async def streetview_browse(
        self,
        *,
        user_id: str,
        path: Optional[str] = None,
        user_roles: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Browse MAP_LAYERS_ROOT for catalog/streetview import paths."""
        listing = self._browse_layers_root(path)
        if listing.get("error"):
            return listing

        try:
            layers = await self.layer_accessible_list(
                user_id,
                user_roles=user_roles,
                layer_kind=None,
                enabled_only=False,
                limit=500,
                offset=0,
            )
        except Exception:
            layers = {"items": []}
        sv_layers = [
            x
            for x in layers.get("items") or []
            if str(x.get("layer_kind") or "").startswith("streetview")
        ]
        import_roots: List[Dict[str, Any]] = []
        root = map_layers_root()
        import_dir = root / "streetview"
        if import_dir.is_dir():
            for child in sorted(import_dir.iterdir()):
                if child.is_dir():
                    import_roots.append(
                        {
                            "path": str(child),
                            "name": child.name,
                        }
                    )
        listing["layers"] = sv_layers
        listing["import_roots"] = import_roots
        listing["default_root"] = str(import_dir)
        return listing

    def streetview_image_path(
        self,
        layer: Dict[str, Any],
        relative_path: str,
    ) -> Optional[Path]:
        root = layer.get("storage_root")
        if not root:
            return None
        base = map_layers_root() / str(root)
        rel = _safe_relative(relative_path.replace("\\", "/"))
        if not rel:
            return None
        full = (base / rel).resolve()
        try:
            full.relative_to(base.resolve())
        except ValueError:
            return None
        if full.is_file() and _JPEG_RE.search(full.name):
            return full
        return None

    _BASEMAP_KINDS = frozenset({"vector_pmtiles", "raster_pmtiles"})

    def pmtiles_file_for_layer(
        self, layer: Dict[str, Any], *, filename: Optional[str] = None
    ) -> Optional[Path]:
        kind = str(layer.get("layer_kind") or "")
        if kind not in self._BASEMAP_KINDS:
            return None
        root = layer.get("storage_root")
        if not root:
            return None
        full = _resolve_under_root(str(root))
        if not full:
            return None
        if filename:
            safe = Path(str(filename).replace("\\", "/")).name
            if not safe.lower().endswith(".pmtiles"):
                return None
            if full.is_dir():
                candidate = full / safe
                return candidate if candidate.is_file() else None
            return None
        if full.is_file() and full.suffix.lower() == ".pmtiles":
            return full
        if full.is_dir():
            sc = layer.get("source_config") or {}
            if isinstance(sc, str):
                try:
                    sc = json.loads(sc)
                except Exception:
                    sc = {}
            preferred = sc.get("pmtiles_file") if isinstance(sc, dict) else None
            if preferred:
                candidate = full / str(preferred)
                if candidate.is_file():
                    return candidate
            matches = sorted(full.glob("*.pmtiles"))
            return matches[0] if matches else None
        return None

    @staticmethod
    def _companion_pmtiles_files(layer_dir: Path) -> Dict[str, Optional[str]]:
        """Discover world-overview / world-full siblings in a PMTiles directory."""
        out: Dict[str, Optional[str]] = {
            "world_overview_file": None,
            "world_full_file": None,
        }
        if not layer_dir.is_dir():
            return out
        try:
            files = sorted(layer_dir.glob("*.pmtiles"))
        except OSError:
            return out
        for f in files:
            lower = f.name.lower()
            if "world-overview" in lower or lower.startswith("world_overview"):
                out["world_overview_file"] = f.name
            elif "world-full" in lower or "world_full" in lower:
                out["world_full_file"] = f.name
        return out

    async def get_user_basemap(
        self, user_id: str, *, user_roles: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            async with self._pool().acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT layer_id::text AS layer_id, updated_at FROM user_map_basemap WHERE user_id = $1",
                    user_id,
                )
        except Exception:
            return {"layer_id": None, "layer": None}
        if not row or not row["layer_id"]:
            return {"layer_id": None, "layer": None}
        layer_id = str(row["layer_id"])
        access = await self.layer_is_accessible(
            user_id=user_id,
            layer_id=layer_id,
            action="read",
            user_roles=user_roles,
        )
        if not access.get("accessible"):
            return {"layer_id": None, "layer": None}
        async with self._pool().acquire() as conn:
            layer_row = await conn.fetchrow("SELECT * FROM map_layers WHERE id = $1", uuid.UUID(layer_id))
        if not layer_row:
            return {"layer_id": None, "layer": None}
        layer = self._row_to_dict(layer_row)
        pmtiles = self.pmtiles_file_for_layer(layer)
        data_ok = False
        if pmtiles:
            try:
                data_ok = pmtiles.is_file() and pmtiles.stat().st_size > 0
            except OSError:
                data_ok = False
        companions = self._companion_pmtiles_files(pmtiles.parent if pmtiles else Path())
        primary_name = pmtiles.name if pmtiles else None
        return {
            "layer_id": layer_id,
            "layer": layer,
            "pmtiles_path": str(pmtiles) if pmtiles else None,
            "pmtiles_file": primary_name,
            "world_overview_file": companions.get("world_overview_file"),
            "world_full_file": companions.get("world_full_file"),
            "data_available": data_ok,
            "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
        }

    async def set_user_basemap_layer(
        self, user_id: str, layer_id: str, *, user_roles: Optional[str] = None
    ) -> Dict[str, Any]:
        access = await self.layer_is_accessible(
            user_id=user_id,
            layer_id=layer_id,
            action="read",
            user_roles=user_roles,
        )
        if not access.get("accessible"):
            raise PermissionError("not_authorized_for_layer")
        async with self._pool().acquire() as conn:
            layer_row = await conn.fetchrow("SELECT * FROM map_layers WHERE id = $1", uuid.UUID(layer_id))
        if not layer_row:
            raise KeyError("layer_not_found")
        layer = self._row_to_dict(layer_row)
        if str(layer.get("layer_kind") or "") not in self._BASEMAP_KINDS:
            raise ValueError("layer_not_pmtiles")
        if not self.pmtiles_file_for_layer(layer):
            raise ValueError("pmtiles_file_not_found")
        async with self._pool().acquire() as conn:
            await conn.execute(
                """
                INSERT INTO user_map_basemap (user_id, layer_id, updated_at)
                VALUES ($1, $2::uuid, now())
                ON CONFLICT (user_id) DO UPDATE
                SET layer_id = EXCLUDED.layer_id, updated_at = now()
                """,
                user_id,
                uuid.UUID(layer_id),
            )
        return await self.get_user_basemap(user_id)

    async def describe_pmtiles_style(
        self,
        layer_id: str,
        *,
        user_id: str,
        user_roles: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Read-only PMTiles bundle metadata for map style (no user_map_basemap)."""
        access = await self.layer_is_accessible(
            user_id=user_id,
            layer_id=layer_id,
            action="read",
            user_roles=user_roles,
        )
        if not access.get("accessible"):
            raise PermissionError("not_authorized_for_layer")
        async with self._pool().acquire() as conn:
            layer_row = await conn.fetchrow("SELECT * FROM map_layers WHERE id = $1", uuid.UUID(layer_id))
        if not layer_row:
            raise KeyError("layer_not_found")
        layer = self._row_to_dict(layer_row)
        if str(layer.get("layer_kind") or "") not in self._BASEMAP_KINDS:
            raise ValueError("layer_not_pmtiles")
        pmtiles = self.pmtiles_file_for_layer(layer)
        data_ok = False
        if pmtiles:
            try:
                data_ok = pmtiles.is_file() and pmtiles.stat().st_size > 0
            except OSError:
                data_ok = False
        companions = self._companion_pmtiles_files(pmtiles.parent if pmtiles else Path())
        primary_name = pmtiles.name if pmtiles else None
        return {
            "layer_id": layer_id,
            "pmtiles_file": primary_name,
            "world_overview_file": companions.get("world_overview_file"),
            "world_full_file": companions.get("world_full_file"),
            "data_available": data_ok,
        }

    async def set_user_basemap_from_path(
        self,
        user_id: str,
        path: str,
        *,
        name: Optional[str] = None,
        user_roles: Optional[str] = None,
    ) -> Dict[str, Any]:
        detected = self.detect_path(path)
        if not detected.get("ok"):
            raise ValueError(detected.get("error") or "detect_failed")
        kind = str(detected.get("layer_kind") or "")
        if kind not in self._BASEMAP_KINDS:
            raise ValueError("not_pmtiles_path")
        rel = str(detected.get("storage_root") or "").strip()
        if rel:
            listed = await self.layer_accessible_list(
                user_id,
                user_roles=user_roles,
                enabled_only=False,
                limit=500,
                offset=0,
            )
            for item in listed.get("items") or []:
                if str(item.get("storage_root") or "").strip() == rel:
                    return await self.set_user_basemap_layer(
                        user_id,
                        str(item["id"]),
                        user_roles=user_roles,
                    )
        layer = await self.find_or_create_layer(
            owner_user_id=user_id,
            name=name or Path(path.strip()).name or "basemap",
            layer_kind=kind,
            storage_root=rel or detected.get("storage_root"),
        )
        return await self.set_user_basemap_layer(
            user_id, str(layer["id"]), user_roles=user_roles
        )

    def effective_zoom_range(self, layer: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        min_z = layer.get("min_zoom")
        max_z = layer.get("max_zoom")
        kind = layer.get("layer_kind") or ""
        root = layer.get("storage_root")
        if kind == "raster_sas_sqlite" and root and (min_z is None or max_z is None):
            mn, mx = sas_sqlite_zoom_range(map_layers_root() / str(root))
            if min_z is None:
                min_z = mn
            if max_z is None:
                max_z = mx
        return min_z, max_z

    def tile_path_for_layer(self, layer: Dict[str, Any], z: int, x: int, y: int, ext: str = "jpg") -> Optional[Path]:
        kind = layer.get("layer_kind") or ""
        root = layer.get("storage_root")
        if not root:
            return None
        base = map_layers_root() / str(root)
        if kind == "raster_sas_sqlite":
            return None
        if kind == "raster_sas":
            return base / f"z{z + 1}" / f"x{x}" / f"y{y}.{ext}"
        if kind in ("raster_xyz", "raster_pmtiles", "raster_mbtiles"):
            if base.is_file():
                return None
            return base / str(z) / str(x) / f"{y}.{ext}"
        return None

    def tile_bytes_for_layer(
        self,
        layer: Dict[str, Any],
        z: int,
        x: int,
        y: int,
        ext: str = "jpg",
    ) -> Optional[Tuple[bytes, str]]:
        kind = layer.get("layer_kind") or ""
        root = layer.get("storage_root")
        if not root:
            return None
        base = map_layers_root() / str(root)
        if kind == "raster_sas_sqlite":
            return sas_sqlite_tile_response(base, z, x, y)
        tile_path = self.tile_path_for_layer(layer, z, x, y, ext=ext)
        if tile_path and tile_path.is_file():
            data = tile_path.read_bytes()
            return data, guess_image_media_type(data)
        return None
