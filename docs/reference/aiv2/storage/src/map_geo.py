"""Geo queries from Mongo tool_outputs (file_meta GPS) intersect accessible artifacts."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .case_importer_support import tool_id_lookup_variants

_META_TOOLS = ("file_meta", "image_meta")
_GEO_TOOL_NAMES: Set[str] = set()
for _t in _META_TOOLS:
    _GEO_TOOL_NAMES.update(tool_id_lookup_variants(_t))

# Auth-scoped geo tile cache: key includes permission fingerprint so revoked
# access cannot resurrect points from another user's / earlier scope cache.
_TILE_CACHE: Dict[str, Tuple[float, bytes, str]] = {}
_TILE_CACHE_TTL_SEC = 30.0
_TILE_CACHE_MAX = 256
_MVT_EXTENT = 4096
_CLUSTER_ZOOM_MAX = 11


def _extract_location(data: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None

    def _pair(lat: Any, lon: Any, *, source: str) -> Optional[Dict[str, Any]]:
        if lat is None or lon is None:
            return None
        try:
            return {
                "latitude": float(lat),
                "longitude": float(lon),
                "source": source,
            }
        except (TypeError, ValueError):
            return None

    loc = data.get("location")
    if isinstance(loc, dict):
        found = _pair(
            loc.get("latitude", loc.get("lat")),
            loc.get("longitude", loc.get("lon", loc.get("lng"))),
            source=str(loc.get("source") or "file_meta"),
        )
        if found:
            return found

    gps = data.get("gps")
    if isinstance(gps, dict):
        found = _pair(
            gps.get("latitude", gps.get("lat")),
            gps.get("longitude", gps.get("lon", gps.get("lng"))),
            source=str(gps.get("source") or "exif"),
        )
        if found:
            return found
    if isinstance(gps, (list, tuple)) and len(gps) >= 2:
        # Common shapes: [lat, lon] or [lon, lat] — prefer |lat|<=90.
        try:
            a, b = float(gps[0]), float(gps[1])
        except (TypeError, ValueError):
            a = b = None  # type: ignore
        if a is not None and b is not None:
            if abs(a) <= 90 and abs(b) <= 180:
                return {"latitude": a, "longitude": b, "source": "embed"}
            if abs(b) <= 90 and abs(a) <= 180:
                return {"latitude": b, "longitude": a, "source": "embed"}

    # Flat keys some tools dump at top-level.
    found = _pair(
        data.get("latitude", data.get("lat")),
        data.get("longitude", data.get("lon", data.get("lng"))),
        source="flat",
    )
    if found:
        return found
    return None


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _in_bounds(lat: float, lon: float, bounds: Dict[str, float]) -> bool:
    return (
        bounds["south"] <= lat <= bounds["north"]
        and bounds["west"] <= lon <= bounds["east"]
    )


def _minimal_row(artifact_id: str, art: Dict[str, Any], loc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": artifact_id,
        "artifact_id": artifact_id,
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "filename": art.get("filename") or art.get("original_filename") or "",
        "mime_type": art.get("mime_type") or "",
    }


def _tile_bounds(z: int, x: int, y: int) -> Dict[str, float]:
    """Web Mercator XYZ tile → WGS84 bounds (north/south/east/west)."""
    n = 2.0 ** int(z)

    def _lat(yt: float) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * yt / n))))

    return {
        "west": int(x) / n * 360.0 - 180.0,
        "east": (int(x) + 1) / n * 360.0 - 180.0,
        "north": _lat(float(y)),
        "south": _lat(float(y) + 1.0),
    }


def _lonlat_to_tile_xy(
    lon: float, lat: float, z: int, tx: int, ty: int, extent: int = _MVT_EXTENT
) -> Tuple[int, int]:
    n = 2.0 ** int(z)
    lat_c = max(min(lat, 85.05112878), -85.05112878)
    x = (lon + 180.0) / 360.0 * n
    sin_lat = math.sin(math.radians(lat_c))
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * n
    px = int((x - int(tx)) * extent)
    py = int((y - int(ty)) * extent)
    return max(0, min(extent, px)), max(0, min(extent, py))


def _encode_varint(value: int) -> bytes:
    out = bytearray()
    v = int(value)
    while v > 0x7F:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v & 0x7F)
    return bytes(out)


def _encode_key(field_number: int, wire_type: int) -> bytes:
    return _encode_varint((field_number << 3) | wire_type)


def _encode_string(field_number: int, value: str) -> bytes:
    raw = value.encode("utf-8")
    return _encode_key(field_number, 2) + _encode_varint(len(raw)) + raw


def _encode_bytes(field_number: int, value: bytes) -> bytes:
    return _encode_key(field_number, 2) + _encode_varint(len(value)) + value


def _encode_uint32(field_number: int, value: int) -> bytes:
    return _encode_key(field_number, 0) + _encode_varint(value)


def _zigzag(n: int) -> int:
    return (n << 1) ^ (n >> 31)


def _encode_point_geometry(px: int, py: int) -> bytes:
    # MoveTo(1) command = (1 << 3) | 1 = 9, then zigzag dx/dy from (0,0).
    geom = _encode_varint(9) + _encode_varint(_zigzag(px)) + _encode_varint(_zigzag(py))
    return geom


def _encode_mvt_layer(
    name: str,
    features: Sequence[Dict[str, Any]],
    *,
    extent: int = _MVT_EXTENT,
) -> bytes:
    """Minimal Mapbox Vector Tile (one layer, Point + optional cluster props)."""
    keys = ["artifact_id", "count", "cluster"]
    values: List[bytes] = []
    value_index: Dict[Tuple[str, Any], int] = {}

    def _value_msg(val: Any) -> bytes:
        if isinstance(val, bool):
            return _encode_key(7, 0) + _encode_varint(1 if val else 0)
        if isinstance(val, int):
            return _encode_key(6, 0) + _encode_varint(int(val))
        s = str(val)
        raw = s.encode("utf-8")
        return _encode_key(1, 2) + _encode_varint(len(raw)) + raw

    def _ensure_value(val: Any) -> int:
        key = (type(val).__name__, val)
        if key in value_index:
            return value_index[key]
        idx = len(values)
        values.append(_value_msg(val))
        value_index[key] = idx
        return idx

    feature_msgs: List[bytes] = []
    for i, feat in enumerate(features):
        px = int(feat["x"])
        py = int(feat["y"])
        tags = bytearray()
        aid = feat.get("artifact_id")
        if aid:
            tags += _encode_varint(0)  # key index artifact_id
            tags += _encode_varint(_ensure_value(str(aid)))
        count = feat.get("count")
        if count is not None:
            tags += _encode_varint(1)
            tags += _encode_varint(_ensure_value(int(count)))
        if feat.get("cluster"):
            tags += _encode_varint(2)
            tags += _encode_varint(_ensure_value(True))

        geom = _encode_point_geometry(px, py)
        msg = (
            _encode_uint32(1, i + 1)  # id
            + _encode_key(2, 2)
            + _encode_varint(len(tags))
            + bytes(tags)
            + _encode_uint32(3, 1)  # GeomType POINT
            + _encode_bytes(4, geom)
        )
        feature_msgs.append(msg)

    layer = bytearray()
    layer += _encode_uint32(15, 2)  # version
    layer += _encode_string(1, name)
    for fm in feature_msgs:
        layer += _encode_bytes(2, fm)
    for k in keys:
        layer += _encode_string(3, k)
    for vm in values:
        layer += _encode_bytes(4, vm)
    layer += _encode_uint32(5, extent)
    # Tile = repeated layers (field 3)
    return _encode_bytes(3, bytes(layer))


def _tile_cache_get(key: str) -> Optional[Tuple[bytes, str]]:
    hit = _TILE_CACHE.get(key)
    if not hit:
        return None
    expires, body, ctype = hit
    if time.time() > expires:
        _TILE_CACHE.pop(key, None)
        return None
    return body, ctype


def _tile_cache_put(key: str, body: bytes, ctype: str) -> None:
    if len(_TILE_CACHE) >= _TILE_CACHE_MAX:
        # Drop oldest ~25%
        ordered = sorted(_TILE_CACHE.items(), key=lambda kv: kv[1][0])
        for k, _ in ordered[: max(1, _TILE_CACHE_MAX // 4)]:
            _TILE_CACHE.pop(k, None)
    _TILE_CACHE[key] = (time.time() + _TILE_CACHE_TTL_SEC, body, ctype)


def _full_row(artifact_id: str, art: Dict[str, Any], loc: Dict[str, Any], doc: Dict[str, Any]) -> Dict[str, Any]:
    data = doc.get("data") if isinstance(doc.get("data"), dict) else {}
    row = _minimal_row(artifact_id, art, loc)
    row.update(
        {
            "case_id": doc.get("case_id") or art.get("case_id"),
            "file_path": doc.get("file_path") or art.get("storage_path"),
            "location_source": loc.get("source"),
            "created_at": art.get("created_at"),
            "tags": (art.get("metadata") or {}).get("tags") if isinstance(art.get("metadata"), dict) else None,
            "camera": data.get("metadata", {}).get("camera") if isinstance(data.get("metadata"), dict) else None,
        }
    )
    return row


class MapGeoService:
    def __init__(self, storage: Any) -> None:
        self.storage = storage

    async def _accessible_artifact_map(
        self,
        user_id: str,
        *,
        limit: int = 50000,
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        offset = 0
        page = 2000
        while len(out) < limit:
            batch = await self.storage.artifact_accessible_list(
                user_id=user_id,
                limit=min(page, limit - len(out)),
                offset=offset,
            )
            items = batch.get("items") or []
            if not items:
                break
            for art in items:
                aid = str(art.get("id") or art.get("artifact_id") or "")
                if aid:
                    out[aid] = art
            offset += len(items)
            if len(items) < page:
                break
        return out

    async def _gps_docs_for_artifacts(
        self,
        artifact_ids: Sequence[str],
    ) -> Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]:
        """artifact_id -> (mongo_doc, location)."""
        mongo = getattr(self.storage, "mongo", None)
        if mongo is None or mongo.db is None or not artifact_ids:
            return {}

        result: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
        ids = list(artifact_ids)
        chunk = 500
        for i in range(0, len(ids), chunk):
            batch = ids[i : i + chunk]
            cursor = mongo.db.tool_outputs.find(
                {
                    "artifact_id": {"$in": batch},
                    "tool_name": {"$in": list(_GEO_TOOL_NAMES)},
                    "channel": {"$in": ["ui", "rawdata", "embed"]},
                }
            )
            docs = await cursor.to_list(length=len(batch) * 4)
            for doc in docs:
                aid = str(doc.get("artifact_id") or "")
                if not aid or aid in result:
                    continue
                loc = _extract_location(doc.get("data"))
                if loc:
                    result[aid] = (doc, loc)
        return result

    async def _collect_geo_rows_from_index(
        self,
        user_id: str,
        *,
        has_gps: str = "true",
        fields: str = "full",
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        pg = getattr(self.storage, "postgres", None)
        if not pg or not pg.pool:
            return None
        want_gps = (has_gps or "true").lower()
        if want_gps == "false":
            return None
        arts = await self._accessible_artifact_map(user_id, limit=50000)
        if not arts:
            return []
        ids = list(arts.keys())
        rows: List[Dict[str, Any]] = []
        chunk = 1000
        try:
            async with pg.pool.acquire() as conn:
                for i in range(0, len(ids), chunk):
                    batch = ids[i : i + chunk]
                    geo_rows = await conn.fetch(
                        """
                        SELECT artifact_id, latitude, longitude, location_source,
                               filename, mime_type, case_id, created_at
                        FROM artifact_geo WHERE artifact_id = ANY($1::text[])
                        """,
                        batch,
                    )
                    for gr in geo_rows:
                        aid = str(gr["artifact_id"])
                        art = arts.get(aid) or {}
                        if filters:
                            if filters.get("case_id") and str(gr.get("case_id") or "") != str(filters["case_id"]):
                                continue
                            if filters.get("mime_type"):
                                mt = (gr.get("mime_type") or art.get("mime_type") or "").lower()
                                if filters["mime_type"].lower() not in mt:
                                    continue
                        loc = {
                            "latitude": float(gr["latitude"]),
                            "longitude": float(gr["longitude"]),
                            "source": gr.get("location_source") or "artifact_geo",
                        }
                        if fields == "minimal":
                            rows.append(_minimal_row(aid, art, loc))
                        else:
                            rows.append(
                                {
                                    **_minimal_row(aid, art, loc),
                                    "case_id": gr.get("case_id") or art.get("case_id"),
                                    "created_at": gr.get("created_at") or art.get("created_at"),
                                    "location_source": loc.get("source"),
                                }
                            )
            if rows or want_gps == "true":
                return rows
        except Exception:
            return None
        return None

    async def _collect_geo_rows(
        self,
        user_id: str,
        *,
        has_gps: str = "true",
        fields: str = "full",
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        want_gps = (has_gps or "true").lower()
        # Always query Postgres artifacts + Mongo tool_outputs for GPS truth.
        # artifact_geo is an optional accelerator only when it already covers the
        # same (or denser) set — a sparse index must never hide Mongo rows.
        arts = await self._accessible_artifact_map(user_id)
        gps_map = await self._gps_docs_for_artifacts(list(arts.keys()))
        indexed = await self._collect_geo_rows_from_index(
            user_id, has_gps=has_gps, fields=fields, filters=filters
        )

        # Use Postgres index only when it is at least as complete as Mongo GPS.
        if indexed and want_gps != "false" and len(gps_map) > 0 and len(indexed) >= len(gps_map):
            if want_gps == "all":
                gps_ids = {r.get("artifact_id") for r in indexed}
                for aid, art in arts.items():
                    if aid not in gps_ids:
                        indexed.append(
                            {
                                "id": aid,
                                "artifact_id": aid,
                                "latitude": None,
                                "longitude": None,
                                "filename": art.get("filename") or art.get("original_filename") or "",
                                "mime_type": art.get("mime_type") or "",
                                "case_id": art.get("case_id"),
                            }
                        )
            return indexed

        rows: List[Dict[str, Any]] = []

        def _append_no_gps(aid: str, art: Dict[str, Any]) -> None:
            if fields == "minimal":
                rows.append(
                    {
                        "id": aid,
                        "artifact_id": aid,
                        "latitude": None,
                        "longitude": None,
                        "filename": art.get("filename") or art.get("original_filename") or "",
                        "mime_type": art.get("mime_type") or "",
                    }
                )
            else:
                rows.append(
                    {
                        "id": aid,
                        "artifact_id": aid,
                        "latitude": None,
                        "longitude": None,
                        "filename": art.get("filename") or art.get("original_filename") or "",
                        "mime_type": art.get("mime_type") or "",
                        "case_id": art.get("case_id"),
                    }
                )

        if want_gps == "false":
            for aid, art in arts.items():
                if aid in gps_map:
                    continue
                if filters:
                    if filters.get("case_id") and str(art.get("case_id") or "") != str(filters["case_id"]):
                        continue
                    if filters.get("mime_type"):
                        mt = (art.get("mime_type") or "").lower()
                        if filters["mime_type"].lower() not in mt:
                            continue
                _append_no_gps(aid, art)
            return rows

        for aid, (doc, loc) in gps_map.items():
            art = arts.get(aid) or {}
            if filters:
                if filters.get("case_id") and str(doc.get("case_id") or art.get("case_id") or "") != str(filters["case_id"]):
                    continue
                if filters.get("mime_type"):
                    mt = (art.get("mime_type") or "").lower()
                    if filters["mime_type"].lower() not in mt:
                        continue
            if want_gps == "all" or want_gps == "true":
                if fields == "minimal":
                    rows.append(_minimal_row(aid, art, loc))
                else:
                    rows.append(_full_row(aid, art, loc, doc))

        if want_gps == "all":
            for aid, art in arts.items():
                if aid not in gps_map:
                    _append_no_gps(aid, art)
        return rows

    async def geo_fingerprint(
        self,
        user_id: str,
        *,
        has_gps: str = "true",
        case_ids: Optional[List[str]] = None,
        mime_types: Optional[List[str]] = None,
        precomputed_rows: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Cheap RBAC-scoped revalidation token (count + max updated_at).

        Never returns coordinates or artifact id lists. Prefers artifact_geo
        index aggregates over accessible ids; when ``precomputed_rows`` is
        provided (e.g. from geo_files), fingerprint matches that live page.
        """
        if precomputed_rows is not None:
            rows = list(precomputed_rows)
            if case_ids:
                allowed = {str(c) for c in case_ids}
                rows = [r for r in rows if str(r.get("case_id") or "") in allowed]
            if mime_types:
                allowed_m = {m.lower() for m in mime_types}
                rows = [r for r in rows if (r.get("mime_type") or "").lower() in allowed_m]
            with_gps = [r for r in rows if r.get("latitude") is not None]
            accessible_count = len(rows)
            gps_count = len(with_gps)
            max_updated = None
            source = "precomputed"
        else:
            arts = await self._accessible_artifact_map(user_id, limit=50000)
            accessible_count = len(arts)
            ids = list(arts.keys())
            gps_count = 0
            max_updated: Optional[str] = None
            source = "empty"

            pg = getattr(self.storage, "postgres", None)
            if ids and pg and getattr(pg, "pool", None):
                try:
                    chunk = 1000
                    async with pg.pool.acquire() as conn:
                        for i in range(0, len(ids), chunk):
                            batch = ids[i : i + chunk]
                            row = await conn.fetchrow(
                                """
                                SELECT COUNT(*)::bigint AS cnt,
                                       MAX(updated_at) AS max_u
                                FROM artifact_geo
                                WHERE artifact_id = ANY($1::text[])
                                """,
                                batch,
                            )
                            if not row:
                                continue
                            gps_count += int(row["cnt"] or 0)
                            mu = row["max_u"]
                            if mu is not None:
                                iso = mu.isoformat() if hasattr(mu, "isoformat") else str(mu)
                                if max_updated is None or iso > max_updated:
                                    max_updated = iso
                    source = "artifact_geo"
                except Exception:
                    source = "fallback"

            if source != "artifact_geo":
                rows = await self._collect_geo_rows(
                    user_id, has_gps=has_gps or "true", fields="minimal"
                )
                if case_ids:
                    allowed = {str(c) for c in case_ids}
                    rows = [r for r in rows if str(r.get("case_id") or "") in allowed]
                if mime_types:
                    allowed_m = {m.lower() for m in mime_types}
                    rows = [r for r in rows if (r.get("mime_type") or "").lower() in allowed_m]
                with_gps = [r for r in rows if r.get("latitude") is not None]
                gps_count = len(with_gps)
                accessible_count = len(rows)
                source = "live_collect"
            elif case_ids:
                allowed = {str(c) for c in case_ids}
                accessible_count = sum(
                    1 for a in arts.values() if str(a.get("case_id") or "") in allowed
                )

        raw = f"{user_id}|{accessible_count}|{gps_count}|{max_updated or ''}|{has_gps or 'true'}"
        if case_ids:
            raw += "|c:" + ",".join(sorted(str(c) for c in case_ids))
        if mime_types:
            raw += "|m:" + ",".join(sorted(str(m).lower() for m in mime_types))
        etag = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
        return {
            "fingerprint": etag,
            "etag": f'"{etag}"',
            "accessible_count": accessible_count,
            "gps_count": gps_count,
            "max_updated_at": max_updated,
            "source": source,
            "permission_key": hashlib.sha256(
                f"{user_id}|{accessible_count}|{gps_count}".encode("utf-8")
            ).hexdigest()[:24],
        }

    async def geo_files(
        self,
        user_id: str,
        *,
        page: int = 1,
        page_size: int = 100,
        has_gps: str = "true",
        fields: str = "full",
        filename: Optional[str] = None,
        case_ids: Optional[List[str]] = None,
        mime_types: Optional[List[str]] = None,
        fmt: str = "list",
    ) -> Dict[str, Any]:
        rows = await self._collect_geo_rows(user_id, has_gps=has_gps, fields=fields)
        if filename:
            needle = filename.lower()
            rows = [r for r in rows if needle in (r.get("filename") or "").lower()]
        if case_ids:
            allowed = {str(c) for c in case_ids}
            rows = [r for r in rows if str(r.get("case_id") or "") in allowed]
        if mime_types:
            allowed_m = {m.lower() for m in mime_types}
            rows = [r for r in rows if (r.get("mime_type") or "").lower() in allowed_m]

        total = len(rows)
        start = max(0, (page - 1) * page_size)
        end = start + page_size
        page_rows = rows[start:end]

        fp = await self.geo_fingerprint(
            user_id,
            has_gps=has_gps,
            case_ids=case_ids,
            mime_types=mime_types,
            precomputed_rows=rows,
        )
        fp_fields = {
            "fingerprint": fp.get("fingerprint"),
            "etag": fp.get("etag"),
            "permission_key": fp.get("permission_key"),
            "gps_count": fp.get("gps_count"),
            "max_updated_at": fp.get("max_updated_at"),
        }

        if (fmt or "list").lower() == "geojson":
            features = []
            for r in page_rows:
                lat, lon = r.get("latitude"), r.get("longitude")
                if lat is None or lon is None:
                    continue
                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                        "properties": {
                            "artifact_id": r.get("artifact_id") or r.get("id"),
                            "filename": r.get("filename"),
                            "mime_type": r.get("mime_type"),
                            "case_id": r.get("case_id"),
                        },
                    }
                )
            geojson = {"type": "FeatureCollection", "features": features}
            return {
                "type": "FeatureCollection",
                "features": features,
                "geojson": geojson,
                "total": total,
                "page": page,
                "per_page": page_size,
                **fp_fields,
            }

        return {
            "files": page_rows,
            "items": page_rows,
            "total": total,
            "page": page,
            "per_page": page_size,
            "page_size": page_size,
            **fp_fields,
        }

    async def geo_tile(
        self,
        user_id: str,
        *,
        z: int,
        x: int,
        y: int,
        fmt: str = "mvt",
    ) -> Dict[str, Any]:
        """Auth-scoped tile for visible z/x/y — live accessible GPS only.

        Cache key includes permission_key derived from the caller's accessible
        GPS fingerprint so tiles are never shared across permission scopes.
        """
        z_i, x_i, y_i = int(z), int(x), int(y)
        if z_i < 0 or z_i > 22 or x_i < 0 or y_i < 0 or x_i >= 2**z_i or y_i >= 2**z_i:
            raise ValueError("invalid_tile")

        fp = await self.geo_fingerprint(user_id, has_gps="true")
        perm = str(fp.get("permission_key") or user_id)
        want_mvt = (fmt or "mvt").lower() in ("mvt", "pbf", "vector")
        cache_key = f"{perm}|{z_i}/{x_i}/{y_i}|{'mvt' if want_mvt else 'geojson'}"
        cached = _tile_cache_get(cache_key)
        if cached:
            body, ctype = cached
            return {
                "content": body,
                "content_type": ctype,
                "fingerprint": fp.get("fingerprint"),
                "permission_key": perm,
                "cached": True,
                "z": z_i,
                "x": x_i,
                "y": y_i,
            }

        bounds = _tile_bounds(z_i, x_i, y_i)
        rows = await self._collect_geo_rows(user_id, has_gps="true", fields="minimal")
        points: List[Dict[str, Any]] = []
        for r in rows:
            lat, lon = r.get("latitude"), r.get("longitude")
            if lat is None or lon is None:
                continue
            try:
                lat_f, lon_f = float(lat), float(lon)
            except (TypeError, ValueError):
                continue
            if not _in_bounds(lat_f, lon_f, bounds):
                continue
            points.append(
                {
                    "artifact_id": r.get("artifact_id") or r.get("id"),
                    "latitude": lat_f,
                    "longitude": lon_f,
                }
            )

        features_out: List[Dict[str, Any]] = []
        if z_i <= _CLUSTER_ZOOM_MAX and points:
            # Grid aggregate inside the tile (auth filter already applied).
            grid = 8
            cells: Dict[Tuple[int, int], Dict[str, Any]] = {}
            lat_span = max(bounds["north"] - bounds["south"], 1e-12)
            lon_span = max(bounds["east"] - bounds["west"], 1e-12)
            for p in points:
                gx = int((p["longitude"] - bounds["west"]) / lon_span * grid)
                gy = int((bounds["north"] - p["latitude"]) / lat_span * grid)
                gx = max(0, min(grid - 1, gx))
                gy = max(0, min(grid - 1, gy))
                cell = cells.setdefault(
                    (gx, gy),
                    {"count": 0, "lat_sum": 0.0, "lon_sum": 0.0, "artifact_id": p["artifact_id"]},
                )
                cell["count"] += 1
                cell["lat_sum"] += p["latitude"]
                cell["lon_sum"] += p["longitude"]
            for cell in cells.values():
                n = cell["count"]
                lat_c = cell["lat_sum"] / n
                lon_c = cell["lon_sum"] / n
                px, py = _lonlat_to_tile_xy(lon_c, lat_c, z_i, x_i, y_i)
                features_out.append(
                    {
                        "x": px,
                        "y": py,
                        "latitude": lat_c,
                        "longitude": lon_c,
                        "count": n,
                        "cluster": n > 1,
                        "artifact_id": None if n > 1 else cell.get("artifact_id"),
                    }
                )
        else:
            for p in points:
                px, py = _lonlat_to_tile_xy(p["longitude"], p["latitude"], z_i, x_i, y_i)
                features_out.append(
                    {
                        "x": px,
                        "y": py,
                        "latitude": p["latitude"],
                        "longitude": p["longitude"],
                        "count": 1,
                        "cluster": False,
                        "artifact_id": p["artifact_id"],
                    }
                )

        if want_mvt:
            body = _encode_mvt_layer("artifact_geo", features_out)
            ctype = "application/vnd.mapbox-vector-tile"
        else:
            gj_features = []
            for f in features_out:
                props: Dict[str, Any] = {"count": f.get("count") or 1, "cluster": bool(f.get("cluster"))}
                if f.get("artifact_id"):
                    props["artifact_id"] = f["artifact_id"]
                gj_features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [float(f["longitude"]), float(f["latitude"])],
                        },
                        "properties": props,
                    }
                )
            body = json.dumps(
                {"type": "FeatureCollection", "features": gj_features},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            ctype = "application/geo+json"

        _tile_cache_put(cache_key, body, ctype)
        return {
            "content": body,
            "content_type": ctype,
            "fingerprint": fp.get("fingerprint"),
            "permission_key": perm,
            "cached": False,
            "feature_count": len(features_out),
            "z": z_i,
            "x": x_i,
            "y": y_i,
        }

    async def geo_within(
        self,
        user_id: str,
        *,
        bounds: Optional[Dict[str, float]] = None,
        center: Optional[Dict[str, float]] = None,
        radius_meters: Optional[float] = None,
        fields: str = "minimal",
        limit: int = 5000,
    ) -> Dict[str, Any]:
        rows = await self._collect_geo_rows(user_id, fields=fields)
        out: List[Dict[str, Any]] = []
        for r in rows:
            lat, lon = r.get("latitude"), r.get("longitude")
            if lat is None or lon is None:
                continue
            try:
                lat_f, lon_f = float(lat), float(lon)
            except (TypeError, ValueError):
                continue
            if bounds and _in_bounds(lat_f, lon_f, bounds):
                out.append(r)
            elif center and radius_meters is not None:
                d = _haversine_m(
                    lat_f,
                    lon_f,
                    float(center["lat"]),
                    float(center["lng"]),
                )
                if d <= float(radius_meters):
                    r = dict(r)
                    r["distance_m"] = round(d, 2)
                    out.append(r)
            if len(out) >= limit:
                break
        return {"files": out[:limit], "count": min(len(out), limit)}

    async def geo_stats(
        self,
        user_id: str,
        *,
        has_gps: Optional[str] = None,
        case_ids: Optional[List[str]] = None,
        mime_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        hg = has_gps if has_gps and has_gps != "all" else "true"
        rows = await self._collect_geo_rows(user_id, has_gps=hg, fields="minimal")
        if case_ids:
            allowed = {str(c) for c in case_ids}
            rows = [r for r in rows if str(r.get("case_id") or "") in allowed]
        if mime_types:
            allowed_m = {m.lower() for m in mime_types}
            rows = [r for r in rows if (r.get("mime_type") or "").lower() in allowed_m]

        with_gps = [r for r in rows if r.get("latitude") is not None]
        lats = [float(r["latitude"]) for r in with_gps]
        lons = [float(r["longitude"]) for r in with_gps]
        bounds = None
        if lats and lons:
            bounds = {
                "north": max(lats),
                "south": min(lats),
                "east": max(lons),
                "west": min(lons),
            }
        by_mime: Dict[str, int] = {}
        for r in with_gps:
            mt = (r.get("mime_type") or "unknown").split(";")[0]
            by_mime[mt] = by_mime.get(mt, 0) + 1
        return {
            "total_files": len(rows),
            "with_gps": len(with_gps),
            "without_gps": len(rows) - len(with_gps),
            "bounds": bounds,
            "by_mime_type": by_mime,
        }

    async def geo_cluster(
        self,
        user_id: str,
        *,
        bounds: Dict[str, float],
        zoom: int = 10,
        grid_size: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        rows = await self._collect_geo_rows(
            user_id, fields="minimal", filters=filters or None
        )
        lat_span = max(bounds["north"] - bounds["south"], 1e-9)
        lon_span = max(bounds["east"] - bounds["west"], 1e-9)
        cells: Dict[str, Dict[str, Any]] = {}

        for r in rows:
            lat, lon = r.get("latitude"), r.get("longitude")
            if lat is None or lon is None:
                continue
            try:
                lat_f, lon_f = float(lat), float(lon)
            except (TypeError, ValueError):
                continue
            if not _in_bounds(lat_f, lon_f, bounds):
                continue
            gx = int((lon_f - bounds["west"]) / lon_span * grid_size)
            gy = int((bounds["north"] - lat_f) / lat_span * grid_size)
            gx = max(0, min(grid_size - 1, gx))
            gy = max(0, min(grid_size - 1, gy))
            key = f"{gx}:{gy}"
            cell = cells.setdefault(
                key,
                {
                    "cell": key,
                    "count": 0,
                    "lat_sum": 0.0,
                    "lon_sum": 0.0,
                    "ids": [],
                },
            )
            cell["count"] += 1
            cell["lat_sum"] += lat_f
            cell["lon_sum"] += lon_f
            if len(cell["ids"]) < 50:
                cell["ids"].append(r.get("artifact_id"))

        clusters = []
        for cell in cells.values():
            n = cell["count"]
            clusters.append(
                {
                    "latitude": cell["lat_sum"] / n,
                    "longitude": cell["lon_sum"] / n,
                    "count": n,
                    "ids": cell["ids"],
                }
            )
        return {"clusters": clusters, "zoom": zoom, "grid_size": grid_size}

    async def geo_details(
        self,
        user_id: str,
        *,
        ids: List[str],
    ) -> Dict[str, Any]:
        arts = await self._accessible_artifact_map(user_id, limit=100000)
        gps_map = await self._gps_docs_for_artifacts(ids)
        details = []
        for aid in ids:
            if aid not in arts:
                continue
            art = arts[aid]
            filename = art.get("filename") or art.get("original_filename") or ""
            mime = art.get("mime_type") or ""
            media_type = (art.get("media_type") or "").strip().lower() or None
            if not mime and filename:
                import mimetypes as _mt

                guessed, _ = _mt.guess_type(str(filename))
                mime = guessed or ""
            if not media_type and mime.startswith("image/"):
                media_type = "image"
            if not media_type and str(filename).lower().endswith(
                (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic")
            ):
                media_type = "image"
                if not mime:
                    mime = "image/jpeg"
            entry: Dict[str, Any] = {
                "id": aid,
                "artifact_id": aid,
                "filename": filename,
                "original_filename": art.get("original_filename") or filename,
                "mime_type": mime or None,
                "type": mime or None,
                "media_type": media_type,
                "storage_path": art.get("storage_path"),
                "file_path": art.get("storage_path"),
                "path": art.get("storage_path"),
                "size": art.get("file_size_bytes") or art.get("size_bytes"),
                "created_at": art.get("created_at"),
                "case_id": art.get("case_id"),
                # Gateway-proxied thumbnail — FE loads with auth (not a bare <img>).
                "thumbnail_url": f"/storage/files/{aid}/thumbnail?width=320&height=320&format=jpeg&quality=80",
            }
            if aid in gps_map:
                doc, loc = gps_map[aid]
                data = doc.get("data") if isinstance(doc.get("data"), dict) else {}
                entry["location"] = loc
                entry["latitude"] = loc.get("latitude")
                entry["longitude"] = loc.get("longitude")
                entry["location_source"] = loc.get("source")
                entry["gps"] = {
                    "latitude": loc.get("latitude"),
                    "longitude": loc.get("longitude"),
                    "source": loc.get("source"),
                }
                entry["case_id"] = doc.get("case_id") or entry.get("case_id")
                # Prefer inline preview from tool_outputs when present (no extra round-trip).
                thumb = data.get("thumbnail") or data.get("thumbnail_url") or data.get("preview_url")
                if not thumb:
                    preview = data.get("preview")
                    if isinstance(preview, str) and preview.strip():
                        thumb = (
                            preview
                            if preview.startswith("data:") or preview.startswith("http")
                            else f"data:image/jpeg;base64,{preview}"
                        )
                    elif isinstance(preview, dict):
                        b64 = preview.get("base64") or preview.get("data") or preview.get("b64")
                        pmime = preview.get("mime_type") or preview.get("mime") or "image/jpeg"
                        if isinstance(b64, str) and b64.strip():
                            thumb = b64 if b64.startswith("data:") else f"data:{pmime};base64,{b64}"
                if thumb:
                    entry["thumbnail"] = thumb
                # Enrich mime from tool payload when artifact row is incomplete.
                if not entry.get("mime_type"):
                    file_info = data.get("file") if isinstance(data.get("file"), dict) else {}
                    meta_dict = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
                    tip = (
                        data.get("mime_type")
                        or file_info.get("mime_type")
                        or meta_dict.get("mime_type")
                    )
                    if tip:
                        entry["mime_type"] = tip
                        entry["type"] = tip
                        if str(tip).startswith("image/"):
                            entry["media_type"] = "image"
                cam = data.get("camera")
                if isinstance(cam, dict):
                    entry["camera"] = cam.get("model") or cam.get("make") or json.dumps(cam, ensure_ascii=False)[:80]
                elif isinstance(cam, str):
                    entry["camera"] = cam
                meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
                if not entry.get("camera") and isinstance(meta.get("camera"), str):
                    entry["camera"] = meta.get("camera")
            details.append(entry)
        return {"files": details, "items": details}

    async def geo_export(
        self,
        user_id: str,
        *,
        fmt: str = "csv",
        ids: Optional[List[str]] = None,
        include_all: bool = False,
    ) -> Dict[str, Any]:
        if include_all:
            rows = await self._collect_geo_rows(user_id, fields="full")
        else:
            rows = (await self.geo_details(user_id, ids=ids or [])).get("items") or []

        fmt_l = (fmt or "csv").lower()
        if fmt_l == "json":
            content = json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")
            ct = "application/json"
        elif fmt_l == "kml":
            parts = [
                '<?xml version="1.0" encoding="UTF-8"?>',
                '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
            ]
            for r in rows:
                lat, lon = r.get("latitude"), r.get("longitude")
                if lat is None or lon is None:
                    continue
                name = (r.get("filename") or r.get("id") or "point").replace("&", "&amp;")
                parts.append(f"<Placemark><name>{name}</name>")
                parts.append(f"<Point><coordinates>{lon},{lat},0</coordinates></Point>")
                parts.append("</Placemark>")
            parts.append("</Document></kml>")
            content = "\n".join(parts).encode("utf-8")
            ct = "application/vnd.google-earth.kml+xml"
        else:
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["artifact_id", "filename", "latitude", "longitude", "mime_type", "case_id"])
            for r in rows:
                w.writerow(
                    [
                        r.get("artifact_id") or r.get("id"),
                        r.get("filename") or "",
                        r.get("latitude"),
                        r.get("longitude"),
                        r.get("mime_type") or "",
                        r.get("case_id") or "",
                    ]
                )
            content = buf.getvalue().encode("utf-8")
            ct = "text/csv"

        return {
            "format": fmt_l,
            "content_type": ct,
            "size_bytes": len(content),
            "content_base64": base64.b64encode(content).decode("ascii"),
        }

    async def geo_timeline(
        self,
        user_id: str,
        *,
        granularity: str = "day",
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        case_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        rows = await self._collect_geo_rows(user_id, fields="full")
        if case_ids:
            allowed = {str(c) for c in case_ids}
            rows = [r for r in rows if str(r.get("case_id") or "") in allowed]

        buckets: Dict[str, int] = {}
        for r in rows:
            if r.get("latitude") is None:
                continue
            ts = r.get("created_at")
            if not ts:
                key = "unknown"
            else:
                s = str(ts)[:10] if granularity == "day" else str(ts)[:7]
                key = s
            buckets[key] = buckets.get(key, 0) + 1
        timeline = [{"period": k, "count": v} for k, v in sorted(buckets.items())]
        return {"granularity": granularity, "timeline": timeline}

    async def geo_movement(
        self,
        user_id: str,
        *,
        stop_radius: int = 100,
        stop_min_files: int = 2,
        limit: int = 10000,
        case_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        rows = await self._collect_geo_rows(user_id, fields="full")
        if case_ids:
            allowed = {str(c) for c in case_ids}
            rows = [r for r in rows if str(r.get("case_id") or "") in allowed]
        points = [r for r in rows if r.get("latitude") is not None][:limit]
        points.sort(key=lambda x: str(x.get("created_at") or ""))

        stops: List[Dict[str, Any]] = []
        current: List[Dict[str, Any]] = []
        for p in points:
            if not current:
                current = [p]
                continue
            lat0 = float(current[0]["latitude"])
            lon0 = float(current[0]["longitude"])
            lat1 = float(p["latitude"])
            lon1 = float(p["longitude"])
            if _haversine_m(lat0, lon0, lat1, lon1) <= stop_radius:
                current.append(p)
            else:
                if len(current) >= stop_min_files:
                    stops.append(
                        {
                            "latitude": sum(float(x["latitude"]) for x in current) / len(current),
                            "longitude": sum(float(x["longitude"]) for x in current) / len(current),
                            "file_count": len(current),
                            "artifact_ids": [x.get("artifact_id") for x in current],
                        }
                    )
                current = [p]
        if len(current) >= stop_min_files:
            stops.append(
                {
                    "latitude": sum(float(x["latitude"]) for x in current) / len(current),
                    "longitude": sum(float(x["longitude"]) for x in current) / len(current),
                    "file_count": len(current),
                    "artifact_ids": [x.get("artifact_id") for x in current],
                }
            )
        return {"stops": stops, "stop_radius_m": stop_radius}
