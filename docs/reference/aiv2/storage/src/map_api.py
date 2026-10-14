"""Map geo + layer catalog HTTP routes for Storage."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from .map_geo import MapGeoService
from .map_layers import MapLayersService, map_layers_root
from .map_rbac import resolve_map_layer_rbac

router = APIRouter(tags=["map"])

_BYTE_RANGE_RE = re.compile(r"^bytes=(\d+)-(\d*)$", re.IGNORECASE)


def _parse_byte_range(range_header: str, file_size: int) -> Optional[Tuple[int, int]]:
    m = _BYTE_RANGE_RE.match((range_header or "").strip())
    if not m or file_size <= 0:
        return None
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else file_size - 1
    if start < 0 or start >= file_size:
        return None
    end = min(end, file_size - 1)
    if end < start:
        return None
    return start, end


def _read_file_range(path: Path, start: int, length: int) -> bytes:
    with open(path, "rb") as handle:
        handle.seek(start)
        return handle.read(length)


def _storage_from_request(request: Request):
    return request.app.state.storage


def _require_user_id(request: Request) -> str:
    uid = request.headers.get("X-User-Id")
    if not uid or not str(uid).strip():
        raise HTTPException(status_code=400, detail="X-User-Id required")
    return str(uid).strip()


def _geo_svc(request: Request) -> MapGeoService:
    return MapGeoService(_storage_from_request(request))


def _layers_svc(request: Request) -> MapLayersService:
    return MapLayersService(_storage_from_request(request))


def _user_roles(request: Request) -> Optional[str]:
    value = request.headers.get("X-User-Roles")
    return value.strip() if value and value.strip() else None


def _map_caller(request: Request) -> tuple[str, Optional[str]]:
    return _require_user_id(request), _user_roles(request)


def _split_csv(value: Optional[str]) -> Optional[List[str]]:
    if not value or not str(value).strip():
        return None
    return [s.strip() for s in str(value).split(",") if s.strip()]


async def _require_shared_import(request: Request) -> None:
    """Block MAP_LAYERS_ROOT register/detect unless caller may import shared layers."""
    user_id, user_roles = _map_caller(request)
    rbac = await resolve_map_layer_rbac(user_id, user_roles=user_roles)
    if not rbac.can_import_shared():
        raise HTTPException(status_code=403, detail="map_layers_import_shared_required")


async def _require_register_api(request: Request) -> None:
    """Block vector_api registration unless caller may register outbound API layers."""
    user_id, user_roles = _map_caller(request)
    rbac = await resolve_map_layer_rbac(user_id, user_roles=user_roles)
    if not rbac.can_register_api():
        raise HTTPException(status_code=403, detail="map_layers_register_api_required")


def _layer_with_probe(svc: MapLayersService, layer: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(layer)
    out["data_available"] = svc.layer_data_available(layer)
    return out


class GeoWithinBody(BaseModel):
    bounds: Optional[Dict[str, float]] = None
    center: Optional[Dict[str, float]] = None
    radiusMeters: Optional[float] = Field(None, alias="radius_meters")
    fields: str = "minimal"
    limit: int = 5000

    model_config = {"populate_by_name": True}


class GeoClusterBody(BaseModel):
    bounds: Dict[str, float]
    zoom: int = 10
    gridSize: int = Field(10, alias="grid_size")
    filters: Optional[Dict[str, Any]] = None

    model_config = {"populate_by_name": True}


class GeoExportBody(BaseModel):
    format: str = "csv"
    ids: Optional[List[str]] = None
    includeAll: bool = Field(False, alias="include_all")

    model_config = {"populate_by_name": True}


class GeoDetailsBody(BaseModel):
    ids: List[str] = Field(default_factory=list)


class LayerCreateBody(BaseModel):
    name: str
    layer_kind: str
    source_type: str = "file"
    storage_root: Optional[str] = None
    source_url: Optional[str] = None
    source_config: Optional[Dict[str, Any]] = None
    bounds: Optional[Any] = None
    min_zoom: Optional[int] = None
    max_zoom: Optional[int] = None


class LayerDetectBody(BaseModel):
    path: str


class LayerDetectUrlBody(BaseModel):
    url: str


class LayerApiProbeBody(BaseModel):
    """Read-only probe of a user-supplied GeoJSON API (no persistence)."""

    source_config: Optional[Dict[str, Any]] = None
    url: Optional[str] = None
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    query: Optional[Dict[str, Any]] = None
    auth: Optional[Dict[str, Any]] = None
    jmespath: Optional[str] = None
    timeout_ms: Optional[int] = None


class LayerPatchBody(BaseModel):
    name: Optional[str] = None
    enabled: Optional[bool] = None
    sort_order: Optional[int] = None
    style: Optional[Dict[str, Any]] = None
    source_config: Optional[Dict[str, Any]] = None
    bounds: Optional[Any] = None
    min_zoom: Optional[int] = None
    max_zoom: Optional[int] = None


# -------------------- Geo --------------------


@router.get("/map/geo/fingerprint")
async def map_geo_fingerprint(
    request: Request,
    has_gps: str = Query("true"),
    case_ids: Optional[str] = None,
    mime_types: Optional[str] = None,
):
    """Cheap RBAC-scoped geo fingerprint for client skip-reload (no point bodies)."""
    user_id = _require_user_id(request)
    fp = await _geo_svc(request).geo_fingerprint(
        user_id,
        has_gps=has_gps,
        case_ids=_split_csv(case_ids),
        mime_types=_split_csv(mime_types),
    )
    headers = {
        "ETag": str(fp.get("etag") or ""),
        "X-Geo-Fingerprint": str(fp.get("fingerprint") or ""),
        "Cache-Control": "private, no-store",
    }
    return Response(
        content=json.dumps(fp, ensure_ascii=False).encode("utf-8"),
        media_type="application/json",
        headers={k: v for k, v in headers.items() if v},
    )


@router.get("/map/geo/files")
async def map_geo_files(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=5000),
    per_page: Optional[int] = Query(None, ge=1, le=5000),
    has_gps: str = Query("true"),
    fields: str = Query("full"),
    filename: Optional[str] = None,
    case_ids: Optional[str] = None,
    mime_types: Optional[str] = None,
    format: Optional[str] = Query(None, alias="format"),
):
    user_id = _require_user_id(request)
    ps = per_page or page_size
    payload = await _geo_svc(request).geo_files(
        user_id,
        page=page,
        page_size=ps,
        has_gps=has_gps,
        fields=fields,
        filename=filename,
        case_ids=_split_csv(case_ids),
        mime_types=_split_csv(mime_types),
        fmt=format or "list",
    )
    headers = {
        "ETag": str(payload.get("etag") or ""),
        "X-Geo-Fingerprint": str(payload.get("fingerprint") or ""),
        "Cache-Control": "private, no-store",
    }
    return Response(
        content=json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"),
        media_type="application/json",
        headers={k: v for k, v in headers.items() if v},
    )


@router.head("/map/geo/files")
async def map_geo_files_head(
    request: Request,
    has_gps: str = Query("true"),
    case_ids: Optional[str] = None,
    mime_types: Optional[str] = None,
):
    """HEAD revalidation: ETag / X-Geo-Fingerprint without a body."""
    user_id = _require_user_id(request)
    fp = await _geo_svc(request).geo_fingerprint(
        user_id,
        has_gps=has_gps,
        case_ids=_split_csv(case_ids),
        mime_types=_split_csv(mime_types),
    )
    return Response(
        status_code=200,
        headers={
            "ETag": str(fp.get("etag") or ""),
            "X-Geo-Fingerprint": str(fp.get("fingerprint") or ""),
            "X-Geo-Total": str(fp.get("gps_count") if fp.get("gps_count") is not None else ""),
            "Cache-Control": "private, no-store",
        },
    )


@router.get("/map/geo/tiles/{z}/{x}/{y}.mvt")
@router.get("/map/geo/tiles/{z}/{x}/{y}")
async def map_geo_tiles(
    request: Request,
    z: int,
    x: int,
    y: int,
    format: Optional[str] = Query(None, alias="format"),
):
    """Auth-scoped progressive tiles (MVT or GeoJSON). Cache key includes permission scope."""
    user_id = _require_user_id(request)
    path = request.url.path
    want = (format or "").lower()
    if not want:
        want = "mvt" if path.endswith(".mvt") else "geojson"
        accept = (request.headers.get("accept") or "").lower()
        if "geo+json" in accept or "application/json" in accept:
            want = "geojson"
        elif "mapbox-vector-tile" in accept or "mvt" in accept or "pbf" in accept:
            want = "mvt"
    try:
        tile = await _geo_svc(request).geo_tile(user_id, z=z, x=x, y=y, fmt=want)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    headers = {
        "Cache-Control": "private, max-age=30",
        "X-Geo-Fingerprint": str(tile.get("fingerprint") or ""),
        "X-Geo-Permission-Key": str(tile.get("permission_key") or ""),
        "X-Geo-Tile-Cached": "1" if tile.get("cached") else "0",
    }
    return Response(
        content=tile["content"],
        media_type=tile["content_type"],
        headers=headers,
    )


@router.get("/map/geo/within")
async def map_geo_within_get(
    request: Request,
    north: Optional[float] = None,
    south: Optional[float] = None,
    east: Optional[float] = None,
    west: Optional[float] = None,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    radius_meters: Optional[float] = None,
    fields: str = "minimal",
    limit: int = Query(5000, ge=1, le=50000),
):
    user_id = _require_user_id(request)
    bounds = None
    center = None
    if north is not None and south is not None and east is not None and west is not None:
        bounds = {"north": north, "south": south, "east": east, "west": west}
    if lat is not None and lng is not None:
        center = {"lat": lat, "lng": lng}
    if not bounds and not center:
        raise HTTPException(status_code=400, detail="Provide bounds or lat/lng")
    return await _geo_svc(request).geo_within(
        user_id,
        bounds=bounds,
        center=center,
        radius_meters=radius_meters,
        fields=fields,
        limit=limit,
    )


@router.post("/map/geo/within")
async def map_geo_within_post(request: Request, body: GeoWithinBody):
    user_id = _require_user_id(request)
    if not body.bounds and not body.center:
        raise HTTPException(status_code=400, detail="Provide bounds or center")
    return await _geo_svc(request).geo_within(
        user_id,
        bounds=body.bounds,
        center=body.center,
        radius_meters=body.radiusMeters,
        fields=body.fields,
        limit=body.limit,
    )


@router.get("/map/geo/stats")
async def map_geo_stats(
    request: Request,
    has_gps: Optional[str] = Query(None, alias="hasGps"),
    case_ids: Optional[str] = Query(None, alias="caseIds"),
    mime_types: Optional[str] = Query(None, alias="mimeTypes"),
):
    user_id = _require_user_id(request)
    stats = await _geo_svc(request).geo_stats(
        user_id,
        has_gps=has_gps,
        case_ids=_split_csv(case_ids),
        mime_types=_split_csv(mime_types),
    )
    return {"stats": stats, **stats}


@router.post("/map/geo/cluster")
async def map_geo_cluster(request: Request, body: GeoClusterBody):
    user_id = _require_user_id(request)
    return await _geo_svc(request).geo_cluster(
        user_id,
        bounds=body.bounds,
        zoom=body.zoom,
        grid_size=body.gridSize,
        filters=body.filters,
    )


@router.get("/map/geo/cluster")
async def map_geo_cluster_get(
    request: Request,
    north: float,
    south: float,
    east: float,
    west: float,
    zoom: int = 10,
    grid_size: int = 10,
):
    user_id = _require_user_id(request)
    return await _geo_svc(request).geo_cluster(
        user_id,
        bounds={"north": north, "south": south, "east": east, "west": west},
        zoom=zoom,
        grid_size=grid_size,
    )


@router.post("/map/geo/details")
async def map_geo_details(request: Request, body: GeoDetailsBody):
    user_id = _require_user_id(request)
    if not body.ids:
        raise HTTPException(status_code=400, detail="ids required")
    return await _geo_svc(request).geo_details(user_id, ids=body.ids)


@router.post("/map/geo/export")
async def map_geo_export(request: Request, body: GeoExportBody):
    user_id = _require_user_id(request)
    return await _geo_svc(request).geo_export(
        user_id,
        fmt=body.format,
        ids=body.ids,
        include_all=body.includeAll,
    )


@router.get("/map/geo/timeline")
async def map_geo_timeline(
    request: Request,
    granularity: str = "day",
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    case_ids: Optional[str] = None,
):
    user_id = _require_user_id(request)
    return await _geo_svc(request).geo_timeline(
        user_id,
        granularity=granularity,
        date_start=date_start,
        date_end=date_end,
        case_ids=_split_csv(case_ids),
    )


@router.get("/map/geo/movement")
async def map_geo_movement(
    request: Request,
    stop_radius: int = Query(100, ge=1),
    stop_min_files: int = Query(2, ge=1),
    limit: int = Query(10000, ge=1, le=100000),
    case_ids: Optional[str] = None,
):
    user_id = _require_user_id(request)
    return await _geo_svc(request).geo_movement(
        user_id,
        stop_radius=stop_radius,
        stop_min_files=stop_min_files,
        limit=limit,
        case_ids=_split_csv(case_ids),
    )


# -------------------- Layers --------------------


@router.get("/users/{user_id}/map/layers/accessible")
async def map_layers_accessible(
    request: Request,
    user_id: str,
    layer_kind: Optional[str] = None,
    enabled_only: bool = True,
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    caller = _require_user_id(request)
    if caller != user_id:
        raise HTTPException(status_code=403, detail="user_id_mismatch")
    return await _layers_svc(request).layer_accessible_list(
        user_id,
        user_roles=_user_roles(request),
        layer_kind=layer_kind,
        enabled_only=enabled_only,
        limit=limit,
        offset=offset,
    )


@router.get("/map/layers")
async def map_layers_list(
    request: Request,
    layer_kind: Optional[str] = None,
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    user_id = _require_user_id(request)
    return await _layers_svc(request).layer_accessible_list(
        user_id,
        user_roles=_user_roles(request),
        layer_kind=layer_kind,
        enabled_only=False,
        limit=limit,
        offset=offset,
    )


@router.post("/map/layers")
async def map_layers_create(request: Request, body: LayerCreateBody):
    user_id = _require_user_id(request)
    if body.storage_root:
        await _require_shared_import(request)
    if str(body.layer_kind or "").strip() == "vector_api":
        await _require_register_api(request)
    svc = _layers_svc(request)
    source_config = dict(body.source_config or {})
    if body.storage_root and not source_config.get("shared"):
        source_config["shared"] = True
    try:
        layer = await svc.find_or_create_layer(
            owner_user_id=user_id,
            name=body.name,
            layer_kind=body.layer_kind,
            source_type=body.source_type,
            storage_root=body.storage_root,
            source_url=body.source_url,
            source_config=source_config or None,
            bounds=body.bounds,
            min_zoom=body.min_zoom,
            max_zoom=body.max_zoom,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if str(layer.get("layer_kind") or "").startswith("streetview"):
        await svc.scan_streetview_layer(str(layer["id"]))
    return _layer_with_probe(svc, layer)


@router.post("/map/layers/api/probe")
async def map_layers_api_probe(request: Request, body: LayerApiProbeBody):
    """Discover/validate a user-supplied GeoJSON API without writing to the catalog."""
    _require_user_id(request)
    from .map_layer_api import probe_api_layer

    if isinstance(body.source_config, dict) and body.source_config:
        cfg = dict(body.source_config)
    else:
        url = (body.url or "").strip()
        if not url:
            raise HTTPException(status_code=400, detail="url_or_source_config_required")
        http: Dict[str, Any] = {
            "method": (body.method or "GET").upper(),
            "url": url,
            "headers": dict(body.headers or {}),
            "query": dict(body.query or {}),
        }
        if body.timeout_ms is not None:
            http["timeout_ms"] = body.timeout_ms
        cfg = {
            "http": http,
            "auth": dict(body.auth or {"type": "none"}),
            "response": {"format": "geojson", "jmespath": body.jmespath},
        }
    return await probe_api_layer(cfg)


@router.post("/map/layers/detect")
async def map_layers_detect(request: Request, body: LayerDetectBody):
    await _require_shared_import(request)
    return _layers_svc(request).detect_path(body.path)


@router.post("/map/layers/detect-url")
async def map_layers_detect_url(request: Request, body: LayerDetectUrlBody):
    _require_user_id(request)
    return _layers_svc(request).detect_url(body.url)


@router.get("/map/layers/browse")
async def map_layers_browse(
    request: Request,
    path: Optional[str] = None,
):
    await _require_shared_import(request)
    listing = _layers_svc(request)._browse_layers_root(path)
    if listing.get("error"):
        raise HTTPException(status_code=400, detail=listing.get("error"))
    return listing


@router.get("/map/layers/{layer_id}")
async def map_layers_get(request: Request, layer_id: str):
    user_id, user_roles = _map_caller(request)
    access = await _layers_svc(request).layer_is_accessible(
        user_id=user_id, layer_id=layer_id, action="read", user_roles=user_roles
    )
    if not access.get("accessible"):
        raise HTTPException(status_code=403, detail="not_authorized_for_layer")
    async with _layers_svc(request)._pool().acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM map_layers WHERE id = $1", uuid.UUID(layer_id))
    if not row:
        raise HTTPException(status_code=404, detail="layer_not_found")
    return _layers_svc(request)._row_to_dict(row)


@router.patch("/map/layers/{layer_id}")
async def map_layers_patch(request: Request, layer_id: str, body: LayerPatchBody):
    user_id, user_roles = _map_caller(request)
    patch = body.model_dump(exclude_none=True)
    try:
        return await _layers_svc(request).update_layer(
            layer_id, user_id=user_id, patch=patch, user_roles=user_roles
        )
    except PermissionError:
        raise HTTPException(status_code=403, detail="not_authorized_for_layer")
    except KeyError:
        raise HTTPException(status_code=404, detail="layer_not_found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/map/layers/{layer_id}")
async def map_layers_delete(
    request: Request,
    layer_id: str,
    purge_data: bool = False,
):
    user_id, user_roles = _map_caller(request)
    try:
        return await _layers_svc(request).delete_layer(
            layer_id, user_id=user_id, purge_data=purge_data, user_roles=user_roles
        )
    except PermissionError:
        raise HTTPException(status_code=403, detail="not_authorized_for_layer")
    except KeyError:
        raise HTTPException(status_code=404, detail="layer_not_found")


@router.post("/map/layers/{layer_id}/scan")
async def map_layers_scan(request: Request, layer_id: str):
    user_id, user_roles = _map_caller(request)
    access = await _layers_svc(request).layer_is_accessible(
        user_id=user_id, layer_id=layer_id, action="write", user_roles=user_roles
    )
    if not access.get("accessible"):
        raise HTTPException(status_code=403, detail="not_authorized_for_layer")
    try:
        return await _layers_svc(request).scan_streetview_layer(layer_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="layer_not_found")


@router.get("/map/layers/{layer_id}/tilejson")
async def map_layers_tilejson(request: Request, layer_id: str):
    user_id, user_roles = _map_caller(request)
    access = await _layers_svc(request).layer_is_accessible(
        user_id=user_id, layer_id=layer_id, action="read", user_roles=user_roles
    )
    if not access.get("accessible"):
        raise HTTPException(status_code=403, detail="not_authorized_for_layer")
    async with _layers_svc(request)._pool().acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM map_layers WHERE id = $1", uuid.UUID(layer_id))
    if not row:
        raise HTTPException(status_code=404, detail="layer_not_found")
    layer = _layers_svc(request)._row_to_dict(row)
    kind = layer.get("layer_kind") or ""
    ext_suffix = ".jpg" if kind.startswith("raster") else ""
    tiles_url = f"/map/tiles/{layer_id}/{{z}}/{{x}}/{{y}}{ext_suffix}"
    min_z, max_z = _layers_svc(request).effective_zoom_range(layer)
    return {
        "tilejson": "3.0.0",
        "name": layer.get("name"),
        "tiles": [tiles_url],
        "minzoom": min_z if min_z is not None else 0,
        "maxzoom": max_z if max_z is not None else 22,
        "bounds": layer.get("bounds"),
        "layer_id": layer_id,
        "layer_kind": kind,
    }


@router.get("/map/layers/{layer_id}/features")
async def map_layers_features(
    request: Request,
    layer_id: str,
    min_lon: Optional[float] = Query(None, alias="minLon"),
    min_lat: Optional[float] = Query(None, alias="minLat"),
    max_lon: Optional[float] = Query(None, alias="maxLon"),
    max_lat: Optional[float] = Query(None, alias="maxLat"),
):
    """GeoJSON for ``vector_api`` / ``vector_geojson`` catalog layers (RBAC-gated)."""
    from .map_layer_api import ApiLayerError, fetch_layer_features

    user_id, user_roles = _map_caller(request)
    svc = _layers_svc(request)
    access = await svc.layer_is_accessible(
        user_id=user_id, layer_id=layer_id, action="read", user_roles=user_roles
    )
    if not access.get("accessible"):
        raise HTTPException(status_code=403, detail="not_authorized_for_layer")
    layer = await svc.get_layer_raw(layer_id)
    if not layer:
        raise HTTPException(status_code=404, detail="layer_not_found")
    kind = str(layer.get("layer_kind") or "")
    if kind == "vector_geojson":
        try:
            return await asyncio.to_thread(svc.load_vector_geojson_features, layer)
        except ApiLayerError as exc:
            raise HTTPException(
                status_code=404 if exc.code == "geojson_file_missing" else 502,
                detail={"code": exc.code, "message": exc.message},
            )
    if kind != "vector_api":
        raise HTTPException(status_code=400, detail="layer_kind_not_vector_overlay")
    bbox = None
    if None not in (min_lon, min_lat, max_lon, max_lat):
        bbox = {
            "minLon": float(min_lon),
            "minLat": float(min_lat),
            "maxLon": float(max_lon),
            "maxLat": float(max_lat),
        }
    try:
        return await fetch_layer_features(
            layer_id=layer_id,
            source_config=layer.get("source_config") or {},
            bbox=bbox,
        )
    except ApiLayerError as exc:
        raise HTTPException(status_code=502, detail={"code": exc.code, "message": exc.message})


@router.get("/map/layers/{layer_id}/assets/{asset_path:path}")
async def map_layer_asset(
    request: Request,
    layer_id: str,
    asset_path: str,
):
    """Serve a sidecar file under a catalog layer root (e.g. thumbnails/*.jpg)."""
    user_id, user_roles = _map_caller(request)
    svc = _layers_svc(request)
    access = await svc.layer_is_accessible(
        user_id=user_id, layer_id=layer_id, action="read", user_roles=user_roles
    )
    if not access.get("accessible"):
        raise HTTPException(status_code=403, detail="not_authorized_for_layer")
    layer = await svc.get_layer_raw(layer_id)
    if not layer:
        raise HTTPException(status_code=404, detail="layer_not_found")
    try:
        path = await asyncio.to_thread(svc.resolve_layer_asset_path, layer, asset_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="asset_not_found")
    return FileResponse(path)


@router.get("/map/streetview/panoramas")
async def map_streetview_panoramas(
    request: Request,
    layer_id: Optional[str] = None,
    folder: Optional[str] = None,
):
    user_id, user_roles = _map_caller(request)
    svc = _layers_svc(request)
    if layer_id:
        try:
            return await svc.streetview_panoramas(
                layer_id, user_id=user_id, folder=folder, user_roles=user_roles
            )
        except PermissionError:
            raise HTTPException(status_code=403, detail="not_authorized_for_layer")
    if folder:
        detected = svc.detect_path(folder)
        if detected.get("ok"):
            kind = str(detected.get("layer_kind") or "")
            if not kind.startswith("streetview"):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "not_streetview_folder",
                        "layer_kind": kind,
                        "path": folder,
                    },
                )
            layer = await svc.find_or_create_layer(
                owner_user_id=user_id,
                name=Path(folder.strip()).name or "streetview",
                layer_kind="streetview_folder",
                storage_root=detected.get("storage_root"),
            )
            return await svc.streetview_panoramas(
                str(layer["id"]), user_id=user_id, user_roles=user_roles
            )
    raise HTTPException(status_code=400, detail="layer_id or folder required")


@router.get("/map/streetview/browse")
async def map_streetview_browse(
    request: Request,
    path: Optional[str] = None,
):
    user_id, user_roles = _map_caller(request)
    return await _layers_svc(request).streetview_browse(
        user_id=user_id, path=path, user_roles=user_roles
    )


@router.get("/map/sat-tile/config")
async def map_sat_tile_config_get(request: Request):
    user_id, user_roles = _map_caller(request)
    svc = _layers_svc(request)
    items: list = []
    for kind in ("raster_sas_sqlite", "raster_sas", "raster_xyz"):
        layers = await svc.layer_accessible_list(
            user_id,
            user_roles=user_roles,
            layer_kind=kind,
            enabled_only=False,
            limit=50,
            offset=0,
        )
        items.extend(layers.get("items") or [])

    def _pick_satellite_imagery(candidates: list) -> Optional[dict]:
        imagery = [
            row
            for row in candidates
            if str(row.get("sas_role") or (row.get("source_config") or {}).get("sas_role") or "")
            == "satellite_imagery"
        ]
        if imagery:
            return imagery[0]
        for row in candidates:
            if row.get("layer_kind") in ("raster_sas_sqlite", "raster_sas"):
                enriched = svc._enrich_sas_metadata(dict(row))
                if enriched.get("sas_role") == "satellite_imagery":
                    return enriched
        return candidates[0] if candidates else None

    layer = _pick_satellite_imagery(items)
    if layer:
        root = map_layers_root()
        return {
            "path": str(root / str(layer.get("storage_root") or "")),
            "layer_id": layer.get("id"),
            "layer": layer,
        }
    return {"path": str(map_layers_root() / "raster"), "layer_id": None}


@router.post("/map/sat-tile/config")
async def map_sat_tile_config_set(request: Request, body: Dict[str, Any] = Body(...)):
    user_id = _require_user_id(request)
    path = str(body.get("path") or "").strip()
    if not path:
        raise HTTPException(status_code=400, detail="path required")
    svc = _layers_svc(request)
    detected = svc.detect_path(path)
    if not detected.get("ok"):
        raise HTTPException(status_code=400, detail=detected.get("error") or "detect_failed")
    layer = await svc.find_or_create_layer(
        owner_user_id=user_id,
        name=body.get("name") or Path(path).name or "sat-raster",
        layer_kind=detected.get("layer_kind") or "raster_sas",
        storage_root=detected.get("storage_root") or path,
    )
    root = map_layers_root()
    return {
        "path": str(root / str(layer.get("storage_root") or "")),
        "layer_id": layer.get("id"),
        "layer": layer,
    }


@router.get("/map/layers/{layer_id}/pmtiles-style")
async def map_layer_pmtiles_style(request: Request, layer_id: str):
    user_id, user_roles = _map_caller(request)
    try:
        return await _layers_svc(request).describe_pmtiles_style(
            layer_id, user_id=user_id, user_roles=user_roles
        )
    except PermissionError:
        raise HTTPException(status_code=403, detail="not_authorized_for_layer")
    except KeyError:
        raise HTTPException(status_code=404, detail="layer_not_found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/map/basemap/config")
async def map_basemap_config_get(request: Request):
    user_id, user_roles = _map_caller(request)
    return await _layers_svc(request).get_user_basemap(user_id, user_roles=user_roles)


@router.post("/map/basemap/config")
async def map_basemap_config_set(request: Request, body: Dict[str, Any] = Body(...)):
    user_id, user_roles = _map_caller(request)
    svc = _layers_svc(request)
    layer_id = body.get("layer_id")
    path = str(body.get("path") or "").strip()
    name = body.get("name")
    try:
        if layer_id:
            return await svc.set_user_basemap_layer(
                user_id, str(layer_id), user_roles=user_roles
            )
        if path:
            return await svc.set_user_basemap_from_path(
                user_id,
                path,
                name=str(name).strip() if name else None,
                user_roles=user_roles,
            )
    except PermissionError:
        raise HTTPException(status_code=403, detail="not_authorized_for_layer")
    except KeyError:
        raise HTTPException(status_code=404, detail="layer_not_found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    raise HTTPException(status_code=400, detail="layer_id or path required")


@router.get("/map/layers/{layer_id}/pmtiles/{filename:path}")
@router.get("/map/layers/{layer_id}/pmtiles")
async def map_layer_pmtiles_file(
    request: Request,
    layer_id: str,
    filename: Optional[str] = None,
):
    user_id, user_roles = _map_caller(request)
    svc = _layers_svc(request)
    access = await svc.layer_is_accessible(
        user_id=user_id, layer_id=layer_id, action="read", user_roles=user_roles
    )
    if not access.get("accessible"):
        raise HTTPException(status_code=403, detail="not_authorized")
    async with svc._pool().acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM map_layers WHERE id = $1", uuid.UUID(layer_id))
    if not row:
        raise HTTPException(status_code=404, detail="layer_not_found")
    layer = svc._row_to_dict(row)
    pmtiles_name = filename.split("/")[-1] if filename else None
    pmtiles_path = svc.pmtiles_file_for_layer(layer, filename=pmtiles_name)
    if not pmtiles_path or not pmtiles_path.is_file():
        raise HTTPException(status_code=404, detail="pmtiles_not_found")

    file_size = pmtiles_path.stat().st_size
    common_headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=3600",
    }
    range_header = request.headers.get("range")
    if range_header:
        parsed = _parse_byte_range(range_header, file_size)
        if parsed is None:
            return Response(
                status_code=416,
                headers={"Content-Range": f"bytes */{file_size}"},
            )
        start, end = parsed
        length = end - start + 1
        data = await asyncio.to_thread(_read_file_range, pmtiles_path, start, length)
        return Response(
            content=data,
            status_code=206,
            media_type="application/vnd.pmtiles",
            headers={
                **common_headers,
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(length),
            },
        )

    return FileResponse(
        pmtiles_path,
        media_type="application/vnd.pmtiles",
        headers=common_headers,
    )


@router.get("/map/tiles/{layer_id}/{z}/{x}/{y}")
@router.get("/map/tiles/{layer_id}/{z}/{x}/{y}.{ext}")
async def map_layer_tile_public(
    request: Request,
    layer_id: str,
    z: int,
    x: int,
    y: int,
    ext: str = "jpg",
):
    """Gateway-facing tile path (alias for /map/layers/{id}/tiles/...)."""
    return await map_layer_tile(request, layer_id, z, x, y, ext=ext)


@router.get("/map/streetview/{layer_id}/{file_path:path}")
async def map_streetview_image(
    request: Request,
    layer_id: str,
    file_path: str,
):
    user_id, user_roles = _map_caller(request)
    svc = _layers_svc(request)
    access = await svc.layer_is_accessible(
        user_id=user_id, layer_id=layer_id, action="read", user_roles=user_roles
    )
    if not access.get("accessible"):
        raise HTTPException(status_code=403, detail="not_authorized_for_layer")
    async with svc._pool().acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM map_layers WHERE id = $1", uuid.UUID(layer_id))
    if not row:
        raise HTTPException(status_code=404, detail="layer_not_found")
    layer = svc._row_to_dict(row)
    img = svc.streetview_image_path(layer, file_path)
    if img and img.is_file():
        return FileResponse(str(img))
    raise HTTPException(status_code=404, detail="image_not_found")


@router.get("/map/layers/{layer_id}/tiles/{z}/{x}/{y}")
async def map_layer_tile(
    request: Request,
    layer_id: str,
    z: int,
    x: int,
    y: int,
    ext: str = "jpg",
):
    user_id, user_roles = _map_caller(request)
    svc = _layers_svc(request)
    access = await svc.layer_is_accessible(
        user_id=user_id, layer_id=layer_id, action="read", user_roles=user_roles
    )
    if not access.get("accessible"):
        raise HTTPException(status_code=403, detail="not_authorized_for_layer")
    async with svc._pool().acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM map_layers WHERE id = $1", uuid.UUID(layer_id))
    if not row:
        raise HTTPException(status_code=404, detail="layer_not_found")
    layer = svc._row_to_dict(row)
    try:
        tile_path = svc.tile_path_for_layer(layer, z, x, y, ext=ext.lstrip("."))
        if tile_path and tile_path.is_file():
            return FileResponse(str(tile_path))
        tile_data = svc.tile_bytes_for_layer(layer, z, x, y, ext=ext.lstrip("."))
    except OSError:
        tile_data = None
    if tile_data:
        blob, media_type = tile_data
        return Response(content=blob, media_type=media_type)
    raise HTTPException(status_code=404, detail="tile_not_found")


def register_map_routes(app) -> None:
    app.include_router(router)
