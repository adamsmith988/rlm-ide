"""User-defined API → GeoJSON map layers (no hardcoded upstream providers).

Probe / fetch / SSRF guards for ``vector_api`` catalog layers. Secrets stay
server-side; clients only ever see ``/map/layers/{id}/features``.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

logger = logging.getLogger(__name__)

MAX_RESPONSE_BYTES = 100 * 1024 * 1024  # 100 MiB
DEFAULT_TIMEOUT_MS = 15_000
DEFAULT_CACHE_TTL_S = 60

_BLOCKED_HOSTS = frozenset(
    {
        "localhost",
        "127.0.0.1",
        "::1",
        "0.0.0.0",
        "metadata.google.internal",
    }
)

# In-process TTL cache: key → (expires_at, geojson dict)
_FEATURES_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}


class ApiLayerError(Exception):
    """Raised for probe/fetch failures with a stable machine code."""

    def __init__(self, code: str, message: str = ""):
        self.code = code
        self.message = message or code
        super().__init__(self.message)


def redact_source_config(config: Any) -> Any:
    """Return a copy of source_config with secrets stripped for API responses."""
    if not isinstance(config, dict):
        return config
    out = dict(config)
    auth = out.get("auth")
    if isinstance(auth, dict):
        auth_out = dict(auth)
        if "value_secret" in auth_out and auth_out.get("value_secret"):
            auth_out["value_secret"] = "***"
            auth_out["has_secret"] = True
        out["auth"] = auth_out
    http_cfg = out.get("http")
    if isinstance(http_cfg, dict):
        http_out = dict(http_cfg)
        headers = http_out.get("headers")
        if isinstance(headers, dict):
            scrubbed = {}
            for k, v in headers.items():
                lk = str(k).lower()
                if lk in ("authorization", "x-api-key", "api-key", "proxy-authorization"):
                    scrubbed[k] = "***"
                else:
                    scrubbed[k] = v
            http_out["headers"] = scrubbed
        out["http"] = http_out
    return out


def validate_vector_api_config(source_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize and validate source_config for vector_api. Raises ApiLayerError."""
    if not isinstance(source_config, dict):
        raise ApiLayerError("invalid_source_config", "source_config object required")
    http_cfg = source_config.get("http")
    if not isinstance(http_cfg, dict):
        raise ApiLayerError("invalid_http", "source_config.http object required")
    url = str(http_cfg.get("url") or "").strip()
    if not url:
        raise ApiLayerError("missing_url", "source_config.http.url required")
    assert_safe_url(url)

    method = str(http_cfg.get("method") or "GET").strip().upper()
    if method not in ("GET", "POST"):
        raise ApiLayerError("unsupported_method", "only GET and POST are allowed")

    timeout_ms = int(http_cfg.get("timeout_ms") or DEFAULT_TIMEOUT_MS)
    timeout_ms = max(1000, min(timeout_ms, 60_000))

    auth = source_config.get("auth") if isinstance(source_config.get("auth"), dict) else {}
    auth_type = str(auth.get("type") or "none").strip().lower()
    if auth_type not in ("none", "bearer", "header", "query"):
        raise ApiLayerError("unsupported_auth", f"unsupported auth.type: {auth_type}")

    response = source_config.get("response") if isinstance(source_config.get("response"), dict) else {}
    fmt = str(response.get("format") or "geojson").strip().lower()
    if fmt != "geojson":
        raise ApiLayerError("unsupported_format", "only response.format=geojson is supported")

    refresh = int(source_config.get("refresh_interval_s") or 300)
    cache_ttl = int(source_config.get("cache_ttl_s") or DEFAULT_CACHE_TTL_S)
    cache_ttl = max(0, min(cache_ttl, 3600))

    viewport = source_config.get("viewport") if isinstance(source_config.get("viewport"), dict) else {}
    normalized = {
        "http": {
            "method": method,
            "url": url,
            "headers": dict(http_cfg.get("headers") or {}) if isinstance(http_cfg.get("headers"), dict) else {},
            "query": dict(http_cfg.get("query") or {}) if isinstance(http_cfg.get("query"), dict) else {},
            "body": http_cfg.get("body"),
            "timeout_ms": timeout_ms,
        },
        "auth": {
            "type": auth_type,
            "header_name": str(auth.get("header_name") or "Authorization"),
            "query_param": str(auth.get("query_param") or "api_key"),
            "value_secret": str(auth.get("value_secret") or ""),
        },
        "response": {
            "format": "geojson",
            "jmespath": response.get("jmespath"),
        },
        "refresh_interval_s": max(0, refresh),
        "cache_ttl_s": cache_ttl,
        "viewport": {
            "enabled": bool(viewport.get("enabled")),
            "bbox_param": str(viewport.get("bbox_param") or "bbox"),
            "bbox_template": str(
                viewport.get("bbox_template") or "{minLon},{minLat},{maxLon},{maxLat}"
            ),
        },
    }
    return normalized


def assert_safe_url(url: str) -> None:
    """Block SSRF to localhost / link-local / private / metadata endpoints."""
    raw = (url or "").strip()
    try:
        parsed = urlparse(raw)
    except Exception as exc:
        raise ApiLayerError("invalid_url", str(exc)) from exc
    if parsed.scheme not in ("http", "https"):
        raise ApiLayerError("invalid_scheme", "only http/https URLs are allowed")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise ApiLayerError("invalid_host", "URL host required")
    if host in _BLOCKED_HOSTS:
        raise ApiLayerError("blocked_host", f"host blocked: {host}")
    if host.endswith(".local") or host.endswith(".internal"):
        raise ApiLayerError("blocked_host", f"host blocked: {host}")

    # Resolve DNS and reject private / link-local / loopback addresses.
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ApiLayerError("dns_failed", f"cannot resolve host: {host}") from exc

    for info in infos:
        sockaddr = info[4]
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ApiLayerError("blocked_ip", f"resolved address blocked: {ip_str}")
        # Cloud metadata classic
        if str(ip) == "169.254.169.254":
            raise ApiLayerError("blocked_metadata_ip", "metadata IP blocked")


def _apply_auth(
    headers: Dict[str, str],
    query: Dict[str, str],
    auth: Dict[str, Any],
) -> None:
    auth_type = str(auth.get("type") or "none").lower()
    secret = str(auth.get("value_secret") or "")
    if auth_type == "none" or not secret:
        return
    if auth_type == "bearer":
        headers["Authorization"] = f"Bearer {secret}"
    elif auth_type == "header":
        name = str(auth.get("header_name") or "Authorization")
        headers[name] = secret
    elif auth_type == "query":
        param = str(auth.get("query_param") or "api_key")
        query[param] = secret


def _build_url(base_url: str, query: Dict[str, Any], bbox: Optional[Dict[str, float]], viewport: Dict[str, Any]) -> str:
    parsed = urlparse(base_url)
    merged = dict(parse_qsl(parsed.query, keep_blank_values=True))
    for k, v in (query or {}).items():
        if v is None:
            continue
        merged[str(k)] = str(v)
    if viewport.get("enabled") and bbox:
        tmpl = str(viewport.get("bbox_template") or "{minLon},{minLat},{maxLon},{maxLat}")
        value = (
            tmpl.replace("{minLon}", str(bbox["minLon"]))
            .replace("{minLat}", str(bbox["minLat"]))
            .replace("{maxLon}", str(bbox["maxLon"]))
            .replace("{maxLat}", str(bbox["maxLat"]))
        )
        param = str(viewport.get("bbox_param") or "bbox")
        merged[param] = value
    new_query = urlencode(merged)
    return urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment)
    )


def _dig(obj: Any, path: Optional[str]) -> Any:
    """Minimal dotted-path extractor (phase-1 stand-in for full JMESPath)."""
    if not path or not str(path).strip():
        return obj
    cur = obj
    for part in str(path).strip().split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise ApiLayerError("jmespath_miss", f"path not found: {path}")
    return cur


def normalize_geojson(payload: Any) -> Dict[str, Any]:
    """Coerce Feature / FeatureCollection / geometry into a FeatureCollection."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ApiLayerError("invalid_json", "response is not JSON") from exc
    if not isinstance(payload, dict):
        raise ApiLayerError("invalid_geojson", "expected a JSON object")

    gtype = str(payload.get("type") or "")
    if gtype == "FeatureCollection":
        features = payload.get("features")
        if not isinstance(features, list):
            raise ApiLayerError("invalid_geojson", "FeatureCollection.features must be a list")
        return {"type": "FeatureCollection", "features": features}
    if gtype == "Feature":
        return {"type": "FeatureCollection", "features": [payload]}
    if gtype in (
        "Point",
        "MultiPoint",
        "LineString",
        "MultiLineString",
        "Polygon",
        "MultiPolygon",
        "GeometryCollection",
    ):
        return {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "geometry": payload, "properties": {}}],
        }
    raise ApiLayerError("invalid_geojson", f"unsupported GeoJSON type: {gtype or 'missing'}")


async def _http_fetch(
    *,
    method: str,
    url: str,
    headers: Dict[str, str],
    body: Any,
    timeout_ms: int,
) -> Tuple[int, bytes, str]:
    assert_safe_url(url)
    timeout = httpx.Timeout(timeout_ms / 1000.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        if method == "POST":
            content = None
            json_body = None
            if isinstance(body, (dict, list)):
                json_body = body
            elif body is not None:
                content = str(body).encode("utf-8")
            resp = await client.post(url, headers=headers, json=json_body, content=content)
        else:
            resp = await client.get(url, headers=headers)
    raw = resp.content or b""
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ApiLayerError("response_too_large", f"response exceeds {MAX_RESPONSE_BYTES} bytes")
    ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
    return resp.status_code, raw, ctype


async def probe_api_layer(source_config: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only probe: fetch once, validate GeoJSON, return suggested config (no persist)."""
    warnings: List[str] = []
    try:
        cfg = validate_vector_api_config(source_config)
    except ApiLayerError as exc:
        return {"ok": False, "error": exc.code, "message": exc.message, "warnings": warnings}

    http_cfg = cfg["http"]
    headers = {str(k): str(v) for k, v in (http_cfg.get("headers") or {}).items()}
    query = {str(k): str(v) for k, v in (http_cfg.get("query") or {}).items()}
    _apply_auth(headers, query, cfg["auth"])
    url = _build_url(http_cfg["url"], query, None, cfg["viewport"])

    try:
        status, raw, ctype = await _http_fetch(
            method=http_cfg["method"],
            url=url,
            headers=headers,
            body=http_cfg.get("body"),
            timeout_ms=int(http_cfg["timeout_ms"]),
        )
    except ApiLayerError as exc:
        return {"ok": False, "error": exc.code, "message": exc.message, "warnings": warnings}
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "error": "upstream_http_error",
            "message": f"{type(exc).__name__}:{exc}",
            "warnings": warnings,
        }

    if status >= 400:
        return {
            "ok": False,
            "error": "upstream_status",
            "message": f"upstream HTTP {status}",
            "status_code": status,
            "warnings": warnings,
        }

    if ctype and "json" not in ctype and "geo+json" not in ctype and "javascript" not in ctype:
        warnings.append(f"unexpected_content_type:{ctype}")

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "error": "invalid_json",
            "message": str(exc),
            "warnings": warnings,
        }

    jmes = cfg["response"].get("jmespath")
    try:
        extracted = _dig(payload, jmes if isinstance(jmes, str) else None)
        geojson = normalize_geojson(extracted)
    except ApiLayerError as exc:
        return {
            "ok": False,
            "error": exc.code,
            "message": exc.message,
            "warnings": warnings,
            "raw_type": type(payload).__name__,
        }

    features = geojson.get("features") or []
    sample_props: List[Dict[str, Any]] = []
    for feat in features[:3]:
        if isinstance(feat, dict) and isinstance(feat.get("properties"), dict):
            sample_props.append(dict(list(feat["properties"].items())[:12]))

    suggested = redact_source_config(cfg)
    # Re-attach empty secret slot so UI knows the shape (without value).
    if isinstance(suggested.get("auth"), dict):
        suggested["auth"]["value_secret"] = ""

    return {
        "ok": True,
        "detected_format": "geojson",
        "feature_count": len(features),
        "sample_properties": sample_props,
        "suggested_source_config": suggested,
        "status_code": status,
        "content_type": ctype,
        "warnings": warnings,
    }


def _cache_key(layer_id: str, bbox: Optional[Dict[str, float]]) -> str:
    if not bbox:
        return f"{layer_id}:full"
    return (
        f"{layer_id}:{bbox.get('minLon')}:{bbox.get('minLat')}:"
        f"{bbox.get('maxLon')}:{bbox.get('maxLat')}"
    )


async def fetch_layer_features(
    *,
    layer_id: str,
    source_config: Dict[str, Any],
    bbox: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Fetch + normalize GeoJSON for a registered vector_api layer (with TTL cache)."""
    cfg = validate_vector_api_config(source_config)
    ttl = int(cfg.get("cache_ttl_s") or 0)
    key = _cache_key(layer_id, bbox if cfg.get("viewport", {}).get("enabled") else None)
    now = time.time()
    if ttl > 0:
        hit = _FEATURES_CACHE.get(key)
        if hit and hit[0] > now:
            return hit[1]

    http_cfg = cfg["http"]
    headers = {str(k): str(v) for k, v in (http_cfg.get("headers") or {}).items()}
    query = {str(k): str(v) for k, v in (http_cfg.get("query") or {}).items()}
    _apply_auth(headers, query, cfg["auth"])
    url = _build_url(
        http_cfg["url"],
        query,
        bbox if cfg.get("viewport", {}).get("enabled") else None,
        cfg["viewport"],
    )

    try:
        status, raw, _ctype = await _http_fetch(
            method=http_cfg["method"],
            url=url,
            headers=headers,
            body=http_cfg.get("body"),
            timeout_ms=int(http_cfg["timeout_ms"]),
        )
    except ApiLayerError:
        raise
    except httpx.HTTPError as exc:
        raise ApiLayerError("upstream_http_error", f"{type(exc).__name__}:{exc}") from exc

    if status >= 400:
        raise ApiLayerError("upstream_status", f"upstream HTTP {status}")

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiLayerError("invalid_json", "response is not JSON") from exc

    jmes = cfg["response"].get("jmespath")
    extracted = _dig(payload, jmes if isinstance(jmes, str) else None)
    geojson = normalize_geojson(extracted)
    geojson.setdefault("metadata", {})
    if isinstance(geojson["metadata"], dict):
        geojson["metadata"]["layer_id"] = layer_id
        geojson["metadata"]["fetched_at"] = int(now)

    if ttl > 0:
        _FEATURES_CACHE[key] = (now + ttl, geojson)
    return geojson


def clear_features_cache(layer_id: Optional[str] = None) -> None:
    """Test helper / invalidate after config update."""
    if layer_id is None:
        _FEATURES_CACHE.clear()
        return
    prefix = f"{layer_id}:"
    for key in list(_FEATURES_CACHE.keys()):
        if key.startswith(prefix):
            del _FEATURES_CACHE[key]
