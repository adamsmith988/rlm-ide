"""
Storage Service - FastAPI Application
سرویس ذخیره‌سازی - مدیریت داده‌ها و آرتیفکت‌ها

Port: 8004
"""

import json
import logging
import os
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException, Query, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .storage import Storage, get_storage, close_storage
from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _content_disposition(disposition: str, filename: Optional[str]) -> str:
    """Build an RFC 6266 / RFC 5987 safe Content-Disposition header.

    HTTP header values are latin-1; non-ASCII filenames (e.g. Persian) crash the
    response with UnicodeEncodeError. We emit an ASCII fallback ``filename`` plus
    a UTF-8 ``filename*`` that compliant clients prefer.
    """
    from urllib.parse import quote

    name = filename or "file"
    ascii_name = name.encode("ascii", "ignore").decode("ascii").strip() or "file"
    ascii_name = ascii_name.replace('"', "").replace("\\", "")
    quoted_utf8 = quote(name, safe="")
    return (
        f"{disposition}; filename=\"{ascii_name}\"; "
        f"filename*=UTF-8''{quoted_utf8}"
    )


def _storage_auth_context(request: Request) -> Dict[str, Optional[str]]:
    """Verified service/user context for internal Storage reads."""
    if not settings.STORAGE_RBAC_ENABLED:
        return {"authorized": "true", "mode": "disabled", "user_id": None, "group_id": None, "case_id": None}

    internal_key = request.headers.get("X-Internal-Key")
    if internal_key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Valid internal storage auth context required")

    service_name = (request.headers.get("X-Service-Name") or "").strip()
    if service_name not in {"tool-runner", "api-gateway", "case-importer", "orchestrator"}:
        raise HTTPException(status_code=403, detail="Service not allowed to access storage")

    return {
        "authorized": "true",
        "mode": "internal",
        "service_name": service_name,
        "user_id": request.headers.get("X-User-Id"),
        "group_id": request.headers.get("X-Group-Id"),
        "case_id": request.headers.get("X-Case-Id"),
    }


def _case_matches_artifact(ctx: Dict[str, Optional[str]], artifact: Dict[str, Any]) -> bool:
    case_id = ctx.get("case_id")
    if not case_id:
        return False
    meta = artifact.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    return case_id in {
        artifact.get("case_id"),
        meta.get("case_id") if isinstance(meta, dict) else None,
        meta.get("case_importer_last_case_id") if isinstance(meta, dict) else None,
    }


async def _require_artifact_access(
    storage: Storage,
    artifact_id: str,
    request: Request,
    *,
    artifact: Optional[Dict[str, Any]] = None,
    action: str = "read",
) -> Dict[str, Any]:
    ctx = _storage_auth_context(request)
    artifact = artifact or await storage.artifact_get(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    user_id = ctx.get("user_id")
    if user_id:
        # A user identity is present → that user's RBAC is authoritative. We do
        # NOT fall back to the X-Case-Id path here: doing so let any caller read
        # another user's artifact just by supplying the artifact's case id.
        if str(artifact.get("user_id") or "") == str(user_id):
            return artifact
        try:
            access = await storage.artifact_is_accessible(
                user_id=str(user_id),
                artifact_id=artifact_id,
                action=action,
            )
        except Exception as exc:
            # Fail closed: an access-check error must not grant access.
            logger.error("artifact_is_accessible failed; denying: %s", exc)
            raise HTTPException(status_code=403, detail="not_authorized_for_artifact")
        if (access or {}).get("accessible"):
            return artifact

        # Case-linked plugin/import artifacts: collaborators with case access may
        # read when metadata.case_id (or artifact.case_id) matches an accessible case.
        # Write/delete still require owner / group_files / override.
        if (action or "read").lower() == "read":
            meta = _artifact_metadata_dict(artifact)
            linked_case = (
                str(artifact.get("case_id") or "").strip()
                or str(meta.get("case_id") or "").strip()
                or str(meta.get("case_importer_last_case_id") or "").strip()
            )
            if linked_case:
                try:
                    case_access = await storage.case_is_accessible(
                        user_id=str(user_id),
                        case_id=linked_case,
                    )
                    if (case_access or {}).get("accessible"):
                        return artifact
                except Exception as exc:
                    logger.warning(
                        "case_is_accessible failed during artifact read for %s: %s",
                        artifact_id,
                        exc,
                    )

        raise HTTPException(status_code=403, detail="not_authorized_for_artifact")

    # No user identity (pure batch / importer service path): allow the
    # case-scoped fallback only. X-Case-Id here is asserted by a trusted internal
    # service that has already validated case membership upstream.
    if _case_matches_artifact(ctx, artifact):
        return artifact

    raise HTTPException(status_code=403, detail="not_authorized_for_artifact")


def _artifact_allowed_by_context(ctx: Dict[str, Optional[str]], artifact: Dict[str, Any]) -> bool:
    user_id = ctx.get("user_id")
    if user_id and str(artifact.get("user_id") or "") == str(user_id):
        return True
    return _case_matches_artifact(ctx, artifact)


def _artifact_metadata_dict(artifact: Dict[str, Any]) -> Dict[str, Any]:
    meta = artifact.get("metadata") or {}
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str):
        try:
            parsed = json.loads(meta)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def _resolve_existing_local_artifact_path(artifact: Dict[str, Any]) -> Optional[str]:
    """
    Recover an on-disk path for legacy/imported artifacts whose stored
    ``storage_path`` no longer exists in the current runtime.

    Resolution order:
    1. Current absolute/relative storage_path
    2. Case-based paths derived from import metadata
    3. Import staging tree using case_importer_rel_path / filename
    """
    if not isinstance(artifact, dict):
        return None

    meta = _artifact_metadata_dict(artifact)
    raw_path = str(artifact.get("storage_path") or "").strip()
    rel_path = str(meta.get("case_importer_rel_path") or "").strip().lstrip("/").replace("\\", "/")
    filename = str(
        meta.get("case_importer_filename")
        or artifact.get("original_filename")
        or Path(raw_path).name
        or ""
    ).strip()

    candidates: List[Path] = []
    seen: Set[str] = set()

    def add_candidate(path_like: Optional[str]) -> None:
        text = str(path_like or "").strip()
        if not text or text.startswith("minio://"):
            return
        candidate = Path(text if text.startswith("/") else f"/data/uploads/{text.lstrip('/')}")
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)

    add_candidate(raw_path)

    case_ids: List[str] = []
    for value in (
        artifact.get("case_id"),
        meta.get("case_id"),
        meta.get("case_importer_last_case_id"),
    ):
        text = str(value or "").strip()
        if text and text not in case_ids:
            case_ids.append(text)

    case_rel = rel_path or filename
    if case_rel:
        for case_id in case_ids:
            add_candidate(f"/data/cases_temp/{case_id}/{case_rel}")
            add_candidate(f"/data/cases/{case_id}/{case_rel}")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate)

    import_root = Path("/data/import_staging")
    if import_root.exists():
        staged_matches: List[Path] = []
        patterns: List[str] = []
        if rel_path:
            patterns.append(f"*/tree/{rel_path}")
        elif filename:
            patterns.append(f"*/tree/**/{filename}")
        for pattern in patterns:
            try:
                staged_matches.extend(p for p in import_root.glob(pattern) if p.is_file())
            except Exception:
                continue
        if staged_matches:
            staged_matches.sort(
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
                reverse=True,
            )
            return str(staged_matches[0])

    return None


def _looks_like_probe_query(query: str) -> bool:
    q = " ".join((query or "").strip().lower().split())
    return q in {"return 1 as ok limit 1", "return 1 as ok", "return 1 limit 1", "return 1"}


def _looks_like_aggregate_stats_record(record: Dict[str, Any]) -> bool:
    """COUNT()/stats rows have no nested case_id for RBAC to scrape."""
    keys = {str(k).lower() for k in record.keys()}
    return bool(
        keys
        & {
            "entity_count",
            "relationship_count",
            "node_count",
            "rel_count",
            "c",
            "count",
        }
    )


def _looks_like_count_query(query: str) -> bool:
    q = " ".join((query or "").strip().lower().split())
    return "count(" in q and "return" in q


def _collect_graph_refs(value: Any) -> Tuple[Set[str], Set[str]]:
    """Recursively collect artifact/case ids from Neo4j record payloads."""
    artifact_ids: Set[str] = set()
    case_ids: Set[str] = set()

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, raw in obj.items():
                lk = str(key).lower().replace("-", "_")
                if lk == "artifact_id" and raw:
                    artifact_ids.add(str(raw))
                elif lk == "case_id" and raw:
                    case_ids.add(str(raw))
                elif lk in {"props", "properties", "data", "metadata"}:
                    visit(raw)
                elif isinstance(raw, (dict, list)):
                    visit(raw)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    visit(value)
    return artifact_ids, case_ids


async def _fetch_case_row(storage: Storage, case_id: str) -> Optional[Dict[str, Any]]:
    if not storage.postgres or not storage.postgres.pool:
        return None
    try:
        async with storage.postgres.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, name, user_id, group_id, session_id, status, progress,
                       metadata, created_at, updated_at
                FROM cases
                WHERE id = $1
                """,
                case_id,
            )
        return dict(row) if row else None
    except Exception as exc:
        logger.warning("fetch case row failed for %s: %s", case_id, exc)
        return None


async def _require_case_access(
    storage: Storage,
    case_id: str,
    request: Request,
    *,
    require_owner: bool = False,
) -> Dict[str, Any]:
    """Storage RBAC via Postgres group tables (same model as artifacts)."""
    ctx = _storage_auth_context(request)
    row = await _fetch_case_row(storage, case_id)
    if not row:
        raise HTTPException(status_code=404, detail="Case not found")

    user_id = ctx.get("user_id")
    if settings.STORAGE_RBAC_ENABLED:
        if not user_id:
            raise HTTPException(status_code=401, detail="user_id_required")
        access = await storage.case_is_accessible(user_id=str(user_id), case_id=case_id)
        if not (access or {}).get("accessible"):
            raise HTTPException(status_code=403, detail="not_authorized_for_case")
        if require_owner:
            nested = (access or {}).get("access") or {}
            if not nested.get("is_owner"):
                raise HTTPException(status_code=403, detail="owner_required")
    return row


async def _case_allowed_by_context(
    storage: Storage,
    ctx: Dict[str, Optional[str]],
    case_id: str,
) -> bool:
    if ctx.get("case_id") and str(ctx["case_id"]) == str(case_id):
        return True
    user_id = ctx.get("user_id")
    if not user_id:
        return False
    try:
        access = await storage.case_is_accessible(user_id=str(user_id), case_id=case_id)
        return bool((access or {}).get("accessible"))
    except Exception as exc:
        logger.warning("case_is_accessible failed for %s: %s", case_id, exc)
        return False


async def _graph_record_allowed(
    storage: Storage,
    ctx: Dict[str, Optional[str]],
    record: Dict[str, Any],
    *,
    artifact_cache: Dict[str, bool],
    case_cache: Dict[str, bool],
) -> bool:
    artifact_ids, case_ids = _collect_graph_refs(record)
    if not artifact_ids and not case_ids:
        # Aggregate COUNT() rows have no nested case_id — authorize via ctx.
        ctx_case = str(ctx.get("case_id") or "").strip()
        if not ctx_case or not _looks_like_aggregate_stats_record(record):
            return False
        if ctx.get("service_name") in {
            "case-importer",
            "api-gateway",
            "orchestrator",
        }:
            return True
        if ctx_case not in case_cache:
            case_cache[ctx_case] = await _case_allowed_by_context(
                storage, ctx, ctx_case
            )
        return bool(case_cache[ctx_case])

    for artifact_id in artifact_ids:
        if artifact_id not in artifact_cache:
            artifact = await storage.artifact_get(artifact_id)
            allowed = False
            if artifact:
                allowed = _artifact_allowed_by_context(ctx, artifact)
                if not allowed and ctx.get("user_id"):
                    try:
                        access = await storage.artifact_is_accessible(
                            user_id=str(ctx["user_id"]),
                            artifact_id=artifact_id,
                        )
                        allowed = bool((access or {}).get("accessible"))
                    except Exception as exc:
                        logger.warning("graph artifact access check failed for %s: %s", artifact_id, exc)
            artifact_cache[artifact_id] = allowed
        if not artifact_cache[artifact_id]:
            return False

    for case_id in case_ids:
        if case_id not in case_cache:
            case_cache[case_id] = await _case_allowed_by_context(storage, ctx, case_id)
        if not case_cache[case_id]:
            return False

    return True


async def _filter_graph_records_for_context(
    storage: Storage,
    ctx: Dict[str, Optional[str]],
    records: List[Dict[str, Any]],
    *,
    query: str,
) -> List[Dict[str, Any]]:
    if not settings.STORAGE_RBAC_ENABLED:
        return records
    if _looks_like_probe_query(query):
        return records
    # COUNT() aggregates scoped by Cypher $case_id + X-Case-Id: keep rows.
    if (
        _looks_like_count_query(query)
        and str(ctx.get("case_id") or "").strip()
        and all(
            isinstance(r, dict) and _looks_like_aggregate_stats_record(r)
            for r in records
        )
    ):
        return list(records)
    artifact_cache: Dict[str, bool] = {}
    case_cache: Dict[str, bool] = {}
    filtered: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if await _graph_record_allowed(
            storage,
            ctx,
            record,
            artifact_cache=artifact_cache,
            case_cache=case_cache,
        ):
            filtered.append(record)
    return filtered

# ==================== Pydantic Models ====================

class PersonCreate(BaseModel):
    full_name: str
    external_id: Optional[str] = None
    aliases: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class FolderCreate(BaseModel):
    name: str
    parent_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class FolderEnsureChain(BaseModel):
    """Create nested folders; ``names`` and ``segments`` are aliases."""

    names: Optional[List[str]] = None
    segments: Optional[List[str]] = None
    parent_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    case_id: Optional[str] = None


class ArtifactCreate(BaseModel):
    storage_path: str
    media_type: str  # image, audio, video, text, document, other
    original_filename: Optional[str] = None
    folder_id: Optional[str] = None
    checksum: Optional[str] = None
    file_size_bytes: Optional[int] = None
    mime_type: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ArtifactMetadataUpdate(BaseModel):
    metadata: Dict[str, Any]


class SessionArtifactAttach(BaseModel):
    artifact_id: Optional[str] = None
    case_id: Optional[str] = None
    path: Optional[str] = None
    storage_path: Optional[str] = None
    filename: Optional[str] = None
    name: Optional[str] = None
    user_id: Optional[str] = None
    limit: Optional[int] = 25


# Note: ToolRunCreate model has been REMOVED - tool executions use /traces/{trace_id}/tools API


class ObservationCreate(BaseModel):
    session_id: str
    observation_type: str
    content: str
    artifact_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class InsightCreate(BaseModel):
    session_id: str
    insight_type: str
    title: str
    content: str
    confidence: Optional[float] = None
    sources: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class FeedbackCreate(BaseModel):
    session_id: str
    rating: str  # positive, negative, neutral
    comment: Optional[str] = None
    message_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class VectorUpsert(BaseModel):
    collection_name: str
    vectors: List[List[float]]
    ids: List[str]
    payloads: Optional[List[Dict[str, Any]]] = None
    dimension: Optional[int] = None


class VectorSearch(BaseModel):
    collection_name: str
    query_vector: List[float]
    limit: int = 10
    filter_dict: Optional[Dict[str, Any]] = None
    score_threshold: Optional[float] = None


class CypherQuery(BaseModel):
    query: str
    parameters: Optional[Dict[str, Any]] = None


class CypherBatch(BaseModel):
    queries: List[Dict[str, Any]]  # [{"query": "...", "parameters": {...}}, ...]


class FaceCreate(BaseModel):
    artifact_id: str
    face_key: str
    bbox: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    person_id: Optional[str] = None
    tag: Optional[str] = None
    thumbnail_base64: Optional[str] = None
    vector_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ImageMetaCreate(BaseModel):
    artifact_id: str
    width_px: Optional[int] = None
    height_px: Optional[int] = None
    dpi: Optional[int] = None
    color_mode: Optional[str] = None
    camera_model: Optional[str] = None
    lens: Optional[str] = None
    iso: Optional[int] = None
    exposure: Optional[str] = None
    focal_length: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AudioMetaCreate(BaseModel):
    artifact_id: str
    duration_sec: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    codec: Optional[str] = None
    bitrate: Optional[int] = None
    transcription: Optional[str] = None
    language: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AudioSegmentsBatch(BaseModel):
    segments: List[Dict[str, Any]]
    replace: bool = True


class ToolRecordsBatch(BaseModel):
    tool_id: str
    records: List[Dict[str, Any]]
    artifact_id: Optional[str] = None
    case_id: Optional[str] = None
    user_id: Optional[str] = None
    run_id: Optional[str] = None


# ==================== Lifespan ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("🚀 Storage service starting...")
    from .config import assert_secure_secrets
    assert_secure_secrets()
    storage = await get_storage()
    app.state.storage = storage
    from .map_api import register_map_routes
    from .map_layers import MapLayersService
    from .artifact_geo import ensure_artifact_geo_schema
    from .plugin_prefs_api import register_plugin_prefs_routes
    from .plugin_user_prefs import PluginUserPrefsService

    register_map_routes(app)
    register_plugin_prefs_routes(app)
    try:
        layers_svc = MapLayersService(storage)
        await layers_svc.ensure_schema()
        pg = getattr(storage, "postgres", None)
        if pg and pg.pool:
            await ensure_artifact_geo_schema(pg.pool)
            try:
                await _ensure_tool_records_schema(pg.pool)
            except Exception as tool_rec_exc:
                logger.warning("tool_records schema ensure skipped: %s", tool_rec_exc)
    except Exception as exc:
        logger.warning("map_layers schema ensure skipped: %s", exc)
    try:
        await PluginUserPrefsService(storage).ensure_schema()
        # Platform UI prefs + Phase C client work (idempotent).
        prefs_svc = PluginUserPrefsService(storage)
        await prefs_svc.register_namespace(
            plugin_id="platform",
            namespace="platform.user_preferences",
            schema_version=1,
            metadata={"description": "Unified UI preferences"},
        )
        await prefs_svc.register_namespace(
            plugin_id="platform",
            namespace="platform.client_work",
            schema_version=1,
            metadata={"description": "Per-user client work product"},
        )
    except Exception as exc:
        logger.warning("plugin_user_prefs schema ensure skipped: %s", exc)
    try:
        from .hf_pipeline_taxonomy import ensure_hf_pipeline_taxonomy_schema

        pg = getattr(storage, "postgres", None)
        if pg and pg.pool:
            await ensure_hf_pipeline_taxonomy_schema(pg.pool)
    except Exception as exc:
        logger.warning("hf_pipeline_taxonomy schema ensure skipped: %s", exc)
    try:
        minio = getattr(storage, "minio", None)
        if minio and hasattr(minio, "ensure_browser_cors"):
            from .config import settings as _st

            origins = getattr(_st, "MINIO_CORS_ORIGINS", None) or ["*"]
            if isinstance(origins, str):
                origins = [o.strip() for o in origins.split(",") if o.strip()]
            ok = await minio.ensure_browser_cors(origins)
            logger.info("MinIO browser CORS %s", "ok" if ok else "skipped")
    except Exception as exc:
        logger.warning("MinIO CORS ensure skipped: %s", exc)
    logger.info("✅ Storage service ready")
    yield
    logger.info("🛑 Storage service shutting down...")
    await close_storage()
    logger.info("✅ Storage service stopped")


# ==================== FastAPI App ====================

app = FastAPI(
    title="Storage Service",
    description="سرویس ذخیره‌سازی - مدیریت داده‌ها و آرتیفکت‌ها",
    version="1.0.0",
    lifespan=lifespan
)

# Direct-to-MinIO upload sessions (presigned PUT)
from .upload_sessions import router as upload_sessions_router
app.include_router(upload_sessions_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes reachable without the internal service key (liveness/probes + API docs).
_STORAGE_PUBLIC_PATHS = {"/health", "/health/live", "/docs", "/openapi.json", "/redoc", "/"}
_STORAGE_ALLOWED_SERVICES = {
    "tool-runner",
    "api-gateway",
    "case-importer",
    "orchestrator",
    "llm-proxy",
    "isomorphic-frontend",
}


@app.middleware("http")
async def _require_internal_service_auth(request: Request, call_next):
    """Global gate: storage is an internal service and must never be reachable by
    untrusted clients. Every route (except health/docs) requires the shared
    internal key + a known service name. This is defense-in-depth on top of the
    per-artifact owner/RBAC checks and assumes :8004 is also network-isolated.
    """
    if request.method == "OPTIONS" or not settings.STORAGE_RBAC_ENABLED:
        return await call_next(request)

    path = request.url.path
    if path in _STORAGE_PUBLIC_PATHS:
        return await call_next(request)

    internal_key = request.headers.get("X-Internal-Key")
    if internal_key != settings.INTERNAL_API_KEY:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=401,
            content={"detail": "Valid internal storage credential required"},
        )
    service_name = (request.headers.get("X-Service-Name") or "").strip()
    if service_name not in _STORAGE_ALLOWED_SERVICES:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=403,
            content={"detail": "Service not allowed to access storage"},
        )
    return await call_next(request)


def get_storage_from_app() -> Storage:
    """Get storage instance from app state"""
    return app.state.storage


# ==================== Health Routes ====================

@app.get("/health/live")
async def health_live():
    """Fast liveness/readiness — must not block on backend pings."""
    return {"status": "alive", "service": "storage", "port": 8004}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    storage = get_storage_from_app()
    status = await storage.health_check()
    return {
        "status": "healthy" if status["healthy"] else "unhealthy",
        "service": "storage",
        "port": 8004,
        **status
    }


# ==================== LLM Control Plane (Admin) ====================

class ExternalEndpointUpsert(BaseModel):
    name: str
    scheme: str = "http"
    host: str
    port: int
    base_path: str = ""
    auth: Optional[Dict[str, Any]] = None
    capabilities: Optional[Dict[str, Any]] = None
    last_probe: Optional[Dict[str, Any]] = None
    is_active: bool = True


class ExternalEndpointPatch(BaseModel):
    """Partial update for an existing endpoint (e.g. host/port migration)."""
    name: Optional[str] = None
    scheme: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    base_path: Optional[str] = None
    auth: Optional[Dict[str, Any]] = None
    capabilities: Optional[Dict[str, Any]] = None
    last_probe: Optional[Dict[str, Any]] = None
    # 0 clears; omit to leave unchanged. Shared by every token on this endpoint.
    requests_per_minute: Optional[int] = None
    is_active: Optional[bool] = None


class ExternalEndpointMergeRequest(BaseModel):
    keep_endpoint_id: str
    drop_endpoint_id: str
    keep_endpoint_name: str
    drop_endpoint_name: str


class EndpointCredentialCreate(BaseModel):
    label: str = "default"
    token: str = ""
    scope: str = "model"  # model-only (endpoint-wide credentials are not stored)
    logical_id: Optional[str] = None
    upstream_model_id: Optional[str] = None
    priority: int = 1000
    weight: int = 1
    requests_per_minute: Optional[int] = None
    is_active: bool = True
    last_probe: Optional[Dict[str, Any]] = None


class EndpointCredentialPatch(BaseModel):
    label: Optional[str] = None
    token: Optional[str] = None
    scope: Optional[str] = None
    logical_id: Optional[str] = None
    upstream_model_id: Optional[str] = None
    priority: Optional[int] = None
    weight: Optional[int] = None
    # 0 clears the gate (stored as NULL); omit to leave unchanged.
    requests_per_minute: Optional[int] = None
    is_active: Optional[bool] = None
    last_probe: Optional[Dict[str, Any]] = None


class EndpointModelRpmPatch(BaseModel):
    """Set one RPM on all credentials for endpoint + logical_id (or upstream)."""

    requests_per_minute: int
    logical_id: Optional[str] = None
    upstream_model_id: Optional[str] = None


class LocalModelUpsert(BaseModel):
    name: str
    # Deprecated: ignored for routing/storage. Prefer metadata.dispatch_key / I/O / api_protocol.
    # Kept optional so older clients still POST without schema errors.
    task: Optional[str] = None
    backend_kind: str  # kserve|external
    kserve_namespace: Optional[str] = None
    kserve_inferenceservice: Optional[str] = None
    external_endpoint_id: Optional[str] = None
    external_model_id: Optional[str] = None
    version: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    is_active: bool = True
    node_id: Optional[str] = None


class LocalModelPatch(BaseModel):
    is_active: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class LlmNodeUpsert(BaseModel):
    id: str
    display_name: Optional[str] = None
    models_root: Optional[str] = None
    has_gpu: bool = True
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = None


class LlmNodePatch(BaseModel):
    display_name: Optional[str] = None
    is_active: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class RouteUpsert(BaseModel):
    route_key: str
    model_name: str
    fallback_model_name: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None
    is_active: bool = True


class RoutePurgeLogicalIds(BaseModel):
    logical_ids: List[str]


class ModelCandidateUpsert(BaseModel):
    storage_path: str
    category: Optional[str] = None
    name: Optional[str] = None
    model_name: Optional[str] = None
    deploy_status: str = "discovered"
    metadata: Optional[Dict[str, Any]] = None


class PoolPolicyUpsert(BaseModel):
    logical_id: str
    canonical_model_key: Optional[str] = None
    min_replicas: int = Field(1, ge=0, le=16)
    max_replicas: int = Field(3, ge=1, le=32)
    target_free_vram_mb: int = Field(4096, ge=512)
    idle_scale_down_sec: int = Field(1800, ge=60)
    enabled: bool = True
    metadata: Optional[Dict[str, Any]] = None


class ModelUsageTouch(BaseModel):
    last_used_at: Optional[str] = None


@app.get("/llm/endpoints")
async def llm_external_endpoints_list():
    storage = get_storage_from_app()
    return await storage.llm_external_endpoints_list()


@app.post("/llm/endpoints")
async def llm_external_endpoint_upsert(data: ExternalEndpointUpsert):
    storage = get_storage_from_app()
    row = await storage.llm_external_endpoint_upsert(**data.model_dump())
    endpoint_id = str(row.get("id") or "")
    if endpoint_id:
        # Drop any leftover whole-endpoint credential rows (model-scoped only).
        try:
            await storage.llm_endpoint_credentials_ensure_default_from_auth(endpoint_id)
        except Exception as exc:  # non-fatal: endpoint upsert already succeeded
            logger.warning("purge_endpoint_scoped_credentials failed for %s: %s", endpoint_id, exc)
    return row


@app.post("/llm/endpoints/merge")
async def llm_external_endpoint_merge(body: ExternalEndpointMergeRequest):
    storage = get_storage_from_app()
    return await storage.llm_external_endpoint_merge_into(**body.model_dump())


@app.delete("/llm/endpoints/{endpoint_id}")
async def llm_external_endpoint_delete(endpoint_id: str):
    storage = get_storage_from_app()
    ok = await storage.llm_external_endpoint_delete(endpoint_id)
    return {"ok": ok, "id": endpoint_id}


@app.patch("/llm/endpoints/{endpoint_id}")
async def llm_external_endpoint_patch(endpoint_id: str, data: ExternalEndpointPatch):
    """Partial update for an endpoint — useful when the host IP changes."""
    storage = get_storage_from_app()
    body = {k: v for k, v in data.model_dump().items() if v is not None}
    row = await storage.llm_external_endpoint_update(endpoint_id, **body)
    if row is None:
        raise HTTPException(status_code=404, detail=f"endpoint_not_found: {endpoint_id}")
    return row


@app.get("/llm/credentials")
async def llm_endpoint_credentials_list_all():
    """All bearer credentials across endpoints (llm-proxy registry sync)."""
    storage = get_storage_from_app()
    return await storage.llm_endpoint_credentials_list_all()


@app.get("/llm/endpoints/{endpoint_id}/credentials")
async def llm_endpoint_credentials_list(endpoint_id: str):
    storage = get_storage_from_app()
    return await storage.llm_endpoint_credentials_list(endpoint_id)


@app.post("/llm/endpoints/{endpoint_id}/credentials")
async def llm_endpoint_credential_create(endpoint_id: str, data: EndpointCredentialCreate):
    from .backends.postgres import LlmCredentialTokenExists, LlmEndpointNotFound

    storage = get_storage_from_app()
    try:
        return await storage.llm_endpoint_credential_create(endpoint_id, **data.model_dump())
    except LlmCredentialTokenExists as exc:
        raise HTTPException(
            status_code=409, detail=f"credential_token_exists: {exc}"
        ) from exc
    except LlmEndpointNotFound as exc:
        raise HTTPException(
            status_code=404, detail=f"endpoint_not_found: {endpoint_id}"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/llm/endpoints/{endpoint_id}/credentials/ensure-default")
async def llm_endpoint_credentials_ensure_default(endpoint_id: str):
    """Purge legacy endpoint-wide credential rows (credentials are model-scoped only)."""
    storage = get_storage_from_app()
    row = await storage.llm_endpoint_credentials_ensure_default_from_auth(endpoint_id)
    return {"ok": True, "endpoint_id": endpoint_id, "credential": row, "created": False}


@app.get("/llm/endpoints/{endpoint_id}/credentials/{cred_id}")
async def llm_endpoint_credential_get(endpoint_id: str, cred_id: str):
    storage = get_storage_from_app()
    row = await storage.llm_endpoint_credential_get(endpoint_id, cred_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"credential_not_found: {cred_id}")
    return row


@app.patch("/llm/endpoints/{endpoint_id}/credentials/{cred_id}")
async def llm_endpoint_credential_patch(
    endpoint_id: str, cred_id: str, data: EndpointCredentialPatch
):
    from .backends.postgres import LlmCredentialTokenExists

    storage = get_storage_from_app()
    body = {k: v for k, v in data.model_dump().items() if v is not None}
    try:
        row = await storage.llm_endpoint_credential_update(endpoint_id, cred_id, **body)
    except LlmCredentialTokenExists as exc:
        raise HTTPException(
            status_code=409, detail=f"credential_token_exists: {exc}"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if row is None:
        raise HTTPException(status_code=404, detail=f"credential_not_found: {cred_id}")
    return row


@app.patch("/llm/endpoints/{endpoint_id}/model-rpm")
async def llm_endpoint_model_rpm_patch(endpoint_id: str, data: EndpointModelRpmPatch):
    """Persist RPM for endpoint + logical_id across matching credential rows."""
    storage = get_storage_from_app()
    try:
        rows = await storage.llm_endpoint_credentials_set_rpm_for_model(
            endpoint_id,
            requests_per_minute=int(data.requests_per_minute),
            logical_id=data.logical_id,
            upstream_model_id=data.upstream_model_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not rows:
        raise HTTPException(
            status_code=404,
            detail="no_credentials_for_model",
        )
    rpm = rows[0].get("requests_per_minute")
    return {
        "endpoint_id": endpoint_id,
        "logical_id": data.logical_id,
        "upstream_model_id": data.upstream_model_id,
        "requests_per_minute": rpm,
        "updated": len(rows),
        "credentials": rows,
    }


@app.delete("/llm/endpoints/{endpoint_id}/credentials/{cred_id}")
async def llm_endpoint_credential_delete(endpoint_id: str, cred_id: str):
    storage = get_storage_from_app()
    ok = await storage.llm_endpoint_credential_delete(endpoint_id, cred_id)
    return {"ok": ok, "endpoint_id": endpoint_id, "id": cred_id}


@app.get("/llm/models")
async def llm_local_models_list():
    storage = get_storage_from_app()
    return await storage.llm_local_models_list()


@app.post("/llm/models")
async def llm_local_model_upsert(data: LocalModelUpsert):
    meta = data.metadata or {}
    backend = str(data.backend_kind or "").lower()
    if backend == "kserve":
        raise HTTPException(
            status_code=400,
            detail="backend_kind_deprecated: use external endpoint import or GPU node deploy",
        )
    lid = str(meta.get("logical_id") or "").strip()
    if not lid:
        raise HTTPException(
            status_code=400,
            detail="logical_id_required: set metadata.logical_id when registering a model",
        )
    node_id = str(data.node_id or meta.get("node_id") or meta.get("node") or "").strip()
    if backend == "external" and not data.external_endpoint_id:
        if not (node_id and node_id != "default"):
            raise HTTPException(
                status_code=400,
                detail="provider_required: external_endpoint_id or managed node_id required",
            )
    storage = get_storage_from_app()
    return await storage.llm_local_model_upsert(**data.model_dump())


@app.delete("/llm/models/{model_name}")
async def llm_local_model_delete(model_name: str, node_id: Optional[str] = None):
    storage = get_storage_from_app()
    ok = await storage.llm_local_model_delete(model_name, node_id=node_id)
    return {"ok": ok, "name": model_name, "node_id": node_id}


@app.patch("/llm/models/{node_id}/{model_name}")
async def llm_local_model_patch(node_id: str, model_name: str, data: LocalModelPatch):
    storage = get_storage_from_app()
    body = {k: v for k, v in data.model_dump().items() if v is not None}
    row = await storage.llm_local_model_patch(node_id=node_id, name=model_name, **body)
    if row is None:
        raise HTTPException(status_code=404, detail=f"model_not_found: {node_id}/{model_name}")
    return row


@app.post("/llm/models/{node_id}/{model_name}/touch")
async def llm_local_model_touch(node_id: str, model_name: str, data: ModelUsageTouch):
    """Merge ``metadata.last_used_at`` for pool idle scale-down (fire-and-forget safe)."""
    from datetime import datetime, timezone

    storage = get_storage_from_app()
    stamp = (data.last_used_at or "").strip() or datetime.now(timezone.utc).isoformat()
    ok = await storage.llm_local_model_touch_usage(
        node_id=node_id,
        name=model_name,
        last_used_at=stamp,
    )
    if not ok:
        raise HTTPException(status_code=404, detail=f"model_not_found: {node_id}/{model_name}")
    return {"ok": True, "node_id": node_id, "name": model_name, "last_used_at": stamp}


@app.get("/llm/nodes")
async def llm_nodes_list():
    storage = get_storage_from_app()
    return await storage.llm_nodes_list()


@app.post("/llm/nodes")
async def llm_node_upsert(data: LlmNodeUpsert):
    storage = get_storage_from_app()
    return await storage.llm_node_upsert(**data.model_dump())


@app.patch("/llm/nodes/{node_id}")
async def llm_node_patch(node_id: str, data: LlmNodePatch):
    storage = get_storage_from_app()
    body = {k: v for k, v in data.model_dump().items() if v is not None}
    row = await storage.llm_node_patch(id=node_id, **body)
    if row is None:
        raise HTTPException(status_code=404, detail=f"node_not_found: {node_id}")
    return row


@app.delete("/llm/nodes/{node_id}")
async def llm_node_delete(node_id: str):
    storage = get_storage_from_app()
    ok = await storage.llm_node_delete(id=node_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"node_not_found: {node_id}")
    return {"ok": True, "id": node_id}


@app.get("/llm/nodes/{node_id}/models")
async def llm_node_models(node_id: str):
    storage = get_storage_from_app()
    return await storage.llm_local_models_by_node(node_id)


@app.get("/llm/routes")
async def llm_routes_list():
    storage = get_storage_from_app()
    return await storage.llm_routes_list()


@app.post("/llm/routes")
async def llm_route_upsert(data: RouteUpsert):
    storage = get_storage_from_app()
    try:
        return await storage.llm_route_upsert(**data.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/llm/routes/purge-logical-ids")
async def llm_routes_purge_logical_ids(body: RoutePurgeLogicalIds):
    storage = get_storage_from_app()
    return await storage.llm_routes_purge_logical_ids(body.logical_ids)


@app.post("/llm/routes/enforce-catalog")
async def llm_routes_enforce_catalog_bindings():
    storage = get_storage_from_app()
    return await storage.llm_routes_enforce_catalog_bindings()


@app.delete("/llm/routes/{route_key}")
async def llm_route_delete(route_key: str):
    storage = get_storage_from_app()
    ok = await storage.llm_route_delete(route_key)
    return {"ok": ok, "route_key": route_key}


@app.get("/llm/pool-policies")
async def llm_pool_policies_list():
    storage = get_storage_from_app()
    return await storage.llm_pool_policies_list()


@app.post("/llm/pool-policies")
async def llm_pool_policy_upsert(data: PoolPolicyUpsert):
    storage = get_storage_from_app()
    return await storage.llm_pool_policy_upsert(**data.model_dump())


@app.delete("/llm/pool-policies/{logical_id}")
async def llm_pool_policy_delete(logical_id: str):
    storage = get_storage_from_app()
    ok = await storage.llm_pool_policy_delete(logical_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"policy_not_found: {logical_id}")
    return {"ok": True, "logical_id": logical_id}


@app.get("/llm/candidates")
async def llm_model_candidates_list():
    storage = get_storage_from_app()
    return await storage.llm_model_candidates_list()


@app.post("/llm/candidates")
async def llm_model_candidate_upsert(data: ModelCandidateUpsert):
    storage = get_storage_from_app()
    return await storage.llm_model_candidate_upsert(**data.model_dump())


@app.get("/llm/gpu-budget")
async def llm_gpu_budget(node: Optional[str] = None, max_fraction: float = 0.95):
    storage = get_storage_from_app()
    return await storage.llm_gpu_budget(node=node, max_fraction=max_fraction)


@app.get("/llm/pipeline-tags")
async def llm_pipeline_tags(
    modality: Optional[str] = None,
    needs_review: Optional[bool] = None,
    include_hidden: bool = False,
):
    """Versioned HF Hub pipeline_tag taxonomy (+ AIv2 I/O enrichment)."""
    from .hf_pipeline_taxonomy import (
        ensure_hf_pipeline_taxonomy_schema,
        hf_pipeline_tags_list,
        hf_taxonomy_meta_get,
    )

    storage = get_storage_from_app()
    pg = getattr(storage, "postgres", None)
    if not pg or not pg.pool:
        return {"items": [], "meta": {}, "error": "postgres_unavailable"}
    await ensure_hf_pipeline_taxonomy_schema(pg.pool)
    items = await hf_pipeline_tags_list(
        pg.pool,
        modality=modality,
        needs_review=needs_review,
        include_hidden=include_hidden,
    )
    meta = await hf_taxonomy_meta_get(pg.pool)
    return {"items": items, "meta": meta, "count": len(items)}


@app.get("/llm/io-modalities")
async def llm_io_modalities():
    """Atomic I/O modality vocabulary for Deploy / binding UIs (DB-backed)."""
    from .hf_pipeline_taxonomy import (
        ensure_hf_pipeline_taxonomy_schema,
        io_modalities_ensure_seeded,
    )

    storage = get_storage_from_app()
    pg = getattr(storage, "postgres", None)
    if not pg or not pg.pool:
        from .hf_pipeline_taxonomy import _modality_catalog_seed

        items = _modality_catalog_seed()
        return {"items": items, "count": len(items), "source": "llm_taxonomy_fallback"}
    await ensure_hf_pipeline_taxonomy_schema(pg.pool)
    items = await io_modalities_ensure_seeded(pg.pool)
    return {"items": items, "count": len(items), "source": "postgres"}


@app.get("/llm/pipeline-tags/{tag_id}")
async def llm_pipeline_tag_get(tag_id: str):
    from .hf_pipeline_taxonomy import (
        ensure_hf_pipeline_taxonomy_schema,
        hf_pipeline_tag_get,
    )
    from aiv2_model_contract.hf_pipeline_io import resolve_pipeline_tag_io

    storage = get_storage_from_app()
    pg = getattr(storage, "postgres", None)
    if pg and pg.pool:
        await ensure_hf_pipeline_taxonomy_schema(pg.pool)
        row = await hf_pipeline_tag_get(pg.pool, tag_id)
        if row:
            return row
    # Accept unknown tags: resolve without DB whitelist rejection.
    return resolve_pipeline_tag_io(tag_id)


# ==================== Upload Routes ====================

from fastapi import File, UploadFile, Form
import os
import re
import uuid
import mimetypes
from datetime import datetime

UPLOAD_DIR = "/data/uploads"

# ==================== Auto-extract text on upload ====================
# When the user uploads a textual file (code / markdown / json / csv / log / ...)
# we eagerly read the content, store it as `metadata.text_extract`, and let the
# orchestrator and downstream tools (`session_read`, `session_search`,
# `session_fileinfo`, `text.read`) reuse it. This matches the OCR-on-upload
# pipeline that already exists for images.
#
# Tunables via env vars so ops can dial it up/down for special deployments.

AUTO_EXTRACT_TEXT_ENABLED: bool = os.environ.get(
    "AUTO_EXTRACT_TEXT_ENABLED", "1"
).strip().lower() not in ("0", "false", "no", "off")
# Files larger than this are NOT auto-extracted (still uploadable). 1 MiB default.
AUTO_EXTRACT_MAX_BYTES: int = int(os.environ.get("AUTO_EXTRACT_MAX_BYTES", str(1024 * 1024)))
# We keep `extracted_text` truncated in metadata so Postgres jsonb stays sane.
AUTO_EXTRACT_MAX_CHARS: int = int(os.environ.get("AUTO_EXTRACT_MAX_CHARS", str(500_000)))

# Mirror modules/tools/session_read/tool.py — keep the lists in sync.
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".log",
    ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xml",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env",
    ".py", ".pyi", ".ipynb",
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".java", ".kt", ".kts", ".scala", ".groovy",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",
    ".cs", ".vb", ".fs",
    ".go", ".rs", ".swift", ".m", ".mm",
    ".rb", ".php", ".pl", ".pm", ".lua", ".r", ".jl",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
    ".sql", ".graphql", ".gql", ".proto",
    ".dockerfile", ".tf", ".tfvars", ".hcl",
    ".vue", ".svelte", ".astro",
    ".diff", ".patch",
}

_EXTRA_TEXT_MIMES = {
    "application/json",
    "application/xml",
    "application/x-yaml",
    "application/yaml",
    "application/x-toml",
    "application/x-sh",
    "application/javascript",
    "application/typescript",
}


def _is_textual_upload(filename: Optional[str], mime_type: Optional[str]) -> bool:
    """Return True iff this looks like a text/code/markdown file we can extract."""
    mt = (mime_type or "").lower()
    if mt.startswith("text/"):
        return True
    if mt in _EXTRA_TEXT_MIMES:
        return True
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext and ext in _TEXT_EXTENSIONS:
            return True
    return False


def _extract_text_from_bytes(
    content: bytes,
    *,
    max_chars: int = AUTO_EXTRACT_MAX_CHARS,
) -> Dict[str, Any]:
    """Decode bytes → dict suitable for metadata.text_extract."""
    encoding = "utf-8"
    try:
        import chardet  # type: ignore
        encoding = chardet.detect(content[:10_000]).get("encoding") or "utf-8"
    except Exception:
        pass
    try:
        text = content.decode(encoding)
    except Exception:
        text = content.decode("utf-8", errors="replace")
        encoding = "utf-8"

    total_chars = len(text)
    truncated = total_chars > max_chars
    if truncated:
        text = text[:max_chars]
    total_lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)

    return {
        "extracted_text": text,
        "encoding": encoding,
        "total_chars": total_chars,
        "stored_chars": len(text),
        "total_lines": total_lines,
        "truncated": truncated,
        "auto_extracted": True,
        "extracted_at": datetime.utcnow().isoformat() + "Z",
        "extractor_version": 1,
    }


async def _auto_extract_text_for_artifact(
    storage_obj,
    *,
    artifact_id: Optional[str],
    content: bytes,
    filename: Optional[str],
    mime_type: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    If `content` looks textual and is within size limits, decode it and merge
    a `text_extract` payload into the artifact's metadata.

    Returns the payload (without the full text) on success, ``None`` otherwise.
    Errors are logged but never raised — auto-extract is best-effort.
    """
    if not AUTO_EXTRACT_TEXT_ENABLED:
        return None
    if not artifact_id:
        return None
    if content is None:
        return None
    if len(content) > AUTO_EXTRACT_MAX_BYTES:
        logger.info(
            "📝 auto-extract: skipping %s — too large (%d > %d bytes)",
            filename, len(content), AUTO_EXTRACT_MAX_BYTES,
        )
        return None
    if not _is_textual_upload(filename, mime_type):
        return None

    try:
        payload = _extract_text_from_bytes(content)
    except Exception as exc:
        logger.warning("📝 auto-extract: decode failed for %s: %s", filename, exc)
        return None

    try:
        await storage_obj.artifact_update_metadata(
            artifact_id, {"text_extract": payload}
        )
    except Exception as exc:
        logger.warning(
            "📝 auto-extract: metadata update failed for %s (%s): %s",
            artifact_id, filename, exc,
        )
        return None

    logger.info(
        "📝 auto-extract: stored text_extract for %s (%s) — %d chars (%s)",
        artifact_id, filename, payload["stored_chars"],
        "truncated" if payload["truncated"] else "full",
    )
    summary = {k: v for k, v in payload.items() if k != "extracted_text"}
    return summary


async def _auto_extract_text_from_local_file(
    storage_obj,
    *,
    artifact_id: str,
    file_path: str,
    filename: Optional[str] = None,
    mime_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Best-effort text_extract from a file already on the shared volume."""
    path = str(file_path or "").strip()
    if not path or path.startswith("minio://") or not os.path.isfile(path):
        return None
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    if size > AUTO_EXTRACT_MAX_BYTES:
        return None
    try:
        with open(path, "rb") as handle:
            content = handle.read()
    except OSError as exc:
        logger.warning("📝 auto-extract: read failed for %s: %s", path, exc)
        return None
    return await _auto_extract_text_for_artifact(
        storage_obj,
        artifact_id=artifact_id,
        content=content,
        filename=filename or os.path.basename(path),
        mime_type=mime_type,
    )

@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    context: Optional[str] = Form(None),
    draft_id: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None),
):
    """
    آپلود فایل‌ها به سرور و ثبت در دیتابیس.
    
    Flow:
    1. اگر MinIO فعال باشد، فایل‌ها در MinIO ذخیره می‌شوند
    2. در غیر این صورت، فایل‌ها در /data/uploads ذخیره می‌شوند
    3. اطلاعات فایل در دیتابیس ثبت می‌شود (artifact)
    4. ارتباط فایل با user_id و session_id ثبت می‌شود
    """
    from .config import settings
    
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    storage = get_storage_from_app()
    
    # Check if MinIO is enabled and connected
    use_minio = storage.minio is not None and storage.minio.is_connected
    
    saved = []
    for file in files:
        # ساخت نام یکتا
        ext = os.path.splitext(file.filename or "file")[1]
        unique_id = uuid.uuid4().hex[:12]
        unique_name = f"{unique_id}{ext}"
        
        # Read file content
        content = await file.read()
        
        # تشخیص MIME type و media type
        mime_type, _ = mimetypes.guess_type(file.filename or "")
        mime_type = mime_type or file.content_type
        
        # Determine media type from mime
        media_type = "other"
        if mime_type:
            if mime_type.startswith("image/"):
                media_type = "image"
            elif mime_type.startswith("audio/"):
                media_type = "audio"
            elif mime_type.startswith("video/"):
                media_type = "video"
            elif mime_type.startswith("text/") or mime_type in ("application/pdf", "application/msword"):
                media_type = "document"
        
        # Storage path depends on backend
        upload_ctx = (context or "chat").strip().lower()
        if use_minio:
            # MinIO: Store with new path structure
            artifact_hex = uuid.uuid4().hex[:12]
            
            if upload_ctx == "messenger":
                draft_part = (draft_id or uuid.uuid4().hex[:12]).strip()
                if user_id and conversation_id:
                    object_name = f"messenger/user_{user_id}/conversation_{conversation_id}/artifact_{artifact_hex}{ext}"
                elif user_id:
                    object_name = f"messenger/user_{user_id}/draft_{draft_part}/artifact_{artifact_hex}{ext}"
                else:
                    object_name = f"messenger/anonymous/draft_{draft_part}/artifact_{artifact_hex}{ext}"
            elif user_id and session_id:
                # Chat upload path: /chat/user_{id}/session_{id}/artifact_{id}
                object_name = f"chat/user_{user_id}/session_{session_id}/artifact_{artifact_hex}{ext}"
            elif session_id:
                object_name = f"chat/session_{session_id}/artifact_{artifact_hex}{ext}"
            elif user_id:
                object_name = f"chat/user_{user_id}/artifact_{artifact_hex}{ext}"
            else:
                object_name = f"chat/anonymous/artifact_{artifact_hex}{ext}"
            
            # Upload to MinIO
            minio_path = await storage.file_upload(
                file_data=content,
                object_name=object_name,
                content_type=mime_type or "application/octet-stream",
                metadata={
                    "original_filename": file.filename or "",
                    "user_id": user_id or "",
                    "session_id": session_id or "",
                    "source": "user_messenger" if upload_ctx == "messenger" else "chat",
                    "draft_id": draft_id or "",
                    "conversation_id": conversation_id or "",
                },
            )
            
            if minio_path:
                storage_path = f"minio://{minio_path}"  # Mark as MinIO path
                logger.info(f"📤 File uploaded to MinIO: {minio_path}")
            else:
                # Fallback to local if MinIO upload fails
                storage_path = os.path.join(UPLOAD_DIR, unique_name)
                with open(storage_path, "wb") as f:
                    f.write(content)
                logger.warning(f"⚠️ MinIO upload failed, saved locally: {storage_path}")
        else:
            # Local filesystem
            storage_path = os.path.join(UPLOAD_DIR, unique_name)
            with open(storage_path, "wb") as f:
                f.write(content)
            logger.info(f"📁 File saved locally: {storage_path}")
        
        # ثبت artifact در دیتابیس با user_id و session_id به صورت ستون مستقل
        artifact_data = await storage.artifact_register(
            storage_path=storage_path,
            media_type=media_type,
            original_filename=file.filename,
            file_size_bytes=len(content),
            mime_type=mime_type,
            user_id=user_id,
            session_id=session_id,
            metadata={
                "upload_name": unique_name,
                "storage_backend": "minio" if use_minio else "local",
                "source": "user_messenger" if upload_ctx == "messenger" else "chat",
                "draft_id": draft_id,
                "conversation_id": conversation_id,
            }
        )
        
        artifact_id = artifact_data.get("artifact_id") if artifact_data else unique_name

        # Auto-extract text content for textual files (mirror OCR-on-upload).
        text_extract_summary = await _auto_extract_text_for_artifact(
            storage,
            artifact_id=artifact_id,
            content=content,
            filename=file.filename,
            mime_type=mime_type,
        )

        saved_entry = {
            "id": artifact_id,
            "path": storage_path,
            "name": file.filename,
            "filename": file.filename,
            "mime_type": mime_type,
            "type": media_type,
            "size": len(content),
            "session_id": session_id,
            "user_id": user_id,
            "backend": "minio" if use_minio else "local",
        }
        if text_extract_summary is not None:
            saved_entry["text_extract"] = text_extract_summary
        saved.append(saved_entry)

        logger.info(f"📁 File uploaded: {file.filename} -> {storage_path} (artifact_id={artifact_id}, user={user_id}, session={session_id})")
    
    return {"saved": saved, "count": len(saved)}


def _infer_upload_media_type(mime_type: Optional[str], filename: Optional[str] = None) -> str:
    """Classify media_type for uploads (includes OOXML document MIME types)."""
    mt = (mime_type or "").strip().lower()
    name = (filename or "").strip().lower()
    if mt.startswith("image/"):
        return "image"
    if mt.startswith("audio/"):
        return "audio"
    if mt.startswith("video/"):
        return "video"
    document_mimes = {
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.oasis.opendocument.text",
        "application/rtf",
        "text/rtf",
    }
    if mt.startswith("text/") or mt in document_mimes:
        return "document"
    if name.endswith((".docx", ".doc", ".pdf", ".odt", ".rtf", ".txt", ".md")):
        return "document"
    return "other"


def _safe_plugin_path_segment(value: Optional[str], *, fallback: str = "_") -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())
    text = text.strip("._-")
    return (text[:80] or fallback)


@app.post("/upload/plugin-output")
async def upload_plugin_output(
    request: Request,
    files: List[UploadFile] = File(...),
    plugin_id: Optional[str] = Form(None),
    tool_id: Optional[str] = Form(None),
    flow_id: Optional[str] = Form(None),
    record_id: Optional[str] = Form(None),
    run_id: Optional[str] = Form(None),
    node_kind: Optional[str] = Form(None),
    case_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    folder_id: Optional[str] = Form(None),
    layout_name: Optional[str] = Form(None),
    layout_template: Optional[str] = Form(None),
):
    """
    Durable plugin/tool blob upload owned by tool-runner.

    Ownership and case scope come from verified internal headers (X-User-Id,
    X-Case-Id), not from tool-supplied Form user_id.

    Optional ``layout_name`` / ``layout_template`` select an allowlisted object-key
    template; Storage re-validates and renders with form/header vars.
    """
    from .config import settings as _settings
    from .persist_layouts import render_layout_template, resolve_layout_template

    ctx = _storage_auth_context(request)
    owner_id = (ctx.get("user_id") or "").strip() or None
    if _settings.STORAGE_RBAC_ENABLED and not owner_id:
        raise HTTPException(status_code=401, detail="user_id_required")

    header_case = (ctx.get("case_id") or "").strip() or None
    form_case = (case_id or "").strip() or None
    tmpl_hint = str(layout_template or "").strip().replace("\\", "/")
    layout_l = str(layout_name or "").strip().lower()
    # User-scoped templates (agent_flow flows/runs, sessions, …) must never be
    # rewritten onto case_importer/ just because X-Case-Id is present.
    user_scoped = layout_l == "sessions" or tmpl_hint.startswith("users/")
    if user_scoped:
        bound_case = None
    else:
        bound_case = (form_case or header_case or "").strip() or None
        if header_case and form_case and form_case != header_case:
            raise HTTPException(status_code=400, detail="case_id_mismatch")

    storage = get_storage_from_app()
    if bound_case:
        await _require_case_access(storage, bound_case, request)

    use_minio = storage.minio is not None and storage.minio.is_connected
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    plugin_seg = _safe_plugin_path_segment(plugin_id or "tools", fallback="tools")
    tool_seg = _safe_plugin_path_segment(tool_id or "tool", fallback="tool")
    flow_seg = _safe_plugin_path_segment(flow_id, fallback="_")
    record_seg = _safe_plugin_path_segment(record_id, fallback="_")
    run_seg = _safe_plugin_path_segment(run_id or uuid.uuid4().hex[:12], fallback=uuid.uuid4().hex[:12])
    node_seg = _safe_plugin_path_segment(node_kind, fallback="")

    try:
        key_template = resolve_layout_template(
            layout_name=layout_name,
            layout_template=layout_template,
            has_case=bool(bound_case),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Belt-and-suspenders: never accept a case_importer key for user-scoped uploads.
    if user_scoped and key_template.replace("\\", "/").startswith("case_importer/"):
        raise HTTPException(
            status_code=400,
            detail="user_scoped_layout_cannot_use_case_importer_path",
        )

    saved: List[Dict[str, Any]] = []
    for file in files:
        original_filename = file.filename or "output.bin"
        ext = os.path.splitext(original_filename)[1]
        safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", original_filename).strip() or f"output{ext}"
        safe_name = safe_name[:180]
        content = await file.read()
        mime_type, _ = mimetypes.guess_type(original_filename)
        mime_type = mime_type or file.content_type or "application/octet-stream"
        media_type = _infer_upload_media_type(mime_type, original_filename)
        artifact_hex = uuid.uuid4().hex[:12]
        unique_name = f"{artifact_hex}_{safe_name}"

        layout_values = {
            "user_id": owner_id or "_anonymous",
            "case_id": bound_case or "",
            "session_id": session_id or "",
            "group_id": ctx.get("group_id") or "",
            "plugin_id": plugin_seg,
            "tool_id": tool_seg,
            "flow_id": flow_seg,
            "record_id": record_seg,
            "run_id": run_seg,
            "hex": artifact_hex,
            "filename": safe_name,
            "node_kind": node_seg or "_",
        }
        try:
            object_name = render_layout_template(key_template, layout_values)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if use_minio:
            minio_path = await storage.file_upload(
                file_data=content,
                object_name=object_name,
                content_type=mime_type,
                metadata={
                    "original_filename": original_filename,
                    "user_id": owner_id or "",
                    "session_id": session_id or "",
                    "source": "plugin_output",
                    "plugin_id": plugin_seg,
                    "tool_id": tool_seg,
                    "flow_id": flow_seg,
                    "record_id": record_seg,
                    "run_id": run_seg,
                    "node_kind": node_seg,
                    "layout": str(layout_name or "").strip() or "default",
                    **({"case_id": bound_case} if bound_case else {}),
                },
            )
            if minio_path:
                storage_path = f"minio://{minio_path}"
            else:
                storage_path = os.path.join(UPLOAD_DIR, unique_name)
                with open(storage_path, "wb") as fh:
                    fh.write(content)
                logger.warning("plugin_output MinIO upload failed; saved locally: %s", storage_path)
        else:
            storage_path = os.path.join(UPLOAD_DIR, unique_name)
            with open(storage_path, "wb") as fh:
                fh.write(content)

        meta: Dict[str, Any] = {
            "upload_name": unique_name,
            "storage_backend": "minio" if use_minio else "local",
            "source": "plugin_output",
            "plugin_id": plugin_seg,
            "tool_id": tool_seg,
            "flow_id": flow_seg if flow_seg != "_" else None,
            "record_id": record_seg if record_seg != "_" else None,
            "run_id": run_seg,
            "node_kind": node_seg or None,
            "producer_user_id": owner_id,
            "group_id": ctx.get("group_id"),
            "layout": str(layout_name or "").strip() or "default",
        }
        if bound_case:
            meta["case_id"] = bound_case
        if session_id:
            meta["session_id"] = session_id
        if str(layout_name or "").strip().lower() == "sessions":
            meta["kind"] = meta.get("kind") or "plugin_session"

        artifact_data = await storage.artifact_register(
            storage_path=storage_path,
            media_type=media_type,
            original_filename=original_filename,
            file_size_bytes=len(content),
            mime_type=mime_type,
            user_id=owner_id,
            session_id=session_id,
            folder_id=folder_id,
            metadata={k: v for k, v in meta.items() if v is not None},
        )
        artifact_id = artifact_data.get("artifact_id") if artifact_data else artifact_hex
        saved.append(
            {
                "id": artifact_id,
                "artifact_id": artifact_id,
                "path": storage_path,
                "name": original_filename,
                "filename": original_filename,
                "mime_type": mime_type,
                "type": media_type,
                "media_type": media_type,
                "size": len(content),
                "session_id": session_id,
                "user_id": owner_id,
                "case_id": bound_case,
                "backend": "minio" if use_minio else "local",
                "source": "plugin_output",
            }
        )
        logger.info(
            "plugin_output uploaded: %s -> %s (artifact_id=%s user=%s case=%s)",
            original_filename,
            storage_path,
            artifact_id,
            owner_id,
            bound_case,
        )

    return {"saved": saved, "count": len(saved), "source": "plugin_output"}


# ==================== Chunked/Resumable Upload ====================

# In-memory storage for chunked uploads (in production, use Redis)
_chunked_uploads: Dict[str, Dict[str, Any]] = {}

class ChunkedUploadInit(BaseModel):
    filename: str
    total_size: int
    total_chunks: int
    mime_type: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None


@app.post("/upload/init")
async def init_chunked_upload(data: ChunkedUploadInit):
    """
    Initialize a chunked/resumable upload.
    
    Returns an upload_id to be used for subsequent chunk uploads.
    """
    upload_id = uuid.uuid4().hex
    
    # Create temp directory for chunks
    temp_dir = f"/data/uploads/.chunks/{upload_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    _chunked_uploads[upload_id] = {
        "filename": data.filename,
        "total_size": data.total_size,
        "total_chunks": data.total_chunks,
        "mime_type": data.mime_type,
        "session_id": data.session_id,
        "user_id": data.user_id,
        "temp_dir": temp_dir,
        "received_chunks": set(),
        "created_at": os.times().elapsed,
    }
    
    logger.info(f"📤 Chunked upload initialized: {upload_id} ({data.filename}, {data.total_chunks} chunks)")
    
    return {
        "upload_id": upload_id,
        "total_chunks": data.total_chunks,
        "status": "initialized"
    }


@app.put("/upload/{upload_id}/chunk/{chunk_index}")
async def upload_chunk(
    upload_id: str,
    chunk_index: int,
    file: UploadFile = File(...)
):
    """
    Upload a single chunk of a chunked upload.
    
    chunk_index is 0-based.
    """
    if upload_id not in _chunked_uploads:
        raise HTTPException(status_code=404, detail="Upload not found or expired")
    
    upload_info = _chunked_uploads[upload_id]
    
    if chunk_index < 0 or chunk_index >= upload_info["total_chunks"]:
        raise HTTPException(status_code=400, detail="Invalid chunk index")
    
    # Save chunk to temp file
    chunk_path = os.path.join(upload_info["temp_dir"], f"chunk_{chunk_index:05d}")
    content = await file.read()
    
    with open(chunk_path, "wb") as f:
        f.write(content)
    
    upload_info["received_chunks"].add(chunk_index)
    
    received = len(upload_info["received_chunks"])
    total = upload_info["total_chunks"]
    
    logger.info(f"📤 Chunk {chunk_index + 1}/{total} received for upload {upload_id}")
    
    return {
        "upload_id": upload_id,
        "chunk_index": chunk_index,
        "received_chunks": received,
        "total_chunks": total,
        "complete": received == total
    }


@app.post("/upload/{upload_id}/complete")
async def complete_chunked_upload(upload_id: str):
    """
    Complete a chunked upload by assembling all chunks.
    
    Returns the created artifact info.
    """
    if upload_id not in _chunked_uploads:
        raise HTTPException(status_code=404, detail="Upload not found or expired")
    
    upload_info = _chunked_uploads[upload_id]
    received = len(upload_info["received_chunks"])
    total = upload_info["total_chunks"]
    
    if received != total:
        missing = set(range(total)) - upload_info["received_chunks"]
        raise HTTPException(
            status_code=400,
            detail=f"Missing {total - received} chunks: {sorted(missing)[:10]}..."
        )
    
    # Assemble chunks into final file
    ext = os.path.splitext(upload_info["filename"])[1]
    unique_id = uuid.uuid4().hex[:12]
    final_name = f"{unique_id}{ext}"
    final_path = os.path.join(UPLOAD_DIR, final_name)
    
    try:
        with open(final_path, "wb") as outfile:
            for i in range(total):
                chunk_path = os.path.join(upload_info["temp_dir"], f"chunk_{i:05d}")
                with open(chunk_path, "rb") as chunk_file:
                    outfile.write(chunk_file.read())
        
        # Get final file size
        file_size = os.path.getsize(final_path)
        
        # Clean up temp chunks
        import shutil
        shutil.rmtree(upload_info["temp_dir"], ignore_errors=True)
        
        # Determine media type
        mime_type = upload_info["mime_type"]
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(upload_info["filename"])
        
        media_type = "other"
        if mime_type:
            if mime_type.startswith("image/"):
                media_type = "image"
            elif mime_type.startswith("audio/"):
                media_type = "audio"
            elif mime_type.startswith("video/"):
                media_type = "video"
            elif mime_type.startswith("text/"):
                media_type = "document"
        
        # Register artifact
        storage = get_storage_from_app()
        artifact_data = await storage.artifact_register(
            storage_path=final_path,
            media_type=media_type,
            original_filename=upload_info["filename"],
            file_size_bytes=file_size,
            mime_type=mime_type,
            user_id=upload_info.get("user_id"),
            session_id=upload_info.get("session_id"),
            metadata={
                "upload_type": "chunked",
                "total_chunks": total,
            }
        )
        
        artifact_id = artifact_data.get("artifact_id") if artifact_data else unique_id

        # Auto-extract text content for textual files (mirror OCR-on-upload).
        text_extract_summary = None
        try:
            if _is_textual_upload(upload_info["filename"], mime_type):
                with open(final_path, "rb") as fh:
                    payload_bytes = fh.read(AUTO_EXTRACT_MAX_BYTES + 1)
                if len(payload_bytes) <= AUTO_EXTRACT_MAX_BYTES:
                    text_extract_summary = await _auto_extract_text_for_artifact(
                        storage,
                        artifact_id=artifact_id,
                        content=payload_bytes,
                        filename=upload_info["filename"],
                        mime_type=mime_type,
                    )
        except Exception as exc:
            logger.warning(
                "📝 auto-extract: chunked-upload extraction failed for %s: %s",
                upload_info["filename"], exc,
            )

        # Clean up tracking
        del _chunked_uploads[upload_id]
        
        logger.info(f"✅ Chunked upload completed: {upload_id} -> {artifact_id}")
        
        return {
            "status": "completed",
            "artifact_id": artifact_id,
            "path": final_path,
            "size": file_size,
            "filename": upload_info["filename"],
            "mime_type": mime_type,
            **({"text_extract": text_extract_summary} if text_extract_summary else {}),
        }
        
    except Exception as e:
        logger.error(f"Failed to assemble chunked upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to assemble file: {str(e)}")


@app.get("/upload/{upload_id}/status")
async def get_upload_status(upload_id: str):
    """
    Get status of a chunked upload.
    """
    if upload_id not in _chunked_uploads:
        raise HTTPException(status_code=404, detail="Upload not found or expired")
    
    info = _chunked_uploads[upload_id]
    return {
        "upload_id": upload_id,
        "filename": info["filename"],
        "total_size": info["total_size"],
        "total_chunks": info["total_chunks"],
        "received_chunks": len(info["received_chunks"]),
        "missing_chunks": sorted(set(range(info["total_chunks"])) - info["received_chunks"]),
        "complete": len(info["received_chunks"]) == info["total_chunks"]
    }


@app.delete("/upload/{upload_id}")
async def cancel_chunked_upload(upload_id: str):
    """
    Cancel and clean up a chunked upload.
    """
    if upload_id not in _chunked_uploads:
        raise HTTPException(status_code=404, detail="Upload not found or expired")
    
    info = _chunked_uploads[upload_id]
    
    # Clean up temp files
    import shutil
    shutil.rmtree(info["temp_dir"], ignore_errors=True)
    
    del _chunked_uploads[upload_id]
    
    logger.info(f"🗑️ Chunked upload cancelled: {upload_id}")
    
    return {"status": "cancelled", "upload_id": upload_id}


@app.post("/upload/case")
async def upload_case_file(
    files: List[UploadFile] = File(...),
    case_id: str = Form(...)
):
    """
    Upload files for a case to MinIO storage.
    Path structure: cases/case_{case_id}/filename
    """
    storage = get_storage_from_app()
    
    # Check if MinIO is enabled
    use_minio = hasattr(storage, 'minio') and storage.minio is not None
    if not use_minio:
        raise HTTPException(
            status_code=503,
            detail="MinIO storage is not enabled"
        )
    
    saved = []
    
    for file in files:
        content = await file.read()
        
        # Detect mime type
        try:
            mime = magic.Magic(mime=True)
            mime_type = mime.from_buffer(content[:1024])
        except:
            mime_type = file.content_type or "application/octet-stream"
        
        # Determine media type
        if mime_type.startswith("image/"):
            media_type = "image"
        elif mime_type.startswith("video/"):
            media_type = "video"
        elif mime_type.startswith("audio/"):
            media_type = "audio"
        elif mime_type.startswith("text/"):
            media_type = "text"
        else:
            media_type = "document"
        
        # Build path: cases/case_{case_id}/original_filename
        original_filename = file.filename or "unknown"
        storage_path = f"cases/case_{case_id}/{original_filename}"
        
        # Upload to MinIO
        try:
            minio_path = await storage.file_upload(
                content=content,
                object_name=storage_path,
                content_type=mime_type
            )
        except Exception as e:
            logger.error(f"❌ MinIO upload failed for {original_filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to upload {original_filename}: {str(e)}")
        
        saved.append({
            "path": storage_path,
            "minio_path": minio_path,
            "name": original_filename,
            "filename": original_filename,
            "mime_type": mime_type,
            "type": media_type,
            "size": len(content),
            "case_id": case_id,
            "backend": "minio",
        })
        
        logger.info(f"📁 Case file uploaded: {original_filename} -> {storage_path} (case_id={case_id})")
    
    return {"saved": saved, "count": len(saved)}


@app.post("/upload/case/path")
async def upload_case_file_from_path(
    case_id: str = Form(...),
    file_paths: str = Form(...)  # JSON array of file paths
):
    """
    Upload files from local paths to MinIO for a case.
    This is used by case-importer which already has files on disk.
    Path structure: cases/case_{case_id}/filename
    """
    import json as json_module
    
    storage = get_storage_from_app()
    
    # Check if MinIO is enabled
    use_minio = hasattr(storage, 'minio') and storage.minio is not None
    if not use_minio:
        raise HTTPException(
            status_code=503,
            detail="MinIO storage is not enabled"
        )
    
    try:
        raw = json_module.loads(file_paths)
    except Exception:
        raise HTTPException(status_code=400, detail="file_paths must be a valid JSON array")

    items: List[Dict[str, Any]] = []
    for entry in raw:
        if isinstance(entry, str):
            items.append({"artifact_id": None, "path": entry, "kind": "file"})
        elif isinstance(entry, dict) and entry.get("path"):
            items.append({
                "artifact_id": entry.get("artifact_id"),
                "path": entry["path"],
                "kind": entry.get("kind", "file"),
            })

    saved = []
    for item in items:
        file_path = item["path"]
        artifact_id = item.get("artifact_id")
        kind = item.get("kind", "file")
        local_path = Path(file_path)
        if not local_path.exists():
            logger.warning(f"⚠️ File not found: {file_path}")
            continue

        content = local_path.read_bytes()
        original_filename = local_path.name

        try:
            mime_type, _ = mimetypes.guess_type(original_filename)
            if not mime_type:
                mime_type = "application/octet-stream"
        except Exception:
            mime_type = "application/octet-stream"

        if mime_type.startswith("image/"):
            media_type = "image"
        elif mime_type.startswith("video/"):
            media_type = "video"
        elif mime_type.startswith("audio/"):
            media_type = "audio"
        elif mime_type.startswith("text/"):
            media_type = "text"
        else:
            media_type = "document"

        if kind == "state":
            object_name = f"case_importer/{case_id}/_state/{original_filename}"
        else:
            base_name = f"case_importer/{case_id}/{original_filename}"
            object_name = base_name
            try:
                if await storage.file_exists(base_name) and artifact_id:
                    suffix = artifact_id.split("-")[0]
                    stem, dot, ext = original_filename.rpartition(".")
                    if dot:
                        unique_name = f"{stem}__{suffix}.{ext}"
                    else:
                        unique_name = f"{original_filename}__{suffix}"
                    object_name = f"case_importer/{case_id}/{unique_name}"
            except Exception as exc:
                logger.warning(f"file_exists check failed for {base_name}: {exc}")

        meta_payload = {
            "case-id": case_id,
            "kind": kind,
            "original-filename-b64": __import__("base64").b64encode(
                original_filename.encode("utf-8")
            ).decode("ascii"),
        }
        if artifact_id:
            meta_payload["artifact-id"] = artifact_id

        try:
            uploaded = await storage.file_upload(
                file_data=content,
                object_name=object_name,
                content_type=mime_type,
                metadata=meta_payload,
            )
        except Exception as e:
            logger.error(f"❌ MinIO upload failed for {original_filename}: {e}")
            continue

        if not uploaded:
            logger.error(f"❌ MinIO upload returned no path for {original_filename}")
            continue

        canonical = uploaded if uploaded.startswith("minio://") else f"minio://{uploaded}"

        saved.append({
            "path": object_name,
            "minio_path": canonical,
            "local_path": str(file_path),
            "artifact_id": artifact_id,
            "kind": kind,
            "name": original_filename,
            "filename": original_filename,
            "mime_type": mime_type,
            "type": media_type,
            "size": len(content),
            "case_id": case_id,
            "backend": "minio",
        })

        logger.info(f"📁 Case file uploaded: {file_path} -> {canonical} (artifact={artifact_id})")

    return {"saved": saved, "count": len(saved)}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Storage Service",
        "version": "1.0.0",
        "description": "سرویس ذخیره‌سازی AIv2",
        "endpoints": {
            "health": "/health",
            "persons": "/persons",
            "folders": "/folders",
            "artifacts": "/artifacts",
            "traces": "/traces",
            "vectors": "/vectors",
            "graph": "/graph",
            "images": "/images",
            "audio": "/audio",
            "chat": "/chat"
        }
    }


# ==================== Person Routes ====================

@app.post("/persons", response_model=Dict[str, Any])
async def create_person(data: PersonCreate):
    """Create a new person"""
    storage = get_storage_from_app()
    result = await storage.person_create(
        full_name=data.full_name,
        external_id=data.external_id,
        aliases=data.aliases,
        metadata=data.metadata
    )
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create person")
    return result


@app.get("/persons/{person_id}", response_model=Dict[str, Any])
async def get_person(person_id: str):
    """Get a person by ID"""
    storage = get_storage_from_app()
    result = await storage.person_get(person_id)
    if not result:
        raise HTTPException(status_code=404, detail="Person not found")
    return result


@app.get("/persons", response_model=List[Dict[str, Any]])
async def list_persons(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None)
):
    """List persons with pagination"""
    storage = get_storage_from_app()
    return await storage.person_list(limit=limit, offset=offset, search=search)


# ==================== Folder Routes ====================

@app.post("/folders", response_model=Dict[str, Any])
async def create_folder(data: FolderCreate):
    """Create or get a folder"""
    storage = get_storage_from_app()
    result = await storage.folder_get_or_create(
        name=data.name,
        parent_id=data.parent_id,
        metadata=data.metadata
    )
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create folder")
    return result


@app.post("/folders/ensure-chain", response_model=Dict[str, Any])
async def ensure_folder_chain(data: FolderEnsureChain):
    """
    Ensure a nested folder chain exists and return the leaf folder.

    Accepts ``names`` or ``segments`` (alias used by case-importer client).
    Optional ``user_id`` / ``case_id`` are stamped onto the leaf folder metadata.
    """
    storage = get_storage_from_app()
    chain = list(data.names or data.segments or [])
    if not chain:
        raise HTTPException(status_code=400, detail="names_required")
    for seg in chain:
        text = str(seg or "").strip()
        if not text or "/" in text or "\\" in text or ".." in text:
            raise HTTPException(status_code=400, detail=f"invalid_folder_segment:{seg}")

    parent_id = data.parent_id
    leaf: Optional[Dict[str, Any]] = None
    path_parts: List[str] = []
    for idx, raw_name in enumerate(chain):
        name = str(raw_name).strip()
        path_parts.append(name)
        meta = None
        if idx == len(chain) - 1:
            meta = dict(data.metadata or {})
            if data.user_id:
                meta.setdefault("user_id", data.user_id)
            if data.case_id:
                meta.setdefault("case_id", data.case_id)
        leaf = await storage.folder_get_or_create(
            name=name,
            parent_id=parent_id,
            metadata=meta,
        )
        if not leaf or not leaf.get("id"):
            raise HTTPException(status_code=500, detail="Failed to ensure folder chain")
        parent_id = str(leaf["id"])

    assert leaf is not None
    return {
        **leaf,
        "folder_id": leaf.get("id"),
        "path": "/".join(path_parts),
        "segments": path_parts,
    }


# ==================== Artifact Routes ====================

@app.post("/artifacts", response_model=Dict[str, Any])
async def register_artifact(data: ArtifactCreate):
    """Register a new artifact"""
    storage = get_storage_from_app()
    result = await storage.artifact_register(
        storage_path=data.storage_path,
        media_type=data.media_type,
        original_filename=data.original_filename,
        folder_id=data.folder_id,
        checksum=data.checksum,
        file_size_bytes=data.file_size_bytes,
        mime_type=data.mime_type,
        user_id=data.user_id,
        session_id=data.session_id,
        metadata=data.metadata
    )
    if not result:
        raise HTTPException(status_code=500, detail="Failed to register artifact")
    artifact_id = result.get("artifact_id") or result.get("id")
    if artifact_id and data.storage_path and not str(data.storage_path).startswith("minio://"):
        await _auto_extract_text_from_local_file(
            storage,
            artifact_id=str(artifact_id),
            file_path=str(data.storage_path),
            filename=data.original_filename,
            mime_type=data.mime_type,
        )
        refreshed = await storage.artifact_get(str(artifact_id))
        if refreshed:
            result = refreshed
    return result


@app.get("/artifacts/lookup-by-checksum", response_model=Dict[str, Any])
async def lookup_artifact_by_checksum(
    checksum: str = Query(..., description="SHA-256 hex digest of file bytes"),
    user_id: Optional[str] = Query(None, description="If set, only match artifacts owned by this user"),
):
    """
    Deterministic duplicate detection: same file bytes => same SHA-256.
    Returns the oldest matching artifact (canonical row for dedup).
    """
    storage = get_storage_from_app()
    if not storage.postgres:
        raise HTTPException(status_code=503, detail="PostgreSQL not available")
    row = await storage.artifact_find_by_checksum(checksum, user_id=user_id)
    if not row:
        raise HTTPException(status_code=404, detail="No artifact with this checksum")
    return row


@app.get("/artifacts/lookup-by-import-locator", response_model=Dict[str, Any])
async def lookup_artifact_by_import_locator(
    relative_path: str = Query(
        ...,
        description="مسیر نسبی از ریشهٔ پوشهٔ import تا فایل (سبک posix)",
    ),
    filename: str = Query(..., description="نام فایل (basename)"),
    user_id: Optional[str] = Query(
        None,
        description="در صورت تنظیم، فقط artifact های همان کاربر",
    ),
):
    """
    تشخیص تکرار سبک: همان (مسیر نسبی + نام فایل) در metadata ثبت‌شدهٔ import قبلی.
    """
    storage = get_storage_from_app()
    if not storage.postgres:
        raise HTTPException(status_code=503, detail="PostgreSQL not available")
    row = await storage.artifact_find_by_import_locator(
        relative_path, filename, user_id=user_id
    )
    if not row:
        raise HTTPException(
            status_code=404,
            detail="No artifact with this import locator",
        )
    return row


@app.get("/artifacts/{artifact_id}", response_model=Dict[str, Any])
async def get_artifact(artifact_id: str, request: Request):
    """Get an artifact by ID"""
    storage = get_storage_from_app()
    result = await storage.artifact_get(artifact_id)
    if not result:
        raise HTTPException(status_code=404, detail="Artifact not found")
    await _require_artifact_access(storage, artifact_id, request, artifact=result)
    return result


@app.delete("/artifacts/{artifact_id}", response_model=Dict[str, Any])
async def delete_artifact(
    artifact_id: str,
    request: Request,
    purge_minio: bool = Query(True),
    purge_all: bool = Query(False, description="Also purge Mongo, Qdrant, Neo4j, PG tool_* refs"),
):
    """Delete an artifact. Use purge_all=true for full cross-DB cleanup."""
    storage = get_storage_from_app()
    # Enforce delete-level access only for user-scoped callers. Internal callers
    # (no X-User-Id, e.g. case-importer cleanup) keep the existing behaviour.
    if _storage_auth_context(request).get("user_id"):
        await _require_artifact_access(storage, artifact_id, request, action="delete")
    if purge_all:
        result = await storage.artifact_purge_full(artifact_id, purge_minio=purge_minio)
    else:
        result = await storage.artifact_delete(artifact_id, purge_minio=purge_minio)
    if not result.get("ok"):
        if result.get("error") == "artifact_not_found":
            raise HTTPException(status_code=404, detail="Artifact not found")
        raise HTTPException(status_code=500, detail=str(result.get("error") or "delete_failed"))
    return result


@app.get("/artifacts/{artifact_id}/tools", response_model=List[Dict[str, Any]])
async def artifact_tools(artifact_id: str, request: Request):
    """List tools with persisted output for this artifact (Postgres + Mongo)."""
    storage = get_storage_from_app()
    await _require_artifact_access(storage, artifact_id, request)
    pg_tools = await storage.tools_for_artifact(artifact_id)
    seen = {str(t.get("tool_id")) for t in pg_tools if t.get("tool_id")}
    if storage.mongo is not None and storage.mongo.db is not None:
        try:
            names = await storage.mongo.db.tool_outputs.distinct(
                "tool_name", {"artifact_id": artifact_id}
            )
            for name in names:
                tid = str(name or "").strip()
                if not tid or tid in seen:
                    continue
                seen.add(tid)
                pg_tools.append(
                    {
                        "tool_id": tid,
                        "source": "mongo",
                        "row_count": None,
                    }
                )
        except Exception as exc:
            logger.warning("mongo tool discovery failed for %s: %s", artifact_id, exc)
    return pg_tools


@app.get("/artifacts/{artifact_id}/everywhere", response_model=Dict[str, Any])
async def artifact_everywhere(
    request: Request,
    artifact_id: str,
    user_id: Optional[str] = Query(None, description="If provided, RBAC is enforced before discovery"),
    include: Optional[str] = Query(
        None,
        description=(
            "Comma-separated subset of backends to scan: "
            "minio,postgres,mongo,qdrant,neo4j. Defaults to all."
        ),
    ),
    per_table_limit: int = Query(200, ge=1, le=2000),
    per_collection_limit: int = Query(200, ge=1, le=2000),
    per_qdrant_limit: int = Query(100, ge=1, le=1000),
    max_relationships: int = Query(200, ge=1, le=2000),
):
    """
    Generic cross-DB discovery for a single artifact.

    Algorithm (independent of any tool):
      1. (Optional) RBAC enforced if `user_id` is supplied.
      2. Postgres: scan information_schema for every table that has an
         `artifact_id` column → SELECT WHERE artifact_id = $1 LIMIT N.
      3. Mongo: list_collection_names → find({"artifact_id": aid}) per coll.
      4. Qdrant: get_collections → scroll(filter artifact_id == aid).
      5. Neo4j: MATCH (n {artifact_id: $aid}) and 1-hop relationships.
      6. MinIO: stat the artifact's storage_path.

    The result tells the UI everything any backend knows about this artifact —
    tool outputs, image/audio metadata, faces, vectors, graph subgraph, S3 stat
    — without the plugin needing per-feature endpoints.
    """
    storage = get_storage_from_app()

    if user_id:
        access = await storage.artifact_is_accessible(
            user_id=user_id, artifact_id=artifact_id
        )
        if not (access or {}).get("accessible"):
            raise HTTPException(status_code=403, detail="not_authorized_for_artifact")
    else:
        await _require_artifact_access(storage, artifact_id, request)

    inc_list: Optional[List[str]] = None
    if include:
        inc_list = [s.strip() for s in include.split(",") if s.strip()]

    return await storage.artifact_everywhere(
        artifact_id,
        include=inc_list,
        per_table_limit=per_table_limit,
        per_collection_limit=per_collection_limit,
        per_qdrant_limit=per_qdrant_limit,
        max_relationships=max_relationships,
    )


# ==================== Case importer (prefs + tool snapshot) ====================


class CaseImporterPrefsUpdate(BaseModel):
    """tool_allowlist: null = all tools; [] = none; non-empty = only those tool ids."""

    tool_allowlist: Optional[List[str]] = None


@app.get("/case-importer/preferences/{user_id}")
async def case_importer_prefs_get(user_id: str, request: Request):
    _storage_auth_context(request)
    storage = get_storage_from_app()
    if not storage.postgres:
        raise HTTPException(status_code=503, detail="PostgreSQL not available")
    row = await storage.case_importer_prefs_get(user_id)
    if not row:
        return {"user_id": user_id, "tool_allowlist": None, "updated_at": None}
    return row


@app.put("/case-importer/preferences/{user_id}")
async def case_importer_prefs_put(
    user_id: str, data: CaseImporterPrefsUpdate, request: Request
):
    _storage_auth_context(request)
    storage = get_storage_from_app()
    if not storage.postgres:
        raise HTTPException(status_code=503, detail="PostgreSQL not available")
    row = await storage.case_importer_prefs_upsert(user_id, data.tool_allowlist)
    if not row:
        raise HTTPException(status_code=500, detail="Failed to save preferences")
    return row


@app.get("/case-importer/artifacts/{artifact_id}/tools/{tool_id}/snapshot")
async def case_importer_tool_snapshot(artifact_id: str, tool_id: str, request: Request):
    """
    Reconstruct persisted channels for one tool run (Mongo tool_outputs + PG metadata tables).
    Used to skip re-execution when checksum matches and tool version is current.
    """
    from .case_importer_support import (
        mongo_find_tool_documents,
        channels_dict_from_storage,
        max_stored_tool_version,
        storage_has_any_tool_output,
        tool_id_lookup_variants,
    )

    storage = get_storage_from_app()
    if not storage.postgres:
        raise HTTPException(status_code=503, detail="PostgreSQL not available")
    await _require_artifact_access(storage, artifact_id, request)

    mongo_docs: List[Dict[str, Any]] = []
    if storage.mongo is not None and storage.mongo.db is not None:
        mongo_docs = await mongo_find_tool_documents(
            storage.mongo.db, artifact_id, tool_id
        )

    pg_rows: List[Dict[str, Any]] = []
    for tid in tool_id_lookup_variants(tool_id):
        rows = await storage.tool_metadata_rows_for_artifact(tid, artifact_id)
        if rows:
            pg_rows = rows
            break
    if not storage_has_any_tool_output(mongo_docs, pg_rows):
        return {
            "ok": True,
            "artifact_id": artifact_id,
            "tool_id": tool_id,
            "has_output": False,
            "stored_tool_version": None,
            "channels": {},
        }

    channels = channels_dict_from_storage(mongo_docs, pg_rows)
    ver = max_stored_tool_version(mongo_docs, pg_rows)
    return {
        "ok": True,
        "artifact_id": artifact_id,
        "tool_id": tool_id,
        "has_output": True,
        "stored_tool_version": ver,
        "channels": channels,
    }


@app.get("/artifacts/{artifact_id}/download")
async def download_artifact(
    request: Request,
    artifact_id: str,
    mode: str = Query("inline", description="'inline' for browser view, 'attachment' for download"),
    redirect: bool = Query(
        False,
        description=(
            "If true and object is on MinIO, return 302 to a short-lived media-edge "
            "presigned URL (browser CDN path). Default false: stream bytes via MinIO "
            "ClusterIP for in-cluster clients (tool-runner, case-importer, gateway proxy)."
        ),
    ),
    range: Optional[str] = None
):
    """
    Download/serve artifact file content.
    
    - mode=inline: Browser displays the file (images, PDFs, etc.)
    - mode=attachment: Browser downloads the file
    - redirect=true: optional CDN offload (302 → media-edge); never the default
    - Supports Range requests for partial content/resumable downloads
    - Includes ETag and Cache-Control headers
    
    Supports both MinIO (S3) and local filesystem storage.
    """
    from fastapi import Header, Request
    from fastapi.responses import StreamingResponse, Response
    import io
    import hashlib
    
    storage = get_storage_from_app()
    artifact = await storage.artifact_get(artifact_id)
    
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    await _require_artifact_access(storage, artifact_id, request, artifact=artifact)
    
    storage_path = artifact.get("storage_path")
    if not storage_path:
        raise HTTPException(status_code=404, detail="File path not found")
    
    # Determine MIME type
    mime_type = artifact.get("mime_type")
    if not mime_type:
        mime_type, _ = mimetypes.guess_type(storage_path)
        mime_type = mime_type or "application/octet-stream"
    
    # Original filename for Content-Disposition
    filename = artifact.get("original_filename") or os.path.basename(storage_path)
    
    # Set Content-Disposition based on mode
    disposition = "inline" if mode == "inline" else "attachment"
    
    # Check if file is stored in MinIO
    if storage_path.startswith("minio://"):
        # MinIO path format: minio://bucket/object_name
        minio_path = storage_path[8:]  # Remove "minio://" prefix

        if not (storage.minio and storage.minio.is_connected):
            # Startup race: MinIO may have been down when storage connected.
            if storage.minio and not storage.minio.is_connected:
                try:
                    await storage.minio.connect()
                except Exception as exc:
                    logger.warning("MinIO reconnect failed: %s", exc)
            if not (storage.minio and storage.minio.is_connected):
                raise HTTPException(status_code=503, detail="MinIO not available")

        object_name = minio_path.split("/", 1)[1] if "/" in minio_path else minio_path

        # Opt-in CDN: browser fetches media-edge directly. In-cluster defaults
        # must stream (public LB is slower/unreachable from pods).
        if redirect:
            presigned_url = await storage.file_get_url(object_name, expires_seconds=900)
            if presigned_url:
                from fastapi.responses import RedirectResponse
                return RedirectResponse(
                    url=presigned_url,
                    status_code=302,
                    headers={
                        "Cache-Control": "private, max-age=60",
                        "Access-Control-Expose-Headers": (
                            "Content-Disposition, Content-Length, ETag, Accept-Ranges, Location"
                        ),
                    },
                )

        content = await storage.file_download(object_name)

        if content:
            etag = hashlib.md5(content[:1024]).hexdigest()
            return StreamingResponse(
                io.BytesIO(content),
                media_type=mime_type,
                headers={
                    "Content-Disposition": _content_disposition(disposition, filename),
                    "Content-Length": str(len(content)),
                    "Access-Control-Expose-Headers": "Content-Disposition, Content-Length, ETag, Accept-Ranges",
                    "Cache-Control": "private, max-age=3600",
                    "ETag": f'"{etag}"',
                    "Accept-Ranges": "bytes",
                },
            )
        logger.error(f"Failed to download from MinIO: {minio_path}")
        raise HTTPException(status_code=404, detail="File not found in MinIO")
    
    # Local filesystem
    resolved_storage_path = _resolve_existing_local_artifact_path(artifact)
    if resolved_storage_path:
        if resolved_storage_path != storage_path:
            logger.warning(
                "Recovered local artifact path for %s: %s -> %s",
                artifact_id,
                storage_path,
                resolved_storage_path,
            )
            try:
                await storage.artifact_update_storage_path(artifact_id, resolved_storage_path)
                artifact["storage_path"] = resolved_storage_path
            except Exception as exc:
                logger.warning("Failed to self-heal storage_path for %s: %s", artifact_id, exc)
        storage_path = resolved_storage_path
    else:
        logger.error(f"File not found at path: {storage_path}")
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    # Get file stats for ETag and size
    stat = os.stat(storage_path)
    file_size = stat.st_size
    etag = hashlib.md5(f"{storage_path}:{stat.st_mtime}:{file_size}".encode()).hexdigest()
    
    # Common headers
    base_headers = {
        "Content-Disposition": _content_disposition(disposition, filename),
        "Access-Control-Expose-Headers": "Content-Disposition, Content-Length, Content-Range, ETag, Accept-Ranges",
        "Cache-Control": "private, max-age=3600",
        "ETag": f'"{etag}"',
        "Accept-Ranges": "bytes"
    }
    
    # Check for Range header (partial content request)
    if range and range.startswith("bytes="):
        try:
            range_spec = range[6:]  # Remove "bytes=" prefix
            
            if "-" in range_spec:
                start_str, end_str = range_spec.split("-", 1)
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1
            else:
                start = int(range_spec)
                end = file_size - 1
            
            # Validate range
            if start >= file_size or end >= file_size or start > end:
                return Response(
                    status_code=416,
                    headers={"Content-Range": f"bytes */{file_size}"}
                )
            
            # Create generator for partial content
            content_length = end - start + 1
            
            def iter_file_range():
                with open(storage_path, "rb") as f:
                    f.seek(start)
                    remaining = content_length
                    chunk_size = 8192
                    while remaining > 0:
                        read_size = min(chunk_size, remaining)
                        data = f.read(read_size)
                        if not data:
                            break
                        remaining -= len(data)
                        yield data
            
            return StreamingResponse(
                iter_file_range(),
                status_code=206,
                media_type=mime_type,
                headers={
                    **base_headers,
                    "Content-Length": str(content_length),
                    "Content-Range": f"bytes {start}-{end}/{file_size}"
                }
            )
        except (ValueError, IndexError):
            # Invalid range format, return full file
            pass
    
    # Return full file
    return FileResponse(
        path=storage_path,
        media_type=mime_type,
        filename=filename,
        headers=base_headers
    )


# ==================== File Management (UX-009 to UX-013) ====================

@app.get("/files/{artifact_id}/thumbnail")
async def get_file_thumbnail(
    request: Request,
    artifact_id: str,
    width: int = Query(200, ge=32, le=800),
    height: int = Query(200, ge=32, le=800),
    format: str = Query("webp", description="Output format: webp, jpeg, png"),
    quality: int = Query(80, ge=10, le=100, description="Image quality (10-100)"),
):
    """
    Get image thumbnail (UX-009).
    
    Generates/returns a thumbnail for image artifacts.
    Supports WebP, JPEG, PNG formats with quality control.
    """
    import hashlib
    
    storage = get_storage_from_app()
    artifact = await storage.artifact_get(artifact_id)
    
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    await _require_artifact_access(storage, artifact_id, request, artifact=artifact)
    
    mime_type = artifact.get("mime_type", "")
    if not mime_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Not an image artifact")
    
    storage_path = artifact.get("storage_path")
    if not storage_path:
        raise HTTPException(status_code=404, detail="File path not found")
    
    if not storage_path.startswith("/"):
        storage_path = f"/data/uploads/{storage_path}"
    
    if not os.path.exists(storage_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    # Normalize format
    format = format.lower()
    if format not in ("webp", "jpeg", "jpg", "png"):
        format = "webp"
    if format == "jpg":
        format = "jpeg"
    
    # Format to extension and mime type mapping
    format_map = {
        "webp": ("webp", "image/webp"),
        "jpeg": ("jpg", "image/jpeg"),
        "png": ("png", "image/png")
    }
    ext, response_mime = format_map[format]
    
    # Check for cached thumbnail
    thumb_dir = "/data/uploads/.thumbnails"
    os.makedirs(thumb_dir, exist_ok=True)
    thumb_filename = f"{artifact_id}_{width}x{height}_q{quality}.{ext}"
    thumb_path = os.path.join(thumb_dir, thumb_filename)
    
    if not os.path.exists(thumb_path):
        # Generate thumbnail using PIL
        try:
            from PIL import Image
            with Image.open(storage_path) as img:
                img.thumbnail((width, height), Image.Resampling.LANCZOS)
                
                # Handle transparency for JPEG
                if format == "jpeg" and img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                
                # Save with format-specific options
                save_kwargs = {"quality": quality}
                if format == "webp":
                    save_kwargs["method"] = 4  # Balance speed/quality
                
                img.save(thumb_path, format.upper(), **save_kwargs)
        except Exception as e:
            logger.error(f"Thumbnail generation failed: {e}")
            raise HTTPException(status_code=500, detail="Thumbnail generation failed")
    
    # Generate ETag from file path and mtime
    stat = os.stat(thumb_path)
    etag = hashlib.md5(f"{thumb_path}:{stat.st_mtime}".encode()).hexdigest()
    
    return FileResponse(
        path=thumb_path,
        media_type=response_mime,
        headers={
            "Cache-Control": "public, max-age=604800, immutable",
            "ETag": f'"{etag}"'
        }
    )


@app.get("/files/{artifact_id}/presigned-url")
async def get_file_presigned_url(
    request: Request,
    artifact_id: str,
    expires: int = Query(900, ge=60, le=3600, description="URL expiry in seconds (max 1h)"),
):
    """
    Get a presigned URL for direct file download.
    
    Only works for files stored in MinIO. For local files, returns the regular download URL.
    """
    storage = get_storage_from_app()
    artifact = await storage.artifact_get(artifact_id)
    
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    await _require_artifact_access(storage, artifact_id, request, artifact=artifact)
    
    storage_path = artifact.get("storage_path")
    if not storage_path:
        raise HTTPException(status_code=404, detail="File path not found")
    
    # Check if file is stored in MinIO
    if storage_path.startswith("minio://") and storage.minio and storage.minio.is_connected:
        minio_path = storage_path[8:]  # Remove "minio://" prefix
        object_name = minio_path.split("/", 1)[1] if "/" in minio_path else minio_path
        
        url = await storage.file_get_url(object_name, expires_seconds=expires)
        if url:
            return {
                "url": url,
                "expires_in": expires,
                "method": "GET"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to generate presigned URL")
    
    # For local files, return the regular download URL
    return {
        "url": f"/artifacts/{artifact_id}/download",
        "expires_in": None,
        "method": "GET",
        "note": "Direct download URL (file stored locally)"
    }


@app.get("/files/{artifact_id}/metadata")
async def get_file_metadata(artifact_id: str, request: Request):
    """
    Get detailed file metadata (UX-010).
    
    Extracts and returns file metadata including:
    - Basic info (size, type, created)
    - Image: dimensions, EXIF data
    - Audio: duration, bitrate, sample rate
    - Video: dimensions, duration, codec
    - Document: page count, word count (if available)
    """
    storage = get_storage_from_app()
    artifact = await storage.artifact_get(artifact_id)
    
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    await _require_artifact_access(storage, artifact_id, request, artifact=artifact)
    
    storage_path = artifact.get("storage_path")
    if not storage_path:
        raise HTTPException(status_code=404, detail="File path not found")
    
    if not storage_path.startswith("/"):
        storage_path = f"/data/uploads/{storage_path}"
    
    if not os.path.exists(storage_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    # Basic metadata
    file_stat = os.stat(storage_path)
    metadata = {
        "artifact_id": artifact_id,
        "filename": artifact.get("original_filename", os.path.basename(storage_path)),
        "mime_type": artifact.get("mime_type"),
        "size_bytes": file_stat.st_size,
        "created_at": artifact.get("created_at"),
        "modified_at": file_stat.st_mtime,
    }
    
    mime_type = artifact.get("mime_type", "")
    
    # Image metadata
    if mime_type.startswith("image/"):
        try:
            from PIL import Image
            with Image.open(storage_path) as img:
                metadata["width"] = img.width
                metadata["height"] = img.height
                metadata["format"] = img.format
                metadata["mode"] = img.mode
                
                # EXIF data
                exif = img.getexif()
                if exif:
                    metadata["exif"] = {
                        "camera": exif.get(271),  # Make
                        "model": exif.get(272),   # Model
                        "datetime": exif.get(306),  # DateTime
                        "orientation": exif.get(274),
                    }
        except Exception as e:
            logger.warning(f"Could not extract image metadata: {e}")
    
    # Audio/Video metadata (using ffprobe if available)
    elif mime_type.startswith(("audio/", "video/")):
        try:
            import subprocess
            import json as json_mod
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", storage_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                probe_data = json_mod.loads(result.stdout)
                fmt = probe_data.get("format", {})
                metadata["duration_seconds"] = float(fmt.get("duration", 0))
                metadata["bitrate"] = int(fmt.get("bit_rate", 0))
                
                for stream in probe_data.get("streams", []):
                    if stream.get("codec_type") == "video":
                        metadata["video_width"] = stream.get("width")
                        metadata["video_height"] = stream.get("height")
                        metadata["video_codec"] = stream.get("codec_name")
                    elif stream.get("codec_type") == "audio":
                        metadata["audio_codec"] = stream.get("codec_name")
                        metadata["sample_rate"] = stream.get("sample_rate")
                        metadata["channels"] = stream.get("channels")
        except Exception as e:
            logger.warning(f"Could not extract media metadata: {e}")
    
    return metadata


@app.put("/files/{artifact_id}/analyze")
async def request_file_analysis(artifact_id: str):
    """
    Request AI analysis of a file (UX-011).
    
    Triggers async analysis job and returns job ID for status tracking.
    Analysis includes: OCR, face detection, content description, etc.
    """
    storage = get_storage_from_app()
    artifact = await storage.artifact_get(artifact_id)
    
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    # Check if analysis already exists
    existing_analysis = artifact.get("analysis")
    if existing_analysis:
        return {
            "status": "completed",
            "analysis": existing_analysis,
            "message": "Analysis already available"
        }
    
    # For now, mark as pending (actual async implementation would use task queue)
    # This is a placeholder that returns immediately
    import uuid
    job_id = str(uuid.uuid4())[:8]
    
    return {
        "status": "pending",
        "job_id": job_id,
        "artifact_id": artifact_id,
        "message": "Analysis queued. Check back later or use /files/{id}/analysis/status"
    }


@app.get("/files/{artifact_id}/preview")
async def get_file_preview(artifact_id: str, page: int = Query(1, ge=1)):
    """
    Get preview for non-image files (UX-012).
    
    For PDFs: Returns first page as image
    For documents: Returns text excerpt
    For code: Returns syntax-highlighted preview
    """
    storage = get_storage_from_app()
    artifact = await storage.artifact_get(artifact_id)
    
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    storage_path = artifact.get("storage_path")
    if not storage_path:
        raise HTTPException(status_code=404, detail="File path not found")
    
    if not storage_path.startswith("/"):
        storage_path = f"/data/uploads/{storage_path}"
    
    if not os.path.exists(storage_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    mime_type = artifact.get("mime_type", "")
    filename = artifact.get("original_filename", "")
    
    # PDF preview
    if mime_type == "application/pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(storage_path)
            if page > len(doc):
                raise HTTPException(status_code=400, detail=f"Page {page} does not exist")
            
            pdf_page = doc[page - 1]
            pix = pdf_page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for quality
            
            # Save to memory
            import io
            img_bytes = pix.tobytes("png")
            
            return StreamingResponse(
                io.BytesIO(img_bytes),
                media_type="image/png",
                headers={"X-Total-Pages": str(len(doc))}
            )
        except ImportError:
            return {"error": "PDF preview not available", "message": "PyMuPDF not installed"}
        except Exception as e:
            logger.error(f"PDF preview failed: {e}")
            raise HTTPException(status_code=500, detail="Preview generation failed")
    
    # Text/code preview
    elif mime_type.startswith("text/") or filename.endswith((".py", ".js", ".json", ".md", ".yaml", ".yml")):
        try:
            with open(storage_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(10000)  # First 10KB
            
            return {
                "content": content,
                "truncated": len(content) >= 10000,
                "mime_type": mime_type,
            }
        except Exception as e:
            logger.error(f"Text preview failed: {e}")
            raise HTTPException(status_code=500, detail="Preview failed")
    
    else:
        return {
            "message": "Preview not available for this file type",
            "mime_type": mime_type,
        }


@app.get("/storage/quota")
async def get_storage_quota(user_id: Optional[str] = Query(None)):
    """
    Get storage quota information (UX-013).
    
    Returns used space, quota limit, and breakdown by file type.
    """
    storage = get_storage_from_app()
    
    # Calculate used storage
    upload_dir = "/data/uploads"
    total_used = 0
    by_type = {"image": 0, "audio": 0, "video": 0, "document": 0, "other": 0}
    file_count = 0
    
    try:
        for root, dirs, files in os.walk(upload_dir):
            # Skip thumbnails directory
            if ".thumbnails" in root:
                continue
            for filename in files:
                filepath = os.path.join(root, filename)
                try:
                    size = os.path.getsize(filepath)
                    total_used += size
                    file_count += 1
                    
                    # Categorize by extension
                    ext = filename.lower().split(".")[-1] if "." in filename else ""
                    if ext in ("jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff"):
                        by_type["image"] += size
                    elif ext in ("mp3", "wav", "ogg", "flac", "m4a", "aac"):
                        by_type["audio"] += size
                    elif ext in ("mp4", "mkv", "avi", "mov", "webm"):
                        by_type["video"] += size
                    elif ext in ("pdf", "doc", "docx", "txt", "md", "xls", "xlsx"):
                        by_type["document"] += size
                    else:
                        by_type["other"] += size
                except OSError:
                    pass
    except Exception as e:
        logger.error(f"Quota calculation failed: {e}")
    
    # Default quota: 10GB (configurable)
    quota_bytes = int(os.getenv("STORAGE_QUOTA_BYTES", 10 * 1024 * 1024 * 1024))
    
    return {
        "used_bytes": total_used,
        "used_human": _format_size(total_used),
        "quota_bytes": quota_bytes,
        "quota_human": _format_size(quota_bytes),
        "available_bytes": max(0, quota_bytes - total_used),
        "usage_percent": round((total_used / quota_bytes) * 100, 2) if quota_bytes > 0 else 0,
        "file_count": file_count,
        "by_type": {k: {"bytes": v, "human": _format_size(v)} for k, v in by_type.items()},
    }


def _format_size(bytes_size: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} PB"


@app.get("/artifacts/by-path/{storage_path:path}", response_model=Dict[str, Any])
async def get_artifact_by_path(storage_path: str):
    """Get an artifact by storage path"""
    storage = get_storage_from_app()
    result = await storage.artifact_get_by_path(storage_path)
    if not result:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return result


@app.get("/artifacts", response_model=List[Dict[str, Any]])
async def list_artifacts(
    folder_id: Optional[str] = Query(None),
    media_type: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """List artifacts with filters"""
    storage = get_storage_from_app()
    return await storage.artifact_list(
        folder_id=folder_id,
        media_type=media_type,
        user_id=user_id,
        session_id=session_id,
        limit=limit,
        offset=offset
    )


# ==================== File access (owner/group/override) ====================


def _split_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    if not str(value).strip():
        return None
    return [s.strip() for s in str(value).split(",") if s.strip()]


@app.get("/users/{user_id}/artifacts/accessible", response_model=Dict[str, Any])
async def list_accessible_artifacts(
    user_id: str,
    folder_id: Optional[str] = Query(None),
    media_type: Optional[str] = Query(None, description="image|video|audio|text|other"),
    session_id: Optional[str] = Query(None),
    search: Optional[str] = Query(None, description="Substring match on filename/storage_path"),
    ownership: str = Query("any", description="any|owner|shared"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sort_by: Optional[str] = Query(None, description="created_at|updated_at|name|size|mime_type|media_type"),
    sort_dir: Optional[str] = Query(None, description="asc|desc"),
    created_from: Optional[str] = Query(None, description="ISO timestamp lower bound"),
    created_to: Optional[str] = Query(None, description="ISO timestamp upper bound"),
    size_min: Optional[int] = Query(None, ge=0, description="Minimum size in bytes"),
    size_max: Optional[int] = Query(None, ge=0, description="Maximum size in bytes"),
    mime_type: Optional[str] = Query(None, description="Single MIME type equality match"),
    mime_types: Optional[str] = Query(None, description="CSV list of MIME types (any-of)"),
    tags_any: Optional[str] = Query(None, description="CSV list of tags (any of)"),
    tags_all: Optional[str] = Query(None, description="CSV list of tags (all of)"),
    folder_prefix: Optional[str] = Query(None, description="Single S3-like virtual prefix"),
    folder_prefixes: Optional[str] = Query(None, description="CSV list of S3-like virtual prefixes (OR)"),
    uploaded_by: Optional[str] = Query(None, description="CSV list of uploader user_ids"),
):
    storage = get_storage_from_app()
    return await storage.artifact_accessible_list(
        user_id=user_id,
        folder_id=folder_id,
        media_type=media_type,
        session_id=session_id,
        search=search,
        ownership=ownership,
        limit=limit,
        offset=offset,
        sort_by=sort_by,
        sort_dir=sort_dir,
        created_from=created_from,
        created_to=created_to,
        size_min=size_min,
        size_max=size_max,
        mime_type=mime_type,
        mime_types=_split_csv(mime_types),
        tags_any=_split_csv(tags_any),
        tags_all=_split_csv(tags_all),
        folder_prefix=folder_prefix,
        folder_prefixes=_split_csv(folder_prefixes),
        uploaded_by=_split_csv(uploaded_by),
    )


@app.get("/users/{user_id}/artifacts/stats", response_model=Dict[str, Any])
async def accessible_artifacts_stats(
    user_id: str,
    folder_id: Optional[str] = Query(None),
    media_type: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    ownership: str = Query("any", description="any|owner|shared"),
    created_from: Optional[str] = Query(None),
    created_to: Optional[str] = Query(None),
    size_min: Optional[int] = Query(None, ge=0),
    size_max: Optional[int] = Query(None, ge=0),
    mime_type: Optional[str] = Query(None),
    mime_types: Optional[str] = Query(None),
    tags_any: Optional[str] = Query(None),
    tags_all: Optional[str] = Query(None),
    folder_prefix: Optional[str] = Query(None),
    folder_prefixes: Optional[str] = Query(None),
    uploaded_by: Optional[str] = Query(None),
):
    storage = get_storage_from_app()
    return await storage.artifact_accessible_stats(
        user_id=user_id,
        folder_id=folder_id,
        media_type=media_type,
        session_id=session_id,
        search=search,
        ownership=ownership,
        created_from=created_from,
        created_to=created_to,
        size_min=size_min,
        size_max=size_max,
        mime_type=mime_type,
        mime_types=_split_csv(mime_types),
        tags_any=_split_csv(tags_any),
        tags_all=_split_csv(tags_all),
        folder_prefix=folder_prefix,
        folder_prefixes=_split_csv(folder_prefixes),
        uploaded_by=_split_csv(uploaded_by),
    )


@app.get("/users/{user_id}/artifacts/{artifact_id}/access-check", response_model=Dict[str, Any])
async def check_artifact_access(
    user_id: str,
    artifact_id: str,
    action: str = Query("read", description="read|write|delete"),
):
    storage = get_storage_from_app()
    return await storage.artifact_is_accessible(
        user_id=user_id, artifact_id=artifact_id, action=action
    )


@app.get("/users/{user_id}/cases/{case_id}/access-check", response_model=Dict[str, Any])
async def check_case_access(user_id: str, case_id: str, request: Request):
    _storage_auth_context(request)
    storage = get_storage_from_app()
    return await storage.case_is_accessible(user_id=user_id, case_id=case_id)


@app.get("/users/{user_id}/folders", response_model=Dict[str, Any])
async def list_accessible_folders(
    user_id: str,
    prefix: str = Query("", description="S3-like prefix (virtual path)"),
    delimiter: str = Query("/", description="Delimiter (usually '/')"),
    limit: int = Query(200, ge=1, le=500),
    offset: int = Query(0, ge=0),
    ownership: str = Query("any", description="any|owner|shared"),
    folder_id: Optional[str] = Query(None),
    media_type: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    created_from: Optional[str] = Query(None),
    created_to: Optional[str] = Query(None),
    size_min: Optional[int] = Query(None, ge=0),
    size_max: Optional[int] = Query(None, ge=0),
    mime_type: Optional[str] = Query(None),
    mime_types: Optional[str] = Query(None),
    tags_any: Optional[str] = Query(None),
    tags_all: Optional[str] = Query(None),
    uploaded_by: Optional[str] = Query(None),
):
    storage = get_storage_from_app()
    return await storage.accessible_folders(
        user_id=user_id,
        prefix=prefix,
        delimiter=delimiter,
        limit=limit,
        offset=offset,
        ownership=ownership,
        folder_id=folder_id,
        media_type=media_type,
        session_id=session_id,
        search=search,
        created_from=created_from,
        created_to=created_to,
        size_min=size_min,
        size_max=size_max,
        mime_type=mime_type,
        mime_types=_split_csv(mime_types),
        tags_any=_split_csv(tags_any),
        tags_all=_split_csv(tags_all),
        uploaded_by=_split_csv(uploaded_by),
    )


@app.get("/users/{user_id}/artifacts/facets", response_model=Dict[str, Any])
async def accessible_facets(
    user_id: str,
    search: Optional[str] = Query(None),
    ownership: str = Query("any", description="any|owner|shared"),
    top_mimes: int = Query(30, ge=1, le=200),
    top_tags: int = Query(30, ge=1, le=200),
    top_uploaders: int = Query(10, ge=1, le=100),
    top_sessions: int = Query(10, ge=1, le=100),
    date_buckets: str = Query("month", description="day|week|month"),
    folder_id: Optional[str] = Query(None),
    media_type: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    created_from: Optional[str] = Query(None),
    created_to: Optional[str] = Query(None),
    size_min: Optional[int] = Query(None, ge=0),
    size_max: Optional[int] = Query(None, ge=0),
    mime_type: Optional[str] = Query(None),
    mime_types: Optional[str] = Query(None),
    tags_any: Optional[str] = Query(None),
    tags_all: Optional[str] = Query(None),
    folder_prefix: Optional[str] = Query(None),
    folder_prefixes: Optional[str] = Query(None),
    uploaded_by: Optional[str] = Query(None),
):
    storage = get_storage_from_app()
    return await storage.accessible_facets(
        user_id=user_id,
        search=search,
        ownership=ownership,
        top_mimes=top_mimes,
        top_tags=top_tags,
        top_uploaders=top_uploaders,
        top_sessions=top_sessions,
        date_buckets=date_buckets,
        folder_id=folder_id,
        media_type=media_type,
        session_id=session_id,
        created_from=created_from,
        created_to=created_to,
        size_min=size_min,
        size_max=size_max,
        mime_type=mime_type,
        mime_types=_split_csv(mime_types),
        tags_any=_split_csv(tags_any),
        tags_all=_split_csv(tags_all),
        folder_prefix=folder_prefix,
        folder_prefixes=_split_csv(folder_prefixes),
        uploaded_by=_split_csv(uploaded_by),
    )


@app.post("/shares", response_model=Dict[str, Any])
async def create_artifact_share(payload: Dict[str, Any], request: Request):
    storage = get_storage_from_app()
    artifact_id = str(payload.get("artifact_id") or "").strip()
    created_by = str(payload.get("created_by") or "").strip()
    expires_sec = int(payload.get("expires_sec") or 3600)
    scope = payload.get("scope") if isinstance(payload.get("scope"), dict) else {}
    if not artifact_id or not created_by:
        raise HTTPException(status_code=400, detail="artifact_id and created_by are required")

    # The caller may only share artifacts they can access. ``created_by`` must
    # match the authenticated user, and that user must have access to the file.
    requester = (request.headers.get("X-User-Id") or "").strip()
    if settings.STORAGE_RBAC_ENABLED and requester:
        if created_by and created_by != requester:
            raise HTTPException(status_code=403, detail="cannot_share_as_another_user")
        await _require_artifact_access(storage, artifact_id, request, action="read")

    share = await storage.share_create(
        artifact_id=artifact_id,
        created_by=created_by,
        expires_sec=expires_sec,
        scope=scope,
    )
    # NOTE: this service is internal; api-gateway should rewrite to /storage/shares/...
    share["resolve_path"] = f"/shares/{share['token']}/resolve"
    share["download_path"] = f"/shares/{share['token']}/download"
    return {"ok": True, "share": share}


@app.post("/shares/{token}/revoke", response_model=Dict[str, Any])
async def revoke_artifact_share(token: str, request: Request):
    storage = get_storage_from_app()
    ok = await storage.share_revoke(token=token)
    await storage.share_audit(
        token_id=None,
        action="revoke",
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        metadata={"ok": ok},
    )
    return {"ok": True, "revoked": bool(ok)}


@app.get("/shares/{token}/resolve", response_model=Dict[str, Any])
async def resolve_artifact_share(token: str, request: Request):
    storage = get_storage_from_app()
    row = await storage.share_get_by_token(token=token)
    if not row:
        await storage.share_audit(
            token_id=None,
            action="deny",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            metadata={"reason": "not_found"},
        )
        raise HTTPException(status_code=404, detail="Share token not found")

    revoked_at = row.get("revoked_at")
    expires_at = row.get("expires_at")
    # We keep these as ISO strings from backend.
    if revoked_at:
        await storage.share_audit(
            token_id=row.get("id"),
            action="deny",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            metadata={"reason": "revoked"},
        )
        raise HTTPException(status_code=410, detail="Share token revoked")

    # naive expiration check (string compare is safe in ISO8601 only if same format; keep simple by trusting DB at download time)
    await storage.share_audit(
        token_id=row.get("id"),
        action="resolve",
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        metadata={},
    )
    return {
        "ok": True,
        "artifact_id": row.get("artifact_id"),
        "expires_at": expires_at,
        "download_path": f"/shares/{token}/download",
    }


@app.get("/shares/{token}/download")
async def download_shared_artifact(token: str, request: Request):
    """
    Public download endpoint for a share token.
    This proxies bytes (no redirect to MinIO), so clients never see MinIO URLs.
    """
    from fastapi.responses import StreamingResponse
    import io

    storage = get_storage_from_app()
    row = await storage.share_get_by_token(token=token)
    if not row:
        await storage.share_audit(
            token_id=None,
            action="deny",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            metadata={"reason": "not_found"},
        )
        raise HTTPException(status_code=404, detail="Share token not found")

    if row.get("revoked_at"):
        await storage.share_audit(
            token_id=row.get("id"),
            action="expired",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            metadata={"reason": "revoked"},
        )
        raise HTTPException(status_code=410, detail="Share token revoked")

    artifact_id = row.get("artifact_id")
    artifact = await storage.artifact_get(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    storage_path = artifact.get("storage_path") or ""
    mime_type = artifact.get("mime_type") or "application/octet-stream"
    filename = artifact.get("original_filename") or os.path.basename(storage_path) or "file"

    try:
        if storage_path.startswith("minio://"):
            minio_path = storage_path[8:]
            object_name = minio_path.split("/", 1)[1] if "/" in minio_path else minio_path
            content = await storage.file_download(object_name)
            if not content:
                raise HTTPException(status_code=404, detail="File not found in MinIO")
            await storage.share_audit(
                token_id=row.get("id"),
                action="download",
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                metadata={"bytes": len(content)},
            )
            return StreamingResponse(
                io.BytesIO(content),
                media_type=mime_type,
                headers={
                    "Content-Disposition": _content_disposition("attachment", filename),
                    "Content-Length": str(len(content)),
                    "Access-Control-Expose-Headers": "Content-Disposition, Content-Length",
                    "Cache-Control": "private, max-age=60",
                },
            )

        # local filesystem fallback
        path = storage_path
        if path and not path.startswith("/"):
            path = f"/data/uploads/{path}"
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="File not found on disk")
        with open(path, "rb") as f:
            content = f.read()
        await storage.share_audit(
            token_id=row.get("id"),
            action="download",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            metadata={"bytes": len(content)},
        )
        return StreamingResponse(
            io.BytesIO(content),
            media_type=mime_type,
            headers={
                "Content-Disposition": _content_disposition("attachment", filename),
                "Content-Length": str(len(content)),
                "Access-Control-Expose-Headers": "Content-Disposition, Content-Length",
                "Cache-Control": "private, max-age=60",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("share download failed: %s", e)
        raise HTTPException(status_code=500, detail="Share download failed")


# 🆕 دریافت فایل‌های یک کاربر خاص
@app.get("/users/{user_id}/artifacts", response_model=List[Dict[str, Any]])
async def get_user_artifacts(
    user_id: str,
    media_type: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """
    دریافت لیست فایل‌های آپلود شده توسط یک کاربر
    
    این endpoint به کاربر امکان می‌دهد همه فایل‌هایی که آپلود کرده را ببیند.
    """
    storage = get_storage_from_app()
    return await storage.artifact_list(
        user_id=user_id,
        media_type=media_type,
        limit=limit,
        offset=offset
    )


# 🆕 دریافت فایل‌های یک جلسه چت خاص
@app.get("/chat/sessions/{session_id}/artifacts", response_model=List[Dict[str, Any]])
async def get_session_artifacts(
    request: Request,
    session_id: str,
    media_type: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """
    دریافت لیست فایل‌های آپلود شده در یک جلسه چت
    
    این endpoint به کاربر امکان می‌دهد همه فایل‌های یک چت را ببیند.
    """
    storage = get_storage_from_app()
    ctx = _storage_auth_context(request)
    user_id = ctx.get("user_id")
    if user_id:
        try:
            result = await storage.artifact_accessible_list(
                user_id=str(user_id),
                session_id=session_id,
                media_type=media_type,
                limit=limit,
                offset=offset,
            )
            return result.get("items", []) if isinstance(result, dict) else []
        except Exception as exc:
            logger.warning("accessible session artifacts failed; owner fallback: %s", exc)
            rows = await storage.artifact_list(
                session_id=session_id,
                media_type=media_type,
                limit=limit,
                offset=offset,
            )
            return [row for row in rows if str(row.get("user_id") or "") == str(user_id)]

    if ctx.get("service_name") not in {"case-importer"}:
        raise HTTPException(status_code=403, detail="user_context_required_for_session_artifacts")

    return await storage.artifact_list(
        session_id=session_id,
        media_type=media_type,
        limit=limit,
        offset=offset
    )


@app.post("/chat/sessions/{session_id}/artifacts/attach", response_model=Dict[str, Any])
async def attach_artifact_to_session(
    session_id: str,
    data: SessionArtifactAttach,
    request: Request,
):
    """
    Link an existing storage artifact into a chat session workspace.

    Used by the orchestrator ``session_attach`` tool so the LLM can pull a
    database file into the current session without re-uploading it.
    """
    storage = get_storage_from_app()

    user_id: Optional[str] = None
    ctx: Dict[str, Any] = {}
    try:
        ctx = _storage_auth_context(request)
        user_id = ctx.get("user_id")
    except HTTPException:
        if settings.STORAGE_RBAC_ENABLED:
            raise

    user_id = user_id or data.user_id
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id_required")

    artifact_id = (data.artifact_id or "").strip()
    path_hint = (data.path or data.storage_path or "").strip()
    filename = (data.filename or data.name or "").strip()
    case_id = (data.case_id or "").strip()
    search_limit = int(data.limit or 25)

    if not any([artifact_id, path_hint, filename, case_id]):
        raise HTTPException(
            status_code=400,
            detail="artifact_locator_required",
        )

    prep = await storage.artifact_prepare_attach(
        user_id=str(user_id),
        artifact_id=artifact_id or None,
        case_id=case_id or None,
        path=path_hint or None,
        filename=filename or None,
        register_if_local_file=True,
        search_limit=max(1, min(search_limit, 50)),
    )

    status = prep.get("status")
    candidates = prep.get("candidates") or []

    if status == "not_found":
        raise HTTPException(
            status_code=404,
            detail="artifact_not_found",
        )

    if status == "ambiguous":
        return {
            "ok": True,
            "status": "ambiguous",
            "session_id": session_id,
            "count": len(candidates),
            "candidates": candidates,
            "message": (
                f"Multiple files ({len(candidates)}) match your criteria. "
                "Call session_attach again with a single artifact_id from the list."
            ),
        }

    resolved = prep.get("artifact")
    if not isinstance(resolved, dict):
        raise HTTPException(status_code=404, detail="artifact_not_found")

    artifact_id = str(resolved.get("id") or "")
    if settings.STORAGE_RBAC_ENABLED:
        await _require_artifact_access(storage, artifact_id, request, artifact=resolved)

    result = await storage.artifact_attach_to_session(
        artifact_id=artifact_id,
        session_id=session_id,
        user_id=str(user_id),
    )
    if not result:
        raise HTTPException(status_code=404, detail="artifact_not_found_or_not_accessible")

    out_name = (
        result.get("original_filename")
        or result.get("filename")
        or result.get("name")
        or "unnamed"
    )
    return {
        "ok": True,
        "status": "attached",
        "artifact_id": str(result.get("id") or artifact_id),
        "session_id": session_id,
        "file": {
            "id": str(result.get("id") or artifact_id),
            "artifact_id": str(result.get("id") or artifact_id),
            "name": out_name,
            "filename": out_name,
            "path": result.get("storage_path"),
            "storage_path": result.get("storage_path"),
            "mime_type": result.get("mime_type"),
            "type": result.get("media_type"),
            "media_type": result.get("media_type"),
            "size_bytes": result.get("file_size_bytes"),
            "case_id": storage._artifact_case_id(result.get("metadata")),
        },
    }


@app.patch("/artifacts/{artifact_id}/metadata", response_model=Dict[str, Any])
async def update_artifact_metadata(artifact_id: str, data: ArtifactMetadataUpdate):
    """Update artifact metadata (merge)"""
    storage = get_storage_from_app()
    result = await storage.artifact_update_metadata(artifact_id, data.metadata)
    if not result:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return result


class ArtifactStoragePathUpdate(BaseModel):
    storage_path: str
    original_filename: Optional[str] = None
    media_type: Optional[str] = None
    mime_type: Optional[str] = None


@app.patch("/artifacts/{artifact_id}/storage-path", response_model=Dict[str, Any])
async def update_artifact_storage_path(artifact_id: str, data: ArtifactStoragePathUpdate):
    """
    Update the canonical storage_path of an artifact.
    Used by case-importer after promoting files from cases_temp to MinIO/local cases,
    and after file_identify adds a detected extension.
    """
    storage = get_storage_from_app()
    if data.original_filename is not None or data.media_type is not None or data.mime_type is not None:
        ok = await storage.artifact_update_storage_identity(
            artifact_id,
            storage_path=data.storage_path,
            original_filename=data.original_filename,
            media_type=data.media_type,
            mime_type=data.mime_type,
        )
    else:
        ok = await storage.artifact_update_storage_path(artifact_id, data.storage_path)
    if not ok:
        raise HTTPException(status_code=404, detail="Artifact not found")
    out: Dict[str, Any] = {"id": artifact_id, "storage_path": data.storage_path}
    if data.original_filename is not None:
        out["original_filename"] = data.original_filename
    if data.media_type is not None:
        out["media_type"] = data.media_type
    if data.mime_type is not None:
        out["mime_type"] = data.mime_type
    return out


class MinioRenameBody(BaseModel):
    src_object: str
    dest_object: str


@app.post("/internal/minio/rename", response_model=Dict[str, Any])
async def rename_minio_object(body: MinioRenameBody):
    """Server-side copy+delete for case_importer objects (extension detection)."""
    storage = get_storage_from_app()
    if not storage.minio or not storage.minio.is_connected:
        raise HTTPException(status_code=503, detail="MinIO not available")
    src = str(body.src_object or "").lstrip("/")
    dest = str(body.dest_object or "").lstrip("/")
    if not src or not dest:
        raise HTTPException(status_code=400, detail="missing_fields")
    if ".." in src.split("/") or ".." in dest.split("/"):
        raise HTTPException(status_code=400, detail="invalid_path")
    if not src.startswith("case_importer/") or not dest.startswith("case_importer/"):
        raise HTTPException(status_code=403, detail="prefix_forbidden")
    if src == dest:
        return {"ok": True, "src_object": src, "dest_object": dest, "noop": True}
    if await storage.file_exists(dest):
        # Prefer keeping the extensioned name unique; overwrite empty/stale targets.
        await storage.file_delete(dest)
    ok = await storage.file_copy(src, dest)
    if not ok:
        raise HTTPException(status_code=500, detail="copy_failed")
    await storage.file_delete(src)
    bucket = getattr(storage.minio, "default_bucket", None) or "uploads"
    return {
        "ok": True,
        "src_object": src,
        "dest_object": dest,
        "storage_path": f"minio://{bucket}/{dest}",
    }


class ArtifactTextExtractRequest(BaseModel):
    file_path: Optional[str] = None


@app.post("/artifacts/{artifact_id}/extract-text", response_model=Dict[str, Any])
async def backfill_artifact_text_extract(
    artifact_id: str,
    data: ArtifactTextExtractRequest = ArtifactTextExtractRequest(),
):
    """Store ``metadata.text_extract`` from an on-disk file (import cleanup / repair)."""
    storage = get_storage_from_app()
    artifact = await storage.artifact_get(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    file_path = (data.file_path or artifact.get("storage_path") or "").strip()
    resolved = file_path
    if not resolved or resolved.startswith("minio://") or not os.path.isfile(resolved):
        resolved = _resolve_existing_local_artifact_path(artifact) or ""
    if not resolved or not os.path.isfile(resolved):
        raise HTTPException(status_code=404, detail="File not found on disk")

    summary = await _auto_extract_text_from_local_file(
        storage,
        artifact_id=artifact_id,
        file_path=resolved,
        filename=artifact.get("original_filename"),
        mime_type=artifact.get("mime_type"),
    )
    if summary is None:
        raise HTTPException(status_code=400, detail="Text extraction not applicable or failed")
    return {"ok": True, "artifact_id": artifact_id, "text_extract": summary}


# ==================== Tool Run Routes (DEPRECATED) ====================
# Note: tool_runs table has been deprecated. Tool executions are now stored
# in agent_tool_executions table via the /traces/* API endpoints.
# These endpoints are kept temporarily for backward compatibility.

# Note: /tool-runs endpoints have been REMOVED
# Tool executions are stored via POST /traces/{trace_id}/tools API


# ==================== Observation Routes ====================

@app.post("/observations")
async def save_observation(data: ObservationCreate):
    """Save an observation"""
    storage = get_storage_from_app()
    result = await storage.save_observation(
        session_id=data.session_id,
        observation_type=data.observation_type,
        content=data.content,
        artifact_id=data.artifact_id,
        metadata=data.metadata
    )
    return {"id": result, "status": "saved"}


@app.get("/observations", response_model=List[Dict[str, Any]])
async def list_observations(
    session_id: str = Query(...),
    observation_type: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """List observations for a session"""
    storage = get_storage_from_app()
    return await storage.list_observations(
        session_id=session_id,
        observation_type=observation_type,
        limit=limit
    )


# ==================== Insight Routes ====================

@app.post("/insights")
async def save_insight(data: InsightCreate):
    """Save an insight"""
    storage = get_storage_from_app()
    result = await storage.save_insight(
        session_id=data.session_id,
        insight_type=data.insight_type,
        title=data.title,
        content=data.content,
        confidence=data.confidence,
        sources=data.sources,
        metadata=data.metadata
    )
    return {"id": result, "status": "saved"}


@app.get("/insights", response_model=List[Dict[str, Any]])
async def list_insights(
    session_id: str = Query(...),
    insight_type: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """List insights for a session"""
    storage = get_storage_from_app()
    return await storage.list_insights(
        session_id=session_id,
        insight_type=insight_type,
        limit=limit
    )


# ==================== Vector Routes ====================

@app.post("/vectors/upsert")
async def upsert_vectors(data: VectorUpsert):
    """Upsert vectors to Qdrant"""
    storage = get_storage_from_app()
    success = await storage.vector_upsert(
        collection_name=data.collection_name,
        vectors=data.vectors,
        ids=data.ids,
        payloads=data.payloads,
        dimension=data.dimension
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to upsert vectors")
    return {"status": "success", "count": len(data.ids)}


@app.get("/vectors/count")
async def count_vectors_by_case(case_id: str = Query(..., min_length=1)):
    """Exact Qdrant point count for a case (all collections, payload.case_id)."""
    storage = get_storage_from_app()
    return await storage.vector_count_by_case(case_id)


@app.post("/vectors/search", response_model=List[Dict[str, Any]])
async def search_vectors(data: VectorSearch, request: Request):
    """Search vectors in Qdrant"""
    storage = get_storage_from_app()
    ctx = _storage_auth_context(request)
    raw_results = await storage.vector_search(
        collection_name=data.collection_name,
        query_vector=data.query_vector,
        limit=max(data.limit, min(data.limit * 5, 100)),
        filter_dict=data.filter_dict,
        score_threshold=data.score_threshold
    )
    filtered: List[Dict[str, Any]] = []
    for result in raw_results:
        payload = result.get("payload") or {}
        artifact_id = payload.get("artifact_id")
        if not artifact_id:
            if ctx.get("service_name") in {"case-importer"} and not ctx.get("user_id"):
                filtered.append(result)
            continue
        artifact = await storage.artifact_get(str(artifact_id))
        if artifact and _artifact_allowed_by_context(ctx, artifact):
            filtered.append(result)
            if len(filtered) >= data.limit:
                break
        elif artifact and ctx.get("user_id"):
            try:
                access = await storage.artifact_is_accessible(
                    user_id=str(ctx["user_id"]),
                    artifact_id=str(artifact_id),
                )
                if (access or {}).get("accessible"):
                    filtered.append(result)
                    if len(filtered) >= data.limit:
                        break
            except Exception as exc:
                logger.warning("vector artifact access check failed for %s: %s", artifact_id, exc)
    return filtered


# ==================== Graph (Neo4j) Routes ====================

@app.post("/graph/query")
async def graph_query(data: CypherQuery, request: Request):
    """Execute a read Cypher query on Neo4j"""
    storage = get_storage_from_app()
    if not storage.neo4j:
        raise HTTPException(status_code=503, detail="Neo4j not configured")
    ctx = _storage_auth_context(request)
    results = await storage.graph_query(data.query, data.parameters)
    results = await _filter_graph_records_for_context(
        storage,
        ctx,
        results,
        query=data.query,
    )
    return {"records": results}


@app.post("/graph/write")
async def graph_write(data: CypherQuery, request: Request):
    """Execute a write Cypher query on Neo4j (authenticated internal callers only)."""
    _storage_auth_context(request)
    storage = get_storage_from_app()
    if not storage.neo4j:
        raise HTTPException(status_code=503, detail="Neo4j not configured")
    return await storage.graph_write(data.query, data.parameters)


@app.post("/graph/batch")
async def graph_write_batch(data: CypherBatch, request: Request):
    """Execute multiple write queries in a single atomic transaction."""
    _storage_auth_context(request)
    storage = get_storage_from_app()
    if not storage.neo4j:
        raise HTTPException(status_code=503, detail="Neo4j not configured")
    return await storage.graph_write_batch(data.queries)


# ==================== Feedback Routes ====================

@app.post("/feedback", response_model=Dict[str, Any])
async def save_feedback(data: FeedbackCreate):
    """Save user feedback"""
    storage = get_storage_from_app()
    result = await storage.save_feedback(
        session_id=data.session_id,
        rating=data.rating,
        comment=data.comment,
        message_text=data.message_text,
        metadata=data.metadata
    )
    return result or {"status": "saved"}


# ==================== Image Repository Routes ====================

@app.post("/images/metadata", response_model=Dict[str, Any])
async def upsert_image_meta(data: ImageMetaCreate):
    """Upsert image metadata"""
    storage = get_storage_from_app()
    if not storage.images:
        raise HTTPException(status_code=503, detail="Image repository not available")
    result = await storage.images.upsert_image_meta(
        artifact_id=data.artifact_id,
        width_px=data.width_px,
        height_px=data.height_px,
        dpi=data.dpi,
        color_mode=data.color_mode,
        camera_model=data.camera_model,
        lens=data.lens,
        iso=data.iso,
        exposure=data.exposure,
        focal_length=data.focal_length,
        metadata=data.metadata
    )
    if not result:
        raise HTTPException(status_code=500, detail="Failed to upsert image metadata")
    return result


@app.get("/images/{artifact_id}/metadata", response_model=Dict[str, Any])
async def get_image_meta(artifact_id: str):
    """Get image metadata"""
    storage = get_storage_from_app()
    if not storage.images:
        raise HTTPException(status_code=503, detail="Image repository not available")
    result = await storage.images.get_image_meta(artifact_id)
    if not result:
        raise HTTPException(status_code=404, detail="Image metadata not found")
    return result


@app.post("/faces", response_model=Dict[str, Any])
async def upsert_face(data: FaceCreate):
    """Upsert face detection result"""
    storage = get_storage_from_app()
    if not storage.images:
        raise HTTPException(status_code=503, detail="Image repository not available")
    result = await storage.images.upsert_face(
        artifact_id=data.artifact_id,
        face_key=data.face_key,
        bbox=data.bbox,
        confidence=data.confidence,
        person_id=data.person_id,
        tag=data.tag,
        thumbnail_base64=data.thumbnail_base64,
        vector_id=data.vector_id,
        metadata=data.metadata
    )
    if not result:
        raise HTTPException(status_code=500, detail="Failed to upsert face")
    return result


@app.get("/faces", response_model=List[Dict[str, Any]])
async def list_faces(
    artifact_id: Optional[str] = Query(None),
    person_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """List faces with filters"""
    storage = get_storage_from_app()
    if not storage.images:
        raise HTTPException(status_code=503, detail="Image repository not available")
    return await storage.images.list_faces(
        artifact_id=artifact_id,
        person_id=person_id,
        limit=limit
    )


@app.patch("/faces/{artifact_id}/{face_key}/link")
async def link_face_to_person(
    artifact_id: str,
    face_key: str,
    person_id: str = Body(..., embed=True)
):
    """Link a face to a person"""
    storage = get_storage_from_app()
    if not storage.images:
        raise HTTPException(status_code=503, detail="Image repository not available")
    success = await storage.images.link_face_to_person(
        artifact_id=artifact_id,
        face_key=face_key,
        person_id=person_id
    )
    if not success:
        raise HTTPException(status_code=404, detail="Face not found")
    return {"status": "linked"}


# ==================== Audio Repository Routes ====================

@app.post("/audio/metadata", response_model=Dict[str, Any])
async def upsert_audio_meta(data: AudioMetaCreate):
    """Upsert audio metadata"""
    storage = get_storage_from_app()
    if not storage.audio:
        raise HTTPException(status_code=503, detail="Audio repository not available")
    result = await storage.audio.upsert_audio_meta(
        artifact_id=data.artifact_id,
        duration_sec=data.duration_sec,
        sample_rate=data.sample_rate,
        channels=data.channels,
        codec=data.codec,
        bitrate=data.bitrate,
        transcription=data.transcription,
        language=data.language,
        metadata=data.metadata
    )
    if not result:
        raise HTTPException(status_code=500, detail="Failed to upsert audio metadata")
    return result


@app.get("/audio/{artifact_id}/metadata", response_model=Dict[str, Any])
async def get_audio_meta(artifact_id: str):
    """Get audio metadata"""
    storage = get_storage_from_app()
    if not storage.audio:
        raise HTTPException(status_code=503, detail="Audio repository not available")
    result = await storage.audio.get_audio_meta(artifact_id)
    if not result:
        raise HTTPException(status_code=404, detail="Audio metadata not found")
    return result


@app.get("/audio/{artifact_id}/transcript")
async def get_audio_transcript(artifact_id: str):
    """Get full transcript for audio"""
    storage = get_storage_from_app()
    if not storage.audio:
        raise HTTPException(status_code=503, detail="Audio repository not available")
    transcript = await storage.audio.get_full_transcript(artifact_id)
    return {"artifact_id": artifact_id, "transcript": transcript}


@app.get("/audio/{artifact_id}/segments", response_model=List[Dict[str, Any]])
async def list_audio_segments(
    artifact_id: str,
    speaker_id: Optional[str] = Query(None),
    limit: int = Query(1000, ge=1, le=10000)
):
    """List audio segments"""
    storage = get_storage_from_app()
    if not storage.audio:
        raise HTTPException(status_code=503, detail="Audio repository not available")
    return await storage.audio.list_segments(
        artifact_id=artifact_id,
        speaker_id=speaker_id,
        limit=limit
    )


@app.post("/audio/{artifact_id}/segments", response_model=Dict[str, Any])
async def upsert_audio_segments(artifact_id: str, data: AudioSegmentsBatch, request: Request):
    """Replace or append typed audio segments (tool-runner projection writer)."""
    storage = get_storage_from_app()
    if not storage.audio:
        raise HTTPException(status_code=503, detail="Audio repository not available")
    await _require_artifact_access(storage, artifact_id, request, action="write")
    if data.replace:
        await storage.audio.delete_segments(artifact_id)
    count = await storage.audio.batch_add_segments(artifact_id, data.segments or [])
    return {
        "artifact_id": artifact_id,
        "count": count,
        "replaced": bool(data.replace),
    }


# ==================== User Routes ====================

class UserCreate(BaseModel):
    username: str
    email: str
    password_hash: Optional[str] = None
    keycloak_id: Optional[str] = None
    display_name: Optional[str] = None
    role_name: str = "user"


class UserUpdate(BaseModel):
    display_name: Optional[str] = None
    email: Optional[str] = None
    avatar_url: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


@app.post("/users", response_model=Dict[str, Any])
async def create_user(data: UserCreate):
    """Create a new user"""
    storage = get_storage_from_app()
    if not storage.users:
        raise HTTPException(status_code=503, detail="User repository not available")
    result = await storage.users.create_user(
        username=data.username,
        email=data.email,
        password_hash=data.password_hash,
        keycloak_id=data.keycloak_id,
        display_name=data.display_name,
        role_name=data.role_name,
    )
    if not result:
        raise HTTPException(status_code=400, detail="User creation failed - username or email may exist")
    return result


@app.get("/users/{user_id}", response_model=Dict[str, Any])
async def get_user(user_id: str):
    """Get user by ID"""
    storage = get_storage_from_app()
    if not storage.users:
        raise HTTPException(status_code=503, detail="User repository not available")
    result = await storage.users.get_user_by_id(user_id)
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    # Remove password hash from response
    result.pop("password_hash", None)
    return result


@app.get("/users/by-username/{username}", response_model=Dict[str, Any])
async def get_user_by_username(username: str):
    """Get user by username"""
    storage = get_storage_from_app()
    if not storage.users:
        raise HTTPException(status_code=503, detail="User repository not available")
    result = await storage.users.get_user_by_username(username)
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    return result


@app.get("/users", response_model=List[Dict[str, Any]])
async def list_users(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
):
    """List users with optional filters"""
    storage = get_storage_from_app()
    if not storage.users:
        raise HTTPException(status_code=503, detail="User repository not available")
    return await storage.users.list_users(
        limit=limit,
        offset=offset,
        search=search,
        role_name=role,
        is_active=is_active,
    )


@app.patch("/users/{user_id}", response_model=Dict[str, Any])
async def update_user(user_id: str, data: UserUpdate):
    """Update user profile"""
    storage = get_storage_from_app()
    if not storage.users:
        raise HTTPException(status_code=503, detail="User repository not available")
    result = await storage.users.update_user(
        user_id=user_id,
        display_name=data.display_name,
        email=data.email,
        avatar_url=data.avatar_url,
        settings=data.settings,
        is_active=data.is_active,
    )
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    result.pop("password_hash", None)
    return result


@app.patch("/users/{user_id}/role", response_model=Dict[str, Any])
async def set_user_role(user_id: str, role_name: str = Body(..., embed=True)):
    """Set user role (admin only)"""
    storage = get_storage_from_app()
    if not storage.users:
        raise HTTPException(status_code=503, detail="User repository not available")
    success = await storage.users.set_user_role(user_id, role_name)
    if not success:
        raise HTTPException(status_code=404, detail="User or role not found")
    return {"status": "updated", "role": role_name}


@app.post("/users/{old_id}/remap/{new_id}")
async def remap_user_id(old_id: str, new_id: str):
    """Remap a user's primary key (e.g., to sync with Keycloak sub)"""
    storage = get_storage_from_app()
    if not storage.users:
        raise HTTPException(status_code=503, detail="User repository not available")
    success = await storage.users.remap_user_id(old_id, new_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found or new ID already exists")
    return {"status": "remapped", "old_id": old_id, "new_id": new_id}


@app.delete("/users/{user_id}")
async def delete_user(user_id: str):
    """Delete a user"""
    storage = get_storage_from_app()
    if not storage.users:
        raise HTTPException(status_code=503, detail="User repository not available")
    success = await storage.users.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"status": "deleted"}


# ==================== Role Routes ====================

@app.get("/roles", response_model=List[Dict[str, Any]])
async def list_roles():
    """List all roles"""
    storage = get_storage_from_app()
    if not storage.roles:
        raise HTTPException(status_code=503, detail="Role repository not available")
    return await storage.roles.list_roles()


@app.get("/roles/{role_name}", response_model=Dict[str, Any])
async def get_role(role_name: str):
    """Get role by name"""
    storage = get_storage_from_app()
    if not storage.roles:
        raise HTTPException(status_code=503, detail="Role repository not available")
    result = await storage.roles.get_role(role_name)
    if not result:
        raise HTTPException(status_code=404, detail="Role not found")
    return result


class RoleCreateBody(BaseModel):
    name: str
    description: Optional[str] = None
    permissions: Optional[List[str]] = None


@app.post("/roles", response_model=Dict[str, Any], status_code=201)
async def create_role(body: RoleCreateBody):
    """Create a custom role"""
    storage = get_storage_from_app()
    if not storage.roles:
        raise HTTPException(status_code=503, detail="Role repository not available")
    result = await storage.roles.create_role(
        name=body.name,
        description=body.description,
        permissions=body.permissions,
    )
    if not result:
        existing = await storage.roles.get_role(body.name)
        if existing:
            raise HTTPException(status_code=409, detail="Role already exists")
        raise HTTPException(status_code=500, detail="Failed to create role")
    return result


# ==================== Chat Session Routes ====================

class ChatSessionCreate(BaseModel):
    user_id: str
    title: str = "چت جدید"
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatSessionUpdate(BaseModel):
    title: Optional[str] = None
    is_archived: Optional[bool] = None
    is_pinned: Optional[bool] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatDockSessionBody(BaseModel):
    user_id: str
    surface: str
    anchor_key: str
    title: Optional[str] = None


@app.get("/chat/stats/summary", response_model=Dict[str, Any])
async def chat_stats_summary():
    """Aggregate chat counts for admin dashboard."""
    storage = get_storage_from_app()
    if not storage.chats:
        raise HTTPException(status_code=503, detail="Chat repository not available")
    return await storage.chats.global_counts()


@app.get("/chat/sessions/dock", response_model=Dict[str, Any])
async def get_chat_dock_session(
    user_id: str = Query(...),
    surface: str = Query(...),
    anchor_key: str = Query(...),
):
    """Lookup floating native-AI dock session for (user, surface, anchor).

    Declared before ``/chat/sessions/{session_id}`` so literal ``dock`` is never
    parsed as a UUID (that previously returned HTTP 500 and broke mini-chat
    after the user deleted the dock session from the hub).
    """
    storage = get_storage_from_app()
    if not storage.chats:
        raise HTTPException(status_code=503, detail="Chat repository not available")
    result = await storage.chats.find_dock_session(
        user_id=user_id,
        surface=surface,
        anchor_key=anchor_key,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Dock session not found")
    return result


@app.post("/chat/sessions/dock", response_model=Dict[str, Any])
async def create_chat_dock_session(data: ChatDockSessionBody):
    """Get or create floating native-AI dock session for a page anchor."""
    storage = get_storage_from_app()
    if not storage.chats:
        raise HTTPException(status_code=503, detail="Chat repository not available")
    result = await storage.chats.get_or_create_dock_session(
        user_id=data.user_id,
        surface=data.surface,
        anchor_key=data.anchor_key,
        title=data.title,
    )
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create dock session")
    return result


@app.post("/chat/sessions/{session_id}/promote-dock", response_model=Dict[str, Any])
async def promote_chat_dock_session(session_id: str):
    """Clear dock flag so the session appears as a normal hub chat."""
    storage = get_storage_from_app()
    if not storage.chats:
        raise HTTPException(status_code=503, detail="Chat repository not available")
    result = await storage.chats.promote_dock_session(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="Session not found")
    return result


@app.post("/chat/sessions", response_model=Dict[str, Any])
async def create_chat_session(data: ChatSessionCreate):
    """Create a new chat session"""
    storage = get_storage_from_app()
    if not storage.chats:
        raise HTTPException(status_code=503, detail="Chat repository not available")
    result = await storage.chats.create_session(
        user_id=data.user_id,
        title=data.title,
        model=data.model,
        system_prompt=data.system_prompt,
        metadata=data.metadata,
    )
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create session")
    return result


@app.get("/chat/sessions/{session_id}", response_model=Dict[str, Any])
async def get_chat_session(session_id: str, include_messages: bool = Query(False)):
    """Get a chat session"""
    storage = get_storage_from_app()
    if not storage.chats:
        raise HTTPException(status_code=503, detail="Chat repository not available")
    
    if include_messages:
        result = await storage.chats.get_session_with_messages(session_id)
    else:
        result = await storage.chats.get_session(session_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Session not found")
    return result


@app.get("/chat/sessions/user/{user_id}", response_model=List[Dict[str, Any]])
async def list_user_sessions(
    user_id: str,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    include_archived: bool = Query(False),
):
    """List chat sessions for a user"""
    storage = get_storage_from_app()
    if not storage.chats:
        raise HTTPException(status_code=503, detail="Chat repository not available")
    return await storage.chats.list_sessions(
        user_id=user_id,
        limit=limit,
        offset=offset,
        include_archived=include_archived,
    )


@app.patch("/chat/sessions/{session_id}", response_model=Dict[str, Any])
async def update_chat_session(session_id: str, data: ChatSessionUpdate):
    """Update a chat session"""
    storage = get_storage_from_app()
    if not storage.chats:
        raise HTTPException(status_code=503, detail="Chat repository not available")
    result = await storage.chats.update_session(
        session_id=session_id,
        title=data.title,
        is_archived=data.is_archived,
        is_pinned=data.is_pinned,
        model=data.model,
        system_prompt=data.system_prompt,
        metadata=data.metadata,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Session not found")
    return result


@app.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session"""
    storage = get_storage_from_app()
    if not storage.chats:
        raise HTTPException(status_code=503, detail="Chat repository not available")
    success = await storage.chats.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


# ==================== Tool / Plugin Sessions (Postgres, no MinIO) ====================

class ToolSessionUpsert(BaseModel):
    user_id: Optional[str] = None
    plugin_id: str
    tool_id: Optional[str] = None
    record_id: str
    run_id: str
    status: Optional[str] = None
    ok: Optional[bool] = None
    error: Optional[str] = None
    output_preview: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


@app.post("/tool-sessions", response_model=Dict[str, Any])
async def upsert_tool_session(data: ToolSessionUpsert, request: Request):
    """Upsert a plugin/tool run session (tool-runner write path). Never case-bound."""
    storage = get_storage_from_app()
    if not storage.tool_sessions:
        raise HTTPException(status_code=503, detail="Tool session repository not available")
    ctx = _storage_auth_context(request)
    user_id = (data.user_id or ctx.get("user_id") or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id_required")
    result = await storage.tool_sessions.upsert(
        user_id=user_id,
        plugin_id=data.plugin_id,
        tool_id=data.tool_id,
        record_id=data.record_id,
        run_id=data.run_id,
        status=data.status,
        ok=data.ok,
        error=data.error,
        output_preview=data.output_preview,
        payload=data.payload,
    )
    if not result:
        raise HTTPException(status_code=500, detail="Failed to upsert tool session")
    return result


@app.get("/tool-sessions", response_model=Dict[str, Any])
async def list_tool_sessions(
    request: Request,
    plugin_id: str = Query(...),
    record_id: str = Query(...),
    user_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    include_payload: bool = Query(False),
):
    """List tool sessions for a plugin record (e.g. agent_flow flow_id)."""
    storage = get_storage_from_app()
    if not storage.tool_sessions:
        raise HTTPException(status_code=503, detail="Tool session repository not available")
    ctx = _storage_auth_context(request)
    uid = (user_id or ctx.get("user_id") or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="user_id_required")
    if settings.STORAGE_RBAC_ENABLED:
        requester = (ctx.get("user_id") or "").strip()
        if requester and requester != uid:
            raise HTTPException(status_code=403, detail="not_authorized")
    items = await storage.tool_sessions.list_sessions(
        user_id=uid,
        plugin_id=plugin_id,
        record_id=record_id,
        limit=limit,
        include_payload=include_payload,
    )
    return {"count": len(items), "items": items}


@app.get("/tool-sessions/by-run", response_model=Dict[str, Any])
async def get_tool_session_by_run(
    request: Request,
    plugin_id: str = Query(...),
    record_id: str = Query(...),
    run_id: str = Query(...),
    user_id: Optional[str] = Query(None),
    include_payload: bool = Query(True),
):
    storage = get_storage_from_app()
    if not storage.tool_sessions:
        raise HTTPException(status_code=503, detail="Tool session repository not available")
    ctx = _storage_auth_context(request)
    uid = (user_id or ctx.get("user_id") or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="user_id_required")
    row = await storage.tool_sessions.get(
        user_id=uid,
        plugin_id=plugin_id,
        record_id=record_id,
        run_id=run_id,
        include_payload=include_payload,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return row


@app.get("/tool-sessions/{session_id}", response_model=Dict[str, Any])
async def get_tool_session(
    session_id: str,
    request: Request,
    include_payload: bool = Query(True),
):
    storage = get_storage_from_app()
    if not storage.tool_sessions:
        raise HTTPException(status_code=503, detail="Tool session repository not available")
    ctx = _storage_auth_context(request)
    row = await storage.tool_sessions.get(
        session_id=session_id,
        user_id=ctx.get("user_id"),
        include_payload=include_payload,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return row


@app.delete("/tool-sessions/{session_id}", response_model=Dict[str, Any])
async def delete_tool_session(session_id: str, request: Request):
    storage = get_storage_from_app()
    if not storage.tool_sessions:
        raise HTTPException(status_code=503, detail="Tool session repository not available")
    ctx = _storage_auth_context(request)
    uid = (ctx.get("user_id") or "").strip() or None
    ok = await storage.tool_sessions.delete(session_id=session_id, user_id=uid)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


# ==================== Share/Export Conversation (UX-006) ====================

from fastapi.responses import StreamingResponse
import io
import json as json_module


def _require_session_owner(session: Dict[str, Any], request: Request) -> None:
    """Ensure the calling user (X-User-Id) owns the chat session.

    Storage is only reachable by trusted services (the global gate), but those
    services forward the end-user identity; we must still stop user A from
    exporting/sharing user B's session by id.
    """
    if not settings.STORAGE_RBAC_ENABLED:
        return
    requester = (request.headers.get("X-User-Id") or "").strip()
    owner = str(session.get("user_id") or "")
    if not requester or (owner and requester != owner):
        raise HTTPException(status_code=403, detail="not_authorized_for_session")


@app.get("/chat/sessions/{session_id}/export")
async def export_chat_session(
    session_id: str,
    request: Request,
    format: str = Query("json", description="Export format: json, txt, md, html"),
):
    """
    Export/share a chat conversation (UX-006).
    
    Formats:
    - json: Full JSON export with all metadata
    - txt: Plain text format
    - md: Markdown format
    - html: HTML format for sharing
    """
    storage = get_storage_from_app()
    if not storage.chats:
        raise HTTPException(status_code=503, detail="Chat repository not available")
    
    # Get session with messages
    session = await storage.chats.get_session_with_messages(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _require_session_owner(session, request)
    
    title = session.get("title", "Chat Conversation")
    messages = session.get("messages", [])
    created_at = session.get("created_at", "")
    
    if format == "json":
        content = json_module.dumps(session, ensure_ascii=False, indent=2, default=str)
        media_type = "application/json"
        filename = f"chat_{session_id}.json"
    
    elif format == "txt":
        lines = [f"=== {title} ===", f"Session: {session_id}", f"Created: {created_at}", ""]
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content_text = msg.get("content", "")
            lines.append(f"[{role}]")
            lines.append(content_text)
            lines.append("")
        content = "\n".join(lines)
        media_type = "text/plain"
        filename = f"chat_{session_id}.txt"
    
    elif format == "md":
        lines = [f"# {title}", "", f"**Session ID:** {session_id}", f"**Created:** {created_at}", "", "---", ""]
        for msg in messages:
            role = msg.get("role", "unknown")
            content_text = msg.get("content", "")
            if role == "user":
                lines.append(f"### 👤 User")
            elif role == "assistant":
                lines.append(f"### 🤖 Assistant")
            else:
                lines.append(f"### {role.title()}")
            lines.append("")
            lines.append(content_text)
            lines.append("")
            lines.append("---")
            lines.append("")
        content = "\n".join(lines)
        media_type = "text/markdown"
        filename = f"chat_{session_id}.md"
    
    elif format == "html":
        html_messages = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content_text = msg.get("content", "").replace("\n", "<br>")
            role_class = "user" if role == "user" else "assistant" if role == "assistant" else "system"
            role_label = "👤 User" if role == "user" else "🤖 Assistant" if role == "assistant" else role.title()
            html_messages.append(f'<div class="message {role_class}"><strong>{role_label}</strong><p>{content_text}</p></div>')
        
        content = f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{ font-family: Vazirmatn, Tahoma, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        .meta {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
        .message {{ margin: 15px 0; padding: 15px; border-radius: 10px; }}
        .user {{ background: #e3f2fd; border-right: 4px solid #2196f3; }}
        .assistant {{ background: #f3e5f5; border-right: 4px solid #9c27b0; }}
        .system {{ background: #fff3e0; border-right: 4px solid #ff9800; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="meta">Session: {session_id} | Created: {created_at}</div>
    {"".join(html_messages)}
</body>
</html>"""
        media_type = "text/html"
        filename = f"chat_{session_id}.html"
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}. Use json, txt, md, or html")
    
    # Return as downloadable file
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type=media_type,
        headers={
            "Content-Disposition": _content_disposition("attachment", filename),
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )


@app.post("/chat/sessions/{session_id}/share")
async def share_chat_session(
    session_id: str,
    request: Request,
    expires_hours: int = Query(24, ge=1, le=168, description="Link expiry in hours (max 7 days)"),
):
    """
    Generate a shareable link for a chat session (UX-006).
    
    Creates a temporary share token that allows viewing the conversation
    without authentication (read-only).
    """
    import hashlib
    import time
    
    storage = get_storage_from_app()
    if not storage.chats:
        raise HTTPException(status_code=503, detail="Chat repository not available")
    
    # Verify session exists
    session = await storage.chats.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _require_session_owner(session, request)
    
    # Generate share token
    timestamp = int(time.time())
    expires_at = timestamp + (expires_hours * 3600)
    token_data = f"{session_id}:{timestamp}:{expires_at}"
    share_token = hashlib.sha256(token_data.encode()).hexdigest()[:16]
    
    # Store share info (could be extended to use a proper shares table)
    # For now, return the token that encodes the session_id
    
    return {
        "session_id": session_id,
        "share_token": share_token,
        "expires_at": expires_at,
        "expires_hours": expires_hours,
        "share_url": f"/shared/chat/{share_token}",
    }


# ==================== Chat Message Routes ====================


class ToolCallItem(BaseModel):
    """یک فراخوانی ابزار توسط LLM"""
    id: Optional[str] = Field("", description="شناسه فراخوانی")
    name: str = Field(..., description="نام ابزار، مثلاً image_meta")
    step: Optional[int] = Field(None, description="شماره مرحله در گراف")
    arguments: Optional[Dict[str, Any]] = Field(None, description="آرگومان‌های ورودی ابزار")


class ToolResultChannels(BaseModel):
    """کانال‌های مختلف خروجی ابزار"""
    llm: Optional[str] = Field(None, description="متن خلاصه برای LLM")
    embed: Optional[Dict[str, Any]] = Field(None, description="داده برای embedding")
    rawdata: Optional[Dict[str, Any]] = Field(None, description="داده خام")


class ToolResultInner(BaseModel):
    """نتیجه داخلی اجرای ابزار"""
    ok: Optional[bool] = None
    data: Optional[Dict[str, Any]] = None
    channels: Optional[ToolResultChannels] = None


class ToolResultData(BaseModel):
    """داده کامل نتیجه ابزار"""
    ok: Optional[bool] = None
    error: Optional[str] = None
    result: Optional[ToolResultInner] = None
    tool_id: Optional[str] = None


class ToolResultItem(BaseModel):
    """یک نتیجه اجرای ابزار"""
    id: Optional[str] = Field("", description="شناسه")
    name: Optional[str] = Field(None, description="نام ابزار")
    ok: bool = Field(..., description="آیا اجرا موفق بود")
    summary: Optional[str] = Field(
        None,
        description="خلاصهٔ متنی خروجی برای UI (از orchestrator save_chat_turn)",
    )
    output: Optional[str] = Field(None, description="همان summary یا متن جایگزین")
    content: Optional[str] = Field(None, description="متن نمایشی (اختیاری)")
    text_for_llm: Optional[str] = Field(None, description="متنی که به LLM feed شده")
    data: Optional[ToolResultData] = Field(None, description="داده کامل نتیجه شامل data.result.channels.llm")
    error: Optional[str] = Field(None, description="پیام خطا در صورت شکست")
    execution_time: Optional[float] = Field(None, description="زمان اجرا بر حسب ثانیه")
    step: Optional[int] = Field(None, description="شماره مرحله در گراف")


class ChatMessageResponse(BaseModel):
    """پیام چت با نتایج کامل ابزار"""
    id: str
    session_id: str
    role: str = Field(..., description="نقش: user | assistant | system | tool")
    content: str
    tool_calls: Optional[List[ToolCallItem]] = Field(None, description="فراخوانی‌های ابزار توسط LLM")
    tool_results: Optional[List[ToolResultItem]] = Field(None, description="نتایج اجرای ابزارها شامل channels.llm")
    thinking: Optional[str] = Field(None, description="زنجیره تفکر مدل (Chain-of-Thought)")
    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    attachments: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None


class ChatMessageCreate(BaseModel):
    session_id: str
    role: str  # user, assistant, system, tool
    content: str
    user_id: Optional[str] = None  # For auto-creating session if needed
    trace_id: Optional[str] = None  # Links to agent_traces for detailed execution info
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None
    thinking: Optional[str] = None
    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    attachments: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


@app.post("/chat/messages", response_model=Dict[str, Any])
async def add_chat_message(data: ChatMessageCreate):
    """Add a message to a chat session"""
    storage = get_storage_from_app()
    if not storage.chats:
        raise HTTPException(status_code=503, detail="Chat repository not available")
    result = await storage.chats.add_message(
        session_id=data.session_id,
        role=data.role,
        content=data.content,
        tool_calls=data.tool_calls,
        tool_results=data.tool_results,
        thinking=data.thinking,
        tokens_prompt=data.tokens_prompt,
        tokens_completion=data.tokens_completion,
        attachments=data.attachments,
        metadata=data.metadata,
        user_id=data.user_id,
        trace_id=data.trace_id,
    )
    if not result:
        raise HTTPException(status_code=500, detail="Failed to add message")
    return result


@app.get(
    "/chat/sessions/{session_id}/messages",
    response_model=List[ChatMessageResponse],
    responses={
        200: {
            "description": "لیست پیام‌های جلسه. هر پیام assistant دارای tool_calls (فراخوانی‌ها) و tool_results (نتایج با channels.llm) است."
        }
    }
)
async def list_session_messages(
    session_id: str,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    include_tool_runs: bool = Query(False, description="[DEPRECATED] This parameter is ignored. Tool runs are included in assistant messages as tool_calls/tool_results."),
):
    """
    List messages in a session.

    Each **assistant** message may contain:
    - `tool_calls`: list of tool invocations made by the LLM
    - `tool_results`: list of tool execution results, each with:
      - `ok`: whether execution succeeded
      - `data.result.channels.llm`: LLM-readable text summary
      - `execution_time`: seconds taken

    Note: `include_tool_runs` parameter is deprecated. Tool data is now embedded in assistant messages.
    For detailed tool executions, use GET /traces/{trace_id}/tools
    """
    storage = get_storage_from_app()
    if not storage.chats:
        raise HTTPException(status_code=503, detail="Chat repository not available")
    
    messages = await storage.chats.list_messages(
        session_id=session_id,
        limit=limit,
        offset=offset,
    )
    
    # Note: include_tool_runs is deprecated - tool data is now in assistant messages
    # The tool_runs table has been removed; use /traces/{trace_id}/tools instead
    return messages


@app.get("/chat/sessions/{session_id}/context", response_model=List[Dict[str, Any]])
async def get_session_context(
    session_id: str,
    request: Request,
    max_messages: int = Query(20, ge=1, le=100),
    max_tokens: Optional[int] = Query(None),
):
    """Get recent messages for LLM context"""
    storage = get_storage_from_app()
    if not storage.chats:
        raise HTTPException(status_code=503, detail="Chat repository not available")
    session = await storage.chats.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    # If caller asserts a user identity, it must own the session (IDOR guard).
    # Gateway is responsible for admin bypass; it forwards X-User-Id for owners.
    if settings.STORAGE_RBAC_ENABLED:
        requester = (request.headers.get("X-User-Id") or "").strip()
        owner = str(session.get("user_id") or "")
        if requester and owner and requester != owner:
            raise HTTPException(status_code=403, detail="not_authorized_for_session")
    return await storage.chats.get_messages_for_context(
        session_id=session_id,
        max_messages=max_messages,
        max_tokens=max_tokens,
    )


@app.delete("/chat/sessions/{session_id}/messages")
async def clear_session_messages(session_id: str, request: Request):
    """Clear all messages in a session"""
    storage = get_storage_from_app()
    if not storage.chats:
        raise HTTPException(status_code=503, detail="Chat repository not available")
    session = await storage.chats.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _require_session_owner(session, request)
    count = await storage.chats.clear_messages(session_id)
    return {"status": "cleared", "count": count}


@app.post("/chat/sessions/{session_id}/auto-title")
async def auto_update_session_title(session_id: str):
    """Automatically update session title from first message"""
    storage = get_storage_from_app()
    if not storage.chats:
        raise HTTPException(status_code=503, detail="Chat repository not available")
    title = await storage.chats.auto_update_title(session_id)
    if not title:
        raise HTTPException(status_code=404, detail="No messages found")
    return {"title": title}


# ==================== Chat Message Edit & Feedback (UX-004, UX-005) ====================

class ChatMessageUpdate(BaseModel):
    content: str


@app.patch("/chat/messages/{message_id}", response_model=Dict[str, Any])
async def update_chat_message(
    message_id: str,
    data: ChatMessageUpdate,
):
    """
    Update a chat message content (UX-004).
    Only updates the content, preserves other fields.
    """
    storage = get_storage_from_app()
    if not storage.chats:
        raise HTTPException(status_code=503, detail="Chat repository not available")
    
    # Update message content
    result = await storage.chats.update_message(message_id, content=data.content)
    if not result:
        raise HTTPException(status_code=404, detail="Message not found")
    
    return result


@app.delete("/chat/messages/{message_id}")
async def delete_chat_message(message_id: str):
    """Delete a specific chat message"""
    storage = get_storage_from_app()
    if not storage.chats:
        raise HTTPException(status_code=503, detail="Chat repository not available")
    
    success = await storage.chats.delete_message(message_id)
    if not success:
        raise HTTPException(status_code=404, detail="Message not found")
    
    return {"status": "deleted", "message_id": message_id}


class MessageFeedbackCreate(BaseModel):
    feedback: str  # "like" or "dislike"
    comment: Optional[str] = None


@app.post("/chat/messages/{message_id}/feedback", response_model=Dict[str, Any])
async def add_message_feedback(
    message_id: str,
    data: MessageFeedbackCreate,
):
    """
    Add feedback to a specific message (UX-005).
    Unlike session-level feedback, this is per-message.
    """
    storage = get_storage_from_app()
    if not storage.chats:
        raise HTTPException(status_code=503, detail="Chat repository not available")
    
    # Validate feedback value
    if data.feedback not in ["like", "dislike"]:
        raise HTTPException(status_code=400, detail="Feedback must be 'like' or 'dislike'")
    
    result = await storage.chats.add_message_feedback(
        message_id=message_id,
        feedback=data.feedback,
        comment=data.comment,
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Message not found")
    
    return result


@app.get("/chat/messages/{message_id}/feedback")
async def get_message_feedback(message_id: str):
    """Get feedback for a specific message"""
    storage = get_storage_from_app()
    if not storage.chats:
        raise HTTPException(status_code=503, detail="Chat repository not available")
    
    result = await storage.chats.get_message_feedback(message_id)
    return result or {"message_id": message_id, "feedback": None}


# ==================== User Messenger (internal messaging) ====================


class MessengerSendRequest(BaseModel):
    to: str
    subject: Optional[str] = None
    body: str = ""
    attachments: Optional[List[Any]] = None
    priority: Optional[str] = "normal"
    client_message_id: Optional[str] = None
    draft: bool = False
    cc: Optional[List[str]] = None
    bcc: Optional[List[str]] = None
    content_type: Optional[str] = None
    voice_duration_ms: Optional[int] = None
    scheduled_at: Optional[str] = None
    reply_to_id: Optional[str] = None


class MessengerPatchRequest(BaseModel):
    read: Optional[bool] = None
    folder: Optional[str] = None
    body: Optional[str] = None
    starred: Optional[bool] = None
    pinned: Optional[bool] = None
    muted: Optional[bool] = None


class MessengerReplyRequest(BaseModel):
    body: str = ""
    attachments: Optional[List[Any]] = None
    client_message_id: Optional[str] = None
    content_type: Optional[str] = None
    voice_duration_ms: Optional[int] = None


def _messenger_user_id(request: Request) -> str:
    ctx = _storage_auth_context(request)
    uid = (ctx.get("user_id") or "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="X-User-Id required")
    return uid


@app.get("/messages")
async def list_user_messages(
    request: Request,
    folder: str = Query("inbox"),
    page: int = Query(1, ge=1),
    limit: int = Query(30, ge=1, le=100),
    q: Optional[str] = Query(None),
):
    storage = get_storage_from_app()
    if not storage.messenger:
        raise HTTPException(status_code=503, detail="Messenger repository not available")
    user_id = _messenger_user_id(request)
    return await storage.messenger.list_messages(
        user_id, folder=folder, page=page, limit=limit, q=q
    )


@app.get("/messages/attachments/library")
async def list_messenger_library(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    search: Optional[str] = Query(None),
    media_type: Optional[str] = Query(None),
):
    storage = get_storage_from_app()
    if not storage.messenger:
        raise HTTPException(status_code=503, detail="Messenger repository not available")
    user_id = _messenger_user_id(request)
    return await storage.messenger.list_attachable_artifacts(
        user_id, page=page, page_size=page_size, search=search, media_type=media_type
    )


@app.get("/messages/search")
async def search_user_messages(
    request: Request,
    q: str = Query(..., min_length=1),
    folder: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(30, ge=1, le=100),
):
    storage = get_storage_from_app()
    if not storage.messenger:
        raise HTTPException(status_code=503, detail="Messenger repository not available")
    user_id = _messenger_user_id(request)
    return await storage.messenger.search_messages(
        user_id, q=q, folder=folder, page=page, limit=limit
    )


@app.get("/messages/{message_id}/replies")
async def list_message_replies(
    message_id: str,
    request: Request,
    limit: int = Query(50, ge=1, le=100),
):
    storage = get_storage_from_app()
    if not storage.messenger:
        raise HTTPException(status_code=503, detail="Messenger repository not available")
    user_id = _messenger_user_id(request)
    return await storage.messenger.list_thread_replies(user_id, message_id, limit=limit)


@app.post("/messages/{message_id}/reply")
async def reply_user_message(message_id: str, data: MessengerReplyRequest, request: Request):
    storage = get_storage_from_app()
    if not storage.messenger:
        raise HTTPException(status_code=503, detail="Messenger repository not available")
    user_id = _messenger_user_id(request)
    try:
        result = await storage.messenger.reply_to_message(
            user_id,
            message_id,
            body=data.body,
            attachments=data.attachments,
            client_message_id=data.client_message_id,
            content_type=data.content_type,
            voice_duration_ms=data.voice_duration_ms,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not result:
        raise HTTPException(status_code=400, detail="Failed to reply")
    return result


@app.get("/messages/{message_id}")
async def get_user_message(message_id: str, request: Request):
    storage = get_storage_from_app()
    if not storage.messenger:
        raise HTTPException(status_code=503, detail="Messenger repository not available")
    user_id = _messenger_user_id(request)
    result = await storage.messenger.get_message(user_id, message_id)
    if not result:
        raise HTTPException(status_code=404, detail="Message not found")
    return result


@app.post("/messages")
async def send_user_message(data: MessengerSendRequest, request: Request):
    storage = get_storage_from_app()
    if not storage.messenger:
        raise HTTPException(status_code=503, detail="Messenger repository not available")
    user_id = _messenger_user_id(request)
    try:
        result = await storage.messenger.send_message(
            user_id,
            to=data.to,
            subject=data.subject,
            body=data.body,
            attachments=data.attachments,
            priority=data.priority or "normal",
            client_message_id=data.client_message_id,
            draft=data.draft,
            cc=data.cc,
            bcc=data.bcc,
            content_type=data.content_type,
            voice_duration_ms=data.voice_duration_ms,
            scheduled_at=data.scheduled_at,
            parent_message_id=data.reply_to_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not result:
        raise HTTPException(status_code=400, detail="Failed to send message")
    return result


@app.patch("/messages/{message_id}")
async def patch_user_message(message_id: str, data: MessengerPatchRequest, request: Request):
    storage = get_storage_from_app()
    if not storage.messenger:
        raise HTTPException(status_code=503, detail="Messenger repository not available")
    user_id = _messenger_user_id(request)
    try:
        result = await storage.messenger.update_message(
            user_id,
            message_id,
            read=data.read,
            folder=data.folder,
            body=data.body,
            starred=data.starred,
            pinned=data.pinned,
            muted=data.muted,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not result:
        raise HTTPException(status_code=404, detail="Message not found")
    return result


@app.post("/messages/{message_id}/resend")
async def resend_user_message(message_id: str, request: Request):
    storage = get_storage_from_app()
    if not storage.messenger:
        raise HTTPException(status_code=503, detail="Messenger repository not available")
    user_id = _messenger_user_id(request)
    try:
        result = await storage.messenger.resend_message(user_id, message_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not result:
        raise HTTPException(status_code=404, detail="Message not found")
    return result


@app.delete("/messages/{message_id}")
async def delete_user_message(message_id: str, request: Request):
    storage = get_storage_from_app()
    if not storage.messenger:
        raise HTTPException(status_code=503, detail="Messenger repository not available")
    user_id = _messenger_user_id(request)
    ok = await storage.messenger.delete_message(user_id, message_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Message not found")
    return {"status": "deleted", "message_id": message_id}


# ==================== Case Importer Routes ====================

class CaseCreate(BaseModel):
    case_id: str
    case_name: str
    user_id: Optional[str] = None
    group_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CaseStatusUpdate(BaseModel):
    status: str  # pending, analyzing, embedding, storing, completed, failed
    progress: Optional[float] = None
    error: Optional[str] = None


class ToolMetadataBatch(BaseModel):
    tool_id: str
    records: List[Dict[str, Any]]


class MongoDocumentsBatch(BaseModel):
    documents: List[Dict[str, Any]]


@app.post("/cases", response_model=Dict[str, Any])
async def create_case(data: CaseCreate):
    """
    ایجاد یک Case جدید (پرونده)
    
    user_id به عنوان owner ذخیره می‌شود.
    """
    storage = get_storage_from_app()
    if not storage.postgres or not storage.postgres.pool:
        raise HTTPException(status_code=503, detail="PostgreSQL not available")
    
    pool = storage.postgres.pool
    try:
        async with pool.acquire() as conn:
            # Create cases table if not exists
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS cases (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    user_id TEXT,
                    group_id TEXT,
                    session_id TEXT,
                    status TEXT DEFAULT 'pending',
                    progress FLOAT DEFAULT 0.0,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Insert case
            await conn.execute("""
                INSERT INTO cases (id, name, user_id, group_id, session_id, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    group_id = COALESCE(EXCLUDED.group_id, cases.group_id),
                    updated_at = NOW()
            """, data.case_id, data.case_name, data.user_id, data.group_id, data.session_id, 
                json.dumps(data.metadata or {}))
            return {
                "id": data.case_id,
                "name": data.case_name,
                "user_id": data.user_id,
                "group_id": data.group_id,
                "session_id": data.session_id,
            }
    except Exception as e:
        logger.error(f"Failed to create case: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/cases/{case_id}/status", response_model=Dict[str, Any])
async def update_case_status(case_id: str, data: CaseStatusUpdate):
    """بروزرسانی وضعیت یک Case"""
    storage = get_storage_from_app()
    if not storage.postgres or not storage.postgres.pool:
        raise HTTPException(status_code=503, detail="PostgreSQL not available")
    
    pool = storage.postgres.pool
    try:
        async with pool.acquire() as conn:
            await conn.execute("""
                UPDATE cases SET
                    status = $2,
                    progress = COALESCE($3, progress),
                    updated_at = NOW()
                WHERE id = $1
            """, case_id, data.status, data.progress)
            
            return {"id": case_id, "status": data.status}
    except Exception as e:
        logger.error(f"Failed to update case status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class CaseMetadataUpdate(BaseModel):
    metadata: Dict[str, Any]


@app.patch("/cases/{case_id}/metadata", response_model=Dict[str, Any])
async def update_case_metadata(case_id: str, data: CaseMetadataUpdate):
    """
    Merge metadata jsonb into a case row.
    Used by case-importer to persist case_root (e.g. minio://cases/case_xxx).
    """
    storage = get_storage_from_app()
    ok = await storage.case_update_metadata(case_id, data.metadata)
    if not ok:
        raise HTTPException(status_code=404, detail="Case not found")
    return {"id": case_id, "metadata": data.metadata}


@app.get("/cases/{case_id}", response_model=Dict[str, Any])
async def get_case(case_id: str, request: Request):
    """دریافت اطلاعات یک Case — RBAC: فقط owner."""
    storage = get_storage_from_app()
    return await _require_case_access(storage, case_id, request)


@app.get("/cases/{case_id}/artifacts", response_model=Dict[str, Any])
async def list_case_artifacts(
    case_id: str,
    request: Request,
    limit: Optional[int] = Query(
        None,
        ge=1,
        description="Optional max rows; omit for the full case artifact roster",
    ),
):
    """List artifacts linked to a case (for agent-flow Start / RunFileInput).

    Excludes plugin session records (``session.json`` / layout=sessions) —
    those are user-scoped like chat sessions, not case evidence.
    """
    storage = get_storage_from_app()
    await _require_case_access(storage, case_id, request)
    items = await storage.case_list_artifacts(case_id, limit=limit)
    return {"case_id": case_id, "count": len(items), "items": items}


@app.get("/cases/{case_id}/tool-sessions", response_model=Dict[str, Any])
async def list_case_tool_sessions(
    case_id: str,
    request: Request,
    limit: int = Query(100, ge=1, le=500),
    plugin_id: Optional[str] = Query(
        None,
        description="Optional plugin filter; omit for all case-visible sessions",
    ),
):
    """Case-visible plugin/tool session outputs (soft-link, opt-in).

    Lists ``tool_sessions`` stamped with ``payload.case_id`` where
    ``expose_to_case`` is true. Deliverable artifacts remain user-scoped
    (``bind_case: false``); this endpoint never binds blobs to the case.
    Case RBAC via ``_require_case_access``; artifact download uses normal
    artifact owner/group RBAC.
    """
    return await _list_case_tool_sessions_payload(
        case_id=case_id,
        request=request,
        limit=limit,
        plugin_id=plugin_id,
    )


@app.get("/cases/{case_id}/agent-flow/sessions", response_model=Dict[str, Any])
async def list_case_agent_flow_sessions(
    case_id: str,
    request: Request,
    limit: int = Query(100, ge=1, le=500),
):
    """Alias: agent_flow-only case-visible sessions (compat)."""
    return await _list_case_tool_sessions_payload(
        case_id=case_id,
        request=request,
        limit=limit,
        plugin_id="agent_flow",
    )


async def _list_case_tool_sessions_payload(
    *,
    case_id: str,
    request: Request,
    limit: int,
    plugin_id: Optional[str],
) -> Dict[str, Any]:
    storage = get_storage_from_app()
    await _require_case_access(storage, case_id, request)
    if not storage.tool_sessions:
        raise HTTPException(status_code=503, detail="Tool session repository not available")
    plugin = str(plugin_id or "").strip() or None
    items = await storage.tool_sessions.list_by_case_id(
        case_id=case_id,
        plugin_id=plugin,
        limit=limit,
        user_id=None,
        require_expose_to_case=True,
    )
    # Drop case-evidence artifact ids. AI Summary is deliverables only.
    case_aids: set[str] = set()
    try:
        if storage.postgres:
            for aid in await storage.postgres.list_artifact_ids_for_case(case_id):
                s = str(aid or "").strip()
                if s:
                    case_aids.add(s)
    except Exception:
        case_aids = set()
    sessions = []
    for row in items:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        files = row.get("files") if isinstance(row.get("files"), list) else []
        if not files and isinstance(payload.get("files"), list):
            files = payload.get("files") or []
        if case_aids and isinstance(files, list):
            files = [
                f
                for f in files
                if isinstance(f, dict)
                and str(f.get("artifact_id") or "").strip() not in case_aids
            ]
        expose_raw = payload.get("expose_to_case")
        expose_on = expose_raw is True or str(expose_raw or "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        sessions.append(
            {
                "session_id": row.get("session_id") or row.get("id"),
                "plugin_id": row.get("plugin_id") or payload.get("plugin_id"),
                "run_id": row.get("run_id"),
                "record_id": row.get("record_id"),
                "flow_id": row.get("flow_id")
                or payload.get("flow_id")
                or row.get("record_id"),
                "status": row.get("status"),
                "ok": row.get("ok"),
                "error": row.get("error"),
                "output_preview": row.get("output_preview") or "",
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
                "files": files,
                "case_id": payload.get("case_id") or case_id,
                "expose_to_case": expose_on,
            }
        )
    return {
        "case_id": case_id,
        "count": len(sessions),
        "sessions": sessions,
    }


@app.delete("/cases/{case_id}", response_model=Dict[str, Any])
async def delete_case(case_id: str, request: Request, purge: bool = Query(True)):
    """
    Purge a case-importer case from PostgreSQL, MongoDB, Qdrant, Neo4j, and MinIO.
    RBAC: only the case owner may delete.
    """
    storage = get_storage_from_app()
    await _require_case_access(storage, case_id, request, require_owner=True)
    if purge:
        result = await storage.case_purge_full(case_id)
    else:
        deleted = False
        if storage.postgres and storage.postgres.pool:
            deleted = await storage.postgres.delete_case_row(case_id)
        result = {"ok": deleted, "case_id": case_id, "case_row_deleted": deleted}
    if not result.get("ok") and not result.get("artifact_ids") and not result.get("case_row_deleted"):
        raise HTTPException(status_code=404, detail="Case not found or purge failed")
    return result


@app.get("/cases", response_model=List[Dict[str, Any]])
async def list_cases(
    request: Request,
    session_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """
    List cases visible to the authenticated user (Postgres group-RBAC).
    """
    storage = get_storage_from_app()
    if not storage.postgres or not storage.postgres.pool:
        raise HTTPException(status_code=503, detail="PostgreSQL not available")

    ctx = _storage_auth_context(request)
    user_id = ctx.get("user_id")
    if settings.STORAGE_RBAC_ENABLED:
        if not user_id:
            raise HTTPException(status_code=401, detail="user_id_required")
        try:
            return await storage.case_accessible_list(
                user_id=str(user_id),
                limit=limit,
                offset=offset,
                session_id=session_id,
                status=status,
            )
        except Exception as e:
            logger.error(f"Failed to list accessible cases: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    pool = storage.postgres.pool
    try:
        conditions = []
        params = []
        idx = 1

        if session_id:
            conditions.append(f"session_id = ${idx}")
            params.append(session_id)
            idx += 1

        if status:
            conditions.append(f"status = ${idx}")
            params.append(status)
            idx += 1

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        params.extend([limit, offset])
        query = f"""
            SELECT id, name, user_id, group_id, session_id, status, progress, created_at, updated_at
            FROM cases
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ${idx} OFFSET ${idx + 1}
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(r) for r in rows]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list cases: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tool-metadata/batch", response_model=Dict[str, Any])
async def store_tool_metadata_batch(data: ToolMetadataBatch):
    """
    ذخیره metadata در PostgreSQL با جدول داینامیک
    
    هر tool_id یک جدول مخصوص داره: tool_{tool_id}
    این منطبق با معماری 5 کاناله در case_review است.
    """
    storage = get_storage_from_app()
    if not storage.postgres or not storage.postgres.pool:
        raise HTTPException(status_code=503, detail="PostgreSQL not available")
    
    pool = storage.postgres.pool
    try:
        # ساخت نام جدول امن
        safe_tool_id = data.tool_id.replace(".", "_").replace("-", "_")
        table_name = f"tool_{safe_tool_id}"
        
        async with pool.acquire() as conn:
            # ایجاد جدول اگر وجود نداره
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id TEXT PRIMARY KEY,
                    artifact_id TEXT,
                    case_id TEXT,
                    item_type TEXT,
                    item_index INTEGER DEFAULT 0,
                    file_name TEXT,
                    file_path TEXT,
                    data JSONB DEFAULT '{{}}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # درج رکوردها
            inserted = 0
            for record in data.records:
                await conn.execute(f"""
                    INSERT INTO {table_name} (id, artifact_id, case_id, item_type, item_index, file_name, file_path, data)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (id) DO UPDATE SET
                        data = EXCLUDED.data,
                        created_at = NOW()
                """,
                    record.get("id"),
                    record.get("artifact_id"),
                    record.get("case_id"),
                    record.get("item_type"),
                    record.get("item_index", 0),
                    record.get("file_name"),
                    record.get("file_path"),
                    json.dumps(record.get("data", {})),
                )
                inserted += 1
            
            return {
                "table": table_name,
                "count": inserted,
                "tool_id": data.tool_id,
            }
    except Exception as e:
        logger.error(f"Failed to store tool metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tool-outputs/search", response_model=Dict[str, Any])
async def search_tool_outputs(
    request: Request,
    q: str = Query(..., min_length=1, description="Substring match on vision/tool text"),
    tool_name: Optional[str] = Query("vision_info"),
    channel: Optional[str] = Query(None),
    limit: int = Query(30, ge=1, le=100),
):
    """Search persisted tool text (Mongo tool_outputs), e.g. vision_info captions."""
    storage = get_storage_from_app()
    ctx = _storage_auth_context(request)
    user_id = ctx.get("user_id")
    if settings.STORAGE_RBAC_ENABLED and not user_id:
        raise HTTPException(status_code=401, detail="user_id_required")
    items = await storage.search_tool_outputs_text(
        q,
        user_id=str(user_id) if user_id else None,
        tool_name=(tool_name or "").strip() or None,
        channel=(channel or "").strip() or None,
        limit=limit,
    )
    return {"items": items, "total_count": len(items), "query": q}


@app.post("/tool-outputs/batch", response_model=Dict[str, Any])
async def store_tool_outputs_batch(data: MongoDocumentsBatch):
    """
    ذخیره خروجی ابزارها در MongoDB (collection: tool_outputs)
    
    این برای کانال‌های rawdata, llm, ui, embed استفاده می‌شود.
    منطبق با معماری 5 کاناله در case_review.
    """
    storage = get_storage_from_app()
    mongo = storage.mongo
    if not mongo:
        raise HTTPException(status_code=503, detail="MongoDB not available")
    
    try:
        collection = mongo.db["tool_outputs"]
        
        # استفاده از bulk write برای کارایی بالاتر
        from pymongo import ReplaceOne
        import json as _json

        # Mongo BSON hard limit 16MB — refuse individual monsters early with a
        # clear error instead of a cryptic bulk_write failure.
        _SAFE = 14 * 1024 * 1024
        safe_docs: List[Dict[str, Any]] = []
        skipped: List[str] = []
        for doc in data.documents:
            if not isinstance(doc, dict):
                continue
            try:
                nbytes = len(
                    _json.dumps(doc, ensure_ascii=False, default=str).encode("utf-8")
                )
            except Exception:
                nbytes = 0
            if nbytes > _SAFE:
                skipped.append(str(doc.get("_id") or "?"))
                logger.error(
                    "Skipping oversized tool_output doc _id=%s approx_bytes=%s",
                    doc.get("_id"),
                    nbytes,
                )
                continue
            safe_docs.append(doc)
        
        operations = []
        inserted_docs = []
        for doc in safe_docs:
            doc_id = doc.get("_id")
            if doc_id:
                operations.append(
                    ReplaceOne({"_id": doc_id}, doc, upsert=True)
                )
            else:
                await collection.insert_one(doc)
                inserted_docs.append(doc)
        
        from .artifact_geo import upsert_artifact_geo_batch, upsert_artifact_geo_from_doc

        # Chunk bulk_write — a single command with tens of thousands of ops
        # can also exceed Mongo message size limits.
        _BULK_CHUNK = 200
        upserted = 0
        modified = 0
        if operations:
            for i in range(0, len(operations), _BULK_CHUNK):
                slice_ops = operations[i : i + _BULK_CHUNK]
                result = await collection.bulk_write(slice_ops)
                upserted += int(result.upserted_count or 0)
                modified += int(result.modified_count or 0)
            try:
                await upsert_artifact_geo_batch(safe_docs, storage)
            except Exception as geo_exc:
                logger.warning("artifact_geo upsert skipped: %s", geo_exc)
            out = {
                "count": upserted + modified,
                "upserted": upserted,
                "modified": modified,
            }
            if skipped:
                out["skipped_oversized"] = skipped
            return out

        for doc in inserted_docs:
            try:
                await upsert_artifact_geo_from_doc(doc, storage)
            except Exception as geo_exc:
                logger.warning("artifact_geo upsert skipped: %s", geo_exc)
        
        out = {"count": len(safe_docs)}
        if skipped:
            out["skipped_oversized"] = skipped
        return out
    except Exception as e:
        logger.error(f"Failed to store tool outputs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _ensure_tool_records_schema(pool) -> None:
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS tool_records (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            artifact_id UUID REFERENCES artifacts(id) ON DELETE CASCADE,
            tool_id TEXT NOT NULL,
            case_id TEXT,
            user_id UUID,
            run_id TEXT,
            record_type TEXT NOT NULL DEFAULT 'row',
            record_index INT NOT NULL DEFAULT 0,
            data JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    await pool.execute(
        "CREATE INDEX IF NOT EXISTS idx_tool_records_artifact ON tool_records(artifact_id)"
    )
    await pool.execute(
        "CREATE INDEX IF NOT EXISTS idx_tool_records_tool ON tool_records(tool_id)"
    )
    await pool.execute(
        "CREATE INDEX IF NOT EXISTS idx_tool_records_case ON tool_records(case_id)"
    )
    await pool.execute(
        "CREATE INDEX IF NOT EXISTS idx_tool_records_run ON tool_records(run_id)"
    )


@app.post("/tool-records/batch", response_model=Dict[str, Any])
async def store_tool_records_batch(data: ToolRecordsBatch, request: Request):
    """Upsert structured rows into the shared tool_records table (no per-tool DDL)."""
    storage = get_storage_from_app()
    if not storage.postgres or not storage.postgres.pool:
        raise HTTPException(status_code=503, detail="PostgreSQL not available")

    ctx = _storage_auth_context(request)
    owner_id = (data.user_id or ctx.get("user_id") or "").strip() or None
    case_id = (data.case_id or ctx.get("case_id") or "").strip() or None
    tool_id = (data.tool_id or "").strip()
    if not tool_id:
        raise HTTPException(status_code=400, detail="tool_id_required")
    if not data.records:
        return {"count": 0, "tool_id": tool_id}

    pool = storage.postgres.pool
    try:
        await _ensure_tool_records_schema(pool)
    except Exception as exc:
        logger.warning("tool_records schema ensure failed: %s", exc)

    from .base import to_uuid

    inserted = 0
    async with pool.acquire() as conn:
        for idx, record in enumerate(data.records):
            if not isinstance(record, dict):
                continue
            artifact_id = (
                str(record.get("artifact_id") or data.artifact_id or "").strip() or None
            )
            record_type = str(record.get("record_type") or record.get("type") or "row")
            record_index = int(record.get("record_index", record.get("index", idx)) or 0)
            payload = record.get("data") if isinstance(record.get("data"), dict) else record
            # Avoid nesting identity keys into data when whole record used as payload
            if payload is record:
                payload = {
                    k: v
                    for k, v in record.items()
                    if k
                    not in {
                        "artifact_id",
                        "record_type",
                        "type",
                        "record_index",
                        "index",
                        "id",
                    }
                }
            row_id = to_uuid(record.get("id")) or uuid.uuid4()
            await conn.execute(
                """
                INSERT INTO tool_records (
                    id, artifact_id, tool_id, case_id, user_id, run_id,
                    record_type, record_index, data
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    data = EXCLUDED.data,
                    record_type = EXCLUDED.record_type,
                    record_index = EXCLUDED.record_index,
                    updated_at = now()
                """,
                row_id,
                to_uuid(artifact_id),
                tool_id,
                case_id or record.get("case_id"),
                to_uuid(owner_id or record.get("user_id")),
                (data.run_id or record.get("run_id") or None),
                record_type,
                record_index,
                json.dumps(payload or {}),
            )
            inserted += 1

    return {"count": inserted, "tool_id": tool_id}


# ==================== User Profile Endpoints ====================
# اندپوینت‌های مدیریت پروفایل کاربران

class UserProfileCreate(BaseModel):
    """Create user profile request."""
    keycloak_id: str
    username: str
    email: str
    profile: Optional[Dict[str, Any]] = None


class UserProfileUpdate(BaseModel):
    """Update user profile request."""
    bio: Optional[str] = None
    timezone: Optional[str] = None
    language: Optional[str] = None
    avatar_url: Optional[str] = None
    notifications: Optional[Dict[str, Any]] = None


class UserActivityLog(BaseModel):
    """Log user activity request."""
    keycloak_id: str
    action: str
    details: Optional[Dict[str, Any]] = None


@app.post("/users/profile")
async def create_user_profile(data: UserProfileCreate):
    """Create or update user profile in PostgreSQL."""
    storage = get_storage_from_app()
    if not storage.users:
        raise HTTPException(status_code=503, detail="User repository not available")
    
    # Extract display_name from profile if provided
    display_name = data.profile.get("display_name") if data.profile else None
    
    result = await storage.users.create_or_update_by_keycloak(
        keycloak_id=data.keycloak_id,
        username=data.username,
        email=data.email,
        display_name=display_name,
    )
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create/update profile")
    
    return result


@app.get("/users/profile/{keycloak_id}")
async def get_user_profile(keycloak_id: str):
    """Get user profile by keycloak ID from PostgreSQL."""
    storage = get_storage_from_app()
    if not storage.users:
        raise HTTPException(status_code=503, detail="User repository not available")
    
    profile = await storage.users.get_user_by_keycloak_id(keycloak_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@app.put("/users/profile/{keycloak_id}")
async def update_user_profile(keycloak_id: str, data: UserProfileUpdate):
    """Update user profile in PostgreSQL."""
    storage = get_storage_from_app()
    if not storage.users:
        raise HTTPException(status_code=503, detail="User repository not available")
    
    # First get user by keycloak_id to get user.id
    user = await storage.users.get_user_by_keycloak_id(keycloak_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update using user.id
    update_data = data.model_dump(exclude_unset=True)
    result = await storage.users.update_user(
        user_id=user["id"],
        display_name=update_data.get("display_name"),
        avatar_url=update_data.get("avatar_url"),
    )
    
    return {"ok": True, "updated": list(update_data.keys())}


@app.delete("/users/profile/{keycloak_id}")
async def delete_user_profile(keycloak_id: str):
    """Delete user profile from PostgreSQL."""
    storage = get_storage_from_app()
    if not storage.users:
        raise HTTPException(status_code=503, detail="User repository not available")
    
    # Get user first
    user = await storage.users.get_user_by_keycloak_id(keycloak_id)
    if user:
        # Mark as inactive instead of deleting
        await storage.users.update_user(user_id=user["id"], is_active=False)
    return {"ok": True}


@app.post("/users/activity")
async def log_user_activity(data: UserActivityLog):
    """Log user activity (audit trail)."""
    # For now, just log to console
    logger.info(f"User activity: {data.keycloak_id} - {data.action} - {data.details}")
    return {"ok": True}


# ==================== Agent Trace Endpoints ====================

class TraceCreate(BaseModel):
    trace_id: str
    session_id: str
    user_message: str
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TraceUpdate(BaseModel):
    status: Optional[str] = None
    complexity: Optional[str] = None
    plan_version: Optional[int] = None
    execution_plan: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None
    synthesis_reasoning: Optional[str] = None
    overall_confidence: Optional[float] = None
    total_steps: Optional[int] = None
    total_tool_calls: Optional[int] = None
    replan_count: Optional[int] = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    completed_at: Optional[str] = None
    total_duration_ms: Optional[int] = None


class TraceStepCreate(BaseModel):
    trace_id: str
    step_number: int
    node_name: str
    task_id: Optional[str] = None
    task_description: Optional[str] = None
    status: Optional[str] = "started"
    input_data: Optional[Dict[str, Any]] = None
    plan_generated: Optional[Dict[str, Any]] = None
    planning_reasoning: Optional[str] = None
    tool_calls_requested: Optional[List[Dict[str, Any]]] = None
    llm_messages_sent: Optional[List[Dict[str, Any]]] = None
    started_at: Optional[str] = None


class TraceStepUpdate(BaseModel):
    status: Optional[str] = None
    output_data: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None
    needs_retry: Optional[bool] = None
    needs_replan: Optional[bool] = None
    synthesis_content: Optional[str] = None
    llm_response: Optional[Dict[str, Any]] = None
    llm_tokens_prompt: Optional[int] = None
    llm_tokens_completion: Optional[int] = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    completed_at: Optional[str] = None
    duration_ms: Optional[int] = None


class ToolExecutionCreate(BaseModel):
    trace_id: str
    tool_name: str
    arguments: Dict[str, Any]
    step_id: Optional[str] = None
    tool_id: Optional[str] = None
    task_id: Optional[str] = None
    status: Optional[str] = "pending"
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    started_at: Optional[str] = None
    retry_count: Optional[int] = 0


class ToolExecutionUpdate(BaseModel):
    status: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    result_summary: Optional[str] = None
    result_sent_to_llm: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: Optional[int] = None
    retry_count: Optional[int] = None


class StreamEventCreate(BaseModel):
    trace_id: str
    event_type: str
    event_data: Dict[str, Any]
    sequence_number: int
    timestamp: Optional[str] = None


class BatchWriteItem(BaseModel):
    type: str  # trace_update, step, step_update, tool_execution, tool_execution_update, stream_event
    data: Dict[str, Any]


class BatchWriteRequest(BaseModel):
    items: List[BatchWriteItem]


@app.post("/traces", response_model=Dict[str, Any])
async def create_trace(data: TraceCreate):
    """Create a new agent trace."""
    storage: Storage = get_storage_from_app()
    if not storage.traces:
        raise HTTPException(status_code=503, detail="Trace repository not available")
    
    result = await storage.traces.create_trace(
        trace_id=data.trace_id,
        session_id=data.session_id,
        user_message=data.user_message,
        user_id=data.user_id,
        metadata=data.metadata
    )
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create trace")
    return result


@app.put("/traces/{trace_id}", response_model=Dict[str, Any])
async def update_trace(trace_id: str, data: TraceUpdate):
    """Update an existing trace."""
    storage: Storage = get_storage_from_app()
    if not storage.traces:
        raise HTTPException(status_code=503, detail="Trace repository not available")
    
    updates = data.model_dump(exclude_unset=True)
    result = await storage.traces.update_trace(trace_id, **updates)
    
    if not result:
        raise HTTPException(status_code=404, detail="Trace not found or no updates applied")
    return result


# ==================== User Feedback API ====================

class TraceFeedbackRequest(BaseModel):
    """User feedback for a trace (thumbs up/down + optional comment)."""
    rating: int = Field(..., ge=-1, le=1, description="1=thumbs up, -1=thumbs down, 0=neutral")
    comment: Optional[str] = Field(None, max_length=2000, description="Optional user comment")


@app.put("/traces/{trace_id}/feedback", response_model=Dict[str, Any])
async def add_trace_feedback(trace_id: str, data: TraceFeedbackRequest):
    """
    Add user feedback (thumbs up/down + comment) to a trace.
    
    This endpoint allows users to rate the quality of AI responses.
    - rating: 1 (thumbs up), -1 (thumbs down), 0 (neutral/reset)
    - comment: Optional text feedback
    """
    storage: Storage = get_storage_from_app()
    if not storage.traces:
        raise HTTPException(status_code=503, detail="Trace repository not available")
    
    result = await storage.traces.add_feedback(
        trace_id=trace_id,
        rating=data.rating,
        comment=data.comment
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Trace not found")
    return result


@app.get("/traces/{trace_id}/feedback", response_model=Dict[str, Any])
async def get_trace_feedback(trace_id: str):
    """Get user feedback for a trace."""
    storage: Storage = get_storage_from_app()
    if not storage.traces:
        raise HTTPException(status_code=503, detail="Trace repository not available")
    
    trace = await storage.traces.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    return {
        "trace_id": trace_id,
        "user_feedback_rating": trace.get("user_feedback_rating"),
        "user_feedback_comment": trace.get("user_feedback_comment"),
        "user_feedback_at": trace.get("user_feedback_at")
    }


@app.get("/traces/{trace_id}", response_model=Dict[str, Any])
async def get_trace(trace_id: str, full: bool = False):
    """Get a trace by ID. Use full=true to include all steps and tool executions."""
    storage: Storage = get_storage_from_app()
    if not storage.traces:
        raise HTTPException(status_code=503, detail="Trace repository not available")
    
    if full:
        result = await storage.traces.get_trace_full(trace_id)
    else:
        result = await storage.traces.get_trace(trace_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Trace not found")
    return result


@app.get("/traces", response_model=List[Dict[str, Any]])
async def list_traces(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """List traces with optional filters."""
    storage: Storage = get_storage_from_app()
    if not storage.traces:
        raise HTTPException(status_code=503, detail="Trace repository not available")
    
    return await storage.traces.list_traces(
        session_id=session_id,
        user_id=user_id,
        status=status,
        limit=limit,
        offset=offset
    )


@app.post("/traces/{trace_id}/steps", response_model=Dict[str, Any])
async def add_trace_step(trace_id: str, data: TraceStepCreate):
    """Add a step to a trace."""
    storage: Storage = get_storage_from_app()
    if not storage.traces:
        raise HTTPException(status_code=503, detail="Trace repository not available")
    
    if data.trace_id != trace_id:
        raise HTTPException(status_code=400, detail="trace_id mismatch")
    
    step_data = data.model_dump(exclude={"trace_id"})
    result = await storage.traces.add_step(
        trace_id=trace_id,
        step_number=step_data.pop("step_number"),
        node_name=step_data.pop("node_name"),
        **step_data
    )
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to add step")
    return result


@app.put("/traces/{trace_id}/steps/{step_number}", response_model=Dict[str, Any])
async def update_trace_step(trace_id: str, step_number: int, data: TraceStepUpdate):
    """Update a step in a trace."""
    storage: Storage = get_storage_from_app()
    if not storage.traces:
        raise HTTPException(status_code=503, detail="Trace repository not available")
    
    updates = data.model_dump(exclude_unset=True)
    result = await storage.traces.update_step(trace_id, step_number, **updates)
    
    if not result:
        raise HTTPException(status_code=404, detail="Step not found or no updates applied")
    return result


@app.get("/traces/{trace_id}/steps", response_model=List[Dict[str, Any]])
async def list_trace_steps(trace_id: str, node_name: Optional[str] = None):
    """List all steps in a trace."""
    storage: Storage = get_storage_from_app()
    if not storage.traces:
        raise HTTPException(status_code=503, detail="Trace repository not available")
    
    return await storage.traces.list_steps(trace_id, node_name=node_name)


@app.post("/traces/{trace_id}/tools", response_model=Dict[str, Any])
async def add_tool_execution(trace_id: str, data: ToolExecutionCreate):
    """Add a tool execution record."""
    storage: Storage = get_storage_from_app()
    if not storage.traces:
        raise HTTPException(status_code=503, detail="Trace repository not available")
    
    if data.trace_id != trace_id:
        raise HTTPException(status_code=400, detail="trace_id mismatch")
    
    tool_data = data.model_dump(exclude={"trace_id"})
    result = await storage.traces.add_tool_execution(
        trace_id=trace_id,
        tool_name=tool_data.pop("tool_name"),
        arguments=tool_data.pop("arguments"),
        **tool_data
    )
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to add tool execution")
    return result


@app.put("/traces/{trace_id}/tools/{tool_id}", response_model=Dict[str, Any])
async def update_tool_execution(trace_id: str, tool_id: str, data: ToolExecutionUpdate):
    """Update a tool execution record."""
    storage: Storage = get_storage_from_app()
    if not storage.traces:
        raise HTTPException(status_code=503, detail="Trace repository not available")
    
    updates = data.model_dump(exclude_unset=True)
    result = await storage.traces.update_tool_execution(trace_id, tool_id, **updates)
    
    if not result:
        raise HTTPException(status_code=404, detail="Tool execution not found or no updates applied")
    return result


@app.get("/traces/{trace_id}/tools", response_model=List[Dict[str, Any]])
async def list_tool_executions(trace_id: str, task_id: Optional[str] = None):
    """List all tool executions in a trace."""
    storage: Storage = get_storage_from_app()
    if not storage.traces:
        raise HTTPException(status_code=503, detail="Trace repository not available")
    
    return await storage.traces.list_tool_executions(trace_id, task_id=task_id)


@app.post("/traces/{trace_id}/events", response_model=Dict[str, Any])
async def add_stream_event(trace_id: str, data: StreamEventCreate):
    """Add a stream event for replay."""
    storage: Storage = get_storage_from_app()
    if not storage.traces:
        raise HTTPException(status_code=503, detail="Trace repository not available")
    
    if data.trace_id != trace_id:
        raise HTTPException(status_code=400, detail="trace_id mismatch")
    
    result = await storage.traces.add_stream_event(
        trace_id=trace_id,
        event_type=data.event_type,
        event_data=data.event_data,
        sequence_number=data.sequence_number,
        timestamp=data.timestamp
    )
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to add stream event")
    return result


@app.get("/traces/{trace_id}/events", response_model=List[Dict[str, Any]])
async def list_stream_events(trace_id: str, after_sequence: int = 0, limit: int = 1000):
    """List stream events for replay."""
    storage: Storage = get_storage_from_app()
    if not storage.traces:
        raise HTTPException(status_code=503, detail="Trace repository not available")
    
    return await storage.traces.list_stream_events(trace_id, after_sequence=after_sequence, limit=limit)


@app.post("/traces/batch", response_model=Dict[str, int])
async def batch_write_traces(data: BatchWriteRequest):
    """Write multiple trace items in a batch."""
    storage: Storage = get_storage_from_app()
    if not storage.traces:
        raise HTTPException(status_code=503, detail="Trace repository not available")
    
    items = [item.model_dump() for item in data.items]
    result = await storage.traces.batch_write(items)
    
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
