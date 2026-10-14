"""Direct-to-MinIO upload session endpoints (presigned PUT / multipart)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Request, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .config import settings
from .storage import get_storage

router = APIRouter(tags=["upload-sessions"])


class UploadSessionCreate(BaseModel):
    session_id: str = Field(..., min_length=8, max_length=128)
    file_id: str = Field(..., min_length=8, max_length=128)
    upload_length: int = Field(..., ge=0)
    relative_path: Optional[str] = None
    content_type: Optional[str] = None
    expires_seconds: int = Field(3600, ge=60, le=86400)


class MultipartPartEtag(BaseModel):
    part_number: int = Field(..., ge=1)
    etag: str = Field(..., min_length=1)


class UploadSessionComplete(BaseModel):
    session_id: str = Field(..., min_length=8, max_length=128)
    file_id: str = Field(..., min_length=8, max_length=128)
    expected_size: Optional[int] = Field(None, ge=0)
    upload_id: Optional[str] = None
    parts: Optional[List[MultipartPartEtag]] = None


def staging_object_key(user_id: str, session_id: str, file_id: str) -> str:
    uid = (user_id or "anon").replace("/", "_")
    sid = session_id.replace("/", "_")
    fid = file_id.replace("/", "_")
    return f"import_staging/{uid}/{sid}/{fid}"


def _multipart_enabled(upload_length: int) -> bool:
    threshold = int(getattr(settings, "UPLOAD_MULTIPART_THRESHOLD_BYTES", 0) or 0)
    return threshold > 0 and upload_length >= threshold


@router.post("/upload-sessions")
async def create_upload_session(body: UploadSessionCreate, request: Request):
    storage = await get_storage()
    user_id = (request.headers.get("X-User-Id") or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-Id required")
    if not storage.minio or not storage.minio.is_connected:
        raise HTTPException(status_code=503, detail="MinIO not available")

    object_name = staging_object_key(user_id, body.session_id, body.file_id)
    bucket = settings.MINIO_BUCKET
    storage_path = f"minio://{bucket}/{object_name}"

    if _multipart_enabled(body.upload_length):
        part_size = int(
            getattr(settings, "UPLOAD_MULTIPART_PART_SIZE_BYTES", 16 * 1024 * 1024)
            or (16 * 1024 * 1024)
        )
        part_size = max(part_size, 5 * 1024 * 1024)
        upload_id = await storage.minio.create_multipart_upload(
            object_name,
            content_type=body.content_type,
        )
        if not upload_id:
            raise HTTPException(status_code=500, detail="multipart_create_failed")
        n_parts = max(1, math.ceil(body.upload_length / part_size))
        # S3 hard limit 10000 parts
        if n_parts > 10000:
            raise HTTPException(status_code=400, detail="too_many_parts")
        parts_out: List[Dict[str, Any]] = []
        for pn in range(1, n_parts + 1):
            url = await storage.minio.get_presigned_upload_part_url(
                object_name,
                upload_id,
                pn,
                expires_seconds=body.expires_seconds,
            )
            if not url:
                await storage.minio.abort_multipart_upload(object_name, upload_id)
                raise HTTPException(status_code=500, detail="part_presign_failed")
            parts_out.append({"part_number": pn, "upload_url": url})
        existing = await storage.minio.list_multipart_parts(object_name, upload_id)
        return {
            "ok": True,
            "mode": "multipart",
            "method": "PUT",
            "upload_id": upload_id,
            "object_name": object_name,
            "storage_path": storage_path,
            "bucket": bucket,
            "upload_length": body.upload_length,
            "part_size": part_size,
            "parts": parts_out,
            "completed_parts": existing,
            "expires_in": body.expires_seconds,
            "relative_path": body.relative_path,
        }

    put_url = await storage.file_get_put_url(
        object_name,
        expires_seconds=body.expires_seconds,
        content_type=body.content_type,
    )
    if not put_url:
        raise HTTPException(status_code=500, detail="Failed to generate presigned PUT URL")

    return {
        "ok": True,
        "mode": "single",
        "method": "PUT",
        "upload_url": put_url,
        "object_name": object_name,
        "storage_path": storage_path,
        "bucket": bucket,
        "upload_length": body.upload_length,
        "expires_in": body.expires_seconds,
        "relative_path": body.relative_path,
    }


@router.post("/upload-sessions/complete")
async def complete_upload_session(body: UploadSessionComplete, request: Request):
    storage = await get_storage()
    user_id = (request.headers.get("X-User-Id") or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-Id required")
    if not storage.minio or not storage.minio.is_connected:
        raise HTTPException(status_code=503, detail="MinIO not available")

    object_name = staging_object_key(user_id, body.session_id, body.file_id)
    bucket = settings.MINIO_BUCKET

    if body.upload_id and body.parts:
        ok = await storage.minio.complete_multipart_upload(
            object_name,
            body.upload_id,
            [p.model_dump() for p in body.parts],
        )
        if not ok:
            raise HTTPException(status_code=400, detail="multipart_complete_failed")

    info = await storage.file_stat(object_name)
    if not info:
        raise HTTPException(status_code=404, detail="object_missing")
    size = int(info.get("size") or 0)
    if body.expected_size is not None and size != int(body.expected_size):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "size_mismatch",
                "expected": body.expected_size,
                "actual": size,
            },
        )
    return {
        "ok": True,
        "complete": True,
        "object_name": object_name,
        "storage_path": f"minio://{bucket}/{object_name}",
        "size": size,
        "etag": info.get("etag"),
        "content_type": info.get("content_type"),
        "mode": "multipart" if body.upload_id else "single",
    }


@router.post("/upload-sessions/abort")
async def abort_upload_session(request: Request, body: Dict[str, Any] = Body(...)):
    storage = await get_storage()
    user_id = (request.headers.get("X-User-Id") or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-Id required")
    if not storage.minio or not storage.minio.is_connected:
        raise HTTPException(status_code=503, detail="MinIO not available")
    session_id = str(body.get("session_id") or "")
    file_id = str(body.get("file_id") or "")
    upload_id = str(body.get("upload_id") or "")
    if not session_id or not file_id or not upload_id:
        raise HTTPException(status_code=400, detail="missing_fields")
    object_name = staging_object_key(user_id, session_id, file_id)
    ok = await storage.minio.abort_multipart_upload(object_name, upload_id)
    return {"ok": ok}


@router.post("/upload-sessions/copy-to-case")
async def copy_staging_object_to_case(request: Request, body: Dict[str, Any] = Body(...)):
    storage = await get_storage()
    user_id = (request.headers.get("X-User-Id") or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-Id required")
    if not storage.minio or not storage.minio.is_connected:
        raise HTTPException(status_code=503, detail="MinIO not available")

    session_id = str(body.get("session_id") or "")
    file_id = str(body.get("file_id") or "")
    case_id = str(body.get("case_id") or "")
    relative_path = str(body.get("relative_path") or "")
    if not session_id or not file_id or not case_id or not relative_path:
        raise HTTPException(status_code=400, detail="missing_fields")

    src = staging_object_key(user_id, session_id, file_id)
    rel = relative_path.replace("\\", "/").lstrip("/")
    if ".." in rel.split("/"):
        raise HTTPException(status_code=400, detail="invalid_path")
    dest = f"case_importer/{case_id}/{rel}"
    ok = await storage.file_copy(src, dest)
    if not ok:
        raise HTTPException(status_code=500, detail="copy_failed")
    bucket = settings.MINIO_BUCKET
    return {
        "ok": True,
        "storage_path": f"minio://{bucket}/{dest}",
        "object_name": dest,
    }


@router.get("/internal/minio/object")
async def get_minio_object_bytes(
    request: Request,
    object_name: str = Query(..., min_length=1),
):
    """Internal byte fetch for case-importer materialize (not for browsers)."""
    storage = await get_storage()
    if not storage.minio or not storage.minio.is_connected:
        raise HTTPException(status_code=503, detail="MinIO not available")
    if not (
        object_name.startswith("import_staging/")
        or object_name.startswith("case_importer/")
    ):
        raise HTTPException(status_code=403, detail="prefix_forbidden")
    data = await storage.file_download(object_name)
    if data is None:
        raise HTTPException(status_code=404, detail="object_missing")
    return Response(content=data, media_type="application/octet-stream")
