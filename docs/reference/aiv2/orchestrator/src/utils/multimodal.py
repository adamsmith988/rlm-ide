"""
Multimodal content preparation for Qwen3.5-VL.
Converts file_context (images/videos) to OpenAI content-parts format
so the LLM can directly understand visual content.
"""
import base64
import logging
import mimetypes
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..config import settings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory bytes cache (LRU, size-bounded, TTL-aware).
# ---------------------------------------------------------------------------
# Continue keeps image bytes in editor memory and re-uses them across turns.
# AIv2 used to re-download the same artifact from Storage on every chat turn
# (and again on every mid-loop multimodal injection), which cost 200–500ms
# per image per round. This cache fixes that.
#
# Keyed by ``artifact_id`` (preferred — stable, RBAC-safe) or storage_path as
# a fallback. Bounded by total bytes so a long-running orchestrator process
# never blows past ``ARTIFACT_BYTES_CACHE_MAX_MB``.

_ARTIFACT_BYTES_CACHE_MAX_BYTES = int(
    getattr(settings, "ARTIFACT_BYTES_CACHE_MAX_MB", 256)
) * 1024 * 1024
_ARTIFACT_BYTES_CACHE_TTL_S = float(
    getattr(settings, "ARTIFACT_BYTES_CACHE_TTL_S", 1800.0)
)


class _BytesCache:
    """Tiny thread-safe LRU keyed by stable identifier."""

    def __init__(self, max_total_bytes: int, ttl_s: float) -> None:
        self._max_total = max_total_bytes
        self._ttl_s = ttl_s
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, Tuple[bytes, float]]" = OrderedDict()
        self._total_bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[bytes]:
        if not key:
            return None
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            data, ts = entry
            if (time.monotonic() - ts) > self._ttl_s:
                self._data.pop(key, None)
                self._total_bytes -= len(data)
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return data

    def put(self, key: str, data: bytes) -> None:
        if not key or not data:
            return
        size = len(data)
        if size > self._max_total:
            return
        with self._lock:
            if key in self._data:
                old, _ = self._data.pop(key)
                self._total_bytes -= len(old)
            self._data[key] = (data, time.monotonic())
            self._total_bytes += size
            while self._total_bytes > self._max_total and self._data:
                _, (evicted, _) = self._data.popitem(last=False)
                self._total_bytes -= len(evicted)


_BYTES_CACHE = _BytesCache(_ARTIFACT_BYTES_CACHE_MAX_BYTES, _ARTIFACT_BYTES_CACHE_TTL_S)

SUPPORTED_IMAGE_MIMES = {
    "image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp",
}
SUPPORTED_VIDEO_MIMES = {
    "video/mp4", "video/webm", "video/mpeg",
}
MAX_IMAGE_SIZE_MB = 10
MAX_VIDEO_SIZE_MB = 50
# Must match upstream multimodal limits (typical image 2, video 1 on node deploys)
MAX_IMAGES_PER_LLM_MESSAGE = 2
MAX_VIDEOS_PER_LLM_MESSAGE = 1


def _storage_headers(auth_context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    headers = {
        "X-Internal-Key": settings.INTERNAL_API_KEY,
        "X-Service-Name": "orchestrator",
    }
    ctx = auth_context or {}
    if ctx.get("user_id"):
        headers["X-User-Id"] = str(ctx["user_id"])
    if ctx.get("group_id"):
        headers["X-Group-Id"] = str(ctx["group_id"])
    if ctx.get("case_id"):
        headers["X-Case-Id"] = str(ctx["case_id"])
    return headers


async def _read_local_file(path: str) -> Optional[bytes]:
    """Read file bytes from local filesystem (/data/uploads)."""
    try:
        p = Path(path)
        if not p.exists():
            log.warning(f"Local file not found: {path}")
            return None
        size_mb = p.stat().st_size / (1024 * 1024)
        if size_mb > MAX_VIDEO_SIZE_MB:
            log.warning(f"File too large for multimodal: {size_mb:.1f}MB > {MAX_VIDEO_SIZE_MB}MB")
            return None
        return p.read_bytes()
    except Exception as e:
        log.error(f"Failed to read local file {path}: {e}")
        return None


async def _download_by_artifact_id(
    artifact_id: str,
    auth_context: Optional[Dict[str, Any]] = None,
) -> Optional[bytes]:
    """Download file content directly via artifact ID."""
    storage_url = settings.STORAGE_URL
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            dl_url = f"{storage_url}/artifacts/{artifact_id}/download"
            dl_resp = await client.get(dl_url, headers=_storage_headers(auth_context))
            if dl_resp.status_code == 200:
                return dl_resp.content
            log.warning(f"Artifact download failed ({dl_resp.status_code}): {artifact_id}")
    except Exception as e:
        log.error(f"Failed to download artifact {artifact_id}: {e}")
    return None


async def get_file_bytes(
    file_path: str,
    artifact_id: Optional[str] = None,
    auth_context: Optional[Dict[str, Any]] = None,
) -> Optional[bytes]:
    """Get file content bytes — prefers artifact_id, falls back to local path.

    First checks the in-memory bytes cache (keyed by ``artifact_id`` when
    available, otherwise ``file_path``). Cache hits skip the Storage HTTP
    round-trip entirely — this is how Continue keeps multi-turn vision
    interactions snappy.
    """
    cache_key = artifact_id or (file_path or "")
    cached = _BYTES_CACHE.get(cache_key) if cache_key else None
    if cached is not None:
        log.debug("📷 artifact bytes cache hit (%d bytes) key=%s", len(cached), cache_key[:12])
        return cached

    if artifact_id:
        data = await _download_by_artifact_id(artifact_id, auth_context=auth_context)
        if data:
            _BYTES_CACHE.put(cache_key, data)
            return data
        log.warning(f"artifact_id download failed, trying path: {file_path}")
    if file_path.startswith("minio://"):
        log.warning(f"Cannot read minio:// path without artifact_id: {file_path}")
        return None
    data = await _read_local_file(file_path)
    if data and cache_key:
        _BYTES_CACHE.put(cache_key, data)
    return data


def artifact_bytes_cache_stats() -> Dict[str, Any]:
    """Expose cache hit/miss counters for telemetry."""
    return {
        "hits": _BYTES_CACHE.hits,
        "misses": _BYTES_CACHE.misses,
        "entries": len(_BYTES_CACHE._data),  # noqa: SLF001
        "total_bytes": _BYTES_CACHE._total_bytes,  # noqa: SLF001
        "max_bytes": _ARTIFACT_BYTES_CACHE_MAX_BYTES,
    }


def _file_to_data_url(content: bytes, mime_type: str) -> str:
    """Convert file bytes to data: URL for OpenAI API."""
    b64 = base64.b64encode(content).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


async def build_multimodal_content(
    text: str,
    file_context: Optional[Dict[str, Any]] = None,
    *,
    visual_embed_order: str = "newest_first",
    auth_context: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Build OpenAI content-parts format for multimodal messages.

    If file_context contains supported images/videos, returns a list
    of content parts. Otherwise returns the plain text string.

    visual_embed_order:
        "newest_first" — reverse artifact order so the latest uploads are embedded
        first (within vLLM image/video caps). "list_order" — use list order as-is.
    """
    if not file_context:
        return text

    all_files = file_context.get("all_files", [])
    raw_list = all_files if all_files else [file_context]
    if visual_embed_order == "newest_first":
        files_to_process = list(reversed(raw_list))
    else:
        files_to_process = raw_list

    def _is_visual_file(f: Dict[str, Any]) -> bool:
        f_path = f.get("path", "")
        f_mime = (f.get("mime_type") or "").lower()
        f_type = (f.get("type") or "").lower()
        if not f_mime:
            f_mime, _ = mimetypes.guess_type(f_path)
            f_mime = f_mime or ""
        return (
            f_type in ("image", "video")
            or f_mime in SUPPORTED_IMAGE_MIMES
            or f_mime in SUPPORTED_VIDEO_MIMES
        )

    # In mixed uploads the primary file may be text/document while a later
    # entry in all_files is an image/video. Check the full list before
    # returning plain text, otherwise visual attachments silently disappear.
    if not any(_is_visual_file(f) for f in files_to_process if isinstance(f, dict)):
        return text

    parts: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    images_added = 0
    videos_added = 0

    for f in files_to_process:
        f_path = f.get("path", "")
        f_mime = f.get("mime_type") or ""
        f_type = (f.get("type") or "").lower()

        if not f_mime:
            f_mime, _ = mimetypes.guess_type(f_path)
            f_mime = f_mime or ""

        f_is_image = f_type == "image" or f_mime in SUPPORTED_IMAGE_MIMES
        f_is_video = f_type == "video" or f_mime in SUPPORTED_VIDEO_MIMES

        if not (f_is_image or f_is_video):
            continue

        if f_is_image and images_added >= MAX_IMAGES_PER_LLM_MESSAGE:
            log.warning(
                "Skipping image for multimodal (vLLM limit %s images): %s",
                MAX_IMAGES_PER_LLM_MESSAGE,
                f.get("name", f_path),
            )
            continue
        if f_is_video and videos_added >= MAX_VIDEOS_PER_LLM_MESSAGE:
            log.warning(
                "Skipping video for multimodal (vLLM limit %s videos): %s",
                MAX_VIDEOS_PER_LLM_MESSAGE,
                f.get("name", f_path),
            )
            continue

        f_artifact_id = f.get("artifact_id") or f.get("id") or f.get("file_id")
        content = await get_file_bytes(
            f_path,
            artifact_id=f_artifact_id,
            auth_context=auth_context,
        )
        if not content:
            log.warning(f"Could not read file for multimodal: {f_path}")
            continue

        data_url = _file_to_data_url(content, f_mime)

        if f_is_image:
            parts.append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })
            images_added += 1
            log.info(f"Added image to multimodal content: {f.get('name', f_path)}")
        elif f_is_video:
            parts.append({
                "type": "video_url",
                "video_url": {"url": data_url}
            })
            videos_added += 1
            log.info(f"Added video to multimodal content: {f.get('name', f_path)}")

    if len(parts) == 1:
        return text

    return parts
