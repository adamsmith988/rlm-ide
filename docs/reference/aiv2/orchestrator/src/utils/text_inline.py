"""
Text-file inlining for the LLM prompt.

Mirrors `multimodal.py` for visual files: small text / code / markdown / json /
csv / log files attached to the session are read once and embedded as a
synthesised "## File Content" system block so the model can reason about their
content directly without paying a tool-call round-trip.

Resolution order for each file:
  1. `metadata.text_extract.extracted_text` (storage service auto-extract).
  2. Local filesystem (`/data/uploads/...`).
  3. Storage `/artifacts/{id}/download` (MinIO / remote).

All limits are configurable via :class:`Settings`:
  * INLINE_TEXT_FILE_THRESHOLD_BYTES  – per-file size cap.
  * INLINE_TEXT_PER_FILE_CHARS        – per-file character cap in the prompt.
  * INLINE_TEXT_TOTAL_CHARS           – total character cap across all files.
  * INLINE_TEXT_MAX_FILES             – max number of files to inline per turn.

The helper is a no-op when the feature flag is disabled, when the session has
no textual files, or when none of them yield any content (e.g. all binary).
"""
from __future__ import annotations

import json
import logging
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..config import settings

log = logging.getLogger(__name__)

# Extensions we treat as "textual" even if MIME type is missing or generic
# (mimetypes.guess_type misses many of these on stripped-down containers).
TEXT_EXTENSIONS: set = {
    ".txt", ".md", ".markdown", ".rst", ".log",
    ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xml", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".env",
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

# Treat these "non-text/" mimes as text too.
EXTRA_TEXT_MIMES: set = {
    "application/json",
    "application/xml",
    "application/x-yaml",
    "application/yaml",
    "application/x-toml",
    "application/x-sh",
    "application/javascript",
    "application/typescript",
}


def _is_text_file(name: str, mime_type: str) -> bool:
    """Best-effort heuristic to decide if a file is textual."""
    mt = (mime_type or "").lower()
    if mt.startswith("text/"):
        return True
    if mt in EXTRA_TEXT_MIMES:
        return True
    if name:
        ext = Path(name).suffix.lower()
        if ext and ext in TEXT_EXTENSIONS:
            return True
    return False


def _decode(data: bytes) -> Tuple[str, str]:
    """Decode bytes → (text, encoding). Falls back to utf-8/replace."""
    try:
        import chardet  # type: ignore
        detected = chardet.detect(data[:10_000]).get("encoding") or "utf-8"
    except Exception:
        detected = "utf-8"
    try:
        return data.decode(detected), detected
    except Exception:
        return data.decode("utf-8", errors="replace"), "utf-8"


def _truncate(text: str, max_chars: int) -> Tuple[str, bool]:
    """Truncate to max_chars, returning (text, was_truncated)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


async def _fetch_artifact(client: httpx.AsyncClient, artifact_id: str) -> Optional[Dict[str, Any]]:
    """Fetch artifact metadata (including text_extract if any)."""
    try:
        resp = await client.get(f"{settings.STORAGE_URL}/artifacts/{artifact_id}")
        if resp.status_code == 200:
            return resp.json()
        log.debug("artifact fetch %s -> %s", artifact_id, resp.status_code)
    except Exception as exc:
        log.debug("artifact fetch failed %s: %s", artifact_id, exc)
    return None


async def _download_artifact(client: httpx.AsyncClient, artifact_id: str) -> Optional[bytes]:
    try:
        resp = await client.get(f"{settings.STORAGE_URL}/artifacts/{artifact_id}/download")
        if resp.status_code == 200:
            return resp.content
        log.debug("artifact download %s -> %s", artifact_id, resp.status_code)
    except Exception as exc:
        log.debug("artifact download failed %s: %s", artifact_id, exc)
    return None


def _read_local(path: str, byte_cap: int) -> Optional[bytes]:
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        size = p.stat().st_size
        if size > byte_cap * 4:
            # File is much larger than the inlining cap; reading the first
            # `byte_cap` bytes is enough — we'll truncate to per-file char cap.
            with p.open("rb") as fh:
                return fh.read(byte_cap * 4)
        return p.read_bytes()
    except Exception as exc:
        log.debug("local read failed %s: %s", path, exc)
        return None


def _format_block(items: List[Dict[str, Any]]) -> str:
    """Render the synthesised system block."""
    parts = [
        "## File Content",
        "The following text/code/document files attached to this session have "
        "been pre-loaded for you. Each block contains the actual file content. "
        "Reason about them directly — you do NOT need to call `text.read` / "
        "`session_read` again unless content is truncated and you need more.",
        "",
    ]

    for it in items:
        name = it.get("name") or "<unknown>"
        ftype = it.get("type") or "text"
        chars = it.get("chars", 0)
        total_chars = it.get("total_chars", chars)
        truncated = it.get("truncated", False)
        encoding = it.get("encoding") or "utf-8"
        source = it.get("source") or "disk"
        fence_lang = it.get("fence_lang") or ""

        head = f"### `{name}` — {ftype} | {chars}/{total_chars} chars | {encoding} | source={source}"
        if truncated:
            head += " | TRUNCATED — call `text.read(filename=\"" + str(name) + "\", start_line=…, end_line=…)` for more"
        parts.append(head)
        parts.append(f"```{fence_lang}")
        parts.append(it.get("content", ""))
        parts.append("```")
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


_FENCE_LANG_BY_EXT: Dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "jsx",
    ".ts": "typescript", ".tsx": "tsx",
    ".json": "json", ".jsonl": "json", ".ndjson": "json",
    ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml",
    ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "scss", ".sass": "sass", ".less": "less",
    ".md": "markdown", ".markdown": "markdown",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".sql": "sql", ".graphql": "graphql", ".gql": "graphql",
    ".java": "java", ".kt": "kotlin", ".scala": "scala",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
    ".cs": "csharp", ".go": "go", ".rs": "rust",
    ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".xml": "xml", ".csv": "csv", ".tsv": "tsv",
    ".diff": "diff", ".patch": "diff",
    ".dockerfile": "dockerfile",
    ".vue": "vue", ".svelte": "svelte", ".astro": "astro",
    ".lua": "lua", ".r": "r", ".jl": "julia",
}


def _fence_lang(name: str) -> str:
    if not name:
        return ""
    return _FENCE_LANG_BY_EXT.get(Path(name).suffix.lower(), "")


def _coerce_metadata(md: Any) -> Dict[str, Any]:
    """Storage returns ``metadata`` as a JSON-serialised string (psycopg jsonb
    fallback). Tolerate both shapes so callers can do dict-lookups freely."""
    if isinstance(md, dict):
        return md
    if isinstance(md, (bytes, bytearray)):
        try:
            md = md.decode("utf-8")
        except Exception:
            return {}
    if isinstance(md, str):
        s = md.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_meta_text(artifact: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """If the storage already auto-extracted text, return its payload."""
    md = _coerce_metadata(artifact.get("metadata"))
    te = md.get("text_extract")
    if isinstance(te, dict):
        text = te.get("extracted_text")
        if isinstance(text, str) and text:
            return te
    return None


async def build_text_files_inline_block(
    file_context: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Inline small text/code/markdown files into a system block.

    Returns a tuple ``(block, inlined_metas)`` where:
      * ``block`` is the synthesised markdown block (or ``None`` when nothing
        could be inlined).
      * ``inlined_metas`` is a list of ``{"name", "artifact_id", "chars",
        "truncated", "source"}`` records — useful for the workspace registry
        so the LLM is told the files are already inlined.

    The caller is responsible for inserting ``block`` near the top of the
    system messages (after the main system prompt and before the user's
    message), exactly the way image/video data: URLs are inserted today.
    """
    if not getattr(settings, "INLINE_TEXT_FILES_ENABLED", True):
        return None, []
    if not file_context:
        return None, []

    threshold = int(getattr(settings, "INLINE_TEXT_FILE_THRESHOLD_BYTES", 32_768))
    per_file_chars = int(getattr(settings, "INLINE_TEXT_PER_FILE_CHARS", 8_000))
    total_chars = int(getattr(settings, "INLINE_TEXT_TOTAL_CHARS", 32_000))
    max_files = int(getattr(settings, "INLINE_TEXT_MAX_FILES", 6))

    if threshold <= 0 or per_file_chars <= 0 or total_chars <= 0 or max_files <= 0:
        return None, []

    raw_files = file_context.get("all_files") or [file_context]
    candidates: List[Dict[str, Any]] = []
    for f in raw_files:
        if not isinstance(f, dict):
            continue
        name = f.get("name") or f.get("filename") or ""
        mime = f.get("mime_type") or ""
        if not mime and name:
            mime, _ = mimetypes.guess_type(name)
            mime = mime or ""
        if not _is_text_file(name, mime):
            continue
        candidates.append({
            "name": name,
            "mime_type": mime,
            "path": f.get("path") or f.get("file_path") or "",
            "artifact_id": f.get("artifact_id") or f.get("id"),
            "type": (f.get("type") or "document"),
        })

    if not candidates:
        return None, []

    # Honour total ordering: keep the **earliest** uploads first so the prompt
    # remains stable across turns. (Reversing would break prompt-cache reuse.)
    candidates = candidates[:max_files]

    items: List[Dict[str, Any]] = []
    inlined_metas: List[Dict[str, Any]] = []
    chars_used = 0
    timeout = httpx.Timeout(connect=2.0, read=8.0, write=8.0, pool=8.0)

    storage_headers = {
        "X-Internal-Key": settings.INTERNAL_API_KEY,
        "X-Service-Name": settings.SERVICE_NAME,
    }
    async with httpx.AsyncClient(timeout=timeout, headers=storage_headers) as client:
        for cand in candidates:
            if chars_used >= total_chars:
                log.debug("text-inline: total cap reached, stopping")
                break

            name = cand["name"]
            artifact_id = cand.get("artifact_id")
            content_bytes: Optional[bytes] = None
            text: Optional[str] = None
            encoding = "utf-8"
            source = "unknown"
            total_chars_file = 0

            # 1. Storage auto-extracted text (no I/O).
            if artifact_id:
                artifact = await _fetch_artifact(client, str(artifact_id))
                if artifact is not None:
                    # Honour storage's recorded size — bail early on huge files.
                    size_bytes = (
                        artifact.get("file_size_bytes")
                        or artifact.get("size_bytes")
                        or 0
                    )
                    if size_bytes and size_bytes > threshold * 16:
                        log.debug(
                            "text-inline: skipping %s (%d bytes > %d × 16)",
                            name, size_bytes, threshold,
                        )
                        continue
                    meta_payload = _extract_meta_text(artifact)
                    if meta_payload is not None:
                        text = meta_payload.get("extracted_text") or ""
                        encoding = meta_payload.get("encoding") or "utf-8"
                        source = "metadata.text_extract"
                        total_chars_file = (
                            meta_payload.get("total_chars")
                            or len(text)
                        )

            # 2. Local filesystem.
            if text is None:
                local_path = cand.get("path") or ""
                if local_path and not local_path.startswith("minio://"):
                    raw = _read_local(local_path, threshold)
                    if raw is not None:
                        if len(raw) > threshold:
                            # Truncate at byte level then decode.
                            raw = raw[:threshold]
                        content_bytes = raw

            # 3. Storage download.
            if text is None and content_bytes is None and artifact_id:
                raw = await _download_artifact(client, str(artifact_id))
                if raw is not None and len(raw) <= threshold * 4:
                    content_bytes = raw[:threshold] if len(raw) > threshold else raw

            if text is None and content_bytes is not None:
                text, encoding = _decode(content_bytes)
                source = "storage_download" if artifact_id else "disk"
                total_chars_file = len(text)

            if not text:
                continue

            # Per-file truncation
            remaining_total = total_chars - chars_used
            slot = min(per_file_chars, max(0, remaining_total))
            if slot <= 0:
                break
            snippet, truncated = _truncate(text, slot)

            items.append({
                "name": name or f"file_{artifact_id or '?'}",
                "type": cand["type"],
                "content": snippet,
                "chars": len(snippet),
                "total_chars": total_chars_file or len(text),
                "truncated": truncated,
                "encoding": encoding,
                "source": source,
                "fence_lang": _fence_lang(name),
            })
            inlined_metas.append({
                "name": name,
                "artifact_id": artifact_id,
                "chars": len(snippet),
                "total_chars": total_chars_file or len(text),
                "truncated": truncated,
                "source": source,
            })
            chars_used += len(snippet)

    if not items:
        return None, []

    block = _format_block(items)
    log.info(
        "📝 text-inline: embedded %d file(s) (%d chars total) into system prompt",
        len(items), chars_used,
    )
    return block, inlined_metas


__all__ = ["build_text_files_inline_block", "TEXT_EXTENSIONS"]
