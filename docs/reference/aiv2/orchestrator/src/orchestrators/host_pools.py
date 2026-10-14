"""Generic pool substrate helpers for orchestrator hosts.

Speaks only inventory rows + on-disk ``pool/`` layout. Callers (chat session,
Agent Flow, jobs, …) normalize their own sources into inventory first — this
module must not import chat-session or agent-flow types.

Bytes live once in a shared artifact cache (keyed by ``artifact_id``). Session
``pool/files/`` entries are hardlinks (or a single copy) into that cache so
reattaching the same case/file does not re-download or fill the disk.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

POOL_DIR = "pool"
_SAFE_FN = re.compile(r"[^A-Za-z0-9._\-+=@]+")
_SAFE_AID = re.compile(r"[^A-Za-z0-9._\-]+")

DownloadFn = Callable[[str, str], Awaitable[Optional[bytes]]]
# (artifact_id, filename_hint) -> bytes | None


def safe_pool_filename(name: str, *, fallback: str = "file") -> str:
    raw = str(name or "").strip() or fallback
    base = os.path.basename(raw.replace("\\", "/")) or fallback
    cleaned = _SAFE_FN.sub("_", base).strip("._") or fallback
    return cleaned[:180]


def artifact_cache_root() -> str:
    """Shared content cache (one blob per artifact_id across all sessions)."""
    env = str(os.environ.get("AIV2_ARTIFACT_CACHE") or "").strip()
    if env:
        return env
    for cand in (
        "/data/tool_workspaces/_artifact_cache",
        "/tmp/aiv2_artifact_cache",
    ):
        parent = os.path.dirname(cand.rstrip("/")) or "/"
        if os.path.isdir(parent) or cand.startswith("/tmp/"):
            return cand
    return "/tmp/aiv2_artifact_cache"


def artifact_cache_path(artifact_id: str, filename: str = "") -> str:
    aid = _SAFE_AID.sub("_", str(artifact_id or "").strip())[:120] or "unknown"
    fn = safe_pool_filename(filename or "blob")
    return os.path.join(artifact_cache_root(), aid[:2] or "_", aid, fn)


def _is_nonempty_file(path: str) -> bool:
    try:
        return bool(path) and os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _same_realpath(a: str, b: str) -> bool:
    try:
        return os.path.realpath(a) == os.path.realpath(b)
    except OSError:
        return False


def link_or_copy(src: str, dest: str) -> str:
    """Prefer hardlink into the session pool; fall back to copy once."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if _same_realpath(src, dest):
        return "same"
    if os.path.lexists(dest):
        try:
            if _is_nonempty_file(dest) and _same_realpath(src, dest):
                return "same"
            os.unlink(dest)
        except OSError:
            pass
    try:
        os.link(src, dest)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dest)
        return "copy"


def empty_inventory(*, note: str = "") -> Dict[str, Any]:
    return {
        "ok": True,
        "files": [],
        "texts": [],
        "case_ids": [],
        "note": note,
        "count": 0,
    }


def write_pool_index(cwd: str, inventory: Mapping[str, Any] | None) -> str:
    """Write ``pool/INDEX.md`` listing files already (or about to be) on disk."""
    inv = inventory if isinstance(inventory, Mapping) else {}
    files = [r for r in (inv.get("files") or []) if isinstance(r, Mapping)]
    texts = [r for r in (inv.get("texts") or []) if isinstance(r, Mapping)]
    note = str(inv.get("note") or "")

    root = os.path.join(str(cwd or ""), POOL_DIR)
    os.makedirs(os.path.join(root, "files"), exist_ok=True)
    inputs_dir = os.path.join(root, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    for row in texts:
        fn = safe_pool_filename(str(row.get("filename") or "input.txt"))
        path = os.path.join(inputs_dir, fn)
        body = str(row.get("content") or "")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body if body.endswith("\n") else body + "\n")
        row["filename"] = fn
        row["path"] = path
        row["relpath"] = os.path.join(POOL_DIR, "inputs", fn)

    lines = [
        "# Data pool",
        "",
        "Files under `pool/files/`; texts under `pool/inputs/`.",
        "Use `ls pool`, `open(...)`, or host `pools.<name>.ls()` / `.resolve_file(...)`.",
        "",
        "## Files",
    ]
    if not files and not texts:
        lines.append("_(empty)_")
    for row in files:
        fn = str(row.get("filename") or row.get("original_filename") or "?")
        aid = str(row.get("artifact_id") or "").strip()
        suffix = f" (`{aid[:8]}…`)" if aid else ""
        lines.append(f"- `{fn}`{suffix}")
    if texts:
        lines.extend(["", "## Texts"])
        for row in texts:
            lines.append(f"- `{row.get('filename') or 'input.txt'}`")
    if note:
        lines.extend(["", f"_note: {note}_"])
    lines.append("")

    index_path = os.path.join(root, "INDEX.md")
    with open(index_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return index_path


def _bind_row(row: Dict[str, Any], *, fn: str, dest: str, rel: str) -> None:
    row["filename"] = fn
    row["path"] = dest
    row["relpath"] = rel


async def ensure_artifact_local(
    *,
    artifact_id: str,
    filename: str,
    dest: str,
    download: Optional[DownloadFn] = None,
    src_path: str = "",
) -> Dict[str, Any]:
    """Ensure ``dest`` points at bytes for ``artifact_id`` (cache → hardlink)."""
    aid = str(artifact_id or "").strip()
    fn = safe_pool_filename(filename or "file")
    cache = artifact_cache_path(aid, fn) if aid else ""

    if _is_nonempty_file(dest):
        return {"ok": True, "path": dest, "via": "session_hit"}

    if cache and _is_nonempty_file(cache):
        via = link_or_copy(cache, dest)
        return {"ok": True, "path": dest, "via": f"cache_{via}"}

    src = str(src_path or "").strip()
    if src and _is_nonempty_file(src):
        if cache:
            os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
            if not _is_nonempty_file(cache):
                link_or_copy(src, cache)
            via = link_or_copy(cache if _is_nonempty_file(cache) else src, dest)
        else:
            via = link_or_copy(src, dest)
        return {"ok": True, "path": dest, "via": f"src_{via}"}

    if aid and download is not None:
        data = await download(aid, fn)
        if data:
            target = cache or dest
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            tmp = target + ".partial"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, target)
            if cache and target == cache:
                link_or_copy(cache, dest)
            return {"ok": True, "path": dest, "via": "download"}

    return {"ok": False, "error": "missing", "path": dest}


async def materialize_pool_files(
    cwd: str,
    inventory: Mapping[str, Any] | None,
    *,
    download: Optional[DownloadFn] = None,
) -> Dict[str, Any]:
    """Bind inventory artifacts under ``pool/files/`` via shared cache.

    Never re-downloads when the session file or global cache already has bytes.
    Mutates inventory file rows with resolved ``path`` / ``filename``.
    """
    inv = inventory if isinstance(inventory, Mapping) else {}
    files = [r for r in (inv.get("files") or []) if isinstance(r, dict)]
    dest_dir = os.path.join(str(cwd or ""), POOL_DIR, "files")
    os.makedirs(dest_dir, exist_ok=True)
    ok_n = 0
    fail_n = 0
    skipped = 0
    cached = 0
    downloaded = 0
    errors: List[str] = []

    for row in files:
        fn = safe_pool_filename(
            str(row.get("filename") or row.get("original_filename") or row.get("name") or "file")
        )
        dest = os.path.join(dest_dir, fn)
        rel = os.path.join(POOL_DIR, "files", fn)
        aid = str(row.get("artifact_id") or "").strip()
        src = str(row.get("path") or "").strip()
        try:
            result = await ensure_artifact_local(
                artifact_id=aid,
                filename=fn,
                dest=dest,
                download=download,
                src_path=src if (src and not _same_realpath(src, dest)) else "",
            )
            if result.get("ok"):
                _bind_row(row, fn=fn, dest=dest, rel=rel)
                ok_n += 1
                via = str(result.get("via") or "")
                if via == "session_hit":
                    skipped += 1
                elif via.startswith("cache_"):
                    cached += 1
                elif via == "download":
                    downloaded += 1
                continue
            fail_n += 1
            errors.append(f"{fn}:missing")
        except Exception as exc:  # noqa: BLE001 — host materialize best-effort
            fail_n += 1
            errors.append(f"{fn}:{type(exc).__name__}")

    write_pool_index(cwd, inv)
    stats = {
        "materialized": ok_n,
        "failed": fail_n,
        "skipped": skipped,
        "cache_hits": cached,
        "downloaded": downloaded,
        "errors": errors,
    }
    if downloaded or fail_n:
        logger.info(
            "pool materialize cwd=%s ok=%s skip=%s cache=%s download=%s fail=%s",
            cwd,
            ok_n,
            skipped,
            cached,
            downloaded,
            fail_n,
        )
    return stats


def inventory_ls_payload(inventory: Mapping[str, Any] | None) -> Dict[str, Any]:
    inv = inventory if isinstance(inventory, Mapping) else {}
    files_out: List[Dict[str, Any]] = []
    for row in inv.get("files") or []:
        if not isinstance(row, Mapping):
            continue
        files_out.append(
            {
                "filename": row.get("filename") or row.get("original_filename"),
                "artifact_id": row.get("artifact_id"),
                "path": row.get("path"),
                "relpath": row.get("relpath"),
                "mime_type": row.get("mime_type"),
                "size_bytes": row.get("size_bytes") or row.get("size"),
            }
        )
    for row in inv.get("texts") or []:
        if not isinstance(row, Mapping):
            continue
        files_out.append(
            {
                "filename": row.get("filename"),
                "path": row.get("path"),
                "relpath": row.get("relpath"),
                "kind": "text",
            }
        )
    return {"ok": True, "files": files_out, "count": len(files_out)}


def lookup_inventory_row(
    inventory: Mapping[str, Any] | None,
    locator: str,
) -> Optional[Dict[str, Any]]:
    inv = inventory if isinstance(inventory, Mapping) else {}
    loc = str(locator or "").strip().replace("\\", "/")
    if not loc:
        return None
    loc_base = os.path.basename(loc.rstrip("/")) or loc
    for row in list(inv.get("files") or []) + list(inv.get("texts") or []):
        if not isinstance(row, Mapping):
            continue
        names = {
            str(row.get("filename") or ""),
            str(row.get("original_filename") or ""),
            str(row.get("relpath") or "").replace("\\", "/"),
            str(row.get("path") or "").replace("\\", "/"),
        }
        names.discard("")
        base = os.path.basename(str(row.get("path") or row.get("relpath") or ""))
        if loc in names or loc_base in names or loc_base == base:
            return dict(row)
        aid = str(row.get("artifact_id") or "").strip()
        if aid and (loc == aid or loc.endswith(aid)):
            return dict(row)
    return None


def make_pool_host_closures(
    *,
    artifact_dir: str,
    pool_defs: Sequence[Mapping[str, Any]],
    union_inventory: Mapping[str, Any],
    download: Optional[DownloadFn] = None,
) -> Dict[str, Any]:
    """Build HostPorts-compatible pool_* async closures from prepared inventories."""
    by_key: Dict[str, Dict[str, Any]] = {}
    for entry in pool_defs:
        if not isinstance(entry, Mapping):
            continue
        key = str(entry.get("pool_key") or entry.get("name") or "").strip()
        if key:
            by_key[key] = dict(entry)

    def _inv_for(pool_key: str) -> Dict[str, Any]:
        entry = by_key.get(str(pool_key or "").strip())
        if entry and isinstance(entry.get("inventory"), dict):
            return entry["inventory"]
        return dict(union_inventory)

    async def pool_list(pool_key: str) -> Dict[str, Any]:
        return inventory_ls_payload(_inv_for(pool_key))

    async def pool_resolve(locator: str, pool_key: str = "") -> Dict[str, Any]:
        loc = str(locator or "").strip().replace("\\", "/")
        loc_base = os.path.basename(loc.rstrip("/")) or loc
        if loc_base.upper() == "INDEX.MD" or loc.endswith("pool/INDEX.md"):
            index_path = os.path.join(artifact_dir, POOL_DIR, "INDEX.md")
            if os.path.isfile(index_path):
                return {
                    "path": index_path,
                    "filename": "INDEX.md",
                    "artifact_id": None,
                    "kind": "text",
                }
        row = lookup_inventory_row(_inv_for(pool_key), locator)
        if not row:
            return {"ok": False, "error": "not_found", "locator": locator}
        path = str(row.get("path") or "").strip()
        if path and os.path.isfile(path):
            return {
                "ok": True,
                "path": path,
                "filename": row.get("filename") or loc_base,
                "artifact_id": row.get("artifact_id"),
                "kind": row.get("kind") or "file",
            }
        # Lazy bind from shared cache / storage when INDEX listed the file
        # but the session hardlink was never created (or was wiped).
        aid = str(row.get("artifact_id") or "").strip()
        fn = safe_pool_filename(
            str(row.get("filename") or row.get("original_filename") or loc_base)
        )
        dest = path or os.path.join(artifact_dir, POOL_DIR, "files", fn)
        if aid or _is_nonempty_file(str(row.get("path") or "")):
            ensured = await ensure_artifact_local(
                artifact_id=aid,
                filename=fn,
                dest=dest,
                download=download,
                src_path=str(row.get("path") or ""),
            )
            if ensured.get("ok"):
                row["path"] = dest
                row["filename"] = fn
                return {
                    "ok": True,
                    "path": dest,
                    "filename": fn,
                    "artifact_id": aid or None,
                    "kind": row.get("kind") or "file",
                }
        return {
            "ok": False,
            "error": "path_missing",
            "locator": locator,
            "artifact_id": row.get("artifact_id"),
        }

    async def pool_search(pool_key: str, query: str) -> Dict[str, Any]:
        q = str(query or "").strip().lower()
        payload = inventory_ls_payload(_inv_for(pool_key))
        files = payload.get("files") or []
        if not q:
            return {"ok": True, "results": files}
        results = []
        for f in files:
            if not isinstance(f, dict):
                continue
            blob = " ".join(
                str(f.get(k) or "")
                for k in ("filename", "artifact_id", "path", "relpath")
            ).lower()
            if q in blob:
                results.append(f)
        return {"ok": True, "results": results, "query": query}

    async def pool_read(
        locator: str = "",
        pool_key: str = "",
        max_bytes: int = 200_000,
        **_kwargs: Any,
    ) -> Dict[str, Any]:
        resolved = await pool_resolve(locator, pool_key=pool_key)
        path = str(resolved.get("path") or "").strip()
        if path and not resolved.get("ok", True) and resolved.get("error"):
            return {"ok": False, "error": resolved.get("error") or "not_found"}
        if not path or not os.path.isfile(path):
            return {"ok": False, "error": resolved.get("error") or "not_found"}
        try:
            with open(path, "rb") as fh:
                data = fh.read(max(1, int(max_bytes)))
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("utf-8", errors="replace")
            return {
                "ok": True,
                "path": path,
                "filename": resolved.get("filename"),
                "text": text,
                "bytes": len(data),
            }
        except OSError as exc:
            return {"ok": False, "error": str(exc)}

    return {
        "pool_list": pool_list,
        "pool_resolve": pool_resolve,
        "pool_search": pool_search,
        "pool_read": pool_read,
    }
