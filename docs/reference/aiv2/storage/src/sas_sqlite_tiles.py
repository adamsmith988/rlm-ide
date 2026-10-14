"""Read tiles from SAS.Planet SQLite3 cache (cache_sqlite/*.sqlitedb).

Path layout per SAS.Planet format 71:
  {root}/z{z+1}/{x/1024}/{y/1024}/{x/256}.{y/256}.sqlitedb

Table ``t`` stores global tile x,y (Web Mercator indices) and blob ``b``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Tuple

SasSqliteRole = Literal["satellite_imagery", "labels_overlay"]

_PNG = b"\x89PNG\r\n\x1a\n"
_JPEG = b"\xff\xd8\xff"


def sas_sqlite_db_path(base: Path, z: int, x: int, y: int) -> Path:
    return base / f"z{z + 1}" / str(x // 1024) / str(y // 1024) / f"{x // 256}.{y // 256}.sqlitedb"


def guess_image_media_type(data: bytes) -> str:
    if data.startswith(_PNG):
        return "image/png"
    if data.startswith(_JPEG):
        return "image/jpeg"
    return "application/octet-stream"


def _cache_dir() -> Path:
    """Writable cache for SAS sqlite DB copies when host mounts cannot read blobs.

    MAP_LAYERS_ROOT is often mounted read-only in k8s, so the legacy default under
    /data/map-layers/.sas-sqlite-cache fails with EROFS. Fall back to /tmp.
    """
    configured = (os.environ.get("SAS_SQLITE_CACHE_DIR") or "").strip()
    candidates = [configured, "/tmp/sas-sqlite-cache", "/var/tmp/sas-sqlite-cache"]
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw)
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except OSError:
            continue
    return Path("/tmp")


def _can_read_tile_table(db: Path) -> bool:
    """Host mounts (NTFS/exFAT) often open the DB but fail on blob reads from ``t``."""
    try:
        conn = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
        conn.execute("SELECT b FROM t LIMIT 1").fetchone()
        conn.close()
        return True
    except sqlite3.Error:
        return False


def sas_sqlite_zoom_range(base: Path) -> Tuple[Optional[int], Optional[int]]:
    """Infer zoom range from SAS folder layout ``z{N}`` (N = zoom + 1)."""
    if not base.is_dir():
        return None, None
    levels: list[int] = []
    for child in base.iterdir():
        if child.is_dir() and child.name.startswith("z") and child.name[1:].isdigit():
            levels.append(int(child.name[1:]) - 1)
    if not levels:
        return None, None
    return min(levels), max(levels)


def _readable_db_path(db: Path) -> Path:
    """Return a path SQLite can read tiles from (copy to PVC when host mount cannot mmap tables)."""
    if _can_read_tile_table(db):
        return db

    try:
        key = hashlib.sha1(str(db.resolve()).encode()).hexdigest()[:24]
        cached = _cache_dir() / f"{key}.sqlitedb"
        if not cached.is_file() or cached.stat().st_mtime < db.stat().st_mtime:
            shutil.copy2(db, cached)
        return cached
    except OSError:
        return db


@lru_cache(maxsize=512)
def _open_ro(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def sas_sqlite_tile_bytes(base: Path, z: int, x: int, y: int) -> Optional[bytes]:
    db = sas_sqlite_db_path(base, z, x, y)
    if not db.is_file():
        return None
    try:
        readable = _readable_db_path(db)
        conn = _open_ro(str(readable.resolve()))
        row = conn.execute(
            "SELECT b FROM t WHERE x = ? AND y = ? AND v = 0 LIMIT 1",
            (x, y),
        ).fetchone()
    except (sqlite3.Error, OSError):
        return None
    if not row or not row[0]:
        return None
    return bytes(row[0])


def sas_sqlite_tile_response(base: Path, z: int, x: int, y: int) -> Optional[Tuple[bytes, str]]:
    blob = sas_sqlite_tile_bytes(base, z, x, y)
    if not blob:
        return None
    return blob, guess_image_media_type(blob)


def looks_like_sas_sqlite_tree(path: Path) -> bool:
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if not child.is_dir() or not child.name.startswith("z") or not child.name[1:].isdigit():
            continue
        for db in child.rglob("*.sqlitedb"):
            return True
    return False


def infer_sas_sqlite_role(base: Path, max_samples: int = 12) -> Optional[SasSqliteRole]:
    """Classify SAS.Planet sqlite cache from sampled tile bytes — not folder names."""
    if not base.is_dir():
        return None

    samples: list[tuple[int, str]] = []
    for z_dir in sorted(base.iterdir()):
        if len(samples) >= max_samples:
            break
        if not z_dir.is_dir() or not z_dir.name.startswith("z") or not z_dir.name[1:].isdigit():
            continue
        z = int(z_dir.name[1:]) - 1
        for db in z_dir.rglob("*.sqlitedb"):
            if len(samples) >= max_samples:
                break
            try:
                readable = _readable_db_path(db)
                conn = _open_ro(str(readable.resolve()))
                rows = conn.execute("SELECT x, y FROM t LIMIT 4").fetchall()
            except sqlite3.Error:
                continue
            for x, y in rows:
                blob = sas_sqlite_tile_bytes(base, z, int(x), int(y))
                if not blob or len(blob) < 50:
                    continue
                samples.append((len(blob), guess_image_media_type(blob)))
                if len(samples) >= max_samples:
                    break

    if not samples:
        return None

    sizes = sorted(size for size, _ in samples)
    median = sizes[len(sizes) // 2]
    png_n = sum(1 for _, media in samples if media == "image/png")
    jpeg_n = len(samples) - png_n

    # Transparent label/road overlays are tiny PNG tiles.
    if png_n >= max(1, (len(samples) * 2) // 3) and median < 2000:
        return "labels_overlay"
    if jpeg_n >= len(samples) // 2 or median >= 1500:
        return "satellite_imagery"
    if png_n > jpeg_n:
        return "labels_overlay"
    return "satellite_imagery"


def infer_sas_sqlite_tile_ext(base: Path, role: Optional[SasSqliteRole] = None) -> str:
    """Infer tile file extension from a sample blob, with role-based fallback."""
    if base.is_dir():
        for z_dir in sorted(base.iterdir(), reverse=True):
            if not z_dir.is_dir() or not z_dir.name.startswith("z") or not z_dir.name[1:].isdigit():
                continue
            z = int(z_dir.name[1:]) - 1
            for db in z_dir.rglob("*.sqlitedb"):
                try:
                    readable = _readable_db_path(db)
                    conn = _open_ro(str(readable.resolve()))
                    row = conn.execute("SELECT x, y FROM t LIMIT 1").fetchone()
                except sqlite3.Error:
                    continue
                if not row:
                    continue
                blob = sas_sqlite_tile_bytes(base, z, int(row[0]), int(row[1]))
                if blob:
                    return "png" if blob.startswith(_PNG) else "jpg"
    if role == "labels_overlay":
        return "png"
    return "jpg"
