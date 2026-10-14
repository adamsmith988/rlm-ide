"""Local session artifact persist + optional output_intents (plugin-local)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Kernel cwd internals stay on local disk — never uploaded, never Direct Reply.
TERMINAL_LOG_NAME = "terminal.jsonl"

_SESSION_INTERNAL_NAMES = frozenset(
    {
        "meta.json",
        "cfg.json",
        "transcript.jsonl",
        "transcript.full.jsonl",
        TERMINAL_LOG_NAME,
        "harness_state.json",
        "kernel-state.dill",
        "kernel-state.dill.json",
        "kernel-snapshot.marker",
    }
)
# Inputs / runtime internals — not agent-produced deliverables.
# ``children`` is scanned via each child session's own artifact_dir.
# ``pool`` is skipped in the general walk; agent *outputs* under ``pool/files/``
# are merged separately against a post-materialize baseline (see
# :func:`snapshot_pool_files` / :func:`write_pool_files_baseline`).
_SKIP_DIR_NAMES = frozenset({"harness", "children", "__pycache__", "case", "pool"})
_POOL_FILES_REL = "pool/files"
_POOL_BASELINE_REL = "pool/.deliverable_baseline.json"
# Soft safety valves only (env-tunable). Agent-written files must not be dropped
# casually — defaults are intentionally high.
_MAX_DELIVERABLE_BYTES = max(
    1,
    int(
        os.environ.get("AIV2_RLM_MAX_DELIVERABLE_BYTES", str(1024 * 1024 * 1024))
        or str(1024 * 1024 * 1024)
    ),
)
_MAX_DELIVERABLE_FILES = max(
    1,
    int(os.environ.get("AIV2_RLM_MAX_DELIVERABLE_FILES", "10000") or "10000"),
)

# kind, mime_type, media_type
_DELIVERABLE_EXT: Dict[str, Tuple[str, str, str]] = {
    ".html": ("html", "text/html", "document"),
    ".htm": ("html", "text/html", "document"),
    ".png": ("image", "image/png", "image"),
    ".jpg": ("image", "image/jpeg", "image"),
    ".jpeg": ("image", "image/jpeg", "image"),
    ".webp": ("image", "image/webp", "image"),
    ".gif": ("image", "image/gif", "image"),
    ".svg": ("image", "image/svg+xml", "image"),
    ".pdf": ("document", "application/pdf", "document"),
    ".csv": ("document", "text/csv", "document"),
    ".md": ("document", "text/markdown", "document"),
    ".txt": ("document", "text/plain", "document"),
    ".json": ("json", "application/json", "document"),
    ".jsonl": ("json", "application/jsonl", "document"),
    ".yaml": ("document", "text/yaml", "document"),
    ".yml": ("document", "text/yaml", "document"),
    ".xml": ("document", "application/xml", "document"),
    ".zip": ("archive", "application/zip", "document"),
    ".tar": ("archive", "application/x-tar", "document"),
    ".gz": ("archive", "application/gzip", "document"),
    ".docx": (
        "document",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "document",
    ),
    ".xlsx": (
        "document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "document",
    ),
    ".pptx": (
        "document",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "document",
    ),
    ".glb": ("mesh", "model/gltf-binary", "model"),
    ".gltf": ("mesh", "model/gltf+json", "model"),
    ".mp3": ("audio", "audio/mpeg", "audio"),
    ".wav": ("audio", "audio/wav", "audio"),
    ".ogg": ("audio", "audio/ogg", "audio"),
    ".mp4": ("video", "video/mp4", "video"),
    ".webm": ("video", "video/webm", "video"),
    ".js": ("file", "text/javascript", "document"),
    ".css": ("file", "text/css", "document"),
    ".py": ("file", "text/x-python", "document"),
    ".ts": ("file", "text/typescript", "document"),
    ".tsx": ("file", "text/tsx", "document"),
    ".sh": ("file", "text/x-shellscript", "document"),
}
_UNKNOWN_DELIVERABLE = ("file", "application/octet-stream", "document")

# root -> (dir signature, candidate records incl. empty files). Re-walk only
# when a directory mtime changed; sizes are re-stat'ed on hit so in-place
# writes still surface.
_LISTING_CACHE: Dict[
    str, Tuple[Tuple[Tuple[str, int], ...], List[Dict[str, Any]], float]
] = {}
_LISTING_CACHE_MAX = 256
_MTIME_SETTLE_SEC = 0.05
_TEXT_CACHE: Dict[str, Tuple[int, int, str]] = {}
_TEXT_CACHE_MAX_BYTES = 64 * 1024 * 1024
_TEXT_CACHE_BYTES = 0
_CACHE_LOCK = threading.Lock()
# Loop-thread and background persist writers must not interleave on one file.
_WRITE_LOCK = threading.Lock()
# Canonical workspace manifests are shared by parent/child sessions.  Each
# manifest owns discovery and upload state for one real directory.
_MANIFESTS: Dict[str, Dict[str, Any]] = {}
_MANIFEST_LOCK = threading.RLock()
_RESERVATION_TTL_SEC = max(
    5.0, float(os.environ.get("AIV2_RLM_PERSIST_RESERVATION_TTL_SEC", "300") or "300")
)


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def slim_messages(
    messages: Any, *, limit: int = 40, content_chars: int = 800
) -> List[Dict[str, Any]]:
    rows = [m for m in (messages or []) if isinstance(m, dict)]
    out: List[Dict[str, Any]] = []
    for row in rows[-max(1, int(limit)) :]:
        slim: Dict[str, Any] = {
            "role": row.get("role"),
            "content": str(row.get("content") or "")[: max(80, int(content_chars))],
        }
        if row.get("name"):
            slim["name"] = row.get("name")
        if row.get("tool_calls"):
            slim["tool_calls"] = True
        out.append(slim)
    return out


def _without_inline_file_bodies(value: Any) -> Any:
    """Copy a persisted snapshot without embedding deliverable bytes/text."""
    if isinstance(value, list):
        return [_without_inline_file_bodies(item) for item in value]
    if not isinstance(value, dict):
        return value
    out = {key: _without_inline_file_bodies(item) for key, item in value.items()}
    is_file = bool(
        out.get("source") == "rlm_kernel"
        or out.get("artifact_id")
        or (out.get("filename") and (out.get("mime_type") or out.get("content_type")))
    )
    if is_file:
        out.pop("text", None)
        out.pop("bytes_b64", None)
    return out


def slim_file_envelopes(rows: Any) -> List[Dict[str, Any]]:
    """Metadata-only file rows safe for frequent status/SSE snapshots."""
    allowed = {
        "filename",
        "relpath",
        "artifact_id",
        "size_bytes",
        "mime_type",
        "content_type",
        "kind",
        "owner",
        "session_id",
        "updated_at",
    }
    return [
        {key: value for key, value in row.items() if key in allowed}
        for row in (rows or [])
        if isinstance(row, dict)
    ]


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with _WRITE_LOCK, open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _read_transcript(path: str, *, limit: int = 40) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict):
                    rows.append(rec)
    except Exception:
        return []
    return slim_messages(rows, limit=limit)


def iter_session_dirs(root: str) -> List[str]:
    out: List[str] = []
    if not root or not os.path.isdir(root):
        return out
    if os.path.isfile(os.path.join(root, "meta.json")):
        out.append(root)
    try:
        names = os.listdir(root)
    except OSError:
        return out
    for name in names:
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if os.path.isfile(os.path.join(path, "meta.json")):
            out.append(path)
        children = os.path.join(path, "children")
        if not os.path.isdir(children):
            continue
        try:
            cnames = os.listdir(children)
        except OSError:
            continue
        for cname in cnames:
            cpath = os.path.join(children, cname)
            if os.path.isfile(os.path.join(cpath, "meta.json")):
                out.append(cpath)
    return out


def terminal_log_path(artifact_dir: str) -> str:
    return os.path.join(str(artifact_dir or ""), TERMINAL_LOG_NAME)


def append_terminal_log(artifact_dir: str, line: str, *, ts: Optional[float] = None) -> None:
    """Append one full terminal line to durable ``terminal.jsonl`` (never trim here)."""
    root = str(artifact_dir or "").strip()
    text = str(line or "")
    if not root or not text:
        return
    try:
        os.makedirs(root, exist_ok=True)
        row = {"ts": float(ts if ts is not None else time.time()), "line": text}
        with _WRITE_LOCK:
            with open(terminal_log_path(root), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def load_terminal_log(artifact_dir: str, *, limit: Optional[int] = None) -> List[str]:
    """Read durable terminal lines (oldest→newest). ``limit`` keeps the tail only."""
    from collections import deque

    path = terminal_log_path(str(artifact_dir or ""))
    if not path or not os.path.isfile(path):
        return []
    keep = int(limit) if limit and int(limit) > 0 else 0
    buf: Any = deque(maxlen=keep) if keep else []
    try:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                s = raw.strip()
                if not s:
                    continue
                line = ""
                try:
                    obj = json.loads(s)
                except Exception:
                    line = s
                else:
                    if isinstance(obj, dict):
                        line = str(obj.get("line") or "").rstrip("\n")
                    elif isinstance(obj, str) and obj.strip():
                        line = obj.strip()
                if line:
                    buf.append(line)
    except Exception:
        return list(buf)[-keep:] if keep else list(buf)
    return list(buf)


def hydrate_log_lines(
    meta: Dict[str, Any], *, limit: Optional[int] = None
) -> Dict[str, Any]:
    """Ensure persisted sessions expose a deploy-style log_lines list.

    Prefer durable ``terminal.jsonl`` (full lines) over truncated meta snapshots.
    ``limit`` caps how many tail lines to keep (default 10000).
    """
    err = str(meta.get("error") or "").strip()
    keep = int(limit) if limit and int(limit) > 0 else 10000

    def _repair(lines: List[str]) -> List[str]:
        out: List[str] = []
        for line in lines:
            s = str(line).rstrip("\n")
            if not s.strip():
                continue
            if s.strip() in ("[error]", "[error] unknown") and err:
                out.append(f"[error] {err}")
            else:
                out.append(s)
        if err and not any(x.startswith("[error]") for x in out):
            out.append(f"[error] {err}")
        return out[-keep:]

    art = str(meta.get("artifact_dir") or meta.get("session_dir") or "").strip()
    durable = load_terminal_log(art, limit=keep) if art else []
    existing = meta.get("log_lines")
    existing_list = (
        [str(x) for x in existing] if isinstance(existing, list) else []
    )
    # Prefer the longer / richer durable file when both exist.
    if durable and (
        not existing_list
        or sum(len(x) for x in durable) >= sum(len(x) for x in existing_list)
    ):
        meta["log_lines"] = _repair(durable)
        return meta
    if existing_list and any(str(x).strip() for x in existing_list):
        meta["log_lines"] = _repair(existing_list)
        return meta
    try:
        from src.orchestrators.rlm.runtime.session import format_rlm_log_line
    except Exception:
        format_rlm_log_line = None  # type: ignore[assignment]
    rebuilt: List[str] = []
    for ev in meta.get("events") or []:
        if not isinstance(ev, dict):
            continue
        kind = str(ev.get("kind") or "event")
        fields = {k: v for k, v in ev.items() if k not in ("ts", "kind", "session_id")}
        if format_rlm_log_line is not None:
            rebuilt.append(format_rlm_log_line(kind, fields))
        else:
            extra = ev.get("detail") or ev.get("error") or ev.get("preview") or ev.get("status") or ""
            rebuilt.append(f"[{kind}] {extra}".rstrip())
    for th in meta.get("thoughts_tail") or []:
        if not isinstance(th, dict):
            continue
        line = f"[{th.get('kind') or 'thought'}] {th.get('text') or ''}".rstrip()
        if line and line not in rebuilt:
            rebuilt.append(line)
    meta["log_lines"] = _repair(rebuilt)
    return meta


def load_persisted_public(
    root: str, session_id: str, *, user_id: str = ""
) -> Optional[Dict[str, Any]]:
    sid = str(session_id or "").strip()
    if not sid:
        return None
    for directory in iter_session_dirs(root):
        meta = _read_json(os.path.join(directory, "meta.json"))
        if not meta:
            continue
        rec_id = str(meta.get("session_id") or os.path.basename(directory) or "")
        if rec_id != sid:
            continue
        owner = str(meta.get("user_id") or "")
        if user_id and owner and owner != user_id:
            continue
        if not meta.get("messages_tail"):
            trans = _read_transcript(os.path.join(directory, "transcript.jsonl"))
            if trans:
                meta["messages_tail"] = trans
        meta["persisted"] = True
        meta["live"] = False
        meta.setdefault("artifact_dir", directory)
        meta.setdefault("session_dir", directory)
        return hydrate_log_lines(meta)
    return None


def list_persisted_public(
    root: str,
    *,
    user_id: str = "",
    flow_id: str = "",
    hydrate_logs: bool = False,
) -> List[Dict[str, Any]]:
    """List cold session meta from disk.

    ``hydrate_logs=False`` (default) skips ``terminal.jsonl`` — required for
    fast session *lists*. Full logs belong on focused session_get / attach.
    """
    out: List[Dict[str, Any]] = []
    want_flow = str(flow_id or "").strip()
    for directory in iter_session_dirs(root):
        meta = _read_json(os.path.join(directory, "meta.json"))
        if not meta or not meta.get("session_id"):
            continue
        owner = str(meta.get("user_id") or "")
        if user_id and owner and owner != user_id:
            continue
        if want_flow and str(meta.get("flow_id") or "") not in ("", want_flow):
            continue
        meta["persisted"] = True
        meta["live"] = False
        meta.setdefault("artifact_dir", directory)
        meta.setdefault("session_dir", directory)
        if hydrate_logs:
            out.append(hydrate_log_lines(meta))
        else:
            # Keep list payloads tiny — drop bulky tails that may sit in meta.
            slim = dict(meta)
            for bulky in (
                "log_lines",
                "thoughts_tail",
                "events",
                "events_tail",
                "messages",
                "messages_tail",
            ):
                slim.pop(bulky, None)
            out.append(slim)
    out.sort(key=lambda x: float(x.get("updated_at") or 0), reverse=True)
    return out

def session_dir_for(root: str, session_id: str, *, user_id: str = "") -> Optional[str]:
    sid = str(session_id or "").strip()
    if not sid:
        return None
    for directory in iter_session_dirs(root):
        meta = _read_json(os.path.join(directory, "meta.json"))
        if not meta:
            continue
        rec_id = str(meta.get("session_id") or os.path.basename(directory) or "")
        if rec_id != sid:
            continue
        owner = str(meta.get("user_id") or "")
        if user_id and owner and owner != user_id:
            continue
        return directory
    return None


def purge_persisted_session(root: str, session_id: str, *, user_id: str = "") -> bool:
    """Remove a local session directory. Does not delete Storage artifacts."""
    directory = session_dir_for(root, session_id, user_id=user_id)
    if not directory:
        return False
    root_abs = os.path.abspath(root)
    dir_abs = os.path.abspath(directory)
    if not root_abs or dir_abs == root_abs:
        return False
    prefix = root_abs if root_abs.endswith(os.sep) else root_abs + os.sep
    if not dir_abs.startswith(prefix):
        return False
    shutil.rmtree(dir_abs, ignore_errors=True)
    return not os.path.isdir(dir_abs)


def clear_session_log_files(artifact_dir: str) -> None:
    """Truncate / remove local transcript + terminal logs for a cold retry."""
    root = str(artifact_dir or "").strip()
    if not root or not os.path.isdir(root):
        return
    for name in (
        "transcript.jsonl",
        "transcript.full.jsonl",
        TERMINAL_LOG_NAME,
        "kernel-state.dill",
        "kernel-state.dill.json",
        "kernel-snapshot.marker",
    ):
        path = os.path.join(root, name)
        try:
            if os.path.isfile(path):
                os.unlink(path)
        except OSError:
            pass
    tmp = os.path.join(root, "tmp")
    if os.path.isdir(tmp):
        try:
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp, exist_ok=True)
        except OSError:
            pass


def wipe_local_deliverable_files(session: Any) -> List[str]:
    """Delete on-disk agent deliverables (keeps pool/case/harness internals).

    Returns deleted absolute paths.
    """
    deleted: List[str] = []
    for rec in list_session_deliverable_files(session):
        path = str(rec.get("path") or "").strip()
        if not path or not os.path.isfile(path):
            continue
        try:
            os.unlink(path)
            deleted.append(path)
        except OSError:
            continue
        # Drop empty parent dirs under the session root (e.g. car-game/).
        try:
            parent = os.path.dirname(path)
            art = str(getattr(session, "artifact_dir", "") or "").strip()
            ws = str(getattr(session, "workspace_dir", "") or "").strip()
            roots = {
                os.path.realpath(p)
                for p in (art, ws)
                if p and os.path.isdir(p)
            }
            while parent and parent not in roots:
                try:
                    os.rmdir(parent)
                except OSError:
                    break
                parent = os.path.dirname(parent)
        except Exception:
            pass
    return deleted


async def wipe_session_storage_artifacts(
    session: Any,
    *,
    extra_artifact_ids: Optional[List[str]] = None,
) -> List[str]:
    """Delete Storage blobs previously stamped for this session's deliverables."""
    aids: List[str] = []
    seen: set[str] = set()

    def _add(aid: Any) -> None:
        s = str(aid or "").strip()
        if s and s not in seen:
            seen.add(s)
            aids.append(s)

    mapping = getattr(session, "deliverable_artifacts", None)
    if isinstance(mapping, dict):
        for aid in mapping.values():
            _add(aid)
    bag = getattr(session, "_superseded_deliverable_artifacts", None)
    if isinstance(bag, list):
        for aid in bag:
            _add(aid)
    for f in getattr(session, "_cached_files", None) or []:
        if isinstance(f, dict):
            _add(f.get("artifact_id"))
    for aid in extra_artifact_ids or []:
        _add(aid)
    # Also scrape meta.json in case memory was restored thin.
    art = str(getattr(session, "artifact_dir", "") or "").strip()
    if art:
        meta = _read_json(os.path.join(art, "meta.json")) or {}
        for f in meta.get("files") or []:
            if isinstance(f, dict):
                _add(f.get("artifact_id"))
    if not aids:
        return []
    try:
        from modules.tools.output_persistence import delete_persisted_blob
    except Exception:
        return []
    uid = str(getattr(session, "user_id", "") or "").strip()
    auth = {"user_id": uid} if uid else {}
    deleted: List[str] = []
    for aid in aids:
        try:
            await delete_persisted_blob(
                aid,
                auth_context=auth,
                service_name="tool-runner",
            )
            deleted.append(aid)
        except Exception:
            logging.getLogger(__name__).debug(
                "retry wipe storage delete skipped aid=%s", aid, exc_info=True
            )
    return deleted


async def wipe_session_for_retry(session: Any) -> Dict[str, Any]:
    """Full cold wipe of one RLM agent node: Storage + disk deliverables + logs.

    Does not delete case ``pool/`` inputs. Clears in-memory deliverable maps on
    ``session``. Caller should also reset transcript / children / kernel.
    """
    storage_deleted = await wipe_session_storage_artifacts(session)
    local_deleted = wipe_local_deliverable_files(session)
    art = str(getattr(session, "artifact_dir", "") or "").strip()
    if art:
        clear_session_log_files(art)
        children_root = os.path.join(art, "children")
        if os.path.isdir(children_root):
            try:
                for name in list(os.listdir(children_root)):
                    path = os.path.join(children_root, name)
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass
    try:
        session.deliverable_artifacts = {}
    except Exception:
        pass
    try:
        session.deliverable_fingerprints = {}
    except Exception:
        pass
    try:
        session._cached_files = []
    except Exception:
        pass
    try:
        session._emitted_deliverable_names = set()
    except Exception:
        pass
    try:
        session._superseded_deliverable_artifacts = []
    except Exception:
        pass
    # Invalidate workspace upload manifests so rewritten files upload again.
    for key in ("workspace_dir", "artifact_dir"):
        raw = str(getattr(session, key, "") or "").strip()
        if not raw:
            continue
        try:
            real = os.path.realpath(raw)
        except OSError:
            continue
        with _MANIFEST_LOCK:
            drop = [
                k for k, m in _MANIFESTS.items() if str(m.get("root") or "") == real
            ]
            for k in drop:
                _MANIFESTS.pop(k, None)
    try:
        persist_session_files(session)
    except Exception:
        pass
    return {
        "storage_deleted": storage_deleted,
        "local_deleted": local_deleted,
    }


def persist_session_files(session: Any) -> None:
    """Write meta.json and a compact transcript snapshot locally."""
    artifact_dir = str(getattr(session, "artifact_dir", "") or "")
    if not artifact_dir:
        return
    os.makedirs(artifact_dir, exist_ok=True)
    public: Dict[str, Any] = {}
    meta_path = os.path.join(artifact_dir, "meta.json")
    if hasattr(session, "public_status"):
        # May run off-loop; the loop thread can mutate registries mid-snapshot.
        for _ in range(3):
            try:
                public = dict(session.public_status() or {})
                break
            except Exception:
                public = {}
    transcript = os.path.join(artifact_dir, "transcript.jsonl")
    messages = list(getattr(session, "messages", []) or [])
    with _WRITE_LOCK:
        if public or not os.path.isfile(meta_path):
            write_json(
                meta_path,
                _without_inline_file_bodies({
                    **public,
                    "run_lifetime": getattr(session, "run_lifetime", None),
                    "model_logical_id": getattr(session, "model_logical_id", None),
                    "harness_path": (
                        session.harness_path()
                        if hasattr(session, "harness_path")
                        else None
                    ),
                }),
            )
        try:
            with open(transcript, "w", encoding="utf-8") as fh:
                for row in messages[-200:]:
                    if isinstance(row, dict):
                        slim = {
                            "role": row.get("role"),
                            "content": str(row.get("content") or "")[:4000],
                        }
                        if row.get("tool_calls"):
                            slim["tool_calls"] = True
                        fh.write(json.dumps(slim, ensure_ascii=False) + "\n")
        except Exception:
            pass
    try:
        from src.orchestrators.rlm.runtime.restore import write_session_cfg

        write_session_cfg(
            artifact_dir,
            dict(getattr(session, "cfg", None) or {}),
            gen_params=dict(getattr(session, "gen_params", None) or {}),
        )
    except Exception:
        pass


def _deliverable_spec_for_name(name: str) -> Tuple[str, str, str]:
    ext = os.path.splitext(name)[1].lower()
    if ext in _DELIVERABLE_EXT:
        return _DELIVERABLE_EXT[ext]
    return _UNKNOWN_DELIVERABLE


def _storage_filename(relpath: str, basename: str) -> str:
    """Stable unique blob name under flows/.../runs/ (preserve nested paths)."""
    rel = str(relpath or basename or "output.bin").replace("\\", "/").strip("/")
    if not rel or rel in (".", ".."):
        rel = basename or "output.bin"
    safe = re.sub(r"[^a-zA-Z0-9._+-]+", "_", rel.replace("/", "__"))
    safe = safe.strip("._") or (basename or "output.bin")
    return safe[:180]


def _skip_dir(name: str) -> bool:
    return name in _SKIP_DIR_NAMES or str(name).startswith(".")


def _dir_signature(root: str) -> Tuple[Tuple[str, int], ...]:
    sig: List[Tuple[str, int]] = []
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            st = os.stat(d)
            with os.scandir(d) as it:
                for entry in it:
                    if _skip_dir(entry.name):
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                    except OSError:
                        continue
        except OSError:
            continue
        sig.append((d, st.st_mtime_ns))
    sig.sort()
    return tuple(sig)


def _scan_deliverable_candidates(root: str) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    seen_real: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _skip_dir(d)]
        for name in filenames:
            if not name or name.startswith(".") or name in _SESSION_INTERNAL_NAMES:
                continue
            path = os.path.join(dirpath, name)
            try:
                real = os.path.realpath(path)
            except OSError:
                continue
            if real in seen_real:
                continue
            try:
                st = os.stat(path)
            except OSError:
                continue
            try:
                rel = os.path.relpath(path, root)
            except ValueError:
                rel = name
            kind, mime, media = _deliverable_spec_for_name(name)
            storage_name = _storage_filename(rel, name)
            seen_real.add(real)
            found.append(
                {
                    "path": path,
                    "filename": storage_name,
                    "display_name": name,
                    "relpath": str(rel).replace("\\", "/"),
                    "kind": kind,
                    "mime_type": mime,
                    "media_type": media,
                    "size_bytes": int(st.st_size),
                    "mtime_ns": int(st.st_mtime_ns),
                }
            )
            if len(found) >= _MAX_DELIVERABLE_FILES:
                return found
    return found


def _refresh_candidate_stats(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in cands:
        try:
            st = os.stat(rec["path"])
        except OSError:
            continue
        rec["size_bytes"] = int(st.st_size)
        rec["mtime_ns"] = int(st.st_mtime_ns)
        out.append(rec)
    return out


def invalidate_deliverable_cache(root: Optional[str] = None) -> None:
    global _TEXT_CACHE_BYTES
    with _CACHE_LOCK:
        if root is None:
            _LISTING_CACHE.clear()
            _TEXT_CACHE.clear()
            _TEXT_CACHE_BYTES = 0
        else:
            _LISTING_CACHE.pop(str(root), None)
    with _MANIFEST_LOCK:
        if root is None:
            _MANIFESTS.clear()
        else:
            real = os.path.realpath(str(root))
            for key in list(_MANIFESTS):
                if _MANIFESTS[key].get("root") == real:
                    _MANIFESTS.pop(key, None)


def _canonical_root_key(root: str, owner: str = "") -> Tuple[str, str]:
    real = os.path.realpath(os.path.abspath(root))
    try:
        st = os.stat(real)
        fingerprint = f"{int(st.st_dev)}:{int(st.st_ino)}"
    except OSError:
        fingerprint = "missing"
    owner_key = str(owner or "").strip()
    return real, f"{real}|{fingerprint}|{owner_key}"


def _manifest_for(root: str, owner: str = "") -> Dict[str, Any]:
    real, key = _canonical_root_key(root, owner)
    manifest = _MANIFESTS.get(key)
    if manifest is None:
        manifest = {
            "key": key,
            "root": real,
            "owner": str(owner or ""),
            "signature": None,
            "candidates": [],
            "scanned_at": 0.0,
            "uploaded": {},
            "reserved": {},
        }
        _MANIFESTS[key] = manifest
    return manifest


def _manifest_files(root: str, owner: str = "") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return one manifest snapshot; discovery is serialized per canonical root."""
    with _MANIFEST_LOCK:
        manifest = _manifest_for(root, owner)
        sig = _dir_signature(manifest["root"])
        if manifest["signature"] != sig:
            manifest["candidates"] = _scan_deliverable_candidates(manifest["root"])
            manifest["signature"] = sig
            manifest["scanned_at"] = time.time()
        else:
            manifest["candidates"] = _refresh_candidate_stats(manifest["candidates"])
        records = [
            dict(rec)
            for rec in manifest["candidates"]
            if 0 <= int(rec.get("size_bytes") or 0) <= _MAX_DELIVERABLE_BYTES
        ]
        return manifest, records


def list_kernel_deliverable_files(artifact_dir: str) -> List[Dict[str, Any]]:
    """User files written under a session root, excluding session internals."""
    root = str(artifact_dir or "").strip()
    if not root or not os.path.isdir(root):
        return []
    _, records = _manifest_files(root)
    return records


def snapshot_workspace_files(root: str) -> Dict[str, Tuple[int, int]]:
    """Map relative path → (mtime_ns, size) for deliverable candidates under root."""
    base = str(root or "").strip()
    if not base or not os.path.isdir(base):
        return {}
    out: Dict[str, Tuple[int, int]] = {}
    for rec in _scan_deliverable_candidates(base):
        rel = str(rec.get("relpath") or "").strip()
        if not rel:
            continue
        out[rel] = (int(rec.get("mtime_ns") or 0), int(rec.get("size_bytes") or 0))
    return out


def diff_workspace_produced_files(
    root: str,
    baseline: Optional[Mapping[str, Any]] = None,
    *,
    max_files: int = 40,
) -> List[Dict[str, Any]]:
    """Files created or updated in the shared workspace since ``baseline`` snapshot.

    Used to tell a parent which paths a child wrote even when the child never
    called ``agent_message.send``.
    """
    current = snapshot_workspace_files(root)
    base = baseline if isinstance(baseline, Mapping) else {}
    produced: List[Dict[str, Any]] = []
    for rel, (mtime_ns, size) in sorted(current.items()):
        prev = base.get(rel)
        if prev is None:
            changed = True
        else:
            try:
                prev_mtime, prev_size = int(prev[0]), int(prev[1])
            except (TypeError, ValueError, IndexError):
                prev_mtime, prev_size = -1, -1
            changed = mtime_ns != prev_mtime or size != prev_size
        if not changed:
            continue
        produced.append(
            {
                "relpath": rel,
                "size_bytes": size,
                "mtime_ns": mtime_ns,
                "new": prev is None,
            }
        )
        if len(produced) >= max(1, int(max_files or 40)):
            break
    return produced


async def list_kernel_deliverable_files_async(
    artifact_dir: str,
) -> List[Dict[str, Any]]:
    """Off-loop equivalent of :func:`list_kernel_deliverable_files`."""
    return await asyncio.to_thread(list_kernel_deliverable_files, artifact_dir)


def snapshot_pool_files(root: str) -> Dict[str, Tuple[int, int]]:
    """Map ``pool/files/<rel>`` → (mtime_ns, size) for materialize baseline."""
    base = str(root or "").strip()
    files_dir = os.path.join(base, "pool", "files") if base else ""
    if not files_dir or not os.path.isdir(files_dir):
        return {}
    out: Dict[str, Tuple[int, int]] = {}
    for dirpath, _dirnames, filenames in os.walk(files_dir):
        for name in filenames:
            if not name or name.startswith("."):
                continue
            path = os.path.join(dirpath, name)
            try:
                st = os.stat(path)
                rel_under = os.path.relpath(path, base).replace("\\", "/")
            except (OSError, ValueError):
                continue
            out[rel_under] = (int(st.st_mtime_ns), int(st.st_size))
    return out


def write_pool_files_baseline(root: str) -> Dict[str, Tuple[int, int]]:
    """Snapshot ``pool/files/`` after materialize so later agent writes are deliverables.

    Materialized case/session inputs stay out of Session Files; new/changed
    files the model writes under ``pool/files/`` are picked up by
    :func:`list_session_deliverable_files`.
    """
    base = str(root or "").strip()
    snap = snapshot_pool_files(base)
    if not base:
        return snap
    path = os.path.join(base, _POOL_BASELINE_REL)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        serial = {k: [int(v[0]), int(v[1])] for k, v in snap.items()}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"schema_version": 1, "files": serial}, fh, ensure_ascii=False)
    except Exception:
        logging.getLogger(__name__).debug(
            "pool deliverable baseline write failed root=%s", base, exc_info=True
        )
    return snap


def load_pool_files_baseline(root: str) -> Dict[str, Tuple[int, int]]:
    base = str(root or "").strip()
    path = os.path.join(base, _POOL_BASELINE_REL) if base else ""
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        files = raw.get("files") if isinstance(raw, Mapping) else None
        if not isinstance(files, Mapping):
            return {}
        out: Dict[str, Tuple[int, int]] = {}
        for key, val in files.items():
            rel = str(key or "").replace("\\", "/").strip()
            if not rel:
                continue
            try:
                out[rel] = (int(val[0]), int(val[1]))
            except (TypeError, ValueError, IndexError):
                continue
        return out
    except Exception:
        return {}


def stamp_pool_files_baseline(session: Any, root: Optional[str] = None) -> Dict[str, Tuple[int, int]]:
    """Write baseline to disk and cache on the session object."""
    art = str(
        root
        or getattr(session, "artifact_dir", "")
        or getattr(session, "workspace_dir", "")
        or ""
    ).strip()
    snap = write_pool_files_baseline(art)
    try:
        session._pool_files_baseline = snap
    except Exception:
        pass
    return snap


def _scan_pool_files_outputs(root: str) -> List[Dict[str, Any]]:
    """Scan ``pool/files/`` for candidate deliverables (agent outputs live here)."""
    base = str(root or "").strip()
    files_dir = os.path.join(base, "pool", "files") if base else ""
    if not files_dir or not os.path.isdir(files_dir):
        return []
    found: List[Dict[str, Any]] = []
    seen_real: set[str] = set()
    for dirpath, _dirnames, filenames in os.walk(files_dir):
        for name in filenames:
            if not name or name.startswith(".") or name in _SESSION_INTERNAL_NAMES:
                continue
            path = os.path.join(dirpath, name)
            try:
                real = os.path.realpath(path)
                st = os.stat(path)
                rel = os.path.relpath(path, base).replace("\\", "/")
            except (OSError, ValueError):
                continue
            if real in seen_real:
                continue
            seen_real.add(real)
            kind, mime, media = _deliverable_spec_for_name(name)
            found.append(
                {
                    "path": path,
                    "filename": _storage_filename(rel, name),
                    "display_name": name,
                    "relpath": rel,
                    "kind": kind,
                    "mime_type": mime,
                    "media_type": media,
                    "size_bytes": int(st.st_size),
                    "mtime_ns": int(st.st_mtime_ns),
                    "_pool_output": True,
                }
            )
            if len(found) >= _MAX_DELIVERABLE_FILES:
                return found
    return found


def _pool_output_is_new_or_changed(
    rec: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> bool:
    rel = str(rec.get("relpath") or "").replace("\\", "/").strip()
    if not rel:
        return False
    prev = baseline.get(rel)
    if prev is None:
        return True
    try:
        prev_mtime, prev_size = int(prev[0]), int(prev[1])
    except (TypeError, ValueError, IndexError):
        return True
    return int(rec.get("mtime_ns") or 0) != prev_mtime or int(
        rec.get("size_bytes") or 0
    ) != prev_size


def list_session_deliverable_files(session: Any) -> List[Dict[str, Any]]:
    """All deliverable files for a session (shared workspace + private artifact dir).

    Also includes **new/changed** files under ``pool/files/`` relative to the
    post-materialize baseline (agent outputs the model claims to have saved).
    """
    roots: List[Tuple[str, str, str]] = []
    workspace = str(getattr(session, "workspace_dir", "") or "").strip()
    session_id = str(getattr(session, "session_id", "") or "").strip()
    user_id = str(getattr(session, "user_id", "") or "").strip()
    for key in ("workspace_dir", "artifact_dir"):
        raw = str(getattr(session, key, "") or "").strip()
        real = os.path.realpath(raw) if raw else ""
        if not raw or not os.path.isdir(raw) or any(real == item[0] for item in roots):
            continue
        role = "workspace" if key == "workspace_dir" else "private"
        # Shared workspaces key on user ownership; private child roots also key
        # on session ownership so their files remain attributable.
        owner = user_id if role == "workspace" else f"{user_id}:{session_id}"
        roots.append((real, owner, role))
    found: List[Dict[str, Any]] = []
    seen_real: set[str] = set()
    for root, owner, role in roots:
        with _MANIFEST_LOCK:
            manifest, records = _manifest_files(root, owner)
        for rec in records:
            try:
                real = os.path.realpath(str(rec.get("path") or ""))
            except OSError:
                continue
            if not real or real in seen_real:
                continue
            seen_real.add(real)
            rec["_manifest_key"] = manifest["key"]
            rec["_root_role"] = role
            rec["_owner_session_id"] = session_id
            found.append(rec)
            if len(found) >= _MAX_DELIVERABLE_FILES:
                return found
        # Agent-written outputs under pool/files/ (inputs excluded via baseline).
        sess_base = getattr(session, "_pool_files_baseline", None)
        baseline: Mapping[str, Any]
        if isinstance(sess_base, Mapping) and sess_base:
            baseline = sess_base
        else:
            baseline = load_pool_files_baseline(root)
        if not baseline and not os.path.isdir(os.path.join(root, "pool", "files")):
            continue
        # No baseline → do not treat pre-existing pool inputs as deliverables.
        if not baseline:
            continue
        for rec in _scan_pool_files_outputs(root):
            if not _pool_output_is_new_or_changed(rec, baseline):
                continue
            try:
                real = os.path.realpath(str(rec.get("path") or ""))
            except OSError:
                continue
            if not real or real in seen_real:
                continue
            if int(rec.get("size_bytes") or 0) > _MAX_DELIVERABLE_BYTES:
                continue
            seen_real.add(real)
            rec["_manifest_key"] = ""
            rec["_root_role"] = role
            rec["_owner_session_id"] = session_id
            found.append(rec)
            if len(found) >= _MAX_DELIVERABLE_FILES:
                return found
    return found


async def list_session_deliverable_files_async(session: Any) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(list_session_deliverable_files, session)


_MAX_INLINE_TEXT_BYTES = 200 * 1024
_INLINE_TEXT_MIMES = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/csv",
        "text/html",
        "text/css",
        "text/javascript",
        "text/x-python",
        "application/json",
    }
)


def kernel_deliverable_envelopes(session: Any) -> List[Dict[str, Any]]:
    """Slim file envelopes for Direct Reply / UI (artifact_id stamped after persist).

    Small text documents also carry inline ``text`` so Direct Reply can render
    content immediately without waiting for Storage artifact_id stamping.
    """
    known = getattr(session, "deliverable_artifacts", None)
    known = known if isinstance(known, dict) else {}
    out: List[Dict[str, Any]] = []
    for rec in list_session_deliverable_files(session):
        env: Dict[str, Any] = {
            "kind": rec["kind"],
            "filename": rec["filename"],
            "display_name": rec.get("display_name") or rec["filename"],
            "relpath": rec.get("relpath") or "",
            "mime_type": rec["mime_type"],
            "content_type": rec["mime_type"],
            "media_type": rec["media_type"],
            "size_bytes": rec["size_bytes"],
            "source": "rlm_kernel",
        }
        aid = str(
            known.get(rec["filename"])
            or known.get(str(rec.get("relpath") or ""))
            or known.get(str(rec.get("display_name") or ""))
            or ""
        ).strip()
        manifest_key = str(rec.get("_manifest_key") or "")
        revision = _fp_tuple(rec.get("size_bytes"), rec.get("mtime_ns"))
        if not aid and manifest_key:
            with _MANIFEST_LOCK:
                manifest = _MANIFESTS.get(manifest_key) or {}
                uploaded = (manifest.get("uploaded") or {}).get(rec.get("relpath"))
                if isinstance(uploaded, dict) and tuple(uploaded.get("revision") or ()) == revision:
                    aid = str(uploaded.get("artifact_id") or "").strip()
        if aid:
            env["artifact_id"] = aid
        mime = str(rec.get("mime_type") or "")
        size = int(rec.get("size_bytes") or 0)
        if mime in _INLINE_TEXT_MIMES and 0 < size <= _MAX_INLINE_TEXT_BYTES:
            text = _read_inline_text(
                str(rec.get("path") or ""), int(rec.get("mtime_ns") or 0), size
            )
            if text is not None:
                env["text"] = text
        out.append(env)
    return out


async def kernel_deliverable_envelopes_async(session: Any) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(kernel_deliverable_envelopes, session)


def _read_inline_text(path: str, mtime_ns: int, size: int) -> Optional[str]:
    if not path:
        return None
    with _CACHE_LOCK:
        hit = _TEXT_CACHE.get(path)
    if hit is not None and hit[0] == mtime_ns and hit[1] == size:
        return hit[2]
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None
    global _TEXT_CACHE_BYTES
    with _CACHE_LOCK:
        old = _TEXT_CACHE.get(path)
        if old is not None:
            _TEXT_CACHE_BYTES -= len(old[2])
        if _TEXT_CACHE_BYTES + len(text) > _TEXT_CACHE_MAX_BYTES:
            _TEXT_CACHE.clear()
            _TEXT_CACHE_BYTES = 0
        _TEXT_CACHE[path] = (mtime_ns, size, text)
        _TEXT_CACHE_BYTES += len(text)
    return text


def sync_session_deliverables(session: Any) -> List[Dict[str, Any]]:
    """Scan kernel cwd, emit [file] once per new name, return UI envelopes."""
    envs = kernel_deliverable_envelopes(session)
    try:
        session._cached_files = list(envs)
    except Exception:
        pass
    seen = getattr(session, "_emitted_deliverable_names", None)
    if not isinstance(seen, set):
        seen = set()
        try:
            session._emitted_deliverable_names = seen
        except Exception:
            pass
    emit = getattr(session, "_emit", None)
    for env in envs:
        name = str(env.get("filename") or "")
        if not name or name in seen:
            continue
        seen.add(name)
        if callable(emit):
            try:
                emit("file", filename=name, preview=name)
            except Exception:
                pass
    return envs


def _fp_tuple(size_bytes: Any, mtime_ns: Any) -> Tuple[int, int]:
    return (int(size_bytes or 0), int(mtime_ns or 0))


def _fp_matches(stored: Any, size_bytes: int, mtime_ns: int) -> bool:
    if stored is None:
        return False
    if isinstance(stored, (list, tuple)) and len(stored) >= 2:
        return int(stored[0] or 0) == size_bytes and int(stored[1] or 0) == mtime_ns
    if isinstance(stored, dict):
        return (
            int(stored.get("size_bytes") or 0) == size_bytes
            and int(stored.get("mtime_ns") or 0) == mtime_ns
        )
    return False


def _lookup_mapped(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        text = str(key or "").strip()
        if text and text in mapping and mapping[text] not in (None, ""):
            return mapping[text]
    return None


def pending_deliverable_intents(session: Any) -> List[Dict[str, Any]]:
    """new_blob intents for kernel files missing from Storage or rewritten on disk.

    A prior artifact_id is not enough: if size/mtime changed since the last
    successful upload, queue another blob so File Explorer gets the final bytes.
    """
    try:
        from modules.tools.output_persistence import (
            intent_new_blob,
            register_persist_failure_hook,
        )
    except Exception:
        return []
    register_persist_failure_hook("rlm_workspace", release_persist_intent_reservation)
    known = getattr(session, "deliverable_artifacts", None)
    known = known if isinstance(known, dict) else {}
    fingerprints = getattr(session, "deliverable_fingerprints", None)
    fingerprints = fingerprints if isinstance(fingerprints, dict) else {}
    flow_id = str(getattr(session, "flow_id", "") or "").strip() or None
    run_id = str(getattr(session, "run_id", "") or "").strip() or None
    intents: List[Dict[str, Any]] = []
    for rec in list_session_deliverable_files(session):
        key = str(rec.get("filename") or "")
        rel = str(rec.get("relpath") or "")
        display = str(rec.get("display_name") or "")
        size_bytes = int(rec.get("size_bytes") or 0)
        mtime_ns = int(rec.get("mtime_ns") or 0)
        aid = str(
            _lookup_mapped(known, key, rel, display) or ""
        ).strip()
        prev_fp = _lookup_mapped(fingerprints, key, rel, display)
        manifest_key = str(rec.get("_manifest_key") or "")
        revision = _fp_tuple(size_bytes, mtime_ns)
        reservation_id = ""
        if manifest_key:
            with _MANIFEST_LOCK:
                manifest = _MANIFESTS.get(manifest_key)
                if manifest is not None:
                    uploaded = manifest["uploaded"].get(rel)
                    if (
                        isinstance(uploaded, dict)
                        and tuple(uploaded.get("revision") or ()) == revision
                        and uploaded.get("artifact_id")
                    ):
                        continue
                    if isinstance(uploaded, dict) and uploaded.get("artifact_id") and not aid:
                        aid = str(uploaded["artifact_id"])
                    reserved = manifest["reserved"].get(rel)
                    now = time.monotonic()
                    if (
                        isinstance(reserved, dict)
                        and tuple(reserved.get("revision") or ()) == revision
                        and now - float(reserved.get("at") or 0) < _RESERVATION_TTL_SEC
                    ):
                        continue
                    reservation_id = f"{threading.get_ident():x}-{time.time_ns():x}"
                    manifest["reserved"][rel] = {
                        "revision": revision,
                        "reservation_id": reservation_id,
                        "at": now,
                    }
        # Skip only when Storage already has this exact on-disk revision.
        if aid and _fp_matches(prev_fp, size_bytes, mtime_ns):
            if manifest_key:
                with _MANIFEST_LOCK:
                    manifest = _MANIFESTS.get(manifest_key)
                    if manifest is not None:
                        manifest["uploaded"][rel] = {
                            "revision": revision,
                            "artifact_id": aid,
                        }
                        manifest["reserved"].pop(rel, None)
            continue
        intents.append(
            intent_new_blob(
                local_path=rec["path"],
                filename=key,
                mime_type=rec["mime_type"],
                media_type=rec["media_type"],
                layout="default",
                node_kind="direct_reply",
                flow_id=flow_id,
                run_id=run_id,
                # Never park agent_flow deliverables under case_importer/…
                bind_case=False,
                metadata={
                    "kind": rec["kind"],
                    "content_type": rec["mime_type"],
                    "source": "rlm_kernel",
                    "relpath": rel,
                    "display_name": display,
                    "rlm_session_id": str(getattr(session, "session_id", "") or ""),
                    "size_bytes": size_bytes,
                    "mtime_ns": mtime_ns,
                    "replaces_artifact_id": aid or None,
                    "workspace_manifest_key": manifest_key,
                    "workspace_revision": list(revision),
                    "workspace_reservation_id": reservation_id,
                    "workspace_root_role": rec.get("_root_role"),
                    "owner_session_id": rec.get("_owner_session_id"),
                    "persist_failure_hook": "rlm_workspace",
                },
            )
        )
    return intents


async def pending_deliverable_intents_async(session: Any) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(pending_deliverable_intents, session)


def release_persist_intent_reservation(intent: Mapping[str, Any]) -> None:
    """Make a failed reserved revision immediately eligible for retry."""
    meta = intent.get("metadata") if isinstance(intent.get("metadata"), Mapping) else {}
    manifest_key = str(meta.get("workspace_manifest_key") or "")
    rel = str(meta.get("relpath") or "")
    reservation_id = str(meta.get("workspace_reservation_id") or "")
    if not manifest_key or not rel:
        return
    with _MANIFEST_LOCK:
        manifest = _MANIFESTS.get(manifest_key)
        reserved = (manifest or {}).get("reserved", {}).get(rel)
        if isinstance(reserved, dict) and (
            not reservation_id or reserved.get("reservation_id") == reservation_id
        ):
            manifest["reserved"].pop(rel, None)


def persist_intents_for_session(session: Any) -> List[Dict[str, Any]]:
    """User-facing kernel files only. Session internals stay on local disk."""
    return pending_deliverable_intents(session)


async def persist_intents_for_session_async(session: Any) -> List[Dict[str, Any]]:
    """Off-loop persistence intent discovery for async runner call sites."""
    return await pending_deliverable_intents_async(session)


def remember_session_deliverables(session: Any, entries: List[Dict[str, Any]]) -> None:
    """Record Storage artifact ids + size/mtime fingerprints for kernel files.

    ``pending_deliverable_intents`` skips only when both the artifact id and the
    on-disk fingerprint still match — rewritten files are uploaded again.
    """
    mapping = getattr(session, "deliverable_artifacts", None)
    if not isinstance(mapping, dict):
        mapping = {}
        try:
            session.deliverable_artifacts = mapping
        except Exception:
            return
    fingerprints = getattr(session, "deliverable_fingerprints", None)
    if not isinstance(fingerprints, dict):
        fingerprints = {}
        try:
            session.deliverable_fingerprints = fingerprints
        except Exception:
            fingerprints = {}
    superseded: List[str] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        filename = str(entry.get("filename") or "").strip()
        artifact_id = str(entry.get("artifact_id") or "").strip()
        meta = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        if filename and artifact_id:
            old = str(mapping.get(filename) or "").strip()
            if old and old != artifact_id:
                superseded.append(old)
            mapping[filename] = artifact_id
        for alt in (
            str(meta.get("relpath") or "").strip(),
            str(meta.get("display_name") or "").strip(),
        ):
            if alt and artifact_id:
                mapping[alt] = artifact_id
        fp = _fp_tuple(meta.get("size_bytes"), meta.get("mtime_ns"))
        # Prefer live stat when path is known so fingerprint matches disk.
        local_path = str(
            entry.get("local_path") or meta.get("local_path") or ""
        ).strip()
        if local_path and os.path.isfile(local_path):
            try:
                st = os.stat(local_path)
                fp = _fp_tuple(st.st_size, st.st_mtime_ns)
            except OSError:
                pass
        if filename:
            fingerprints[filename] = fp
        for alt in (
            str(meta.get("relpath") or "").strip(),
            str(meta.get("display_name") or "").strip(),
        ):
            if alt:
                fingerprints[alt] = fp
        replaced = str(meta.get("replaces_artifact_id") or "").strip()
        if replaced and replaced != artifact_id:
            superseded.append(replaced)
        manifest_key = str(meta.get("workspace_manifest_key") or "")
        relpath = str(meta.get("relpath") or "")
        revision_raw = meta.get("workspace_revision")
        revision = (
            tuple(revision_raw)
            if isinstance(revision_raw, (list, tuple)) and len(revision_raw) >= 2
            else fp
        )
        if manifest_key and relpath and artifact_id:
            with _MANIFEST_LOCK:
                manifest = _MANIFESTS.get(manifest_key)
                if manifest is not None:
                    manifest["uploaded"][relpath] = {
                        "revision": tuple(revision[:2]),
                        "artifact_id": artifact_id,
                        "owner_session_id": meta.get("owner_session_id"),
                    }
                    manifest["reserved"].pop(relpath, None)
    if superseded:
        bag = getattr(session, "_superseded_deliverable_artifacts", None)
        if not isinstance(bag, list):
            bag = []
            try:
                session._superseded_deliverable_artifacts = bag
            except Exception:
                bag = []
        for aid in superseded:
            if aid and aid not in bag:
                bag.append(aid)


def attach_persist_intents(payload: Dict[str, Any], session: Any) -> Dict[str, Any]:
    persist_session_files(session)
    intents = persist_intents_for_session(session)
    if not intents or not isinstance(payload, dict):
        return payload
    payload["output_intents"] = intents
    channels = payload.get("channels")
    if isinstance(channels, dict):
        channels["storage"] = True
    return payload
