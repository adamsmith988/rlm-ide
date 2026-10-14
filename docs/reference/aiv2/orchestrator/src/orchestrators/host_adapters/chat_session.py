"""Hub-chat host adapter: session files → generic pool HostPorts.

This is how the chatbot *uses* an orchestrator that declares ``pools.*`` —
not part of any engine. Agent Flow keeps its own data_pool host wiring.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.orchestrators import host_pools
from src.orchestrators.host_adapters.case_mentions import (
    build_case_mentions_inventory,
    merge_inventories,
)

logger = logging.getLogger(__name__)


def _row_from_entry(entry: Mapping[str, Any], *, source: str) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, Mapping):
        return None
    aid = entry.get("artifact_id") or entry.get("id")
    path = (
        entry.get("path")
        or entry.get("storage_path")
        or entry.get("file_path")
        or ""
    )
    name = (
        entry.get("name")
        or entry.get("filename")
        or entry.get("original_filename")
        or (os.path.basename(str(path)) if path else "")
        or (str(aid) if aid else "")
        or "file"
    )
    if not aid and not path:
        return None
    return {
        "filename": host_pools.safe_pool_filename(str(name)),
        "original_filename": str(name),
        "artifact_id": str(aid).strip() if aid else None,
        "path": str(path or "").strip() or None,
        "mime_type": entry.get("mime_type") or entry.get("content_type"),
        "size_bytes": entry.get("size") or entry.get("size_bytes") or entry.get("file_size_bytes"),
        "case_id": entry.get("case_id"),
        "source": source,
    }


def inventory_from_session_files(
    file_registry: Optional[Sequence[Mapping[str, Any]]] = None,
    artifacts: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Normalize chat session registry + fresh uploads into a generic inventory."""
    inv = host_pools.empty_inventory(note="hub_chat_session_files")
    seen: set[str] = set()
    files: List[Dict[str, Any]] = []

    def _add(row: Optional[Dict[str, Any]]) -> None:
        if not row:
            return
        key = str(row.get("artifact_id") or row.get("path") or row.get("filename") or "")
        if not key or key in seen:
            return
        seen.add(key)
        files.append(row)

    for art in artifacts or []:
        _add(_row_from_entry(art, source="fresh_turn"))
    for entry in file_registry or []:
        _add(_row_from_entry(entry, source="session_registry"))

    inv["files"] = files
    inv["count"] = len(files)
    return inv


def make_session_artifact_download(
    auth_context: Optional[Dict[str, Any]] = None,
):
    """Storage fetch used by pool materialize + lazy ``pool_resolve``."""

    async def _download(artifact_id: str, filename_hint: str) -> Optional[bytes]:
        del filename_hint
        try:
            from src.utils.multimodal import _download_by_artifact_id

            return await _download_by_artifact_id(
                artifact_id, auth_context=auth_context or {}
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat session pool download failed aid=%s: %s", artifact_id, exc)
            return None

    return _download


async def build_chat_session_pool(
    *,
    artifact_dir: str,
    file_registry: Optional[Sequence[Mapping[str, Any]]] = None,
    artifacts: Optional[Sequence[Mapping[str, Any]]] = None,
    pool_name: str = "session",
    auth_context: Optional[Dict[str, Any]] = None,
    request_context: Optional[Mapping[str, Any]] = None,
    case_inventory: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Materialize session + optional case-mention files; return pool_defs/inventory/closures."""
    os.makedirs(artifact_dir, exist_ok=True)
    session_inv = inventory_from_session_files(file_registry, artifacts)

    if isinstance(case_inventory, Mapping) and (
        case_inventory.get("files")
        or case_inventory.get("texts")
        or case_inventory.get("case_ids")
    ):
        case_inv: Dict[str, Any] = dict(case_inventory)
    else:
        user_id = str((auth_context or {}).get("user_id") or "").strip()
        case_inv = await build_case_mentions_inventory(
            request_context, user_id=user_id
        )

    inv = merge_inventories(session_inv, case_inv)
    _download = make_session_artifact_download(auth_context)

    mat = {"materialized": 0, "failed": 0}
    if inv.get("files") or inv.get("texts"):
        try:
            mat = await host_pools.materialize_pool_files(
                artifact_dir, inv, download=_download
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("chat session pool materialize failed: %s", exc)
            host_pools.write_pool_index(artifact_dir, inv)
    else:
        host_pools.write_pool_index(artifact_dir, inv)

    # Snapshot pool/files after materialize so agent-written outputs become
    # Session Files deliverables (AF parity).
    try:
        from src.orchestrators.rlm.runtime.persist import write_pool_files_baseline

        write_pool_files_baseline(artifact_dir)
    except Exception:
        logger.debug("chat session pool baseline stamp skipped", exc_info=True)

    pool_defs: List[Dict[str, Any]] = [
        {
            "name": pool_name,
            "display_name": pool_name,
            "pool_key": pool_name,
            "inventory": inv,
            "files": (host_pools.inventory_ls_payload(inv).get("files") or []),
        }
    ]
    # Extra named pool when case outputs are present (parity with agent-flow data_pool).
    if case_inv.get("files") or case_inv.get("texts") or case_inv.get("case_ids"):
        pool_defs.append(
            {
                "name": "case",
                "display_name": "case",
                "pool_key": "case",
                "inventory": case_inv,
                "files": (host_pools.inventory_ls_payload(case_inv).get("files") or []),
            }
        )

    closures = host_pools.make_pool_host_closures(
        artifact_dir=artifact_dir,
        pool_defs=pool_defs,
        union_inventory=inv,
        download=_download,
    )
    logger.info(
        "chat session pool ready dir=%s files=%s texts=%s cases=%s "
        "ok=%s skip=%s cache=%s download=%s failed=%s",
        artifact_dir,
        len(inv.get("files") or []),
        len(inv.get("texts") or []),
        inv.get("case_ids"),
        mat.get("materialized"),
        mat.get("skipped"),
        mat.get("cache_hits"),
        mat.get("downloaded"),
        mat.get("failed"),
    )
    return pool_defs, inv, closures
