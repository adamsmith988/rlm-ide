"""Hub-chat: apply @case mention outputs into a data-pool-like inventory.

Mirrors agent-flow Case node + data_pool ingest without importing the runner.
When ``chat_mentions_v1.cases[].outputs`` is present, resolve selected
artifacts / kinds / tool evidence and merge into the session pool so RLM
(and native attach) can work with those files.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.orchestrators import host_pools

logger = logging.getLogger(__name__)

CHAT_MENTIONS_CONTEXT_KEY = "chat_mentions_v1"
CASE_BINDINGS_FILENAME = "hub_case_bindings.json"
CASE_INVENTORY_FILENAME = "hub_case_inventory.json"
# Survives orchestrator container restarts (Postgres chat_sessions.metadata).
SESSION_CASE_BINDINGS_META_KEY = "hub_case_bindings_v1"

# Cap for merging case files into the chat session registry / attach path.
# Default 0: case outputs stay in the RLM ``case`` pool only — never dump into
# Session Files / native attach UI. Set AIV2_CHAT_CASE_SESSION_ATTACH_CAP > 0
# only for legacy native-tools debugging.
_SESSION_REGISTRY_CAP = max(
    0, int(os.environ.get("AIV2_CHAT_CASE_SESSION_ATTACH_CAP") or "0")
)

# Soft cap for durable inventory file rows (artifacts / kinds). Evidence texts
# are kept in full. Oversized cases keep a sample + ``files_truncated`` flag;
# RLM can still use text_search scope=case / pool tools for the rest.
_INVENTORY_FILE_CAP = max(
    0, int(os.environ.get("AIV2_CHAT_CASE_INVENTORY_FILE_CAP") or "250")
)


def hub_chat_artifact_dir(session_id: str, user_id: str) -> str:
    """Stable per-chat RLM workspace (same path as ``rlm_chat_stream``)."""
    try:
        from modules.tools.run_workspace import workspace_dir

        base = workspace_dir(
            "orchestrator_rlm",
            str(user_id or "user"),
            str(session_id or "session"),
            "chat",
        )
        path = os.path.join(base, "rlm")
        os.makedirs(path, exist_ok=True)
        return path
    except Exception:
        root = (
            str(os.environ.get("AIV2_RLM_CHAT_ROOT") or "").strip()
            or "/tmp/aiv2_rlm_chat"
        )
        uid = re.sub(r"[^A-Za-z0-9._-]+", "_", str(user_id or "user"))[:80] or "user"
        sid = (
            re.sub(r"[^A-Za-z0-9._-]+", "_", str(session_id or "session"))[:180]
            or "session"
        )
        path = os.path.join(root, uid, sid, "rlm")
        os.makedirs(path, exist_ok=True)
        return path


def inventory_has_cases(inventory: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(inventory, Mapping):
        return False
    if any(str(x).strip() for x in (inventory.get("case_ids") or [])):
        return True
    if inventory.get("files") or inventory.get("texts"):
        # Prefer explicit case_ids; fall back to files tagged with case_id.
        for row in list(inventory.get("files") or []) + list(inventory.get("texts") or []):
            if isinstance(row, Mapping) and str(row.get("case_id") or "").strip():
                return True
    return False


def merge_case_selections(
    base: Sequence[Mapping[str, Any]],
    new: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge case selections by case_id (later wins)."""
    by_id: Dict[str, Dict[str, Any]] = {}
    for src in (base, new):
        for row in src or []:
            if not isinstance(row, Mapping):
                continue
            cid = str(row.get("case_id") or row.get("id") or "").strip()
            if not cid:
                continue
            by_id[cid] = {
                "case_id": cid,
                "label": str(row.get("label") or cid),
                "outputs": list(row.get("outputs") or []),
                "emit_mode": str(row.get("emit_mode") or "batch"),
            }
    return list(by_id.values())


def _bindings_disk_payload(
    selections: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "selections": merge_case_selections([], selections),
    }


def save_case_bindings(
    artifact_dir: str,
    selections: Sequence[Mapping[str, Any]],
) -> None:
    path = os.path.join(str(artifact_dir or ""), CASE_BINDINGS_FILENAME)
    if not artifact_dir:
        return
    try:
        os.makedirs(artifact_dir, exist_ok=True)
        payload = _bindings_disk_payload(selections)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("case_mentions: save bindings failed: %s", exc)


def load_case_bindings(artifact_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(str(artifact_dir or ""), CASE_BINDINGS_FILENAME)
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, Mapping):
            return []
        sels = raw.get("selections")
        if not isinstance(sels, list):
            return []
        return merge_case_selections([], sels)
    except Exception as exc:  # noqa: BLE001
        logger.warning("case_mentions: load bindings failed: %s", exc)
        return []


def _clear_case_bindings_file(artifact_dir: str) -> None:
    path = os.path.join(str(artifact_dir or ""), CASE_BINDINGS_FILENAME)
    if not path or not os.path.isfile(path):
        return
    try:
        os.remove(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("case_mentions: clear bindings file failed: %s", exc)


def _storage_auth_headers(user_id: str = "") -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    try:
        from src.config import settings

        key = str(getattr(settings, "INTERNAL_API_KEY", "") or "").strip()
        if key:
            headers["X-Internal-Key"] = key
        headers["X-Service-Name"] = str(
            getattr(settings, "SERVICE_NAME", "") or "orchestrator"
        )
    except Exception:  # noqa: BLE001
        pass
    uid = str(user_id or "").strip()
    if uid:
        headers["X-User-Id"] = uid
    return headers


async def fetch_case_bindings_from_storage(
    *,
    session_id: str,
    user_id: str = "",
) -> List[Dict[str, Any]]:
    """Load bindings from chat_sessions.metadata (survives container restart)."""
    sid = str(session_id or "").strip()
    if not sid:
        return []
    try:
        import httpx
        from src.config import settings

        url = f"{str(settings.STORAGE_URL).rstrip('/')}/chat/sessions/{sid}"
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url, headers=_storage_auth_headers(user_id))
        if resp.status_code >= 400:
            logger.warning(
                "case_mentions: storage get session failed status=%s session=%s",
                resp.status_code,
                sid,
            )
            return []
        data = resp.json()
        if not isinstance(data, Mapping):
            return []
        meta = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
        raw = meta.get(SESSION_CASE_BINDINGS_META_KEY)
        if not isinstance(raw, Mapping):
            return []
        sels = raw.get("selections")
        if not isinstance(sels, list):
            return []
        return merge_case_selections([], sels)
    except Exception as exc:  # noqa: BLE001
        logger.warning("case_mentions: storage load bindings failed: %s", exc)
        return []


async def persist_case_bindings_to_storage(
    *,
    session_id: str,
    user_id: str = "",
    selections: Sequence[Mapping[str, Any]],
) -> bool:
    """Merge bindings into chat_sessions.metadata (Postgres-backed)."""
    sid = str(session_id or "").strip()
    if not sid:
        return False
    payload = {
        SESSION_CASE_BINDINGS_META_KEY: _bindings_disk_payload(selections),
    }
    try:
        import httpx
        from src.config import settings

        url = f"{str(settings.STORAGE_URL).rstrip('/')}/chat/sessions/{sid}"
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.patch(
                url,
                json={"metadata": payload},
                headers=_storage_auth_headers(user_id),
            )
        if resp.status_code >= 400:
            logger.warning(
                "case_mentions: storage patch bindings failed status=%s session=%s",
                resp.status_code,
                sid,
            )
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("case_mentions: storage persist bindings failed: %s", exc)
        return False


async def load_case_bindings_durable(
    *,
    session_id: str,
    user_id: str,
    artifact_dir: str,
) -> List[Dict[str, Any]]:
    """Disk cache first; fall back to session metadata and hydrate disk."""
    local = load_case_bindings(artifact_dir)
    if local:
        return local
    remote = await fetch_case_bindings_from_storage(
        session_id=session_id, user_id=user_id
    )
    if remote:
        save_case_bindings(artifact_dir, remote)
        logger.info(
            "case_mentions hydrated bindings from storage session=%s cases=%s",
            session_id,
            [s.get("case_id") for s in remote],
        )
    return remote


async def persist_case_bindings_durable(
    *,
    session_id: str,
    user_id: str,
    selections: Sequence[Mapping[str, Any]],
    artifact_dir: str,
) -> None:
    """Write bindings to disk cache and session metadata."""
    merged = merge_case_selections([], selections)
    if merged:
        save_case_bindings(artifact_dir, merged)
    else:
        _clear_case_bindings_file(artifact_dir)
    ok = await persist_case_bindings_to_storage(
        session_id=session_id,
        user_id=user_id,
        selections=merged,
    )
    if not ok:
        logger.error(
            "case_mentions: durable storage persist FAILED session=%s cases=%s "
            "(disk cache may still work until restart)",
            session_id,
            [s.get("case_id") for s in merged],
        )


def _clear_case_inventory_file(artifact_dir: str) -> None:
    path = os.path.join(str(artifact_dir or ""), CASE_INVENTORY_FILENAME)
    if not path or not os.path.isfile(path):
        return
    try:
        os.remove(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("case_mentions: clear inventory failed: %s", exc)


def delete_case_binding(artifact_dir: str, case_id: str) -> List[Dict[str, Any]]:
    """Remove one case from durable bindings; return remaining selections."""
    cid = str(case_id or "").strip()
    prior = load_case_bindings(artifact_dir)
    if not cid:
        return prior
    remaining = [s for s in prior if str(s.get("case_id") or "").strip() != cid]
    if not remaining:
        path = os.path.join(str(artifact_dir or ""), CASE_BINDINGS_FILENAME)
        try:
            if path and os.path.isfile(path):
                os.remove(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("case_mentions: remove bindings file failed: %s", exc)
        _clear_case_inventory_file(artifact_dir)
        return []
    save_case_bindings(artifact_dir, remaining)
    return remaining


def _output_handles(outputs: Sequence[Mapping[str, Any]]) -> List[str]:
    handles: List[str] = []
    for row in outputs or []:
        if not isinstance(row, Mapping):
            continue
        h = str(row.get("handle") or "").strip()
        if h and h not in handles:
            handles.append(h)
    return handles


def _counts_for_case(
    inventory: Optional[Mapping[str, Any]],
    case_id: str,
) -> Tuple[int, int]:
    cid = str(case_id or "").strip()
    if not isinstance(inventory, Mapping) or not cid:
        return 0, 0
    files = 0
    texts = 0
    for row in inventory.get("files") or []:
        if isinstance(row, Mapping) and str(row.get("case_id") or "").strip() == cid:
            files += 1
    for row in inventory.get("texts") or []:
        if isinstance(row, Mapping) and str(row.get("case_id") or "").strip() == cid:
            texts += 1
    # Fallback when inventory only has union totals for a single case.
    if files == 0 and texts == 0:
        case_ids = [
            str(x).strip() for x in (inventory.get("case_ids") or []) if str(x).strip()
        ]
        if case_ids == [cid]:
            files = len([r for r in (inventory.get("files") or []) if isinstance(r, Mapping)])
            texts = len([r for r in (inventory.get("texts") or []) if isinstance(r, Mapping)])
    return files, texts


def summarize_case_bindings(
    selections: Sequence[Mapping[str, Any]],
    inventory: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """API-facing binding rows (handles + counts, no file/text payloads)."""
    out: List[Dict[str, Any]] = []
    for sel in selections or []:
        if not isinstance(sel, Mapping):
            continue
        cid = str(sel.get("case_id") or "").strip()
        if not cid:
            continue
        handles = _output_handles(list(sel.get("outputs") or []))
        file_count, evidence_count = _counts_for_case(inventory, cid)
        out.append(
            {
                "case_id": cid,
                "label": str(sel.get("label") or cid),
                "outputs": list(sel.get("outputs") or []),
                "handles": handles,
                "emit_mode": str(sel.get("emit_mode") or "batch"),
                "file_count": file_count,
                "evidence_count": evidence_count,
            }
        )
    return out


async def list_case_bindings_for_session(
    *,
    session_id: str,
    user_id: str,
    artifact_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Return durable bindings + counts for the FE Bound Cases UI."""
    art = str(artifact_dir or "").strip() or hub_chat_artifact_dir(session_id, user_id)
    selections = await load_case_bindings_durable(
        session_id=session_id, user_id=user_id, artifact_dir=art
    )
    inventory = load_case_inventory(art)
    if inventory is None and selections:
        inventory = await build_case_mentions_inventory(
            _context_from_case_selections(selections),
            user_id=user_id,
        )
        if inventory_has_cases(inventory):
            save_case_inventory(art, inventory, selections=selections)
    bindings = summarize_case_bindings(selections, inventory)
    return {
        "ok": True,
        "session_id": str(session_id or ""),
        "bindings": bindings,
        "count": len(bindings),
        "files_truncated": bool(
            isinstance(inventory, Mapping) and inventory.get("files_truncated")
        ),
        "files_total": (
            int(inventory.get("files_total") or 0)
            if isinstance(inventory, Mapping)
            else 0
        ),
    }


async def unbind_case_for_session(
    *,
    session_id: str,
    user_id: str,
    case_id: str,
    artifact_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Drop one case binding and rebuild (or clear) durable inventory."""
    art = str(artifact_dir or "").strip() or hub_chat_artifact_dir(session_id, user_id)
    cid = str(case_id or "").strip()
    prior = await load_case_bindings_durable(
        session_id=session_id, user_id=user_id, artifact_dir=art
    )
    remaining = [
        s for s in prior if str(s.get("case_id") or "").strip() != cid
    ] if cid else list(prior)
    await persist_case_bindings_durable(
        session_id=session_id,
        user_id=user_id,
        selections=remaining,
        artifact_dir=art,
    )
    if not remaining:
        _clear_case_inventory_file(art)
        return await list_case_bindings_for_session(
            session_id=session_id, user_id=user_id, artifact_dir=art
        )
    inv = await build_case_mentions_inventory(
        _context_from_case_selections(remaining),
        user_id=user_id,
    )
    if inventory_has_cases(inv):
        save_case_inventory(art, inv, selections=remaining)
    else:
        _clear_case_inventory_file(art)
    return await list_case_bindings_for_session(
        session_id=session_id, user_id=user_id, artifact_dir=art
    )


def save_case_inventory(
    artifact_dir: str,
    inventory: Mapping[str, Any],
    *,
    selections: Optional[Sequence[Mapping[str, Any]]] = None,
) -> None:
    path = os.path.join(str(artifact_dir or ""), CASE_INVENTORY_FILENAME)
    if not artifact_dir or not inventory_has_cases(inventory):
        return
    try:
        os.makedirs(artifact_dir, exist_ok=True)
        payload: Dict[str, Any] = {
            "schema_version": 1,
            "inventory": {
                "files": list(inventory.get("files") or []),
                "texts": list(inventory.get("texts") or []),
                "case_ids": list(inventory.get("case_ids") or []),
                "count": inventory.get("count"),
                "note": inventory.get("note") or "hub_chat_case_mentions",
                "files_truncated": bool(inventory.get("files_truncated")),
                "files_total": inventory.get("files_total"),
            },
        }
        if selections is not None:
            payload["selections"] = merge_case_selections([], selections)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("case_mentions: save inventory failed: %s", exc)


def load_case_inventory(artifact_dir: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(str(artifact_dir or ""), CASE_INVENTORY_FILENAME)
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, Mapping):
            return None
        inv = raw.get("inventory")
        if not isinstance(inv, Mapping):
            return None
        out = host_pools.empty_inventory(note=str(inv.get("note") or "hub_chat_case_restored"))
        out["files"] = [dict(r) for r in (inv.get("files") or []) if isinstance(r, Mapping)]
        out["texts"] = [dict(r) for r in (inv.get("texts") or []) if isinstance(r, Mapping)]
        out["case_ids"] = [
            str(x).strip() for x in (inv.get("case_ids") or []) if str(x).strip()
        ]
        out["count"] = len(out["files"]) + len(out["texts"])
        if inv.get("files_truncated"):
            out["files_truncated"] = True
            try:
                out["files_total"] = int(inv.get("files_total") or len(out["files"]))
            except (TypeError, ValueError):
                out["files_total"] = len(out["files"])
        return out if inventory_has_cases(out) else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("case_mentions: load inventory failed: %s", exc)
        return None


async def restore_or_build_case_inventory(
    *,
    context: Optional[Mapping[str, Any]],
    user_id: str,
    session_id: str,
    artifact_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Build from this-turn mentions, else reload durable bindings/inventory.

    Mentions are only present on the turn the user picks a case; later turns
    restore from disk cache or ``chat_sessions.metadata`` (survives restart).
    """
    art = str(artifact_dir or "").strip() or hub_chat_artifact_dir(session_id, user_id)
    selections = extract_case_mention_selections(context)
    if selections:
        # Merge with any prior bindings so adding a second case keeps the first.
        prior = await load_case_bindings_durable(
            session_id=session_id, user_id=user_id, artifact_dir=art
        )
        merged = merge_case_selections(prior, selections)
        inv = await build_case_mentions_inventory(
            _context_from_case_selections(merged),
            user_id=user_id,
        )
        await persist_case_bindings_durable(
            session_id=session_id,
            user_id=user_id,
            selections=merged,
            artifact_dir=art,
        )
        save_case_inventory(art, inv, selections=merged)
        return inv

    cached = load_case_inventory(art)
    if cached is not None:
        logger.info(
            "case_mentions restored inventory from disk cases=%s files=%s",
            cached.get("case_ids"),
            len(cached.get("files") or []),
        )
        return cached

    prior = await load_case_bindings_durable(
        session_id=session_id, user_id=user_id, artifact_dir=art
    )
    if prior:
        inv = await build_case_mentions_inventory(
            _context_from_case_selections(prior),
            user_id=user_id,
        )
        if inventory_has_cases(inv):
            save_case_inventory(art, inv, selections=prior)
            return inv

    return host_pools.empty_inventory(note="hub_chat_case_mentions")


def _context_from_case_selections(
    selections: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        CHAT_MENTIONS_CONTEXT_KEY: {
            "schema_version": 1,
            "cases": [
                {
                    "id": str(s.get("case_id") or ""),
                    "label": str(s.get("label") or s.get("case_id") or ""),
                    "outputs": list(s.get("outputs") or []),
                    "emit_mode": str(s.get("emit_mode") or "batch"),
                }
                for s in selections
                if str(s.get("case_id") or "").strip()
            ],
        }
    }


def extract_case_mention_selections(
    context: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Return ``[{case_id, label, outputs, emit_mode}, ...]`` with non-empty outputs."""
    if not isinstance(context, Mapping):
        return []
    mentions = context.get(CHAT_MENTIONS_CONTEXT_KEY)
    if not isinstance(mentions, Mapping):
        return []
    cases = mentions.get("cases")
    if not isinstance(cases, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in cases:
        if not isinstance(row, Mapping):
            continue
        cid = str(row.get("id") or row.get("case_id") or "").strip()
        if not cid:
            continue
        outputs = row.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            # Legacy chip without dialog — treat as all artifacts.
            outputs = [{"handle": "artifacts", "mode": "artifacts"}]
        out.append(
            {
                "case_id": cid,
                "label": str(row.get("label") or cid).strip() or cid,
                "outputs": list(outputs),
                "emit_mode": str(row.get("emit_mode") or "batch").strip() or "batch",
            }
        )
    return out


def _evidence_to_text(item: Mapping[str, Any], *, handle: str) -> Optional[Dict[str, Any]]:
    try:
        from modules.plugins.agent_flow.case_node import format_case_evidence_text
    except Exception:
        format_case_evidence_text = None  # type: ignore

    body = ""
    if format_case_evidence_text is not None:
        try:
            body = format_case_evidence_text(item) or ""
        except Exception:
            body = ""
    if not body:
        channels = item.get("channels") if isinstance(item.get("channels"), Mapping) else {}
        for key in ("rawdata", "metadata", "llm"):
            chunk = channels.get(key) if isinstance(channels, Mapping) else None
            if chunk is None:
                continue
            if isinstance(chunk, str) and chunk.strip():
                body = chunk.strip()
                break
            try:
                import json as _json

                body = _json.dumps(chunk, ensure_ascii=False, default=str)[:120_000]
                break
            except Exception:
                continue
    if not body:
        return None
    tool = str(item.get("tool_id") or "").strip()
    if not tool and "tool:" in handle:
        tool = handle.split("tool:")[-1].strip()
    src_name = str(
        item.get("source_filename")
        or item.get("filename")
        or item.get("source_artifact_id")
        or "evidence"
    ).strip()
    filename = host_pools.safe_pool_filename(
        f"{tool or 'evidence'}__{src_name}.md", fallback="evidence.md"
    )
    return {
        "kind": "text",
        "source": "case_tool_evidence",
        "handle": handle,
        "content": body[:200_000],
        "filename": filename,
        "media_type": "text",
        "tool_id": tool or None,
        "source_artifact_id": str(item.get("source_artifact_id") or "").strip() or None,
        "case_id": str(item.get("case_id") or "").strip() or None,
    }


def _ref_to_file_row(ref: Mapping[str, Any], *, case_id: str) -> Optional[Dict[str, Any]]:
    aid = str(ref.get("artifact_id") or ref.get("id") or "").strip()
    if not aid:
        return None
    name = (
        ref.get("original_filename")
        or ref.get("filename")
        or ref.get("name")
        or aid
    )
    return {
        "filename": host_pools.safe_pool_filename(str(name)),
        "original_filename": str(name),
        "artifact_id": aid,
        "path": None,
        "mime_type": ref.get("mime_type") or ref.get("content_type"),
        "size_bytes": ref.get("file_size_bytes") or ref.get("size_bytes") or ref.get("size"),
        "media_type": ref.get("media_type") or ref.get("kind"),
        "kind": ref.get("kind") or ref.get("media_type"),
        "case_id": case_id,
        "source": "case_mention",
    }


async def resolve_case_selection_inventory(
    *,
    case_id: str,
    user_id: str,
    outputs: Sequence[Mapping[str, Any]],
    label: str = "",
) -> Dict[str, Any]:
    """Resolve one case selection into pool inventory files + texts."""
    inv = host_pools.empty_inventory(
        note=f"case_mention:{label or case_id}"
    )
    cid = (case_id or "").strip()
    uid = (user_id or "").strip()
    if not cid or not uid:
        return inv

    try:
        from modules.plugins.agent_flow.case_node import resolve_case_handle_outputs
        from modules.plugins.agent_flow.case_schema import normalize_case_outputs
    except Exception as exc:
        logger.warning("case_mentions: agent_flow case_node unavailable: %s", exc)
        # Fallback: fetch all artifacts only.
        try:
            from modules.plugins.agent_flow.case_artifacts import fetch_case_artifacts

            raw = await fetch_case_artifacts(cid, user_id=uid)
        except Exception as exc2:
            logger.warning("case_mentions: fetch_case_artifacts failed: %s", exc2)
            inv["case_ids"] = [cid]
            inv["ok"] = False
            inv["note"] = f"fetch_failed:{exc2}"
            return inv
        files: List[Dict[str, Any]] = []
        for row in raw or []:
            if isinstance(row, Mapping):
                fr = _ref_to_file_row(row, case_id=cid)
                if fr:
                    files.append(fr)
        inv["files"] = files
        inv["case_ids"] = [cid]
        inv["count"] = len(files)
        return inv

    specs = normalize_case_outputs(list(outputs))
    if not specs:
        specs = [{"handle": "artifacts", "mode": "artifacts"}]

    try:
        handles = await resolve_case_handle_outputs(
            case_id=cid,
            user_id=uid,
            outputs=specs,
        )
    except Exception as exc:
        logger.warning(
            "case_mentions: resolve_case_handle_outputs failed case=%s: %s",
            cid,
            exc,
        )
        inv["case_ids"] = [cid]
        inv["ok"] = False
        inv["note"] = f"resolve_failed:{exc}"
        return inv

    files: List[Dict[str, Any]] = []
    texts: List[Dict[str, Any]] = []
    seen_aids: set[str] = set()

    for handle, value in (handles or {}).items():
        h = str(handle or "")
        is_evidence = h.startswith("tool:") or "/tool:" in h
        items = value if isinstance(value, list) else ([value] if value else [])
        for item in items:
            if not isinstance(item, Mapping):
                continue
            if is_evidence:
                text_row = _evidence_to_text(item, handle=h)
                if text_row:
                    text_row["case_id"] = cid
                    texts.append(text_row)
                continue
            fr = _ref_to_file_row(item, case_id=cid)
            if not fr:
                continue
            aid = str(fr.get("artifact_id") or "")
            if aid and aid in seen_aids:
                continue
            if aid:
                seen_aids.add(aid)
            files.append(fr)

    inv["files"] = files
    inv["texts"] = texts
    inv["case_ids"] = [cid]
    inv["count"] = len(files) + len(texts)
    return inv


async def build_case_mentions_inventory(
    context: Optional[Mapping[str, Any]],
    *,
    user_id: str,
) -> Dict[str, Any]:
    """Union inventory for all case mentions with outputs in ``context``."""
    selections = extract_case_mention_selections(context)
    union = host_pools.empty_inventory(note="hub_chat_case_mentions")
    if not selections:
        return union

    files: List[Dict[str, Any]] = []
    texts: List[Dict[str, Any]] = []
    case_ids: List[str] = []
    seen_aids: set[str] = set()
    seen_text_keys: set[str] = set()

    for sel in selections:
        part = await resolve_case_selection_inventory(
            case_id=sel["case_id"],
            user_id=user_id,
            outputs=sel["outputs"],
            label=sel.get("label") or "",
        )
        for cid in part.get("case_ids") or []:
            text = str(cid).strip()
            if text and text not in case_ids:
                case_ids.append(text)
        for row in part.get("files") or []:
            if not isinstance(row, Mapping):
                continue
            aid = str(row.get("artifact_id") or "").strip()
            key = aid or str(row.get("filename") or "")
            if not key or key in seen_aids:
                continue
            seen_aids.add(key)
            files.append(dict(row))
        for row in part.get("texts") or []:
            if not isinstance(row, Mapping):
                continue
            key = f"{row.get('filename')}:{row.get('handle')}:{row.get('source_artifact_id')}"
            if key in seen_text_keys:
                continue
            seen_text_keys.add(key)
            texts.append(dict(row))

    union["files"] = files
    union["texts"] = texts
    union["case_ids"] = case_ids
    if _INVENTORY_FILE_CAP and len(files) > _INVENTORY_FILE_CAP:
        union["files_total"] = len(files)
        union["files_truncated"] = True
        union["files"] = files[:_INVENTORY_FILE_CAP]
        logger.info(
            "case_mentions inventory truncated files=%s kept=%s texts=%s",
            union["files_total"],
            _INVENTORY_FILE_CAP,
            len(texts),
        )
    union["count"] = len(union["files"]) + len(texts)
    logger.info(
        "case_mentions inventory cases=%s files=%s texts=%s truncated=%s",
        len(case_ids),
        len(union["files"]),
        len(texts),
        bool(union.get("files_truncated")),
    )
    return union


def merge_inventories(*parts: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge pool inventories (files / texts / case_ids) with dedupe."""
    out = host_pools.empty_inventory(note="merged")
    files: List[Dict[str, Any]] = []
    texts: List[Dict[str, Any]] = []
    case_ids: List[str] = []
    seen_f: set[str] = set()
    seen_t: set[str] = set()
    notes: List[str] = []

    for part in parts:
        if not isinstance(part, Mapping):
            continue
        note = str(part.get("note") or "").strip()
        if note:
            notes.append(note)
        for cid in part.get("case_ids") or []:
            text = str(cid).strip()
            if text and text not in case_ids:
                case_ids.append(text)
        for row in part.get("files") or []:
            if not isinstance(row, Mapping):
                continue
            key = str(row.get("artifact_id") or row.get("path") or row.get("filename") or "")
            if not key or key in seen_f:
                continue
            seen_f.add(key)
            files.append(dict(row))
        for row in part.get("texts") or []:
            if not isinstance(row, Mapping):
                continue
            key = f"{row.get('filename')}:{row.get('handle')}:{row.get('content', '')[:64]}"
            if key in seen_t:
                continue
            seen_t.add(key)
            texts.append(dict(row))

    out["files"] = files
    out["texts"] = texts
    out["case_ids"] = case_ids
    out["count"] = len(files) + len(texts)
    if notes:
        out["note"] = " + ".join(notes[:6])
    return out


def case_files_for_session_registry(
    inventory: Mapping[str, Any],
    *,
    cap: int = _SESSION_REGISTRY_CAP,
) -> List[Dict[str, Any]]:
    """Slim registry rows for native tools (capped)."""
    rows: List[Dict[str, Any]] = []
    for row in list(inventory.get("files") or [])[: max(0, int(cap))]:
        if not isinstance(row, Mapping):
            continue
        aid = str(row.get("artifact_id") or "").strip()
        if not aid:
            continue
        rows.append(
            {
                "artifact_id": aid,
                "id": aid,
                "name": row.get("original_filename") or row.get("filename") or aid,
                "filename": row.get("original_filename") or row.get("filename") or aid,
                "mime_type": row.get("mime_type"),
                "media_type": row.get("media_type") or row.get("kind"),
                "case_id": row.get("case_id"),
                "source": "case_mention",
            }
        )
    return rows


def case_mentions_prompt_addon(inventory: Mapping[str, Any]) -> str:
    """Extra host-prompt lines when case outputs were materialized."""
    files = [r for r in (inventory.get("files") or []) if isinstance(r, Mapping)]
    texts = [r for r in (inventory.get("texts") or []) if isinstance(r, Mapping)]
    case_ids = [str(x).strip() for x in (inventory.get("case_ids") or []) if str(x).strip()]
    if not files and not texts and not case_ids:
        return ""
    lines = [
        "CASE OUTPUTS LOADED (host materialised — prefer pool / workspace over re-attach)",
        f"- case_ids: {', '.join(case_ids) if case_ids else '(none)'}",
        f"- pool files: {len(files)}"
        + (
            f" (sample of {inventory.get('files_total')}; truncated)"
            if inventory.get("files_truncated")
            else ""
        ),
        f"- pool evidence texts: {len(texts)}",
    ]
    if files:
        sample = ", ".join(
            str(r.get("original_filename") or r.get("filename") or r.get("artifact_id"))
            for r in files[:8]
        )
        lines.append(f"- sample files: {sample}")
    if texts:
        sample = ", ".join(str(r.get("filename") or "evidence.md") for r in texts[:6])
        lines.append(f"- evidence: {sample}")
    lines.append(
        "  → RLM: use pool tools / `ls pool` / open files under pool/. "
        "Case files are NOT attached as Session Files — use the case pool. "
        "Native: text_search scope=case still works; registry attach is disabled by default."
    )
    return "\n".join(lines)
