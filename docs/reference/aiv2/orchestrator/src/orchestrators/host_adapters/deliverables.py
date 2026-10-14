"""Hub-chat host adapter: RLM kernel deliverables → chat session files.

Engines stay host-agnostic. After an RLM run the hub host uploads new/changed
workspace files to Storage and attaches them into the chat session so the
Files panel / session_files tools see them.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_TOOL_ID = "orchestrator_rlm"


def _runtime_session(orch_session: Any) -> Any:
    """Unwrap RlmOrchestratorSession → underlying RlmSession when present."""
    inner = getattr(orch_session, "_session", None)
    return inner if inner is not None else orch_session


def _sessions_to_flush(root: Any) -> List[Any]:
    out: List[Any] = []
    seen: set[int] = set()

    def _add(s: Any) -> None:
        if s is None:
            return
        key = id(s)
        if key in seen:
            return
        seen.add(key)
        out.append(s)

    _add(root)
    children = getattr(root, "_child_sessions", None)
    if isinstance(children, dict):
        for child in children.values():
            _add(child)
    return out


async def _upload_intents(
    intents: List[Dict[str, Any]],
    *,
    user_id: str,
    chat_session_id: str,
    run_id: str,
) -> List[Dict[str, Any]]:
    if not intents:
        return []
    try:
        from modules.tools.output_persistence import MAX_INTENTS, persist_outputs
    except Exception:
        logger.warning("deliverable upload: persist_outputs unavailable", exc_info=True)
        return []

    try:
        from src.config import settings

        internal_key = getattr(settings, "INTERNAL_API_KEY", None)
        service_name = getattr(settings, "SERVICE_NAME", None) or "orchestrator"
        # Ensure STORAGE_URL is visible to modules.tools helpers.
        if getattr(settings, "STORAGE_URL", None) and not os.environ.get("STORAGE_URL"):
            os.environ["STORAGE_URL"] = str(settings.STORAGE_URL)
    except Exception:
        internal_key = None
        service_name = "orchestrator"

    chunk_size = max(1, int(MAX_INTENTS))
    # persist_outputs requires an explicit plugin_id for new_blob (tool-runner
    # taxonomy segment). Hub RLM is not an AF plugin — stamp a stable host id.
    auth = {
        "user_id": str(user_id),
        "session_id": str(chat_session_id),
        "run_id": str(run_id or chat_session_id),
        "plugin_id": "orchestrator_rlm",
    }
    entries: List[Dict[str, Any]] = []
    for offset in range(0, len(intents), chunk_size):
        chunk = [dict(i) for i in intents[offset : offset + chunk_size] if isinstance(i, dict)]
        if not chunk:
            continue
        for intent in chunk:
            intent["bind_case"] = False
            intent.setdefault("layout", "default")
            intent.setdefault("plugin_id", "orchestrator_rlm")
            if run_id and not intent.get("run_id"):
                intent["run_id"] = run_id
        shim: Dict[str, Any] = {
            "output_intents": chunk,
            "run_id": str(run_id or chat_session_id),
        }
        try:
            await persist_outputs(
                tool_id=_TOOL_ID,
                result=shim,
                auth_context=auth,
                persist_profile={"plugin_id": "orchestrator_rlm", "bind_case": False},
                strict=False,
                internal_api_key=internal_key,
                service_name=service_name,
            )
        except Exception:
            logger.warning(
                "RLM deliverable upload failed session=%s offset=%s",
                chat_session_id,
                offset,
                exc_info=True,
            )
            continue
        persisted = shim.get("persisted")
        for e in persisted if isinstance(persisted, list) else []:
            if (
                isinstance(e, dict)
                and str(e.get("op") or "") == "new_blob"
                and str(e.get("artifact_id") or "").strip()
            ):
                entries.append(e)
    return entries


async def _attach_to_chat(
    *,
    chat_session_id: str,
    user_id: str,
    entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not entries:
        return []
    try:
        from src.agent_turn import AgentTurnArtifact, attach_result_artifact
    except Exception:
        logger.warning("deliverable attach: agent_turn unavailable", exc_info=True)
        return []

    attached: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        aid = str(entry.get("artifact_id") or "").strip()
        if not aid or aid in seen:
            continue
        seen.add(aid)
        art = AgentTurnArtifact(
            artifact_id=aid,
            filename=str(entry.get("filename") or "") or None,
            mime_type=str(entry.get("mime_type") or "") or None,
        )
        try:
            row = await attach_result_artifact(
                session_id=chat_session_id,
                user_id=user_id,
                artifact=art,
            )
            if isinstance(row, dict) and (row.get("id") or row.get("path")):
                attached.append(row)
        except Exception:
            logger.warning(
                "RLM deliverable attach failed session=%s artifact=%s",
                chat_session_id,
                aid,
                exc_info=True,
            )
    return attached


async def sync_rlm_deliverables_to_chat(
    *,
    orch_session: Any,
    chat_session_id: str,
    user_id: str,
    run_id: str = "",
) -> List[Dict[str, Any]]:
    """Upload RLM-produced files and link them into the chat session workspace.

    Returns attached file dicts suitable for ``session_file_attached`` SSE.
    Soft-fails: returns ``[]`` on any import/upload error so the chat answer
    still completes.
    """
    sid = str(chat_session_id or "").strip()
    uid = str(user_id or "").strip()
    if not sid or not uid:
        return []

    root = _runtime_session(orch_session)
    try:
        from src.orchestrators.rlm.runtime.persist import (
            persist_intents_for_session_async,
            persist_session_files,
            remember_session_deliverables,
        )
    except Exception:
        logger.warning("deliverable sync: persist helpers unavailable", exc_info=True)
        return []

    all_entries: List[Dict[str, Any]] = []
    for sess in _sessions_to_flush(root):
        try:
            persist_session_files(sess)
        except Exception:
            logger.debug("persist_session_files skipped", exc_info=True)
        try:
            intents = await persist_intents_for_session_async(sess)
        except Exception:
            logger.debug("pending deliverable intents failed", exc_info=True)
            continue
        if not intents:
            continue
        uploaded = await _upload_intents(
            intents,
            user_id=uid,
            chat_session_id=sid,
            run_id=str(run_id or sid),
        )
        if not uploaded:
            continue
        by_name = {
            str(i.get("filename") or ""): i
            for i in intents
            if isinstance(i, dict) and i.get("filename")
        }
        enriched: List[Dict[str, Any]] = []
        for e in uploaded:
            row = dict(e)
            src = by_name.get(str(e.get("filename") or ""))
            if isinstance(src, dict):
                if isinstance(src.get("metadata"), dict):
                    row["metadata"] = dict(src["metadata"])
                if src.get("local_path"):
                    row["local_path"] = src["local_path"]
            enriched.append(row)
            all_entries.append(row)
        try:
            remember_session_deliverables(sess, enriched)
            persist_session_files(sess)
        except Exception:
            logger.debug("remember deliverables failed", exc_info=True)

    return await _attach_to_chat(
        chat_session_id=sid,
        user_id=uid,
        entries=all_entries,
    )


def attached_sse_events(
    attached: List[Dict[str, Any]],
    *,
    base_registry_size: int = 0,
) -> List[Dict[str, Any]]:
    """Map attach rows → hub ``session_file_attached`` events."""
    events: List[Dict[str, Any]] = []
    size = max(0, int(base_registry_size or 0))
    for row in attached or []:
        if not isinstance(row, dict):
            continue
        size += 1
        events.append(
            {
                "type": "session_file_attached",
                "file": row,
                "registry_size": size,
            }
        )
    return events
