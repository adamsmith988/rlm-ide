"""Record an Agent Flow result into a normal chat session.

Agent Flow plugin sessions/runs are untouched. This module only:
- links result artifacts into the *chat* workspace
- appends user + assistant messages with ``metadata.source=agent_flow``

Used after the frontend finishes a real agent-flow run so the orchestrator
can later discuss those results via ``/chat/stream``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from .config import settings

log = logging.getLogger(__name__)


class AgentTurnRef(BaseModel):
    flow_id: str
    flow_name: Optional[str] = None
    run_id: str
    flow_session_id: Optional[str] = None


class AgentTurnArtifact(BaseModel):
    artifact_id: str
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: Optional[int] = None


class AgentTurnAttachment(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    path: Optional[str] = None
    mime_type: Optional[str] = None
    size: Optional[int] = None


class AgentTurnRequest(BaseModel):
    user_id: str
    user_content: str = ""
    assistant_content: str = ""
    user_attachments: List[AgentTurnAttachment] = Field(default_factory=list)
    result_artifacts: List[AgentTurnArtifact] = Field(default_factory=list)
    agent: AgentTurnRef


class AgentTurnResponse(BaseModel):
    user_message: Dict[str, Any]
    assistant_message: Dict[str, Any]
    attached_artifacts: List[Dict[str, Any]] = Field(default_factory=list)


def _storage_headers(user_id: Optional[str] = None) -> Dict[str, str]:
    headers = {
        "X-Internal-Key": settings.INTERNAL_API_KEY,
        "X-Service-Name": settings.SERVICE_NAME,
    }
    if user_id:
        headers["X-User-Id"] = str(user_id)
    return headers


async def _storage_request(
    method: str,
    endpoint: str,
    *,
    json: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    timeout: float = 15.0,
) -> Any:
    url = f"{settings.STORAGE_URL}{endpoint}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.request(
            method=method,
            url=url,
            json=json,
            params=params,
            headers=_storage_headers(user_id),
        )
        if response.status_code >= 400:
            response.raise_for_status()
        if response.status_code == 204 or not response.content:
            return None
        return response.json()


def _normalize_user_attachment(att: AgentTurnAttachment) -> Dict[str, Any]:
    aid = (att.id or att.path or "").strip()
    return {
        "id": aid or None,
        "path": (att.path or aid or "").strip() or None,
        "name": (att.name or "").strip() or None,
        "mime_type": att.mime_type,
        "size": att.size,
    }


def _attachment_from_file_info(
    file_info: Dict[str, Any],
    fallback: Optional[AgentTurnArtifact] = None,
) -> Dict[str, Any]:
    aid = str(
        file_info.get("id")
        or file_info.get("artifact_id")
        or (fallback.artifact_id if fallback else "")
        or ""
    ).strip()
    name = str(
        file_info.get("name")
        or file_info.get("filename")
        or (fallback.filename if fallback else "")
        or aid
    ).strip()
    mime = (
        file_info.get("mime_type")
        or (fallback.mime_type if fallback else None)
    )
    size = file_info.get("size") or file_info.get("size_bytes")
    if size is None and fallback is not None:
        size = fallback.size_bytes
    return {
        "id": aid,
        "path": aid,
        "name": name or aid,
        "mime_type": mime,
        "size": size,
    }


async def attach_result_artifact(
    *,
    session_id: str,
    user_id: str,
    artifact: AgentTurnArtifact,
) -> Dict[str, Any]:
    """Link an existing storage artifact into the chat session workspace."""
    aid = (artifact.artifact_id or "").strip()
    if not aid:
        raise ValueError("artifact_id_required")
    payload: Dict[str, Any] = {
        "user_id": str(user_id),
        "artifact_id": aid,
    }
    if artifact.filename:
        payload["filename"] = artifact.filename
    resp = await _storage_request(
        "POST",
        f"/chat/sessions/{session_id}/artifacts/attach",
        json=payload,
        user_id=user_id,
    )
    if not isinstance(resp, dict):
        return _attachment_from_file_info({}, artifact)
    if resp.get("status") == "ambiguous":
        # Prefer exact artifact_id; if storage still returns ambiguous, keep locator.
        log.warning(
            "agent_turn attach ambiguous session=%s artifact_id=%s",
            session_id,
            aid,
        )
        return _attachment_from_file_info({}, artifact)
    file_info = resp.get("file") if isinstance(resp.get("file"), dict) else {}
    if not file_info:
        file_info = {
            "id": resp.get("artifact_id") or aid,
            "artifact_id": resp.get("artifact_id") or aid,
            "filename": artifact.filename,
            "mime_type": artifact.mime_type,
            "size_bytes": artifact.size_bytes,
        }
    return _attachment_from_file_info(file_info, artifact)


async def get_chat_session(session_id: str, *, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    try:
        result = await _storage_request(
            "GET",
            f"/chat/sessions/{session_id}",
            user_id=user_id,
        )
    except httpx.HTTPStatusError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return None
        raise
    return result if isinstance(result, dict) else None


def session_owned_by(session: Dict[str, Any], user_id: str) -> bool:
    owner = str(session.get("user_id") or "").strip()
    want = str(user_id or "").strip()
    return bool(owner) and bool(want) and owner == want


async def record_agent_turn(
    *,
    session_id: str,
    request: AgentTurnRequest,
    allow_foreign_session: bool = False,
) -> AgentTurnResponse:
    """Persist one agent-backed chat turn (user + assistant) into storage."""
    sid = (session_id or "").strip()
    if not sid:
        raise ValueError("session_id_required")
    uid = (request.user_id or "").strip()
    if not uid:
        raise ValueError("user_id_required")
    if not (request.agent.flow_id or "").strip():
        raise ValueError("flow_id_required")
    if not (request.agent.run_id or "").strip():
        raise ValueError("run_id_required")

    session = await get_chat_session(sid, user_id=uid)
    if not session:
        raise LookupError("session_not_found")
    if not allow_foreign_session and not session_owned_by(session, uid):
        raise PermissionError("session_access_denied")

    attached: List[Dict[str, Any]] = []
    seen_aids: set[str] = set()
    for art in request.result_artifacts:
        aid = (art.artifact_id or "").strip()
        if not aid or aid in seen_aids:
            continue
        seen_aids.add(aid)
        try:
            row = await attach_result_artifact(
                session_id=sid, user_id=uid, artifact=art
            )
            attached.append(row)
        except Exception as exc:
            log.warning(
                "agent_turn attach failed session=%s artifact_id=%s error=%s",
                session_id,
                aid,
                exc,
            )
            # Still expose the locator on the assistant message so UI can link it.
            attached.append(_attachment_from_file_info({}, art))

    user_attachments = [
        _normalize_user_attachment(a)
        for a in request.user_attachments
        if (a.id or a.path or a.name)
    ]

    user_content = (request.user_content or "").strip() or "(agent run)"
    assistant_content = (request.assistant_content or "").strip()

    metadata = {
        "source": "agent_flow",
        "flow_id": request.agent.flow_id,
        "flow_name": request.agent.flow_name,
        "run_id": request.agent.run_id,
        "flow_session_id": request.agent.flow_session_id,
    }

    from .chat_history import get_chat_history_client

    client = get_chat_history_client()
    user_message = await client.add_message(
        session_id=sid,
        role="user",
        content=user_content,
        attachments=user_attachments or None,
        user_id=uid,
    )
    if not user_message:
        raise RuntimeError("failed_to_save_user_message")

    assistant_message = await client.add_message(
        session_id=sid,
        role="assistant",
        content=assistant_content,
        attachments=attached or None,
        metadata=metadata,
        user_id=uid,
    )
    if not assistant_message:
        raise RuntimeError("failed_to_save_assistant_message")

    log.info(
        "agent_turn recorded session=%s flow_id=%s run_id=%s attached=%s",
        sid,
        request.agent.flow_id,
        request.agent.run_id,
        len(attached),
    )
    return AgentTurnResponse(
        user_message=user_message,
        assistant_message=assistant_message,
        attached_artifacts=attached,
    )
