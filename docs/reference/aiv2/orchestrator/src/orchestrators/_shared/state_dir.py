"""Durable per-chat-session directories for controller state."""

from __future__ import annotations

import os
import re
import tempfile

from src.config import settings

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _root() -> str:
    configured = str(getattr(settings, "ORCHESTRATOR_STATE_ROOT", "") or "").strip()
    if configured:
        return configured
    workspace = str(os.environ.get("TOOL_RUNNER_WORKSPACE_ROOT") or "").strip()
    if workspace:
        return os.path.join(workspace, "orchestrator_state")
    return os.path.join(tempfile.gettempdir(), "aiv2_orchestrator_state")


def session_state_dir(engine_id: str, user_id: str, session_id: str) -> str:
    uid = _SAFE.sub("_", str(user_id or "user"))[:80] or "user"
    sid = _SAFE.sub("_", str(session_id or "session"))[:180] or "session"
    path = os.path.join(_root(), _SAFE.sub("_", engine_id), uid, sid)
    os.makedirs(path, exist_ok=True)
    return path
