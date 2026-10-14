"""Hub-chat host adapter: durable vs ephemeral engine sessions.

Engines declare ``session_binding`` on their manifest. Hosts decide when to
``create``/``run`` vs ``continue`` so durable engines (RLM kernel/memory) are
not restarted on every chat message.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionBinding:
    mode: str
    key_from: str = "chat_session_id"
    engine_session_prefix: str = ""
    first_op: str = "run"
    next_op: str = "continue"
    stores: Optional[List[str]] = None
    raw: Optional[Dict[str, Any]] = None

    @property
    def is_durable(self) -> bool:
        return self.mode == "durable"

    @property
    def is_ephemeral_turn(self) -> bool:
        return self.mode == "ephemeral_turn"


def read_session_binding(manifest: Any) -> Optional[SessionBinding]:
    """Parse session_binding from a Manifest or raw dict."""
    if manifest is None:
        return None
    if isinstance(manifest, Mapping):
        raw = manifest.get("session_binding")
        memory = list(manifest.get("memory") or [])
    else:
        raw = getattr(manifest, "session_binding", None)
        memory = list(getattr(manifest, "memory", None) or [])
    if not isinstance(raw, Mapping):
        # Infer durable when lifecycle/memory already advertise kernel retention.
        lifecycle = (
            str(manifest.get("lifecycle") or "")
            if isinstance(manifest, Mapping)
            else str(getattr(manifest, "lifecycle", "") or "")
        )
        if lifecycle == "daemon" or "kernel" in {str(m).strip() for m in memory}:
            return SessionBinding(
                mode="durable",
                key_from="chat_session_id",
                engine_session_prefix="rlm_chat_",
                first_op="run",
                next_op="continue",
                stores=[str(m) for m in memory] or ["session", "kernel"],
            )
        return None
    mode = str(raw.get("mode") or "").strip()
    if not mode:
        return None
    stores_raw = raw.get("stores")
    stores: Optional[List[str]] = None
    if isinstance(stores_raw, list):
        stores = [str(x).strip() for x in stores_raw if str(x or "").strip()]
    return SessionBinding(
        mode=mode,
        key_from=str(raw.get("key_from") or "chat_session_id").strip()
        or "chat_session_id",
        engine_session_prefix=str(raw.get("engine_session_prefix") or "").strip(),
        first_op=str(raw.get("first_op") or "run").strip() or "run",
        next_op=str(raw.get("next_op") or "continue").strip() or "continue",
        stores=stores,
        raw=dict(raw),
    )


def _safe_key(raw: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw or "").strip())
    return (text[:180] or "session")


def engine_session_id_for(
    binding: Optional[SessionBinding],
    *,
    chat_session_id: str,
    orchestrator_id: str = "",
) -> str:
    """Stable engine session id derived from the host chat session."""
    chat = _safe_key(chat_session_id)
    if binding is None or binding.is_ephemeral_turn:
        # Caller may still append entropy; this is only a base.
        return chat
    prefix = binding.engine_session_prefix or (
        f"{_safe_key(orchestrator_id)}_" if orchestrator_id else ""
    )
    return f"{prefix}{chat}"


def rebind_host_ports(runtime_session: Any, host: Any) -> None:
    """Refresh host callables on a live RlmSession between chat turns."""
    if runtime_session is None or host is None:
        return
    for attr, value in (
        ("chat_fn", getattr(host, "chat", None)),
        ("chat_stream_fn", getattr(host, "chat_stream", None)),
        ("tool_runner", getattr(host, "tools", None)),
        ("pool_search", getattr(host, "pool_search", None)),
        ("pool_resolve", getattr(host, "pool_resolve", None)),
        ("pool_list", getattr(host, "pool_list", None)),
        ("pool_read", getattr(host, "pool_read", None)),
        ("on_event", getattr(host, "events", None)),
        ("spawn_child", getattr(host, "spawn_child", None)),
        ("model_finder", getattr(host, "model_finder", None)),
    ):
        if value is not None and hasattr(runtime_session, attr):
            try:
                setattr(runtime_session, attr, value)
            except Exception:
                logger.debug("rebind_host_ports skip attr=%s", attr, exc_info=True)


def lookup_live_runtime_session(engine_session_id: str) -> Any:
    """Return a live RlmSession from the process daemon, or None."""
    sid = str(engine_session_id or "").strip()
    if not sid:
        return None
    try:
        from src.orchestrators.rlm.runtime.daemon import RlmDaemon

        sess = RlmDaemon.instance().get(sid)
    except Exception:
        logger.debug("session_binding daemon lookup failed", exc_info=True)
        return None
    if sess is None:
        return None
    if getattr(sess, "_closed", False):
        return None
    return sess


def hub_rlm_engine_session_id(chat_session_id: str) -> str:
    """Stable RLM engine id for a hub chat session (matches rlm_chat_stream)."""
    return engine_session_id_for(
        SessionBinding(
            mode="durable",
            key_from="chat_session_id",
            engine_session_prefix="rlm_chat_",
            first_op="run",
            next_op="continue",
            stores=["session", "kernel"],
        ),
        chat_session_id=chat_session_id,
        orchestrator_id="rlm",
    )


def hub_rlm_live_snapshot(chat_session_id: str) -> Dict[str, Any]:
    """AF-style live snapshot for hub RLM chat (logs + answer + status).

    Returns ``{ok, live, status, engine_session_id, snapshot|None, …}``.
    Used when the user leaves the chat page and returns — the SSE pipe is gone
    but the durable kernel may still be running (or finished with logs on disk).
    """
    chat = str(chat_session_id or "").strip()
    engine_sid = hub_rlm_engine_session_id(chat) if chat else ""
    empty: Dict[str, Any] = {
        "ok": True,
        "live": False,
        "status": "absent",
        "chat_session_id": chat,
        "engine_session_id": engine_sid,
        "snapshot": None,
    }
    if not chat or not engine_sid:
        return empty

    sess = lookup_live_runtime_session(engine_sid)
    if sess is None:
        # Also try daemon.get even if _closed — finished sessions may still hold logs.
        try:
            from src.orchestrators.rlm.runtime.daemon import RlmDaemon

            sess = RlmDaemon.instance().get(engine_sid)
        except Exception:
            sess = None
    if sess is None:
        return empty

    try:
        snap = sess.live_progress_snapshot()
    except Exception as exc:
        logger.warning("hub_rlm_live_snapshot failed sid=%s: %s", engine_sid, exc)
        return {
            **empty,
            "ok": False,
            "status": "error",
            "error": str(exc)[:500],
        }

    status = str(snap.get("status") or getattr(sess, "status", "") or "").strip() or "unknown"
    live = bool(snap.get("live"))
    # Treat pause/idle chat lifetime as "waiting for next user message" not running.
    if status in ("paused", "idle", "waiting"):
        live = False
    logs = snap.get("log_lines") if isinstance(snap.get("log_lines"), list) else []
    return {
        "ok": True,
        "live": live,
        "status": status,
        "chat_session_id": chat,
        "engine_session_id": engine_sid,
        "error": snap.get("error"),
        "last_content": str(snap.get("last_content") or ""),
        "last_reasoning": str(snap.get("last_reasoning") or ""),
        "log_lines": list(logs),
        "log_count": len(logs),
        "files": snap.get("files") if isinstance(snap.get("files"), list) else [],
        "updated_at": snap.get("updated_at"),
        "snapshot": snap,
    }


def register_runtime_session(runtime_session: Any) -> None:
    """Register a newly created RlmSession with the process daemon."""
    if runtime_session is None:
        return
    try:
        from src.orchestrators.rlm.runtime.daemon import RlmDaemon

        RlmDaemon.instance().register(runtime_session)
    except Exception:
        logger.warning("session_binding daemon register failed", exc_info=True)
