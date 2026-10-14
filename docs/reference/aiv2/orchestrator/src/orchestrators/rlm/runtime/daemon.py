"""Process-local RLM session daemon / supervisor."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

from src.orchestrators.rlm.runtime.constants import idle_eviction_minutes
from src.orchestrators.rlm.schema_utils import is_resident_run_lifetime
from src.orchestrators.rlm.runtime.persist import (
    list_persisted_public,
    persist_session_files,
)
from src.orchestrators.rlm.runtime.session import RlmSession

logger = logging.getLogger(__name__)


class RlmDaemon:
    """Singleton-ish supervisor for durable RLM sessions in this process."""

    _instance: Optional["RlmDaemon"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._sessions: Dict[str, RlmSession] = {}
        # One flow run may host multiple RLM agents + children that share run_id.
        self._by_run: Dict[str, List[str]] = {}
        self._mu = threading.RLock()
        self._bg_tasks: Dict[str, asyncio.Task] = {}
        self._hb_tasks: Dict[str, asyncio.Task] = {}
        self._evict_task: Optional[asyncio.Task] = None
        explicit = str(os.environ.get("AIV2_RLM_SESSION_DIR") or "").strip()
        if explicit:
            self.root_dir = explicit
        else:
            try:
                from modules.tools.run_workspace import namespace_scratch_root

                # Durable under TOOL_RUNNER_WORKSPACE_ROOT when set (PVC).
                self.root_dir = os.path.join(
                    namespace_scratch_root("agent_flow"), "rlm"
                )
            except Exception:
                self.root_dir = os.path.join(
                    tempfile.gettempdir(), "aiv2_rlm_sessions"
                )
        os.makedirs(self.root_dir, exist_ok=True)

    @classmethod
    def instance(cls) -> "RlmDaemon":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        with cls._lock:
            inst = cls._instance
            cls._instance = None
        if inst is None:
            return
        for task in list(inst._hb_tasks.values()) + list(inst._bg_tasks.values()):
            if task and not task.done():
                try:
                    if not task.get_loop().is_closed():
                        task.cancel()
                except RuntimeError:
                    pass
        if inst._evict_task and not inst._evict_task.done():
            try:
                if not inst._evict_task.get_loop().is_closed():
                    inst._evict_task.cancel()
            except RuntimeError:
                pass
        try:
            from src.orchestrators.rlm.runtime.harness import HarnessStore

            HarnessStore.clear_shared_for_tests()
        except Exception:
            pass

    @staticmethod
    def harness_key_part(raw: Any, *, fallback: str = "_none") -> str:
        """Filesystem-safe segment for harness path keys."""
        import re

        text = str(raw or "").strip() or fallback
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
        return (text[:180] or fallback)

    def harness_path_for(
        self,
        *,
        flow_id: str = "",
        user_id: str = "",
        node_id: str = "",
        model_logical_id: str = "",
    ) -> str:
        """Durable refine harness JSON keyed by user + flow + node + model.

        Path:
        ``{root}/flows/{flow}/users/{user}/nodes/{node}/models/{model}/harness_state.json``
        """
        flow_key = self.harness_key_part(flow_id)
        user_key = self.harness_key_part(user_id)
        node_key = self.harness_key_part(node_id)
        model_key = self.harness_key_part(model_logical_id)
        path = os.path.join(
            self.root_dir,
            "flows",
            flow_key,
            "users",
            user_key,
            "nodes",
            node_key,
            "models",
            model_key,
            "harness_state.json",
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def harness_paths(self, *, flow_id: str = "", user_id: str = "") -> dict[str, str]:
        """Deprecated multi-scope paths (kept for tests/migration readers)."""
        flow_key = self.harness_key_part(flow_id)
        user_key = self.harness_key_part(user_id)
        flow_path = os.path.join(
            self.root_dir, "flows", flow_key, "users", user_key, "harness_state.json"
        )
        user_path = os.path.join(self.root_dir, "users", user_key, "harness_state.json")
        global_path = os.path.join(
            self.root_dir, "flows", flow_key, "shared", "harness_state.json"
        )
        self._migrate_legacy_flow_harness(flow_id=flow_key, dest=global_path)
        os.makedirs(os.path.dirname(flow_path), exist_ok=True)
        os.makedirs(os.path.dirname(user_path), exist_ok=True)
        os.makedirs(os.path.dirname(global_path), exist_ok=True)
        return {"flow": flow_path, "user": user_path, "global": global_path}

    def _migrate_legacy_flow_harness(self, *, flow_id: str, dest: str) -> None:
        legacy = os.path.join(self.root_dir, "flows", flow_id, "harness_state.json")
        try:
            if os.path.isfile(legacy) and not os.path.isfile(dest):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                os.replace(legacy, dest)
        except OSError:
            pass

    def _ensure_evictor(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._evict_task and not self._evict_task.done():
            return
        self._evict_task = loop.create_task(self._evict_loop())

    async def _evict_loop(self) -> None:
        while True:
            if not self._sessions:
                return
            await asyncio.sleep(60)
            if not self._sessions:
                return
            try:
                await self.evict_idle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("rlm evict failed: %s", exc)

    async def evict_idle(self) -> None:
        limit_m = idle_eviction_minutes()
        now = time.time()
        with self._mu:
            items = list(self._sessions.items())
            live_ids = set(self._sessions.keys())
        for sid, sess in items:
            if is_resident_run_lifetime(sess.run_lifetime):
                continue
            if sess.status in ("running",):
                continue
            parent = str(getattr(sess, "parent_session_id", None) or "").strip()
            if parent and parent in live_ids:
                # Keep children while their parent is still live.
                continue
            idle_for = (now - float(sess.updated_at if sess.updated_at is not None else now)) / 60.0
            if idle_for >= limit_m:
                await self.shutdown(sid, reason="idle_eviction", pop=True)

    def get(self, session_id: str) -> Optional[RlmSession]:
        with self._mu:
            return self._sessions.get(str(session_id or ""))

    def sessions_for_run(self, run_id: str) -> List[RlmSession]:
        """Live sessions sharing ``run_id`` (root-first, then updated_at desc)."""
        rid = str(run_id or "").strip()
        if not rid:
            return []
        with self._mu:
            sids = list(self._by_run.get(rid) or [])
            sessions = [self._sessions[s] for s in sids if s in self._sessions]
        roots = [s for s in sessions if not str(getattr(s, "parent_session_id", None) or "").strip()]
        children = [s for s in sessions if s not in roots]
        children.sort(key=lambda s: float(getattr(s, "updated_at", 0) or 0), reverse=True)
        return roots + children

    def primary_for_run(
        self, run_id: str, *, node_id: str = ""
    ) -> Optional[RlmSession]:
        """Prefer root matching node_id, else any root, else most recent."""
        sessions = self.sessions_for_run(run_id)
        if not sessions:
            return None
        want_node = str(node_id or "").strip()
        roots = [
            s
            for s in sessions
            if not str(getattr(s, "parent_session_id", None) or "").strip()
        ]
        if want_node:
            for sess in roots:
                if str(getattr(sess, "node_id", None) or "").strip() == want_node:
                    return sess
        if roots:
            return max(roots, key=lambda s: float(getattr(s, "updated_at", 0) or 0))
        return max(sessions, key=lambda s: float(getattr(s, "updated_at", 0) or 0))

    def session_for_agent(
        self, run_id: str, *, node_id: str
    ) -> Optional[RlmSession]:
        """Live root session for one Agent node in a conversation (never a child)."""
        rid = str(run_id or "").strip()
        nid = str(node_id or "").strip()
        if not rid or not nid:
            return None
        for sess in self.sessions_for_run(rid):
            if str(getattr(sess, "parent_session_id", None) or "").strip():
                continue
            if str(getattr(sess, "node_id", None) or "").strip() != nid:
                continue
            if getattr(sess, "_closed", False):
                continue
            return sess
        return None

    async def shutdown_agent_sessions(
        self, run_id: str, *, node_id: str, reason: str = "replace"
    ) -> None:
        """Shut down prior root sessions for the same agent slot (ready_gate)."""
        rid = str(run_id or "").strip()
        nid = str(node_id or "").strip()
        if not rid or not nid:
            return
        for sess in list(self.sessions_for_run(rid)):
            if str(getattr(sess, "parent_session_id", None) or "").strip():
                continue
            if str(getattr(sess, "node_id", None) or "").strip() != nid:
                continue
            try:
                await self.shutdown(sess.session_id, reason=reason, pop=True)
            except Exception:
                logger.warning(
                    "rlm shutdown_agent_sessions failed id=%s", sess.session_id
                )

    def get_by_run(self, run_id: str) -> Optional[RlmSession]:
        """Back-compat: primary session for the run."""
        return self.primary_for_run(run_id)

    def list_sessions(
        self,
        *,
        user_id: Optional[str] = None,
        flow_id: Optional[str] = None,
        include_persisted: bool = True,
        hydrate_logs: bool = False,
    ) -> List[Dict[str, Any]]:
        """List RLM sessions for a user/flow.

        ``include_persisted``: also scan cold disk meta (off for flow session
        list merge — Postgres already has those runs).
        ``hydrate_logs``: read ``terminal.jsonl`` (expensive; only for focused
        inspect paths, never for list polls).
        """
        out = []
        with self._mu:
            values = list(self._sessions.values())
        for sess in values:
            if user_id and sess.user_id != user_id:
                continue
            if flow_id and sess.flow_id != flow_id:
                continue
            out.append(sess.public_status())
        if not include_persisted:
            out.sort(key=lambda x: float(x.get("updated_at") or 0), reverse=True)
            return out
        live_ids = {str(row.get("session_id") or "") for row in out}
        try:
            for rec in list_persisted_public(
                self.root_dir,
                user_id=str(user_id or ""),
                flow_id=str(flow_id or ""),
                hydrate_logs=hydrate_logs,
            ):
                sid = str(rec.get("session_id") or "")
                if not sid or sid in live_ids:
                    continue
                out.append(rec)
        except Exception:
            pass
        out.sort(key=lambda x: float(x.get("updated_at") or 0), reverse=True)
        return out
    def register(self, session: RlmSession) -> None:
        self._ensure_evictor()
        with self._mu:
            self._sessions[session.session_id] = session
            if session.run_id:
                rid = str(session.run_id)
                lst = self._by_run.setdefault(rid, [])
                if session.session_id not in lst:
                    lst.append(session.session_id)
        try:
            persist_session_files(session)
        except Exception:
            pass

    def _pop(self, session_id: str) -> None:
        with self._mu:
            sess = self._sessions.pop(session_id, None)
            if sess and sess.run_id:
                rid = str(sess.run_id)
                lst = self._by_run.get(rid) or []
                if session_id in lst:
                    lst = [x for x in lst if x != session_id]
                    if lst:
                        self._by_run[rid] = lst
                    else:
                        self._by_run.pop(rid, None)
            self._bg_tasks.pop(session_id, None)
            hb = self._hb_tasks.pop(session_id, None)
            empty = not self._sessions
        if hb and not hb.done():
            hb.cancel()
        if empty:
            if self._evict_task and not self._evict_task.done():
                self._evict_task.cancel()
            self._evict_task = None

    async def shutdown(
        self,
        session_id: str,
        *,
        reason: str = "shutdown",
        pop: bool = True,
    ) -> Dict[str, Any]:
        sess = self.get(session_id)
        if not sess:
            return {"ok": False, "error": "not_found", "session_id": session_id}
        # Mark before cancelling so awaiters can distinguish user shutdown from soft cancel.
        setattr(sess, "_shutdown_requested", True)
        for cid in list(getattr(sess, "_child_sessions", {}) or {}):
            if cid != session_id and self.get(cid) is not None:
                await self.shutdown(cid, reason="parent_shutdown", pop=True)
        with self._mu:
            task = self._bg_tasks.pop(session_id, None)
            hb = self._hb_tasks.pop(session_id, None)
        if task and not task.done():
            task.cancel()
        if hb and not hb.done():
            hb.cancel()
        try:
            await sess.shutdown()
        except Exception as exc:
            logger.warning("rlm shutdown failed id=%s err=%s", session_id, exc)
        status = sess.public_status()
        status["shutdown_reason"] = reason
        if pop:
            self._pop(session_id)
        return {"ok": True, "session": status}

    def artifact_dir_for(self, session_id: str) -> str:
        path = os.path.join(self.root_dir, session_id)
        os.makedirs(path, exist_ok=True)
        return path

    def _ensure_heartbeat_loop(self, session: RlmSession) -> None:
        if session.run_lifetime != "heartbeat":
            return
        sid = session.session_id
        with self._mu:
            existing = self._hb_tasks.get(sid)
            if existing and not existing.done():
                return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        with self._mu:
            self._hb_tasks[sid] = loop.create_task(self._heartbeat_loop(session))

    async def _heartbeat_loop(self, session: RlmSession) -> None:
        while not session._closed and self.get(session.session_id) is not None:
            try:
                due = session.jobs.due_heartbeats()
                if due and session.status not in ("running",):
                    rec = due[0]
                    prompt = (
                        str(rec.get("prompt") or "").strip()
                        or session.heartbeat_prompt
                        or (
                            session.jobs.goal_get() or {}
                        ).get("objective")
                        or "Continue the active goal. Report via agent_message if blocked."
                    )
                    session._emit("heartbeat_tick", heartbeat_id=rec.get("id"))
                    await session.continue_with_message(str(prompt))
                    # continue_with_message already runs the auto-refine checkpoint.
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "rlm heartbeat failed id=%s err=%s", session.session_id, exc
                )
                session.status = "error"
                session.error = str(exc)
            await asyncio.sleep(5)

    async def run_session_background(self, session: RlmSession, *, base_system: str) -> asyncio.Task:
        self.register(session)

        async def _runner() -> str:
            try:
                out = await session.run_turns(base_system=base_system)
                if session.run_lifetime == "heartbeat":
                    self._ensure_heartbeat_loop(session)
                return out
            except Exception as exc:
                logger.warning(
                    "rlm background session failed id=%s err=%s", session.session_id, exc
                )
                try:
                    from src.orchestrators.rlm.runtime.session import (
                        format_exception_for_log,
                    )

                    session.error = format_exception_for_log(exc)
                except Exception:
                    session.error = str(exc)
                session.status = "error"
                # Do not mask failures as empty success — runner must see the error.
                raise

        task = asyncio.create_task(_runner())
        with self._mu:
            self._bg_tasks[session.session_id] = task
        return task

    async def run_session_inline(self, session: RlmSession, *, base_system: str) -> str:
        self.register(session)
        out = await session.run_turns(base_system=base_system)
        if session.run_lifetime == "heartbeat":
            self._ensure_heartbeat_loop(session)
        return out
