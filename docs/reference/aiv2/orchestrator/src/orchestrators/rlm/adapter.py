"""RLM drop-in adapter — implements the stable orchestrator contract.

Does not import agent_flow. Host injects chat/tools/pools via HostPorts.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from src.orchestrators.contract import (
    Manifest,
    RunEnvelope,
    RunResult,
    ensure_host_ready,
    normalize_manifest,
    validate_manifest,
)
from src.orchestrators.host import HostPorts
from src.orchestrators.rlm.schema_utils import (
    agent_model_is_explicit,
    normalize_run_lifetime,
    normalize_tool_id_list,
)


def _load_manifest() -> Manifest:
    path = Path(__file__).resolve().parent / "manifest.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return validate_manifest(normalize_manifest(raw))


def _extract_assistant(resp: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal OpenAI-style assistant extraction for host chat responses."""
    if not isinstance(resp, dict):
        return {"role": "assistant", "content": str(resp or "")}
    choices = resp.get("choices") or []
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        if isinstance(msg, dict):
            return dict(msg)
    if "message" in resp and isinstance(resp["message"], dict):
        return dict(resp["message"])
    content = resp.get("content") or resp.get("text") or ""
    return {"role": "assistant", "content": content}


class RlmOrchestratorSession:
    """Thin session wrapper around ``RlmSession`` using injected HostPorts."""

    def __init__(self, session: Any, envelope: RunEnvelope) -> None:
        self._session = session
        self._envelope = envelope
        self._last_text: Optional[str] = None

    def _result_from_session(self, text: Any) -> RunResult:
        """Map runtime session status/text — never pretend soft-fail is completed."""
        sess = self._session
        raw = text if isinstance(text, str) else str(text or "")
        if not str(raw or "").strip() and hasattr(sess, "_visible_session_output"):
            try:
                raw = str(sess._visible_session_output() or "")  # noqa: SLF001
            except Exception:
                pass
        self._last_text = raw
        status = str(getattr(sess, "status", "") or "").strip().lower() or "completed"
        if status not in ("completed", "incomplete", "error", "cancelled", "running"):
            status = "completed"
        # Running after a finished turn means the loop returned normally.
        if status == "running":
            status = "completed"
        return RunResult(
            status=status,
            output={"text": self._last_text},
            artifacts=[],
            extensions={
                "rlm_session_id": getattr(sess, "session_id", None),
                "rlm_status": status,
                "rlm_error": getattr(sess, "error", None),
                "rlm_stop_reason": getattr(sess, "_autonomous_stop_reason", None),
            },
        )

    async def run(self) -> RunResult:
        text = await self._session.run_turns()
        return self._result_from_session(text)

    async def continue_with(self, message: Any) -> RunResult:
        text = await self._session.continue_with_message(message)
        return self._result_from_session(text)

    async def status(self) -> Dict[str, Any]:
        sess = self._session
        if hasattr(sess, "public_status"):
            return dict(sess.public_status() or {})
        return {
            "rlm_session_id": getattr(sess, "session_id", None),
            "request_id": self._envelope.request_id,
        }

    async def shutdown(self) -> None:
        sess = self._session
        if hasattr(sess, "shutdown"):
            await sess.shutdown()
        elif hasattr(sess, "close"):
            maybe = sess.close()
            if hasattr(maybe, "__await__"):
                await maybe

    async def events(self) -> AsyncIterator[Dict[str, Any]]:
        # Event streaming is host-driven via HostPorts.events; yield nothing by default.
        if False:  # pragma: no cover
            yield {}
        return


class RlmOrchestratorEngine:
    """OrchestratorEngine implementation for the RLM drop-in."""

    def __init__(self, manifest: Optional[Manifest] = None) -> None:
        self._manifest = manifest or _load_manifest()

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    async def create_session(
        self,
        envelope: RunEnvelope,
        host: HostPorts,
    ) -> RlmOrchestratorSession:
        ensure_host_ready(self._manifest, host)
        if host.chat is None or host.tools is None:
            raise ValueError("rlm requires host ports: chat, tools")

        from src.orchestrators.rlm.runtime.session import RlmSession

        cfg = dict(envelope.config or {})
        if isinstance(envelope.input, dict):
            user_content = envelope.input.get("user_content", envelope.input.get("message"))
            if "cfg" in envelope.input and isinstance(envelope.input["cfg"], dict):
                merged = dict(envelope.input["cfg"])
                merged.update(cfg)
                cfg = merged
        else:
            user_content = envelope.input

        if not agent_model_is_explicit(cfg.get("model")):
            # Allow empty for structural create in tests; runtime will still need a model.
            pass

        if "run_lifetime" in cfg:
            cfg["run_lifetime"] = normalize_run_lifetime(cfg.get("run_lifetime"))
        if "tool_ids" in cfg:
            cfg["tool_ids"] = normalize_tool_id_list(cfg.get("tool_ids"))

        artifact_dir = str(cfg.get("artifact_dir") or "").strip()
        workspace_dir = str(cfg.get("workspace_dir") or artifact_dir).strip()
        if not artifact_dir:
            # Ephemeral dirs — caller should pass durable paths in production.
            import tempfile

            artifact_dir = tempfile.mkdtemp(prefix="rlm_orch_")
            workspace_dir = workspace_dir or artifact_dir

        session_id = str(cfg.get("session_id") or envelope.session_id or uuid.uuid4())
        run_id = str(cfg.get("run_id") or envelope.run_id or envelope.request_id or session_id)

        session = RlmSession(
            cfg=cfg,
            user_content=user_content,
            session_id=session_id,
            run_id=run_id,
            artifact_dir=artifact_dir,
            workspace_dir=workspace_dir,
            chat_fn=host.chat,
            extract_assistant_fn=_extract_assistant,
            gen_params=dict(cfg.get("gen_params") or {}),
            tool_runner=host.tools,
            pool_search=host.pool_search,
            pool_resolve=host.pool_resolve,
            pool_list=host.pool_list,
            pool_read=host.pool_read,
            pool_defs=cfg.get("pool_defs") if isinstance(cfg.get("pool_defs"), list) else None,
            spawn_child=host.spawn_child,
            model_finder=host.model_finder,
            on_event=host.events,
            chat_stream_fn=host.chat_stream,
        )
        try:
            from .runtime.persist import (
                load_pool_files_baseline,
                write_pool_files_baseline,
            )
            import os as _os

            baseline_path = _os.path.join(
                artifact_dir, "pool", ".deliverable_baseline.json"
            )
            if not _os.path.isfile(baseline_path):
                write_pool_files_baseline(artifact_dir)
            session._pool_files_baseline = load_pool_files_baseline(artifact_dir)
        except Exception:
            session._pool_files_baseline = {}
        return RlmOrchestratorSession(session, envelope)


def get_engine() -> RlmOrchestratorEngine:
    return RlmOrchestratorEngine()


# Aliases for registry discovery
Engine = RlmOrchestratorEngine
create_engine = get_engine
