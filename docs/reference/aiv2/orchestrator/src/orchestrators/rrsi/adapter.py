"""RRSI drop-in adapter — the contract surface.

Over the contract the policy is the host ``chat`` port steered by the harness
(prompt block as system message, ``engine_config`` limited to the manifest's
``contract_policy_keys``); hub chat uses any registered chat engine instead
(see chat_stream). Envelope input: ``{"question": str, "context_messages": [...]}``.
State lives in ``config.state_dir`` or the per-session orchestrator state root.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Dict, List, Optional

from src.orchestrators.contract import Manifest, RunEnvelope, RunResult, ensure_host_ready
from src.orchestrators.host import HostPorts

from .._shared.llm_roles import host_chat_llm
from .._shared.manifest_config import package_manifest, resolve_config
from .._shared.state_dir import session_state_dir
from .core.config import RRSIConfig
from .runtime import harness as H
from .runtime.controller import RrsiController, RrsiSettings, TurnOutcome, build_evolve_set
from .runtime.domain import Task, Trial, default_domain
from .runtime.state import RrsiState


def _question_and_context(raw: Any) -> tuple[str, List[Dict[str, Any]]]:
    if isinstance(raw, dict):
        q = raw.get("question", raw.get("message", raw.get("user_content", "")))
        ctx = [m for m in (raw.get("context_messages") or []) if isinstance(m, dict)]
        return str(q or ""), ctx
    return str(raw or ""), []


class RrsiOrchestratorSession:
    def __init__(self, envelope: RunEnvelope, host: HostPorts, manifest: Manifest) -> None:
        self._envelope = envelope
        self._manifest = manifest
        self._host = host
        cfg = resolve_config(manifest, envelope.config)
        self._settings = RrsiSettings.from_config(cfg)
        self._domain = default_domain()
        llm = host_chat_llm(host.chat)
        policy_keys = {str(k) for k in (cfg.get("contract_policy_keys") or [])}
        root = str(cfg.get("state_dir") or "").strip() or session_state_dir(
            manifest.id, envelope.tenant_id or "contract", envelope.session_id or envelope.request_id
        )
        self._state = RrsiState(root)
        self._controller = RrsiController(
            self._settings, RRSIConfig.from_mapping(cfg), self._domain, llm, self._policy(llm), self._state,
            allowed_engine_keys=policy_keys, known_tool_ids=set(),
            on_log=lambda line: self._emit({"kind": "log", "line": line}),
            on_progress=lambda p: self._emit({"kind": "progress", **p}),
        )
        self._question, self._context = _question_and_context(envelope.input)
        self._last: Optional[TurnOutcome] = None
        self._status = "created"

    def _emit(self, payload: Dict[str, Any]) -> None:
        if self._host.events is not None:
            self._host.events({"orchestrator": self._manifest.id, **payload})

    def _policy(self, llm):
        async def run(harness: H.Harness, task: Task, index: int) -> Trial:
            started = time.monotonic()
            messages: List[Dict[str, Any]] = []
            block = H.prompt_block(self._domain, harness)
            if block:
                messages.append({"role": "system", "content": block})
            messages += H.trim_history(task.context, harness.get("history_turns"))
            messages.append({"role": "user", "content": task.request})
            ecfg = dict(harness.get("engine_config") or {})
            reply = await llm(
                messages,
                temperature=ecfg.pop("temperature", None),
                max_tokens=ecfg.pop("max_tokens", None),
                sampling=ecfg or None,
            )
            return Trial(task_id=task.id, index=index, answer=reply.text if reply.ok else "",
                         error="" if reply.ok else reply.error, tokens=reply.tokens,
                         elapsed_sec=round(time.monotonic() - started, 3))

        return run

    async def search(self) -> RunResult:
        self._status = "searching"
        tasks, current_id = build_evolve_set(self._question, self._context, self._settings.replay_turns,
                                             self._settings.replay_max_request_chars)
        self._last = await self._controller.turn(tasks, current_id)
        self._current_task = tasks[current_id]
        self._status = "searched"
        return RunResult(status="completed", output=self._summary())

    async def select(self) -> RunResult:
        out = self._last
        if out is None:
            return RunResult(status="error", error="search has not run")
        answer = out.answer
        if answer is None or answer.failed:
            answer = await self._controller.run_trial(out.harness, self._current_task, 0)
        if answer.failed:
            self._status = "error"
            return RunResult(status="error", error=answer.error or "empty answer", output=self._summary())
        self._status = "completed"
        return RunResult(status="completed", output={"text": answer.answer, **self._summary()})

    def _summary(self) -> Dict[str, Any]:
        out = self._last
        if out is None:
            return {}
        return {
            "harness": out.harness,
            "harness_version": out.version,
            "served_only": out.served_only,
            "rounds": [
                {"t": r.t, "accepted": r.accepted, "delta": r.delta, "b_t": r.budget, "sigma_t": r.sigma,
                 "S_incumbent": r.incumbent.ev.S, "S_next": r.next_incumbent.ev.S,
                 "decisions": [d.to_json() for d in r.decisions]}
                for r in out.rounds
            ],
        }

    async def run(self) -> RunResult:
        await self.search()
        return await self.select()

    async def continue_with(self, message: Any) -> RunResult:
        question, ctx = _question_and_context(message)
        prev = self._question
        answer = (self._last.answer.answer if self._last and self._last.answer else "")
        if prev:
            self._context = list(self._context) + [{"role": "user", "content": prev}]
            if answer:
                self._context.append({"role": "assistant", "content": answer})
        self._context += ctx
        self._question = question
        return await self.run()

    async def status(self) -> Dict[str, Any]:
        fr = self._state.frontier() or {}
        return {
            "orchestrator": self._manifest.id,
            "request_id": self._envelope.request_id,
            "status": self._status,
            "round": fr.get("t"),
            "incumbent": fr.get("incumbent"),
            "state_dir": str(self._state.root),
        }

    async def shutdown(self) -> None:
        self._status = "shutdown"

    async def events(self) -> AsyncIterator[Dict[str, Any]]:
        if False:  # pragma: no cover — events go through HostPorts.events
            yield {}
        return


class RrsiOrchestratorEngine:
    def __init__(self, manifest: Optional[Manifest] = None) -> None:
        self._manifest = manifest or package_manifest(__file__)

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    async def create_session(self, envelope: RunEnvelope, host: HostPorts) -> RrsiOrchestratorSession:
        ensure_host_ready(self._manifest, host)
        return RrsiOrchestratorSession(envelope, host, self._manifest)


def get_engine() -> RrsiOrchestratorEngine:
    return RrsiOrchestratorEngine()


Engine = RrsiOrchestratorEngine
create_engine = get_engine
