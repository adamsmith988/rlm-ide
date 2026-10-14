"""SRLM drop-in adapter — SelfReflect over the contract's ``chat`` HostPort.

Envelope input: ``{"question": str, "context_messages": [...]}`` (or a bare
string). Config: any key of the manifest's ``config_schema``; missing keys take
the manifest defaults. ``peer_engine`` applies to hub chat only — over the
contract the host chat port is the sampler.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from src.orchestrators.contract import Manifest, RunEnvelope, RunResult, ensure_host_ready
from src.orchestrators.host import HostPorts

from .._shared.llm_roles import host_chat_llm
from .._shared.manifest_config import package_manifest, resolve_config
from .runtime.controller import Sample, SelfReflectController, SrlmSettings, answer_distribution


def _question_and_context(raw: Any) -> tuple[str, List[Dict[str, Any]]]:
    if isinstance(raw, dict):
        question = raw.get("question", raw.get("message", raw.get("user_content", "")))
        ctx = raw.get("context_messages") or []
        return str(question or ""), [m for m in ctx if isinstance(m, dict)]
    return str(raw or ""), []


class SrlmOrchestratorSession:
    def __init__(self, envelope: RunEnvelope, host: HostPorts, manifest: Manifest) -> None:
        self._envelope = envelope
        self._manifest = manifest
        settings = SrlmSettings.from_config(resolve_config(manifest, envelope.config))
        self._controller = SelfReflectController(
            settings, host_chat_llm(host.chat), on_progress=self._emit
        )
        self._host = host
        self._question, self._context = _question_and_context(envelope.input)
        self._samples: List[Sample] = []
        self._status = "created"
        self._last: Optional[Dict[str, Any]] = None

    def _emit(self, payload: Dict[str, Any]) -> None:
        if self._host.events is not None:
            self._host.events({"orchestrator": self._manifest.id, **payload})

    async def search(self) -> RunResult:
        self._status = "searching"
        self._samples = await self._controller.search(
            self._controller.direct_sampler(self._question, self._context)
        )
        self._status = "searched"
        return RunResult(
            status="completed",
            output={
                "samples": [s.to_dict() for s in self._samples],
                "distribution": answer_distribution(self._samples),
            },
        )

    async def select(self) -> RunResult:
        try:
            text, plan, tokens = await self._controller.select(
                self._question, self._context, self._samples
            )
        except Exception as exc:  # noqa: BLE001
            self._status = "error"
            return RunResult(status="error", error=str(exc))
        self._status = "completed"
        self._last = {
            "text": text,
            "strategy": plan.strategy,
            "analysis": plan.analysis or None,
            "samples": [s.to_dict() for s in self._samples],
            "distribution": answer_distribution(self._samples),
            "tokens": tokens,
        }
        return RunResult(status="completed", output=self._last)

    async def run(self) -> RunResult:
        if self._controller.settings.needs_samples and not self._samples:
            await self.search()
        return await self.select()

    async def continue_with(self, message: Any) -> RunResult:
        question, ctx = _question_and_context(message)
        if self._last and self._question:
            self._context = list(self._context) + [
                {"role": "user", "content": self._question},
                {"role": "assistant", "content": self._last["text"]},
            ]
        self._context = list(self._context) + ctx
        self._question = question
        self._samples = []
        return await self.run()

    async def status(self) -> Dict[str, Any]:
        return {
            "orchestrator": self._manifest.id,
            "request_id": self._envelope.request_id,
            "status": self._status,
            "n_samples": len(self._samples),
            "strategy": self._controller.settings.summary_strategy,
        }

    async def shutdown(self) -> None:
        self._status = "shutdown"

    async def events(self) -> AsyncIterator[Dict[str, Any]]:
        if False:  # pragma: no cover — events go through HostPorts.events
            yield {}
        return


class SrlmOrchestratorEngine:
    def __init__(self, manifest: Optional[Manifest] = None) -> None:
        self._manifest = manifest or package_manifest(__file__)

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    async def create_session(self, envelope: RunEnvelope, host: HostPorts) -> SrlmOrchestratorSession:
        ensure_host_ready(self._manifest, host)
        return SrlmOrchestratorSession(envelope, host, self._manifest)


def get_engine() -> SrlmOrchestratorEngine:
    return SrlmOrchestratorEngine()


Engine = SrlmOrchestratorEngine
create_engine = get_engine
