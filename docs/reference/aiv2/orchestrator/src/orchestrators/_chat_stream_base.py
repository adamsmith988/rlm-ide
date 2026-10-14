"""Shared stub engine for chat stream orchestrators (selection via /chat/stream)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from src.orchestrators.contract import (
    Manifest,
    RunEnvelope,
    RunResult,
    normalize_manifest,
    validate_manifest,
)
from src.orchestrators.host import HostPorts


def load_package_manifest(package_dir: Path) -> Manifest:
    raw = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    return validate_manifest(normalize_manifest(raw))


class ChatStreamSession:
    """Chat engines run on /chat/stream, not create_session — stub for registry."""

    def __init__(self, orchestrator_id: str) -> None:
        self._id = orchestrator_id

    async def run(self) -> RunResult:
        return RunResult(
            status="use_chat_stream",
            output={
                "message": (
                    f"Orchestrator {self._id} is a chat stream engine. "
                    "POST /chat/stream with orchestrator_id."
                )
            },
        )

    async def continue_with(self, message: Any) -> RunResult:
        return await self.run()

    async def status(self) -> Dict[str, Any]:
        return {"orchestrator_id": self._id, "mode": "chat_stream"}

    async def shutdown(self) -> None:
        return None

    async def events(self) -> AsyncIterator[Dict[str, Any]]:
        if False:  # pragma: no cover
            yield {}
        return


class ChatStreamEngine:
    def __init__(self, package_dir: Path) -> None:
        self._manifest = load_package_manifest(package_dir)

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    async def create_session(
        self,
        envelope: RunEnvelope,
        host: HostPorts,
    ) -> ChatStreamSession:
        return ChatStreamSession(self._manifest.id)
