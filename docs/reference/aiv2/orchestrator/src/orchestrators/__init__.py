"""Drop-in orchestrator / agent-harness registry for the orchestrator service.

New engines are added as ``src/orchestrators/<id>/`` with ``manifest.json`` +
``adapter.py``. Core chat routes under ``/chat*`` are untouched.
"""

from __future__ import annotations

from .contract import (
    API_VERSION,
    SCHEMA_VERSION,
    Manifest,
    OrchestratorEngine,
    OrchestratorSession,
    RunEnvelope,
    RunResult,
    normalize_manifest,
    validate_manifest,
)
from .host import HostPortCatalog, HostPorts, negotiate_host_ports
from .registry import OrchestratorRegistry, get_registry

__all__ = [
    "API_VERSION",
    "SCHEMA_VERSION",
    "HostPortCatalog",
    "HostPorts",
    "Manifest",
    "OrchestratorEngine",
    "OrchestratorRegistry",
    "OrchestratorSession",
    "RunEnvelope",
    "RunResult",
    "get_registry",
    "negotiate_host_ports",
    "normalize_manifest",
    "validate_manifest",
]
