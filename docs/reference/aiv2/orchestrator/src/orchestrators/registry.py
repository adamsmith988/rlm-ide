"""Dynamic drop-in orchestrator registry.

Scans ``src/orchestrators/*/manifest.json`` (skips ``_*`` / ``_archived``).
No hardcoded orchestrator ids — RLM appears only because its folder exists.
"""

from __future__ import annotations

import importlib
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .contract import (
    Manifest,
    OrchestratorEngine,
    normalize_manifest,
    validate_engine_operations,
    validate_manifest,
)

logger = logging.getLogger(__name__)


def _package_root() -> Path:
    return Path(__file__).resolve().parent


@dataclass
class RegisteredOrchestrator:
    manifest: Manifest
    engine: Optional[OrchestratorEngine] = None
    package_dir: Optional[Path] = None
    load_error: Optional[str] = None

    @property
    def id(self) -> str:
        return self.manifest.id

    @property
    def ready(self) -> bool:
        return self.engine is not None and not self.load_error


@dataclass
class OrchestratorRegistry:
    root: Path = field(default_factory=_package_root)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _by_id: Dict[str, RegisteredOrchestrator] = field(default_factory=dict, repr=False)
    _loaded: bool = field(default=False, repr=False)

    def discover(self, *, force: bool = False) -> None:
        with self._lock:
            if self._loaded and not force:
                return
            found: Dict[str, RegisteredOrchestrator] = {}
            if not self.root.is_dir():
                self._by_id = found
                self._loaded = True
                return
            for child in sorted(self.root.iterdir()):
                if not child.is_dir():
                    continue
                name = child.name
                if name.startswith("_") or name == "_archived":
                    continue
                manifest_path = child / "manifest.json"
                if not manifest_path.is_file():
                    continue
                try:
                    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest = validate_manifest(normalize_manifest(raw))
                except Exception as exc:  # noqa: BLE001 — isolate bad packages
                    logger.warning(
                        "orchestrator manifest skip dir=%s err=%s",
                        child.name,
                        exc,
                    )
                    continue
                if manifest.id != child.name and manifest.id != name:
                    # Prefer folder name mismatch as soft warning; require id match folder.
                    logger.warning(
                        "orchestrator id/folder mismatch id=%s folder=%s",
                        manifest.id,
                        child.name,
                    )
                engine = None
                load_error = None
                adapter_path = child / "adapter.py"
                if adapter_path.is_file():
                    try:
                        engine = self._load_adapter(child, manifest)
                        validate_engine_operations(engine)
                        if engine.manifest.id != manifest.id:
                            raise ValueError(
                                f"adapter manifest.id={engine.manifest.id!r} "
                                f"!= file id={manifest.id!r}"
                            )
                    except Exception as exc:  # noqa: BLE001
                        load_error = str(exc)
                        logger.warning(
                            "orchestrator adapter load failed id=%s err=%s",
                            manifest.id,
                            exc,
                        )
                        engine = None
                found[manifest.id] = RegisteredOrchestrator(
                    manifest=manifest,
                    engine=engine,
                    package_dir=child,
                    load_error=load_error,
                )
            self._by_id = found
            self._loaded = True

    def _load_adapter(self, package_dir: Path, manifest: Manifest) -> OrchestratorEngine:
        # Import as src.orchestrators.<folder>.adapter
        folder = package_dir.name
        module_name = f"src.orchestrators.{folder}.adapter"
        mod = importlib.import_module(module_name)
        factory = getattr(mod, "get_engine", None) or getattr(mod, "create_engine", None)
        if callable(factory):
            engine = factory()
        else:
            cls = getattr(mod, "Engine", None) or getattr(mod, "Adapter", None)
            if cls is None:
                raise ValueError(f"{module_name}: need get_engine() or Engine class")
            engine = cls()
        if not isinstance(engine, OrchestratorEngine) and not (
            hasattr(engine, "manifest") and hasattr(engine, "create_session")
        ):
            raise ValueError(f"{module_name}: engine does not satisfy OrchestratorEngine")
        return engine  # type: ignore[return-value]

    def reload(self) -> None:
        self.discover(force=True)

    def list_orchestrators(
        self,
        *,
        capability: Optional[str] = None,
        operation: Optional[str] = None,
        kind: Optional[str] = None,
        ready_only: bool = False,
    ) -> List[RegisteredOrchestrator]:
        self.discover()
        caps = [c.strip() for c in (capability or "").split(",") if c.strip()]
        ops = [o.strip() for o in (operation or "").split(",") if o.strip()]
        kind_f = str(kind or "").strip()
        out: List[RegisteredOrchestrator] = []
        for reg in self._by_id.values():
            if ready_only and not reg.ready:
                continue
            m = reg.manifest
            if kind_f and m.kind != kind_f:
                continue
            if caps and not all(c in m.capabilities for c in caps):
                continue
            if ops and not all(o in m.operations for o in ops):
                continue
            out.append(reg)
        out.sort(key=lambda r: r.manifest.id)
        return out

    def get(self, orchestrator_id: str) -> Optional[RegisteredOrchestrator]:
        self.discover()
        return self._by_id.get(str(orchestrator_id or "").strip())

    def require(self, orchestrator_id: str) -> RegisteredOrchestrator:
        reg = self.get(orchestrator_id)
        if reg is None:
            raise KeyError(f"orchestrator not found: {orchestrator_id}")
        return reg

    def negotiate(
        self,
        *,
        required_capabilities: Sequence[str] | None = None,
        required_operations: Sequence[str] | None = None,
        kind: Optional[str] = None,
        ready_only: bool = True,
    ) -> List[RegisteredOrchestrator]:
        """Rank matches by how many optional extras they cover (stable sort by id)."""
        caps = [str(c).strip() for c in (required_capabilities or ()) if str(c).strip()]
        ops = [str(o).strip() for o in (required_operations or ()) if str(o).strip()]
        matches = self.list_orchestrators(
            capability=",".join(caps) if caps else None,
            operation=",".join(ops) if ops else None,
            kind=kind,
            ready_only=ready_only,
        )
        # Prefer engines that declare more of the requested capabilities (already filtered).
        matches.sort(
            key=lambda r: (
                -len(set(r.manifest.capabilities) & set(caps)) if caps else 0,
                -len(r.manifest.capabilities),
                r.manifest.id,
            )
        )
        return matches

    def resolve_depends_on(self, orchestrator_id: str) -> List[str]:
        """Return depends_on ids; does not auto-load — caller checks presence."""
        reg = self.require(orchestrator_id)
        return list(reg.manifest.depends_on)

    def missing_dependencies(self, orchestrator_id: str) -> List[str]:
        self.discover()
        reg = self.require(orchestrator_id)
        missing = []
        for dep in reg.manifest.depends_on:
            if dep not in self._by_id:
                missing.append(dep)
        return missing


_REGISTRY: Optional[OrchestratorRegistry] = None
_REGISTRY_LOCK = threading.Lock()


def get_registry(*, root: Optional[Path] = None) -> OrchestratorRegistry:
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None or (root is not None and root != _REGISTRY.root):
            _REGISTRY = OrchestratorRegistry(root=root or _package_root())
        return _REGISTRY
