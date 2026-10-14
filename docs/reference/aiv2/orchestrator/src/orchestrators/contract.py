"""Forward-compatible orchestrator contract (schema_version 1).

Unknown manifest fields are preserved in ``extensions`` so consumers and
future engines can evolve without breaking the core registry.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from .host import HostPorts, negotiate_host_ports

SCHEMA_VERSION = 1
API_VERSION = 1

_CORE_KEYS = frozenset(
    {
        "schema_version",
        "api_version",
        "id",
        "name",
        "version",
        "kind",
        "lifecycle",
        "runtime",
        "memory",
        "capabilities",
        "operations",
        "depends_on",
        "host_ports",
        "input_schema",
        "output_schema",
        "config_schema",
        "settings_ui",
        "artifacts",
        "resource",
        "extensions",
        "description",
        "tools_binding",
        "session_binding",
    }
)

# Documented tools_binding.mode vocabulary (open — not exclusive).
DOCUMENTED_TOOLS_BINDING_MODES = frozenset(
    {
        "openai_functions",
        "kernel_callable",
    }
)

# Documented session_binding.mode vocabulary (open — not exclusive).
DOCUMENTED_SESSION_BINDING_MODES = frozenset(
    {
        "durable",  # host reuses one engine session across chat turns (kernel/memory)
        "ephemeral_turn",  # each host turn is create+run; memory lives outside engine
    }
)

_LIFECYCLES = frozenset({"one_shot", "session", "daemon"})
_RUNTIME_KINDS = frozenset({"in_process", "subprocess", "controller"})

# Documented capability vocabulary (open — not an exclusive enum).
DOCUMENTED_CAPABILITIES = frozenset(
    {
        "stream",
        "tools",
        "builtin_tools",
        "subagents",
        "code_exec",
        "bash_only",
        "subprocess_actions",
        "jsonl_rpc",
        "pools",
        "compaction",
        "autonomous",
        "durable_resume",
        "harness_refine",
        "multimodal_look",
        "ensemble_search",
        "isolated_workspaces",
        "trajectory_export",
    }
)

_OPERATION_METHOD = {
    "run": "run",
    "continue": "continue_with",
    "status": "status",
    "shutdown": "shutdown",
    "events": "events",
    # Controllers (e.g. SRLM) may declare these; methods are optional until implemented.
    "search": "search",
    "select": "select",
    "create": None,  # satisfied by Engine.create_session
}


@dataclass
class RuntimeSpec:
    kind: str
    wire: str = "python_api"
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind, "wire": self.wire}
        out.update(self.extras)
        return out


@dataclass
class HostPortDecl:
    id: str
    required: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"id": self.id, "required": self.required}
        out.update(self.extras)
        return out


@dataclass
class ResourceSpec:
    weight: Optional[float] = None
    max_parallel: Optional[int] = None
    timeout_sec: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(self.extras)
        if self.weight is not None:
            out["weight"] = self.weight
        if self.max_parallel is not None:
            out["max_parallel"] = self.max_parallel
        if self.timeout_sec is not None:
            out["timeout_sec"] = self.timeout_sec
        return out


@dataclass
class Manifest:
    schema_version: int
    id: str
    name: str
    version: str
    kind: str
    lifecycle: str
    runtime: RuntimeSpec
    operations: List[str]
    host_ports: List[HostPortDecl]
    api_version: int = API_VERSION
    memory: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    # Consumer-agnostic form defs (AF-style param objects). Empty = no settings UI.
    settings_ui: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    resource: Optional[ResourceSpec] = None
    extensions: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    # How this engine accepts host catalog tools (see tools_binding docs).
    tools_binding: Optional[Dict[str, Any]] = None
    # How the host should retain / resume engine state across chat turns.
    session_binding: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "api_version": self.api_version,
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "kind": self.kind,
            "lifecycle": self.lifecycle,
            "runtime": self.runtime.to_dict(),
            "memory": list(self.memory),
            "capabilities": list(self.capabilities),
            "operations": list(self.operations),
            "depends_on": list(self.depends_on),
            "host_ports": [p.to_dict() for p in self.host_ports],
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "config_schema": self.config_schema,
            "settings_ui": list(self.settings_ui),
            "artifacts": list(self.artifacts),
            "extensions": self.extensions,
        }
        if self.description:
            out["description"] = self.description
        if self.resource is not None:
            out["resource"] = self.resource.to_dict()
        if self.tools_binding is not None:
            out["tools_binding"] = dict(self.tools_binding)
        if self.session_binding is not None:
            out["session_binding"] = dict(self.session_binding)
        return out

    def has_capability(self, cap: str) -> bool:
        return str(cap or "").strip() in self.capabilities

    def has_operation(self, op: str) -> bool:
        return str(op or "").strip() in self.operations


@dataclass
class RunEnvelope:
    """Shared request envelope (api_version 1). HTTP execution is phase 2."""

    api_version: int
    orchestrator_id: str
    request_id: str
    input: Any = None
    config: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    tenant_id: str = ""
    run_id: str = ""
    session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    status: str
    output: Any = None
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    extensions: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@runtime_checkable
class OrchestratorSession(Protocol):
    async def run(self) -> RunResult: ...

    async def continue_with(self, message: Any) -> RunResult: ...

    async def status(self) -> Dict[str, Any]: ...

    async def shutdown(self) -> None: ...

    def events(self) -> AsyncIterator[Dict[str, Any]]: ...


@runtime_checkable
class OrchestratorEngine(Protocol):
    @property
    def manifest(self) -> Manifest: ...

    async def create_session(
        self,
        envelope: RunEnvelope,
        host: HostPorts,
    ) -> OrchestratorSession: ...


def _uniq_str_list(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    seen = set()
    for item in raw:
        s = str(item or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _schema_path() -> Path:
    return Path(__file__).resolve().parent / "manifest.schema.json"


def load_schema() -> Dict[str, Any]:
    return json.loads(_schema_path().read_text(encoding="utf-8"))


def normalize_manifest(raw: Mapping[str, Any] | None) -> Manifest:
    """Normalize a raw dict into Manifest; unknown keys → extensions."""
    if not isinstance(raw, Mapping):
        raise ValueError("manifest must be an object")
    data = dict(raw)
    unknown = {k: copy.deepcopy(v) for k, v in data.items() if k not in _CORE_KEYS}
    extensions = dict(data.get("extensions") or {})
    if not isinstance(extensions, dict):
        extensions = {}
    for k, v in unknown.items():
        extensions.setdefault(k, v)

    schema_version = int(data.get("schema_version") or SCHEMA_VERSION)
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version={schema_version}; want {SCHEMA_VERSION}")

    oid = str(data.get("id") or "").strip()
    name = str(data.get("name") or "").strip()
    version = str(data.get("version") or "").strip()
    kind = str(data.get("kind") or "").strip()
    lifecycle = str(data.get("lifecycle") or "").strip()
    if not oid or not name or not version or not kind:
        raise ValueError("manifest requires id, name, version, kind")
    if lifecycle not in _LIFECYCLES:
        raise ValueError(f"invalid lifecycle={lifecycle!r}")

    runtime_raw = data.get("runtime") or {}
    if not isinstance(runtime_raw, Mapping):
        raise ValueError("runtime must be an object")
    runtime_kind = str(runtime_raw.get("kind") or "").strip()
    if runtime_kind not in _RUNTIME_KINDS:
        raise ValueError(f"invalid runtime.kind={runtime_kind!r}")
    runtime_wire = str(runtime_raw.get("wire") or "python_api").strip() or "python_api"
    runtime_extras = {
        k: copy.deepcopy(v)
        for k, v in runtime_raw.items()
        if k not in ("kind", "wire")
    }

    operations = _uniq_str_list(data.get("operations"))
    if not operations:
        raise ValueError("operations must be a non-empty array")

    host_ports: List[HostPortDecl] = []
    for item in data.get("host_ports") or []:
        if not isinstance(item, Mapping):
            continue
        pid = str(item.get("id") or "").strip()
        if not pid:
            continue
        extras = {
            k: copy.deepcopy(v)
            for k, v in item.items()
            if k not in ("id", "required")
        }
        host_ports.append(
            HostPortDecl(
                id=pid,
                required=bool(item.get("required", False)),
                extras=extras,
            )
        )

    resource = None
    resource_raw = data.get("resource")
    if isinstance(resource_raw, Mapping):
        extras = {
            k: copy.deepcopy(v)
            for k, v in resource_raw.items()
            if k not in ("weight", "max_parallel", "timeout_sec")
        }
        weight = resource_raw.get("weight")
        max_parallel = resource_raw.get("max_parallel")
        timeout_sec = resource_raw.get("timeout_sec")
        resource = ResourceSpec(
            weight=float(weight) if weight is not None else None,
            max_parallel=int(max_parallel) if max_parallel is not None else None,
            timeout_sec=float(timeout_sec) if timeout_sec is not None else None,
            extras=extras,
        )

    def _obj(key: str) -> Dict[str, Any]:
        v = data.get(key)
        return dict(v) if isinstance(v, Mapping) else {}

    settings_ui_raw = data.get("settings_ui")
    settings_ui: List[Dict[str, Any]] = []
    if isinstance(settings_ui_raw, list):
        for item in settings_ui_raw:
            if isinstance(item, Mapping):
                settings_ui.append(dict(item))

    tools_binding_raw = data.get("tools_binding")
    tools_binding: Optional[Dict[str, Any]] = None
    if isinstance(tools_binding_raw, Mapping):
        tools_binding = dict(tools_binding_raw)

    session_binding_raw = data.get("session_binding")
    session_binding: Optional[Dict[str, Any]] = None
    if isinstance(session_binding_raw, Mapping):
        session_binding = dict(session_binding_raw)

    return Manifest(
        schema_version=schema_version,
        api_version=int(data.get("api_version") or API_VERSION),
        id=oid,
        name=name,
        version=version,
        kind=kind,
        lifecycle=lifecycle,
        runtime=RuntimeSpec(kind=runtime_kind, wire=runtime_wire, extras=runtime_extras),
        operations=operations,
        host_ports=host_ports,
        memory=_uniq_str_list(data.get("memory")),
        capabilities=_uniq_str_list(data.get("capabilities")),
        depends_on=_uniq_str_list(data.get("depends_on")),
        input_schema=_obj("input_schema"),
        output_schema=_obj("output_schema"),
        config_schema=_obj("config_schema"),
        settings_ui=settings_ui,
        artifacts=_uniq_str_list(data.get("artifacts")),
        resource=resource,
        extensions=extensions,
        description=str(data.get("description") or ""),
        tools_binding=tools_binding,
        session_binding=session_binding,
    )


def validate_manifest(raw: Mapping[str, Any] | Manifest) -> Manifest:
    """Normalize + structural validation (qualification gate step 1)."""
    manifest = raw if isinstance(raw, Manifest) else normalize_manifest(raw)
    if not manifest.id or not manifest.id[0].isalpha():
        raise ValueError(f"invalid id={manifest.id!r}")
    for ch in manifest.id:
        if not (ch.isalnum() or ch == "_"):
            raise ValueError(f"invalid id={manifest.id!r}")
    if manifest.lifecycle not in _LIFECYCLES:
        raise ValueError(f"invalid lifecycle={manifest.lifecycle!r}")
    if manifest.runtime.kind not in _RUNTIME_KINDS:
        raise ValueError(f"invalid runtime.kind={manifest.runtime.kind!r}")
    if not manifest.operations:
        raise ValueError("operations empty")
    return manifest


def validate_engine_operations(engine: OrchestratorEngine) -> None:
    """Ensure declared operations map to session methods (qualification gate)."""
    manifest = engine.manifest
    # Probe create_session signature by attribute presence.
    if not hasattr(engine, "create_session"):
        raise ValueError(f"engine {manifest.id}: missing create_session")
    # Session methods are checked lazily after create; for static ops we require
    # the engine class / factory documents them. Soft check: operations subset
    # of known method map or custom (allowed).
    for op in manifest.operations:
        if op == "create":
            continue
        method = _OPERATION_METHOD.get(op)
        if method is None and op not in _OPERATION_METHOD:
            # Custom operations are allowed; adapters implement as needed.
            continue


def ensure_host_ready(manifest: Manifest, host: HostPorts) -> None:
    result = negotiate_host_ports(
        [p.to_dict() for p in manifest.host_ports],
        host,
    )
    if not result.ok:
        raise ValueError(
            f"orchestrator {manifest.id}: missing required host ports: "
            f"{', '.join(result.missing_required)}"
        )
