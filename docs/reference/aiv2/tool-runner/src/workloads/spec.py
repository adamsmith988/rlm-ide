"""Declarative contract for work that may outlive an HTTP request."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple


def _tuple(value: Any, default: Tuple[str, ...] = ()) -> Tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError("expected a string or sequence")
    return tuple(str(item).strip() for item in value if str(item).strip())


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    backoff_sec: float = 0.0
    max_backoff_sec: float = 0.0
    retry_on: Tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, raw: Any) -> "RetryPolicy":
        data = _mapping(raw, "retry")
        policy = cls(
            max_attempts=int(data.get("max_attempts", 1)),
            backoff_sec=float(data.get("backoff_sec", 0.0)),
            max_backoff_sec=float(data.get("max_backoff_sec", 0.0)),
            retry_on=_tuple(data.get("retry_on")),
        )
        if policy.max_attempts < 1 or policy.backoff_sec < 0 or policy.max_backoff_sec < 0:
            raise ValueError("retry values must be non-negative and max_attempts >= 1")
        return policy


@dataclass(frozen=True)
class IdempotencyPolicy:
    mode: str = "none"
    key_fields: Tuple[str, ...] = ()
    ttl_sec: float = 0.0

    @classmethod
    def from_mapping(cls, raw: Any) -> "IdempotencyPolicy":
        data = _mapping(raw, "idempotency")
        policy = cls(
            mode=str(data.get("mode") or "none").strip().lower(),
            key_fields=_tuple(data.get("key_fields")),
            ttl_sec=float(data.get("ttl_sec", 0.0)),
        )
        if policy.ttl_sec < 0:
            raise ValueError("idempotency.ttl_sec must be non-negative")
        return policy


@dataclass(frozen=True)
class AffinityPolicy:
    selectors: Tuple[str, ...] = ()
    id_fields: Tuple[str, ...] = ()
    dynamic_id_fields: Tuple[str, ...] = ()
    tool_ids: frozenset[str] = field(default_factory=frozenset)
    tool_prefixes: Tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, raw: Any) -> "AffinityPolicy":
        data = _mapping(raw, "affinity")
        return cls(
            selectors=_tuple(data.get("selectors")),
            id_fields=_tuple(data.get("id_fields")),
            dynamic_id_fields=_tuple(data.get("dynamic_id_fields")),
            tool_ids=frozenset(_tuple(data.get("tool_ids"))),
            tool_prefixes=_tuple(data.get("tool_prefixes") or data.get("prefixes")),
        )

    def matches_tool(self, tool_id: str) -> bool:
        tid = str(tool_id or "").strip()
        return bool(tid) and (
            tid in self.tool_ids
            or any(tid.startswith(prefix) for prefix in self.tool_prefixes)
        )


@dataclass(frozen=True)
class CheckpointPolicy:
    mode: str = "none"
    interval_sec: float = 0.0
    resume: bool = False

    @classmethod
    def from_mapping(cls, raw: Any) -> "CheckpointPolicy":
        data = _mapping(raw, "checkpoint")
        return cls(
            mode=str(data.get("mode") or "none").strip().lower(),
            interval_sec=float(data.get("interval_sec", 0.0)),
            resume=bool(data.get("resume", False)),
        )


@dataclass(frozen=True)
class CancelPolicy:
    mode: str = "cooperative"
    grace_sec: float = 0.0

    @classmethod
    def from_mapping(cls, raw: Any) -> "CancelPolicy":
        data = _mapping(raw, "cancel")
        return cls(
            mode=str(data.get("mode") or "cooperative").strip().lower(),
            grace_sec=float(data.get("grace_sec", 0.0)),
        )


@dataclass(frozen=True)
class SandboxPolicy:
    mode: str = "process"
    network: str = "inherit"
    filesystem: str = "inherit"
    executable_allowlist: Tuple[str, ...] = ()
    env_allowlist: Tuple[str, ...] = ("PATH", "LANG", "LC_ALL", "TZ")
    output_limit_bytes: int = 2 * 1024 * 1024
    required_capability: str = ""

    @classmethod
    def from_mapping(cls, raw: Any) -> "SandboxPolicy":
        data = _mapping(raw, "sandbox")
        return cls(
            mode=str(data.get("mode") or "process").strip().lower(),
            network=str(data.get("network") or "inherit").strip().lower(),
            filesystem=str(data.get("filesystem") or "inherit").strip().lower(),
            executable_allowlist=_tuple(data.get("executable_allowlist")),
            env_allowlist=_tuple(
                data.get("env_allowlist"),
                ("PATH", "LANG", "LC_ALL", "TZ"),
            ),
            output_limit_bytes=max(
                4096, int(data.get("output_limit_bytes", 2 * 1024 * 1024))
            ),
            required_capability=str(
                data.get("required_capability") or ""
            ).strip(),
        )


@dataclass(frozen=True)
class WorkloadSpec:
    """Runtime-neutral workload metadata loaded from a tool manifest."""

    tool_id: str
    execution_mode: str = "inline"
    backend: str = "local"
    job_id_fields: Tuple[str, ...] = ("job_id", "run_id")
    owner_namespace: str = "workloads"
    resource_class: str = "default"
    weight: float = 1.0
    soft_timeout_sec: float = 0.0
    hard_timeout_sec: float = 0.0
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    idempotency: IdempotencyPolicy = field(default_factory=IdempotencyPolicy)
    affinity: AffinityPolicy = field(default_factory=AffinityPolicy)
    checkpoint: CheckpointPolicy = field(default_factory=CheckpointPolicy)
    cancel: CancelPolicy = field(default_factory=CancelPolicy)
    sandbox: SandboxPolicy = field(default_factory=SandboxPolicy)

    @classmethod
    def from_manifest(cls, tool_id: str, metadata: Mapping[str, Any]) -> "WorkloadSpec":
        raw = metadata.get("workload")
        data = _mapping(raw, "workload")
        if not data:
            raise ValueError("manifest has no workload declaration")
        timeouts = _mapping(data.get("timeouts"), "workload.timeouts")
        spec = cls(
            tool_id=str(tool_id or metadata.get("id") or "").strip(),
            execution_mode=str(
                data.get("execution_mode") or data.get("mode") or "inline"
            ).strip().lower(),
            backend=str(data.get("backend") or "local").strip().lower(),
            job_id_fields=_tuple(
                data.get("job_id_fields") or data.get("job_id_keys"),
                ("job_id", "run_id"),
            ),
            owner_namespace=str(data.get("owner_namespace") or "workloads").strip(),
            resource_class=str(data.get("resource_class") or "default").strip(),
            weight=float(data.get("weight", 1.0)),
            soft_timeout_sec=float(
                data.get("soft_timeout_sec", timeouts.get("soft_sec", 0.0))
            ),
            hard_timeout_sec=float(
                data.get("hard_timeout_sec", timeouts.get("hard_sec", 0.0))
            ),
            retry=RetryPolicy.from_mapping(data.get("retry")),
            idempotency=IdempotencyPolicy.from_mapping(data.get("idempotency")),
            affinity=AffinityPolicy.from_mapping(data.get("affinity")),
            checkpoint=CheckpointPolicy.from_mapping(data.get("checkpoint")),
            cancel=CancelPolicy.from_mapping(data.get("cancel")),
            sandbox=SandboxPolicy.from_mapping(data.get("sandbox")),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.tool_id:
            raise ValueError("workload tool_id is required")
        if not self.job_id_fields:
            raise ValueError("workload job_id_fields cannot be empty")
        if self.weight <= 0:
            raise ValueError("workload weight must be positive")
        if self.soft_timeout_sec < 0 or self.hard_timeout_sec < 0:
            raise ValueError("workload timeouts must be non-negative")
        if (
            self.soft_timeout_sec
            and self.hard_timeout_sec
            and self.soft_timeout_sec > self.hard_timeout_sec
        ):
            raise ValueError("soft timeout cannot exceed hard timeout")
        if self.backend in {"process", "subprocess"}:
            if not self.sandbox.executable_allowlist:
                raise ValueError(
                    "subprocess workload requires sandbox.executable_allowlist"
                )
            if self.sandbox.filesystem != "workspace":
                raise ValueError(
                    "subprocess workload requires sandbox.filesystem=workspace"
                )

    @property
    def detached(self) -> bool:
        return self.execution_mode in {"detached", "deferred", "queued"}

    # Compatibility for the current route layer; the unified runtime consumes
    # job_id_fields and affinity directly.
    @property
    def job_id_keys(self) -> Tuple[str, ...]:
        return self.job_id_fields

    @property
    def affinity_tool_ids(self) -> frozenset[str]:
        return self.affinity.tool_ids

    @property
    def affinity_prefixes(self) -> Tuple[str, ...]:
        return self.affinity.tool_prefixes

    def matches_affinity(self, tool_id: str) -> bool:
        return self.affinity.matches_tool(tool_id)
