"""Generic workload contract and registry."""

from .registry import (
    WorkloadRegistration,
    WorkloadRegistry,
    extract_job_id,
    is_workload,
    load_from_tool_registry,
    workload_registry,
)
from .spec import (
    AffinityPolicy,
    CancelPolicy,
    CheckpointPolicy,
    IdempotencyPolicy,
    RetryPolicy,
    SandboxPolicy,
    WorkloadSpec,
)

__all__ = [
    "AffinityPolicy",
    "CancelPolicy",
    "CheckpointPolicy",
    "IdempotencyPolicy",
    "RetryPolicy",
    "SandboxPolicy",
    "WorkloadRegistration",
    "WorkloadRegistry",
    "WorkloadSpec",
    "extract_job_id",
    "is_workload",
    "load_from_tool_registry",
    "workload_registry",
]
