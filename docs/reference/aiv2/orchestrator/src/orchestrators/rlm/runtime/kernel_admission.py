"""Dynamic RLM kernel admission from live CPU/RAM headroom.

Replaces fixed per-worker/per-user kernel caps. When the pod has spare
capacity, new kernels start immediately. Under pressure, users share slots
proportionally instead of hitting a hard per-user number.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


def _positive_float(raw: object, default: float) -> float:
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        value = default
    return max(0.0, value)


def _positive_int(raw: object, default: int) -> int:
    try:
        value = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        value = default
    return max(1, value)


@dataclass(frozen=True)
class KernelAdmissionPolicy:
    min_free_mb: int
    mem_per_kernel_mb: int
    mem_pressure_max: float
    cpu_load_max: float

    @classmethod
    def from_env(cls) -> "KernelAdmissionPolicy":
        return cls(
            min_free_mb=_positive_int(os.environ.get("RLM_KERNEL_MIN_FREE_MB"), 512),
            mem_per_kernel_mb=_positive_int(
                os.environ.get("RLM_KERNEL_MEM_PER_KERNEL_MB"), 256
            ),
            mem_pressure_max=_positive_float(
                os.environ.get("RLM_KERNEL_MEM_PRESSURE_MAX"), 0.90
            ),
            cpu_load_max=_positive_float(
                os.environ.get("RLM_KERNEL_CPU_LOAD_MAX"), 1.25
            ),
        )


@dataclass(frozen=True)
class ResourceSnapshot:
    load_ratio: float
    mem_used_ratio: float
    available_mb: int
    total_mb: int

    def pressure_high(self, policy: KernelAdmissionPolicy) -> bool:
        return (
            self.mem_used_ratio >= policy.mem_pressure_max
            or self.load_ratio >= policy.cpu_load_max
            or self.available_mb < policy.min_free_mb + policy.mem_per_kernel_mb
        )

    def allows_new_kernel(
        self,
        active_kernels: int,
        policy: KernelAdmissionPolicy,
    ) -> bool:
        if self.available_mb < policy.min_free_mb + policy.mem_per_kernel_mb:
            return False
        if self.mem_used_ratio >= policy.mem_pressure_max:
            return False
        if self.load_ratio >= policy.cpu_load_max:
            return False
        dynamic_cap = max(1, self.available_mb // policy.mem_per_kernel_mb)
        return active_kernels < dynamic_cap


def _read_int(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _memory_from_cgroup() -> Optional[tuple[float, int, int]]:
    root = Path("/sys/fs/cgroup")
    current = _read_int(root / "memory.current")
    if current is None:
        return None
    try:
        max_raw = (root / "memory.max").read_text().strip()
    except OSError:
        return None
    if max_raw == "max":
        return None
    try:
        limit = int(max_raw)
    except ValueError:
        return None
    if limit <= 0:
        return None
    used_ratio = min(1.0, current / limit)
    avail_mb = max(0, (limit - current) // (1024 * 1024))
    total_mb = limit // (1024 * 1024)
    return used_ratio, avail_mb, total_mb


def _memory_from_proc() -> tuple[float, int, int]:
    mem_total = 0
    mem_available = 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_total = int(line.split()[1]) // 1024
            elif line.startswith("MemAvailable:"):
                mem_available = int(line.split()[1]) // 1024
    except OSError:
        pass
    if mem_total <= 0:
        return 0.0, 4096, 4096
    used_ratio = max(0.0, min(1.0, 1.0 - (mem_available / mem_total)))
    return used_ratio, max(0, mem_available), mem_total


def _load_ratio() -> float:
    cpus = os.cpu_count() or 1
    try:
        return os.getloadavg()[0] / cpus
    except (AttributeError, OSError):
        return 0.0


def read_resource_snapshot() -> ResourceSnapshot:
    mem = _memory_from_cgroup() or _memory_from_proc()
    used_ratio, avail_mb, total_mb = mem
    return ResourceSnapshot(
        load_ratio=_load_ratio(),
        mem_used_ratio=used_ratio,
        available_mb=avail_mb,
        total_mb=total_mb,
    )


def fair_share_allows(
    *,
    uid: str,
    active: int,
    by_user: dict[str, int],
    snap: ResourceSnapshot,
    policy: KernelAdmissionPolicy,
) -> bool:
    """Under resource pressure, spread kernels across users instead of FIFO."""
    if not snap.pressure_high(policy):
        return True
    n_users = max(1, len(by_user))
    # ceil(active / n_users): grows with load but never below 1 for idle users.
    fair = max(1, (active + n_users - 1) // n_users)
    return by_user.get(uid, 0) < fair


# Test hook — override in unit tests.
_resource_reader: Callable[[], ResourceSnapshot] = read_resource_snapshot


def set_resource_reader_for_tests(
    reader: Optional[Callable[[], ResourceSnapshot]],
) -> None:
    global _resource_reader
    _resource_reader = reader or read_resource_snapshot


def current_resource_snapshot() -> ResourceSnapshot:
    return _resource_reader()
