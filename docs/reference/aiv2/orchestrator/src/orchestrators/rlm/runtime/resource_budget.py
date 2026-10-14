"""Optional per-run RLM fan-out counters.

Real backpressure for kernels is ``KernelGovernor`` (live CPU/RAM + fair
share). The fixed per-run child/LLM caps below default to **unlimited** (0)
so a single quiet user is not artificially throttled; set env/cfg only if you
need an explicit hard ceiling for tests or emergencies.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Union


def _non_negative(raw: Any, default: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(0, value)


def _positive_float(raw: Any, default: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default
    return max(0.1, value)


def _limit(cfg: Dict[str, Any], key: str, env: str, default: int) -> int:
    """0 means unlimited (no per-run ceiling)."""
    return _non_negative(cfg.get(key, os.environ.get(env, default)), default)


async def _wait_slot(
    wait_fn,
    *,
    timeout_sec: float,
    timeout_error: str,
) -> None:
    """Wait until ``wait_fn`` returns; ``timeout_sec <= 0`` waits forever."""
    timeout = float(timeout_sec)
    if timeout <= 0:
        await wait_fn()
        return
    try:
        await asyncio.wait_for(wait_fn(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(timeout_error) from exc


@dataclass
class BudgetSnapshot:
    max_total_children: int
    max_concurrent_children: int
    total_children: int
    active_children: int
    max_concurrent_llm_calls: int
    active_llm_calls: int


class RunResourceBudget:
    """Shared by every root/child session carrying the same ``run_id``.

    Defaults are unlimited; optional cfg/env ceilings remain for tests.
    """

    def __init__(self, run_id: str, cfg: Dict[str, Any]) -> None:
        self.run_id = run_id
        # 0 = unlimited. KernelGovernor is the real capacity gate.
        self.max_total_children = _limit(
            cfg, "rlm_max_children_per_run", "RLM_MAX_CHILDREN_PER_RUN", 0
        )
        self.max_concurrent_children = _limit(
            cfg, "rlm_max_concurrent_children", "RLM_MAX_CONCURRENT_CHILDREN", 0
        )
        self.max_concurrent_llm_calls = _limit(
            cfg,
            "rlm_max_concurrent_llm_calls",
            "RLM_MAX_CONCURRENT_LLM_CALLS",
            0,
        )
        self.total_children = 0
        self.active_children = 0
        self.active_llm_calls = 0
        self.last_used = time.monotonic()
        self._condition: Optional[asyncio.Condition] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _cond(self) -> asyncio.Condition:
        loop = asyncio.get_running_loop()
        if self._condition is None or self._loop is not loop:
            self._loop = loop
            self._condition = asyncio.Condition()
        return self._condition

    def _total_capped(self) -> bool:
        return self.max_total_children > 0 and self.total_children >= self.max_total_children

    def _concurrent_full(self) -> bool:
        return (
            self.max_concurrent_children > 0
            and self.active_children >= self.max_concurrent_children
        )

    def _llm_full(self) -> bool:
        return (
            self.max_concurrent_llm_calls > 0
            and self.active_llm_calls >= self.max_concurrent_llm_calls
        )

    async def reserve_child(self, *, timeout_sec: float = 0.0) -> None:
        cond = self._cond()
        async with cond:
            if self._total_capped():
                raise RuntimeError(
                    "rlm_child_budget_exhausted:"
                    f"total={self.total_children}:max={self.max_total_children}"
                )

            async def _wait_for_slot() -> None:
                while self._concurrent_full():
                    await cond.wait()

            await _wait_slot(
                _wait_for_slot,
                timeout_sec=timeout_sec,
                timeout_error=(
                    "rlm_child_concurrency_timeout:"
                    f"active={self.active_children}:max={self.max_concurrent_children}"
                ),
            )
            if self._total_capped():
                raise RuntimeError(
                    "rlm_child_budget_exhausted:"
                    f"total={self.total_children}:max={self.max_total_children}"
                )
            self.total_children += 1
            self.active_children += 1
            self.last_used = time.monotonic()

    async def release_child(self) -> None:
        cond = self._cond()
        async with cond:
            self.active_children = max(0, self.active_children - 1)
            self.last_used = time.monotonic()
            cond.notify_all()

    async def reserve_llm(self, *, timeout_sec: float = 0.0) -> None:
        """Reserve one concurrent LLM slot when a ceiling is configured.

        Cumulative work is bounded by each session's ``rlm_max_turns``. Default
        is unlimited; gateway/admission handle real backpressure.
        """
        cond = self._cond()
        async with cond:
            async def _wait_for_slot() -> None:
                while self._llm_full():
                    await cond.wait()

            await _wait_slot(
                _wait_for_slot,
                timeout_sec=timeout_sec,
                timeout_error=(
                    "rlm_llm_concurrency_timeout:"
                    f"active={self.active_llm_calls}:"
                    f"max={self.max_concurrent_llm_calls}"
                ),
            )
            self.active_llm_calls += 1
            self.last_used = time.monotonic()

    async def release_llm(self) -> None:
        cond = self._cond()
        async with cond:
            self.active_llm_calls = max(0, self.active_llm_calls - 1)
            self.last_used = time.monotonic()
            cond.notify_all()

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            max_total_children=self.max_total_children,
            max_concurrent_children=self.max_concurrent_children,
            total_children=self.total_children,
            active_children=self.active_children,
            max_concurrent_llm_calls=self.max_concurrent_llm_calls,
            active_llm_calls=self.active_llm_calls,
        )


class KernelGovernor:
    """Live CPU/RAM admission for RLM kernel subprocesses."""

    def __init__(self) -> None:
        from src.orchestrators.rlm.runtime.kernel_admission import (
            KernelAdmissionPolicy,
            current_resource_snapshot,
            fair_share_allows,
        )

        self._policy = KernelAdmissionPolicy.from_env()
        self._read_resources = current_resource_snapshot
        self._fair_share_allows = fair_share_allows
        self.active = 0
        self.waiting = 0
        self.by_user: Dict[str, int] = {}
        self._condition: Optional[asyncio.Condition] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue_poll_sec = _positive_float(
            os.environ.get("RLM_KERNEL_QUEUE_POLL_SEC"), 2.0
        )

    def _cond(self) -> asyncio.Condition:
        loop = asyncio.get_running_loop()
        if self._condition is None or self._loop is not loop:
            self._loop = loop
            self._condition = asyncio.Condition()
        return self._condition

    def _may_acquire(self, uid: str) -> bool:
        snap = self._read_resources()
        if not snap.allows_new_kernel(self.active, self._policy):
            return False
        return self._fair_share_allows(
            uid=uid,
            active=self.active,
            by_user=self.by_user,
            snap=snap,
            policy=self._policy,
        )

    def _waiting_status(self, uid: str) -> Dict[str, Any]:
        snap = self._read_resources()
        return {
            "user_id": uid,
            "waiting": self.waiting,
            "active": self.active,
            "user_active": self.by_user.get(uid, 0),
            "users": len(self.by_user),
            "load_ratio": round(snap.load_ratio, 3),
            "mem_used_ratio": round(snap.mem_used_ratio, 3),
            "available_mb": snap.available_mb,
        }

    async def _emit_waiting(
        self,
        on_waiting: Callable[[Dict[str, Any]], Union[Awaitable[Any], Any]],
        uid: str,
    ) -> None:
        payload = self._waiting_status(uid)
        result = on_waiting(payload)
        if asyncio.iscoroutine(result):
            await result

    def _grant(self, uid: str) -> None:
        self.active += 1
        self.by_user[uid] = self.by_user.get(uid, 0) + 1

    async def acquire(
        self,
        user_id: str,
        *,
        on_waiting: Optional[
            Callable[[Dict[str, Any]], Union[Awaitable[Any], Any]]
        ] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        """Wait in queue until CPU/RAM allow a kernel — no hard timeout failure."""
        uid = str(user_id or "").strip() or "__anonymous__"
        cond = self._cond()
        poll = max(0.5, float(self._queue_poll_sec))
        async with cond:
            if self._may_acquire(uid):
                self._grant(uid)
                return

            self.waiting += 1
            try:
                while not self._may_acquire(uid):
                    if cancel_check and cancel_check():
                        raise asyncio.CancelledError("rlm_kernel_queue_cancelled")
                    if on_waiting:
                        await self._emit_waiting(on_waiting, uid)
                    try:
                        await asyncio.wait_for(cond.wait(), timeout=poll)
                    except asyncio.TimeoutError:
                        # Re-check live CPU/RAM and emit another waiting pulse.
                        pass
                self._grant(uid)
            finally:
                self.waiting = max(0, self.waiting - 1)

    async def release(self, user_id: str) -> None:
        uid = str(user_id or "").strip() or "__anonymous__"
        cond = self._cond()
        async with cond:
            self.active = max(0, self.active - 1)
            count = self.by_user.get(uid, 0) - 1
            if count > 0:
                self.by_user[uid] = count
            else:
                self.by_user.pop(uid, None)
            cond.notify_all()

    def snapshot(self) -> Dict[str, Any]:
        snap = self._read_resources()
        return {
            "active": self.active,
            "waiting": self.waiting,
            "users": len(self.by_user),
            "load_ratio": round(snap.load_ratio, 3),
            "mem_used_ratio": round(snap.mem_used_ratio, 3),
            "available_mb": snap.available_mb,
            "policy": {
                "min_free_mb": self._policy.min_free_mb,
                "mem_per_kernel_mb": self._policy.mem_per_kernel_mb,
                "mem_pressure_max": self._policy.mem_pressure_max,
                "cpu_load_max": self._policy.cpu_load_max,
            },
        }


_RUN_BUDGETS: Dict[str, RunResourceBudget] = {}
_KERNEL_GOVERNOR = KernelGovernor()


def get_run_budget(run_id: Any, cfg: Dict[str, Any]) -> RunResourceBudget:
    key = str(run_id or "").strip()
    if not key:
        # Session ids are assigned immediately after this call site; callers
        # without a run id should not accidentally share one global bucket.
        key = f"anonymous:{id(cfg)}"
    budget = _RUN_BUDGETS.get(key)
    if budget is None:
        if len(_RUN_BUDGETS) >= 1024:
            idle = sorted(
                (
                    (k, b)
                    for k, b in _RUN_BUDGETS.items()
                    if b.active_children == 0 and b.active_llm_calls == 0
                ),
                key=lambda item: item[1].last_used,
            )
            for old_key, _ in idle[: max(1, len(idle) // 4)]:
                _RUN_BUDGETS.pop(old_key, None)
        budget = RunResourceBudget(key, cfg)
        _RUN_BUDGETS[key] = budget
    return budget


def kernel_governor() -> KernelGovernor:
    return _KERNEL_GOVERNOR


def reset_resource_budgets_for_tests() -> None:
    global _KERNEL_GOVERNOR
    from src.orchestrators.rlm.runtime.kernel_admission import (
        set_resource_reader_for_tests,
    )

    _RUN_BUDGETS.clear()
    set_resource_reader_for_tests(None)
    _KERNEL_GOVERNOR = KernelGovernor()
