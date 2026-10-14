"""Plugin-neutral workload and post-persistence hook registry."""

from __future__ import annotations

import importlib
import inspect
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from .spec import WorkloadSpec

logger = logging.getLogger(__name__)
PersistStampHook = Callable[[Dict[str, Any]], Any]
LifecycleHook = Callable[[], Any]
JobHook = Callable[[str], Any]
EventStreamHook = Callable[[str, Mapping[str, Any]], Any]


@dataclass(frozen=True)
class WorkloadRegistration:
    spec: WorkloadSpec
    persist_stamp_hook: Optional[PersistStampHook] = None
    startup_hook: Optional[LifecycleHook] = None
    shutdown_hook: Optional[LifecycleHook] = None
    status_hook: Optional[JobHook] = None
    cancel_hook: Optional[JobHook] = None
    checkpoint_hook: Optional[JobHook] = None
    event_stream_hook: Optional[EventStreamHook] = None


class WorkloadRegistry:
    def __init__(self) -> None:
        self._registrations: Dict[str, WorkloadRegistration] = {}

    def clear(self) -> None:
        self._registrations.clear()

    def register(
        self,
        spec: WorkloadSpec,
        *,
        persist_stamp_hook: Optional[PersistStampHook] = None,
        startup_hook: Optional[LifecycleHook] = None,
        shutdown_hook: Optional[LifecycleHook] = None,
        status_hook: Optional[JobHook] = None,
        cancel_hook: Optional[JobHook] = None,
        checkpoint_hook: Optional[JobHook] = None,
        event_stream_hook: Optional[EventStreamHook] = None,
    ) -> WorkloadRegistration:
        spec.validate()
        registration = WorkloadRegistration(
            spec,
            persist_stamp_hook,
            startup_hook,
            shutdown_hook,
            status_hook,
            cancel_hook,
            checkpoint_hook,
            event_stream_hook,
        )
        self._registrations[spec.tool_id] = registration
        return registration

    def unregister(self, tool_id: str) -> None:
        self._registrations.pop(str(tool_id or "").strip(), None)

    def get(self, tool_id: str) -> Optional[WorkloadRegistration]:
        return self._registrations.get(str(tool_id or "").strip())

    def spec_for(self, tool_id: str) -> Optional[WorkloadSpec]:
        registration = self.get(tool_id)
        return registration.spec if registration else None

    def specs(self) -> Tuple[WorkloadSpec, ...]:
        return tuple(
            self._registrations[tool_id].spec
            for tool_id in sorted(self._registrations)
        )

    async def stamp_persisted(self, tool_id: str, result: Dict[str, Any]) -> None:
        registration = self.get(tool_id)
        hook = registration.persist_stamp_hook if registration else None
        if hook is None:
            return
        value = hook(result)
        if inspect.isawaitable(value):
            await value

    async def startup(self) -> None:
        await self._run_lifecycle("startup_hook", reversed_order=False)

    async def shutdown(self) -> None:
        await self._run_lifecycle("shutdown_hook", reversed_order=True)

    async def status(self, tool_id: str, job_id: str) -> Optional[Dict[str, Any]]:
        registration = self.get(tool_id)
        hook = registration.status_hook if registration else None
        if hook is None:
            return None
        value = hook(job_id)
        if inspect.isawaitable(value):
            value = await value
        return dict(value) if isinstance(value, Mapping) else None

    async def cancel(self, tool_id: str, job_id: str) -> bool:
        return bool(await self._run_job_hook(tool_id, job_id, "cancel_hook"))

    async def checkpoint(self, tool_id: str, job_id: str) -> bool:
        registration = self.get(tool_id)
        if registration is None or registration.checkpoint_hook is None:
            return False
        await self._run_job_hook(tool_id, job_id, "checkpoint_hook")
        return True

    def event_stream(
        self,
        tool_id: str,
        job_id: str,
        context: Mapping[str, Any],
    ) -> Any:
        registration = self.get(tool_id)
        hook = registration.event_stream_hook if registration else None
        return hook(job_id, context) if hook is not None else None

    async def _run_job_hook(self, tool_id: str, job_id: str, field: str) -> Any:
        registration = self.get(tool_id)
        hook = getattr(registration, field) if registration else None
        if hook is None:
            return None
        value = hook(job_id)
        if inspect.isawaitable(value):
            value = await value
        return value

    async def _run_lifecycle(self, field: str, *, reversed_order: bool) -> None:
        registrations = list(self._registrations.values())
        if reversed_order:
            registrations.reverse()
        for registration in registrations:
            hook = getattr(registration, field)
            if hook is None:
                continue
            value = hook()
            if inspect.isawaitable(value):
                await value


def _load_entrypoint(reference: str) -> Callable[..., Any]:
    module_name, separator, attribute = str(reference or "").strip().partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("workload adapter must use 'module.path:function' syntax")
    callback = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(callback):
        raise ValueError(f"workload adapter is not callable: {reference}")
    return callback


def _iter_adapters(tool_registry: Any) -> Iterable[Tuple[str, Any]]:
    listed = tool_registry.list() if hasattr(tool_registry, "list") else {}
    for tool_id in listed:
        if hasattr(tool_registry, "get_adapter"):
            adapter = tool_registry.get_adapter(tool_id)
        elif hasattr(tool_registry, "get"):
            adapter = tool_registry.get(tool_id)
        else:
            adapter = None
        if adapter is not None:
            yield str(tool_id), adapter


def load_from_tool_registry(
    tool_registry: Any,
    *,
    target: Optional[WorkloadRegistry] = None,
) -> WorkloadRegistry:
    """Rebuild registrations from discovered tool/plugin manifest metadata."""
    registry = target or workload_registry
    registry.clear()
    for tool_id, adapter in _iter_adapters(tool_registry):
        metadata = getattr(adapter, "meta", None)
        if not isinstance(metadata, Mapping) or not isinstance(
            metadata.get("workload"), Mapping
        ):
            continue
        try:
            spec = WorkloadSpec.from_manifest(tool_id, metadata)
            adapter_ref = metadata["workload"].get("adapter")
            if adapter_ref:
                callback = _load_entrypoint(str(adapter_ref))
                callback(registry=registry, spec=spec, metadata=metadata)
                if registry.spec_for(tool_id) is None:
                    raise ValueError("workload adapter did not register its spec")
            else:
                registry.register(spec)
        except Exception as exc:
            logger.warning("Invalid workload declaration for %s: %s", tool_id, exc)
    return registry


workload_registry = WorkloadRegistry()


def is_workload(tool_id: str) -> bool:
    spec = workload_registry.spec_for(tool_id)
    return bool(spec and spec.detached)


def extract_job_id(tool_id: str, args: Optional[Mapping[str, Any]]) -> str:
    spec = workload_registry.spec_for(tool_id)
    fields = spec.job_id_fields if spec else ("job_id", "run_id")
    if isinstance(args, Mapping):
        for field_name in fields:
            value = str(args.get(field_name) or "").strip()
            if value:
                return value
    return str(uuid.uuid4())
