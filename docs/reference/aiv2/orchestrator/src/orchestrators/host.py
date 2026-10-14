"""Host port catalog and negotiation for orchestrator engines.

Known ports are documented, not exclusive — engines may declare custom ports
in their manifest. Required ports must be present on ``HostPorts`` before
``create_session``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence

# Documented vocabulary (open — custom ids allowed).
KNOWN_HOST_PORTS = frozenset(
    {
        "chat",
        "chat_stream",
        "tools",
        "pools.search",
        "pools.resolve",
        "pools.list",
        "pools.read",
        "events",
        "spawn_child",
        "model_finder",
        "environment",
        "process.spawn",
    }
)

ChatFn = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
ChatStreamFn = Callable[..., Awaitable[Dict[str, Any]]]
ToolFn = Callable[[str, Dict[str, Any]], Awaitable[Any]]
PoolSearchFn = Callable[[str, str], Awaitable[Dict[str, Any]]]
PoolResolveFn = Callable[[str], Awaitable[Dict[str, Any]]]
PoolListFn = Callable[[str], Awaitable[Dict[str, Any]]]
PoolReadFn = Callable[..., Awaitable[Dict[str, Any]]]
EventFn = Callable[[Dict[str, Any]], None]
EnvironmentFn = Callable[[str], Awaitable[Dict[str, Any]]]
ProcessSpawnFn = Callable[..., Any]


@dataclass
class HostPorts:
    """Injected host capabilities for an orchestrator session."""

    chat: Optional[ChatFn] = None
    chat_stream: Optional[ChatStreamFn] = None
    tools: Optional[ToolFn] = None
    pool_search: Optional[PoolSearchFn] = None
    pool_resolve: Optional[PoolResolveFn] = None
    pool_list: Optional[PoolListFn] = None
    pool_read: Optional[PoolReadFn] = None
    events: Optional[EventFn] = None
    spawn_child: Optional[Callable[..., Awaitable[Any]]] = None
    model_finder: Optional[Callable[..., Awaitable[List[Dict[str, Any]]]]] = None
    environment: Optional[EnvironmentFn] = None
    process_spawn: Optional[ProcessSpawnFn] = None
    # Forward-compatible bag for custom / future ports.
    extras: Dict[str, Any] = field(default_factory=dict)

    def has(self, port_id: str) -> bool:
        pid = str(port_id or "").strip()
        if not pid:
            return False
        mapping = {
            "chat": self.chat,
            "chat_stream": self.chat_stream,
            "tools": self.tools,
            "pools.search": self.pool_search,
            "pools.resolve": self.pool_resolve,
            "pools.list": self.pool_list,
            "pools.read": self.pool_read,
            "events": self.events,
            "spawn_child": self.spawn_child,
            "model_finder": self.model_finder,
            "environment": self.environment,
            "process.spawn": self.process_spawn,
        }
        if pid in mapping:
            return mapping[pid] is not None
        return pid in self.extras and self.extras[pid] is not None

    def get(self, port_id: str, default: Any = None) -> Any:
        pid = str(port_id or "").strip()
        mapping = {
            "chat": self.chat,
            "chat_stream": self.chat_stream,
            "tools": self.tools,
            "pools.search": self.pool_search,
            "pools.resolve": self.pool_resolve,
            "pools.list": self.pool_list,
            "pools.read": self.pool_read,
            "events": self.events,
            "spawn_child": self.spawn_child,
            "model_finder": self.model_finder,
            "environment": self.environment,
            "process.spawn": self.process_spawn,
        }
        if pid in mapping:
            val = mapping[pid]
            return default if val is None else val
        return self.extras.get(pid, default)


@dataclass(frozen=True)
class HostPortCatalog:
    """Static documentation of known host ports (not an exclusive allow-list)."""

    known: frozenset[str] = KNOWN_HOST_PORTS

    def is_known(self, port_id: str) -> bool:
        return str(port_id or "").strip() in self.known


@dataclass(frozen=True)
class HostNegotiationResult:
    ok: bool
    missing_required: List[str] = field(default_factory=list)
    provided: List[str] = field(default_factory=list)


def negotiate_host_ports(
    declared: Sequence[Mapping[str, Any]] | None,
    host: HostPorts,
) -> HostNegotiationResult:
    """Ensure every required host_port from the manifest is present on ``host``."""
    missing: List[str] = []
    provided: List[str] = []
    for item in declared or ():
        if not isinstance(item, Mapping):
            continue
        pid = str(item.get("id") or "").strip()
        if not pid:
            continue
        required = bool(item.get("required", False))
        if host.has(pid):
            provided.append(pid)
        elif required:
            missing.append(pid)
    return HostNegotiationResult(
        ok=not missing,
        missing_required=missing,
        provided=provided,
    )
