"""Hub-chat host adapter: resolve catalog tools per orchestrator tools_binding.

Engines declare *how* they accept tools (manifest.tools_binding). The host
decides *which* tools fit the turn/page and injects them accordingly.
Tool-runner manifests are never rewritten.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

ToolFn = Callable[[str, Dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True)
class ToolsBinding:
    mode: str
    config_key: str = "tool_ids"
    host_port: str = "tools"
    selection: str = "host_provided"
    surface: Optional[Dict[str, Any]] = None
    raw: Optional[Dict[str, Any]] = None

    @property
    def is_kernel_callable(self) -> bool:
        return self.mode == "kernel_callable"

    @property
    def is_openai_functions(self) -> bool:
        return self.mode == "openai_functions"

    @property
    def namespace(self) -> str:
        """Kernel name a ``kernel_callable`` engine binds the tools under (``surface.namespace``)."""
        from src.orchestrators._shared.tool_surface import surface_namespace

        return surface_namespace(self.surface, default=self.host_port)


def read_tools_binding(manifest: Any) -> Optional[ToolsBinding]:
    """Parse tools_binding from a Manifest or raw dict."""
    raw: Any = None
    if manifest is None:
        return None
    if isinstance(manifest, Mapping):
        raw = manifest.get("tools_binding")
        caps = manifest.get("capabilities") or []
    else:
        raw = getattr(manifest, "tools_binding", None)
        caps = getattr(manifest, "capabilities", None) or []
    if not isinstance(raw, Mapping):
        return None
    mode = str(raw.get("mode") or "").strip()
    if not mode:
        return None
    # Engines without tools capability should not receive a binding.
    if caps and "tools" not in {str(c).strip() for c in caps}:
        return None
    surface = raw.get("surface") if isinstance(raw.get("surface"), Mapping) else None
    return ToolsBinding(
        mode=mode,
        config_key=str(raw.get("config_key") or "tool_ids").strip() or "tool_ids",
        host_port=str(raw.get("host_port") or "tools").strip() or "tools",
        selection=str(raw.get("selection") or "host_provided").strip() or "host_provided",
        surface=dict(surface) if surface else None,
        raw=dict(raw),
    )


def _tool_id(row: Mapping[str, Any]) -> str:
    return str(row.get("id") or row.get("name") or "").strip()


def _normalize_id_list(raw: Any) -> List[str]:
    if not isinstance(raw, (list, tuple)):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for item in raw:
        tid = str(item or "").strip()
        if not tid or tid in seen:
            continue
        seen.add(tid)
        out.append(tid)
    return out


def _intersect_allowed(ids: Sequence[str], allowed_manifest: Sequence[Mapping[str, Any]]) -> List[str]:
    allowed = {_tool_id(t) for t in allowed_manifest if _tool_id(t)}
    if not allowed:
        return []
    return [tid for tid in ids if tid in allowed]


async def resolve_tool_ids_for_run(
    *,
    binding: Optional[ToolsBinding],
    tool_executor: Any,
    user_id: str = "",
    user_message: str = "",
    orchestrator_config: Optional[Mapping[str, Any]] = None,
    file_registry: Optional[Sequence[Mapping[str, Any]]] = None,
    artifacts: Optional[Sequence[Mapping[str, Any]]] = None,
    active_plugin: Optional[str] = None,
    discourse_focus: Optional[Mapping[str, Any]] = None,
    current_file_context: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Resolve allowlist of tool ids for a hub run.

    Override: non-empty ``orchestrator_config[config_key]`` wins (RBAC-intersected).
    Otherwise for ``host_provided`` / ``kernel_callable``: same selection as native
    chat (``select_llm_tools_for_turn`` + RBAC).
    """
    if binding is None:
        return []

    cfg = dict(orchestrator_config or {}) if isinstance(orchestrator_config, Mapping) else {}
    config_key = binding.config_key
    override = _normalize_id_list(cfg.get(config_key))

    from src.auth.permissions_client import resolve_allowed_tools
    from src.tools.llm_tool_selection import page_allows_tool, select_llm_tools_for_turn

    try:
        manifest = await tool_executor.get_tools_manifest()
    except Exception:
        logger.warning("tools_binding: failed to load tools manifest", exc_info=True)
        manifest = []
    if not isinstance(manifest, list):
        manifest = []

    allowed = await resolve_allowed_tools(manifest, user_id or None)
    if not isinstance(allowed, list):
        allowed = []

    if override:
        page = str(active_plugin).strip() if active_plugin else None
        return _intersect_allowed(override, [t for t in allowed if page_allows_tool(t, page)])

    # openai_functions + host_turn: native engine selects itself; hub kernel path
    # still benefits from the same selection when asked.
    if binding.selection not in ("host_provided", "host_turn") and not binding.is_kernel_callable:
        return []

    registry: List[Dict[str, Any]] = []
    for row in list(file_registry or []) + list(artifacts or []):
        if isinstance(row, Mapping):
            registry.append(dict(row))

    selected = select_llm_tools_for_turn(
        allowed,
        user_message=user_message or "",
        current_file_context=dict(current_file_context) if isinstance(current_file_context, Mapping) else None,
        session_file_registry=registry or None,
        discourse_focus=dict(discourse_focus) if isinstance(discourse_focus, Mapping) else None,
        active_plugin=str(active_plugin).strip() if active_plugin else None,
    )
    ids = [_tool_id(t) for t in selected if _tool_id(t)]
    # Preserve order, drop empties/dupes already handled by selection.
    return ids


def host_owned_tool_args() -> frozenset:
    """Argument names the host fills from the authenticated chat."""
    from src.config import settings

    raw = str(getattr(settings, "TOOL_HOST_OWNED_ARGS", "") or "")
    return frozenset(a.strip() for a in raw.split(",") if a.strip())


def _without_host_args(spec: Mapping[str, Any], owned: frozenset) -> Dict[str, Any]:
    fn = dict(spec.get("function") or {})
    params = fn.get("parameters")
    if isinstance(params, Mapping) and isinstance(params.get("properties"), Mapping):
        params = dict(params)
        params["properties"] = {k: v for k, v in params["properties"].items() if k not in owned}
        if isinstance(params.get("required"), list):
            params["required"] = [r for r in params["required"] if r not in owned]
        fn["parameters"] = params
    return {**dict(spec), "function": fn}


async def kernel_tool_specs(tool_executor: Any, tool_ids: Sequence[str]) -> List[Dict[str, Any]]:
    """OpenAI-style schemas for ``tool_ids`` as a kernel-callable engine sees them.

    Canonical tool-runner schemas first (same source as native function
    calling), manifest parse for any the runner did not return; host-owned
    arguments are removed because the host port supplies them.
    """
    ids = _normalize_id_list(list(tool_ids or []))
    if not ids:
        return []
    specs: List[Dict[str, Any]] = []
    try:
        specs = list(await tool_executor.get_openai_tools_for_ids(ids) or [])
    except Exception:
        logger.warning("kernel_tool_specs: canonical schemas unavailable", exc_info=True)
    have = {str((s.get("function") or {}).get("name") or "") for s in specs if isinstance(s, Mapping)}
    missing = [t for t in ids if t not in have]
    if missing:
        try:
            manifest = await tool_executor.get_tools_manifest()
            rows = [r for r in manifest or [] if isinstance(r, Mapping) and _tool_id(r) in set(missing)]
            specs += list(tool_executor.get_openai_tools_format(rows) or [])
        except Exception:
            logger.warning("kernel_tool_specs: manifest fallback failed", exc_info=True)
    order = {t: i for i, t in enumerate(ids)}
    owned = host_owned_tool_args()
    out = [_without_host_args(s, owned) for s in specs if isinstance(s, Mapping)]
    return sorted(out, key=lambda s: order.get(str((s.get("function") or {}).get("name") or ""), len(order)))


def make_tools_host_port(
    tool_executor: Any,
    *,
    session_id: str,
    user_id: str = "",
    user_message: str = "",
) -> ToolFn:
    """Build HostPorts.tools callable backed by ToolExecutor.

    The chat turn's tool context is bound here: kernels serve tool calls from
    tasks that may predate this turn.
    """
    from src.tools.side_effects import current_turn, use_turn

    owned = host_owned_tool_args()
    turn = current_turn()

    async def _run(tool_id: str, args: Dict[str, Any]) -> Any:
        clean = {k: v for k, v in dict(args or {}).items() if k not in owned}
        with use_turn(turn or current_turn()):
            return await tool_executor.execute(
                tool_id,
                clean,
                session_id=session_id,
                user_id=user_id,
                user_message=user_message,
            )

    return _run
