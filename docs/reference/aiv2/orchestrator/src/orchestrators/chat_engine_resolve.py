"""Resolve which chat stream engine to use from orchestrator_id + settings.

Chat engines are whatever the drop-in registry discovers with
``extensions.chat_stream: true`` and an ``extensions.stream_entry`` of the form
``"package.module:function"`` in their manifest. Adding an engine is adding a
folder; nothing here names one.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any, Callable, Dict, FrozenSet, Mapping, Optional

from ..config import settings
from ..host_context import (
    ENGINE_FIELD,
    PAGE_FIELD,
    extract_host_envelopes,
    host_envelopes,
    resolve_active_plugin,
)
from .contract import Manifest
from .registry import get_registry

_entry_cache: Dict[str, Callable[..., Any]] = {}
_entry_lock = threading.Lock()


def _extensions(manifest: Optional[Manifest]) -> Mapping[str, Any]:
    ext = getattr(manifest, "extensions", None)
    return ext if isinstance(ext, Mapping) else {}


def stream_entry(manifest: Optional[Manifest], *, full: bool = False) -> str:
    """``module:function`` serving /chat/stream (or /chat/stream/full when ``full``)."""
    ext = _extensions(manifest)
    if not ext.get("chat_stream"):
        return ""
    if full:
        return str(ext.get("stream_entry_full") or ext.get("stream_entry") or "").strip()
    return str(ext.get("stream_entry") or "").strip()


def wraps_engines(manifest: Optional[Manifest]) -> bool:
    """Controllers that run other chat engines (must not be nested inside each other)."""
    return bool(_extensions(manifest).get("wraps_engines"))


def chat_stream_orchestrator_ids() -> FrozenSet[str]:
    """Ids of ready registry entries that can serve hub chat."""
    reg = get_registry()
    return frozenset(
        r.id for r in reg.list_orchestrators(ready_only=True) if stream_entry(r.manifest)
    )


def chat_stream_manifest(orchestrator_id: str) -> Optional[Manifest]:
    reg = get_registry().get(orchestrator_id)
    if reg is None or not reg.ready or not stream_entry(reg.manifest):
        return None
    return reg.manifest


def _configured_or_fallback(configured: str) -> str:
    ids = chat_stream_orchestrator_ids()
    oid = str(configured or "").strip()
    if oid in ids or not ids:
        return oid
    plain = sorted(i for i in ids if not wraps_engines(chat_stream_manifest(i)))
    return (plain or sorted(ids))[0]


def default_chat_orchestrator_id() -> str:
    return _configured_or_fallback(settings.DEFAULT_CHAT_ORCHESTRATOR)


def _dock_engine_by_page() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for pair in str(getattr(settings, "DOCK_CHAT_ORCHESTRATOR_BY_PAGE", "") or "").split(","):
        page, _, oid = pair.partition("=")
        if page.strip() and oid.strip():
            out[page.strip()] = oid.strip()
    return out


def dock_chat_orchestrator_id(context: Optional[Mapping[str, Any]] = None) -> str:
    """Engine for a page mini-chat: the page envelope's own choice, the per-page setting, else the dock default."""
    ids = chat_stream_orchestrator_ids()
    envelopes = [env for env in extract_host_envelopes(context) if env]
    for env in envelopes:
        asked = str(env.get(ENGINE_FIELD) or "").strip()
        if asked in ids:
            return asked
    by_page = _dock_engine_by_page()
    if by_page:
        pages = [str(env.get(PAGE_FIELD) or "").strip() for env in envelopes]
        pages.append(str(resolve_active_plugin(context) or "").strip())
        for page in pages:
            if page and by_page.get(page) in ids:
                return by_page[page]
    return _configured_or_fallback(settings.DOCK_CHAT_ORCHESTRATOR)


def is_floating_dock_context(context: Optional[Mapping[str, Any]]) -> bool:
    """True when the request carries a page envelope (page mini-chat / dock, not hub /ai-chat)."""
    return bool(host_envelopes(context))


def resolve_chat_orchestrator_id(requested: Optional[str]) -> str:
    oid = str(requested or "").strip()
    if oid and oid in chat_stream_orchestrator_ids():
        return oid
    # Unknown / archived / not-ready id — keep the default rather than crashing the turn.
    return default_chat_orchestrator_id()


def resolve_chat_orchestrator_for_request(
    requested: Optional[str],
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    """Hub may select any chat engine; a floating dock/mini-chat never inherits the hub's
    top-level choice and uses :func:`dock_chat_orchestrator_id` instead."""
    if is_floating_dock_context(context):
        return dock_chat_orchestrator_id(context)
    return resolve_chat_orchestrator_id(requested)


def import_entry(entry: str) -> Callable[..., Any]:
    """Import ``package.module:function`` (cached)."""
    with _entry_lock:
        hit = _entry_cache.get(entry)
        if hit is not None:
            return hit
    module_name, _, attr = str(entry or "").partition(":")
    if not module_name or not attr:
        raise ValueError(f"invalid stream_entry {entry!r}; want 'module:function'")
    fn = getattr(importlib.import_module(module_name), attr)
    if not callable(fn):
        raise ValueError(f"stream_entry {entry!r} is not callable")
    with _entry_lock:
        _entry_cache[entry] = fn
    return fn


def stream_agent_for_id(orchestrator_id: str, *, full: bool = False) -> Callable[..., Any]:
    """Strict lookup (no fallback) — used by controllers that run other engines."""
    manifest = chat_stream_manifest(orchestrator_id)
    if manifest is None:
        raise KeyError(f"chat stream orchestrator not available: {orchestrator_id!r}")
    return import_entry(stream_entry(manifest, full=full))


def load_stream_agent(
    orchestrator_id: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> Callable[..., Any]:
    oid = resolve_chat_orchestrator_for_request(orchestrator_id, context)
    return stream_agent_for_id(oid)


def load_stream_agent_full(
    orchestrator_id: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> Callable[..., Any]:
    oid = resolve_chat_orchestrator_for_request(orchestrator_id, context)
    return stream_agent_for_id(oid, full=True)


async def release_chat_session(session_id: str, *, user_id: str = "", reason: str = "") -> Dict[str, Any]:
    """Drop live in-memory state every chat engine holds for ``session_id``.

    Engines opt in with ``extensions.release_entry`` (``module:async_function``,
    called with ``session_id``, ``user_id``, ``reason``). All engines are asked,
    because controllers may have run another engine for this chat.
    """
    reg = get_registry()
    out: Dict[str, Any] = {}
    for r in reg.list_orchestrators(ready_only=True):
        entry = str(_extensions(r.manifest).get("release_entry") or "").strip()
        if not entry:
            continue
        try:
            out[r.id] = await import_entry(entry)(session_id=session_id, user_id=user_id, reason=reason)
        except Exception as exc:  # noqa: BLE001 — one engine must not block the others
            out[r.id] = {"ok": False, "error": str(exc)[:200]}
    return out
