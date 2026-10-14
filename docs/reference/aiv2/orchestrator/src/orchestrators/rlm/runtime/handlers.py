"""Host-side RPC handlers for RLM kernel host_request calls."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any, Dict, List, Set, TYPE_CHECKING

from src.orchestrators.rlm.runtime.constants import HOST_REQUEST_TYPES, safe_ident
from src.orchestrators.rlm.runtime.prompts import format_child_task_prompt

if TYPE_CHECKING:
    from src.orchestrators.rlm.runtime.session import RlmSession

logger = logging.getLogger(__name__)


async def dispatch(session: "RlmSession", request_type: str, payload: Dict[str, Any]) -> Any:
    rtype = str(request_type or "").strip()
    if rtype not in HOST_REQUEST_TYPES:
        raise RuntimeError(f"unknown_host_request:{rtype}")
    handler = _HANDLERS.get(rtype)
    if not handler:
        raise RuntimeError(f"unhandled_host_request:{rtype}")
    return await handler(session, payload or {})


async def _rlm_run(session: "RlmSession", payload: Dict[str, Any]) -> Dict[str, Any]:
    if session.depth >= session.max_depth:
        raise RuntimeError(
            f"max_depth_exceeded: depth={session.depth} max={session.max_depth}"
        )
    if not session.spawn_child:
        raise RuntimeError("spawn_child_unavailable")
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise RuntimeError("missing_prompt")
    name = str(payload.get("name") or "").strip()
    if not name:
        name = f"child_{uuid.uuid4().hex[:8]}"
    name = safe_ident(name, fallback="child")[:64]
    existing = {str(c.get("name") or "") for c in session.children.values()}
    if name in existing:
        raise RuntimeError(f"duplicate_child_name:{name}")
    model = payload.get("model")
    child_model = str(model).strip() if model else session.model
    if model and not child_model:
        raise RuntimeError("invalid_model")
    found: List[Dict[str, Any]] = []
    if model and session.model_finder:
        found = await session.model_finder(child_model, 20)
        ids = {
            str(item.get("id") or item.get("label") or "").strip()
            for item in (found or [])
            if isinstance(item, dict)
        }
        if child_model not in ids:
            raise RuntimeError(f"unknown_model:{child_model}")

    # Track fan-out for this run (optional ceilings; default unlimited).
    # Real capacity gating is KernelGovernor at kernel start.
    budget = session._resource_budget
    await budget.reserve_child()
    child_id = f"sub_{uuid.uuid4().hex[:10]}"
    child_dir = os.path.join(session.artifact_dir, "children", child_id)
    try:
        await asyncio.to_thread(os.makedirs, child_dir, exist_ok=True)
    except BaseException:
        await budget.release_child()
        raise
    # Children share the parent's IPython workspace so relative paths the parent
    # prepared (``work/batches/4.json``) resolve. Session-private state still
    # lives under ``children/<id>/`` (transcript, kernel snapshot, harness).
    parent_workspace = str(
        getattr(session, "workspace_dir", None) or session.artifact_dir or ""
    ).strip() or session.artifact_dir
    handle = {
        "rlm_child_id": child_id,
        "name": name,
        "session_dir": child_dir,
        "workspace_dir": parent_workspace,
        "model": child_model,
    }

    child_cfg = dict(session.cfg)
    child_cfg["model"] = child_model
    child_cfg["tool_ids"] = list(session.subagent_tool_ids)
    child_cfg["subagent_tool_ids"] = list(session.subagent_tool_ids)
    if child_model == session.model and session.context_length:
        child_cfg["context_length"] = session.context_length
        if session.max_output_tokens:
            child_cfg["max_output_tokens"] = session.max_output_tokens
    elif model and session.model_finder:
        for item in found or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("id") or "").strip() == child_model:
                if item.get("context_length"):
                    child_cfg["context_length"] = item.get("context_length")
                if item.get("max_output_tokens"):
                    child_cfg["max_output_tokens"] = item.get("max_output_tokens")
                break
    child_cfg["system_prompt"] = str(session.cfg.get("system_prompt") or "")
    # Doctrine lives in build_system_prompt(depth>0); do not stuff it into base.
    child_cfg["_rlm_child_name"] = name
    parent_label = str(
        session.cfg.get("_rlm_child_name")
        or getattr(session, "node_id", None)
        or "parent"
    ).strip() or "parent"
    child_cfg["_rlm_parent_name"] = parent_label

    labeled_prompt = format_child_task_prompt(prompt)

    try:
        child = await session.spawn_child(
            parent=session,
            prompt=labeled_prompt,
            cfg=child_cfg,
            name=name,
            child_id=child_id,
            artifact_dir=child_dir,
            workspace_dir=parent_workspace,
            depth=session.depth + 1,
        )
    except BaseException:
        await budget.release_child()
        raise
    child._parent_name = parent_label  # type: ignore[attr-defined]

    async def _run_child() -> None:
        try:
            await child.run_turns(base_system=str(child_cfg.get("system_prompt") or ""))
        finally:
            async def _cleanup() -> None:
                try:
                    await asyncio.to_thread(child.persist_local)
                except Exception:
                    logger.warning(
                        "child local persistence failed child=%s", child_id, exc_info=True
                    )
                try:
                    await child.stop_kernel(dump_snapshot=True)
                finally:
                    child.compact_completed_child()
                    await budget.release_child()

            await asyncio.shield(_cleanup())

    task = asyncio.create_task(_run_child())
    session.register_child(handle, child, task)
    session._emit("child_admitted", **handle)
    return handle


async def _rlm_list(session: "RlmSession", payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"subagents": session.list_subagents()}


async def _rlm_delete(session: "RlmSession", payload: Dict[str, Any]) -> Dict[str, Any]:
    return await session.delete_subagent(str(payload.get("target") or ""))


async def _rlm_find_models(session: "RlmSession", payload: Dict[str, Any]) -> Dict[str, Any]:
    if not session.model_finder:
        return {"models": [{"id": session.model, "label": session.model}]}
    models = await session.model_finder(
        str(payload.get("query") or ""),
        int(payload.get("limit") or 8),
    )
    return {"models": models}


async def _call_pool_resolve(session: "RlmSession", locator: str, pool_key: str) -> Any:
    """Call ``pool_resolve`` pool-scoped, tolerating older unscoped callables."""
    try:
        return await session.pool_resolve(locator, pool_key=pool_key)
    except TypeError as exc:
        if "pool_key" not in str(exc):
            raise
        return await session.pool_resolve(locator)


def _inside_session_dir(path: str, root: str) -> bool:
    if not root:
        return False
    try:
        real = os.path.realpath(path)
        real_root = os.path.realpath(root)
    except OSError:
        return False
    return real == real_root or real.startswith(real_root + os.sep)


async def _resolve_look_path(
    session: "RlmSession",
    locator: str,
    *,
    pool: str = "",
) -> str:
    loc = str(locator or "").strip()
    if not loc:
        raise RuntimeError("look_missing_path")
    root = str(
        getattr(session, "workspace_dir", None)
        or getattr(session, "artifact_dir", "")
        or ""
    )
    # A locator is kernel-controlled input, so a direct path hit is only honored
    # inside this session's dir. Anything else falls through to pool resolution
    # by basename instead of reading an arbitrary host file.
    direct = loc if os.path.isabs(loc) else os.path.join(root, loc)
    if _inside_session_dir(direct, root) and os.path.isfile(direct):
        return os.path.realpath(direct)
    base = os.path.basename(loc.rstrip("/"))
    pool_join = os.path.join(root, "pool", "files", base)
    if _inside_session_dir(pool_join, root) and os.path.isfile(pool_join):
        return pool_join

    want_pool = str(pool or "").strip()
    pool_keys: List[str] = []
    if want_pool:
        for entry in session.pool_defs or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            key = str(entry.get("pool_key") or "").strip()
            if want_pool in (name, key):
                pool_keys.append(key or name)
                break
        if not pool_keys:
            raise RuntimeError(f"look_unknown_pool:{want_pool}")
    else:
        for entry in session.pool_defs or []:
            if not isinstance(entry, dict):
                continue
            key = str(entry.get("pool_key") or entry.get("name") or "").strip()
            if key:
                pool_keys.append(key)
        if not pool_keys:
            pool_keys = [""]

    hits: List[tuple[str, str]] = []
    seen: Set[str] = set()

    def _remember(key: str, resolved: Any) -> None:
        if not isinstance(resolved, dict) or resolved.get("ok") is False:
            return
        path = str(resolved.get("path") or "").strip()
        if not path or not os.path.isfile(path):
            return
        # The union fallback can reach one artifact through several pool keys —
        # that is the same file, not a collision. Only distinct artifacts clash.
        aid = str(resolved.get("artifact_id") or "").strip()
        ident = aid or os.path.realpath(path)
        if ident in seen:
            return
        seen.add(ident)
        hits.append((key or "_", path))

    if session.pool_resolve:
        for key in pool_keys:
            try:
                resolved = await _call_pool_resolve(session, loc, key)
            except Exception:
                continue
            _remember(key, resolved)
    if not hits and session.pool_read:
        for key in pool_keys:
            try:
                got = await session.pool_read(key, loc)
            except Exception:
                continue
            _remember(key, got)
    if len(hits) == 1:
        return hits[0][1]
    if len(hits) > 1:
        names = ",".join(sorted({h[0] for h in hits}))
        raise RuntimeError(f"look_ambiguous:{loc}:{names}")
    raise RuntimeError(f"look_file_not_found:{loc}")


async def _rlm_look(session: "RlmSession", payload: Dict[str, Any]) -> Dict[str, Any]:
    from src.orchestrators.rlm.runtime.native_media import (
        coerce_locators,
        look_file,
        look_turn_limit_hint,
        native_look_kinds,
        normalize_input_modalities,
        partition_look_call,
    )

    locators = coerce_locators(payload)
    if not locators:
        raise RuntimeError("look_missing_path")
    pool = str(payload.get("pool") or "").strip()
    modalities = normalize_input_modalities(session.cfg.get("input_modalities"))
    native = native_look_kinds(modalities)
    candidates: List[Dict[str, Any]] = []
    for loc in locators:
        path = await _resolve_look_path(session, loc, pool=pool)
        rec = look_file(path, modalities=modalities)
        candidates.append(
            {
                "filename": str(rec.get("filename") or os.path.basename(path)),
                "kind": str(rec.get("kind") or ""),
                "bytes": rec.get("bytes"),
                "encoded_bytes": rec.get("encoded_bytes"),
                "path": rec.get("path"),
                "part": rec["part"],
            }
        )
    accept, skipped = partition_look_call(
        candidates,
        already_pending=getattr(session, "_pending_looks", []) or [],
    )
    queued: List[Dict[str, Any]] = []
    for item in accept:
        session.queue_look(
            filename=str(item.get("filename") or "file"),
            kind=str(item.get("kind") or ""),
            part=item["part"],
        )
        queued.append(
            {
                "filename": item.get("filename"),
                "kind": item.get("kind"),
                "bytes": item.get("bytes"),
                "path": item.get("path"),
            }
        )
    skipped_out = [
        {
            "filename": s.get("filename"),
            "kind": s.get("kind"),
            "reason": s.get("skip_reason"),
        }
        for s in skipped
    ]
    hint = (
        "queued file(s) appear on the next chat turn, not in this ipython result; "
        + look_turn_limit_hint()
    )
    if skipped_out:
        names = ", ".join(str(s.get("filename") or "?") for s in skipped_out)
        hint += (
            f". Skipped this call (call look again later, ≤2 images / ≤1 video): {names}"
        )
    return {
        "ok": True,
        "queued": queued,
        "skipped": skipped_out,
        "native_modalities": sorted(native),
        "hint": hint,
    }


def _root_session(session: "RlmSession") -> "RlmSession":
    parent = getattr(session, "_parent_ref", None)
    return parent if parent is not None else session


def _child_name_for(session: "RlmSession") -> str:
    parent = getattr(session, "_parent_ref", None)
    if parent is not None:
        for rec in parent.children.values():
            if rec.get("rlm_child_id") == session.session_id or rec.get(
                "session_dir"
            ) == session.artifact_dir:
                return str(rec.get("name") or session.session_id)
    return session.session_id


def _nudge_if_idle(target: "RlmSession", text: str) -> None:
    if getattr(target, "_flow_released", False):
        return
    if target.status in ("completed", "idle") and str(text or "").strip():
        asyncio.create_task(target.continue_with_message(text))


async def _agent_message_send(session: "RlmSession", payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text") or "")
    role = str(payload.get("receiver_role") or "parent").strip().lower()
    receiver_name = str(payload.get("receiver_name") or "").strip()
    if role == "parent":
        if not session.parent_session_id:
            session.enqueue_message(
                text=text, from_role="self", from_name=session.session_id, to_role="self"
            )
            return {"ok": True, "routed": "self"}
        parent = getattr(session, "_parent_ref", None)
        if parent is None:
            raise RuntimeError("parent_unavailable")
        parent.enqueue_message(
            text=text,
            from_role="child",
            from_name=_child_name_for(session),
            to_role="parent",
        )
        # Prime: count parent-directed replies for completed_without_reply.
        if int(getattr(session, "depth", 0) or 0) > 0:
            session._parent_reply_count = int(getattr(session, "_parent_reply_count", 0) or 0) + 1
            session._replied_to_parent_since_task = True
        return {"ok": True, "routed": "parent"}
    if role == "child":
        if not receiver_name:
            raise RuntimeError("receiver_name_required")
        target = None
        for cid, rec in session.children.items():
            if str(rec.get("name") or "") == receiver_name or cid == receiver_name:
                target = session._child_sessions.get(cid)
                break
        if target is None:
            raise RuntimeError(f"child_not_found:{receiver_name}")
        target.enqueue_message(
            text=text,
            from_role="parent",
            from_name="parent",
            to_role="child",
        )
        # Prime: a new parent task resets "replied since task" (count stays).
        target._replied_to_parent_since_task = False
        _nudge_if_idle(target, text)
        return {"ok": True, "routed": "child", "receiver_name": receiver_name}
    if role in ("sibling", "peer"):
        if not receiver_name:
            raise RuntimeError("receiver_name_required")
        siblings = session.sibling_map()
        target = siblings.get(receiver_name)
        if target is None or target.session_id == session.session_id:
            raise RuntimeError(f"sibling_not_found:{receiver_name}")
        target.enqueue_message(
            text=text,
            from_role="sibling",
            from_name=_child_name_for(session),
            to_role="sibling",
        )
        _nudge_if_idle(target, text)
        return {"ok": True, "routed": "sibling", "receiver_name": receiver_name}
    raise RuntimeError(f"invalid_receiver_role:{role}")


async def _agent_message_inbox(session: "RlmSession", payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"messages": session.drain_python_inbox()}


def _agent_listing(session: "RlmSession") -> List[Dict[str, Any]]:
    root = _root_session(session)
    agents = [
        {
            "role": "parent" if root.parent_session_id is None else "child",
            "name": "parent" if root.parent_session_id is None else _child_name_for(root),
            "session_id": root.session_id,
            "status": root.status,
        }
    ]
    for rec in root.list_subagents():
        agents.append(
            {
                "role": "child",
                "name": rec.get("name"),
                "session_id": rec.get("rlm_child_id"),
                "status": rec.get("status"),
            }
        )
    return agents


async def _agent_message_list_agents(session: "RlmSession", payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"agents": _agent_listing(session)}


def _resolve_observe_target(session: "RlmSession", target: str) -> "RlmSession":
    sel = str(target or "").strip()
    if not sel or sel in ("self", session.session_id, _child_name_for(session)):
        return session
    root = _root_session(session)
    if sel in ("parent", "root") or sel == root.session_id:
        return root
    siblings = session.sibling_map()
    found = siblings.get(sel)
    if found is None:
        raise RuntimeError(f"observe_target_not_found:{sel}")
    return found


async def _agent_observe_list(session: "RlmSession", payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"agents": _agent_listing(session)}


async def _agent_observe_get(session: "RlmSession", payload: Dict[str, Any]) -> Dict[str, Any]:
    target = _resolve_observe_target(session, str(payload.get("target") or ""))
    return target.public_status()


async def _agent_observe_recent(session: "RlmSession", payload: Dict[str, Any]) -> Dict[str, Any]:
    target = _resolve_observe_target(session, str(payload.get("target") or ""))
    limit = max(1, min(100, int(payload.get("limit") or 20)))
    return {"events": list(target.events[-limit:])}


async def _pools_search(session: "RlmSession", payload: Dict[str, Any]) -> Any:
    if not session.pool_search:
        raise RuntimeError("pool_search_unavailable")
    pool_key = str(payload.get("pool_key") or "").strip()
    query = str(payload.get("query") or "").strip()
    if not query and getattr(session, "pool_list", None):
        return await session.pool_list(pool_key)
    return await session.pool_search(pool_key, query)


async def _pools_resolve_file(session: "RlmSession", payload: Dict[str, Any]) -> Any:
    if not session.pool_resolve:
        raise RuntimeError("pool_resolve_unavailable")
    locator = str(payload.get("locator") or "")
    pool_key = str(payload.get("pool_key") or "").strip()
    return await _call_pool_resolve(session, locator, pool_key)


async def _pools_list(session: "RlmSession", payload: Dict[str, Any]) -> Any:
    if not getattr(session, "pool_list", None):
        raise RuntimeError("pool_list_unavailable")
    return await session.pool_list(str(payload.get("pool_key") or "").strip())


async def _tools_call(session: "RlmSession", payload: Dict[str, Any]) -> Any:
    args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
    return await session.call_project_tool(str(payload.get("tool_id") or ""), args)


async def _tools_describe(session: "RlmSession", payload: Dict[str, Any]) -> Any:
    tool_id = str(payload.get("tool_id") or "").strip()
    if tool_id not in set(session.tool_ids):
        raise PermissionError(f"tool_not_allowed:{tool_id}")
    for spec in session.tool_specs:
        fn = spec.get("function") if isinstance(spec, dict) else None
        if isinstance(fn, dict) and str(fn.get("name") or "") == tool_id:
            return {
                "name": tool_id,
                "description": str(fn.get("description") or ""),
                "parameters": fn.get("parameters") or {},
            }
    return {"name": tool_id, "description": "", "parameters": {}}


async def _harness_get(session: "RlmSession", payload: Dict[str, Any]) -> Any:
    return session.harness.get()


async def _harness_list(session: "RlmSession", payload: Dict[str, Any]) -> Any:
    return {"entries": session.harness.list_entries(payload.get("kind"))}


async def _harness_upsert(session: "RlmSession", payload: Dict[str, Any]) -> Any:
    return session.harness.upsert(
        kind=str(payload.get("kind") or ""),
        content=str(payload.get("content") or ""),
        entry_id=payload.get("entry_id"),
        title=str(payload.get("title") or ""),
        meta=payload.get("meta") if isinstance(payload.get("meta"), dict) else None,
    )


async def _harness_delete(session: "RlmSession", payload: Dict[str, Any]) -> Any:
    ok = session.harness.delete(
        kind=str(payload.get("kind") or ""),
        entry_id=str(payload.get("entry_id") or ""),
    )
    return {"ok": ok}


async def _goal_get(session: "RlmSession", payload: Dict[str, Any]) -> Any:
    return {"goal": session.jobs.goal_get()}


async def _goal_create(session: "RlmSession", payload: Dict[str, Any]) -> Any:
    return session.jobs.goal_create(
        str(payload.get("objective") or ""),
        token_budget=int(payload.get("token_budget") or 0),
    )


async def _goal_complete(session: "RlmSession", payload: Dict[str, Any]) -> Any:
    return session.jobs.goal_complete()


async def _hb_list(session: "RlmSession", payload: Dict[str, Any]) -> Any:
    return {"heartbeats": session.jobs.heartbeat_list()}


async def _hb_create(session: "RlmSession", payload: Dict[str, Any]) -> Any:
    return session.jobs.heartbeat_create(
        interval_sec=int(payload.get("interval_sec") or 300),
        prompt=str(payload.get("prompt") or ""),
        name=str(payload.get("name") or ""),
    )


async def _hb_delete(session: "RlmSession", payload: Dict[str, Any]) -> Any:
    ok = session.jobs.heartbeat_delete(str(payload.get("target") or ""))
    return {"ok": ok}


async def _compact_status(session: "RlmSession", payload: Dict[str, Any]) -> Any:
    return {"messages": len(session.messages), "kernel_intact": True}


async def _compact_run(session: "RlmSession", payload: Dict[str, Any]) -> Any:
    custom = str(payload.get("instructions") or payload.get("custom_instructions") or "")
    try:
        result = await session.maybe_compact(force=True, custom_instructions=custom)
    except Exception:
        result = session.compact_messages(keep_last=int(payload.get("keep_last") or 12))
    if session.refine_enabled:
        try:
            from src.orchestrators.rlm.runtime.auto_refine import maybe_auto_refine

            refine_out = await maybe_auto_refine(session, reason="compact")
            result["refine"] = refine_out
        except Exception as exc:
            result["refine"] = f"skipped:{exc}"
    return result


async def _refine_status(session: "RlmSession", payload: Dict[str, Any]) -> Any:
    snaps = list(session.harness.state.get("snapshots") or [])
    return {
        "enabled": session.refine_enabled,
        "auto_enabled": session.refine_auto_enabled,
        "turn_interval": session.refine_turn_interval,
        "cooldown_sec": session.refine_cooldown_sec,
        "on_compact": session.refine_on_compact,
        "harness_path": session.harness_path(),
        "node_id": session.node_id,
        "model_logical_id": session.model_logical_id or session.model,
        "turns_since_refine": session._assistant_turns_since_refine,
        "last_review_at": session._last_refine_review_at,
        "compact_pending": session._compact_refine_pending,
        "history": session.merged_refinement_history(limit=10),
        "snapshots": [
            {"id": s.get("id"), "label": s.get("label"), "ts": s.get("ts")}
            for s in snaps
            if isinstance(s, dict)
        ][-10:],
    }


async def _refine_run(session: "RlmSession", payload: Dict[str, Any]) -> Any:
    from src.orchestrators.rlm.runtime.refine import refine_session

    return await refine_session(
        session,
        chat_fn=session.chat_fn,
        extract_assistant_fn=session.extract_assistant_fn,
        model=payload.get("model"),
        instructions=payload.get("instructions"),
        skip_review=True,
    )


_HANDLERS = {
    "rlm.run": _rlm_run,
    "rlm.list_subagents": _rlm_list,
    "rlm.delete_subagent": _rlm_delete,
    "rlm.find_models": _rlm_find_models,
    "rlm.look": _rlm_look,
    "agent_message.send": _agent_message_send,
    "agent_message.inbox": _agent_message_inbox,
    "agent_message.list_agents": _agent_message_list_agents,
    "agent_observe.list": _agent_observe_list,
    "agent_observe.get": _agent_observe_get,
    "agent_observe.recent": _agent_observe_recent,
    "pools.search": _pools_search,
    "pools.resolve_file": _pools_resolve_file,
    "pools.list": _pools_list,
    "tools.call": _tools_call,
    "tools.describe": _tools_describe,
    "harness.get": _harness_get,
    "harness.list": _harness_list,
    "harness.upsert": _harness_upsert,
    "harness.delete": _harness_delete,
    "goal.get": _goal_get,
    "goal.create": _goal_create,
    "goal.complete": _goal_complete,
    "rlm_heartbeat.list": _hb_list,
    "rlm_heartbeat.create": _hb_create,
    "rlm_heartbeat.delete": _hb_delete,
    "compact.status": _compact_status,
    "compact.run": _compact_run,
    "refine.status": _refine_status,
    "refine.run": _refine_run,
}
