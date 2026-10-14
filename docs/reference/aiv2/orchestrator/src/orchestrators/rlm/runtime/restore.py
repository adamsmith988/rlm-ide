"""Cold-restore an RLM session from its durable run-workspace artifact dir.

Live ``RlmDaemon`` state is process-local. After tool-runner restart / idle
eviction the FE still holds ``session_id`` (+ usually ``run_id`` /
``artifact_dir``). Attach / continue / retry must resurrect the session from
disk before running more turns.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_CFG_NAME = "cfg.json"


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def write_session_cfg(artifact_dir: str, cfg: Dict[str, Any], *, gen_params: Optional[Dict[str, Any]] = None) -> None:
    """Persist enough agent config to cold-restore the session later."""
    root = str(artifact_dir or "").strip()
    if not root:
        return
    try:
        os.makedirs(root, exist_ok=True)
        payload = {
            "cfg": dict(cfg or {}),
            "gen_params": dict(gen_params or {}),
        }
        path = os.path.join(root, _CFG_NAME)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        logger.debug("write_session_cfg failed dir=%s", root, exc_info=True)


def load_session_cfg(artifact_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data = _read_json(os.path.join(str(artifact_dir or ""), _CFG_NAME)) or {}
    cfg = data.get("cfg") if isinstance(data.get("cfg"), dict) else {}
    gen = data.get("gen_params") if isinstance(data.get("gen_params"), dict) else {}
    return dict(cfg), dict(gen)


def resolve_artifact_dir(
    session_id: str,
    *,
    user_id: str = "",
    artifact_dir: str = "",
    run_id: str = "",
    flow_id: str = "",
    node_id: str = "",
) -> Optional[str]:
    """Locate the on-disk session directory for ``session_id``."""
    sid = str(session_id or "").strip()
    if not sid:
        return None

    hinted = str(artifact_dir or "").strip()
    if hinted and os.path.isfile(os.path.join(hinted, "meta.json")):
        meta = _read_json(os.path.join(hinted, "meta.json")) or {}
        rec = str(meta.get("session_id") or os.path.basename(hinted) or "").strip()
        if rec in ("", sid):
            owner = str(meta.get("user_id") or "")
            if not user_id or not owner or owner == user_id:
                return os.path.realpath(hinted)

    from src.orchestrators.rlm.runtime.daemon import RlmDaemon
    from src.orchestrators.rlm.runtime.persist import (
        load_persisted_public,
        session_dir_for,
    )

    daemon = RlmDaemon.instance()
    found = session_dir_for(daemon.root_dir, sid, user_id=user_id)
    if found:
        return found
    rec = load_persisted_public(daemon.root_dir, sid, user_id=user_id)
    if rec and rec.get("artifact_dir"):
        return str(rec["artifact_dir"])

    # Prefer the known run workspace (cheap) over a full PVC walk.
    rid = str(run_id or "").strip()
    fid = str(flow_id or "").strip()
    uid = str(user_id or "").strip()
    if uid and fid and rid:
        try:
            try:
                from modules.plugins.agent_flow.session_disk_hydrate import resolve_run_workspace
            except ImportError as e:
                raise RuntimeError(
                    "RLM cold restore requires host adapter (session_disk_hydrate); "
                    "not available in orchestrator service copy"
                ) from e

            ws = resolve_run_workspace(user_id=uid, flow_id=fid, run_id=rid)
        except Exception:
            ws = ""
        if ws:
            rlm_root = os.path.join(ws, "rlm")
            found = session_dir_for(rlm_root, sid, user_id=user_id)
            if found:
                return found
            # Node folder naming is usually rlm/<node_id>/ even when session_id differs.
            nid = str(node_id or "").strip()
            if nid:
                cand = os.path.join(rlm_root, nid)
                meta = _read_json(os.path.join(cand, "meta.json")) or {}
                if str(meta.get("session_id") or "") == sid:
                    return cand
            # Scan only this run's rlm tree (shallow).
            try:
                for name in os.listdir(rlm_root):
                    cand = os.path.join(rlm_root, name)
                    meta = _read_json(os.path.join(cand, "meta.json")) or {}
                    if str(meta.get("session_id") or "") == sid:
                        return cand
                    children = os.path.join(cand, "children")
                    if not os.path.isdir(children):
                        continue
                    for cname in os.listdir(children):
                        cpath = os.path.join(children, cname)
                        cmeta = _read_json(os.path.join(cpath, "meta.json")) or {}
                        if str(cmeta.get("session_id") or "") == sid:
                            return cpath
            except OSError:
                pass
    return None


async def _cfg_from_flow_node(
    *,
    flow_id: str,
    node_id: str,
    user_id: str,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    fid = str(flow_id or "").strip()
    nid = str(node_id or "").strip()
    if not fid or not nid:
        return {}
    try:
        try:
            from modules.plugins.agent_flow._shared import resolve_flow_access
        except ImportError as e:
            raise RuntimeError(
                "RLM cold restore requires host adapter (resolve_flow_access); "
                "not available in orchestrator service copy"
            ) from e

        flow, _owner, access_err = await resolve_flow_access(fid, user_id, **kwargs)
        if access_err or not isinstance(flow, dict):
            return {}
        definition = flow.get("definition") if isinstance(flow.get("definition"), dict) else {}
        nodes = definition.get("nodes") if isinstance(definition.get("nodes"), list) else []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            if str(node.get("id") or "") != nid:
                continue
            data = node.get("data") if isinstance(node.get("data"), dict) else {}
            cfg = data.get("config")
            return dict(cfg) if isinstance(cfg, dict) else {}
    except Exception:
        logger.debug("cfg_from_flow_node failed", exc_info=True)
    return {}


async def restore_session_from_disk(
    session_id: str,
    user_id: str,
    *,
    kwargs: Optional[Dict[str, Any]] = None,
    artifact_dir: str = "",
    run_id: str = "",
    flow_id: str = "",
    node_id: str = "",
) -> Optional[Any]:
    """Rebuild an ``RlmSession`` from disk and register it with the daemon."""
    from src.orchestrators.rlm.runtime.daemon import RlmDaemon
    from src.orchestrators.rlm.runtime.session import RlmSession
    try:
        from modules.plugins.agent_flow.runner import FlowRunner
    except ImportError as e:
        raise RuntimeError(
            "RLM cold restore requires host adapter (FlowRunner); "
            "provide a host restore hook instead of using agent_flow restore in-process"
        ) from e

    sid = str(session_id or "").strip()
    uid = str(user_id or "").strip()
    if not sid or not uid:
        return None

    daemon = RlmDaemon.instance()
    live = daemon.get(sid)
    if live is not None:
        return live

    art = resolve_artifact_dir(
        sid,
        user_id=uid,
        artifact_dir=artifact_dir,
        run_id=run_id,
        flow_id=flow_id,
        node_id=node_id,
    )
    if not art:
        return None

    meta = _read_json(os.path.join(art, "meta.json")) or {}
    owner = str(meta.get("user_id") or "").strip()
    if owner and owner != uid:
        return None

    cfg, gen_params = load_session_cfg(art)
    meta_flow = str(meta.get("flow_id") or flow_id or "").strip()
    meta_node = str(meta.get("node_id") or node_id or "").strip()
    meta_run = str(meta.get("run_id") or run_id or "").strip()
    if not cfg:
        cfg = await _cfg_from_flow_node(
            flow_id=meta_flow,
            node_id=meta_node,
            user_id=uid,
            kwargs=dict(kwargs or {}),
        )
    if not cfg:
        cfg = {}
    model = str(cfg.get("model") or meta.get("model") or meta.get("model_logical_id") or "").strip()
    if model:
        cfg.setdefault("model", model)
    if meta.get("context_length") and not cfg.get("context_length"):
        cfg["context_length"] = meta.get("context_length")
    if meta.get("max_output_tokens") and not cfg.get("max_output_tokens"):
        cfg["max_output_tokens"] = meta.get("max_output_tokens")
    cfg.setdefault("daemon_enabled", True)
    cfg.setdefault("execution_mode", "rlm")

    helper = FlowRunner(
        flow={"id": meta_flow or "rlm"},
        definition={"nodes": [], "edges": []},
        run_input={},
        kwargs={**(kwargs or {}), "user_id": uid},
        run_id=meta_run or None,
    )

    async def tool_runner(tool_id: str, args: Dict[str, Any]) -> Any:
        return await helper._run_tool({"tool_id": tool_id, "args": dict(args or {})})

    async def pool_list(_pool_key: str) -> Dict[str, Any]:
        return {"ok": True, "files": []}

    async def pool_read(
        _pool_key: str, locator: str, *, max_chars: int = 20000
    ) -> Dict[str, Any]:
        path = os.path.join(art, str(locator or "").lstrip("/"))
        if not os.path.isfile(path):
            # Common relative paths under the workspace.
            for rel in (
                str(locator or ""),
                os.path.join("pool", "files", os.path.basename(str(locator or ""))),
                os.path.join("work", os.path.basename(str(locator or ""))),
            ):
                cand = os.path.join(art, rel)
                if os.path.isfile(cand):
                    path = cand
                    break
        if not os.path.isfile(path):
            return {"ok": False, "error": "not_found", "detail": str(locator or "")}
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read(max(1, int(max_chars)))
        except Exception as exc:
            return {"ok": False, "error": "read_failed", "detail": str(exc)}
        return {
            "ok": True,
            "path": path,
            "filename": os.path.basename(path),
            "preview": text,
        }

    async def pool_resolve(locator: str, *, pool_key: str = "") -> Dict[str, Any]:
        del pool_key
        got = await pool_read("", locator)
        if not got.get("ok"):
            raise RuntimeError(str(got.get("error") or "not_found"))
        return {
            "path": got.get("path"),
            "filename": got.get("filename"),
            "artifact_id": None,
        }

    async def pool_search(_pool_key: str, _query: str) -> Dict[str, Any]:
        return {"ok": True, "results": []}

    async def model_finder(query: str, limit: int = 8) -> List[Dict[str, Any]]:
        try:
            try:
                from modules.plugins.agent_flow.catalog_models.tool import run as catalog_models_run
            except ImportError as e:
                raise RuntimeError(
                    "RLM cold restore requires host adapter (catalog_models); "
                    "not available in orchestrator service copy"
                ) from e

            result = await catalog_models_run(search=query or None)
            payload = result.get("data") if isinstance(result, dict) else result
            items = payload.get("items") if isinstance(payload, dict) else []
            out: List[Dict[str, Any]] = []
            for it in items or []:
                if not isinstance(it, dict):
                    continue
                mid = str(it.get("id") or "").strip()
                if not mid:
                    continue
                out.append(
                    {
                        "id": mid,
                        "label": str(it.get("display_name") or mid),
                        "task": it.get("task"),
                        "context_length": it.get("context_length"),
                        "max_output_tokens": it.get("max_output_tokens"),
                    }
                )
                if len(out) >= max(1, int(limit or 8)):
                    break
            return out
        except Exception:
            return []

    async def spawn_child(**_kw: Any) -> RlmSession:
        raise RuntimeError(
            "spawn_child_unavailable_after_restore: retry from a live agent run "
            "if new subagents are required"
        )

    hooks: Dict[str, Any] = {}

    def on_event(ev: Dict[str, Any]) -> None:
        """Push live RLM snapshots onto the flow progress bus for FE polls/SSE."""
        s = hooks.get("sess")
        rid = str(meta_run or getattr(s, "run_id", "") or "").strip()
        if not rid or not isinstance(ev, dict):
            return
        try:
            try:
                from modules.plugins.agent_flow._shared import progress_set
            except ImportError:
                def progress_set(*_a, **_k):
                    return None

            nid = str(meta_node or getattr(s, "node_id", "") or "").strip()
            rlm_snap = ev.get("rlm") if isinstance(ev.get("rlm"), dict) else None
            if rlm_snap is None and s is not None and hasattr(s, "live_progress_snapshot"):
                try:
                    rlm_snap = s.live_progress_snapshot()
                except Exception:
                    rlm_snap = None
            payload: Dict[str, Any] = {
                "status": "running",
                "user_id": uid,
                "flow_id": meta_flow or None,
                "current_kind": "agent",
                "error": None,
            }
            if nid:
                payload["current_node"] = nid
                payload["current_nodes"] = [nid]
            if s is not None:
                payload["rlm_session_id"] = s.session_id
                payload["rlm_status"] = s.status
            if rlm_snap:
                payload["rlm"] = rlm_snap
            progress_set(rid, **payload)
        except Exception:
            logger.debug("restore on_event progress_set failed", exc_info=True)

    workspace = str(meta.get("workspace_dir") or art).strip() or art
    sess = RlmSession(
        cfg=cfg,
        user_content=None,
        depth=int(meta.get("depth") or 0),
        session_id=sid,
        parent_session_id=meta.get("parent_session_id"),
        run_id=meta_run or None,
        flow_id=meta_flow,
        user_id=uid,
        node_id=meta_node,
        artifact_dir=art,
        workspace_dir=workspace,
        chat_fn=helper._internal_chat,
        chat_stream_fn=helper._internal_chat_stream,
        extract_assistant_fn=helper._extract_assistant_message,
        gen_params=gen_params or helper._gen_params(cfg),
        tool_runner=tool_runner,
        pool_search=pool_search,
        pool_resolve=pool_resolve,
        pool_list=pool_list,
        pool_read=pool_read,
        pool_defs=[],
        spawn_child=spawn_child,
        model_finder=model_finder,
        on_event=on_event,
    )
    hooks["sess"] = sess

    # Carry over terminal fields so FE/restore know the prior failure until
    # continue/retry clears them.
    st = str(meta.get("status") or "").strip()
    if st:
        sess.status = st
    if meta.get("error"):
        sess.error = str(meta.get("error"))
    if meta.get("last_content"):
        sess.last_content = str(meta.get("last_content") or "")
    if meta.get("last_reasoning"):
        sess.last_reasoning = str(meta.get("last_reasoning") or "")
    logs = meta.get("log_lines")
    if isinstance(logs, list) and logs:
        try:
            sess.log_lines = [str(x) for x in logs[-200:]]
        except Exception:
            pass
    # Mark closed=False so attach can reopen even if meta said terminal.
    sess._closed = False

    base_system = str(cfg.get("system_prompt") or cfg.get("system") or "")
    hydrated = sess.hydrate_messages_from_disk(base_system=base_system)
    if not hydrated:
        # Fall back to compact transcript.jsonl if full transcript missing.
        try:
            tpath = os.path.join(art, "transcript.jsonl")
            rows: List[Dict[str, Any]] = []
            if os.path.isfile(tpath):
                with open(tpath, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(row, dict):
                            rows.append(row)
            if rows:
                system = sess.build_system_prompt(base_system)
                msgs: List[Dict[str, Any]] = []
                if system:
                    msgs.append({"role": "system", "content": system})
                for row in rows:
                    if isinstance(row, dict) and row.get("role"):
                        msgs.append(
                            {
                                "role": row.get("role"),
                                "content": str(row.get("content") or ""),
                            }
                        )
                if len(msgs) > (1 if system else 0):
                    sess.messages = msgs
                    sess._transcript_written = len(sess.messages)
                    sess._restored_from_disk = True
                    hydrated = True
        except Exception:
            pass

    if not hydrated and not sess.messages:
        logger.warning("restore_session empty transcript sid=%s dir=%s", sid, art)

    daemon.register(sess)
    logger.info(
        "rlm session restored from disk sid=%s status=%s dir=%s hydrated=%s",
        sid,
        sess.status,
        art,
        hydrated,
    )
    return sess
