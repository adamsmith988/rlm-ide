"""Hub-chat host adapter for orchestrator_id=rlm.

Consumer-only wiring: injects HostPorts (chat stream, events, pools from
session files) and maps engine events onto hub SSE shapes. Does not change
Agent Flow or default native_tools chat.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import os
import re
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from ..llm.gateway_llm_client import (
    _merge_vllm_extra_body,
    _normalize_messages_for_vllm,
    _params_with_max_tokens,
    get_gateway_llm_client,
)
from ..tools.executor import get_tool_executor
from ._shared.host_context import page_prompt_block, prompt_blocks, replace_host_context
from ._shared.manifest_config import manifest_defaults
from .contract import RunEnvelope
from .host import HostPorts
from .host_adapters.chat_session import (
    build_chat_session_pool,
    make_session_artifact_download,
)
from .host_adapters.rlm_chat_memory import record_served_turn, resume_or_seed_live_memory
from .host_adapters.deliverables import (
    attached_sse_events,
    sync_rlm_deliverables_to_chat,
)
from .host_adapters.hub_final_answer import resolve_hub_final_answer
from .host_adapters.hub_sse import HubEventBridge
from .host_adapters.llm_retry import call_with_llm_retry, notify_llm_reconnecting
from .host_adapters.session_binding import (
    engine_session_id_for,
    lookup_live_runtime_session,
    read_session_binding,
    rebind_host_ports,
    register_runtime_session,
)
from .host_adapters.tools_binding import (
    kernel_tool_specs,
    make_tools_host_port,
    read_tools_binding,
    resolve_tool_ids_for_run,
)
from .rlm.adapter import RlmOrchestratorEngine, RlmOrchestratorSession
from .rlm.runtime.chat_stream import (
    apply_chat_delta,
    completion_from_stream_state,
    maybe_await_delta,
)

logger = logging.getLogger(__name__)


def _vllm_result_to_openai(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert orchestrator GatewayLLMClient.chat() result → OpenAI chat.completion dict."""
    if not isinstance(result, dict):
        return {
            "choices": [{"message": {"role": "assistant", "content": str(result or "")}}],
        }
    if result.get("choices"):
        return result
    content = str(result.get("content") or "")
    msg: Dict[str, Any] = {"role": "assistant", "content": content}
    reasoning = result.get("reasoning_content") or result.get("reasoning")
    if reasoning:
        msg["reasoning_content"] = reasoning
    tool_calls = result.get("tool_calls") or []
    if tool_calls:
        oai_calls = []
        for i, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            args = tc.get("arguments")
            if isinstance(args, dict):
                args_s = json.dumps(args, ensure_ascii=False)
            else:
                args_s = str(args or "{}")
            oai_calls.append(
                {
                    "id": str(tc.get("id") or f"call_{i}"),
                    "type": "function",
                    "function": {
                        "name": str(tc.get("name") or ""),
                        "arguments": args_s,
                    },
                }
            )
        if oai_calls:
            msg["tool_calls"] = oai_calls
    usage = {}
    if result.get("tokens_prompt") is not None:
        usage["prompt_tokens"] = result["tokens_prompt"]
    if result.get("tokens_completion") is not None:
        usage["completion_tokens"] = result["tokens_completion"]
    out: Dict[str, Any] = {
        "choices": [
            {
                "index": 0,
                "message": msg,
                # Keep provider finish_reason as-is (incl. None). Defaulting to
                # ``stop`` disabled length-continuation when the proxy omitted it.
                "finish_reason": result.get("finish_reason"),
            }
        ],
    }
    if usage:
        out["usage"] = usage
    if not result.get("ok", True):
        out["error"] = result.get("error") or "llm_error"
    return out


def _delta_tool_calls_to_dicts(tool_calls: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not tool_calls:
        return out
    for tc in tool_calls:
        if isinstance(tc, dict):
            out.append(tc)
            continue
        try:
            idx = getattr(tc, "index", 0)
            fn = getattr(tc, "function", None)
            fn_dict = {
                "name": getattr(fn, "name", None) or "",
                "arguments": getattr(fn, "arguments", None) or "",
            }
            out.append(
                {
                    "index": idx,
                    "id": getattr(tc, "id", None) or "",
                    "type": getattr(tc, "type", None) or "function",
                    "function": fn_dict,
                }
            )
        except Exception:
            continue
    return out



# Alias kept for local readability
_HubEventBridge = HubEventBridge



def _durable_artifact_dir(session_id: str, user_id: str) -> str:
    """Stable per-chat workspace (never mkdtemp — that breaks resume/continue)."""
    try:
        from modules.tools.run_workspace import workspace_dir

        base = workspace_dir(
            "orchestrator_rlm",
            str(user_id or "user"),
            str(session_id or "session"),
            "chat",
        )
        path = os.path.join(base, "rlm")
        os.makedirs(path, exist_ok=True)
        return path
    except Exception:
        # Orchestrator image has no ``modules`` package — keep a stable path
        # under /tmp (or AIV2_RLM_CHAT_ROOT) keyed by user+chat session.
        root = (
            str(os.environ.get("AIV2_RLM_CHAT_ROOT") or "").strip()
            or "/tmp/aiv2_rlm_chat"
        )
        uid = re.sub(r"[^A-Za-z0-9._-]+", "_", str(user_id or "user"))[:80] or "user"
        sid = re.sub(r"[^A-Za-z0-9._-]+", "_", str(session_id or "session"))[:180] or "session"
        path = os.path.join(root, uid, sid, "rlm")
        os.makedirs(path, exist_ok=True)
        return path


async def release_rlm_chat(session_id: str, *, user_id: str = "", reason: str = "", **_: Any) -> Dict[str, Any]:
    """Shut down and evict this chat's live RLM sessions (state on disk stays; the next turn resumes)."""
    from .host_adapters.session_binding import hub_rlm_engine_session_id
    from .rlm.runtime.daemon import RlmDaemon

    daemon = RlmDaemon.instance()
    ids = {s.session_id for s in daemon.sessions_for_run(session_id)}
    engine_sid = hub_rlm_engine_session_id(session_id)
    if daemon.get(engine_sid) is not None:
        ids.add(engine_sid)
    released = []
    for sid in sorted(ids):
        res = await daemon.shutdown(sid, reason=reason or "chat_closed", pop=True)
        if res.get("ok"):
            released.append(sid)
    return {"ok": True, "released": released}


async def stream_agent_rlm(
    user_message: str,
    session_id: str,
    artifacts: Optional[List[Dict[str, Any]]] = None,
    user_id: str = "default",
    max_steps: Optional[int] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    has_fresh_artifacts: bool = False,
    conversation_state: Any = None,
    history_summary: str = "",
    file_registry: Optional[List[Dict[str, Any]]] = None,
    pinned_facts: Optional[Dict[str, Any]] = None,
    workspace_runtime: Optional[Dict[str, Any]] = None,
    discourse_focus: Optional[Dict[str, Any]] = None,
    task_checklist_state: Optional[Dict[str, Any]] = None,
    rule_policies: Optional[Dict[str, Any]] = None,
    active_plugin: Optional[str] = None,
    host_page_prompt: Optional[str] = None,
    rag_context: Optional[List[Dict[str, Any]]] = None,
    rag_prompt_block: str = "",
    orchestrator_config: Optional[Dict[str, Any]] = None,
    request_context: Optional[Dict[str, Any]] = None,
    case_inventory: Optional[Dict[str, Any]] = None,
    **_kwargs: Any,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Run one RLM session for a hub chat turn with live SSE-shaped events."""
    del (
        has_fresh_artifacts,
        conversation_state,
        history_summary,
        pinned_facts,
        workspace_runtime,
        task_checklist_state,
        rule_policies,
        rag_context,
    )
    host_block_kwargs = {
        "rag_prompt_block": rag_prompt_block,
        "page_prompt_block": page_prompt_block(active_plugin, host_page_prompt),
        **_kwargs,
    }
    # ``chat_history`` is kept — AF-parity fallback when daemon/disk miss.
    # Prefer explicit kwargs; fall back to leftovers from workspace.to_engine_kwargs.
    if request_context is None and isinstance(_kwargs.get("request_context"), dict):
        request_context = _kwargs.get("request_context")
    if case_inventory is None and isinstance(_kwargs.get("case_inventory"), dict):
        case_inventory = _kwargs.get("case_inventory")
    trace_id = str(uuid.uuid4())
    yield {
        "type": "start",
        "trace_id": trace_id,
        "session_id": session_id,
        "engine": "rlm",
        "timestamp": _dt.datetime.utcnow().isoformat(),
    }

    llm_client = get_gateway_llm_client()
    tool_executor = get_tool_executor()
    event_q: asyncio.Queue = asyncio.Queue()
    bridge = HubEventBridge()

    def on_event(ev: Dict[str, Any]) -> None:
        try:
            event_q.put_nowait(ev if isinstance(ev, dict) else {"kind": "event", "raw": ev})
        except Exception:
            pass

    def _on_llm_retry(**kwargs: Any) -> str:
        return notify_llm_reconnecting(on_event=on_event, **kwargs)

    async def _chat_once(body: Dict[str, Any]) -> Dict[str, Any]:
        messages = list(body.get("messages") or [])
        tools = body.get("tools")
        temperature = body.get("temperature")
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
        result = await llm_client.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=body.get("tool_choice"),
            extra_body=body.get("extra_body"),
        )
        if isinstance(result, dict) and not result.get("ok", True):
            # Surface backend outages instead of an empty assistant turn that
            # collapses to the hub's generic «پاسخی تولید نشد.»
            raise RuntimeError(str(result.get("error") or "llm_error"))
        return _vllm_result_to_openai(result if isinstance(result, dict) else {})

    async def chat_fn(body: Dict[str, Any]) -> Dict[str, Any]:
        return await call_with_llm_retry(
            lambda: _chat_once(body),
            on_retry=_on_llm_retry,
        )

    async def chat_stream_fn(body: Dict[str, Any], on_delta=None) -> Dict[str, Any]:
        """OpenAI SSE → assembled completion; invoke on_delta for live UI."""

        async def _stream_once() -> Dict[str, Any]:
            current_model = await llm_client._resolve_model()  # noqa: SLF001
            if not (current_model or "").strip():
                return await _chat_once(body)

            messages = _normalize_messages_for_vllm(list(body.get("messages") or []))
            temperature = body.get("temperature")
            max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
            params = _params_with_max_tokens(
                {
                    "model": current_model,
                    "messages": messages,
                    "temperature": temperature
                    if temperature is not None
                    else getattr(llm_client, "temperature", 0.7),
                    "stream": True,
                },
                max_tokens,
                getattr(llm_client, "max_tokens", None),
            )
            params["extra_body"] = _merge_vllm_extra_body(body.get("extra_body"))
            tools = body.get("tools")
            if tools:
                params["tools"] = tools
                if body.get("tool_choice") is not None:
                    params["tool_choice"] = body.get("tool_choice")

            state: Dict[str, Any] = {"content": "", "reasoning": "", "tool_calls": []}
            # Ask proxy/backends for usage on the final chunk so length-stitch can
            # detect max-token cuts even when finish_reason was historically "stop".
            params["stream_options"] = {"include_usage": True}
            try:
                stream = await llm_client.create_completion(**params)
                async for chunk in stream:
                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        try:
                            state["usage"] = {
                                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                                "completion_tokens": getattr(
                                    usage, "completion_tokens", None
                                ),
                                "total_tokens": getattr(usage, "total_tokens", None),
                            }
                        except Exception:
                            pass
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    delta = choice.delta
                    delta_dict: Dict[str, Any] = {}
                    content = getattr(delta, "content", None)
                    if content:
                        delta_dict["content"] = content
                    reasoning = (
                        getattr(delta, "reasoning", None)
                        or getattr(delta, "reasoning_content", None)
                    )
                    if reasoning:
                        delta_dict["reasoning"] = reasoning
                    tc = getattr(delta, "tool_calls", None)
                    if tc:
                        delta_dict["tool_calls"] = _delta_tool_calls_to_dicts(tc)
                    finish = getattr(choice, "finish_reason", None)
                    norm = apply_chat_delta(
                        state,
                        delta_dict,
                        finish_reason=str(finish) if finish else None,
                    )
                    await maybe_await_delta(on_delta, norm)
                assembled = completion_from_stream_state(state)
                fr = str(
                    ((assembled.get("choices") or [{}])[0] or {}).get("finish_reason")
                    or ""
                )
                if fr in ("length", "max_tokens") or (
                    fr in ("", "stop")
                    and isinstance(state.get("usage"), dict)
                    and state["usage"].get("completion_tokens")
                ):
                    logger.info(
                        "RLM chat_stream finish=%s completion_tokens=%s content_chars=%s",
                        fr or "none",
                        (state.get("usage") or {}).get("completion_tokens"),
                        len(str(state.get("content") or "")),
                    )
                return assembled
            except Exception as exc:
                # Context overflow cannot be fixed by non-stream retry — let the
                # session compact/recover path handle it (avoids a second 400).
                try:
                    from .rlm.runtime.context_budget import looks_like_context_overflow

                    if looks_like_context_overflow(str(exc)):
                        raise
                except ImportError:
                    pass
                err_l = str(exc or "").lower()
                # Dead backend pool: raise so outer AF-parity reconnect owns the
                # wait (non-stream retry on the same dead peers only burns time).
                if "all_backends_failed" in err_l or "chat_model_unavailable" in err_l:
                    logger.error(
                        "RLM chat_stream backend outage (reconnect path): %s", exc
                    )
                    raise
                logger.warning("RLM chat_stream fallback to non-stream: %s", exc)
                return await _chat_once(body)

        return await call_with_llm_retry(_stream_once, on_retry=_on_llm_retry)

    tool_runner = make_tools_host_port(
        tool_executor,
        session_id=session_id,
        user_id=user_id,
        user_message=user_message,
    )

    model = ""
    try:
        model = await llm_client._resolve_model()  # noqa: SLF001
    except Exception:
        model = str(getattr(llm_client, "model", "") or "").strip()
    if not model:
        model = "default"

    artifact_dir = _durable_artifact_dir(session_id, user_id)
    # Mentions are turn-scoped; restore durable case pool on continue turns.
    from .host_adapters.case_mentions import (
        inventory_has_cases,
        restore_or_build_case_inventory,
        save_case_inventory,
    )

    try:
        if not inventory_has_cases(case_inventory):
            case_inventory = await restore_or_build_case_inventory(
                context=request_context,
                user_id=user_id,
                session_id=session_id,
                artifact_dir=artifact_dir,
            )
        elif inventory_has_cases(case_inventory):
            # Persist what workspace already resolved (covers first turn).
            save_case_inventory(artifact_dir, case_inventory)
    except Exception:
        logger.debug("RLM case inventory restore skipped", exc_info=True)

    pool_defs, _inv, pool_closures = await build_chat_session_pool(
        artifact_dir=artifact_dir,
        file_registry=file_registry,
        artifacts=artifacts,
        pool_name="session",
        auth_context={"user_id": user_id},
        request_context=request_context,
        case_inventory=case_inventory,
    )

    engine = RlmOrchestratorEngine()
    tools_bind = read_tools_binding(engine.manifest)
    sess_bind = read_session_binding(engine.manifest)
    user_cfg = dict(orchestrator_config or {}) if isinstance(orchestrator_config, dict) else {}
    # Catalog window (AF parity): compact @ ~90% of this model's context_length.
    try:
        from .host_adapters.model_catalog import resolve_model_context_window

        window_decl = (getattr(engine.manifest, "extensions", None) or {}).get("context_window")
        user_cfg, _meta = await resolve_model_context_window(
            str(user_cfg.get("model") or model),
            cfg=user_cfg,
            min_context=(window_decl or {}).get("min_tokens") if isinstance(window_decl, dict) else None,
        )
    except Exception:
        logger.warning("RLM catalog context_length stamp failed", exc_info=True)
    tool_ids: List[str] = []
    try:
        tool_ids = await resolve_tool_ids_for_run(
            binding=tools_bind,
            tool_executor=tool_executor,
            user_id=user_id,
            user_message=user_message,
            orchestrator_config=user_cfg,
            file_registry=file_registry,
            artifacts=artifacts,
            active_plugin=active_plugin,
            discourse_focus=discourse_focus if isinstance(discourse_focus, dict) else None,
        )
    except Exception:
        logger.warning(
            "RLM tools_binding resolve failed session=%s; continuing with empty allowlist",
            session_id,
            exc_info=True,
        )
        tool_ids = []
    logger.info(
        "📦 RLM tools selected session=%s page=%s: %d %s",
        session_id, active_plugin or "-", len(tool_ids), tool_ids,
    )
    tool_specs: List[Dict[str, Any]] = []
    if tool_ids:
        try:
            tool_specs = await kernel_tool_specs(tool_executor, tool_ids)
        except Exception:
            logger.warning("RLM tool schemas unavailable session=%s", session_id, exc_info=True)

    config_key = tools_bind.config_key if tools_bind else "tool_ids"
    # Stable engine session id when durable — never mint a new UUID per message.
    engine_sid = engine_session_id_for(
        sess_bind, chat_session_id=session_id, orchestrator_id="rlm"
    )
    # Hub multi-turn memory: chat lifetime unless the user explicitly chose heartbeat.
    raw_lifetime = str(user_cfg.get("run_lifetime") or "chat").strip().lower()
    if raw_lifetime in ("heartbeat", "always_on", "alwayson", "resident"):
        run_lifetime = "heartbeat"
    else:
        run_lifetime = "chat"
    # Host-owned paths / pools / tool allowlist always win over client config.
    cfg: Dict[str, Any] = {
        **user_cfg,
        "model": str(user_cfg.get("model") or model),
        "run_lifetime": run_lifetime,
        "artifact_dir": artifact_dir,
        "workspace_dir": artifact_dir,
        "session_id": engine_sid,
        "run_id": session_id,
        "pool_defs": pool_defs,
        "user_id": user_id,
        config_key: list(tool_ids),
        "tool_specs": tool_specs,
        "tool_namespace": tools_bind.namespace if tools_bind else None,
    }
    # Refresh live session window when catalog learned a value this turn.
    try:
        ctx_len = int(cfg.get("context_length") or 0)
    except (TypeError, ValueError):
        ctx_len = 0
    if ctx_len > 0:
        try:
            live_early = lookup_live_runtime_session(engine_sid)
            if live_early is not None:
                prev = int(getattr(live_early, "context_length", 0) or 0)
                if prev != ctx_len:
                    live_early.context_length = ctx_len
                    if hasattr(live_early, "cfg") and isinstance(live_early.cfg, dict):
                        live_early.cfg["context_length"] = ctx_len
        except Exception:
            logger.debug("live context_length stamp skipped", exc_info=True)
    try:
        if inventory_has_cases(case_inventory):
            cfg["case_inventory"] = case_inventory
    except Exception:
        pass
    if max_steps is not None and "rlm_max_turns" not in cfg:
        cfg["rlm_max_turns"] = int(max_steps)
    try:
        block_selection = cfg.get(
            "host_context_blocks", manifest_defaults(engine.manifest).get("host_context_blocks", "*")
        )
    except Exception:
        block_selection = "*"
    host_blocks = prompt_blocks(host_block_kwargs, block_selection)
    if host_blocks:
        cfg["system_prompt"] = replace_host_context(str(cfg.get("system_prompt") or ""), host_blocks)

    host = HostPorts(
        chat=chat_fn,
        chat_stream=chat_stream_fn,
        tools=tool_runner,
        pool_search=pool_closures.get("pool_search"),
        pool_resolve=pool_closures.get("pool_resolve"),
        pool_list=pool_closures.get("pool_list"),
        pool_read=pool_closures.get("pool_read"),
        events=on_event,
    )
    envelope = RunEnvelope(
        api_version=1,
        orchestrator_id="rlm",
        request_id=trace_id,
        input={"user_content": user_message},
        config=cfg,
        meta={"user_id": user_id, "session_id": session_id},
        run_id=session_id,
        session_id=session_id,
        tenant_id=user_id,
    )

    try:
        live = (
            lookup_live_runtime_session(engine_sid)
            if sess_bind is None or sess_bind.is_durable
            else None
        )
        base_system = str(cfg.get("system_prompt") or "")
        continuing = live is not None
        if continuing:
            rebind_host_ports(live, host)
            # Do not replay prior terminal / answer / reasoning into this turn's SSE.
            bridge.seed_from_live_session(live)
            # Refresh allowlist / pools on the live session for this turn.
            # If this turn rebuilt pools without a case pool, keep the prior one.
            try:
                prev_pools = list(getattr(live, "pool_defs", None) or [])
                has_case = any(
                    isinstance(p, dict)
                    and str(p.get("name") or p.get("pool_key") or "").strip().lower()
                    == "case"
                    for p in pool_defs
                )
                if not has_case:
                    for p in prev_pools:
                        if (
                            isinstance(p, dict)
                            and str(p.get("name") or p.get("pool_key") or "")
                            .strip()
                            .lower()
                            == "case"
                        ):
                            pool_defs = list(pool_defs) + [dict(p)]
                            # Rebuild closures so pool_* host ports see the case pool.
                            try:
                                from . import host_pools as _hp

                                pool_closures = _hp.make_pool_host_closures(
                                    artifact_dir=artifact_dir,
                                    pool_defs=pool_defs,
                                    union_inventory=_inv,
                                    download=make_session_artifact_download(
                                        {"user_id": user_id}
                                    ),
                                )
                                host.pool_search = pool_closures.get("pool_search")
                                host.pool_resolve = pool_closures.get("pool_resolve")
                                host.pool_list = pool_closures.get("pool_list")
                                host.pool_read = pool_closures.get("pool_read")
                            except Exception:
                                logger.debug(
                                    "case pool closure rebuild skipped",
                                    exc_info=True,
                                )
                            break
                # Prefer live.cfg case_inventory if turn still empty.
                if (
                    hasattr(live, "cfg")
                    and isinstance(live.cfg, dict)
                    and not inventory_has_cases(case_inventory)
                ):
                    prev_ci = live.cfg.get("case_inventory")
                    if inventory_has_cases(prev_ci if isinstance(prev_ci, dict) else None):
                        case_inventory = dict(prev_ci)  # type: ignore[arg-type]
                if hasattr(live, "cfg") and isinstance(live.cfg, dict) and tools_bind:
                    live.cfg["tool_namespace"] = tools_bind.namespace
                if hasattr(live, "refresh_tool_surface"):
                    await live.refresh_tool_surface(tool_ids, tool_specs)
                else:
                    live.tool_ids = list(tool_ids)
                live.pool_defs = pool_defs
                try:
                    from .rlm.runtime.persist import load_pool_files_baseline

                    live._pool_files_baseline = load_pool_files_baseline(artifact_dir)
                except Exception:
                    pass
                if hasattr(live, "cfg") and isinstance(live.cfg, dict):
                    live.cfg[config_key] = list(tool_ids)
                    live.cfg["pool_defs"] = pool_defs
                    live.cfg["run_lifetime"] = run_lifetime
                    if inventory_has_cases(case_inventory):
                        live.cfg["case_inventory"] = case_inventory
                # Keep resident chat lifetime if the live session was created older.
                if str(getattr(live, "run_lifetime", "") or "") not in (
                    "chat",
                    "heartbeat",
                ):
                    live.run_lifetime = run_lifetime
            except Exception:
                logger.debug("live session tool/pool refresh skipped", exc_info=True)
            # Previous turn may have left status=error; clear stamps so continue works.
            live_status = str(getattr(live, "status", "") or "").strip().lower()
            if live_status in ("error", "cancelled", "incomplete"):
                try:
                    if hasattr(live, "_reset_failure_flags_for_resume"):
                        live._reset_failure_flags_for_resume()
                        logger.info(
                            "RLM reset failure flags before continue chat=%s status_was=%s",
                            session_id,
                            live_status,
                        )
                except Exception:
                    logger.warning(
                        "RLM failure-flag reset failed chat=%s", session_id, exc_info=True
                    )
            # Warm daemon: if memory is empty (zombie), hydrate/seed first.
            mem_mode, user_on_msgs = resume_or_seed_live_memory(
                live,
                user_message=user_message,
                base_system=base_system,
                chat_history=chat_history,
                self_compact=_kwargs.get("self_compact"),
            )
            if hasattr(live, "refresh_host_context"):
                try:
                    live.refresh_host_context(host_blocks)
                except Exception:
                    logger.debug("RLM host context refresh skipped", exc_info=True)
            session = RlmOrchestratorSession(live, envelope)
            tool_runs_start = len(getattr(live, "tool_runs", None) or [])
            cont_msg: Any = "" if user_on_msgs else user_message
            run_task = asyncio.create_task(session.continue_with(cont_msg))
            logger.info(
                "RLM durable continue chat=%s engine_sid=%s memory=%s "
                "kernel_alive=%s msgs=%s logs=%s",
                session_id,
                engine_sid,
                mem_mode,
                bool(getattr(getattr(live, "kernel", None), "alive", False)),
                len(getattr(live, "messages", None) or []),
                len(getattr(live, "log_lines", None) or []),
            )
        else:
            session = await engine.create_session(envelope, host)
            live_new = getattr(session, "_session", None)
            tool_runs_start = len(getattr(live_new, "tool_runs", None) or [])
            register_runtime_session(live_new)
            if (
                live_new is not None
                and hasattr(live_new, "cfg")
                and isinstance(live_new.cfg, dict)
                and inventory_has_cases(case_inventory)
            ):
                live_new.cfg["case_inventory"] = case_inventory
            # AF ladder on daemon miss: transcript disk → hub chat_history → cold.
            mem_mode, user_on_msgs = resume_or_seed_live_memory(
                live_new,
                user_message=user_message,
                base_system=base_system,
                chat_history=chat_history,
                self_compact=_kwargs.get("self_compact"),
            )
            if mem_mode in ("resumed", "reseeded", "seeded"):
                # User turn already on messages; do not cold-run from user_content alone.
                try:
                    live_new.user_content = None
                except Exception:
                    pass
                run_task = asyncio.create_task(
                    session.continue_with("" if user_on_msgs else user_message)
                )
                logger.info(
                    "RLM durable resume chat=%s engine_sid=%s memory=%s msgs=%s",
                    session_id,
                    engine_sid,
                    mem_mode,
                    len(getattr(live_new, "messages", None) or []),
                )
            else:
                run_task = asyncio.create_task(session.run())
                logger.info(
                    "RLM durable create+run chat=%s engine_sid=%s",
                    session_id,
                    engine_sid,
                )

        while not run_task.done():
            try:
                ev = await asyncio.wait_for(event_q.get(), timeout=0.15)
            except asyncio.TimeoutError:
                await asyncio.sleep(0)
                continue
            for mapped in bridge.map_event(ev):
                yield mapped

        # Drain remaining events
        while not event_q.empty():
            try:
                ev = event_q.get_nowait()
            except asyncio.QueueEmpty:
                break
            for mapped in bridge.map_event(ev):
                yield mapped

        for mapped in bridge.close_phases():
            yield mapped

        result = await run_task
        text = ""
        if result and isinstance(result.output, dict):
            text = str(result.output.get("text") or "")
        elif result and result.output is not None:
            text = str(result.output)

        # Prefer the engine's stitched answer when longer than streamed partials
        # (length-continuation may finish after the UI cursor stopped).
        streamed = str(bridge.collected_answer or "")
        engine_text = str(text or "")
        if engine_text and len(engine_text) > len(streamed):
            if not bridge._answer_started:
                yield {"type": "answer_start", "step": 1}
            if streamed and engine_text.startswith(streamed):
                yield {"type": "answer", "content": engine_text[len(streamed) :]}
            else:
                yield {"type": "answer", "content": engine_text}
            bridge.collected_answer = engine_text

        # Upload kernel-produced files and attach them to the chat session
        # workspace so the Files panel / session_files tools see them.
        attached_files: List[Dict[str, Any]] = []
        try:
            attached_files = await sync_rlm_deliverables_to_chat(
                orch_session=session,
                chat_session_id=session_id,
                user_id=user_id,
                run_id=session_id,
            )
        except Exception:
            logger.warning(
                "RLM deliverable→session sync failed session=%s",
                session_id,
                exc_info=True,
            )
            attached_files = []
        base_reg = len(file_registry or []) + len(artifacts or [])
        for ev in attached_sse_events(
            attached_files, base_registry_size=base_reg
        ):
            yield ev

        runtime = getattr(session, "_session", None)
        err_hint = ""
        if runtime is not None:
            err_hint = str(getattr(runtime, "error", "") or "")
        if not err_hint and result is not None:
            ext0 = getattr(result, "extensions", None) or {}
            if isinstance(ext0, dict):
                err_hint = str(ext0.get("rlm_error") or "")
        final_text, rlm_status = resolve_hub_final_answer(
            streamed=str(bridge.collected_answer or ""),
            engine_text=engine_text,
            runtime_session=runtime,
            thinking_text=str(bridge.collected_thinking or ""),
            error_hint=err_hint,
        )
        # Never yield an empty / placeholder RLM done — main.py would stamp junk.
        from .host_adapters.hub_final_answer import (
            ensure_nonempty_hub_answer,
            is_placeholder_answer,
        )

        if is_placeholder_answer(final_text):
            final_text = ensure_nonempty_hub_answer(
                final_text,
                thinking=str(bridge.collected_thinking or ""),
                error=err_hint,
                status=str(rlm_status or ""),
            )
            rlm_status = rlm_status or "incomplete"
            logger.warning(
                "RLM hub placeholder final coerced session=%s status=%s err=%s think=%s",
                session_id,
                rlm_status,
                bool(err_hint),
                len(str(bridge.collected_thinking or "")),
            )

        # AF parity: stream the notice as answer so the FE shows it live.
        # FE *appends* answer deltas — never emit a second full answer on top of
        # a placeholder (that glued «see process log» + Persian notice).
        prev = str(bridge.collected_answer or "")
        if final_text and final_text != prev:
            if is_placeholder_answer(prev):
                # Replace logically for done/FINAL; skip streaming another chunk
                # when the UI already painted the stub (FINAL replaces display).
                bridge.collected_answer = final_text
            else:
                if not bridge._answer_started:
                    yield {"type": "answer_start", "step": 1}
                    bridge._answer_started = True
                if prev and final_text.startswith(prev):
                    piece = final_text[len(prev) :]
                    if piece:
                        yield {"type": "answer", "content": piece}
                else:
                    yield {"type": "answer", "content": final_text}
                bridge.collected_answer = final_text
        elif final_text and not prev:
            if not bridge._answer_started:
                yield {"type": "answer_start", "step": 1}
                bridge._answer_started = True
            yield {"type": "answer", "content": final_text}
            bridge.collected_answer = final_text

        try:
            record_served_turn(runtime, user_message, final_text)
        except Exception:
            logger.debug("RLM served-turn record skipped", exc_info=True)
        extensions = dict(result.extensions if result else {}) or {}
        if attached_files:
            extensions["attached_session_files"] = len(attached_files)
        extensions["engine_session_id"] = engine_sid
        extensions["session_continued"] = bool(continuing)
        if rlm_status:
            extensions["rlm_status"] = rlm_status
        if result is not None:
            extensions.setdefault("run_status", getattr(result, "status", None))
        logger.info(
            "RLM hub done session=%s status=%s answer_chars=%s think_chars=%s",
            session_id,
            rlm_status or getattr(result, "status", None),
            len(str(final_text or "")),
            len(str(bridge.collected_thinking or "")),
        )
        done_ev: Dict[str, Any] = {
            "type": "done",
            "trace_id": trace_id,
            "session_id": session_id,
            "engine": "rlm",
            "final_answer": final_text,
            "answer": final_text,
            "thinking": bridge.collected_thinking or None,
            "tool_runs": list((getattr(runtime, "tool_runs", None) or [])[tool_runs_start:]),
            "extensions": extensions,
        }
        if rlm_status in ("error", "incomplete"):
            done_ev["status"] = rlm_status
        if err_hint:
            done_ev["error"] = err_hint[:2000]
        yield done_ev
    except Exception as exc:
        logger.exception("RLM chat stream failed: %s", exc)
        err_s = str(exc or "")
        err_l = err_s.lower()
        if "all_backends_failed" in err_l:
            msg = (
                "مدل در دسترس نیست (همهٔ بک‌اندهای LLM از کار افتاده‌اند یا unhealthy هستند). "
                "لطفاً وضعیت llm-gateway / مدل‌ها را بررسی کنید و دوباره تلاش کنید."
            )
        elif "chat_model_unavailable" in err_l:
            msg = "مدل چت برای این سرویس bind نشده یا در دسترس نیست."
        elif "temporary failure in name resolution" in err_l or "nameresolutionerror" in err_l:
            msg = "خطای شبکه/DNS در دسترسی به سرور مدل. اتصال بک‌اند LLM را بررسی کنید."
        else:
            msg = f"RLM orchestrator error: {exc}"
        yield {"type": "error", "message": msg}
        yield {"type": "answer", "content": msg}
        yield {
            "type": "done",
            "trace_id": trace_id,
            "session_id": session_id,
            "engine": "rlm",
            "final_answer": msg,
            "answer": msg,
            "error": err_s,
        }
