"""Hub chat stream for the RRSI controller.

Evolve turns run ``rounds_per_turn`` RRSI rounds on the conversation's evolve
set and answer with the best-judged trial of the accepted harness. Serve turns
(no rounds due, or the horizon T is spent) stream the policy engine live under
the incumbent harness.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import nullcontext
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple

from src.tools.side_effects import rehearsal

from .._shared.engine_turn import run_engine_turn, validate_inner_engine
from .._shared.hub_stream import HubEmitter, TaskOutcome, run_with_events
from .._shared.llm_roles import gateway_llm
from .._shared.manifest_config import own_config_keys, package_manifest, passthrough_config, resolve_config
from .._shared.state_dir import session_state_dir
from ..chat_engine_resolve import chat_stream_manifest, default_chat_orchestrator_id, stream_agent_for_id
from .core.config import RRSIConfig
from .runtime import harness as H
from .runtime.controller import RrsiController, RrsiSettings, TurnOutcome, build_evolve_set
from .runtime.domain import Task, Trial, default_domain
from .runtime.state import RrsiState

logger = logging.getLogger(__name__)


async def _tool_catalog(
    user_id: str, active_plugin: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Optional[Set[str]]]:
    """Catalog tools this user may use on this page (same RBAC and page gate as the host)."""
    try:
        from src.auth.permissions_client import resolve_allowed_tools
        from src.tools.executor import get_tool_executor
        from src.tools.llm_tool_selection import page_allows_tool

        manifest = await get_tool_executor().get_tools_manifest()
        allowed = await resolve_allowed_tools(manifest if isinstance(manifest, list) else [], user_id or None)
    except Exception:  # noqa: BLE001 — without a catalog, tool_ids edits are refused
        logger.warning("rrsi: tool catalog unavailable", exc_info=True)
        return [], None
    rows = []
    for t in allowed or []:
        if not page_allows_tool(t, active_plugin):
            continue
        tid = str((t or {}).get("id") or (t or {}).get("name") or "").strip()
        if tid:
            rows.append({"id": tid, "description": str((t or {}).get("description") or "")})
    return rows, {r["id"] for r in rows}


def _compose(
    harness: H.Harness,
    task: Task,
    *,
    domain,
    base_kwargs: Dict[str, Any],
    base_config: Dict[str, Any],
    host_max_steps: Optional[int],
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[int]]:
    kwargs = dict(base_kwargs)
    history = base_kwargs.get("chat_history") if task.current else task.context
    kwargs["user_message"] = task.request
    kwargs["chat_history"] = H.trim_history(list(history or []), harness.get("history_turns"))
    block = H.prompt_block(domain, harness)
    base_prompt = str(base_kwargs.get("host_page_prompt") or "").strip()
    kwargs["host_page_prompt"] = "\n\n".join(x for x in (base_prompt, block) if x) or None
    ocfg = {**base_config, **dict(harness.get("engine_config") or {})}
    if harness.get("tool_ids"):
        ocfg["tool_ids"] = list(harness["tool_ids"])
    return kwargs, ocfg, harness.get("max_steps") or host_max_steps


def _round_summary(out: TurnOutcome) -> List[Dict[str, Any]]:
    return [
        {
            "t": r.t,
            "accepted": r.accepted,
            "delta": round(r.delta, 6),
            "b_t": r.budget,
            "sigma_t": r.sigma,
            "S_incumbent": round(r.incumbent.ev.S, 6),
            "S_next": round(r.next_incumbent.ev.S, 6),
            "decisions": [d.to_json() for d in r.decisions],
            "top_modes": [m.get("mode") for m in (r.report.get("failure_modes") or [])[:5]],
        }
        for r in out.rounds
    ]


async def stream_agent_rrsi(
    user_message: str,
    session_id: str,
    max_steps: Optional[int] = None,
    orchestrator_config: Optional[Dict[str, Any]] = None,
    **engine_kwargs: Any,
) -> AsyncGenerator[Dict[str, Any], None]:
    manifest = package_manifest(__file__)
    emitter = HubEmitter(manifest.id, session_id)
    yield emitter.start()
    domain = default_domain()
    user_id = str(engine_kwargs.get("user_id") or "default")
    try:
        cfg = resolve_config(manifest, orchestrator_config)
        st = RrsiSettings.from_config(cfg)
        rcfg = RRSIConfig.from_mapping(cfg)
        policy_id = st.policy_engine or default_chat_orchestrator_id()
        validate_inner_engine(policy_id, caller_id=manifest.id)
        policy_manifest = chat_stream_manifest(policy_id)
        allowed_keys = own_config_keys(policy_manifest) if policy_manifest is not None else set()
    except Exception as exc:  # noqa: BLE001
        for ev in emitter.fail(str(exc)):
            yield ev
        return

    base_config = {**passthrough_config(manifest, orchestrator_config), **st.policy_config}
    base_kwargs = dict(engine_kwargs, user_message=user_message, session_id=session_id)
    catalog, known_tools = await _tool_catalog(user_id, engine_kwargs.get("active_plugin"))
    state = RrsiState(session_state_dir(manifest.id, user_id, session_id))
    queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()

    def log(line: str) -> None:
        if st.show_round_log:
            for ev in emitter.log(line):
                queue.put_nowait(ev)

    def progress(payload: Dict[str, Any]) -> None:
        if payload.get("phase") == "evaluate":
            done, total = payload.get("completed", 0), payload.get("total", 1)
            queue.put_nowait(emitter.progress(
                f"{payload.get('job')}: {done}/{total} trials", progress=done / max(total, 1),
                phase="evaluate", job=payload.get("job"),
            ))

    # Context checkpoints of this request's trials; the served trial's becomes the turn's.
    checkpoints: Dict[int, Tuple[Trial, Dict[str, Any]]] = {}

    async def policy(harness: H.Harness, task: Task, index: int) -> Trial:
        kwargs, ocfg, steps = _compose(harness, task, domain=domain, base_kwargs=base_kwargs,
                                       base_config=base_config, host_max_steps=max_steps)
        # Replays of earlier requests must not repeat their actions for real.
        with (nullcontext() if task.current else rehearsal()):
            turn = await run_engine_turn(
                policy_id, kwargs, orchestrator_config=ocfg, max_steps=steps,
                session_suffix=f"{manifest.id}_{task.id}_{index}_{uuid.uuid4().hex[:6]}",
            )
        trial = Trial(task_id=task.id, index=index, answer=turn.text if turn.ok else "",
                      error="" if turn.ok else (turn.error or "empty answer"),
                      tokens=turn.tokens, tool_runs=turn.tool_runs, elapsed_sec=turn.elapsed_sec)
        if task.current and turn.self_compact:
            checkpoints[id(trial)] = (trial, turn.self_compact)
        return trial

    controller = RrsiController(
        st, rcfg, domain, gateway_llm(), policy, state,
        allowed_engine_keys=allowed_keys, known_tool_ids=known_tools, tool_catalog=catalog,
        on_log=log, on_progress=progress,
    )

    try:
        tasks, current_id = build_evolve_set(user_message, engine_kwargs.get("chat_history"), st.replay_turns,
                                             st.replay_max_request_chars)
        instructions = str(engine_kwargs.get("persona_prompt_block") or "")
        for task in tasks.values():
            task.instructions = instructions
        outcome = TaskOutcome()
        async for ev in run_with_events(controller.turn(tasks, current_id), queue, outcome):
            yield ev
        if outcome.error is not None:
            raise outcome.error
        result: TurnOutcome = outcome.value
        extensions: Dict[str, Any] = {
            "orchestrator": manifest.id,
            "rrsi": {
                "policy_engine": policy_id,
                "harness_version": result.version,
                "harness": result.harness,
                "rounds": _round_summary(result),
                "served_only": result.served_only,
                "evolve_set": [t.id for t in tasks.values()],
                "state_dir": str(state.root),
            },
        }
        answer = result.answer
        if answer is not None and not answer.failed:
            for ev in emitter.log(f"answer · best judged trial (reward {answer.reward:.3f}) of harness v{result.version}"):
                yield ev
            for ev in emitter.answer(answer.answer):
                yield ev
            extensions["rrsi"]["answer_reward"] = round(answer.reward, 6)
            extensions["rrsi"]["answer_verdicts"] = answer.verdicts
            done = emitter.done(answer.answer, tool_runs=answer.tool_runs, extensions=extensions)
            served = checkpoints.get(id(answer))
            if served is not None and served[0] is answer:
                done["self_compact"] = served[1]
                # The served trial ran off-stream: surface its context compactions like a live turn.
                for event in served[1].get("events") or []:
                    if isinstance(event, dict):
                        yield {**event, "type": "context_compacted"}
            yield done
            return

        # Serve: stream the policy engine live under the incumbent harness.
        if not result.served_only:
            for ev in emitter.log("no successful judged trial for this request; answering live with the incumbent harness"):
                yield ev
        kwargs, ocfg, steps = _compose(result.harness, tasks[current_id], domain=domain, base_kwargs=base_kwargs,
                                       base_config=base_config, host_max_steps=max_steps)
        for ev in emitter.close_thinking():
            yield ev
        fn = stream_agent_for_id(policy_id)
        async for ev in fn(max_steps=steps, orchestrator_config=ocfg, **kwargs):
            et = ev.get("type")
            if et == "start":
                continue
            if et == "done":
                ext = dict(ev.get("extensions") or {})
                ext.update(extensions)
                ev = {**ev, "engine": manifest.id, "extensions": ext,
                      "thinking": (emitter.thinking + str(ev.get("thinking") or "")) or None}
            yield ev
    except Exception as exc:  # noqa: BLE001
        logger.exception("RRSI chat stream failed: %s", exc)
        for ev in emitter.fail(str(exc), extensions={"rrsi": {"state_dir": str(state.root)}}):
            yield ev
