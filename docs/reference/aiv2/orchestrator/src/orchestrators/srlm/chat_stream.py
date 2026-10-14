"""Hub chat stream for the SRLM (SelfReflect) controller.

search: K answers — sampled from the bound chat model (paper setting) or, when
``peer_engine`` is set, one full turn of that chat engine per sample.
select: the configured summary strategy, streamed as the answer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from .._shared.engine_turn import run_engine_turn, validate_inner_engine
from .._shared.host_context import page_prompt_block, prompt_blocks, render_blocks
from .._shared.hub_stream import HubEmitter, StreamedReply, TaskOutcome, run_with_events, stream_llm
from .._shared.llm_roles import gateway_llm
from .._shared.manifest_config import package_manifest, passthrough_config, resolve_config
from .runtime import strategies as S
from .runtime.controller import (
    Sample,
    SelfReflectController,
    SrlmResult,
    SrlmSettings,
    answer_distribution,
    _sum_tokens,
)

logger = logging.getLogger(__name__)

_PREVIEW_CHARS = 280


def _preview(text: str) -> str:
    one_line = " ".join(str(text or "").split())
    return one_line if len(one_line) <= _PREVIEW_CHARS else one_line[: _PREVIEW_CHARS - 1] + "…"


async def stream_agent_srlm(
    user_message: str,
    session_id: str,
    max_steps: Optional[int] = None,
    orchestrator_config: Optional[Dict[str, Any]] = None,
    **engine_kwargs: Any,
) -> AsyncGenerator[Dict[str, Any], None]:
    manifest = package_manifest(__file__)
    emitter = HubEmitter(manifest.id, session_id)
    yield emitter.start()
    try:
        cfg = resolve_config(manifest, orchestrator_config)
        st = SrlmSettings.from_config(cfg)
        if st.peer_engine:
            validate_inner_engine(st.peer_engine, caller_id=manifest.id)
    except Exception as exc:  # noqa: BLE001 — bad config is reported to the user
        for ev in emitter.fail(str(exc)):
            yield ev
        return

    queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()

    def on_progress(payload: Dict[str, Any]) -> None:
        phase = payload.get("phase")
        if phase == "search" and "sample" in payload:
            sample = payload["sample"]
            done, total = payload.get("completed", 0), payload.get("total", st.num_samples)
            head = f"[{done}/{total}] sample {sample['index'] + 1}"
            if sample.get("ok"):
                line = f"{head}: {_preview(sample['text'])}" if st.show_samples else f"{head}: ok"
            else:
                line = f"{head}: failed — {sample.get('error')}"
            for ev in emitter.log(line):
                queue.put_nowait(ev)
            queue.put_nowait(
                emitter.progress(
                    f"sampled {done}/{total}",
                    progress=done / max(total, 1),
                    phase="search",
                )
            )
        elif phase == "search":
            source = st.peer_engine or "chat model"
            for ev in emitter.log(
                f"SelfReflect · sampling {payload.get('total')} answers from {source} "
                f"(T={st.sample_temperature}, strategy={st.summary_strategy})"
            ):
                queue.put_nowait(ev)

    llm = gateway_llm()
    controller = SelfReflectController(st, llm, on_progress=on_progress)
    compactor = _self_compactor(orchestrator_config, engine_kwargs.get("self_compact"), st.history_turns)
    if compactor is not None:
        try:
            context, sc_events = await compactor.prepare_history(
                engine_kwargs["self_compact"].get("messages") or [],
                user_message,
                max_turns=st.history_turns,
            )
        except Exception as exc:  # noqa: BLE001 — fall back to the plain recent-turns window
            logger.warning("SRLM SelfCompact context failed, using recent turns: %s", exc)
            compactor, sc_events = None, []
            context = controller.context_messages(engine_kwargs.get("chat_history"))
        for ev in sc_events:
            yield ev
            if ev.get("type") == "context_compacted":
                for out in emitter.log(
                    f"context compacted ({ev.get('trigger')}) · {ev.get('tokens_before')}→{ev.get('tokens_after')} tokens"
                ):
                    yield out
    else:
        context = controller.context_messages(engine_kwargs.get("chat_history"))
    # A peer engine renders the host blocks into its own samples; SRLM's selection step
    # writes the final reply itself, so it always needs them (persona, clock, memory, …).
    block_kwargs = {
        "page_prompt_block": page_prompt_block(
            engine_kwargs.get("active_plugin"), engine_kwargs.get("host_page_prompt")
        ),
        **engine_kwargs,
    }
    select_host_text = render_blocks(prompt_blocks(block_kwargs, cfg.get("host_context_blocks")))
    host_text = "" if st.peer_engine else select_host_text
    base_context = list(context)
    context = _with_host_text(base_context, host_text)
    tool_runs: List[Dict[str, Any]] = []

    sampler = None
    if st.peer_engine:
        peer_cfg = {**passthrough_config(manifest, orchestrator_config), **st.peer_config}
        peer_kwargs = dict(engine_kwargs, user_message=user_message, session_id=session_id)

        async def sampler(index: int) -> Sample:
            def on_peer_event(ev: Dict[str, Any]) -> None:
                if ev.get("type") == "tool_start":
                    for out in emitter.log(f"  sample {index + 1} · tool {ev.get('tool_name') or ev.get('tool_id')}"):
                        queue.put_nowait(out)

            turn = await run_engine_turn(
                st.peer_engine,
                peer_kwargs,
                orchestrator_config=peer_cfg,
                max_steps=max_steps,
                session_suffix=f"{manifest.id}_{emitter.trace_id[:8]}_{index}",
                on_event=on_peer_event,
            )
            tool_runs.extend(turn.tool_runs)
            return Sample(
                index=index,
                text=turn.text if turn.ok else "",
                error=turn.error if not turn.ok else "",
                tokens=turn.tokens,
                tool_runs=turn.tool_runs,
                elapsed_sec=turn.elapsed_sec,
            )

    samples: List[Sample] = []
    try:
        if st.needs_samples:
            outcome = TaskOutcome()
            async for ev in run_with_events(
                controller.search(sampler or controller.direct_sampler(user_message, context)),
                queue,
                outcome,
            ):
                yield ev
            if outcome.error is not None:
                raise outcome.error
            samples = list(outcome.value or [])
            overflow = _overflow_error(samples, compactor) if sampler is None else ""
            if overflow:
                anchor = {"role": "user", "content": user_message}
                messages, ev = await compactor.recover_overflow(
                    base_context + [anchor], tools=None, anchor=anchor, step=2, error_text=overflow
                )
                yield ev
                for out in emitter.log(
                    f"context compacted (provider_overflow) · {ev.get('tokens_before')}→{ev.get('tokens_after')} tokens; resampling"
                ):
                    yield out
                context = _with_host_text(_compacted_context(compactor, messages, anchor), host_text)
                outcome = TaskOutcome()
                async for ev in run_with_events(
                    controller.search(controller.direct_sampler(user_message, context)), queue, outcome
                ):
                    yield ev
                if outcome.error is not None:
                    raise outcome.error
                samples = list(outcome.value or [])

        dist = answer_distribution(samples)
        if samples:
            for ev in emitter.log(
                f"distribution · {dist['n_ok']}/{dist['n_samples']} ok, "
                f"{dist['n_distinct']} distinct, top share {dist['top_share']}"
            ):
                yield ev

        outcome = TaskOutcome()
        async for ev in run_with_events(
            controller.plan_select(
                user_message, _with_host_text(context, select_host_text) if st.peer_engine else context, samples
            ),
            queue,
            outcome,
        ):
            yield ev
        if outcome.error is not None:
            raise outcome.error
        plan = outcome.value
        yield emitter.progress(f"summarizing ({plan.strategy})", phase="select")
        for ev in emitter.log(f"select · {plan.strategy}"):
            yield ev

        select_tokens = plan.tokens
        if plan.text is not None:
            final_text = plan.text
        else:
            # Strategies whose raw output is post-processed go to the process log first.
            live = plan.strategy != S.COT and not plan.enable_thinking
            reply = StreamedReply()
            async for ev in stream_llm(
                emitter,
                plan.messages or [],
                reply,
                as_answer=live,
                temperature=plan.temperature,
                max_tokens=plan.max_tokens,
                enable_thinking=plan.enable_thinking,
                sampling=plan.sampling or None,
            ):
                yield ev
            if reply.error and not reply.text.strip():
                raise RuntimeError(f"summary generation failed: {reply.error}")
            final_text = controller.finalize(plan, reply.text)
        for ev in emitter.finish_answer(final_text):
            yield ev

        result = SrlmResult(
            text=final_text,
            strategy=plan.strategy,
            samples=samples,
            distribution=dist,
            analysis=plan.analysis,
            tokens=_sum_tokens([s.tokens for s in samples] + [select_tokens]),
        )
        extensions: Dict[str, Any] = {
            "orchestrator": manifest.id,
            "srlm": {
                "strategy": result.strategy,
                "num_samples": st.num_samples,
                "peer_engine": st.peer_engine or None,
                "sample_temperature": st.sample_temperature,
                "summary_temperature": plan.temperature,
                "distribution": result.distribution,
                "samples": [s.to_dict() for s in samples] if st.show_samples else None,
                "analysis": result.analysis or None,
                "tokens_known": result.tokens,
            },
        }
        yield _with_checkpoint(emitter.done(final_text, tool_runs=tool_runs, extensions=extensions), compactor)
    except Exception as exc:  # noqa: BLE001
        logger.exception("SRLM chat stream failed: %s", exc)
        for ev in emitter.fail(str(exc), extensions={"srlm": {"distribution": answer_distribution(samples)}}):
            yield _with_checkpoint(ev, compactor) if ev.get("type") == "done" else ev


def _self_compactor(orchestrator_config: Optional[Dict[str, Any]], carried: Any, history_turns: int) -> Any:
    """SelfCompact for the session context; None keeps the recent-turns window."""
    if history_turns <= 0 or not isinstance(carried, dict):
        return None
    from src.context.self_compact import SelfCompactController
    from src.llm.gateway_llm_client import get_gateway_llm_client

    try:
        return SelfCompactController.for_turn(
            orchestrator_config, llm_client=get_gateway_llm_client(), carried=carried
        )
    except Exception as exc:  # noqa: BLE001 — bad override disables it for this turn only
        logger.warning("SRLM SelfCompact disabled for this turn: %s", exc)
        return None


def _with_host_text(context: List[Dict[str, Any]], host_text: str) -> List[Dict[str, Any]]:
    if not host_text:
        return context
    context = list(context)
    if context and context[0].get("role") == "system" and isinstance(context[0].get("content"), str):
        context[0] = {**context[0], "content": f"{context[0]['content']}\n\n{host_text}"}
    else:
        context.insert(0, {"role": "system", "content": host_text})
    return context


def _overflow_error(samples: List[Sample], compactor: Any) -> str:
    """The provider overflow error when every sample failed on it and a retry is left."""
    if compactor is None or not samples or any(s.ok for s in samples) or not compactor.can_retry():
        return ""
    errors = [str(s.error or "") for s in samples]
    return next((e for e in errors if compactor.is_overflow(e)), "")


def _compacted_context(compactor: Any, messages: List[Dict[str, Any]], anchor: Dict[str, Any]) -> List[Dict[str, Any]]:
    from src.context.self_compact import render_summary_section

    if compactor.new_summary:
        return [{"role": "system", "content": render_summary_section(compactor.summary, compactor.prompts)}]
    out = [m for m in messages if m is not anchor]
    if out and out[-1].get("role") == "user" and out[-1].get("content") == anchor["content"]:
        out = out[:-1]
    return out


def _with_checkpoint(done: Dict[str, Any], compactor: Any) -> Dict[str, Any]:
    if compactor is not None:
        done["self_compact"] = compactor.done_payload()
    return done
