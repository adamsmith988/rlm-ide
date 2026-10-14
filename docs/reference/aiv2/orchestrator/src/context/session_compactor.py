"""Session compaction — one host-side decision for every chat engine.

Before a turn is handed to any engine the host measures the conversation that
engine will receive (carried checkpoint + stored turns since it + the new user
message) against the serving model, as the LLM catalog describes it:
``context_length`` and ``max_output_tokens``, stamped onto the orchestrator
config by ``prepare_orchestrator_config``. Past the trigger it compacts the way
the RLM runtime does: the oldest whole turns fold into a structured checkpoint
(updated from the previous checkpoint, never restarted), the newest turns stay
verbatim, and a cut falls only between turns, so a tool call always keeps its
result.

Every engine therefore starts a turn from the same memory, whichever engine
served the earlier turns; the engines' own compactors only guard growth inside
the turn (``owner`` on the carried trajectory). The checkpoint is stored on
the turn's assistant row (``metadata.self_compact``) and
:func:`~src.context.self_compact.split_trajectory` rebuilds it next turn.

Knobs: ``settings.SESSION_COMPACT_*``, per chat
``orchestrator_config["session_compact_<name>"]``.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..config import settings
from .self_compact import SelfCompactController, SelfCompactSettings, _coerce, _positive_int

log = logging.getLogger(__name__)

CONFIG_PREFIX = "session_compact_"
OWNER = "session"
_BUNDLED_SUMMARIZER = Path(__file__).with_name("session_compact_prompts") / "summarizer.md"
_PAYLOAD_KEYS = ("summary", "kept_turns", "kept_from_id", "compacted_by")


@dataclass(frozen=True)
class SessionCompactSettings:
    enabled: bool
    history_share: float
    trigger_ratio: float
    keep_recent_ratio: float
    max_output_ratio: float
    max_rows_ratio: float
    summarizer_file: str

    @classmethod
    def resolve(cls, overrides: Optional[Mapping[str, Any]] = None) -> "SessionCompactSettings":
        """``settings.SESSION_COMPACT_<NAME>`` overridden by ``session_compact_<name>`` keys."""
        ov = overrides if isinstance(overrides, Mapping) else {}
        values: Dict[str, Any] = {}
        for f in fields(cls):
            base = getattr(settings, f"SESSION_COMPACT_{f.name.upper()}")
            raw = ov.get(f"{CONFIG_PREFIX}{f.name}")
            values[f.name] = _coerce(raw, base) if raw not in (None, "") else base
        return cls(**values)


@dataclass(frozen=True)
class Budget:
    """Token budget of one turn, from the serving model's catalog window."""

    window: int
    output_reserve: int
    history: int
    trigger: int
    keep: int


def budget_for(orchestrator_config: Optional[Mapping[str, Any]], st: SessionCompactSettings) -> Budget:
    oc = orchestrator_config if isinstance(orchestrator_config, Mapping) else {}
    window = _positive_int(oc.get("context_length")) or _positive_int(oc.get("max_model_len"))
    if not window:
        window = _positive_int(settings.SELF_COMPACT_FALLBACK_CONTEXT_TOKENS) or int(
            settings.STREAM_CONTEXT_CHAR_BUDGET / settings.SELF_COMPACT_CHARS_PER_TOKEN
        )
    reserve = _positive_int(oc.get("max_output_tokens")) or _positive_int(settings.SELF_COMPACT_OUTPUT_RESERVE_TOKENS)
    reserve = min(reserve, int(window * st.max_output_ratio))
    history = max(0, int((window - reserve) * st.history_share))
    return Budget(
        window=window,
        output_reserve=reserve,
        history=history,
        trigger=int(history * st.trigger_ratio),
        keep=int(history * st.keep_recent_ratio),
    )


def turn_spans(rows: Sequence[Mapping[str, Any]]) -> List[Tuple[int, int]]:
    """``(start, end)`` row ranges, one per turn (a user row and what follows it).
    Rows before the first user row form a leading span of their own."""
    starts = [i for i, r in enumerate(rows) if r.get("role") == "user"]
    if not starts:
        return [(0, len(rows))] if rows else []
    spans: List[Tuple[int, int]] = [(0, starts[0])] if starts[0] > 0 else []
    for n, s in enumerate(starts):
        spans.append((s, starts[n + 1] if n + 1 < len(starts) else len(rows)))
    return spans


def plan_cut(turn_tokens: Sequence[int], keep: int, newest_limit: int = 0) -> int:
    """How many leading turns fold into the checkpoint.

    The newest turns stay verbatim while they fit ``keep``; the newest turn
    alone also stays when it fits ``newest_limit``, so a follow-up still sees
    the exchange it answers. At least one turn folds (the caller already
    decided the history is too large)."""
    n = len(turn_tokens)
    kept, used = 0, 0
    while kept < n and used + int(turn_tokens[n - kept - 1]) <= keep:
        used += int(turn_tokens[n - kept - 1])
        kept += 1
    if not kept and n > 1 and newest_limit and int(turn_tokens[-1]) <= newest_limit:
        kept = 1
    return max(1, n - kept) if n else 0


@lru_cache(maxsize=8)
def _summarizer_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def summarizer_path(st: SessionCompactSettings) -> str:
    """The checkpoint prompt file; engines compacting inside the turn use it too."""
    return st.summarizer_file or str(_BUNDLED_SUMMARIZER)


def checkpoint_prompt(path: str) -> str:
    return _summarizer_text(path)


@dataclass
class SessionCompaction:
    summary: str
    kept_rows: List[Dict[str, Any]]
    kept_turns: int
    kept_from_id: str
    trigger: str
    tokens_before: int
    tokens_after: int
    budget: Budget
    compactions: int
    token_scale: float

    def event(self) -> Dict[str, Any]:
        return {
            "type": "context_compacted",
            "step": 0,
            "trigger": self.trigger,
            "compacted_by": OWNER,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "context_window": self.budget.window,
            "max_output_tokens": self.budget.output_reserve,
            "kept_turns": self.kept_turns,
            "summary_chars": len(self.summary),
        }

    def payload(self) -> Dict[str, Any]:
        """What the turn's assistant row stores (``metadata.self_compact``)."""
        ev = {k: v for k, v in self.event().items() if k != "type"}
        out: Dict[str, Any] = {
            "summary": self.summary,
            "kept_turns": self.kept_turns,
            "compacted_by": OWNER,
            "compactions": self.compactions,
            "context_window": self.budget.window,
            "token_scale": round(self.token_scale, 4),
            "events": [ev],
        }
        if self.kept_from_id:
            out["kept_from_id"] = self.kept_from_id
        return out


async def compact_session(
    trajectory: Mapping[str, Any],
    rows: List[Dict[str, Any]],
    *,
    user_message: str,
    orchestrator_config: Optional[Mapping[str, Any]],
    llm_client: Any,
    expand: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
    rows_limit: int = 0,
) -> Optional[SessionCompaction]:
    """Compact ``rows`` (stored user/assistant rows since the carried checkpoint)
    when they no longer fit this turn's budget; ``None`` when they do.

    ``expand`` turns stored rows into the chat messages engines receive (tool
    calls with their results), so the estimate measures what is actually sent.
    """
    st = SessionCompactSettings.resolve(orchestrator_config)
    if not st.enabled or not rows:
        return None
    budget = budget_for(orchestrator_config, st)
    state = trajectory.get("state") if isinstance(trajectory.get("state"), Mapping) else {}
    carried = str(trajectory.get("summary") or "")
    ctrl = SelfCompactController(
        SelfCompactSettings.resolve(orchestrator_config),
        llm_client=llm_client,
        context_tokens=budget.window,
        output_reserve=budget.output_reserve,
        carried={"summary": carried, "state": state},
    )
    ctrl.prompts = dataclasses.replace(ctrl.prompts, summarizer=_summarizer_text(summarizer_path(st)))

    spans = turn_spans(rows)
    turns = [expand(rows[s:e]) for s, e in spans]
    turn_tokens = [ctrl.est.tokens(t) if t else 0 for t in turns]
    fixed = ctrl.est.tokens([{"role": "user", "content": str(user_message or "")}])
    if carried:
        fixed += ctrl.est.tokens([{"role": "system", "content": carried}])
    before = sum(turn_tokens) + fixed

    trigger = ""
    if before >= budget.trigger:
        trigger = "window"
    elif rows_limit and len(rows) >= int(rows_limit * st.max_rows_ratio):
        trigger = "rows"
    if not trigger:
        return None

    fold = plan_cut(turn_tokens, budget.keep, newest_limit=budget.history - fixed)
    head = [m for t in turns[:fold] for m in t]
    if not head:
        return None
    summary = await ctrl.summarize(head, None, None)
    if not summary:
        log.warning("session compaction: summarizer returned nothing; keeping the history as is")
        return None

    kept_rows = rows[spans[fold][0]:] if fold < len(spans) else []
    after = sum(turn_tokens[fold:]) + ctrl.est.tokens([{"role": "system", "content": summary}])
    after += ctrl.est.tokens([{"role": "user", "content": str(user_message or "")}])
    result = SessionCompaction(
        summary=summary,
        kept_rows=kept_rows,
        kept_turns=sum(1 for r in kept_rows if r.get("role") == "user"),
        kept_from_id=str(kept_rows[0].get("id") or "") if kept_rows else "",
        trigger=trigger,
        tokens_before=before,
        tokens_after=after,
        budget=budget,
        compactions=int(state.get("compactions") or 0) + 1,
        token_scale=ctrl.est.scale,
    )
    log.info(
        "session compaction %s: tokens %s→%s window=%s out=%s folded_turns=%s kept_turns=%s summary_chars=%s",
        trigger, before, after, budget.window, budget.output_reserve, fold, result.kept_turns, len(summary),
    )
    return result


def merge_turn_payload(
    session: Optional[Mapping[str, Any]],
    engine: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Combine this turn's session checkpoint with the engine's ``done.self_compact``.

    An engine summary written later in the same turn already covers the
    session checkpoint (the engine carried it), so it wins; otherwise the
    session checkpoint is what the next turn rebuilds from.
    """
    if not session:
        return dict(engine) if engine else None
    if not engine:
        return dict(session)
    events = list(session.get("events") or []) + list(engine.get("events") or [])
    if engine.get("summary"):
        out = dict(engine)
    else:
        out = {**engine, **{k: session[k] for k in _PAYLOAD_KEYS if k in session}}
    out["compactions"] = max(int(session.get("compactions") or 0), int(engine.get("compactions") or 0))
    if events:
        out["events"] = events
    return out


__all__ = [
    "Budget",
    "OWNER",
    "SessionCompactSettings",
    "SessionCompaction",
    "budget_for",
    "compact_session",
    "merge_turn_payload",
    "plan_cut",
    "turn_spans",
]
