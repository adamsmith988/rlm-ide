"""SelfReflect controller: search (sample K answers) then select (summarize).

Mirrors ``generate_summaries.py`` of the reference repo for one question:

1. ``search`` — for strategies conditioned on the answer distribution
   (``sample_and_summarize``, ``concatenated_answers``), draw K answers with the
   plain question prompt at the conditioning temperature
   (``sample_and_summarize_generation``). A sampler may instead run a whole
   chat engine per sample (peer engines with tools).
2. ``select`` — build the strategy prompt and generate the summary at the
   summary temperature (thinking sampling when enabled for eligible
   strategies), then apply the strategy's post-processing.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional

from ..._shared.llm_roles import LLMCall
from . import strategies as S

ProgressFn = Callable[[Dict[str, Any]], None]


@dataclass
class SrlmSettings:
    num_samples: int
    sample_temperature: float
    sample_max_tokens: int
    summary_strategy: str
    summary_temperature: float
    summary_max_tokens: int
    enable_thinking_for_summaries: bool
    thinking_sampling: Dict[str, Any]
    max_parallel: int
    history_turns: int
    sample_timeout_sec: int
    peer_engine: str
    peer_config: Dict[str, Any]
    show_samples: bool

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "SrlmSettings":
        missing = [f for f in cls.__dataclass_fields__ if f not in cfg]
        if missing:
            raise ValueError(f"srlm config missing keys (manifest defaults?): {missing}")
        out = cls(**{f: cfg[f] for f in cls.__dataclass_fields__})
        out.summary_strategy = S.validate(out.summary_strategy)
        out.num_samples = max(1, int(out.num_samples))
        out.max_parallel = max(1, int(out.max_parallel))
        out.history_turns = max(0, int(out.history_turns))
        out.peer_engine = str(out.peer_engine or "").strip()
        out.peer_config = dict(out.peer_config or {})
        out.thinking_sampling = dict(out.thinking_sampling or {})
        return out

    @property
    def needs_samples(self) -> bool:
        return self.summary_strategy in S.NEEDS_SAMPLES

    @property
    def thinking_for_summary(self) -> bool:
        return bool(self.enable_thinking_for_summaries) and self.summary_strategy in S.THINKING_ELIGIBLE


@dataclass
class Sample:
    index: int
    text: str = ""
    error: str = ""
    tokens: Optional[int] = None
    tool_runs: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.text.strip())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "ok": self.ok,
            "error": self.error or None,
            "tokens": self.tokens,
            "elapsed_sec": self.elapsed_sec,
        }


Sampler = Callable[[int], Awaitable[Sample]]


@dataclass
class SelectPlan:
    strategy: str
    messages: Optional[List[Dict[str, Any]]] = None
    text: Optional[str] = None  # precomputed (no summary LLM call)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    enable_thinking: bool = False
    sampling: Dict[str, Any] = field(default_factory=dict)
    analysis: str = ""
    tokens: Optional[int] = None


@dataclass
class SrlmResult:
    text: str
    strategy: str
    samples: List[Sample]
    distribution: Dict[str, Any]
    analysis: str = ""
    tokens: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "strategy": self.strategy,
            "samples": [s.to_dict() for s in self.samples],
            "distribution": self.distribution,
            "analysis": self.analysis or None,
            "tokens": self.tokens,
        }


_NORM_WS = re.compile(r"\s+")
_NORM_EDGE = re.compile(r"^[\s\"'`.,;:!?()\[\]]+|[\s\"'`.,;:!?()\[\]]+$")


def _normalize_answer(text: str) -> str:
    return _NORM_EDGE.sub("", _NORM_WS.sub(" ", str(text or "").strip().lower()))


def answer_distribution(samples: List[Sample]) -> Dict[str, Any]:
    """Empirical distribution over (normalized) sampled answers."""
    ok = [s for s in samples if s.ok]
    counts = Counter(_normalize_answer(s.text) for s in ok)
    n = len(ok)
    support = [
        {"answer": a, "count": c, "share": round(c / n, 4)}
        for a, c in counts.most_common()
    ] if n else []
    return {
        "n_samples": len(samples),
        "n_ok": n,
        "n_failed": len(samples) - n,
        "n_distinct": len(counts),
        "top_share": support[0]["share"] if support else None,
        "support": support,
    }


def _sum_tokens(values: List[Optional[int]]) -> Optional[int]:
    known = [int(v) for v in values if isinstance(v, (int, float))]
    return sum(known) if known else None


class SelfReflectController:
    def __init__(
        self,
        settings: SrlmSettings,
        llm: LLMCall,
        *,
        on_progress: Optional[ProgressFn] = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self._on_progress = on_progress

    def _progress(self, **payload: Any) -> None:
        if self._on_progress is not None:
            self._on_progress(payload)

    # --------------------------------------------------------------- context
    def context_messages(self, history: Optional[List[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
        turns = int(self.settings.history_turns)
        if turns <= 0 or not history:
            return []
        msgs = [
            {"role": str(m.get("role")), "content": str(m.get("content") or "")}
            for m in history
            if isinstance(m, Mapping)
            and m.get("role") in ("user", "assistant")
            and str(m.get("content") or "").strip()
        ]
        return msgs[-2 * turns :]

    def direct_sampler(self, question: str, context: List[Dict[str, Any]]) -> Sampler:
        """``sample_and_summarize_generation``: plain question prompt, conditioning temperature."""
        st = self.settings
        messages = list(context) + [{"role": "user", "content": S.greedy_prompt(question)}]

        async def sample(index: int) -> Sample:
            started = time.monotonic()
            reply = await self.llm(
                messages,
                temperature=st.sample_temperature,
                max_tokens=st.sample_max_tokens or None,
                enable_thinking=False,
            )
            return Sample(
                index=index,
                text=reply.text if reply.ok else "",
                error="" if reply.ok else reply.error,
                tokens=reply.tokens,
                elapsed_sec=round(time.monotonic() - started, 3),
            )

        return sample

    # ---------------------------------------------------------------- search
    async def search(self, sampler: Sampler) -> List[Sample]:
        st = self.settings
        sem = asyncio.Semaphore(st.max_parallel)
        done = 0

        async def one(i: int) -> Sample:
            nonlocal done
            async with sem:
                try:
                    if st.sample_timeout_sec and st.sample_timeout_sec > 0:
                        out = await asyncio.wait_for(sampler(i), timeout=st.sample_timeout_sec)
                    else:
                        out = await sampler(i)
                except asyncio.TimeoutError:
                    out = Sample(index=i, error=f"timeout after {st.sample_timeout_sec}s")
                except Exception as exc:  # noqa: BLE001 — a failed sample is missing, not fatal
                    out = Sample(index=i, error=f"{type(exc).__name__}: {exc}")
            done += 1
            self._progress(
                phase="search",
                completed=done,
                total=st.num_samples,
                sample=out.to_dict(),
            )
            return out

        self._progress(phase="search", completed=0, total=st.num_samples)
        results = await asyncio.gather(*(one(i) for i in range(st.num_samples)))
        return sorted(results, key=lambda s: s.index)

    # ---------------------------------------------------------------- select
    async def plan_select(
        self,
        question: str,
        context: List[Dict[str, Any]],
        samples: List[Sample],
    ) -> SelectPlan:
        st = self.settings
        strategy = st.summary_strategy
        answers = [s.text for s in samples if s.ok]
        if strategy in S.NEEDS_SAMPLES and not answers:
            raise RuntimeError("no sampled answer succeeded; cannot summarize the distribution")
        plan = SelectPlan(strategy=strategy, max_tokens=st.summary_max_tokens or None)
        if strategy in S.NO_LLM:
            plan.text = S.concatenated_answers(answers)
            return plan
        plan.enable_thinking = st.thinking_for_summary
        if plan.enable_thinking:
            sampling = dict(st.thinking_sampling)
            plan.temperature = float(sampling.pop("temperature", st.summary_temperature))
            plan.sampling = sampling
        else:
            plan.temperature = st.summary_temperature
        if strategy == S.TWOSTEP_COT:
            first = list(context) + [{"role": "user", "content": S.cot_analysis_prompt(question)}]
            reply = await self.llm(
                first,
                temperature=st.summary_temperature,
                max_tokens=st.summary_max_tokens or None,
                enable_thinking=False,
            )
            if not reply.ok:
                raise RuntimeError(f"two-step analysis failed: {reply.error}")
            plan.analysis = reply.text
            plan.tokens = reply.tokens
            plan.messages = first + [
                {"role": "assistant", "content": reply.text},
                {"role": "user", "content": S.twostep_summary_prompt()},
            ]
            return plan
        plan.messages = list(context) + [
            {"role": "user", "content": S.summary_user_prompt(strategy, question, answers)}
        ]
        return plan

    def finalize(self, plan: SelectPlan, raw_text: str) -> str:
        return S.postprocess(plan.strategy, raw_text, thinking_enabled=plan.enable_thinking)

    async def select(
        self,
        question: str,
        context: List[Dict[str, Any]],
        samples: List[Sample],
    ) -> tuple[str, SelectPlan, Optional[int]]:
        plan = await self.plan_select(question, context, samples)
        self._progress(phase="select", strategy=plan.strategy)
        if plan.text is not None:
            return plan.text, plan, plan.tokens
        reply = await self.llm(
            plan.messages or [],
            temperature=plan.temperature,
            max_tokens=plan.max_tokens,
            enable_thinking=plan.enable_thinking,
            sampling=plan.sampling or None,
        )
        if not reply.ok:
            raise RuntimeError(f"summary generation failed: {reply.error}")
        return self.finalize(plan, reply.text), plan, _sum_tokens([plan.tokens, reply.tokens])

    # ------------------------------------------------------------------- run
    async def run(
        self,
        question: str,
        history: Optional[List[Mapping[str, Any]]] = None,
        sampler: Optional[Sampler] = None,
    ) -> SrlmResult:
        context = self.context_messages(history)
        samples: List[Sample] = []
        if self.settings.needs_samples:
            samples = await self.search(sampler or self.direct_sampler(question, context))
        text, plan, select_tokens = await self.select(question, context, samples)
        return SrlmResult(
            text=text,
            strategy=plan.strategy,
            samples=samples,
            distribution=answer_distribution(samples),
            analysis=plan.analysis,
            tokens=_sum_tokens([s.tokens for s in samples] + [select_tokens]),
        )
