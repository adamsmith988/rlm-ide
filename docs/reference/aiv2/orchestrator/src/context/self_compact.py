"""SelfCompact — rubric-gated context compaction for chat trajectories.

Implements Li et al., "Self-Compacting Language Model Agents" (arXiv 2606.23525):
at round boundaries the same model judges a rubric appended to a *copy* of the
trajectory; only a COMPRESS verdict fires the summarizer, whose output replaces
the trajectory so generation resumes from
``[system, latest_user_message, assistant_summary, user_continue]``.
The paper's gates (ROUND / TOKENS / CAP / PERIOD) and token-ratio backstop
apply, plus two guards a chat service needs so a turn never stops: a pre-send
window guard and compact-and-retry on provider context-overflow errors.

Prompts and fire rules live in ``self_compact_prompts/<rubric>/``; every knob
comes from ``settings.SELF_COMPACT_*`` and can be overridden per chat with
``orchestrator_config["self_compact_<name>"]``.

Persistence: compaction never deletes stored messages. The engine returns a
``self_compact`` dict on ``done`` that the host stores in the assistant row's
metadata; :func:`split_trajectory` uses it to rebuild the next turn's context
as ``summary + messages from the compacting turn's user message onward``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import dataclasses
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..config import settings

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).with_name("self_compact_prompts")
CONFIG_PREFIX = "self_compact_"
_RUBRIC_NAME = re.compile(r"^[a-z0-9_-]+$")
_INPUT_TOKENS_RE = re.compile(r"(\d+)\s+(?:input|prompt)\s+tokens", re.I)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)

COMPRESS = "COMPRESS"
CONTINUE = "CONTINUE"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelfCompactSettings:
    enabled: bool
    rubric: str
    probe_period: int
    min_rounds: int
    probe_min_ratio: float
    max_rubric_per_turn: int
    backstop_ratio: float
    backstop_min_trajectory_ratio: float
    hard_ratio: float
    max_reserve_ratio: float
    output_reserve_tokens: int
    fallback_context_tokens: int
    chars_per_token: float
    image_tokens: int
    max_token_scale: float
    scale_backoff: float
    probe_max_tokens: int
    summary_max_tokens: int
    summary_chunk_ratio: float
    summary_parallel: int
    summary_head_max_ratio: float
    keep_carried_values: bool
    cap_completion: bool
    temperature: float
    judge_thinking: bool
    max_overflow_retries: int
    history_limit: int

    @classmethod
    def resolve(cls, overrides: Optional[Mapping[str, Any]] = None) -> "SelfCompactSettings":
        """``settings.SELF_COMPACT_<NAME>`` overridden by ``self_compact_<name>`` keys."""
        ov = overrides if isinstance(overrides, Mapping) else {}
        values: Dict[str, Any] = {}
        for f in fields(cls):
            base = getattr(settings, f"SELF_COMPACT_{f.name.upper()}")
            raw = ov.get(f"{CONFIG_PREFIX}{f.name}")
            values[f.name] = _coerce(raw, base) if raw not in (None, "") else base
        st = cls(**values)
        if not _RUBRIC_NAME.match(st.rubric) or not (PROMPTS_DIR / st.rubric).is_dir():
            raise ValueError(f"unknown SelfCompact rubric {st.rubric!r}")
        return st


def _coerce(raw: Any, like: Any) -> Any:
    if isinstance(like, bool):
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)
    if isinstance(like, int):
        return int(raw)
    if isinstance(like, float):
        return float(raw)
    return str(raw)


# ---------------------------------------------------------------------------
# Prompts and verdict parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptSet:
    rubric: str
    summarizer: str
    continue_prompt: str
    labels: Tuple[str, ...]
    fire_any: Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...]
    summary_section: str
    carried_values: str = ""


@lru_cache(maxsize=16)
def load_prompts(name: str) -> PromptSet:
    base = PROMPTS_DIR / name
    rule = json.loads((base / "rule.json").read_text(encoding="utf-8"))
    clauses = tuple(
        (tuple(c.get("yes") or ()), tuple(c.get("no") or ()))
        for c in rule.get("fire_any") or ()
    )
    if not clauses:
        raise ValueError(f"SelfCompact rubric {name!r} has no fire rule")
    return PromptSet(
        rubric=(base / "rubric.md").read_text(encoding="utf-8").strip(),
        summarizer=(base / "summarizer.md").read_text(encoding="utf-8").strip(),
        continue_prompt=(base / "continue.md").read_text(encoding="utf-8").strip(),
        labels=tuple(rule.get("labels") or ()),
        fire_any=clauses,
        summary_section=(PROMPTS_DIR / "summary_section.md").read_text(encoding="utf-8").strip(),
        carried_values=_optional_prompt(base / "carried_values.md", PROMPTS_DIR / "carried_values.md"),
    )


def _optional_prompt(*paths: Path) -> str:
    for path in paths:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return ""


_VALUE_TOKEN = re.compile(r"[^\s,;:()\[\]\"'`]*\d[^\s,;:()\[\]\"'`]*")


def value_tokens(line: str) -> List[str]:
    """Tokens that carry a value: anything containing a digit (codes, ids, amounts, dates)."""
    return [t.strip(".") for t in _VALUE_TOKEN.findall(str(line or "")) if t.strip(".")]


def dropped_value_lines(previous: str, current: str) -> List[str]:
    """Lines of ``previous`` none of whose value tokens survive anywhere in ``current``."""
    lost: List[str] = []
    for line in str(previous or "").splitlines():
        tokens = value_tokens(line)
        if tokens and not any(t in current for t in tokens):
            lost.append(line.strip())
    return lost


@dataclass
class Verdict:
    decision: str
    answers: Dict[str, bool]
    evidence: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "answers": {k: ("Y" if v else "N") for k, v in self.answers.items()},
        }


def parse_verdict(text: str, prompts: PromptSet) -> Verdict:
    """Apply the rubric's fire rule. A Y without evidence counts as N."""
    body = _THINK_BLOCK.sub("", str(text or ""))
    answers: Dict[str, bool] = {}
    evidence: Dict[str, str] = {}
    for label in prompts.labels:
        m = re.search(
            rf"^[ \t>*#`_-]*{re.escape(label)}\b[^:\n]*:[ \t]*\**[ \t]*(YES|NO|Y|N)\b\**[ \t]*(?:--|—|–|-|:)?[ \t]*(.*)$",
            body,
            re.I | re.M,
        )
        if not m:
            answers[label] = False
            continue
        ev = m.group(2).strip().strip("*").strip()
        evidence[label] = ev
        answers[label] = m.group(1).upper().startswith("Y") and bool(ev)
    fire = any(
        all(answers.get(lbl, False) for lbl in yes) and not any(answers.get(lbl, False) for lbl in no)
        for yes, no in prompts.fire_any
    )
    return Verdict(COMPRESS if fire else CONTINUE, answers, evidence)


def clean_model_text(text: str) -> str:
    return _THINK_BLOCK.sub("", str(text or "")).strip()


def render_summary_section(summary: str, prompts: PromptSet) -> str:
    return prompts.summary_section.replace("{summary}", summary.strip())


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TokenEstimator:
    """Char-based estimate, rescaled whenever the provider reports real counts."""

    def __init__(self, chars_per_token: float, image_tokens: int, max_scale: float = 4.0) -> None:
        self.chars_per_token = max(0.5, float(chars_per_token))
        self.image_tokens = max(0, int(image_tokens))
        self.max_scale = max(1.0, float(max_scale))
        self.scale = 1.0

    def set_scale(self, value: Any) -> None:
        try:
            self.scale = min(self.max_scale, max(1.0 / self.max_scale, float(value)))
        except (TypeError, ValueError):
            pass

    def raw(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> float:
        chars = 0
        images = 0
        for m in messages or []:
            if not isinstance(m, dict):
                continue
            c = m.get("content")
            if isinstance(c, str):
                chars += len(c)
            elif isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        chars += len(str(part.get("text") or ""))
                    elif isinstance(part, dict):
                        images += 1
                    else:
                        chars += len(str(part))
            elif c is not None:
                chars += len(str(c))
            if m.get("tool_calls"):
                chars += len(json.dumps(m["tool_calls"], ensure_ascii=False, default=str))
        if tools:
            chars += len(json.dumps(tools, ensure_ascii=False, default=str))
        return chars / self.chars_per_token + images * self.image_tokens

    def tokens(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> int:
        return int(self.raw(messages, tools) * self.scale)

    def calibrate(self, observed_tokens: Any, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> None:
        try:
            observed = float(observed_tokens)
        except (TypeError, ValueError):
            return
        raw = self.raw(messages, tools)
        if observed > 0 and raw > 0:
            self.set_scale(observed / raw)


# ---------------------------------------------------------------------------
# Trajectory persistence helpers
# ---------------------------------------------------------------------------


def _meta(msg: Mapping[str, Any]) -> Dict[str, Any]:
    meta = msg.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except ValueError:
            return {}
    return meta if isinstance(meta, dict) else {}


def _kept_tail_start(rows: List[Dict[str, Any]], turn_start: int, sc: Mapping[str, Any]) -> int:
    """First row of the verbatim tail a checkpoint kept before its compacting turn."""
    kept_id = str(sc.get("kept_from_id") or "")
    if kept_id:
        for k in range(turn_start - 1, -1, -1):
            if str(rows[k].get("id") or "") == kept_id:
                return k
    want = _positive_int(sc.get("kept_turns"))
    start = turn_start
    k = turn_start - 1
    while want and k >= 0:
        if rows[k].get("role") == "user":
            start, want = k, want - 1
        k -= 1
    return start


def split_trajectory(raw_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Locate the latest checkpoint in stored rows (oldest first).

    Returns ``summary`` (str), ``messages`` (rows from the compacting turn's
    user message onward, or all rows when there is no checkpoint), ``state``
    (latest persisted controller state) and ``rounds`` (assistant rows in the
    trajectory). A checkpoint written by session compaction also kept the turns
    before the compacting one verbatim (``kept_from_id`` / ``kept_turns``); the
    trajectory then starts at the first of them.
    """
    rows = [m for m in (raw_messages or []) if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    state: Optional[Dict[str, Any]] = None
    summary = ""
    start = 0
    for i in range(len(rows) - 1, -1, -1):
        if rows[i].get("role") != "assistant":
            continue
        sc = _meta(rows[i]).get("self_compact")
        if not isinstance(sc, dict):
            continue
        if state is None:
            state = sc
        if isinstance(sc.get("summary"), str) and sc["summary"].strip():
            summary = sc["summary"].strip()
            j = i - 1
            while j >= 0 and rows[j].get("role") != "user":
                j -= 1
            start = _kept_tail_start(rows, j if j >= 0 else i, sc)
            break
    trajectory = rows[start:]
    return {
        "summary": summary,
        "messages": trajectory,
        "state": dict(state or {}),
        "rounds": sum(1 for m in trajectory if m.get("role") == "assistant"),
    }


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _overflow_checks():
    """Provider overflow detection shared with RLM (stdlib-only module, loaded
    by path so the RLM package and its runtime deps are not imported)."""
    try:
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "orchestrators" / "rlm" / "runtime" / "context_budget.py"
        spec = importlib.util.spec_from_file_location("_self_compact_context_budget", path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod.looks_like_context_overflow, mod.parse_context_overflow_error
    except Exception:  # noqa: BLE001 — degrade to substring detection
        needles = ("context length", "maximum context", "max_model_len", "too many tokens", "prompt is too long")

        def _looks(text: str) -> bool:
            low = str(text or "").lower()
            return any(n in low for n in needles)

        return _looks, (lambda _text: None)


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(content or "")


def _shorten(text: str, keep_chars: int, marker: str) -> str:
    if len(text) <= keep_chars:
        return text
    head = max(0, keep_chars * 2 // 3)
    tail = max(0, keep_chars - head)
    return f"{text[:head]}\n{marker}\n{text[len(text) - tail:]}" if tail else f"{text[:head]}\n{marker}"


class SelfCompactController:
    """Per-turn SelfCompact state for one chat engine run."""

    def __init__(
        self,
        cfg: SelfCompactSettings,
        *,
        llm_client: Any,
        context_tokens: int,
        output_reserve: int,
        carried: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.cfg = cfg
        self.prompts = load_prompts(cfg.rubric)
        self.llm = llm_client
        self.context_tokens = int(context_tokens)
        self.output_reserve = int(output_reserve)
        self.est = TokenEstimator(cfg.chars_per_token, cfg.image_tokens, cfg.max_token_scale)
        carried = carried if isinstance(carried, Mapping) else {}
        state = carried.get("state") if isinstance(carried.get("state"), Mapping) else {}
        self.summary: str = str(carried.get("summary") or "")
        # Set when the host already sized the history for this turn (session
        # compaction): the first round then only gets the hard window guard.
        self.history_owner: str = str(carried.get("owner") or "")
        # The owner's checkpoint prompt: compaction inside the turn writes the same checkpoint.
        owner_summarizer = str(carried.get("owner_summarizer") or "")
        if self.history_owner and owner_summarizer:
            from .session_compactor import checkpoint_prompt

            try:
                self.prompts = dataclasses.replace(self.prompts, summarizer=checkpoint_prompt(owner_summarizer))
            except OSError:
                log.warning("self_compact: owner summarizer %s unreadable; keeping the rubric's", owner_summarizer)
        self.trajectory_messages: int = len(carried.get("messages") or [])
        self.rounds_since_compact: int = int(carried.get("rounds") or 0)
        if "rounds_since_probe" in state:
            self.rounds_since_probe = int(state.get("rounds_since_probe") or 0) + 1
        else:
            self.rounds_since_probe = self.rounds_since_compact
        self.compactions_total: int = int(state.get("compactions") or 0)
        self.est.set_scale(state.get("token_scale") or 1.0)
        self.rubric_fired_this_turn = 0
        self.overflow_retries = 0
        self.new_summary = False
        self.events: List[Dict[str, Any]] = []
        self._section_msg: Optional[Dict[str, Any]] = None
        self._anchor_msg: Optional[Dict[str, Any]] = None
        self._last_step = 0
        self._volatile: List[str] = []
        self._looks_overflow, self._parse_overflow = _overflow_checks()

    # ---- construction -------------------------------------------------

    @classmethod
    def for_turn(
        cls,
        orchestrator_config: Optional[Mapping[str, Any]],
        *,
        llm_client: Any,
        carried: Optional[Mapping[str, Any]] = None,
    ) -> Optional["SelfCompactController"]:
        cfg = SelfCompactSettings.resolve(orchestrator_config)
        if not cfg.enabled:
            return None
        oc = orchestrator_config if isinstance(orchestrator_config, Mapping) else {}
        ctx = _positive_int(oc.get("context_length")) or _positive_int(oc.get("max_model_len"))
        if not ctx:
            ctx = cfg.fallback_context_tokens or int(settings.STREAM_CONTEXT_CHAR_BUDGET / cfg.chars_per_token)
        reserve = (
            _positive_int(oc.get("max_output_tokens"))
            or _positive_int(getattr(llm_client, "max_tokens", None))
            or cfg.output_reserve_tokens
        )
        reserve = min(reserve, int(ctx * cfg.max_reserve_ratio))
        return cls(cfg, llm_client=llm_client, context_tokens=ctx, output_reserve=reserve, carried=carried)

    # ---- message helpers ----------------------------------------------

    def attach_summary(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Insert the carried summary as a system section after the leading system run."""
        if not self.summary:
            return messages
        i = 0
        while i < len(messages) and messages[i].get("role") == "system":
            i += 1
        self._section_msg = {"role": "system", "content": render_summary_section(self.summary, self.prompts)}
        return messages[:i] + [self._section_msg] + messages[i:]

    def exclude_from_summary(self, *texts: str) -> None:
        """System-prompt sections rebuilt every turn (recalled memory, retrieved
        passages): the summarizer must not fold them into the conversation."""
        self._volatile.extend(t.strip() for t in texts if t and t.strip())

    def _without_volatile(self, view: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self._volatile:
            return view
        out: List[Dict[str, Any]] = []
        for m in view:
            content = m.get("content")
            if m.get("role") == "system" and isinstance(content, str):
                for text in self._volatile:
                    content = content.replace(text, "")
                content = re.sub(r"\n{3,}", "\n\n", content).strip()
                if not content:
                    continue
                m = {**m, "content": content}
            out.append(m)
        return out

    def _anchor(self, messages: List[Dict[str, Any]], anchor: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """The turn's user message: kept verbatim through every compaction."""
        for cand in (self._anchor_msg, anchor):
            if cand is not None and any(m is cand for m in messages):
                self._anchor_msg = cand
                return cand
        for m in reversed(messages):
            if m.get("role") == "user":
                self._anchor_msg = m
                return m
        return None

    @property
    def hard_limit(self) -> int:
        return int(self.context_tokens * self.cfg.hard_ratio) - self.output_reserve

    def completion_budget(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> Optional[int]:
        """max_tokens for the next round: what is left of the window (the provider
        may serve a larger one than configured), never below the output reserve."""
        if not self.cfg.cap_completion:
            return None
        room = int(self.context_tokens * self.cfg.hard_ratio) - self.est.tokens(messages, tools)
        budget = max(self.output_reserve, room)
        client_cap = _positive_int(getattr(self.llm, "max_tokens", None))
        return min(budget, client_cap) if client_cap else budget

    def usage(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        tok = self.est.tokens(messages, tools)
        return {
            "context_tokens": tok,
            "context_window": self.context_tokens,
            "context_usage_pct": round(100.0 * tok / max(1, self.context_tokens), 1),
        }

    def _judge_kwargs(self, tools: Optional[List[Dict[str, Any]]], max_tokens: int) -> Dict[str, Any]:
        kw: Dict[str, Any] = {
            "temperature": self.cfg.temperature,
            "max_tokens": max_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": bool(self.cfg.judge_thinking)}},
        }
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = "none"
        return kw

    # ---- rubric probe and summarizer ----------------------------------

    def _judge_view(self, messages: List[Dict[str, Any]], anchor: Optional[Dict[str, Any]], prompt: str) -> List[Dict[str, Any]]:
        """Trajectory + appended instruction. A still-unanswered turn message at
        the end is left out: it is kept verbatim, and the model would answer it
        instead of following the instruction."""
        body = messages[:-1] if messages and anchor is not None and messages[-1] is anchor else messages
        return body + [{"role": "user", "content": prompt}]

    async def probe(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        anchor: Optional[Dict[str, Any]] = None,
    ) -> Optional[Verdict]:
        view = self._judge_view(messages, anchor, self.prompts.rubric)
        error = ""
        try:
            res = await self.llm.chat(messages=view, **self._judge_kwargs(tools, self.cfg.probe_max_tokens))
        except Exception as exc:  # noqa: BLE001 — a failed probe means CONTINUE
            res, error = None, str(exc)
        if not isinstance(res, dict) or not res.get("ok"):
            error = error or str((res or {}).get("error") if isinstance(res, dict) else res)
            self.learn_from_error(error, view, tools, self.cfg.probe_max_tokens)
            log.warning("self_compact probe failed: %s", error[:300])
            return None
        self.est.calibrate(res.get("tokens_prompt"), view, tools)
        return parse_verdict(str(res.get("content") or ""), self.prompts)

    def learn_from_error(
        self,
        error_text: str,
        view: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_output: int = 0,
    ) -> bool:
        """Take the real window / prompt size from a provider overflow error.

        vLLM stops counting at ``max_model_len - max_tokens + 1`` and says
        "at least N"; such a count is only a floor, so the scale backs off
        beyond it instead of settling on it.
        """
        text = str(error_text or "")
        if not self.is_overflow(text):
            return False
        info = self._parse_overflow(text) or {}
        learned = _positive_int(info.get("context_length"))
        if learned and learned < self.context_tokens:
            self.context_tokens = learned
            self.output_reserve = min(self.output_reserve, int(learned * self.cfg.max_reserve_ratio))
        observed = _positive_int(info.get("input_tokens")) or _positive_int(info.get("requested"))
        if not observed:
            m = _INPUT_TOKENS_RE.search(text)
            observed = _positive_int(m.group(1)) if m else 0
        previous = self.est.scale
        if observed:
            self.est.calibrate(observed, view, tools)
        lower_bound = not observed or bool(
            re.search(rf"at\s+least\s+{observed}\b", text, re.I)
            or (max_output and observed >= self.context_tokens - max_output)
        )
        if lower_bound:
            self.est.set_scale(max(self.est.scale, previous) * self.cfg.scale_backoff)
        return True

    async def summarize(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]], anchor: Optional[Dict[str, Any]]) -> str:
        summary = await self._summarize_raw(messages, tools, anchor)
        return self._keep_carried_values(summary) if summary else summary

    def _keep_carried_values(self, summary: str) -> str:
        """Re-attach carried-summary lines whose values the new summary lost."""
        template = self.prompts.carried_values
        if not (self.cfg.keep_carried_values and template and self.summary):
            return summary
        lost = dropped_value_lines(self.summary, summary)
        if not lost:
            return summary
        log.info("self_compact re-attached %d carried value line(s) the summarizer dropped", len(lost))
        return f"{summary.rstrip()}\n\n" + template.replace("{lines}", "\n".join(lost))

    async def _summarize_raw(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]], anchor: Optional[Dict[str, Any]]) -> str:
        """One summarizer call, or — when the trajectory is larger than
        ``summary_chunk_ratio`` of the window — consecutive calls over turn
        chunks, each extending the running summary (small models drop facts
        when one call has to digest several long documents)."""
        messages, tools = self._summary_inputs(messages, tools)
        chunks = self._summary_chunks(messages, tools, anchor)
        if len(chunks) <= 1:
            messages = self._with_carried_summary(messages)
            return await self._summarize_view(self._judge_view(messages, anchor, self.prompts.summarizer), tools, anchor)
        lead = next((i for i, m in enumerate(messages) if m.get("role") != "system"), len(messages))
        head = [m for m in messages[:lead] if m is not self._section_msg]
        if self.cfg.summary_parallel > 1:
            return await self._summarize_parallel(head, chunks, tools, anchor)
        running = self.summary
        for n, chunk in enumerate(chunks, 1):
            section = [{"role": "system", "content": render_summary_section(running, self.prompts)}] if running else []
            view = head + section + chunk + [{"role": "user", "content": self.prompts.summarizer}]
            out = await self._summarize_view(view, tools, anchor)
            if not out:
                log.warning("self_compact chunked summary failed at chunk %d/%d", n, len(chunks))
                return ""
            running = out
        return running

    def _with_carried_summary(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """The next summary replaces the carried one, so the summarizer must see it.

        Engines may rebuild their system messages each round, so presence is checked by
        content rather than by the section message object.
        """
        carried = (self.summary or "").strip()
        if not carried:
            return messages
        if any(m.get("role") == "system" and carried in _text_of(m.get("content")) for m in messages):
            return messages
        lead = next((i for i, m in enumerate(messages) if m.get("role") != "system"), len(messages))
        section = {"role": "system", "content": render_summary_section(carried, self.prompts)}
        return messages[:lead] + [section] + messages[lead:]

    def _summary_inputs(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        """Summarizer views normally reuse the turn's system prompt and tool schemas
        (prompt-cache friendly). When those alone take more than
        ``summary_head_max_ratio`` of the window they would crowd out the
        conversation being summarized, so they are left out (the carried summary
        section stays)."""
        ratio = self.cfg.summary_head_max_ratio
        lead = next((i for i, m in enumerate(messages) if m.get("role") != "system"), len(messages))
        head = [m for m in messages[:lead] if m is not self._section_msg]
        if ratio <= 0 or not head or self.est.tokens(head, tools) <= int(self.context_tokens * ratio):
            return messages, tools
        kept = [m for m in messages[:lead] if m is self._section_msg]
        return kept + messages[lead:], None

    async def _summarize_parallel(
        self,
        head: List[Dict[str, Any]],
        chunks: List[List[Dict[str, Any]]],
        tools: Optional[List[Dict[str, Any]]],
        anchor: Optional[Dict[str, Any]],
    ) -> str:
        """Summarize chunks concurrently (bounded), then merge the partial
        summaries — in order, after the carried summary — in one call."""
        sem = asyncio.Semaphore(max(1, self.cfg.summary_parallel))
        ask = [{"role": "user", "content": self.prompts.summarizer}]

        async def one(chunk: List[Dict[str, Any]]) -> str:
            async with sem:
                return await self._summarize_view(head + chunk + ask, tools, anchor)

        partials = await asyncio.gather(*(one(c) for c in chunks))
        failed = [n for n, p in enumerate(partials, 1) if not p]
        if failed:
            log.warning("self_compact parallel summary failed at chunk(s) %s of %d", failed, len(chunks))
            return ""
        sections = [self.summary] if self.summary else []
        sections.extend(partials)
        view = head + [
            {"role": "system", "content": render_summary_section(s, self.prompts)} for s in sections
        ] + ask
        return await self._summarize_view(view, tools, anchor)

    def _summary_chunks(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        anchor: Optional[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """Trajectory (no system run, no trailing anchor) packed into chunks of
        whole turns, each within ``summary_chunk_ratio`` of the window."""
        limit = int(self.context_tokens * self.cfg.summary_chunk_ratio)
        body = messages[:-1] if messages and anchor is not None and messages[-1] is anchor else list(messages)
        start = next((i for i, m in enumerate(body) if m.get("role") != "system"), len(body))
        body = [m for m in body[start:] if m is not self._section_msg]
        if limit <= 0 or not body or self.est.tokens(body, tools) <= limit:
            return [body]
        turns: List[List[Dict[str, Any]]] = []
        for m in body:
            if m.get("role") == "user" or not turns:
                turns.append([m])
            else:
                turns[-1].append(m)
        chunks: List[List[Dict[str, Any]]] = []
        for turn in turns:
            if chunks and self.est.tokens(chunks[-1] + turn, tools) <= limit:
                chunks[-1].extend(turn)
            else:
                chunks.append(list(turn))
        return chunks

    async def _summarize_view(
        self,
        view: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        anchor: Optional[Dict[str, Any]],
    ) -> str:
        view = self._without_volatile(view)
        prior_replies = {
            _text_of(m.get("content")).strip().lower()
            for m in view
            if m.get("role") == "assistant" and _text_of(m.get("content")).strip()
        }
        # Retry without tool schemas when the model echoes a prior reply
        # instead of summarizing (small models follow earlier format rules);
        # refit (recalibrating each time) when the provider reports the view
        # itself overflowed.
        attempts: List[Optional[List[Dict[str, Any]]]] = [tools, None] if tools else [None]
        refits_left = self.cfg.max_overflow_retries
        i = 0
        while i < len(attempts):
            attempt_tools = attempts[i]
            budget = int(self.context_tokens * self.cfg.hard_ratio) - self.cfg.summary_max_tokens
            if self.est.tokens(view, attempt_tools) > budget:
                carried = (self.summary or "").strip()
                pinned = [m for m in view if carried and m.get("role") == "system"
                          and carried in _text_of(m.get("content"))]
                view = self.fit(view, attempt_tools, budget, keep=[view[-1], anchor, *pinned], track_anchor=False)
            error = ""
            try:
                res = await self.llm.chat(
                    messages=view, **self._judge_kwargs(attempt_tools, self.cfg.summary_max_tokens)
                )
            except Exception as exc:  # noqa: BLE001
                res, error = None, str(exc)
            if not isinstance(res, dict) or not res.get("ok"):
                error = error or str((res or {}).get("error") if isinstance(res, dict) else res)
                if refits_left and self.learn_from_error(error, view, attempt_tools, self.cfg.summary_max_tokens):
                    refits_left -= 1
                    log.warning(
                        "self_compact summarizer view overflowed (provider=%s, estimate now %s, budget %s, scale %.2f); refitting",
                        (self._parse_overflow(error) or {}).get("input_tokens"),
                        self.est.tokens(view, attempt_tools), budget, self.est.scale,
                    )
                    continue
                log.warning("self_compact summarizer failed: %s", error[:300])
                return ""
            self.est.calibrate(res.get("tokens_prompt"), view, attempt_tools)
            summary = clean_model_text(res.get("content") or "")
            lines = [ln.strip().lower() for ln in summary.splitlines() if ln.strip()]
            if lines and not all(ln in prior_replies for ln in lines):
                return summary
            log.warning("self_compact summarizer echoed a prior reply (%d chars); %s",
                        len(summary), "retrying without tools" if attempt_tools else "keeping trajectory")
            i += 1
        return ""

    def fit(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        budget: int,
        *,
        keep: Optional[List[Optional[Dict[str, Any]]]] = None,
        track_anchor: bool = True,
    ) -> List[Dict[str, Any]]:
        """Shrink the largest message bodies (middle cut) until the estimate fits ``budget``.

        Messages in ``keep`` are cut only after everything else; the leading
        system message is cut last. Image parts become a text marker.
        """
        protected = [k for k in (keep or []) if k is not None]
        out = list(messages)
        originals = list(messages)
        marker = "[… trimmed to fit the model context window …]"

        def replace(i: int, content: str) -> None:
            out[i] = {**out[i], "content": content}
            if track_anchor and originals[i] is self._anchor_msg:
                self._anchor_msg = out[i]

        def order(i: int) -> Tuple[int, int]:
            m = originals[i]
            tier = 2 if (i == 0 and m.get("role") == "system") else (1 if any(m is k for k in protected) else 0)
            return (tier, -len(_text_of(out[i].get("content"))))

        for _ in range(64):
            over = self.est.tokens(out, tools) - budget
            if over <= 0:
                break
            idx = sorted(range(len(out)), key=order)
            changed = False
            for i in idx:
                content = out[i].get("content")
                if isinstance(content, list):
                    replace(i, _text_of(content) + "\n[attachment omitted to fit the context window]")
                    changed = True
                    break
                text = _text_of(content)
                if len(text) > 400:
                    cut_chars = int(over * self.est.chars_per_token / max(self.est.scale, 1e-6)) + 200
                    replace(i, _shorten(text, max(200, len(text) - cut_chars), marker))
                    changed = True
                    break
            if not changed:
                break
        return out

    # ---- round hook -----------------------------------------------------

    def _advance(self, step: int) -> None:
        if step > self._last_step:
            if self._last_step:
                delta = step - self._last_step
                self.rounds_since_compact += delta
                self.rounds_since_probe += delta
            self._last_step = step

    def _rebuild(self, messages: List[Dict[str, Any]], anchor: Optional[Dict[str, Any]], summary: str) -> List[Dict[str, Any]]:
        sys_msgs: List[Dict[str, Any]] = []
        for m in messages:
            if m.get("role") != "system":
                break
            if m is not self._section_msg:
                sys_msgs.append(m)
        self._section_msg = None
        tail: List[Dict[str, Any]] = [anchor] if anchor is not None else []
        return sys_msgs + tail + [
            {"role": "assistant", "content": summary},
            {"role": "user", "content": self.prompts.continue_prompt},
        ]

    async def _compress(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        anchor: Optional[Dict[str, Any]],
        *,
        step: int,
        trigger: str,
        verdict: Optional[Verdict] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        before = self.est.tokens(messages, tools)
        has_trajectory = any(
            m.get("role") != "system" and m is not anchor for m in messages
        ) or self._section_msg is not None
        summary = await self.summarize(messages, tools, anchor) if has_trajectory else ""
        event: Dict[str, Any] = {
            "type": "context_compacted",
            "step": step,
            "trigger": trigger,
            "tokens_before": before,
            "context_window": self.context_tokens,
        }
        if verdict is not None:
            event["verdict"] = verdict.to_dict()
        if summary:
            messages = self._rebuild(messages, anchor, summary)
            self.summary = summary
            self.new_summary = True
            self.compactions_total += 1
            self.rounds_since_compact = 0
            self.rounds_since_probe = 0
            event["summary_chars"] = len(summary)
        else:
            event["summary_chars"] = 0
            event["fallback"] = "summarizer_failed" if has_trajectory else "nothing_to_summarize"
        if self.est.tokens(messages, tools) > self.hard_limit:
            messages = self.fit(messages, tools, self.hard_limit, keep=[self._anchor_msg])
            event.setdefault("fallback", "trimmed")
        event["tokens_after"] = self.est.tokens(messages, tools)
        self.events.append({k: v for k, v in event.items() if k != "type"})
        log.info(
            "self_compact %s step=%s tokens %s→%s window=%s summary_chars=%s",
            trigger, step, before, event["tokens_after"], self.context_tokens, event["summary_chars"],
        )
        return messages, event

    async def before_round(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]],
        anchor: Optional[Dict[str, Any]],
        step: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run the gates for this round; returns (messages, events to emit)."""
        self._advance(step)
        anchor = self._anchor(messages, anchor)
        cfg = self.cfg
        events: List[Dict[str, Any]] = []

        if self.history_owner and step <= 1:
            if self.est.tokens(messages, tools) <= self.hard_limit:
                return messages, events
            messages, ev = await self._compress(messages, tools, anchor, step=step, trigger="window_guard")
            self.trajectory_messages = 0
            return messages, [ev]

        trigger = self._forced_trigger(messages, tools)
        if trigger:
            messages, ev = await self._compress(messages, tools, anchor, step=step, trigger=trigger)
            self.trajectory_messages = 0
            return messages, [ev]

        tokens = self.est.tokens(messages, tools)
        gates = (
            self.rounds_since_compact >= cfg.min_rounds
            and tokens >= cfg.probe_min_ratio * self.context_tokens
            and self.rubric_fired_this_turn < cfg.max_rubric_per_turn
            and self.rounds_since_probe >= cfg.probe_period
        )
        if not gates:
            return messages, events
        verdict = await self.probe(messages, tools, anchor)
        self.rounds_since_probe = 0
        probe_ev = {
            "type": "context_probe",
            "step": step,
            "decision": verdict.decision if verdict else CONTINUE,
            "answers": verdict.to_dict()["answers"] if verdict else {},
            "tokens": tokens,
            "context_window": self.context_tokens,
        }
        events.append(probe_ev)
        self.events.append({k: v for k, v in probe_ev.items() if k != "type"})
        if verdict and verdict.decision == COMPRESS:
            self.rubric_fired_this_turn += 1
            messages, ev = await self._compress(messages, tools, anchor, step=step, trigger="rubric", verdict=verdict)
            self.trajectory_messages = 0
            events.append(ev)
        elif verdict is None:
            # A failed probe may have taught us the real prompt size.
            trigger = self._forced_trigger(messages, tools)
            if trigger:
                messages, ev = await self._compress(messages, tools, anchor, step=step, trigger=trigger)
                self.trajectory_messages = 0
                events.append(ev)
        return messages, events

    def _compressible_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """What a summary could replace: everything but the leading system run and the turn's anchor."""
        lead = next((i for i, m in enumerate(messages) if m.get("role") != "system"), len(messages))
        rest = [m for m in messages[lead:] if m is not self._anchor_msg]
        return self.est.tokens(rest) if rest else 0

    def _forced_trigger(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> str:
        tokens = self.est.tokens(messages, tools)
        if tokens > self.hard_limit:
            return "window_guard"
        if tokens >= self.cfg.backstop_ratio * self.context_tokens and (
            self._compressible_tokens(messages) >= self.cfg.backstop_min_trajectory_ratio * self.context_tokens
        ):
            return "backstop"
        if self.trajectory_messages and self.trajectory_messages >= int(self.cfg.history_limit * self.cfg.hard_ratio):
            return "history_backstop"
        return ""

    async def prepare_history(
        self,
        history: List[Mapping[str, Any]],
        user_message: str,
        *,
        max_turns: int = 0,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Context for engines that make plain completions instead of a tool loop.

        ``history`` is the trajectory since the last checkpoint (the carried
        summary covers everything older). When it holds more than
        ``max_turns`` prior turns they are compacted instead of dropped.
        Returns (context messages without the turn message, events).
        """
        turns = [
            {"role": str(m.get("role")), "content": _text_of(m.get("content"))}
            for m in history or []
            if isinstance(m, Mapping)
            and m.get("role") in ("user", "assistant")
            and _text_of(m.get("content")).strip()
        ]
        anchor: Dict[str, Any] = {"role": "user", "content": str(user_message or "")}
        messages = self.attach_summary(turns + [anchor])
        prior = sum(1 for m in turns if m["role"] == "assistant")
        if max_turns and prior > max_turns and not self.history_owner:
            self._advance(1)
            messages, ev = await self._compress(messages, None, self._anchor(messages, anchor), step=1, trigger="history_turns")
            self.trajectory_messages = 0
            events = [ev]
        else:
            messages, events = await self.before_round(messages, tools=None, anchor=anchor, step=1)
        if self.new_summary:
            return [{"role": "system", "content": render_summary_section(self.summary, self.prompts)}], events
        return [m for m in messages if m is not self._anchor_msg], events

    # ---- provider overflow --------------------------------------------

    def is_overflow(self, error_text: str) -> bool:
        return bool(self._looks_overflow(str(error_text or "")))

    def can_retry(self) -> bool:
        return self.overflow_retries < self.cfg.max_overflow_retries

    async def recover_overflow(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]],
        anchor: Optional[Dict[str, Any]],
        step: int,
        error_text: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Learn the real window from the error, compact, and return messages to retry with."""
        self.overflow_retries += 1
        self.learn_from_error(error_text, messages, tools)
        anchor = self._anchor(messages, anchor)
        return await self._compress(messages, tools, anchor, step=step, trigger="provider_overflow")

    # ---- persistence ----------------------------------------------------

    def done_payload(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "rubric": self.cfg.rubric,
            "rounds_since_probe": self.rounds_since_probe,
            "compactions": self.compactions_total,
            "context_window": self.context_tokens,
            "token_scale": round(self.est.scale, 4),
        }
        if self.new_summary and self.summary:
            out["summary"] = self.summary
        if self.events:
            out["events"] = self.events
        return out


def _positive_int(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


__all__ = [
    "COMPRESS",
    "CONTINUE",
    "PromptSet",
    "SelfCompactController",
    "SelfCompactSettings",
    "TokenEstimator",
    "Verdict",
    "load_prompts",
    "parse_verdict",
    "render_summary_section",
    "split_trajectory",
]
