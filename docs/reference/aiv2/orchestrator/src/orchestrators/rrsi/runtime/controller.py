"""RRSI rounds over a conversation's evolve set (Algorithms 1 and 2 of the paper).

Per chat turn the evolve set D is the current request plus a bounded replay of
earlier requests of the same conversation. One round t:

  1. Evaluate(H_t, D, k)            incumbent trials, judged per rubric
  2. delta                          cfg.delta, else bootstrap calibration (floored)
  3. F_t <- Analyze(H_t, D)         three-lens report over worst / best traces
  4. b_t, sigma_t, T_t, U_t, E_t, B_t from the vendored schedule / History
  5. m candidates: propose -> critic (bounded repair) -> tag -> smoke (validate)
  6. Evaluate(H', D, k) for screened candidates
  7. select_round: floor S_t - delta (S* on this D), cost rule, guards, argmax
  8. H_{t+1}, history, attribution, cumulative-gain trajectory for sigma

Because D changes between turns, scores are only compared within a round on the
same D: S* for the floor is the incumbent's score on D, and the stall flag reads
the cumulative accepted gain.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..._shared.llm_roles import LLMCall
from ..core.calibrate import calibrate
from ..core.components import normalize
from ..core.config import RRSIConfig
from ..core.evaluate import EvalResult, TaskResult, aggregate
from ..core.history import exploration, stall_flag
from ..core.schedule import edit_budget
from ..core.selection import Candidate, Decision, select_round
from . import harness as H
from . import judge as J
from . import roles as R
from .domain import ChatDomain, Task, Trial
from .state import RrsiState

LogFn = Callable[[str], None]
ProgressFn = Callable[[Dict[str, Any]], None]
# Runs one raw policy turn of ``task`` under ``harness`` (no retry / review).
PolicyRunner = Callable[[H.Harness, Task, int], Awaitable[Trial]]

VARIANT_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class RrsiSettings:
    policy_engine: str
    policy_config: Dict[str, Any]
    rounds_per_turn: int
    replay_turns: int
    replay_max_request_chars: int
    role_text_max_chars: int
    delta_floor: float
    trial_parallel: int
    trial_timeout_sec: int
    proposer_max_turns: int
    role_temperature: float
    role_max_tokens: int
    role_enable_thinking: bool
    judge_temperature: float
    judge_max_tokens: int
    json_retries: int
    rubric_min_criteria: int
    rubric_max_criteria: int
    leak_ngram: int
    frozen_engine_keys: List[str]
    estimate_tokens_when_missing: bool
    chars_per_token: float
    history_render_rows: int
    initial_harness: Dict[str, Any]
    show_round_log: bool

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "RrsiSettings":
        missing = [f for f in cls.__dataclass_fields__ if f not in cfg]
        if missing:
            raise ValueError(f"rrsi config missing keys (manifest defaults?): {missing}")
        out = cls(**{f: cfg[f] for f in cls.__dataclass_fields__})
        out.policy_engine = str(out.policy_engine or "").strip()
        out.policy_config = dict(out.policy_config or {})
        out.initial_harness = dict(out.initial_harness or {})
        out.frozen_engine_keys = [str(k) for k in (out.frozen_engine_keys or [])]
        out.rounds_per_turn = max(0, int(out.rounds_per_turn))
        out.replay_turns = max(0, int(out.replay_turns))
        out.replay_max_request_chars = max(0, int(out.replay_max_request_chars or 0))
        out.role_text_max_chars = max(0, int(out.role_text_max_chars or 0))
        out.trial_parallel = max(1, int(out.trial_parallel))
        return out

    def role_params(self) -> J.RoleParams:
        return J.RoleParams(self.role_temperature, self.role_max_tokens or None,
                            self.role_enable_thinking, self.json_retries, self.role_text_max_chars)

    def judge_params(self) -> J.RoleParams:
        return J.RoleParams(self.judge_temperature, self.judge_max_tokens or None,
                            False, self.json_retries, self.role_text_max_chars)


def task_id_for(request: str, position: int) -> str:
    digest = hashlib.sha1(f"{position}\x00{request}".encode("utf-8")).hexdigest()[:10]
    return f"q{position}_{digest}"


def build_evolve_set(user_message: str, history: Optional[Sequence[Mapping[str, Any]]],
                     replay_turns: int, max_request_chars: int = 0) -> Tuple[Dict[str, Task], str]:
    """D = current request + up to ``replay_turns`` earlier requests (with their context).

    Earlier requests longer than ``max_request_chars`` (0 = no limit) are not replayed.
    """
    rows = [
        {"role": str(m.get("role")), "content": str(m.get("content") or "")}
        for m in (history or [])
        if isinstance(m, Mapping) and m.get("role") in ("user", "assistant")
        and str(m.get("content") or "").strip()
    ]
    user_positions = [
        i for i, m in enumerate(rows)
        if m["role"] == "user" and not (max_request_chars and len(m["content"]) > max_request_chars)
    ]
    tasks: Dict[str, Task] = {}
    for pos in user_positions[-replay_turns:] if replay_turns else []:
        request = rows[pos]["content"]
        tid = task_id_for(request, pos)
        tasks[tid] = Task(id=tid, request=request, context=rows[:pos])
    current_id = task_id_for(user_message, len(rows))
    tasks[current_id] = Task(id=current_id, request=user_message, context=rows, current=True)
    return tasks, current_id


@dataclass
class Evaluation:
    ev: EvalResult
    trials: Dict[str, List[Trial]]


@dataclass
class RoundOutcome:
    t: int
    accepted: Optional[str]
    decisions: List[Decision]
    incumbent: Evaluation
    next_incumbent: Evaluation
    delta: float
    budget: int
    sigma: int
    report: Dict[str, Any]


@dataclass
class TurnOutcome:
    harness: H.Harness
    version: int
    rounds: List[RoundOutcome] = field(default_factory=list)
    answer: Optional[Trial] = None
    served_only: bool = False


class RrsiController:
    def __init__(
        self,
        settings: RrsiSettings,
        cfg: RRSIConfig,
        domain: ChatDomain,
        llm: LLMCall,
        policy: PolicyRunner,
        state: RrsiState,
        *,
        allowed_engine_keys: Optional[Set[str]],
        known_tool_ids: Optional[Set[str]],
        tool_catalog: Sequence[Mapping[str, Any]] = (),
        on_log: Optional[LogFn] = None,
        on_progress: Optional[ProgressFn] = None,
    ) -> None:
        self.st = settings
        self.cfg = cfg
        self.domain = domain
        self.llm = llm
        self.policy = policy
        self.state = state
        self.allowed_engine_keys = allowed_engine_keys
        self.known_tool_ids = known_tool_ids
        self.tool_catalog = list(tool_catalog)
        self._log = on_log or (lambda _m: None)
        self._progress = on_progress or (lambda _p: None)

    # ------------------------------------------------------------ frontier --
    def ensure_frontier(self) -> Dict[str, Any]:
        fr = self.state.frontier()
        if fr is not None:
            return fr
        h0 = H.normalize_harness(self.domain, self.st.initial_harness)
        problems = H.validate(self.domain, h0, allowed_engine_keys=self.allowed_engine_keys,
                              frozen_engine_keys=set(self.st.frozen_engine_keys),
                              known_tool_ids=self.known_tool_ids)
        if problems:
            raise ValueError(f"initial_harness invalid: {problems}")
        self.state.save_harness(0, h0)
        fr = {
            "domain": self.domain.name,
            "incumbent": {"version": 0, "t": 0, "S": None, "C": None},
            "t": 0,
            "trajectory": [{"t": 0, "G": 0.0}],
            "config": self.cfg.dump(),
        }
        self.state.save_frontier(fr)
        self.state.history.append({"t": 0, "variant": "-", "edit_id": None, "component": None,
                                   "hypothesis": "H_0 baseline", "outcome": "BASELINE",
                                   "accepted": True, "delta_S": None, "delta_C": None, "bundle": 0})
        return fr

    def incumbent_harness(self, fr: Mapping[str, Any]) -> H.Harness:
        h = self.state.load_harness(int(fr["incumbent"]["version"]))
        return H.normalize_harness(self.domain, h)

    # ------------------------------------------------------------ evaluate --
    async def _rubric(self, task: Task) -> List[Dict[str, Any]]:
        cached = self.state.rubrics().get(task.id)
        if cached:
            return cached
        rubric = await J.make_rubric(self.llm, self.domain, task, self.st.judge_params(),
                                     min_criteria=self.st.rubric_min_criteria,
                                     max_criteria=self.st.rubric_max_criteria)
        self.state.save_rubric(task.id, rubric)
        return rubric

    async def _review(self, harness: H.Harness, task: Task, trial: Trial) -> Trial:
        cfg = harness.get("subagent_review") or {}
        if not cfg.get("enabled") or trial.failed:
            return trial
        prompt = self.domain.texts["reviewer"].format(
            instructions=str(cfg.get("instructions") or "").strip(),
            context=J._context_text(task),
            request=task.request,
            draft=trial.answer,
        )
        reply = await self.llm([{"role": "user", "content": prompt}],
                               temperature=self.st.role_temperature,
                               max_tokens=self.st.role_max_tokens or None,
                               enable_thinking=False)
        if reply.ok and reply.text.strip():
            trial.answer = reply.text.strip()
            trial.reviewed = True
            if reply.tokens is not None and trial.tokens is not None:
                trial.tokens += reply.tokens
        return trial

    async def run_trial(self, harness: H.Harness, task: Task, index: int) -> Trial:
        async def once() -> Trial:
            try:
                if self.st.trial_timeout_sec and self.st.trial_timeout_sec > 0:
                    return await asyncio.wait_for(self.policy(harness, task, index),
                                                  timeout=self.st.trial_timeout_sec)
                return await self.policy(harness, task, index)
            except asyncio.TimeoutError:
                return Trial(task_id=task.id, index=index, error=f"timeout after {self.st.trial_timeout_sec}s")
            except Exception as exc:  # noqa: BLE001 — a crashed trial is a measurement
                return Trial(task_id=task.id, index=index, error=f"{type(exc).__name__}: {exc}")

        trial = await once()
        if trial.failed and harness.get("retry_on_failure"):
            trial = await once()
        trial = await self._review(harness, task, trial)
        if trial.tokens is None and self.st.estimate_tokens_when_missing and not trial.failed:
            trial.tokens = round(len(trial.answer) / max(self.st.chars_per_token, 0.1), 1)
        return trial

    async def evaluate(self, harness: H.Harness, tasks: Mapping[str, Task], job: str) -> Evaluation:
        k = max(1, int(self.cfg.k))
        rubrics = {tid: await self._rubric(task) for tid, task in tasks.items()}
        sem = asyncio.Semaphore(self.st.trial_parallel)
        total = len(tasks) * k
        done = 0

        async def one(task: Task, j: int) -> Trial:
            nonlocal done
            async with sem:
                trial = await self.run_trial(harness, task, j)
                try:
                    await J.grade(self.llm, self.domain, task, rubrics[task.id], trial, self.st.judge_params())
                except Exception as exc:  # noqa: BLE001 — infrastructure: counts as missing
                    trial.graded = False
                    trial.reward = 0.0
                    trial.error = trial.error or f"judge failed: {exc}"
            done += 1
            self._progress({"phase": "evaluate", "job": job, "completed": done, "total": total})
            return trial

        results = await asyncio.gather(*(one(task, j) for task in tasks.values() for j in range(k)))
        trials: Dict[str, List[Trial]] = {tid: [] for tid in tasks}
        for tr in results:
            trials[tr.task_id].append(tr)
        per_task: Dict[str, TaskResult] = {}
        failed = 0
        for tid, rows in trials.items():
            rows.sort(key=lambda x: x.index)
            weight = rows[0].weight if rows else 1.0
            per_task[tid] = TaskResult(
                rewards=[r.reward for r in rows],
                weights=[r.weight or weight for r in rows],
                tokens=[r.tokens for r in rows],
                missing=sum(1 for r in rows if r.failed or not r.graded),
            )
            failed += sum(1 for r in rows if r.failed)
        ev = aggregate(job, k, per_task, {"failed_trials": failed})
        return Evaluation(ev=ev, trials=trials)

    def _invalid(self, ev: EvalResult) -> bool:
        return ev.missing > self.cfg.invalid_missing_frac * ev.n_expected

    # -------------------------------------------------------------- traces --
    def build_traces(self, evaluation: Evaluation) -> Dict[str, Trial]:
        ranked = sorted(evaluation.ev.per_task.items(), key=lambda kv: kv[1].mean)
        fails = [t for t, _ in ranked[: self.cfg.n_fail_traces]]
        wins = [t for t, _ in ranked[-self.cfg.n_success_traces:] if t not in fails] if self.cfg.n_success_traces else []
        out: Dict[str, Trial] = {}
        for tid in fails + wins:
            rows = evaluation.trials.get(tid) or []
            if not rows:
                continue
            pick = min(rows, key=lambda r: r.reward) if tid in fails else max(rows, key=lambda r: r.reward)
            out[tid] = pick
        return out

    def _delta(self, ev: EvalResult) -> Tuple[float, Dict[str, Any]]:
        if self.cfg.delta is not None:
            return float(self.cfg.delta), {"method": "configured"}
        try:
            cal = calibrate([ev], z=self.cfg.delta_z)
        except Exception as exc:  # noqa: BLE001
            return float(self.st.delta_floor), {"method": "floor", "error": str(exc)}
        return max(float(self.st.delta_floor), float(cal["delta"])), cal

    # --------------------------------------------------------------- draft --
    async def _draft(self, t: int, vid: str, *, harness: H.Harness, tasks: Dict[str, Task],
                     inc: Evaluation, traces: Dict[str, Trial], report: Dict[str, Any],
                     budget: int, explore: Dict[str, Any], reserved: bool,
                     prune: List[Dict[str, Any]], hist_rows: List[Dict[str, Any]],
                     corpus: List[str]) -> Tuple[Candidate, Optional[H.Harness]]:
        rel = f"r{t}/{vid}"
        frozen = set(self.st.frozen_engine_keys)
        variant_brief = (f"You are variant {vid} of round {t}. {self.cfg.m} variants are "
                         f"drafted independently from the same incumbent this round "
                         f"and each is evaluated on the full evolve set; the best "
                         f"admissible one becomes H_{t + 1}.")
        common = dict(
            harness=harness, tasks=tasks, traces=traces, trials=inc.trials, report=report,
            history_rows=hist_rows, scoreboard=self.state.scoreboard(), budget=budget,
            explore=explore, reserved_slot=reserved, prune_set=prune, variant_brief=variant_brief,
            params=self.st.role_params(), max_turns=self.st.proposer_max_turns,
            allowed_engine_keys=self.allowed_engine_keys, frozen_engine_keys=frozen,
            known_tool_ids=self.known_tool_ids, tool_catalog=self.tool_catalog,
        )

        def finish(gate: str, detail: str = "", edits=None) -> Tuple[Candidate, None]:
            self.state.write(f"{rel}/prep.json", {"gate_failure": gate, "detail": detail, "edits": edits or []})
            self._log(f"  {vid}: {gate} {detail[:200]}")
            return Candidate(vid, edits or [], gate_failure=gate, detail=detail), None

        prop = await R.propose(self.llm, self.domain, **common)
        self.state.write(f"{rel}/proposal.json", prop)
        if prop["status"] != "done":
            return finish("no_proposal", str(prop.get("status")))

        verdict: Dict[str, Any] = {}
        task_ids = list(tasks)
        for attempt in range(1 + int(self.cfg.repair_rounds)):
            cand_h = prop["harness"]
            diff = H.render_diff(harness, cand_h)
            self.state.write(f"{rel}/diff.json", {"diff": diff})
            verdict = await R.review(
                self.llm, self.domain, diff=diff, added=H.added_text(harness, cand_h),
                summary=prop.get("mechanism") or "", targets_mode=prop.get("targets_mode") or "",
                edits=prop.get("edits") or [], corpus=corpus, task_ids=task_ids,
                leak_ngram=self.st.leak_ngram, params=self.st.role_params(),
            )
            if verdict.get("verdict") == "accept":
                tagged = [self._tag(harness, e) for e in prop.get("edits") or []]
                if reserved and explore.get("untried") and not any(c in explore["untried"] for c in tagged):
                    verdict = {"verdict": "reject",
                               "reasons": [f"this variant holds a RESERVED EXPLORATION SLOT: at least one edit "
                                           f"must be on a never-exercised component from {explore['untried']}, "
                                           f"judged by the DIFF, and none is"],
                               "risk_notes": verdict.get("risk_notes")}
            self.state.write(f"{rel}/critic_a{attempt}.json", verdict)
            if verdict.get("verdict") == "accept" or attempt >= int(self.cfg.repair_rounds):
                break
            self._log(f"  {vid}: critic objections (repair {attempt + 1}/{self.cfg.repair_rounds}): "
                      f"{str(verdict.get('reasons'))[:200]}")
            prop = await R.propose(self.llm, self.domain, **common, repair_brief={
                "reasons": verdict.get("reasons"), "risk_notes": verdict.get("risk_notes"),
                "your_declared_edits": prop.get("edits"), "previous_candidate": prop.get("harness"),
            })
            self.state.write(f"{rel}/proposal_r{attempt + 1}.json", prop)
            if prop["status"] != "done":
                break
        edits = prop.get("edits") or []
        if verdict.get("verdict") != "accept" or prop.get("status") != "done":
            return finish("critic_reject", str(verdict.get("reasons"))[:600], edits)

        for e in edits:
            e["component"] = self._tag(harness, e)
        cand_h = prop["harness"]
        smoke = H.validate(self.domain, cand_h, allowed_engine_keys=self.allowed_engine_keys,
                           frozen_engine_keys=frozen, known_tool_ids=self.known_tool_ids)
        if smoke:
            return finish("smoke_fail", json.dumps(smoke)[:600], edits)
        self.state.write(f"{rel}/prep.json", {"edits": edits, "harness": cand_h,
                                              "mechanism": prop.get("mechanism")})
        self._log(f"  {vid}: {len(edits)} edit(s) on {[e['component'] for e in edits]}")
        return Candidate(vid, edits, diff_path=f"{rel}/diff.json"), cand_h

    def _tag(self, harness: H.Harness, edit: Mapping[str, Any]) -> str:
        """Validate the declared component against this edit's own diff."""
        single, _ = H.apply_changes(self.domain, harness, edit.get("changes") or [])
        edit_diff = H.render_diff(harness, single)
        return normalize(edit.get("component"), edit_diff, self.domain.component_signals)

    # --------------------------------------------------------------- round --
    async def round(self, t: int, tasks: Dict[str, Task], inc: Optional[Evaluation] = None) -> RoundOutcome:
        cfg = self.cfg
        fr = self.ensure_frontier()
        harness = self.incumbent_harness(fr)
        rdir = f"r{t}"
        if inc is None:
            self._log(f"round {t}: evaluating incumbent v{fr['incumbent']['version']} on {len(tasks)} task(s) x k={cfg.k}")
            inc = await self.evaluate(harness, tasks, f"r{t}_incumbent")
        self.state.write(f"{rdir}/incumbent_eval.json", inc.ev.to_json())
        delta, cal = self._delta(inc.ev)
        S_star = inc.ev.S
        self._log(f"round {t}: incumbent S={inc.ev.S:.4f} C={inc.ev.C} delta={delta:.4f}")

        traces = self.build_traces(inc)
        prior = self.state.global_analysis()
        try:
            report = await R.analyze(self.llm, self.domain, tasks, inc.trials, traces,
                                     self.st.role_params(), prior=prior)
        except Exception as exc:  # noqa: BLE001 — proposer still works from the traces
            report = {"failure_modes": [], "capability_gaps": [], "success_habits": [], "error": str(exc)}
        self.state.write(f"{rdir}/analysis_report.json", report)
        self.state.save_global_analysis({"failure_modes": report.get("failure_modes"),
                                         "success_habits": report.get("success_habits")})
        self._log(f"round {t}: top modes {[m.get('mode') for m in report.get('failure_modes', [])[:5]]}")

        budget = edit_budget(t, cfg.T, cfg.b_min, cfg.b_max)
        traj = [float(x["G"]) for x in fr["trajectory"]]
        sigma = stall_flag(traj, t, cfg.w, delta)
        tried = self.state.history.tried()
        explore = exploration(t, sigma, tried, cfg.m_draft)
        prune = self.state.history.prune_set(t, cfg.n_prune)
        hist_rows = self.state.history.render(self.st.history_render_rows)
        self.state.write(f"{rdir}/directives.json", {
            "t": t, "b_t": budget, "sigma_t": sigma, "tried": sorted(tried), "explore": explore,
            "prune_set": prune, "delta": delta, "calibration": cal, "S_star": S_star,
        })
        self._log(f"round {t}: b_t={budget} sigma_t={sigma} untried={explore['untried']} "
                  f"prune={[p['component'] for p in prune]}")

        corpus = [task.request for task in tasks.values()]
        corpus += [c.get("text", "") for rub in self.state.rubrics().values() for c in rub]
        corpus += [tr.answer for rows in inc.trials.values() for tr in rows if tr.answer]

        drafts = await asyncio.gather(*(
            self._draft(t, VARIANT_LABELS[v], harness=harness, tasks=tasks, inc=inc, traces=traces,
                        report=report, budget=budget, explore=explore,
                        reserved=bool(sigma and explore["untried"] and v >= cfg.m - cfg.m_draft),
                        prune=prune, hist_rows=hist_rows, corpus=corpus)
            for v in range(cfg.m)
        ))
        cands = [c for c, _ in drafts]
        cand_harness = {c.variant: h for c, h in drafts if h is not None}
        cand_trials: Dict[str, Evaluation] = {}

        live = [c for c in cands if c.gate_failure is None]
        sem = asyncio.Semaphore(max(1, int(cfg.eval_parallel)))

        async def measure(c: Candidate) -> None:
            async with sem:
                job = f"r{t}{c.variant}"
                self._log(f"  {c.variant}: evaluating")
                res = await self.evaluate(cand_harness[c.variant], tasks, job)
                if self._invalid(res.ev):
                    self._log(f"  {c.variant}: {res.ev.missing}/{res.ev.n_expected} trials missing; retrying once")
                    res = await self.evaluate(cand_harness[c.variant], tasks, job)
                if self._invalid(res.ev):
                    c.gate_failure = "eval_invalid"
                    c.detail = f"{res.ev.missing}/{res.ev.n_expected} trials missing after a retry"
                    return
                c.ev = res.ev
                cand_trials[c.variant] = res
                self.state.write(f"r{t}/{c.variant}/eval.json", res.ev.to_json())

        await asyncio.gather(*(measure(c) for c in live))

        counts = self.state.history.incumbent_component_counts()
        winner, decisions = select_round(cands, inc.ev, S_star, delta, cfg, counts, guard_fn=self.domain.guards)
        self.state.write(f"{rdir}/decisions.json", [d.to_json() for d in decisions])
        for c, dec in zip(cands, decisions):
            if c.ev is None:
                self.state.history.append_candidate(t, c.variant, c.edits, c.gate_failure or "not_evaluated",
                                                    None, None, False, None, None, c.diff_path, c.detail)
                continue
            outcome = "ACCEPTED" if c is winner else ("LOST" if dec.admissible else "REJECTED")
            self.state.history.append_candidate(t, c.variant, c.edits, outcome, dec.delta_S, dec.delta_C,
                                                c is winner, dec.S, dec.C, c.diff_path, dec.reason)
            self.state.append_attribution(self._attribute(t, c, inc.ev))
            self._log(f"  {c.variant}: S={dec.S:.4f} dS={dec.delta_S:+.4f} dC={dec.delta_C:+.3f} "
                      f"nu={dec.novelty} -> {outcome}: {dec.reason}")

        G = traj[-1] if traj else 0.0
        if winner is not None:
            version = int(fr["incumbent"]["version"]) + 1
            self.state.save_harness(version, cand_harness[winner.variant])
            fr["incumbent"] = {"version": version, "t": t + 1, "S": winner.ev.S, "C": winner.ev.C,
                               "variant": winner.variant}
            G += float(winner.ev.S - inc.ev.S)
            nxt = cand_trials[winner.variant]
            self._log(f"round {t}: ACCEPTED {winner.variant} -> harness v{version} S={winner.ev.S:.4f}")
        else:
            fr["incumbent"] = {**fr["incumbent"], "t": t + 1, "S": inc.ev.S, "C": inc.ev.C}
            nxt = inc
            self._log(f"round {t}: no admissible candidate; H_{t + 1} = H_{t}")
        fr["t"] = t + 1
        fr["trajectory"] = [x for x in fr["trajectory"] if int(x["t"]) <= t] + [{"t": t + 1, "G": round(G, 6)}]
        self.state.save_frontier(fr)
        return RoundOutcome(t=t, accepted=winner.variant if winner else None, decisions=decisions,
                            incumbent=inc, next_incumbent=nxt, delta=delta, budget=budget,
                            sigma=sigma, report=report)

    def _attribute(self, t: int, c: Candidate, inc: EvalResult) -> List[Dict[str, Any]]:
        thr = self.domain.regression_threshold(self.cfg.k)
        cand = c.ev
        rows = []
        for e in c.edits:
            pred = [str(p) for p in (e.get("predicted_affected") or [])
                    if str(p) in inc.per_task and str(p) in cand.per_task]
            hits = [p for p in pred if cand.per_task[p].mean > inc.per_task[p].mean]
            drops = [x for x in inc.per_task if x in cand.per_task
                     and cand.per_task[x].mean - inc.per_task[x].mean <= -thr]
            rows.append({"t": t, "variant": c.variant, "edit_id": e.get("id"),
                         "component": e.get("component"),
                         "hypothesis": str(e.get("hypothesis"))[:160],
                         "n_predicted": len(pred), "predicted_hit": hits,
                         "hit_rate": round(len(hits) / len(pred), 2) if pred else None,
                         "unpredicted_regressions": [d for d in drops if d not in pred][:12]})
        return rows

    # ---------------------------------------------------------------- turn --
    def rounds_this_turn(self, fr: Mapping[str, Any]) -> int:
        remaining = max(0, int(self.cfg.T) - int(fr.get("t", 0)))
        return min(self.st.rounds_per_turn, remaining)

    async def turn(self, tasks: Dict[str, Task], current_id: str) -> TurnOutcome:
        fr = self.ensure_frontier()
        n = self.rounds_this_turn(fr)
        if n <= 0:
            return TurnOutcome(harness=self.incumbent_harness(fr),
                               version=int(fr["incumbent"]["version"]), served_only=True)
        outcomes: List[RoundOutcome] = []
        inc: Optional[Evaluation] = None
        for _ in range(n):
            fr = self.ensure_frontier()
            out = await self.round(int(fr["t"]), tasks, inc)
            outcomes.append(out)
            inc = out.next_incumbent
        fr = self.ensure_frontier()
        rows = (inc.trials.get(current_id) if inc else None) or []
        ok_rows = [r for r in rows if not r.failed]
        answer = max(ok_rows, key=lambda r: (r.reward, -r.index)) if ok_rows else (rows[0] if rows else None)
        return TurnOutcome(harness=self.incumbent_harness(fr), version=int(fr["incumbent"]["version"]),
                           rounds=outcomes, answer=answer)
