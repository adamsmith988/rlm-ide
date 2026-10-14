"""The three search roles of Algorithm 1 for the hub-chat domain.

analyze  F_t <- Analyze(H_t, D)        three-lens report over the incumbent's traces
propose  C ~ P_reg(. | H_t, F_t, L_t, b_t, E_t)   strict-JSON protocol, done() validated
review   Critic(H_t, H') -> {0,1}      deterministic precheck + LLM review
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from ..._shared.llm_roles import LLMCall, extract_json
from ..core.components import K
from . import harness as H
from .domain import ChatDomain, Task, Trial
from .judge import RoleParams, ask_json

_WORD = re.compile(r"\w+", re.U)


# ------------------------------------------------------------------ analyst --
def task_table(domain: ChatDomain, tasks: Dict[str, Task], trials: Dict[str, List[Trial]]) -> str:
    rows = []
    for tid, task in tasks.items():
        tr = trials.get(tid) or []
        mean = (sum(t.reward for t in tr) / len(tr)) if tr else 0.0
        rows.append((mean, domain.task_row(task, tr)))
    return "\n".join(r for _, r in sorted(rows, key=lambda x: x[0]))


async def analyze(
    llm: LLMCall,
    domain: ChatDomain,
    tasks: Dict[str, Task],
    trials: Dict[str, List[Trial]],
    traces: Dict[str, Trial],
    params: RoleParams,
    *,
    prior: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    system = domain.texts["analyst_system"].format(domain_brief=domain.briefs["analyst"])
    prior = dict(prior or {})
    rendered = [
        domain.render_trace(tasks[tid], trial, detail=False)
        for tid, trial in sorted(traces.items(), key=lambda kv: kv[1].reward)
    ]
    prompt = "\n\n".join([
        "=== TASK TABLE (lowest scores first) ===",
        task_table(domain, tasks, trials),
        "=== PRIOR FAILURE-MODE NAMES (for stable naming) ===",
        json.dumps([{"mode": m.get("mode"), "description": m.get("description")}
                    for m in (prior.get("failure_modes") or [])], ensure_ascii=False, indent=1),
        "=== PRIOR SUCCESS-HABIT NAMES (for stable naming) ===",
        json.dumps([{"habit": h.get("habit"), "description": h.get("description")}
                    for h in (prior.get("success_habits") or [])], ensure_ascii=False, indent=1),
        "=== RENDERED TRACES ===",
        "\n\n".join(rendered),
        "=== TASK ===",
        "Produce the report.",
    ])
    raw = await ask_json(llm, system, prompt, params)
    if not isinstance(raw, dict):
        raise ValueError("analyst reply is not an object")
    report = {k: [x for x in (raw.get(k) or []) if isinstance(x, dict)]
              for k in ("failure_modes", "capability_gaps", "success_habits")}
    report["failure_modes"].sort(key=lambda m: -(m.get("n_tasks") or 0))
    return report


# ----------------------------------------------------------------- proposer --
_REQUIRED_EDIT_KEYS = ("id", "component", "changes", "hypothesis", "targets_mode",
                       "predicted_affected", "retroactive_check")


def _schema_for_prompt(domain: ChatDomain, allowed_engine_keys: Optional[Set[str]],
                       frozen: Set[str], tool_catalog: Sequence[Mapping[str, Any]]) -> str:
    fields = {}
    for name, spec in domain.fields.items():
        row = {k: v for k, v in spec.items() if k != "default"}
        if name == "engine_config":
            row["allowed_keys"] = sorted((allowed_engine_keys or set()) - frozen)
        if name == "tool_ids":
            row["catalog"] = [
                {"id": t.get("id"), "description": str(t.get("description") or "")[:140]}
                for t in tool_catalog
            ]
        fields[name] = row
    return json.dumps(fields, ensure_ascii=False, indent=1)


async def propose(
    llm: LLMCall,
    domain: ChatDomain,
    *,
    harness: H.Harness,
    tasks: Dict[str, Task],
    traces: Dict[str, Trial],
    trials: Dict[str, List[Trial]],
    report: Dict[str, Any],
    history_rows: List[Dict[str, Any]],
    scoreboard: List[Dict[str, Any]],
    budget: int,
    explore: Dict[str, Any],
    reserved_slot: bool,
    prune_set: List[Dict[str, Any]],
    variant_brief: str,
    params: RoleParams,
    max_turns: int,
    allowed_engine_keys: Optional[Set[str]],
    frozen_engine_keys: Set[str],
    known_tool_ids: Optional[Set[str]],
    tool_catalog: Sequence[Mapping[str, Any]],
    repair_brief: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Draft one candidate. Returns {status, summary, edits, harness, log}."""
    system = domain.texts["proposer_system"].format(
        domain_brief=domain.briefs["proposer"],
        harness_schema=_schema_for_prompt(domain, allowed_engine_keys, frozen_engine_keys, tool_catalog),
        components=" | ".join(K),
    )
    skill_md, patterns_md = domain.constitution()
    explore_text = explore.get("text", "")
    if reserved_slot:
        explore_text += ("\n\nRESERVED EXPLORATION SLOT: this variant holds one. At "
                         "least one of your edits MUST be on a never-exercised "
                         f"component from: {explore.get('untried')}.")
    prune_text = json.dumps(prune_set, ensure_ascii=False, indent=1) if prune_set else "(none)"
    context = "\n\n".join([
        "=== CONSTITUTION ===", skill_md,
        "=== PATTERN LIBRARY ===", patterns_md,
        *(["=== THIS VARIANT'S BRIEF ===", variant_brief] if variant_brief else []),
        "=== EDIT HISTORY L_t (every measured edit: component, hypothesis, "
        "Delta S, Delta C, accepted). A rejected mechanism is negative evidence; "
        "do not redraw it unchanged. An accepted one carries the gain it "
        "produced; refine what has known credit, not what merely preceded a "
        "rise. ===",
        json.dumps(history_rows, ensure_ascii=False, indent=1),
        "=== ATTRIBUTION SCOREBOARD (how past edits' predictions fared; "
        "unpredicted_regressions are tasks an edit likely broke) ===",
        json.dumps(scoreboard or [], ensure_ascii=False, indent=1),
        "=== EXPLORATION DIRECTIVES E_t ===", explore_text,
        "=== COMPONENTS TO PRUNE B_t (exercised, no strictly improving edit in "
        "the recent window; remove the accepted machinery listed, it has "
        "stopped earning its place) ===", prune_text,
        "=== THREE-LENS ANALYSIS REPORT F_t (failure modes ranked; capability "
        "gaps often need tool/plumbing fixes; success_habits are behaviors "
        "your change MUST NOT break) ===",
        json.dumps(report, ensure_ascii=False, indent=1),
        "=== TASK LIST (ids for predicted_affected / read_trace) ===",
        task_table(domain, tasks, trials),
        "=== CURRENT HARNESS H_t ===",
        json.dumps(harness, ensure_ascii=False, indent=1),
        "=== THIS ROUND'S EDIT BUDGET b_t ===",
        f"You may ship AT MOST {budget} independent edit(s) in this candidate "
        f"(the budget anneals over the run: early rounds explore, late rounds "
        f"make single attributable changes). Ship fewer if the evidence "
        f"supports fewer.",
        "=== TASK ===",
("REPAIR ROUND: your previous candidate and its declared edits are in "
         "the objections payload. The reviewer raised the objections below. "
         "Fix ONLY what the objections require (remove leaked content, split "
         "or re-declare edits, or drop the offending part), then call done with "
         "the FULL corrected edits array; changes are applied to CURRENT "
         "HARNESS H_t, not to the previous candidate.\n\n"
         "=== REVIEWER OBJECTIONS ===\n"
         + json.dumps(repair_brief, ensure_ascii=False, indent=1) + "\nFirst action:")
        if repair_brief else
        "Address the highest-impact failure modes / capability gaps within your "
        "edit budget, then call done with the edits array. First action:",
    ])

    log: List[Dict[str, Any]] = []
    transcript = ""
    for _turn in range(max(1, max_turns)):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "=== ROUND CONTEXT ===\n" + context
             + "\n\n=== INTERACTION LOG ===\n" + transcript
             + "\nReply with exactly one JSON action object."},
        ]
        reply = await llm(messages, temperature=params.temperature, max_tokens=params.max_tokens,
                          enable_thinking=params.enable_thinking)
        if not reply.ok:
            transcript += f"\n[result] ERROR: model call failed ({reply.error})"
            continue
        try:
            act = extract_json(reply.text)
        except ValueError:
            transcript += f"\n[you] {reply.text[:500]}\n[result] ERROR: not valid JSON"
            continue
        if isinstance(act, list):
            act = next((x for x in act if isinstance(x, dict)), None)
        if not isinstance(act, dict):
            transcript += "\n[you] (non-object)\n[result] ERROR: reply with EXACTLY ONE JSON action object"
            continue
        a = act.get("action")
        log.append({k: (v if len(str(v)) < 300 else str(v)[:300] + "...") for k, v in act.items()})
        if a == "read_trace":
            tid = str(act.get("task_id") or "")
            trial = traces.get(tid)
            result = (domain.render_trace(tasks[tid], trial, detail=True)
                      if trial is not None and tid in tasks else
                      f"ERROR: no trace for {tid} (valid ids: {sorted(traces)})")
            transcript += f"\n[you] {json.dumps(act)[:300]}\n[result] {result[:30000]}"
            continue
        if a != "done":
            transcript += (f"\n[you] {json.dumps(act)[:300]}\n[result] ERROR: unknown action {a}; "
                           f"there is no abort action — use read_trace or done")
            continue

        edits = [e for e in (act.get("edits") or []) if isinstance(e, dict)]
        problems: List[str] = []
        if not edits:
            problems.append("edits array is empty")
        if len(edits) > budget:
            problems.append(f"{len(edits)} edits exceed the budget b_t = {budget}")
        candidate = harness
        for e in edits:
            miss = [f for f in _REQUIRED_EDIT_KEYS if not e.get(f)]
            if miss:
                problems.append(f"edit {e.get('id', '?')} missing {miss}")
            comp = str(e.get("component") or "").lower()
            if comp and comp not in K:
                problems.append(f"edit {e.get('id')} component {comp!r} not in {K}")
            for ch in e.get("changes") or []:
                fcomp = domain.field_component(str((ch or {}).get("field") or ""))
                if fcomp and comp and fcomp != comp:
                    problems.append(f"edit {e.get('id')} is tagged {comp!r} but changes "
                                    f"field {ch.get('field')!r} of component {fcomp!r}")
            candidate, ch_problems = H.apply_changes(domain, candidate, e.get("changes") or [])
            problems += [f"edit {e.get('id')}: {p}" for p in ch_problems]
        if reserved_slot and explore.get("untried") and edits and not any(
                str(e.get("component", "")).lower() in explore["untried"] for e in edits):
            problems.append("this variant holds a RESERVED EXPLORATION SLOT: at least one edit "
                            f"must be on a never-exercised component from {explore['untried']}")
        if not problems and not H.changed_fields(harness, candidate):
            problems.append("the edits change nothing in the harness")
        if not problems:
            problems += H.validate(domain, candidate, allowed_engine_keys=allowed_engine_keys,
                                   frozen_engine_keys=frozen_engine_keys, known_tool_ids=known_tool_ids)
        if problems:
            transcript += (f"\n[you] done\n[result] ERROR: {problems}. Call done again fixed "
                           f"(drop or merge edits if over budget; add the required edit if a "
                           f"slot is reserved).")
            continue
        for e in edits:
            e["component"] = str(e.get("component", "")).lower()
            e.setdefault("mechanism", e.get("hypothesis"))
        return {
            "status": "done",
            "summary": act.get("summary"),
            "edits": edits,
            "harness": candidate,
            "mechanism": act.get("summary") or "; ".join(str(e.get("hypothesis")) for e in edits)[:200],
            "targets_mode": ", ".join(str(e.get("targets_mode")) for e in edits)[:200],
            "log": log,
        }
    return {"status": "max_turns", "edits": [], "harness": harness, "log": log}


# ------------------------------------------------------------------- critic --
def _ngrams(text: str, n: int) -> Set[str]:
    words = [w.lower() for w in _WORD.findall(text or "")]
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)} if len(words) >= n else set()


def leakage_hits(added: str, corpus: Sequence[str], n: int) -> List[str]:
    """Word n-grams the candidate shares with evolve-set requests, rubrics or answers."""
    if n <= 0:
        return []
    mine = _ngrams(added, n)
    hits: List[str] = []
    for doc in corpus:
        shared = mine & _ngrams(doc, n)
        hits.extend(sorted(shared)[:3])
    return hits[:5]


def precheck(domain: ChatDomain, diff: str, added: str, corpus: Sequence[str], task_ids: Sequence[str],
             leak_ngram: int) -> List[str]:
    hits: List[str] = []
    for pat, why in domain.critic_patterns:
        if re.search(pat, added):
            hits.append(why)
    for tid in task_ids:
        if tid and tid in added:
            hits.append(f"task id {tid} in harness")
    overlap = leakage_hits(added, corpus, leak_ngram)
    if overlap:
        hits.append(f"copies evolve-set text ({leak_ngram}-gram overlap): {overlap}")
    return hits


async def review(
    llm: LLMCall,
    domain: ChatDomain,
    *,
    diff: str,
    added: str,
    summary: str,
    targets_mode: str,
    edits: List[Dict[str, Any]],
    corpus: Sequence[str],
    task_ids: Sequence[str],
    leak_ngram: int,
    params: RoleParams,
) -> Dict[str, Any]:
    hard = precheck(domain, diff, added, corpus, task_ids, leak_ngram)
    if hard:
        return {"verdict": "reject", "reasons": [f"precheck: {h}" for h in hard], "risk_notes": []}
    if not diff.strip():
        return {"verdict": "reject", "reasons": ["empty diff"], "risk_notes": []}
    system = domain.texts["critic_system"].format(domain_brief=domain.briefs["critic"])
    payload = (
        f"CANDIDATE SUMMARY: {summary}\nTARGETS: {targets_mode}\n\n"
        f"=== DECLARED EDITS (independent changes in this candidate) ===\n"
        f"{json.dumps(edits or [], ensure_ascii=False, indent=1)[:20_000]}\n\n"
        f"=== DIFF ===\n{diff[:120_000]}"
    )
    try:
        v = await ask_json(llm, system, payload, params)
    except ValueError as exc:
        return {"verdict": "reject", "reasons": [f"critic output unparseable: {exc}"], "risk_notes": []}
    if isinstance(v, dict) and v.get("verdict") in ("accept", "reject"):
        return v
    return {"verdict": "reject", "reasons": [f"critic verdict missing: {str(v)[:200]}"], "risk_notes": []}
