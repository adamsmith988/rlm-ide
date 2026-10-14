"""Reward r(x, tau) in [0, 1] for chat requests: a rubric is written once per
task, then each trial is graded per criterion. reward = passed weight / total
weight and the trial's weight is the total (the paper's rubric-domain
weighting), so S_hat is the fraction of criteria passed."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..._shared.llm_roles import LLMCall, extract_json
from .domain import ChatDomain, Task, Trial


@dataclass
class RoleParams:
    temperature: Optional[float]
    max_tokens: Optional[int]
    enable_thinking: Optional[bool]
    json_retries: int
    text_max_chars: int = 0  # request / answer clip in role prompts (0 = no limit)


def clip_text(text: str, limit: int) -> str:
    """Head and tail of ``text`` when longer than ``limit`` (0 = unchanged)."""
    text = str(text or "")
    if limit <= 0 or len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return f"{text[:half]}\n[…]\n{text[-half:]}"


async def ask_json(llm: LLMCall, system: Optional[str], prompt: str, params: RoleParams) -> Any:
    """One JSON-returning role call with bounded re-asks on unparseable output."""
    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    last = ""
    for _ in range(max(1, params.json_retries)):
        reply = await llm(
            messages,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            enable_thinking=params.enable_thinking,
        )
        if not reply.ok:
            last = f"llm error: {reply.error}"
            continue
        last = reply.text
        try:
            return extract_json(reply.text)
        except ValueError:
            messages = messages + [
                {"role": "assistant", "content": reply.text[:4000]},
                {"role": "user", "content": "ERROR: not valid JSON. Reply with the STRICT JSON object only."},
            ]
    raise ValueError(f"no valid JSON after {params.json_retries} attempts: {last[:200]}")


def _context_text(task: Task) -> str:
    if not task.context:
        return "(none)"
    return "\n".join(
        f"[{m.get('role')}] {str(m.get('content') or '')[:1500]}" for m in task.context[-8:]
    )


def _instructions_text(task: Task) -> str:
    return str(getattr(task, "instructions", "") or "").strip() or "(none)"


async def make_rubric(
    llm: LLMCall, domain: ChatDomain, task: Task, params: RoleParams, *, min_criteria: int, max_criteria: int
) -> List[Dict[str, Any]]:
    prompt = domain.texts["judge_rubric"].format(
        domain_brief=domain.briefs["judge"],
        min_criteria=min_criteria,
        max_criteria=max_criteria,
        instructions=_instructions_text(task),
        context=_context_text(task),
        request=clip_text(task.request, params.text_max_chars),
    )
    raw = await ask_json(llm, None, prompt, params)
    items = raw.get("criteria") if isinstance(raw, dict) else raw
    out: List[Dict[str, Any]] = []
    for i, c in enumerate(items or []):
        if isinstance(c, dict) and str(c.get("text") or "").strip():
            try:
                weight = float(c.get("weight", 1) or 1)
            except (TypeError, ValueError):
                weight = 1.0
            out.append({"id": str(c.get("id") or f"c{i + 1}"), "text": str(c["text"]).strip(), "weight": max(weight, 0.0)})
        elif isinstance(c, str) and c.strip():
            out.append({"id": f"c{i + 1}", "text": c.strip(), "weight": 1.0})
    out = out[:max_criteria]
    if not out:
        raise ValueError("rubric came back empty")
    return out


async def grade(
    llm: LLMCall, domain: ChatDomain, task: Task, rubric: List[Dict[str, Any]], trial: Trial, params: RoleParams
) -> Trial:
    total = sum(float(c.get("weight", 1)) for c in rubric) or 1.0
    trial.weight = total
    if trial.failed:
        trial.reward = 0.0
        trial.verdicts = [{"id": c["id"], "text": c["text"], "pass": False, "why": "no answer"} for c in rubric]
        trial.graded = True
        return trial
    prompt = domain.texts["judge_grade"].format(
        domain_brief=domain.briefs["judge"],
        instructions=_instructions_text(task),
        request=clip_text(task.request, params.text_max_chars),
        rubric=json.dumps([{"id": c["id"], "text": c["text"]} for c in rubric], ensure_ascii=False, indent=1),
        tool_runs=(clip_text(json.dumps(trial.tool_runs[:20], ensure_ascii=False), params.text_max_chars)
                   if trial.tool_runs else "(none)"),
        answer=clip_text(trial.answer, params.text_max_chars),
    )
    raw = await ask_json(llm, None, prompt, params)
    verdicts = raw.get("verdicts") if isinstance(raw, dict) else raw
    by_id = {str(v.get("id")): v for v in (verdicts or []) if isinstance(v, dict)}
    passed = 0.0
    out: List[Dict[str, Any]] = []
    for c in rubric:
        v = by_id.get(c["id"]) or {}
        ok = v.get("pass") is True or str(v.get("pass")).strip().lower() == "true"
        if ok:
            passed += float(c.get("weight", 1))
        out.append({"id": c["id"], "text": c["text"], "pass": ok, "why": str(v.get("why") or "")[:300]})
    trial.verdicts = out
    trial.reward = passed / total
    trial.graded = True
    return trial
