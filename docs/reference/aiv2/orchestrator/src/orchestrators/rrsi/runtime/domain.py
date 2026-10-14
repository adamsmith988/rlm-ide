"""Hub-chat domain for RRSI: the Domain interface of the paper, backed by data
files under ``../domain`` (schema, briefs, prompts, constitution, patterns)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DOMAIN_DIR = Path(__file__).resolve().parent.parent / "domain"


@dataclass
class Task:
    """One evolve-set item: a user request with the conversation before it."""

    id: str
    request: str
    context: List[Dict[str, Any]] = field(default_factory=list)
    current: bool = False
    instructions: str = ""  # the chat's operator instructions (persona / system prompt)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "request": self.request, "context": self.context, "current": self.current,
                "instructions": self.instructions}


@dataclass
class Trial:
    task_id: str
    index: int
    answer: str = ""
    error: str = ""
    tokens: Optional[float] = None
    tool_runs: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_sec: float = 0.0
    reward: float = 0.0
    weight: float = 1.0
    verdicts: List[Dict[str, Any]] = field(default_factory=list)
    graded: bool = False
    reviewed: bool = False

    @property
    def failed(self) -> bool:
        return bool(self.error) or not self.answer.strip()

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Trial":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})


def _read(name: str) -> str:
    return (DOMAIN_DIR / name).read_text(encoding="utf-8")


class ChatDomain:
    name = "hub_chat"

    def __init__(self, directory: Optional[Path] = None) -> None:
        d = Path(directory or DOMAIN_DIR)
        self.dir = d
        self.schema: Dict[str, Any] = json.loads((d / "harness_schema.json").read_text(encoding="utf-8"))
        self.briefs: Dict[str, str] = json.loads((d / "briefs.json").read_text(encoding="utf-8"))
        self.critic_patterns: List[Tuple[str, str]] = [
            (str(p), str(why)) for p, why in json.loads((d / "critic_patterns.json").read_text(encoding="utf-8"))
        ]
        self.texts: Dict[str, str] = {
            p.stem: p.read_text(encoding="utf-8") for p in sorted(d.glob("*.md"))
        }

    # ------------------------------------------------------------ schema --
    @property
    def fields(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.schema.get("fields") or {})

    def field_component(self, name: str) -> Optional[str]:
        spec = self.fields.get(name)
        return str(spec.get("component")) if spec else None

    @property
    def component_signals(self) -> List[Tuple[str, List[str]]]:
        """Diff regexes -> component tag, derived from the schema."""
        by_comp: Dict[str, List[str]] = {}
        for fname, spec in self.fields.items():
            by_comp.setdefault(str(spec.get("component")), []).append(
                rf"(?m)^[+-] {re.escape(fname)} = "
            )
        return list(by_comp.items())

    def constitution(self) -> Tuple[str, str]:
        return self.texts["constitution"], self.texts["patterns"]

    # ---------------------------------------------------------- evidence --
    def regression_threshold(self, k: int) -> float:
        return 1.0 / max(1, k)

    def guards(self, incumbent: Any, candidate: Any) -> List[str]:
        """Non-compensatory: a candidate may not fail (crash / empty) more trials."""
        inc_failed = int((incumbent.extra or {}).get("failed_trials", 0))
        cand_failed = int((candidate.extra or {}).get("failed_trials", 0))
        if cand_failed > inc_failed:
            return [f"more failed policy trials than the incumbent ({cand_failed} > {inc_failed})"]
        return []

    def task_row(self, task: Task, trials: List[Trial]) -> str:
        mean = (sum(t.reward for t in trials) / len(trials)) if trials else 0.0
        failed = sum(1 for t in trials if t.failed)
        return (
            f"{task.id} | mean_reward={mean:.3f} | trials={len(trials)} | failed={failed} | "
            f"request={json.dumps(task.request[:160], ensure_ascii=False)}"
        )

    def render_trace(self, task: Task, trial: Trial, *, detail: bool = False) -> str:
        cap = None if detail else 2500
        answer = trial.answer if cap is None else trial.answer[:cap]
        request = task.request if cap is None else task.request[:cap]
        lines = [
            f"=== TASK {task.id} (trial {trial.index}) ===",
            "REQUEST: " + request,
        ]
        if task.context:
            ctx = task.context[-6:]
            lines.append("EARLIER TURNS: " + json.dumps(ctx, ensure_ascii=False)[: (None if detail else 1500)])
        if trial.tool_runs:
            lines.append("TOOL RUNS: " + json.dumps(trial.tool_runs, ensure_ascii=False)[:1500])
        if trial.error:
            lines.append("ERROR: " + trial.error)
        lines.append("ANSWER:\n" + (answer or "(empty)"))
        lines.append(f"=== GRADING: reward={trial.reward:.3f} weight={trial.weight} ===")
        for v in trial.verdicts:
            lines.append(f"- [{'PASS' if v.get('pass') else 'FAIL'}] {v.get('text') or v.get('id')}: {v.get('why', '')}")
        return "\n".join(lines)


@lru_cache(maxsize=1)
def default_domain() -> ChatDomain:
    return ChatDomain()
