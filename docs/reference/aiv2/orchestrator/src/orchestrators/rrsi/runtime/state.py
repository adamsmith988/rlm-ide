"""Durable per-conversation RRSI state (the paper's runs/<domain>/ layout).

    harness/v{n}.json     every incumbent version H_n
    frontier.json         incumbent pointer, cumulative-gain trajectory, round counter
    history.jsonl         L_t (vendored History)
    attribution.jsonl     prediction scoreboard
    global_analysis.json  stable failure-mode / habit names
    rubrics.json          judge rubric per task id
    r{t}/                 round artifacts (report, directives, variants, decisions)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.history import History


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


class RrsiState:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.history = History(self.root / "history.jsonl")

    # --------------------------------------------------------------- harness --
    def harness_path(self, version: int) -> Path:
        return self.root / "harness" / f"v{version}.json"

    def save_harness(self, version: int, harness: Dict[str, Any]) -> None:
        _write_json(self.harness_path(version), harness)

    def load_harness(self, version: int) -> Optional[Dict[str, Any]]:
        return _read_json(self.harness_path(version), None)

    # -------------------------------------------------------------- frontier --
    def frontier(self) -> Optional[Dict[str, Any]]:
        return _read_json(self.root / "frontier.json", None)

    def save_frontier(self, fr: Dict[str, Any]) -> None:
        _write_json(self.root / "frontier.json", fr)

    # -------------------------------------------------------------- evidence --
    def global_analysis(self) -> Dict[str, Any]:
        return _read_json(self.root / "global_analysis.json", {})

    def save_global_analysis(self, data: Dict[str, Any]) -> None:
        _write_json(self.root / "global_analysis.json", data)

    def rubrics(self) -> Dict[str, Any]:
        return _read_json(self.root / "rubrics.json", {})

    def save_rubric(self, task_id: str, rubric: List[Dict[str, Any]]) -> None:
        data = self.rubrics()
        data[task_id] = rubric
        _write_json(self.root / "rubrics.json", data)

    def scoreboard(self, n: int = 20) -> List[Dict[str, Any]]:
        p = self.root / "attribution.jsonl"
        if not p.is_file():
            return []
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        return rows[-n:]

    def append_attribution(self, rows: List[Dict[str, Any]]) -> None:
        p = self.root / "attribution.jsonl"
        with open(p, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------------------------------------------------------------- rounds --
    def round_dir(self, t: int) -> Path:
        d = self.root / f"r{t}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write(self, rel: str, data: Any) -> None:
        _write_json(self.root / rel, data)
