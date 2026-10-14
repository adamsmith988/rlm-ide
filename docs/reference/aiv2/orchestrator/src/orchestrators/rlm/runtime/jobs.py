"""In-process goal / heartbeat / schedule jobs for an RLM session."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from src.orchestrators.rlm.runtime.autonomous import AutonomousRuntimeState


class SessionJobs:
    def __init__(self) -> None:
        self.goal: Optional[Dict[str, Any]] = None
        self.heartbeats: Dict[str, Dict[str, Any]] = {}
        self.schedules: Dict[str, Dict[str, Any]] = {}
        # Populated by RlmSession from cfg; kept here for status/handlers.
        self.autonomous: Optional[AutonomousRuntimeState] = None

    def goal_get(self) -> Optional[Dict[str, Any]]:
        return dict(self.goal) if self.goal else None

    def goal_create(self, objective: str, *, token_budget: int = 0) -> Dict[str, Any]:
        self.goal = {
            "id": f"goal_{uuid.uuid4().hex[:8]}",
            "objective": str(objective or "").strip(),
            "status": "active",
            "token_budget": int(token_budget or 0),
            "tokens_used": 0,
            "continuations_used": 0,
            "created_at": time.time(),
        }
        return dict(self.goal)

    def goal_complete(self) -> Dict[str, Any]:
        if not self.goal:
            return {"ok": False, "error": "no_goal"}
        self.goal["status"] = "complete"
        self.goal["completed_at"] = time.time()
        return dict(self.goal)

    def goal_is_active(self) -> bool:
        g = self.goal
        return bool(g) and str(g.get("status") or "") == "active" and bool(
            str(g.get("objective") or "").strip()
        )

    def goal_add_usage(self, *, tokens: int = 0) -> None:
        if not self.goal or str(self.goal.get("status") or "") != "active":
            return
        self.goal["tokens_used"] = int(self.goal.get("tokens_used") or 0) + max(0, int(tokens))

    def goal_mark_continuation(self) -> None:
        if not self.goal or str(self.goal.get("status") or "") != "active":
            return
        self.goal["continuations_used"] = int(self.goal.get("continuations_used") or 0) + 1

    def goal_mark_budget_limited(self) -> None:
        if not self.goal:
            return
        if str(self.goal.get("status") or "") == "active":
            self.goal["status"] = "budget_limited"

    def heartbeat_create(
        self, *, interval_sec: int, prompt: str = "", name: str = ""
    ) -> Dict[str, Any]:
        hid = f"hb_{uuid.uuid4().hex[:8]}"
        rec = {
            "id": hid,
            "name": name or hid,
            "interval_sec": max(30, int(interval_sec or 300)),
            "prompt": str(prompt or ""),
            "next_at": time.time() + max(30, int(interval_sec or 300)),
            "created_at": time.time(),
        }
        self.heartbeats[hid] = rec
        return dict(rec)

    def heartbeat_list(self) -> List[Dict[str, Any]]:
        return [dict(v) for v in self.heartbeats.values()]

    def heartbeat_delete(self, target: str) -> bool:
        tid = str(target or "").strip()
        if tid in self.heartbeats:
            self.heartbeats.pop(tid, None)
            return True
        for hid, rec in list(self.heartbeats.items()):
            if str(rec.get("name") or "") == tid:
                self.heartbeats.pop(hid, None)
                return True
        return False

    def due_heartbeats(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        ts = float(now if now is not None else time.time())
        due = []
        for rec in self.heartbeats.values():
            if float(rec.get("next_at") or 0) <= ts:
                rec["next_at"] = ts + max(30, int(rec.get("interval_sec") or 300))
                due.append(dict(rec))
        return due
