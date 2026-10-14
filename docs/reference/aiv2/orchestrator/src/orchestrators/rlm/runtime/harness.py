"""Continual Harness ledger: supplemental notes/memories/specs + refine snapshots."""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

ENTRY_KINDS = frozenset({"note", "memory", "skill_desc", "subagent_spec"})
KIND_ALIASES = {
    "note": "note",
    "prompt": "note",
    "memory": "memory",
    "skill_desc": "skill_desc",
    "skill": "skill_desc",
    "subagent_spec": "subagent_spec",
    "subagent": "subagent_spec",
}
_MAX_EVENTS = 200
_MAX_REFINEMENTS = 50


def normalize_entry_kind(raw: Any) -> str:
    v = str(raw or "").strip().lower().replace("-", "_")
    mapped = KIND_ALIASES.get(v, v)
    if mapped not in ENTRY_KINDS:
        raise ValueError(f"invalid_kind:{raw}")
    return mapped

_STORES: Dict[str, "HarnessStore"] = {}
_REGISTRY_LOCK = threading.Lock()


def empty_harness_state() -> Dict[str, Any]:
    return {
        "version": 1,
        "notes": [],
        "memories": [],
        "skill_descs": [],
        "subagent_specs": [],
        "events": [],
        "snapshots": [],
        "refinements": [],
    }


def _bucket_for_kind(kind: str) -> str:
    return {
        "note": "notes",
        "memory": "memories",
        "skill_desc": "skill_descs",
        "subagent_spec": "subagent_specs",
    }[kind]


class HarnessStore:
    """In-memory harness with optional JSON path persistence."""

    def __init__(self, *, path: Optional[str] = None, initial: Optional[Dict[str, Any]] = None):
        self.path = path
        self._lock = threading.RLock()
        if initial and isinstance(initial, dict):
            self.state = copy.deepcopy(initial)
        elif path:
            self.state = self._load(path)
        else:
            self.state = empty_harness_state()

    @classmethod
    def shared(cls, path: str) -> "HarnessStore":
        """One store instance per realpath for flow/user scoped harness files."""
        key = os.path.realpath(str(path or "").strip() or ".")
        with _REGISTRY_LOCK:
            existing = _STORES.get(key)
            if existing is not None:
                return existing
            store = cls(path=key)
            _STORES[key] = store
            return store

    @classmethod
    def clear_shared_for_tests(cls) -> None:
        with _REGISTRY_LOCK:
            _STORES.clear()

    @staticmethod
    def _load(path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                base = empty_harness_state()
                for k in base:
                    if k in data:
                        base[k] = data[k]
                return base
        except FileNotFoundError:
            pass
        except Exception:
            pass
        return empty_harness_state()

    def _trim_events(self) -> None:
        events = list(self.state.get("events") or [])
        if len(events) > _MAX_EVENTS:
            self.state["events"] = events[-_MAX_EVENTS:]

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            try:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                tmp = (
                    f"{self.path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
                )
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(self.state, fh, ensure_ascii=False, indent=2)
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())
                    except OSError:
                        pass
                os.replace(tmp, self.path)
            except Exception:
                try:
                    if "tmp" in locals() and os.path.isfile(tmp):
                        os.unlink(tmp)
                except Exception:
                    pass

    def get(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self.state)

    def list_entries(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            if kind:
                if kind not in ENTRY_KINDS:
                    return []
                return copy.deepcopy(list(self.state.get(_bucket_for_kind(kind)) or []))
            out: List[Dict[str, Any]] = []
            for k in ENTRY_KINDS:
                for item in self.state.get(_bucket_for_kind(k)) or []:
                    if isinstance(item, dict):
                        out.append({**item, "kind": k})
            return out

    def upsert(
        self,
        *,
        kind: str,
        content: str,
        entry_id: Optional[str] = None,
        title: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            kind = str(kind or "").strip().lower()
            if kind not in ENTRY_KINDS:
                raise ValueError(f"invalid_harness_kind:{kind}")
            text = str(content or "").strip()
            if not text:
                raise ValueError("empty_harness_content")
            if len(text) > 8000:
                text = text[:8000]
            bucket = _bucket_for_kind(kind)
            items: List[Dict[str, Any]] = list(self.state.get(bucket) or [])
            eid = str(entry_id or "").strip() or f"{kind}_{uuid.uuid4().hex[:10]}"
            now = time.time()
            entry = {
                "id": eid,
                "title": str(title or "")[:200],
                "content": text,
                "meta": meta if isinstance(meta, dict) else {},
                "updated_at": now,
            }
            replaced = False
            for i, old in enumerate(items):
                if isinstance(old, dict) and str(old.get("id") or "") == eid:
                    entry["created_at"] = old.get("created_at") or now
                    items[i] = entry
                    replaced = True
                    break
            if not replaced:
                entry["created_at"] = now
                items.append(entry)
            self.state[bucket] = items
            self.state.setdefault("events", []).append(
                {
                    "ts": now,
                    "op": "upsert",
                    "kind": kind,
                    "id": eid,
                }
            )
            self._trim_events()
            self.save()
            return copy.deepcopy(entry)

    def delete(self, *, kind: str, entry_id: str) -> bool:
        with self._lock:
            kind = str(kind or "").strip().lower()
            if kind not in ENTRY_KINDS:
                return False
            bucket = _bucket_for_kind(kind)
            eid = str(entry_id or "").strip()
            items = list(self.state.get(bucket) or [])
            new_items = [
                x for x in items if not (isinstance(x, dict) and str(x.get("id")) == eid)
            ]
            if len(new_items) == len(items):
                return False
            self.state[bucket] = new_items
            self.state.setdefault("events", []).append(
                {"ts": time.time(), "op": "delete", "kind": kind, "id": eid}
            )
            self._trim_events()
            self.save()
            return True

    def record_refinement(self, event: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            rec = dict(event or {})
            rec.setdefault("id", f"refine_{uuid.uuid4().hex[:10]}")
            rec.setdefault("ts", time.time())
            items = list(self.state.get("refinements") or [])
            items.append(rec)
            if len(items) > _MAX_REFINEMENTS:
                items = items[-_MAX_REFINEMENTS:]
            self.state["refinements"] = items
            self.save()
            return copy.deepcopy(rec)

    def list_refinements(self, *, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            items = [x for x in (self.state.get("refinements") or []) if isinstance(x, dict)]
            return copy.deepcopy(items[-max(1, int(limit)) :])

    def snapshot(self, *, label: str = "") -> str:
        with self._lock:
            sid = f"snap_{uuid.uuid4().hex[:12]}"
            snap = {
                "id": sid,
                "label": str(label or "")[:200],
                "ts": time.time(),
                "state": copy.deepcopy(
                    {
                        "notes": self.state.get("notes") or [],
                        "memories": self.state.get("memories") or [],
                        "skill_descs": self.state.get("skill_descs") or [],
                        "subagent_specs": self.state.get("subagent_specs") or [],
                    }
                ),
            }
            snaps = list(self.state.get("snapshots") or [])
            snaps.append(snap)
            self.state["snapshots"] = snaps[-20:]
            self.state.setdefault("events", []).append(
                {"ts": time.time(), "op": "snapshot", "id": sid, "label": label}
            )
            self._trim_events()
            self.save()
            return sid

    def rollback(self, snapshot_id: str) -> bool:
        with self._lock:
            sid = str(snapshot_id or "").strip()
            for snap in reversed(list(self.state.get("snapshots") or [])):
                if not isinstance(snap, dict):
                    continue
                if str(snap.get("id") or "") != sid:
                    continue
                body = snap.get("state") if isinstance(snap.get("state"), dict) else {}
                for key in ("notes", "memories", "skill_descs", "subagent_specs"):
                    self.state[key] = copy.deepcopy(body.get(key) or [])
                self.state.setdefault("events", []).append(
                    {"ts": time.time(), "op": "rollback", "id": sid}
                )
                self._trim_events()
                self.save()
                return True
            return False

    def overlay_text(self, *, max_chars: int = 6000) -> str:
        """Render supplemental harness text for system context injection."""
        with self._lock:
            parts: List[str] = []
            for kind in ("note", "memory", "skill_desc", "subagent_spec"):
                for item in self.state.get(_bucket_for_kind(kind)) or []:
                    if not isinstance(item, dict):
                        continue
                    title = str(item.get("title") or "").strip()
                    content = str(item.get("content") or "").strip()
                    if not content:
                        continue
                    head = f"[{kind}] {title}".strip() if title else f"[{kind}]"
                    parts.append(f"{head}\n{content}")
            text = "\n\n".join(parts).strip()
            if len(text) > max_chars:
                return text[: max_chars - 20] + "\n…[truncated]"
            return text

    def overview_for_prompt(self, *, max_entries: int = 40, max_content: int = 240) -> str:
        with self._lock:
            return overview_for_prompt(
                self.state, max_entries=max_entries, max_content=max_content
            )


def overview_for_prompt(
    state: Dict[str, Any],
    *,
    max_entries: int = 40,
    max_content: int = 240,
) -> str:
    lines: List[str] = []
    for kind in ("note", "memory", "skill_desc", "subagent_spec"):
        entries = [
            x
            for x in (state.get(_bucket_for_kind(kind)) or [])
            if isinstance(x, dict)
        ]
        lines.append(f"{kind}: {len(entries)}")
        for entry in entries[:max_entries]:
            content = re.sub(r"\s+", " ", str(entry.get("content") or "")).strip()
            if len(content) > max_content:
                content = content[:max_content]
            title = str(entry.get("title") or entry.get("id") or "")[:80]
            eid = str(entry.get("id") or "")
            lines.append(f"- [{eid}] {title}: {content}")
        if len(entries) > max_entries:
            lines.append(f"- +{len(entries) - max_entries} more {kind} entries")
    refinements = [x for x in (state.get("refinements") or []) if isinstance(x, dict)]
    lines.append(f"recent refinements: {len(refinements)}")
    for event in refinements[-5:]:
        trigger = str(event.get("trigger") or event.get("summary") or "")[:180]
        changes = event.get("changes") or []
        change_s = ", ".join(str(c) for c in changes[:8]) if isinstance(changes, list) else ""
        lines.append(f"- [{event.get('id')}] {trigger}: {change_s}")
    return "\n".join(lines).strip() or "No saved harness entries yet."


def history_for_prompt(events: List[Dict[str, Any]], *, limit: int = 20) -> str:
    items = [e for e in (events or []) if isinstance(e, dict)][-limit:]
    if not items:
        return "No prior refinement history."
    parts: List[str] = []
    for item in items:
        edits = item.get("changes") or item.get("applied") or []
        if isinstance(edits, list):
            edit_s = ", ".join(str(x)[:80] for x in edits[:12])
        else:
            edit_s = str(edits)[:200]
        outcome = str(item.get("outcome") or item.get("expectedOutcome") or "")[:240]
        summary = str(item.get("summary") or item.get("trigger") or "")[:240]
        parts.append(
            f"[{item.get('id')}] {summary}\n{edit_s}\nExpected outcome: {outcome}"
        )
    return "\n\n".join(parts)


def merge_overlay_text(
    stores: List[Optional["HarnessStore"]],
    *,
    labels: Optional[List[str]] = None,
    max_chars: int = 6000,
) -> str:
    """Union global ∪ user ∪ flow ∪ session harness text for system injection."""
    parts: List[str] = []
    for i, store in enumerate(stores):
        if store is None:
            continue
        body = store.overlay_text(max_chars=max_chars)
        if not body:
            continue
        label = ""
        if labels and i < len(labels) and labels[i]:
            label = str(labels[i])
        parts.append(f"### {label}\n{body}" if label else body)
    text = "\n\n".join(parts).strip()
    if len(text) > max_chars:
        return text[: max_chars - 20] + "\n…[truncated]"
    return text
