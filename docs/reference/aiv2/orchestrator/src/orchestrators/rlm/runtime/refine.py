"""Refine harness from an RLM session trajectory (Prime-aligned)."""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.orchestrators.rlm.runtime.harness import (
    history_for_prompt,
    normalize_entry_kind,
)
from src.orchestrators.rlm.runtime.session import RlmSession

logger = logging.getLogger(__name__)

ChatFn = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]

_REFINE_PROMPT = """\
You are AIv2's /refine continual harness subsystem.

Your job is to improve the editable continual harness state from the current trajectory.
Never rewrite the base system prompt. Emit precise Create, Update, or Delete edits.

Continual harness kinds (use these exact kind names):
- note: supplemental prompt notes / narrow behavioral policy addendums that
  change how this agent should behave on future runs.
- memory: durable, cross-run lessons only — repeated failures, reusable tactics,
  stable preferences/constraints, or decisions that will still matter on the
  NEXT unrelated task for this same node+model.
- skill_desc: reusable Python call descriptions. Create/update MUST include
  `reference` (type/import/callable/call_pattern) and `arguments`.
- subagent_spec: reusable delegation specs (purpose, when to invoke).

This harness is injected into EVERY future run of this node+model. It is NOT a
session scratchpad. Record only what improves future performance.

DO NOT store as memory/note (leave edits empty instead):
- one-shot facts, specs, IDs, filenames, artifact paths, case details, or
  entity attributes that appeared only in this trajectory
- task-specific content the agent can re-read from pool/files/transcript
- transient observations, intermediate results, or "what we found this time"
- anything useful only for finishing the current user request

Test before each create/update: "Would this still help on a different future
task with different inputs?" If no / unclear → do not store it.
Prefer DELETE of existing one-off or stale memories over adding new ones.
Prefer an empty edits array over speculative or one-off memories.

Return ONLY JSON:
{
  "summary": "one sentence",
  "rationale": "why these edits are justified by trajectory evidence",
  "expectedOutcome": "what should improve on future runs",
  "edits": [
    {
      "action": "create|update|delete",
      "kind": "note|memory|skill_desc|subagent_spec",
      "id": "stable id for update/delete, optional for create",
      "title": "required for create/update",
      "content": "required for create/update",
      "reference": {},
      "arguments": {},
      "reason": "why this edit helps FUTURE runs (not just this task)"
    }
  ]
}
Max 5 edits. Each content <= 1500 chars. Evidence-backed only.
If no useful cross-run edit is justified, return an empty edits array with a rationale.
"""


async def refine_session(
    session: RlmSession,
    *,
    chat_fn: ChatFn,
    extract_assistant_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    model: Optional[str] = None,
    instructions: Optional[str] = None,
    skip_review: bool = False,
    write_scope: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply harness refine to the session's node+model store.

    ``write_scope`` is ignored (legacy); all writes go to node+model harness.
    """
    _ = skip_review, write_scope
    if not session.refine_enabled:
        return {"ok": False, "error": "refine_disabled"}

    store = session.harness
    snap_id = store.snapshot(label="pre_refine")
    trajectory = {
        "last_content": session.last_content,
        "status": session.status,
        "events": session.events[-80:],
        "node_id": session.node_id,
        "model_logical_id": session.model_logical_id or session.model,
        "harness_overview": store.overview_for_prompt(),
        "harness": store.list_entries(),
        "refinement_history": history_for_prompt(session.merged_refinement_history()),
        "messages_tail": _slim_messages(session.messages[-20:]),
    }
    user_bits = [json.dumps(trajectory, ensure_ascii=False)[:60000]]
    if instructions:
        user_bits.insert(
            0,
            f"<user_refine_instructions>\n{instructions}\n</user_refine_instructions>",
        )
    body = {
        "messages": [
            {"role": "system", "content": _REFINE_PROMPT},
            {"role": "user", "content": "\n\n".join(user_bits)},
        ],
        "stream": False,
        "temperature": 0.2,
        "model": model or session.model,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if session.context_length:
        from src.orchestrators.rlm.runtime.context_budget import apply_context_budget

        apply_context_budget(
            body,
            session.context_length,
            max_output_tokens=session.max_output_tokens,
        )
    data = await chat_fn(body)
    msg = extract_assistant_fn(data)
    raw = str(msg.get("content") or "")
    parsed = _parse_proposal(raw)
    edits = parsed.get("edits") or []
    applied: List[Dict[str, Any]] = []
    errors: List[str] = []
    for edit in edits[:5]:
        try:
            applied.append(_apply_edit(store, edit))
        except Exception as exc:
            errors.append(str(exc))
    changes = [
        f"{a.get('op') or a.get('action')} {a.get('kind')}:"
        f"{a.get('entry_id') or (a.get('entry') or {}).get('id')}"
        for a in applied
    ]
    store.record_refinement(
        {
            "trigger": parsed.get("summary") or "refine",
            "summary": parsed.get("summary") or "",
            "changes": changes,
            "evidence": parsed.get("rationale") or "",
            "outcome": parsed.get("expectedOutcome") or "",
            "scope": "node_model",
            "node_id": session.node_id,
            "model_logical_id": session.model_logical_id or session.model,
        }
    )
    return {
        "ok": True,
        "snapshot_id": snap_id,
        "applied": applied,
        "errors": errors,
        "rationale": parsed.get("rationale") or "",
        "summary": parsed.get("summary") or "",
        "expectedOutcome": parsed.get("expectedOutcome") or "",
        "write_scope": "node_model",
        "harness_path": session.harness_path(),
        "skip_review": True,
    }


def rollback_refine(session: RlmSession, snapshot_id: str) -> Dict[str, Any]:
    ok = session.harness.rollback(str(snapshot_id or ""))
    return {"ok": ok, "snapshot_id": snapshot_id}


def _slim_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    out = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        item = {"role": m.get("role"), "content": str(m.get("content") or "")[:2000]}
        if m.get("tool_calls"):
            item["tool_calls"] = True
        out.append(item)
    return out


def _parse_proposal(raw: str) -> Dict[str, Any]:
    from src.orchestrators.rlm.runtime.refine_review import _extract_json_object

    data = _extract_json_object(raw)
    edits = data.get("edits") if isinstance(data.get("edits"), list) else []
    return {
        "summary": str(data.get("summary") or "")[:500],
        "rationale": str(data.get("rationale") or "")[:500],
        "expectedOutcome": str(data.get("expectedOutcome") or "")[:500],
        "edits": [e for e in edits if isinstance(e, dict)],
    }


def _apply_edit(store: Any, edit: Dict[str, Any]) -> Dict[str, Any]:
    action = str(edit.get("action") or edit.get("op") or "upsert").strip().lower()
    if action in ("upsert", "create", "update"):
        op = "upsert" if action != "delete" else "delete"
    elif action == "delete":
        op = "delete"
    else:
        op = "upsert"
    kind = normalize_entry_kind(edit.get("kind"))
    entry_id = edit.get("id") or edit.get("entry_id")
    if op == "delete":
        ok = store.delete(kind=kind, entry_id=str(entry_id or ""))
        return {
            "op": "delete",
            "action": "delete",
            "kind": kind,
            "ok": ok,
            "entry_id": entry_id,
        }
    content = str(edit.get("content") or "").strip()
    if len(content) > 1500:
        content = content[:1500]
    meta: Dict[str, Any] = {}
    if isinstance(edit.get("metadata"), dict):
        meta.update(edit["metadata"])
    if kind == "skill_desc":
        ref = edit.get("reference")
        args = edit.get("arguments")
        if not isinstance(ref, dict) or not ref:
            raise ValueError("skill_desc_missing_reference")
        meta["reference"] = ref
        meta["arguments"] = args if isinstance(args, dict) else {}
    entry = store.upsert(
        kind=kind,
        content=content,
        entry_id=entry_id,
        title=str(edit.get("title") or "")[:200],
        meta=meta or None,
    )
    return {
        "op": "upsert",
        "action": action,
        "kind": kind,
        "entry": entry,
        "entry_id": entry.get("id"),
    }
