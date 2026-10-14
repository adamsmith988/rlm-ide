"""Prime-like auto-refine review gate (shouldRefine before /refine)."""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Dict, Optional

from src.orchestrators.rlm.runtime.harness import (
    history_for_prompt,
    overview_for_prompt,
)

logger = logging.getLogger(__name__)

ChatFn = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]

AUTO_REFINE_REVIEW_SYSTEM_PROMPT = """\
You are AIv2's automatic /refine review gate.

Decide whether this checkpoint should run /refine. Auto /refine writes the \
continual harness for THIS agent node + THIS model logical_id only. Approve \
ONLY when the trajectory has a reusable lesson for FUTURE unrelated runs of \
the same node+model (repeated failures, stable tactics, lasting constraints).

Reject and set shouldRefine=false for:
- one-shot facts, specs, IDs, filenames, artifact paths, or case details
- observations useful only for finishing the current user request
- transient tool outputs or intermediate results
- unsupported hypotheses / speculative lessons

Do not suggest promoting lessons to other nodes or models.
Prefer shouldRefine=false when unsure.

Return JSON only:
{
  "shouldRefine": true|false,
  "rationale": "short reason",
  "instructions": "optional concise instructions for /refine if shouldRefine is true"
}
"""


def _extract_json_object(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p2 = p.strip()
            if p2.startswith("json"):
                p2 = p2[4:].strip()
            if p2.startswith("{"):
                text = p2
                break
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


def parse_auto_refine_review(raw: str) -> Dict[str, Any]:
    record = _extract_json_object(raw)
    return {
        "shouldRefine": record.get("shouldRefine") is True,
        "rationale": str(record.get("rationale") or "No rationale provided.")[:500],
        "instructions": (
            str(record.get("instructions")).strip()
            if isinstance(record.get("instructions"), str)
            else None
        )
        or None,
    }


def _conversation_tail(messages: list, *, max_chars: int = 40_000) -> str:
    lines = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        if role == "system":
            continue
        content = str(m.get("content") or "")[:2000]
        lines.append(f"{role}: {content}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


async def review_auto_refine(
    session: Any,
    *,
    chat_fn: ChatFn,
    extract_assistant_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    reason: str,
    turns_since_last_review: int,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    store = getattr(session, "harness", None)
    harness_text = ""
    if store is not None:
        harness_text = store.overview_for_prompt()
    if not harness_text:
        harness_text = overview_for_prompt(store.get() if store else {})
    history = history_for_prompt(session.merged_refinement_history())
    user_prompt = "\n\n".join(
        [
            f"<trigger>\n{reason}; {turns_since_last_review} assistant turns since last auto-refine review\n</trigger>",
            f"<node_id>{getattr(session, 'node_id', '')}</node_id>",
            f"<model_logical_id>{getattr(session, 'model_logical_id', None) or getattr(session, 'model', '')}</model_logical_id>",
            f"<current_harness_state>\n{harness_text}\n</current_harness_state>",
            f"<refinement_history>\n{history}\n</refinement_history>",
            f"<conversation>\n{_conversation_tail(session.messages)}\n</conversation>",
            "Return shouldRefine=true when the trajectory contains evidence useful to future turns of this same node+model.",
        ]
    )
    body: Dict[str, Any] = {
        "messages": [
            {"role": "system", "content": AUTO_REFINE_REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt[:60000]},
        ],
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 4096,
        "model": model or session.model,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if session.context_length:
        from src.orchestrators.rlm.runtime.context_budget import apply_context_budget

        apply_context_budget(body, session.context_length, max_output_tokens=4096)
    data = await chat_fn(body)
    msg = extract_assistant_fn(data)
    raw = str(msg.get("content") or "")
    return parse_auto_refine_review(raw)
