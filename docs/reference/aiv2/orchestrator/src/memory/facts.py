"""Long-term fact distillation: one turn + related known facts → add/update/delete ops."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..config import settings

log = logging.getLogger(__name__)

DEFAULT_FACT_PROMPT = """You maintain a long-term memory of durable facts about ONE user, shared across all of their chats.

Given the latest exchange and the facts already known, return JSON only:
{"add": ["<new fact>", ...], "update": [{"ref": <number>, "fact": "<corrected fact>"}], "delete": [<number>, ...]}

Store only durable, user-specific information worth recalling in a future chat:
identity (name, role, employer, location), stable preferences (language, answer style,
formats), ongoing projects/cases/goals with their identifiers, decisions, deadlines and
commitments, and explicit "remember this" requests.
Do NOT store: greetings, one-off questions, general knowledge, the assistant's own
suggestions, or anything already known unchanged. Text the user pasted, attached, or asked
to summarize/translate/analyse is material, not information about the user — never turn its
contents into facts about the user. Codes, numbers, dates, amounts and values that belong to a
document, file, note or task in this chat (an access code, a station code, a contract date, a
computed total) stay in that chat: store one only when the user says it is personally theirs
("my badge number is ..."). Likewise the details of a single request or transaction the
assistant is handling (a booking's route, dates and traveller count, an order to cancel, an
amount to transfer) belong to that request, not to the user: do not store them unless the user
states them as a standing preference ("I always fly from Porto").

Rules:
- Each fact is one short self-contained sentence naming its subject ("The user ...").
- Write each fact in the same language the user wrote in.
- If the exchange changes or contradicts a known fact, use "update" with that fact's ref
  (or "delete" when the user retracts it) instead of adding a duplicate.
- When nothing qualifies return {"add": [], "update": [], "delete": []}."""


@dataclass
class FactOps:
    add: List[str] = field(default_factory=list)
    update: List[Dict[str, Any]] = field(default_factory=list)  # {"id", "fact"}
    delete: List[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.add or self.update or self.delete)


def _clip(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return f"{text[:half]}\n…\n{text[-half:]}"


_NUMERIC_TOKEN = re.compile(r"[^\W_]*\d[^\s,;:!?()\[\]{}\"'«»]*")


def _norm(text: str) -> str:
    return "".join(str(unicodedata.digit(c, c)) if c.isdigit() else c for c in str(text or "")).casefold()


def _numeric_tokens(text: str) -> List[str]:
    return [t.rstrip(".-/") for t in _NUMERIC_TOKEN.findall(_norm(text)) if t.rstrip(".-/")]


def grounded_in_user(fact: str, user_message: str) -> bool:
    """True when every number-bearing token of ``fact`` also occurs in the user's message."""
    source = _norm(user_message)
    return all(tok in source for tok in _numeric_tokens(fact))


def _ground(ops: FactOps, user_message: str) -> FactOps:
    ops.add = [f for f in ops.add if grounded_in_user(f, user_message)]
    ops.update = [u for u in ops.update if grounded_in_user(u["fact"], user_message)]
    return ops


def _parse(raw: Any, known: List[Dict[str, Any]], max_items: int) -> FactOps:
    ops = FactOps()
    if not isinstance(raw, dict):
        return ops

    def ref_id(ref: Any) -> str:
        try:
            idx = int(ref) - 1
        except (TypeError, ValueError):
            return ""
        return str(known[idx]["id"]) if 0 <= idx < len(known) else ""

    for fact in raw.get("add") or []:
        if isinstance(fact, str) and fact.strip():
            ops.add.append(fact.strip())
    for item in raw.get("update") or []:
        if isinstance(item, dict) and isinstance(item.get("fact"), str) and item["fact"].strip():
            mid = ref_id(item.get("ref"))
            if mid:
                ops.update.append({"id": mid, "fact": item["fact"].strip()})
            else:
                ops.add.append(item["fact"].strip())
    for ref in raw.get("delete") or []:
        mid = ref_id(ref)
        if mid:
            ops.delete.append(mid)
    ops.add = ops.add[:max_items]
    ops.update = ops.update[:max_items]
    return ops


_fact_clients: Dict[tuple, Any] = {}


def _fact_llm() -> Any:
    from ..llm.gateway_llm_client import GatewayLLMClient
    from ..orchestrators._shared.llm_roles import gateway_llm

    key = ((settings.MEMORY_BINDING_SERVICE or "orchestrator").strip(), (settings.MEMORY_FACT_PURPOSE or "chat").strip())
    if key not in _fact_clients:
        _fact_clients[key] = GatewayLLMClient(service=key[0], purpose=key[1])
    return gateway_llm(_fact_clients[key])


def fact_prompt() -> str:
    return (settings.MEMORY_FACT_PROMPT or "").strip() or DEFAULT_FACT_PROMPT


async def distill_facts(
    user_message: str, assistant_response: str, known: List[Dict[str, Any]],
) -> FactOps:
    """``known`` items: ``{"id", "content"}``; refs in the reply are 1-based indexes into it."""
    from ..orchestrators._shared.llm_roles import extract_json

    limit = int(settings.MEMORY_FACT_MAX_INPUT_CHARS or 8000)
    known_block = "\n".join(f"{i}. {k['content']}" for i, k in enumerate(known, 1)) or "(none)"
    user_block = (
        f"Known facts:\n{known_block}\n\n"
        f"User said:\n{_clip(user_message, limit)}\n\n"
        f"Assistant replied:\n{_clip(assistant_response, limit // 4)}"
    )
    reply = await _fact_llm()(
        [{"role": "system", "content": fact_prompt()}, {"role": "user", "content": user_block}],
        temperature=settings.MEMORY_FACT_TEMPERATURE,
        max_tokens=int(settings.MEMORY_FACT_MAX_TOKENS or 0) or None,
        enable_thinking=settings.MEMORY_FACT_ENABLE_THINKING,
    )
    if not reply.ok:
        log.warning("memory: fact distillation failed: %s", reply.error)
        return FactOps()
    try:
        raw = extract_json(reply.text)
    except ValueError:
        log.debug("memory: fact distillation returned no JSON: %s", reply.text[:200])
        return FactOps()
    if isinstance(raw, dict) and "facts" in raw and "add" not in raw:
        raw = {"add": raw.get("facts")}
    ops = _parse(raw, known, int(settings.MEMORY_FACT_MAX_PER_TURN or 8))
    return _ground(ops, user_message) if settings.MEMORY_FACT_REQUIRE_USER_GROUNDING else ops


def describe_ops(ops: FactOps) -> str:
    return json.dumps({"add": len(ops.add), "update": len(ops.update), "delete": len(ops.delete)})
