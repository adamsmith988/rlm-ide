"""
Pinned Conversation Facts
=========================

A deterministic anchor that survives every summarization / window-trim pass.
Whatever happens to the rolling chat window, these facts are *always*
re-injected into the system prompt so the LLM cannot "forget":

- **Goal**          : the user's first non-greeting request in the session
- **Constraints**   : "must / must not / فقط / حتما / نباید / avoid / بدون"
- **Decisions**     : explicit assistant commitments ("I'll do X", "تصمیم گرفتم")
- **Files**         : compact list of session files (from registry)

The point: the LLM may compress / paraphrase the rolling history, but the
*pinned* fields are short, structured and stable across turns — exactly the
"milestone retention" pattern that production LangGraph agents use to fight
mid-conversation drift and repetition.

Implementation notes
--------------------
* Pure heuristics / regex by default — never fails, never adds latency.
* Optional LLM enrichment behind `PINNED_FACTS_LLM_ENABLED` (off by default)
  with a hard timeout and graceful fallback.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .config import settings

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Patterns                                                                    #
# --------------------------------------------------------------------------- #

_GREETING_PATTERN = re.compile(
    r"^\s*(سلام|درود|صبح بخیر|عصر بخیر|شب بخیر|خسته نباشید|hi|hello|hey|"
    r"good morning|good evening|howdy|hola|greetings|what'?s up|"
    r"ممنون|مرسی|تشکر|سپاس|thanks|thank you|thx|ty)\b",
    re.IGNORECASE | re.UNICODE,
)

# Keywords that signal a constraint / requirement on the user side.
_CONSTRAINT_PATTERNS = [
    re.compile(r"\b(must|should|need to|have to|require[ds]?|only|don't|do not|avoid|never|always|prefer)\b",
               re.IGNORECASE),
    re.compile(r"(باید|نباید|حتما|حتماً|فقط|تنها|بدون|ترجیحاً|ترجیحا|اجبار|الزام)",
               re.UNICODE),
]

# Keywords that signal an explicit decision / commitment by the assistant.
_DECISION_PATTERNS = [
    re.compile(r"\b(I will|I'll|I'm going to|let's|we'll|decision is|chose|chosen|decided|i decided)\b",
               re.IGNORECASE),
    re.compile(r"(تصمیم گرفتم|انتخاب کردم|انجام می‌دهم|انجام میدم|پیش می‌رم|قرار شد|ادامه می‌دهیم)",
               re.UNICODE),
]

_QUESTION_PATTERN = re.compile(r"[?؟]\s*$")

# Local stopword-ish set used only by the "is this user message substantive?"
# check below. Kept tiny on purpose — its only job is to avoid picking a
# pure follow-up like "ممنون توضیح بده" as the conversation goal.
_FOLLOWUP_MARKERS = {
    "بیشتر", "کمتر", "دیگه", "دیگر", "ادامه", "بازم", "توضیح", "مثال",
    "بده", "بزن", "بنویس", "بگو", "لطفا", "لطفاً", "ممنون", "مرسی",
    "more", "less", "another", "again", "explain", "example", "examples",
    "tell", "show", "give", "please", "thanks", "thank",
}


def _looks_substantive(text: str) -> bool:
    """A user message is considered substantive when it carries at least
    two non-courtesy tokens of length >= 3 — i.e. it introduces actual
    content rather than being a pure continuation marker."""
    if not text:
        return False
    body = text.strip()
    if len(body) < 6:
        return False
    if _is_greeting(body):
        return False
    tokens = re.findall(r"[\w\u0600-\u06FF]+", body.lower())
    real = [t for t in tokens if len(t) >= 3 and t not in _FOLLOWUP_MARKERS]
    return len(real) >= 2


# --------------------------------------------------------------------------- #
# Data class                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class PinnedFacts:
    """Stable anchor for conversation coherence."""
    goal: str = ""
    constraints: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    source: str = "heuristic"   # "heuristic" | "llm" | "merged" | "empty"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def is_empty(self) -> bool:
        return not (
            self.goal
            or self.constraints
            or self.decisions
            or self.files
            or self.open_questions
        )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _is_greeting(text: str) -> bool:
    """Return True only when the *whole* line is a greeting/thanks.

    "سلام دانته نویسنده چه کتاب‌هایی هست؟" must NOT count as a greeting,
    otherwise the goal-extraction loop would skip it and the pinned-facts
    anchor would keep pointing at the original (now-stale) topic — exactly
    the regression that produced the admin/admin123 verbatim-repeat bug.
    """
    if not text:
        return True
    head = text.strip().splitlines()[0]
    m = _GREETING_PATTERN.match(head)
    if not m:
        return False
    rest = head[m.end():].strip(" \t,،.!?؟:،\u200c-")
    return len(rest) < 4


def _first_sentence(text: str, max_chars: int = 240) -> str:
    if not text:
        return ""
    text = text.strip().replace("\n", " ").replace("\r", " ")
    # Prefer first ?/.|!/؟ — gives a self-contained statement.
    m = re.search(r"[\.\!\?\؟]", text)
    if m:
        return text[: m.end()].strip()[:max_chars]
    return text[:max_chars]


def _matches_any(text: str, patterns: List[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)


def _dedup_keep_order(items: List[str], cap: int) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for it in items:
        key = re.sub(r"\s+", " ", it).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it.strip())
        if len(out) >= cap:
            break
    return out


# --------------------------------------------------------------------------- #
# Heuristic extractor (default, never fails)                                  #
# --------------------------------------------------------------------------- #

def extract_pinned_facts_heuristic(
    chat_history: List[Dict[str, Any]],
    current_message: Optional[str] = None,
    file_registry: Optional[List[Dict[str, Any]]] = None,
    conversation_state: Any = None,
) -> PinnedFacts:
    """
    Pure-Python deterministic extraction. Zero LLM calls, zero failure modes.

    Args:
        chat_history: list of `{"role": ..., "content": ..., "attachments": ...}`
                      dicts in chronological order (oldest first).
        current_message: the user's message for this turn (not yet in history).
        file_registry: chronological session-file list.
        conversation_state: optional ``ConversationState``. When ``topic_changed``
                      is True, the pinned anchor is *reset* to the new topic so
                      we don't drag the stale goal/files into the new prompt.
    """
    facts = PinnedFacts(source="heuristic")

    topic_changed = bool(getattr(conversation_state, "topic_changed", False))

    # --- Goal -------------------------------------------------------------- #
    # The "current goal" is the user's *current* focus, not the topic that
    # started the session 30 turns ago. We pick the MOST RECENT substantial
    # user message (≥2 non-courtesy tokens). This handles three regressions
    # the older "first message wins" rule produced:
    #   1. After a topic switch the new question becomes the anchor.
    #   2. After a long conversation the goal tracks the user's current focus.
    #   3. After a buggy LLM turn (e.g. the admin/admin123 King-Klou repeat)
    #      the user's re-asked question becomes the goal again.
    user_messages = [m for m in chat_history if m.get("role") == "user"]
    candidate: Optional[str] = None
    if topic_changed and current_message and not _is_greeting(current_message):
        candidate = current_message.strip()
    else:
        for msg in reversed(user_messages):
            text = (msg.get("content") or "").strip()
            if not _looks_substantive(text):
                continue
            candidate = text
            break
        if not candidate and current_message and not _is_greeting(current_message):
            candidate = current_message.strip()
    if candidate:
        facts.goal = _first_sentence(candidate, max_chars=280)

    # On a topic switch we deliberately drop the historical
    # constraints/decisions/files — they belong to the previous topic and
    # would otherwise keep biasing the LLM. Constraints the user *re-states*
    # in the current message are still picked up by the loop below.
    if topic_changed:
        facts.source = "heuristic+topic_reset"

    # --- Constraints ------------------------------------------------------- #
    # Only look at user messages — assistant text rarely contains *new* user
    # constraints. Cap at config.PINNED_FACTS_MAX_CONSTRAINTS. On a topic
    # switch we only consider the *current* message; older constraints
    # belonged to the previous topic.
    cap_c = getattr(settings, "PINNED_FACTS_MAX_CONSTRAINTS", 6)
    raw_constraints: List[str] = []
    if not topic_changed:
        for msg in user_messages:
            text = (msg.get("content") or "").strip()
            if not text:
                continue
            for line in re.split(r"[\n\.؟\?\!]+", text):
                line = line.strip()
                if 4 <= len(line) <= 220 and _matches_any(line, _CONSTRAINT_PATTERNS):
                    raw_constraints.append(line)
    if current_message:
        for line in re.split(r"[\n\.؟\?\!]+", current_message):
            line = line.strip()
            if 4 <= len(line) <= 220 and _matches_any(line, _CONSTRAINT_PATTERNS):
                raw_constraints.append(line)
    facts.constraints = _dedup_keep_order(raw_constraints, cap_c)

    # --- Decisions --------------------------------------------------------- #
    # Decisions are commitments made *in* a topic. On topic switch they are
    # stale by construction.
    cap_d = getattr(settings, "PINNED_FACTS_MAX_DECISIONS", 6)
    raw_decisions: List[str] = []
    if not topic_changed:
        for msg in chat_history:
            if msg.get("role") != "assistant":
                continue
            text = (msg.get("content") or "").strip()
            if not text:
                continue
            for line in re.split(r"[\n\.؟\?\!]+", text):
                line = line.strip()
                if 6 <= len(line) <= 220 and _matches_any(line, _DECISION_PATTERNS):
                    raw_decisions.append(line)
    facts.decisions = _dedup_keep_order(raw_decisions, cap_d)

    # --- Open questions ---------------------------------------------------- #
    # Recent assistant questions that the user hasn't yet answered.
    raw_questions: List[str] = []
    last_user_idx = -1
    for i in range(len(chat_history) - 1, -1, -1):
        if chat_history[i].get("role") == "user":
            last_user_idx = i
            break
    tail = chat_history[last_user_idx + 1:] if last_user_idx >= 0 else chat_history
    for msg in tail:
        if msg.get("role") != "assistant":
            continue
        text = (msg.get("content") or "").strip()
        for line in text.splitlines():
            line = line.strip()
            if 6 <= len(line) <= 200 and _QUESTION_PATTERN.search(line):
                raw_questions.append(line)
    facts.open_questions = _dedup_keep_order(raw_questions, 4)

    # --- Files ------------------------------------------------------------- #
    # On topic switch the prior session files are *also* irrelevant for
    # anchoring — they keep dragging the LLM back into the old topic. The
    # workspace block (rendered separately) still surfaces them so the model
    # can choose to load them if the user mentions them again.
    if file_registry and not topic_changed:
        compact: List[str] = []
        for f in file_registry[-8:]:  # most recent 8 (already small)
            ordinal = f.get("ordinal", "?")
            name = f.get("name", "?")
            ftype = f.get("type", "?")
            compact.append(f"#{ordinal} {name} ({ftype})")
        facts.files = compact

    if facts.is_empty():
        facts.source = "empty"
    return facts


# --------------------------------------------------------------------------- #
# Optional LLM enrichment                                                     #
# --------------------------------------------------------------------------- #

_LLM_PINNED_FACTS_PROMPT = (
    "You are a structured information extractor. From the conversation below, "
    "produce a JSON object with EXACTLY these keys: goal, constraints, "
    "decisions, open_questions. Each list item must be a short, self-contained "
    "factual statement. Do not invent. If unknown, leave empty. JSON only, no markdown."
)


async def _enrich_with_llm(
    facts: PinnedFacts,
    chat_history: List[Dict[str, Any]],
    current_message: Optional[str],
    llm_client,
) -> PinnedFacts:
    """Best-effort LLM enrichment with hard timeout and graceful fallback."""
    if not llm_client:
        return facts

    timeout = getattr(settings, "PINNED_FACTS_LLM_TIMEOUT", 6.0)

    # Compact transcript — cap each message to keep the call fast.
    lines: List[str] = []
    for m in chat_history[-30:]:
        role = "User" if m.get("role") == "user" else "Assistant"
        content = (m.get("content") or "")[:400]
        if content:
            lines.append(f"{role}: {content}")
    if current_message:
        lines.append(f"User (current): {current_message[:600]}")
    transcript = "\n".join(lines)

    try:
        result = await asyncio.wait_for(
            llm_client.chat(
                messages=[
                    {"role": "system", "content": _LLM_PINNED_FACTS_PROMPT + " /no_think"},
                    {
                        "role": "user",
                        "content": (
                            "Extract pinned facts as JSON.\n\n"
                            "JSON schema:\n"
                            '{"goal":"...","constraints":["..."],'
                            '"decisions":["..."],"open_questions":["..."]}\n\n'
                            f"Conversation:\n{transcript}"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=400,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        log.warning("PinnedFacts LLM enrichment timed out — keeping heuristic facts")
        return facts
    except Exception as e:
        log.warning(f"PinnedFacts LLM enrichment failed: {e} — keeping heuristic facts")
        return facts

    if not result or not result.get("ok"):
        return facts

    raw = (result.get("content") or "").strip()
    data = _safe_json(raw)
    if not isinstance(data, dict):
        return facts

    # Merge — LLM-found items take precedence but we keep heuristic ones
    # the LLM might have missed.
    cap_c = getattr(settings, "PINNED_FACTS_MAX_CONSTRAINTS", 6)
    cap_d = getattr(settings, "PINNED_FACTS_MAX_DECISIONS", 6)

    goal = (data.get("goal") or "").strip() or facts.goal
    constraints = _dedup_keep_order(
        [str(x).strip() for x in (data.get("constraints") or []) if x] + facts.constraints,
        cap_c,
    )
    decisions = _dedup_keep_order(
        [str(x).strip() for x in (data.get("decisions") or []) if x] + facts.decisions,
        cap_d,
    )
    open_questions = _dedup_keep_order(
        [str(x).strip() for x in (data.get("open_questions") or []) if x] + facts.open_questions,
        4,
    )
    return PinnedFacts(
        goal=goal[:280],
        constraints=constraints,
        decisions=decisions,
        open_questions=open_questions,
        files=facts.files,
        source="merged",
    )


def _safe_json(raw: str) -> Optional[Any]:
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

async def extract_pinned_facts(
    chat_history: List[Dict[str, Any]],
    current_message: Optional[str] = None,
    file_registry: Optional[List[Dict[str, Any]]] = None,
    llm_client: Optional[Any] = None,
    conversation_state: Any = None,
) -> PinnedFacts:
    """
    Extract pinned conversation facts.

    Always returns a PinnedFacts object — even an empty one when the
    conversation has nothing to anchor yet. Callers can render it via
    ``render_pinned_facts_block`` directly.

    When ``conversation_state.topic_changed`` is True the anchor is reset
    to the new topic so we never pin a stale goal/file across topic switches.
    """
    if not getattr(settings, "PINNED_FACTS_ENABLED", True):
        return PinnedFacts(source="disabled")

    facts = extract_pinned_facts_heuristic(
        chat_history=chat_history or [],
        current_message=current_message,
        file_registry=file_registry,
        conversation_state=conversation_state,
    )

    if getattr(settings, "PINNED_FACTS_LLM_ENABLED", False) and llm_client is not None:
        try:
            facts = await _enrich_with_llm(facts, chat_history or [], current_message, llm_client)
        except Exception as e:
            log.warning(f"PinnedFacts enrichment swallowed: {e}")

    return facts


def render_pinned_facts_block(facts: Optional[PinnedFacts | Dict[str, Any]]) -> str:
    """
    Render PinnedFacts as a stable system-prompt block.

    Returns empty string when there is nothing useful to anchor.
    """
    if facts is None:
        return ""
    if isinstance(facts, dict):
        try:
            facts = PinnedFacts(**{k: v for k, v in facts.items() if k in PinnedFacts.__dataclass_fields__})
        except Exception:
            return ""
    if not isinstance(facts, PinnedFacts) or facts.is_empty():
        return ""

    lines: List[str] = ["## Pinned Conversation Facts (always honor these)"]
    lines.append(
        "These are the *anchor* facts of this conversation. Whatever was "
        "summarised or trimmed elsewhere, these are still true and must guide "
        "your reply. Do NOT contradict them, do NOT re-ask the user about "
        "them, and do NOT repeat answers you already gave."
    )

    if facts.goal:
        lines.append(f"\n**Goal:** {facts.goal}")

    if facts.constraints:
        lines.append("\n**Constraints / requirements:**")
        for c in facts.constraints:
            lines.append(f"- {c}")

    if facts.decisions:
        lines.append("\n**Decisions already made:**")
        for d in facts.decisions:
            lines.append(f"- {d}")

    if facts.open_questions:
        lines.append("\n**Open questions awaiting the user:**")
        for q in facts.open_questions:
            lines.append(f"- {q}")

    if facts.files:
        lines.append("\n**Session files referenced:**")
        for f in facts.files:
            lines.append(f"- {f}")

    return "\n".join(lines)


__all__ = [
    "PinnedFacts",
    "extract_pinned_facts",
    "extract_pinned_facts_heuristic",
    "render_pinned_facts_block",
]
