"""
Repetition / Loop Detection
===========================

Two complementary mechanisms used together to fight mid-chat drift:

1. **Proactive guard (prompt-level)** — once a conversation crosses
   ``ANTI_LOOP_TURN_THRESHOLD`` turns, we inject a small system block that
   reminds the LLM not to repeat itself and to honour the pinned facts.

2. **Reactive detector (post-generation)** — compare the new assistant
   message against the most recent assistant message(s) using cheap n-gram
   similarity. If similarity exceeds the threshold we mark the turn as
   "repeated" so the caller can log / surface / repair it.

Both functions are flag-gated and dependency-free (pure stdlib + regex).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import settings

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tokenisation                                                                #
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _ngrams(tokens: List[str], n: int) -> List[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


# --------------------------------------------------------------------------- #
# Proactive guard                                                             #
# --------------------------------------------------------------------------- #

ANTI_LOOP_BLOCK = """## Anti-Loop / Coherence Guard
You have already exchanged several turns with this user. Before responding:

1. Do **NOT** repeat content from your previous answers verbatim or paraphrased.
2. Do **NOT** re-ask the user about information already present in the
   "Pinned Conversation Facts" block above.
3. Do **NOT** restart the topic from scratch — *continue* from the latest
   user message and the most recent decision in the pinned facts.
4. If you find yourself about to give a generic / circular reply, stop and
   instead acknowledge what is already established (briefly) and then advance
   the conversation with concrete information or a clarifying question that
   was NOT asked before.
5. Match the user's language exactly.
"""


def should_inject_anti_loop(
    conversation_state: Optional[Any],
    *,
    threshold: Optional[int] = None,
) -> bool:
    """Whether to inject the proactive anti-loop block."""
    if not getattr(settings, "ANTI_LOOP_GUARD_ENABLED", True):
        return False
    if conversation_state is None:
        return False
    turn = getattr(conversation_state, "turn_number", 0) or 0
    thr = threshold if threshold is not None else getattr(
        settings, "ANTI_LOOP_TURN_THRESHOLD", 4
    )
    return turn >= thr


def render_anti_loop_block(
    conversation_state: Optional[Any] = None,
    *,
    threshold: Optional[int] = None,
) -> str:
    """Render the anti-loop guard block when conditions are met."""
    if not should_inject_anti_loop(conversation_state, threshold=threshold):
        return ""
    return ANTI_LOOP_BLOCK


# --------------------------------------------------------------------------- #
# Reactive detector                                                           #
# --------------------------------------------------------------------------- #

@dataclass
class RepetitionSignal:
    """Result of a repetition check."""
    is_repetition: bool
    similarity: float
    compared_to_turn: Optional[int] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_repetition": self.is_repetition,
            "similarity": round(self.similarity, 3),
            "compared_to_turn": self.compared_to_turn,
            "reason": self.reason,
        }


def detect_repetition(
    new_assistant_text: str,
    chat_history: List[Dict[str, Any]],
    *,
    look_back: int = 3,
    threshold: Optional[float] = None,
) -> RepetitionSignal:
    """
    Compare a freshly-generated assistant answer against the previous
    ``look_back`` assistant answers using 3-gram Jaccard similarity.

    Returns a RepetitionSignal — never raises.
    """
    if not getattr(settings, "REPETITION_DETECTION_ENABLED", True):
        return RepetitionSignal(False, 0.0, reason="disabled")

    if not new_assistant_text or not chat_history:
        return RepetitionSignal(False, 0.0, reason="empty")

    thr = (
        threshold
        if threshold is not None
        else getattr(settings, "REPETITION_SIMILARITY_THRESHOLD", 0.82)
    )

    new_tokens = _tokens(new_assistant_text)
    if len(new_tokens) < 8:
        # Too short — even legitimate confirmations look "similar".
        return RepetitionSignal(False, 0.0, reason="too_short")

    new_grams = _ngrams(new_tokens, 3) or _ngrams(new_tokens, 2)

    # Walk history from newest to oldest and stop after `look_back` assistant
    # turns. Compare against each.
    best_sim = 0.0
    best_turn: Optional[int] = None
    seen_assistant = 0
    turn_idx = sum(1 for m in chat_history if m.get("role") == "user")

    for msg in reversed(chat_history):
        role = msg.get("role")
        if role == "user":
            turn_idx -= 1
            continue
        if role != "assistant":
            continue
        seen_assistant += 1
        prev_text = (msg.get("content") or "").strip()
        if not prev_text:
            if seen_assistant >= look_back:
                break
            continue

        prev_tokens = _tokens(prev_text)
        prev_grams = _ngrams(prev_tokens, 3) or _ngrams(prev_tokens, 2)
        sim = _jaccard(new_grams, prev_grams)
        if sim > best_sim:
            best_sim = sim
            best_turn = max(turn_idx, 0)

        if seen_assistant >= look_back:
            break

    is_rep = best_sim >= thr
    return RepetitionSignal(
        is_repetition=is_rep,
        similarity=best_sim,
        compared_to_turn=best_turn,
        reason="threshold_exceeded" if is_rep else "below_threshold",
    )


__all__ = [
    "ANTI_LOOP_BLOCK",
    "RepetitionSignal",
    "detect_repetition",
    "render_anti_loop_block",
    "should_inject_anti_loop",
]
