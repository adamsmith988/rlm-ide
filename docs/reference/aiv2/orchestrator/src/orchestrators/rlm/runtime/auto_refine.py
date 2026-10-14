"""Prime-like auto-refine scheduler (turn interval + post-compact + cooldown)."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

from src.orchestrators.rlm.schema_utils import coerce_bool

if TYPE_CHECKING:
    from src.orchestrators.rlm.runtime.session import RlmSession

logger = logging.getLogger(__name__)


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def apply_env_overrides(session: "RlmSession") -> None:
    """Override node cfg with env fallbacks only when cfg omitted the key."""
    cfg = session.cfg if isinstance(session.cfg, dict) else {}
    if "refine_auto_enabled" not in cfg:
        raw = os.environ.get("AIV2_RLM_REFINE_AUTO")
        if raw is not None and str(raw).strip() != "":
            session.refine_auto_enabled = coerce_bool(raw, default=True)
    if "refine_turn_interval" not in cfg:
        n = _env_int("AIV2_RLM_REFINE_TURN_INTERVAL")
        if n is not None:
            session.refine_turn_interval = max(1, min(500, n))
    if "refine_cooldown_sec" not in cfg:
        n = _env_int("AIV2_RLM_REFINE_COOLDOWN_SEC")
        if n is not None:
            session.refine_cooldown_sec = max(0, min(86400, n))
    if "refine_on_compact" not in cfg:
        raw = os.environ.get("AIV2_RLM_REFINE_ON_COMPACT")
        if raw is not None and str(raw).strip() != "":
            session.refine_on_compact = coerce_bool(raw, default=True)


def auto_refine_allowed(session: "RlmSession") -> bool:
    if not session.refine_enabled or not session.refine_auto_enabled:
        return False
    if int(getattr(session, "depth", 0) or 0) != 0:
        return False
    return True


def _under_cooldown(session: "RlmSession") -> bool:
    last = float(getattr(session, "_last_refine_review_at", 0) or 0)
    if last <= 0:
        return False
    return (time.time() - last) < float(session.refine_cooldown_sec or 0)


async def maybe_auto_refine(
    session: "RlmSession",
    *,
    reason: str = "turn_interval",
) -> Dict[str, Any]:
    """Run review gate then refine when due. No-ops when disabled/cooldown."""
    if not auto_refine_allowed(session):
        if reason == "compact":
            session._compact_refine_pending = False
        return {"ok": True, "skipped": "not_allowed"}

    if getattr(session, "_refine_in_progress", False):
        if reason == "compact":
            session._compact_refine_pending = True
        return {"ok": True, "skipped": "busy"}

    if reason == "compact":
        if not session.refine_on_compact:
            session._compact_refine_pending = False
            reason = "turn_interval"
        elif not session._compact_refine_pending:
            return {"ok": True, "skipped": "no_compact_pending"}

    if reason == "turn_interval":
        if session._assistant_turns_since_refine < session.refine_turn_interval:
            return {"ok": True, "skipped": "interval"}

    if _under_cooldown(session):
        if reason == "compact":
            session._compact_refine_pending = True
        return {"ok": True, "skipped": "cooldown"}

    session._refine_in_progress = True
    try:
        from src.orchestrators.rlm.runtime.refine import refine_session
        from src.orchestrators.rlm.runtime.refine_review import review_auto_refine

        review = await review_auto_refine(
            session,
            chat_fn=session.chat_fn,
            extract_assistant_fn=session.extract_assistant_fn,
            reason=reason,
            turns_since_last_review=session._assistant_turns_since_refine,
        )
        session._last_refine_review_at = time.time()
        if not review.get("shouldRefine"):
            session._assistant_turns_since_refine = 0
            if reason == "compact":
                session._compact_refine_pending = False
            return {
                "ok": True,
                "skipped": "review_declined",
                "review": review,
            }

        result = await refine_session(
            session,
            chat_fn=session.chat_fn,
            extract_assistant_fn=session.extract_assistant_fn,
            instructions=str(review.get("instructions") or "") or None,
            skip_review=True,
        )
        session._assistant_turns_since_refine = 0
        if reason == "compact":
            session._compact_refine_pending = False
        result["review"] = review
        result["trigger"] = reason
        return result
    except Exception as exc:
        logger.warning("auto-refine failed: %s", exc)
        session._last_refine_review_at = time.time()
        return {"ok": False, "error": str(exc)[:400]}
    finally:
        session._refine_in_progress = False
