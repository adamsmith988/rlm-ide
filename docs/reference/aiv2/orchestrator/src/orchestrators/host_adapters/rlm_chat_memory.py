"""AF-parity memory ladder for hub RLM chat (warm → disk → hub conversation).

The hub conversation (stored turns, sized by session compaction: checkpoint
summary + verbatim tail) is the memory every engine shares. RLM also keeps its
own transcript (code cells, REPL output) per chat; when other engines served
turns in between, those turns are appended to it before RLM continues, and
when the hub conversation no longer reaches RLM's own last turn the transcript
is reseeded from the hub conversation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

HUB_TURNS_FILE = "hub_turns.json"
_KEY_CHARS = 400
_MAX_TRACKED = 200
_SPACE = re.compile(r"\s+")


def messages_have_memory(messages: Any) -> bool:
    """True when transcript has at least one non-system turn the model can recall."""
    if not isinstance(messages, list):
        return False
    for m in messages:
        if not isinstance(m, dict):
            continue
        if str(m.get("role") or "").strip() == "system":
            continue
        return True
    return False


def hub_history_prior_turns(
    chat_history: Optional[List[Dict[str, Any]]],
    *,
    current_user: str = "",
) -> List[Dict[str, Any]]:
    """Normalize hub ``chat_history`` into user/assistant rows."""
    if not isinstance(chat_history, list) or not chat_history:
        return []
    rows = list(chat_history)
    cur = str(current_user or "").strip()
    if (
        cur
        and rows
        and isinstance(rows[-1], dict)
        and str(rows[-1].get("role") or "").strip() == "user"
        and str(rows[-1].get("content") or "").strip() == cur
    ):
        rows = rows[:-1]
    out: List[Dict[str, Any]] = []
    for m in rows:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "").strip()
        if role not in ("user", "assistant"):
            continue
        content = m.get("content")
        if content is None or content == "":
            continue
        out.append({"role": role, "content": content})
    return out


def hub_conversation(
    self_compact: Optional[Mapping[str, Any]],
    chat_history: Optional[List[Dict[str, Any]]],
    *,
    current_user: str = "",
) -> Tuple[str, List[Dict[str, Any]]]:
    """``(checkpoint summary, prior user/assistant rows)`` the host prepared for this turn."""
    if isinstance(self_compact, Mapping) and (self_compact.get("messages") or self_compact.get("summary")):
        return (
            str(self_compact.get("summary") or ""),
            hub_history_prior_turns(list(self_compact.get("messages") or []), current_user=current_user),
        )
    return "", hub_history_prior_turns(chat_history, current_user=current_user)


def summary_message(summary: str) -> Optional[Dict[str, Any]]:
    """The checkpoint rendered the way every engine shows it."""
    text = str(summary or "").strip()
    if not text:
        return None
    from src.config import settings
    from src.context.self_compact import load_prompts, render_summary_section

    return {"role": "system", "content": render_summary_section(text, load_prompts(settings.SELF_COMPACT_RUBRIC))}


# ---- which hub turns RLM served ----------------------------------------------


def turn_key(text: Any) -> str:
    norm = _SPACE.sub(" ", str(text or "")).strip().lower()[:_KEY_CHARS]
    return hashlib.sha1(norm.encode("utf-8")).hexdigest() if norm else ""


def _turns_path(live: Any) -> str:
    base = str(getattr(live, "artifact_dir", "") or "")
    return os.path.join(base, HUB_TURNS_FILE) if base else ""


def served_turns(live: Any) -> Dict[str, List[str]]:
    """Keys of the user messages and answers RLM handled in this chat."""
    cached = getattr(live, "_hub_turns", None)
    if isinstance(cached, dict):
        return cached
    data: Dict[str, List[str]] = {"users": [], "answers": []}
    path = _turns_path(live)
    if path and os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            data = {k: [str(x) for x in raw.get(k) or []] for k in ("users", "answers")}
        except (OSError, ValueError, AttributeError):
            logger.debug("RLM hub turn record unreadable: %s", path, exc_info=True)
    try:
        live._hub_turns = data
    except Exception:  # noqa: BLE001 — read-only stand-ins
        pass
    return data


def record_served_turn(live: Any, user_message: str, answer: str) -> None:
    """Remember that RLM answered this turn, so later turns can find where it left off."""
    if live is None:
        return
    data = served_turns(live)
    for name, text in (("users", user_message), ("answers", answer)):
        key = turn_key(text)
        if key:
            data[name] = (data.get(name, []) + [key])[-_MAX_TRACKED:]
    path = _turns_path(live)
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
    except OSError:
        logger.debug("RLM hub turn record not written: %s", path, exc_info=True)


def last_served_index(prior: List[Dict[str, Any]], served: Mapping[str, List[str]]) -> int:
    """Index of the last hub row RLM itself answered (-1 when none is in ``prior``)."""
    answers, users = set(served.get("answers") or []), set(served.get("users") or [])
    for i in range(len(prior) - 1, -1, -1):
        if prior[i].get("role") != "assistant":
            continue
        if turn_key(prior[i].get("content")) in answers:
            return i
        if i and prior[i - 1].get("role") == "user" and turn_key(prior[i - 1].get("content")) in users:
            return i
    return -1


def sync_with_hub(live: Any, *, prior: List[Dict[str, Any]]) -> Tuple[str, int]:
    """Bring RLM's own transcript up to date with turns other engines served.

    Returns ``(mode, appended)``: ``untracked`` (no record of RLM's turns),
    ``in_sync``, ``appended`` or ``stale`` (the hub conversation no longer
    reaches RLM's last turn — the caller reseeds).
    """
    served = served_turns(live)
    if not (served.get("answers") or served.get("users")):
        return "untracked", 0
    if not prior:
        return "in_sync", 0
    last = last_served_index(prior, served)
    if last < 0:
        return "stale", 0
    missing = [dict(m) for m in prior[last + 1:]]
    if not missing:
        return "in_sync", 0
    live.messages.extend(missing)
    return "appended", len(missing)


# ---- ladder ----------------------------------------------------------------


def seed_session_from_hub_history(
    live: Any,
    *,
    chat_history: Optional[List[Dict[str, Any]]],
    user_message: str,
    base_system: str,
    summary: str = "",
    prior: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Seed ``live.messages`` from the hub conversation when disk hydrate is empty."""
    if prior is None:
        prior = hub_history_prior_turns(chat_history, current_user=user_message)
    checkpoint = summary_message(summary)
    if not prior and checkpoint is None:
        return False
    seeded: List[Dict[str, Any]] = []
    try:
        system = (
            live.build_system_prompt(base_system)
            if hasattr(live, "build_system_prompt")
            else base_system
        )
    except Exception:
        system = base_system
    if system:
        seeded.append({"role": "system", "content": system})
    if checkpoint is not None:
        seeded.append(checkpoint)
    seeded.extend(prior)
    um = str(user_message or "").strip()
    if um:
        seeded.append({"role": "user", "content": um})
    live.messages = seeded
    live.user_content = None
    try:
        tpath = os.path.join(
            str(getattr(live, "artifact_dir", "") or ""), "transcript.full.jsonl"
        )
        if os.path.isfile(tpath) and os.path.getsize(tpath) > 64:
            live._transcript_written = len(live.messages)
    except Exception:
        pass
    return True


def _synced(live: Any, prior: List[Dict[str, Any]], summary: str, user_message: str, base_system: str, mode: str) -> Tuple[str, bool]:
    """Apply :func:`sync_with_hub` after a warm/disk resume; reseed when stale."""
    sync, n = sync_with_hub(live, prior=prior)
    if sync == "stale" and seed_session_from_hub_history(
        live, chat_history=None, user_message=user_message, base_system=base_system, summary=summary, prior=prior,
    ):
        logger.info("RLM transcript reseeded from the hub conversation (%d prior rows)", len(prior))
        return "reseeded", bool(str(user_message or "").strip())
    if sync == "appended":
        logger.info("RLM transcript caught up with %d hub rows served by other engines", n)
    return mode, False


def resume_or_seed_live_memory(
    live: Any,
    *,
    user_message: str,
    base_system: str,
    chat_history: Optional[List[Dict[str, Any]]],
    self_compact: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, bool]:
    """AF-parity ladder: warm → disk hydrate → hub conversation → empty.

    Returns ``(mode, user_already_on_messages)`` where mode is one of
    ``warm`` | ``resumed`` | ``reseeded`` | ``seeded`` | ``empty``.
    When ``user_already_on_messages`` is True, caller must ``continue_with("")``.
    """
    if live is None:
        return "empty", False
    summary, prior = hub_conversation(self_compact, chat_history, current_user=user_message)
    if messages_have_memory(getattr(live, "messages", None)):
        return _synced(live, prior, summary, user_message, base_system, "warm")
    hydrated = False
    try:
        if hasattr(live, "hydrate_messages_from_disk"):
            hydrated = bool(live.hydrate_messages_from_disk(base_system=base_system))
    except Exception:
        logger.debug("RLM disk hydrate failed", exc_info=True)
        hydrated = False
    if hydrated:
        mode, user_on = _synced(live, prior, summary, user_message, base_system, "resumed")
        if user_on:
            return mode, True
        um = str(user_message or "").strip()
        if um:
            live.messages.append({"role": "user", "content": um})
        live.user_content = None
        return mode, bool(um)
    if seed_session_from_hub_history(
        live,
        chat_history=chat_history,
        user_message=user_message,
        base_system=base_system,
        summary=summary,
        prior=prior,
    ):
        return "seeded", bool(str(user_message or "").strip())
    return "empty", False
