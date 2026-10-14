"""Resolve hub RLM final answer so empty turns never collapse to «پاسخی تولید نشد.»

Mirrors Agent Flow: soft ``incomplete`` / overflow keep a user-visible notice;
hard LLM outages surface a clear message instead of an empty ``done``.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from src.config import settings

CONTEXT_OVERFLOW_NOTICE = (
    "Context window is full and automatic compaction could not free "
    "enough space for another model turn. Summarize or start a fresh "
    "chat, then continue."
)

INCOMPLETE_FALLBACK_NOTICE = settings.HUB_NOTICE_INCOMPLETE

# Legacy English stubs — never show these to hub users again.
PROCESS_LOG_FALLBACK = "_(RLM finished — see process log)_"
EMPTY_ANSWER_FALLBACK = "_(RLM finished with empty answer)_"
INTERRUPTED_FALLBACK = "_(turn interrupted before a final answer)_"

_PLACEHOLDER_ANSWERS = frozenset(
    {
        "",
        "پاسخی تولید نشد.",
        "پاسخی تولید نشد",
        PROCESS_LOG_FALLBACK,
        EMPTY_ANSWER_FALLBACK,
        INTERRUPTED_FALLBACK,
        "(RLM finished — see process log)",
        "(RLM finished with empty answer)",
    }
)


def is_placeholder_answer(text: Any) -> bool:
    s = str(text or "").strip()
    if not s:
        return True
    if s in _PLACEHOLDER_ANSWERS:
        return True
    low = s.lower()
    # Exact English stubs only. Do NOT treat "stub + real Persian notice"
    # as a pure placeholder — ensure_nonempty_hub_answer unglues those.
    if low in {str(p).strip().lower() for p in _PLACEHOLDER_ANSWERS if str(p).strip()}:
        return True
    if s in ("پاسخی تولید نشد.", "پاسخی تولید نشد"):
        return True
    if low.startswith("_(rlm finished") and "کار ناتمام" not in s and "مدل" not in s:
        return True
    if (
        "see process log" in low
        and len(s) < 80
        and "کار ناتمام" not in s
        and "مدل در دسترس" not in s
    ):
        return True
    return False


def is_llm_gateway_outage(detail: Any) -> bool:
    text = str(detail or "").lower()
    if not text:
        return False
    needles = (
        "all_backends_failed",
        "chat_model_unavailable",
        "catalog_unavailable",
        "catalog is temporarily unavailable",
        "llm gateway catalog",
        "temporary failure in name resolution",
        "nameresolutionerror",
        "no address associated with hostname",
        "connecterror",
        "model gateway 503",
        "[reconnecting]",
        "reconnecting, retry",
    )
    return any(n in text for n in needles)


def friendly_llm_outage_notice(detail: Any = None) -> str:
    text = str(detail or "")
    low = text.lower()
    if (
        "all_backends_failed" in low
        or "connecterror" in low
        or "no address associated" in low
        or "model gateway 503" in low
        or "reconnecting" in low
    ):
        return (
            "مدل در دسترس نیست (بک‌اندهای LLM قطع یا unhealthy هستند — اغلب DNS/OSBI). "
            "وضعیت جلسه حفظ شده؛ پس از برگشتن مدل پیام را دوباره بفرستید یا Continue بزنید."
        )
    if "chat_model_unavailable" in low:
        return (
            "مدل چت برای این سرویس bind نشده یا در دسترس نیست. "
            "وضعیت جلسه حفظ شده؛ پس از رفع مشکل ادامه دهید."
        )
    if "temporary failure in name resolution" in low or "nameresolutionerror" in low:
        return (
            "خطای شبکه/DNS در دسترسی به سرور مدل. "
            "وضعیت جلسه حفظ شده؛ پس از رفع اتصال ادامه دهید."
        )
    if "catalog" in low and "unavailable" in low:
        return (
            "کاتالوگ مدل موقتاً در دسترس نیست. "
            "وضعیت جلسه حفظ شده؛ کمی بعد دوباره تلاش کنید."
        )
    if text.strip():
        clipped = text.strip()[:400]
        return (
            f"RLM با خطای مدل/پراکسی متوقف شد: {clipped}\n"
            "وضعیت جلسه حفظ شده؛ می‌توانید ادامه دهید."
        )
    return (
        "مدل پاسخ نداد یا قطع شد. وضعیت جلسه حفظ شده؛ "
        "پس از پایدار شدن مدل دوباره تلاش کنید."
    )


def incomplete_notice(reason: Any = None) -> str:
    why = str(reason or "").strip()
    if "context_overflow" in why:
        return CONTEXT_OVERFLOW_NOTICE
    if why and why not in ("idle", "completed", "running"):
        return settings.HUB_NOTICE_INCOMPLETE_REASON.replace("{reason}", why)
    return INCOMPLETE_FALLBACK_NOTICE


def _this_turn_visible(sess: Any) -> str:
    """Only this-turn stamps — never replay a prior hub placeholder from history."""
    for candidate in (
        str(getattr(sess, "last_content", "") or "").strip(),
        str(getattr(sess, "_last_quiescent_turn_content", "") or "").strip(),
    ):
        if candidate and not is_placeholder_answer(candidate):
            return candidate
    return ""


def resolve_hub_final_answer(
    *,
    streamed: str = "",
    engine_text: str = "",
    runtime_session: Any = None,
    thinking_text: str = "",
    error_hint: str = "",
) -> Tuple[str, Optional[str]]:
    """Return ``(final_answer, status)`` — never a hub placeholder."""
    status = None
    if runtime_session is not None:
        status = str(getattr(runtime_session, "status", "") or "") or None

    for candidate in (streamed, engine_text):
        text = str(candidate or "").strip()
        if text and not is_placeholder_answer(text):
            return text, status

    sess = runtime_session
    err = (
        str(getattr(sess, "error", "") or "").strip() if sess else ""
    ) or str(error_hint or "").strip()
    reason = (
        str(getattr(sess, "_autonomous_stop_reason", "") or "").strip() if sess else ""
    )
    st = str(status or "").strip().lower()
    outage_flag = bool(getattr(sess, "_llm_gateway_outage_seen", False)) if sess else False

    visible = _this_turn_visible(sess) if sess is not None else ""
    if visible:
        return visible, status

    logs = getattr(sess, "log_lines", None) if sess is not None else None
    log_blob = ""
    if isinstance(logs, list):
        log_blob = "\n".join(str(x) for x in logs if str(x or "").strip())
    thinking = str(thinking_text or "").strip() or (
        str(getattr(sess, "last_reasoning", "") or "").strip() if sess else ""
    )
    signal = "\n".join(x for x in (err, reason, log_blob, thinking) if x)

    if (
        outage_flag
        or is_llm_gateway_outage(signal)
        or reason == "llm_gateway_outage"
    ):
        return friendly_llm_outage_notice(signal), st or "incomplete"

    if st == "incomplete" or (reason and reason not in ("idle", "completed")):
        return incomplete_notice(reason or st), st or "incomplete"

    if st == "error" or err:
        return friendly_llm_outage_notice(err or signal), st or "error"

    if thinking or log_blob:
        # Real work happened but no user-visible answer — soft incomplete.
        return INCOMPLETE_FALLBACK_NOTICE, st or "incomplete"

    return "", status


def ensure_nonempty_hub_answer(
    answer: str,
    *,
    thinking: str = "",
    error: str = "",
    status: str = "",
) -> str:
    """Last-line defense for ``main.py`` / live finalize."""
    text = str(answer or "").strip()
    if text and not is_placeholder_answer(text):
        # Streamed concat may glue a legacy English stub + a real notice.
        if "see process log" in text.lower() or "پاسخی تولید نشد" in text:
            # Prefer the Persian/notice tail when present.
            for marker in (
                "مدل در دسترس نیست",
                "کار ناتمام ماند",
                "Context window is full",
            ):
                idx = text.find(marker)
                if idx >= 0:
                    return text[idx:].strip()
            # Drop English stub prefix if glued.
            for stub in (
                PROCESS_LOG_FALLBACK,
                EMPTY_ANSWER_FALLBACK,
                INTERRUPTED_FALLBACK,
                "(RLM finished — see process log)",
                "(RLM finished with empty answer)",
            ):
                if text.startswith(stub):
                    rest = text[len(stub) :].strip()
                    if rest and not is_placeholder_answer(rest):
                        return rest
        else:
            return text
    err = str(error or "").strip()
    st = str(status or "").strip().lower()
    think = str(thinking or "").strip()
    signal = "\n".join(x for x in (err, think) if x)
    if is_llm_gateway_outage(signal):
        return friendly_llm_outage_notice(signal)
    if err:
        return friendly_llm_outage_notice(err)
    if st == "incomplete":
        return incomplete_notice(st)
    if think:
        return INCOMPLETE_FALLBACK_NOTICE
    return friendly_llm_outage_notice(err or think)
