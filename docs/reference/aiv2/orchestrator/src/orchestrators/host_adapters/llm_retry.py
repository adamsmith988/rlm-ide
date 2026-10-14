"""Agent Flow–parity LLM reconnect for hub RLM chat ports.

Transient model-gateway 503s / ``all_backends_failed`` and mid-stream transport
drops are retried with backoff so a short OSBI/DNS blip does not collapse the
turn into the hub's empty «پاسخی تولید نشد.» answer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Match modules/plugins/agent_flow/runner.py
LLM_503_RETRY_ATTEMPTS = 12
LLM_503_RETRY_DELAY_SEC = 10.0
LLM_TRANSPORT_RETRY_ATTEMPTS = 5
LLM_TRANSPORT_RETRY_DELAY_SEC = 3.0

OnRetry = Callable[..., Any]
AsyncCall = Callable[[], Awaitable[Any]]


def looks_like_catalog_unavailable(detail: Any) -> bool:
    text = str(detail or "").lower()
    return (
        "catalog_unavailable" in text
        or "catalog is temporarily unavailable" in text
        or "llm gateway catalog" in text
    )


def _overflow_or_too_large(text: str) -> bool:
    """Never burn reconnect retries on context / entity-too-large failures."""
    try:
        from src.orchestrators.rlm.runtime.context_budget import (
            looks_like_context_overflow,
            looks_like_request_too_large,
            parse_context_overflow_error,
        )

        if parse_context_overflow_error(text):
            return True
        if looks_like_request_too_large(text) or looks_like_context_overflow(text):
            return True
    except Exception:
        pass
    low = str(text or "").lower()
    return any(
        s in low
        for s in (
            "context length",
            "maximum context",
            "max_model_len",
            "maximum model length",
            "decoder prompt",
            "request_too_large",
            "entity too large",
        )
    )


def is_retryable_llm_503(exc: BaseException) -> bool:
    """True for transient model/gateway outages (not overflow / hard 4xx)."""
    text = str(exc or "")
    low = text.lower()
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    has_503_shape = (
        status == 503
        or "llm error 503" in low
        or "llm stream error" in low
        # Hub OpenAI client surfaces proxy aggregate failures without AF's prefix.
        or "all_backends_failed" in low
        or "chat_model_unavailable" in low
        or looks_like_catalog_unavailable(text)
    )
    if not has_503_shape:
        return False
    if _overflow_or_too_large(text):
        return False
    return True


def is_retryable_llm_transport(exc: BaseException) -> bool:
    """True for mid-stream / connect drops that deserve a fresh chat attempt."""
    if is_retryable_llm_503(exc):
        return False  # longer 503 plan owns these
    name = type(exc).__name__.lower()
    text = str(exc or "").lower()
    cause = getattr(exc, "__cause__", None)
    cause_name = type(cause).__name__.lower() if cause is not None else ""
    cause_text = str(cause or "").lower()
    blob = f"{name} {text} {cause_name} {cause_text}"
    needles = (
        "readerror",
        "remoteprotocolerror",
        "writeerror",
        "connecterror",
        "connecttimeout",
        "pooltimeout",
        "networkerror",
        "connectionreset",
        "connection aborted",
        "server disconnected",
        "incomplete read",
        "broken pipe",
        "llm stream failed",
    )
    return any(n in blob for n in needles)


def llm_retry_plan(exc: BaseException) -> Optional[Tuple[int, float, str]]:
    """Return ``(max_attempts, wait_sec, kind)`` when ``exc`` is retryable."""
    if is_retryable_llm_503(exc):
        return (
            LLM_503_RETRY_ATTEMPTS,
            LLM_503_RETRY_DELAY_SEC,
            "model gateway 503",
        )
    if is_retryable_llm_transport(exc):
        return (
            LLM_TRANSPORT_RETRY_ATTEMPTS,
            LLM_TRANSPORT_RETRY_DELAY_SEC,
            "LLM transport",
        )
    return None


def reconnect_preview(
    *,
    failed_attempt: int,
    max_attempts: int,
    wait_sec: float,
    kind: str = "model gateway 503",
) -> str:
    next_attempt = int(failed_attempt) + 1
    label = str(kind or "LLM").strip() or "LLM"
    return (
        f"{label} — reconnecting, "
        f"retry {next_attempt}/{max_attempts} in {int(wait_sec)}s"
    )


def notify_llm_reconnecting(
    *,
    failed_attempt: int,
    max_attempts: int,
    wait_sec: float,
    kind: str = "model gateway 503",
    exc: Optional[BaseException] = None,
    on_event: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> str:
    """Surface a non-error reconnecting line (one terminal path, no duplicate)."""
    del exc  # reserved for richer diagnostics
    preview = reconnect_preview(
        failed_attempt=failed_attempt,
        max_attempts=max_attempts,
        wait_sec=wait_sec,
        kind=kind,
    )
    emitted = False
    try:
        from src.orchestrators.host_adapters.session_binding import (
            lookup_live_runtime_session,
        )
        from src.orchestrators.rlm.runtime.session import ACTIVE_RLM_SESSION_ID

        sid = str(ACTIVE_RLM_SESSION_ID.get() or "").strip()
        if sid:
            sess = lookup_live_runtime_session(sid)
            if sess is not None:
                # Single path: append via _emit (durable terminal + on_event with
                # rlm.log_lines). HubEventBridge drains that once — do not also
                # upsert+on_event (that doubled / fragmented thinking cards).
                try:
                    try:
                        sess._llm_gateway_outage_seen = True  # noqa: SLF001
                    except Exception:
                        pass
                    sess._emit(  # noqa: SLF001
                        "llm_reconnect",
                        preview=preview,
                        attempt=int(failed_attempt) + 1,
                        max_attempts=int(max_attempts),
                        wait_sec=float(wait_sec),
                        persist=False,
                    )
                    emitted = True
                except Exception:
                    emitted = False
    except Exception:
        emitted = False
    if not emitted and on_event is not None:
        try:
            on_event(
                {
                    "kind": "llm_reconnect",
                    "preview": preview,
                    "attempt": int(failed_attempt) + 1,
                    "max_attempts": int(max_attempts),
                    "wait_sec": float(wait_sec),
                }
            )
        except Exception:
            pass
    return preview


async def call_with_llm_retry(
    call: AsyncCall,
    *,
    on_retry: Optional[OnRetry] = None,
) -> Any:
    """Retry transient model 503s and stream transport drops (AF parity)."""
    last_exc: Optional[BaseException] = None
    ceiling = max(LLM_503_RETRY_ATTEMPTS, LLM_TRANSPORT_RETRY_ATTEMPTS)
    for attempt in range(1, ceiling + 1):
        try:
            return await call()
        except Exception as exc:
            last_exc = exc
            plan = llm_retry_plan(exc)
            if plan is None:
                raise
            plan_max, wait_sec, kind = plan
            if attempt >= plan_max:
                raise
            if on_retry is not None:
                try:
                    maybe = on_retry(
                        failed_attempt=attempt,
                        max_attempts=plan_max,
                        wait_sec=wait_sec,
                        kind=kind,
                        exc=exc,
                    )
                    if hasattr(maybe, "__await__"):
                        await maybe  # type: ignore[misc]
                except Exception:
                    pass
            logger.warning(
                "RLM LLM %s retry %s/%s in %ss: %s",
                kind,
                attempt,
                plan_max,
                wait_sec,
                exc,
            )
            await asyncio.sleep(wait_sec)
    assert last_exc is not None
    raise last_exc
