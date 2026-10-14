"""Hub RLM final-answer resolve — never empty «پاسخی تولید نشد» on known failures."""

from __future__ import annotations

from src.orchestrators.host_adapters.hub_final_answer import (
    CONTEXT_OVERFLOW_NOTICE,
    ensure_nonempty_hub_answer,
    friendly_llm_outage_notice,
    incomplete_notice,
    is_llm_gateway_outage,
    is_placeholder_answer,
    resolve_hub_final_answer,
)
from src.orchestrators.host_adapters.hub_sse import HubEventBridge


def test_gateway_outage_classifier():
    assert is_llm_gateway_outage(
        "all_backends_failed: openai_compat:osbi:ConnectError: dns"
    )
    assert is_llm_gateway_outage("model gateway 503 — reconnecting, retry 2/12")
    assert is_llm_gateway_outage("[reconnecting] model gateway 503")
    assert not is_llm_gateway_outage("maximum context length exceeded")


def test_placeholder_detector():
    assert is_placeholder_answer("پاسخی تولید نشد.")
    assert is_placeholder_answer("_(RLM finished — see process log)_")
    assert is_placeholder_answer("")
    assert not is_placeholder_answer("گزارش نهایی شخصیت‌ها")


def test_resolve_ignores_stale_placeholder_in_history():
    """Prior hub failures left «پاسخی تولید نشد» on messages — must not replay."""

    class _Sess:
        status = "idle"
        error = None
        _autonomous_stop_reason = None
        log_lines = ["[reconnecting] model gateway 503 — retry 2/12 in 10s"]
        last_content = ""
        last_reasoning = ""
        messages = [
            {"role": "assistant", "content": "پاسخی تولید نشد."},
            {"role": "user", "content": "again"},
        ]

        def _visible_session_output(self):
            # Old helper walked history and returned the placeholder — hub must ignore.
            return "پاسخی تولید نشد."

    text, status = resolve_hub_final_answer(
        runtime_session=_Sess(),
        thinking_text="[reconnecting] model gateway 503 — retry 1/12",
    )
    assert "مدل در دسترس نیست" in text
    assert not is_placeholder_answer(text)
    assert "see process log" not in text.lower()


def test_resolve_incomplete_overflow_notice():
    class _Sess:
        status = "incomplete"
        error = "overflow"
        _autonomous_stop_reason = "context_overflow"
        log_lines = ["[incomplete] context overflow"]
        last_content = ""
        last_reasoning = ""
        messages = []

    text, status = resolve_hub_final_answer(runtime_session=_Sess())
    assert text == CONTEXT_OVERFLOW_NOTICE
    assert status == "incomplete"


def test_ensure_replaces_persian_empty_with_outage():
    out = ensure_nonempty_hub_answer(
        "پاسخی تولید نشد.",
        thinking="[reconnecting] model gateway 503\nall_backends_failed: osbi",
    )
    assert "مدل در دسترس نیست" in out
    assert out != "پاسخی تولید نشد."
    assert "see process log" not in out.lower()


def test_ensure_unglues_process_log_prefix():
    glued = (
        "_(RLM finished — see process log)_"
        "کار ناتمام ماند. وضعیت جلسه حفظ شده؛ می‌توانید ادامه دهید."
    )
    out = ensure_nonempty_hub_answer(glued)
    assert out.startswith("کار ناتمام ماند")
    assert "see process log" not in out.lower()


def test_ensure_no_english_process_log():
    out = ensure_nonempty_hub_answer("", thinking="[kernel] ready\n[ipython] x=1")
    assert "see process log" not in out.lower()
    assert "ناتمام" in out or "ادامه" in out
    assert "no_visible_answer" not in out


def test_hub_sse_run_error_emits_answer():
    bridge = HubEventBridge()
    notice = friendly_llm_outage_notice("all_backends_failed: x")
    out = bridge.map_event(
        {
            "kind": "run_error",
            "error": "all_backends_failed: x",
            "notice": notice,
            "status": "incomplete",
        }
    )
    types = [e["type"] for e in out]
    assert "error" in types
    assert "answer" in types
    assert bridge.collected_answer == notice


def test_incomplete_notice_helper():
    assert "context" in incomplete_notice("context_overflow").lower() or "Context" in incomplete_notice(
        "context_overflow"
    )
