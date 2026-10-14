import datetime as dt

from src.config import settings
from src.context import turn_guidance as tg

NOW = dt.datetime(2026, 10, 10, 23, 30, tzinfo=dt.timezone.utc)


def test_clock_uses_client_zone_when_valid_else_setting(monkeypatch):
    monkeypatch.setattr(settings, "HOST_CLOCK_TIMEZONE", "UTC")
    assert "Saturday, 2026-10-10 (UTC)" in tg.clock_block({}, now=NOW)
    assert "Sunday, 2026-10-11 (Asia/Tehran)" in tg.clock_block({"timezone": "Asia/Tehran"}, now=NOW)
    assert "(UTC)" in tg.clock_block({"timezone": "Not/AZone"}, now=NOW)
    monkeypatch.setattr(settings, "HOST_CLOCK_TEMPLATE", "")
    assert tg.clock_block({}, now=NOW) == ""


def test_persona_from_session_system_prompt(monkeypatch):
    assert tg.persona_block("") == ""
    assert "You are Sunny." in tg.persona_block("You are Sunny.")
    monkeypatch.setattr(settings, "HOST_PERSONA_MAX_CHARS", 5)
    assert tg.persona_block("You are Sunny.").endswith("You a…")


def test_loop_guard_after_repeated_user_messages(monkeypatch):
    monkeypatch.setattr(settings, "USER_LOOP_REPEATS", 3)
    same = "I still don't understand how to reset my password"
    hist = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": same},
        {"role": "assistant", "content": "Click reset."},
    ]
    assert tg.repeated_user_turns(same, hist) == 2
    assert tg.loop_guard_block(same, hist) == ""
    hist += [{"role": "user", "content": same + "!"}, {"role": "assistant", "content": "Click reset."}]
    block = tg.loop_guard_block(same, hist)
    assert "3 times" in block and "human" in block
    assert tg.loop_guard_block("What is the weather?", hist) == ""


def test_build_drops_empty_blocks():
    out = tg.build_guidance_blocks(user_message="hi", chat_history=[], session_record={"system_prompt": None})
    assert set(out) == {"clock"}
