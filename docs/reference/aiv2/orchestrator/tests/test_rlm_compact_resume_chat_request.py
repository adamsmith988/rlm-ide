"""RLM post-compaction nudge keeps the hub chat turn's request verbatim and last."""

from src.orchestrators.rlm.runtime.compact_resume import build_continue_nudge

CHECKPOINT = "## Goal\nRun the bakery.\n\n## Next Steps\n1. Check stock.\n"


def test_chat_request_is_appended_last_and_verbatim():
    body = build_continue_nudge(CHECKPOINT, current_request="Reply with exactly: noted 3")
    assert body.rstrip().endswith("following any format it asks for.")
    assert "Reply with exactly: noted 3" in body
    assert body.index("Reply with exactly: noted 3") > body.index("## Goal")


def test_without_request_the_nudge_is_unchanged():
    assert build_continue_nudge(CHECKPOINT) == build_continue_nudge(CHECKPOINT, current_request="  ")
    assert "current message" not in build_continue_nudge(CHECKPOINT)
