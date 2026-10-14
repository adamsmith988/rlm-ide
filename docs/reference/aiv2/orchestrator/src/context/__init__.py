"""Context package for orchestrator."""

from .compaction import maybe_compact_messages, generate_conversation_summary
from .transcript_search import (
    build_transcript_search_hint,
    search_transcript_via_session_search,
    should_offer_transcript_search,
)

__all__ = [
    "maybe_compact_messages",
    "generate_conversation_summary",
    "build_transcript_search_hint",
    "search_transcript_via_session_search",
    "should_offer_transcript_search",
]
