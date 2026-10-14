"""Native corpus RAG for the orchestrator (preflight retrieve → prompt inject)."""

from .prompt import format_rag_prompt_block


def __getattr__(name: str):
    if name in {"RagResult", "retrieve_for_prompt"}:
        from .client import RagResult, retrieve_for_prompt

        return {"RagResult": RagResult, "retrieve_for_prompt": retrieve_for_prompt}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RagResult",
    "retrieve_for_prompt",
    "format_rag_prompt_block",
]
