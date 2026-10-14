"""Format retrieved corpus hits for injection into the system prompt."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


_RAG_HEADER = """## Retrieved corpus context
Use the following indexed corpus snippets as evidence when answering.
Cite sources by [n]. If evidence is weak or unrelated, say so; do not invent \
facts from outside these snippets when the question is knowledge-seeking."""


def _truncate(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if max_chars <= 0 or len(t) <= max_chars:
        return t
    return t[: max(0, max_chars - 3)].rstrip() + "..."


def format_rag_prompt_block(
    hits: List[Dict[str, Any]],
    *,
    max_chars_per_hit: int = 1200,
) -> str:
    """Build the system-prompt section for corpus RAG hits.

    Each hit may include: ``text``, ``filename``, ``artifact_id``, ``score``.
    Returns empty string when there are no usable hits.
    """
    blocks: List[str] = []
    for i, hit in enumerate(hits or [], 1):
        if not isinstance(hit, dict):
            continue
        text = _truncate(
            str(
                hit.get("text")
                or hit.get("full_text")
                or hit.get("text_preview")
                or ""
            ),
            max_chars_per_hit,
        )
        if not text:
            continue
        filename = str(hit.get("filename") or "").strip() or "unknown"
        artifact_id = str(hit.get("artifact_id") or "").strip() or "-"
        score = hit.get("combined_score")
        if score is None:
            score = hit.get("score")
        try:
            score_s = f"{float(score):.3f}"
        except (TypeError, ValueError):
            score_s = "-"
        blocks.append(
            f"[{i}] file={filename} artifact_id={artifact_id} score={score_s}\n{text}"
        )
    if not blocks:
        return ""
    return _RAG_HEADER + "\n\n" + "\n\n".join(blocks)


def extract_hit_text(payload: Optional[Dict[str, Any]], result: Optional[Dict[str, Any]] = None) -> str:
    """Best-effort text extraction from a Storage/Qdrant search row."""
    payload = payload if isinstance(payload, dict) else {}
    result = result if isinstance(result, dict) else {}
    for key in ("full_text", "text", "text_preview"):
        val = result.get(key) or payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""
