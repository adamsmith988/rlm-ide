"""Boundary-aware text chunking (stdlib only).

Splits long text into overlapping windows that prefer to break on paragraph,
line, sentence, then word boundaries, so a fact never straddles two chunks
without also appearing whole in one of them (given ``overlap`` ≥ fact length).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

_BREAKS = ("\n\n", "\n", ". ", "? ", "! ", "؟ ", "。", "; ", ", ", " ")
_WS = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class TextChunk:
    index: int
    start: int
    end: int
    text: str


def _cut_point(text: str, start: int, hard_end: int, min_end: int) -> int:
    if hard_end >= len(text):
        return len(text)
    window = text[min_end:hard_end]
    for sep in _BREAKS:
        pos = window.rfind(sep)
        if pos != -1:
            return min_end + pos + len(sep)
    return hard_end


def chunk_text(text: str, *, max_chars: int, overlap: int = 0, max_chunks: int = 0) -> List[TextChunk]:
    """Overlapping chunks of at most ``max_chars``; ``max_chunks`` ≤ 0 means unbounded.

    When ``max_chunks`` would be exceeded the window grows instead of dropping
    text, so the whole input is always covered.
    """
    src = str(text or "")
    if not src.strip():
        return []
    max_chars = max(1, int(max_chars))
    overlap = max(0, min(int(overlap), max_chars // 2))
    if max_chunks and max_chunks > 0:
        step = max_chars - overlap
        needed = 1 + max(0, -(-(len(src) - max_chars) // max(1, step)))
        if needed > max_chunks:
            max_chars = -(-len(src) // max_chunks) + overlap
    out: List[TextChunk] = []
    start = 0
    while start < len(src):
        hard_end = min(len(src), start + max_chars)
        end = _cut_point(src, start, hard_end, start + max(1, (max_chars * 2) // 3))
        piece = _WS.sub(" ", src[start:end]).strip()
        if piece:
            out.append(TextChunk(index=len(out), start=start, end=end, text=piece))
        if end >= len(src):
            break
        nxt = max(start + 1, end - overlap)
        if overlap:
            ws = src.find(" ", nxt, end)
            if ws != -1:
                nxt = ws + 1
        start = nxt
    return out
