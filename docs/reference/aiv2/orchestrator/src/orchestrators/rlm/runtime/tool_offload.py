"""Spill large tool/ipython results out of the LLM transcript onto disk."""

from __future__ import annotations

import hashlib
import os
import re
from typing import Optional, Tuple

DEFAULT_OFFLOAD_CHARS = 16384

# Worker bash truncate appends this so the model can open the full dump.
_FULL_OUTPUT_LINE_RE = re.compile(
    r"^\[Full output:\s*(.+?)\]\s*$",
    re.MULTILINE,
)


def offload_threshold_chars() -> int:
    raw = (os.environ.get("AIV2_RLM_TOOL_OFFLOAD_CHARS") or "").strip()
    if not raw:
        return DEFAULT_OFFLOAD_CHARS
    try:
        return max(4096, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_OFFLOAD_CHARS


def _safe_ext(kind: str) -> str:
    k = re.sub(r"[^a-z0-9]+", "", str(kind or "txt").lower()) or "txt"
    return k[:12]


def extract_full_output_notes(text: str) -> str:
    """Keep bash ``[Full output: …]`` lines when the surrounding body is offloaded."""
    notes = []
    seen = set()
    for match in _FULL_OUTPUT_LINE_RE.finditer(str(text or "")):
        line = match.group(0).strip()
        if line and line not in seen:
            seen.add(line)
            notes.append(line)
    return "\n".join(notes)


def spill_content_to_offload(
    artifact_dir: str,
    content: str,
    *,
    kind: str = "bash",
) -> Optional[str]:
    """Always write ``content`` under ``artifact_dir/offload/``.

    Returns a session-relative path (``offload/<digest>.<ext>``) or ``None``
    when there is no durable directory / empty content.
    """
    text = str(content or "")
    if not text:
        return None
    root = str(artifact_dir or "").strip()
    if not root:
        return None
    off_dir = os.path.join(root, "offload")
    try:
        os.makedirs(off_dir, exist_ok=True)
    except OSError:
        return None
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    rel = os.path.join("offload", f"{digest}.{_safe_ext(kind)}")
    abs_path = os.path.join(root, rel)
    if not os.path.isfile(abs_path):
        try:
            with open(abs_path, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError:
            return None
    return rel


def offload_large_tool_content(
    artifact_dir: str,
    content: str,
    *,
    kind: str = "ipython",
) -> Tuple[str, Optional[str]]:
    """If ``content`` exceeds the threshold, write it under ``artifact_dir/offload/``.

    Returns ``(model_facing_text, absolute_path_or_None)``.
    Terminal / worker logs should already hold the full body before this runs.

    When the body already contains bash ``[Full output: …]`` spill markers, those
    lines are preserved on the stub so the model still sees the recoverable path
    after host-side offload of the truncated preview.
    """
    text = str(content or "")
    limit = offload_threshold_chars()
    if len(text) <= limit:
        return text, None
    full_notes = extract_full_output_notes(text)
    root = str(artifact_dir or "").strip()
    if not root:
        # No durable dir — hard-truncate with a hint rather than ballooning context.
        keep = max(2048, limit - 80)
        stub = (
            text[:keep]
            + f"\n…[truncated {len(text) - keep} chars — no artifact_dir for offload]"
        )
        if full_notes:
            stub = stub.rstrip() + "\n" + full_notes
        return stub, None
    off_dir = os.path.join(root, "offload")
    os.makedirs(off_dir, exist_ok=True)
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    rel = os.path.join("offload", f"{digest}.{_safe_ext(kind)}")
    abs_path = os.path.join(root, rel)
    if not os.path.isfile(abs_path):
        with open(abs_path, "w", encoding="utf-8") as fh:
            fh.write(text)
    stub = (
        f"[offloaded {len(text)} chars → {rel}; "
        f"use open/Path/skills.grep_file — do not re-paste]"
    )
    if full_notes:
        stub = stub + "\n" + full_notes
    return stub, abs_path
