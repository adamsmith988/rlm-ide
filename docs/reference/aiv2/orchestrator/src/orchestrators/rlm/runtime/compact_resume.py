"""Parse compaction checkpoints and build post-compact resume guidance."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

_SECTION_RE = re.compile(
    r"(?im)^##\s*(Goal|Progress|Next Steps|Dead Ends|Artifacts\s*&\s*Paths|"
    r"Constraints\s*&\s*Preferences|Key Decisions|Critical Context|"
    r"Media\s*/\s*Looks Already Reviewed|Kernel\s*/\s*REPL|"
    r"Subagents\s*&\s*Parallel Work)\s*$"
)

COMPACTION_RESUME_PREFIX = (
    "The conversation history before this point was compacted. "
    "Resume from the checkpoint below — do not restate the whole plan, "
    "do not re-explore Dead Ends, and take the first concrete Next Step "
    "(prefer an `ipython` action). Kernel variables and `offload/` / `pool/` "
    "files remain available.\n\n"
)

CHAT_TURN_REQUEST_SECTION = (
    "## The user's current message (reply to this now)\n"
    "{request}\n\n"
    "The checkpoint above is background context — do not repeat or restate it. "
    "Reply to the user's current message, following any format it asks for."
)


def resolve_compact_model(cfg: Any, session_model: str) -> str:
    """``cfg.compact_model`` → env ``AIV2_RLM_COMPACT_MODEL`` → session model."""
    model = ""
    if isinstance(cfg, dict):
        model = str(cfg.get("compact_model") or "").strip()
    if not model:
        model = str(os.environ.get("AIV2_RLM_COMPACT_MODEL") or "").strip()
    if not model:
        model = str(session_model or "").strip()
    return model


def parse_checkpoint_sections(summary: str) -> Dict[str, str]:
    """Split a structured compaction summary into named section bodies."""
    text = str(summary or "").strip()
    out: Dict[str, str] = {}
    if not text:
        return out
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        out["raw"] = text
        return out
    for i, m in enumerate(matches):
        name = re.sub(r"\s+", " ", m.group(1).strip()).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        out[name] = body
    return out


def _bullet_lines(section_body: str, *, limit: int = 8) -> List[str]:
    lines: List[str] = []
    for raw in str(section_body or "").splitlines():
        s = raw.strip()
        if not s or s.lower() in ("(none)", "- (none)", "* (none)"):
            continue
        if s.startswith(("#",)):
            continue
        lines.append(s)
        if len(lines) >= limit:
            break
    return lines


def build_continue_nudge(summary: str, current_request: str = "") -> str:
    """User message injected after compact so the model resumes executably.

    ``current_request`` (hub chat) is the turn's user message, kept verbatim
    last so the model answers it instead of echoing the checkpoint.
    """
    sections = parse_checkpoint_sections(summary)
    parts: List[str] = [COMPACTION_RESUME_PREFIX.rstrip(), ""]
    goal = sections.get("goal") or ""
    if goal:
        parts.append("## Goal")
        parts.append(goal.strip()[:800])
        parts.append("")
    next_steps = _bullet_lines(sections.get("next steps") or "", limit=6)
    if next_steps:
        parts.append("## Next Steps (do these now)")
        for line in next_steps:
            parts.append(line if line[:1].isdigit() or line.startswith(("-", "*")) else f"- {line}")
        parts.append("")
    else:
        parts.append("## Next Steps (do these now)")
        parts.append("1. Take one concrete `ipython` action toward the Goal.")
        parts.append("")
    dead = _bullet_lines(sections.get("dead ends") or "", limit=6)
    if dead:
        parts.append("## Dead Ends (do NOT retry)")
        for line in dead:
            parts.append(line if line.startswith(("-", "*")) else f"- {line}")
        parts.append("")
    media = _bullet_lines(sections.get("media / looks already reviewed") or "", limit=8)
    if media:
        parts.append("## Already reviewed (do NOT re-look)")
        for line in media:
            parts.append(line if line.startswith(("-", "*")) else f"- {line}")
        parts.append("")
    parts.append(
        "Kernel state survived compaction. Large prior tool dumps may live under "
        "`offload/` — open by path; do not paste dumps into chat."
    )
    request = str(current_request or "").strip()
    if request:
        parts.append("")
        parts.append(CHAT_TURN_REQUEST_SECTION.format(request=request))
    return "\n".join(parts).strip() + "\n"


def why_not_memory_content(summary: str) -> str:
    """Harness memory body for durable why-not / dead ends."""
    sections = parse_checkpoint_sections(summary)
    bits: List[str] = []
    goal = (sections.get("goal") or "").strip()
    if goal:
        bits.append(f"Goal: {goal[:500]}")
    dead = _bullet_lines(sections.get("dead ends") or "", limit=12)
    if dead:
        bits.append("Dead ends:")
        bits.extend(dead)
    next_steps = _bullet_lines(sections.get("next steps") or "", limit=6)
    if next_steps:
        bits.append("Next steps at compact:")
        bits.extend(next_steps)
    if not bits:
        raw = str(summary or "").strip()[:1500]
        return raw or "(empty checkpoint)"
    return "\n".join(bits)[:8000]


_LOOK_PATH_RE = re.compile(
    r"""(?:rlm\.)?look\s*\(\s*['"]([^'"]+)['"]""",
    re.IGNORECASE,
)


def extract_look_path(code: str) -> Optional[str]:
    m = _LOOK_PATH_RE.search(str(code or ""))
    if not m:
        return None
    return str(m.group(1) or "").strip() or None
