"""Bundled helper skills preloaded into the RLM worker namespace.

These stay inside the plugin package — no global venv or modules/tools install.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional


def read_text(path: str, *, max_chars: int = 20000) -> str:
    """Read a text file with an explicit truncation marker (never silent truncate).

    Large files are *not* rejected — callers get the head plus a hint to continue
    via ``read_lines`` / ``grep_file`` / ``file_stats``. Default cap is 20k chars
    so the model does not dump multi-MB bodies into the chat context.
    """
    cap = max(1, int(max_chars))
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read(cap + 1)
    if len(text) <= cap:
        return text
    text = text[:cap]
    size_note = ""
    try:
        size_note = f", file_size={os.path.getsize(path)}"
    except OSError:
        pass
    return (
        text
        + f"\n…[truncated at {cap} chars{size_note} — not EOF; "
        f"use skills.read_lines(path, start=N, count=M), skills.grep_file(...), "
        f"or skills.file_stats(path); keep large text in variables, do not re-paste]"
    )


def write_text(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(str(text or ""))
    return path


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str, data: Any) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return path


def list_dir(path: str = ".") -> List[str]:
    try:
        return sorted(os.listdir(path or "."))
    except Exception:
        return []


def session_cwd() -> str:
    return str(os.environ.get("PWD") or os.getcwd())


def preview(value: Any, *, max_chars: int = 4000) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) > max_chars:
        return text[: max_chars - 16] + "\n…[truncated]"
    return text


def read_lines(
    path: str,
    *,
    start: int = 1,
    count: int = 100,
    max_chars: int = 20000,
) -> str:
    """Read a line window from a text file (1-based ``start``).

    When more content remains, append an explicit footer with ``next start=N`` so
    the model can paginate instead of assuming EOF.
    """
    lo = max(1, int(start))
    take = max(1, int(count))
    cap = max(1, int(max_chars))
    lines: List[str] = []
    used = 0
    last_idx = lo - 1
    stopped_by = ""
    has_more = False
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for idx, line in enumerate(fh, start=1):
            if idx < lo:
                continue
            if len(lines) >= take:
                # Iterator yielded a line past the window → more content exists.
                stopped_by = "count"
                has_more = True
                break
            if used + len(line) > cap:
                remain = cap - used
                if remain > 0:
                    lines.append(line[:remain] + "\n…[truncated]")
                    last_idx = idx
                stopped_by = "max_chars"
                has_more = True
                break
            lines.append(line)
            used += len(line)
            last_idx = idx
    body = "".join(lines)
    if not has_more:
        return body
    next_start = last_idx + 1
    return (
        body
        + f"\n…[window end at line {last_idx}; next start={next_start}; "
        f"stopped_by={stopped_by} — not EOF; call skills.read_lines again or "
        f"skills.grep_file / skills.file_stats]"
    )


def grep_file(
    path: str,
    pattern: str,
    *,
    max_matches: int = 50,
    context: int = 0,
    ignore_case: bool = False,
) -> List[str]:
    """Return matching lines (optionally with ±context) capped at ``max_matches``."""
    flags = re.IGNORECASE if ignore_case else 0
    rx = re.compile(str(pattern or ""), flags)
    ctx = max(0, int(context))
    limit = max(1, int(max_matches))
    hits: List[str] = []
    window: List[str] = []
    capped = False
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for idx, line in enumerate(fh, start=1):
            window.append(line)
            if len(window) > (2 * ctx + 1):
                window.pop(0)
            if not rx.search(line):
                continue
            if ctx:
                start_no = idx - len(window) + 1
                block = []
                for off, wline in enumerate(window):
                    block.append(f"{start_no + off}:{wline.rstrip()}")
                hits.append("\n".join(block))
            else:
                hits.append(f"{idx}:{line.rstrip()}")
            if len(hits) >= limit:
                # Peek whether more matches exist so we can mark the cap.
                for more in fh:
                    if rx.search(more):
                        capped = True
                        break
                break
    if capped:
        hits.append(
            f"…[grep capped at {limit} matches — refine pattern, raise max_matches, "
            f"or continue from a later line window; not exhaustive]"
        )
    return hits


def file_stats(path: str, *, max_line_scan: int = 500000) -> Dict[str, Any]:
    """Cheap metadata so the model can choose grep/window reads over full dumps."""
    st = os.stat(path)
    size_bytes = int(st.st_size)
    line_count: Optional[int] = 0
    truncated_scan = False
    with open(path, "rb") as fh:
        chunk = fh.read(8192)
        if b"\x00" in chunk[:2048]:
            return {
                "path": path,
                "size_bytes": size_bytes,
                "line_count": None,
                "binary": True,
            }
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for idx, _ in enumerate(fh, start=1):
            if idx > max(1, int(max_line_scan)):
                truncated_scan = True
                line_count = None
                break
            line_count = idx
    out: Dict[str, Any] = {
        "path": path,
        "size_bytes": size_bytes,
        "line_count": line_count,
        "binary": False,
    }
    if truncated_scan:
        out["line_count_truncated"] = True
    return out


__all__ = [
    "file_stats",
    "grep_file",
    "list_dir",
    "preview",
    "read_json",
    "read_lines",
    "read_text",
    "session_cwd",
    "write_json",
    "write_text",
]
