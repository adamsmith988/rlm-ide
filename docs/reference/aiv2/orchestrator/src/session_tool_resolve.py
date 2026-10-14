"""
Server-side resolution for session file tools (Continue-style preprocessArgs).

Fills file_id / line ranges before tool-runner validation when the model
omits locators or asks to continue reading.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_SESSION_READ_TOOLS = frozenset({
    "session_read_file",
    "session_read",
    "plugin_session_read",
})


def user_wants_case_lookup(text: str) -> bool:
    """User asks for a case/file name in storage (پرونده), not graph relations."""
    s = (text or "").strip().lower()
    if not s:
        return False
    markers = (
        "پرونده",
        "نام پرونده",
        "نام فایل",
        "case name",
        "what is the name",
        "چه نام",
        "چه اسمی",
        "نام دارد",
        "نامش",
        "کدوم پرونده",
    )
    return any(m in s for m in markers)


def extract_attach_filename_hint(text: str) -> Optional[str]:
    """Best-effort person/case token for session_attach(filename=...)."""
    s = (text or "").strip()
    if not s:
        return None
    skip = frozenset(
        {
            "پرونده",
            "نام",
            "فایل",
            "چه",
            "دارد",
            "است",
            "باشد",
            "the",
            "case",
            "file",
        }
    )
    for pattern in (
        r"پرونده\s*(?:ی|ی)?\s*([\u0600-\u06FF]{2,24})",
        r"([\u0600-\u06FF]{2,24})\s+چه\s+نام",
        r"برای\s+([\u0600-\u06FF]{2,24})",
        r"([\u0600-\u06FF]{2,24})\s+در\s+دیتابیس",
    ):
        m = re.search(pattern, s, re.I)
        if m:
            token = (m.group(1) or "").strip()
            if token and token not in skip:
                return token
    for m in re.finditer(r"[\u0600-\u06FF]{2,24}", s):
        token = m.group(0).strip()
        if token not in skip and len(token) >= 2:
            return token
    return None


def user_wants_file_discovery(text: str) -> bool:
    """User wants to locate which file contains something — not read a known file."""
    s = (text or "").strip().lower()
    if not s:
        return False
    if "پیدا کن در" in s or "find in" in s:
        return False
    markers = (
        "find the file",
        "which file",
        "what file",
        "locate the file",
        "file that contains",
        "contains",
        "فایلی که",
        "کدوم فایل",
        "چه فایلی",
        "کدام فایل",
        "پیدا کن",
        "پیدا کردن",
        "توش هست",
        "توشه",
        "داخلش",
        "داخلشه",
        "where is the file",
    )
    return any(m in s for m in markers)


def user_wants_file_read(text: str) -> bool:
    s = (text or "").strip().lower()
    if not s:
        return False
    if user_wants_file_discovery(s):
        return False
    markers = (
        "بخون", "بخوان", "بخونش", "بخوانش", "read it", "read the", "read file",
        "open the file", "show content", "file content", "محتوا", "محتوای",
        "contents of", "what is in", "what's in", "بررسی کن", "بررسی",
    )
    if any(m in s for m in markers):
        return True
    if re.search(r"(?:خط|line)\s*\d+", s, re.I):
        return True
    return False


def user_wants_more_file_content(text: str) -> bool:
    s = (text or "").strip().lower()
    if not s:
        return False
    markers = (
        "بیشتر", "ادامه", "بقیه", "more", "continue", "next page", "next lines",
        "rest of", "باقی", "صفحه بعد",
    )
    return any(m in s for m in markers)


def parse_line_range_from_user_message(text: str) -> Optional[Dict[str, int]]:
    s = text or ""
    m = re.search(r"خط\s*(?:و)?\s*(\d+)\s*(?:تا|الی|-)\s*(\d+)", s, re.I)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        if start > end:
            start, end = end, start
        return {"start_line": start, "end_line": end}
    m = re.search(r"lines?\s*(\d+)\s*(?:to|-)\s*(\d+)", s, re.I)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        if start > end:
            start, end = end, start
        return {"start_line": start, "end_line": end}
    m = re.search(r"(?:خط|line)\s*(\d+)\b", s, re.I)
    if m:
        n = int(m.group(1))
        return {"start_line": n, "end_line": n}
    return None


def history_has_readable_file_context(
    chat_history: Optional[List[Dict[str, Any]]],
) -> bool:
    if not chat_history:
        return False
    for row in reversed(chat_history):
        if not isinstance(row, dict):
            continue
        if row.get("role") != "assistant":
            continue
        for tr in row.get("tool_results") or []:
            if not isinstance(tr, dict):
                continue
            name = (tr.get("name") or tr.get("tool_id") or "").strip()
            if name not in _SESSION_READ_TOOLS:
                continue
            summary = (tr.get("summary") or "").strip()
            if tr.get("ok") and summary and "lines " in summary and " of " in summary:
                return True
            if summary.startswith("📄") and "lines " in summary:
                return True
    return False


def infer_next_line_range_from_history(
    chat_history: Optional[List[Dict[str, Any]]],
    *,
    page_size: int = 1000,
) -> Optional[Dict[str, int]]:
    if not chat_history:
        return None
    pat = re.compile(
        r"lines\s+(\d+)\s*[-–]\s*(\d+)\s+of\s+(\d+)",
        re.IGNORECASE,
    )
    for row in reversed(chat_history):
        if not isinstance(row, dict):
            continue
        chunks: List[str] = []
        if row.get("role") == "tool":
            chunks.append(str(row.get("content") or ""))
        if row.get("role") == "assistant":
            for tr in row.get("tool_results") or []:
                if isinstance(tr, dict):
                    chunks.append(str(tr.get("summary") or ""))
        for text in chunks:
            m = pat.search(text)
            if not m:
                continue
            start, end, total = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if end >= total:
                return None
            next_start = end + 1
            next_end = min(end + page_size, total)
            return {"start_line": next_start, "end_line": next_end}
    return None


def resolve_line_range_for_read(
    user_message: str,
    chat_history: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, int]]:
    explicit = parse_line_range_from_user_message(user_message)
    if explicit:
        return explicit
    if user_wants_more_file_content(user_message):
        return infer_next_line_range_from_history(chat_history)
    return None


def should_preloop_session_read(
    user_message: str,
    chat_history: Optional[List[Dict[str, Any]]],
) -> bool:
    if user_wants_file_read(user_message):
        return True
    if user_wants_more_file_content(user_message) and history_has_readable_file_context(
        chat_history
    ):
        return True
    return False


def _registry_entry_for_focus(
    file_registry: List[Dict[str, Any]],
    artifact_id: str,
) -> Optional[Dict[str, Any]]:
    aid = str(artifact_id or "").strip()
    if not aid:
        return None
    for entry in file_registry or []:
        if not isinstance(entry, dict):
            continue
        eid = str(
            entry.get("artifact_id") or entry.get("id") or entry.get("file_id") or ""
        )
        if eid == aid:
            return entry
    return None


def resolve_read_file_args(
    *,
    user_message: str,
    chat_history: Optional[List[Dict[str, Any]]],
    file_registry: Optional[List[Dict[str, Any]]],
    discourse_focus: Optional[Dict[str, Any]] = None,
    explicit_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build args for session_read_file: merge explicit model args with focus,
    smart_resolve output, and continuation line ranges.
    """
    args = dict(explicit_args or {})
    has_locator = any(
        args.get(k)
        for k in ("file_id", "artifact_id", "filename", "name", "path")
    )

    if user_wants_file_discovery(user_message):
        line_range = resolve_line_range_for_read(user_message, chat_history)
        if line_range:
            args.update(line_range)
        return args

    if not has_locator and isinstance(discourse_focus, dict):
        ids = discourse_focus.get("active_artifact_ids") or []
        if len(ids) == 1:
            args["file_id"] = str(ids[0])
            has_locator = True
        elif discourse_focus.get("last_text_file_id"):
            args["file_id"] = str(discourse_focus["last_text_file_id"])
            has_locator = True

    if not has_locator and file_registry and len(file_registry) == 1:
        only = file_registry[0]
        if isinstance(only, dict):
            fid = only.get("artifact_id") or only.get("id") or only.get("file_id")
            if fid:
                args["file_id"] = str(fid)
                has_locator = True

    line_range = resolve_line_range_for_read(user_message, chat_history)
    if line_range:
        args.update(line_range)

    return args


def hydrate_read_args_from_focus(
    tool_name: str,
    args: Dict[str, Any],
    *,
    discourse_focus: Optional[Dict[str, Any]] = None,
    file_registry: Optional[List[Dict[str, Any]]] = None,
    user_message: str = "",
) -> Dict[str, Any]:
    """Fill missing file_id when tool is session_read_file and focus is unambiguous."""
    if tool_name not in _SESSION_READ_TOOLS:
        return args
    if user_wants_file_discovery(user_message):
        return args
    out = dict(args)
    has_locator = any(
        out.get(k) for k in ("file_id", "artifact_id", "filename", "name", "path")
    )
    if has_locator:
        return out
    resolved = resolve_read_file_args(
        user_message="",
        chat_history=None,
        file_registry=file_registry,
        discourse_focus=discourse_focus,
        explicit_args=out,
    )
    return resolved
