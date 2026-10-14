"""Append-only full RLM transcript (Prime-like, linear chain).

Source of truth for multi-turn chat continuity. The slim ``transcript.jsonl``
written by ``persist_session_files`` remains a UI projection only.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

TRANSCRIPT_VERSION = 1
FULL_TRANSCRIPT_NAME = "transcript.full.jsonl"

_MESSAGE_ROLES = frozenset({"system", "user", "assistant", "tool"})


def full_transcript_path(artifact_dir: str) -> str:
    return os.path.join(str(artifact_dir or ""), FULL_TRANSCRIPT_NAME)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_entry_id() -> str:
    return secrets.token_hex(4)


def write_session_header(
    artifact_dir: str,
    *,
    session_id: str,
    run_id: str = "",
    flow_id: str = "",
    node_id: str = "",
    user_id: str = "",
    rlm_depth: int = 0,
) -> None:
    path = full_transcript_path(artifact_dir)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(artifact_dir or ".", exist_ok=True)
    header = {
        "type": "session",
        "version": TRANSCRIPT_VERSION,
        "id": str(session_id or ""),
        "run_id": str(run_id or ""),
        "flow_id": str(flow_id or ""),
        "node_id": str(node_id or ""),
        "user_id": str(user_id or ""),
        "cwd": str(artifact_dir or ""),
        "timestamp": _iso_now(),
        "rlm_depth": int(rlm_depth or 0),
    }
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")


def _last_entry_id(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    last: Optional[str] = None
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict) and rec.get("id") and rec.get("type") != "session":
                    last = str(rec["id"])
    except Exception:
        return last
    return last


def append_entry(artifact_dir: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    path = full_transcript_path(artifact_dir)
    os.makedirs(artifact_dir or ".", exist_ok=True)
    row = dict(entry)
    if not row.get("id"):
        row["id"] = new_entry_id()
    if "parent_id" not in row:
        row["parent_id"] = _last_entry_id(path)
    if not row.get("ts"):
        row["ts"] = _iso_now()
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return row


def append_message(artifact_dir: str, message: Dict[str, Any]) -> Dict[str, Any]:
    """Persist one OpenAI-shaped message verbatim (tool_calls / tool_call_id kept)."""
    if not isinstance(message, dict):
        return {}
    msg = {k: v for k, v in message.items() if k != "_omit_from_transcript"}
    return append_entry(
        artifact_dir,
        {"type": "message", "message": msg},
    )


def append_custom_message(
    artifact_dir: str,
    *,
    custom_type: str,
    content: Any,
    display: bool = False,
) -> Dict[str, Any]:
    return append_entry(
        artifact_dir,
        {
            "type": "custom_message",
            "custom_type": str(custom_type or ""),
            "content": content,
            "display": bool(display),
        },
    )


def append_compaction(
    artifact_dir: str,
    *,
    summary: str,
    first_kept_entry_id: str,
    tokens_before: int = 0,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return append_entry(
        artifact_dir,
        {
            "type": "compaction",
            "summary": str(summary or ""),
            "first_kept_entry_id": str(first_kept_entry_id or ""),
            "tokens_before": int(tokens_before or 0),
            "details": dict(details or {}),
        },
    )


def append_kernel_state(
    artifact_dir: str,
    *,
    op: str,
    saved: Optional[List[str]] = None,
    restored: Optional[List[str]] = None,
    skipped: Optional[List[Any]] = None,
    failed: Optional[List[Any]] = None,
    bytes_size: int = 0,
) -> Dict[str, Any]:
    return append_entry(
        artifact_dir,
        {
            "type": "kernel_state",
            "op": str(op or ""),
            "saved": list(saved or []),
            "restored": list(restored or []),
            "skipped": list(skipped or []),
            "failed": list(failed or []),
            "bytes": int(bytes_size or 0),
        },
    )


def load_entries(artifact_dir: str) -> List[Dict[str, Any]]:
    path = full_transcript_path(artifact_dir)
    if not os.path.isfile(path):
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except Exception:
        return out
    return out


def _entry_to_message(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    et = str(entry.get("type") or "")
    if et == "message":
        msg = entry.get("message")
        return dict(msg) if isinstance(msg, dict) else None
    if et == "custom_message":
        content = entry.get("content")
        if content is None or content == "":
            return None
        return {"role": "user", "content": content}
    if et == "compaction":
        summary = str(entry.get("summary") or "").strip()
        if not summary:
            return None
        return {
            "role": "user",
            "content": f"[context checkpoint]\n{summary}",
            "_compaction": True,
        }
    return None


def repair_tool_pairs(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop orphan tool results; strip tool_calls whose results were all dropped."""
    msgs = [dict(m) for m in messages if isinstance(m, dict)]
    call_ids: set[str] = set()
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if isinstance(tc, dict) and tc.get("id"):
                call_ids.add(str(tc["id"]))
    kept: List[Dict[str, Any]] = []
    result_ids: set[str] = set()
    for m in msgs:
        if m.get("role") == "tool":
            tid = str(m.get("tool_call_id") or "").strip()
            if tid and tid not in call_ids:
                continue
            if tid:
                result_ids.add(tid)
            kept.append(m)
            continue
        kept.append(m)
    out: List[Dict[str, Any]] = []
    for m in kept:
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            out.append(m)
            continue
        tcs = [
            tc
            for tc in (m.get("tool_calls") or [])
            if isinstance(tc, dict)
            and (not tc.get("id") or str(tc["id"]) in result_ids)
        ]
        row = dict(m)
        if tcs:
            row["tool_calls"] = tcs
        else:
            row.pop("tool_calls", None)
        out.append(row)
    return out


def build_session_context(
    entries: Sequence[Dict[str, Any]],
    *,
    system_prompt: str = "",
) -> Dict[str, Any]:
    """Rebuild LLM messages from durable transcript (summary-first after compaction)."""
    rows = [e for e in entries if isinstance(e, dict) and e.get("type") != "session"]
    compaction_idx = -1
    for i, e in enumerate(rows):
        if e.get("type") == "compaction":
            compaction_idx = i

    def collect(slice_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        msgs: List[Dict[str, Any]] = []
        for e in slice_rows:
            m = _entry_to_message(e)
            if m:
                msgs.append(m)
        return msgs

    if compaction_idx < 0:
        messages = collect(rows)
    else:
        compaction = rows[compaction_idx]
        first_kept = str(compaction.get("first_kept_entry_id") or "").strip()
        retained: List[Dict[str, Any]] = []
        found = not first_kept
        for e in rows[:compaction_idx]:
            if first_kept and str(e.get("id") or "") == first_kept:
                found = True
            if not found:
                continue
            m = _entry_to_message(e)
            if m and not m.get("_compaction"):
                retained.append(m)
        summary_msg = _entry_to_message(compaction)
        tail = collect(rows[compaction_idx + 1 :])
        messages = ([summary_msg] if summary_msg else []) + retained + tail

    messages = repair_tool_pairs(messages)
    # Drop embedded system messages; caller re-anchors with fresh system prompt.
    messages = [m for m in messages if str(m.get("role") or "") != "system"]
    for m in messages:
        m.pop("_compaction", None)
    if system_prompt:
        messages = [{"role": "system", "content": str(system_prompt)}] + messages
    return {
        "messages": messages,
        "retained_message_count": len(messages),
        "has_compaction": compaction_idx >= 0,
    }


def hydrate_messages_from_disk(
    artifact_dir: str, *, system_prompt: str = ""
) -> List[Dict[str, Any]]:
    ctx = build_session_context(load_entries(artifact_dir), system_prompt=system_prompt)
    return list(ctx.get("messages") or [])


def sync_messages_to_transcript(
    artifact_dir: str,
    messages: Sequence[Dict[str, Any]],
    *,
    already_written: int = 0,
) -> int:
    """Append any new in-memory messages not yet written. Returns new write count."""
    written = max(0, int(already_written or 0))
    rows = [m for m in messages if isinstance(m, dict)]
    for msg in rows[written:]:
        append_message(artifact_dir, msg)
        written += 1
    return written
