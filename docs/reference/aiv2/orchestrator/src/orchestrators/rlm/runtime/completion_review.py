"""Host-owned completion review at quiescent boundaries (before status=completed)."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ChatFn = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]

MAX_COMPLETION_REVIEWS_PER_RUN = 12
# Tiny first-chunk files that only exist to satisfy `test -f …`.
STUB_DELIVERABLE_MAX_BYTES = 512
# Host-read excerpts embedded into the completion-review prompt (no tools).
REVIEW_FILE_HEAD_CHARS = 12000
REVIEW_FILE_TAIL_CHARS = 3000
REVIEW_FILES_TOTAL_CHARS = 40000
REVIEW_MAX_FILES = 8
_STUB_CONTENT_RE = re.compile(
    r"(?is)\b("
    r"TODO|FIXME|TBD|placeholder|stub|"
    r"to be (?:filled|expanded|written|completed)|"
    r"first chunk|expand (?:this|me|later)|outline only"
    r")\b"
)

COMPLETION_REVIEW_JSON_SCHEMA = """\
Return JSON only:
{
  "complete": true|false,
  "reason": "short reason",
  "missing": ["optional concrete next steps if complete is false"]
}
"""

# Kept for detectors / older imports; prefer build_completion_review_system_prompt().
COMPLETION_REVIEW_SYSTEM_PROMPT = (
    "You are the host completion reviewer for an RLM agent run.\n\n"
    + COMPLETION_REVIEW_JSON_SCHEMA
)


def build_completion_review_system_prompt(
    *,
    user_goal: str,
    gate_commands: Optional[List[str]] = None,
    artifact_required: bool = False,
    has_acceptance_contract: bool = False,
) -> str:
    """Build a run-specific system prompt: real gates and/or the process goal."""
    goal = str(user_goal or "").strip() or "(empty user task)"
    cmds = [str(c).strip() for c in (gate_commands or []) if str(c).strip()]
    lines = [
        "You are the host completion reviewer for an RLM agent run.",
        "",
        "Decide whether the ORIGINAL USER TASK / PROCESS GOAL is fully done "
        "based only on the evidence the host provides. Be skeptical of "
        "premature finishes and stub/first-chunk deliverables.",
        "",
        "<process_goal>",
        goal[:8000],
        "</process_goal>",
        "",
    ]
    if has_acceptance_contract:
        lines.extend(
            [
                "An acceptance checklist and host probe results are provided.",
                "Do NOT re-judge deterministic/structural probes that already PASS or FAIL.",
                "Focus on: (1) whether the worker claim matches the goal, "
                "(2) open semantic checklist items, (3) obvious stub/partial work.",
                "If blocking probes already failed, the host will reject regardless "
                "of your vote — still return complete=false.",
                "",
            ]
        )
    if cmds:
        listed = "\n".join(f"- `{c}`" for c in cmds[:32])
        lines.extend(
            [
                "Configured quality gates for this run (shell checks):",
                listed,
                "",
                "Important about these gates:",
                "- gate_result=passed means ONLY that those shell commands succeeded.",
                "- It is necessary host evidence, NOT proof the process_goal is done.",
                "- If a gate is a weak check (e.g. file exists) and the deliverable "
                "still looks like a stub/outline/first chunk, or the assistant still "
                "plans to expand/fill/rewrite it, return complete=false.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "No quality gates are configured for this run.",
                "Judge completion strictly against <process_goal> above and the "
                "host evidence (assistant text + checklist/claim/probes).",
                "Do not treat a short progress note or an empty/stub file as done "
                "unless process_goal is clearly fully satisfied.",
                "",
            ]
        )
    lines.extend(
        [
            f"artifact_required={bool(artifact_required)}",
            "- If artifact_required is true, complete=true only when deliverable "
            "evidence exists AND process_goal looks fully satisfied (not a partial draft).",
            "- If artifact_required is false, a clear user-facing answer may be enough.",
            "- Never invent files or gate results that are not listed in the evidence.",
            "- When <deliverable_contents> is present, read those host-provided "
            "excerpts carefully. If a file is marked truncated, do not assume the "
            "unread middle is complete enough for process_goal.",
            "- When <deliverable_contents> is (omitted), do not invent file bodies; "
            "rely on checklist, probes, and claim.",
            "",
            COMPLETION_REVIEW_JSON_SCHEMA.strip(),
        ]
    )
    return "\n".join(lines)


def _extract_json_object(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p2 = p.strip()
            if p2.startswith("json"):
                p2 = p2[4:].strip()
            if p2.startswith("{"):
                text = p2
                break
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


def parse_completion_review(raw: str) -> Dict[str, Any]:
    record = _extract_json_object(raw)
    missing_raw = record.get("missing")
    missing: List[str] = []
    if isinstance(missing_raw, list):
        missing = [str(x).strip() for x in missing_raw if str(x).strip()][:8]
    elif isinstance(missing_raw, str) and missing_raw.strip():
        missing = [missing_raw.strip()]
    return {
        "complete": record.get("complete") is True,
        "reason": str(record.get("reason") or "No rationale provided.")[:500],
        "missing": missing,
    }


def build_completion_review_rejection_message(review: Dict[str, Any]) -> Dict[str, Any]:
    reason = str(review.get("reason") or "The host completion review rejected finishing.")
    missing = review.get("missing") if isinstance(review.get("missing"), list) else []
    lines = [
        "[completion review] The host reviewer decided the task is NOT finished yet.",
        f"Reason: {reason}",
    ]
    if missing:
        lines.append("Still missing:")
        for item in missing:
            lines.append(f"- {item}")
    lines.append(
        "Continue working. Address the missing items. If a workspace deliverable "
        "file is required, write it in the next ipython cell, then summarize the path."
    )
    return {
        "role": "user",
        "content": "\n".join(lines),
        "_rlm_continuation": "completion_review",
        "_rlm_continuation_reason": "completion_review_rejected",
    }


def build_unchanged_deliverable_continuation(prev_reason: str = "") -> Dict[str, Any]:
    """Nudge when evidence/files did not change since the last review reject."""
    reason = str(prev_reason or "previous completion review still applies").strip()
    return {
        "role": "user",
        "content": (
            "[completion review] Deliverable is UNCHANGED since the last rejection "
            f"({reason}).\n"
            "You must make tangible progress NOW in the next ipython cell:\n"
            "1) Open/read the workspace deliverable path.\n"
            "2) Append or rewrite substantial content toward the full user task "
            "(not a TODO/stub/outline).\n"
            "3) Save the file, then briefly confirm the path and what you added.\n"
            "Do not only plan or narrate — write the file."
        ),
        "_rlm_continuation": "completion_review",
        "_rlm_continuation_reason": "completion_review_unchanged",
    }


def completion_review_evidence_fingerprint(
    *,
    evidence: str,
    gate_result: Optional[str],
    file_records: List[Dict[str, Any]],
    extra: str = "",
) -> str:
    """Stable fingerprint so identical deliverables do not burn review budget.

    When workspace files exist, fingerprint is gate + file size/mtime/preview
    only — assistant chatter changes must not re-trigger a full review.
    ``extra`` carries acceptance checklist/probe/claim fragments when present.
    """
    import hashlib

    rows: List[str] = [f"gate={gate_result or 'none'}"]
    file_rows: List[str] = []
    for rec in file_records or []:
        if not isinstance(rec, dict):
            continue
        name = str(rec.get("relpath") or rec.get("filename") or "").strip()
        if not name:
            continue
        size = rec.get("size_bytes")
        mtime = rec.get("mtime_ns")
        preview = str(rec.get("preview") or rec.get("excerpt_head") or "")[:400]
        file_rows.append(f"{name}|{size}|{mtime}|{preview}")
    if file_rows:
        rows.extend(sorted(file_rows))
    else:
        rows.append(f"evidence={str(evidence or '').strip()[:2500]}")
    extra_s = str(extra or "").strip()
    if extra_s:
        rows.append(f"acceptance={extra_s[:4000]}")
    raw = "\n".join(rows)
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:32]


def should_skip_completion_review(
    *,
    artifact_required: bool,
    evidence_text: str,
    host_evidence_ok: bool,
) -> bool:
    """Cheap chat exits (e.g. greetings) skip the extra LLM call."""
    if artifact_required:
        return False
    if not host_evidence_ok:
        return True
    return len(str(evidence_text or "").strip()) < 200


def looks_like_stub_deliverable(
    *,
    size_bytes: Optional[int],
    preview: str = "",
    artifact_required: bool = True,
) -> bool:
    """True for tiny/placeholder files that only exist to pass a gate."""
    if not artifact_required:
        return False
    try:
        size = int(size_bytes or 0)
    except (TypeError, ValueError):
        size = 0
    text = str(preview or "").strip()
    if size <= 0:
        return False
    # Empty / whitespace-only tiny files.
    if size < 8 and not text:
        return True
    if size <= STUB_DELIVERABLE_MAX_BYTES and _STUB_CONTENT_RE.search(text):
        return True
    return False


def any_stub_deliverable(
    records: List[Dict[str, Any]],
    *,
    artifact_required: bool = True,
) -> bool:
    if not artifact_required:
        return False
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        preview = str(
            rec.get("preview") or rec.get("excerpt_head") or rec.get("head") or ""
        ).strip()
        if looks_like_stub_deliverable(
            size_bytes=rec.get("size_bytes"),
            preview=preview,
            artifact_required=True,
        ):
            return True
    return False


def _resolve_deliverable_path(rec: Dict[str, Any], workspace_dir: str = "") -> str:
    path = str(rec.get("path") or "").strip()
    if path and os.path.isfile(path):
        return path
    name = str(rec.get("relpath") or rec.get("filename") or "").strip()
    root = str(workspace_dir or "").strip()
    if name and root:
        candidate = os.path.join(root, name)
        if os.path.isfile(candidate):
            return candidate
    return ""


def _read_text_excerpt(
    path: str,
    *,
    head_chars: int,
    tail_chars: int,
) -> Tuple[str, str, bool, int]:
    """Return (head, tail, truncated, size_bytes)."""
    try:
        size = int(os.path.getsize(path))
    except OSError:
        return "", "", False, 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(max(0, int(head_chars)))
            truncated = bool(fh.read(1))
        tail = ""
        if truncated and int(tail_chars) > 0:
            with open(path, "rb") as bf:
                bf.seek(0, os.SEEK_END)
                end = bf.tell()
                take = min(end, max(int(tail_chars) * 4, int(tail_chars)))
                bf.seek(max(0, end - take))
                raw_tail = bf.read()
            tail = raw_tail.decode("utf-8", errors="replace")
            if len(tail) > int(tail_chars):
                tail = tail[-int(tail_chars) :]
            # Tiny files / overlap: don't repeat the same window.
            if tail and head and tail in head:
                tail = ""
        return head, tail, truncated, size
    except OSError:
        return "", "", False, size


def load_deliverable_excerpts_for_review(
    records: List[Dict[str, Any]],
    *,
    workspace_dir: str = "",
    max_files: int = REVIEW_MAX_FILES,
    head_chars: int = REVIEW_FILE_HEAD_CHARS,
    tail_chars: int = REVIEW_FILE_TAIL_CHARS,
    total_chars: int = REVIEW_FILES_TOTAL_CHARS,
) -> Dict[str, Any]:
    """Host-read file excerpts for the completion reviewer (no tool use)."""
    summaries: List[str] = []
    enriched: List[Dict[str, Any]] = []
    blocks: List[str] = []
    used = 0
    budget = max(1000, int(total_chars or REVIEW_FILES_TOTAL_CHARS))
    limit = max(1, int(max_files or REVIEW_MAX_FILES))

    for rec in (records or [])[: limit * 2]:
        if len(enriched) >= limit or used >= budget:
            break
        if not isinstance(rec, dict):
            continue
        name = str(rec.get("relpath") or rec.get("filename") or "").strip()
        if not name:
            continue
        path = _resolve_deliverable_path(rec, workspace_dir)
        size = rec.get("size_bytes")
        head = ""
        tail = ""
        truncated = False
        if path:
            head, tail, truncated, size_read = _read_text_excerpt(
                path,
                head_chars=max(256, int(head_chars)),
                tail_chars=max(0, int(tail_chars)),
            )
            if size is None:
                size = size_read
        else:
            head = str(rec.get("preview") or rec.get("head") or "").strip()

        # Keep room for at least a short head of this file.
        remaining = budget - used
        if remaining < 200:
            break
        head_budget = min(len(head), remaining)
        head_out = head[:head_budget]
        used += len(head_out)
        tail_out = ""
        if truncated and tail and used < budget:
            tail_budget = min(len(tail), budget - used)
            # Avoid duplicating if file is small enough that head≈whole.
            if tail_budget > 0 and tail[-tail_budget:] not in head_out:
                tail_out = tail[-tail_budget:]
                used += len(tail_out)

        size_s = f"{size}B" if size is not None else "?"
        flag = "truncated=yes" if truncated or len(head) > len(head_out) else "truncated=no"
        if truncated and not tail_out and len(head) > len(head_out):
            flag = "truncated=yes"
        summary = f"{name} ({size_s}, {flag})"
        summaries.append(summary)

        parts = [
            f'<file path="{name}" size="{size_s}" {flag}>',
            "<head>",
            head_out if head_out else "(empty or unreadable)",
            "</head>",
        ]
        if tail_out:
            parts.extend(["<tail>", tail_out, "</tail>"])
        parts.append("</file>")
        blocks.append("\n".join(parts))

        enriched.append(
            {
                "relpath": name,
                "path": path,
                "size_bytes": size,
                "preview": (head_out or head)[:800],
                "excerpt_head": head_out,
                "excerpt_tail": tail_out,
                "truncated": truncated or len(head) > len(head_out),
            }
        )

    contents_block = "\n\n".join(blocks) if blocks else "(none)"
    return {
        "summaries": summaries,
        "records": enriched,
        "contents_block": contents_block,
    }


def body_is_completion_review(body: Dict[str, Any]) -> bool:
    """True when this chat request is the host completion-review call."""
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        text = str(m.get("content") or "")
        if "completion reviewer" in text.lower() or "<artifact_required>" in text:
            return True
    return False


def completion_review_accept_payload() -> Dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '{"complete": true, "reason": "ok", "missing": []}',
                }
            }
        ]
    }


async def review_completion(
    session: Any,
    *,
    chat_fn: ChatFn,
    extract_assistant_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    user_goal: str,
    evidence_text: str,
    deliverable_files: List[str],
    artifact_required: bool,
    gate_result: Optional[str] = None,
    gate_commands: Optional[List[str]] = None,
    model: Optional[str] = None,
    deliverable_details: Optional[List[str]] = None,
    deliverable_contents: Optional[str] = None,
    acceptance_checklist: Optional[str] = None,
    worker_claim: Optional[str] = None,
    probe_results: Optional[str] = None,
    include_excerpts: bool = True,
) -> Dict[str, Any]:
    cmds = [str(c).strip() for c in (gate_commands or []) if str(c).strip()]
    files_block = (
        "\n".join(f"- {n}" for n in (deliverable_details or deliverable_files or [])[:40])
        or "(none)"
    )
    contents_block = str(deliverable_contents or "").strip() or "(none)"
    gate_cmds_block = (
        "\n".join(f"- `{c}`" for c in cmds[:32]) if cmds else "(none configured)"
    )
    gate_note = (
        "Note: gate_result=passed only means the configured shell gates succeeded. "
        "It does NOT mean process_goal is done."
        if cmds
        else "Note: no gates configured — judge only against process_goal + evidence."
    )
    has_contract = bool(
        str(acceptance_checklist or "").strip()
        and str(acceptance_checklist).strip()
        not in ("(no acceptance contract)", "(empty checklist)")
    )
    system_prompt = build_completion_review_system_prompt(
        user_goal=user_goal,
        gate_commands=cmds,
        artifact_required=artifact_required,
        has_acceptance_contract=has_contract,
    )
    parts = [
        f"<artifact_required>{bool(artifact_required)}</artifact_required>",
        f"<gate_result>{gate_result or 'none'}</gate_result>",
        f"<configured_gates>\n{gate_cmds_block}\n</configured_gates>",
        gate_note,
        f"<process_goal>\n{str(user_goal or '').strip() or '(empty)'}\n</process_goal>",
        f"<user_task>\n{str(user_goal or '').strip() or '(empty)'}\n</user_task>",
        f"<assistant_evidence>\n{str(evidence_text or '')[:8000]}\n</assistant_evidence>",
        f"<workspace_deliverable_files>\n{files_block}\n</workspace_deliverable_files>",
    ]
    if has_contract or acceptance_checklist is not None:
        parts.append(
            f"<acceptance_checklist>\n{str(acceptance_checklist or '(none)')}\n"
            "</acceptance_checklist>"
        )
    if probe_results is not None:
        parts.append(
            f"<probe_results>\n{str(probe_results or '(none)')}\n</probe_results>"
        )
    if worker_claim is not None:
        parts.append(
            f"<worker_claim>\n{str(worker_claim or '(missing)')}\n</worker_claim>"
        )
    if include_excerpts:
        parts.extend(
            [
                (
                    "Host-read file excerpts follow. truncated=yes means unread middle "
                    "exists — do not invent that middle content."
                ),
                f"<deliverable_contents>\n{contents_block[:REVIEW_FILES_TOTAL_CHARS]}\n"
                "</deliverable_contents>",
            ]
        )
    else:
        parts.append("<deliverable_contents>(omitted)</deliverable_contents>")
    parts.append("Return JSON with complete/reason/missing only.")
    user_prompt = "\n\n".join(parts)
    body: Dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt[:90000]},
        ],
        "stream": False,
        "temperature": 0.1,
        "max_tokens": 1024,
        "model": model or getattr(session, "model", None),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    ctx = getattr(session, "context_length", None)
    if ctx:
        try:
            from src.orchestrators.rlm.runtime.context_budget import (
                apply_context_budget,
            )

            apply_context_budget(body, int(ctx), max_output_tokens=1024)
        except Exception:
            pass
    data = await chat_fn(body)
    msg = extract_assistant_fn(data)
    raw = str(msg.get("content") or "")
    parsed = parse_completion_review(raw)
    if not parsed["complete"] and not _extract_json_object(raw):
        body["messages"] = list(body["messages"]) + [
            {
                "role": "user",
                "content": "Your previous reply was not valid JSON. Reply with JSON only.",
            }
        ]
        try:
            data2 = await chat_fn(body)
            msg2 = extract_assistant_fn(data2)
            parsed = parse_completion_review(str(msg2.get("content") or ""))
        except Exception:
            logger.warning("completion review retry failed", exc_info=True)
            parsed = {
                "complete": False,
                "reason": "completion_review_parse_failed",
                "missing": [
                    "Provide a valid completion decision after fixing outputs."
                ],
            }
    return parsed
