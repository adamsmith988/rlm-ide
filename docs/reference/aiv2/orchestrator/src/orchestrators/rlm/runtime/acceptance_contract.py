"""Host-owned Acceptance Contract for RLM one_shot completion.

Gates remain necessary host evidence. This layer adds a goal-derived checklist,
deterministic/structural probes, a worker claim, and a slim judge for semantic
items — filling the gap when users lack strong Verifiers / gate scripts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ChatFn = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]

SOFT_CHECKLIST_CAP = 20
PROBE_CMD_TIMEOUT_SEC = 60

EVIDENCE_DETERMINISTIC = "deterministic"
EVIDENCE_STRUCTURAL = "structural"
EVIDENCE_SEMANTIC = "semantic"
PRIORITY_BLOCKING = "blocking"
PRIORITY_SOFT = "soft"

MODE_CHAT = "chat"
MODE_ARTIFACT = "artifact"

VERDICT_ACCEPT = "accept"
VERDICT_REJECT = "reject"
VERDICT_NEED_CLAIM = "need_claim"

ACCEPTANCE_CONTRACT_JSON_SCHEMA = """\
Return JSON only:
{
  "mode": "chat"|"artifact",
  "items": [
    {
      "id": "snake_id",
      "text": "binary checkable requirement from the user goal",
      "priority": "blocking"|"soft",
      "evidence_kind": "deterministic"|"structural"|"semantic",
      "hint": "optional path, command, or structural cue"
    }
  ]
}
"""

ACCEPTANCE_CONTRACT_SYSTEM_PROMPT = (
    "You extract an acceptance checklist for an RLM agent run.\n\n"
    "Rules:\n"
    "- Only requirements explicitly stated or strongly implied by the user goal.\n"
    "- Do NOT invent extra polish, tests, or features the user did not ask for.\n"
    "- Each item must be binary (pass/fail), short, and concrete.\n"
    "- Prefer deterministic/structural evidence when a file or structure is enough.\n"
    "- Use semantic only when a human/LLM judgment is required.\n"
    "- Mark must-have requirements as blocking; nice-to-haves as soft.\n"
    "- No minimum item count. Cap around "
    f"{SOFT_CHECKLIST_CAP} by merging related soft items if needed.\n"
    "- hint MUST be one of: empty, deliverable_file, html_title_or_h1, md_heading, "
    "json_parse, csv_nonempty, a relative file path, or a single POSIX "
    "`test -f path` / `test -s path` command. Never PowerShell, never free-form "
    "English, never Compare/Count/Check helpers, never complex shell pipelines.\n\n"
    + ACCEPTANCE_CONTRACT_JSON_SCHEMA
)

# Built-in probe tokens (never executed as shell).
KNOWN_PROBE_HINTS = frozenset(
    {
        "deliverable_file",
        "html_title_or_h1",
        "md_heading",
        "json_parse",
        "csv_nonempty",
    }
)

# Only these POSIX test forms may run under /bin/sh.
_ALLOWED_POSIX_TEST_RE = re.compile(
    r"^test\s+-[fsde]\s+(\S+)\s*$"
)

_HOST_PROBE_DEFECT_RE = re.compile(
    "(?is)(command not found|:\\s*not found|syntax error|word unexpected|"
    "unexpected EOF|unexpected \\(|Permission denied)"
)

WORKER_CLAIM_JSON_SCHEMA = """\
When finishing an artifact task, include a final JSON claim (alone or after a short note):
{
  "done": true,
  "artifacts": ["relative/path.ext"],
  "checklist_status": [
    {"id": "item_id", "satisfied": true, "evidence": "short proof"}
  ],
  "why_complete": "one short sentence",
  "residual_risks": [],
  "commands_run": []
}
"""

_FILE_GOAL_RE = re.compile(
    r"(?is)\b("
    r"write|create|generate|produce|save|export|build|make|"
    r"گزارش|بنویس|بساز|ذخیره|فایل|html|markdown|\.md|\.html|\.csv|\.json|\.pdf"
    r")\b"
)
_ID_SAFE_RE = re.compile(r"[^a-z0-9_]+")


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


def _hint_is_path_like(hint: str) -> bool:
    h = str(hint or "").strip()
    if not h or any(c in h for c in ";|&`$<>()"):
        return False
    if " " in h and not h.startswith("test "):
        return False
    return bool(os.path.splitext(h)[1]) or "/" in h


def _hint_is_host_runnable(hint: str) -> bool:
    """True when hint can be checked in-process or via a safe POSIX test."""
    h = str(hint or "").strip()
    if not h:
        return True
    if h in KNOWN_PROBE_HINTS or "deliverable" in h.lower():
        return True
    m = _ALLOWED_POSIX_TEST_RE.match(h)
    if m:
        path_arg = str(m.group(1) or "")
        if any(c in path_arg for c in ";|&`$<>()"):
            return False
        return True
    return _hint_is_path_like(h)


def looks_like_host_probe_defect(detail: str, *, returncode: Optional[int] = None) -> bool:
    """Shell parse / command-not-found failures are host defects, not worker bugs."""
    text = str(detail or "")
    if returncode in (127, 126, 2) and _HOST_PROBE_DEFECT_RE.search(text):
        return True
    if _HOST_PROBE_DEFECT_RE.search(text):
        # Exit 127 often embedded as "command failed (127): Compare: not found"
        if "command failed (127)" in text or "command failed (126)" in text:
            return True
        if ": not found" in text.lower() or "syntax error" in text.lower():
            return True
    return False


def _slug_id(text: str, *, fallback: str, used: set[str]) -> str:
    base = _ID_SAFE_RE.sub("_", str(text or "").lower()).strip("_")
    if not base:
        base = fallback
    base = base[:48]
    candidate = base
    n = 2
    while candidate in used:
        candidate = f"{base}_{n}"
        n += 1
    used.add(candidate)
    return candidate


def infer_run_mode(
    user_content: str,
    *,
    pools: Optional[Sequence[Any]] = None,
    gates: bool = False,
    strict: bool = False,
) -> str:
    """Heuristic chat vs artifact. Host flags always force artifact."""
    if strict or gates or list(pools or []):
        return MODE_ARTIFACT
    text = str(user_content or "").strip()
    if not text:
        return MODE_CHAT
    if _FILE_GOAL_RE.search(text) and len(text) > 40:
        return MODE_ARTIFACT
    return MODE_CHAT


def sanitize_checklist_items(
    raw_items: Any,
    *,
    soft_cap: int = SOFT_CHECKLIST_CAP,
) -> List[Dict[str, Any]]:
    """Normalize LLM/heuristic items. No minimum; soft-cap extras → soft/merge."""
    if not isinstance(raw_items, list):
        return []
    used_ids: set[str] = set()
    out: List[Dict[str, Any]] = []
    for idx, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or raw.get("requirement") or "").strip()
        if not text:
            continue
        priority = str(raw.get("priority") or PRIORITY_BLOCKING).strip().lower()
        if priority not in (PRIORITY_BLOCKING, PRIORITY_SOFT):
            priority = PRIORITY_BLOCKING
        kind = str(raw.get("evidence_kind") or raw.get("kind") or EVIDENCE_SEMANTIC)
        kind = kind.strip().lower()
        if kind not in (
            EVIDENCE_DETERMINISTIC,
            EVIDENCE_STRUCTURAL,
            EVIDENCE_SEMANTIC,
        ):
            kind = EVIDENCE_SEMANTIC
        hint = str(raw.get("hint") or "").strip()[:500]
        # Free-form / PowerShell / invented probe text is not host-runnable.
        # Demote to semantic so the judge handles it — never /bin/sh-exec it.
        if (
            kind in (EVIDENCE_DETERMINISTIC, EVIDENCE_STRUCTURAL)
            and hint
            and not _hint_is_host_runnable(hint)
        ):
            kind = EVIDENCE_SEMANTIC
        item_id = str(raw.get("id") or "").strip().lower()
        if not item_id or not re.match(r"^[a-z][a-z0-9_]{0,63}$", item_id):
            item_id = _slug_id(text, fallback=f"item_{idx+1}", used=used_ids)
        else:
            item_id = _slug_id(item_id, fallback=f"item_{idx+1}", used=used_ids)
        out.append(
            {
                "id": item_id,
                "text": text[:400],
                "priority": priority,
                "evidence_kind": kind,
                "hint": hint,
            }
        )
    cap = max(1, int(soft_cap or SOFT_CHECKLIST_CAP))
    if len(out) <= cap:
        return out
    kept = out[:cap]
    overflow = out[cap:]
    # Soften overflow that would otherwise vanish: fold into last soft item note.
    extra_bits = [f"{x['id']}: {x['text']}" for x in overflow[:8]]
    if kept:
        last = dict(kept[-1])
        if last.get("priority") != PRIORITY_SOFT:
            last["priority"] = PRIORITY_SOFT
        note = last.get("hint") or ""
        merged = (note + " | also: " + "; ".join(extra_bits)).strip(" |")
        last["hint"] = merged[:500]
        kept[-1] = last
    return kept


def fallback_acceptance_contract(
    user_content: str,
    *,
    mode: str = MODE_ARTIFACT,
) -> Dict[str, Any]:
    """Deterministic contract when LLM extraction is unavailable."""
    goal = str(user_content or "").strip()
    items: List[Dict[str, Any]] = []
    if mode == MODE_ARTIFACT:
        items.append(
            {
                "id": "deliverable_present",
                "text": "A non-stub workspace deliverable file exists for the user task.",
                "priority": PRIORITY_BLOCKING,
                "evidence_kind": EVIDENCE_DETERMINISTIC,
                "hint": "deliverable_file",
            }
        )
        if re.search(r"(?i)\.html?\b|html\b", goal):
            items.append(
                {
                    "id": "html_structure",
                    "text": "HTML deliverable has a document structure (title or h1).",
                    "priority": PRIORITY_BLOCKING,
                    "evidence_kind": EVIDENCE_STRUCTURAL,
                    "hint": "html_title_or_h1",
                }
            )
        if re.search(r"(?i)\.md\b|markdown\b|report\b|گزارش", goal):
            items.append(
                {
                    "id": "doc_has_headings",
                    "text": "Document deliverable has at least one heading and body content.",
                    "priority": PRIORITY_BLOCKING,
                    "evidence_kind": EVIDENCE_STRUCTURAL,
                    "hint": "md_heading",
                }
            )
        items.append(
            {
                "id": "goal_satisfied",
                "text": "Deliverable content satisfies the stated user goal (not a stub/outline).",
                "priority": PRIORITY_BLOCKING,
                "evidence_kind": EVIDENCE_SEMANTIC,
                "hint": "",
            }
        )
    return {
        "mode": mode if mode in (MODE_CHAT, MODE_ARTIFACT) else MODE_ARTIFACT,
        "items": sanitize_checklist_items(items),
        "generated_at": "",
        "source": "fallback",
        "goal_preview": goal[:500],
    }


def parse_acceptance_contract_payload(
    raw: str,
    *,
    default_mode: str = MODE_ARTIFACT,
) -> Dict[str, Any]:
    record = _extract_json_object(raw)
    mode = str(record.get("mode") or default_mode).strip().lower()
    if mode not in (MODE_CHAT, MODE_ARTIFACT):
        mode = default_mode
    items = sanitize_checklist_items(record.get("items"))
    if not items and mode == MODE_ARTIFACT:
        return fallback_acceptance_contract("", mode=mode)
    return {
        "mode": mode,
        "items": items,
        "generated_at": "",
        "source": "llm",
        "goal_preview": "",
    }


async def build_acceptance_contract(
    session: Any,
    *,
    chat_fn: ChatFn,
    extract_assistant_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    user_goal: str,
    mode: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """One LLM call to extract checklist; falls back to heuristic on failure."""
    import time

    goal = str(user_goal or "").strip()
    if mode == MODE_CHAT:
        return {
            "mode": MODE_CHAT,
            "items": [],
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "chat_skip",
            "goal_preview": goal[:500],
        }
    user_prompt = "\n\n".join(
        [
            "<acceptance_contract_request>",
            "Extract an acceptance checklist for this user goal.",
            "</acceptance_contract_request>",
            f"<process_goal>\n{goal[:8000] or '(empty)'}\n</process_goal>",
            f"<forced_mode>{MODE_ARTIFACT}</forced_mode>",
            "Return JSON only.",
        ]
    )
    body: Dict[str, Any] = {
        "messages": [
            {"role": "system", "content": ACCEPTANCE_CONTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt[:30000]},
        ],
        "stream": False,
        "temperature": 0.1,
        "max_tokens": 2048,
        "model": model or getattr(session, "model", None),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    ctx = getattr(session, "context_length", None)
    if ctx:
        try:
            from src.orchestrators.rlm.runtime.context_budget import (
                apply_context_budget,
            )

            apply_context_budget(body, int(ctx), max_output_tokens=2048)
        except Exception:
            pass
    try:
        data = await chat_fn(body)
        msg = extract_assistant_fn(data)
        raw = str(msg.get("content") or "")
        parsed = parse_acceptance_contract_payload(raw, default_mode=MODE_ARTIFACT)
        if not parsed.get("items"):
            parsed = fallback_acceptance_contract(goal, mode=MODE_ARTIFACT)
            parsed["source"] = "fallback_empty_llm"
    except Exception:
        logger.warning("acceptance contract LLM build failed", exc_info=True)
        parsed = fallback_acceptance_contract(goal, mode=MODE_ARTIFACT)
        parsed["source"] = "fallback_error"
    parsed["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    parsed["goal_preview"] = goal[:500]
    parsed["mode"] = MODE_ARTIFACT
    return parsed


def body_is_acceptance_contract(body: Dict[str, Any]) -> bool:
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        text = str(m.get("content") or "")
        low = text.lower()
        if (
            "<acceptance_contract_request>" in low
            or "extract an acceptance checklist for this user goal" in low
            or "you extract an acceptance checklist" in low
        ):
            return True
    return False


def parse_worker_claim(text: str) -> Optional[Dict[str, Any]]:
    """Parse the last JSON object that looks like a worker completion claim."""
    raw = str(text or "")
    if not raw.strip():
        return None
    # Prefer the last JSON object in the text (claim often trails a short note).
    candidates: List[Dict[str, Any]] = []
    depth = 0
    start = -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    chunk = raw[start : i + 1]
                    try:
                        data = json.loads(chunk)
                        if isinstance(data, dict):
                            candidates.append(data)
                    except json.JSONDecodeError:
                        pass
                    start = -1
    record: Dict[str, Any] = {}
    for cand in reversed(candidates):
        if (
            "done" in cand
            or "checklist_status" in cand
            or "why_complete" in cand
            or "artifacts" in cand
        ):
            record = cand
            break
    if not record:
        return None
    status_raw = record.get("checklist_status")
    status: List[Dict[str, Any]] = []
    if isinstance(status_raw, list):
        for row in status_raw[:40]:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("id") or "").strip()
            if not sid:
                continue
            status.append(
                {
                    "id": sid,
                    "satisfied": row.get("satisfied") is True,
                    "evidence": str(row.get("evidence") or "")[:500],
                }
            )
    arts = record.get("artifacts")
    artifacts: List[str] = []
    if isinstance(arts, list):
        artifacts = [str(a).strip() for a in arts if str(a).strip()][:40]
    elif isinstance(arts, str) and arts.strip():
        artifacts = [arts.strip()]
    risks = record.get("residual_risks")
    residual: List[str] = []
    if isinstance(risks, list):
        residual = [str(x).strip() for x in risks if str(x).strip()][:12]
    cmds = record.get("commands_run")
    commands: List[str] = []
    if isinstance(cmds, list):
        commands = [str(x).strip() for x in cmds if str(x).strip()][:20]
    return {
        "done": record.get("done") is True,
        "artifacts": artifacts,
        "checklist_status": status,
        "why_complete": str(record.get("why_complete") or "")[:800],
        "residual_risks": residual,
        "commands_run": commands,
    }


def _list_deliverables(workspace_dir: str) -> List[Dict[str, Any]]:
    try:
        from src.orchestrators.rlm.runtime.persist import (
            list_kernel_deliverable_files,
        )

        return [
            r
            for r in (list_kernel_deliverable_files(workspace_dir) or [])
            if isinstance(r, dict)
        ]
    except Exception:
        return []


def _read_file_text(path: str, *, limit: int = 200_000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(max(0, int(limit)))
    except OSError:
        return ""


def _resolve_hint_path(workspace_dir: str, hint: str) -> str:
    root = str(workspace_dir or "").strip()
    h = str(hint or "").strip()
    if not root or not h:
        return ""
    # Ignore special probe hints that are not paths.
    if h in (
        "deliverable_file",
        "html_title_or_h1",
        "md_heading",
        "json_parse",
        "csv_nonempty",
    ):
        return ""
    # Strip shell-ish prefixes.
    if h.startswith("test -f "):
        h = h[len("test -f ") :].strip().strip("'\"")
    candidate = h if os.path.isabs(h) else os.path.join(root, h)
    if os.path.isfile(candidate):
        return candidate
    # Basename search among deliverables.
    base = os.path.basename(h)
    for rec in _list_deliverables(root):
        name = str(rec.get("relpath") or rec.get("filename") or "")
        if name == h or os.path.basename(name) == base:
            path = str(rec.get("path") or "")
            if path and os.path.isfile(path):
                return path
            joined = os.path.join(root, name)
            if os.path.isfile(joined):
                return joined
    return ""


def _probe_deliverable_present(workspace_dir: str) -> Tuple[bool, str]:
    """Host check: a non-empty deliverable file exists.

    Content quality (stub vs finished report) is judged by the AI reviewer.
    """
    recs = _list_deliverables(workspace_dir)
    if not recs:
        return False, "no deliverable files in workspace"
    for rec in recs:
        try:
            size = int(rec.get("size_bytes") or 0)
        except (TypeError, ValueError):
            size = 0
        path = str(rec.get("path") or "")
        if size <= 0 and path and os.path.isfile(path):
            try:
                size = int(os.path.getsize(path))
            except OSError:
                size = 0
        if size > 0:
            name = str(rec.get("relpath") or rec.get("filename") or path)
            return True, f"deliverable present: {name} ({size}B)"
    return False, "only empty deliverables found"


def _probe_html_structure(workspace_dir: str, hint: str) -> Tuple[bool, str]:
    path = _resolve_hint_path(workspace_dir, hint)
    candidates: List[str] = []
    if path:
        candidates.append(path)
    else:
        for rec in _list_deliverables(workspace_dir):
            name = str(rec.get("relpath") or "").lower()
            if name.endswith(".html") or name.endswith(".htm"):
                p = str(rec.get("path") or "")
                if p:
                    candidates.append(p)
                else:
                    candidates.append(
                        os.path.join(workspace_dir, str(rec.get("relpath") or ""))
                    )
    if not candidates:
        return False, "no html deliverable found"
    for path in candidates:
        text = _read_file_text(path)
        if not text.strip():
            continue
        has_title = bool(re.search(r"(?is)<title[^>]*>\s*\S+", text))
        has_h1 = bool(re.search(r"(?is)<h1[^>]*>\s*\S+", text))
        if has_title or has_h1:
            return True, f"html structure ok in {os.path.basename(path)}"
    return False, "html missing title/h1"


def _probe_md_heading(workspace_dir: str, hint: str) -> Tuple[bool, str]:
    path = _resolve_hint_path(workspace_dir, hint)
    candidates: List[str] = []
    if path:
        candidates.append(path)
    else:
        for rec in _list_deliverables(workspace_dir):
            name = str(rec.get("relpath") or "").lower()
            if name.endswith(".md") or name.endswith(".txt"):
                p = str(rec.get("path") or "")
                if p:
                    candidates.append(p)
                else:
                    candidates.append(
                        os.path.join(workspace_dir, str(rec.get("relpath") or ""))
                    )
    if not candidates:
        return False, "no markdown/text deliverable found"
    for path in candidates:
        text = _read_file_text(path)
        if re.search(r"(?m)^#{1,6}\s+\S+", text) and len(text.strip()) > 40:
            return True, f"heading+body ok in {os.path.basename(path)}"
    return False, "markdown missing heading or body too short"


def _probe_json_parse(workspace_dir: str, hint: str) -> Tuple[bool, str]:
    path = _resolve_hint_path(workspace_dir, hint)
    if not path:
        for rec in _list_deliverables(workspace_dir):
            name = str(rec.get("relpath") or "").lower()
            if name.endswith(".json"):
                path = str(rec.get("path") or "") or os.path.join(
                    workspace_dir, str(rec.get("relpath") or "")
                )
                break
    if not path or not os.path.isfile(path):
        return False, "no json deliverable found"
    try:
        json.loads(_read_file_text(path))
        return True, f"json parses: {os.path.basename(path)}"
    except json.JSONDecodeError as exc:
        return False, f"json parse failed: {exc}"


def _probe_csv_nonempty(workspace_dir: str, hint: str) -> Tuple[bool, str]:
    path = _resolve_hint_path(workspace_dir, hint)
    if not path:
        for rec in _list_deliverables(workspace_dir):
            name = str(rec.get("relpath") or "").lower()
            if name.endswith(".csv"):
                path = str(rec.get("path") or "") or os.path.join(
                    workspace_dir, str(rec.get("relpath") or "")
                )
                break
    if not path or not os.path.isfile(path):
        return False, "no csv deliverable found"
    text = _read_file_text(path, limit=50_000)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return True, f"csv has header+rows: {os.path.basename(path)}"
    return False, "csv empty or header-only"


def _probe_shell_hint(workspace_dir: str, hint: str) -> Tuple[bool, str, bool]:
    """Run a narrow POSIX ``test`` probe.

    Returns ``(ok, detail, host_invalid)``. ``host_invalid`` means the command
    is not a usable host probe (worker cannot fix it by editing artifacts).
    """
    cmd = str(hint or "").strip()
    if not cmd or not workspace_dir:
        return False, "empty command hint", True
    if not _ALLOWED_POSIX_TEST_RE.match(cmd):
        return False, f"non-portable probe hint skipped: {cmd[:120]}", True
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            timeout=PROBE_CMD_TIMEOUT_SEC,
        )
        rc = int(proc.returncode)
        if rc == 0:
            return True, f"command ok: {cmd[:120]}", False
        err = (proc.stderr or proc.stdout or "").strip()[:240]
        detail = f"command failed ({rc}): {err or cmd[:80]}"
        if looks_like_host_probe_defect(detail, returncode=rc):
            return False, detail, True
        return False, detail, False
    except subprocess.TimeoutExpired:
        return False, f"command timed out: {cmd[:80]}", False
    except Exception as exc:
        return False, f"command error: {exc}", True


def _run_one_probe(item: Dict[str, Any], workspace_dir: str) -> Dict[str, Any]:
    item_id = str(item.get("id") or "")
    kind = str(item.get("evidence_kind") or EVIDENCE_SEMANTIC)
    hint = str(item.get("hint") or "").strip()
    if kind == EVIDENCE_SEMANTIC:
        return {
            "item_id": item_id,
            "pass": None,
            "detail": "semantic — deferred to judge",
            "skipped": True,
        }
    ok = False
    detail = "unhandled probe"
    host_invalid = False
    if kind == EVIDENCE_DETERMINISTIC:
        if _ALLOWED_POSIX_TEST_RE.match(hint):
            ok, detail, host_invalid = _probe_shell_hint(workspace_dir, hint)
        elif hint in ("", "deliverable_file") or "deliverable" in hint.lower():
            ok, detail = _probe_deliverable_present(workspace_dir)
        elif hint in KNOWN_PROBE_HINTS:
            # Known tokens without a dedicated branch → deliverable presence.
            ok, detail = _probe_deliverable_present(workspace_dir)
        elif _hint_is_path_like(hint) or "test -f" in hint:
            path = _resolve_hint_path(workspace_dir, hint)
            if path:
                ok, detail = True, f"file exists: {os.path.basename(path)}"
            else:
                ok, detail = _probe_deliverable_present(workspace_dir)
        elif hint:
            # Never exec free-form English / PowerShell as /bin/sh.
            detail = f"non-portable probe hint skipped: {hint[:120]}"
            host_invalid = True
        else:
            ok, detail = _probe_deliverable_present(workspace_dir)
    elif kind == EVIDENCE_STRUCTURAL:
        hlow = hint.lower()
        if hint in ("html_title_or_h1",) or "html" in hlow or hint.endswith(
            (".html", ".htm")
        ):
            ok, detail = _probe_html_structure(workspace_dir, hint)
        elif hint in ("json_parse",) or "json" in hlow or hint.endswith(".json"):
            ok, detail = _probe_json_parse(workspace_dir, hint)
        elif hint in ("csv_nonempty",) or "csv" in hlow or hint.endswith(".csv"):
            ok, detail = _probe_csv_nonempty(workspace_dir, hint)
        elif (
            hint in ("md_heading",)
            or "md" in hlow
            or "heading" in hlow
            or hint.endswith(".md")
        ):
            ok, detail = _probe_md_heading(workspace_dir, hint)
        elif hint and not _hint_is_host_runnable(hint):
            detail = f"non-portable structural hint skipped: {hint[:120]}"
            host_invalid = True
        else:
            # Generic: prefer html then md then json based on workspace files.
            html_ok, html_d = _probe_html_structure(workspace_dir, hint)
            if html_ok:
                ok, detail = html_ok, html_d
            else:
                md_ok, md_d = _probe_md_heading(workspace_dir, hint)
                if md_ok:
                    ok, detail = md_ok, md_d
                else:
                    ok, detail = False, f"structural probe failed ({html_d}; {md_d})"
    if host_invalid:
        return {
            "item_id": item_id,
            "pass": None,
            "detail": str(detail)[:500],
            "skipped": True,
            "host_invalid": True,
        }
    return {
        "item_id": item_id,
        "pass": bool(ok),
        "detail": str(detail)[:500],
        "skipped": False,
    }


def run_checklist_probes(
    contract: Optional[Dict[str, Any]],
    workspace_dir: str,
) -> List[Dict[str, Any]]:
    if not isinstance(contract, dict):
        return []
    items = contract.get("items") if isinstance(contract.get("items"), list) else []
    results: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        results.append(_run_one_probe(item, workspace_dir))
    return results


def blocking_probe_failures(
    contract: Optional[Dict[str, Any]],
    probes: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    items = {}
    if isinstance(contract, dict):
        for it in contract.get("items") or []:
            if isinstance(it, dict) and it.get("id"):
                items[str(it["id"])] = it
    fails: List[Dict[str, Any]] = []
    for pr in probes or []:
        if not isinstance(pr, dict) or pr.get("skipped"):
            continue
        if pr.get("pass") is True:
            continue
        item = items.get(str(pr.get("item_id") or ""))
        if not item:
            continue
        if str(item.get("priority") or PRIORITY_BLOCKING) != PRIORITY_BLOCKING:
            continue
        # Only host-probed kinds can hard-fail here.
        if str(item.get("evidence_kind") or "") == EVIDENCE_SEMANTIC:
            continue
        fails.append(
            {
                "id": item.get("id"),
                "text": item.get("text"),
                "detail": pr.get("detail"),
            }
        )
    return fails


def all_blocking_probes_green(
    contract: Optional[Dict[str, Any]],
    probes: Sequence[Dict[str, Any]],
) -> bool:
    """True when every runnable blocking deterministic/structural probe passed.

    Host-invalid / skipped probes (non-portable hints) are ignored so a broken
    ``Compare``/PowerShell-style checklist cannot block completion forever.
    """
    if not isinstance(contract, dict):
        return False
    items = [it for it in (contract.get("items") or []) if isinstance(it, dict)]
    blocking = [
        it
        for it in items
        if str(it.get("priority") or PRIORITY_BLOCKING) == PRIORITY_BLOCKING
        and str(it.get("evidence_kind") or "")
        in (EVIDENCE_DETERMINISTIC, EVIDENCE_STRUCTURAL)
    ]
    if not blocking:
        return False
    by_id = {
        str(p.get("item_id")): p
        for p in (probes or [])
        if isinstance(p, dict)
    }
    runnable = 0
    for it in blocking:
        pr = by_id.get(str(it.get("id")))
        if pr and pr.get("skipped") and (
            pr.get("host_invalid") or pr.get("pass") is None
        ):
            continue
        runnable += 1
        if not pr or pr.get("pass") is not True:
            return False
    return runnable > 0


def open_semantic_blocking_items(
    contract: Optional[Dict[str, Any]],
    probes: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    probe_pass = {
        str(p.get("item_id")): p
        for p in (probes or [])
        if isinstance(p, dict)
    }
    out: List[Dict[str, Any]] = []
    if not isinstance(contract, dict):
        return out
    for it in contract.get("items") or []:
        if not isinstance(it, dict):
            continue
        if str(it.get("priority") or "") != PRIORITY_BLOCKING:
            continue
        pr = probe_pass.get(str(it.get("id") or ""))
        # Host-invalid deterministic hints are judged semantically.
        if pr and pr.get("host_invalid") and pr.get("skipped"):
            out.append(it)
            continue
        if str(it.get("evidence_kind") or "") != EVIDENCE_SEMANTIC:
            # Already decided by probe; not open for judge.
            continue
        out.append(it)
    return out


def claim_contradicts_probes(
    claim: Optional[Dict[str, Any]],
    probes: Sequence[Dict[str, Any]],
) -> List[str]:
    if not isinstance(claim, dict):
        return []
    failed = {
        str(p.get("item_id"))
        for p in (probes or [])
        if isinstance(p, dict) and p.get("pass") is False and not p.get("skipped")
    }
    bad: List[str] = []
    for row in claim.get("checklist_status") or []:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("id") or "")
        if row.get("satisfied") is True and sid in failed:
            bad.append(sid)
    return bad


def evaluate_host_verdict(
    *,
    contract: Optional[Dict[str, Any]],
    gate_result: Optional[str],
    probes: Sequence[Dict[str, Any]],
    claim: Optional[Dict[str, Any]],
    judge: Optional[Dict[str, Any]],
    artifact_required: bool,
    gates_configured: bool = False,
) -> Dict[str, Any]:
    """Host authority over completion. Judge cannot override red probes."""
    if not artifact_required:
        if judge and judge.get("complete") is True:
            return {"verdict": VERDICT_ACCEPT, "reason": "chat_complete"}
        if judge is None:
            return {"verdict": VERDICT_ACCEPT, "reason": "chat_skip"}
        return {
            "verdict": VERDICT_REJECT,
            "reason": str((judge or {}).get("reason") or "chat_review_rejected"),
        }

    if gates_configured and gate_result not in (None, "passed"):
        return {
            "verdict": VERDICT_REJECT,
            "reason": f"gate_{gate_result or 'failed'}",
        }

    fails = blocking_probe_failures(contract, probes)
    if fails:
        return {
            "verdict": VERDICT_REJECT,
            "reason": "acceptance_probe_failed",
            "missing": [
                f"{f.get('id')}: {f.get('detail') or f.get('text')}" for f in fails
            ],
        }

    if not isinstance(claim, dict):
        return {"verdict": VERDICT_NEED_CLAIM, "reason": "acceptance_claim_missing"}

    contradictions = claim_contradicts_probes(claim, probes)
    if contradictions:
        return {
            "verdict": VERDICT_REJECT,
            "reason": "acceptance_claim_contradiction",
            "missing": [f"claim marked satisfied but probe failed: {x}" for x in contradictions],
        }

    semantic_open = open_semantic_blocking_items(contract, probes)
    if not semantic_open and all_blocking_probes_green(contract, probes):
        # Near-Prime path: strong probes + claim, no LLM needed.
        if claim.get("done") is True or claim.get("checklist_status"):
            return {
                "verdict": VERDICT_ACCEPT,
                "reason": "probes_and_claim",
            }

    if judge is None:
        return {
            "verdict": VERDICT_REJECT,
            "reason": "completion_review_pending",
        }
    if judge.get("complete") is True:
        return {"verdict": VERDICT_ACCEPT, "reason": "judge_accept"}
    return {
        "verdict": VERDICT_REJECT,
        "reason": str(judge.get("reason") or "completion_review_rejected"),
        "missing": list(judge.get("missing") or [])[:8],
    }


def build_probe_failed_continuation(failures: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    lines = [
        "[acceptance probes] Host checklist probes failed. The run is NOT finished.",
        "Fix these blocking items:",
    ]
    host_side = 0
    for f in failures[:12]:
        detail = str(f.get("detail") or "failed")
        lines.append(f"- {f.get('id')}: {f.get('text') or ''} — {detail}")
        if looks_like_host_probe_defect(detail) or "non-portable probe" in detail:
            host_side += 1
    if host_side and host_side >= len(list(failures or [])[:12]):
        lines.append(
            "NOTE: These look like host-side probe/shell defects (not artifact bugs). "
            "Do not keep rewriting the deliverable for Compare/Count/syntax errors. "
            "Re-emit an honest worker claim; the host will judge semantically."
        )
    else:
        lines.append(
            "Continue working in the workspace. After fixing, summarize progress "
            "and include a final worker claim JSON when done."
        )
    return {
        "role": "user",
        "content": "\n".join(lines),
        "_rlm_continuation": "acceptance",
        "_rlm_continuation_reason": "acceptance_probe_failed",
    }


def build_claim_required_continuation(
    contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ids: List[str] = []
    if isinstance(contract, dict):
        for it in contract.get("items") or []:
            if isinstance(it, dict) and it.get("id"):
                ids.append(str(it["id"]))
    id_hint = ", ".join(ids[:16]) if ids else "item ids from the acceptance checklist"
    return {
        "role": "user",
        "content": (
            "[acceptance claim] Host terminal evidence exists, but a valid worker "
            "claim JSON was not found in your last reply.\n"
            "Emit a final claim JSON now (schema below) after any short note. "
            f"Checklist ids: {id_hint}\n\n"
            + WORKER_CLAIM_JSON_SCHEMA
        ),
        "_rlm_continuation": "acceptance",
        "_rlm_continuation_reason": "acceptance_claim_missing",
    }


def build_claim_contradiction_continuation(ids: Sequence[str]) -> Dict[str, Any]:
    listed = ", ".join(str(x) for x in ids[:12])
    return {
        "role": "user",
        "content": (
            "[acceptance claim] Your claim marked items satisfied that host probes "
            f"still fail: {listed}.\n"
            "Fix the workspace deliverables, then send an honest claim JSON."
        ),
        "_rlm_continuation": "acceptance",
        "_rlm_continuation_reason": "acceptance_claim_contradiction",
    }


def format_checklist_for_prompt(
    contract: Optional[Dict[str, Any]],
    probes: Sequence[Dict[str, Any]],
) -> str:
    if not isinstance(contract, dict):
        return "(no acceptance contract)"
    probe_map = {
        str(p.get("item_id")): p
        for p in (probes or [])
        if isinstance(p, dict)
    }
    lines: List[str] = []
    for it in contract.get("items") or []:
        if not isinstance(it, dict):
            continue
        sid = str(it.get("id") or "")
        pr = probe_map.get(sid) or {}
        if pr.get("skipped"):
            status = "open(semantic)"
        elif pr.get("pass") is True:
            status = "PASS"
        elif pr.get("pass") is False:
            status = "FAIL"
        else:
            status = "unknown"
        lines.append(
            f"- [{status}] ({it.get('priority')}/{it.get('evidence_kind')}) "
            f"{sid}: {it.get('text')}"
            + (f" — {pr.get('detail')}" if pr.get("detail") else "")
        )
    return "\n".join(lines) if lines else "(empty checklist)"


def format_claim_for_prompt(claim: Optional[Dict[str, Any]]) -> str:
    if not isinstance(claim, dict):
        return "(missing)"
    try:
        return json.dumps(claim, ensure_ascii=False, indent=2)[:4000]
    except Exception:
        return str(claim)[:4000]


def format_probes_for_prompt(probes: Sequence[Dict[str, Any]]) -> str:
    if not probes:
        return "(none)"
    lines = []
    for p in probes:
        if not isinstance(p, dict):
            continue
        if p.get("skipped"):
            st = "skip"
        elif p.get("pass") is True:
            st = "pass"
        else:
            st = "fail"
        lines.append(f"- {p.get('item_id')}: {st} — {p.get('detail')}")
    return "\n".join(lines) if lines else "(none)"


def acceptance_evidence_extra(
    *,
    contract: Optional[Dict[str, Any]],
    probes: Sequence[Dict[str, Any]],
    claim: Optional[Dict[str, Any]],
) -> str:
    """Stable fragment for completion-review fingerprinting."""
    parts: List[str] = []
    if isinstance(contract, dict):
        for it in contract.get("items") or []:
            if isinstance(it, dict):
                parts.append(
                    f"c:{it.get('id')}|{it.get('priority')}|{it.get('evidence_kind')}"
                )
    for p in probes or []:
        if isinstance(p, dict):
            parts.append(
                f"p:{p.get('item_id')}|{p.get('pass')}|{str(p.get('detail') or '')[:80]}"
            )
    if isinstance(claim, dict):
        parts.append(
            "claim:"
            + hashlib.sha256(
                json.dumps(claim, sort_keys=True, ensure_ascii=False).encode(
                    "utf-8", errors="replace"
                )
            ).hexdigest()[:16]
        )
    return "\n".join(parts)
