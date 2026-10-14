"""Rule applicability and system-message merge (Continue-compatible)."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from .globmatch import path_matches_globs
from .paths import extract_content_for_path, extract_paths_from_code_blocks
from .types import MarkdownRule, RuleMetadata, RulePolicies

log = logging.getLogger(__name__)


def normalize_rule_policies(raw: Any) -> RulePolicies:
    """Parse ``context[\"rule_policies\"]`` from API requests (``on`` / ``off`` per rule name)."""
    out: RulePolicies = {}
    if not isinstance(raw, dict):
        return out
    for key, val in raw.items():
        if not isinstance(key, str) or not key.strip():
            continue
        s = str(val).strip().lower() if val is not None else ""
        if s in ("on", "off"):
            out[key.strip()] = s  # type: ignore[assignment]
    return out


def _content_matches_regex(file_content: str, regex: Any) -> bool:
    if not file_content:
        return False
    if not regex:
        return True
    patterns = regex if isinstance(regex, list) else [regex]
    if not patterns:
        return True
    for pat in patterns:
        try:
            if re.search(str(pat), file_content, re.DOTALL):
                return True
        except re.error as e:
            log.warning("Invalid rule regex %s: %s", pat, e)
    return False


def _is_file_in_rule_directory(file_path: str, rule_dir: str) -> bool:
    """Loose 'under directory' check (Continue ``isFileInDirectory`` style)."""
    fp = file_path.replace("\\", "/")
    rd = rule_dir.replace("\\", "/").rstrip("/")
    if not rd:
        return True
    # basename containment heuristic from Continue
    dir_name = rd.split("/")[-1]
    if dir_name and f"{dir_name}/" in fp:
        return True
    try:
        return Path(fp).resolve().is_relative_to(Path(rd).resolve())
    except (ValueError, OSError):
        return fp.startswith(rd + "/") or fp == rd


def is_global_rule(rule: MarkdownRule) -> bool:
    if rule.always_apply is True:
        return True
    if rule.always_apply is False:
        return False
    stem = Path(rule.source_file).stem.upper()
    if stem in ("AGENTS", "AGENT", "CLAUDE"):
        return True
    if rule.is_rules_directory_root_file and not rule.globs and not rule.regex:
        return True
    return False


def _check_globs_and_regex(
    rule: MarkdownRule,
    file_paths: List[str],
    file_contents: Dict[str, str],
) -> bool:
    if not rule.globs:
        matching = file_paths
    else:
        matching = [fp for fp in file_paths if path_matches_globs(fp, rule.globs)]
    if not matching:
        return False
    if rule.regex:
        for fp in matching:
            content = file_contents.get(fp)
            if not content:
                continue
            if _content_matches_regex(content, rule.regex):
                return True
        return False
    return len(matching) > 0


def should_apply_rule(
    rule: MarkdownRule,
    file_paths: List[str],
    rule_policies: RulePolicies,
    file_contents: Dict[str, str],
) -> bool:
    policy = rule_policies.get(rule.name)
    if policy == "off":
        return False

    if is_global_rule(rule):
        return True

    if not file_paths:
        return False

    # Root of a rules pack (e.g. ``AGENTS.md`` next to other pack rules): workspace-wide globs/regex.
    is_root = rule.is_rules_directory_root_file
    rule_dir = str(Path(rule.source_file).resolve().parent)

    if not is_root:
        files_in_rule_dir = [
            fp for fp in file_paths if _is_file_in_rule_directory(fp, rule_dir)
        ]
        return _check_globs_and_regex(rule, files_in_rule_dir, file_contents)

    if (
        rule.always_apply is False
        and rule.globs is None
        and rule.regex is None
    ):
        return False

    return _check_globs_and_regex(rule, file_paths, file_contents)


def collect_paths_for_matching(
    user_message: str,
    file_context: Optional[Dict[str, Any]],
    file_registry: Optional[List[Dict[str, Any]]],
) -> Tuple[List[str], Dict[str, str]]:
    """Paths + optional text snippets for regex (inlined session text)."""
    paths: List[str] = []
    contents: Dict[str, str] = {}

    if user_message:
        for p in extract_paths_from_code_blocks(user_message):
            if p not in paths:
                paths.append(p)
            c = extract_content_for_path(user_message, p)
            if c:
                contents[p] = c

    if file_context:
        for key in ("path", "name"):
            v = file_context.get(key)
            if isinstance(v, str) and v and v not in paths:
                paths.append(v)
        for f in file_context.get("all_files") or []:
            if not isinstance(f, dict):
                continue
            for key in ("path", "name"):
                v = f.get(key)
                if isinstance(v, str) and v and v not in paths:
                    paths.append(v)
            if f.get("text_content_inlined") and f.get("name"):
                # small excerpt only for regex
                te = f.get("text_extract") or ""
                if isinstance(te, str) and te:
                    nm = f.get("name")
                    if isinstance(nm, str):
                        contents[nm] = te[:8000]

    if file_registry:
        for entry in file_registry:
            if not isinstance(entry, dict):
                continue
            for key in ("path", "name"):
                v = entry.get(key)
                if isinstance(v, str) and v and v not in paths:
                    paths.append(v)

    return paths, contents


def get_applicable_rules(
    *,
    user_message: str,
    available_rules: List[MarkdownRule],
    file_context: Optional[Dict[str, Any]],
    file_registry: Optional[List[Dict[str, Any]]],
    rule_policies: RulePolicies,
) -> List[MarkdownRule]:
    file_paths, file_contents = collect_paths_for_matching(
        user_message, file_context, file_registry
    )
    out: List[MarkdownRule] = []
    for rule in available_rules:
        if rule.invokable:
            continue
        try:
            if should_apply_rule(rule, file_paths, rule_policies, file_contents):
                out.append(rule)
        except Exception as e:
            log.warning("Rule %s applicability error: %s", rule.name, e)
    return out


def merge_applicable_rules_into_chat_messages(
    messages: List[Dict[str, Any]],
    applicable: List[MarkdownRule],
) -> Tuple[List[Dict[str, Any]], List[RuleMetadata]]:
    """
    Append rule bodies to the first ``role: system`` message (Qwen-style single system).
    If there is no system message, prepends one.
    """
    if not applicable:
        return messages, []

    meta = [
        RuleMetadata(
            name=r.name,
            source_file=r.source_file,
            description=r.description,
            slug=Path(r.source_file).stem,
        )
        for r in applicable
    ]
    out: List[Dict[str, Any]] = [dict(m) for m in messages]
    extra = _rules_block(applicable)
    for i, m in enumerate(out):
        if m.get("role") == "system":
            base = m.get("content") or ""
            out[i] = {
                **m,
                "content": f"{base}\n\n{extra}".strip() if base.strip() else extra,
            }
            return out, meta
    return [{"role": "system", "content": extra}] + out, meta


def merge_rules_into_first_system_message(
    system_messages: List[Dict[str, str]],
    applicable: List[MarkdownRule],
) -> Tuple[List[Dict[str, str]], List[RuleMetadata]]:
    """Append rule bodies to the first system message (Qwen single-system constraint)."""
    merged, meta = merge_applicable_rules_into_chat_messages(
        cast(List[Dict[str, Any]], system_messages),
        applicable,
    )
    return cast(List[Dict[str, str]], merged), meta


def _rules_block(rules: List[MarkdownRule]) -> str:
    parts = [
        "## Session rules (markdown policies)",
        "",
        "The following rules apply to this turn. Follow them strictly.",
        "",
    ]
    for r in rules:
        header = f"### Rule: {r.name}"
        if r.description:
            header += f"\n_{r.description}_"
        parts.append(header)
        parts.append("")
        parts.append(r.rule.strip())
        parts.append("")
    return "\n".join(parts).strip()
