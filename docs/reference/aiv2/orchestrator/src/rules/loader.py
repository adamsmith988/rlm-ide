"""Load ``*.md`` rules from configured directories."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Tuple

from ..config import settings
from .markdown import normalize_globs, parse_markdown_rule, rule_name_from_frontmatter
from .types import MarkdownRule

log = logging.getLogger(__name__)

_cache: Tuple[float, Tuple[MarkdownRule, ...], Tuple[str, ...]] = (
    0.0,
    (),
    (),
)


def get_rule_directory_roots() -> List[Path]:
    """Configured rules roots (absolute paths). Empty if disabled or unset."""
    if not getattr(settings, "RULES_ENABLED", True):
        return []
    raw = getattr(settings, "RULES_MARKDOWN_DIRS", "") or ""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        # Default: ``orchestrator/agent_rules`` next to this package (works in dev & typical Docker layouts).
        bundled = Path(__file__).resolve().parent.parent.parent / "agent_rules"
        if bundled.is_dir():
            return [bundled.resolve()]
        return []
    out: List[Path] = []
    for p in parts:
        path = Path(p).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        out.append(path.resolve())
    return out


def _rules_roots_signature(roots: List[Path]) -> Tuple[str, ...]:
    sig: List[str] = []
    for r in roots:
        try:
            m = r.stat().st_mtime_ns
        except OSError:
            m = 0
        sig.append(f"{r}:{m}")
    return tuple(sig)


def load_markdown_rules(*, force: bool = False) -> List[MarkdownRule]:
    """
    Scan ``RULES_MARKDOWN_DIRS`` for ``*.md`` and parse frontmatter + body.
    Cached briefly (``RULES_CACHE_TTL_SEC``).
    """
    global _cache
    if not getattr(settings, "RULES_ENABLED", True):
        return []

    roots = get_rule_directory_roots()
    if not roots:
        return []

    sig = _rules_roots_signature(roots)
    ttl = float(getattr(settings, "RULES_CACHE_TTL_SEC", 30))
    now = time.monotonic()
    if (
        not force
        and _cache[1]
        and _cache[2] == sig
        and (now - _cache[0]) < ttl
    ):
        return list(_cache[1])

    rules: List[MarkdownRule] = []
    for rules_root in roots:
        if not rules_root.is_dir():
            log.debug("Rules dir missing (skip): %s", rules_root)
            continue
        for path in sorted(rules_root.rglob("*.md")):
            if path.name.startswith("."):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                log.warning("Cannot read rule file %s: %s", path, e)
                continue
            fm, body = parse_markdown_rule(text)
            name = rule_name_from_frontmatter(fm, path)
            try:
                rel_root = path.parent.resolve().relative_to(rules_root.resolve())
                is_root_file = len(rel_root.parts) == 0
            except ValueError:
                is_root_file = path.parent.resolve() == rules_root.resolve()

            globs = normalize_globs(
                fm.get("globs"),
                rule_file_dir=path.parent,
                rules_root=rules_root,
            )
            rule = MarkdownRule(
                name=name,
                rule=body,
                source_file=str(path.resolve()),
                globs=globs,
                regex=fm.get("regex"),
                description=fm.get("description"),
                always_apply=fm.get("alwaysApply"),
                invokable=fm.get("invokable"),
                is_rules_directory_root_file=is_root_file,
                extra={k: v for k, v in fm.items() if k not in (
                    "globs", "regex", "name", "description", "alwaysApply", "invokable",
                )},
            )
            rules.append(rule)

    _cache = (now, tuple(rules), sig)
    log.info("Loaded %s markdown rule(s) from %s", len(rules), roots)
    return rules
