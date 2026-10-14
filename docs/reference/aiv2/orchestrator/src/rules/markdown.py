"""Parse markdown rules with YAML frontmatter (Continue-compatible)."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Tuple

log = logging.getLogger(__name__)

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


def parse_markdown_rule(content: str) -> Tuple[Dict[str, Any], str]:
    """
    Split ``---`` YAML frontmatter from markdown body.
    Mirrors ``parseMarkdownRule`` in @continuedev/config-yaml.
    """
    normalized = content.replace("\r\n", "\n")
    parts = re.split(r"^---\s*$", normalized, maxsplit=2, flags=re.MULTILINE)
    if len(parts) >= 3:
        fm_raw = parts[1]
        body = parts[2]
        if yaml is None:
            log.warning("PyYAML not installed; ignoring rule frontmatter")
            return {}, body.strip()
        try:
            frontmatter = yaml.safe_load(fm_raw) or {}
            if not isinstance(frontmatter, dict):
                frontmatter = {}
        except Exception as exc:
            log.warning("Invalid YAML frontmatter in rule: %s", exc)
            frontmatter = {}
        return frontmatter, body.strip()
    return {}, normalized.strip()


def rule_name_from_frontmatter(frontmatter: Dict[str, Any], file_path: Path) -> str:
    name = frontmatter.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    stem = file_path.stem
    if stem:
        return stem
    return "rule"


def normalize_globs(
    globs: Any,
    *,
    rule_file_dir: Path,
    rules_root: Path,
) -> Any:
    """
    If the rule file lives outside ``.continue``-style meta dirs, prepend
    ``**/`` to relative globs so they match anywhere under workspace (Continue behaviour sketch).
    For server-side we prepend ``**/`` only when the pattern has no path sep and no leading **.
    """
    if globs is None:
        return None
    try:
        rel_to_root = rule_file_dir.resolve().relative_to(rules_root.resolve())
    except ValueError:
        rel_to_root = None

    def one(g: str) -> str:
        g = str(g).replace("\\", "/").strip()
        if not g:
            return g
        if g.startswith("!"):
            return "!" + one(g[1:])
        if g.startswith("**"):
            return g
        if "/" in g or g.startswith("/"):
            return g
        # bare pattern like *.py → match anywhere
        return "**/" + g

    if rel_to_root is not None and len(rel_to_root.parts) > 0:
        # Nested rule under rules root: bias patterns toward subtree (optional).
        pass

    if isinstance(globs, list):
        return [one(str(x)) for x in globs]
    return one(str(globs))
