"""Glob path matching (globstar) for rules — Continue-compatible semantics."""
from __future__ import annotations

from typing import List, Union

try:
    from wcmatch import fnmatch as wfnmatch
except ImportError:  # pragma: no cover
    wfnmatch = None  # type: ignore

GlobsInput = Union[str, List[str]]
_FLAGS = None


def _flags():
    global _FLAGS
    if _FLAGS is not None:
        return _FLAGS
    if wfnmatch is None:
        _FLAGS = 0
    else:
        _FLAGS = wfnmatch.GLOBSTAR | wfnmatch.DOTGLOB | wfnmatch.EXTMATCH
    return _FLAGS


def _one_match(path: str, pattern: str) -> bool:
    path = path.replace("\\", "/")
    pattern = pattern.replace("\\", "/")
    base = path.split("/")[-1]
    if wfnmatch is not None:
        fl = _flags()
        if wfnmatch.fnmatch(path, pattern, flags=fl) or wfnmatch.fnmatch(
            base, pattern, flags=fl
        ):
            return True
        # ``**/foo`` from normalized rules + basename-only paths (e.g. ``script.py``).
        if pattern.startswith("**/"):
            return wfnmatch.fnmatch(base, pattern[3:], flags=fl)
        return False
    import fnmatch as std

    if std.fnmatch(path, pattern) or std.fnmatch(base, pattern):
        return True
    if pattern.startswith("**/"):
        return std.fnmatch(base, pattern[3:])
    return False


def path_matches_globs(file_path: str, globs: GlobsInput) -> bool:
    """
    - Single pattern: if starts with ``!``, path must NOT match the rest.
    - List: positives ORed; negatives ANDed (exclude); if only negatives, match unless any negative hits.
    """
    if globs is None:
        return True
    path = file_path.replace("\\", "/")

    if isinstance(globs, str):
        g = globs.strip()
        if g.startswith("!"):
            return not _one_match(path, g[1:])
        return _one_match(path, g)

    patterns: List[str] = [str(x) for x in globs if str(x).strip()]
    if not patterns:
        return True

    positive = [p for p in patterns if not p.startswith("!")]
    negative = [p[1:] for p in patterns if p.startswith("!")]

    if not positive:
        return not any(_one_match(path, p) for p in negative)

    return any(_one_match(path, p) for p in positive) and not any(
        _one_match(path, p) for p in negative
    )
