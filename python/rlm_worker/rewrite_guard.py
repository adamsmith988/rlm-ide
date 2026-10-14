"""Stops "fix by rewriting the whole file" from REPL cells.

Cells get a guarded `open()` (and `Path.write_text` while a cell runs). Rewriting an existing source
file in full while changing only a few lines is allowed once per file with a warning, then refused
with the exact lines to change through fs.edit_file. Same rule and thresholds as fs.write_file;
the host sends the thresholds in the bootstrap message.
"""

from __future__ import annotations

import difflib
import io
import os
import pathlib
import sys
import weakref
from typing import Any, Callable, Dict, List, Optional, Tuple

SOURCE_EXT = {
    ".py", ".pyi", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".sh", ".bash", ".go", ".rs", ".java", ".kt", ".scala",
    ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".lua", ".r", ".jl", ".html", ".css", ".scss",
    ".vue", ".svelte", ".sql",
}
MAX_BYTES = 2 * 1024 * 1024


def rewrite_stats(before: str, after: str) -> Tuple[int, int, int, float, List[Tuple[int, int]]]:
    """(before_lines, after_lines, changed_lines, ratio, changed 1-based ranges of `after`)."""
    a = before.split("\n")
    b = after.split("\n")
    kept = 0
    ranges: List[Tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            kept += i2 - i1
            continue
        lo, hi = (j1 + 1, j2) if j2 > j1 else (j1 + 1, j1 + 1)
        if ranges and lo <= ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], max(hi, ranges[-1][1]))
        else:
            ranges.append((lo, hi))
    changed = max(len(a), len(b)) - kept
    return len(a), len(b), changed, changed / max(1, len(a), len(b)), ranges


def describe_ranges(r: List[Tuple[int, int]], limit: int = 6) -> str:
    parts = [str(x) if x == y else f"{x}-{y}" for x, y in r[:limit]]
    return ", ".join(parts) + (f" and {len(r) - limit} more" if len(r) > limit else "")


class _PendingRewrite(io.StringIO):
    """Collects what the cell writes; decides at close whether it reaches the disk."""

    def __init__(self, guard: "RewriteGuard", path: str, before: str, encoding: Optional[str], newline: Optional[str], name: Any) -> None:
        super().__init__()
        self._guard = guard
        self._path = path
        self._before = before
        self._encoding = encoding or "utf-8"
        self._newline = newline
        self.name = name
        self.mode = "w"
        guard.pending.add(self)

    def close(self) -> None:
        if self.closed:
            return
        content = self.getvalue()
        super().close()
        self._guard.pending.discard(self)
        refusal = self._guard.decide(self._path, self._before, content)
        if refusal:
            raise PermissionError(refusal)
        with self._guard.real_open(self._path, "w", encoding=self._encoding, newline=self._newline) as f:
            f.write(content)


class RewriteGuard:
    def __init__(self, root: str, min_lines: int, max_changed: float, real_open: Callable[..., Any]) -> None:
        self.root = os.path.realpath(root)
        self.min_lines = min_lines
        self.max_changed = max_changed
        self.real_open = real_open
        self.counts: Dict[str, int] = {}
        # Weak, so `open(p, "w").write(s)` is settled as soon as the file object is dropped.
        self.pending: "weakref.WeakSet[_PendingRewrite]" = weakref.WeakSet()
        self.refusals: List[str] = []
        self._orig_write_text = pathlib.Path.write_text

    def _rel(self, path: str) -> str:
        return os.path.relpath(path, self.root)

    def _watched(self, file: Any) -> Optional[Tuple[str, str]]:
        if not isinstance(file, (str, os.PathLike)):
            return None
        path = os.path.realpath(os.fspath(file))
        if not path.startswith(self.root + os.sep) or os.path.splitext(path)[1].lower() not in SOURCE_EXT:
            return None
        try:
            if not os.path.isfile(path) or os.path.getsize(path) > MAX_BYTES:
                return None
            with self.real_open(path, "r", encoding="utf-8") as f:
                before = f.read()
        except (OSError, UnicodeDecodeError):
            return None
        if before.count("\n") + 1 < self.min_lines:
            return None
        return path, before

    def decide(self, path: str, before: str, after: str) -> Optional[str]:
        """None to write; otherwise the refusal message. Warns (to the cell's stderr) on the first small-fix rewrite."""
        n_before, n_after, changed, ratio, ranges = rewrite_stats(before, after)
        if n_before < self.min_lines or changed == 0 or ratio > self.max_changed:
            return None
        rel = self._rel(path)
        n = self.counts.get(path, 0) + 1
        self.counts[path] = n
        where = f"lines {describe_ranges(ranges)} of your new content"
        if n == 1:
            print(
                f"note: you rewrote {rel} in full but changed only {changed} of {n_after} lines ({where}). "
                f'Next time change just those lines with fs.edit_file(path="{rel}", old_string=..., new_string=...); '
                "further full rewrites of this file that change this little will be refused.",
                file=sys.stderr,
            )
            return None
        msg = (
            f"not written: this rewrite of {rel} changes only {changed} of {n_after} lines ({where}). "
            f'Change just those lines with fs.edit_file(path="{rel}", old_string=..., new_string=...), one call per changed block. '
            "If the file really has to be replaced as a whole, use fs.write_file(..., overwrite=True)."
        )
        self.refusals.append(msg)
        return msg

    def open(self, file: Any, mode: str = "r", buffering: int = -1, encoding: Optional[str] = None, errors: Optional[str] = None, newline: Optional[str] = None, closefd: bool = True, opener: Any = None) -> Any:
        if "w" in mode and "b" not in mode and "+" not in mode and closefd and opener is None:
            hit = self._watched(file)
            if hit:
                return _PendingRewrite(self, hit[0], hit[1], encoding, newline, file)
        return self.real_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

    def install(self) -> None:
        guard = self
        orig = self._orig_write_text

        def write_text(path_self: pathlib.Path, data: str, encoding: Optional[str] = None, errors: Optional[str] = None, newline: Optional[str] = None) -> int:
            if guard._watched(path_self) is None:
                kwargs = {"encoding": encoding, "errors": errors}
                if newline is not None:
                    kwargs["newline"] = newline
                return orig(path_self, data, **kwargs)
            with guard.open(path_self, "w", encoding=encoding, newline=newline) as f:
                return f.write(data)

        pathlib.Path.write_text = write_text  # type: ignore[assignment]

    def uninstall(self) -> Optional[str]:
        """Restores Path.write_text, settles files the cell left open, and returns the refusals of this cell."""
        pathlib.Path.write_text = self._orig_write_text  # type: ignore[assignment]
        for p in list(self.pending):
            try:
                p.close()
            except PermissionError:
                pass
        out = "\n".join(dict.fromkeys(self.refusals)) or None
        self.refusals = []
        return out
