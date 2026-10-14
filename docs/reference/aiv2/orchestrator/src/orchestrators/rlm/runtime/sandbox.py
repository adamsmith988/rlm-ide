"""Process-level filesystem guard for the RLM IPython worker subprocess."""

from __future__ import annotations

import builtins
import glob as _glob_mod
import os
import re
import sys
from pathlib import Path as _PathClass
from typing import Any, Callable, List, Optional

_SESSION_ROOT: Optional[str] = None
_RO_ROOTS: List[str] = []
_INSTALLED = False
_ORIG: dict[str, Any] = {}


class FsGuardError(PermissionError):
    """Raised when kernel code tries to access paths outside the session root."""


def session_root() -> Optional[str]:
    return _SESSION_ROOT


def _real(path: str | os.PathLike[str]) -> str:
    fn = _ORIG.get("realpath") or os.path.realpath
    try:
        return fn(os.path.expanduser(os.fspath(path)))
    except OSError:
        abspath = _ORIG.get("abspath") or os.path.abspath
        return abspath(os.fspath(path))


def _resolve(path: str | os.PathLike[str]) -> str:
    p = os.fspath(path)
    if not os.path.isabs(p):
        p = os.path.join(os.getcwd(), p)
    return _real(p)


def _collect_ro_roots() -> List[str]:
    """System / app trees the worker may read (stdlib, site-packages, /app)."""
    out: List[str] = []
    for p in ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/app"):
        if os.path.exists(p):
            out.append(_real(p))
    for entry in sys.path:
        if entry and entry != "." and os.path.exists(entry):
            out.append(_real(entry))
    venv = os.environ.get("VIRTUAL_ENV")
    if venv and os.path.isdir(venv):
        out.append(_real(venv))
    # Prefer shorter parents
    norms = sorted(set(out), key=lambda x: (len(x), x))
    kept: List[str] = []
    for p in norms:
        if any(p == k or p.startswith(k + os.sep) for k in kept):
            continue
        kept.append(p)
    return kept


def assert_path_allowed(path: str | os.PathLike[str]) -> str:
    """Allow session cwd (RW intent) plus RO system/app trees for imports."""
    root = _SESSION_ROOT
    if not root:
        return _resolve(path)
    resolved = _resolve(path)
    root_r = _real(root)
    if resolved == root_r or resolved.startswith(root_r + os.sep):
        return resolved
    for ro in _RO_ROOTS:
        if resolved == ro or resolved.startswith(ro + os.sep):
            return resolved
    raise FsGuardError(f"filesystem_access_denied:{resolved}")


def _wrap_path_arg(fn: Callable[..., Any]) -> Callable[..., Any]:
    def guarded(*args: Any, **kwargs: Any) -> Any:
        if args:
            assert_path_allowed(args[0])
        return fn(*args, **kwargs)

    return guarded


def _wrap_path_method(fn: Callable[..., Any]) -> Callable[..., Any]:
    def guarded(self: _PathClass, *args: Any, **kwargs: Any) -> Any:
        assert_path_allowed(self)
        return fn(self, *args, **kwargs)

    return guarded


def install_process_fs_guard(session_root: str) -> None:
    """Restrict this worker process to *session_root* + RO system trees (idempotent)."""
    global _SESSION_ROOT, _RO_ROOTS, _INSTALLED
    _SESSION_ROOT = _real(session_root)
    _RO_ROOTS = _collect_ro_roots()
    if _INSTALLED:
        return
    _INSTALLED = True

    _ORIG["realpath"] = os.path.realpath
    _ORIG["abspath"] = os.path.abspath
    _ORIG["open"] = builtins.open

    def guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)):
            assert_path_allowed(file)
        else:
            name = getattr(file, "name", None)
            if isinstance(name, str) and name:
                assert_path_allowed(name)
        return _ORIG["open"](file, mode, *args, **kwargs)

    builtins.open = guarded_open  # type: ignore[assignment]

    _ORIG["os_open"] = os.open

    def guarded_os_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if dir_fd is None:
            assert_path_allowed(path)
        return _ORIG["os_open"](path, flags, mode, dir_fd=dir_fd)

    os.open = guarded_os_open  # type: ignore[assignment]

    for name in (
        "listdir",
        "scandir",
        "remove",
        "unlink",
        "rmdir",
        "mkdir",
        "makedirs",
        "rename",
        "replace",
        "chmod",
    ):
        if hasattr(os, name):
            _ORIG[f"os_{name}"] = getattr(os, name)
            setattr(os, name, _wrap_path_arg(_ORIG[f"os_{name}"]))

    _ORIG["os_walk"] = os.walk
    os.walk = _wrap_path_arg(_ORIG["os_walk"])  # type: ignore[assignment]

    _ORIG["os_chdir"] = os.chdir
    os.chdir = _wrap_path_arg(_ORIG["os_chdir"])  # type: ignore[assignment]

    _ORIG["glob"] = _glob_mod.glob

    def guarded_glob(pathname: str, *args: Any, **kwargs: Any) -> list[str]:
        if os.path.isabs(pathname):
            head = pathname.split("*", 1)[0].split("[", 1)[0].rstrip("/") or pathname
            assert_path_allowed(head)
        return _ORIG["glob"](pathname, *args, **kwargs)

    _glob_mod.glob = guarded_glob  # type: ignore[assignment]

    for meth in (
        "read_text",
        "read_bytes",
        "write_text",
        "write_bytes",
        "open",
        "iterdir",
        "glob",
        "rglob",
        "unlink",
        "mkdir",
        "rmdir",
        "rename",
        "replace",
        "touch",
    ):
        orig = getattr(_PathClass, meth, None)
        if callable(orig):
            key = f"path_{meth}"
            if key not in _ORIG:
                _ORIG[key] = orig
            setattr(_PathClass, meth, _wrap_path_method(_ORIG[key]))


def restore_process_fs_guard() -> None:
    """Restore patched builtins/os/pathlib (for tests)."""
    global _SESSION_ROOT, _RO_ROOTS, _INSTALLED
    if _ORIG.get("open"):
        builtins.open = _ORIG["open"]  # type: ignore[assignment]
    if _ORIG.get("os_open"):
        os.open = _ORIG["os_open"]  # type: ignore[assignment]
    for name in (
        "listdir",
        "scandir",
        "remove",
        "unlink",
        "rmdir",
        "mkdir",
        "makedirs",
        "rename",
        "replace",
        "chmod",
        "walk",
        "chdir",
    ):
        key = f"os_{name}"
        if _ORIG.get(key):
            setattr(os, name, _ORIG[key])
    if _ORIG.get("glob"):
        _glob_mod.glob = _ORIG["glob"]  # type: ignore[assignment]
    for meth in (
        "read_text",
        "read_bytes",
        "write_text",
        "write_bytes",
        "open",
        "iterdir",
        "glob",
        "rglob",
        "unlink",
        "mkdir",
        "rmdir",
        "rename",
        "replace",
        "touch",
    ):
        key = f"path_{meth}"
        if _ORIG.get(key):
            setattr(_PathClass, meth, _ORIG[key])
    _SESSION_ROOT = None
    _RO_ROOTS = []
    _INSTALLED = False


# Only host *data* escapes. Do NOT list /usr /bin /lib — models often invoke
# `/usr/bin/grep` and that must stay allowed inside the session sandbox.
_BASH_ABS = re.compile(
    r"(?<![\w./-])"
    r"(/(?:home|root|etc)(?:/[^\s;&|<>\"']*)?"
    r"|/tmp/aiv2_rlm_sessions/[^\s;&|<>\"']+)"
)


_BASH_TOKEN = re.compile(r"[^\s;&|<>\"'()=]+")


def _escapes_root(resolved: str, root: str) -> bool:
    return resolved != root and not resolved.startswith(root + os.sep)


def bash_script_allowed(
    script: str, session_root: str, *, cwd: Optional[str] = None
) -> Optional[str]:
    """Return an error code when *script* references paths outside *session_root*."""
    root = _real(session_root)
    text = str(script or "")
    for match in _BASH_ABS.finditer(text):
        fragment = match.group(1).rstrip("'\"")
        token = fragment.split()[0] if " " in fragment else fragment
        try:
            resolved = _real(token)
        except OSError:
            resolved = token
        if _escapes_root(resolved, root):
            return f"bash_path_denied:{fragment}"
    base = _real(cwd) if cwd else root
    if _escapes_root(base, root):
        base = root
    for match in _BASH_TOKEN.finditer(text):
        token = match.group(0)
        if ".." not in token.replace("\\", "/").split("/"):
            continue
        try:
            resolved = _real(os.path.join(base, token))
        except OSError:
            continue
        if _escapes_root(resolved, root):
            return f"bash_path_denied:{token}"
    return None
