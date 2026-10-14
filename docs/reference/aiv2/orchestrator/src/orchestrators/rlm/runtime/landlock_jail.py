"""Linux Landlock FS jail for the RLM worker (works without user namespaces).

Applied inside the worker process after bootstrap. Rules are inherited by
children (``%%bash`` / ``subprocess``), so this is a real kernel boundary even
when bubblewrap cannot create namespaces (AppArmor / Docker default seccomp).
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from ctypes import Structure, byref, c_int, c_uint64, sizeof
from typing import Iterable, List, Optional, Sequence, Set

logger = logging.getLogger(__name__)

# x86_64 / aarch64 landlock syscall numbers (stable since introduction).
_SYS_CREATE = 444
_SYS_ADD = 445
_SYS_RESTRICT = 446

_LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
_LANDLOCK_RULE_PATH_BENEATH = 1

_ACCESS_FS_EXECUTE = 1 << 0
_ACCESS_FS_WRITE_FILE = 1 << 1
_ACCESS_FS_READ_FILE = 1 << 2
_ACCESS_FS_READ_DIR = 1 << 3
_ACCESS_FS_REMOVE_DIR = 1 << 4
_ACCESS_FS_REMOVE_FILE = 1 << 5
_ACCESS_FS_MAKE_CHAR = 1 << 6
_ACCESS_FS_MAKE_DIR = 1 << 7
_ACCESS_FS_MAKE_REG = 1 << 8
_ACCESS_FS_MAKE_SOCK = 1 << 9
_ACCESS_FS_MAKE_FIFO = 1 << 10
_ACCESS_FS_MAKE_BLOCK = 1 << 11
_ACCESS_FS_MAKE_SYM = 1 << 12
# ABI ≥ 2
_ACCESS_FS_REFER = 1 << 13
# ABI ≥ 3
_ACCESS_FS_TRUNCATE = 1 << 14

_FS_READ = (
    _ACCESS_FS_EXECUTE
    | _ACCESS_FS_READ_FILE
    | _ACCESS_FS_READ_DIR
)
_FS_WRITE = (
    _ACCESS_FS_WRITE_FILE
    | _ACCESS_FS_REMOVE_DIR
    | _ACCESS_FS_REMOVE_FILE
    | _ACCESS_FS_MAKE_CHAR
    | _ACCESS_FS_MAKE_DIR
    | _ACCESS_FS_MAKE_REG
    | _ACCESS_FS_MAKE_SOCK
    | _ACCESS_FS_MAKE_FIFO
    | _ACCESS_FS_MAKE_BLOCK
    | _ACCESS_FS_MAKE_SYM
    | _ACCESS_FS_TRUNCATE
)
_FS_RW = _FS_READ | _FS_WRITE | _ACCESS_FS_REFER

_PR_SET_NO_NEW_PRIVS = 38

_INSTALLED = False

# Captured at import time so Landlock rule setup still works after the process
# FS guard patches ``os.open`` (otherwise adding /usr rules silently fails and
# ``import datetime`` gets EACCES).
_OS_OPEN = os.open
_OS_CLOSE = os.close


class _RulesetAttr(Structure):
    _fields_ = [("handled_access_fs", c_uint64)]


class _PathBeneathAttr(Structure):
    _fields_ = [
        ("allowed_access", c_uint64),
        ("parent_fd", c_int),
    ]


def _libc():
    return ctypes.CDLL(None, use_errno=True)


def landlock_abi_version() -> int:
    libc = _libc()
    syscall = libc.syscall
    syscall.restype = c_int
    ver = syscall(_SYS_CREATE, None, 0, _LANDLOCK_CREATE_RULESET_VERSION)
    return int(ver) if ver >= 0 else -1


def landlock_available() -> bool:
    try:
        return landlock_abi_version() >= 1
    except Exception:
        return False


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def landlock_enabled() -> bool:
    """Default on when the kernel supports Landlock. ``AIV2_RLM_LANDLOCK=0`` disables."""
    if not landlock_available():
        return False
    return _env_flag("AIV2_RLM_LANDLOCK", default=True)


def _realpath(path: str) -> str:
    try:
        return os.path.realpath(path)
    except OSError:
        return os.path.abspath(path)


def _collect_ro_roots(extra: Sequence[str] | None = None) -> List[str]:
    out: List[str] = []
    for p in ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc"):
        if os.path.exists(p):
            out.append(p)
    # Intentionally omit /proc and /dev from Landlock allowlists — kernel
    # special FS access is still available for open fds; path-beneath on /proc
    # would otherwise widen ``/proc/self/root/...`` toward allowlisted trees only,
    # which is fine, but sibling-process poking is unnecessary for the worker.
    for entry in sys.path:
        if entry and entry != "." and os.path.exists(entry):
            out.append(entry)
    venv = os.environ.get("VIRTUAL_ENV")
    if venv and os.path.isdir(venv):
        out.append(venv)
    exe = _realpath(sys.executable)
    prefix = os.path.dirname(os.path.dirname(exe))
    if prefix and os.path.isdir(prefix):
        out.append(prefix)
    # Repo root
    here = os.path.abspath(os.path.dirname(__file__))
    cur = here
    for _ in range(8):
        if os.path.isdir(os.path.join(cur, "modules", "plugins")):
            out.append(cur)
            break
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    if extra:
        out.extend(extra)
    # Dedupe preferring shorter (parent) paths
    norms = sorted({_realpath(p) for p in out if p and os.path.exists(p)}, key=lambda x: (len(x), x))
    kept: List[str] = []
    for p in norms:
        if any(p == k or p.startswith(k + os.sep) for k in kept):
            continue
        kept.append(p)
    return kept


def install_landlock_fs_jail(
    session_root: str,
    *,
    extra_ro: Sequence[str] | None = None,
) -> bool:
    """Restrict this process (and children) to *session_root* RW + allowlisted RO paths.

    Returns True when rules were applied. Idempotent within the process.
    """
    global _INSTALLED
    if _INSTALLED:
        return True
    if not landlock_enabled():
        return False

    session = _realpath(session_root)
    if not session or not os.path.isdir(session):
        logger.warning("landlock skipped: bad session_root %r", session_root)
        return False

    libc = _libc()
    syscall = libc.syscall
    syscall.restype = c_int
    prctl = libc.prctl
    prctl.restype = c_int

    handled = _FS_RW
    abi = landlock_abi_version()
    if abi < 2:
        handled &= ~_ACCESS_FS_REFER
    if abi < 3:
        handled &= ~_ACCESS_FS_TRUNCATE

    attr = _RulesetAttr(handled)
    ruleset_fd = syscall(_SYS_CREATE, byref(attr), sizeof(attr), 0)
    if ruleset_fd < 0:
        logger.warning("landlock_create_ruleset failed errno=%s", ctypes.get_errno())
        return False

    def _add_path(path: str, access: int) -> None:
        nonlocal ruleset_fd
        rp = _realpath(path)
        if not os.path.exists(rp):
            return
        flags = os.O_PATH | os.O_CLOEXEC
        if hasattr(os, "O_DIRECTORY") and os.path.isdir(rp):
            flags |= os.O_DIRECTORY
        try:
            parent_fd = _OS_OPEN(rp, flags)
        except OSError:
            return
        try:
            rule = _PathBeneathAttr(access & handled, parent_fd)
            rc = syscall(
                _SYS_ADD,
                ruleset_fd,
                _LANDLOCK_RULE_PATH_BENEATH,
                byref(rule),
                0,
            )
            if rc != 0:
                logger.debug(
                    "landlock_add_rule failed path=%s errno=%s",
                    rp,
                    ctypes.get_errno(),
                )
        finally:
            _OS_CLOSE(parent_fd)

    try:
        for ro in _collect_ro_roots(extra_ro):
            # Do not grant RO on parent of session that would include sibling sessions.
            if ro == session or session.startswith(ro.rstrip("/") + os.sep):
                # If session is under this RO root (e.g. /tmp), skip the broad parent;
                # session gets its own RW rule below.
                if ro in ("/tmp", "/var", "/var/tmp") or ro.endswith("aiv2_rlm_sessions"):
                    continue
            _add_path(ro, _FS_READ | (_ACCESS_FS_EXECUTE))
        _add_path(session, handled)

        # Required before restrict_self
        if prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            logger.warning("prctl(NO_NEW_PRIVS) failed errno=%s", ctypes.get_errno())
            os.close(ruleset_fd)
            return False

        rc = syscall(_SYS_RESTRICT, ruleset_fd, 0)
        if rc != 0:
            logger.warning("landlock_restrict_self failed errno=%s", ctypes.get_errno())
            os.close(ruleset_fd)
            return False
        _INSTALLED = True
        logger.info("rlm landlock FS jail active session=%s abi=%s", session, abi)
        return True
    finally:
        try:
            os.close(ruleset_fd)
        except OSError:
            pass


def reset_landlock_for_tests() -> None:
    global _INSTALLED
    _INSTALLED = False
