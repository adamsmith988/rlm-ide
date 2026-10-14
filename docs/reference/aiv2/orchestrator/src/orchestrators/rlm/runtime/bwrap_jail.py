"""Bubblewrap (bwrap) OS jail for the RLM IPython worker subprocess.

The Python monkey-patch in ``sandbox.py`` is defense-in-depth only. Real isolation
comes from running the entire worker inside a mount/user namespace so ``io.open``,
``subprocess``, and ``%%bash`` cannot see host paths outside the session dir.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from typing import Iterable, List, Optional, Sequence, Set

logger = logging.getLogger(__name__)

WORKER_MODULE = "src.orchestrators.rlm.runtime.worker"

# Paths we never want as a whole-tree bind (even if they appear on sys.path).
_BLOCKED_ROOTS = frozenset(
    {
        "/home",
        "/root",
        "/var",
        "/opt",
        "/srv",
        "/mnt",
        "/media",
        "/boot",
        "/data",
    }
)


def bwrap_binary() -> Optional[str]:
    return shutil.which("bwrap")


def bwrap_probe() -> bool:
    """True when a minimal bwrap sandbox can actually start on this host.

    Binary presence is not enough: AppArmor ``apparmor_restrict_unprivileged_userns``
    and Docker's default seccomp often return ``Permission denied`` on uid maps.
    """
    binary = bwrap_binary()
    if not binary:
        return False
    try:
        import subprocess

        cmd = [
            binary,
            "--die-with-parent",
            "--ro-bind",
            "/usr",
            "/usr",
            "--dev",
            "/dev",
            "--bind",
            "/tmp",
            "/tmp",
        ]
        # Merged-/usr distros need lib/bin reachable
        if os.path.isdir("/lib"):
            cmd.extend(["--ro-bind", "/lib", "/lib"])
        elif os.path.islink("/lib"):
            cmd.extend(["--symlink", "usr/lib", "/lib"])
        if os.path.isdir("/bin"):
            cmd.extend(["--ro-bind", "/bin", "/bin"])
        elif os.path.islink("/bin"):
            cmd.extend(["--symlink", "usr/bin", "/bin"])
        true_bin = "/usr/bin/true" if os.path.isfile("/usr/bin/true") else "/bin/true"
        cmd.extend(["--", true_bin])
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return proc.returncode == 0
    except Exception:
        return False


_PROBE_CACHE: Optional[bool] = None


def bwrap_available() -> bool:
    """Usable bwrap (binary + successful probe)."""
    global _PROBE_CACHE
    if _PROBE_CACHE is None:
        _PROBE_CACHE = bool(bwrap_binary()) and bwrap_probe()
    return _PROBE_CACHE


def reset_bwrap_probe_cache_for_tests() -> None:
    global _PROBE_CACHE
    _PROBE_CACHE = None


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def bwrap_enabled() -> bool:
    """Whether to attempt bwrap when the binary exists.

    Default: on when ``bwrap`` is on PATH. Override with ``AIV2_RLM_BWRAP=0``.
    """
    if not bwrap_available():
        return False
    return _env_flag("AIV2_RLM_BWRAP", default=True)


def bwrap_required() -> bool:
    """Fail kernel start if jail cannot be applied (prod)."""
    return _env_flag("AIV2_RLM_REQUIRE_BWRAP", default=False)


def _realpath(path: str) -> str:
    try:
        return os.path.realpath(path)
    except OSError:
        return os.path.abspath(path)


def _is_blocked_tree(path: str) -> bool:
    """True if *path* is exactly a blocked root (e.g. /home), not a subpath we need."""
    rp = _realpath(path).rstrip("/") or "/"
    return rp in _BLOCKED_ROOTS


def _collect_ro_paths(
    *,
    python_executable: str,
    extra: Sequence[str] | None = None,
) -> List[str]:
    """Minimal RO bind set so CPython + project modules import cleanly."""
    candidates: List[str] = []

    # Standard FHS / Ubuntu merged-/usr layout
    for p in ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc"):
        if os.path.exists(p):
            candidates.append(p)

    # Interpreter + venv prefix (may live under /home/.../venv — bind that prefix only)
    exe = _realpath(python_executable)
    candidates.append(exe)
    prefix = os.path.dirname(os.path.dirname(exe))  # .../bin/python → prefix
    if prefix and os.path.isdir(prefix):
        candidates.append(prefix)
    venv = os.environ.get("VIRTUAL_ENV")
    if venv and os.path.isdir(venv):
        candidates.append(venv)

    for entry in sys.path:
        if not entry or not isinstance(entry, str):
            continue
        if entry == ".":
            continue
        if not os.path.exists(entry):
            continue
        # Prefer the path itself; never bind /home as a whole.
        if _is_blocked_tree(entry):
            continue
        candidates.append(entry)

    # Repo / app roots commonly needed for ``modules.plugins...``
    for hint in (
        os.environ.get("PYTHONPATH", ""),
        "/app",
        "/app/modules",
        "/app/packages",
    ):
        for part in str(hint).split(os.pathsep):
            part = part.strip()
            if part and os.path.exists(part) and not _is_blocked_tree(part):
                candidates.append(part)

    # Walk up from this file to find the AIv2 repo root (contains ``modules/``).
    here = os.path.abspath(os.path.dirname(__file__))
    cur = here
    for _ in range(8):
        if os.path.isdir(os.path.join(cur, "modules", "plugins")):
            candidates.append(cur)
            candidates.append(os.path.join(cur, "modules"))
            break
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    if extra:
        candidates.extend(extra)

    # Minimal identity files (passwd may be needed by getpwuid in some libs)
    for p in (
        "/etc/passwd",
        "/etc/group",
        "/etc/nsswitch.conf",
        "/etc/hosts",
        "/etc/resolv.conf",
        "/etc/ssl",
        "/etc/ca-certificates",
        "/etc/localtime",
        "/etc/timezone",
    ):
        if os.path.exists(p):
            candidates.append(p)

    return _dedupe_mount_paths(candidates)


def _dedupe_mount_paths(paths: Iterable[str]) -> List[str]:
    """Prefer parent binds; drop children already covered; skip blocked roots."""
    normalized: List[str] = []
    seen: Set[str] = set()
    for raw in paths:
        if not raw:
            continue
        rp = _realpath(raw)
        if rp in seen:
            continue
        if _is_blocked_tree(rp):
            continue
        # Never bind the entire sessions parent — only the per-session dir is RW.
        if rp.rstrip("/") == "/tmp" or rp.endswith("/aiv2_rlm_sessions"):
            continue
        if not os.path.exists(rp):
            continue
        seen.add(rp)
        normalized.append(rp)

    # Sort shortest-first so parents come before children; keep parent, drop child.
    normalized.sort(key=lambda p: (len(p), p))
    kept: List[str] = []
    for p in normalized:
        if any(p == parent or p.startswith(parent + os.sep) for parent in kept):
            continue
        kept.append(p)
    return kept


def _append_ro_bind(cmd: List[str], path: str) -> None:
    """Bind *path* read-only, creating parent dirs via --dir as needed."""
    rp = _realpath(path)
    if not os.path.exists(rp):
        return
    # Ensure parent directories exist inside the sandbox for deep binds
    # (bwrap creates the mount point; parents of custom paths under /home need --dir).
    parent = os.path.dirname(rp)
    parts: List[str] = []
    while parent and parent not in ("/", ""):
        parts.append(parent)
        nxt = os.path.dirname(parent)
        if nxt == parent:
            break
        parent = nxt
    # Create from root downward for paths under blocked roots (e.g. ~/.local/...)
    for d in reversed(parts):
        if d in _BLOCKED_ROOTS or any(
            d.startswith(br + os.sep) for br in _BLOCKED_ROOTS
        ):
            # Only mkdir the intermediate dirs we need under /home etc.
            if d not in cmd:  # cheap membership; dirs added as --dir args
                pass
    # Add --dir for each ancestor under a blocked root so the leaf can mount.
    for d in reversed(parts):
        under_blocked = any(d == br or d.startswith(br + os.sep) for br in _BLOCKED_ROOTS)
        if under_blocked:
            cmd.extend(["--dir", d])

    if os.path.islink(path) and not os.path.isdir(rp):
        # Follow to real target for files; for dirs use realpath already.
        pass
    if os.path.isdir(rp):
        cmd.extend(["--ro-bind", rp, rp])
    else:
        cmd.extend(["--ro-bind", rp, rp])


def build_worker_command(
    *,
    python_executable: str,
    session_dir: str,
    module: str = WORKER_MODULE,
    extra_ro: Sequence[str] | None = None,
) -> List[str]:
    """Return ``bwrap … -- python -m <module>`` argv for the worker.

    Raises ``RuntimeError`` if ``bwrap`` is missing.
    """
    binary = bwrap_binary()
    if not binary:
        raise RuntimeError("bwrap_unavailable")

    session = _realpath(session_dir)
    if not session or not os.path.isdir(session):
        raise RuntimeError(f"bwrap_bad_session_dir:{session_dir}")

    py = _realpath(python_executable) if python_executable else sys.executable
    if not os.path.isfile(py) and not os.path.islink(python_executable or ""):
        py = python_executable or sys.executable

    cmd: List[str] = [
        binary,
        "--die-with-parent",
        "--unshare-pid",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/var/tmp",
    ]
    # Network namespace is best-effort: some hosts reject loopback setup
    # (RTM_NEWADDR). FS isolation is the security boundary we need.
    if _env_flag("AIV2_RLM_BWRAP_UNSHARE_NET", default=False):
        cmd.append("--unshare-net")

    # Symlink helpers for merged-/usr distros (best-effort; ignore if exists as dir)
    for link, target in (
        ("/bin", "usr/bin"),
        ("/sbin", "usr/sbin"),
        ("/lib", "usr/lib"),
        ("/lib64", "usr/lib64"),
    ):
        # Prefer real binds from _collect; symlinks only if path missing as dir
        if not os.path.isdir(link) and os.path.islink(link):
            cmd.extend(["--symlink", target, link])

    for path in _collect_ro_paths(python_executable=py, extra=extra_ro):
        if path == session or path.startswith(session + os.sep):
            continue
        _append_ro_bind(cmd, path)

    # Session workspace: read-write, only this directory under sessions root.
    # Create parent dirs if session lives under /tmp/...
    parent = os.path.dirname(session)
    ancestors: List[str] = []
    cur = parent
    while cur and cur not in ("/", ""):
        ancestors.append(cur)
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    for d in reversed(ancestors):
        # /tmp is already tmpfs — still need nested dirs for bind mount point
        if d in ("/tmp", "/var", "/var/tmp"):
            continue
        cmd.extend(["--dir", d])
    if parent.startswith("/tmp/") or parent == "/tmp":
        # nested under tmpfs /tmp
        for d in reversed(ancestors):
            if d.startswith("/tmp/") or d == "/tmp":
                cmd.extend(["--dir", d])

    cmd.extend(["--bind", session, session])
    cmd.extend(["--chdir", session])
    cmd.extend(["--setenv", "HOME", session])
    cmd.extend(["--setenv", "TMPDIR", "/tmp"])
    cmd.extend(["--setenv", "TMP", "/tmp"])
    cmd.extend(["--setenv", "TEMP", "/tmp"])
    cmd.extend(["--setenv", "PYTHONUNBUFFERED", "1"])
    cmd.extend(["--setenv", "PYTHONDONTWRITEBYTECODE", "1"])

    # Preserve PYTHONPATH from host so modules resolve the same way.
    pypath = os.environ.get("PYTHONPATH")
    if pypath:
        cmd.extend(["--setenv", "PYTHONPATH", pypath])
    # Session-local pip userbase (writable under the session bind).
    for key in (
        "PYTHONUSERBASE",
        "PIP_USER",
        "PATH",
        "AIV2_RLM_ALLOW_PIP",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "no_proxy",
    ):
        val = os.environ.get(key)
        if val:
            cmd.extend(["--setenv", key, val])

    cmd.extend(["--", py, "-m", module])
    return cmd


def resolve_worker_argv(
    *,
    python_executable: str,
    session_dir: str,
    module: str = WORKER_MODULE,
) -> tuple[List[str], bool]:
    """Return (argv, jailed).

    If jail is disabled/unavailable and not required, returns the plain
    ``[python, -m, module]`` command. If required and unavailable, raises.
    """
    plain = [python_executable, "-m", module]
    usable = bwrap_available()
    if bwrap_required() and not usable:
        raise RuntimeError("bwrap_required_but_disabled_or_missing")
    if not bwrap_enabled():
        if not usable:
            logger.warning("rlm bwrap unavailable; spawning worker without OS jail")
        return plain, False
    try:
        return build_worker_command(
            python_executable=python_executable,
            session_dir=session_dir,
            module=module,
        ), True
    except Exception as exc:
        if bwrap_required():
            raise RuntimeError(f"bwrap_required_failed:{exc}") from exc
        logger.warning("rlm bwrap jail build failed (%s); falling back to ungated worker", exc)
        return plain, False
