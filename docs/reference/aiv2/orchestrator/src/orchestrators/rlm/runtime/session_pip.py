"""Session-local pip installs for RLM Agent kernels.

System ``site-packages`` stay read-only under bwrap/Landlock. When enabled
(``AIV2_RLM_ALLOW_PIP``, default on), packages install into
``{session}/.rlm-pip`` via ``PYTHONUSERBASE`` / ``pip install --user``.
"""

from __future__ import annotations

import os
import sys
from typing import Dict


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def pip_install_allowed() -> bool:
    """Whether Agent kernels may run ``pip install`` into the session userbase."""
    return _env_flag("AIV2_RLM_ALLOW_PIP", default=True)


def session_pip_userbase(session_dir: str) -> str:
    return os.path.join(os.path.realpath(session_dir), ".rlm-pip")


def session_pip_site_packages(session_dir: str) -> str:
    base = session_pip_userbase(session_dir)
    ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return os.path.join(base, "lib", ver, "site-packages")


def configure_session_pip(session_dir: str) -> Dict[str, str]:
    """Create the session userbase and return env vars for the worker process.

    Empty dict when pip installs are disabled.
    """
    if not pip_install_allowed():
        return {}
    sess = os.path.realpath(session_dir)
    if not sess or not os.path.isdir(sess):
        return {}
    userbase = session_pip_userbase(sess)
    site_pkgs = session_pip_site_packages(sess)
    bin_dir = os.path.join(userbase, "bin")
    try:
        os.makedirs(site_pkgs, exist_ok=True)
        os.makedirs(bin_dir, exist_ok=True)
    except OSError:
        return {}

    env: Dict[str, str] = {
        "PYTHONUSERBASE": userbase,
        "PIP_USER": "1",
    }
    # Prefer user site over a host PYTHONNOUSERSITE inherit.
    env["PYTHONNOUSERSITE"] = ""

    path_parts = [p for p in str(os.environ.get("PATH") or "").split(os.pathsep) if p]
    if bin_dir not in path_parts:
        path_parts.insert(0, bin_dir)
    env["PATH"] = os.pathsep.join(path_parts)

    py_parts = [p for p in str(os.environ.get("PYTHONPATH") or "").split(os.pathsep) if p]
    if site_pkgs not in py_parts:
        py_parts.insert(0, site_pkgs)
    env["PYTHONPATH"] = os.pathsep.join(py_parts)
    return env


def apply_session_pip_env(session_dir: str) -> Dict[str, str]:
    """Configure pip userbase and apply env vars into ``os.environ``."""
    updates = configure_session_pip(session_dir)
    for key, val in updates.items():
        if key == "PYTHONNOUSERSITE" and not val:
            os.environ.pop("PYTHONNOUSERSITE", None)
            continue
        if key == "PYTHONPATH":
            # Merge with any path already prepared by the host (app root, etc.).
            existing = [p for p in str(os.environ.get("PYTHONPATH") or "").split(os.pathsep) if p]
            site_pkgs = session_pip_site_packages(session_dir)
            merged = [site_pkgs] + [p for p in existing if p != site_pkgs]
            os.environ["PYTHONPATH"] = os.pathsep.join(merged)
            updates["PYTHONPATH"] = os.environ["PYTHONPATH"]
            continue
        if key == "PATH":
            existing = [p for p in str(os.environ.get("PATH") or "").split(os.pathsep) if p]
            bin_dir = os.path.join(session_pip_userbase(session_dir), "bin")
            merged = [bin_dir] + [p for p in existing if p != bin_dir]
            os.environ["PATH"] = os.pathsep.join(merged)
            updates["PATH"] = os.environ["PATH"]
            continue
        os.environ[key] = val
    return updates
