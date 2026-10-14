"""Kernel namespace snapshot / restore helpers."""

from __future__ import annotations

import json
import logging
import types
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Keys that must not be pickled into snapshots (host bridges, modules).
_SKIP_PREFIXES = ("_",)
_SKIP_NAMES = frozenset(
    {
        "rlm",
        "agent_message",
        "agent_observe",
        "pools",
        "tools",
        "host_request",
        "aiv2_rlm",
        "skills",
        "RLMSpawnHandle",
        "In",
        "Out",
        "exit",
        "quit",
        "get_ipython",
        "open",
    }
)


def filter_user_namespace(ns: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, val in (ns or {}).items():
        if not isinstance(key, str):
            continue
        if key in _SKIP_NAMES or key.startswith(_SKIP_PREFIXES):
            continue
        if callable(val) and not isinstance(val, type):
            continue
        if isinstance(val, types.ModuleType):
            continue
        if isinstance(val, types.SimpleNamespace):
            continue
        mod = getattr(type(val), "__module__", "") or ""
        if "rlm_runtime.worker" in mod:
            continue
        out[key] = val
    return out


def _dump_json_subset(data: Dict[str, Any], path: str) -> bool:
    serializable: Dict[str, Any] = {}
    for k, v in data.items():
        try:
            json.dumps(v)
            serializable[k] = v
        except Exception:
            serializable[k] = repr(v)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(serializable, fh, ensure_ascii=False, indent=2)
        return True
    except Exception as exc:
        logger.warning("rlm snapshot json failed: %s", exc)
        return False


def dump_snapshot(ns: Dict[str, Any], path: str) -> bool:
    """Best-effort snapshot. JSON subset always; dill/cloudpickle when possible."""
    data = filter_user_namespace(ns)
    json_ok = _dump_json_subset(data, path + ".json")
    try:
        import dill  # type: ignore

        with open(path, "wb") as fh:
            dill.dump(data, fh)
        return True
    except Exception:
        pass
    try:
        import cloudpickle  # type: ignore

        with open(path, "wb") as fh:
            cloudpickle.dump(data, fh)
        return True
    except Exception:
        pass
    return json_ok


def load_snapshot(path: str) -> Optional[Dict[str, Any]]:
    for loader in (_load_dill, _load_cloudpickle):
        data = loader(path)
        if isinstance(data, dict):
            return data
    try:
        with open(path + ".json", "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _load_dill(path: str) -> Optional[Dict[str, Any]]:
    try:
        import dill  # type: ignore

        with open(path, "rb") as fh:
            data = dill.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _load_cloudpickle(path: str) -> Optional[Dict[str, Any]]:
    try:
        import cloudpickle  # type: ignore

        with open(path, "rb") as fh:
            data = cloudpickle.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        return None
