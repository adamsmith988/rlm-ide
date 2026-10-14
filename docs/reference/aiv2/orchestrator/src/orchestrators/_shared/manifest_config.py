"""Resolve an engine's runtime config from its manifest.

Defaults live in the manifest (``settings_ui[*].default`` and
``config_schema.properties[*].default``) so the UI, the HTTP discovery API and
the runtime read one source. User values from ``orchestrator_config`` win and
are coerced to the declared type; unknown keys are kept for pass-through.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set

from ..contract import Manifest, normalize_manifest, validate_manifest


@lru_cache(maxsize=None)
def _load_manifest_file(path: str) -> Manifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_manifest(normalize_manifest(raw))


def package_manifest(module_file: str) -> Manifest:
    """Manifest of the drop-in package containing ``module_file``."""
    here = Path(module_file).resolve().parent
    for folder in (here, *here.parents):
        candidate = folder / "manifest.json"
        if candidate.is_file():
            return _load_manifest_file(str(candidate))
        if folder.name == "orchestrators":
            break
    raise FileNotFoundError(f"no manifest.json above {module_file}")

_INT_TYPES = {"int", "integer"}
_FLOAT_TYPES = {"float", "number"}
_BOOL_TYPES = {"boolean", "bool"}
_OBJECT_TYPES = {"object", "json"}


def _declared_types(manifest: Manifest) -> Dict[str, str]:
    out: Dict[str, str] = {}
    props = (manifest.config_schema or {}).get("properties") or {}
    if isinstance(props, Mapping):
        for name, spec in props.items():
            if isinstance(spec, Mapping) and spec.get("type"):
                out[str(name)] = str(spec.get("type"))
    for item in manifest.settings_ui or []:
        name = str(item.get("name") or "").strip()
        if name and item.get("type"):
            out[name] = str(item.get("type"))
    return out


def manifest_defaults(manifest: Manifest) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    props = (manifest.config_schema or {}).get("properties") or {}
    if isinstance(props, Mapping):
        for name, spec in props.items():
            if isinstance(spec, Mapping) and "default" in spec:
                out[str(name)] = spec["default"]
    for item in manifest.settings_ui or []:
        name = str(item.get("name") or "").strip()
        if name and "default" in item:
            out[name] = item["default"]
    return out


def own_config_keys(manifest: Manifest) -> Set[str]:
    """Keys this engine consumes (everything else may be forwarded to inner engines)."""
    return set(manifest_defaults(manifest)) | set(_declared_types(manifest))


def _coerce(value: Any, type_name: str, default: Any) -> Any:
    if value is None:
        return default
    try:
        if type_name in _INT_TYPES:
            return int(value)
        if type_name in _FLOAT_TYPES:
            return float(value)
        if type_name in _BOOL_TYPES:
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if type_name in _OBJECT_TYPES:
            if isinstance(value, str):
                return json.loads(value) if value.strip() else default
            return value
        if type_name == "array":
            return list(value) if isinstance(value, (list, tuple)) else default
    except (TypeError, ValueError):
        return default
    return value


def resolve_config(manifest: Manifest, user_cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    defaults = manifest_defaults(manifest)
    types = _declared_types(manifest)
    out: Dict[str, Any] = dict(defaults)
    for key, value in dict(user_cfg or {}).items():
        out[key] = _coerce(value, types.get(key, ""), defaults.get(key))
    return out


def passthrough_config(manifest: Manifest, user_cfg: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """User config minus this engine's own keys (model stamps, tool_ids, …)."""
    own = own_config_keys(manifest)
    return {k: v for k, v in dict(user_cfg or {}).items() if k not in own}
