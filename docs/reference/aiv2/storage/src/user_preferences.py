"""Platform ``user_preferences`` namespace sync + gateway-facing helpers.

Registers ``platform.user_preferences`` with Storage (same dynamic table pattern
as plugin prefs). Document shape is UI-only:

  locale / appearance / nav / uiFlags

``nav`` is a single personal sidebar overlay (pins/hide/order/shortcuts) — not
scoped to workspace/group. Legacy ``nav.byWorkspace`` is flattened on read.

Sensitive client work (drafts, topology, report docs) is Phase C — not here.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

PLATFORM_PLUGIN_ID = "platform"
PLATFORM_USER_PREFS_NAMESPACE = "platform.user_preferences"
PLATFORM_USER_PREFS_SCHEMA_VERSION = 1

EMPTY_NAV_OVERLAY: Dict[str, Any] = {
    "schemaVersion": 1,
    "pinnedIds": [],
    "hiddenIds": [],
    "orderOverrides": {},
    "shortcuts": {},
}

DEFAULT_USER_PREFERENCES: Dict[str, Any] = {
    "locale": {"language": "en", "direction": "ltr"},
    "appearance": {
        "theme": "light",
        "colorPreset": "blue",
        "layout": "hydrogen",
        "berylliumSidebarExpanded": True,
    },
    "nav": copy.deepcopy(EMPTY_NAV_OVERLAY),
    "uiFlags": {"onboarding": {}, "lab": {}},
}


def deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge; patch wins for scalars/lists."""
    out = copy.deepcopy(base) if isinstance(base, dict) else {}
    if not isinstance(patch, dict):
        return out
    for key, value in patch.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _as_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(x) for x in value if str(x).strip()]


def normalize_nav_overlay(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize a nav overlay; flatten legacy ``byWorkspace`` if present."""
    if not isinstance(raw, dict):
        return copy.deepcopy(EMPTY_NAV_OVERLAY)

    # Flat personal overlay (current shape).
    if any(k in raw for k in ("pinnedIds", "hiddenIds", "orderOverrides", "shortcuts")):
        return {
            "schemaVersion": 1,
            "pinnedIds": _as_str_list(raw.get("pinnedIds")),
            "hiddenIds": _as_str_list(raw.get("hiddenIds")),
            "orderOverrides": (
                {str(k): int(v) for k, v in raw["orderOverrides"].items()}
                if isinstance(raw.get("orderOverrides"), dict)
                else {}
            ),
            "shortcuts": (
                {str(k): str(v) for k, v in raw["shortcuts"].items() if str(v).strip()}
                if isinstance(raw.get("shortcuts"), dict)
                else {}
            ),
        }

    # Legacy workspace-scoped overlays → one personal overlay.
    by_ws = raw.get("byWorkspace")
    if isinstance(by_ws, dict) and by_ws:
        merged = copy.deepcopy(EMPTY_NAV_OVERLAY)
        order: Dict[str, int] = {}
        shortcuts: Dict[str, str] = {}
        for overlay in by_ws.values():
            if not isinstance(overlay, dict):
                continue
            for pid in _as_str_list(overlay.get("pinnedIds")):
                if pid not in merged["pinnedIds"]:
                    merged["pinnedIds"].append(pid)
            for hid in _as_str_list(overlay.get("hiddenIds")):
                if hid not in merged["hiddenIds"]:
                    merged["hiddenIds"].append(hid)
            if isinstance(overlay.get("orderOverrides"), dict):
                order.update({str(k): int(v) for k, v in overlay["orderOverrides"].items()})
            if isinstance(overlay.get("shortcuts"), dict):
                shortcuts.update(
                    {str(k): str(v) for k, v in overlay["shortcuts"].items() if str(v).strip()}
                )
        merged["orderOverrides"] = order
        merged["shortcuts"] = shortcuts
        return merged

    return copy.deepcopy(EMPTY_NAV_OVERLAY)


def normalize_user_preferences(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge stored prefs onto defaults and drop unknown top-level keys."""
    merged = deep_merge(DEFAULT_USER_PREFERENCES, raw if isinstance(raw, dict) else {})
    return {
        "locale": merged.get("locale")
        if isinstance(merged.get("locale"), dict)
        else copy.deepcopy(DEFAULT_USER_PREFERENCES["locale"]),
        "appearance": merged.get("appearance")
        if isinstance(merged.get("appearance"), dict)
        else copy.deepcopy(DEFAULT_USER_PREFERENCES["appearance"]),
        "nav": normalize_nav_overlay(merged.get("nav") if isinstance(merged.get("nav"), dict) else {}),
        "uiFlags": merged.get("uiFlags")
        if isinstance(merged.get("uiFlags"), dict)
        else copy.deepcopy(DEFAULT_USER_PREFERENCES["uiFlags"]),
    }


def sanitize_nav_pins(
    prefs: Dict[str, Any],
    *,
    allowed_pin_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Keep only pins that are in the caller's allowed catalog (when provided).

    ``allowed_pin_ids`` is typically ``plugin:{plugin_id}`` plus any non-plugin
    catalog ids the FE may pin. When ``None``, pins are left as-is (caller has
    not supplied a ceiling yet).
    """
    out = normalize_user_preferences(prefs)
    nav = normalize_nav_overlay(out.get("nav") if isinstance(out.get("nav"), dict) else {})
    if allowed_pin_ids is None:
        out["nav"] = nav
        return out

    allowed: Set[str] = {str(x).strip() for x in allowed_pin_ids if str(x).strip()}
    out["nav"] = {
        **nav,
        "pinnedIds": [p for p in nav.get("pinnedIds") or [] if str(p) in allowed],
        "hiddenIds": [h for h in (nav.get("hiddenIds") or []) if isinstance(h, str)],
    }
    return out
