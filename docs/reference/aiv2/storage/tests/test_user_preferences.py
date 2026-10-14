"""Unit tests for platform user_preferences helpers (no live DB)."""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.storage.src.user_preferences import (
    DEFAULT_USER_PREFERENCES,
    normalize_nav_overlay,
    normalize_user_preferences,
    sanitize_nav_pins,
)


def test_normalize_fills_defaults():
    out = normalize_user_preferences({"locale": {"language": "fa"}})
    assert out["locale"]["language"] == "fa"
    assert out["locale"]["direction"] == "ltr"
    assert out["appearance"]["layout"] == DEFAULT_USER_PREFERENCES["appearance"]["layout"]
    assert out["nav"]["pinnedIds"] == []
    assert "documents" not in out
    assert "byWorkspace" not in out["nav"]


def test_normalize_flattens_legacy_by_workspace():
    out = normalize_nav_overlay(
        {
            "byWorkspace": {
                "g1": {"pinnedIds": ["plugin:audio_panel"], "hiddenIds": ["nav.x"]},
                "g2": {"pinnedIds": ["plugin:audio_panel", "plugin:map"], "hiddenIds": []},
            }
        }
    )
    assert out["pinnedIds"] == ["plugin:audio_panel", "plugin:map"]
    assert out["hiddenIds"] == ["nav.x"]
    assert "byWorkspace" not in out


def test_sanitize_nav_pins_filters_allowed():
    prefs = {
        "nav": {
            "pinnedIds": ["plugin:audio_panel", "plugin:evil"],
            "hiddenIds": [],
        }
    }
    out = sanitize_nav_pins(prefs, allowed_pin_ids=["plugin:audio_panel"])
    assert out["nav"]["pinnedIds"] == ["plugin:audio_panel"]
