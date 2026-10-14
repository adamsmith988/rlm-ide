"""Pure-logic tests for PluginUserPrefsService helpers (no DB needed)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.storage.src.plugin_user_prefs import PluginUserPrefsService, safe_table_name, _normalize_namespace


def test_safe_table_name_basic():
    assert safe_table_name("map_explorer.layer_stack") == "user_prefs_map_explorer_layer_stack"


def test_safe_table_name_sanitizes_separators():
    assert safe_table_name("Foo.Bar-Baz") == "user_prefs_foo_bar_baz"


def test_safe_table_name_rejects_empty():
    with pytest.raises(ValueError):
        safe_table_name("")
    with pytest.raises(ValueError):
        safe_table_name("...")


def test_normalize_namespace_ok():
    assert _normalize_namespace("map_explorer.layer_stack") == "map_explorer.layer_stack"


def test_normalize_namespace_rejects_invalid():
    for bad in ("", "Bad Name", "has space", "weird!chars", ".leading"):
        with pytest.raises(ValueError):
            _normalize_namespace(bad)


def test_registry_ddl_matches_init_script():
    init_path = ROOT / "init-scripts" / "13-plugin-user-prefs-registry.sql"
    assert init_path.is_file()
    from_file = init_path.read_text(encoding="utf-8")
    from_service = PluginUserPrefsService._registry_sql()
    assert "plugin_pref_namespaces" in from_file
    assert "plugin_pref_namespaces" in from_service
    # Fallback must define the same core table when init-script is not mounted.
    assert "CREATE TABLE IF NOT EXISTS plugin_pref_namespaces" in from_service
