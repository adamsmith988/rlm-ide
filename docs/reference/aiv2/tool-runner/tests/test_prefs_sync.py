"""Discovery of plugin user_prefs namespaces from plugin.json manifests."""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules.plugins.discovery import build_user_pref_namespaces


def test_map_explorer_namespace_discovered():
    plugins_root = os.path.join(ROOT, "modules", "plugins")
    namespaces = build_user_pref_namespaces(plugins_root)
    by_ns = {n["namespace"]: n for n in namespaces}
    assert "map_explorer.layer_stack" in by_ns
    entry = by_ns["map_explorer.layer_stack"]
    assert entry["plugin_id"] == "map_explorer"
    assert entry["schema_version"] == 1


def test_custom_manifest_parsing(tmp_path):
    pdir = tmp_path / "demo_plugin"
    pdir.mkdir()
    (pdir / "plugin.json").write_text(
        json.dumps(
            {
                "id": "demo_plugin",
                "user_prefs": [
                    {"namespace": "filters", "schema_version": 2, "description": "x"},
                    {"namespace": "", "schema_version": 1},  # skipped (no namespace)
                    "not-an-object",  # skipped
                ],
            }
        ),
        encoding="utf-8",
    )
    out = build_user_pref_namespaces(str(tmp_path))
    assert len(out) == 1
    assert out[0]["namespace"] == "demo_plugin.filters"
    assert out[0]["schema_version"] == 2
    assert out[0]["metadata"].get("description") == "x"
