"""Unit tests for user-defined vector_api map layers (no hardcoded providers)."""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.storage.src.map_layer_api import (  # noqa: E402
    ApiLayerError,
    assert_safe_url,
    clear_features_cache,
    normalize_geojson,
    probe_api_layer,
    redact_source_config,
    validate_vector_api_config,
)
from services.storage.src.map_rbac import MapLayerRbac  # noqa: E402


class TestSsrfGuards:
    def test_blocks_localhost(self) -> None:
        with pytest.raises(ApiLayerError) as ei:
            assert_safe_url("http://localhost/geojson")
        assert ei.value.code == "blocked_host"

    def test_blocks_loopback_ip(self) -> None:
        with pytest.raises(ApiLayerError) as ei:
            assert_safe_url("http://127.0.0.1:8080/data")
        assert ei.value.code in {"blocked_host", "blocked_ip"}

    def test_blocks_metadata_ip_literal(self) -> None:
        with pytest.raises(ApiLayerError) as ei:
            assert_safe_url("http://169.254.169.254/latest/meta-data")
        assert ei.value.code in {"blocked_host", "blocked_metadata_ip", "blocked_ip"}

    def test_rejects_non_http(self) -> None:
        with pytest.raises(ApiLayerError) as ei:
            assert_safe_url("file:///etc/passwd")
        assert ei.value.code == "invalid_scheme"


class TestNormalizeGeojson:
    def test_feature_collection(self) -> None:
        fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [1, 2]},
                    "properties": {"a": 1},
                }
            ],
        }
        out = normalize_geojson(fc)
        assert out["type"] == "FeatureCollection"
        assert len(out["features"]) == 1

    def test_single_feature(self) -> None:
        feat = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [0, 0]},
            "properties": {},
        }
        out = normalize_geojson(feat)
        assert len(out["features"]) == 1

    def test_rejects_non_geojson(self) -> None:
        with pytest.raises(ApiLayerError) as ei:
            normalize_geojson({"hello": "world"})
        assert ei.value.code == "invalid_geojson"


class TestValidateConfig:
    def test_requires_url(self) -> None:
        with pytest.raises(ApiLayerError):
            validate_vector_api_config({"http": {"method": "GET"}})

    def test_normalizes_defaults(self) -> None:
        # Use example.com which resolves publicly; may fail DNS in airgap — skip if so
        try:
            cfg = validate_vector_api_config(
                {"http": {"url": "https://example.com/geo.json", "method": "get"}}
            )
        except ApiLayerError as exc:
            if exc.code == "dns_failed":
                pytest.skip("DNS unavailable")
            raise
        assert cfg["http"]["method"] == "GET"
        assert cfg["response"]["format"] == "geojson"
        assert cfg["auth"]["type"] == "none"


class TestRedact:
    def test_strips_secret(self) -> None:
        red = redact_source_config(
            {
                "http": {"url": "https://example.com", "headers": {"Authorization": "Bearer x"}},
                "auth": {"type": "bearer", "value_secret": "super-secret"},
            }
        )
        assert red["auth"]["value_secret"] == "***"
        assert red["http"]["headers"]["Authorization"] == "***"


class TestRegisterApiPermission:
    def test_super_admin_can_register(self) -> None:
        rbac = MapLayerRbac(user_id="u1", is_super_admin=True)
        assert rbac.can_register_api()

    def test_grant_required(self) -> None:
        rbac = MapLayerRbac(user_id="u1")
        assert not rbac.can_register_api()
        rbac2 = MapLayerRbac(
            user_id="u1",
            granted_permissions=frozenset({"map_layers:register_api"}),
        )
        assert rbac2.can_register_api()


class TestProbeLocalhostRejected:
    def test_probe_blocked_before_fetch(self) -> None:
        clear_features_cache()
        result = asyncio.run(
            probe_api_layer(
                {
                    "http": {"url": "http://127.0.0.1/geo", "method": "GET"},
                    "auth": {"type": "none"},
                    "response": {"format": "geojson"},
                }
            )
        )
        assert result["ok"] is False
        assert result["error"] in {"blocked_host", "blocked_ip", "invalid_http", "missing_url"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
