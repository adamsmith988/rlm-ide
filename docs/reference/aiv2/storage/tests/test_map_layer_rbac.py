"""Unit tests for map layer RBAC resolution (auth-service as source of truth)."""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.storage.src.map_rbac import (  # noqa: E402
    MapLayerRbac,
    layer_ids_from_permissions,
)


class TestLayerIdsFromPermissions:
    def test_extracts_map_layer_uuids(self) -> None:
        perms = {"map_layer:abc", "chat:read", "map_layer:def"}
        assert layer_ids_from_permissions(perms) == {"abc", "def"}


class TestMapLayerRbac:
    def test_super_admin_reads_any(self) -> None:
        rbac = MapLayerRbac(user_id="u1", is_super_admin=True)
        assert rbac.can_read_layer("any-id", owner_user_id="other")

    def test_owner_reads_without_grant(self) -> None:
        rbac = MapLayerRbac(user_id="u1")
        assert rbac.can_read_layer("lid", owner_user_id="u1")

    def test_grant_allows_non_owner(self) -> None:
        rbac = MapLayerRbac(user_id="u1", granted_layer_ids=frozenset({"lid"}))
        assert rbac.can_read_layer("lid", owner_user_id="other")

    def test_deny_wins_over_grant(self) -> None:
        rbac = MapLayerRbac(
            user_id="u1",
            granted_layer_ids=frozenset({"lid"}),
            denied_layer_ids=frozenset({"lid"}),
        )
        assert not rbac.can_read_layer("lid", owner_user_id="other")

    def test_grant_does_not_allow_mutate(self) -> None:
        rbac = MapLayerRbac(user_id="u1", granted_layer_ids=frozenset({"lid"}))
        assert not rbac.can_mutate_layer("lid", owner_user_id="other")
        assert rbac.can_mutate_layer("lid", owner_user_id="u1")

    def test_register_api_permission(self) -> None:
        assert not MapLayerRbac(user_id="u1").can_register_api()
        assert MapLayerRbac(user_id="u1", is_super_admin=True).can_register_api()
        assert MapLayerRbac(
            user_id="u1",
            granted_permissions=frozenset({"map_layers:register_api"}),
        ).can_register_api()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
