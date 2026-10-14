import asyncio
import os
import sys

SERVICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from src.auth import permissions_client as pc
from src.auth.permissions_client import (
    UserPermissions,
    filter_tools_by_permission,
    permission_matches,
    resolve_allowed_tools,
)


def _tools(*names):
    return [{"name": n} for n in names]


def test_permission_matches_wildcard():
    assert permission_matches("tools:execute:embed_face", {"tools:execute:*"})
    assert not permission_matches("tools:execute:embed_face", {"tools:view"})


def test_can_use_tools_gate():
    chat_only = UserPermissions(user_id="u", permissions={"chat:read", "chat:write"})
    assert not chat_only.can_use_tools()
    full = UserPermissions(user_id="u", permissions={"chat:use_tools", "tools:execute"})
    assert full.can_use_tools()


def test_can_use_tools_denied():
    user = UserPermissions(
        user_id="u",
        permissions={"chat:use_tools"},
        denied={"chat:use_tools"},
    )
    assert not user.can_use_tools()


def test_can_execute_tool_coarse_and_deny():
    user = UserPermissions(
        user_id="u",
        permissions={"tools:execute"},
        denied={"tools:execute:embed_face"},
    )
    assert not user.can_execute_tool("embed_image")
    assert not user.can_execute_tool("embed_face")


def test_can_execute_tool_wildcard_and_specific():
    wildcard = UserPermissions(user_id="u", permissions={"tools:execute:*"})
    assert wildcard.can_execute_tool("embed_image")
    specific = UserPermissions(user_id="u", permissions={"tools:execute:embed_face"})
    assert specific.can_execute_tool("embed_face")
    assert not specific.can_execute_tool("embed_image")


def test_filter_tools_specific_grant():
    perms = UserPermissions(user_id="u", permissions={"tools:execute:embed_face"})
    out = filter_tools_by_permission(_tools("embed_face", "embed_image"), perms)
    assert [t["name"] for t in out] == ["embed_face"]


def test_filter_tools_coarse_grant_does_not_pass_all():
    perms = UserPermissions(user_id="u", permissions={"tools:execute"})
    out = filter_tools_by_permission(_tools("a", "b", "c"), perms)
    assert out == []


def test_filter_tools_wildcard_grant_passes_all():
    perms = UserPermissions(user_id="u", permissions={"tools:execute:*"})
    out = filter_tools_by_permission(_tools("a", "b", "c"), perms)
    assert len(out) == 3


def test_resolve_allowed_tools_chat_only(monkeypatch):
    monkeypatch.setattr(pc.settings, "TOOL_PERMISSION_ENFORCEMENT", True)

    async def fake_perms(user_id, roles=None, force_refresh=False):
        return UserPermissions(user_id=user_id, permissions={"chat:read"})

    monkeypatch.setattr(pc, "get_user_permissions", fake_perms)
    out = asyncio.run(resolve_allowed_tools(_tools("a", "b"), "u1"))
    assert out == []


def test_resolve_allowed_tools_filters(monkeypatch):
    monkeypatch.setattr(pc.settings, "TOOL_PERMISSION_ENFORCEMENT", True)

    async def fake_perms(user_id, roles=None, force_refresh=False):
        return UserPermissions(
            user_id=user_id,
            permissions={"chat:use_tools", "tools:execute:embed_image"},
            denied={"tools:execute:embed_face"},
        )

    monkeypatch.setattr(pc, "get_user_permissions", fake_perms)
    out = asyncio.run(resolve_allowed_tools(_tools("embed_face", "embed_image"), "u1"))
    assert [t["name"] for t in out] == ["embed_image"]


def test_resolve_allowed_tools_enforcement_disabled(monkeypatch):
    monkeypatch.setattr(pc.settings, "TOOL_PERMISSION_ENFORCEMENT", False)
    out = asyncio.run(resolve_allowed_tools(_tools("a", "b"), "u1"))
    assert len(out) == 2
