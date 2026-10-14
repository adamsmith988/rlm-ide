import asyncio
import os
import sys

import pytest
from fastapi import HTTPException
from starlette.requests import Request

SERVICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from src import authz


def _make_request(headers: dict[str, str]) -> Request:
    raw_headers = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    scope = {"type": "http", "method": "GET", "path": "/", "headers": raw_headers}
    return Request(scope)


def test_internal_context_accepts_trusted_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(authz.settings, "INTERNAL_API_KEY", "k-test")
    monkeypatch.setattr(authz.settings, "TOOL_RUNNER_TRUSTED_SERVICES", "llm-gateway,orchestrator")
    req = _make_request(
        {
            "x-internal-key": "k-test",
            "x-service-name": "llm-gateway",
            "x-user-id": "u1",
        }
    )
    ctx = authz._internal_context(req, authz.PERMISSION_TOOLS_VIEW)
    assert ctx is not None
    assert ctx.mode == "internal"
    assert ctx.principal == "service:llm-gateway"
    assert ctx.user_id == "u1"


def test_internal_context_rejects_untrusted_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(authz.settings, "INTERNAL_API_KEY", "k-test")
    monkeypatch.setattr(authz.settings, "TOOL_RUNNER_TRUSTED_SERVICES", "orchestrator")
    req = _make_request({"x-internal-key": "k-test", "x-service-name": "llm-gateway"})
    with pytest.raises(HTTPException) as ex:
        authz._internal_context(req, authz.PERMISSION_TOOLS_VIEW)
    assert ex.value.status_code == 403


# ---------------------------------------------------------------------------
# Deny-aware tool execution semantics
# ---------------------------------------------------------------------------

def test_tool_execute_allowed_coarse_grant() -> None:
    assert not authz._tool_execute_allowed("embed_face", {"tools:execute"}, set())


def test_tool_execute_allowed_specific_grant() -> None:
    assert authz._tool_execute_allowed("embed_face", {"tools:execute:embed_face"}, set())
    assert not authz._tool_execute_allowed("other", {"tools:execute:embed_face"}, set())


def test_tool_execute_allowed_wildcard_grant() -> None:
    assert authz._tool_execute_allowed("anything", {"tools:execute:*"}, set())


def test_tool_execute_specific_deny_overrides_wildcard_grant() -> None:
    assert not authz._tool_execute_allowed(
        "embed_face", {"tools:execute:*"}, {"tools:execute:embed_face"}
    )
    assert authz._tool_execute_allowed(
        "embed_image", {"tools:execute:*"}, {"tools:execute:embed_face"}
    )


def test_tool_execute_coarse_deny_blocks_all() -> None:
    assert not authz._tool_execute_allowed("x", {"tools:execute"}, {"tools:execute"})


# ---------------------------------------------------------------------------
# Impersonation gate: internal caller on behalf of a user
# ---------------------------------------------------------------------------

def _internal_exec_request() -> Request:
    return _make_request(
        {
            "x-internal-key": "k-test",
            "x-service-name": "orchestrator",
            "x-user-id": "u1",
        }
    )


def _setup_impersonation(monkeypatch: pytest.MonkeyPatch, trusted: str = "orchestrator") -> None:
    monkeypatch.setattr(authz.settings, "INTERNAL_API_KEY", "k-test")
    monkeypatch.setattr(authz.settings, "TOOL_RUNNER_RBAC_ENABLED", True)
    monkeypatch.setattr(authz.settings, "TOOL_RUNNER_RBAC_FINE_GRAINED", True)
    monkeypatch.setattr(authz.settings, "TOOL_RUNNER_RBAC_IMPERSONATION", True)
    monkeypatch.setattr(authz.settings, "TOOL_RUNNER_TRUSTED_SERVICES", trusted)


def test_impersonation_denies_when_user_lacks_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_impersonation(monkeypatch)

    async def fake_eff(user_id, roles, group_id=None):
        return {"granted": {"tools:execute:other"}, "denied": set(), "is_super_admin": False}

    monkeypatch.setattr(authz, "_fetch_internal_effective", fake_eff)

    with pytest.raises(HTTPException) as ex:
        asyncio.run(authz.authorize_tool_execute(_internal_exec_request(), "embed_face"))
    assert ex.value.status_code == 403


def test_impersonation_allows_with_specific_grant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_impersonation(monkeypatch)

    async def fake_eff(user_id, roles, group_id=None):
        return {
            "granted": {"tools:execute:embed_face"},
            "denied": set(),
            "is_super_admin": False,
        }

    monkeypatch.setattr(authz, "_fetch_internal_effective", fake_eff)

    ctx = asyncio.run(authz.authorize_tool_execute(_internal_exec_request(), "embed_face"))
    assert ctx.allowed


def test_impersonation_denies_without_specific_or_wildcard_grant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_impersonation(monkeypatch)

    async def fake_eff(user_id, roles, group_id=None):
        return {
            "granted": {"tools:execute"},
            "denied": set(),
            "is_super_admin": False,
        }

    monkeypatch.setattr(authz, "_fetch_internal_effective", fake_eff)

    with pytest.raises(HTTPException) as ex:
        asyncio.run(authz.authorize_tool_execute(_internal_exec_request(), "embed_face"))
    assert ex.value.status_code == 403


def test_internal_batch_without_user_id_bypasses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # case-importer style call: internal key, no X-User-Id → legacy bypass.
    _setup_impersonation(monkeypatch, trusted="case-importer")

    req = _make_request({"x-internal-key": "k-test", "x-service-name": "case-importer"})
    ctx = asyncio.run(authz.authorize_tool_execute(req, "embed_face"))
    assert ctx.allowed


# ---------------------------------------------------------------------------
# Effective-permission TTL cache + shared HTTP client
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Stands in for the shared httpx.AsyncClient; records calls."""

    is_closed = False

    def __init__(self, responder) -> None:
        self._responder = responder
        self.calls: list[tuple[str, dict]] = []

    async def get(self, url, headers=None):
        self.calls.append((url, dict(headers or {})))
        return self._responder(url, headers or {})


def _install_fake_client(monkeypatch: pytest.MonkeyPatch, responder) -> _FakeClient:
    fake = _FakeClient(responder)
    monkeypatch.setattr(authz, "_get_http_client", lambda: fake)
    return fake


@pytest.fixture(autouse=True)
def _clear_effective_cache():
    authz.reset_effective_cache()
    yield
    authz.reset_effective_cache()


def _ok_payload(url, headers):
    return _FakeResponse(
        {
            "permissions": ["tools:execute:embed_face"],
            "denied_permissions": [],
            "is_super_admin": False,
            "groups": {"g1": {"permissions": ["tools:execute:group_only"]}},
        }
    )


def test_effective_cache_hit_within_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(authz.settings, "TOOL_RUNNER_RBAC_EFFECTIVE_CACHE_TTL_SEC", 10.0)
    fake = _install_fake_client(monkeypatch, _ok_payload)

    first = asyncio.run(authz._fetch_internal_effective("u1", "admin", group_id=None))
    second = asyncio.run(authz._fetch_internal_effective("u1", "admin", group_id=None))

    assert len(fake.calls) == 1
    assert first is second
    assert "tools:execute:embed_face" in first["granted"]


def test_effective_cache_key_includes_roles_and_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(authz.settings, "TOOL_RUNNER_RBAC_EFFECTIVE_CACHE_TTL_SEC", 10.0)
    fake = _install_fake_client(monkeypatch, _ok_payload)

    base = asyncio.run(authz._fetch_internal_effective("u1", "admin", group_id=None))
    other_roles = asyncio.run(authz._fetch_internal_effective("u1", "viewer", group_id=None))
    with_group = asyncio.run(authz._fetch_internal_effective("u1", "admin", group_id="g1"))
    other_user = asyncio.run(authz._fetch_internal_effective("u2", "admin", group_id=None))

    assert len(fake.calls) == 4
    assert "tools:execute:group_only" not in base["granted"]
    assert "tools:execute:group_only" in with_group["granted"]
    assert other_roles is not base and other_user is not base


def test_effective_cache_expires_after_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(authz.settings, "TOOL_RUNNER_RBAC_EFFECTIVE_CACHE_TTL_SEC", 5.0)
    fake = _install_fake_client(monkeypatch, _ok_payload)

    clock = {"now": 1000.0}
    monkeypatch.setattr(authz.time, "monotonic", lambda: clock["now"])

    asyncio.run(authz._fetch_internal_effective("u1", None, group_id=None))
    clock["now"] += 4.9
    asyncio.run(authz._fetch_internal_effective("u1", None, group_id=None))
    assert len(fake.calls) == 1

    clock["now"] += 0.2  # past the 5s TTL
    asyncio.run(authz._fetch_internal_effective("u1", None, group_id=None))
    assert len(fake.calls) == 2


def test_effective_cache_disabled_when_ttl_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(authz.settings, "TOOL_RUNNER_RBAC_EFFECTIVE_CACHE_TTL_SEC", 0)
    fake = _install_fake_client(monkeypatch, _ok_payload)

    asyncio.run(authz._fetch_internal_effective("u1", None, group_id=None))
    asyncio.run(authz._fetch_internal_effective("u1", None, group_id=None))
    assert len(fake.calls) == 2


def test_effective_cache_does_not_cache_failures_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_impersonation(monkeypatch)
    monkeypatch.setattr(authz.settings, "TOOL_RUNNER_RBAC_FAIL_OPEN", False)
    monkeypatch.setattr(authz.settings, "TOOL_RUNNER_RBAC_EFFECTIVE_CACHE_TTL_SEC", 10.0)

    state = {"fail": True}

    def responder(url, headers):
        if state["fail"]:
            raise ConnectionError("auth-service down")
        return _ok_payload(url, headers)

    fake = _install_fake_client(monkeypatch, responder)

    # Cache miss + transport failure → 503 (fail-closed), nothing cached.
    with pytest.raises(HTTPException) as ex:
        asyncio.run(authz.authorize_tool_execute(_internal_exec_request(), "embed_face"))
    assert ex.value.status_code == 503
    assert len(fake.calls) == 1

    # Auth-service recovers → next call refetches (failure was not cached).
    state["fail"] = False
    ctx = asyncio.run(authz.authorize_tool_execute(_internal_exec_request(), "embed_face"))
    assert ctx.allowed
    assert len(fake.calls) == 2

    # Now served from cache even though auth-service is down again.
    state["fail"] = True
    ctx = asyncio.run(authz.authorize_tool_execute(_internal_exec_request(), "embed_face"))
    assert ctx.allowed
    assert len(fake.calls) == 2


def test_effective_cache_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(authz.settings, "TOOL_RUNNER_RBAC_EFFECTIVE_CACHE_TTL_SEC", 10.0)
    monkeypatch.setattr(authz.settings, "TOOL_RUNNER_RBAC_EFFECTIVE_CACHE_MAX_ENTRIES", 3)
    _install_fake_client(monkeypatch, _ok_payload)

    for uid in ("a", "b", "c", "d"):
        asyncio.run(authz._fetch_internal_effective(uid, None, group_id=None))

    assert len(authz._effective_cache) <= 3


def test_shared_http_client_is_reused_and_closable() -> None:
    asyncio.run(authz.aclose_http_client())
    first = authz._get_http_client()
    second = authz._get_http_client()
    assert first is second
    assert not first.is_closed

    asyncio.run(authz.aclose_http_client())
    assert first.is_closed
    third = authz._get_http_client()
    assert third is not first
    asyncio.run(authz.aclose_http_client())

