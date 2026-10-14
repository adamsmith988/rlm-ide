"""Unit tests for multi-token llm_endpoint_credentials storage (fake asyncpg pool)."""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import sys
import uuid

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.storage.src.backends.postgres import (  # noqa: E402
    LlmCredentialTokenExists,
    LlmEndpointNotFound,
    PostgresBackend,
)

EP_ID = "11111111-1111-1111-1111-111111111111"
CRED_ID = "22222222-2222-2222-2222-222222222222"


class UniqueViolationError(Exception):
    """Name-compatible stand-in for asyncpg's unique violation."""


class ForeignKeyViolationError(Exception):
    """Name-compatible stand-in for asyncpg's FK violation."""


class _NullCtx:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


class FakeConn:
    """Records queries and replays queued results in call order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple]] = []
        self.fetch_results: list = []
        self.fetchrow_results: list = []
        self.execute_results: list = []
        self.raise_next: Exception | None = None

    def _maybe_raise(self) -> None:
        if self.raise_next is not None:
            exc, self.raise_next = self.raise_next, None
            raise exc

    async def fetch(self, query, *args):
        self.calls.append(("fetch", query, args))
        self._maybe_raise()
        return self.fetch_results.pop(0) if self.fetch_results else []

    async def fetchrow(self, query, *args):
        self.calls.append(("fetchrow", query, args))
        self._maybe_raise()
        return self.fetchrow_results.pop(0) if self.fetchrow_results else None

    async def execute(self, query, *args):
        self.calls.append(("execute", query, args))
        self._maybe_raise()
        return self.execute_results.pop(0) if self.execute_results else "DELETE 0"

    def transaction(self):
        return _NullCtx()


class _ConnCtx:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *exc):
        return False


class FakePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _ConnCtx(self.conn)


def make_backend() -> tuple[PostgresBackend, FakeConn]:
    backend = PostgresBackend(database_url=None)
    conn = FakeConn()
    backend.pool = FakePool(conn)
    return backend, conn


def cred_row(**over) -> dict:
    row = {
        "id": uuid.UUID(CRED_ID),
        "endpoint_id": uuid.UUID(EP_ID),
        "label": "default",
        "token": "sk-abc",
        "scope": "model",
        "logical_id": "gemma4-31",
        "upstream_model_id": "gemma4-31",
        "priority": 1000,
        "weight": 1,
        "requests_per_minute": None,
        "is_active": True,
        "last_probe": "{}",
        "created_at": dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
        "updated_at": dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
    }
    row.update(over)
    return row


class TestSchema:
    def test_declares_table_and_indexes(self) -> None:
        from services.storage.src.schemas import CORE_SCHEMA

        assert "CREATE TABLE IF NOT EXISTS llm_endpoint_credentials" in CORE_SCHEMA
        assert "REFERENCES llm_external_endpoints(id) ON DELETE CASCADE" in CORE_SCHEMA
        assert "idx_llm_endpoint_credentials_ep" in CORE_SCHEMA
        assert "DROP INDEX IF EXISTS idx_llm_endpoint_credentials_ep_label" in CORE_SCHEMA
        assert "DROP INDEX IF EXISTS idx_llm_endpoint_credentials_ep_label_scope" in CORE_SCHEMA
        assert (
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_llm_endpoint_credentials_ep_model_token"
            in CORE_SCHEMA
        )
        assert "lower(COALESCE(NULLIF(btrim(logical_id), ''), ''))" in CORE_SCHEMA
        assert ",\n    token\n  )" in CORE_SCHEMA or "token\n  )" in CORE_SCHEMA
        assert "requests_per_minute INTEGER" in CORE_SCHEMA
        assert "ADD COLUMN IF NOT EXISTS requests_per_minute" in CORE_SCHEMA

    def test_endpoint_name_uniqueness_untouched(self) -> None:
        from services.storage.src.schemas import CORE_SCHEMA

        assert "name TEXT NOT NULL UNIQUE" in CORE_SCHEMA


class TestList:
    def test_scopes_to_endpoint_and_normalizes_row(self) -> None:
        backend, conn = make_backend()
        conn.fetch_results.append(
            [cred_row(label="alt", last_probe='{"ok": true}')]
        )
        rows = asyncio.run(backend.llm_endpoint_credentials_list(EP_ID))

        assert len(rows) == 1
        assert rows[0]["id"] == CRED_ID
        assert rows[0]["endpoint_id"] == EP_ID
        assert rows[0]["last_probe"] == {"ok": True}
        _, query, args = conn.calls[0]
        assert "WHERE endpoint_id = $1" in query
        assert "ORDER BY priority ASC, lower(label) ASC" in query
        assert args == (uuid.UUID(EP_ID),)

    def test_invalid_endpoint_id_returns_empty(self) -> None:
        backend, conn = make_backend()
        assert asyncio.run(backend.llm_endpoint_credentials_list("not-a-uuid")) == []
        assert conn.calls == []

    def test_list_all_keeps_endpoint_id(self) -> None:
        backend, conn = make_backend()
        conn.fetch_results.append([cred_row(), cred_row(label="alt")])
        rows = asyncio.run(backend.llm_endpoint_credentials_list_all())

        assert [r["endpoint_id"] for r in rows] == [EP_ID, EP_ID]
        _, query, args = conn.calls[0]
        assert "WHERE" not in query
        assert "ORDER BY endpoint_id, priority ASC, lower(label) ASC" in query
        assert args == ()


class TestGet:
    def test_matches_endpoint_and_credential(self) -> None:
        backend, conn = make_backend()
        conn.fetchrow_results.append(cred_row())
        row = asyncio.run(backend.llm_endpoint_credential_get(EP_ID, CRED_ID))

        assert row is not None and row["token"] == "sk-abc"
        _, query, args = conn.calls[0]
        assert "WHERE endpoint_id = $1 AND id = $2" in query
        assert args == (uuid.UUID(EP_ID), uuid.UUID(CRED_ID))

    def test_missing_returns_none(self) -> None:
        backend, _ = make_backend()
        assert asyncio.run(backend.llm_endpoint_credential_get(EP_ID, CRED_ID)) is None


class TestCreate:
    def test_normalizes_scope_label_and_blank_text(self) -> None:
        backend, conn = make_backend()
        conn.fetchrow_results.append(cred_row(scope="model", logical_id="qwen-chat"))
        row = asyncio.run(
            backend.llm_endpoint_credential_create(
                EP_ID,
                label="  ",
                token="sk-2",
                scope="BOGUS",
                logical_id="qwen-chat",
                upstream_model_id="   ",
                priority="5",
                weight="2",
            )
        )

        assert row["logical_id"] == "qwen-chat"
        _, query, args = conn.calls[0]
        assert "INSERT INTO llm_endpoint_credentials" in query
        assert args[0] == uuid.UUID(EP_ID)
        assert args[1] == "default"  # blank label falls back
        assert args[2] == "sk-2"
        assert args[3] == "model"  # credentials are always model-scoped
        assert args[4] == "qwen-chat"
        assert args[5] == "qwen-chat"  # blank upstream falls back to logical_id
        assert args[6] == 5 and args[7] == 2
        assert args[8] is None  # requests_per_minute default
        assert args[9] is True
        assert args[10] == "{}"

    def test_create_persists_requests_per_minute(self) -> None:
        backend, conn = make_backend()
        conn.fetchrow_results.append(cred_row(requests_per_minute=20))
        asyncio.run(
            backend.llm_endpoint_credential_create(
                EP_ID,
                label="t",
                token="sk-rpm",
                logical_id="qwen-chat",
                requests_per_minute=20,
            )
        )
        assert conn.calls[0][2][8] == 20

    def test_model_scope_is_preserved(self) -> None:
        backend, conn = make_backend()
        conn.fetchrow_results.append(cred_row(scope="model"))
        asyncio.run(
            backend.llm_endpoint_credential_create(
                EP_ID, label="gpu-a", token="sk-3", scope="Model", logical_id="gemma4-31"
            )
        )
        assert conn.calls[0][2][3] == "model"

    def test_requires_logical_or_upstream(self) -> None:
        backend, conn = make_backend()
        with pytest.raises(ValueError, match="model_scope_requires"):
            asyncio.run(
                backend.llm_endpoint_credential_create(EP_ID, label="a", token="sk")
            )
        assert conn.calls == []

    def test_token_conflict_raises(self) -> None:
        backend, conn = make_backend()
        conn.raise_next = UniqueViolationError("dup")
        with pytest.raises(LlmCredentialTokenExists) as ei:
            asyncio.run(
                backend.llm_endpoint_credential_create(
                    EP_ID, label="alt", token="sk", logical_id="gemma4-31"
                )
            )
        assert "token_tail" in str(ei.value)

    def test_unknown_endpoint_raises_not_found(self) -> None:
        backend, conn = make_backend()
        conn.raise_next = ForeignKeyViolationError("fk")
        with pytest.raises(LlmEndpointNotFound):
            asyncio.run(
                backend.llm_endpoint_credential_create(
                    EP_ID, label="alt", token="sk", logical_id="gemma4-31"
                )
            )

    def test_invalid_endpoint_id_raises_not_found(self) -> None:
        backend, conn = make_backend()
        with pytest.raises(LlmEndpointNotFound):
            asyncio.run(
                backend.llm_endpoint_credential_create(
                    "nope", label="a", token="sk", logical_id="gemma4-31"
                )
            )
        assert conn.calls == []


class TestUpdate:
    def test_patches_only_supplied_fields(self) -> None:
        backend, conn = make_backend()
        conn.fetchrow_results.append(cred_row(token="sk-new", is_active=False))
        row = asyncio.run(
            backend.llm_endpoint_credential_update(
                EP_ID, CRED_ID, token="sk-new", is_active=False
            )
        )

        assert row is not None and row["token"] == "sk-new"
        _, query, args = conn.calls[0]
        set_clause = query.split("SET", 1)[1].split("WHERE", 1)[0]
        assert "token = $1" in set_clause and "is_active = $2" in set_clause
        assert "label" not in set_clause and "priority" not in set_clause
        assert "updated_at = now()" in set_clause
        assert "WHERE endpoint_id = $3 AND id = $4" in query
        assert args == ("sk-new", False, uuid.UUID(EP_ID), uuid.UUID(CRED_ID))

    def test_patches_requests_per_minute(self) -> None:
        backend, conn = make_backend()
        conn.fetchrow_results.append(cred_row(requests_per_minute=20))
        asyncio.run(
            backend.llm_endpoint_credential_update(
                EP_ID, CRED_ID, requests_per_minute=20
            )
        )
        _, query, args = conn.calls[0]
        assert "requests_per_minute = $1" in query
        assert args[0] == 20

    def test_zero_rpm_clears_column(self) -> None:
        backend, conn = make_backend()
        conn.fetchrow_results.append(cred_row(requests_per_minute=None))
        asyncio.run(
            backend.llm_endpoint_credential_update(
                EP_ID, CRED_ID, requests_per_minute=0
            )
        )
        assert conn.calls[0][2][0] is None

    def test_blank_scope_field_clears_column(self) -> None:
        backend, conn = make_backend()
        conn.fetchrow_results.append(cred_row())
        asyncio.run(
            backend.llm_endpoint_credential_update(EP_ID, CRED_ID, logical_id="")
        )
        assert conn.calls[0][2][0] is None

    def test_blank_label_rejected(self) -> None:
        backend, conn = make_backend()
        with pytest.raises(ValueError):
            asyncio.run(
                backend.llm_endpoint_credential_update(EP_ID, CRED_ID, label="   ")
            )
        assert conn.calls == []

    def test_empty_patch_reads_current_row(self) -> None:
        backend, conn = make_backend()
        conn.fetchrow_results.append(cred_row())
        row = asyncio.run(backend.llm_endpoint_credential_update(EP_ID, CRED_ID))

        assert row is not None and row["id"] == CRED_ID
        assert "SELECT" in conn.calls[0][1]

    def test_token_conflict_raises(self) -> None:
        backend, conn = make_backend()
        conn.raise_next = UniqueViolationError("dup")
        with pytest.raises(LlmCredentialTokenExists):
            asyncio.run(
                backend.llm_endpoint_credential_update(EP_ID, CRED_ID, token="taken")
            )

    def test_missing_row_returns_none(self) -> None:
        backend, _ = make_backend()
        assert (
            asyncio.run(
                backend.llm_endpoint_credential_update(EP_ID, CRED_ID, token="x")
            )
            is None
        )


class TestSetRpmForModel:
    def test_updates_matching_logical_id(self) -> None:
        backend, conn = make_backend()
        conn.fetch_results.append([cred_row(requests_per_minute=20)])
        rows = asyncio.run(
            backend.llm_endpoint_credentials_set_rpm_for_model(
                EP_ID,
                requests_per_minute=20,
                logical_id="gemma4-31",
            )
        )
        assert len(rows) == 1
        assert rows[0]["requests_per_minute"] == 20
        _, query, args = conn.calls[0]
        assert "UPDATE llm_endpoint_credentials" in query
        assert "requests_per_minute = $2" in query
        assert "lower(coalesce(logical_id, '')) = lower($3)" in query
        assert args[0] == uuid.UUID(EP_ID)
        assert args[1] == 20
        assert args[2] == "gemma4-31"

    def test_requires_logical_or_upstream(self) -> None:
        backend, conn = make_backend()
        with pytest.raises(ValueError, match="logical_id_or_upstream_required"):
            asyncio.run(
                backend.llm_endpoint_credentials_set_rpm_for_model(
                    EP_ID, requests_per_minute=10
                )
            )
        assert conn.calls == []

    def test_zero_clears(self) -> None:
        backend, conn = make_backend()
        conn.fetch_results.append([cred_row(requests_per_minute=None)])
        asyncio.run(
            backend.llm_endpoint_credentials_set_rpm_for_model(
                EP_ID,
                requests_per_minute=0,
                logical_id="gemma4-31",
            )
        )
        assert conn.calls[0][2][1] is None


class TestDelete:
    def test_true_when_row_removed(self) -> None:
        backend, conn = make_backend()
        conn.execute_results.append("DELETE 1")
        assert asyncio.run(backend.llm_endpoint_credential_delete(EP_ID, CRED_ID)) is True
        assert "WHERE endpoint_id = $1 AND id = $2" in conn.calls[0][1]

    def test_false_when_nothing_removed(self) -> None:
        backend, conn = make_backend()
        conn.execute_results.append("DELETE 0")
        assert asyncio.run(backend.llm_endpoint_credential_delete(EP_ID, CRED_ID)) is False

    def test_false_for_invalid_ids(self) -> None:
        backend, conn = make_backend()
        assert asyncio.run(backend.llm_endpoint_credential_delete("x", "y")) is False
        assert conn.calls == []


class TestEnsureDefaultFromAuth:
    def test_purges_endpoint_scoped_rows(self) -> None:
        backend, conn = make_backend()
        conn.execute_results.append("DELETE 2")
        row = asyncio.run(
            backend.llm_endpoint_credentials_ensure_default_from_auth(EP_ID)
        )
        assert row is None
        _, query, args = conn.calls[0]
        assert "DELETE FROM llm_endpoint_credentials" in query
        assert args == (uuid.UUID(EP_ID),)

    def test_none_for_invalid_endpoint(self) -> None:
        backend, conn = make_backend()
        assert (
            asyncio.run(backend.llm_endpoint_credentials_ensure_default_from_auth("nope"))
            is None
        )
        assert conn.calls == []


class StubStorage:
    """Minimal Storage stand-in for route-level assertions."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.create_error: Exception | None = None
        self.rows: list[dict] = []
        self.row: dict | None = None
        self.deleted = True
        self.ensured: dict | None = None

    async def llm_endpoint_credentials_list(self, endpoint_id):
        self.calls.append(("list", endpoint_id))
        return self.rows

    async def llm_endpoint_credentials_list_all(self):
        self.calls.append(("list_all",))
        return self.rows

    async def llm_endpoint_credential_get(self, endpoint_id, cred_id):
        self.calls.append(("get", endpoint_id, cred_id))
        return self.row

    async def llm_endpoint_credential_create(self, endpoint_id, **kwargs):
        self.calls.append(("create", endpoint_id, kwargs))
        if self.create_error is not None:
            raise self.create_error
        return self.row

    async def llm_endpoint_credential_update(self, endpoint_id, cred_id, **kwargs):
        self.calls.append(("update", endpoint_id, cred_id, kwargs))
        return self.row

    async def llm_endpoint_credential_delete(self, endpoint_id, cred_id):
        self.calls.append(("delete", endpoint_id, cred_id))
        return self.deleted

    async def llm_endpoint_credentials_ensure_default_from_auth(self, endpoint_id):
        self.calls.append(("ensure", endpoint_id))
        return self.ensured


@pytest.fixture()
def client_and_stub():
    from fastapi.testclient import TestClient

    from services.storage.src.config import settings
    from services.storage.src.main import app

    stub = StubStorage()
    previous = getattr(app.state, "storage", None)
    app.state.storage = stub
    # Storage sits behind a global internal-service gate; credential routes are
    # covered by it like every other /llm route.
    headers = {
        "X-Internal-Key": settings.INTERNAL_API_KEY,
        "X-Service-Name": "api-gateway",
    }
    try:
        yield TestClient(app, headers=headers), stub
    finally:
        app.state.storage = previous


def test_credential_routes_require_internal_key() -> None:
    from fastapi.testclient import TestClient

    from services.storage.src.config import settings
    from services.storage.src.main import app

    if not settings.STORAGE_RBAC_ENABLED:
        pytest.skip("storage RBAC gate disabled in this environment")
    client = TestClient(app)
    assert client.get(f"/llm/endpoints/{EP_ID}/credentials").status_code == 401
    assert client.get("/llm/credentials").status_code == 401


class TestRoutes:
    BASE = f"/llm/endpoints/{EP_ID}/credentials"

    def test_list_returns_rows(self, client_and_stub) -> None:
        client, stub = client_and_stub
        stub.rows = [{"id": CRED_ID, "endpoint_id": EP_ID, "label": "default"}]
        res = client.get(self.BASE)
        assert res.status_code == 200
        assert res.json()[0]["label"] == "default"
        assert stub.calls == [("list", EP_ID)]

    def test_list_all_route(self, client_and_stub) -> None:
        client, stub = client_and_stub
        stub.rows = [{"id": CRED_ID, "endpoint_id": EP_ID}]
        res = client.get("/llm/credentials")
        assert res.status_code == 200
        assert res.json()[0]["endpoint_id"] == EP_ID
        assert stub.calls == [("list_all",)]

    def test_create_applies_defaults(self, client_and_stub) -> None:
        client, stub = client_and_stub
        stub.row = {"id": CRED_ID, "endpoint_id": EP_ID, "label": "alt"}
        res = client.post(self.BASE, json={"label": "alt", "token": "sk"})
        assert res.status_code == 200
        payload = stub.calls[0][2]
        assert payload["scope"] == "model"
        assert payload["priority"] == 1000 and payload["weight"] == 1
        assert payload["is_active"] is True
        assert payload["logical_id"] is None

    def test_create_token_conflict_is_409(self, client_and_stub) -> None:
        client, stub = client_and_stub
        stub.create_error = LlmCredentialTokenExists("dup")
        res = client.post(self.BASE, json={"label": "alt", "token": "sk"})
        assert res.status_code == 409
        assert "credential_token_exists" in res.json()["detail"]

    def test_create_unknown_endpoint_is_404(self, client_and_stub) -> None:
        client, stub = client_and_stub
        stub.create_error = LlmEndpointNotFound(EP_ID)
        res = client.post(self.BASE, json={"label": "alt", "token": "sk"})
        assert res.status_code == 404
        assert "endpoint_not_found" in res.json()["detail"]

    def test_get_missing_is_404(self, client_and_stub) -> None:
        client, stub = client_and_stub
        stub.row = None
        res = client.get(f"{self.BASE}/{CRED_ID}")
        assert res.status_code == 404
        assert "credential_not_found" in res.json()["detail"]

    def test_patch_forwards_only_supplied_fields(self, client_and_stub) -> None:
        client, stub = client_and_stub
        stub.row = {"id": CRED_ID, "endpoint_id": EP_ID, "token": "sk-new"}
        res = client.patch(f"{self.BASE}/{CRED_ID}", json={"token": "sk-new"})
        assert res.status_code == 200
        assert stub.calls[0][3] == {"token": "sk-new"}

    def test_patch_missing_is_404(self, client_and_stub) -> None:
        client, stub = client_and_stub
        stub.row = None
        res = client.patch(f"{self.BASE}/{CRED_ID}", json={"token": "x"})
        assert res.status_code == 404

    def test_delete_reports_result(self, client_and_stub) -> None:
        client, stub = client_and_stub
        stub.deleted = False
        res = client.delete(f"{self.BASE}/{CRED_ID}")
        assert res.status_code == 200
        assert res.json() == {"ok": False, "endpoint_id": EP_ID, "id": CRED_ID}

    def test_ensure_default_route(self, client_and_stub) -> None:
        client, stub = client_and_stub
        stub.ensured = {"id": CRED_ID, "label": "default"}
        res = client.post(f"{self.BASE}/ensure-default")
        assert res.status_code == 200
        body = res.json()
        # Route always reports created=False: ensure-default now purges only.
        assert body["created"] is False
        assert body["credential"]["label"] == "default"
        assert stub.calls == [("ensure", EP_ID)]

    def test_ensure_default_reports_nothing_to_do(self, client_and_stub) -> None:
        client, stub = client_and_stub
        stub.ensured = None
        body = client.post(f"{self.BASE}/ensure-default").json()
        assert body["created"] is False and body["credential"] is None

    def test_ensure_default_not_shadowed_by_cred_id_route(self, client_and_stub) -> None:
        """POST .../ensure-default must not be parsed as a {cred_id} path."""
        client, stub = client_and_stub
        stub.ensured = None
        client.post(f"{self.BASE}/ensure-default")
        assert stub.calls == [("ensure", EP_ID)]


class TestEndpointUpsertMigratesAuth:
    def test_upsert_triggers_ensure_default(self, client_and_stub) -> None:
        client, stub = client_and_stub

        async def _upsert(**kwargs):
            stub.calls.append(("upsert", kwargs["name"]))
            return {"id": EP_ID, "name": kwargs["name"]}

        stub.llm_external_endpoint_upsert = _upsert
        stub.ensured = {"id": CRED_ID, "label": "default"}

        res = client.post(
            "/llm/endpoints",
            json={
                "name": "ep-a",
                "scheme": "http",
                "host": "10.0.0.5",
                "port": 8000,
                "auth": {"type": "bearer", "token": "sk-legacy"},
            },
        )
        assert res.status_code == 200
        assert res.json()["id"] == EP_ID
        assert stub.calls == [("upsert", "ep-a"), ("ensure", EP_ID)]

    def test_upsert_survives_credential_failure(self, client_and_stub) -> None:
        client, stub = client_and_stub

        async def _upsert(**kwargs):
            return {"id": EP_ID, "name": kwargs["name"]}

        async def _boom(endpoint_id):
            raise RuntimeError("pg down")

        stub.llm_external_endpoint_upsert = _upsert
        stub.llm_endpoint_credentials_ensure_default_from_auth = _boom

        res = client.post(
            "/llm/endpoints",
            json={"name": "ep-a", "scheme": "http", "host": "h", "port": 1},
        )
        assert res.status_code == 200
        assert res.json()["id"] == EP_ID


class TestFacadeAndRoutes:
    METHODS = (
        "llm_endpoint_credentials_list",
        "llm_endpoint_credentials_list_all",
        "llm_endpoint_credential_get",
        "llm_endpoint_credential_create",
        "llm_endpoint_credential_update",
        "llm_endpoint_credential_delete",
        "llm_endpoint_credentials_ensure_default_from_auth",
    )

    def test_storage_facade_exposes_methods(self) -> None:
        from services.storage.src.storage import Storage

        for name in self.METHODS:
            assert callable(getattr(Storage, name, None)), name

    def test_routes_registered(self) -> None:
        from services.storage.src.main import app

        registered = {
            (path, method)
            for route in app.routes
            for path in [getattr(route, "path", "")]
            for method in getattr(route, "methods", set()) or set()
        }
        base = "/llm/endpoints/{endpoint_id}/credentials"
        assert ("/llm/credentials", "GET") in registered
        assert (base, "GET") in registered
        assert (base, "POST") in registered
        assert (f"{base}/ensure-default", "POST") in registered
        assert (f"{base}/{{cred_id}}", "GET") in registered
        assert (f"{base}/{{cred_id}}", "PATCH") in registered
        assert (f"{base}/{{cred_id}}", "DELETE") in registered
