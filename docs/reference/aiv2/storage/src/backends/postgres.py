"""
PostgreSQL backend for structured metadata and relations.
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..base import BaseBackend, DbExecutor, to_uuid, json_dumps, now_iso

try:
    import asyncpg
except ImportError:
    asyncpg = None

log = logging.getLogger(__name__)


def _coerce_meta_dict(value: Any) -> Dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _registry_logical_id(row: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(row, dict):
        return None
    meta = _coerce_meta_dict(row.get("metadata"))
    lid = str(meta.get("logical_id") or "").strip()
    if not lid or lid.lower() == "model":
        return None
    return lid


def _registry_has_provider(row: Dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    if str(row.get("backend_kind") or "").lower() == "kserve":
        return False
    if not _registry_logical_id(row):
        return False
    if row.get("external_endpoint_id"):
        return True
    meta = _coerce_meta_dict(row.get("metadata"))
    node_id = str(row.get("node_id") or meta.get("node_id") or meta.get("node") or "").strip()
    if node_id and node_id != "default":
        if meta.get("kind") in ("managed_node", "remote_agent"):
            return True
        if str(meta.get("origin") or "") == "node-auto":
            return True
        name = str(row.get("name") or "")
        if name.startswith(f"{node_id}:"):
            return True
        if meta.get("inference_url") or meta.get("deploy_status"):
            return True
    return False


def catalog_logical_ids_from_models(models: List[Dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for row in models:
        if not _registry_has_provider(row):
            continue
        lid = _registry_logical_id(row)
        if lid:
            out.add(lid)
    return out


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class LlmCredentialTokenExists(Exception):
    """Same bearer token already registered for this endpoint + logical_id."""


# Back-compat alias (older callers / tests may still import the label name).
LlmCredentialLabelExists = LlmCredentialTokenExists


class LlmEndpointNotFound(Exception):
    """The referenced llm_external_endpoints row does not exist."""


_LLM_ENDPOINT_COLS = (
    "id, name, scheme, host, port, base_path, auth, capabilities, last_probe, "
    "requests_per_minute, is_active, created_at, updated_at"
)

_LLM_CREDENTIAL_COLS = (
    "id, endpoint_id, label, token, scope, logical_id, upstream_model_id, "
    "priority, weight, requests_per_minute, is_active, last_probe, "
    "created_at, updated_at"
)


def _credential_rpm_or_null(value: Any) -> Optional[int]:
    """Normalize RPM: blank/0/negative → NULL (unlimited); else positive int."""
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _endpoint_row_to_dict(row: Any) -> Dict[str, Any]:
    d = dict(row)
    d["id"] = str(d["id"])
    return d

_LLM_CREDENTIAL_SCOPES = ("model",)


def _credential_scope(value: Any) -> str:
    """Credentials are always model-scoped (endpoint-wide tokens are not stored)."""
    _ = value
    return "model"


def _credential_text_or_null(value: Any) -> Optional[str]:
    """Empty/whitespace strings clear the column so callers can unset a scope."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _credential_row_to_dict(row: Any) -> Dict[str, Any]:
    d = dict(row)
    d["id"] = str(d["id"])
    if d.get("endpoint_id") is not None:
        d["endpoint_id"] = str(d["endpoint_id"])
    d["last_probe"] = _coerce_meta_dict(d.get("last_probe"))
    return d


class PostgresBackend(BaseBackend):
    """
    PostgreSQL backend for:
    - Artifact metadata (files, paths, checksums)
    - Persons and entities
    - Folders and hierarchies
    - Relations between artifacts
    - Tool runs and analysis results
    """

    def __init__(self, database_url: Optional[str] = None):
        super().__init__()
        self.database_url = database_url
        self.pool: Any = None
        # Repositories (images/audio/…) use `self.pg.executor.*`; health checks in
        # older images used `postgres.executor.fetchval`. Thin wrapper over pool.
        self.executor: Optional[DbExecutor] = None

    async def connect(self) -> None:
        """Initialize connection pool."""
        if self.database_url and asyncpg and self.pool is None:
            log.info("Connecting to PostgreSQL...")
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10
            )
            await self._apply_schema()
            self.executor = DbExecutor(self.pool)
            self._connected = True
            log.info("✅ PostgreSQL connected")

    async def disconnect(self) -> None:
        """Close connection pool."""
        self.executor = None
        if self.pool is not None:
            await self.pool.close()
            self.pool = None
            self._connected = False

    async def _apply_schema(self) -> None:
        """Apply database schema."""
        if self.pool is None:
            return
        
        from ..schemas import CORE_SCHEMA, AUTH_CHAT_SCHEMA
        async with self.pool.acquire() as conn:
            await conn.execute(CORE_SCHEMA)
            await conn.execute(AUTH_CHAT_SCHEMA)
        log.info("PostgreSQL schema applied (core + auth/chat)")

    # ---------------------------------------------------------------- LLM admin

    async def llm_external_endpoint_upsert(self, *, name: str, scheme: str, host: str, port: int, base_path: str = "", auth: Optional[Dict[str, Any]] = None, capabilities: Optional[Dict[str, Any]] = None, last_probe: Optional[Dict[str, Any]] = None, is_active: bool = True) -> Dict[str, Any]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO llm_external_endpoints (name, scheme, host, port, base_path, auth, capabilities, last_probe, is_active)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8::jsonb, $9)
                ON CONFLICT (name) DO UPDATE
                  SET scheme = EXCLUDED.scheme,
                      host = EXCLUDED.host,
                      port = EXCLUDED.port,
                      base_path = EXCLUDED.base_path,
                      auth = EXCLUDED.auth,
                      capabilities = EXCLUDED.capabilities,
                      last_probe = EXCLUDED.last_probe,
                      is_active = EXCLUDED.is_active,
                      updated_at = now()
                RETURNING id, name, scheme, host, port, base_path, auth, capabilities, last_probe, requests_per_minute, is_active, created_at, updated_at
                """,
                name,
                scheme,
                host,
                int(port),
                base_path or "",
                json_dumps(auth or {}),
                json_dumps(capabilities or {}),
                json_dumps(last_probe or {}),
                bool(is_active),
            )
        return _endpoint_row_to_dict(row)

    async def llm_external_endpoints_list(self) -> List[Dict[str, Any]]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {_LLM_ENDPOINT_COLS} FROM llm_external_endpoints ORDER BY updated_at DESC"
            )
        return [_endpoint_row_to_dict(r) for r in rows]

    async def llm_external_endpoint_delete(self, endpoint_id: str) -> bool:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        async with self.pool.acquire() as conn:
            res = await conn.execute(
                "DELETE FROM llm_external_endpoints WHERE id = $1",
                to_uuid(endpoint_id),
            )
        return res.startswith("DELETE ") and not res.endswith(" 0")

    async def llm_external_endpoint_find_by_address(
        self,
        *,
        scheme: str,
        host: str,
        port: int,
        base_path: str = "",
    ) -> Optional[Dict[str, Any]]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, name, scheme, host, port, base_path, auth, capabilities, last_probe,
                       is_active, created_at, updated_at
                FROM llm_external_endpoints
                WHERE lower(scheme) = lower($1)
                  AND lower(host) = lower($2)
                  AND port = $3
                  AND rtrim(coalesce(base_path, ''), '/') = rtrim(coalesce($4::text, ''), '/')
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                scheme or "http",
                host,
                int(port),
                base_path or "",
            )
        if not row:
            return None
        d = dict(row)
        d["id"] = str(d["id"])
        return d

    async def llm_external_endpoint_merge_into(
        self,
        *,
        keep_endpoint_id: str,
        drop_endpoint_id: str,
        keep_endpoint_name: str,
        drop_endpoint_name: str,
    ) -> Dict[str, Any]:
        """Repoint models from duplicate endpoint row, then delete the duplicate."""
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        keep_uuid = to_uuid(keep_endpoint_id)
        drop_uuid = to_uuid(drop_endpoint_id)
        repointed = 0
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(
                    """
                    SELECT id, name, node_id, metadata
                    FROM llm_local_models
                    WHERE external_endpoint_id = $1
                    """,
                    drop_uuid,
                )
                for row in rows:
                    old_name = str(row["name"] or "")
                    new_name = old_name
                    suffix = f"@{drop_endpoint_name}"
                    if suffix in old_name:
                        new_name = (
                            old_name[: -len(suffix)] + f"@{keep_endpoint_name}"
                        )
                    meta_raw = row["metadata"]
                    if isinstance(meta_raw, str):
                        try:
                            meta = json.loads(meta_raw) if meta_raw else {}
                        except Exception:
                            meta = {}
                    else:
                        meta = dict(meta_raw or {})
                    backend = str(meta.get("backend") or "")
                    if backend == f"openai_compat:{drop_endpoint_name}":
                        meta["backend"] = f"openai_compat:{keep_endpoint_name}"
                    await conn.execute(
                        """
                        UPDATE llm_local_models
                        SET external_endpoint_id = $1,
                            name = $2,
                            metadata = $3::jsonb,
                            updated_at = now()
                        WHERE id = $4
                        """,
                        keep_uuid,
                        new_name,
                        json_dumps(meta),
                        row["id"],
                    )
                    repointed += 1
                deleted = await conn.execute(
                    "DELETE FROM llm_external_endpoints WHERE id = $1",
                    drop_uuid,
                )
        return {
            "ok": deleted.startswith("DELETE ") and not deleted.endswith(" 0"),
            "repointed_models": repointed,
            "keep_endpoint_id": keep_endpoint_id,
            "drop_endpoint_id": drop_endpoint_id,
        }

    async def llm_external_endpoint_update(
        self,
        endpoint_id: str,
        *,
        name: Optional[str] = None,
        scheme: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        base_path: Optional[str] = None,
        auth: Optional[Dict[str, Any]] = None,
        capabilities: Optional[Dict[str, Any]] = None,
        last_probe: Optional[Dict[str, Any]] = None,
        requests_per_minute: Optional[int] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """Patch one endpoint by id; only non-None fields are updated.

        Pass ``requests_per_minute=0`` to clear the shared RPM gate (NULL).
        ``name`` may be updated so labels stay aligned with ``{node}-{runtime}-{port}``.
        """
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        sets: List[str] = []
        params: List[Any] = []
        idx = 1
        if name is not None:
            sets.append(f"name = ${idx}"); params.append(str(name).strip()); idx += 1
        if scheme is not None:
            sets.append(f"scheme = ${idx}"); params.append(scheme); idx += 1
        if host is not None:
            sets.append(f"host = ${idx}"); params.append(host); idx += 1
        if port is not None:
            sets.append(f"port = ${idx}"); params.append(int(port)); idx += 1
        if base_path is not None:
            sets.append(f"base_path = ${idx}"); params.append(base_path); idx += 1
        if auth is not None:
            sets.append(f"auth = ${idx}::jsonb"); params.append(json_dumps(auth)); idx += 1
        if capabilities is not None:
            sets.append(f"capabilities = ${idx}::jsonb"); params.append(json_dumps(capabilities)); idx += 1
        if last_probe is not None:
            sets.append(f"last_probe = ${idx}::jsonb"); params.append(json_dumps(last_probe)); idx += 1
        if requests_per_minute is not None:
            sets.append(f"requests_per_minute = ${idx}")
            params.append(_credential_rpm_or_null(requests_per_minute))
            idx += 1
        if is_active is not None:
            sets.append(f"is_active = ${idx}"); params.append(bool(is_active)); idx += 1
        if not sets:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"SELECT {_LLM_ENDPOINT_COLS} FROM llm_external_endpoints WHERE id = $1",
                    to_uuid(endpoint_id),
                )
            return _endpoint_row_to_dict(row) if row else None
        sets.append("updated_at = now()")
        params.append(to_uuid(endpoint_id))
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"UPDATE llm_external_endpoints SET {', '.join(sets)} WHERE id = ${idx} "
                f"RETURNING {_LLM_ENDPOINT_COLS}",
                *params,
            )
        return _endpoint_row_to_dict(row) if row else None

    # ------------------------------------------------- LLM endpoint credentials

    async def llm_endpoint_credentials_list(self, endpoint_id: str) -> List[Dict[str, Any]]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        ep_uuid = to_uuid(endpoint_id)
        if ep_uuid is None:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT {_LLM_CREDENTIAL_COLS}
                FROM llm_endpoint_credentials
                WHERE endpoint_id = $1
                ORDER BY priority ASC, lower(label) ASC
                """,
                ep_uuid,
            )
        return [_credential_row_to_dict(r) for r in rows]

    async def llm_endpoint_credentials_list_all(self) -> List[Dict[str, Any]]:
        """Every credential row across endpoints — used by llm-proxy registry sync."""
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT {_LLM_CREDENTIAL_COLS}
                FROM llm_endpoint_credentials
                ORDER BY endpoint_id, priority ASC, lower(label) ASC
                """
            )
        return [_credential_row_to_dict(r) for r in rows]

    async def llm_endpoint_credential_get(
        self, endpoint_id: str, cred_id: str
    ) -> Optional[Dict[str, Any]]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        ep_uuid = to_uuid(endpoint_id)
        cred_uuid = to_uuid(cred_id)
        if ep_uuid is None or cred_uuid is None:
            return None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT {_LLM_CREDENTIAL_COLS}
                FROM llm_endpoint_credentials
                WHERE endpoint_id = $1 AND id = $2
                """,
                ep_uuid,
                cred_uuid,
            )
        return _credential_row_to_dict(row) if row else None

    async def llm_endpoint_credential_create(
        self,
        endpoint_id: str,
        *,
        label: str,
        token: str,
        scope: str = "model",
        logical_id: Optional[str] = None,
        upstream_model_id: Optional[str] = None,
        priority: int = 1000,
        weight: int = 1,
        requests_per_minute: Optional[int] = None,
        is_active: bool = True,
        last_probe: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        ep_uuid = to_uuid(endpoint_id)
        if ep_uuid is None:
            raise LlmEndpointNotFound(endpoint_id)
        label_text = str(label or "").strip() or "default"
        scope_norm = _credential_scope(scope)
        logical_norm = _credential_text_or_null(logical_id) or _credential_text_or_null(
            upstream_model_id
        )
        if not logical_norm:
            raise ValueError("model_scope_requires_logical_or_upstream_id")
        token_text = str(token or "").strip()
        if not token_text:
            raise ValueError("token_empty: bearer token is required")
        rpm = _credential_rpm_or_null(requests_per_minute)
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    INSERT INTO llm_endpoint_credentials
                        (endpoint_id, label, token, scope, logical_id, upstream_model_id,
                         priority, weight, requests_per_minute, is_active, last_probe)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)
                    RETURNING {_LLM_CREDENTIAL_COLS}
                    """,
                    ep_uuid,
                    label_text,
                    token_text,
                    scope_norm,
                    logical_norm,
                    _credential_text_or_null(upstream_model_id) or logical_norm,
                    int(priority),
                    int(weight),
                    rpm,
                    bool(is_active),
                    json_dumps(last_probe or {}),
                )
        except Exception as exc:
            name = type(exc).__name__
            if name == "UniqueViolationError":
                raise LlmCredentialTokenExists(
                    f"logical_id={logical_norm} token_tail=…{(token_text[-4:] if token_text else '')}"
                ) from exc
            if name == "ForeignKeyViolationError":
                raise LlmEndpointNotFound(endpoint_id) from exc
            raise
        return _credential_row_to_dict(row)

    async def llm_endpoint_credential_update(
        self,
        endpoint_id: str,
        cred_id: str,
        *,
        label: Optional[str] = None,
        token: Optional[str] = None,
        scope: Optional[str] = None,
        logical_id: Optional[str] = None,
        upstream_model_id: Optional[str] = None,
        priority: Optional[int] = None,
        weight: Optional[int] = None,
        requests_per_minute: Optional[int] = None,
        is_active: Optional[bool] = None,
        last_probe: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Patch one credential; only non-None fields are written.

        Pass ``requests_per_minute=0`` (or negative) to clear the RPM gate (NULL).
        """
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        ep_uuid = to_uuid(endpoint_id)
        cred_uuid = to_uuid(cred_id)
        if ep_uuid is None or cred_uuid is None:
            return None
        sets: List[str] = []
        params: List[Any] = []
        idx = 1
        if label is not None:
            text = str(label).strip()
            if not text:
                raise ValueError("label_empty: credential label cannot be blank")
            sets.append(f"label = ${idx}"); params.append(text); idx += 1
        if token is not None:
            sets.append(f"token = ${idx}"); params.append(str(token).strip()); idx += 1
        if scope is not None:
            sets.append(f"scope = ${idx}"); params.append(_credential_scope(scope)); idx += 1
        if logical_id is not None:
            sets.append(f"logical_id = ${idx}"); params.append(_credential_text_or_null(logical_id)); idx += 1
        if upstream_model_id is not None:
            sets.append(f"upstream_model_id = ${idx}"); params.append(_credential_text_or_null(upstream_model_id)); idx += 1
        if priority is not None:
            sets.append(f"priority = ${idx}"); params.append(int(priority)); idx += 1
        if weight is not None:
            sets.append(f"weight = ${idx}"); params.append(int(weight)); idx += 1
        if requests_per_minute is not None:
            sets.append(f"requests_per_minute = ${idx}")
            params.append(_credential_rpm_or_null(requests_per_minute))
            idx += 1
        if is_active is not None:
            sets.append(f"is_active = ${idx}"); params.append(bool(is_active)); idx += 1
        if last_probe is not None:
            sets.append(f"last_probe = ${idx}::jsonb"); params.append(json_dumps(last_probe)); idx += 1
        if not sets:
            return await self.llm_endpoint_credential_get(endpoint_id, cred_id)
        sets.append("updated_at = now()")
        params.append(ep_uuid)
        params.append(cred_uuid)
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"UPDATE llm_endpoint_credentials SET {', '.join(sets)} "
                    f"WHERE endpoint_id = ${idx} AND id = ${idx + 1} "
                    f"RETURNING {_LLM_CREDENTIAL_COLS}",
                    *params,
                )
        except Exception as exc:
            if type(exc).__name__ == "UniqueViolationError":
                raise LlmCredentialTokenExists(
                    f"duplicate bearer for endpoint+logical_id (label={label or ''})"
                ) from exc
            raise
        return _credential_row_to_dict(row) if row else None

    async def llm_endpoint_credentials_set_rpm_for_model(
        self,
        endpoint_id: str,
        *,
        requests_per_minute: int,
        logical_id: Optional[str] = None,
        upstream_model_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Set one RPM on every credential for endpoint + logical_id (or upstream).

        Matching rows: same endpoint and (logical_id match OR upstream_model_id
        match). Does not touch credentials bound to other models on the endpoint.
        Pass ``requests_per_minute=0`` to clear (NULL).
        """
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        ep_uuid = to_uuid(endpoint_id)
        if ep_uuid is None:
            return []
        lid = _credential_text_or_null(logical_id)
        upstream = _credential_text_or_null(upstream_model_id)
        if not lid and not upstream:
            raise ValueError("logical_id_or_upstream_required")
        rpm = _credential_rpm_or_null(requests_per_minute)
        clauses: List[str] = []
        params: List[Any] = [ep_uuid, rpm]
        # $1 endpoint, $2 rpm value
        if lid:
            params.append(lid)
            clauses.append(f"lower(coalesce(logical_id, '')) = lower(${len(params)})")
        if upstream:
            params.append(upstream)
            clauses.append(
                f"lower(coalesce(upstream_model_id, '')) = lower(${len(params)})"
            )
        where_model = " OR ".join(clauses)
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                UPDATE llm_endpoint_credentials
                SET requests_per_minute = $2, updated_at = now()
                WHERE endpoint_id = $1
                  AND ({where_model})
                RETURNING {_LLM_CREDENTIAL_COLS}
                """,
                *params,
            )
        return [_credential_row_to_dict(r) for r in rows]

    async def llm_endpoint_credential_delete(self, endpoint_id: str, cred_id: str) -> bool:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        ep_uuid = to_uuid(endpoint_id)
        cred_uuid = to_uuid(cred_id)
        if ep_uuid is None or cred_uuid is None:
            return False
        async with self.pool.acquire() as conn:
            res = await conn.execute(
                "DELETE FROM llm_endpoint_credentials WHERE endpoint_id = $1 AND id = $2",
                ep_uuid,
                cred_uuid,
            )
        return res.startswith("DELETE ") and not res.endswith(" 0")

    async def llm_endpoint_credentials_ensure_default_from_auth(
        self, endpoint_id: str
    ) -> Optional[Dict[str, Any]]:
        """Purge legacy endpoint-wide credential rows (no longer supported).

        Historically this migrated ``auth.token`` into a ``default`` / scope=endpoint
        row. That model is retired: credentials are model-scoped only. Calling this
        cleans leftover whole-endpoint rows and returns ``None``.
        """
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        ep_uuid = to_uuid(endpoint_id)
        if ep_uuid is None:
            return None
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                DELETE FROM llm_endpoint_credentials
                 WHERE endpoint_id = $1
                   AND (
                     lower(COALESCE(scope, '')) <> 'model'
                     OR logical_id IS NULL
                     OR btrim(logical_id) = ''
                   )
                """,
                ep_uuid,
            )
        return None
    async def llm_local_model_upsert(
        self,
        *,
        name: str,
        task: Optional[str] = None,
        backend_kind: str,
        kserve_namespace: Optional[str] = None,
        kserve_inferenceservice: Optional[str] = None,
        external_endpoint_id: Optional[str] = None,
        external_model_id: Optional[str] = None,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_active: bool = True,
        node_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        endpoint_uuid = to_uuid(external_endpoint_id) if external_endpoint_id else None
        meta = dict(metadata or {})
        # Prefer persisted dispatch_key; legacy task param ignored.
        dispatch = str(meta.get("dispatch_key") or "").strip().lower()
        if not dispatch:
            try:
                from aiv2_model_contract.introduction import dispatch_key_from_serve

                dispatch = str(
                    dispatch_key_from_serve(
                        api_protocol=str(meta.get("api_protocol") or ""),
                        ui_profile=str(meta.get("ui_profile") or ""),
                        endpoint_path=str(meta.get("endpoint_path") or ""),
                        input_modalities=list(meta.get("input_modalities") or []),
                        output_modalities=list(meta.get("output_modalities") or []),
                        runtime=str(meta.get("runtime") or ""),
                        api=str(meta.get("api") or ""),
                    )
                    or "unknown"
                )
            except Exception:
                dispatch = "unknown"
            if dispatch and dispatch != "unknown":
                meta["dispatch_key"] = dispatch
        _ = task  # Deprecated: ignored. Prefer metadata.dispatch_key.
        nid = node_id or meta.get("node_id") or meta.get("node") or "default"
        old_logical_id: Optional[str] = None
        if self.pool:
            async with self.pool.acquire() as conn:
                old_row = await conn.fetchrow(
                    "SELECT metadata FROM llm_local_models WHERE node_id = $1 AND name = $2",
                    nid,
                    name,
                )
                if old_row:
                    old_logical_id = _registry_logical_id(dict(old_row))
        async with self.pool.acquire() as conn:
            await conn.execute(
                "ALTER TABLE llm_local_models ADD COLUMN IF NOT EXISTS node_id TEXT"
            )
            await conn.execute(
                "ALTER TABLE llm_local_models DROP CONSTRAINT IF EXISTS llm_local_models_name_key"
            )
            # Drop legacy task column if present (migration 17); keep upsert task-free.
            await conn.execute(
                "ALTER TABLE llm_local_models DROP COLUMN IF EXISTS task"
            )
            row = await conn.fetchrow(
                """
                INSERT INTO llm_local_models (
                    name, backend_kind, kserve_namespace, kserve_inferenceservice,
                    external_endpoint_id, external_model_id, version, metadata, is_active, node_id
                )
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9,$10)
                ON CONFLICT (node_id, name) DO UPDATE
                  SET backend_kind = EXCLUDED.backend_kind,
                      kserve_namespace = EXCLUDED.kserve_namespace,
                      kserve_inferenceservice = EXCLUDED.kserve_inferenceservice,
                      external_endpoint_id = EXCLUDED.external_endpoint_id,
                      external_model_id = EXCLUDED.external_model_id,
                      version = EXCLUDED.version,
                      metadata = EXCLUDED.metadata,
                      is_active = EXCLUDED.is_active,
                      node_id = EXCLUDED.node_id,
                      updated_at = now()
                RETURNING id, name, backend_kind, kserve_namespace, kserve_inferenceservice,
                          external_endpoint_id, external_model_id, version, metadata, is_active,
                          node_id, created_at, updated_at
                """,
                name,
                backend_kind,
                kserve_namespace,
                kserve_inferenceservice,
                endpoint_uuid,
                external_model_id,
                version,
                json_dumps(meta),
                bool(is_active),
                nid,
            )
        d = dict(row)
        d["id"] = str(d["id"])
        if d.get("external_endpoint_id"):
            d["external_endpoint_id"] = str(d["external_endpoint_id"])
        new_logical_id = _registry_logical_id(d)
        if old_logical_id and old_logical_id != new_logical_id:
            models = await self.llm_local_models_list()
            catalog = catalog_logical_ids_from_models(models)
            if old_logical_id not in catalog:
                await self.llm_routes_purge_logical_ids([old_logical_id])
        return d

    async def llm_local_model_patch(
        self,
        *,
        node_id: str,
        name: str,
        is_active: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        sets: List[str] = []
        params: List[Any] = []
        idx = 1
        if is_active is not None:
            sets.append(f"is_active = ${idx}")
            params.append(bool(is_active))
            idx += 1
        if metadata is not None:
            sets.append(f"metadata = ${idx}::jsonb")
            params.append(json_dumps(metadata))
            idx += 1
        if not sets:
            return await self.llm_local_model_get(node_id=node_id, name=name)
        old_row = await self.llm_local_model_get(node_id=node_id, name=name)
        old_logical_id = _registry_logical_id(old_row) if old_row else None
        sets.append("updated_at = now()")
        params.extend([node_id, name])
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE llm_local_models
                SET {", ".join(sets)}
                WHERE node_id = ${idx} AND name = ${idx + 1}
                RETURNING id, name, backend_kind, kserve_namespace, kserve_inferenceservice,
                          external_endpoint_id, external_model_id, version, metadata, is_active,
                          node_id, created_at, updated_at
                """,
                *params,
            )
        if not row:
            return None
        d = dict(row)
        d["id"] = str(d["id"])
        if d.get("external_endpoint_id"):
            d["external_endpoint_id"] = str(d["external_endpoint_id"])
        new_logical_id = _registry_logical_id(d)
        if old_logical_id and old_logical_id != new_logical_id:
            models = await self.llm_local_models_list()
            catalog = catalog_logical_ids_from_models(models)
            if old_logical_id not in catalog:
                await self.llm_routes_purge_logical_ids([old_logical_id])
        return d

    async def llm_local_model_get(self, *, node_id: str, name: str) -> Optional[Dict[str, Any]]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, name, backend_kind, kserve_namespace, kserve_inferenceservice,
                       external_endpoint_id, external_model_id, version, metadata, is_active,
                       node_id, created_at, updated_at
                FROM llm_local_models WHERE node_id = $1 AND name = $2
                """,
                node_id,
                name,
            )
        if not row:
            return None
        d = dict(row)
        d["id"] = str(d["id"])
        if d.get("external_endpoint_id"):
            d["external_endpoint_id"] = str(d["external_endpoint_id"])
        return d

    async def llm_local_models_list(self) -> List[Dict[str, Any]]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, name, backend_kind, kserve_namespace, kserve_inferenceservice,
                       external_endpoint_id, external_model_id, version, metadata, is_active,
                       node_id, created_at, updated_at
                FROM llm_local_models ORDER BY node_id, updated_at DESC
                """
            )
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["id"] = str(d["id"])
            if d.get("external_endpoint_id"):
                d["external_endpoint_id"] = str(d["external_endpoint_id"])
            out.append(d)
        return out

    async def llm_local_model_delete(
        self, model_name: str, *, node_id: Optional[str] = None
    ) -> bool:
        """Delete a registered model by node + name (or name only for legacy)."""
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        removed_logical_ids: set[str] = set()
        async with self.pool.acquire() as conn:
            if node_id:
                rows = await conn.fetch(
                    """
                    SELECT metadata FROM llm_local_models
                    WHERE node_id = $1 AND name = $2
                    """,
                    node_id,
                    model_name,
                )
                res = await conn.execute(
                    "DELETE FROM llm_local_models WHERE node_id = $1 AND name = $2",
                    node_id,
                    model_name,
                )
            else:
                rows = await conn.fetch(
                    "SELECT metadata FROM llm_local_models WHERE name = $1",
                    model_name,
                )
                res = await conn.execute(
                    "DELETE FROM llm_local_models WHERE name = $1",
                    model_name,
                )
            for row in rows:
                lid = _registry_logical_id(dict(row))
                if lid:
                    removed_logical_ids.add(lid)
        deleted = res.startswith("DELETE ") and not res.endswith(" 0")
        if deleted and removed_logical_ids:
            models = await self.llm_local_models_list()
            catalog = catalog_logical_ids_from_models(models)
            purge = sorted(lid for lid in removed_logical_ids if lid not in catalog)
            if purge:
                await self.llm_routes_purge_logical_ids(purge)
        return deleted

    async def llm_nodes_list(self) -> List[Dict[str, Any]]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, display_name, models_root, has_gpu, is_active, metadata,
                       created_at, updated_at
                FROM llm_nodes ORDER BY id
                """
            )
        return [dict(r) for r in rows]

    async def llm_node_upsert(
        self,
        *,
        id: str,
        display_name: Optional[str] = None,
        models_root: Optional[str] = None,
        has_gpu: bool = True,
        is_active: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO llm_nodes (id, display_name, models_root, has_gpu, is_active, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                ON CONFLICT (id) DO UPDATE
                  SET display_name = CASE
                        WHEN COALESCE(llm_nodes.metadata, '{}'::jsonb)->'admin_overrides'->>'display_name' IS NOT NULL
                         AND btrim(COALESCE(llm_nodes.metadata, '{}'::jsonb)->'admin_overrides'->>'display_name') <> ''
                        THEN llm_nodes.metadata->'admin_overrides'->>'display_name'
                        ELSE COALESCE(EXCLUDED.display_name, llm_nodes.display_name)
                      END,
                      models_root = COALESCE(EXCLUDED.models_root, llm_nodes.models_root),
                      has_gpu = EXCLUDED.has_gpu,
                      is_active = CASE
                        WHEN COALESCE(llm_nodes.metadata, '{}'::jsonb)->'admin_overrides' ? 'is_active'
                        THEN (llm_nodes.metadata->'admin_overrides'->>'is_active')::boolean
                        ELSE EXCLUDED.is_active
                      END,
                      metadata = (
                        COALESCE(llm_nodes.metadata, '{}'::jsonb)
                          || COALESCE(EXCLUDED.metadata, '{}'::jsonb)
                      ) || CASE
                        WHEN COALESCE(llm_nodes.metadata, '{}'::jsonb)->'admin_overrides'->>'host' IS NOT NULL
                        THEN jsonb_build_object(
                          'host', llm_nodes.metadata->'admin_overrides'->>'host',
                          'public_host', llm_nodes.metadata->'admin_overrides'->>'host',
                          'inference_host', llm_nodes.metadata->'admin_overrides'->>'host',
                          'port', COALESCE(
                            NULLIF(llm_nodes.metadata->'admin_overrides'->>'port', '')::int,
                            NULLIF(COALESCE(llm_nodes.metadata, '{}'::jsonb)->>'port', '')::int,
                            8020
                          ),
                          'agent_url', 'http://'
                            || (llm_nodes.metadata->'admin_overrides'->>'host')
                            || ':'
                            || COALESCE(
                              NULLIF(llm_nodes.metadata->'admin_overrides'->>'port', ''),
                              NULLIF(COALESCE(llm_nodes.metadata, '{}'::jsonb)->>'port', ''),
                              '8020'
                            )
                        )
                        ELSE '{}'::jsonb
                      END,
                      updated_at = now()
                RETURNING id, display_name, models_root, has_gpu, is_active, metadata,
                          created_at, updated_at
                """,
                id,
                display_name,
                models_root,
                bool(has_gpu),
                bool(is_active),
                json_dumps(metadata or {}),
            )
        return dict(row)

    async def llm_node_patch(
        self,
        *,
        id: str,
        display_name: Optional[str] = None,
        is_active: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        sets: List[str] = ["updated_at = now()"]
        params: List[Any] = []
        idx = 1
        if display_name is not None:
            sets.append(f"display_name = ${idx}")
            params.append(display_name)
            idx += 1
        if is_active is not None:
            sets.append(f"is_active = ${idx}")
            params.append(bool(is_active))
            idx += 1
        if metadata is not None:
            sets.append(
                f"metadata = COALESCE(metadata, '{{}}'::jsonb) || ${idx}::jsonb"
            )
            params.append(json_dumps(metadata))
            idx += 1
        params.append(id)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE llm_nodes SET {", ".join(sets)}
                WHERE id = ${idx}
                RETURNING id, display_name, models_root, has_gpu, is_active, metadata,
                          created_at, updated_at
                """,
                *params,
            )
        return dict(row) if row else None

    async def llm_node_delete(self, *, id: str) -> bool:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        async with self.pool.acquire() as conn:
            res = await conn.execute("DELETE FROM llm_nodes WHERE id = $1", id)
        return res.startswith("DELETE ") and not res.endswith(" 0")

    async def llm_local_models_by_node(self, node_id: str) -> List[Dict[str, Any]]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT name, backend_kind, external_endpoint_id, external_model_id,
                       metadata, is_active, node_id, created_at, updated_at
                FROM llm_local_models WHERE node_id = $1 ORDER BY name
                """,
                node_id,
            )
        return [dict(r) for r in rows]

    async def llm_routes_purge_logical_ids(
        self, logical_ids: List[str]
    ) -> Dict[str, Any]:
        """Delete bindings whose primary model is a removed catalog logical_id."""
        ids = sorted({str(x).strip() for x in logical_ids if str(x).strip()})
        if not ids or not self.pool:
            return {"deleted_routes": [], "cleared_fallbacks": []}
        async with self.pool.acquire() as conn:
            deleted_rows = await conn.fetch(
                "DELETE FROM llm_routes WHERE model_name = ANY($1::text[]) RETURNING route_key",
                ids,
            )
            cleared_rows = await conn.fetch(
                """
                UPDATE llm_routes
                SET fallback_model_name = NULL, updated_at = now()
                WHERE fallback_model_name = ANY($1::text[])
                RETURNING route_key
                """,
                ids,
            )
        return {
            "deleted_routes": sorted({str(r["route_key"]) for r in deleted_rows}),
            "cleared_fallbacks": sorted({str(r["route_key"]) for r in cleared_rows}),
        }

    async def llm_routes_enforce_catalog_bindings(self) -> Dict[str, Any]:
        """Drop route rows that do not bind to a catalog logical_id."""
        if not self.pool:
            return {"deleted_routes": [], "cleared_fallbacks": []}
        models = await self.llm_local_models_list()
        catalog = catalog_logical_ids_from_models(models)
        routes = await self.llm_routes_list()
        orphan_keys: List[str] = []
        stale_fallback_keys: List[str] = []
        for route in routes:
            if not isinstance(route, dict):
                continue
            rk = str(route.get("route_key") or "").strip()
            if not rk:
                continue
            primary = str(route.get("model_name") or "").strip()
            if primary not in catalog:
                orphan_keys.append(rk)
                continue
            fallback = str(route.get("fallback_model_name") or "").strip()
            if fallback and fallback not in catalog:
                stale_fallback_keys.append(rk)
        deleted: List[str] = []
        cleared: List[str] = []
        async with self.pool.acquire() as conn:
            if orphan_keys:
                rows = await conn.fetch(
                    "DELETE FROM llm_routes WHERE route_key = ANY($1::text[]) RETURNING route_key",
                    orphan_keys,
                )
                deleted = sorted({str(r["route_key"]) for r in rows})
            for rk in stale_fallback_keys:
                if rk in orphan_keys:
                    continue
                row = await conn.fetchrow(
                    """
                    UPDATE llm_routes
                    SET fallback_model_name = NULL, updated_at = now()
                    WHERE route_key = $1
                    RETURNING route_key
                    """,
                    rk,
                )
                if row:
                    cleared.append(str(row["route_key"]))
        return {
            "deleted_routes": deleted,
            "cleared_fallbacks": sorted(set(cleared)),
            "orphan_routes": sorted(set(orphan_keys)),
        }

    async def llm_route_upsert(self, *, route_key: str, model_name: str, fallback_model_name: Optional[str] = None, constraints: Optional[Dict[str, Any]] = None, is_active: bool = True) -> Dict[str, Any]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        models = await self.llm_local_models_list()
        catalog = catalog_logical_ids_from_models(models)
        primary = str(model_name or "").strip()
        if primary not in catalog:
            raise ValueError(f"catalog_logical_id_required: {primary}")
        # A route binds exactly one logical id. The legacy fallback field is
        # accepted for wire compatibility but never persisted.
        _ = fallback_model_name
        fallback = None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO llm_routes (route_key, model_name, fallback_model_name, constraints, is_active)
                VALUES ($1,$2,$3,$4::jsonb,$5)
                ON CONFLICT (route_key) DO UPDATE
                  SET model_name = EXCLUDED.model_name,
                      fallback_model_name = EXCLUDED.fallback_model_name,
                      constraints = EXCLUDED.constraints,
                      is_active = EXCLUDED.is_active,
                      updated_at = now()
                RETURNING id, route_key, model_name, fallback_model_name, constraints, is_active, created_at, updated_at
                """,
                route_key,
                model_name,
                fallback_model_name,
                json_dumps(constraints or {}),
                bool(is_active),
            )
        d = dict(row)
        d["id"] = str(d["id"])
        return d

    async def llm_routes_list(self) -> List[Dict[str, Any]]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, route_key, model_name, fallback_model_name, constraints, is_active, created_at, updated_at FROM llm_routes ORDER BY updated_at DESC"
            )
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["id"] = str(d["id"])
            out.append(d)
        return out

    async def llm_route_delete(self, route_key: str) -> bool:
        """Delete a route entry by its unique route_key (e.g. service binding)."""
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        async with self.pool.acquire() as conn:
            res = await conn.execute(
                "DELETE FROM llm_routes WHERE route_key = $1",
                route_key,
            )
        return res.startswith("DELETE ") and not res.endswith(" 0")

    async def llm_pool_policies_list(self) -> List[Dict[str, Any]]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT logical_id, canonical_model_key, min_replicas, max_replicas,
                       target_free_vram_mb, idle_scale_down_sec, enabled, metadata, updated_at
                FROM llm_pool_policies
                ORDER BY logical_id
                """
            )
        return [dict(r) for r in rows]

    async def llm_pool_policy_upsert(
        self,
        *,
        logical_id: str,
        canonical_model_key: Optional[str] = None,
        min_replicas: int = 1,
        max_replicas: int = 3,
        target_free_vram_mb: int = 4096,
        idle_scale_down_sec: int = 1800,
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO llm_pool_policies (
                    logical_id, canonical_model_key, min_replicas, max_replicas,
                    target_free_vram_mb, idle_scale_down_sec, enabled, metadata
                )
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb)
                ON CONFLICT (logical_id) DO UPDATE
                  SET canonical_model_key = EXCLUDED.canonical_model_key,
                      min_replicas = EXCLUDED.min_replicas,
                      max_replicas = EXCLUDED.max_replicas,
                      target_free_vram_mb = EXCLUDED.target_free_vram_mb,
                      idle_scale_down_sec = EXCLUDED.idle_scale_down_sec,
                      enabled = EXCLUDED.enabled,
                      metadata = EXCLUDED.metadata,
                      updated_at = now()
                RETURNING logical_id, canonical_model_key, min_replicas, max_replicas,
                          target_free_vram_mb, idle_scale_down_sec, enabled, metadata, updated_at
                """,
                logical_id,
                canonical_model_key,
                int(min_replicas),
                int(max_replicas),
                int(target_free_vram_mb),
                int(idle_scale_down_sec),
                bool(enabled),
                json_dumps(metadata or {}),
            )
        return dict(row)

    async def llm_pool_policy_delete(self, logical_id: str) -> bool:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        async with self.pool.acquire() as conn:
            res = await conn.execute(
                "DELETE FROM llm_pool_policies WHERE logical_id = $1",
                logical_id,
            )
        return res.startswith("DELETE ") and not res.endswith(" 0")

    async def llm_local_model_touch_usage(
        self,
        *,
        node_id: str,
        name: str,
        last_used_at: str,
    ) -> bool:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        async with self.pool.acquire() as conn:
            res = await conn.execute(
                """
                UPDATE llm_local_models
                SET metadata = COALESCE(metadata, '{}'::jsonb) || jsonb_build_object('last_used_at', $3::text),
                    updated_at = now()
                WHERE node_id = $1 AND name = $2
                """,
                node_id,
                name,
                last_used_at,
            )
        return res.startswith("UPDATE ") and not res.endswith(" 0")

    async def llm_model_candidate_upsert(
        self,
        *,
        storage_path: str,
        category: Optional[str] = None,
        name: Optional[str] = None,
        model_name: Optional[str] = None,
        deploy_status: str = "discovered",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO llm_model_candidates
                  (storage_path, category, name, model_name, deploy_status, metadata)
                VALUES ($1,$2,$3,$4,$5,$6::jsonb)
                ON CONFLICT (storage_path) DO UPDATE
                  SET category = COALESCE(EXCLUDED.category, llm_model_candidates.category),
                      name = COALESCE(EXCLUDED.name, llm_model_candidates.name),
                      model_name = COALESCE(EXCLUDED.model_name, llm_model_candidates.model_name),
                      deploy_status = EXCLUDED.deploy_status,
                      metadata = EXCLUDED.metadata,
                      updated_at = now()
                RETURNING id, storage_path, category, name, model_name, deploy_status, metadata, created_at, updated_at
                """,
                storage_path,
                category,
                name,
                model_name,
                deploy_status,
                json_dumps(metadata or {}),
            )
        d = dict(row)
        d["id"] = str(d["id"])
        return d

    async def llm_model_candidates_list(self) -> List[Dict[str, Any]]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, storage_path, category, name, model_name, deploy_status, metadata, created_at, updated_at "
                "FROM llm_model_candidates ORDER BY updated_at DESC"
            )
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["id"] = str(d["id"])
            out.append(d)
        return out

    async def llm_gpu_budget(self, *, node: Optional[str] = None, max_fraction: float = 0.95) -> Dict[str, Any]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT name, metadata, is_active, node_id
                FROM llm_local_models WHERE is_active = TRUE
                """
            )
        allocated = 0.0
        entries: List[Dict[str, Any]] = []
        for r in rows:
            meta = r["metadata"] or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            row_node = r.get("node_id") or meta.get("node_id") or meta.get("node")
            if node and row_node and row_node != node:
                continue
            frac = meta.get("gpu_memory_fraction")
            if frac is None:
                frac = meta.get("gpu_memory_utilization")
            try:
                fval = float(frac) if frac is not None else 0.0
            except (TypeError, ValueError):
                fval = 0.0
            if fval <= 0:
                continue
            allocated += fval
            entries.append(
                {
                    "name": r["name"],
                    "dispatch_key": (meta.get("dispatch_key") or meta.get("task") or ""),
                    "gpu_memory_fraction": fval,
                    "deploy_status": meta.get("deploy_status", "deployed"),
                }
            )
        remaining = max(0.0, max_fraction - allocated)
        return {
            "max_fraction": max_fraction,
            "allocated_fraction": round(allocated, 4),
            "remaining_fraction": round(remaining, 4),
            "over_budget": allocated > max_fraction,
            "models": entries,
            "node": node,
        }

    # ------------------------------------------------------------------ Persons
    async def person_create(
        self,
        name: str,
        *,
        external_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create or update a person record."""
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO persons (full_name, external_id, metadata)
                VALUES ($1, $2, $3::jsonb)
                ON CONFLICT (external_id) WHERE external_id IS NOT NULL
                DO UPDATE SET full_name = EXCLUDED.full_name,
                              metadata = EXCLUDED.metadata,
                              updated_at = now()
                RETURNING id, full_name, external_id, metadata, created_at
                """,
                name,
                external_id,
                json_dumps(metadata),
            )

        return {
            "id": str(row["id"]),
            "name": row["full_name"],
            "external_id": row["external_id"],
            "metadata": row["metadata"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }

    async def person_get(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Get person by ID or external_id."""
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")

        async with self.pool.acquire() as conn:
            try:
                person_uuid = uuid.UUID(identifier)
                row = await conn.fetchrow(
                    "SELECT id, full_name, external_id, metadata, created_at FROM persons WHERE id = $1",
                    person_uuid,
                )
            except ValueError:
                row = await conn.fetchrow(
                    "SELECT id, full_name, external_id, metadata, created_at FROM persons WHERE external_id = $1",
                    identifier,
                )

        if not row:
            return None

        return {
            "id": str(row["id"]),
            "name": row["full_name"],
            "external_id": row["external_id"],
            "metadata": row["metadata"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }

    async def person_list(self, q: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List all persons, optionally filtered by name."""
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")

        async with self.pool.acquire() as conn:
            if q:
                rows = await conn.fetch(
                    "SELECT id, full_name, external_id, metadata, created_at FROM persons WHERE full_name ILIKE $1 LIMIT $2",
                    f"%{q}%", limit
                )
            else:
                rows = await conn.fetch(
                    "SELECT id, full_name, external_id, metadata, created_at FROM persons LIMIT $1",
                    limit
                )

        return [
            {
                "id": str(r["id"]),
                "name": r["full_name"],
                "external_id": r["external_id"],
                "metadata": r["metadata"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ]

    # ------------------------------------------------------------------ Folders
    async def folder_get_or_create(
        self,
        name: str,
        *,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get or create folder."""
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")

        parent_uuid = to_uuid(parent_id)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO folders (name, parent_id, metadata)
                VALUES ($1, $2, $3::jsonb)
                ON CONFLICT (parent_id, name)
                DO UPDATE SET updated_at = now()
                RETURNING id, name, parent_id, metadata, created_at
                """,
                name,
                parent_uuid,
                json_dumps(metadata),
            )

        return {
            "id": str(row["id"]),
            "name": row["name"],
            "parent_id": str(row["parent_id"]) if row["parent_id"] else None,
            "metadata": row["metadata"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }

    # ------------------------------------------------------------------ Artifacts
    async def artifact_upsert(
        self,
        path: str,
        media_type: str,
        *,
        external_ref: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        checksum: Optional[str] = None,
        folder_id: Optional[str] = None,
        original_filename: Optional[str] = None,
        file_size_bytes: Optional[int] = None,
        mime_type: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Insert or update artifact, return ID."""
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")

        folder_uuid = to_uuid(folder_id)
        user_uuid = to_uuid(user_id)
        session_uuid = to_uuid(session_id)
        async with self.pool.acquire() as conn:
            artifact_id = await conn.fetchval(
                """
                INSERT INTO artifacts (
                    media_type, storage_path, external_ref, checksum, 
                    metadata, folder_id, original_filename, file_size_bytes, mime_type,
                    user_id, session_id
                )
                VALUES ($1::media_kind, $2, $3, $4, $5::jsonb, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (storage_path)
                DO UPDATE SET 
                    metadata = COALESCE(artifacts.metadata, '{}') || EXCLUDED.metadata,
                    checksum = COALESCE(EXCLUDED.checksum, artifacts.checksum),
                    folder_id = COALESCE(EXCLUDED.folder_id, artifacts.folder_id),
                    user_id = COALESCE(EXCLUDED.user_id, artifacts.user_id),
                    session_id = COALESCE(EXCLUDED.session_id, artifacts.session_id),
                    updated_at = now()
                RETURNING id
                """,
                media_type,
                path,
                external_ref,
                checksum,
                json_dumps(metadata),
                folder_uuid,
                original_filename,
                file_size_bytes,
                mime_type,
                user_uuid,
                session_uuid,
            )
        return str(artifact_id)

    async def artifact_get(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Get artifact by ID."""
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, media_type, storage_path, external_ref, checksum, 
                       metadata, folder_id, original_filename, file_size_bytes, 
                       mime_type, user_id, session_id, created_at, updated_at
                FROM artifacts WHERE id = $1
                """,
                to_uuid(artifact_id),
            )

        if not row:
            return None

        return {
            "id": str(row["id"]),
            "media_type": row["media_type"],
            "storage_path": row["storage_path"],
            "external_ref": row["external_ref"],
            "checksum": row["checksum"],
            "metadata": row["metadata"],
            "folder_id": str(row["folder_id"]) if row["folder_id"] else None,
            "original_filename": row["original_filename"],
            "file_size_bytes": row["file_size_bytes"],
            "mime_type": row["mime_type"],
            "user_id": str(row["user_id"]) if row["user_id"] else None,
            "session_id": str(row["session_id"]) if row["session_id"] else None,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        }

    async def artifact_list(
        self,
        folder_id: Optional[str] = None,
        media_type: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List artifacts with optional filters."""
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")

        conditions = []
        params = []
        idx = 1

        if folder_id:
            conditions.append(f"folder_id = ${idx}")
            params.append(to_uuid(folder_id))
            idx += 1

        if media_type:
            conditions.append(f"media_type = ${idx}::media_kind")
            params.append(media_type)
            idx += 1

        if user_id:
            conditions.append(f"user_id = ${idx}")
            params.append(to_uuid(user_id))
            idx += 1

        if session_id:
            conditions.append(self._artifact_in_session_sql("artifacts", idx))
            params.append(to_uuid(session_id))
            idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        params.extend([limit, offset])

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, media_type, storage_path, original_filename, 
                       file_size_bytes, mime_type, user_id, session_id, created_at
                FROM artifacts 
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ${idx} OFFSET ${idx + 1}
                """,
                *params,
            )

        return [
            {
                "id": str(r["id"]),
                "media_type": r["media_type"],
                "storage_path": r["storage_path"],
                "original_filename": r["original_filename"],
                "file_size_bytes": r["file_size_bytes"],
                "mime_type": r["mime_type"],
                "user_id": str(r["user_id"]) if r["user_id"] else None,
                "session_id": str(r["session_id"]) if r["session_id"] else None,
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ]

    async def artifact_merge_metadata(
        self, artifact_id: str, patch: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Merge metadata into artifact. Returns {\"id\", \"metadata\"} or None if row missing."""
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")

        aid = to_uuid(artifact_id)
        if not aid:
            return None

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE artifacts
                SET metadata = COALESCE(metadata, '{}') || $2::jsonb,
                    updated_at = now()
                WHERE id = $1
                RETURNING id, metadata
                """,
                aid,
                json_dumps(patch),
            )

        if not row:
            return None

        meta = row["metadata"]
        if hasattr(meta, "items"):
            meta = dict(meta)
        return {"id": str(row["id"]), "metadata": meta}

    async def artifact_update_storage_path(self, artifact_id: str, storage_path: str) -> bool:
        """Update the storage_path field of an artifact (e.g. after MinIO/local promotion)."""
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE artifacts
                SET storage_path = $2,
                    updated_at = now()
                WHERE id = $1
                RETURNING id
                """,
                to_uuid(artifact_id),
                storage_path,
            )
            return bool(row)

    async def artifact_update_storage_identity(
        self,
        artifact_id: str,
        *,
        storage_path: Optional[str] = None,
        original_filename: Optional[str] = None,
        media_type: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> bool:
        """Update storage_path and/or display identity after extension detection."""
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")

        sets: List[str] = ["updated_at = now()"]
        args: List[Any] = [to_uuid(artifact_id)]
        idx = 2
        if storage_path is not None:
            sets.append(f"storage_path = ${idx}")
            args.append(storage_path)
            idx += 1
        if original_filename is not None:
            sets.append(f"original_filename = ${idx}")
            args.append(original_filename)
            idx += 1
        if media_type is not None:
            sets.append(f"media_type = ${idx}")
            args.append(media_type)
            idx += 1
        if mime_type is not None:
            sets.append(f"mime_type = ${idx}")
            args.append(mime_type)
            idx += 1
        if len(args) == 1:
            return False

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE artifacts
                SET {", ".join(sets)}
                WHERE id = $1
                RETURNING id
                """,
                *args,
            )
            return bool(row)

    @staticmethod
    def _artifact_in_session_sql(alias: str, param_idx: int) -> str:
        """Match artifacts owned by a chat session (primary or linked)."""
        a = alias.strip() or "artifacts"
        return f"""(
            {a}.session_id = ${param_idx}
            OR EXISTS (
                SELECT 1
                FROM jsonb_array_elements_text(
                    COALESCE({a}.metadata->'attached_sessions', '[]'::jsonb)
                ) AS elem
                WHERE elem = ${param_idx}::text
            )
        )"""

    async def artifact_attach_to_session(
        self,
        *,
        artifact_id: str,
        session_id: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Link an existing DB artifact into a chat session workspace."""
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")

        access = await self.artifact_is_accessible(user_id=user_id, artifact_id=artifact_id)
        if not access.get("accessible"):
            return None

        aid = to_uuid(artifact_id)
        sid = to_uuid(session_id)
        if not aid or not sid:
            return None

        artifact = await self.artifact_get(artifact_id)
        if not artifact:
            return None

        current_session = str(artifact.get("session_id") or "")
        target_session = str(session_id)
        meta = artifact.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}

        attached = [str(x) for x in (meta.get("attached_sessions") or []) if x]
        if current_session == target_session or target_session in attached:
            return artifact

        async with self.pool.acquire() as conn:
            if not current_session:
                await conn.execute(
                    """
                    UPDATE artifacts
                    SET session_id = $2,
                        updated_at = now()
                    WHERE id = $1
                    """,
                    aid,
                    sid,
                )
            else:
                merged = sorted(set(attached + [target_session]))
                await conn.execute(
                    """
                    UPDATE artifacts
                    SET metadata = COALESCE(metadata, '{}'::jsonb)
                        || jsonb_build_object('attached_sessions', $2::jsonb),
                        updated_at = now()
                    WHERE id = $1
                    """,
                    aid,
                    json_dumps(merged),
                )

        return await self.artifact_get(artifact_id)

    def _artifact_row_to_dict(self, row: Any) -> Dict[str, Any]:
        meta = row["metadata"]
        if hasattr(meta, "items"):
            meta = dict(meta)
        return {
            "id": str(row["id"]),
            "media_type": row["media_type"],
            "storage_path": row["storage_path"],
            "external_ref": row["external_ref"],
            "checksum": row["checksum"],
            "metadata": meta,
            "folder_id": str(row["folder_id"]) if row["folder_id"] else None,
            "original_filename": row["original_filename"],
            "file_size_bytes": row["file_size_bytes"],
            "mime_type": row["mime_type"],
            "user_id": str(row["user_id"]) if row["user_id"] else None,
            "session_id": str(row["session_id"]) if row["session_id"] else None,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        }

    async def artifact_get_by_path(self, storage_path: str) -> Optional[Dict[str, Any]]:
        """Exact match on artifacts.storage_path."""
        if not self.pool:
            return None
        path = (storage_path or "").strip()
        if not path:
            return None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, media_type, storage_path, external_ref, checksum,
                       metadata, folder_id, original_filename, file_size_bytes,
                       mime_type, user_id, session_id, created_at, updated_at
                FROM artifacts
                WHERE storage_path = $1
                LIMIT 1
                """,
                path,
            )
        if not row:
            return None
        return self._artifact_row_to_dict(row)

    @staticmethod
    def _escape_ilike(value: str) -> str:
        return (value or "").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    @staticmethod
    def _import_locator_candidates(path_hint: str) -> List[tuple[str, str]]:
        """Derive (relative_path, filename) pairs from absolute/temp paths."""
        norm = (path_hint or "").replace("\\", "/").strip()
        if not norm:
            return []
        out: List[tuple[str, str]] = []
        basename = Path(norm).name
        if basename:
            out.append((basename, basename))
        if "cases_temp/" in norm:
            tail = norm.split("cases_temp/", 1)[1]
            parts = [p for p in tail.split("/") if p]
            if len(parts) >= 2:
                rel = "/".join(parts[1:])
                fn = parts[-1]
                if rel and fn:
                    out.append((rel, fn))
            elif len(parts) == 1:
                out.append((parts[0], parts[0]))
        if "cases/" in norm:
            tail = norm.split("cases/", 1)[1]
            parts = [p for p in tail.split("/") if p]
            if len(parts) >= 2:
                rel = "/".join(parts[1:])
                fn = parts[-1]
                if rel and fn:
                    out.append((rel, fn))
        # De-dupe preserving order
        seen: set[tuple[str, str]] = set()
        unique: List[tuple[str, str]] = []
        for pair in out:
            if pair not in seen:
                seen.add(pair)
                unique.append(pair)
        return unique

    async def artifact_find_candidates_by_path_hint(
        self,
        path_hint: str,
        *,
        user_id: Optional[str] = None,
        filename: Optional[str] = None,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """Return artifact rows matching a filesystem/storage path hint (best-effort)."""
        if not self.pool:
            return []
        raw = (path_hint or "").strip()
        if not raw and not (filename or "").strip():
            return []

        basename = (filename or Path(raw).name if raw else "").strip()
        candidates: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        async def _add_row(row: Any) -> None:
            if not row:
                return
            aid = str(row["id"])
            if aid in seen_ids:
                return
            seen_ids.add(aid)
            candidates.append(self._artifact_row_to_dict(row))

        async with self.pool.acquire() as conn:
            if raw:
                exact = await conn.fetchrow(
                    """
                    SELECT id, media_type, storage_path, external_ref, checksum,
                           metadata, folder_id, original_filename, file_size_bytes,
                           mime_type, user_id, session_id, created_at, updated_at
                    FROM artifacts WHERE storage_path = $1 LIMIT 1
                    """,
                    raw,
                )
                await _add_row(exact)

                esc_tail = self._escape_ilike(raw[-240:])
                rows = await conn.fetch(
                    f"""
                    SELECT id, media_type, storage_path, external_ref, checksum,
                           metadata, folder_id, original_filename, file_size_bytes,
                           mime_type, user_id, session_id, created_at, updated_at
                    FROM artifacts
                    WHERE storage_path ILIKE '%' || $1 || '%' ESCAPE '\\'
                    ORDER BY
                      CASE WHEN storage_path = $2 THEN 0
                           WHEN storage_path LIKE '%' || $3 THEN 1
                           ELSE 2 END,
                      created_at DESC
                    LIMIT $4
                    """,
                    esc_tail,
                    raw,
                    self._escape_ilike(basename) if basename else esc_tail,
                    limit,
                )
                for row in rows:
                    await _add_row(row)

            if basename:
                esc_bn = self._escape_ilike(basename)
                rows = await conn.fetch(
                    f"""
                    SELECT id, media_type, storage_path, external_ref, checksum,
                           metadata, folder_id, original_filename, file_size_bytes,
                           mime_type, user_id, session_id, created_at, updated_at
                    FROM artifacts
                    WHERE original_filename = $1
                       OR original_filename ILIKE '%' || $2 || '%' ESCAPE '\\'
                       OR storage_path ILIKE '%' || $2 || '%' ESCAPE '\\'
                    ORDER BY created_at DESC
                    LIMIT $3
                    """,
                    basename,
                    esc_bn,
                    limit,
                )
                for row in rows:
                    await _add_row(row)

            uid = to_uuid(user_id) if user_id else None
            for rel, fn in self._import_locator_candidates(raw):
                if uid:
                    row = await conn.fetchrow(
                        """
                        SELECT id, media_type, storage_path, external_ref, checksum,
                               metadata, folder_id, original_filename, file_size_bytes,
                               mime_type, user_id, session_id, created_at, updated_at
                        FROM artifacts
                        WHERE user_id = $1
                          AND COALESCE(metadata->>'case_importer_rel_path', '') = $2
                          AND COALESCE(metadata->>'case_importer_filename', '') = $3
                        ORDER BY created_at ASC
                        LIMIT 1
                        """,
                        uid,
                        rel,
                        fn,
                    )
                else:
                    row = await conn.fetchrow(
                        """
                        SELECT id, media_type, storage_path, external_ref, checksum,
                               metadata, folder_id, original_filename, file_size_bytes,
                               mime_type, user_id, session_id, created_at, updated_at
                        FROM artifacts
                        WHERE COALESCE(metadata->>'case_importer_rel_path', '') = $1
                          AND COALESCE(metadata->>'case_importer_filename', '') = $2
                        ORDER BY created_at ASC
                        LIMIT 1
                        """,
                        rel,
                        fn,
                    )
                await _add_row(row)

        return candidates[:limit]

    _ARTIFACT_SELECT = """
        SELECT id, media_type, storage_path, external_ref, checksum,
               metadata, folder_id, original_filename, file_size_bytes,
               mime_type, user_id, session_id, created_at, updated_at
        FROM artifacts
    """

    async def artifact_search_by_filters(
        self,
        *,
        artifact_id: Optional[str] = None,
        case_id: Optional[str] = None,
        path: Optional[str] = None,
        filename: Optional[str] = None,
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        """Search artifacts by id, case, path, and/or filename (deduped, capped)."""
        if not self.pool:
            return []

        cap = max(1, min(int(limit or 25), 50))
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()

        def _push(row: Any) -> None:
            if not row:
                return
            aid = str(row["id"])
            if aid in seen:
                return
            seen.add(aid)
            out.append(self._artifact_row_to_dict(row))

        aid = (artifact_id or "").strip()
        if aid:
            row = await self.artifact_get(aid)
            if row:
                return [row]

        cid = (case_id or "").strip()
        fn = (filename or "").strip()
        path_hint = (path or "").strip()

        if path_hint or fn:
            for row in await self.artifact_find_candidates_by_path_hint(
                path_hint,
                filename=fn or None,
                limit=cap,
            ):
                if len(out) >= cap:
                    break
                if row.get("id") not in seen:
                    seen.add(str(row["id"]))
                    out.append(row)

        if cid or fn:
            conditions: List[str] = []
            params: List[Any] = []
            idx = 1

            if cid:
                esc_cid = self._escape_ilike(cid)
                conditions.append(
                    f"""(
                        COALESCE(metadata->>'case_id', '') = ${idx}
                        OR COALESCE(metadata->>'case_importer_last_case_id', '') = ${idx}
                        OR storage_path ILIKE '%' || ${idx + 1} || '%' ESCAPE '\\'
                    )"""
                )
                params.extend([cid, esc_cid])
                idx += 2

            if fn:
                esc_fn = self._escape_ilike(fn)
                conditions.append(
                    f"""(
                        original_filename = ${idx}
                        OR original_filename ILIKE '%' || ${idx + 1} || '%' ESCAPE '\\'
                        OR COALESCE(metadata->>'case_importer_filename', '') = ${idx}
                        OR COALESCE(metadata->>'case_importer_filename', '') ILIKE '%' || ${idx + 1} || '%' ESCAPE '\\'
                        OR storage_path ILIKE '%' || ${idx + 1} || '%' ESCAPE '\\'
                    )"""
                )
                params.extend([fn, esc_fn])
                idx += 2

            where = " AND ".join(conditions)
            params.append(cap)
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    {self._ARTIFACT_SELECT}
                    WHERE {where}
                    ORDER BY created_at DESC
                    LIMIT ${idx}
                    """,
                    *params,
                )
            for row in rows:
                if len(out) >= cap:
                    break
                _push(row)

        return out[:cap]

    async def case_merge_metadata(self, case_id: str, patch: Dict[str, Any]) -> bool:
        """Merge metadata jsonb into a case row (used for case_root, etc.)."""
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE cases
                SET metadata = COALESCE(metadata, '{}') || $2::jsonb,
                    updated_at = now()
                WHERE id = $1
                RETURNING id
                """,
                case_id,
                json_dumps(patch),
            )
            return bool(row)

    # ------------------------------------------------------------------ Accessible artifacts (RBAC)
    async def artifact_is_accessible(
        self,
        *,
        user_id: str,
        artifact_id: str,
        action: str = "read",
    ) -> Dict[str, Any]:
        """Check if a user can access an artifact for ``action`` (read/write/delete).

        Owner and group membership grant access; a per-user file override grants
        access only when its ``permissions`` array contains the requested action.
        """
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")

        uid = to_uuid(user_id)
        aid = to_uuid(artifact_id)
        action = (action or "read").lower()

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    EXISTS (
                        SELECT 1 FROM artifacts a
                        WHERE a.id = $2 AND a.user_id = $1
                    ) AS is_owner,
                    EXISTS (
                        SELECT 1
                        FROM group_memberships gm
                        JOIN group_files gf ON gf.group_id = gm.group_id
                        WHERE gm.user_id = $1 AND gf.artifact_id = $2
                    ) AS is_group,
                    EXISTS (
                        SELECT 1
                        FROM user_file_overrides ufo
                        WHERE ufo.user_id = $1
                          AND ufo.artifact_id = $2
                          AND $3 = ANY(ufo.permissions)
                          AND (ufo.expires_at IS NULL OR ufo.expires_at > now())
                    ) AS is_override
                """,
                uid,
                aid,
                action,
            )

        access = {
            "is_owner": bool(row["is_owner"]) if row else False,
            "is_group": bool(row["is_group"]) if row else False,
            "is_override": bool(row["is_override"]) if row else False,
        }
        return {"accessible": any(access.values()), "access": access}

    # ------------------------------------------------------------------ Accessible cases (RBAC)
    async def case_is_accessible(
        self,
        *,
        user_id: str,
        case_id: str,
    ) -> Dict[str, Any]:
        """
        Check if a user can read a case (owner / group_cases / case.group_id / override).
        Same data-driven model as ``artifact_is_accessible``.
        """
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")

        uid = to_uuid(user_id)
        cid = (case_id or "").strip()
        if not cid:
            return {
                "accessible": False,
                "access": {
                    "is_owner": False,
                    "is_group_assigned": False,
                    "is_group_member": False,
                    "is_override": False,
                },
            }

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    EXISTS (
                        SELECT 1 FROM cases c
                        WHERE c.id = $2 AND c.user_id = $1::text
                    ) AS is_owner,
                    EXISTS (
                        SELECT 1
                        FROM group_memberships gm
                        JOIN group_cases gc ON gc.group_id = gm.group_id
                        WHERE gm.user_id = $3 AND gc.case_id = $2
                    ) AS is_group_assigned,
                    EXISTS (
                        SELECT 1
                        FROM group_memberships gm
                        JOIN cases c ON c.id = $2
                        WHERE gm.user_id = $3
                          AND c.group_id IS NOT NULL
                          AND c.group_id = gm.group_id::text
                    ) AS is_group_member,
                    EXISTS (
                        SELECT 1
                        FROM user_case_overrides uco
                        WHERE uco.user_id = $3
                          AND uco.case_id = $2
                          AND 'read' = ANY(uco.permissions)
                          AND (uco.expires_at IS NULL OR uco.expires_at > now())
                    ) AS is_override
                """,
                user_id,
                cid,
                uid,
            )

        access = {
            "is_owner": bool(row["is_owner"]) if row else False,
            "is_group_assigned": bool(row["is_group_assigned"]) if row else False,
            "is_group_member": bool(row["is_group_member"]) if row else False,
            "is_override": bool(row["is_override"]) if row else False,
        }
        is_group = access["is_group_assigned"] or access["is_group_member"]
        return {
            "accessible": any(
                [access["is_owner"], is_group, access["is_override"]]
            ),
            "access": {**access, "is_group": is_group},
        }

    async def case_accessible_list(
        self,
        *,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        session_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List cases visible to ``user_id`` via group-RBAC tables."""
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")

        uid = to_uuid(user_id)
        params: List[Any] = [user_id, uid]
        idx = 3
        filters: List[str] = []

        if session_id:
            filters.append(f"c.session_id = ${idx}")
            params.append(session_id)
            idx += 1
        if status:
            filters.append(f"c.status = ${idx}")
            params.append(status)
            idx += 1

        where_filters = " AND ".join(filters) if filters else "TRUE"
        params.extend([limit, offset])
        limit_param = idx
        offset_param = idx + 1

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                WITH accessible AS (
                    SELECT c.*
                    FROM cases c
                    WHERE (
                        c.user_id = $1::text
                        OR EXISTS (
                            SELECT 1
                            FROM group_memberships gm
                            JOIN group_cases gc ON gc.group_id = gm.group_id
                            WHERE gm.user_id = $2 AND gc.case_id = c.id
                        )
                        OR EXISTS (
                            SELECT 1
                            FROM group_memberships gm
                            WHERE gm.user_id = $2
                              AND c.group_id IS NOT NULL
                              AND c.group_id = gm.group_id::text
                        )
                        OR EXISTS (
                            SELECT 1
                            FROM user_case_overrides uco
                            WHERE uco.user_id = $2
                              AND uco.case_id = c.id
                              AND 'read' = ANY(uco.permissions)
                              AND (uco.expires_at IS NULL OR uco.expires_at > now())
                        )
                    )
                )
                SELECT
                    c.id, c.name, c.user_id, c.group_id, c.session_id,
                    c.status, c.progress, c.metadata, c.created_at, c.updated_at,
                    (c.user_id = $1::text) AS access_is_owner,
                    EXISTS (
                        SELECT 1
                        FROM group_memberships gm
                        JOIN group_cases gc ON gc.group_id = gm.group_id
                        WHERE gm.user_id = $2 AND gc.case_id = c.id
                    ) AS access_is_group_assigned,
                    EXISTS (
                        SELECT 1
                        FROM group_memberships gm
                        WHERE gm.user_id = $2
                          AND c.group_id IS NOT NULL
                          AND c.group_id = gm.group_id::text
                    ) AS access_is_group_member
                FROM accessible c
                WHERE {where_filters}
                ORDER BY c.created_at DESC
                LIMIT ${limit_param} OFFSET ${offset_param}
                """,
                *params,
            )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Helpers shared across `accessible*` queries
    # ------------------------------------------------------------------
    @staticmethod
    def _virtual_key_expr(alias: str = "a") -> str:
        """SQL expression that yields the S3-like virtual key for storage_path."""
        return (
            f"CASE "
            f"WHEN {alias}.storage_path LIKE 'minio://%' "
            f"  THEN regexp_replace({alias}.storage_path, '^minio://[^/]+/', '') "
            f"WHEN {alias}.storage_path LIKE '/data/uploads/%' "
            f"  THEN regexp_replace({alias}.storage_path, '^/data/uploads/', '') "
            f"ELSE {alias}.storage_path END"
        )

    @staticmethod
    def _virtual_dir_expr(alias: str = "a") -> str:
        """SQL expression that yields the directory portion of the virtual key (no trailing /)."""
        key = PostgresBackend._virtual_key_expr(alias)
        return (
            f"CASE WHEN position('/' in reverse({key})) > 0 "
            f"THEN substring({key} from 1 for char_length({key}) - position('/' in reverse({key}))) "
            f"ELSE '' END"
        )

    def _accessible_list_filters(
        self,
        *,
        params: List[Any],
        idx: int,
        folder_id: Optional[str] = None,
        media_type: Optional[str] = None,
        session_id: Optional[str] = None,
        search: Optional[str] = None,
        ownership: str = "any",
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
        size_min: Optional[int] = None,
        size_max: Optional[int] = None,
        mime_type: Optional[str] = None,
        mime_types: Optional[Sequence[str]] = None,
        tags_any: Optional[Sequence[str]] = None,
        tags_all: Optional[Sequence[str]] = None,
        folder_prefix: Optional[str] = None,
        folder_prefixes: Optional[Sequence[str]] = None,
        uploaded_by: Optional[Sequence[str]] = None,
    ) -> tuple[List[str], int]:
        """
        Build WHERE conditions for filters on the `accessible` CTE.
        Mutates `params` in-place; returns (conditions, next_idx).
        """
        conditions: List[str] = []

        if folder_id:
            conditions.append(f"a.folder_id = ${idx}")
            params.append(to_uuid(folder_id))
            idx += 1
        if media_type:
            conditions.append(f"a.media_type = ${idx}::media_kind")
            params.append(media_type)
            idx += 1
        if session_id:
            conditions.append(self._artifact_in_session_sql("a", idx))
            params.append(to_uuid(session_id))
            idx += 1
        if created_from:
            conditions.append(f"a.created_at >= ${idx}::timestamptz")
            params.append(created_from)
            idx += 1
        if created_to:
            conditions.append(f"a.created_at <= ${idx}::timestamptz")
            params.append(created_to)
            idx += 1
        if size_min is not None:
            conditions.append(f"a.file_size_bytes >= ${idx}")
            params.append(int(size_min))
            idx += 1
        if size_max is not None:
            conditions.append(f"a.file_size_bytes <= ${idx}")
            params.append(int(size_max))
            idx += 1
        if mime_type:
            conditions.append(f"a.mime_type = ${idx}")
            params.append(mime_type)
            idx += 1
        m_types = [str(m).strip() for m in (mime_types or []) if str(m).strip()]
        if m_types:
            conditions.append(f"a.mime_type = ANY(${idx}::text[])")
            params.append(m_types)
            idx += 1
        if search:
            s = (search or "").strip()
            if s:
                conditions.append(
                    f"(a.original_filename ILIKE ${idx} OR a.storage_path ILIKE ${idx})"
                )
                params.append(f"%{s}%")
                idx += 1

        any_tags = [str(t).strip() for t in (tags_any or []) if str(t).strip()]
        if any_tags:
            conditions.append(f"(a.metadata->'tags') ?| ${idx}::text[]")
            params.append(any_tags)
            idx += 1
        all_tags = [str(t).strip() for t in (tags_all or []) if str(t).strip()]
        if all_tags:
            conditions.append(f"(a.metadata->'tags') ?& ${idx}::text[]")
            params.append(all_tags)
            idx += 1

        prefixes: List[str] = []
        if folder_prefix is not None and str(folder_prefix).strip():
            prefixes.append(str(folder_prefix).strip())
        for p in (folder_prefixes or []):
            sp = str(p or "").strip()
            if sp:
                prefixes.append(sp)
        if prefixes:
            key_expr = self._virtual_key_expr("a")
            sub_clauses: List[str] = []
            for pfx in prefixes:
                norm = pfx.lstrip("/").rstrip("/")
                # Match folder root (==) or any child path under it (LIKE 'norm/%')
                sub_clauses.append(
                    f"({key_expr} = ${idx} OR {key_expr} LIKE ${idx + 1})"
                )
                params.extend([norm, f"{norm}/%"])
                idx += 2
            conditions.append("(" + " OR ".join(sub_clauses) + ")")

        uploaders = [str(u).strip() for u in (uploaded_by or []) if str(u).strip()]
        if uploaders:
            conditions.append(f"a.user_id = ANY(${idx}::uuid[])")
            params.append([to_uuid(u) for u in uploaders])
            idx += 1

        own = (ownership or "any").strip().lower()
        if own == "owner":
            conditions.append("a.user_id = $1")
        elif own == "shared":
            conditions.append("a.user_id IS DISTINCT FROM $1")

        return conditions, idx

    async def artifact_accessible_list(
        self,
        *,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        sort_by: Optional[str] = None,
        sort_dir: Optional[str] = None,
        folder_id: Optional[str] = None,
        media_type: Optional[str] = None,
        session_id: Optional[str] = None,
        search: Optional[str] = None,
        ownership: str = "any",  # any|owner|shared
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
        size_min: Optional[int] = None,
        size_max: Optional[int] = None,
        mime_type: Optional[str] = None,
        mime_types: Optional[Sequence[str]] = None,
        tags_any: Optional[Sequence[str]] = None,
        tags_all: Optional[Sequence[str]] = None,
        folder_prefix: Optional[str] = None,
        folder_prefixes: Optional[Sequence[str]] = None,
        uploaded_by: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")

        uid = to_uuid(user_id)

        # Sorting (whitelist)
        sb = (sort_by or "created_at").strip().lower()
        sd = (sort_dir or "desc").strip().lower()
        if sd not in ("asc", "desc"):
            sd = "desc"
        order_map = {
            "created_at": "a.created_at",
            "updated_at": "COALESCE(a.updated_at, a.created_at)",
            "name": "a.original_filename",
            "original_filename": "a.original_filename",
            "size": "a.file_size_bytes",
            "file_size_bytes": "a.file_size_bytes",
            "mime_type": "a.mime_type",
            "media_type": "a.media_type",
        }
        order_col = order_map.get(sb, "a.created_at")
        order_clause = f"{order_col} {sd}, a.id DESC"

        params: List[Any] = [uid]
        conditions, idx = self._accessible_list_filters(
            params=params,
            idx=2,
            folder_id=folder_id,
            media_type=media_type,
            session_id=session_id,
            search=search,
            ownership=ownership,
            created_from=created_from,
            created_to=created_to,
            size_min=size_min,
            size_max=size_max,
            mime_type=mime_type,
            mime_types=mime_types,
            tags_any=tags_any,
            tags_all=tags_all,
            folder_prefix=folder_prefix,
            folder_prefixes=folder_prefixes,
            uploaded_by=uploaded_by,
        )

        where_filters = " AND ".join(conditions) if conditions else "TRUE"

        params.extend([limit, offset])
        limit_param = idx
        offset_param = idx + 1

        v_dir = self._virtual_dir_expr("a")

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                WITH accessible AS (
                    SELECT a.*
                    FROM artifacts a
                    WHERE (
                        a.user_id = $1
                        OR EXISTS (
                            SELECT 1
                            FROM group_memberships gm
                            JOIN group_files gf ON gf.group_id = gm.group_id
                            WHERE gm.user_id = $1 AND gf.artifact_id = a.id
                        )
                        OR EXISTS (
                            SELECT 1
                            FROM user_file_overrides ufo
                            WHERE ufo.user_id = $1
                              AND ufo.artifact_id = a.id
                              AND 'read' = ANY(ufo.permissions)
                              AND (ufo.expires_at IS NULL OR ufo.expires_at > now())
                        )
                    )
                )
                SELECT
                    a.id, a.media_type, a.storage_path, a.original_filename,
                    a.file_size_bytes, a.mime_type, a.user_id, a.session_id,
                    a.folder_id, a.checksum,
                    a.created_at, a.updated_at,
                    a.metadata,
                    {v_dir} AS folder_path,
                    COUNT(*) OVER() AS total_count,
                    (a.user_id = $1) AS access_is_owner,
                    EXISTS (
                        SELECT 1
                        FROM group_memberships gm
                        JOIN group_files gf ON gf.group_id = gm.group_id
                        WHERE gm.user_id = $1 AND gf.artifact_id = a.id
                    ) AS access_is_group,
                    EXISTS (
                        SELECT 1
                        FROM user_file_overrides ufo
                        WHERE ufo.user_id = $1
                          AND ufo.artifact_id = a.id
                          AND 'read' = ANY(ufo.permissions)
                          AND (ufo.expires_at IS NULL OR ufo.expires_at > now())
                    ) AS access_is_override
                FROM accessible a
                WHERE {where_filters}
                ORDER BY {order_clause}
                LIMIT ${limit_param} OFFSET ${offset_param}
                """,
                *params,
            )

        total_count = int(rows[0]["total_count"]) if rows else 0
        items: List[Dict[str, Any]] = []
        for r in rows:
            meta = r["metadata"] or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            tags = meta.get("tags") if isinstance(meta, dict) else None
            if not isinstance(tags, list):
                tags = []
            items.append(
                {
                    "id": str(r["id"]),
                    "media_type": r["media_type"],
                    "storage_path": r["storage_path"],
                    "original_filename": r["original_filename"],
                    "file_size_bytes": r["file_size_bytes"],
                    "mime_type": r["mime_type"],
                    "user_id": str(r["user_id"]) if r["user_id"] else None,
                    "session_id": str(r["session_id"]) if r["session_id"] else None,
                    "folder_id": str(r["folder_id"]) if r["folder_id"] else None,
                    "folder_path": r["folder_path"] or "",
                    "checksum": r["checksum"],
                    "tags": tags,
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                    "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
                    "metadata": meta if isinstance(meta, dict) else {},
                    "access": {
                        "is_owner": bool(r["access_is_owner"]),
                        "is_group": bool(r["access_is_group"]),
                        "is_override": bool(r["access_is_override"]),
                    },
                }
            )
        return {"items": items, "total_count": total_count}

    async def artifact_accessible_stats(
        self,
        *,
        user_id: str,
        folder_id: Optional[str] = None,
        media_type: Optional[str] = None,
        session_id: Optional[str] = None,
        search: Optional[str] = None,
        ownership: str = "any",
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
        size_min: Optional[int] = None,
        size_max: Optional[int] = None,
        mime_type: Optional[str] = None,
        mime_types: Optional[Sequence[str]] = None,
        tags_any: Optional[Sequence[str]] = None,
        tags_all: Optional[Sequence[str]] = None,
        folder_prefix: Optional[str] = None,
        folder_prefixes: Optional[Sequence[str]] = None,
        uploaded_by: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Total bytes/count grouped by media_type for accessible artifacts."""
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        uid = to_uuid(user_id)

        params: List[Any] = [uid]
        conditions, idx = self._accessible_list_filters(
            params=params,
            idx=2,
            folder_id=folder_id,
            media_type=media_type,
            session_id=session_id,
            search=search,
            ownership=ownership,
            created_from=created_from,
            created_to=created_to,
            size_min=size_min,
            size_max=size_max,
            mime_type=mime_type,
            mime_types=mime_types,
            tags_any=tags_any,
            tags_all=tags_all,
            folder_prefix=folder_prefix,
            folder_prefixes=folder_prefixes,
            uploaded_by=uploaded_by,
        )

        where_filters = " AND ".join(conditions) if conditions else "TRUE"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                WITH accessible AS (
                    SELECT a.*
                    FROM artifacts a
                    WHERE (
                        a.user_id = $1
                        OR EXISTS (
                            SELECT 1
                            FROM group_memberships gm
                            JOIN group_files gf ON gf.group_id = gm.group_id
                            WHERE gm.user_id = $1 AND gf.artifact_id = a.id
                        )
                        OR EXISTS (
                            SELECT 1
                            FROM user_file_overrides ufo
                            WHERE ufo.user_id = $1
                              AND ufo.artifact_id = a.id
                              AND 'read' = ANY(ufo.permissions)
                              AND (ufo.expires_at IS NULL OR ufo.expires_at > now())
                        )
                    )
                )
                SELECT
                    COALESCE(a.media_type::text, 'other') AS media_type,
                    COUNT(*) AS file_count,
                    COALESCE(SUM(a.file_size_bytes), 0)::bigint AS total_bytes
                FROM accessible a
                WHERE {where_filters}
                GROUP BY 1
                ORDER BY total_bytes DESC
                """,
                *params,
            )

        by_type: Dict[str, Any] = {}
        total_bytes = 0
        total_files = 0
        for r in rows:
            mt = str(r["media_type"] or "other")
            fc = int(r["file_count"] or 0)
            tb = int(r["total_bytes"] or 0)
            by_type[mt] = {"bytes": tb, "count": fc}
            total_bytes += tb
            total_files += fc
        return {"total": {"bytes": total_bytes, "count": total_files}, "by_type": by_type}

    async def accessible_folders(
        self,
        *,
        user_id: str,
        prefix: str = "",
        delimiter: str = "/",
        limit: int = 200,
        offset: int = 0,
        ownership: str = "any",
        folder_id: Optional[str] = None,
        media_type: Optional[str] = None,
        session_id: Optional[str] = None,
        search: Optional[str] = None,
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
        size_min: Optional[int] = None,
        size_max: Optional[int] = None,
        mime_type: Optional[str] = None,
        mime_types: Optional[Sequence[str]] = None,
        tags_any: Optional[Sequence[str]] = None,
        tags_all: Optional[Sequence[str]] = None,
        uploaded_by: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Virtual folder listing over artifact storage_path keys (S3-like prefix+delimiter).
        Returns immediate child folders under prefix and immediate-file aggregates.
        Accepts the full set of `artifact_accessible_list` filters; folder counts
        reflect those filters.
        """
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")

        uid = to_uuid(user_id)
        delim = (delimiter or "/").strip()
        if len(delim) != 1:
            delim = "/"

        pfx = (prefix or "").lstrip(delim)
        if pfx and not pfx.endswith(delim):
            pfx = pfx + delim

        key_expr = self._virtual_key_expr("a")

        # Build common filter clauses (excluding folder_prefix / prefix; we apply prefix below).
        params: List[Any] = [uid, pfx]
        idx = 3
        conditions, idx = self._accessible_list_filters(
            params=params,
            idx=idx,
            folder_id=folder_id,
            media_type=media_type,
            session_id=session_id,
            search=search,
            ownership=ownership,
            created_from=created_from,
            created_to=created_to,
            size_min=size_min,
            size_max=size_max,
            mime_type=mime_type,
            mime_types=mime_types,
            tags_any=tags_any,
            tags_all=tags_all,
            folder_prefix=None,  # the prefix is handled explicitly via $2
            folder_prefixes=None,
            uploaded_by=uploaded_by,
        )

        extra_clause = (" AND " + " AND ".join(conditions)) if conditions else ""

        params.extend([limit, offset])
        limit_param = idx
        offset_param = idx + 1

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                WITH accessible AS (
                    SELECT a.*
                    FROM artifacts a
                    WHERE (
                        a.user_id = $1
                        OR EXISTS (
                            SELECT 1
                            FROM group_memberships gm
                            JOIN group_files gf ON gf.group_id = gm.group_id
                            WHERE gm.user_id = $1 AND gf.artifact_id = a.id
                        )
                        OR EXISTS (
                            SELECT 1
                            FROM user_file_overrides ufo
                            WHERE ufo.user_id = $1
                              AND ufo.artifact_id = a.id
                              AND 'read' = ANY(ufo.permissions)
                              AND (ufo.expires_at IS NULL OR ufo.expires_at > now())
                        )
                    )
                ),
                scoped AS (
                    SELECT
                        a.file_size_bytes,
                        {key_expr} AS key
                    FROM accessible a
                    WHERE ({key_expr} LIKE $2 || '%') {extra_clause}
                ),
                children AS (
                    SELECT
                        split_part(substr(key, char_length($2) + 1), '{delim}', 1) AS child,
                        position('{delim}' in substr(key, char_length($2) + 1)) > 0 AS is_folder,
                        file_size_bytes
                    FROM scoped
                    WHERE key IS NOT NULL AND key <> $2
                )
                SELECT
                    child AS name,
                    ($2 || child || '{delim}') AS prefix,
                    COUNT(*) FILTER (WHERE is_folder) AS file_count_recursive,
                    COALESCE(SUM(file_size_bytes) FILTER (WHERE is_folder), 0)::bigint AS total_bytes_recursive
                FROM children
                WHERE is_folder AND child <> ''
                GROUP BY child
                ORDER BY child ASC
                LIMIT ${limit_param} OFFSET ${offset_param}
                """,
                *params,
            )

            # Immediate files under prefix (no further `/`)
            file_summary = await conn.fetchrow(
                f"""
                WITH accessible AS (
                    SELECT a.*
                    FROM artifacts a
                    WHERE (
                        a.user_id = $1
                        OR EXISTS (
                            SELECT 1
                            FROM group_memberships gm
                            JOIN group_files gf ON gf.group_id = gm.group_id
                            WHERE gm.user_id = $1 AND gf.artifact_id = a.id
                        )
                        OR EXISTS (
                            SELECT 1
                            FROM user_file_overrides ufo
                            WHERE ufo.user_id = $1
                              AND ufo.artifact_id = a.id
                              AND 'read' = ANY(ufo.permissions)
                              AND (ufo.expires_at IS NULL OR ufo.expires_at > now())
                        )
                    )
                ),
                scoped AS (
                    SELECT
                        a.file_size_bytes,
                        {key_expr} AS key
                    FROM accessible a
                    WHERE ({key_expr} LIKE $2 || '%') {extra_clause}
                )
                SELECT
                    COUNT(*) FILTER (
                        WHERE position('{delim}' in substr(key, char_length($2) + 1)) = 0
                    ) AS immediate_file_count,
                    COALESCE(SUM(file_size_bytes) FILTER (
                        WHERE position('{delim}' in substr(key, char_length($2) + 1)) = 0
                    ), 0)::bigint AS immediate_file_bytes,
                    COUNT(*) AS recursive_file_count,
                    COALESCE(SUM(file_size_bytes), 0)::bigint AS recursive_file_bytes
                FROM scoped
                """,
                *params[:idx - 1],  # drop limit/offset for the summary query
            )

        return {
            "prefix": pfx,
            "delimiter": delim,
            "folders": [
                {
                    "name": r["name"],
                    "prefix": r["prefix"],
                    "file_count_recursive": int(r["file_count_recursive"] or 0),
                    "total_bytes_recursive": int(r["total_bytes_recursive"] or 0),
                }
                for r in rows
            ],
            "summary": {
                "immediate_file_count": int(file_summary["immediate_file_count"] or 0),
                "immediate_file_bytes": int(file_summary["immediate_file_bytes"] or 0),
                "recursive_file_count": int(file_summary["recursive_file_count"] or 0),
                "recursive_file_bytes": int(file_summary["recursive_file_bytes"] or 0),
            } if file_summary else None,
        }

    async def accessible_facets(
        self,
        *,
        user_id: str,
        search: Optional[str] = None,
        ownership: str = "any",
        top_mimes: int = 30,
        top_tags: int = 30,
        top_uploaders: int = 10,
        top_sessions: int = 10,
        date_buckets: str = "month",  # day | week | month
        folder_id: Optional[str] = None,
        media_type: Optional[str] = None,
        session_id: Optional[str] = None,
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
        size_min: Optional[int] = None,
        size_max: Optional[int] = None,
        mime_type: Optional[str] = None,
        mime_types: Optional[Sequence[str]] = None,
        tags_any: Optional[Sequence[str]] = None,
        tags_all: Optional[Sequence[str]] = None,
        folder_prefix: Optional[str] = None,
        folder_prefixes: Optional[Sequence[str]] = None,
        uploaded_by: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Rich facets used by file_manager plugin: mime, tags, uploaders, sessions, sizes, dates."""
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")

        uid = to_uuid(user_id)

        params: List[Any] = [uid]
        conditions, _ = self._accessible_list_filters(
            params=params,
            idx=2,
            folder_id=folder_id,
            media_type=media_type,
            session_id=session_id,
            search=search,
            ownership=ownership,
            created_from=created_from,
            created_to=created_to,
            size_min=size_min,
            size_max=size_max,
            mime_type=mime_type,
            mime_types=mime_types,
            tags_any=tags_any,
            tags_all=tags_all,
            folder_prefix=folder_prefix,
            folder_prefixes=folder_prefixes,
            uploaded_by=uploaded_by,
        )

        where_filters = " AND ".join(conditions) if conditions else "TRUE"

        top_m = int(max(1, min(top_mimes, 200)))
        top_t = int(max(1, min(top_tags, 200)))
        top_u = int(max(1, min(top_uploaders, 100)))
        top_s = int(max(1, min(top_sessions, 100)))

        bucket = (date_buckets or "month").strip().lower()
        if bucket not in ("day", "week", "month"):
            bucket = "month"

        accessible_cte = """
            WITH accessible AS (
                SELECT a.*
                FROM artifacts a
                WHERE (
                    a.user_id = $1
                    OR EXISTS (
                        SELECT 1
                        FROM group_memberships gm
                        JOIN group_files gf ON gf.group_id = gm.group_id
                        WHERE gm.user_id = $1 AND gf.artifact_id = a.id
                    )
                    OR EXISTS (
                        SELECT 1
                        FROM user_file_overrides ufo
                        WHERE ufo.user_id = $1
                          AND ufo.artifact_id = a.id
                          AND 'read' = ANY(ufo.permissions)
                          AND (ufo.expires_at IS NULL OR ufo.expires_at > now())
                    )
                )
            )
        """

        async with self.pool.acquire() as conn:
            total = await conn.fetchrow(
                f"""
                {accessible_cte}
                SELECT
                    COUNT(*) AS total_count,
                    COALESCE(SUM(a.file_size_bytes),0)::bigint AS total_bytes,
                    COUNT(*) FILTER (WHERE a.user_id = $1) AS owner_count,
                    COUNT(*) FILTER (WHERE a.user_id IS DISTINCT FROM $1) AS shared_count,
                    MIN(a.created_at) AS earliest_created,
                    MAX(a.created_at) AS latest_created,
                    MIN(a.file_size_bytes) AS smallest_bytes,
                    MAX(a.file_size_bytes) AS largest_bytes
                FROM accessible a
                WHERE {where_filters}
                """,
                *params,
            )
            mime_rows = await conn.fetch(
                f"""
                {accessible_cte}
                SELECT COALESCE(a.mime_type, 'unknown') AS key, COUNT(*) AS count,
                       COALESCE(SUM(a.file_size_bytes), 0)::bigint AS bytes
                FROM accessible a
                WHERE {where_filters}
                GROUP BY 1
                ORDER BY count DESC
                LIMIT {top_m}
                """,
                *params,
            )
            media_rows = await conn.fetch(
                f"""
                {accessible_cte}
                SELECT COALESCE(a.media_type::text, 'other') AS key, COUNT(*) AS count,
                       COALESCE(SUM(a.file_size_bytes), 0)::bigint AS bytes
                FROM accessible a
                WHERE {where_filters}
                GROUP BY 1
                ORDER BY count DESC
                """,
                *params,
            )
            tag_rows = await conn.fetch(
                f"""
                {accessible_cte}
                SELECT t.tag::text AS key, COUNT(*) AS count
                FROM accessible a, jsonb_array_elements_text(COALESCE(a.metadata->'tags', '[]'::jsonb)) AS t(tag)
                WHERE {where_filters}
                GROUP BY 1
                ORDER BY count DESC
                LIMIT {top_t}
                """,
                *params,
            )
            uploader_rows = await conn.fetch(
                f"""
                {accessible_cte}
                SELECT a.user_id::text AS key, COUNT(*) AS count,
                       COALESCE(SUM(a.file_size_bytes), 0)::bigint AS bytes
                FROM accessible a
                WHERE {where_filters} AND a.user_id IS NOT NULL
                GROUP BY 1
                ORDER BY count DESC
                LIMIT {top_u}
                """,
                *params,
            )
            session_rows = await conn.fetch(
                f"""
                {accessible_cte}
                SELECT a.session_id::text AS key, COUNT(*) AS count,
                       COALESCE(SUM(a.file_size_bytes), 0)::bigint AS bytes
                FROM accessible a
                WHERE {where_filters} AND a.session_id IS NOT NULL
                GROUP BY 1
                ORDER BY count DESC
                LIMIT {top_s}
                """,
                *params,
            )
            size_rows = await conn.fetch(
                f"""
                {accessible_cte}
                SELECT
                    CASE
                        WHEN a.file_size_bytes < 102400 THEN '<100KB'
                        WHEN a.file_size_bytes < 1048576 THEN '100KB-1MB'
                        WHEN a.file_size_bytes < 10485760 THEN '1-10MB'
                        WHEN a.file_size_bytes < 104857600 THEN '10-100MB'
                        WHEN a.file_size_bytes < 1073741824 THEN '100MB-1GB'
                        ELSE '>1GB'
                    END AS key,
                    COUNT(*) AS count,
                    COALESCE(SUM(a.file_size_bytes), 0)::bigint AS bytes
                FROM accessible a
                WHERE {where_filters}
                GROUP BY 1
                ORDER BY MIN(a.file_size_bytes)
                """,
                *params,
            )
            date_rows = await conn.fetch(
                f"""
                {accessible_cte}
                SELECT date_trunc('{bucket}', a.created_at) AS bucket, COUNT(*) AS count
                FROM accessible a
                WHERE {where_filters}
                GROUP BY 1
                ORDER BY 1 DESC
                LIMIT 24
                """,
                *params,
            )

        return {
            "total": {
                "count": int(total["total_count"] or 0),
                "bytes": int(total["total_bytes"] or 0),
                "owner_count": int(total["owner_count"] or 0),
                "shared_count": int(total["shared_count"] or 0),
                "earliest_created": total["earliest_created"].isoformat() if total["earliest_created"] else None,
                "latest_created": total["latest_created"].isoformat() if total["latest_created"] else None,
                "smallest_bytes": int(total["smallest_bytes"]) if total["smallest_bytes"] is not None else None,
                "largest_bytes": int(total["largest_bytes"]) if total["largest_bytes"] is not None else None,
            },
            "facets": {
                "mime_type": [
                    {"key": str(r["key"]), "count": int(r["count"]), "bytes": int(r["bytes"])}
                    for r in mime_rows
                ],
                "media_type": [
                    {"key": str(r["key"]), "count": int(r["count"]), "bytes": int(r["bytes"])}
                    for r in media_rows
                ],
                "tags": [
                    {"key": str(r["key"]), "count": int(r["count"])}
                    for r in tag_rows
                ],
                "uploaders": [
                    {"key": str(r["key"]), "count": int(r["count"]), "bytes": int(r["bytes"])}
                    for r in uploader_rows
                ],
                "sessions": [
                    {"key": str(r["key"]), "count": int(r["count"]), "bytes": int(r["bytes"])}
                    for r in session_rows
                ],
                "size_buckets": [
                    {"key": str(r["key"]), "count": int(r["count"]), "bytes": int(r["bytes"])}
                    for r in size_rows
                ],
                "date_histogram": [
                    {"bucket": r["bucket"].isoformat() if r["bucket"] else None, "count": int(r["count"])}
                    for r in date_rows
                ],
            },
            "_meta": {
                "date_buckets": bucket,
                # legacy shape kept for backward compatibility (mime_type as object)
                "mime_type_map": {str(r["key"]): int(r["count"]) for r in mime_rows},
            },
        }

    # ------------------------------------------------------------------ Share tokens (artifacts)
    async def share_token_create(
        self,
        *,
        artifact_id: str,
        created_by: str,
        expires_sec: int = 3600,
        scope: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")

        aid = to_uuid(artifact_id)
        uid = to_uuid(created_by)
        expires = int(max(60, min(expires_sec, 7 * 24 * 3600)))

        raw = secrets.token_urlsafe(32)
        token_hash = _sha256_hex(raw)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO artifact_share_tokens (token_hash, artifact_id, created_by, scope, expires_at)
                VALUES ($1, $2, $3, COALESCE($4::jsonb,'{}'::jsonb), now() + ($5::int || ' seconds')::interval)
                RETURNING id, artifact_id, created_by, scope, expires_at, revoked_at, created_at
                """,
                token_hash,
                aid,
                uid,
                json_dumps(scope or {}),
                expires,
            )

        return {
            "token": raw,
            "token_id": str(row["id"]),
            "artifact_id": str(row["artifact_id"]),
            "created_by": str(row["created_by"]),
            "scope": row["scope"],
            "expires_at": row["expires_at"].isoformat() if row["expires_at"] else None,
            "revoked_at": row["revoked_at"].isoformat() if row["revoked_at"] else None,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }

    async def share_token_get_by_hash(self, *, token: str) -> Optional[Dict[str, Any]]:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        th = _sha256_hex(token)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, token_hash, artifact_id, created_by, scope, expires_at, revoked_at, created_at
                FROM artifact_share_tokens
                WHERE token_hash = $1
                """,
                th,
            )
        if not row:
            return None
        return {
            "id": str(row["id"]),
            "artifact_id": str(row["artifact_id"]),
            "created_by": str(row["created_by"]),
            "scope": row["scope"],
            "expires_at": row["expires_at"].isoformat() if row["expires_at"] else None,
            "revoked_at": row["revoked_at"].isoformat() if row["revoked_at"] else None,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }

    async def share_token_revoke(self, *, token: str) -> bool:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        th = _sha256_hex(token)
        async with self.pool.acquire() as conn:
            res = await conn.execute(
                """
                UPDATE artifact_share_tokens
                SET revoked_at = now()
                WHERE token_hash = $1 AND revoked_at IS NULL
                """,
                th,
            )
        # asyncpg returns "UPDATE <n>"
        return res.startswith("UPDATE ") and not res.endswith(" 0")

    async def share_audit_log(
        self,
        *,
        token_id: Optional[str],
        action: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")
        tid = to_uuid(token_id) if token_id else None
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO artifact_share_audit (token_id, action, ip_address, user_agent, metadata)
                VALUES ($1, $2, $3, $4, COALESCE($5::jsonb,'{}'::jsonb))
                """,
                tid,
                (action or "unknown")[:32],
                ip_address,
                user_agent,
                json_dumps(metadata or {}),
            )

    async def artifact_find_first_by_checksum(
        self,
        checksum: str,
        *,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Find the earliest artifact with this SHA-256 content hash.
        If user_id is set, restrict to rows where artifacts.user_id matches.
        Used for deterministic deduplication (same bytes => same identity).
        """
        if not self.pool or not checksum:
            return None

        user_uuid = to_uuid(user_id) if user_id else None
        async with self.pool.acquire() as conn:
            if user_uuid:
                row = await conn.fetchrow(
                    """
                    SELECT id, media_type, storage_path, external_ref, checksum,
                           metadata, folder_id, original_filename, file_size_bytes,
                           mime_type, user_id, session_id, created_at, updated_at
                    FROM artifacts
                    WHERE checksum = $1 AND user_id = $2
                    ORDER BY created_at ASC
                    LIMIT 1
                    """,
                    checksum,
                    user_uuid,
                )
            else:
                row = await conn.fetchrow(
                    """
                    SELECT id, media_type, storage_path, external_ref, checksum,
                           metadata, folder_id, original_filename, file_size_bytes,
                           mime_type, user_id, session_id, created_at, updated_at
                    FROM artifacts
                    WHERE checksum = $1
                    ORDER BY created_at ASC
                    LIMIT 1
                    """,
                    checksum,
                )

        if not row:
            return None

        return {
            "id": str(row["id"]),
            "media_type": row["media_type"],
            "storage_path": row["storage_path"],
            "external_ref": row["external_ref"],
            "checksum": row["checksum"],
            "metadata": row["metadata"],
            "folder_id": str(row["folder_id"]) if row["folder_id"] else None,
            "original_filename": row["original_filename"],
            "file_size_bytes": row["file_size_bytes"],
            "mime_type": row["mime_type"],
            "user_id": str(row["user_id"]) if row["user_id"] else None,
            "session_id": str(row["session_id"]) if row["session_id"] else None,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        }

    async def artifact_find_first_by_import_locator(
        self,
        relative_path: str,
        filename: str,
        *,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        نخستین artifact با همان مسیر نسبی از ریشهٔ import و نام فایل در metadata.
        سبک‌تر از هش محتوا؛ برای dedup case-importer.
        """
        if not self.pool:
            return None
        rel = (relative_path or "").strip()
        fn = (filename or "").strip()
        if not rel or not fn:
            return None

        user_uuid = to_uuid(user_id) if user_id else None
        async with self.pool.acquire() as conn:
            if user_uuid:
                row = await conn.fetchrow(
                    """
                    SELECT id, media_type, storage_path, external_ref, checksum,
                           metadata, folder_id, original_filename, file_size_bytes,
                           mime_type, user_id, session_id, created_at, updated_at
                    FROM artifacts
                    WHERE user_id = $1
                      AND COALESCE(metadata->>'case_importer_rel_path', '') = $2
                      AND COALESCE(metadata->>'case_importer_filename', '') = $3
                    ORDER BY created_at ASC
                    LIMIT 1
                    """,
                    user_uuid,
                    rel,
                    fn,
                )
            else:
                row = await conn.fetchrow(
                    """
                    SELECT id, media_type, storage_path, external_ref, checksum,
                           metadata, folder_id, original_filename, file_size_bytes,
                           mime_type, user_id, session_id, created_at, updated_at
                    FROM artifacts
                    WHERE COALESCE(metadata->>'case_importer_rel_path', '') = $1
                      AND COALESCE(metadata->>'case_importer_filename', '') = $2
                    ORDER BY created_at ASC
                    LIMIT 1
                    """,
                    rel,
                    fn,
                )

        if not row:
            return None

        return {
            "id": str(row["id"]),
            "media_type": row["media_type"],
            "storage_path": row["storage_path"],
            "external_ref": row["external_ref"],
            "checksum": row["checksum"],
            "metadata": row["metadata"],
            "folder_id": str(row["folder_id"]) if row["folder_id"] else None,
            "original_filename": row["original_filename"],
            "file_size_bytes": row["file_size_bytes"],
            "mime_type": row["mime_type"],
            "user_id": str(row["user_id"]) if row["user_id"] else None,
            "session_id": str(row["session_id"]) if row["session_id"] else None,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        }

    @staticmethod
    def _safe_tool_table_name(tool_id: str) -> Optional[str]:
        safe = tool_id.replace(".", "_").replace("-", "_").lower()
        if not re.fullmatch(r"[a-z0-9_]+", safe):
            return None
        name = f"tool_{safe}"
        if not re.fullmatch(r"tool_[a-z0-9_]+", name):
            return None
        return name

    async def find_artifact_everywhere(
        self,
        artifact_id: str,
        *,
        per_table_limit: int = 200,
        skip_tables: Optional[Sequence[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generic cross-table discovery: find ALL rows referencing this artifact_id
        across the entire database, regardless of which tool/feature wrote them.

        Algorithm:
          1) Query information_schema.columns for every table that has a column
             named `artifact_id` (any schema except system schemas).
          2) For each such table, run `SELECT * WHERE artifact_id = $1 LIMIT N`.
          3) Return a dict keyed by `<schema>.<table>` (or just `<table>` for public)
             with row_count + sample rows.

        Notes:
          - Table names are validated with a strict regex before being spliced
            into SQL, so this is safe.
          - The artifacts row itself (artifacts.id == artifact_id) is NOT returned
            here (callers fetch it via artifact_get); we only return SUPPLEMENTARY
            rows from other tables that *reference* the artifact.
          - Works for `tool_*` tables, `artifact_images`, `artifact_audio`,
            `artifact_audio_segments`, `artifact_faces`, `shares`, and any future
            table with an `artifact_id` column — without code changes.
        """
        if not self.pool:
            return {}

        aid_uuid = to_uuid(artifact_id)
        if aid_uuid is None:
            return {}

        skip = {s.lower() for s in (skip_tables or [])}
        per_table_limit = max(1, min(int(per_table_limit or 200), 2000))

        async with self.pool.acquire() as conn:
            try:
                cols = await conn.fetch(
                    """
                    SELECT table_schema, table_name, data_type, udt_name
                    FROM information_schema.columns
                    WHERE column_name = 'artifact_id'
                      AND table_schema NOT IN ('information_schema','pg_catalog')
                    """
                )
            except Exception as exc:
                log.warning("find_artifact_everywhere: discovery failed: %s", exc)
                return {}

            results: Dict[str, Dict[str, Any]] = {}
            for col in cols:
                schema = col["table_schema"]
                tname = col["table_name"]
                udt = (col["udt_name"] or "").lower()

                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema or ""):
                    continue
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tname or ""):
                    continue
                key = tname if schema == "public" else f"{schema}.{tname}"
                if tname.lower() in skip or key.lower() in skip:
                    continue

                bind_value: Any = aid_uuid if udt == "uuid" else str(aid_uuid)
                qualified = f'"{schema}"."{tname}"'

                try:
                    count_row = await conn.fetchrow(
                        f'SELECT COUNT(*)::bigint AS c FROM {qualified} WHERE artifact_id = $1',
                        bind_value,
                    )
                    rc = int((count_row and count_row["c"]) or 0)
                    if rc <= 0:
                        continue
                    rows = await conn.fetch(
                        f'SELECT * FROM {qualified} WHERE artifact_id = $1 LIMIT $2',
                        bind_value,
                        per_table_limit,
                    )
                except Exception as exc:
                    log.debug(
                        "find_artifact_everywhere: %s skipped (%s)", qualified, exc
                    )
                    continue

                serialised: List[Dict[str, Any]] = []
                for r in rows:
                    item: Dict[str, Any] = {}
                    for k, v in dict(r).items():
                        if isinstance(v, uuid.UUID):
                            item[k] = str(v)
                        elif hasattr(v, "isoformat"):
                            try:
                                item[k] = v.isoformat()
                            except Exception:
                                item[k] = str(v)
                        elif isinstance(v, (bytes, bytearray, memoryview)):
                            try:
                                item[k] = bytes(v).decode("utf-8", errors="replace")
                            except Exception:
                                item[k] = str(v)
                        elif isinstance(v, str) and v and v[0] in "[{":
                            # asyncpg returns jsonb columns as strings by default;
                            # try to parse so the UI gets structured data.
                            try:
                                item[k] = json.loads(v)
                            except Exception:
                                item[k] = v
                        else:
                            item[k] = v
                    serialised.append(item)

                results[key] = {
                    "row_count": rc,
                    "returned": len(serialised),
                    "truncated": rc > len(serialised),
                    "rows": serialised,
                }
            return results

    async def tools_for_artifact(self, artifact_id: str) -> List[Dict[str, Any]]:
        """
        Discover which dynamic tool_* tables contain rows for a given artifact_id,
        and return per-tool aggregates (count, last item_index, last created_at).

        Used by `plugin_file_manager_tools_for_artifact` to power per-file UIs.
        """
        if not self.pool:
            return []
        async with self.pool.acquire() as conn:
            tables = await conn.fetch(
                """
                SELECT c.relname AS table_name
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public'
                  AND c.relkind IN ('r','p')
                  AND c.relname LIKE 'tool_%'
                ORDER BY c.relname
                """
            )
            results: List[Dict[str, Any]] = []
            for t in tables:
                tname = t["table_name"]
                # Validate name conservatively before splicing into SQL
                if not re.fullmatch(r"tool_[a-z0-9_]+", tname):
                    continue
                try:
                    row = await conn.fetchrow(
                        f"""
                        SELECT
                            COUNT(*) AS row_count,
                            MAX(item_index) AS last_item_index,
                            MAX(created_at) AS last_created_at
                        FROM {tname}
                        WHERE artifact_id = $1
                        """,
                        artifact_id,
                    )
                except Exception:
                    continue
                rc = int((row and row["row_count"]) or 0)
                if rc <= 0:
                    continue
                tool_id = tname[len("tool_"):]
                results.append({
                    "tool_id": tool_id,
                    "tool_table": tname,
                    "row_count": rc,
                    "last_item_index": int(row["last_item_index"] or 0) if row["last_item_index"] is not None else None,
                    "last_created_at": row["last_created_at"].isoformat() if row["last_created_at"] else None,
                })
        return results

    async def tool_metadata_rows_for_artifact(
        self,
        tool_id: str,
        artifact_id: str,
    ) -> List[Dict[str, Any]]:
        """Load metadata channel rows from dynamic tool_* table."""
        if not self.pool:
            return []

        table = self._safe_tool_table_name(tool_id)
        if not table:
            return []

        async with self.pool.acquire() as conn:
            try:
                rows = await conn.fetch(
                    f"""
                    SELECT id, artifact_id, case_id, item_type, item_index,
                           file_name, file_path, data, created_at
                    FROM {table}
                    WHERE artifact_id = $1
                    ORDER BY item_index ASC
                    """,
                    artifact_id,
                )
            except Exception:
                return []

        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({
                "id": r["id"],
                "artifact_id": r["artifact_id"],
                "case_id": r["case_id"],
                "item_type": r["item_type"],
                "item_index": r["item_index"],
                "file_name": r["file_name"],
                "file_path": r["file_path"],
                "data": r["data"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            })
        return out

    async def case_importer_prefs_get(self, user_id: str) -> Optional[Dict[str, Any]]:
        if not self.pool or not user_id:
            return None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT user_id, tool_allowlist, updated_at
                FROM case_importer_user_preferences
                WHERE user_id = $1
                """,
                user_id,
            )
        if not row:
            return None
        allow = row["tool_allowlist"]
        if isinstance(allow, str):
            try:
                allow = json.loads(allow)
            except json.JSONDecodeError:
                pass
        return {
            "user_id": row["user_id"],
            "tool_allowlist": allow,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        }

    async def case_importer_prefs_upsert(
        self,
        user_id: str,
        tool_allowlist: Optional[Any],
    ) -> Dict[str, Any]:
        if not self.pool or not user_id:
            raise RuntimeError("PostgreSQL not connected or missing user_id")
        # asyncpg JSONB bind expects a JSON string for list/dict values (not raw list).
        allow_json = (
            json.dumps(tool_allowlist, ensure_ascii=False)
            if tool_allowlist is not None
            else None
        )
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO case_importer_user_preferences (user_id, tool_allowlist, updated_at)
                VALUES ($1, $2::jsonb, NOW())
                ON CONFLICT (user_id) DO UPDATE SET
                    tool_allowlist = EXCLUDED.tool_allowlist,
                    updated_at = NOW()
                """,
                user_id,
                allow_json,
            )
        return (await self.case_importer_prefs_get(user_id)) or {
            "user_id": user_id,
            "tool_allowlist": tool_allowlist,
        }

    # Plugin session blobs (users/.../plugins/.../sessions/.../session.json) are
    # chat-style durable state — never case evidence / RunFileInput fodder.
    _CASE_ARTIFACT_EXCLUDE_SESSIONS = """
                AND COALESCE(original_filename, '') <> 'session.json'
                AND COALESCE(metadata->>'kind', '') <> 'plugin_session'
                AND COALESCE(metadata->>'layout', '') <> 'sessions'
                AND NOT (
                    COALESCE(storage_path, '') LIKE '%/plugins/%'
                    AND COALESCE(storage_path, '') LIKE '%/sessions/%'
                )
        """

    async def list_artifact_ids_for_case(
        self,
        case_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[str]:
        """Artifact UUIDs linked to a case-importer case."""
        cid = (case_id or "").strip()
        if not self.pool or not cid:
            return []
        esc = self._escape_ilike(cid)
        params: List[Any] = [cid, esc]
        limit_sql = ""
        if limit is not None:
            cap = max(1, int(limit))
            params.append(cap)
            limit_sql = f" LIMIT ${len(params)}"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id FROM artifacts
                WHERE (
                   COALESCE(metadata->>'case_id', '') = $1
                   OR COALESCE(metadata->>'case_importer_last_case_id', '') = $1
                   OR storage_path ILIKE '%' || $2 || '%' ESCAPE '\\'
                )
                {self._CASE_ARTIFACT_EXCLUDE_SESSIONS}
                {limit_sql}
                """,
                *params,
            )
        return [str(r["id"]) for r in rows]

    async def list_artifacts_for_case(
        self,
        case_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Artifact rows linked to a case (id, mime, media_type, filename).

        ``limit=None`` returns every matching row (no hard cap).
        """
        cid = (case_id or "").strip()
        if not self.pool or not cid:
            return []
        esc = self._escape_ilike(cid)
        params: List[Any] = [cid, esc]
        limit_sql = ""
        if limit is not None:
            cap = max(1, int(limit))
            params.append(cap)
            limit_sql = f" LIMIT ${len(params)}"
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, media_type, mime_type, original_filename, storage_path,
                       file_size_bytes
                FROM artifacts
                WHERE (
                   COALESCE(metadata->>'case_id', '') = $1
                   OR COALESCE(metadata->>'case_importer_last_case_id', '') = $1
                   OR storage_path ILIKE '%' || $2 || '%' ESCAPE '\\'
                )
                {self._CASE_ARTIFACT_EXCLUDE_SESSIONS}
                ORDER BY created_at ASC NULLS LAST, id
                {limit_sql}
                """,
                *params,
            )
        out: List[Dict[str, Any]] = []
        for row in rows:
            aid = str(row["id"])
            out.append(
                {
                    "id": aid,
                    "artifact_id": aid,
                    "media_type": row["media_type"],
                    "mime_type": row["mime_type"],
                    "original_filename": row["original_filename"],
                    "filename": row["original_filename"],
                    "storage_path": row["storage_path"],
                    "file_size_bytes": row["file_size_bytes"],
                }
            )
        return out

    async def delete_case_row(self, case_id: str) -> bool:
        if not self.pool:
            return False
        async with self.pool.acquire() as conn:
            res = await conn.execute("DELETE FROM cases WHERE id = $1", case_id)
        return res.startswith("DELETE ") and not res.endswith(" 0")

    async def purge_column_references(
        self,
        column: str,
        value: str,
        *,
        skip_tables: Optional[Sequence[str]] = None,
    ) -> Dict[str, int]:
        """
        DELETE all rows where `column` = value across every table that defines it.
        Table/column names are validated before interpolation.
        """
        if not self.pool or not value:
            return {}
        col = (column or "").strip()
        if col not in ("artifact_id", "case_id"):
            return {}

        skip = {s.lower() for s in (skip_tables or [])}
        counts: Dict[str, int] = {}

        async with self.pool.acquire() as conn:
            try:
                cols = await conn.fetch(
                    """
                    SELECT table_schema, table_name, udt_name
                    FROM information_schema.columns
                    WHERE column_name = $1
                      AND table_schema NOT IN ('information_schema','pg_catalog')
                    """,
                    col,
                )
            except Exception as exc:
                log.warning("purge_column_references discovery failed: %s", exc)
                return {}

            bind_uuid = to_uuid(value) if col == "artifact_id" else None
            for row in cols:
                schema = row["table_schema"]
                tname = row["table_name"]
                udt = (row["udt_name"] or "").lower()
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema or ""):
                    continue
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tname or ""):
                    continue
                if tname.lower() in skip:
                    continue
                qualified = f'"{schema}"."{tname}"'
                bind_value: Any = bind_uuid if (udt == "uuid" and bind_uuid) else str(value)
                try:
                    res = await conn.execute(
                        f"DELETE FROM {qualified} WHERE {col} = $1",
                        bind_value,
                    )
                    if res.startswith("DELETE "):
                        n = int(res.split()[-1])
                        if n > 0:
                            key = tname if schema == "public" else f"{schema}.{tname}"
                            counts[key] = n
                except Exception as exc:
                    log.debug("purge_column_references %s skipped: %s", qualified, exc)
        return counts

    # Note: tool_runs methods have been REMOVED - use agent_tool_executions via TraceRepository

    # ------------------------------------------------------------------ Feedback
    async def feedback_save(
        self,
        *,
        session_id: Optional[str],
        rating: str,
        comment: Optional[str] = None,
        message_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Save user feedback."""
        if not self.pool:
            raise RuntimeError("PostgreSQL not connected")

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO feedback (session_id, rating, comment, message_text, metadata)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                RETURNING id, created_at
                """,
                session_id,
                rating,
                comment,
                message_text,
                json_dumps(metadata),
            )

        return {
            "id": str(row["id"]),
            "session_id": session_id,
            "rating": rating,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }
