"""Tool / plugin run sessions — Postgres only (no MinIO blobs)."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


class ToolSessionRepository:
    """CRUD for ``tool_sessions`` (user-scoped durable plugin/tool runs)."""

    def __init__(self, postgres):
        self.postgres = postgres

    async def _resolve_user_id(self, user_id: str) -> Optional[str]:
        if not self.postgres.pool:
            return None
        uid = str(user_id or "").strip()
        if not uid:
            return None
        async with self.postgres.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM users WHERE keycloak_id = $1",
                uid,
            )
            if row:
                return str(row["id"])
            try:
                import uuid as uuid_module

                uuid_module.UUID(uid)
            except (ValueError, AttributeError):
                log.warning("Could not resolve user_id: %s", uid)
                return None
            row = await conn.fetchrow(
                "SELECT id FROM users WHERE id = $1::uuid",
                uid,
            )
            if row:
                return str(row["id"])
        log.warning("Could not resolve user_id: %s", uid)
        return None

    @staticmethod
    def _row_to_dict(row: Any, *, include_payload: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        has_payload_col = False
        try:
            raw = row["payload"]
            has_payload_col = True
        except (KeyError, TypeError):
            raw = None
        if has_payload_col:
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except Exception:
                    raw = {}
            if isinstance(raw, dict):
                payload = raw
        files: List[Any] = []
        if has_payload_col:
            files = payload.get("files") if isinstance(payload.get("files"), list) else []
        out: Dict[str, Any] = {
            "id": str(row["id"]),
            "session_id": str(row["id"]),
            "user_id": str(row["user_id"]),
            "plugin_id": row["plugin_id"],
            "tool_id": row["tool_id"],
            "record_id": row["record_id"],
            "run_id": row["run_id"],
            "status": row["status"],
            "ok": row["ok"],
            "error": row["error"],
            "output_preview": row["output_preview"] or "",
            "created_at": row["created_at"].timestamp()
            if hasattr(row["created_at"], "timestamp")
            else row["created_at"],
            "updated_at": row["updated_at"].timestamp()
            if hasattr(row["updated_at"], "timestamp")
            else row["updated_at"],
            "files": files,
            "rlm_session_id": payload.get("rlm_session_id") if has_payload_col else None,
            "flow_id": (
                (payload.get("flow_id") or row["record_id"])
                if has_payload_col
                else row["record_id"]
            ),
        }
        if include_payload and has_payload_col:
            out["payload"] = payload
        return out

    async def upsert(
        self,
        *,
        user_id: str,
        plugin_id: str,
        record_id: str,
        run_id: str,
        tool_id: Optional[str] = None,
        status: Optional[str] = None,
        ok: Optional[bool] = None,
        error: Optional[str] = None,
        output_preview: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.postgres.pool:
            return None
        resolved = await self._resolve_user_id(user_id)
        if not resolved:
            log.error("Cannot upsert tool_session: user not found for id %s", user_id)
            return None
        plugin = str(plugin_id or "").strip()
        record = str(record_id or "").strip()
        run = str(run_id or "").strip()
        if not plugin or not record or not run:
            log.error("tool_session upsert missing plugin/record/run")
            return None
        body = payload if isinstance(payload, dict) else {}
        # Stamp top-level case_id from conversation soft-link so Cases UI can
        # query by case without requiring a schema migration.
        if not str(body.get("case_id") or "").strip():
            conv = body.get("conversation")
            if isinstance(conv, dict):
                cid = str(conv.get("case_id") or "").strip()
                if cid:
                    body = {**body, "case_id": cid}
        preview = str(output_preview if output_preview is not None else body.get("output_preview") or "")[
            :2000
        ]
        status_val = status if status is not None else body.get("status")
        ok_val = ok if ok is not None else body.get("ok")
        err_val = error if error is not None else body.get("error")
        async with self.postgres.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO tool_sessions (
                    user_id, plugin_id, tool_id, record_id, run_id,
                    status, ok, error, output_preview, payload
                )
                VALUES (
                    $1::uuid, $2, $3, $4, $5,
                    $6, $7, $8, $9, $10::jsonb
                )
                ON CONFLICT (user_id, plugin_id, record_id, run_id) DO UPDATE SET
                    tool_id = COALESCE(EXCLUDED.tool_id, tool_sessions.tool_id),
                    -- Mid-run checkpoints must never clobber a terminal upsert
                    -- that already landed (async race after run_end / finish).
                    status = CASE
                        WHEN lower(COALESCE(tool_sessions.status, '')) IN
                             ('success', 'error', 'cancelled', 'canceled')
                         AND lower(COALESCE(EXCLUDED.status, '')) = 'running'
                        THEN tool_sessions.status
                        ELSE EXCLUDED.status
                    END,
                    ok = CASE
                        WHEN lower(COALESCE(tool_sessions.status, '')) IN
                             ('success', 'error', 'cancelled', 'canceled')
                         AND lower(COALESCE(EXCLUDED.status, '')) = 'running'
                        THEN tool_sessions.ok
                        ELSE EXCLUDED.ok
                    END,
                    error = CASE
                        WHEN lower(COALESCE(tool_sessions.status, '')) IN
                             ('success', 'error', 'cancelled', 'canceled')
                         AND lower(COALESCE(EXCLUDED.status, '')) = 'running'
                        THEN tool_sessions.error
                        ELSE EXCLUDED.error
                    END,
                    output_preview = CASE
                        WHEN lower(COALESCE(tool_sessions.status, '')) IN
                             ('success', 'error', 'cancelled', 'canceled')
                         AND lower(COALESCE(EXCLUDED.status, '')) = 'running'
                        THEN tool_sessions.output_preview
                        ELSE EXCLUDED.output_preview
                    END,
                    payload = CASE
                        WHEN lower(COALESCE(tool_sessions.status, '')) IN
                             ('success', 'error', 'cancelled', 'canceled')
                         AND lower(COALESCE(EXCLUDED.status, '')) = 'running'
                        THEN tool_sessions.payload
                        ELSE EXCLUDED.payload
                    END,
                    updated_at = now()
                RETURNING *
                """,
                resolved,
                plugin,
                str(tool_id or "").strip() or None,
                record,
                run,
                str(status_val or "").strip() or None,
                ok_val,
                str(err_val).strip() if err_val else None,
                preview,
                json.dumps(body, ensure_ascii=False, default=str),
            )
        return self._row_to_dict(row) if row else None

    async def list_sessions(
        self,
        *,
        user_id: str,
        plugin_id: str,
        record_id: str,
        limit: int = 100,
        include_payload: bool = False,
    ) -> List[Dict[str, Any]]:
        if not self.postgres.pool:
            return []
        resolved = await self._resolve_user_id(user_id)
        if not resolved:
            return []
        cap = max(1, min(int(limit or 100), 500))
        # Never SELECT payload here when include_payload=False. Agent-flow
        # payloads are multi-MB JSONB (trace + rlm logs); Postgres TOASTs the
        # whole value, so even payload->>'x' would detoast every row and make
        # the Run Monitor sessions list crawl.
        if include_payload:
            select_sql = "SELECT * FROM tool_sessions"
        else:
            select_sql = """
                SELECT id, user_id, plugin_id, tool_id, record_id, run_id,
                       status, ok, error, output_preview, created_at, updated_at
                FROM tool_sessions
            """
        async with self.postgres.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                {select_sql}
                WHERE user_id = $1::uuid
                  AND plugin_id = $2
                  AND record_id = $3
                ORDER BY created_at DESC NULLS LAST, id
                LIMIT $4
                """,
                resolved,
                str(plugin_id or "").strip(),
                str(record_id or "").strip(),
                cap,
            )
        return [
            self._row_to_dict(r, include_payload=include_payload) for r in rows
        ]

    async def list_by_case_id(
        self,
        *,
        case_id: str,
        plugin_id: Optional[str] = None,
        limit: int = 100,
        user_id: Optional[str] = None,
        require_expose_to_case: bool = True,
    ) -> List[Dict[str, Any]]:
        """List tool_sessions soft-linked to ``case_id`` (Case AI Summary).

        Matches top-level ``payload.case_id`` or legacy
        ``payload.conversation.case_id``. When ``require_expose_to_case`` is
        true (default), only rows with ``expose_to_case`` truthy are returned.

        ``plugin_id=None`` returns all matching plugins. Returns slim rows
        (files roster only, no full trace/output).
        """
        if not self.postgres.pool:
            return []
        cid = str(case_id or "").strip()
        if not cid:
            return []
        plugin = str(plugin_id or "").strip() or None
        cap = max(1, min(int(limit or 100), 500))
        resolved: Optional[str] = None
        if user_id:
            resolved = await self._resolve_user_id(user_id)
            if not resolved:
                return []
        # Opt-in only: expose_to_case must be set on the envelope.
        expose_sql = ""
        if require_expose_to_case:
            expose_sql = """
                      AND lower(COALESCE(payload->>'expose_to_case', ''))
                            IN ('true', '1', 'yes')
            """
        slim_payload = """
                           jsonb_build_object(
                             'case_id', COALESCE(
                               NULLIF(payload->>'case_id', ''),
                               NULLIF(payload->'conversation'->>'case_id', '')
                             ),
                             'expose_to_case', COALESCE(payload->'expose_to_case', 'false'::jsonb),
                             'flow_id', COALESCE(payload->>'flow_id', record_id),
                             'files', COALESCE(payload->'files', '[]'::jsonb),
                             'output_preview', COALESCE(payload->>'output_preview', '')
                           ) AS payload
        """
        async with self.postgres.pool.acquire() as conn:
            if resolved and plugin:
                rows = await conn.fetch(
                    f"""
                    SELECT id, user_id, plugin_id, tool_id, record_id, run_id,
                           status, ok, error, output_preview, created_at, updated_at,
                           {slim_payload}
                    FROM tool_sessions
                    WHERE user_id = $1::uuid
                      AND plugin_id = $2
                      AND (
                        payload->>'case_id' = $3
                        OR payload->'conversation'->>'case_id' = $3
                      )
                      {expose_sql}
                    ORDER BY created_at DESC NULLS LAST, id
                    LIMIT $4
                    """,
                    resolved,
                    plugin,
                    cid,
                    cap,
                )
            elif resolved and not plugin:
                rows = await conn.fetch(
                    f"""
                    SELECT id, user_id, plugin_id, tool_id, record_id, run_id,
                           status, ok, error, output_preview, created_at, updated_at,
                           {slim_payload}
                    FROM tool_sessions
                    WHERE user_id = $1::uuid
                      AND (
                        payload->>'case_id' = $2
                        OR payload->'conversation'->>'case_id' = $2
                      )
                      {expose_sql}
                    ORDER BY created_at DESC NULLS LAST, id
                    LIMIT $3
                    """,
                    resolved,
                    cid,
                    cap,
                )
            elif plugin:
                rows = await conn.fetch(
                    f"""
                    SELECT id, user_id, plugin_id, tool_id, record_id, run_id,
                           status, ok, error, output_preview, created_at, updated_at,
                           {slim_payload}
                    FROM tool_sessions
                    WHERE plugin_id = $1
                      AND (
                        payload->>'case_id' = $2
                        OR payload->'conversation'->>'case_id' = $2
                      )
                      {expose_sql}
                    ORDER BY created_at DESC NULLS LAST, id
                    LIMIT $3
                    """,
                    plugin,
                    cid,
                    cap,
                )
            else:
                rows = await conn.fetch(
                    f"""
                    SELECT id, user_id, plugin_id, tool_id, record_id, run_id,
                           status, ok, error, output_preview, created_at, updated_at,
                           {slim_payload}
                    FROM tool_sessions
                    WHERE (
                        payload->>'case_id' = $1
                        OR payload->'conversation'->>'case_id' = $1
                      )
                      {expose_sql}
                    ORDER BY created_at DESC NULLS LAST, id
                    LIMIT $2
                    """,
                    cid,
                    cap,
                )
        return [self._row_to_dict(r, include_payload=True) for r in rows]

    async def get(
        self,
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        plugin_id: Optional[str] = None,
        record_id: Optional[str] = None,
        run_id: Optional[str] = None,
        include_payload: bool = True,
    ) -> Optional[Dict[str, Any]]:
        if not self.postgres.pool:
            return None
        sid = str(session_id or "").strip()
        async with self.postgres.pool.acquire() as conn:
            if sid:
                row = await conn.fetchrow(
                    "SELECT * FROM tool_sessions WHERE id = $1::uuid",
                    sid,
                )
                if not row:
                    return None
                if user_id:
                    resolved = await self._resolve_user_id(user_id)
                    if resolved and str(row["user_id"]) != resolved:
                        return None
                return self._row_to_dict(row, include_payload=include_payload)

            resolved = await self._resolve_user_id(str(user_id or ""))
            plugin = str(plugin_id or "").strip()
            record = str(record_id or "").strip()
            run = str(run_id or "").strip()
            if not (resolved and plugin and record and run):
                return None
            row = await conn.fetchrow(
                """
                SELECT * FROM tool_sessions
                WHERE user_id = $1::uuid
                  AND plugin_id = $2
                  AND record_id = $3
                  AND run_id = $4
                """,
                resolved,
                plugin,
                record,
                run,
            )
        return self._row_to_dict(row, include_payload=include_payload) if row else None

    async def delete(
        self,
        *,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        if not self.postgres.pool:
            return False
        sid = str(session_id or "").strip()
        if not sid:
            return False
        async with self.postgres.pool.acquire() as conn:
            if user_id:
                resolved = await self._resolve_user_id(user_id)
                if not resolved:
                    return False
                res = await conn.execute(
                    """
                    DELETE FROM tool_sessions
                    WHERE id = $1::uuid AND user_id = $2::uuid
                    """,
                    sid,
                    resolved,
                )
            else:
                res = await conn.execute(
                    "DELETE FROM tool_sessions WHERE id = $1::uuid",
                    sid,
                )
        return res.startswith("DELETE ") and not res.endswith(" 0")
