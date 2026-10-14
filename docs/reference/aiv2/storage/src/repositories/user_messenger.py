"""
User Messenger Repository — internal user-to-user messaging.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

log = logging.getLogger(__name__)

_ATTACHMENT_SPEC = Union[str, Dict[str, Any]]
_VALID_FOLDERS = frozenset({"inbox", "sent", "drafts", "archived", "trash"})
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str, limit: int = 200) -> str:
    plain = _HTML_TAG_RE.sub(" ", text or "")
    plain = " ".join(plain.split())
    return plain[:limit]


def _dm_pair_key(user_a: str, user_b: str) -> str:
    a, b = sorted([str(user_a), str(user_b)])
    return f"{a}:{b}"


def _record_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[key]
    except (KeyError, TypeError, IndexError):
        return default


def _normalize_attachments(raw: Any) -> List[Dict[str, Any]]:
    if not raw:
        return []
    items: List[Any]
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = [raw]
    out: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            aid = item.strip()
            if aid:
                out.append({"artifact_id": aid, "source": "upload", "share_mode": "direct"})
        elif isinstance(item, dict):
            aid = str(item.get("artifact_id") or item.get("id") or "").strip()
            if not aid:
                continue
            source = str(item.get("source") or "upload").strip().lower()
            if source not in ("upload", "file_manager"):
                source = "file_manager" if source == "library" else "upload"
            share_mode = str(item.get("share_mode") or "direct").strip().lower()
            if share_mode not in ("direct", "share_link"):
                share_mode = "direct"
            spec: Dict[str, Any] = {
                "artifact_id": aid,
                "source": source,
                "share_mode": share_mode,
            }
            if item.get("expires_sec") is not None:
                spec["expires_sec"] = int(item["expires_sec"])
            out.append(spec)
    return out


class UserMessengerRepository:
    def __init__(self, postgres):
        self.postgres = postgres

    async def _resolve_user_id(self, user_id: str) -> Optional[str]:
        if not self.postgres.pool:
            return None
        async with self.postgres.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM users WHERE keycloak_id = $1",
                user_id,
            )
            if row:
                return str(row["id"])
            try:
                uuid.UUID(user_id)
                row = await conn.fetchrow(
                    "SELECT id FROM users WHERE id = $1::uuid",
                    user_id,
                )
                if row:
                    return str(row["id"])
            except (ValueError, AttributeError):
                pass
        return None

    async def _user_summary(self, user_id: str) -> Optional[Dict[str, Any]]:
        if not self.postgres.pool:
            return None
        async with self.postgres.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, username, email, display_name, avatar_url
                FROM users WHERE id = $1::uuid
                """,
                user_id,
            )
        if not row:
            return None
        display = row["display_name"] or row["username"] or row["email"]
        return {
            "id": str(row["id"]),
            "username": row["username"],
            "email": row["email"],
            "display_name": row["display_name"],
            "name": display,
            "avatar_url": row["avatar_url"],
            "avatar": row["avatar_url"],
        }

    @staticmethod
    def _parse_metadata(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    @staticmethod
    def _merge_message_meta(row: Any, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meta = UserMessengerRepository._parse_metadata(_record_get(row, "metadata"))
        extra: Dict[str, Any] = {}
        if meta.get("content_type"):
            extra["content_type"] = meta["content_type"]
        if meta.get("voice_duration_ms") is not None:
            extra["voice_duration_ms"] = meta["voice_duration_ms"]
        if meta.get("scheduled_at"):
            extra["scheduled_at"] = meta["scheduled_at"]
        if meta.get("cc"):
            extra["cc"] = meta["cc"]
        if meta.get("bcc"):
            extra["bcc"] = meta["bcc"]
        if state:
            if state.get("is_starred") is not None:
                extra["starred"] = bool(state["is_starred"])
            if state.get("is_pinned") is not None:
                extra["pinned"] = bool(state["is_pinned"])
            if state.get("is_muted") is not None:
                extra["muted"] = bool(state["is_muted"])
        return extra

    async def _get_or_create_dm_conversation(
        self,
        conn,
        *,
        sender_id: str,
        recipient_id: str,
        subject: Optional[str] = None,
    ) -> str:
        pair = _dm_pair_key(sender_id, recipient_id)
        row = await conn.fetchrow(
            "SELECT id FROM messenger_conversations WHERE dm_pair_key = $1",
            pair,
        )
        if row:
            return str(row["id"])
        row = await conn.fetchrow(
            """
            INSERT INTO messenger_conversations (type, subject, dm_pair_key, created_by)
            VALUES ('dm', $1, $2, $3::uuid)
            RETURNING id
            """,
            subject,
            pair,
            sender_id,
        )
        return str(row["id"])

    async def _message_access(self, user_id: str, message_id: str) -> bool:
        if not self.postgres.pool:
            return False
        async with self.postgres.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT 1 FROM messenger_message_states st
                WHERE st.message_id = $1::uuid AND st.user_id = $2::uuid
                  AND st.deleted_at IS NULL
                """,
                message_id,
                user_id,
            )
            return row is not None

    async def list_messages(
        self,
        user_id: str,
        *,
        folder: str = "inbox",
        page: int = 1,
        limit: int = 30,
        q: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved = await self._resolve_user_id(user_id)
        if not resolved:
            return {"items": [], "unread_count": 0, "total": 0, "page": page, "limit": limit}

        folder = (folder or "inbox").strip().lower()
        if folder not in _VALID_FOLDERS:
            folder = "inbox"
        page = max(1, int(page))
        limit = max(1, min(int(limit), 100))
        offset = (page - 1) * limit

        if not self.postgres.pool:
            return {"items": [], "unread_count": 0, "total": 0, "page": page, "limit": limit}

        search_sql = ""
        params: List[Any] = [resolved, folder]
        if q and q.strip():
            params.append(f"%{q.strip()}%")
            search_sql = f" AND (m.subject ILIKE ${len(params)} OR m.preview ILIKE ${len(params)} OR m.body ILIKE ${len(params)})"

        async with self.postgres.pool.acquire() as conn:
            total = await conn.fetchval(
                f"""
                SELECT COUNT(*)
                FROM messenger_messages m
                JOIN messenger_message_states st
                  ON st.message_id = m.id AND st.user_id = $1::uuid
                WHERE st.folder = $2 AND st.deleted_at IS NULL
                {search_sql}
                """,
                *params,
            )
            unread = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM messenger_message_states st
                WHERE st.user_id = $1::uuid AND st.folder = 'inbox'
                  AND st.is_read = false AND st.deleted_at IS NULL
                """,
                resolved,
            )
            rows = await conn.fetch(
                f"""
                SELECT m.id, m.conversation_id, m.sender_id, m.recipient_id,
                       m.subject, m.preview, m.priority, m.delivery_status,
                       m.metadata, m.created_at, st.is_read,
                       st.is_starred, st.is_pinned, st.is_muted,
                       EXISTS (
                         SELECT 1 FROM messenger_message_attachments ma
                         WHERE ma.message_id = m.id
                       ) AS has_attachments
                FROM messenger_messages m
                JOIN messenger_message_states st
                  ON st.message_id = m.id AND st.user_id = $1::uuid
                WHERE st.folder = $2 AND st.deleted_at IS NULL
                {search_sql}
                ORDER BY m.created_at DESC
                LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
                """,
                *params,
                limit,
                offset,
            )

        items = []
        for row in rows:
            item = {
                "id": str(row["id"]),
                "conversation_id": str(row["conversation_id"]),
                "from": str(row["sender_id"]),
                "to": str(row["recipient_id"]),
                "subject": row["subject"] or "",
                "preview": row["preview"] or "",
                "read": bool(row["is_read"]),
                "priority": row["priority"],
                "delivery_status": row["delivery_status"],
                "has_attachments": bool(row["has_attachments"]),
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            }
            item.update(self._merge_message_meta(row, {
                "is_starred": _record_get(row, "is_starred"),
                "is_pinned": _record_get(row, "is_pinned"),
                "is_muted": _record_get(row, "is_muted"),
            }))
            items.append(item)

        return {
            "items": items,
            "unread_count": int(unread or 0),
            "total": int(total or 0),
            "page": page,
            "limit": limit,
        }

    async def _enrich_attachments(self, message_id: str) -> List[Dict[str, Any]]:
        if not self.postgres.pool:
            return []
        async with self.postgres.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT ma.artifact_id, ma.source, ma.share_mode, ma.metadata,
                       a.original_filename, a.mime_type, a.media_type, a.file_size_bytes,
                       ast.token_hash IS NOT NULL AS has_share
                FROM messenger_message_attachments ma
                JOIN artifacts a ON a.id = ma.artifact_id
                LEFT JOIN artifact_share_tokens ast ON ast.id = ma.share_token_id
                WHERE ma.message_id = $1::uuid
                ORDER BY ma.sort_order, ma.artifact_id
                """,
                message_id,
            )
        out: List[Dict[str, Any]] = []
        for row in rows:
            aid = str(row["artifact_id"])
            meta = row["metadata"] if isinstance(row["metadata"], dict) else {}
            if isinstance(row["metadata"], str):
                try:
                    meta = json.loads(row["metadata"])
                except Exception:
                    meta = {}
            item = {
                "artifact_id": aid,
                "source": row["source"],
                "share_mode": row["share_mode"],
                "name": row["original_filename"] or "",
                "filename": row["original_filename"] or "",
                "mime_type": row["mime_type"],
                "media_type": row["media_type"],
                "size": row["file_size_bytes"],
                "download_url": f"/storage/files/{aid}/download",
                "thumbnail_url": f"/storage/files/{aid}/thumbnail?width=200&height=200",
            }
            if meta.get("share_url"):
                item["share_url"] = meta["share_url"]
            if meta.get("share_expires_at"):
                item["share_expires_at"] = meta["share_expires_at"]
            out.append(item)
        return out

    async def get_message(self, user_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        resolved = await self._resolve_user_id(user_id)
        if not resolved or not await self._message_access(resolved, message_id):
            return None
        if not self.postgres.pool:
            return None
        async with self.postgres.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT m.*, st.is_read, st.folder,
                       st.is_starred, st.is_pinned, st.is_muted
                FROM messenger_messages m
                JOIN messenger_message_states st
                  ON st.message_id = m.id AND st.user_id = $2::uuid
                WHERE m.id = $1::uuid AND st.deleted_at IS NULL
                """,
                message_id,
                resolved,
            )
        if not row:
            return None
        from_user = await self._user_summary(str(row["sender_id"]))
        to_user = await self._user_summary(str(row["recipient_id"]))
        attachments = await self._enrich_attachments(message_id)
        result = {
            "id": str(row["id"]),
            "conversation_id": str(row["conversation_id"]),
            "from": from_user or {"id": str(row["sender_id"])},
            "to": to_user or {"id": str(row["recipient_id"])},
            "subject": row["subject"] or "",
            "body": row["body"] or "",
            "body_format": row["body_format"],
            "read": bool(row["is_read"]),
            "folder": row["folder"],
            "priority": row["priority"],
            "delivery_status": row["delivery_status"],
            "attachments": attachments,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }
        result.update(self._merge_message_meta(row, {
            "is_starred": _record_get(row, "is_starred"),
            "is_pinned": _record_get(row, "is_pinned"),
            "is_muted": _record_get(row, "is_muted"),
        }))
        return result

    async def _validate_artifact(
        self,
        sender_id: str,
        artifact_id: str,
        source: str,
    ) -> Tuple[bool, Optional[str]]:
        art = await self.postgres.artifact_get(artifact_id)
        if not art:
            return False, "artifact_not_found"
        if source == "upload":
            if str(art.get("user_id") or "") != sender_id:
                meta = art.get("metadata") or {}
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                if meta.get("source") != "user_messenger" or str(art.get("user_id") or "") != sender_id:
                    return False, "artifact_not_owned"
        else:
            access = await self.postgres.artifact_is_accessible(user_id=sender_id, artifact_id=artifact_id)
            if not access.get("accessible"):
                return False, "artifact_not_accessible"
        return True, None

    async def _grant_file_access(
        self,
        conn,
        *,
        recipient_id: str,
        artifact_id: str,
        sender_id: str,
        message_id: str,
        share_mode: str,
        expires_sec: int = 86400,
    ) -> Optional[str]:
        if share_mode == "share_link":
            share = await self.postgres.share_token_create(
                artifact_id=artifact_id,
                created_by=sender_id,
                expires_sec=expires_sec,
                scope={"messenger_message_id": message_id, "recipient_id": recipient_id},
            )
            token = share.get("token")
            share_token_id = share.get("token_id")
            return json.dumps({
                "share_token_id": share_token_id,
                "share_url": f"/storage/shares/{token}/download" if token else None,
                "share_expires_at": share.get("expires_at"),
            })
        await conn.execute(
            """
            INSERT INTO user_file_overrides (user_id, artifact_id, permissions, reason, created_by)
            VALUES ($1::uuid, $2::uuid, ARRAY['read'], $3, $4)
            ON CONFLICT (user_id, artifact_id) DO UPDATE SET
                permissions = EXCLUDED.permissions,
                reason = EXCLUDED.reason,
                created_by = EXCLUDED.created_by
            """,
            recipient_id,
            artifact_id,
            f"messenger:message:{message_id}",
            sender_id,
        )
        return None

    async def send_message(
        self,
        sender_id: str,
        *,
        to: str,
        subject: Optional[str] = None,
        body: str = "",
        attachments: Optional[Sequence[_ATTACHMENT_SPEC]] = None,
        priority: str = "normal",
        client_message_id: Optional[str] = None,
        draft: bool = False,
        parent_message_id: Optional[str] = None,
        cc: Optional[Sequence[str]] = None,
        bcc: Optional[Sequence[str]] = None,
        content_type: Optional[str] = None,
        voice_duration_ms: Optional[int] = None,
        scheduled_at: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        resolved_sender = await self._resolve_user_id(sender_id)
        if not resolved_sender:
            log.error("send_message: sender not found %s", sender_id)
            return None

        parent_row = None
        thread_root_id = None
        resolved_recipient = await self._resolve_user_id(to) if (to or "").strip() else None
        if parent_message_id and not draft:
            if not self.postgres.pool:
                return None
            async with self.postgres.pool.acquire() as conn:
                parent_row = await conn.fetchrow(
                    """
                    SELECT id, conversation_id, sender_id, recipient_id, thread_root_id, subject
                    FROM messenger_messages WHERE id = $1::uuid
                    """,
                    parent_message_id,
                )
            if not parent_row:
                raise ValueError("parent_not_found")
            if str(parent_row["sender_id"]) != resolved_sender and str(parent_row["recipient_id"]) != resolved_sender:
                raise ValueError("parent_access_denied")
            resolved_recipient = (
                str(parent_row["recipient_id"])
                if str(parent_row["sender_id"]) == resolved_sender
                else str(parent_row["sender_id"])
            )
            thread_root_id = str(parent_row["thread_root_id"] or parent_row["id"])
        elif not draft and not resolved_recipient:
            log.error("send_message: recipient not found %s", to)
            return None

        priority = (priority or "normal").strip().lower()
        if priority not in ("low", "normal", "high"):
            priority = "normal"

        attach_specs = _normalize_attachments(attachments)
        for spec in attach_specs:
            ok, err = await self._validate_artifact(resolved_sender, spec["artifact_id"], spec["source"])
            if not ok:
                raise ValueError(err or "invalid_attachment")

        if not self.postgres.pool:
            return None

        preview = _strip_html(body)
        delivery = "pending" if draft else "sent"

        metadata: Dict[str, Any] = {}
        if cc:
            metadata["cc"] = [str(x).strip() for x in cc if str(x).strip()]
        if bcc:
            metadata["bcc"] = [str(x).strip() for x in bcc if str(x).strip()]
        if content_type:
            metadata["content_type"] = str(content_type).strip()
        if voice_duration_ms is not None:
            metadata["voice_duration_ms"] = int(voice_duration_ms)
        if scheduled_at:
            metadata["scheduled_at"] = str(scheduled_at).strip()

        async with self.postgres.pool.acquire() as conn:
            async with conn.transaction():
                if client_message_id:
                    existing = await conn.fetchrow(
                        """
                        SELECT id, created_at FROM messenger_messages
                        WHERE sender_id = $1::uuid AND client_message_id = $2
                        """,
                        resolved_sender,
                        client_message_id,
                    )
                    if existing:
                        return {
                            "id": str(existing["id"]),
                            "created_at": existing["created_at"].isoformat() if existing["created_at"] else None,
                            "duplicate": True,
                        }

                conv_id = None
                if parent_row:
                    conv_id = str(parent_row["conversation_id"])
                    if not subject:
                        subject = parent_row["subject"]
                elif resolved_recipient and not draft:
                    conv_id = await self._get_or_create_dm_conversation(
                        conn,
                        sender_id=resolved_sender,
                        recipient_id=resolved_recipient,
                        subject=subject,
                    )
                elif draft:
                    row = await conn.fetchrow(
                        """
                        INSERT INTO messenger_conversations (type, subject, created_by)
                        VALUES ('dm', $1, $2::uuid)
                        RETURNING id
                        """,
                        subject,
                        resolved_sender,
                    )
                    conv_id = str(row["id"])
                    resolved_recipient = resolved_recipient or resolved_sender
                else:
                    return None

                msg_row = await conn.fetchrow(
                    """
                    INSERT INTO messenger_messages (
                        conversation_id, sender_id, recipient_id, client_message_id,
                        parent_message_id, thread_root_id,
                        subject, body, preview, priority, delivery_status, metadata
                    )
                    VALUES ($1::uuid, $2::uuid, $3::uuid, $4, $5::uuid, $6::uuid, $7, $8, $9, $10, $11, $12::jsonb)
                    RETURNING id, created_at
                    """,
                    conv_id,
                    resolved_sender,
                    resolved_recipient,
                    client_message_id,
                    parent_message_id if parent_row else None,
                    thread_root_id,
                    subject,
                    body,
                    preview,
                    priority,
                    delivery,
                    json.dumps(metadata),
                )
                message_id = str(msg_row["id"])

                if parent_message_id and parent_row:
                    await conn.execute(
                        """
                        UPDATE messenger_messages
                        SET reply_count = reply_count + 1, updated_at = now()
                        WHERE id = $1::uuid
                        """,
                        parent_message_id,
                    )

                sender_folder = "drafts" if draft else "sent"
                await conn.execute(
                    """
                    INSERT INTO messenger_message_states (message_id, user_id, folder, is_read)
                    VALUES ($1::uuid, $2::uuid, $3, $4)
                    """,
                    message_id,
                    resolved_sender,
                    sender_folder,
                    True,
                )
                if not draft and resolved_recipient != resolved_sender:
                    await conn.execute(
                        """
                        INSERT INTO messenger_message_states (message_id, user_id, folder, is_read)
                        VALUES ($1::uuid, $2::uuid, 'inbox', false)
                        """,
                        message_id,
                        resolved_recipient,
                    )

                for idx, spec in enumerate(attach_specs):
                    meta_json = None
                    share_token_id = None
                    if not draft and resolved_recipient:
                        grant_meta = await self._grant_file_access(
                            conn,
                            recipient_id=resolved_recipient,
                            artifact_id=spec["artifact_id"],
                            sender_id=resolved_sender,
                            message_id=message_id,
                            share_mode=spec["share_mode"],
                            expires_sec=int(spec.get("expires_sec") or 86400),
                        )
                        if grant_meta:
                            meta = json.loads(grant_meta)
                            share_token_id = meta.get("share_token_id")
                            meta_json = json.dumps({
                                "share_url": meta.get("share_url"),
                                "share_expires_at": meta.get("share_expires_at"),
                            })
                    await conn.execute(
                        """
                        INSERT INTO messenger_message_attachments (
                            message_id, artifact_id, source, share_mode, share_token_id, sort_order, metadata
                        )
                        VALUES ($1::uuid, $2::uuid, $3, $4, $5::uuid, $6, COALESCE($7::jsonb, '{}'::jsonb))
                        """,
                        message_id,
                        spec["artifact_id"],
                        spec["source"],
                        spec["share_mode"],
                        share_token_id,
                        idx,
                        meta_json,
                    )

                await conn.execute(
                    "UPDATE messenger_conversations SET updated_at = now() WHERE id = $1::uuid",
                    conv_id,
                )

        return {
            "id": message_id,
            "created_at": msg_row["created_at"].isoformat() if msg_row["created_at"] else None,
        }

    async def update_message(
        self,
        user_id: str,
        message_id: str,
        *,
        read: Optional[bool] = None,
        folder: Optional[str] = None,
        body: Optional[str] = None,
        starred: Optional[bool] = None,
        pinned: Optional[bool] = None,
        muted: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        resolved = await self._resolve_user_id(user_id)
        if not resolved or not await self._message_access(resolved, message_id):
            return None
        if not self.postgres.pool:
            return None

        folder_norm = (folder or "").strip().lower() if folder is not None else None
        if folder_norm and folder_norm not in _VALID_FOLDERS:
            raise ValueError("invalid_folder")

        state_updates = []
        params: List[Any] = []
        idx = 1
        if read is not None:
            state_updates.append(f"is_read = ${idx}")
            params.append(bool(read))
            idx += 1
        if folder_norm:
            state_updates.append(f"folder = ${idx}")
            params.append(folder_norm)
            idx += 1
        if starred is not None:
            state_updates.append(f"is_starred = ${idx}")
            params.append(bool(starred))
            idx += 1
        if pinned is not None:
            state_updates.append(f"is_pinned = ${idx}")
            params.append(bool(pinned))
            idx += 1
        if muted is not None:
            state_updates.append(f"is_muted = ${idx}")
            params.append(bool(muted))
            idx += 1

        async with self.postgres.pool.acquire() as conn:
            if body is not None:
                sender_row = await conn.fetchrow(
                    "SELECT sender_id FROM messenger_messages WHERE id = $1::uuid",
                    message_id,
                )
                if not sender_row or str(sender_row["sender_id"]) != resolved:
                    raise ValueError("edit_not_allowed")
                preview = _strip_html(body)
                await conn.execute(
                    """
                    UPDATE messenger_messages
                    SET body = $1, preview = $2, edited_at = now(), updated_at = now()
                    WHERE id = $3::uuid
                    """,
                    body,
                    preview,
                    message_id,
                )

            row = None
            if state_updates:
                state_updates.append("updated_at = now()")
                params.extend([message_id, resolved])
                sql = f"""
                    UPDATE messenger_message_states
                    SET {", ".join(state_updates)}
                    WHERE message_id = ${idx}::uuid AND user_id = ${idx + 1}::uuid
                    RETURNING message_id
                """
                row = await conn.fetchrow(sql, *params)
                if read:
                    await conn.execute(
                        """
                        UPDATE messenger_messages SET delivery_status = 'read', updated_at = now()
                        WHERE id = $1::uuid AND recipient_id = $2::uuid
                        """,
                        message_id,
                        resolved,
                    )
            elif body is None:
                return await self.get_message(resolved, message_id)

        if state_updates and not row:
            return None
        return await self.get_message(resolved, message_id)

    async def delete_message(self, user_id: str, message_id: str) -> bool:
        resolved = await self._resolve_user_id(user_id)
        if not resolved or not await self._message_access(resolved, message_id):
            return False
        if not self.postgres.pool:
            return False
        async with self.postgres.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    UPDATE messenger_message_states
                    SET deleted_at = now(), folder = 'trash', updated_at = now()
                    WHERE message_id = $1::uuid AND user_id = $2::uuid
                    """,
                    message_id,
                    resolved,
                )
                tokens = await conn.fetch(
                    """
                    SELECT ma.share_token_id
                    FROM messenger_message_attachments ma
                    WHERE ma.message_id = $1::uuid AND ma.share_token_id IS NOT NULL
                    """,
                    message_id,
                )
                for tok in tokens:
                    if tok["share_token_id"]:
                        await conn.execute(
                            """
                            UPDATE artifact_share_tokens SET revoked_at = now()
                            WHERE id = $1::uuid AND revoked_at IS NULL
                            """,
                            tok["share_token_id"],
                        )
        return True

    async def list_attachable_artifacts(
        self,
        user_id: str,
        *,
        page: int = 1,
        page_size: int = 25,
        search: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved = await self._resolve_user_id(user_id)
        if not resolved:
            return {"items": [], "paging": {"page": page, "page_size": page_size, "total_count": 0}}
        offset = (max(1, page) - 1) * max(1, min(page_size, 100))
        return await self.postgres.artifact_accessible_list(
            user_id=resolved,
            limit=max(1, min(page_size, 100)),
            offset=offset,
            search=search,
            media_type=media_type,
            ownership="any",
        )

    async def search_messages(
        self,
        user_id: str,
        *,
        q: str,
        folder: Optional[str] = None,
        page: int = 1,
        limit: int = 30,
    ) -> Dict[str, Any]:
        folder_param = folder if folder else "inbox"
        result = await self.list_messages(
            user_id, folder=folder_param, page=page, limit=limit, q=q
        )
        result["query"] = q
        if folder:
            result["folder"] = folder
        else:
            result["folder"] = None
            # search all folders user can see
            if not self.postgres.pool:
                return result
            resolved = await self._resolve_user_id(user_id)
            if not resolved:
                return result
            limit_i = max(1, min(int(limit), 100))
            offset = (max(1, page) - 1) * limit_i
            term = f"%{q.strip()}%"
            async with self.postgres.pool.acquire() as conn:
                total = await conn.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM messenger_messages m
                    JOIN messenger_message_states st ON st.message_id = m.id AND st.user_id = $1::uuid
                    WHERE st.deleted_at IS NULL
                      AND (m.subject ILIKE $2 OR m.preview ILIKE $2 OR m.body ILIKE $2)
                    """,
                    resolved,
                    term,
                )
                rows = await conn.fetch(
                    """
                    SELECT m.id, m.conversation_id, m.sender_id, m.recipient_id,
                           m.subject, m.preview, m.priority, m.reply_count, st.folder,
                           st.is_read, m.created_at
                    FROM messenger_messages m
                    JOIN messenger_message_states st ON st.message_id = m.id AND st.user_id = $1::uuid
                    WHERE st.deleted_at IS NULL
                      AND (m.subject ILIKE $2 OR m.preview ILIKE $2 OR m.body ILIKE $2)
                    ORDER BY m.created_at DESC
                    LIMIT $3 OFFSET $4
                    """,
                    resolved,
                    term,
                    limit_i,
                    offset,
                )
            items = [
                {
                    "id": str(r["id"]),
                    "conversation_id": str(r["conversation_id"]),
                    "from": str(r["sender_id"]),
                    "to": str(r["recipient_id"]),
                    "subject": r["subject"] or "",
                    "preview": r["preview"] or "",
                    "read": bool(r["is_read"]),
                    "folder": r["folder"],
                    "reply_count": r["reply_count"],
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                }
                for r in rows
            ]
            result = {
                "items": items,
                "total": int(total or 0),
                "page": page,
                "limit": limit_i,
                "query": q,
                "folder": None,
            }
        return result

    async def list_thread_replies(
        self,
        user_id: str,
        message_id: str,
        *,
        limit: int = 50,
    ) -> Dict[str, Any]:
        resolved = await self._resolve_user_id(user_id)
        if not resolved or not await self._message_access(resolved, message_id):
            return {"items": [], "total": 0}
        if not self.postgres.pool:
            return {"items": [], "total": 0}
        async with self.postgres.pool.acquire() as conn:
            root = await conn.fetchval(
                """
                SELECT COALESCE(thread_root_id, id) FROM messenger_messages WHERE id = $1::uuid
                """,
                message_id,
            )
            rows = await conn.fetch(
                """
                SELECT m.id, m.sender_id, m.recipient_id, m.body, m.preview, m.created_at
                FROM messenger_messages m
                WHERE (m.thread_root_id = $1::uuid OR m.id = $1::uuid)
                  AND m.id != $2::uuid
                ORDER BY m.created_at ASC
                LIMIT $3
                """,
                root,
                message_id,
                max(1, min(limit, 100)),
            )
        return {
            "thread_root_id": str(root) if root else message_id,
            "items": [
                {
                    "id": str(r["id"]),
                    "from": str(r["sender_id"]),
                    "to": str(r["recipient_id"]),
                    "preview": r["preview"] or "",
                    "body": r["body"] or "",
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                }
                for r in rows
            ],
            "total": len(rows),
        }

    async def reply_to_message(
        self,
        user_id: str,
        message_id: str,
        *,
        body: str,
        attachments: Optional[Sequence[_ATTACHMENT_SPEC]] = None,
        client_message_id: Optional[str] = None,
        content_type: Optional[str] = None,
        voice_duration_ms: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        return await self.send_message(
            user_id,
            to="",
            body=body,
            attachments=attachments,
            client_message_id=client_message_id,
            parent_message_id=message_id,
            content_type=content_type,
            voice_duration_ms=voice_duration_ms,
        )

    async def resend_message(self, user_id: str, message_id: str) -> Optional[Dict[str, Any]]:
        resolved = await self._resolve_user_id(user_id)
        if not resolved:
            return None
        msg = await self.get_message(resolved, message_id)
        if not msg:
            return None
        to_raw = msg.get("to")
        to_id = to_raw.get("id") if isinstance(to_raw, dict) else str(to_raw or "")
        if not to_id:
            raise ValueError("missing_recipient")
        attach_ids = [
            str(a.get("artifact_id") or a.get("id"))
            for a in (msg.get("attachments") or [])
            if a.get("artifact_id") or a.get("id")
        ]
        return await self.send_message(
            resolved,
            to=to_id,
            subject=msg.get("subject"),
            body=msg.get("body") or "",
            attachments=attach_ids or None,
            priority=msg.get("priority") or "normal",
        )
