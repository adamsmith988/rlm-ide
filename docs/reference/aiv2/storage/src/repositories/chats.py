"""
Chat Repository
مدیریت چت‌ها و پیام‌ها در PostgreSQL
"""
from __future__ import annotations

import logging
import json
from typing import Any, Dict, List, Optional
from datetime import datetime

log = logging.getLogger(__name__)


class ChatRepository:
    """Repository for chat sessions and messages."""
    
    def __init__(self, postgres):
        self.postgres = postgres
    
    # =========================================================================
    # USER ID RESOLUTION - supports both keycloak_id and users.id
    # =========================================================================
    
    async def _resolve_user_id(self, user_id: str) -> Optional[str]:
        """
        Resolve user_id to PostgreSQL users.id.
        Input can be either keycloak_id or users.id.
        Returns the users.id or None if not found.
        """
        if not self.postgres.pool:
            return None
        
        async with self.postgres.pool.acquire() as conn:
            # First try as keycloak_id
            row = await conn.fetchrow(
                "SELECT id FROM users WHERE keycloak_id = $1",
                user_id
            )
            if row:
                return str(row["id"])
            
            # Then try as users.id directly (only if it looks like a UUID)
            try:
                import uuid as uuid_module
                uuid_module.UUID(user_id)  # Validate UUID format
                row = await conn.fetchrow(
                    "SELECT id FROM users WHERE id = $1::uuid",
                    user_id
                )
                if row:
                    return str(row["id"])
            except (ValueError, AttributeError):
                pass  # Not a valid UUID, skip
            
            log.warning(f"Could not resolve user_id: {user_id}")
            return None
    
    # =========================================================================
    # CHAT SESSIONS
    # =========================================================================
    
    async def create_session(
        self,
        user_id: str,
        title: str = "چت جدید",
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create a new chat session."""
        if not self.postgres.pool:
            return None
        
        # Resolve user_id (supports keycloak_id or users.id)
        resolved_user_id = await self._resolve_user_id(user_id)
        if not resolved_user_id:
            log.error(f"Cannot create session: user not found for id {user_id}")
            return None
        
        async with self.postgres.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO chat_sessions (user_id, title, model, system_prompt, metadata)
                VALUES ($1::uuid, $2, $3, $4, $5::jsonb)
                RETURNING id, user_id, title, model, system_prompt, is_archived, is_pinned, 
                          metadata, created_at, updated_at
                """,
                resolved_user_id, title, model, system_prompt, json.dumps(metadata or {})
            )
            
            if not row:
                return None
            
            return self._row_to_session(row)
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a chat session by ID."""
        if not self.postgres.pool:
            return None
        
        async with self.postgres.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, user_id, title, model, system_prompt, is_archived, is_pinned, 
                       metadata, created_at, updated_at
                FROM chat_sessions
                WHERE id = $1::uuid
                """,
                session_id
            )
            
            if not row:
                return None
            
            return self._row_to_session(row)
    
    async def get_session_with_messages(
        self,
        session_id: str,
        message_limit: int = 100,
    ) -> Optional[Dict[str, Any]]:
        """Get session with its messages."""
        session = await self.get_session(session_id)
        if not session:
            return None
        
        messages = await self.list_messages(session_id, limit=message_limit)
        session["messages"] = messages
        return session
    
    async def list_sessions(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        include_archived: bool = False,
    ) -> List[Dict[str, Any]]:
        """List chat sessions for a user."""
        if not self.postgres.pool:
            return []
        
        # Resolve user_id (supports keycloak_id or users.id)
        resolved_user_id = await self._resolve_user_id(user_id)
        if not resolved_user_id:
            log.warning(f"Cannot list sessions: user not found for id {user_id}")
            return []
        
        async with self.postgres.pool.acquire() as conn:
            if include_archived:
                rows = await conn.fetch(
                    """
                    SELECT id, user_id, title, model, system_prompt, is_archived, is_pinned, 
                           metadata, created_at, updated_at
                    FROM chat_sessions
                    WHERE user_id = $1::uuid
                    ORDER BY is_pinned DESC, updated_at DESC
                    LIMIT $2 OFFSET $3
                    """,
                    resolved_user_id, limit, offset
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, user_id, title, model, system_prompt, is_archived, is_pinned, 
                           metadata, created_at, updated_at
                    FROM chat_sessions
                    WHERE user_id = $1::uuid AND is_archived = FALSE
                    ORDER BY is_pinned DESC, updated_at DESC
                    LIMIT $2 OFFSET $3
                    """,
                    resolved_user_id, limit, offset
                )
            
            return [self._row_to_session(row) for row in rows]

    async def find_dock_session(
        self,
        user_id: str,
        surface: str,
        anchor_key: str,
    ) -> Optional[Dict[str, Any]]:
        """Find the active dock session for (user, surface, anchor_key)."""
        if not self.postgres.pool:
            return None
        resolved_user_id = await self._resolve_user_id(user_id)
        if not resolved_user_id:
            return None
        surface = (surface or "").strip()
        anchor_key = (anchor_key or "").strip()
        if not surface or not anchor_key:
            return None
        async with self.postgres.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, user_id, title, model, system_prompt, is_archived, is_pinned,
                       metadata, created_at, updated_at
                FROM chat_sessions
                WHERE user_id = $1::uuid
                  AND COALESCE(is_archived, FALSE) = FALSE
                  AND COALESCE(metadata->>'is_dock_session', 'false') = 'true'
                  AND metadata->>'surface' = $2
                  AND metadata->>'anchor_key' = $3
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                resolved_user_id,
                surface,
                anchor_key,
            )
            return self._row_to_session(row) if row else None

    async def get_or_create_dock_session(
        self,
        user_id: str,
        surface: str,
        anchor_key: str,
        title: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return existing dock session or create one with dock metadata."""
        existing = await self.find_dock_session(user_id, surface, anchor_key)
        if existing:
            return existing
        surface = (surface or "").strip() or "general"
        anchor_key = (anchor_key or "").strip() or f"{surface}:/"
        meta = {
            "chat_mode": "surface",
            "surface": surface,
            "anchor_key": anchor_key,
            "is_dock_session": True,
        }
        return await self.create_session(
            user_id=user_id,
            title=title or f"Assistant · {surface}",
            metadata=meta,
        )

    async def promote_dock_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Clear dock flag so the session behaves as a normal hub chat."""
        return await self.update_session(
            session_id=session_id,
            metadata={"is_dock_session": False},
        )
    
    async def update_session(
        self,
        session_id: str,
        title: Optional[str] = None,
        is_archived: Optional[bool] = None,
        is_pinned: Optional[bool] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update a chat session."""
        if not self.postgres.pool:
            return None
        
        updates = []
        params = []
        idx = 1
        
        if title is not None:
            updates.append(f"title = ${idx}")
            params.append(title)
            idx += 1
        
        if is_archived is not None:
            updates.append(f"is_archived = ${idx}")
            params.append(is_archived)
            idx += 1
        
        if is_pinned is not None:
            updates.append(f"is_pinned = ${idx}")
            params.append(is_pinned)
            idx += 1
        
        if model is not None:
            updates.append(f"model = ${idx}")
            params.append(model)
            idx += 1
        
        if system_prompt is not None:
            updates.append(f"system_prompt = ${idx}")
            params.append(system_prompt)
            idx += 1
        
        if metadata is not None:
            updates.append(f"metadata = metadata || ${idx}::jsonb")
            params.append(json.dumps(metadata))
            idx += 1
        
        if not updates:
            return await self.get_session(session_id)
        
        params.append(session_id)
        
        async with self.postgres.pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE chat_sessions
                SET {', '.join(updates)}
                WHERE id = ${idx}::uuid
                RETURNING id, user_id, title, model, system_prompt, is_archived, is_pinned, 
                          metadata, created_at, updated_at
                """,
                *params
            )
            
            if not row:
                return None
            
            return self._row_to_session(row)
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and all its messages."""
        if not self.postgres.pool:
            return False
        
        async with self.postgres.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM chat_sessions WHERE id = $1::uuid",
                session_id
            )
            return "DELETE 1" in result
    
    async def count_sessions(self, user_id: str, include_archived: bool = False) -> int:
        """Count sessions for a user."""
        if not self.postgres.pool:
            return 0
        
        # Resolve user_id (supports keycloak_id or users.id)
        resolved_user_id = await self._resolve_user_id(user_id)
        if not resolved_user_id:
            return 0
        
        async with self.postgres.pool.acquire() as conn:
            if include_archived:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM chat_sessions WHERE user_id = $1::uuid",
                    resolved_user_id
                )
            else:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM chat_sessions WHERE user_id = $1::uuid AND is_archived = FALSE",
                    resolved_user_id
                )
            return count or 0
    
    # =========================================================================
    # MESSAGES
    # =========================================================================
    
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict]] = None,
        tool_results: Optional[List[Dict]] = None,
        thinking: Optional[str] = None,
        tokens_prompt: Optional[int] = None,
        tokens_completion: Optional[int] = None,
        attachments: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
        user_id: Optional[str] = None,  # Added for auto-create session
        trace_id: Optional[str] = None,  # Direct trace reference
    ) -> Optional[Dict[str, Any]]:
        """Add a message to a session. Auto-creates session if it doesn't exist."""
        if not self.postgres.pool:
            return None
        
        async with self.postgres.pool.acquire() as conn:
            # Check if session exists
            session_exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM chat_sessions WHERE id = $1::uuid)",
                session_id
            )
            
            # Auto-create session if it doesn't exist
            if not session_exists:
                log.info(f"Auto-creating session {session_id} for user {user_id or 'unknown'}")
                # Insert session with minimal info - will be associated with user from metadata or default
                resolved_user_id = None
                if user_id:
                    resolved_user_id = await self._resolve_user_id(user_id)
                
                if resolved_user_id:
                    await conn.execute(
                        """
                        INSERT INTO chat_sessions (id, user_id, title, model, metadata)
                        VALUES ($1::uuid, $2::uuid, $3, $4, $5::jsonb)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        session_id, resolved_user_id, "چت جدید", None, json.dumps({})
                    )
                else:
                    # Create session without user (will use first user as fallback)
                    first_user = await conn.fetchval("SELECT id FROM users LIMIT 1")
                    if first_user:
                        await conn.execute(
                            """
                            INSERT INTO chat_sessions (id, user_id, title, model, metadata)
                            VALUES ($1::uuid, $2::uuid, $3, $4, $5::jsonb)
                            ON CONFLICT (id) DO NOTHING
                            """,
                            session_id, first_user, "چت جدید", None, json.dumps({})
                        )
            
            row = await conn.fetchrow(
                """
                INSERT INTO chat_messages 
                    (session_id, role, content, trace_id, tool_calls, tool_results, thinking,
                     tokens_prompt, tokens_completion, attachments, metadata)
                VALUES ($1::uuid, $2, $3, $4, $5::jsonb, $6::jsonb, $7, $8, $9, $10::jsonb, $11::jsonb)
                RETURNING id, session_id, role, content, trace_id, tool_calls, tool_results, thinking,
                          tokens_prompt, tokens_completion, attachments, metadata, created_at
                """,
                session_id, role, content, trace_id,
                json.dumps(tool_calls) if tool_calls else None,
                json.dumps(tool_results) if tool_results else None,
                thinking, tokens_prompt, tokens_completion,
                json.dumps(attachments or []),
                json.dumps(metadata or {})
            )
            
            if not row:
                return None
            
            # Update session updated_at
            await conn.execute(
                "UPDATE chat_sessions SET updated_at = now() WHERE id = $1::uuid",
                session_id
            )
            
            return self._row_to_message(row)
    
    async def list_messages(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
        before_id: Optional[str] = None,
        after_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List messages in a session."""
        if not self.postgres.pool:
            return []
        
        async with self.postgres.pool.acquire() as conn:
            if before_id:
                rows = await conn.fetch(
                    """
                    SELECT id, session_id, role, content, trace_id, tool_calls, tool_results, thinking,
                           tokens_prompt, tokens_completion, attachments, metadata, created_at
                    FROM chat_messages
                    WHERE session_id = $1::uuid AND id < $2::uuid
                    ORDER BY created_at DESC
                    LIMIT $3
                    """,
                    session_id, before_id, limit
                )
                rows = list(reversed(rows))
            elif after_id:
                rows = await conn.fetch(
                    """
                    SELECT id, session_id, role, content, trace_id, tool_calls, tool_results, thinking,
                           tokens_prompt, tokens_completion, attachments, metadata, created_at
                    FROM chat_messages
                    WHERE session_id = $1::uuid AND id > $2::uuid
                    ORDER BY created_at ASC
                    LIMIT $3
                    """,
                    session_id, after_id, limit
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, session_id, role, content, trace_id, tool_calls, tool_results, thinking,
                           tokens_prompt, tokens_completion, attachments, metadata, created_at
                    FROM chat_messages
                    WHERE session_id = $1::uuid
                    ORDER BY created_at ASC
                    LIMIT $2 OFFSET $3
                    """,
                    session_id, limit, offset
                )
            
            return [self._row_to_message(row) for row in rows]
    
    async def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get a single message."""
        if not self.postgres.pool:
            return None
        
        async with self.postgres.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, session_id, role, content, trace_id, tool_calls, tool_results, thinking,
                       tokens_prompt, tokens_completion, attachments, metadata, created_at
                FROM chat_messages
                WHERE id = $1::uuid
                """,
                message_id
            )
            
            if not row:
                return None
            
            return self._row_to_message(row)
    
    async def delete_message(self, message_id: str) -> bool:
        """Delete a message."""
        if not self.postgres.pool:
            return False
        
        async with self.postgres.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM chat_messages WHERE id = $1::uuid",
                message_id
            )
            return "DELETE 1" in result
    
    async def clear_messages(self, session_id: str) -> int:
        """Clear all messages in a session."""
        if not self.postgres.pool:
            return 0
        
        async with self.postgres.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM chat_messages WHERE session_id = $1::uuid",
                session_id
            )
            # Extract count from "DELETE X"
            try:
                return int(result.split()[1])
            except (IndexError, ValueError):
                return 0
    
    async def global_counts(self) -> Dict[str, int]:
        """Aggregate chat session and message counts for admin dashboard."""
        if not self.postgres.pool:
            return {"total_chat_sessions": 0, "total_messages": 0}
        async with self.postgres.pool.acquire() as conn:
            sessions_row = await conn.fetchrow(
                "SELECT COUNT(*)::int AS c FROM chat_sessions"
            )
            messages_row = await conn.fetchrow(
                "SELECT COUNT(*)::int AS c FROM chat_messages"
            )
        return {
            "total_chat_sessions": int(sessions_row["c"]) if sessions_row else 0,
            "total_messages": int(messages_row["c"]) if messages_row else 0,
        }

    async def count_messages(self, session_id: str) -> int:
        """Count messages in a session."""
        if not self.postgres.pool:
            return 0
        
        async with self.postgres.pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM chat_messages WHERE session_id = $1::uuid",
                session_id
            )
            return count or 0
    
    async def get_messages_for_context(
        self,
        session_id: str,
        max_messages: int = 20,
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get recent messages for LLM context.
        Returns messages in chronological order, limited by count or tokens.
        """
        if not self.postgres.pool:
            return []
        
        async with self.postgres.pool.acquire() as conn:
            # Get recent messages (reverse order)
            rows = await conn.fetch(
                """
                SELECT id, session_id, role, content, trace_id, tool_calls, tool_results, thinking,
                       tokens_prompt, tokens_completion, attachments, metadata, created_at
                FROM chat_messages
                WHERE session_id = $1::uuid
                ORDER BY created_at DESC
                LIMIT $2
                """,
                session_id, max_messages
            )
            
            # Reverse to get chronological order
            rows = list(reversed(rows))
            
            if max_tokens:
                # Simple token estimation (rough: 4 chars = 1 token)
                selected = []
                total_tokens = 0
                for row in reversed(rows):
                    content = row["content"] or ""
                    estimated_tokens = len(content) // 4
                    if total_tokens + estimated_tokens <= max_tokens:
                        selected.insert(0, row)
                        total_tokens += estimated_tokens
                    else:
                        break
                rows = selected
            
            return [self._row_to_message(row) for row in rows]
    
    # =========================================================================
    # AUTO-TITLE
    # =========================================================================
    
    async def auto_update_title(self, session_id: str) -> Optional[str]:
        """
        Automatically update session title based on first user message.
        Returns the new title.
        """
        if not self.postgres.pool:
            return None
        
        async with self.postgres.pool.acquire() as conn:
            # Get first user message
            row = await conn.fetchrow(
                """
                SELECT content FROM chat_messages
                WHERE session_id = $1::uuid AND role = 'user'
                ORDER BY created_at ASC
                LIMIT 1
                """,
                session_id
            )
            
            if not row or not row["content"]:
                return None
            
            # Create title from first message (max 50 chars)
            content = row["content"].strip()
            title = content[:47] + "..." if len(content) > 50 else content
            
            # Update session
            await conn.execute(
                "UPDATE chat_sessions SET title = $1 WHERE id = $2::uuid",
                title, session_id
            )
            
            return title
    
    # =========================================================================
    # MESSAGE EDIT & FEEDBACK (UX-004, UX-005)
    # =========================================================================
    
    async def update_message(
        self,
        message_id: str,
        content: str,
    ) -> Optional[Dict[str, Any]]:
        """Update message content (UX-004)."""
        if not self.postgres.pool:
            return None
        
        async with self.postgres.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE chat_messages
                SET content = $1, updated_at = NOW()
                WHERE id = $2::uuid
                RETURNING *
                """,
                content, message_id
            )
            
            if not row:
                return None
            
            return self._row_to_message(row)
    
    async def add_message_feedback(
        self,
        message_id: str,
        feedback: str,
        comment: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Add feedback to a message (UX-005)."""
        if not self.postgres.pool:
            return None
        
        async with self.postgres.pool.acquire() as conn:
            # Check if message exists
            exists = await conn.fetchval(
                "SELECT 1 FROM chat_messages WHERE id = $1::uuid",
                message_id
            )
            if not exists:
                return None
            
            # Upsert feedback (use metadata field or create a separate table)
            # For simplicity, store in message metadata
            row = await conn.fetchrow(
                """
                UPDATE chat_messages
                SET metadata = COALESCE(metadata, '{}')::jsonb || $1::jsonb,
                    updated_at = NOW()
                WHERE id = $2::uuid
                RETURNING *
                """,
                json.dumps({"feedback": feedback, "feedback_comment": comment}),
                message_id
            )
            
            if not row:
                return None
            
            return {
                "message_id": str(row["id"]),
                "feedback": feedback,
                "comment": comment,
                "created_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            }
    
    async def get_message_feedback(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get feedback for a message."""
        if not self.postgres.pool:
            return None
        
        async with self.postgres.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, metadata FROM chat_messages WHERE id = $1::uuid",
                message_id
            )
            
            if not row:
                return None
            
            metadata = row["metadata"] or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            
            return {
                "message_id": str(row["id"]),
                "feedback": metadata.get("feedback"),
                "comment": metadata.get("feedback_comment"),
            }
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _row_to_session(self, row) -> Dict[str, Any]:
        """Convert database row to session dict."""
        metadata = row["metadata"] or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}
        out = {
            "id": str(row["id"]),
            "user_id": str(row["user_id"]),
            "title": row["title"],
            "model": row["model"],
            "system_prompt": row["system_prompt"],
            "is_archived": row["is_archived"],
            "is_pinned": row["is_pinned"],
            "metadata": metadata,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        }
        # Flatten dock fields for FE ChatSession shape
        if isinstance(metadata, dict):
            if metadata.get("chat_mode"):
                out["chat_mode"] = metadata.get("chat_mode")
            if metadata.get("surface") is not None:
                out["surface"] = metadata.get("surface")
            if metadata.get("anchor_key") is not None:
                out["anchor_key"] = metadata.get("anchor_key")
            if "is_dock_session" in metadata:
                out["is_dock_session"] = bool(metadata.get("is_dock_session"))
        return out
    
    def _row_to_message(self, row) -> Dict[str, Any]:
        """Convert database row to message dict."""
        # Parse JSONB fields که ممکنه string برگردن
        attachments = row["attachments"]
        if isinstance(attachments, str):
            try:
                attachments = json.loads(attachments)
            except:
                attachments = []
        
        metadata = row["metadata"]
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        
        tool_calls = row["tool_calls"]
        if isinstance(tool_calls, str):
            try:
                tool_calls = json.loads(tool_calls)
            except:
                tool_calls = None
        
        tool_results = row["tool_results"]
        if isinstance(tool_results, str):
            try:
                tool_results = json.loads(tool_results)
            except:
                tool_results = None
        
        return {
            "id": str(row["id"]),
            "session_id": str(row["session_id"]),
            "role": row["role"],
            "content": row["content"],
            "trace_id": row.get("trace_id"),
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "thinking": row["thinking"],
            "tokens_prompt": row["tokens_prompt"],
            "tokens_completion": row["tokens_completion"],
            "attachments": attachments or [],
            "metadata": metadata or {},
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }
