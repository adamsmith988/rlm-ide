"""
User Repository
مدیریت کاربران در PostgreSQL
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import hashlib
import secrets

log = logging.getLogger(__name__)


class UserRepository:
    """Repository for user operations."""
    
    def __init__(self, postgres):
        self.postgres = postgres
    
    async def create_user(
        self,
        username: str,
        email: str,
        password_hash: Optional[str] = None,
        keycloak_id: Optional[str] = None,
        display_name: Optional[str] = None,
        role_name: str = "user",
    ) -> Optional[Dict[str, Any]]:
        """Create a new user."""
        if not self.postgres.pool:
            return None
        
        async with self.postgres.pool.acquire() as conn:
            # Get role ID
            role_row = await conn.fetchrow(
                "SELECT id FROM roles WHERE name = $1",
                role_name
            )
            role_id = role_row["id"] if role_row else None
            
            row = await conn.fetchrow(
                """
                INSERT INTO users (id, username, email, password_hash, keycloak_id, display_name, role_id)
                VALUES (COALESCE($4::uuid, gen_random_uuid()), $1, $2, $3, $4, $5, $6)
                ON CONFLICT (username) DO NOTHING
                RETURNING id, username, email, display_name, role_id, is_active, created_at
                """,
                username, email, password_hash, keycloak_id, display_name or username, role_id
            )
            
            if not row:
                return None
            
            return {
                "id": str(row["id"]),
                "username": row["username"],
                "email": row["email"],
                "display_name": row["display_name"],
                "role_id": str(row["role_id"]) if row["role_id"] else None,
                "is_active": row["is_active"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            }
    
    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        if not self.postgres.pool:
            return None
        
        async with self.postgres.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT u.id, u.username, u.email, u.password_hash, u.keycloak_id, 
                       u.display_name, u.avatar_url, u.is_active, u.last_login_at,
                       u.settings, u.metadata, u.created_at, u.updated_at,
                       r.name as role_name, r.permissions as role_permissions
                FROM users u
                LEFT JOIN roles r ON u.role_id = r.id
                WHERE u.id = $1
                """,
                user_id if isinstance(user_id, str) else str(user_id)
            )
            
            if not row:
                return None
            
            return self._row_to_user(row)
    
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username."""
        if not self.postgres.pool:
            return None
        
        async with self.postgres.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT u.id, u.username, u.email, u.password_hash, u.keycloak_id, 
                       u.display_name, u.avatar_url, u.is_active, u.last_login_at,
                       u.settings, u.metadata, u.created_at, u.updated_at,
                       r.name as role_name, r.permissions as role_permissions
                FROM users u
                LEFT JOIN roles r ON u.role_id = r.id
                WHERE u.username = $1
                """,
                username
            )
            
            if not row:
                return None
            
            return self._row_to_user(row)
    
    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email."""
        if not self.postgres.pool:
            return None
        
        async with self.postgres.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT u.id, u.username, u.email, u.password_hash, u.keycloak_id, 
                       u.display_name, u.avatar_url, u.is_active, u.last_login_at,
                       u.settings, u.metadata, u.created_at, u.updated_at,
                       r.name as role_name, r.permissions as role_permissions
                FROM users u
                LEFT JOIN roles r ON u.role_id = r.id
                WHERE u.email = $1
                """,
                email
            )
            
            if not row:
                return None
            
            return self._row_to_user(row)
    
    async def get_user_by_keycloak_id(self, keycloak_id: str) -> Optional[Dict[str, Any]]:
        """Get user by Keycloak ID."""
        if not self.postgres.pool:
            return None
        
        async with self.postgres.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT u.id, u.username, u.email, u.password_hash, u.keycloak_id, 
                       u.display_name, u.avatar_url, u.is_active, u.last_login_at,
                       u.settings, u.metadata, u.created_at, u.updated_at,
                       r.name as role_name, r.permissions as role_permissions
                FROM users u
                LEFT JOIN roles r ON u.role_id = r.id
                WHERE u.keycloak_id = $1
                """,
                keycloak_id
            )
            
            if not row:
                return None
            
            return self._row_to_user(row)
    
    async def create_or_update_by_keycloak(
        self,
        keycloak_id: str,
        username: str,
        email: str,
        display_name: Optional[str] = None,
        role_name: str = "user",
    ) -> Optional[Dict[str, Any]]:
        """
        Create or update user by Keycloak ID.
        Used for syncing users from Keycloak.
        """
        if not self.postgres.pool:
            return None
        
        async with self.postgres.pool.acquire() as conn:
            # Get role ID
            role_row = await conn.fetchrow(
                "SELECT id FROM roles WHERE name = $1",
                role_name
            )
            role_id = role_row["id"] if role_row else None
            
            row = await conn.fetchrow(
                """
                INSERT INTO users (username, email, keycloak_id, display_name, role_id)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (keycloak_id) DO UPDATE SET
                    username = EXCLUDED.username,
                    email = EXCLUDED.email,
                    display_name = COALESCE(EXCLUDED.display_name, users.display_name),
                    updated_at = now()
                RETURNING id, username, email, keycloak_id, display_name, role_id, is_active, created_at, updated_at
                """,
                username, email, keycloak_id, display_name or username, role_id
            )
            
            if not row:
                return None
            
            return {
                "id": str(row["id"]),
                "username": row["username"],
                "email": row["email"],
                "keycloak_id": row["keycloak_id"],
                "display_name": row["display_name"],
                "role_id": str(row["role_id"]) if row["role_id"] else None,
                "is_active": row["is_active"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            }

    async def update_user(
        self,
        user_id: str,
        display_name: Optional[str] = None,
        email: Optional[str] = None,
        avatar_url: Optional[str] = None,
        settings: Optional[Dict] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update user profile."""
        if not self.postgres.pool:
            return None
        
        updates = []
        params = []
        idx = 1
        
        if display_name is not None:
            updates.append(f"display_name = ${idx}")
            params.append(display_name)
            idx += 1
        
        if email is not None:
            updates.append(f"email = ${idx}")
            params.append(email)
            idx += 1
        
        if avatar_url is not None:
            updates.append(f"avatar_url = ${idx}")
            params.append(avatar_url)
            idx += 1
        
        if settings is not None:
            updates.append(f"settings = settings || ${idx}::jsonb")
            params.append(settings)
            idx += 1
        
        if is_active is not None:
            updates.append(f"is_active = ${idx}")
            params.append(is_active)
            idx += 1
        
        if not updates:
            return await self.get_user_by_id(user_id)
        
        params.append(user_id)
        
        async with self.postgres.pool.acquire() as conn:
            import json
            # Convert dict to JSON string for jsonb
            processed_params = []
            for p in params:
                if isinstance(p, dict):
                    processed_params.append(json.dumps(p))
                else:
                    processed_params.append(p)
            
            await conn.execute(
                f"UPDATE users SET {', '.join(updates)} WHERE id = ${idx}::uuid",
                *processed_params
            )
        
        return await self.get_user_by_id(user_id)
    
    async def update_last_login(self, user_id: str) -> None:
        """Update last login timestamp."""
        if not self.postgres.pool:
            return
        
        async with self.postgres.pool.acquire() as conn:
            await conn.execute(
                "UPDATE users SET last_login_at = now() WHERE id = $1::uuid",
                user_id
            )
    
    async def update_password(self, user_id: str, password_hash: str) -> bool:
        """Update user password."""
        if not self.postgres.pool:
            return False
        
        async with self.postgres.pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE users SET password_hash = $1 WHERE id = $2::uuid",
                password_hash, user_id
            )
            return "UPDATE 1" in result
    
    async def list_users(
        self,
        limit: int = 100,
        offset: int = 0,
        search: Optional[str] = None,
        role_name: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """List users with optional filters."""
        if not self.postgres.pool:
            return []
        
        conditions = ["1=1"]
        params = []
        idx = 1
        
        if search:
            conditions.append(f"(u.username ILIKE ${idx} OR u.email ILIKE ${idx} OR u.display_name ILIKE ${idx})")
            params.append(f"%{search}%")
            idx += 1
        
        if role_name:
            conditions.append(f"r.name = ${idx}")
            params.append(role_name)
            idx += 1
        
        if is_active is not None:
            conditions.append(f"u.is_active = ${idx}")
            params.append(is_active)
            idx += 1
        
        params.extend([limit, offset])
        
        async with self.postgres.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT u.id, u.username, u.email, u.display_name, u.avatar_url, 
                       u.is_active, u.last_login_at, u.created_at,
                       r.name as role_name
                FROM users u
                LEFT JOIN roles r ON u.role_id = r.id
                WHERE {' AND '.join(conditions)}
                ORDER BY u.created_at DESC
                LIMIT ${idx} OFFSET ${idx + 1}
                """,
                *params
            )
            
            return [
                {
                    "id": str(row["id"]),
                    "username": row["username"],
                    "email": row["email"],
                    "display_name": row["display_name"],
                    "avatar_url": row["avatar_url"],
                    "is_active": row["is_active"],
                    "role": row["role_name"],
                    "last_login_at": row["last_login_at"].isoformat() if row["last_login_at"] else None,
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                }
                for row in rows
            ]
    
    async def count_users(
        self,
        search: Optional[str] = None,
        role_name: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> int:
        """Count users with optional filters."""
        if not self.postgres.pool:
            return 0
        
        conditions = ["1=1"]
        params = []
        idx = 1
        
        if search:
            conditions.append(f"(u.username ILIKE ${idx} OR u.email ILIKE ${idx})")
            params.append(f"%{search}%")
            idx += 1
        
        if role_name:
            conditions.append(f"r.name = ${idx}")
            params.append(role_name)
            idx += 1
        
        if is_active is not None:
            conditions.append(f"u.is_active = ${idx}")
            params.append(is_active)
            idx += 1
        
        async with self.postgres.pool.acquire() as conn:
            count = await conn.fetchval(
                f"""
                SELECT COUNT(*) FROM users u
                LEFT JOIN roles r ON u.role_id = r.id
                WHERE {' AND '.join(conditions)}
                """,
                *params
            )
            return count or 0
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        if not self.postgres.pool:
            return False
        
        async with self.postgres.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM users WHERE id = $1::uuid",
                user_id
            )
            return "DELETE 1" in result
    
    async def set_user_role(self, user_id: str, role_name: str) -> bool:
        """Set user role."""
        if not self.postgres.pool:
            return False
        
        async with self.postgres.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE users SET role_id = (SELECT id FROM roles WHERE name = $1)
                WHERE id = $2::uuid
                """,
                role_name, user_id
            )
            return "UPDATE 1" in result
    
    async def log_activity(
        self,
        user_id: str,
        action: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict] = None,
    ) -> None:
        """Log user activity."""
        if not self.postgres.pool:
            return
        
        import json
        async with self.postgres.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO user_activity_log (user_id, action, ip_address, user_agent, details)
                VALUES ($1::uuid, $2, $3, $4, $5::jsonb)
                """,
                user_id, action, ip_address, user_agent, json.dumps(details or {})
            )
    
    def _row_to_user(self, row) -> Dict[str, Any]:
        """Convert database row to user dict."""
        return {
            "id": str(row["id"]),
            "username": row["username"],
            "email": row["email"],
            "password_hash": row["password_hash"],
            "keycloak_id": row["keycloak_id"],
            "display_name": row["display_name"],
            "avatar_url": row["avatar_url"],
            "is_active": row["is_active"],
            "last_login_at": row["last_login_at"].isoformat() if row["last_login_at"] else None,
            "settings": row["settings"] or {},
            "metadata": row["metadata"] or {},
            "role": row["role_name"],
            "permissions": row["role_permissions"] or [],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        }

    async def remap_user_id(self, old_id: str, new_id: str) -> bool:
        """
        Remap a user's primary key (e.g., to match Keycloak sub).
        Updates the users table and all FK references atomically.

        Used after Keycloak migration when the local users.id doesn't match the
        Keycloak user.sub – we rewire the legacy row to the new id so existing
        cases / chat_sessions / artifacts stay attached to the same user.
        """
        if not self.postgres.pool:
            return False

        async with self.postgres.pool.acquire() as conn:
            async with conn.transaction():
                old_user = await conn.fetchrow("SELECT id FROM users WHERE id = $1", old_id)
                if not old_user:
                    return False

                existing = await conn.fetchrow("SELECT id FROM users WHERE id = $1", new_id)
                if existing:
                    # New id already taken – cannot remap.
                    return False

                # Discover every FK pointing to users.id and rewrite them in bulk
                # so we don't need to enumerate tables manually.
                fk_rows = await conn.fetch(
                    """
                    SELECT con.conname        AS constraint_name,
                           cl.relname         AS table_name,
                           att.attname        AS column_name
                    FROM pg_constraint con
                    JOIN pg_class cl       ON cl.oid = con.conrelid
                    JOIN pg_class fcl      ON fcl.oid = con.confrelid
                    JOIN pg_attribute att  ON att.attrelid = con.conrelid
                                          AND att.attnum = ANY(con.conkey)
                    WHERE con.contype = 'f'
                      AND fcl.relname = 'users'
                      AND cl.relname <> 'users'
                    """
                )

                # Drop FKs, update rows, re-add FKs.
                for r in fk_rows:
                    await conn.execute(
                        f'ALTER TABLE "{r["table_name"]}" DROP CONSTRAINT IF EXISTS "{r["constraint_name"]}"'
                    )

                await conn.execute("UPDATE users SET id = $1 WHERE id = $2", new_id, old_id)

                for r in fk_rows:
                    await conn.execute(
                        f'UPDATE "{r["table_name"]}" SET "{r["column_name"]}" = $1 WHERE "{r["column_name"]}" = $2',
                        new_id, old_id,
                    )

                for r in fk_rows:
                    await conn.execute(
                        f'ALTER TABLE "{r["table_name"]}" '
                        f'ADD CONSTRAINT "{r["constraint_name"]}" '
                        f'FOREIGN KEY ("{r["column_name"]}") REFERENCES users(id) ON DELETE CASCADE'
                    )

                return True


# ============================================================================
# ROLE REPOSITORY
# ============================================================================

class RoleRepository:
    """Repository for role operations."""
    
    def __init__(self, postgres):
        self.postgres = postgres
    
    async def list_roles(self) -> List[Dict[str, Any]]:
        """List all roles."""
        if not self.postgres.pool:
            return []
        
        async with self.postgres.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, name, description, permissions, is_system, created_at FROM roles ORDER BY name"
            )
            return [
                {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "description": row["description"],
                    "permissions": row["permissions"] or [],
                    "is_system": row["is_system"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                }
                for row in rows
            ]
    
    async def get_role(self, role_name: str) -> Optional[Dict[str, Any]]:
        """Get role by name."""
        if not self.postgres.pool:
            return None
        
        async with self.postgres.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, name, description, permissions, is_system FROM roles WHERE name = $1",
                role_name
            )
            if not row:
                return None
            return {
                "id": str(row["id"]),
                "name": row["name"],
                "description": row["description"],
                "permissions": row["permissions"] or [],
                "is_system": row["is_system"],
            }
    
    async def create_role(
        self,
        name: str,
        description: Optional[str] = None,
        permissions: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create a new role."""
        if not self.postgres.pool:
            return None
        
        import json
        async with self.postgres.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO roles (name, description, permissions)
                VALUES ($1, $2, $3::jsonb)
                ON CONFLICT (name) DO NOTHING
                RETURNING id, name, description, permissions
                """,
                name, description, json.dumps(permissions or [])
            )
            if not row:
                return None
            return {
                "id": str(row["id"]),
                "name": row["name"],
                "description": row["description"],
                "permissions": row["permissions"] or [],
            }
    
    async def update_role_permissions(
        self,
        role_name: str,
        permissions: List[str],
    ) -> bool:
        """Update role permissions."""
        if not self.postgres.pool:
            return False
        
        import json
        async with self.postgres.pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE roles SET permissions = $1::jsonb WHERE name = $2 AND is_system = FALSE",
                json.dumps(permissions), role_name
            )
            return "UPDATE 1" in result
