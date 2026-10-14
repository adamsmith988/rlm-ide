"""
Base classes and utilities for storage module.
کلاس‌های پایه و ابزارها
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Union

try:
    import asyncpg
except ImportError:
    asyncpg = None


JSONDict = Dict[str, Any]


def to_uuid(value: Optional[Union[str, uuid.UUID]]) -> Optional[uuid.UUID]:
    """Convert string or UUID to UUID object. Returns None for invalid values."""
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError):
        return None


def json_dumps(data: Optional[JSONDict]) -> str:
    """Safe JSON serialization."""
    return json.dumps(data or {}, ensure_ascii=False, default=str)


def now_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.utcnow().isoformat()


class DbExecutor:
    """Thin wrapper to unify asyncpg operations."""

    def __init__(self, pool) -> None:
        self.pool = pool

    async def execute(self, query: str, *args) -> None:
        """Execute a query without returning results."""
        if self.pool:
            async with self.pool.acquire() as conn:
                await conn.execute(query, *args)
        else:
            raise RuntimeError("PostgreSQL connection required")

    async def executemany(self, query: str, args_iter) -> None:
        """Execute query with multiple argument sets."""
        if self.pool:
            async with self.pool.acquire() as conn:
                await conn.executemany(query, list(args_iter))
        else:
            raise RuntimeError("PostgreSQL connection required")

    async def fetchrow(self, query: str, *args):
        """Fetch a single row."""
        if self.pool:
            async with self.pool.acquire() as conn:
                return await conn.fetchrow(query, *args)
        else:
            raise RuntimeError("PostgreSQL connection required")

    async def fetchval(self, query: str, *args):
        """Fetch a single value."""
        if self.pool:
            async with self.pool.acquire() as conn:
                return await conn.fetchval(query, *args)
        else:
            raise RuntimeError("PostgreSQL connection required")

    async def fetch(self, query: str, *args):
        """Fetch multiple rows."""
        if self.pool:
            async with self.pool.acquire() as conn:
                return await conn.fetch(query, *args)
        else:
            raise RuntimeError("PostgreSQL connection required")


class BaseBackend:
    """Base class for storage backends."""

    def __init__(self) -> None:
        self._connected = False

    async def connect(self) -> None:
        """Initialize connection."""
        raise NotImplementedError

    async def disconnect(self) -> None:
        """Close connection."""
        raise NotImplementedError

    @property
    def is_connected(self) -> bool:
        """Check if backend is connected."""
        return self._connected
