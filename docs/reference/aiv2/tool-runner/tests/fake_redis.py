from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple


class FakeRedis:
    """In-memory Redis subset shared by generic queue tests."""

    def __init__(self) -> None:
        self.kv: Dict[str, str] = {}
        self.hashes: Dict[str, Dict[str, str]] = {}
        self.streams: Dict[str, List[Tuple[str, Dict[str, str]]]] = {}
        self.groups: Dict[Tuple[str, str], int] = {}
        self.pending: Dict[Tuple[str, str], Dict[str, Tuple[str, float]]] = {}
        self.ttl: Dict[str, float] = {}
        self.calls: List[Tuple[str, Any]] = []
        self.fail_next: Optional[Exception] = None
        self.closed = False
        self._stream_seq = 0

    def _call(self, op: str, *args: Any) -> None:
        self.calls.append((op, args))
        if self.fail_next is not None:
            exc, self.fail_next = self.fail_next, None
            raise exc

    def _sweep(self) -> None:
        now = time.monotonic()
        for key, expires in list(self.ttl.items()):
            if expires <= now:
                self.ttl.pop(key, None)
                self.kv.pop(key, None)
                self.hashes.pop(key, None)

    async def ping(self) -> bool:
        self._call("ping")
        return True

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        self._call("set", key)
        self.kv[key] = str(value)
        if ex is not None:
            self.ttl[key] = time.monotonic() + float(ex)
        return True

    async def get(self, key: str) -> Optional[str]:
        self._sweep()
        return self.kv.get(key)

    async def expire(self, key: str, seconds: int) -> bool:
        self.ttl[key] = time.monotonic() + float(seconds)
        return True

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            for store in (self.kv, self.hashes, self.streams):
                if key in store:
                    store.pop(key, None)
                    removed += 1
            self.ttl.pop(key, None)
        return removed

    async def hset(self, key: str, mapping: Dict[str, str]) -> int:
        self._call("hset", key)
        self.hashes.setdefault(key, {}).update(
            {str(name): str(value) for name, value in mapping.items()}
        )
        return len(mapping)

    async def hgetall(self, key: str) -> Dict[str, str]:
        self._sweep()
        return dict(self.hashes.get(key) or {})

    async def xgroup_create(
        self, key: str, groupname: str, id: str = "0", mkstream: bool = False
    ) -> bool:
        marker = (key, groupname)
        if marker in self.groups:
            raise RuntimeError("BUSYGROUP")
        self.streams.setdefault(key, [])
        self.groups[marker] = 0
        self.pending[marker] = {}
        return True

    async def xadd(
        self,
        key: str,
        fields: Dict[str, str],
        maxlen: Optional[int] = None,
        approximate: bool = True,
    ) -> str:
        self._call("xadd", key)
        self._stream_seq += 1
        message_id = f"{self._stream_seq}-0"
        self.streams.setdefault(key, []).append(
            (message_id, {str(name): str(value) for name, value in fields.items()})
        )
        if maxlen and len(self.streams[key]) > maxlen:
            self.streams[key] = self.streams[key][-maxlen:]
        return message_id

    async def xreadgroup(
        self,
        groupname: str,
        consumername: str,
        streams: Dict[str, str],
        count: Optional[int] = None,
        block: Optional[int] = None,
    ) -> Any:
        self._call("xreadgroup", tuple(streams))
        deadline = time.monotonic() + float(block or 0) / 1000
        while True:
            for key in streams:
                marker = (key, groupname)
                index = self.groups.get(marker, 0)
                entries = self.streams.get(key, [])
                if index < len(entries):
                    message_id, fields = entries[index]
                    self.groups[marker] = index + 1
                    self.pending.setdefault(marker, {})[message_id] = (
                        consumername,
                        time.monotonic(),
                    )
                    return [(key, [(message_id, dict(fields))])]
            if not block or time.monotonic() >= deadline:
                return []
            await asyncio.sleep(0.005)

    async def xack(self, key: str, group: str, *ids: str) -> int:
        pending = self.pending.setdefault((key, group), {})
        return sum(pending.pop(message_id, None) is not None for message_id in ids)

    async def xdel(self, key: str, *ids: str) -> int:
        entries = self.streams.get(key, [])
        wanted = set(ids)
        for marker, index in list(self.groups.items()):
            if marker[0] == key:
                removed_before = sum(
                    message_id in wanted for message_id, _ in entries[:index]
                )
                self.groups[marker] = max(0, index - removed_before)
        kept = [row for row in entries if row[0] not in wanted]
        self.streams[key] = kept
        return len(entries) - len(kept)

    async def xautoclaim(
        self,
        key: str,
        group: str,
        consumer: str,
        min_idle_time: int,
        start_id: str,
        count: int = 1,
    ) -> Any:
        marker = (key, group)
        now = time.monotonic()
        by_id = dict(self.streams.get(key, []))
        claimed = []
        for message_id, (_, delivered) in list(
            self.pending.get(marker, {}).items()
        ):
            if (now - delivered) * 1000 >= min_idle_time:
                self.pending[marker][message_id] = (consumer, now)
                claimed.append((message_id, dict(by_id[message_id])))
                if len(claimed) >= count:
                    break
        return ("0-0", claimed, [])

    async def xlen(self, key: str) -> int:
        return len(self.streams.get(key) or [])

    async def xinfo_groups(self, key: str) -> Any:
        return [
            {
                "name": group,
                "pending": len(self.pending.get((stream, group), {})),
                "lag": len(self.streams.get(stream, [])) - index,
            }
            for (stream, group), index in self.groups.items()
            if stream == key
        ]

    async def aclose(self) -> None:
        self.closed = True

    def ttl_of(self, key: str) -> Optional[float]:
        expires = self.ttl.get(key)
        return None if expires is None else expires - time.monotonic()
