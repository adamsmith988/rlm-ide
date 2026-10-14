"""Qdrant-backed chat memory.

* Short-term (scope ``session``): every turn is chunked and embedded, so facts
  anywhere inside a long message stay retrievable later in the same chat.
* Long-term (scope ``user``): durable facts distilled per turn in the
  background and shared across all of the user's chats.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..config import settings
from ..orchestrators._shared.text_chunks import chunk_text
from .embedder import EmbeddingUnavailable, embed_texts
from .facts import FactOps, describe_ops, distill_facts
from .store import QdrantMemoryStore, build_filter, match

log = logging.getLogger(__name__)

SHORT_TERM = "short_term"
LONG_TERM = "long_term"
PAIRED_ROLE = "turn"  # one chunk holding a short turn's question and answer


class MemoryScope:
    """انواع scope حافظه."""
    SESSION = "session"      # حافظه کوتاه‌مدت - فقط این session
    USER = "user"            # حافظه بلندمدت - همه session های کاربر
    COMBINED = "combined"    # ترکیب هر دو


class MemoryCategory:
    """دسته‌بندی حافظه‌ها."""
    PREFERENCE = "preference"
    PERSONAL_INFO = "personal"
    CONTEXT = "context"
    FEEDBACK = "feedback"
    DECISION = "decision"
    TASK = "task"
    FACT = "fact"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def qdrant_base_url() -> str:
    url = (settings.QDRANT_URL or "").strip()
    return url or f"http://{settings.QDRANT_HOST}:{int(settings.QDRANT_PORT)}"


class AdvancedMemoryClient:
    def __init__(self, store: Optional[QdrantMemoryStore] = None):
        self.store = store or QdrantMemoryStore(
            qdrant_base_url(),
            prefix=settings.MEMORY_COLLECTION,
            api_key=settings.QDRANT_API_KEY,
            timeout=float(settings.MEMORY_TIMEOUT_S or 20.0),
        )
        self._user_locks: Dict[str, asyncio.Lock] = {}
        self._background: set[asyncio.Task] = set()
        self.last_error: str = ""

    @property
    def enabled(self) -> bool:
        return bool(settings.MEMORY_ENABLED)

    # ---------------- internals ----------------

    async def _embed(self, texts: List[str]) -> tuple[str, List[List[float]]]:
        model, vectors = await embed_texts(texts)
        name = self.store.collection_for(model, len(vectors[0])) if vectors else ""
        if name:
            await self.store.ensure(name, len(vectors[0]))
        return name, vectors

    def _fail(self, what: str, exc: BaseException) -> None:
        self.last_error = f"{what}: {exc}"
        if isinstance(exc, EmbeddingUnavailable):
            log.warning("memory %s skipped: %s", what, exc)
        else:
            log.error("memory %s failed: %s", what, exc)

    @staticmethod
    def _record(point: Dict[str, Any], source: str) -> Dict[str, Any]:
        p = point.get("payload") or {}
        return {
            "id": str(point.get("id", "")),
            "content": p.get("content", ""),
            "score": float(point.get("score") or 0.0),
            "source": source,
            "created_at": p.get("created_at"),
            "updated_at": p.get("updated_at"),
            "metadata": {
                k: p.get(k)
                for k in ("session_id", "memory_type", "kind", "category", "role", "turn_id",
                          "chunk_index", "chunk_count")
                if p.get(k) is not None
            } | dict(p.get("metadata") or {}),
            "categories": [p["category"]] if p.get("category") else [],
        }

    def _point(self, vector: List[float], *, content: str, user_id: str, session_id: str,
               memory_type: str, kind: str, category: str, extra: Optional[Dict[str, Any]] = None,
               metadata: Optional[Dict[str, Any]] = None, point_id: Optional[str] = None) -> Dict[str, Any]:
        now = _now()
        payload = {
            "content": content, "user_id": str(user_id), "session_id": str(session_id or ""),
            "memory_type": memory_type, "kind": kind, "category": category,
            "created_at": now, "updated_at": now, "metadata": dict(metadata or {}), "history": [],
        }
        payload.update(extra or {})
        return {"id": point_id or str(uuid.uuid4()), "vector": vector, "payload": payload}

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def drain(self, timeout: float = 60.0) -> None:
        """Wait for background fact distillation (tests / shutdown)."""
        if self._background:
            await asyncio.wait(list(self._background), timeout=timeout)

    # ---------------- recall ----------------

    async def _search(self, query: str, flt: Dict[str, Any], limit: int, source: str,
                      embedded: Optional[tuple[str, List[float]]] = None) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        if embedded is None:
            name, vecs = await self._embed([query[: int(settings.MEMORY_QUERY_MAX_CHARS or 4000)]])
            embedded = (name, vecs[0])
        name, vec = embedded
        hits = await self.store.search(
            name, vec, flt=flt, limit=limit, score_threshold=settings.MEMORY_MIN_SCORE,
        )
        return [self._record(h, source) for h in hits]

    @staticmethod
    def _fact_source(user_message: Optional[str]) -> bool:
        """Whether a turn may yield long-term facts; oversized messages are pasted material."""
        text = (user_message or "").strip()
        limit = int(settings.MEMORY_FACT_MAX_USER_CHARS or 0)
        return bool(settings.MEMORY_FACT_EXTRACTION and text and (limit <= 0 or len(text) <= limit))

    @staticmethod
    def _paired_turn(user_message: str, assistant_text: str) -> str:
        limit = int(settings.MEMORY_PAIR_TURN_MAX_CHARS or 0)
        template = str(settings.MEMORY_PAIR_TURN_TEMPLATE or "")
        user, assistant = user_message.strip(), assistant_text.strip()
        if limit <= 0 or not template or not user or not assistant:
            return ""
        text = template.replace("{user}", user).replace("{assistant}", assistant)
        return text if len(text) <= limit else ""

    @staticmethod
    def _session_filter(user_id: str, session_id: str) -> Dict[str, Any]:
        return build_filter(must=[match("user_id", str(user_id)), match("session_id", str(session_id)),
                                  match("memory_type", SHORT_TERM)])

    @staticmethod
    def _user_filter(user_id: str) -> Dict[str, Any]:
        return build_filter(must=[match("user_id", str(user_id)), match("memory_type", LONG_TERM)])

    async def get_session_memories(self, query: str, user_id: str, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.enabled or not query.strip():
            return []
        try:
            return await self._search(query, self._session_filter(user_id, session_id), limit, "session")
        except Exception as exc:  # noqa: BLE001
            self._fail("session search", exc)
            return []

    async def get_user_memories(self, query: str, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not self.enabled or not query.strip():
            return []
        try:
            return await self._search(query, self._user_filter(user_id), limit, "user")
        except Exception as exc:  # noqa: BLE001
            self._fail("user search", exc)
            return []

    async def get_relevant_memories(
        self, query: str, user_id: str, session_id: str,
        scope: str = MemoryScope.COMBINED, limit: int = 10,
    ) -> Dict[str, List[Dict[str, Any]]]:
        result: Dict[str, List[Dict[str, Any]]] = {"session": [], "user": [], "combined": []}
        if not self.enabled or not str(query or "").strip():
            return result
        try:
            name, vecs = await self._embed([query[: int(settings.MEMORY_QUERY_MAX_CHARS or 4000)]])
            embedded = (name, vecs[0])
            user_limit = min(limit, int(settings.MEMORY_USER_LIMIT or limit))
            jobs = []
            if scope in (MemoryScope.SESSION, MemoryScope.COMBINED) and session_id:
                jobs.append(("session", self._search(query, self._session_filter(user_id, session_id), limit, "session", embedded)))
            if scope in (MemoryScope.USER, MemoryScope.COMBINED):
                jobs.append(("user", self._search(query, self._user_filter(user_id), user_limit, "user", embedded)))
            for (key, _), hits in zip(jobs, await asyncio.gather(*(j for _, j in jobs))):
                result[key] = hits
            combined = result["user"] + result["session"]
            combined.sort(key=lambda m: m.get("score", 0.0), reverse=True)
            result["combined"] = combined[:limit]
            self.last_error = ""
        except Exception as exc:  # noqa: BLE001
            self._fail("recall", exc)
            result["error"] = str(exc)  # type: ignore[assignment]
        return result

    # ---------------- write ----------------

    async def save_conversation(
        self, user_id: str, session_id: str, user_message: str, assistant_response: str,
        category: str = MemoryCategory.CONTEXT, is_long_term: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self.enabled:
            return False
        turn_id = str(uuid.uuid4())
        memory_type = LONG_TERM if is_long_term else SHORT_TERM
        size = int(settings.MEMORY_CHUNK_CHARS or 1200)
        overlap = int(settings.MEMORY_CHUNK_OVERLAP or 0)
        cap = int(settings.MEMORY_MAX_CHUNKS_PER_TURN or 0)
        assistant_text = (assistant_response or "")[: int(settings.MEMORY_ASSISTANT_MAX_CHARS or 4000)]
        pieces: List[tuple[str, int, int, str]] = []
        paired = self._paired_turn(user_message or "", assistant_text)
        if paired:
            pieces.append((PAIRED_ROLE, 0, 1, paired))
        else:
            for role, text in (("user", user_message or ""), ("assistant", assistant_text)):
                chunks = chunk_text(text, max_chars=size, overlap=overlap, max_chunks=cap)
                pieces.extend((role, c.index, len(chunks), c.text) for c in chunks)
        if not pieces:
            return False
        try:
            name, vectors = await self._embed([p[3] for p in pieces])
            points = [
                self._point(
                    vec, content=text, user_id=user_id, session_id=session_id,
                    memory_type=memory_type, kind="turn", category=category, metadata=metadata,
                    extra={"role": role, "turn_id": turn_id, "chunk_index": idx, "chunk_count": total},
                )
                for (role, idx, total, text), vec in zip(pieces, vectors)
            ]
            await self.store.upsert(name, points)
        except Exception as exc:  # noqa: BLE001
            self._fail("save", exc)
            return False
        if self._fact_source(user_message):
            self._spawn(self._learn_facts(user_id, session_id, user_message, assistant_response or ""))
        return True

    def save_conversation_background(self, **kwargs: Any) -> None:
        """Fire-and-forget ``save_conversation`` (survives client disconnects)."""
        if self.enabled:
            self._spawn(self.save_conversation(**kwargs))

    async def _learn_facts(self, user_id: str, session_id: str, user_message: str, assistant_response: str) -> None:
        lock = self._user_locks.setdefault(str(user_id), asyncio.Lock())
        async with lock:
            try:
                probe = user_message[: int(settings.MEMORY_QUERY_MAX_CHARS or 4000)]
                known = await self.get_user_memories(probe, user_id, limit=int(settings.MEMORY_FACT_KNOWN_LIMIT or 8))
                ops = await distill_facts(user_message, assistant_response, known)
                if not ops.empty:
                    await self._apply_fact_ops(user_id, session_id, ops)
                    log.info("memory: facts for user=%s %s", user_id, describe_ops(ops))
            except Exception as exc:  # noqa: BLE001
                self._fail("fact distillation", exc)

    async def _apply_fact_ops(self, user_id: str, session_id: str, ops: FactOps) -> None:
        for mid in ops.delete:
            await self.delete_memory(mid, user_id=user_id)
        for item in ops.update:
            await self.update_memory(item["id"], item["fact"], user_id=user_id)
        for fact in ops.add:
            await AdvancedMemoryClient.add_memory(self, fact, user_id, session_id, category=MemoryCategory.FACT,
                                                  is_long_term=True, kind="fact")

    async def add_memory(
        self, content: str, user_id: str, session_id: str,
        category: str = MemoryCategory.CONTEXT, is_long_term: bool = False,
        metadata: Optional[Dict[str, Any]] = None, kind: str = "note",
    ) -> bool:
        content = str(content or "").strip()
        if not self.enabled or not content:
            return False
        try:
            name, vectors = await self._embed([content])
            if is_long_term:
                dup = await self.store.search(
                    name, vectors[0], flt=self._user_filter(user_id), limit=1,
                    score_threshold=settings.MEMORY_FACT_DEDUP_SCORE,
                )
                if dup:
                    return await self._rewrite(name, dup[0], content, vectors[0])
            point = self._point(
                vectors[0], content=content, user_id=user_id, session_id=session_id,
                memory_type=LONG_TERM if is_long_term else SHORT_TERM, kind=kind,
                category=category, metadata=metadata,
            )
            await self.store.upsert(name, [point])
            return True
        except Exception as exc:  # noqa: BLE001
            self._fail("add", exc)
            return False

    async def save_preference(self, user_id: str, session_id: str, preference: str,
                              metadata: Optional[Dict[str, Any]] = None) -> bool:
        return await self.add_memory(preference, user_id, session_id, category=MemoryCategory.PREFERENCE,
                                     is_long_term=True, metadata=metadata)

    async def _rewrite(self, name: str, point: Dict[str, Any], content: str, vector: List[float]) -> bool:
        payload = dict(point.get("payload") or {})
        if payload.get("content") == content:
            return True
        history = list(payload.get("history") or [])
        history.append({"content": payload.get("content"), "updated_at": payload.get("updated_at")})
        keep = max(0, int(settings.MEMORY_HISTORY_KEEP or 0))
        payload.update(content=content, updated_at=_now(), history=history[-keep:] if keep else [])
        await self.store.upsert(name, [{"id": point["id"], "vector": vector, "payload": payload}])
        return True

    async def _locate(self, memory_id: str, user_id: Optional[str]) -> Optional[tuple[str, Dict[str, Any]]]:
        must = [match("user_id", str(user_id))] if user_id else []
        for name in await self.store.collections():
            pts = await self.store.scroll(name, flt=build_filter(must=must, ids=[memory_id]), limit=1)
            if pts:
                return name, pts[0]
        return None

    async def update_memory(self, memory_id: str, new_content: str, user_id: Optional[str] = None) -> bool:
        if not self.enabled or not str(new_content or "").strip():
            return False
        try:
            found = await self._locate(memory_id, user_id)
            if not found:
                return False
            name, vectors = await self._embed([new_content.strip()])
            if name != found[0]:
                return False
            return await self._rewrite(name, found[1], new_content.strip(), vectors[0])
        except Exception as exc:  # noqa: BLE001
            self._fail("update", exc)
            return False

    # ---------------- management ----------------

    async def _scroll_all(self, flt: Dict[str, Any], limit: int, source: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for name in await self.store.collections():
            out.extend(self._record(p, source) for p in await self.store.scroll(name, flt=flt, limit=limit - len(out)))
            if len(out) >= limit:
                break
        out.sort(key=lambda m: (m.get("created_at") or "", m["metadata"].get("chunk_index", 0)))
        return out

    async def get_all_session_memories(self, user_id: str, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            return await self._scroll_all(self._session_filter(user_id, session_id), limit, "session")
        except Exception as exc:  # noqa: BLE001
            self._fail("list session", exc)
            return []

    async def get_all_user_memories(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            return await self._scroll_all(self._user_filter(user_id), limit, "user")
        except Exception as exc:  # noqa: BLE001
            self._fail("list user", exc)
            return []

    async def _delete(self, flt: Dict[str, Any], what: str) -> bool:
        if not self.enabled:
            return False
        try:
            for name in await self.store.collections():
                await self.store.delete(name, flt=flt)
            return True
        except Exception as exc:  # noqa: BLE001
            self._fail(what, exc)
            return False

    async def delete_memory(self, memory_id: str, user_id: Optional[str] = None) -> bool:
        must = [match("user_id", str(user_id))] if user_id else []
        return await self._delete(build_filter(must=must, ids=[memory_id]), "delete")

    async def clear_session_memories(self, user_id: str, session_id: str) -> bool:
        """Drop the chat's short-term memory; long-term user facts are kept."""
        return await self._delete(self._session_filter(user_id, session_id), "clear session")

    async def clear_user_memories(self, user_id: str) -> bool:
        return await self._delete(build_filter(must=[match("user_id", str(user_id))]), "clear user")

    async def get_memory_history(self, memory_id: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            found = await self._locate(memory_id, user_id)
            return list(((found[1].get("payload") or {}).get("history")) or []) if found else []
        except Exception as exc:  # noqa: BLE001
            self._fail("history", exc)
            return []

    async def status(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"enabled": self.enabled, "backend": "qdrant", "url": qdrant_base_url()}
        if not self.enabled:
            return info
        try:
            info["collections"] = await self.store.collections()
            info["qdrant"] = "ok"
        except Exception as exc:  # noqa: BLE001
            info["qdrant"] = f"error: {exc}"
        try:
            from .embedder import bound_embed_model

            info["embed_model"] = await bound_embed_model()
        except Exception as exc:  # noqa: BLE001
            info["embed_model"] = None
            info["embed_error"] = str(exc)
        info["ready"] = info.get("qdrant") == "ok" and bool(info.get("embed_model"))
        if self.last_error:
            info["last_error"] = self.last_error
        return info


class MemoryClient(AdvancedMemoryClient):
    """Backward-compatible API (flat lists)."""

    async def get_relevant_memories(  # type: ignore[override]
        self, query: str, user_id: str, session_id: str, limit: int = 5,
    ) -> List[Dict[str, Any]]:
        result = await super().get_relevant_memories(query, user_id, session_id, MemoryScope.COMBINED, limit)
        return result.get("combined", [])

    async def save_message(self, user_id: str, session_id: str, user_message: str,
                           assistant_response: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        return await self.save_conversation(user_id, session_id, user_message, assistant_response, metadata=metadata)

    async def add_memory(self, content: str, user_id: str, metadata: Optional[Dict[str, Any]] = None, **kw: Any) -> bool:  # type: ignore[override]
        session_id = (metadata or {}).get("session_id", "default")
        return await super().add_memory(content, user_id, session_id, metadata=metadata, **kw)

    async def get_all_memories(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        return await self.get_all_user_memories(user_id=user_id, limit=limit)
