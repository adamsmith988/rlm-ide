"""Native corpus RAG client: embed → Storage ANN (global) → rerank → top-k."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from ..config import llm_gateway_v1_url, settings
from ..memory.embedder import post_embeddings
from .prompt import extract_hit_text, format_rag_prompt_block

log = logging.getLogger(__name__)

_ANN_WEIGHT = 0.4
_RERANK_WEIGHT = 0.6


def _llm_gateway_auth_headers() -> Dict[str, str]:
    """Auth for in-cluster llm-gateway (Bearer = INTERNAL_API_KEY)."""
    key = (settings.INTERNAL_API_KEY or "").strip()
    if not key:
        return {"Authorization": "Bearer not-needed"}
    return {
        "Authorization": f"Bearer {key}",
        "X-Internal-Key": key,
    }


@dataclass
class RagResult:
    """Outcome of a native RAG preflight retrieve."""

    hits: List[Dict[str, Any]] = field(default_factory=list)
    prompt_block: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.hits)


def combine_scores(ann_score: float, rerank_score: float) -> float:
    """Weighted blend matching text_search: 0.4 ANN + 0.6 rerank."""
    return _ANN_WEIGHT * float(ann_score) + _RERANK_WEIGHT * float(rerank_score)


def rank_and_trim(
    candidates: List[Dict[str, Any]],
    *,
    top_k: int,
    reranked: bool,
) -> List[Dict[str, Any]]:
    """Sort by combined_score (or ann score) and keep top_k."""
    key = "combined_score" if reranked else "score"
    ranked = sorted(
        candidates,
        key=lambda r: float(r.get(key) or r.get("score") or 0.0),
        reverse=True,
    )
    return ranked[: max(0, int(top_k))]


def _storage_headers(auth_context: Optional[Dict[str, Any]], user_id: str) -> Dict[str, str]:
    headers = {
        "X-Internal-Key": settings.INTERNAL_API_KEY,
        "X-Service-Name": settings.SERVICE_NAME,
        "Content-Type": "application/json",
    }
    ctx = dict(auth_context or {})
    uid = str(ctx.get("user_id") or user_id or "").strip()
    if uid:
        headers["X-User-Id"] = uid
    if ctx.get("group_id"):
        headers["X-Group-Id"] = str(ctx["group_id"])
    # Intentionally omit X-Case-Id: native RAG is global corpus search.
    return headers


def _normalize_candidate(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(row, dict):
        return None
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
    text = extract_hit_text(payload, row)
    if not text:
        return None
    artifact_id = str(
        row.get("artifact_id") or payload.get("artifact_id") or ""
    ).strip() or None
    filename = str(
        row.get("filename")
        or payload.get("filename")
        or payload.get("original_filename")
        or ""
    ).strip()
    if not filename:
        path = str(
            payload.get("storage_path")
            or payload.get("filepath")
            or payload.get("path")
            or ""
        ).strip()
        if path:
            filename = Path(path).name
    try:
        ann_score = float(row.get("score") or 0.0)
    except (TypeError, ValueError):
        ann_score = 0.0
    return {
        "text": text,
        "text_preview": text[:500],
        "artifact_id": artifact_id,
        "filename": filename or "unknown",
        "score": ann_score,
        "ann_score": ann_score,
        "chunk_index": payload.get("chunk_index"),
        "payload": payload,
    }


def _parse_rerank_scores(data: Any, n_docs: int) -> List[float]:
    scores = [0.0] * n_docs
    rows = data if isinstance(data, list) else (
        (data or {}).get("results") or (data or {}).get("data") or []
    )
    if not isinstance(rows, list):
        return scores
    indexed = any(isinstance(r, dict) and "index" in r for r in rows)
    if indexed:
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                idx = int(row.get("index", -1))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < n_docs:
                try:
                    scores[idx] = float(
                        row.get("score") or row.get("relevance_score") or 0.0
                    )
                except (TypeError, ValueError):
                    scores[idx] = 0.0
        return scores
    if len(rows) == n_docs:
        for i, row in enumerate(rows):
            if isinstance(row, (int, float)):
                scores[i] = float(row)
            elif isinstance(row, dict):
                try:
                    scores[i] = float(
                        row.get("score") or row.get("relevance_score") or 0.0
                    )
                except (TypeError, ValueError):
                    pass
    return scores


async def retrieve_for_prompt(
    query: str,
    *,
    user_id: str,
    auth_context: Optional[Dict[str, Any]] = None,
) -> RagResult:
    """Run global corpus RAG and return hits + prompt block (soft-fail)."""
    t0 = time.monotonic()
    meta: Dict[str, Any] = {
        "enabled": True,
        "reason": None,
        "reranked": False,
        "hit_count": 0,
        "candidate_count": 0,
        "timings_ms": {},
    }
    q = (query or "").strip()
    if not q:
        meta["reason"] = "empty_query"
        return RagResult(meta=meta)
    q_max = int(settings.NATIVE_RAG_QUERY_MAX_CHARS or 0)
    if q_max > 0 and len(q) > q_max:
        # Embedding/rerank models have a token window; keep the head and tail
        # (where the question usually is) of an oversized message.
        q = f"{q[: q_max // 2]}\n…\n{q[-(q_max - q_max // 2):]}"
        meta["query_clipped"] = True

    if not settings.NATIVE_RAG_ENABLED:
        meta["enabled"] = False
        meta["reason"] = "disabled"
        return RagResult(meta=meta)

    timeout = float(settings.NATIVE_RAG_TIMEOUT_S or 20.0)
    top_k = max(1, int(settings.NATIVE_RAG_TOP_K or 5))
    candidate_limit = max(top_k, int(settings.NATIVE_RAG_CANDIDATE_LIMIT or 25))
    collection = (settings.NATIVE_RAG_COLLECTION or "tool_embed_text").strip()
    score_threshold = settings.NATIVE_RAG_SCORE_THRESHOLD
    max_chars = int(settings.NATIVE_RAG_MAX_CHARS_PER_HIT or 1200)

    try:
        from ..llm.binding_client import get_binding_client

        binding = get_binding_client("orchestrator")
        embed_model = (await binding.get_model("rag_embed")).strip()
        rerank_model = (await binding.get_model("rag_rerank", optional=True)).strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("native_rag: binding resolve failed: %s", exc)
        meta["reason"] = f"binding_resolve_failed:{exc}"
        meta["timings_ms"]["total"] = int((time.monotonic() - t0) * 1000)
        return RagResult(meta=meta)

    meta["embed_bound"] = bool(embed_model)
    meta["rerank_bound"] = bool(rerank_model)

    if not embed_model:
        meta["reason"] = "binding_missing"
        meta["timings_ms"]["total"] = int((time.monotonic() - t0) * 1000)
        return RagResult(meta=meta)

    # Without rerank binding: ANN-only path (still inject top_k into the prompt).
    use_rerank = bool(rerank_model)
    fetch_limit = candidate_limit if use_rerank else top_k

    gateway_v1 = llm_gateway_v1_url()
    gateway_root = (settings.LLM_PROXY_URL or "").rstrip("/")
    storage_url = (settings.STORAGE_URL or "").rstrip("/")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # ---- 1) Embed query ----
            t_embed = time.monotonic()
            emb_resp = await post_embeddings(
                client, f"{gateway_v1}/embeddings", {"model": embed_model, "input": q}, _llm_gateway_auth_headers(),
            )
            emb_data = emb_resp.json() or {}
            rows = emb_data.get("data") or []
            if not rows or not isinstance(rows[0], dict):
                meta["reason"] = "embed_empty"
                meta["timings_ms"]["embed"] = int((time.monotonic() - t_embed) * 1000)
                meta["timings_ms"]["total"] = int((time.monotonic() - t0) * 1000)
                return RagResult(meta=meta)
            vector = rows[0].get("embedding")
            if not isinstance(vector, list) or not vector:
                meta["reason"] = "embed_empty"
                meta["timings_ms"]["embed"] = int((time.monotonic() - t_embed) * 1000)
                meta["timings_ms"]["total"] = int((time.monotonic() - t0) * 1000)
                return RagResult(meta=meta)
            meta["timings_ms"]["embed"] = int((time.monotonic() - t_embed) * 1000)

            # ---- 2) Global ANN via Storage ----
            t_search = time.monotonic()
            body: Dict[str, Any] = {
                "collection_name": collection,
                "query_vector": vector,
                "limit": fetch_limit,
            }
            if score_threshold is not None:
                body["score_threshold"] = float(score_threshold)
            search_resp = await client.post(
                f"{storage_url}/vectors/search",
                json=body,
                headers=_storage_headers(auth_context, user_id),
            )
            search_resp.raise_for_status()
            raw_hits = search_resp.json() or []
            if not isinstance(raw_hits, list):
                raw_hits = []
            candidates: List[Dict[str, Any]] = []
            for row in raw_hits:
                norm = _normalize_candidate(row)
                if norm:
                    candidates.append(norm)
            meta["candidate_count"] = len(candidates)
            meta["timings_ms"]["search"] = int((time.monotonic() - t_search) * 1000)

            if not candidates:
                meta["reason"] = "no_results"
                meta["timings_ms"]["total"] = int((time.monotonic() - t0) * 1000)
                return RagResult(meta=meta)

            # ---- 3) Optional rerank ----
            reranked = False
            if use_rerank:
                try:
                    t_rerank = time.monotonic()
                    docs = [c["text"] for c in candidates]
                    rerank_resp = await client.post(
                        f"{gateway_root}/rerank",
                        json={"model": rerank_model, "query": q, "texts": docs},
                        headers=_llm_gateway_auth_headers(),
                    )
                    rerank_resp.raise_for_status()
                    rerank_scores = _parse_rerank_scores(rerank_resp.json(), len(docs))
                    for i, cand in enumerate(candidates):
                        rs = rerank_scores[i] if i < len(rerank_scores) else 0.0
                        cand["rerank_score"] = float(rs)
                        cand["combined_score"] = combine_scores(cand["ann_score"], rs)
                    reranked = True
                    meta["timings_ms"]["rerank"] = int((time.monotonic() - t_rerank) * 1000)
                except Exception as rerank_exc:  # noqa: BLE001
                    log.warning(
                        "native_rag: rerank failed, falling back to ANN: %s", rerank_exc
                    )
                    meta["rerank_error"] = f"{type(rerank_exc).__name__}:{rerank_exc}"
                    for cand in candidates:
                        cand["combined_score"] = float(cand.get("ann_score") or 0.0)
            else:
                for cand in candidates:
                    cand["combined_score"] = float(cand.get("ann_score") or 0.0)

            meta["reranked"] = reranked
            hits = rank_and_trim(candidates, top_k=top_k, reranked=reranked)
            for h in hits:
                h.pop("payload", None)

            prompt_block = format_rag_prompt_block(
                hits, max_chars_per_hit=max_chars
            )
            meta["hit_count"] = len(hits)
            if hits:
                meta["reason"] = "ok" if reranked else "ok_ann_only"
            else:
                meta["reason"] = "no_results_after_trim"
            meta["timings_ms"]["total"] = int((time.monotonic() - t0) * 1000)
            return RagResult(hits=hits, prompt_block=prompt_block, meta=meta)

    except Exception as exc:  # noqa: BLE001 — soft-fail
        log.warning("native_rag: retrieve failed: %s", exc)
        meta["reason"] = f"error:{type(exc).__name__}:{exc}"
        meta["timings_ms"]["total"] = int((time.monotonic() - t0) * 1000)
        return RagResult(meta=meta)
