"""Host: resolve model context window from the LLM catalog (Agent-Flow parity).

Agent Flow stamps ``context_length`` / ``max_output_tokens`` from the catalog
entry onto each Agent node before RLM starts. Hub orchestrators must do the
same so compact triggers at ~90% of *that model's* window — never a global
hardcoded token cap.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

import httpx

from src.config import settings
from src.orchestrators.rlm.runtime.context_budget import (
    coerce_context_length,
    coerce_positive_int,
)

logger = logging.getLogger(__name__)

_CACHE_TTL_SEC = 60.0
_catalog_cache: Dict[str, Any] = {"at": 0.0, "by_id": {}}


def _internal_catalog_headers() -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "X-Internal-Key": str(getattr(settings, "INTERNAL_API_KEY", "") or ""),
        "X-Service-Name": "orchestrator",
    }
    secret = str(getattr(settings, "TOOL_RUNNER_INTERNAL_SECRET", "") or "").strip()
    if not secret:
        import os

        secret = str(os.environ.get("TOOL_RUNNER_INTERNAL_SECRET") or "").strip()
    if secret:
        headers["X-Tool-Runner-Internal-Secret"] = secret
    return headers


async def _download_catalog_rows() -> list:
    """Prefer tool-runner ``/internal/llm/catalog`` (same as AF); fall back to llm-proxy."""
    rows: list = []
    tool_base = str(getattr(settings, "TOOL_RUNNER_URL", "") or "").rstrip("/")
    if tool_base:
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                resp = await client.get(
                    f"{tool_base}/internal/llm/catalog",
                    headers=_internal_catalog_headers(),
                )
            if resp.status_code == 200:
                body = resp.json() or {}
                raw = body.get("data") if isinstance(body, dict) else body
                if isinstance(raw, list) and raw:
                    return raw
        except Exception:
            logger.debug("tool-runner catalog fetch failed", exc_info=True)

    proxy = str(getattr(settings, "LLM_PROXY_URL", "") or "").rstrip("/")
    if proxy:
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                resp = await client.get(
                    f"{proxy}/catalog",
                    headers=_internal_catalog_headers(),
                )
            if resp.status_code == 200:
                body = resp.json() or {}
                raw = body.get("data") if isinstance(body, dict) else body
                if isinstance(raw, list):
                    return raw
        except Exception:
            logger.debug("llm-proxy catalog fetch failed", exc_info=True)
    return rows


async def _catalog_by_id(*, force: bool = False) -> Dict[str, Dict[str, Any]]:
    now = time.time()
    cached = _catalog_cache.get("by_id")
    if (
        not force
        and isinstance(cached, dict)
        and cached
        and (now - float(_catalog_cache.get("at") or 0.0)) < _CACHE_TTL_SEC
    ):
        return cached  # type: ignore[return-value]

    rows = await _download_catalog_rows()
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        lid = str(row.get("logical_id") or row.get("id") or "").strip()
        if lid:
            by_id[lid] = dict(row)
            by_id.setdefault(lid.lower(), dict(row))
    if by_id:
        _catalog_cache["by_id"] = by_id
        _catalog_cache["at"] = now
    elif isinstance(cached, dict) and cached:
        return cached  # type: ignore[return-value]
    return by_id


async def fetch_catalog_model(logical_id: str) -> Optional[Dict[str, Any]]:
    """Lookup one logical model (Agent-Flow ``fetch_catalog_model`` parity)."""
    lid = str(logical_id or "").strip()
    if not lid or lid.lower() in ("auto", "default"):
        return None
    by_id = await _catalog_by_id()
    hit = by_id.get(lid) or by_id.get(lid.lower())
    return dict(hit) if isinstance(hit, dict) else None


def input_modalities_from_catalog(entry: Optional[Mapping[str, Any]]) -> Optional[FrozenSet[str]]:
    """Input kinds the model accepts (``input_modalities``, else ``capabilities.supports_vision``).

    ``None`` when the catalog row says nothing, so callers keep their own default.
    """
    if not isinstance(entry, Mapping):
        return None
    raw = entry.get("input_modalities")
    if isinstance(raw, (list, tuple, set, frozenset)) and raw:
        return frozenset(str(m).strip().lower() for m in raw if str(m).strip())
    caps = entry.get("capabilities") if isinstance(entry.get("capabilities"), Mapping) else {}
    vision = caps.get("supports_vision", entry.get("supports_vision"))
    if isinstance(vision, bool):
        return frozenset({"text", "image"} if vision else {"text"})
    return None


async def model_input_modalities(logical_id: str) -> Optional[FrozenSet[str]]:
    try:
        return input_modalities_from_catalog(await fetch_catalog_model(logical_id))
    except Exception:
        logger.debug("model modality lookup failed model=%s", logical_id, exc_info=True)
        return None


def model_window_from_catalog(entry: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Slim meta: ``context_length`` / ``max_output_tokens`` from a catalog row."""
    if not isinstance(entry, Mapping):
        return {}
    meta: Dict[str, Any] = {
        "logical_id": str(entry.get("logical_id") or entry.get("id") or "").strip(),
        "context_length": entry.get("context_length") or entry.get("max_model_len"),
        "max_model_len": entry.get("max_model_len") or entry.get("context_length"),
        "max_output_tokens": entry.get("max_output_tokens"),
    }
    rs = entry.get("run_schema") if isinstance(entry.get("run_schema"), dict) else {}
    if rs:
        meta["run_schema"] = rs
        if not meta.get("context_length"):
            meta["context_length"] = (
                rs.get("context_length") or rs.get("max_model_len") or rs.get("max_seq_len")
            )
        if not meta.get("max_output_tokens"):
            meta["max_output_tokens"] = rs.get("max_output_tokens")
    caps = entry.get("capabilities") if isinstance(entry.get("capabilities"), dict) else {}
    if caps and not meta.get("context_length"):
        meta["context_length"] = caps.get("context_length") or caps.get("max_model_len")
    return meta


def stamp_context_window(
    cfg: Optional[Mapping[str, Any]],
    meta: Optional[Mapping[str, Any]] = None,
    *,
    min_context: Optional[int] = None,
) -> Dict[str, Any]:
    """Copy catalog/deploy window onto orchestrator cfg (AF ``_stamp_context_window``).

    ``min_context`` is the smallest window the engine's harness works in: a
    configured window is not lowered past it (the model's own window still wins).
    """
    out = dict(cfg or {})
    # A configured window may lower the model's window, never raise it.
    configured, catalog = coerce_context_length(out), coerce_context_length(meta)
    floor = coerce_positive_int(out.get("min_context_length"), min_context)
    if configured and floor and configured < floor:
        logger.warning("context_length %s below engine minimum %s; using the minimum", configured, floor)
        configured = floor
    ctx = min(configured, catalog) if configured and catalog else (configured or catalog)
    if ctx:
        out["context_length"] = int(ctx)
    max_out = coerce_positive_int(
        (meta or {}).get("max_output_tokens") if isinstance(meta, Mapping) else None,
        out.get("max_output_tokens"),
    )
    if max_out:
        out["max_output_tokens"] = int(max_out)
    return out


async def resolve_model_context_window(
    model: str,
    *,
    cfg: Optional[Mapping[str, Any]] = None,
    min_context: Optional[int] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Fetch catalog entry for ``model`` and stamp window fields onto ``cfg``.

    Returns ``(stamped_cfg, catalog_meta_or_None)``.
    """
    base = dict(cfg or {})
    mid = str(model or base.get("model") or "").strip()
    entry = await fetch_catalog_model(mid) if mid else None
    meta = model_window_from_catalog(entry)
    stamped = stamp_context_window(base, meta, min_context=min_context)
    if stamped.get("context_length"):
        logger.info(
            "catalog context_length model=%s window=%s max_out=%s",
            mid,
            stamped.get("context_length"),
            stamped.get("max_output_tokens"),
        )
    else:
        logger.warning(
            "catalog context_length missing model=%s — compact/budget disabled until "
            "overflow recovery learns the window from the provider",
            mid or "(empty)",
        )
    return stamped, (meta or None)


async def prepare_orchestrator_config(
    orchestrator_config: Optional[Mapping[str, Any]],
    *,
    model_hint: str = "",
) -> Dict[str, Any]:
    """Stamp catalog context window onto hub ``orchestrator_config`` for engines."""
    cfg = dict(orchestrator_config or {})
    mid = str(cfg.get("model") or model_hint or "").strip()
    if not mid or mid.lower() in ("auto", "default"):
        try:
            from src.llm.gateway_llm_client import get_gateway_llm_client

            mid = str(await get_gateway_llm_client()._resolve_model() or "").strip()  # noqa: SLF001
        except Exception:
            mid = mid if mid.lower() not in ("auto", "default") else ""
    stamped, _ = await resolve_model_context_window(mid, cfg=cfg)
    if mid and not stamped.get("model"):
        stamped["model"] = mid
    return stamped


__all__ = [
    "fetch_catalog_model",
    "model_window_from_catalog",
    "prepare_orchestrator_config",
    "resolve_model_context_window",
    "stamp_context_window",
]
