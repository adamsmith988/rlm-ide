"""
Validate that a logical model's modalities match a tool/binding profile.

Replaces coarse purpose→task checks with input/output modality matching
(HuggingFace-style model cards).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from aiv2_model_contract.llm_taxonomy import (
    infer_api_from_modalities,
    merge_llm_profiles,
    modalities_compatible,
    normalize_llm_profile,
)

from .config import settings

log = logging.getLogger(__name__)

# Refresh interval for a successful catalog snapshot.
_CATALOG_TTL = 30.0
# Keep serving the last good snapshot when refresh fails (high concurrency / gateway load).
_CATALOG_STALE_MAX = 300.0
_CATALOG_FETCH_RETRIES = 3
_CATALOG_FETCH_RETRY_DELAY = 0.15

_catalog_cache: Dict[str, Any] = {"at": 0.0, "by_id": {}}
_catalog_lock = asyncio.Lock()


def _catalog_age(now: float) -> float:
    return now - float(_catalog_cache.get("at") or 0.0)


def _has_cached_catalog() -> bool:
    cached = _catalog_cache.get("by_id")
    return isinstance(cached, dict) and bool(cached)


async def _download_catalog() -> Dict[str, Dict[str, Any]]:
    """Fetch logical_id → catalog entry from llm-gateway (no cache writes)."""
    base = settings.LLM_PROXY_URL.rstrip("/")
    by_id: Dict[str, Dict[str, Any]] = {}
    last_exc: Optional[Exception] = None
    key = (settings.INTERNAL_API_KEY or "").strip()
    headers = {"X-Internal-Key": key} if key else {}
    for attempt in range(_CATALOG_FETCH_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{base}/catalog", headers=headers or None)
            if resp.status_code != 200:
                last_exc = RuntimeError(f"catalog_http_{resp.status_code}")
                continue
            data = resp.json() or {}
            for entry in data.get("data") or []:
                if isinstance(entry, dict):
                    lid = str(entry.get("logical_id") or "").strip()
                    if lid:
                        by_id[lid] = entry
                    name = str(entry.get("name") or "").strip()
                    if name and name not in by_id:
                        by_id[name] = entry
            if by_id:
                return by_id
            last_exc = RuntimeError("catalog_empty")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
        if attempt + 1 < _CATALOG_FETCH_RETRIES:
            await asyncio.sleep(_CATALOG_FETCH_RETRY_DELAY * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    return by_id


async def _fetch_catalog(*, force: bool = False) -> Dict[str, Dict[str, Any]]:
    """Return logical_id → catalog entry, using stale-while-revalidate under load."""
    now = time.time()
    if not force and _has_cached_catalog() and _catalog_age(now) < _CATALOG_TTL:
        cached = _catalog_cache.get("by_id")
        return cached if isinstance(cached, dict) else {}

    async with _catalog_lock:
        now = time.time()
        if not force and _has_cached_catalog() and _catalog_age(now) < _CATALOG_TTL:
            cached = _catalog_cache.get("by_id")
            return cached if isinstance(cached, dict) else {}

        try:
            by_id = await _download_catalog()
        except Exception as exc:  # noqa: BLE001
            stale = _catalog_cache.get("by_id")
            if isinstance(stale, dict) and stale and _catalog_age(now) < _CATALOG_STALE_MAX:
                log.warning(
                    "llm_model_validator: catalog fetch failed (%s); using stale snapshot "
                    "(%d models, age=%.1fs)",
                    exc,
                    len(stale),
                    _catalog_age(now),
                )
                return stale
            log.warning("llm_model_validator: catalog fetch failed: %s", exc)
            return {}

        _catalog_cache["at"] = now
        _catalog_cache["by_id"] = by_id
        return by_id


async def resolve_model_entry(
    model: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], bool]:
    """Return (catalog_entry, logical_id, catalog_loaded).

    ``catalog_loaded`` is False when the gateway catalog could not be loaded and
    there is no stale snapshot to search — distinct from a known-missing model.
    """
    catalog = await _fetch_catalog()
    if not catalog:
        return None, None, False

    entry = catalog.get(model)
    if entry:
        return entry, model, True

    for entry in catalog.values():
        for b in entry.get("backends") or []:
            if isinstance(b, dict) and b.get("name") == model:
                return entry, str(entry.get("logical_id") or model), True
    return None, None, True


# Legacy: purpose from /internal/llm/* path → default modality profile
def profile_from_gateway_purpose(purpose: str) -> Dict[str, Any]:
    from aiv2_model_contract.llm_taxonomy import _API_DEFAULT_PROFILES

    base = _API_DEFAULT_PROFILES.get(purpose.strip().lower())
    if base:
        return normalize_llm_profile(base)
    return normalize_llm_profile({"purpose": purpose, "api": purpose})


async def validate_model_for_profile(
    model: str,
    profile: Dict[str, Any],
    *,
    tool_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return None if OK, else error dict for HTTP 400/503."""
    prof = normalize_llm_profile(profile)
    req_in = prof["input_modalities"]
    req_out = prof["output_modalities"]
    api = prof.get("api") or "chat"

    model = (model or "").strip()
    if not model:
        return {
            "error": "model_missing",
            "message": (
                f"No model configured for this tool LLM profile "
                f"(in={req_in}, out={req_out})"
                + (f" on tool '{tool_id}'." if tool_id else ".")
                + " Set binding via PUT /admin/tools/{tool_id}/binding with "
                "input_modalities and output_modalities."
            ),
            "required_profile": prof,
            "tool_id": tool_id,
            "hint": (
                f"GET /admin/llm/catalog?suggest=1"
                f"&input_modalities={','.join(req_in)}"
                f"&output_modalities={','.join(req_out)}"
            ),
        }

    entry, logical_id, catalog_loaded = await resolve_model_entry(model)
    if not entry:
        if not catalog_loaded:
            return {
                "error": "catalog_unavailable",
                "message": (
                    "LLM gateway catalog is temporarily unavailable; "
                    "retry shortly (this is not a missing-model error)."
                ),
                "model": model,
                "required_profile": prof,
                "tool_id": tool_id,
            }
        return {
            "error": "unknown_model",
            "message": f"Model '{model}' is not in the llm-gateway catalog. Register it first.",
            "model": model,
            "required_profile": prof,
            "tool_id": tool_id,
        }

    model_in = entry.get("input_modalities") or []
    model_out = entry.get("output_modalities") or []
    if not model_in and not model_out:
        # Legacy rows without modalities — prefer dispatch_key vs api, else accept.
        wire = (entry.get("dispatch_key") or "").lower()
        api_wires = {
            "chat": {"chat"},
            "embed": {"embeddings", "embed"},
            "image": {"images", "image"},
        }
        allowed = api_wires.get(api, {api})
        if wire and wire not in allowed:
            return {
                "error": "model_dispatch_mismatch",
                "message": (
                    f"Model '{logical_id}' has dispatch_key '{wire}' but profile requires api '{api}'. "
                    "Re-import the model with input/output modalities, or pick another model."
                ),
                "model": logical_id,
                "model_dispatch_key": wire,
                "required_profile": prof,
                "tool_id": tool_id,
            }
        return None

    if not modalities_compatible(model_in, model_out, req_in, req_out):
        return {
            "error": "model_modality_mismatch",
            "message": (
                f"Model '{logical_id}' supports in={model_in} out={model_out} but this call "
                f"requires in={req_in} out={req_out}. "
                f"Use GET /admin/llm/catalog?suggest=1&input_modalities={','.join(req_in)}"
                f"&output_modalities={','.join(req_out)}"
            ),
            "model": logical_id,
            "model_input_modalities": model_in,
            "model_output_modalities": model_out,
            "required_profile": prof,
            "tool_id": tool_id,
            "catalog_entry": {
                "logical_id": logical_id,
                "display_name": entry.get("display_name"),
                "pipeline_tag": entry.get("pipeline_tag"),
            },
        }
    return None


async def validate_model_for_purpose(
    model: str,
    purpose: str,
    *,
    tool_id: Optional[str] = None,
    binding_constraints: Optional[Dict[str, Any]] = None,
    tool_profile: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Validate using merged tool profile + binding constraints + gateway purpose."""
    prof = merge_llm_profiles(
        profile_from_gateway_purpose(purpose),
        binding_constraints,
        tool_profile,
    )
    return await validate_model_for_profile(model, prof, tool_id=tool_id)


async def validate_binding_model(
    model: str,
    input_modalities: List[str],
    output_modalities: List[str],
    *,
    api: Optional[str] = None,
) -> None:
    """Raise ValueError if binding model does not fit modality profile (admin)."""
    prof = normalize_llm_profile({
        "input_modalities": input_modalities,
        "output_modalities": output_modalities,
        "api": api or infer_api_from_modalities(input_modalities, output_modalities),
    })
    err = await validate_model_for_profile(model, prof, tool_id=None)
    if err:
        raise ValueError(err.get("message") or err.get("error"))


def reset_catalog_cache_for_tests() -> None:
    """Clear in-process catalog cache (tests only)."""
    _catalog_cache["at"] = 0.0
    _catalog_cache["by_id"] = {}
