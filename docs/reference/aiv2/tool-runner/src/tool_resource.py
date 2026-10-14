"""Derive JobAdmission resource_class/weight from tool metadata (generic).

No tool-id hardcoding: callers pass adapter/meta fields from tool.json.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

_LLM_BACKENDS = frozenset(
    {
        "vllm",
        "openai",
        "openai_compat",
        "llm",
        "chat",
    }
)
_LLM_CAPS = frozenset(
    {
        "vision",
        "chat",
        "llm",
        "caption",
        "ocr",
        "video",
        "asr",
        "transcribe",
    }
)
# Embedding / vector tools hit vllm_embed or triton pools (capacity → gpu),
# not the chat AIMD ``llm`` class — even when backend is labeled ``vllm``.
_EMBED_CAPS = frozenset(
    {
        "embedding",
        "embeddings",
        "embed",
        "text_vectorization",
        "image_embedding",
        "text_embedding",
        "face_embedding",
        "face_recognition",
        "vector_search",
        "cross_modal_search",
    }
)


def resolve_tool_resource(
    *,
    backend: str = "local",
    vram_mb: int = 0,
    capabilities: Optional[Sequence[Any]] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, int]:
    """Return ``(resource_class, resource_weight)`` for JobAdmission.

    Precedence:
    1. Explicit ``meta.resource_class`` (+ optional ``resource_weight``)
    2. Embedding capabilities / embed api → ``gpu`` (follows vllm_embed/triton capacity)
    3. ``vram_mb > 0`` with a local/GPU backend → ``gpu``
    4. LLM-ish backend or capabilities → ``llm``
    5. else ``cpu``
    """
    meta = meta or {}
    explicit = str(meta.get("resource_class") or "").strip().lower()
    if explicit:
        try:
            weight = max(1, int(meta.get("resource_weight") or 1))
        except (TypeError, ValueError):
            weight = 1
        return explicit, weight

    backend_l = str(backend or "local").strip().lower()
    try:
        vram = int(vram_mb or 0)
    except (TypeError, ValueError):
        vram = 0
    caps = {str(c).strip().lower() for c in (capabilities or []) if c is not None}
    llm_profile = meta.get("llm_profile") if isinstance(meta.get("llm_profile"), Mapping) else {}
    api = str(llm_profile.get("api") or meta.get("api") or "").strip().lower()

    if caps.intersection(_EMBED_CAPS) or api in ("embed", "embedding", "embeddings"):
        return "gpu", 1
    if backend_l in ("triton",) and (
        "embed" in "".join(caps) or api.startswith("embed")
    ):
        return "gpu", 1

    if vram > 0 and backend_l not in ("vllm", "cpu", "worker", "openai", "openai_compat"):
        # Heavier GPU tools can weigh more than 1; keep it simple and stable.
        return "gpu", max(1, min(8, (vram + 1023) // 1024))

    if backend_l in _LLM_BACKENDS or caps.intersection(_LLM_CAPS):
        return "llm", 1

    return "cpu", 1


def resolve_tool_resource_from_adapter(adapter: Any) -> Tuple[str, int]:
    """Convenience wrapper for ToolAdapter-like objects."""
    if adapter is None:
        return "cpu", 1
    meta = getattr(adapter, "meta", None) or {}
    if not isinstance(meta, Mapping):
        meta = {}
    return resolve_tool_resource(
        backend=getattr(adapter, "backend", "local") or "local",
        vram_mb=getattr(adapter, "vram_mb", 0) or 0,
        capabilities=getattr(adapter, "capabilities", None) or [],
        meta=meta,
    )
