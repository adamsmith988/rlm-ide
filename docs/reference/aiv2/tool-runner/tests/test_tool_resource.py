"""Generic resource_class derivation — no tool-id hardcoding."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

SERVICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from src.tool_resource import (  # noqa: E402
    resolve_tool_resource,
    resolve_tool_resource_from_adapter,
)


def test_explicit_meta_resource_class_wins() -> None:
    assert resolve_tool_resource(
        backend="local",
        vram_mb=8192,
        capabilities=["vision"],
        meta={"resource_class": "cpu", "resource_weight": 3},
    ) == ("cpu", 3)


def test_vram_local_maps_to_gpu() -> None:
    cls, weight = resolve_tool_resource(backend="local", vram_mb=2048, capabilities=[])
    assert cls == "gpu"
    assert weight >= 1


def test_llm_backend_and_caps_map_to_llm() -> None:
    assert resolve_tool_resource(backend="vllm", vram_mb=0, capabilities=[])[0] == "llm"
    assert resolve_tool_resource(backend="local", vram_mb=0, capabilities=["vision"])[0] == "llm"
    assert resolve_tool_resource(backend="openai", vram_mb=0, capabilities=[])[0] == "llm"


def test_embedding_caps_map_to_gpu_not_llm() -> None:
    assert resolve_tool_resource(
        backend="vllm", vram_mb=500, capabilities=["embedding", "text_vectorization"]
    ) == ("gpu", 1)
    assert resolve_tool_resource(
        backend="triton", vram_mb=0, capabilities=["image_embedding"]
    )[0] == "gpu"
    assert resolve_tool_resource(
        backend="vllm",
        vram_mb=0,
        capabilities=[],
        meta={"llm_profile": {"api": "embed"}},
    )[0] == "gpu"


def test_explicit_gpu_resource_class_on_embed_meta() -> None:
    assert resolve_tool_resource(
        backend="vllm",
        capabilities=["embedding"],
        meta={"resource_class": "gpu"},
    ) == ("gpu", 1)


def test_default_is_cpu() -> None:
    assert resolve_tool_resource(backend="local", vram_mb=0, capabilities=["file"]) == (
        "cpu",
        1,
    )


def test_adapter_wrapper() -> None:
    adapter = SimpleNamespace(
        backend="vllm",
        vram_mb=0,
        capabilities=["chat"],
        meta={},
    )
    assert resolve_tool_resource_from_adapter(adapter) == ("llm", 1)
    assert resolve_tool_resource_from_adapter(None) == ("cpu", 1)
