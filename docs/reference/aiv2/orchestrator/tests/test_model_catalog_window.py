"""Catalog → orchestrator context_length stamping (Agent-Flow parity)."""

from __future__ import annotations

from src.orchestrators.host_adapters.model_catalog import (
    model_window_from_catalog,
    stamp_context_window,
)


def test_model_window_from_catalog_prefers_context_length():
    meta = model_window_from_catalog(
        {
            "logical_id": "gemma4-31",
            "context_length": 30000,
            "max_output_tokens": 4096,
        }
    )
    assert meta["context_length"] == 30000
    assert meta["max_output_tokens"] == 4096


def test_model_window_from_catalog_falls_back_to_max_model_len():
    meta = model_window_from_catalog(
        {"logical_id": "qwen", "max_model_len": 8192}
    )
    assert meta["context_length"] == 8192
    assert meta["max_model_len"] == 8192


def test_stamp_context_window_onto_cfg():
    cfg = stamp_context_window(
        {"run_lifetime": "chat", "model": "gemma4-31"},
        {"context_length": 30000, "max_output_tokens": 2048},
    )
    assert cfg["context_length"] == 30000
    assert cfg["max_output_tokens"] == 2048
    assert cfg["run_lifetime"] == "chat"


def test_stamp_keeps_existing_when_meta_empty():
    cfg = stamp_context_window({"context_length": 16000}, {})
    assert cfg["context_length"] == 16000


def test_engine_minimum_window_bounds_a_lowered_window():
    meta = {"context_length": 32768}
    assert stamp_context_window({"context_length": 8192}, meta, min_context=16384)["context_length"] == 16384
    assert stamp_context_window({"context_length": 8192}, {"context_length": 12000}, min_context=16384)["context_length"] == 12000
    assert stamp_context_window({"context_length": 20000}, meta, min_context=16384)["context_length"] == 20000
    assert stamp_context_window({"context_length": 8192, "min_context_length": 4096}, meta, min_context=16384)["context_length"] == 8192


def test_configured_window_may_only_lower_the_model_window():
    assert stamp_context_window({"context_length": 8192}, {"context_length": 32768})["context_length"] == 8192
    assert stamp_context_window({"context_length": 65536}, {"context_length": 32768})["context_length"] == 32768


def test_stamp_run_schema_window():
    meta = model_window_from_catalog(
        {
            "logical_id": "x",
            "run_schema": {"max_seq_len": 4096, "max_output_tokens": 512},
        }
    )
    cfg = stamp_context_window({}, meta)
    assert cfg["context_length"] == 4096
    assert cfg["max_output_tokens"] == 512
