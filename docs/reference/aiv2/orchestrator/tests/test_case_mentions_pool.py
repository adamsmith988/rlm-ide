"""Tests for hub chat case-mention → pool inventory helpers."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

from src.orchestrators.host_adapters.case_mentions import (
    case_files_for_session_registry,
    case_mentions_prompt_addon,
    delete_case_binding,
    extract_case_mention_selections,
    inventory_has_cases,
    list_case_bindings_for_session,
    load_case_bindings,
    load_case_inventory,
    merge_case_selections,
    merge_inventories,
    restore_or_build_case_inventory,
    save_case_bindings,
    save_case_inventory,
    summarize_case_bindings,
    unbind_case_for_session,
)
from src.orchestrators import host_pools
import src.orchestrators.host_adapters.case_mentions as case_mentions_mod


def test_extract_case_mention_selections_with_outputs():
    ctx = {
        "chat_mentions_v1": {
            "cases": [
                {
                    "id": "cas_1",
                    "label": "Demo",
                    "outputs": [
                        {"handle": "artifacts", "mode": "artifacts"},
                        {"handle": "kind:image", "mode": "kind", "kind": "image"},
                    ],
                    "emit_mode": "batch",
                }
            ]
        }
    }
    sels = extract_case_mention_selections(ctx)
    assert len(sels) == 1
    assert sels[0]["case_id"] == "cas_1"
    assert len(sels[0]["outputs"]) == 2


def test_extract_case_mention_legacy_defaults_to_all_artifacts():
    ctx = {"chat_mentions_v1": {"cases": [{"id": "cas_x", "label": "X"}]}}
    sels = extract_case_mention_selections(ctx)
    assert len(sels) == 1
    assert sels[0]["outputs"][0]["mode"] == "artifacts"


def test_merge_inventories_dedupes_files():
    a = host_pools.empty_inventory(note="a")
    a["files"] = [{"artifact_id": "a1", "filename": "one.txt"}]
    a["case_ids"] = ["cas_1"]
    b = host_pools.empty_inventory(note="b")
    b["files"] = [
        {"artifact_id": "a1", "filename": "one.txt"},
        {"artifact_id": "a2", "filename": "two.txt"},
    ]
    b["texts"] = [{"filename": "ev.md", "content": "hi", "handle": "tool:x"}]
    merged = merge_inventories(a, b)
    assert len(merged["files"]) == 2
    assert len(merged["texts"]) == 1
    assert merged["case_ids"] == ["cas_1"]


def test_case_files_for_session_registry_cap():
    inv = host_pools.empty_inventory()
    inv["files"] = [
        {"artifact_id": f"aid_{i}", "original_filename": f"f{i}.bin", "case_id": "cas"}
        for i in range(10)
    ]
    rows = case_files_for_session_registry(inv, cap=3)
    assert len(rows) == 3
    assert rows[0]["source"] == "case_mention"


def test_case_files_for_session_registry_default_cap_zero():
    """Default attach cap is 0 — case files must not dump into Session Files."""
    inv = host_pools.empty_inventory()
    inv["files"] = [
        {"artifact_id": "aid_1", "original_filename": "f.bin", "case_id": "cas"}
    ]
    assert case_files_for_session_registry(inv) == []
    assert case_mentions_mod._SESSION_REGISTRY_CAP == 0


def test_case_mentions_prompt_addon():
    inv = {
        "files": [{"original_filename": "a.jpg", "artifact_id": "1"}],
        "texts": [{"filename": "tool__a.md"}],
        "case_ids": ["cas_1"],
    }
    text = case_mentions_prompt_addon(inv)
    assert "cas_1" in text
    assert "pool files: 1" in text
    assert "NOT attached as Session Files" in text


def test_durable_case_bindings_roundtrip(tmp_path):
    art = str(tmp_path)
    sels = [
        {
            "case_id": "cas_1",
            "label": "Demo",
            "outputs": [{"handle": "artifacts", "mode": "artifacts"}],
            "emit_mode": "batch",
        }
    ]
    save_case_bindings(art, sels)
    loaded = load_case_bindings(art)
    assert len(loaded) == 1
    assert loaded[0]["case_id"] == "cas_1"
    assert os.path.isfile(os.path.join(art, "hub_case_bindings.json"))


def test_durable_case_inventory_roundtrip(tmp_path):
    art = str(tmp_path)
    inv = host_pools.empty_inventory(note="t")
    inv["files"] = [{"artifact_id": "a1", "filename": "x.bin", "case_id": "cas_1"}]
    inv["texts"] = [
        {
            "filename": "asr__clip.md",
            "content": "hello",
            "handle": "tool:asr",
            "source": "case_tool_evidence",
            "case_id": "cas_1",
        }
    ]
    inv["case_ids"] = ["cas_1"]
    inv["count"] = 2
    inv["files_truncated"] = True
    inv["files_total"] = 99
    save_case_inventory(art, inv, selections=[{"case_id": "cas_1", "outputs": []}])
    loaded = load_case_inventory(art)
    assert loaded is not None
    assert inventory_has_cases(loaded)
    assert loaded["case_ids"] == ["cas_1"]
    assert loaded["files"][0]["artifact_id"] == "a1"
    assert loaded["texts"][0]["source"] == "case_tool_evidence"
    assert loaded["files_truncated"] is True
    assert loaded["files_total"] == 99


def test_merge_case_selections_later_wins():
    base = [{"case_id": "a", "label": "A", "outputs": [{"handle": "x"}], "emit_mode": "batch"}]
    new = [
        {"case_id": "a", "label": "A2", "outputs": [{"handle": "y"}], "emit_mode": "batch"},
        {"case_id": "b", "label": "B", "outputs": [{"handle": "z"}], "emit_mode": "batch"},
    ]
    merged = merge_case_selections(base, new)
    by_id = {m["case_id"]: m for m in merged}
    assert by_id["a"]["label"] == "A2"
    assert by_id["a"]["outputs"][0]["handle"] == "y"
    assert "b" in by_id


def test_inventory_has_cases_empty():
    assert not inventory_has_cases(None)
    assert not inventory_has_cases({})
    assert not inventory_has_cases(host_pools.empty_inventory())
    inv = host_pools.empty_inventory()
    inv["case_ids"] = ["cas"]
    assert inventory_has_cases(inv)


def test_summarize_case_bindings_counts_files_and_evidence():
    sels = [
        {
            "case_id": "cas_1",
            "label": "Demo",
            "outputs": [
                {"handle": "artifacts", "mode": "artifacts"},
                {"handle": "tool:asr", "mode": "tool_evidence"},
            ],
            "emit_mode": "batch",
        }
    ]
    inv = host_pools.empty_inventory()
    inv["files"] = [
        {"artifact_id": "a1", "case_id": "cas_1"},
        {"artifact_id": "a2", "case_id": "cas_1"},
    ]
    inv["texts"] = [
        {"filename": "e.md", "case_id": "cas_1", "source": "case_tool_evidence"}
    ]
    inv["case_ids"] = ["cas_1"]
    rows = summarize_case_bindings(sels, inv)
    assert len(rows) == 1
    assert rows[0]["file_count"] == 2
    assert rows[0]["evidence_count"] == 1
    assert "tool:asr" in rows[0]["handles"]
    assert "artifacts" in rows[0]["handles"]


def test_delete_case_binding_clears_files(tmp_path):
    art = str(tmp_path)
    save_case_bindings(
        art,
        [
            {
                "case_id": "cas_1",
                "label": "A",
                "outputs": [{"handle": "artifacts"}],
                "emit_mode": "batch",
            },
            {
                "case_id": "cas_2",
                "label": "B",
                "outputs": [{"handle": "tool:x"}],
                "emit_mode": "batch",
            },
        ],
    )
    remaining = delete_case_binding(art, "cas_1")
    assert len(remaining) == 1
    assert remaining[0]["case_id"] == "cas_2"
    remaining2 = delete_case_binding(art, "cas_2")
    assert remaining2 == []
    assert not os.path.isfile(os.path.join(art, "hub_case_bindings.json"))


@pytest.mark.asyncio
async def test_restore_or_build_without_mention_uses_bindings(tmp_path):
    art = str(tmp_path)
    sels = [
        {
            "case_id": "cas_1",
            "label": "Demo",
            "outputs": [{"handle": "tool:asr", "mode": "tool_evidence"}],
            "emit_mode": "batch",
        }
    ]
    save_case_bindings(art, sels)
    fake_inv = host_pools.empty_inventory(note="fake")
    fake_inv["case_ids"] = ["cas_1"]
    fake_inv["texts"] = [
        {
            "filename": "asr__a.md",
            "content": "transcript",
            "handle": "tool:asr",
            "source": "case_tool_evidence",
            "case_id": "cas_1",
        }
    ]
    fake_inv["count"] = 1

    with patch(
        "src.orchestrators.host_adapters.case_mentions.build_case_mentions_inventory",
        new=AsyncMock(return_value=fake_inv),
    ) as build_mock, patch(
        "src.orchestrators.host_adapters.case_mentions.persist_case_bindings_to_storage",
        new=AsyncMock(return_value=True),
    ):
        restored = await restore_or_build_case_inventory(
            context={},
            user_id="u1",
            session_id="s1",
            artifact_dir=art,
        )
        assert build_mock.await_count == 1
        assert restored["texts"][0]["source"] == "case_tool_evidence"
        # Second call should hit inventory cache without rebuild.
        build_mock.reset_mock()
        cached = await restore_or_build_case_inventory(
            context={},
            user_id="u1",
            session_id="s1",
            artifact_dir=art,
        )
        assert build_mock.await_count == 0
        assert cached["texts"][0]["content"] == "transcript"


@pytest.mark.asyncio
async def test_restore_hydrates_bindings_from_storage_after_disk_wipe(tmp_path):
    """After restart, disk cache is gone — reload from session metadata."""
    art = str(tmp_path / "rlm")
    os.makedirs(art, exist_ok=True)
    remote_sels = [
        {
            "case_id": "cas_1",
            "label": "Demo",
            "outputs": [{"handle": "artifacts", "mode": "artifacts"}],
            "emit_mode": "batch",
        }
    ]
    fake_inv = host_pools.empty_inventory(note="from_storage")
    fake_inv["case_ids"] = ["cas_1"]
    fake_inv["files"] = [
        {"artifact_id": "a1", "filename": "x.bin", "case_id": "cas_1"}
    ]
    fake_inv["count"] = 1

    with patch(
        "src.orchestrators.host_adapters.case_mentions.fetch_case_bindings_from_storage",
        new=AsyncMock(return_value=remote_sels),
    ), patch(
        "src.orchestrators.host_adapters.case_mentions.build_case_mentions_inventory",
        new=AsyncMock(return_value=fake_inv),
    ) as build_mock, patch(
        "src.orchestrators.host_adapters.case_mentions.persist_case_bindings_to_storage",
        new=AsyncMock(return_value=True),
    ):
        restored = await restore_or_build_case_inventory(
            context={},
            user_id="u1",
            session_id="s1",
            artifact_dir=art,
        )
    assert build_mock.await_count == 1
    assert restored["files"][0]["artifact_id"] == "a1"
    # Disk cache hydrated for next local hit
    assert load_case_bindings(art)[0]["case_id"] == "cas_1"


@pytest.mark.asyncio
async def test_list_and_unbind_case_bindings(tmp_path):
    art = str(tmp_path)
    inv = host_pools.empty_inventory()
    inv["files"] = [{"artifact_id": "a1", "case_id": "cas_1", "filename": "f.bin"}]
    inv["texts"] = [
        {
            "filename": "t.md",
            "case_id": "cas_1",
            "source": "case_tool_evidence",
            "content": "x",
        }
    ]
    inv["case_ids"] = ["cas_1"]
    inv["count"] = 2
    sels = [
        {
            "case_id": "cas_1",
            "label": "Demo",
            "outputs": [
                {"handle": "artifacts"},
                {"handle": "tool:asr"},
            ],
            "emit_mode": "batch",
        }
    ]
    save_case_bindings(art, sels)
    save_case_inventory(art, inv, selections=sels)

    listed = await list_case_bindings_for_session(
        session_id="s1", user_id="u1", artifact_dir=art
    )
    assert listed["count"] == 1
    assert listed["bindings"][0]["file_count"] == 1
    assert listed["bindings"][0]["evidence_count"] == 1

    with patch(
        "src.orchestrators.host_adapters.case_mentions.build_case_mentions_inventory",
        new=AsyncMock(return_value=host_pools.empty_inventory()),
    ), patch(
        "src.orchestrators.host_adapters.case_mentions.persist_case_bindings_to_storage",
        new=AsyncMock(return_value=True),
    ) as persist_mock, patch(
        "src.orchestrators.host_adapters.case_mentions.fetch_case_bindings_from_storage",
        new=AsyncMock(return_value=[]),
    ):
        after = await unbind_case_for_session(
            session_id="s1",
            user_id="u1",
            case_id="cas_1",
            artifact_dir=art,
        )
    assert after["count"] == 0
    assert load_case_bindings(art) == []
    assert persist_mock.await_count >= 1


@pytest.mark.asyncio
async def test_inventory_file_cap_keeps_evidence(monkeypatch):
    monkeypatch.setattr(case_mentions_mod, "_INVENTORY_FILE_CAP", 2)

    async def fake_resolve(**kwargs):
        inv = host_pools.empty_inventory()
        inv["case_ids"] = [kwargs["case_id"]]
        inv["files"] = [
            {
                "artifact_id": f"a{i}",
                "filename": f"{i}.bin",
                "case_id": kwargs["case_id"],
            }
            for i in range(5)
        ]
        inv["texts"] = [
            {
                "filename": "ev.md",
                "content": "ok",
                "handle": "tool:x",
                "source": "case_tool_evidence",
                "case_id": kwargs["case_id"],
            }
        ]
        inv["count"] = 6
        return inv

    with patch(
        "src.orchestrators.host_adapters.case_mentions.resolve_case_selection_inventory",
        new=AsyncMock(side_effect=fake_resolve),
    ):
        from src.orchestrators.host_adapters.case_mentions import (
            build_case_mentions_inventory,
        )

        ctx = {
            "chat_mentions_v1": {
                "cases": [
                    {
                        "id": "cas_1",
                        "label": "Big",
                        "outputs": [
                            {"handle": "artifacts"},
                            {"handle": "tool:x"},
                        ],
                    }
                ]
            }
        }
        inv = await build_case_mentions_inventory(ctx, user_id="u1")
    assert inv["files_truncated"] is True
    assert inv["files_total"] == 5
    assert len(inv["files"]) == 2
    assert len(inv["texts"]) == 1
    assert inv["texts"][0]["source"] == "case_tool_evidence"
