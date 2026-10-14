"""Unit tests for MinIO metadata ASCII sanitization."""

from __future__ import annotations

from urllib.parse import quote

from services.storage.src.backends.minio import _ascii_safe_metadata


def test_ascii_safe_metadata_leaves_ascii():
    out = _ascii_safe_metadata(
        {"original_filename": "report.docx", "user_id": "u1"}
    )
    assert out == {"original_filename": "report.docx", "user_id": "u1"}


def test_ascii_safe_metadata_quotes_non_ascii():
    name = "تقرير الفرقة 64.docx"
    out = _ascii_safe_metadata({"original_filename": name, "source": "chat"})
    assert out is not None
    assert out["source"] == "chat"
    assert out["original_filename"] == quote(name, safe="")
    out["original_filename"].encode("ascii")  # must not raise


def test_ascii_safe_metadata_empty():
    assert _ascii_safe_metadata(None) is None
    assert _ascii_safe_metadata({}) is None
