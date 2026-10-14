"""Unit tests for user_messenger repository helpers."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.repositories.user_messenger import (  # noqa: E402
    _dm_pair_key,
    _normalize_attachments,
    _strip_html,
)


def test_strip_html():
    assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"


def test_dm_pair_key_sorted():
    a = "11111111-1111-1111-1111-111111111111"
    b = "22222222-2222-2222-2222-222222222222"
    assert _dm_pair_key(a, b) == _dm_pair_key(b, a)


def test_normalize_attachments_string():
    out = _normalize_attachments(["abc-123"])
    assert len(out) == 1
    assert out[0]["artifact_id"] == "abc-123"
    assert out[0]["source"] == "upload"


def test_normalize_attachments_file_manager():
    out = _normalize_attachments([
        {"artifact_id": "x", "source": "file_manager", "share_mode": "share_link", "expires_sec": 3600}
    ])
    assert out[0]["source"] == "file_manager"
    assert out[0]["share_mode"] == "share_link"
    assert out[0]["expires_sec"] == 3600
