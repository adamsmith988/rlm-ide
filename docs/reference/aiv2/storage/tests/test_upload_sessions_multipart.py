"""Unit tests for multipart threshold / part planning helpers."""

from __future__ import annotations

import math
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import settings
from src.upload_sessions import _multipart_enabled, staging_object_key


class TestUploadSessionHelpers(unittest.TestCase):
    def test_staging_object_key_sanitizes(self):
        key = staging_object_key("u/1", "sess/a", "fid/b")
        self.assertEqual(key, "import_staging/u_1/sess_a/fid_b")

    def test_multipart_threshold(self):
        with patch.object(settings, "UPLOAD_MULTIPART_THRESHOLD_BYTES", 32 * 1024 * 1024):
            self.assertFalse(_multipart_enabled(1024))
            self.assertTrue(_multipart_enabled(32 * 1024 * 1024))
        with patch.object(settings, "UPLOAD_MULTIPART_THRESHOLD_BYTES", 0):
            self.assertFalse(_multipart_enabled(100 * 1024 * 1024))

    def test_part_count_math(self):
        length = 50 * 1024 * 1024
        part = 16 * 1024 * 1024
        self.assertEqual(math.ceil(length / part), 4)


class TestMultipartCreateMock(unittest.IsolatedAsyncioTestCase):
    async def test_create_returns_multipart_shape(self):
        from src import upload_sessions as us

        minio = MagicMock()
        minio.is_connected = True
        minio.create_multipart_upload = AsyncMock(return_value="upid-1")
        minio.get_presigned_upload_part_url = AsyncMock(
            side_effect=lambda *a, **k: f"http://media/part-{a[2]}"
        )
        minio.list_multipart_parts = AsyncMock(return_value=[])
        storage = MagicMock()
        storage.minio = minio

        with patch.object(us, "get_storage", AsyncMock(return_value=storage)):
            with patch.object(settings, "UPLOAD_MULTIPART_THRESHOLD_BYTES", 1024):
                with patch.object(
                    settings, "UPLOAD_MULTIPART_PART_SIZE_BYTES", 5 * 1024 * 1024
                ):
                    with patch.object(settings, "MINIO_BUCKET", "uploads"):
                        req = MagicMock()
                        req.headers = {"X-User-Id": "user-1"}
                        body = us.UploadSessionCreate(
                            session_id="12345678-aaaa-bbbb-cccc-123456789012",
                            file_id="aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
                            upload_length=12 * 1024 * 1024,
                        )
                        out = await us.create_upload_session(body, req)
        self.assertEqual(out["mode"], "multipart")
        self.assertEqual(out["upload_id"], "upid-1")
        self.assertEqual(len(out["parts"]), 3)


if __name__ == "__main__":
    unittest.main()
