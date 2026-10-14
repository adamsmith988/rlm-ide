"""Download defaults to byte stream; CDN 302 only when redirect=true."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1] / "src" / "main.py"


class TestDownloadRedirectOptIn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = MAIN.read_text(encoding="utf-8")

    def test_download_query_has_redirect_default_false(self):
        self.assertIn("redirect: bool = Query(", self.src)
        # Default must be False (stream), not True.
        self.assertRegex(
            self.src,
            r"redirect:\s*bool\s*=\s*Query\(\s*\n\s*False,",
        )

    def test_presigned_302_only_under_redirect_flag(self):
        self.assertIn("if redirect:", self.src)
        self.assertIn("RedirectResponse(", self.src)
        # Must not auto-redirect for api-gateway service name alone.
        self.assertNotIn('redirect_ok = service_name in {"", "api-gateway"}', self.src)

    def test_download_handler_parses(self):
        ast.parse(self.src)


if __name__ == "__main__":
    unittest.main()
