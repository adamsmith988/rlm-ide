"""Unit tests for orchestrator native corpus RAG."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

SERVICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from src.rag.client import (  # noqa: E402
    RagResult,
    combine_scores,
    rank_and_trim,
    retrieve_for_prompt,
)
from src.rag.prompt import format_rag_prompt_block  # noqa: E402


def test_combine_scores_weights():
    assert abs(combine_scores(1.0, 0.0) - 0.4) < 1e-9
    assert abs(combine_scores(0.0, 1.0) - 0.6) < 1e-9
    assert abs(combine_scores(0.5, 0.5) - 0.5) < 1e-9


def test_rank_and_trim_top_k():
    cands = [
        {"text": "a", "score": 0.2, "combined_score": 0.2},
        {"text": "b", "score": 0.9, "combined_score": 0.9},
        {"text": "c", "score": 0.5, "combined_score": 0.5},
        {"text": "d", "score": 0.7, "combined_score": 0.7},
        {"text": "e", "score": 0.1, "combined_score": 0.1},
        {"text": "f", "score": 0.8, "combined_score": 0.8},
    ]
    top = rank_and_trim(cands, top_k=5, reranked=True)
    assert len(top) == 5
    assert [h["text"] for h in top] == ["b", "f", "d", "c", "a"]


def test_format_rag_prompt_block_citations_and_truncation():
    hits = [
        {
            "text": "hello world " * 200,
            "filename": "doc.txt",
            "artifact_id": "aid-1",
            "combined_score": 0.91,
        },
        {
            "text": "second snippet",
            "filename": "other.md",
            "artifact_id": "aid-2",
            "score": 0.5,
        },
    ]
    block = format_rag_prompt_block(hits, max_chars_per_hit=80)
    assert "## Retrieved corpus context" in block
    assert "[1] file=doc.txt artifact_id=aid-1" in block
    assert "[2] file=other.md artifact_id=aid-2" in block
    assert "..." in block
    assert "second snippet" in block


def test_format_rag_prompt_block_empty():
    assert format_rag_prompt_block([]) == ""
    assert format_rag_prompt_block([{"text": ""}]) == ""


def test_native_prompt_helper_block_shape():
    block = format_rag_prompt_block(
        [{"text": "body", "filename": "x", "artifact_id": "y", "combined_score": 0.9}]
    )
    assert block.startswith("## Retrieved corpus context")
    assert "[1] file=x artifact_id=y" in block
    assert "body" in block


def test_retrieve_binding_missing():
    async def _run():
        binding = MagicMock()
        binding.get_model = AsyncMock(side_effect=["", "rerank-model"])
        with patch("src.llm.binding_client.get_binding_client", return_value=binding), patch(
            "src.rag.client.settings"
        ) as mock_settings:
            mock_settings.NATIVE_RAG_ENABLED = True
            mock_settings.NATIVE_RAG_TIMEOUT_S = 5
            mock_settings.NATIVE_RAG_TOP_K = 5
            mock_settings.NATIVE_RAG_CANDIDATE_LIMIT = 25
            mock_settings.NATIVE_RAG_COLLECTION = "tool_embed_text"
            mock_settings.NATIVE_RAG_SCORE_THRESHOLD = 0.3
            mock_settings.NATIVE_RAG_MAX_CHARS_PER_HIT = 1200
            result = await retrieve_for_prompt("query", user_id="u1")
        assert isinstance(result, RagResult)
        assert result.hits == []
        assert result.meta.get("reason") == "binding_missing"
        assert result.meta.get("embed_bound") is False

    asyncio.run(_run())


def test_retrieve_ann_only_when_rerank_unbound():
    async def _run():
        ann_rows = [
            {
                "id": f"p{i}",
                "score": 0.9 - i * 0.1,
                "payload": {
                    "artifact_id": f"a{i}",
                    "filename": f"f{i}.txt",
                    "text_preview": f"snippet {i}",
                },
            }
            for i in range(3)
        ]
        fake = _FakeClient(
            [
                _FakeResp(200, {"data": [{"embedding": [0.1, 0.2]}]}),
                _FakeResp(200, ann_rows),
            ]
        )
        binding = MagicMock()
        # embed bound, rerank missing
        binding.get_model = AsyncMock(side_effect=["embed-m", ""])

        with patch(
            "src.llm.binding_client.get_binding_client", return_value=binding
        ), patch("src.rag.client.httpx.AsyncClient", return_value=fake), patch(
            "src.rag.client.settings"
        ) as mock_settings, patch(
            "src.rag.client.llm_gateway_v1_url", return_value="http://gw/v1"
        ):
            mock_settings.NATIVE_RAG_ENABLED = True
            mock_settings.NATIVE_RAG_TIMEOUT_S = 5
            mock_settings.NATIVE_RAG_TOP_K = 5
            mock_settings.NATIVE_RAG_CANDIDATE_LIMIT = 25
            mock_settings.NATIVE_RAG_COLLECTION = "tool_embed_text"
            mock_settings.NATIVE_RAG_SCORE_THRESHOLD = 0.3
            mock_settings.NATIVE_RAG_MAX_CHARS_PER_HIT = 1200
            mock_settings.STORAGE_URL = "http://storage:8004"
            mock_settings.LLM_PROXY_URL = "http://gw"
            mock_settings.INTERNAL_API_KEY = "k"
            mock_settings.SERVICE_NAME = "orchestrator"
            mock_settings.INTERNAL_API_KEY = "not-needed"

            result = await retrieve_for_prompt("q", user_id="u1")

        assert result.meta.get("reason") == "ok_ann_only"
        assert result.meta.get("reranked") is False
        assert result.meta.get("rerank_bound") is False
        assert len(result.hits) == 3
        assert "## Retrieved corpus context" in result.prompt_block
        assert "snippet 0" in result.prompt_block
        # No /rerank call when unbound
        assert not any("/rerank" in c["url"] for c in fake.calls)

    asyncio.run(_run())


class _FakeResp:
    def __init__(self, status: int, payload: Any):
        self.status_code = status
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http_{self.status_code}")

    def json(self) -> Any:
        return self._payload


class _FakeClient:
    def __init__(self, responses: List[_FakeResp]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url: str, json: Optional[Dict] = None, headers: Optional[Dict] = None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        if not self._responses:
            raise RuntimeError("no_more_responses")
        return self._responses.pop(0)


def test_retrieve_happy_path_top5_rerank():
    async def _run():
        ann_rows = []
        for i in range(6):
            ann_rows.append(
                {
                    "id": f"p{i}",
                    "score": 0.1 * (i + 1),
                    "payload": {
                        "artifact_id": f"a{i}",
                        "filename": f"f{i}.txt",
                        "text_preview": f"document number {i} about cats",
                    },
                }
            )
        rerank_rows = [{"index": i, "score": float(i) / 10.0} for i in range(6)]

        fake = _FakeClient(
            [
                _FakeResp(200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]}),
                _FakeResp(200, ann_rows),
                _FakeResp(200, rerank_rows),
            ]
        )
        binding = MagicMock()
        binding.get_model = AsyncMock(side_effect=["embed-m", "rerank-m"])

        with patch(
            "src.llm.binding_client.get_binding_client", return_value=binding
        ), patch("src.rag.client.httpx.AsyncClient", return_value=fake), patch(
            "src.rag.client.settings"
        ) as mock_settings, patch(
            "src.rag.client.llm_gateway_v1_url", return_value="http://gw/v1"
        ):
            mock_settings.NATIVE_RAG_ENABLED = True
            mock_settings.NATIVE_RAG_TIMEOUT_S = 5
            mock_settings.NATIVE_RAG_TOP_K = 5
            mock_settings.NATIVE_RAG_CANDIDATE_LIMIT = 25
            mock_settings.NATIVE_RAG_COLLECTION = "tool_embed_text"
            mock_settings.NATIVE_RAG_SCORE_THRESHOLD = 0.3
            mock_settings.NATIVE_RAG_MAX_CHARS_PER_HIT = 1200
            mock_settings.STORAGE_URL = "http://storage:8004"
            mock_settings.LLM_PROXY_URL = "http://gw"
            mock_settings.INTERNAL_API_KEY = "k"
            mock_settings.SERVICE_NAME = "orchestrator"
            mock_settings.INTERNAL_API_KEY = "not-needed"

            result = await retrieve_for_prompt(
                "cats",
                user_id="user-1",
                auth_context={"user_id": "user-1"},
            )

        assert result.meta.get("reason") == "ok"
        assert result.meta.get("reranked") is True
        assert len(result.hits) == 5
        assert "## Retrieved corpus context" in result.prompt_block
        search_call = next(c for c in fake.calls if "/vectors/search" in c["url"])
        assert "filter_dict" not in (search_call["json"] or {})
        assert search_call["json"]["collection_name"] == "tool_embed_text"
        assert search_call["headers"].get("X-User-Id") == "user-1"
        assert "X-Case-Id" not in search_call["headers"]

    asyncio.run(_run())


def test_retrieve_empty_storage_soft_fail():
    async def _run():
        fake = _FakeClient(
            [
                _FakeResp(200, {"data": [{"embedding": [0.1]}]}),
                _FakeResp(200, []),
            ]
        )
        binding = MagicMock()
        binding.get_model = AsyncMock(side_effect=["embed-m", "rerank-m"])

        with patch(
            "src.llm.binding_client.get_binding_client", return_value=binding
        ), patch("src.rag.client.httpx.AsyncClient", return_value=fake), patch(
            "src.rag.client.settings"
        ) as mock_settings, patch(
            "src.rag.client.llm_gateway_v1_url", return_value="http://gw/v1"
        ):
            mock_settings.NATIVE_RAG_ENABLED = True
            mock_settings.NATIVE_RAG_TIMEOUT_S = 5
            mock_settings.NATIVE_RAG_TOP_K = 5
            mock_settings.NATIVE_RAG_CANDIDATE_LIMIT = 25
            mock_settings.NATIVE_RAG_COLLECTION = "tool_embed_text"
            mock_settings.NATIVE_RAG_SCORE_THRESHOLD = None
            mock_settings.NATIVE_RAG_MAX_CHARS_PER_HIT = 1200
            mock_settings.STORAGE_URL = "http://storage:8004"
            mock_settings.LLM_PROXY_URL = "http://gw"
            mock_settings.INTERNAL_API_KEY = "k"
            mock_settings.SERVICE_NAME = "orchestrator"
            mock_settings.INTERNAL_API_KEY = "not-needed"

            result = await retrieve_for_prompt("q", user_id="u")

        assert result.hits == []
        assert result.meta.get("reason") == "no_results"

    asyncio.run(_run())


def test_retrieve_network_error_soft_fail():
    async def _run():
        class BoomClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, *a, **k):
                raise ConnectionError("down")

        binding = MagicMock()
        binding.get_model = AsyncMock(side_effect=["embed-m", "rerank-m"])

        with patch(
            "src.llm.binding_client.get_binding_client", return_value=binding
        ), patch("src.rag.client.httpx.AsyncClient", return_value=BoomClient()), patch(
            "src.rag.client.settings"
        ) as mock_settings, patch(
            "src.rag.client.llm_gateway_v1_url", return_value="http://gw/v1"
        ):
            mock_settings.NATIVE_RAG_ENABLED = True
            mock_settings.NATIVE_RAG_TIMEOUT_S = 5
            mock_settings.NATIVE_RAG_TOP_K = 5
            mock_settings.NATIVE_RAG_CANDIDATE_LIMIT = 25
            mock_settings.NATIVE_RAG_COLLECTION = "tool_embed_text"
            mock_settings.NATIVE_RAG_SCORE_THRESHOLD = 0.3
            mock_settings.NATIVE_RAG_MAX_CHARS_PER_HIT = 1200
            mock_settings.STORAGE_URL = "http://storage:8004"
            mock_settings.LLM_PROXY_URL = "http://gw"
            mock_settings.INTERNAL_API_KEY = "k"
            mock_settings.SERVICE_NAME = "orchestrator"
            mock_settings.INTERNAL_API_KEY = "not-needed"

            result = await retrieve_for_prompt("q", user_id="u")

        assert result.hits == []
        assert str(result.meta.get("reason", "")).startswith("error:")

    asyncio.run(_run())
