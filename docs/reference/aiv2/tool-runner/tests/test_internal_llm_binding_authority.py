import asyncio
import json
from unittest.mock import AsyncMock, patch

from starlette.requests import Request

from src.routes.internal_llm import _build_binding_chain, _maybe_rewrite_model


def _request(headers=None):
    raw_headers = [
        (str(k).lower().encode(), str(v).encode())
        for k, v in (headers or {}).items()
    ]
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/llm/rerank",
            "headers": raw_headers,
        }
    )


def test_bound_tool_overrides_explicit_payload_model():
    request = _request(
        {
            "X-AIv2-Tool-Id": "plugin_smart_search_text",
            "X-AIv2-Plugin-Id": "smart_search",
        }
    )
    resolver = AsyncMock(
        return_value=("bound-logical-id", "plugin.smart_search.rerank", {})
    )

    async def _run():
        with patch(
            "src.routes.internal_llm._resolve_model_for",
            new=resolver,
        ):
            return await _maybe_rewrite_model(
                request,
                json.dumps({"model": "attempted-bypass", "query": "q"}).encode(),
                purpose="rerank",
            )

    body, route_key = asyncio.run(_run())
    assert json.loads(body)["model"] == "bound-logical-id"
    assert route_key == "plugin.smart_search.rerank"
    resolver.assert_awaited_once()


def test_pre_resolved_binding_slot_honors_explicit_model_and_route_key():
    """Dual-bind tools pin slot logical_id + route_key; path purpose must not win."""
    request = _request(
        {
            "X-AIv2-Tool-Id": "audio_transcribe",
            "X-AIv2-Route-Key": "tool.audio_transcribe.diarize",
        }
    )
    resolver = AsyncMock(
        return_value=("asr-bound-should-not-win", "tool.audio_transcribe.asr", {})
    )

    async def _run():
        with patch("src.routes.internal_llm._resolve_model_for", new=resolver):
            return await _maybe_rewrite_model(
                request,
                json.dumps(
                    {"model": "nvidia-nemotron-3-diarization", "query": "unused"}
                ).encode(),
                purpose="asr",
            )

    body, route_key = asyncio.run(_run())
    assert json.loads(body)["model"] == "nvidia-nemotron-3-diarization"
    assert route_key == "tool.audio_transcribe.diarize"
    resolver.assert_not_awaited()


def test_unbound_internal_caller_can_keep_explicit_model():
    body, route_key = asyncio.run(
        _maybe_rewrite_model(
            _request(),
            json.dumps({"model": "direct-logical-id", "query": "q"}).encode(),
            purpose="rerank",
        )
    )
    assert json.loads(body)["model"] == "direct-logical-id"
    assert route_key is None


def test_unbound_tool_header_falls_back_to_explicit_model():
    """RLM compact: tool id header present, no admin binding, Agent model set."""
    request = _request(
        {
            "X-AIv2-Tool-Id": "plugin_agent_flow_run",
            "X-AIv2-Plugin-Id": "agent_flow",
        }
    )
    resolver = AsyncMock(return_value=(None, None, {}))

    async def _run():
        with patch("src.routes.internal_llm._resolve_model_for", new=resolver):
            return await _maybe_rewrite_model(
                request,
                json.dumps(
                    {
                        "model": "gemma4-31",
                        "messages": [{"role": "user", "content": "summarize"}],
                    }
                ).encode(),
                purpose="chat",
            )

    body, route_key = asyncio.run(_run())
    assert json.loads(body)["model"] == "gemma4-31"
    assert route_key is None
    # Agent Flow with an explicit node model must not hit binding resolve.
    resolver.assert_not_awaited()


def test_agent_flow_never_overrides_node_model_with_binding():
    """User-picked Agent/Model node model wins; no X-AIv2-Route-Key."""
    request = _request(
        {
            "X-AIv2-Tool-Id": "plugin_agent_flow_run",
            "X-AIv2-Plugin-Id": "agent_flow",
        }
    )
    resolver = AsyncMock(
        return_value=("bound-should-not-win", "tool.agent_flow.chat", {})
    )

    async def _run():
        with patch("src.routes.internal_llm._resolve_model_for", new=resolver):
            return await _maybe_rewrite_model(
                request,
                json.dumps(
                    {
                        "model": "qwen3-8-27",
                        "route_key": "tool.agent_flow.chat",
                        "messages": [],
                    }
                ).encode(),
                purpose="chat",
            )

    body, route_key = asyncio.run(_run())
    payload = json.loads(body)
    assert payload["model"] == "qwen3-8-27"
    assert "route_key" not in payload
    assert route_key is None
    resolver.assert_not_awaited()


def test_unbound_tool_auto_model_still_errors():
    request = _request(
        {
            "X-AIv2-Tool-Id": "plugin_agent_flow_run",
            "X-AIv2-Plugin-Id": "agent_flow",
        }
    )
    resolver = AsyncMock(return_value=(None, None, {}))

    async def _run():
        with patch("src.routes.internal_llm._resolve_model_for", new=resolver):
            return await _maybe_rewrite_model(
                request,
                json.dumps({"model": "auto", "messages": []}).encode(),
                purpose="chat",
            )

    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as ei:
        asyncio.run(_run())
    assert ei.value.status_code == 503
    assert ei.value.detail["error"] == "binding_not_configured"


def test_query_rewrite_binding_overrides_explicit_payload_model():
    request = _request(
        {
            "X-AIv2-Tool-Id": "plugin_smart_search",
            "X-AIv2-Plugin-Id": "smart_search",
        }
    )
    resolver = AsyncMock(
        return_value=(
            "bound-rewrite-logical-id",
            "plugin.smart_search.query_rewrite",
            {},
        )
    )

    async def _run():
        with patch("src.routes.internal_llm._resolve_model_for", new=resolver):
            return await _maybe_rewrite_model(
                request,
                json.dumps(
                    {"model": "attempted-bypass", "messages": [{"role": "user", "content": "q"}]}
                ).encode(),
                purpose="query_rewrite",
            )

    body, route_key = asyncio.run(_run())
    assert json.loads(body)["model"] == "bound-rewrite-logical-id"
    assert route_key == "plugin.smart_search.query_rewrite"
    resolver.assert_awaited_once_with(
        "query_rewrite", "plugin_smart_search", "smart_search"
    )


def test_query_rewrite_chain_cannot_be_widened_by_generic_llm_routes():
    assert _build_binding_chain(
        "query_rewrite", "plugin_smart_search", "smart_search"
    ) == [
        "tool.plugin_smart_search.query_rewrite",
        "plugin.smart_search.query_rewrite",
    ]
