"""
دروازهٔ داخلی LLM — فقط داخل شبکهٔ سرویس / همان کانتینر tool-runner

ابزارها و پلاگین‌ها مستقیماً llm-proxy یا IP نود را صدا نمی‌زنند؛ به این مسیرها
می‌زنند. Binding فقط ``logical_id`` را وصل می‌کند؛ تخصیص backend کار LB است:

  plugin/tool → /internal/llm/* → llm-proxy (LB by logical_id) → backend

POST /internal/llm/chat/completions      →  {LLM_PROXY_URL}/v1/chat/completions
POST /internal/llm/query_rewrite         →  {LLM_PROXY_URL}/v1/chat/completions
POST /internal/llm/embeddings           →  {LLM_PROXY_URL}/v1/embeddings
POST /internal/llm/images/generations   →  {LLM_PROXY_URL}/v1/images/generations
POST /internal/llm/infer                →  {LLM_PROXY_URL}/v1/infer
POST /internal/llm/rerank               →  {LLM_PROXY_URL}/rerank
POST /internal/llm/audio/transcriptions →  {LLM_PROXY_URL}/v1/audio/transcriptions
POST /internal/llm/audio/diarize        →  {LLM_PROXY_URL}/v1/speech/diarize
POST /internal/llm/audio/speech         →  {LLM_PROXY_URL}/v1/audio/speech
POST /internal/llm/custom/generate      →  {LLM_PROXY_URL}/v1/custom/generate
POST /internal/llm/audio/voiceprints    →  {LLM_PROXY_URL}/v1/audio/voiceprints
GET  /internal/llm/catalog              →  {LLM_PROXY_URL}/catalog
GET  /internal/llm/binding/resolve      →  {LLM_PROXY_URL}/binding/resolve
*    /internal/llm/comfy/{path}         →  {LLM_PROXY_URL}/v1/comfy/{path}

Tool/Plugin model binding
-------------------------
هر ابزار می‌تواند model را ``"auto"`` (یا کاملاً حذف) بفرستد. آنگاه forwarder
بر اساس headerهای ``X-AIv2-Tool-Id`` / ``X-AIv2-Plugin-Id`` و purpose مشتق‌شده
از path، زنجیرهٔ زیر را از ``llm-gateway`` resolve می‌کند و model را پر می‌کند:

    tool.<tool_id>.llm
    → tool.<tool_id>.<purpose>
    → plugin.<plugin_id>.llm
    → plugin.<plugin_id>.<purpose>

تنظیم این bindingها توسط ادمین از طریق ``api-gateway`` انجام می‌شود. اگر هیچ
binding ای نباشد ولی model در body پر باشد، همان فرستاده می‌شود.

امنیت:
- اگر TOOL_RUNNER_INTERNAL_SECRET غیرخالی باشد، هدر X-Tool-Runner-Internal-Secret اجباری است
  (توصیه برای production و برای dev وقتی پورت tool-runner به میزبان publish می‌شود).
- اگر کاملاً خالی/حذف باشد، فقط loopback پذیرفته می‌شود (فقط وقتی مطمئنید پورت عمومی نیست).

پشتیبانی stream=true برای RLM live reasoning: SSE را از llm-proxy عبور می‌دهد.

صف (queue) در llm-proxy است (ConcurrencyLimiter): tool-runner یک درخواست HTTP با
timeout بلند باز نگه می‌دارد تا llm-proxy نوبت بدهد — retry روی 503 نمی‌زند.
Retry فقط برای connect refused / TCP blip قبل از پاسخ HTTP است.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from ..config import settings
from aiv2_model_contract.llm_taxonomy import merge_llm_profiles

from ..llm_model_validator import profile_from_gateway_purpose
from ..llm_upstream import (
    infra_retry_attempts,
    infra_retry_delay_sec,
    upstream_get,
    upstream_post,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/llm",
    tags=["internal-llm"],
    include_in_schema=False,
)

_ALLOWED_LOOPBACK = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})
_AUTO_TOKENS = {"", "auto", "default", None}


def _llm_upstream_headers(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Headers for llm-gateway (requires X-Internal-Key)."""
    headers = dict(base or {})
    key = (settings.INTERNAL_API_KEY or "").strip()
    if key:
        headers["X-Internal-Key"] = key
    return headers


async def verify_internal_llm_access(request: Request) -> None:
    secret = (settings.TOOL_RUNNER_INTERNAL_SECRET or "").strip()
    if secret:
        got = (request.headers.get("X-Tool-Runner-Internal-Secret") or "").strip()
        if got != secret:
            logger.warning("internal_llm: invalid secret (host=%s)", request.client.host if request.client else "?")
            raise HTTPException(status_code=403, detail="invalid_internal_secret")
        return

    host = (request.client.host if request.client else "") or ""
    if host not in _ALLOWED_LOOPBACK:
        logger.warning("internal_llm: rejected host=%s (set TOOL_RUNNER_INTERNAL_SECRET for remote)", host)
        raise HTTPException(
            status_code=403,
            detail="internal_llm_requires_loopback_or_TOOL_RUNNER_INTERNAL_SECRET",
        )


def _llm_proxy_chat_url() -> str:
    return f"{settings.LLM_PROXY_URL.rstrip('/')}/v1/chat/completions"


def _llm_proxy_embeddings_url() -> str:
    return f"{settings.LLM_PROXY_URL.rstrip('/')}/v1/embeddings"


def _llm_proxy_images_generations_url() -> str:
    return f"{settings.LLM_PROXY_URL.rstrip('/')}/v1/images/generations"


def _llm_proxy_comfy_url(path: str) -> str:
    cleaned = (path or "").lstrip("/")
    return f"{settings.LLM_PROXY_URL.rstrip('/')}/v1/comfy/{cleaned}"


def _llm_proxy_infer_url() -> str:
    return f"{settings.LLM_PROXY_URL.rstrip('/')}/v1/infer"


def _llm_proxy_rerank_url() -> str:
    return f"{settings.LLM_PROXY_URL.rstrip('/')}/rerank"


def _llm_proxy_audio_transcriptions_url() -> str:
    return f"{settings.LLM_PROXY_URL.rstrip('/')}/v1/audio/transcriptions"


def _llm_proxy_audio_speech_url() -> str:
    return f"{settings.LLM_PROXY_URL.rstrip('/')}/v1/audio/speech"


def _llm_proxy_custom_generate_url() -> str:
    return f"{settings.LLM_PROXY_URL.rstrip('/')}/v1/custom/generate"


def _llm_proxy_audio_diarize_url() -> str:
    return f"{settings.LLM_PROXY_URL.rstrip('/')}/v1/speech/diarize"


def _llm_proxy_audio_voiceprints_url() -> str:
    return f"{settings.LLM_PROXY_URL.rstrip('/')}/v1/audio/voiceprints"


def _llm_proxy_catalog_url() -> str:
    return f"{settings.LLM_PROXY_URL.rstrip('/')}/catalog"


def _llm_proxy_binding_resolve_url() -> str:
    return f"{settings.LLM_PROXY_URL.rstrip('/')}/binding/resolve"


async def _forward_json(
    *,
    request: Request,
    upstream_url: str,
    purpose: str,
    timeout: httpx.Timeout,
    log_label: str,
    body: Optional[bytes] = None,
) -> Response:
    if body is None:
        body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty_body")
    body, route_key = await _maybe_rewrite_model(request, body, purpose=purpose)
    headers = _llm_upstream_headers({"Content-Type": "application/json"})
    if route_key:
        headers["X-AIv2-Route-Key"] = route_key
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await upstream_post(
                client,
                upstream_url,
                content=body,
                headers=headers,
                log_label=log_label,
            )
    except httpx.ConnectError as exc:
        logger.error("internal_llm %s: upstream connect failed: %s", log_label, exc)
        raise HTTPException(status_code=502, detail=f"llm_proxy_unreachable: {exc}") from exc
    except httpx.TimeoutException as exc:
        logger.error("internal_llm %s: upstream timeout: %s", log_label, exc)
        raise HTTPException(status_code=504, detail=f"llm_proxy_timeout: {exc}") from exc
    ct = resp.headers.get("content-type", "application/json")
    return Response(content=resp.content, status_code=resp.status_code, media_type=ct)


async def _resolve_form_model(
    request: Request, *, purpose: str, current_model: Optional[str]
) -> Tuple[str, Optional[str]]:
    tool_id = (request.headers.get("X-AIv2-Tool-Id") or "").strip() or None
    plugin_id = (request.headers.get("X-AIv2-Plugin-Id") or "").strip() or None
    route_key_hdr = (request.headers.get("X-AIv2-Route-Key") or "").strip() or None
    explicit = (
        str(current_model).strip()
        if current_model is not None
        and str(current_model).strip().lower() not in _AUTO_TOKENS
        else ""
    )
    # Agent Flow: node-selected model only — never attach a route_key.
    if _caller_picks_node_model(tool_id, plugin_id):
        if explicit:
            return explicit, None
        model, _matched, _binding_constraints = await _resolve_model_for(
            purpose, tool_id, plugin_id
        )
        if model:
            return str(model).strip(), None
        raise HTTPException(
            status_code=503,
            detail={
                "error": "binding_not_configured",
                "tool_id": tool_id,
                "plugin_id": plugin_id,
                "tried_chain": _build_binding_chain(purpose, tool_id, plugin_id),
            },
        )
    # Dual-bind tools: caller already resolved a slot (e.g. .diarize) and pins
    # that logical_id + route_key. Do not overwrite with path-purpose binding.
    if explicit and route_key_hdr:
        logger.info(
            "internal_llm multipart: tool=%s plugin=%s → model=%s (bound slot %s)",
            tool_id, plugin_id, explicit, route_key_hdr,
        )
        return explicit, route_key_hdr
    needs_resolve = bool(tool_id or plugin_id) or not explicit
    matched: Optional[str] = None
    if needs_resolve:
        model, matched, _binding_constraints = await _resolve_model_for(
            purpose, tool_id, plugin_id
        )
        if model:
            logger.info(
                "internal_llm multipart: tool=%s plugin=%s purpose=%s → model=%s (via %s)",
                tool_id, plugin_id, purpose, model, matched,
            )
            return str(model).strip(), matched
        if explicit:
            # No binding configured — honor explicit body model (RLM / courier).
            return explicit, None
        raise HTTPException(
            status_code=503,
            detail={
                "error": "binding_not_configured",
                "tool_id": tool_id,
                "plugin_id": plugin_id,
                "tried_chain": _build_binding_chain(purpose, tool_id, plugin_id),
            },
        )
    return explicit, None


def _body_wants_stream(body: bytes) -> bool:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid_json_body")
    return isinstance(payload, dict) and payload.get("stream") is True


async def _forward_sse(
    *,
    request: Request,
    upstream_url: str,
    purpose: str,
    timeout: httpx.Timeout,
    log_label: str,
    body: bytes,
) -> Response:
    """Proxy an upstream ``text/event-stream`` response without buffering."""
    if not body:
        raise HTTPException(status_code=400, detail="empty_body")
    body, route_key = await _maybe_rewrite_model(request, body, purpose=purpose)
    headers = _llm_upstream_headers({"Content-Type": "application/json", "Accept": "text/event-stream"})
    if route_key:
        headers["X-AIv2-Route-Key"] = route_key

    client = httpx.AsyncClient(timeout=timeout)

    resp: Optional[httpx.Response] = None
    attempts = infra_retry_attempts()
    delay = infra_retry_delay_sec()
    try:
        for attempt in range(attempts):
            req = client.build_request(
                "POST", upstream_url, content=body, headers=headers
            )
            try:
                resp = await client.send(req, stream=True)
                break
            except httpx.ConnectError as exc:
                if attempt + 1 >= attempts:
                    await client.aclose()
                    logger.error(
                        "internal_llm %s: upstream connect failed: %s", log_label, exc
                    )
                    raise HTTPException(
                        status_code=502, detail=f"llm_proxy_unreachable: {exc}"
                    ) from exc
                logger.warning(
                    "internal_llm %s: stream connect infra retry %d/%d",
                    log_label,
                    attempt + 1,
                    attempts,
                )
                await asyncio.sleep(delay)
            except httpx.TimeoutException as exc:
                await client.aclose()
                logger.error("internal_llm %s: upstream timeout: %s", log_label, exc)
                raise HTTPException(
                    status_code=504, detail=f"llm_proxy_timeout: {exc}"
                ) from exc
            except Exception:
                await client.aclose()
                raise
    except HTTPException:
        raise

    if resp is None:
        await client.aclose()
        raise HTTPException(status_code=502, detail="llm_proxy_unreachable")

    if resp.status_code >= 400:
        try:
            err_body = await resp.aread()
        finally:
            await resp.aclose()
            await client.aclose()
        ct = resp.headers.get("content-type", "application/json")
        return Response(content=err_body, status_code=resp.status_code, media_type=ct)

    async def event_gen():
        try:
            async for chunk in resp.aiter_bytes():
                if chunk:
                    yield chunk
        except httpx.TimeoutException as exc:
            logger.error("internal_llm %s: mid-stream timeout: %s", log_label, exc)
            yield (
                b'data: {"error":{"message":"llm_proxy_timeout","type":"timeout"}}\n\n'
                b"data: [DONE]\n\n"
            )
        finally:
            await resp.aclose()
            await client.aclose()

    media = resp.headers.get("content-type") or "text/event-stream"
    return StreamingResponse(
        event_gen(),
        status_code=resp.status_code,
        media_type=media,
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _tool_llm_profile_from_registry(tool_id: Optional[str]) -> Optional[Dict]:
    if not tool_id:
        return None
    try:
        from ..utils.registry_wrapper import get_registry

        wrapper = get_registry()
        adapter = None
        inner = getattr(wrapper, "tool_registry", None)
        if inner is not None:
            getter = getattr(inner, "get_adapter", None) or getattr(inner, "get", None)
            if callable(getter):
                adapter = getter(tool_id)
        if adapter and getattr(adapter, "meta", None):
            from aiv2_model_contract.llm_taxonomy import infer_tool_llm_profile

            return infer_tool_llm_profile(adapter.meta)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "internal_llm: tool profile lookup failed for %s: %s", tool_id, exc
        )
    return None


def _caller_picks_node_model(
    tool_id: Optional[str], plugin_id: Optional[str]
) -> bool:
    """Agent Flow nodes pick any catalog model — no tool/plugin route binding.

    Identity headers are still useful for logging/auth, but they must never
    override ``body.model`` or attach ``X-AIv2-Route-Key``.
    """
    if str(plugin_id or "").strip() == "agent_flow":
        return True
    tid = str(tool_id or "").strip()
    return tid.startswith("plugin_agent_flow")


def _build_binding_chain(
    purpose: str, tool_id: Optional[str], plugin_id: Optional[str]
) -> List[str]:
    """Ordered route keys: modality-first ``.llm`` then legacy ``.<api>``."""
    chain: List[str] = []
    strict_route = purpose == "query_rewrite"
    if tool_id:
        if not strict_route:
            chain.append(f"tool.{tool_id}.llm")
        chain.append(f"tool.{tool_id}.{purpose}")
    if plugin_id:
        if not strict_route:
            chain.append(f"plugin.{plugin_id}.llm")
        chain.append(f"plugin.{plugin_id}.{purpose}")
    return chain


async def _resolve_model_for(
    purpose: str, tool_id: Optional[str], plugin_id: Optional[str]
) -> Tuple[Optional[str], Optional[str], Dict]:
    """Returns (model, matched_key, binding_constraints)."""
    chain = _build_binding_chain(purpose, tool_id, plugin_id)
    url = f"{settings.LLM_PROXY_URL.rstrip('/')}/binding/resolve"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await upstream_get(
                client,
                url,
                params={"keys": ",".join(chain)},
                headers=_llm_upstream_headers(),
                log_label="binding-resolve",
            )
        if resp.status_code == 200:
            data = resp.json() or {}
            constraints = data.get("constraints") if isinstance(data.get("constraints"), dict) else {}
            return (data.get("model") or None), data.get("route_key"), constraints
        logger.warning(
            "internal_llm: binding resolve HTTP %s (%s): %s",
            resp.status_code,
            chain,
            (resp.text or "")[:200],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("internal_llm: binding resolve failed (%s): %s", chain, exc)
    return None, None, {}


async def _maybe_rewrite_model(
    request: Request, body: bytes, *, purpose: str
) -> Tuple[bytes, Optional[str]]:
    """Resolve ``auto`` via tool/plugin binding; keep explicit models when unbound.

    When a binding exists for the caller, it wins over ``body.model`` (tool
    authority) — except Agent Flow, where each Agent/Model node picks any
    catalog model and must never be overridden or pinned by a route_key.
    """
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body, None
    if not isinstance(payload, dict):
        return body, None

    tool_id = (request.headers.get("X-AIv2-Tool-Id") or "").strip() or None
    plugin_id = (request.headers.get("X-AIv2-Plugin-Id") or "").strip() or None
    route_key_hdr = (request.headers.get("X-AIv2-Route-Key") or "").strip() or None

    current_model = payload.get("model")
    explicit = (
        str(current_model).strip()
        if current_model is not None
        and str(current_model).strip().lower() not in _AUTO_TOKENS
        else ""
    )
    # Agent Flow: always honor the node-selected logical id; never attach a
    # route_key (missing keys empty the backend list in llm-proxy).
    if _caller_picks_node_model(tool_id, plugin_id):
        payload.pop("route_key", None)
        if explicit:
            payload["model"] = explicit
            logger.info(
                "internal_llm: agent_flow tool=%s → model=%s (node selection)",
                tool_id,
                explicit,
            )
            return json.dumps(payload).encode("utf-8"), None
        model, _matched, _binding_constraints = await _resolve_model_for(
            purpose, tool_id, plugin_id
        )
        if model:
            payload["model"] = model
            logger.info(
                "internal_llm: agent_flow tool=%s purpose=%s → model=%s "
                "(auto fill, no route_key)",
                tool_id,
                purpose,
                model,
            )
            return json.dumps(payload).encode("utf-8"), None
        tool_profile = _tool_llm_profile_from_registry(tool_id)
        prof = merge_llm_profiles(
            profile_from_gateway_purpose(purpose), tool_profile
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "binding_not_configured",
                "message": (
                    "Agent Flow node has no model selected and no default "
                    f"binding for tool '{tool_id or '?'}'."
                ),
                "tool_id": tool_id,
                "plugin_id": plugin_id,
                "suggested_profile": prof,
                "tried_chain": _build_binding_chain(purpose, tool_id, plugin_id),
            },
        )

    # Dual-bind / pre-resolved slot: honor explicit logical_id + route_key.
    if explicit and route_key_hdr:
        payload.pop("route_key", None)
        payload["model"] = explicit
        logger.info(
            "internal_llm: tool=%s plugin=%s → model=%s (bound slot %s)",
            tool_id, plugin_id, explicit, route_key_hdr,
        )
        return json.dumps(payload).encode("utf-8"), route_key_hdr

    # Attempt binding whenever a tool/plugin identity is present (authority),
    # or when the body asked for auto/default.
    needs_resolve = bool(tool_id or plugin_id) or not explicit

    tool_profile = _tool_llm_profile_from_registry(tool_id)
    matched: Optional[str] = None

    if needs_resolve:
        model, matched, _binding_constraints = await _resolve_model_for(
            purpose, tool_id, plugin_id
        )
        if model:
            payload["model"] = model
            logger.info(
                "internal_llm: tool=%s plugin=%s purpose=%s → model=%s (via %s)",
                tool_id, plugin_id, purpose, model, matched,
            )
        elif explicit:
            # Unbound tool/plugin with an explicit model (e.g. RLM session model).
            payload["model"] = explicit
            matched = None
            logger.info(
                "internal_llm: tool=%s plugin=%s purpose=%s → model=%s (explicit, no binding)",
                tool_id, plugin_id, purpose, explicit,
            )
        else:
            prof = merge_llm_profiles(
                profile_from_gateway_purpose(purpose), tool_profile
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "binding_not_configured",
                    "message": (
                        f"No LLM binding for tool '{tool_id or '?'}' "
                        f"(profile in={prof['input_modalities']} "
                        f"out={prof['output_modalities']}). Configure via "
                        f"PUT /admin/tools/{tool_id or '<tool_id>'}/binding"
                    ),
                    "tool_id": tool_id,
                    "plugin_id": plugin_id,
                    "suggested_profile": prof,
                    "tried_chain": _build_binding_chain(purpose, tool_id, plugin_id),
                },
            )
    else:
        payload["model"] = explicit

    return json.dumps(payload).encode("utf-8"), matched


@router.post("/chat/completions")
async def forward_chat_completions(
    request: Request,
    _: None = Depends(verify_internal_llm_access),
) -> Response:
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty_body")
    timeout = httpx.Timeout(settings.LLM_FORWARD_TIMEOUT_SEC, connect=20.0)
    if _body_wants_stream(body):
        return await _forward_sse(
            request=request,
            upstream_url=_llm_proxy_chat_url(),
            purpose="chat",
            timeout=timeout,
            log_label="chat-stream",
            body=body,
        )
    return await _forward_json(
        request=request,
        upstream_url=_llm_proxy_chat_url(),
        purpose="chat",
        timeout=timeout,
        log_label="chat",
        body=body,
    )


@router.post("/query_rewrite")
async def forward_query_rewrite(
    request: Request,
    _: None = Depends(verify_internal_llm_access),
) -> Response:
    """Chat-compatible generation with an authoritative query_rewrite route."""
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty_body")
    if _body_wants_stream(body):
        raise HTTPException(status_code=400, detail="query_rewrite_stream_not_supported")
    return await _forward_json(
        request=request,
        upstream_url=_llm_proxy_chat_url(),
        purpose="query_rewrite",
        timeout=httpx.Timeout(settings.LLM_FORWARD_TIMEOUT_SEC, connect=20.0),
        log_label="query-rewrite",
        body=body,
    )


@router.post("/embeddings")
async def forward_embeddings(
    request: Request,
    _: None = Depends(verify_internal_llm_access),
) -> Response:
    return await _forward_json(
        request=request,
        upstream_url=_llm_proxy_embeddings_url(),
        purpose="embed",
        timeout=httpx.Timeout(settings.LLM_EMBED_FORWARD_TIMEOUT_SEC),
        log_label="embeddings",
    )


@router.post("/images/generations")
async def forward_images_generations(
    request: Request,
    _: None = Depends(verify_internal_llm_access),
) -> Response:
    return await _forward_json(
        request=request,
        upstream_url=_llm_proxy_images_generations_url(),
        purpose="image",
        timeout=httpx.Timeout(settings.LLM_IMAGE_FORWARD_TIMEOUT_SEC),
        log_label="images",
    )


@router.post("/infer")
async def forward_infer(
    request: Request,
    _: None = Depends(verify_internal_llm_access),
) -> Response:
    """Triton / OpenAI-tensor infer — tools must not call llm-proxy directly."""
    return await _forward_json(
        request=request,
        upstream_url=_llm_proxy_infer_url(),
        purpose="infer",
        timeout=httpx.Timeout(settings.LLM_FORWARD_TIMEOUT_SEC),
        log_label="infer",
    )


@router.post("/rerank")
async def forward_rerank(
    request: Request,
    _: None = Depends(verify_internal_llm_access),
) -> Response:
    return await _forward_json(
        request=request,
        upstream_url=_llm_proxy_rerank_url(),
        purpose="rerank",
        timeout=httpx.Timeout(settings.LLM_EMBED_FORWARD_TIMEOUT_SEC),
        log_label="rerank",
    )


@router.post("/audio/transcriptions")
async def forward_audio_transcriptions(
    request: Request,
    _: None = Depends(verify_internal_llm_access),
) -> Response:
    return await _forward_audio_multipart(
        request,
        purpose="asr",
        upstream_url=_llm_proxy_audio_transcriptions_url(),
        log_label="audio-transcriptions",
    )


@router.post("/audio/speech")
async def forward_audio_speech(
    request: Request,
    _: None = Depends(verify_internal_llm_access),
) -> Response:
    """OpenAI-compatible TTS — JSON body, often binary audio response."""
    return await _forward_json(
        request=request,
        upstream_url=_llm_proxy_audio_speech_url(),
        purpose="tts",
        timeout=httpx.Timeout(settings.LLM_FORWARD_TIMEOUT_SEC),
        log_label="tts",
    )


@router.post("/custom/generate")
async def forward_custom_generate(
    request: Request,
    _: None = Depends(verify_internal_llm_access),
) -> Response:
    """custom-library courier — llm-proxy resolves catalog endpoint_path."""
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type.lower():
        return await _forward_multipart_custom(request)
    return await _forward_json(
        request=request,
        upstream_url=_llm_proxy_custom_generate_url(),
        purpose="custom",
        timeout=httpx.Timeout(settings.LLM_FORWARD_TIMEOUT_SEC),
        log_label="custom",
    )


async def _forward_multipart_custom(request: Request) -> Response:
    """Pass file-in catalog uploads through as the original multipart bytes.

    Do not call ``request.form()``: Starlette needs python-multipart at import
    time, and re-encoding would drop the worker's ``File(...)`` part.
    Model identity travels on ``X-AIv2-Model`` (set by the courier).
    """
    body = await request.body()
    content_type = request.headers.get("content-type") or ""
    model_hdr = (request.headers.get("X-AIv2-Model") or "").strip() or None
    model, route_key = await _resolve_form_model(
        request, purpose="custom", current_model=model_hdr
    )
    headers: Dict[str, str] = _llm_upstream_headers({
        "content-type": content_type,
        "X-AIv2-Model": model,
    })
    for name in (
        "X-AIv2-Endpoint-Path",
        "X-AIv2-Dispatch-Key",
        "X-AIv2-Task",  # legacy read alias for one release
        "X-User-Id",
        "X-Case-Id",
    ):
        val = (request.headers.get(name) or "").strip()
        if val:
            headers[name] = val
    # Prefer Dispatch-Key; mirror Task → Dispatch-Key when only legacy header present.
    if not headers.get("X-AIv2-Dispatch-Key") and headers.get("X-AIv2-Task"):
        headers["X-AIv2-Dispatch-Key"] = headers["X-AIv2-Task"]
    headers.pop("X-AIv2-Task", None)
    if route_key:
        headers["X-AIv2-Route-Key"] = route_key
    timeout = httpx.Timeout(settings.LLM_FORWARD_TIMEOUT_SEC, connect=20.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await upstream_post(
                client,
                _llm_proxy_custom_generate_url(),
                content=body,
                headers=headers,
                log_label="custom-multipart",
            )
    except httpx.ConnectError as exc:
        logger.error("internal_llm custom multipart: upstream connect failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"llm_proxy_unreachable: {exc}") from exc
    except httpx.TimeoutException as exc:
        logger.error("internal_llm custom multipart: upstream timeout: %s", exc)
        raise HTTPException(status_code=504, detail=f"llm_proxy_timeout: {exc}") from exc
    ct = resp.headers.get("content-type", "application/json")
    return Response(content=resp.content, status_code=resp.status_code, media_type=ct)


async def _forward_audio_multipart(
    request: Request,
    *,
    purpose: str,
    upstream_url: str,
    log_label: str,
) -> Response:
    form = await request.form()
    model, route_key = await _resolve_form_model(
        request,
        purpose=purpose,
        current_model=str(form.get("model") or "") or None,
    )
    data = {k: str(v) for k, v in form.items() if k != "file" and not hasattr(v, "read")}
    data["model"] = model
    upload = form.get("file")
    files = None
    if upload is not None and hasattr(upload, "read"):
        content = await upload.read()
        filename = getattr(upload, "filename", None) or "audio"
        content_type = getattr(upload, "content_type", None) or "application/octet-stream"
        files = {"file": (filename, content, content_type)}
    timeout = httpx.Timeout(settings.LLM_FORWARD_TIMEOUT_SEC, connect=20.0)
    headers: Dict[str, str] = _llm_upstream_headers()
    if route_key:
        headers["X-AIv2-Route-Key"] = route_key
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await upstream_post(
                client,
                upstream_url,
                data=data,
                files=files,
                headers=headers or None,
                log_label=log_label,
            )
    except httpx.ConnectError as exc:
        raise HTTPException(status_code=502, detail=f"llm_proxy_unreachable: {exc}") from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail=f"llm_proxy_timeout: {exc}") from exc
    ct = resp.headers.get("content-type", "application/json")
    return Response(content=resp.content, status_code=resp.status_code, media_type=ct)


@router.post("/audio/diarize")
async def forward_audio_diarize(
    request: Request,
    _: None = Depends(verify_internal_llm_access),
) -> Response:
    return await _forward_audio_multipart(
        request,
        purpose="diarize",
        upstream_url=_llm_proxy_audio_diarize_url(),
        log_label="audio-diarize",
    )


@router.post("/audio/voiceprints")
async def forward_audio_voiceprints(
    request: Request,
    _: None = Depends(verify_internal_llm_access),
) -> Response:
    return await _forward_audio_multipart(
        request,
        purpose="voiceprint",
        upstream_url=_llm_proxy_audio_voiceprints_url(),
        log_label="audio-voiceprints",
    )


@router.get("/catalog")
async def forward_catalog(
    request: Request,
    _: None = Depends(verify_internal_llm_access),
) -> Response:
    """Logical model catalog — plugins/tools must not call llm-proxy /catalog."""
    timeout = httpx.Timeout(30.0, connect=10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await upstream_get(
                client,
                _llm_proxy_catalog_url(),
                params=dict(request.query_params),
                headers=_llm_upstream_headers(),
                log_label="catalog",
            )
    except httpx.ConnectError as exc:
        raise HTTPException(status_code=502, detail=f"llm_proxy_unreachable: {exc}") from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail=f"llm_proxy_timeout: {exc}") from exc
    ct = resp.headers.get("content-type", "application/json")
    return Response(content=resp.content, status_code=resp.status_code, media_type=ct)


@router.get("/binding/resolve")
async def forward_binding_resolve(
    request: Request,
    _: None = Depends(verify_internal_llm_access),
) -> Response:
    """Resolve route_key chain → logical_id (binding only; LB still assigns backend)."""
    timeout = httpx.Timeout(10.0, connect=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await upstream_get(
                client,
                _llm_proxy_binding_resolve_url(),
                params=dict(request.query_params),
                headers=_llm_upstream_headers(),
                log_label="binding-resolve",
            )
    except httpx.ConnectError as exc:
        raise HTTPException(status_code=502, detail=f"llm_proxy_unreachable: {exc}") from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail=f"llm_proxy_timeout: {exc}") from exc
    ct = resp.headers.get("content-type", "application/json")
    return Response(content=resp.content, status_code=resp.status_code, media_type=ct)


@router.api_route("/comfy/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def forward_comfy(
    path: str,
    request: Request,
    _: None = Depends(verify_internal_llm_access),
) -> Response:
    """Forward ComfyUI calls by logical_id ``model`` — never by node IP."""
    from urllib.parse import urlencode

    model = str(request.query_params.get("model") or "").strip()
    raw = b""
    if request.method.upper() in ("POST", "PUT", "PATCH"):
        raw = await request.body()
        if not model and raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    model = str(parsed.get("model") or "").strip()
            except json.JSONDecodeError:
                pass
    if not model:
        model = str(request.headers.get("x-model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model_required")

    qs = dict(request.query_params)
    qs["model"] = model
    url = _llm_proxy_comfy_url(path)
    query = urlencode(qs, doseq=True)
    if query:
        url = f"{url}?{query}"

    timeout = httpx.Timeout(settings.LLM_IMAGE_FORWARD_TIMEOUT_SEC, connect=20.0)
    headers = _llm_upstream_headers({"Content-Type": request.headers.get("content-type") or "application/json"})
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(
                request.method.upper(),
                url,
                content=raw or None,
                headers=headers,
            )
    except httpx.ConnectError as exc:
        logger.error("internal_llm comfy: upstream connect failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"llm_proxy_unreachable: {exc}") from exc
    except httpx.TimeoutException as exc:
        logger.error("internal_llm comfy: upstream timeout: %s", exc)
        raise HTTPException(status_code=504, detail=f"llm_proxy_timeout: {exc}") from exc

    ct = resp.headers.get("content-type", "application/json")
    return Response(content=resp.content, status_code=resp.status_code, media_type=ct)
