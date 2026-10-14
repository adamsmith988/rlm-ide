"""LLM calls for controller roles (sampler, summarizer, judge, proposer, …).

Two backends behind one signature:

* ``gateway_llm`` — the orchestrator's binding-resolved gateway client (hub chat).
* ``host_chat_llm`` — the ``chat`` HostPort of the orchestrator contract.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


@dataclass
class LLMReply:
    text: str
    reasoning: str = ""
    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    ok: bool = True
    error: str = ""
    finish_reason: str = ""

    @property
    def tokens(self) -> Optional[int]:
        if self.tokens_prompt is None and self.tokens_completion is None:
            return None
        return int(self.tokens_prompt or 0) + int(self.tokens_completion or 0)


LLMCall = Callable[..., Awaitable[LLMReply]]


def strip_reasoning(text: str) -> str:
    """Drop ``<think>…</think>``; keep what follows a dangling ``</think>``."""
    out = _THINK_BLOCK.sub("", str(text or ""))
    if "</think>" in out:
        out = out.split("</think>")[-1]
    return out.strip()


def extract_json(text: str) -> Any:
    """First JSON object/array in an LLM reply (fences and reasoning tolerated)."""
    body = strip_reasoning(text)
    candidates: List[str] = [m.group(1) for m in _FENCE.finditer(body)] + [body]
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            start = cand.find(opener)
            end = cand.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(cand[start : end + 1])
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"no JSON found in reply: {body[:200]!r}")


def _thinking_body(enable_thinking: Optional[bool]) -> Optional[Dict[str, Any]]:
    if enable_thinking is None:
        return None
    return {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}


def _sampling_body(extra: Optional[Dict[str, Any]], enable_thinking: Optional[bool]) -> Optional[Dict[str, Any]]:
    body: Dict[str, Any] = dict(extra or {})
    think = _thinking_body(enable_thinking)
    if think:
        body.update(think)
    return body or None


def gateway_llm(client: Any = None) -> LLMCall:
    async def call(
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        sampling: Optional[Dict[str, Any]] = None,
    ) -> LLMReply:
        from src.llm.gateway_llm_client import get_gateway_llm_client

        llm = client or get_gateway_llm_client()
        res = await llm.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens or None,
            extra_body=_sampling_body(sampling, enable_thinking),
        )
        if not isinstance(res, dict) or not res.get("ok", False):
            err = str((res or {}).get("error") or "llm_error") if isinstance(res, dict) else "llm_error"
            return LLMReply(text="", ok=False, error=err)
        return LLMReply(
            text=strip_reasoning(res.get("content") or ""),
            reasoning=str(res.get("reasoning_content") or ""),
            tokens_prompt=res.get("tokens_prompt"),
            tokens_completion=res.get("tokens_completion"),
            finish_reason=str(res.get("finish_reason") or ""),
        )

    return call


async def gateway_llm_stream(
    messages: List[Dict[str, Any]],
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
    sampling: Optional[Dict[str, Any]] = None,
    client: Any = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Hub-shaped events (thinking*/answer*/done/error) from the gateway stream."""
    from src.llm.gateway_llm_client import get_gateway_llm_client

    llm = client or get_gateway_llm_client()
    async for ev in llm.chat_stream(
        messages,
        temperature=temperature,
        max_tokens=max_tokens or None,
        extra_body=_sampling_body(sampling, enable_thinking),
    ):
        yield ev


def host_chat_llm(chat_fn: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]) -> LLMCall:
    """Adapt the contract's OpenAI-shaped ``chat`` HostPort."""

    async def call(
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        sampling: Optional[Dict[str, Any]] = None,
    ) -> LLMReply:
        body: Dict[str, Any] = {"messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens:
            body["max_tokens"] = max_tokens
        extra = _sampling_body(sampling, enable_thinking)
        if extra:
            body["extra_body"] = extra
        try:
            resp = await chat_fn(body)
        except Exception as exc:  # noqa: BLE001 — surface as a failed reply
            return LLMReply(text="", ok=False, error=f"{type(exc).__name__}: {exc}")
        msg: Dict[str, Any] = {}
        usage: Dict[str, Any] = {}
        if isinstance(resp, dict):
            choices = resp.get("choices") or []
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message") or {}
            elif isinstance(resp.get("message"), dict):
                msg = resp["message"]
            else:
                msg = {"content": resp.get("content") or resp.get("text") or ""}
            usage = resp.get("usage") or {}
        return LLMReply(
            text=strip_reasoning(str(msg.get("content") or "")),
            reasoning=str(msg.get("reasoning_content") or msg.get("reasoning") or ""),
            tokens_prompt=usage.get("prompt_tokens"),
            tokens_completion=usage.get("completion_tokens"),
        )

    return call
