"""LLM client package — OpenAI-compat client for in-cluster llm-gateway."""

__all__ = ["GatewayLLMClient", "get_gateway_llm_client"]


def __getattr__(name: str):
    if name in {"GatewayLLMClient", "get_gateway_llm_client"}:
        from .gateway_llm_client import GatewayLLMClient, get_gateway_llm_client

        return {
            "GatewayLLMClient": GatewayLLMClient,
            "get_gateway_llm_client": get_gateway_llm_client,
        }[name]
    raise AttributeError(name)
