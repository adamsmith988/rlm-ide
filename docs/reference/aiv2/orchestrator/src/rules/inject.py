"""Inject applicable markdown rules into OpenAI-style message lists."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..config import settings
from .engine import get_applicable_rules, merge_applicable_rules_into_chat_messages, normalize_rule_policies
from .loader import get_rule_directory_roots, load_markdown_rules
from .types import RuleMetadata

log = logging.getLogger(__name__)


def inject_rules_into_openai_messages(
    messages: List[Dict[str, Any]],
    *,
    user_message: str,
    file_context: Optional[Dict[str, Any]] = None,
    file_registry: Optional[List[Dict[str, Any]]] = None,
    rule_policies: Any = None,
) -> Tuple[List[Dict[str, Any]], List[RuleMetadata]]:
    """
    Load rules, compute applicability for ``user_message`` + file context, merge into
    the first ``role: system`` message. Returns a shallow-copied message list when rules apply.
    """
    if not settings.RULES_ENABLED or not get_rule_directory_roots():
        return messages, []
    try:
        available = load_markdown_rules()
        if not available:
            return messages, []
        applicable = get_applicable_rules(
            user_message=user_message or "",
            available_rules=available,
            file_context=file_context,
            file_registry=file_registry,
            rule_policies=normalize_rule_policies(rule_policies),
        )
        if not applicable:
            return messages, []
        return merge_applicable_rules_into_chat_messages(messages, applicable)
    except Exception as e:
        log.warning("inject_rules_into_openai_messages failed: %s", e)
        return messages, []
