"""Continue-style markdown rules (frontmatter, globs, regex, per-rule on/off)."""
from __future__ import annotations

from .engine import (
    get_applicable_rules,
    merge_applicable_rules_into_chat_messages,
    merge_rules_into_first_system_message,
    normalize_rule_policies,
    should_apply_rule,
)
from .inject import inject_rules_into_openai_messages
from .loader import get_rule_directory_roots, load_markdown_rules
from .types import MarkdownRule, RuleMetadata, RulePolicies

__all__ = [
    "MarkdownRule",
    "RuleMetadata",
    "RulePolicies",
    "get_applicable_rules",
    "get_rule_directory_roots",
    "inject_rules_into_openai_messages",
    "load_markdown_rules",
    "merge_applicable_rules_into_chat_messages",
    "merge_rules_into_first_system_message",
    "normalize_rule_policies",
    "should_apply_rule",
]
