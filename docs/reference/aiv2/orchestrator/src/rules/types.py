"""Types for Continue-compatible markdown rules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

RulePolicy = Literal["on", "off"]
RulePolicies = Dict[str, RulePolicy]

Globs = Union[str, List[str]]
RegexPattern = Union[str, List[str]]


@dataclass
class MarkdownRule:
    """One rule loaded from a ``.md`` file (body + frontmatter)."""

    name: str
    rule: str  # markdown body injected into system prompt
    source_file: str
    globs: Optional[Globs] = None
    regex: Optional[RegexPattern] = None
    description: Optional[str] = None
    always_apply: Optional[bool] = None
    invokable: Optional[bool] = None
    # True if file is a direct child of a configured rules root dir (like AGENTS.md).
    is_rules_directory_root_file: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleMetadata:
    """Subset of rule fields for logging / clients (no full body)."""

    name: str
    source_file: str
    description: Optional[str] = None
    slug: Optional[str] = None
