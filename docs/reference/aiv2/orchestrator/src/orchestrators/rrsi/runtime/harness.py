"""The evolvable harness H: a JSON document validated against the domain schema.

Edits are ``changes`` (set / append / remove on a schema field); the diff d' is
rendered one ``- field = <json>`` / ``+ field = <json>`` pair per changed field
so the vendored component classifier can attribute it.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .domain import ChatDomain

Harness = Dict[str, Any]


def default_harness(domain: ChatDomain) -> Harness:
    return {name: copy.deepcopy(spec.get("default")) for name, spec in domain.fields.items()}


def normalize_harness(domain: ChatDomain, raw: Optional[Mapping[str, Any]]) -> Harness:
    out = default_harness(domain)
    for name, value in dict(raw or {}).items():
        if name in out:
            out[name] = copy.deepcopy(value)
    return out


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def render_diff(old: Harness, new: Harness, fields: Optional[Iterable[str]] = None) -> str:
    names = list(fields) if fields is not None else sorted(set(old) | set(new))
    lines: List[str] = []
    for name in names:
        a, b = old.get(name), new.get(name)
        if _dump(a) == _dump(b):
            continue
        lines.append(f"- {name} = {_dump(a)}")
        lines.append(f"+ {name} = {_dump(b)}")
    return "\n".join(lines)


def changed_fields(old: Harness, new: Harness) -> List[str]:
    return [n for n in sorted(set(old) | set(new)) if _dump(old.get(n)) != _dump(new.get(n))]


def added_text(old: Harness, new: Harness) -> str:
    """All text the candidate introduces (for leakage screening)."""
    parts: List[str] = []
    for name in changed_fields(old, new):
        parts.append(_dump(new.get(name)))
    return "\n".join(parts)


def apply_changes(
    domain: ChatDomain, harness: Harness, changes: Sequence[Mapping[str, Any]]
) -> Tuple[Harness, List[str]]:
    out = copy.deepcopy(harness)
    problems: List[str] = []
    for ch in changes or []:
        if not isinstance(ch, Mapping):
            problems.append(f"change is not an object: {ch!r}"[:200])
            continue
        name = str(ch.get("field") or "")
        op = str(ch.get("op") or "set").lower()
        spec = domain.fields.get(name)
        if spec is None:
            problems.append(f"unknown harness field {name!r}")
            continue
        value = ch.get("value")
        if op == "set":
            out[name] = copy.deepcopy(value)
        elif op in ("append", "remove"):
            if spec.get("type") != "array":
                problems.append(f"{op} only applies to array fields; {name!r} is {spec.get('type')}")
                continue
            cur = list(out.get(name) or [])
            if op == "append":
                cur.extend(copy.deepcopy(value) if isinstance(value, list) else [copy.deepcopy(value)])
            else:
                drop = value if isinstance(value, list) else [value]
                cur = [x for x in cur if _dump(x) not in {_dump(d) for d in drop}]
            out[name] = cur
        else:
            problems.append(f"unknown op {op!r} on {name!r}")
    return out, problems


def validate(
    domain: ChatDomain,
    harness: Harness,
    *,
    allowed_engine_keys: Optional[Set[str]],
    frozen_engine_keys: Set[str],
    known_tool_ids: Optional[Set[str]],
) -> List[str]:
    """Smoke check: the candidate must be a well-formed, runnable harness."""
    problems: List[str] = []
    for name, spec in domain.fields.items():
        value = harness.get(name)
        typ = spec.get("type")
        if value is None:
            if spec.get("default") is not None:
                problems.append(f"{name} must not be null")
            continue
        if typ == "string":
            if not isinstance(value, str):
                problems.append(f"{name} must be a string")
            elif spec.get("max_chars") and len(value) > int(spec["max_chars"]):
                problems.append(f"{name} exceeds {spec['max_chars']} chars")
        elif typ == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                problems.append(f"{name} must be an integer or null")
            else:
                if "min" in spec and value < spec["min"]:
                    problems.append(f"{name} < {spec['min']}")
                if "max" in spec and value > spec["max"]:
                    problems.append(f"{name} > {spec['max']}")
        elif typ == "boolean":
            if not isinstance(value, bool):
                problems.append(f"{name} must be a boolean")
        elif typ == "object":
            if not isinstance(value, dict):
                problems.append(f"{name} must be an object")
                continue
            keys = spec.get("item_keys")
            if keys:
                extra = sorted(set(value) - set(keys))
                if extra:
                    problems.append(f"{name} has unknown keys {extra}; allowed {keys}")
        elif typ == "array":
            if not isinstance(value, list):
                problems.append(f"{name} must be an array")
                continue
            if spec.get("max_items") and len(value) > int(spec["max_items"]):
                problems.append(f"{name} has more than {spec['max_items']} items")
            for item in value:
                if spec.get("items") == "string" and not isinstance(item, str):
                    problems.append(f"{name} items must be strings")
                    break
                if spec.get("items") == "object":
                    if not isinstance(item, dict):
                        problems.append(f"{name} items must be objects")
                        break
                    missing = [k for k in spec.get("item_keys") or [] if not str(item.get(k) or "").strip()]
                    if missing:
                        problems.append(f"{name} item missing {missing}")
                        break
                if spec.get("max_chars_each") and len(_dump(item)) > int(spec["max_chars_each"]):
                    problems.append(f"{name} item exceeds {spec['max_chars_each']} chars")
                    break

    cfg = harness.get("engine_config")
    if isinstance(cfg, dict) and cfg:
        frozen = sorted(k for k in cfg if k in frozen_engine_keys)
        if frozen:
            problems.append(f"engine_config keys are frozen: {frozen}")
        if allowed_engine_keys is not None:
            unknown = sorted(k for k in cfg if k not in allowed_engine_keys and k not in frozen_engine_keys)
            if unknown:
                problems.append(
                    f"engine_config keys not declared by the policy engine: {unknown}; "
                    f"allowed {sorted(allowed_engine_keys - frozen_engine_keys)}"
                )
    tools = harness.get("tool_ids")
    if isinstance(tools, list) and tools:
        if known_tool_ids is None:
            problems.append("tool catalog unavailable; tool_ids cannot be validated")
        else:
            unknown_tools = sorted(t for t in tools if t not in known_tool_ids)
            if unknown_tools:
                problems.append(f"unknown tool ids {unknown_tools}")
    return problems


def prompt_block(domain: ChatDomain, harness: Harness) -> str:
    spec = domain.schema.get("prompt_block") or {}
    sections = spec.get("sections") or {}
    parts: List[str] = []
    for name, heading in sections.items():
        value = harness.get(name)
        if not value:
            continue
        if isinstance(value, str):
            body = value.strip()
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            body = "\n".join(
                "- " + " | ".join(f"{k}: {str(item.get(k, '')).strip()}" for k in item)
                for item in value
            )
        elif isinstance(value, list):
            body = "\n".join(f"- {str(x).strip()}" for x in value)
        else:
            body = _dump(value)
        if body:
            parts.append(f"{heading}\n{body}")
    if not parts:
        return ""
    return "\n\n".join([str(spec.get("header") or "").strip(), *parts]).strip()


def trim_history(history: List[Dict[str, Any]], turns: Optional[int]) -> List[Dict[str, Any]]:
    if turns is None:
        return list(history)
    if turns <= 0:
        return []
    user_idx = [i for i, m in enumerate(history) if isinstance(m, dict) and m.get("role") == "user"]
    if len(user_idx) <= turns:
        return list(history)
    return list(history[user_idx[-turns]:])
