"""Extract file-like paths from user text (code fences) — Continue-style."""
from __future__ import annotations

import re
from typing import List, Set


def extract_paths_from_code_blocks(content: str) -> List[str]:
    """Rough port of ``extractPathsFromCodeBlocks`` (Continue)."""
    if not content:
        return []
    paths: List[str] = []
    seen: Set[str] = set()
    for m in re.finditer(r"```[^\n]+", content):
        block_start = m.group(0)
        # After opening fence: ``lang`` or ``lang path/to/file.ext`` or ``file.ext``
        inner = block_start[3:].strip()
        if not inner:
            continue
        for token in inner.split():
            filename = token.strip(",").strip()
            if "://" in filename:
                continue
            if not re.search(r"\.[a-zA-Z0-9]+$", filename):
                continue
            if filename not in seen:
                seen.add(filename)
                paths.append(filename)
    return paths


def extract_content_for_path(content: str, rel_path: str) -> str:
    """Best-effort: return fenced block that mentions ``rel_path`` in opener."""
    if not content or not rel_path:
        return ""
    lines = content.split("\n")
    in_block = False
    opener_has_path = False
    buf: List[str] = []
    for line in lines:
        if line.startswith("```"):
            if in_block:
                if opener_has_path:
                    return "\n".join(buf)
                in_block = False
                buf = []
                opener_has_path = False
            else:
                in_block = True
                opener_has_path = rel_path in line or rel_path.split("/")[-1] in line
                buf = []
            continue
        if in_block:
            buf.append(line)
    return ""
