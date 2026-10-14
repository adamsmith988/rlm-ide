"""
Helpers for case-importer: reconstruct tool channel payloads from persisted rows.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Legacy dot-notation ids still present in Mongo tool_name fields.
_LEGACY_TOOL_ID_ALIASES: Dict[str, str] = {
    "file.identify": "file_identify",
    "file.meta": "file_meta",
    "file.secure": "file_secure",
    "image.meta": "image_meta",
    "image.faces": "image_faces",
    "graph.extract": "graph_extract",
    "audio.transcribe": "audio_transcribe",
    "text.summarize": "text_summarize",
}


def tool_id_lookup_variants(tool_id: str) -> List[str]:
    """All tool_name / table lookup ids to try for one logical tool."""
    tid = (tool_id or "").strip()
    out: List[str] = []

    def add(val: Optional[str]) -> None:
        if val and val not in out:
            out.append(val)

    add(tid)
    add(_LEGACY_TOOL_ID_ALIASES.get(tid))
    for legacy, canonical in _LEGACY_TOOL_ID_ALIASES.items():
        if tid == canonical:
            add(legacy)
    if "." in tid and not tid.startswith("plugin."):
        add(tid.replace(".", "_"))
    elif "_" in tid and not tid.startswith("plugin_"):
        head, tail = tid.split("_", 1)
        if head in {"file", "image", "audio", "text", "graph", "embed", "vision"}:
            add(f"{head}.{tail}")
    if tid.startswith("plugin."):
        add("plugin_" + tid[len("plugin."):].replace(".", "_"))
    elif tid.startswith("plugin_"):
        add("plugin." + tid[len("plugin_"):].replace("_", "."))
    return out


async def mongo_find_tool_documents(db, artifact_id: str, tool_id: str) -> List[Dict[str, Any]]:
    """All tool_outputs documents for one artifact + tool (alias-aware)."""
    if db is None:
        return []
    for tid in tool_id_lookup_variants(tool_id):
        cursor = db.tool_outputs.find({"artifact_id": artifact_id, "tool_name": tid})
        docs = await cursor.to_list(length=1000)
        if docs:
            out: List[Dict[str, Any]] = []
            for d in docs:
                item = dict(d)
                item.pop("_id", None)
                out.append(item)
            return out
    return []


def max_stored_tool_version(docs: List[Dict[str, Any]], pg_rows: List[Dict[str, Any]]) -> Optional[str]:
    """Best-effort max version string from mongo docs and PG metadata rows."""
    versions: List[str] = []
    for doc in docs:
        v = doc.get("tool_version")
        if v:
            versions.append(str(v))
        data = doc.get("data")
        if isinstance(data, dict) and data.get("_ci_tool_version"):
            versions.append(str(data["_ci_tool_version"]))
    for row in pg_rows:
        data = row.get("data")
        if isinstance(data, dict) and data.get("_ci_tool_version"):
            versions.append(str(data["_ci_tool_version"]))
    if not versions:
        return None
    return max(versions, key=_version_sort_key)


def _version_sort_key(v: str) -> tuple:
    parts: List[Any] = []
    for p in str(v).strip().lower().lstrip("v").split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(p)
    return tuple(parts)


def channels_dict_from_storage(
    mongo_docs: List[Dict[str, Any]],
    pg_metadata_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a `result["channels"]`-shaped dict for case-importer ToolOutputChannels.from_result.
    """
    channels: Dict[str, Any] = {}
    rawdata: List[Dict[str, Any]] = []
    embed_items: List[Dict[str, Any]] = []

    for doc in mongo_docs:
        ch = doc.get("channel")
        data = doc.get("data")
        if ch == "rawdata":
            row: Dict[str, Any] = {}
            if isinstance(data, dict):
                row.update(data)
            elif data is not None:
                row["data"] = data
            row.setdefault("type", doc.get("item_type", "default"))
            row.setdefault("index", doc.get("item_index", 0))
            rawdata.append(row)
        elif ch == "llm" and isinstance(data, dict):
            channels["llm"] = data
        elif ch == "ui" and isinstance(data, dict):
            channels["ui"] = data
        elif ch == "embed":
            if isinstance(data, dict):
                embed_items.append(data)
            elif data is not None:
                embed_items.append({"type": doc.get("item_type", "default"), "index": doc.get("item_index", 0), "data": data})

    if rawdata:
        channels["rawdata"] = rawdata

    if embed_items:
        is_vec = any(
            isinstance(it, dict) and it.get("vector")
            for it in embed_items
        )
        channels["embed"] = {"is_vector": is_vec, "items": embed_items}

    if pg_metadata_rows:
        meta_list: List[Dict[str, Any]] = []
        for row in pg_metadata_rows:
            data = row.get("data")
            if not isinstance(data, dict):
                data = {}
            entry = {k: v for k, v in data.items() if k not in ("type", "index")}
            entry["type"] = row.get("item_type", "default")
            entry["index"] = row.get("item_index", 0)
            meta_list.append(entry)
        channels["metadata"] = meta_list

    return channels


def storage_has_any_tool_output(mongo_docs: List[Dict[str, Any]], pg_rows: List[Dict[str, Any]]) -> bool:
    return bool(mongo_docs) or bool(pg_rows)
