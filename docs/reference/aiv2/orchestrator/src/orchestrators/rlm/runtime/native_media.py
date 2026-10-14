"""Catalog-gated native look: attach pool files to the next RLM chat turn.

Data Pool files stay on disk. ``rlm.look`` is the opt-in courier onto
``message.content`` (image/audio/video) when the Agent model's catalog
``input_modalities`` actually accept that kind. Text stays ``skills.read_text``.
Documents/archives still need tools.
"""

from __future__ import annotations

import base64
import io
import mimetypes
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

try:
    from modules.plugins.agent_flow.case_artifacts import classify_case_media
except Exception:  # pragma: no cover — optional outside agent_flow
    def classify_case_media(*_a, **_k):
        return {"kind": "unknown", "media_type": ""}

try:
    from modules.plugins.agent_flow.media_persist import sniff_bytes
except Exception:  # pragma: no cover
    def sniff_bytes(data: bytes = b"", *_a, **_k):
        return {"mime": "application/octet-stream", "ext": ""}

LOOK_KINDS = frozenset({"image", "audio", "video"})
_MODALITY_ALIASES = {
    "image": "image",
    "images": "image",
    "vision": "image",
    "audio": "audio",
    "voice": "audio",
    "speech": "audio",
    "video": "video",
    "text": "text",
    "chat": "text",
    "prompt": "text",
}
# Per-file raw cap (env override). Was 12MiB — remote gateways 413 on multi-image
# base64 bodies long before the token budget notices.
MAX_LOOK_BYTES = int(os.environ.get("AIV2_RLM_LOOK_MAX_BYTES", str(2 * 1024 * 1024)))
# Target encoded (base64 data-url payload) size after optional JPEG resize.
MAX_LOOK_ENCODED_BYTES = int(
    os.environ.get("AIV2_RLM_LOOK_MAX_ENCODED_BYTES", str(int(1.5 * 1024 * 1024)))
)
# Sum of encoded media parts in one chat turn. Default ~3MiB (≈ two resized
# looks). Set AIV2_RLM_LOOK_TURN_ENCODED_BYTES=0 to disable proactive byte cap.
def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(str(raw).strip())
    except ValueError:
        return int(default)


MAX_LOOK_TURN_ENCODED_BYTES = _env_int(
    "AIV2_RLM_LOOK_TURN_ENCODED_BYTES", 3 * 1024 * 1024
)
MAX_LOOK_BATCH = 8
# Hard caps per LLM turn — OSBI / remote APIs 413 when many base64 media land
# in one body. Remaining looks stay queued for the next turn.
LOOK_MAX_IMAGES_PER_TURN = max(
    0, int(os.environ.get("RLM_LOOK_MAX_IMAGES_PER_TURN", "2") or "2")
)
LOOK_MAX_VIDEOS_PER_TURN = max(
    0, int(os.environ.get("RLM_LOOK_MAX_VIDEOS_PER_TURN", "1") or "1")
)
LOOK_MAX_AUDIO_PER_TURN = max(
    0, int(os.environ.get("RLM_LOOK_MAX_AUDIO_PER_TURN", "2") or "2")
)

_MEDIA_PART_TYPES = frozenset(
    {"image_url", "image", "video_url", "video", "audio_url", "audio", "input_audio"}
)


def normalize_input_modalities(raw: Any) -> Set[str]:
    out: Set[str] = set()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple, set)):
        return out
    for item in raw:
        key = _MODALITY_ALIASES.get(str(item or "").strip().lower(), "")
        if key:
            out.add(key)
    return out


def native_look_kinds(modalities: Iterable[str]) -> Set[str]:
    return {str(m).strip().lower() for m in modalities} & LOOK_KINDS


def classify_local_file(path: str) -> Tuple[str, str]:
    """Return (kind, mime) from magic bytes, then filename."""
    name = os.path.basename(path or "")
    mime = ""
    kind = "other"
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096)
        sniffed_kind, sniffed_mime, _ = sniff_bytes(head)
        mime = sniffed_mime
        if sniffed_kind in LOOK_KINDS or sniffed_kind in ("document", "archive", "json"):
            kind = "text" if sniffed_kind == "json" else sniffed_kind
        else:
            kind = classify_case_media(mime=mime, filename=name)
    except OSError:
        kind = classify_case_media(filename=name)
        guessed, _ = mimetypes.guess_type(name)
        mime = guessed or ""
    if not mime:
        guessed, _ = mimetypes.guess_type(name)
        mime = guessed or "application/octet-stream"
    if kind == "document" and str(mime).startswith("image/"):
        kind = "image"
    return kind, mime


def _b64_len(raw: bytes) -> int:
    return (len(raw) + 2) // 3 * 4


def maybe_resize_image_bytes(
    raw: bytes,
    *,
    mime: str = "",
    max_encoded_bytes: int = MAX_LOOK_ENCODED_BYTES,
) -> Tuple[bytes, str]:
    """Shrink image bytes so base64 payload stays under ``max_encoded_bytes``.

    Always decodes the image first so truncated/corrupt files fail at
    ``rlm.look`` instead of later at the VLM. Uses Pillow when available.
    """
    cap = max(16 * 1024, int(max_encoded_bytes))
    out_mime = (mime or "application/octet-stream").strip() or "application/octet-stream"
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        if _b64_len(raw) <= cap:
            # No Pillow: cannot verify; still allow small payloads through.
            return raw, out_mime
        raise RuntimeError(
            f"look_too_large_no_resize:{_b64_len(raw)}>{cap}:{exc}"
        ) from exc

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:
        raise RuntimeError(f"look_image_decode_failed:{exc}") from exc

    if _b64_len(raw) <= cap:
        return raw, out_mime

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")

    width, height = img.size
    # Iteratively scale + JPEG quality until under cap (Prime-style).
    for scale in (1.0, 0.75, 0.5, 0.35, 0.25, 0.15):
        if scale < 1.0:
            nw = max(32, int(width * scale))
            nh = max(32, int(height * scale))
            resample = getattr(
                getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS
            )
            candidate = img.resize((nw, nh), resample)
        else:
            candidate = img
        for quality in (85, 70, 55, 40, 28):
            buf = io.BytesIO()
            candidate.save(buf, format="JPEG", quality=quality, optimize=True)
            out = buf.getvalue()
            if _b64_len(out) <= cap:
                return out, "image/jpeg"
    raise RuntimeError(
        f"look_too_large:{_b64_len(raw)}>{cap}:resize_exhausted"
    )


def content_part_for_bytes(*, kind: str, mime: str, raw: bytes) -> Dict[str, Any]:
    mime = (mime or "application/octet-stream").strip() or "application/octet-stream"
    b64 = base64.b64encode(raw).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    if kind == "video":
        return {"type": "video_url", "video_url": {"url": data_url}}
    if kind == "audio":
        return {"type": "audio_url", "audio_url": {"url": data_url}}
    return {"type": "image_url", "image_url": {"url": data_url}}


def part_encoded_bytes(part: Any) -> int:
    """Approximate ASCII size of a native media content part (data-url body)."""
    if not isinstance(part, dict):
        return 0
    for key in ("image_url", "video_url", "audio_url"):
        blob = part.get(key)
        if isinstance(blob, dict):
            url = str(blob.get("url") or "")
            if url:
                return len(url)
        elif isinstance(blob, str) and blob:
            return len(blob)
    data = part.get("data")
    if isinstance(data, str):
        return len(data)
    return 0


def requeue_looks_from_messages(
    messages: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Pull native look media out of history into a list + text stubs.

    Returns ``(rewritten_messages, extracted_looks)``. Callers that run this
    *before compaction* should apply the rewritten messages and **discard**
    ``extracted_looks`` (already shown — re-queueing blocks the pending queue).
    Overflow recovery may still re-queue deferred items via other helpers.
    """
    pending: List[Dict[str, Any]] = []
    out: List[Dict[str, Any]] = []
    for raw in messages or []:
        if not isinstance(raw, Mapping):
            continue
        msg = dict(raw)
        content = msg.get("content")
        if not isinstance(content, list) or not any(
            _is_media_content_part(p) for p in content
        ):
            out.append(msg)
            continue
        texts: List[str] = []
        names: List[str] = []
        header = ""
        media_parts: List[Dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                t = str(part.get("text") or "").strip()
                if t:
                    texts.append(t)
                if t.startswith("[rlm.look]"):
                    header = t
                    rest = t.replace("[rlm.look]", "", 1).split("\n", 1)[0]
                    for bit in rest.split(","):
                        name = bit.strip().split(" (", 1)[0].strip()
                        if name:
                            names.append(name)
            elif _is_media_content_part(part):
                media_parts.append(part)
        for i, part in enumerate(media_parts):
            ptype = str(part.get("type") or "")
            kind = "image"
            if "video" in ptype:
                kind = "video"
            elif "audio" in ptype:
                kind = "audio"
            fname = names[i] if i < len(names) else f"{kind}_{i + 1}"
            pending.append(
                {
                    "filename": fname,
                    "kind": kind,
                    "part": part,
                    "encoded_bytes": part_encoded_bytes(part),
                }
            )
        note = (
            "[rlm.look] prior look pixels removed before context compact "
            "(already shown — not re-queued; remaining pending looks continue)."
        )
        if header:
            note = header.split("\n", 1)[0] + "\n" + note
        elif texts:
            note = texts[0][:500] + "\n" + note
        msg["content"] = note
        out.append(msg)
    return out, pending


def look_file(
    path: str,
    *,
    modalities: Iterable[str],
    max_bytes: int = MAX_LOOK_BYTES,
    max_encoded_bytes: int = MAX_LOOK_ENCODED_BYTES,
) -> Dict[str, Any]:
    """Load a local file as a native chat part, or raise RuntimeError."""
    loc = str(path or "").strip()
    if not loc or not os.path.isfile(loc):
        raise RuntimeError(f"look_file_not_found:{loc or '?'}")
    size = os.path.getsize(loc)
    if size <= 0:
        raise RuntimeError(f"look_empty_file:{os.path.basename(loc)}")
    kind, mime = classify_local_file(loc)
    native = native_look_kinds(normalize_input_modalities(list(modalities)))
    if kind not in LOOK_KINDS:
        raise RuntimeError(
            f"look_not_native_kind:{kind}:{os.path.basename(loc)}"
        )
    if kind not in native:
        have = ",".join(sorted(native)) or "none"
        raise RuntimeError(
            f"model_cannot_look:{kind}:native={have}:{os.path.basename(loc)}"
        )
    # Images may be larger on disk — we resize after read. Non-images hard-cap.
    read_cap = max(1, int(max_bytes))
    if kind == "image":
        # Allow reading up to 12MiB raw so we can still JPEG-downsize.
        read_cap = max(read_cap, 12 * 1024 * 1024)
    if size > read_cap:
        raise RuntimeError(
            f"look_too_large:{os.path.basename(loc)}:{size}>{int(read_cap)}"
        )
    with open(loc, "rb") as fh:
        raw = fh.read()
    out_mime = mime
    if kind == "image":
        raw, out_mime = maybe_resize_image_bytes(
            raw, mime=mime, max_encoded_bytes=max_encoded_bytes
        )
    elif _b64_len(raw) > max(1, int(max_encoded_bytes)):
        raise RuntimeError(
            f"look_too_large:{os.path.basename(loc)}:{_b64_len(raw)}>{int(max_encoded_bytes)}"
        )
    part = content_part_for_bytes(kind=kind, mime=out_mime, raw=raw)
    return {
        "ok": True,
        "kind": kind,
        "mime": out_mime,
        "filename": os.path.basename(loc),
        "bytes": len(raw),
        "encoded_bytes": part_encoded_bytes(part),
        "path": loc,
        "part": part,
    }


def look_inventory_note(
    filename: str,
    media_type: str,
    *,
    native: Set[str],
) -> str:
    fn = str(filename or "").strip()
    kind = str(media_type or "").strip().lower()
    if not fn:
        return ""
    if kind in native:
        return f"  - {fn} ({kind}, native: await rlm.look({fn!r}))"
    if kind:
        return f"  - {fn} ({kind})"
    return f"  - {fn}"


def coerce_locators(payload: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for key in ("locators", "paths", "files"):
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            out.append(raw.strip())
        elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
            for item in raw:
                text = str(item or "").strip()
                if text:
                    out.append(text)
    for key in ("path", "filename", "locator", "file"):
        text = str(payload.get(key) or "").strip()
        if text:
            out.append(text)
    seen: Set[str] = set()
    uniq: List[str] = []
    for item in out:
        if item in seen:
            continue
        seen.add(item)
        uniq.append(item)
    return uniq[:MAX_LOOK_BATCH]


def look_per_turn_caps() -> Dict[str, int]:
    """Hard per-kind caps for one chat turn (env-tunable)."""
    return {
        "image": LOOK_MAX_IMAGES_PER_TURN,
        "video": LOOK_MAX_VIDEOS_PER_TURN,
        "audio": LOOK_MAX_AUDIO_PER_TURN,
    }


def look_turn_limit_hint() -> str:
    caps = look_per_turn_caps()
    return (
        f"at most {caps['image']} image(s) and {caps['video']} video(s) "
        "per look call and per chat turn (do not dump many files in one look)"
    )


def remaining_look_kind_slots(
    pending: Sequence[Mapping[str, Any]],
) -> Dict[str, int]:
    """How many more of each kind fit in the *next* inject under hard caps."""
    caps = look_per_turn_caps()
    used = {k: 0 for k in caps}
    for raw in pending:
        if not isinstance(raw, Mapping):
            continue
        kind = str(raw.get("kind") or "").strip().lower()
        if kind in used:
            used[kind] += 1
    return {k: max(0, int(caps[k]) - int(used[k])) for k in caps}


def select_look_batch_for_turn(
    pending: Sequence[Mapping[str, Any]],
    *,
    context_length: Optional[int] = None,
    total_budget: Optional[int] = None,
    max_encoded_bytes: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split pending looks into (inject_now, defer) under per-kind + window + byte caps.

    Order is preserved. Images ≤ ``LOOK_MAX_IMAGES_PER_TURN``, videos ≤
    ``LOOK_MAX_VIDEOS_PER_TURN``, audio ≤ ``LOOK_MAX_AUDIO_PER_TURN``. An extra
    total-parts budget (context window) and encoded-byte budget may defer more
    even when kind caps still have room — avoids remote 413 on multi-base64 bodies.
    """
    from src.orchestrators.rlm.runtime.context_budget import media_parts_per_turn

    caps = look_per_turn_caps()
    if total_budget is None:
        override = os.environ.get("RLM_LOOK_MAX_PER_TURN")
        if override and str(override).strip().isdigit():
            budget = max(1, int(override))
        else:
            budget = int(media_parts_per_turn(context_length))
    else:
        budget = max(0, int(total_budget))

    byte_cap = (
        int(max_encoded_bytes)
        if max_encoded_bytes is not None
        else int(MAX_LOOK_TURN_ENCODED_BYTES)
    )
    # 0 / negative → no proactive byte budget (413 recovery handles overflow).
    byte_cap = int(byte_cap)
    enforce_bytes = byte_cap > 0
    if enforce_bytes:
        byte_cap = max(0, byte_cap)

    taken: List[Dict[str, Any]] = []
    deferred: List[Dict[str, Any]] = []
    used = {"image": 0, "video": 0, "audio": 0}
    used_bytes = 0
    for raw in pending:
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        kind = str(item.get("kind") or "").strip().lower()
        part = item.get("part")
        enc = int(item.get("encoded_bytes") or 0) or part_encoded_bytes(part)
        if enc > 0:
            item["encoded_bytes"] = enc
        over_bytes = bool(enforce_bytes and enc and used_bytes + enc > byte_cap)
        if kind not in caps:
            if len(taken) >= budget or over_bytes:
                deferred.append(item)
                continue
            taken.append(item)
            used_bytes += enc
            continue
        if used[kind] >= caps[kind] or len(taken) >= budget or over_bytes:
            deferred.append(item)
            continue
        taken.append(item)
        used[kind] += 1
        used_bytes += enc
    return taken, deferred


def partition_look_call(
    candidates: Sequence[Mapping[str, Any]],
    *,
    already_pending: Sequence[Mapping[str, Any]] = (),
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split one ``rlm.look`` call into (queue_now, skipped_this_call).

    Files that already fill the next inject's kind caps still queue for later
    turns (they go into ``queue_now`` after the first-turn slice is filled via
    the session pending list). Only locators beyond a generous multi-turn
    backlog are skipped so a single tool call cannot dump dozens of base64
    payloads into memory at once.
    """
    caps = look_per_turn_caps()
    # Allow up to 3 turns of backlog per kind from one call (then skip).
    backlog = {k: max(1, int(caps[k])) * 3 for k in caps}
    pending_used = {k: 0 for k in caps}
    for raw in already_pending:
        if not isinstance(raw, Mapping):
            continue
        kind = str(raw.get("kind") or "").strip().lower()
        if kind in pending_used:
            pending_used[kind] += 1

    accept: List[Dict[str, Any]] = []
    skip: List[Dict[str, Any]] = []
    call_used = {k: 0 for k in caps}
    for raw in candidates:
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        kind = str(item.get("kind") or "").strip().lower()
        if kind not in caps:
            accept.append(item)
            continue
        # Per-call soft ceiling: never accept more than one full turn of each
        # kind in a single rlm.look invocation (rest must be a later call).
        if call_used[kind] >= caps[kind]:
            skip.append({**item, "skip_reason": f"look_call_cap:{kind}"})
            continue
        if pending_used[kind] + call_used[kind] >= backlog[kind]:
            skip.append({**item, "skip_reason": f"look_backlog_cap:{kind}"})
            continue
        accept.append(item)
        call_used[kind] += 1
    return accept, skip


def _is_media_content_part(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    kind = str(part.get("type") or "").strip().lower()
    return kind in _MEDIA_PART_TYPES


def strip_look_media_from_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    keep_last_media_message: bool = False,
) -> Tuple[List[Dict[str, Any]], int]:
    """Replace native media parts with short text notes (files stay on disk).

    When ``keep_last_media_message`` is True, the most recent user message that
    still carries media parts is left intact (current look turn). Older look
    payloads are stripped so subsequent turns do not re-send multi-MB base64.
    """
    msgs = [dict(m) for m in messages if isinstance(m, Mapping)]
    media_indexes: List[int] = []
    for i, msg in enumerate(msgs):
        content = msg.get("content")
        if isinstance(content, list) and any(_is_media_content_part(p) for p in content):
            media_indexes.append(i)
    keep_idx = media_indexes[-1] if keep_last_media_message and media_indexes else None
    stripped = 0
    out: List[Dict[str, Any]] = []
    for i, msg in enumerate(msgs):
        content = msg.get("content")
        if not isinstance(content, list) or i == keep_idx:
            out.append(msg)
            continue
        if not any(_is_media_content_part(p) for p in content):
            out.append(msg)
            continue
        texts: List[str] = []
        names: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                t = str(part.get("text") or "").strip()
                if t:
                    texts.append(t)
                # Pull filenames from look header lines when present.
                if t.startswith("[rlm.look]"):
                    names.append(t)
            elif _is_media_content_part(part):
                stripped += 1
        note = (
            "[rlm.look] older image pixels were dropped from this prompt to fit "
            "the window (files remain on disk). Do NOT re-look the same paths — "
            "write notes from what you already observed, or look only at new files."
        )
        if names:
            note = names[0] + "\n" + note
        elif texts:
            note = texts[0][:500] + "\n" + note
        new_msg = dict(msg)
        new_msg["content"] = note
        out.append(new_msg)
    return out, stripped


def _data_url_to_bytes(url: str) -> Tuple[bytes, str]:
    text = str(url or "")
    mime = "application/octet-stream"
    if text.startswith("data:") and ";base64," in text:
        header, _, b64 = text.partition(";base64,")
        mime = header[5:] or mime
        return base64.b64decode(b64), mime
    return b"", mime


def split_latest_look_to_one(
    messages: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    """Keep only the first media part in the latest look message; return deferred.

    Deferred items are suitable for ``session._pending_looks`` so remaining
    images are shown on later turns after a 413.
    """
    msgs = [dict(m) for m in messages if isinstance(m, Mapping)]
    idx = None
    for i in range(len(msgs) - 1, -1, -1):
        content = msgs[i].get("content")
        if isinstance(content, list) and sum(
            1 for p in content if _is_media_content_part(p)
        ) > 1:
            idx = i
            break
    if idx is None:
        return msgs, [], False

    content = list(msgs[idx].get("content") or [])
    texts = [p for p in content if isinstance(p, dict) and p.get("type") == "text"]
    media = [p for p in content if _is_media_content_part(p)]
    keep = media[0]
    rest = media[1:]
    header = ""
    if texts:
        header = str(texts[0].get("text") or "")
    # Prefer first filename token from header when splitting.
    names = []
    if header.startswith("[rlm.look]"):
        rest_line = header.split("\n", 1)[0]
        names = [
            bit.strip()
            for bit in rest_line.replace("[rlm.look]", "").split(",")
            if bit.strip()
        ]
    new_parts: List[Any] = []
    keep_name = names[0] if names else "file"
    new_parts.append(
        {
            "type": "text",
            "text": (
                f"[rlm.look] {keep_name} (413 recovery — sending one file now; "
                f"{len(rest)} more queued for later turns)"
            ),
        }
    )
    new_parts.append(keep)
    msgs[idx] = dict(msgs[idx])
    msgs[idx]["content"] = new_parts

    deferred: List[Dict[str, Any]] = []
    for i, part in enumerate(rest):
        kind = "image"
        ptype = str(part.get("type") or "")
        if "video" in ptype:
            kind = "video"
        elif "audio" in ptype:
            kind = "audio"
        fname = names[i + 1] if i + 1 < len(names) else f"{kind}_{i + 2}"
        # Strip trailing " (image)" style hints from header names.
        fname = fname.split(" (", 1)[0].strip() or fname
        deferred.append(
            {
                "filename": fname,
                "kind": kind,
                "part": part,
                "encoded_bytes": part_encoded_bytes(part),
            }
        )
    return msgs, deferred, True


def recompress_look_media_in_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    max_encoded_bytes: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Re-encode image_* parts under a tighter base64 budget. Returns (msgs, n)."""
    cap = max(16 * 1024, int(max_encoded_bytes))
    msgs = [dict(m) for m in messages if isinstance(m, Mapping)]
    changed = 0
    for msg in msgs:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_content: List[Any] = []
        touched = False
        for part in content:
            if not _is_media_content_part(part):
                new_content.append(part)
                continue
            ptype = str(part.get("type") or "")
            # Only recompress still images; leave audio/video as-is.
            if ptype not in ("image_url", "image"):
                new_content.append(part)
                continue
            blob = part.get("image_url") if isinstance(part.get("image_url"), dict) else {}
            url = str((blob or {}).get("url") or part.get("url") or "")
            raw, mime = _data_url_to_bytes(url)
            if not raw:
                new_content.append(part)
                continue
            try:
                out, out_mime = maybe_resize_image_bytes(
                    raw, mime=mime, max_encoded_bytes=cap
                )
            except Exception:
                new_content.append(part)
                continue
            if out == raw and _b64_len(raw) <= cap:
                new_content.append(part)
                continue
            new_part = content_part_for_bytes(kind="image", mime=out_mime, raw=out)
            new_content.append(new_part)
            touched = True
            changed += 1
        if touched:
            msg["content"] = new_content
    return msgs, changed
