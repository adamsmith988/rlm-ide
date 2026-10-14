"""Long input: read an oversized user message with parallel sub-agent workers.

A message larger than ``trigger_ratio`` × the model window is split into
overlapping parts. One checklist call lists the user's tasks from the
message's head and tail; then one worker per part (bounded concurrency)
extracts notes for those tasks and reports any instructions found in its
part. Engines receive a digest — tasks, verbatim head, per-part notes,
verbatim tail — rendered from ``long_input_prompts/``. When the notes exceed
the digest budget they are reduced by another worker pass.

Every knob is ``settings.LONG_INPUT_<NAME>``, overridable per chat with
``orchestrator_config["long_input_<name>"]``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from ..config import settings

log = logging.getLogger(__name__)

CONFIG_PREFIX = "long_input_"
DEFAULT_PROMPTS_DIR = Path(__file__).with_name("long_input_prompts")
PROMPT_FILES = ("checklist", "checklist_input", "worker", "worker_input", "default_task", "digest", "part")


@dataclass(frozen=True)
class LongInputSettings:
    enabled: bool
    trigger_ratio: float
    trigger_tokens: int
    chunk_ratio: float
    chunk_overlap_chars: int
    max_chunks: int
    max_parallel: int
    keep_head_chars: int
    keep_tail_chars: int
    digest_max_ratio: float
    max_reduce_passes: int
    max_tasks: int
    worker_max_tokens: int
    checklist_max_tokens: int
    temperature: float
    enable_thinking: bool
    chars_per_token: float
    fallback_context_tokens: int
    binding_service: str
    purpose: str
    prompts_dir: str

    @classmethod
    def resolve(cls, overrides: Optional[Mapping[str, Any]] = None) -> "LongInputSettings":
        ov = overrides if isinstance(overrides, Mapping) else {}
        values: Dict[str, Any] = {}
        for f in fields(cls):
            base = getattr(settings, f"LONG_INPUT_{f.name.upper()}")
            raw = ov.get(f"{CONFIG_PREFIX}{f.name}")
            values[f.name] = _coerce(raw, base) if raw not in (None, "") else base
        return cls(**values)


def _coerce(raw: Any, like: Any) -> Any:
    if isinstance(like, bool):
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)
    if isinstance(like, int):
        return int(raw)
    if isinstance(like, float):
        return float(raw)
    return str(raw)


def _positive_int(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


@lru_cache(maxsize=8)
def load_prompts(prompts_dir: str = "") -> Dict[str, str]:
    """Prompt files from ``prompts_dir``; any file it lacks comes from the bundled defaults."""
    out: Dict[str, str] = {}
    for name in PROMPT_FILES:
        path = Path(prompts_dir) / f"{name}.md" if prompts_dir else None
        if path is None or not path.is_file():
            path = DEFAULT_PROMPTS_DIR / f"{name}.md"
        out[name] = path.read_text(encoding="utf-8").strip()
    return out


def context_tokens(cfg: LongInputSettings, orchestrator_config: Optional[Mapping[str, Any]]) -> int:
    oc = orchestrator_config if isinstance(orchestrator_config, Mapping) else {}
    return (
        _positive_int(oc.get("context_length"))
        or _positive_int(oc.get("max_model_len"))
        or cfg.fallback_context_tokens
        or int(settings.STREAM_CONTEXT_CHAR_BUDGET / max(cfg.chars_per_token, 0.1))
    )


@dataclass
class LongInputResult:
    message: str
    tasks: List[str] = field(default_factory=list)
    parts: int = 0
    failed_parts: int = 0
    reduce_passes: int = 0
    original_chars: int = 0

    def meta(self) -> Dict[str, Any]:
        return {
            "original_chars": self.original_chars,
            "digest_chars": len(self.message),
            "parts": self.parts,
            "failed_parts": self.failed_parts,
            "reduce_passes": self.reduce_passes,
            "tasks": list(self.tasks),
        }


class LongInputReader:
    def __init__(self, cfg: LongInputSettings, *, window_tokens: int, llm: Any = None) -> None:
        self.cfg = cfg
        self.window_tokens = max(1, int(window_tokens))
        self.prompts = load_prompts(cfg.prompts_dir)
        self._llm = llm

    @classmethod
    def for_turn(
        cls, orchestrator_config: Optional[Mapping[str, Any]], *, llm: Any = None
    ) -> Optional["LongInputReader"]:
        cfg = LongInputSettings.resolve(orchestrator_config)
        if not cfg.enabled:
            return None
        return cls(cfg, window_tokens=context_tokens(cfg, orchestrator_config), llm=llm)

    # ---- sizing -----------------------------------------------------

    def tokens(self, text: str) -> int:
        return int(len(text or "") / max(self.cfg.chars_per_token, 0.1)) + 1

    @property
    def trigger_tokens(self) -> int:
        return self.cfg.trigger_tokens or int(self.window_tokens * self.cfg.trigger_ratio)

    def needed(self, message: str) -> bool:
        return bool(message) and self.tokens(message) >= max(1, self.trigger_tokens)

    def _chars(self, ratio: float) -> int:
        return max(1, int(self.window_tokens * ratio * self.cfg.chars_per_token))

    # ---- LLM --------------------------------------------------------

    def _call(self) -> Any:
        if self._llm is None:
            from ..llm.gateway_llm_client import GatewayLLMClient
            from ..orchestrators._shared.llm_roles import gateway_llm

            self._llm = gateway_llm(
                GatewayLLMClient(service=self.cfg.binding_service, purpose=self.cfg.purpose)
            )
        return self._llm

    async def _ask_json(self, system: str, user: str, max_tokens: int) -> Optional[Any]:
        from ..orchestrators._shared.llm_roles import extract_json

        reply = await self._call()(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.cfg.temperature,
            max_tokens=max_tokens or None,
            enable_thinking=self.cfg.enable_thinking,
        )
        if not reply.ok:
            raise RuntimeError(reply.error or "llm_error")
        try:
            return extract_json(reply.text)
        except ValueError:
            return reply.text

    # ---- stages -----------------------------------------------------

    async def checklist(self, message: str) -> List[str]:
        head, tail = self._head_tail(message)
        body = self.prompts["checklist_input"].format(head=head, tail=tail) if tail else head
        try:
            raw = await self._ask_json(self.prompts["checklist"], body, self.cfg.checklist_max_tokens)
        except Exception as exc:  # noqa: BLE001 — workers still report tasks
            log.warning("long_input: checklist failed: %s", exc)
            return []
        tasks = raw.get("tasks") if isinstance(raw, dict) else raw if isinstance(raw, list) else []
        return self._merge_tasks([], tasks)

    async def read_parts(self, parts: List[str], tasks: List[str]) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(max(1, self.cfg.max_parallel))
        task_block = "\n".join(f"- {t}" for t in tasks or [self.prompts["default_task"]])

        async def one(i: int, text: str) -> Dict[str, Any]:
            user = self.prompts["worker_input"].format(
                tasks=task_block, index=i + 1, total=len(parts), text=text
            )
            async with sem:
                try:
                    raw = await self._ask_json(self.prompts["worker"], user, self.cfg.worker_max_tokens)
                except Exception as exc:  # noqa: BLE001 — reported per part
                    return {"ok": False, "error": str(exc), "notes": "", "tasks": []}
            if isinstance(raw, dict):
                notes = raw.get("notes")
                if not isinstance(notes, str):
                    notes = json.dumps(notes, ensure_ascii=False) if notes else ""
                return {"ok": True, "notes": notes.strip(), "tasks": raw.get("tasks") or []}
            return {"ok": True, "notes": str(raw or "").strip(), "tasks": []}

        return list(await asyncio.gather(*(one(i, p) for i, p in enumerate(parts))))

    async def read(self, message: str) -> Optional[LongInputResult]:
        """Digest for ``message``, or None when it is small or every worker failed."""
        if not self.needed(message):
            return None
        parts = self._split(message)
        tasks = await self.checklist(message)
        results = await self.read_parts(parts, tasks)
        if not any(r["ok"] for r in results):
            log.warning("long_input: every worker failed (%s); keeping the original message", results[0].get("error"))
            return None
        for r in results:
            tasks = self._merge_tasks(tasks, r.get("tasks"))
        notes = [self._part_notes(r, parts[i]) for i, r in enumerate(results)]

        budget = self._chars(self.cfg.digest_max_ratio)
        passes = 0
        while sum(len(n) for n in notes) > budget and passes < self.cfg.max_reduce_passes and len(notes) > 1:
            passes += 1
            grouped = self._split("\n\n".join(notes))
            reduced = await self.read_parts(grouped, tasks)
            if not any(r["ok"] for r in reduced):
                break
            notes = [self._part_notes(r, grouped[i]) for i, r in enumerate(reduced)]
        notes = self._fit(notes, budget)

        total = len(notes)
        rendered = "\n\n".join(
            self.prompts["part"].format(index=i + 1, total=total, notes=n) for i, n in enumerate(notes)
        )
        head, tail = self._head_tail(message)
        digest = self.prompts["digest"].format(
            chars=len(message),
            parts=len(parts),
            tasks="\n".join(f"{i}. {t}" for i, t in enumerate(tasks, 1)) or "-",
            head=head,
            notes=rendered,
            tail=tail or "-",
        )
        return LongInputResult(
            message=digest,
            tasks=tasks,
            parts=len(parts),
            failed_parts=sum(1 for r in results if not r["ok"]),
            reduce_passes=passes,
            original_chars=len(message),
        )

    # ---- helpers ----------------------------------------------------

    def _split(self, text: str) -> List[str]:
        from ..orchestrators._shared.text_chunks import chunk_text

        chunks = chunk_text(
            text,
            max_chars=self._chars(self.cfg.chunk_ratio),
            overlap=self.cfg.chunk_overlap_chars,
            max_chunks=self.cfg.max_chunks,
        )
        return [text[c.start : c.end] for c in chunks] or [text]

    def _head_tail(self, message: str) -> tuple:
        h, t = max(0, self.cfg.keep_head_chars), max(0, self.cfg.keep_tail_chars)
        if len(message) <= h + t:
            return message, ""
        return message[:h], (message[-t:] if t else "")

    def _merge_tasks(self, tasks: List[str], extra: Any) -> List[str]:
        out = list(tasks)
        seen = {" ".join(t.lower().split()) for t in out}
        for t in extra if isinstance(extra, list) else []:
            text = " ".join(str(t or "").split())
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                out.append(text)
        return out[: max(1, self.cfg.max_tasks)]

    def _part_notes(self, result: Dict[str, Any], source: str) -> str:
        if result.get("ok") and result.get("notes"):
            return result["notes"]
        # A failed worker keeps a verbatim excerpt of its part rather than nothing.
        keep = max(1, self._chars(self.cfg.digest_max_ratio) // 8)
        return source if len(source) <= keep else f"{source[: keep // 2]}\n[…]\n{source[-keep // 2 :]}"

    @staticmethod
    def _fit(notes: List[str], budget: int) -> List[str]:
        total = sum(len(n) for n in notes)
        if total <= budget or not notes:
            return notes
        share = max(1, budget // len(notes))
        return [n if len(n) <= share else f"{n[: share - 1]}…" for n in notes]
