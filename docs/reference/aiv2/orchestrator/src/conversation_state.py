"""
Conversation State Tracker
Zero-LLM-call heuristic analysis of chat history to extract
structured metadata about the conversation flow.

Tracks: turn count, greeting status, active topic, topic transitions,
file context relevance, conversation phase, and language.
"""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

log = logging.getLogger(__name__)

# Greeting patterns (Persian + English)
_GREETING_PATTERNS = re.compile(
    r"^(سلام|درود|صبح بخیر|عصر بخیر|شب بخیر|خسته نباشید"
    r"|hello|hi|hey|good morning|good evening|good afternoon"
    r"|howdy|greetings|what'?s up|hola)",
    re.IGNORECASE | re.UNICODE,
)

_THANKS_PATTERNS = re.compile(
    r"^(ممنون|مرسی|تشکر|سپاس|دستت درد نکنه|دمت گرم"
    r"|thanks|thank you|thx|ty|cheers|much appreciated)",
    re.IGNORECASE | re.UNICODE,
)

# Persian + Latin "letter boundary": disallow either set on either side.
# Without this, "سند" (document) matches inside "نویسنده" (writer) and
# "نوشته" inside any longer compound — which used to break topic detection
# (admin/admin123 regression).
_LB_BEFORE = r"(?<![\w\u0600-\u06FF])"
_LB_AFTER  = r"(?![\w\u0600-\u06FF])"

_FILE_KEYWORDS = re.compile(
    _LB_BEFORE
    + r"(?:فایل|تصویر|عکس|ویدیو|فیلم|صوت|صدا|آهنگ|سند|مدرک|پی‌دی‌اف"
    r"|file|image|photo|picture|video|audio|document|pdf|upload"
    r"|آپلود|دانلود|ضمیمه|attachment|img|pic"
    r"|بخون|بخوان|بخونش|بخوانش|read\b|show content|file content|محتوا)"
    + _LB_AFTER,
    re.IGNORECASE | re.UNICODE,
)

_FILE_REFERENCE_KEYWORDS = re.compile(
    _LB_BEFORE
    + r"(?:این عکس|این تصویر|این فایل|این ویدیو|این صدا|این صوت"
    r"|همین فایل|همین عکس|همین تصویر|همین ویدیو"
    r"|همون فایل|همون عکس|همون تصویر|همون ویدیو"
    r"|عکسی که|تصویری که|فایلی که|ویدیویی که"
    r"|this image|this file|this photo|this video|the image|the file"
    r"|that image|that file|uploaded file|attached)"
    + _LB_AFTER,
    re.IGNORECASE | re.UNICODE,
)

_SEARCH_KEYWORDS = re.compile(
    r"(جستجو|سرچ|پیدا کن|بگرد|search|find|look up|query|lookup"
    r"|دنبال.*بگرد|توی.*پایگاه|database)",
    re.IGNORECASE | re.UNICODE,
)

# Punctuation / whitespace that sits between a leading "salam," and the
# substantive part of the message. Used by ``_strip_leading_courtesy``.
_LEADING_PUNCT = re.compile(r"^[\s,.،؛؟!?\-:،\u200c]+", re.UNICODE)


_STOPWORDS = {
    # Persian — connectors, pronouns, copulas
    "و", "که", "از", "به", "در", "را", "این", "آن", "با", "تا",
    "تو", "من", "ما", "شما", "خودم", "خودش", "خود", "هم", "همه",
    "هست", "هستی", "هستم", "بود", "بودم", "بودی", "می", "میشه",
    "نمی", "ها", "های", "هایی", "چه", "چی", "چرا", "چگونه", "اگه",
    "اگر", "اما", "ولی", "پس", "یک", "یکی", "یا", "یعنی", "بر",
    "برای", "روی", "زیر", "بعد", "قبل", "حالا", "الان", "همین",
    "همون", "موضوع", "نکته", "نکات",
    # Persian — generic continuation / courtesy markers that should NOT
    # count as substantive content. Without these, "بیشتر توضیح بده"
    # is mistaken for a topic switch.
    "بیشتر", "کمتر", "دیگه", "دیگر", "ادامه", "ادامش", "بازم",
    "توضیح", "توضیحات", "مثال", "مثلا", "مثلاً", "نمونه", "بده",
    "بزن", "بنویس", "بگو", "بفرما", "بفرمایید", "لطف", "لطفا",
    "لطفاً", "ممنون", "میشه", "بشه", "خوبه", "خواهشا", "بپرسم",
    "سوال", "سؤال", "پرسش", "جواب", "پاسخ",
    # English — connectors, copulas, pronouns
    "the", "a", "an", "of", "is", "are", "was", "were", "in", "on",
    "to", "for", "and", "or", "but", "so", "as", "be", "by", "at",
    "this", "that", "these", "those", "it", "its", "if", "with",
    "from", "you", "your", "i", "we", "our", "they", "them",
    # English — continuation / courtesy markers
    "more", "less", "another", "again", "explain", "explanation",
    "example", "examples", "tell", "show", "give", "please",
    "thanks", "thank", "ok", "okay", "sure", "continue", "expand",
    "elaborate", "detail", "details", "question", "answer",
}


def _meaningful_tokens(text: str) -> set[str]:
    """Persian/Latin-aware bag of substantive tokens for topic-overlap checks."""
    if not text:
        return set()
    base = _strip_leading_courtesy(text.lower().strip()) or text.lower().strip()
    out: set[str] = set()
    for tok in re.findall(r"[\w\u0600-\u06FF]+", base):
        if len(tok) >= 3 and tok not in _STOPWORDS:
            out.add(tok)
    return out


def _detect_topic_shift_by_content(
    chat_history: List[Dict[str, Any]],
    current_message: str,
) -> bool:
    """
    Detect a topic switch when the current message has *zero* substantive
    tokens in common with the recent conversational tail.

    Conservative on purpose: this fires only when there are at least two
    substantive (non-stopword, non-continuation-marker) tokens in the new
    message AND the message itself is not trivially short. False negatives
    here are much cheaper than false positives — see "بیشتر توضیح بده"
    regression which used to trip the older version.
    """
    if not current_message:
        return False
    body = _strip_leading_courtesy(current_message.strip()) or current_message.strip()
    if len(body) < 12:
        return False
    cur = _meaningful_tokens(current_message)
    if len(cur) < 2:
        return False
    tail = chat_history[-8:] if chat_history else []
    if not tail:
        return False
    rec: set[str] = set()
    for m in tail:
        rec |= _meaningful_tokens(m.get("content") or "")
    if not rec:
        return False
    return cur.isdisjoint(rec)


def _strip_leading_courtesy(text: str) -> str:
    """
    Remove a leading greeting/thanks token from ``text`` and return what's left.

    The point of this helper is *topic detection*, not formatting:
    when a user writes "سلام، دانته نویسنده چه کتاب‌هایی هست؟" the message
    is fundamentally a question about Dante — the leading "سلام" should not
    suppress topic-change detection.

    Returns the original text when stripping would leave nothing useful
    (i.e. the message really is just a greeting).
    """
    if not text:
        return text
    stripped = text.strip()
    m = _GREETING_PATTERNS.match(stripped)
    if not m:
        m = _THANKS_PATTERNS.match(stripped)
    if not m:
        return text
    rest = stripped[m.end():]
    rest = _LEADING_PUNCT.sub("", rest)
    # Only strip when there is real content after the greeting; otherwise
    # the greeting *is* the message and we want classifiers to see it.
    return rest if len(rest) >= 4 else text


@dataclass
class ConversationState:
    """Structured metadata about the current conversation state."""
    turn_number: int = 1
    greeting_done: bool = False
    active_topic: str = "general_chat"
    topic_changed: bool = False
    file_discussed: bool = False
    file_last_mentioned_turn: int = 0
    language: str = "fa"
    conversation_phase: str = "opening"
    previous_topics: List[str] = field(default_factory=list)


def _detect_language(text: str) -> str:
    """Detect fa vs en from character distribution."""
    persian_chars = len(re.findall(r"[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]", text))
    latin_chars = len(re.findall(r"[a-zA-Z]", text))
    if persian_chars + latin_chars == 0:
        return "fa"
    return "fa" if persian_chars >= latin_chars else "en"


def _classify_topic(
    text: str,
    has_file_context: bool,
    file_context_type: Optional[str] = None,
) -> str:
    """Classify a message's topic from keywords and context.

    Note on courtesy prefixes:
        A leading "سلام" / "hello" / "ممنون" must not hide the real topic.
        We classify on the *content* that follows the greeting; only when
        the message is *purely* a greeting/thanks do we return those classes.
        This is the fix for the "سلام دانته نویسنده ..." drift bug.
    """
    text_lower = text.lower().strip()
    inner = _strip_leading_courtesy(text_lower)
    courtesy_stripped = inner != text_lower

    # Pure courtesy: stripping yielded nothing useful → message *is* the courtesy.
    if not courtesy_stripped:
        if _GREETING_PATTERNS.match(text_lower):
            return "greeting"
        if _THANKS_PATTERNS.match(text_lower):
            return "thanks"

    target = inner if courtesy_stripped else text_lower

    if _SEARCH_KEYWORDS.search(target):
        return "search"
    if has_file_context or _FILE_REFERENCE_KEYWORDS.search(target):
        if file_context_type in ("image", "vision"):
            return "image_analysis"
        if file_context_type == "audio":
            return "audio_analysis"
        if file_context_type == "video":
            return "video_analysis"
        return "file_analysis"
    if _FILE_KEYWORDS.search(target):
        return "file_discussion"
    return "general_chat"


def _is_greeting(text: str) -> bool:
    return bool(_GREETING_PATTERNS.match(text.strip()))


def analyze_conversation(
    chat_history: List[Dict[str, Any]],
    current_message: str,
    has_file_context: bool = False,
    file_context_type: Optional[str] = None,
) -> ConversationState:
    """
    Analyze chat history and current message to produce ConversationState.

    Args:
        chat_history: List of {"role": ..., "content": ...} dicts (oldest first)
        current_message: The new user message (not yet in chat_history)
        has_file_context: Whether a file/artifact is attached to this turn
        file_context_type: Type of attached file (image, audio, video, ...)

    Returns:
        ConversationState with all fields populated.
    """
    # --- Count turns (each user msg = 1 turn) ---
    user_messages = [m for m in chat_history if m.get("role") == "user"]
    turn_number = len(user_messages) + 1  # +1 for the current message

    # --- Detect greeting in history ---
    greeting_done = False
    for msg in chat_history[:6]:  # only check first 6 messages (3 turns)
        content = msg.get("content", "")
        if _is_greeting(content):
            greeting_done = True
            break

    # --- Track file mentions per turn ---
    file_discussed = False
    file_last_mentioned_turn = 0
    turn_idx = 0
    for msg in chat_history:
        if msg.get("role") == "user":
            turn_idx += 1
        content = msg.get("content", "")
        has_att = bool(msg.get("attachments"))
        if has_att or _FILE_KEYWORDS.search(content) or _FILE_REFERENCE_KEYWORDS.search(content):
            file_discussed = True
            file_last_mentioned_turn = turn_idx

    # also check current message
    if has_file_context or _FILE_KEYWORDS.search(current_message) or _FILE_REFERENCE_KEYWORDS.search(current_message):
        file_discussed = True
        file_last_mentioned_turn = turn_number

    # --- Topic classification ---
    current_topic = _classify_topic(current_message, has_file_context, file_context_type)

    # Determine previous topic from last assistant+user exchange
    previous_topics: List[str] = []
    prev_topic = "general_chat"
    for msg in chat_history:
        if msg.get("role") == "user":
            t = _classify_topic(msg.get("content", ""), bool(msg.get("attachments")))
            if t not in ("greeting", "thanks"):
                prev_topic = t
                if t not in previous_topics:
                    previous_topics.append(t)

    topic_changed = (
        current_topic != prev_topic
        and current_topic not in ("greeting", "thanks")
        and prev_topic not in ("greeting", "thanks")
        and turn_number > 1
    )

    # Content-based shift detector: even if both turns classify the same
    # (e.g. both "general_chat"), zero token overlap with the recent tail
    # means the user truly switched subjects. This catches "سلام دانته ..."
    # after a long image discussion where the classifier alone can't tell.
    if (
        not topic_changed
        and turn_number > 1
        and current_topic not in ("greeting", "thanks")
    ):
        if _detect_topic_shift_by_content(chat_history, current_message):
            topic_changed = True

    # --- Conversation phase ---
    if turn_number <= 1:
        phase = "opening"
    elif turn_number <= 3:
        phase = "early"
    elif topic_changed:
        phase = "topic_switch"
    else:
        phase = "active"

    # --- Language ---
    language = _detect_language(current_message)

    state = ConversationState(
        turn_number=turn_number,
        greeting_done=greeting_done,
        active_topic=current_topic,
        topic_changed=topic_changed,
        file_discussed=file_discussed,
        file_last_mentioned_turn=file_last_mentioned_turn,
        language=language,
        conversation_phase=phase,
        previous_topics=previous_topics,
    )

    log.debug(
        f"ConversationState: turn={state.turn_number} phase={state.conversation_phase} "
        f"topic={state.active_topic} changed={state.topic_changed} "
        f"greeting_done={state.greeting_done} file_discussed={state.file_discussed}"
    )
    return state



# ==================== File Reference Resolution ====================
# LLM-first approach: the LLM resolves which file(s) the user means.
# Heuristic regex is ONLY a fast fallback when LLM is unavailable.

import asyncio
import json as _json

_FILE_REF_RESOLVER_PROMPT = """You are a file-reference resolver.
Given a user message and the chronological list of files in the session,
determine which file(s) the user is referring to.

Rules:
- If the user is NOT talking about any file at all, return target="none".
- "this file" / "این فایل" / "همین" usually means the LATEST file.
- Ordinal references ("first image", "اولین عکس") refer to upload order.
- "all files" / "همه فایل‌ها" / "compare" / "مقایسه" means ALL files.
- If ambiguous, prefer the latest file.
- Respond with JSON ONLY. No explanation.

JSON schema:
{"target":"specific|all|none","file_indices":[1],"intent":"analyze|compare|summarize|discuss","file_type_filter":null}

- target: "specific" = one or more specific files, "all" = every file, "none" = not about files
- file_indices: list of ordinal numbers (#) from the file list (only when target="specific")
- intent: what the user wants to do with the file(s)
- file_type_filter: "image"|"audio"|"video"|"document"|null"""


@dataclass
class FileReferenceIntent:
    """What the user intends regarding files in the current message."""
    target: str = "none"          # specific | all | none
    file_indices: List[int] = field(default_factory=list)
    file_type_filter: Optional[str] = None
    is_multi_file_analysis: bool = False
    intent: str = "none"          # analyze | compare | summarize | discuss | none
    resolved_by: str = "none"     # "llm" | "heuristic" | "none"


async def smart_resolve_file_reference(
    message: str,
    file_registry: List[Dict[str, Any]],
    llm_client=None,
    timeout: float = 2.0,
) -> FileReferenceIntent:
    """
    LLM-first file reference resolution.

    1. Sends user message + file registry to LLM for intelligent resolution.
    2. Falls back to lightweight heuristic only on timeout/failure.
    """
    if not file_registry:
        return FileReferenceIntent()

    # --- Primary: LLM resolution ---
    if llm_client:
        try:
            intent = await asyncio.wait_for(
                _llm_resolve(message, file_registry, llm_client),
                timeout=timeout,
            )
            if intent:
                intent.resolved_by = "llm"
                log.info(
                    f"🧠 LLM file-ref: target={intent.target} "
                    f"indices={intent.file_indices} intent={intent.intent}"
                )
                return intent
        except asyncio.TimeoutError:
            log.warning("LLM file-ref resolution timed out, falling back to heuristic")
        except Exception as e:
            log.warning(f"LLM file-ref resolution failed: {e}, falling back to heuristic")

    # --- Fallback: heuristic ---
    intent = _heuristic_resolve(message, file_registry)
    intent.resolved_by = "heuristic"
    return intent


async def _llm_resolve(
    message: str,
    file_registry: List[Dict[str, Any]],
    llm_client,
) -> Optional[FileReferenceIntent]:
    """Use LLM to understand which file(s) the user is referring to."""
    file_lines = "\n".join(
        f"  #{f['ordinal']} {f['name']} — {f.get('type', '?')} (upload turn {f['turn']})"
        for f in file_registry
    )

    user_prompt = (
        f"Session files (chronological):\n{file_lines}\n\n"
        f"User message: {message}\n\n"
        "Which file(s) is the user referring to? JSON only."
    )

    result = await llm_client.chat(
        messages=[
            {"role": "system", "content": _FILE_REF_RESOLVER_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    if not result.get("ok"):
        return None

    raw = result.get("content", "").strip()
    data = _extract_json(raw)
    if not data:
        return None

    target = data.get("target", "none")
    file_indices = data.get("file_indices") or []
    intent_str = data.get("intent", "none")
    ftype = data.get("file_type_filter")

    if target == "none":
        return FileReferenceIntent(target="none", intent=intent_str)

    is_multi = (
        target == "all"
        or len(file_indices) > 1
        or intent_str in ("compare", "summarize")
    )

    return FileReferenceIntent(
        target=target,
        file_indices=file_indices,
        file_type_filter=ftype,
        is_multi_file_analysis=is_multi,
        intent=intent_str,
    )


_ORDINAL_MAP_FA = {
    "اولین": 1, "اولی": 1, "اول": 1, "یکم": 1, "نخستین": 1,
    "دومین": 2, "دومی": 2, "دوم": 2,
    "سومین": 3, "سومی": 3, "سوم": 3,
    "چهارمین": 4, "چهارمی": 4, "چهارم": 4,
    "پنجمین": 5, "پنجمی": 5, "پنجم": 5,
    "آخرین": -1, "آخری": -1, "آخر": -1,
}

_ORDINAL_MAP_EN = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "last": -1, "latest": -1, "most recent": -1,
}

_ALL_FILES_PATTERN = re.compile(
    r"(همه\s*فایل|تمام\s*فایل|کل\s*فایل|all\s*files|every\s*file|مقایسه|compare)",
    re.IGNORECASE | re.UNICODE,
)


def _heuristic_resolve(
    message: str,
    file_registry: List[Dict[str, Any]],
) -> FileReferenceIntent:
    """
    Fast regex fallback — used ONLY when LLM is unavailable.
    Handles ordinal references, "all files", and defaults to latest.
    """
    if not file_registry:
        return FileReferenceIntent()

    text = message.strip().lower()

    if _ALL_FILES_PATTERN.search(text):
        return FileReferenceIntent(
            target="all",
            file_indices=[f["ordinal"] for f in file_registry],
            is_multi_file_analysis=True,
            intent="analyze",
        )

    for word, ordinal in _ORDINAL_MAP_FA.items():
        if word in text:
            if ordinal == -1:
                idx = file_registry[-1]["ordinal"]
            else:
                idx = ordinal if ordinal <= len(file_registry) else file_registry[-1]["ordinal"]
            return FileReferenceIntent(
                target="specific",
                file_indices=[idx],
                intent="analyze",
            )

    for word, ordinal in _ORDINAL_MAP_EN.items():
        if word in text:
            if ordinal == -1:
                idx = file_registry[-1]["ordinal"]
            else:
                idx = ordinal if ordinal <= len(file_registry) else file_registry[-1]["ordinal"]
            return FileReferenceIntent(
                target="specific",
                file_indices=[idx],
                intent="analyze",
            )

    if _FILE_REFERENCE_KEYWORDS.search(text) or _FILE_KEYWORDS.search(text):
        return FileReferenceIntent(
            target="specific",
            file_indices=[file_registry[-1]["ordinal"]],
            intent="analyze",
        )

    # Deictic without explicit file keyword — defer to DiscourseFocus, not latest file
    try:
        from .discourse_focus import has_deictic_reference
        if has_deictic_reference(message):
            return FileReferenceIntent(target="none", intent="analyze")
    except ImportError:
        pass

    return FileReferenceIntent()


def resolve_artifacts_from_intent(
    intent: FileReferenceIntent,
    file_registry: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Given a FileReferenceIntent and the session file registry,
    return the artifact dicts that should be loaded.
    """
    if not file_registry or intent.target == "none":
        return []

    if intent.target == "all":
        pool = file_registry
        if intent.file_type_filter:
            filtered = [f for f in file_registry if f.get("type") == intent.file_type_filter]
            pool = filtered or file_registry
        return list(pool)

    if intent.target == "specific" and intent.file_indices:
        result = []
        for idx in intent.file_indices:
            for f in file_registry:
                if f.get("ordinal") == idx:
                    result.append(f)
                    break
        return result if result else [file_registry[-1]]

    return [file_registry[-1]]


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response (handles markdown fences)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        return _json.loads(cleaned)
    except _json.JSONDecodeError:
        import re as _re
        m = _re.search(r"\{[^{}]*\}", cleaned, _re.DOTALL)
        if m:
            try:
                return _json.loads(m.group(0))
            except _json.JSONDecodeError:
                pass
    return None


def message_references_files(text: str) -> bool:
    """Public helper: True iff ``text`` contains an explicit file mention.

    Used by the orchestrator to decide whether to auto-attach previous
    session artifacts on a follow-up turn. Without this gate the previous
    image is re-attached to *every* turn — including unrelated questions —
    which is one of the root causes of the admin/admin123 drift.
    """
    if not text:
        return False
    body = _strip_leading_courtesy(text.lower().strip()) or text.lower().strip()
    if _FILE_REFERENCE_KEYWORDS.search(body) or _FILE_KEYWORDS.search(body):
        return True
    # Deictic-only references («این کارش بکن») — resolved via DiscourseFocus
    try:
        from .discourse_focus import has_deictic_reference
        return bool(has_deictic_reference(text))
    except ImportError:
        return False


def message_references_files_or_deictic(text: str) -> bool:
    """Alias used by session_workspace — explicit file mention or deictic «this/این»."""
    return message_references_files(text)


__all__ = [
    "ConversationState",
    "analyze_conversation",
    "FileReferenceIntent",
    "smart_resolve_file_reference",
    "resolve_artifacts_from_intent",
    "message_references_files",
]
