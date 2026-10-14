"""
File Context Detection
Smart detection of file context using hybrid keyword + LLM approach

Key techniques:
1. Direct keyword matching (high confidence)
2. Demonstrative + action patterns (medium confidence)  
3. Question pattern detection (medium confidence)
4. Short message file noun detection (low confidence)
5. LLM-based classification (async, for ambiguous cases)

📦 Tool lists are now dynamic — loaded from tool-runner service at startup.
   No more hardcoded tool names!
"""
import mimetypes
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import asyncio

log = logging.getLogger(__name__)


# ==========================================
# Dynamic Tool Cache — populated from tool-runner
# ==========================================

# این cache در startup توسط main.py پر میشه
_tools_by_category_cache: Dict[str, List[str]] = {}
_file_based_tools_cache: Set[str] = set()


def set_tools_by_category_cache(data: Dict[str, List[str]]) -> None:
    """
    Set the tools-by-category cache.
    Called from main.py during startup.
    
    Args:
        data: {"image": ["vision_info", ...], "audio": [...], ...}
    """
    global _tools_by_category_cache
    _tools_by_category_cache = data
    log.info(f"✅ Tools by category cache set: {list(data.keys())}")


def set_file_based_tools_cache(tools: List[str]) -> None:
    """
    Set the file-based tools cache.
    Called from main.py during startup.
    
    Args:
        tools: List of tool IDs that require a file
    """
    global _file_based_tools_cache
    _file_based_tools_cache = set(tools)
    log.info(f"✅ File-based tools cache set: {len(_file_based_tools_cache)} tools")


def get_tools_by_category_cache() -> Dict[str, List[str]]:
    """Get the cached tools-by-category mapping."""
    return _tools_by_category_cache


# ==========================================
# LLM-based File Reference Detection
# ==========================================
async def llm_file_reference_check(message: str, llm_client=None) -> Tuple[bool, float, str]:
    """
    Use LLM (Qwen3) to determine if message references a file.
    
    This is more accurate than keyword matching but slower.
    Use for ambiguous cases or as a verification step.
    
    Args:
        message: User message to classify
        llm_client: Optional GatewayLLMClient instance
        
    Returns:
        Tuple of (references_file, confidence, reasoning)
    """
    if llm_client is None:
        return False, 0.0, "No LLM client available"
    
    prompt = f"""Analyze this message and determine if it's asking about or referring to an uploaded file, image, audio, or document.

Message: "{message}"

Answer with ONLY one of these:
- YES: if the message is asking about, referring to, or requesting analysis of a file/image/audio/video
- NO: if the message is a general greeting, question, or conversation not about any file

Examples:
- "این عکس چیه؟" → YES (asking about an image)
- "توضیح بده" → YES (requesting description of something visual)
- "سلام" → NO (greeting)
- "اسم من علی هست" → NO (self-introduction)
- "what is in this picture?" → YES (asking about image)
- "hello how are you?" → NO (greeting)

Your answer (YES or NO):"""

    try:
        response = await llm_client.generate(
            prompt=prompt,
            max_tokens=10,
            temperature=0.1,
        )
        
        answer = response.strip().upper()
        
        if "YES" in answer:
            return True, 0.9, "LLM classified as file reference"
        elif "NO" in answer:
            return False, 0.9, "LLM classified as non-file reference"
        else:
            return False, 0.5, f"LLM gave unclear answer: {answer}"
            
    except Exception as e:
        log.warning(f"LLM file reference check failed: {e}")
        return False, 0.0, f"Error: {e}"


async def smart_file_reference_check(
    message: str, 
    llm_client=None,
    use_llm_for_uncertain: bool = True
) -> Tuple[bool, float, str]:
    """
    Smart file reference detection combining keyword + LLM.
    
    Strategy:
    1. First, run fast keyword-based detection
    2. If confident (True), return immediately
    3. If uncertain, optionally use LLM for verification
    
    Args:
        message: User message
        llm_client: Optional GatewayLLMClient for LLM-based check
        use_llm_for_uncertain: Whether to use LLM for ambiguous cases
        
    Returns:
        Tuple of (references_file, confidence, method)
    """
    # Step 1: Fast keyword-based detection
    keyword_result = message_references_file(message)
    
    if keyword_result:
        # High confidence positive - return immediately
        return True, 0.95, "keyword_detection"
    
    # Step 2: For negative results, check if message is ambiguous
    # Ambiguous = short message that might be a file reference
    msg_clean = message.strip()
    word_count = len(msg_clean.split())
    
    # Very short messages (1-3 words) that aren't clear negatives might be file refs
    ambiguous = word_count <= 3 and not _is_clear_non_file_message(message)
    
    if ambiguous and use_llm_for_uncertain and llm_client is not None:
        # Step 3: Use LLM for verification
        llm_result, llm_confidence, llm_reason = await llm_file_reference_check(message, llm_client)
        return llm_result, llm_confidence, f"llm_verified: {llm_reason}"
    
    return False, 0.85, "keyword_negative"


def _is_clear_non_file_message(message: str) -> bool:
    """Check if message is clearly NOT a file reference (greetings, self-intro, etc.)"""
    msg_lower = message.lower().strip()
    
    # Clear greetings
    greetings = [
        'سلام', 'درود', 'صبح بخیر', 'شب بخیر', 'خوبی', 'چطوری',
        'hello', 'hi', 'hey', 'good morning', 'good night', 'how are you'
    ]
    
    # Self-introductions
    intros = [
        'اسم من', 'نام من', 'من هستم', 'my name is', 'i am', "i'm"
    ]
    
    # General questions
    general_q = [
        'چکار میکنی', 'کی هستی', 'what can you do', 'who are you', 'help me'
    ]
    
    for g in greetings:
        if msg_lower.startswith(g) or msg_lower == g:
            return True
            
    for i in intros:
        if i in msg_lower:
            return True
            
    for q in general_q:
        if q in msg_lower:
            return True
    
    return False


# ==========================================
# Optional: Semantic Similarity Support (Legacy)
# ==========================================
_sentence_model = None
_file_reference_embeddings = None
_semantic_init_tried = False
_semantic_available = False

def _init_semantic_model():
    """Lazily initialize sentence transformer model for semantic similarity."""
    global _sentence_model, _file_reference_embeddings, _semantic_init_tried, _semantic_available
    
    # Only try once
    if _semantic_init_tried:
        return _semantic_available
    
    _semantic_init_tried = True
        
    try:
        from sentence_transformers import SentenceTransformer
        import os
        
        # Force CPU to avoid GPU memory issues
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # Use multilingual model for Persian/English support
        _sentence_model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2',
            device='cpu'
        )
        
        # Pre-compute embeddings for file reference sentences
        file_reference_examples = [
            # Persian
            "این عکس چیه؟",
            "این تصویر را تحلیل کن",
            "در این فایل چی هست؟",
            "توضیح بده این چیه",
            "متن این عکس را بخوان",
            "چهره های توی این عکس کیا هستن؟",
            # English
            "What is in this image?",
            "Describe this picture",
            "Analyze this file",
            "What does this photo show?",
            "Read the text in this image",
            "Who are the people in this picture?",
        ]
        _file_reference_embeddings = _sentence_model.encode(file_reference_examples)
        
        _semantic_available = True
        log.info("✅ Semantic model loaded for file reference detection (CPU mode)")
        return True
    except ImportError:
        log.debug("sentence-transformers not available, using keyword-only detection")
        return False
    except Exception as e:
        log.warning(f"Failed to load semantic model: {e}")
        return False


def semantic_file_reference_score(message: str, threshold: float = 0.65) -> Tuple[bool, float]:
    """
    Check if message semantically refers to a file using embedding similarity.
    
    NOTE: This is a FALLBACK method. Keyword detection is preferred.
    Only use semantic similarity for ambiguous cases.
    
    Args:
        message: User message
        threshold: Minimum similarity score (0-1)
        
    Returns:
        Tuple of (is_file_reference, max_similarity_score)
    """
    if not _init_semantic_model() or _sentence_model is None:
        return False, 0.0
    
    try:
        import numpy as np
        msg_embedding = _sentence_model.encode([message])
        similarities = np.dot(msg_embedding, _file_reference_embeddings.T)[0]
        max_sim = float(np.max(similarities))
        
        # Higher threshold to avoid false positives
        return max_sim >= threshold, max_sim
    except Exception as e:
        log.debug(f"Semantic check failed: {e}")
        return False, 0.0


# ==========================================
# MIME Type Categories
# ==========================================


# دسته‌بندی MIME types
MIME_CATEGORIES = {
    "image": [
        "image/jpeg", "image/png", "image/gif", "image/webp", 
        "image/bmp", "image/tiff", "image/svg+xml"
    ],
    "audio": [
        "audio/mpeg", "audio/wav", "audio/ogg", "audio/flac",
        "audio/aac", "audio/mp4", "audio/webm", "audio/x-wav"
    ],
    "video": [
        "video/mp4", "video/webm", "video/avi", "video/mkv",
        "video/mov", "video/quicktime", "video/x-msvideo"
    ],
    "document": [
        "application/pdf", "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain", "text/markdown", "text/csv"
    ]
}


# FILE_BASED_TOOLS حذف شد - الان از _file_based_tools_cache استفاده میشه
# که در startup از tool-runner پر میشه


def get_mime_category(mime_type: str) -> Optional[str]:
    """دسته‌بندی MIME type."""
    if not mime_type:
        return None
    
    mime_lower = mime_type.lower()
    
    for category, types in MIME_CATEGORIES.items():
        if mime_lower in types:
            return category
    
    # fallback بر اساس prefix
    if mime_lower.startswith("image/"):
        return "image"
    elif mime_lower.startswith("audio/"):
        return "audio"
    elif mime_lower.startswith("video/"):
        return "video"
    elif mime_lower.startswith("text/"):
        return "document"
    
    return None


def detect_file_context(artifacts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    تشخیص context فایل از artifacts.
    
    Args:
        artifacts: لیست artifacts با path و mime_type
    
    Returns:
        Dict با type, name, path و اطلاعات اضافی یا None
        اگر چند فایل باشه، all_files هم برگردانده می‌شود
    """
    if not artifacts:
        return None
    
    # ==========================================
    # Process ALL artifacts for multi-file support
    # ==========================================
    all_files = []
    for artifact in artifacts:
        file_info = _process_single_artifact(artifact)
        if file_info:
            all_files.append(file_info)
    
    if not all_files:
        return None
    
    # Primary file context (first one - backward compatible)
    primary = all_files[0]
    
    result = {
        # کلیدهای اصلی (از فایل اول)
        "type": primary["type"],
        "name": primary["name"],
        "path": primary["path"],
        "mime_type": primary.get("mime_type"),
        "artifact_id": primary.get("artifact_id"),
        # کلید جدید: همه فایل‌ها
        "all_files": all_files,
        "file_count": len(all_files),
        "is_multi_file": len(all_files) > 1,
    }
    
    log.info(f"📁 File context: {len(all_files)} file(s), primary: {primary['type']} - {primary['path']}")
    return result


def _process_single_artifact(artifact: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single artifact and return file info."""
    # Support multiple path field names: path, file_path, storage_path
    file_path = artifact.get("path") or artifact.get("file_path") or artifact.get("storage_path")
    mime_type = artifact.get("mime_type") or artifact.get("content_type")
    # Support multiple filename fields: name, filename, original_filename
    file_name = artifact.get("name") or artifact.get("filename") or artifact.get("original_filename")
    file_type = artifact.get("type")  # ممکنه مستقیم داده شده باشه
    
    # Check for auto-detected extension info
    extension_detected = artifact.get("_extension_detected", False)
    original_filename_no_ext = artifact.get("_original_filename_no_ext")
    detected_mime = artifact.get("_detected_mime")
    extension_unknown = artifact.get("_extension_unknown", False)
    
    if not file_path:
        return None
    
    # Normalize path: keep minio:// URLs as-is, normalize local paths
    if file_path.startswith("minio://"):
        # MinIO path - keep as-is
        pass
    elif not file_path.startswith("/"):
        if file_path.startswith("uploads/"):
            file_path = f"/data/{file_path}"
        else:
            file_path = f"/data/uploads/{file_path}"
    
    # اگر mime_type نداریم، حدس بزن
    if not mime_type:
        mime_type, _ = mimetypes.guess_type(file_path)
    
    # اگر type مستقیم داده نشده، از category استفاده کن
    category = file_type or get_mime_category(mime_type)
    
    # اگه هنوز نداریم، از name حدس بزن
    if not category and file_name:
        ext = file_name.lower().split('.')[-1] if '.' in file_name else ''
        if ext in ('jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff'):
            category = "image"
        elif ext in ('mp3', 'wav', 'ogg', 'flac', 'm4a'):
            category = "audio"
        elif ext in ('mp4', 'mkv', 'avi', 'mov', 'webm'):
            category = "video"
        elif ext in ('pdf', 'doc', 'docx', 'txt', 'md'):
            category = "document"
    
    # اگه هنوز category نداریم، unknown
    if not category:
        has_ext = "." in (file_name or "")
        if not has_ext:
            category = "unknown"
        else:
            category = "document"  # default
    
    result = {
        "type": category,
        "name": file_name or file_path.split('/')[-1],
        "path": file_path,
        "mime_type": mime_type,
        "artifact_id": artifact.get("id") or artifact.get("artifact_id") or artifact.get("file_id"),
        "has_extension": "." in (file_name or file_path.split('/')[-1]),
    }
    
    # Add auto-detection info if available
    if extension_detected:
        result["extension_detected"] = True
        result["original_name_no_ext"] = original_filename_no_ext
        result["detected_mime"] = detected_mime
    elif extension_unknown:
        result["extension_unknown"] = True
    
    return result


def get_relevant_tools(file_context: Optional[Dict[str, Any]]) -> List[str]:
    """
    دریافت ابزارهای مرتبط با context فایل.
    
    از cache داینامیک استفاده می‌کنه که در startup از tool-runner پر شده.
    
    Args:
        file_context: خروجی detect_file_context
    
    Returns:
        لیست tool_ids مرتبط
    """
    if not file_context:
        return []
    
    # از "type" استفاده کن (کلید جدید)
    category = file_context.get("type") or file_context.get("category")
    
    # از cache استفاده کن - نه hardcoded dict
    return _tools_by_category_cache.get(category, [])


def is_file_based_tool(tool_id: str) -> bool:
    """آیا ابزار file-based است؟ از cache داینامیک استفاده می‌کنه."""
    return tool_id in _file_based_tools_cache


def format_file_context_for_prompt(file_context: Optional[Dict[str, Any]]) -> str:
    """فرمت کردن context فایل برای پرامپت."""
    if not file_context:
        return ""
    
    # از "type" استفاده کن (کلید جدید)
    category = file_context.get("type") or file_context.get("category", "unknown")
    name = file_context.get("name") or file_context.get("path", "")
    
    category_messages = {
        "image": f"🖼️ تصویر ضمیمه شده: {name}",
        "audio": f"🎵 فایل صوتی ضمیمه شده: {name}",
        "video": f"🎬 ویدیو ضمیمه شده: {name}",
        "document": f"📄 سند ضمیمه شده: {name}"
    }
    
    return category_messages.get(category, f"📎 فایل ضمیمه شده: {name}")


def message_references_file(message: str) -> bool:
    """
    Intelligent detection of file/image references in messages.
    
    Uses a hybrid approach:
    1. Direct file keyword detection (high confidence)
    2. Demonstrative + action pattern (medium confidence)
    3. Question words + context (low confidence - needs demonstrative)
    
    The key insight: Most file references contain BOTH:
    - A demonstrative (this, این, اون) or possessive (my, آن)
    - A file-related noun (image, عکس, photo) OR
    - An action that requires a file (describe, توضیح بده)
    
    This avoids false positives like:
    - "My name is X" (no file noun, no file action)
    - "Hello" (no demonstrative, no file context)
    
    Args:
        message: User message text
        
    Returns:
        True if message references a file
    """
    if not message:
        return False
    
    msg = message.strip()
    msg_lower = msg.lower()
    
    # ==========================================
    # Layer 1: Direct file keywords (HIGH confidence)
    # Words that almost always mean "the file"
    # ==========================================
    direct_file_keywords = [
        # Persian - direct references
        'این عکس', 'این تصویر', 'این فایل', 'این صوت', 'این ویدیو',
        'اون عکس', 'اون تصویر', 'همین عکس', 'همین تصویر',
        'عکس بالا', 'تصویر بالا', 'فایل بالا',
        'عکسی که', 'تصویری که', 'فایلی که',
        'عکس رو', 'تصویر رو', 'فایل رو',
        'عکس را', 'تصویر را', 'فایل را',
        # English - direct references
        'this image', 'this picture', 'this photo', 'this file',
        'this audio', 'this video', 'this document',
        'the image', 'the picture', 'the photo', 'the file',
        'above image', 'above picture', 'above photo',
        'attached image', 'attached file', 'attached photo',
        'uploaded image', 'uploaded file', 'uploaded photo',
    ]
    
    for keyword in direct_file_keywords:
        if keyword in msg_lower or keyword in msg:
            log.debug(f"📎 File reference: direct keyword '{keyword}'")
            return True
    
    # ==========================================
    # Layer 2: Demonstrative + File action patterns (MEDIUM confidence)
    # "این چیه" + file action context
    # ==========================================
    
    # Persian demonstratives
    persian_demonstratives = ['این', 'اون', 'آن', 'همین', 'همون']
    # English demonstratives 
    english_demonstratives = ['this', 'that', 'these', 'those', 'it']
    
    has_persian_demo = any(d in msg for d in persian_demonstratives)
    has_english_demo = any(d in msg_lower.split() for d in english_demonstratives)
    
    # File action verbs - require a file to make sense
    persian_file_actions = [
        'توضیح', 'تحلیل', 'بررسی', 'آنالیز', 'تشخیص', 'شناسایی',
        'استخراج', 'نوشته', 'متن', 'چهره', 'صورت', 'افراد'
    ]
    english_file_actions = [
        'describe', 'analyze', 'analyse', 'detect', 'identify',
        'extract', 'recognize', 'read', 'transcribe', 'ocr',
        'faces', 'people', 'objects', 'text in'
    ]
    
    has_persian_action = any(a in msg for a in persian_file_actions)
    has_english_action = any(a in msg_lower for a in english_file_actions)
    
    # Pattern: Demonstrative + File Action
    if (has_persian_demo and has_persian_action):
        log.debug("📎 File reference: Persian demonstrative + action")
        return True
    
    if (has_english_demo and has_english_action):
        log.debug("📎 File reference: English demonstrative + action")
        return True
    
    # ==========================================
    # Layer 2.5: Standalone file action commands (MEDIUM confidence)
    # Very short messages with just action verbs like "توضیح بده"
    # ==========================================
    standalone_action_patterns = [
        # Persian standalone actions (MUST be short/focused)
        r'^(توضیح|تحلیل|بررسی|آنالیز|تشخیص|شناسایی)\s*(بده|کن|بکن)$',
        r'^(متن|نوشته|چهره|صورت)\s*(رو|را|ها)?\s*(بخون|استخراج|شناسایی)\s*(کن|بده)?$',
        r'^(چهره|صورت|افراد)[\u200c\s]*(ها)?[\u200c\s]*(رو|را)?[\u200c\s]*(شناسایی|تشخیص)[\u200c\s]*(کن|بده)?$',
        # Also match "چهره‌ها رو شناسایی کن" anywhere in message
        r'(چهره|صورت|افراد)[\u200c\s]*(ها)?[\u200c\s]*(رو|را)?[\u200c\s]*(شناسایی|تشخیص)',
        # English standalone actions (MUST include "it" or "this")
        r'^(describe|analyze|explain|detect|identify|read)\s+(it|this)$',
    ]
    
    for pattern in standalone_action_patterns:
        if re.search(pattern, msg.strip(), re.IGNORECASE):
            log.debug(f"📎 File reference: standalone action pattern")
            return True
    
    # ==========================================
    # Layer 3: Question words + "this" context (MEDIUM confidence)
    # "این چیه؟" or "What is this?"
    # ==========================================
    
    # Persian question patterns with demonstrative
    persian_question_patterns = [
        r'این\s*چی', r'این\s*چه', r'اون\s*چی', r'اون\s*چه',
        r'چی\s*(توی|در|داخل)', r'کی\s*(توی|در|داخل)',
        r'چه\s*چیزی', r'چه\s*کسی',
    ]
    
    # English question patterns with demonstrative
    english_question_patterns = [
        r'what\s*(is|\'s|do\s*you\s*see)\s*(this|that|in|here)',
        r'who\s*(is|\'s)\s*(this|that|in)',
        r'can\s*you\s*(see|describe|analyze)',
        r'tell\s*me\s*(about|what)',
        r'what\s*do\s*you\s*see',
        r'(see|look\s*at)\s*(here|this)',
    ]
    
    for pattern in persian_question_patterns:
        if re.search(pattern, msg, re.IGNORECASE):
            log.debug(f"📎 File reference: Persian question pattern '{pattern}'")
            return True
            
    for pattern in english_question_patterns:
        if re.search(pattern, msg_lower, re.IGNORECASE):
            log.debug(f"📎 File reference: English question pattern '{pattern}'")
            return True
    
    # ==========================================
    # Layer 4: File type nouns WITHOUT demonstrative (LOW confidence)
    # Only if the message is very short and focused
    # ==========================================
    
    # Very short messages (< 4 words) with file nouns are likely references
    word_count = len(msg.split())
    if word_count <= 5:
        short_msg_file_nouns = [
            'عکس', 'تصویر', 'فایل', 'image', 'picture', 'photo', 'file'
        ]
        for noun in short_msg_file_nouns:
            if noun in msg or noun in msg_lower:
                # Check it's not part of a name introduction
                intro_patterns = ['اسم', 'نام', 'name is', 'i am', 'i\'m', 'my name']
                if not any(intro in msg_lower for intro in intro_patterns):
                    log.debug(f"📎 File reference: short message with file noun '{noun}'")
                    return True
    
    # ==========================================
    # Layer 5: Semantic Similarity (FALLBACK - DISABLED FOR NOW)
    # Use sentence embeddings if available
    # NOTE: Currently disabled due to high false positive rate
    # Re-enable after better model tuning
    # ==========================================
    # Disabled: semantic similarity was causing too many false positives
    # is_semantic_ref, similarity = semantic_file_reference_score(message, threshold=0.75)
    # if is_semantic_ref:
    #     log.debug(f"📎 File reference: semantic similarity {similarity:.2f}")
    #     return True
    
    # No file reference detected
    return False


def is_followup_instruction(message: str) -> bool:
    """
    Detect if a message is a follow-up/continuation instruction
    that implicitly refers to the previous context (including files).
    
    Examples:
    - "بله همه این موارد رو انجام بده" (yes do all these things)
    - "آره انجام بده" (yeah do it)
    - "باشه" (ok)
    - "ادامه بده" (continue)
    - "yes, go ahead"
    - "do it"
    
    This function does NOT check if previous context actually had files.
    The caller should only use this when there are no current artifacts
    and should combine it with checking session attachments.
    """
    if not message:
        return False
    
    msg = message.strip()
    msg_lower = msg.lower().strip()
    
    # ==========================================
    # Layer 1: Pure affirmations (standalone)
    # ==========================================
    pure_affirmations = [
        # Persian
        'بله', 'آره', 'آری', 'اوکی', 'باشه', 'حتما', 'بلی',
        'اره', 'ok', 'اکی', 'خب', 'خوبه',
        # English
        'yes', 'yeah', 'yep', 'yup', 'ok', 'okay', 'sure',
        'go ahead', 'do it', 'please', 'alright', 'right',
        'proceed', 'confirm', 'agreed', 'affirmative',
    ]
    
    # Check standalone (exact match or with punctuation)
    msg_clean = re.sub(r'[.!?،؟\s]+$', '', msg_lower).strip()
    if msg_clean in pure_affirmations:
        log.debug(f"📎 Follow-up: pure affirmation '{msg_clean}'")
        return True
    
    # ==========================================
    # Layer 2: Affirmation + action patterns
    # ==========================================
    persian_followup_patterns = [
        # بله + action
        r'(بله|آره|آری|اره|باشه|اوکی|خب)\s*.*(انجام|اجرا|شروع|ادامه|بررسی|تحلیل)',
        # action commands (continuation)
        r'^(انجام|اجرا|شروع|ادامه)\s*(بده|کن|بکن|بزن)',
        # "همه رو انجام بده" / "همه موارد"
        r'(همه|تمام|کل)\s*.*(انجام|اجرا|بررسی|رو|را)',
        # "لطفا انجام بده"
        r'لطفا\s*(انجام|اجرا|شروع|ادامه|بررسی)',
        # "بزن بریم" / "شروع کن"
        r'(بزن\s*بریم|شروع\s*کن|ادامه\s*بده)',
        # "این کارها رو بکن" / "این موارد"
        r'(این|اون)\s*(کار|مورد|آیتم)(ها|ها\s*رو|ها\s*را)',
        # "مرحله بعد" / "قدم بعدی"
        r'(مرحله|قدم|گام)\s*(بعد|بعدی)',
    ]
    
    english_followup_patterns = [
        # affirmation + action
        r'(yes|yeah|yep|sure|ok|okay)\s*.*(do|run|start|continue|proceed|go|execute|analyze)',
        # continuation commands
        r'^(do|run|start|continue|proceed|execute|go\s*ahead|carry\s*on)',
        # "do all of them" / "all of the above"
        r'(all|every)\s*(of)?\s*(them|these|those|the\s*above|that|it)',
        # "please do it" / "please continue"
        r'please\s*(do|run|start|continue|proceed|go|execute|analyze)',
        # "next step" / "go on"
        r'(next\s*step|go\s*on|move\s*on|keep\s*going)',
    ]
    
    for pattern in persian_followup_patterns:
        if re.search(pattern, msg, re.IGNORECASE):
            log.debug(f"📎 Follow-up: Persian pattern '{pattern}'")
            return True
    
    for pattern in english_followup_patterns:
        if re.search(pattern, msg_lower, re.IGNORECASE):
            log.debug(f"📎 Follow-up: English pattern '{pattern}'")
            return True
    
    # ==========================================
    # Layer 3: Very short messages with action verbs (< 6 words)
    # ==========================================
    word_count = len(msg.split())
    if word_count <= 5:
        short_actions = [
            'انجام', 'اجرا', 'ادامه', 'بررسی', 'تحلیل',
            'do', 'run', 'execute', 'continue', 'analyze', 'start',
        ]
        for action in short_actions:
            if action in msg or action in msg_lower:
                log.debug(f"📎 Follow-up: short message with action '{action}'")
                return True
    
    return False


def extract_file_path_from_message(message: str) -> Optional[str]:
    """
    استخراج مسیر فایل از متن پیام کاربر.
    
    Args:
        message: متن پیام کاربر
        
    Returns:
        مسیر فایل یا None
    """
    if not message:
        return None
    
    # الگوهای مختلف برای تشخیص مسیر فایل
    patterns = [
        # مسیر مطلق Unix/Linux: /path/to/file.ext
        r'(/(?:data|home|tmp|var|uploads?|files?|storage)[^\s\u200c\u200d،؛:\"\'<>|\*\?]+\.\w{2,5})',
        # مسیر با /data/uploads (رایج در این سیستم)
        r'(/data/uploads/[^\s\u200c\u200d،؛:\"\'<>|\*\?]+)',
        # مسیر نسبی با نقطه: ./file.ext یا ../file.ext
        r'(\.{1,2}/[^\s\u200c\u200d،؛:\"\'<>|\*\?]+\.\w{2,5})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            path = match.group(1).strip()
            # حذف کاراکترهای اضافی از انتها
            path = re.sub(r'[،؛\.\s]+$', '', path)
            log.info(f"📁 Extracted file path from message: {path}")
            return path
    
    return None


def detect_file_context_from_message(message: str) -> Optional[Dict[str, Any]]:
    """
    تشخیص context فایل از متن پیام.
    
    Args:
        message: متن پیام کاربر
        
    Returns:
        Dict با type, name, path یا None
    """
    file_path = extract_file_path_from_message(message)
    if not file_path:
        return None
    
    # حدس زدن MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    category = get_mime_category(mime_type)
    
    # اگه از MIME نشد، از extension حدس بزن
    if not category:
        ext = file_path.lower().split('.')[-1] if '.' in file_path else ''
        if ext in ('jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff', 'heic'):
            category = "image"
        elif ext in ('mp3', 'wav', 'ogg', 'flac', 'm4a', 'aac'):
            category = "audio"
        elif ext in ('mp4', 'mkv', 'avi', 'mov', 'webm'):
            category = "video"
        elif ext in ('pdf', 'doc', 'docx', 'txt', 'md', 'csv', 'xlsx'):
            category = "document"
        else:
            category = "file"  # نوع نامشخص
    
    file_name = file_path.split('/')[-1]
    
    log.info(f"📁 File context from message: {category} - {file_path}")
    return {
        "type": category,
        "name": file_name,
        "path": file_path,
        "mime_type": mime_type,
        "source": "message"  # نشان‌دهنده اینکه از پیام extract شده
    }
