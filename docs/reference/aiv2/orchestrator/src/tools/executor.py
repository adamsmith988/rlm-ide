"""
Tool Executor - اجرای ابزارها از طریق سرویس Tool Runner
"""
import asyncio
import logging
import re
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..config import settings

log = logging.getLogger(__name__)

_SESSION_ATTACH_NOT_FOUND_MARKERS = (
    "no accessible file found",
    "artifact_not_found",
)


def _unwrap_tool_runner_payload(data: Any) -> Dict[str, Any]:
    """Return the inner tool module payload from a tool-runner HTTP envelope."""
    if not isinstance(data, dict):
        return {}
    inner = data.get("result")
    if isinstance(inner, dict) and (
        inner.get("channels") is not None
        or inner.get("ok") is not None
        or inner.get("error") is not None
    ):
        return inner
    if data.get("channels") is not None or data.get("ok") is not None:
        return data
    return {}


def _tool_failure_llm_text(failed_result: Dict[str, Any]) -> str:
    """Best-effort LLM channel text from a failed tool-runner response."""
    err_text = str(failed_result.get("error") or "")
    payload = _unwrap_tool_runner_payload(failed_result.get("data"))
    channels = payload.get("channels") if isinstance(payload.get("channels"), dict) else {}
    llm_text = str(channels.get("llm") or "") if channels else ""
    return f"{err_text}\n{llm_text}".strip()


def _is_session_attach_not_found(failed_result: Dict[str, Any]) -> bool:
    combined = _tool_failure_llm_text(failed_result).lower()
    return any(marker in combined for marker in _SESSION_ATTACH_NOT_FOUND_MARKERS)


def session_attach_stale_error_text(content: str) -> bool:
    """True when tool/history text is the terminal session_attach not-found message."""
    return "no accessible file found" in (content or "").lower()


SESSION_ATTACH_HISTORY_GUIDANCE = (
    "session_attach could not find that file in storage (earlier attempt). "
    "Retry session_attach with case_id / artifact_id / exact path, or use session_search_workspace "
    "only if files already exist in this chat. Do not loop session_files on an empty workspace."
)


# A real artifact id looks like ``art_<hex>`` or a v4 UUID. We use this
# to tell whether the model passed a real ``file_id`` or smuggled in a
# filename (which has a dot+extension and is rarely UUID-shaped).
_CHOICE_LIST_RE = re.compile(r"[A-Za-z0-9_.-]+(?:\s*\|\s*[A-Za-z0-9_.-]+)+")

_UUID_LIKE_RE = re.compile(
    r"^(?:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r"|art_[0-9a-fA-F]{6,}|artifact_[0-9a-fA-F]{6,}|file_[0-9a-fA-F]{6,})$"
)


# ==================== Agent Type Constants ====================

AGENT_TYPES = ["vision", "text", "audio", "file", "analysis", "general"]


class ToolExecutor:
    """
    کلاس اجرای ابزارها از طریق سرویس tool-runner.
    - پشتیبانی از اجرای موازی
    - مدیریت timeout
    - نرمال‌سازی آرگومان‌ها
    - پشتیبانی از Multi-Agent Architecture
    """
    
    def __init__(self, tool_runner_url: Optional[str] = None):
        self.tool_runner_url = tool_runner_url or settings.TOOL_RUNNER_URL
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self._multi_agent_cache: Optional[Dict[str, Any]] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        # Short-lived cache of ``/chat/sessions/{id}/artifacts`` listings,
        # keyed by session_id. Each entry is (artifacts_list, expires_at).
        # The TTL is intentionally tiny — long enough to cover the multi-tool
        # back-to-back hydration burst within a single chat turn, short
        # enough that a session_attach in this turn is visible to the next.
        self._session_artifacts_cache: Dict[str, Tuple[List[Dict[str, Any]], float]] = {}
        self._session_artifacts_ttl_s: float = float(
            getattr(settings, "TOOL_SESSION_ARTIFACTS_CACHE_TTL_S", 15.0)
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=300.0)
        return self._http_client

    def invalidate_session_artifacts_cache(self, session_id: Optional[str] = None) -> None:
        """Drop the cached artifact listing for a session (or all sessions).

        Call after operations that mutate the session's artifact set —
        ``session_attach`` is the obvious example.
        """
        if session_id is None:
            self._session_artifacts_cache.clear()
            return
        self._session_artifacts_cache.pop(session_id, None)

    async def _cached_session_artifacts(
        self,
        session_id: str,
        *,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Return the session's artifact listing, fetching once per TTL window."""
        import time as _time

        cached = self._session_artifacts_cache.get(session_id)
        now = _time.monotonic()
        if cached and cached[1] > now:
            return cached[0]

        try:
            client = await self._get_client()
            resp = await client.get(
                f"{settings.STORAGE_URL}/chat/sessions/{session_id}/artifacts",
                params={"limit": 1000, "offset": 0},
                headers=self._tool_runner_headers(user_id=user_id, group_id=group_id),
            )
            resp.raise_for_status()
            artifacts = resp.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("session artifacts fetch failed for %s: %s", session_id, exc)
            return None

        if not isinstance(artifacts, list):
            return None

        self._session_artifacts_cache[session_id] = (
            artifacts,
            now + self._session_artifacts_ttl_s,
        )
        return artifacts

    def _tool_runner_headers(
        self,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> Dict[str, str]:
        headers = {
            "X-Internal-Key": settings.INTERNAL_API_KEY,
            "X-Service-Name": settings.SERVICE_NAME,
        }
        if user_id:
            headers["X-User-Id"] = user_id
        if group_id:
            headers["X-Group-Id"] = group_id
        return headers
    
    async def close(self):
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
    
    async def get_tools_manifest(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """دریافت لیست ابزارهای در دسترس از سرویس tool-runner."""
        if self._tools_cache and not force_refresh:
            return self._tools_cache
        
        try:
            client = await self._get_client()
            # ``/tools`` is a lightweight index and does not include channels
            # or parameter metadata. The Cline XML prompt needs the full
            # manifest so it can filter by output channel and render args.
            response = await client.get(
                f"{self.tool_runner_url}/tools/manifest",
                headers=self._tool_runner_headers(),
            )
            response.raise_for_status()
            data = response.json()
            
            # Handle different response formats.
            tools_data = data.get("tools", data) if isinstance(data, dict) else data
            
            # If tools is a dict (name -> info), convert to list
            if isinstance(tools_data, dict):
                self._tools_cache = [
                    {"name": name, **info} if isinstance(info, dict) else {"name": name}
                    for name, info in tools_data.items()
                ]
            elif isinstance(tools_data, list):
                self._tools_cache = tools_data
            else:
                self._tools_cache = []
            
            log.info(f"📦 Loaded {len(self._tools_cache)} tools from tool-runner")
            return self._tools_cache
        except Exception as exc:
            log.error(f"🔴 Failed to get tools manifest: {exc}")
            return []
    
    async def get_multi_agent_manifest(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        دریافت manifest دسته‌بندی شده برای معماری Multi-Agent.
        
        Returns:
            {
                "agents": {
                    "vision": {"description": ..., "tools": [...]},
                    "text": {"description": ..., "tools": [...]},
                    ...
                },
                "all_tools": [...],
                "category_mapping": {"tool_name": "agent_type", ...}
            }
        """
        if self._multi_agent_cache and not force_refresh:
            return self._multi_agent_cache
        
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.tool_runner_url}/tools/agents/manifest",
                headers=self._tool_runner_headers(),
            )
            response.raise_for_status()
            self._multi_agent_cache = response.json()
            
            # Log agent tool counts
            agents = self._multi_agent_cache.get("agents", {})
            for agent_type, agent_data in agents.items():
                tool_count = len(agent_data.get("tools", []))
                if tool_count > 0:
                    log.info(f"🤖 Agent '{agent_type}': {tool_count} tools")
            
            return self._multi_agent_cache
        except Exception as exc:
            log.error(f"🔴 Failed to get multi-agent manifest: {exc}")
            return {"agents": {}, "all_tools": [], "category_mapping": {}}

    async def get_tools_by_file_requirement(self) -> Dict[str, Any]:
        """
        دریافت دسته‌بندی ابزارها بر اساس نیاز به فایل.
        
        Returns:
            {"file_required": [...], "no_file_required": [...], "file_optional": [...]}
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.tool_runner_url}/tools/by-file-requirement",
                headers=self._tool_runner_headers(),
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            log.error(f"🔴 Failed to get tools by file requirement: {exc}")
            return {"file_required": [], "no_file_required": [], "file_optional": []}

    async def get_prompt_friendly_tools_info(self) -> str:
        """
        دریافت متن داینامیک برای prompt.
        
        Returns:
            متن Markdown با دسته‌بندی ابزارها
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.tool_runner_url}/tools/prompt-info",
                headers=self._tool_runner_headers(),
            )
            response.raise_for_status()
            data = response.json()
            return data.get("tools_info", "")
        except Exception as exc:
            log.error(f"🔴 Failed to get prompt-friendly tools info: {exc}")
            return ""

    async def get_plugin_manifest(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """``plugin.json`` of one plugin as the tool-runner serves it.

        ``{}`` when the tool-runner has no such plugin, ``None`` when it could not be asked.
        """
        pid = str(plugin_id or "").strip()
        if not pid:
            return {}
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.tool_runner_url}/plugins/{pid}",
                headers=self._tool_runner_headers(),
            )
            if response.status_code == 404:
                return {}
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            log.warning("tool-runner plugin manifest %s unavailable: %s", pid, exc)
            return None

    async def get_agent_prompt(self, category: str, channel: str = "llm") -> Dict[str, Any]:
        """
        دریافت prompt داینامیک برای یک agent.
        
        Args:
            category: دسته‌بندی (image, audio, text, ...)
            channel: کانال (default: llm)
        
        Returns:
            {"role": "...", "capabilities": [...], "tools": [...], "prompt": "..."}
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.tool_runner_url}/tools/agent-prompt/{category}",
                params={"channel": channel},
                headers=self._tool_runner_headers(),
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            log.error(f"🔴 Failed to get agent prompt for {category}: {exc}")
            return {}

    async def get_all_agent_prompts(self, channel: str = "llm") -> Dict[str, Dict[str, Any]]:
        """
        دریافت prompts همه agent ها.
        
        برای cache کردن در startup.
        
        Args:
            channel: کانال (default: llm)
        
        Returns:
            {"image": {...}, "audio": {...}, ...}
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.tool_runner_url}/tools/agent-prompts",
                params={"channel": channel},
                headers=self._tool_runner_headers(),
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            log.error(f"🔴 Failed to get all agent prompts: {exc}")
            return {}

    async def get_tools_by_category(
        self, 
        category: str = None, 
        channel: str = "llm",
        include_general: bool = True
    ) -> List[Dict[str, Any]]:
        """
        دریافت ابزارها بر اساس دسته‌بندی و کانال.
        
        مثال: برای فایل عکس → category=image, channel=llm
        
        Args:
            category: دسته‌بندی (image, audio, text, face, file, embed, entity, geo)
            channel: کانال (llm, ui, metadata, rawdata, embed) - default: llm
            include_general: شامل ابزارهای عمومی هم بشه؟ - default: true
        
        Returns:
            لیست ابزارها در فرمت OpenAI tools
        """
        try:
            client = await self._get_client()
            params = {
                "channel": channel,
                "include_general": str(include_general).lower()
            }
            if category:
                params["category"] = category
            
            response = await client.get(
                f"{self.tool_runner_url}/tools/by-category",
                params=params,
                headers=self._tool_runner_headers(),
            )
            response.raise_for_status()
            data = response.json()
            return data.get("tools", [])
        except Exception as exc:
            log.error(f"🔴 Failed to get tools by category: {exc}")
            return []

    async def get_tools_without_file_requirement(self, channel: str = "llm") -> List[Dict[str, Any]]:
        """
        دریافت ابزارهایی که نیاز به فایل ندارن.
        
        برای وقتی که کاربر فایلی آپلود نکرده - فقط ابزارهایی مثل:
        - text_search (جستجو در دیتابیس)
        - enrich_entity (غنی‌سازی موجودیت)
        - text.summarize (با متن ورودی)
        
        همچنین ابزارهای ``file_optional`` را شامل می‌شود (path اختیاری؛ بدون آپلود هم
        با آرگومان‌های دیگر قابل فراخوانی‌اند). اگر طبقه‌بندی رجیستری خالی برگردد،
        از همهٔ ابزارهای کانال به‌جز آن‌هایی که صریحاً ``file_required`` هستند
        fallback می‌زنیم تا LLM هرگز بدون هیچ tool نماند.
        
        Args:
            channel: کانال (default: llm)
        
        Returns:
            لیست ابزارها در فرمت OpenAI tools
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.tool_runner_url}/tools/by-file-requirement",
                params={"channel": channel},
                headers=self._tool_runner_headers(),
            )
            response.raise_for_status()
            data = response.json()

            all_tools = await self._get_openai_tools_cached(channel)
            if not all_tools:
                log.warning("📦 OpenAI tools list empty from tool-runner — LLM will have no tools")
                return []

            def _fname(t: Dict[str, Any]) -> str:
                return (t.get("function") or {}).get("name") or ""

            by_name = {_fname(t): t for t in all_tools if _fname(t)}

            file_required_names = {
                x.get("name")
                for x in data.get("file_required", [])
                if isinstance(x, dict) and x.get("name")
            }
            allow_names = set()
            for key in ("no_file_required", "file_optional"):
                for x in data.get(key) or []:
                    if isinstance(x, dict) and x.get("name"):
                        allow_names.add(x["name"])

            openai_tools = [by_name[n] for n in sorted(allow_names) if n in by_name]

            if not openai_tools and file_required_names:
                # Registry split left nothing in no_file/optional (mis-configured args, etc.)
                openai_tools = [t for t in all_tools if _fname(t) not in file_required_names]
                if openai_tools:
                    log.warning(
                        "📦 No-file tool pick was empty — falling back to all tools "
                        "except file_required (%s excluded)",
                        len(file_required_names),
                    )

            if not openai_tools:
                log.warning(
                    "📦 Still no tools for no-file context — exposing full llm tool list (%s)",
                    len(all_tools),
                )
                openai_tools = list(all_tools)

            log.info(
                "📦 Tools for chat without upload: %s",
                [_fname(t) for t in openai_tools],
            )
            return openai_tools

        except Exception as exc:
            log.error(f"🔴 Failed to get tools without file requirement: {exc}")
            return await self._get_openai_tools_cached(channel)

    async def _get_openai_tools_cached(self, channel: str = "llm") -> List[Dict[str, Any]]:
        """دریافت ابزارها در فرمت OpenAI (با cache)."""
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.tool_runner_url}/tools/openai",
                params={"channel": channel},
                headers=self._tool_runner_headers(),
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            log.error(f"🔴 Failed to get OpenAI tools: {exc}")
            return []

    async def get_available_categories(self) -> List[str]:
        """
        دریافت لیست همه دسته‌بندی‌های موجود.
        
        Returns:
            لیست دسته‌بندی‌ها
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.tool_runner_url}/tools/categories",
                headers=self._tool_runner_headers(),
            )
            response.raise_for_status()
            data = response.json()
            return data.get("categories", [])
        except Exception as exc:
            log.error(f"🔴 Failed to get available categories: {exc}")
            return []

    async def get_tools_for_filetype(self, mime_type: str, channel: str = "llm") -> List[Dict[str, Any]]:
        """
        دریافت ابزارهای مناسب برای یک نوع فایل خاص.
        
        این متد از tool-runner/for-filetype استفاده می‌کنه که
        بر اساس file_types در tool.json فیلتر می‌کنه.
        
        Args:
            mime_type: نوع MIME فایل (مثل "image/jpeg")
            channel: کانال (default: llm)
        
        Returns:
            لیست ابزارها با اطلاعات کامل
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.tool_runner_url}/tools/for-filetype",
                params={"mime": mime_type, "channel": channel},
                headers=self._tool_runner_headers(),
            )
            response.raise_for_status()
            data = response.json()
            tools = data.get("tools", [])
            log.info(f"📦 Tools for {mime_type}: {[t.get('id') for t in tools]}")
            return tools
        except Exception as exc:
            log.error(f"🔴 Failed to get tools for filetype {mime_type}: {exc}")
            return []

    async def get_category_map(self, channel: str = "llm") -> Dict[str, List[str]]:
        """
        دریافت mapping کامل category به tool_id ها.
        
        برای populate کردن cache در file_context.py.
        
        Args:
            channel: فیلتر کانال (default: llm)
        
        Returns:
            {"image": ["vision_info", ...], "audio": [...], ...}
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.tool_runner_url}/tools/category-map",
                params={"channel": channel},
                headers=self._tool_runner_headers(),
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            log.error(f"🔴 Failed to get category map: {exc}")
            return {}

    async def get_file_based_tools(self, channel: str = "llm") -> List[str]:
        """
        دریافت لیست ابزارهایی که نیاز به فایل دارند.
        
        برای populate کردن cache در file_context.py.
        
        Args:
            channel: فیلتر کانال (default: llm)
        
        Returns:
            لیست tool_id ها
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.tool_runner_url}/tools/file-based-tools",
                params={"channel": channel},
                headers=self._tool_runner_headers(),
            )
            response.raise_for_status()
            data = response.json()
            return data.get("tools", [])
        except Exception as exc:
            log.error(f"🔴 Failed to get file-based tools: {exc}")
            return []

    async def map_file_type_to_category(self, file_type: str) -> str:
        """
        تبدیل نوع فایل به دسته‌بندی ابزار.
        
        Args:
            file_type: MIME type یا extension یا نوع کلی
        
        Returns:
            نام دسته‌بندی
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.tool_runner_url}/tools/map-file-type",
                params={"file_type": file_type},
                headers=self._tool_runner_headers(),
            )
            response.raise_for_status()
            data = response.json()
            return data.get("category", "file")
        except Exception as exc:
            log.error(f"🔴 Failed to map file type to category: {exc}")
            # Fallback to local mapping
            return self._local_map_file_type_to_category(file_type)

    def _local_map_file_type_to_category(self, file_type: str) -> str:
        """Fallback local mapping for file type to category."""
        ft_lower = file_type.lower()
        
        if "/" in ft_lower:
            main_type = ft_lower.split("/")[0]
            if main_type in {"image", "audio", "video", "text"}:
                return main_type
        
        image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
        audio_exts = {".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a"}
        video_exts = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm"}
        text_exts = {".txt", ".pdf", ".doc", ".docx", ".rtf", ".md"}
        
        if ft_lower in image_exts or "image" in ft_lower:
            return "image"
        if ft_lower in audio_exts or "audio" in ft_lower:
            return "audio"
        if ft_lower in video_exts or "video" in ft_lower:
            return "video"
        if ft_lower in text_exts or "text" in ft_lower:
            return "text"
        
        return "file"

    async def get_tools_for_file_context(
        self, 
        file_context: Optional[Dict[str, Any]] = None,
        channel: str = "llm"
    ) -> List[Dict[str, Any]]:
        """
        دریافت ابزارهای مناسب بر اساس file context.
        
        - اگر فایل نیست → فقط ابزارهای بدون نیاز به فایل (text_search, enrich_entity)
        - اگر فایل عکس → ابزارهای image + ابزارهای عمومی (file, text, entity)
        - اگر فایل صوت → ابزارهای audio + ابزارهای عمومی
        
        Args:
            file_context: اطلاعات فایل (type, mime_type, path, ...)
            channel: کانال (default: llm)
        
        Returns:
            لیست ابزارها در فرمت OpenAI tools
        """
        if not file_context:
            # No file - فقط ابزارهای بدون نیاز به فایل
            # این ابزارها path required ندارن (مثل text_search, enrich_entity)
            return await self.get_tools_without_file_requirement(channel=channel)
        
        # Determine category from file context
        file_type = file_context.get("type", "").lower()
        mime_type = file_context.get("mime_type", "").lower()
        
        # Use mime_type first, then file_type
        type_to_use = mime_type if mime_type else file_type
        if not type_to_use:
            type_to_use = "file"
        
        category = await self.map_file_type_to_category(type_to_use)
        
        log.info(f"📁 File context type: {type_to_use} → category: {category}")

        picked = await self.get_tools_by_category(
            category=category,
            channel=channel,
            include_general=True  # Always include general tools
        )
        if picked:
            return picked

        log.warning(
            "📁 get_tools_by_category returned empty for category=%s — falling back to all llm tools",
            category,
        )
        all_t = await self._get_openai_tools_cached(channel)
        return all_t if all_t else picked

    async def get_agent_tools(self, agent_type: str) -> List[Dict[str, Any]]:
        """
        دریافت ابزارهای یک agent خاص.
        
        Args:
            agent_type: نوع agent (vision, text, audio, file, analysis, general)
        
        Returns:
            لیست ابزارهای آن agent
        """
        manifest = await self.get_multi_agent_manifest()
        agent_data = manifest.get("agents", {}).get(agent_type, {})
        return agent_data.get("tools", [])
    
    def get_tool_agent_type(self, tool_name: str) -> str:
        """
        تعیین نوع agent برای یک ابزار.
        
        Args:
            tool_name: نام ابزار
        
        Returns:
            نوع agent (vision, text, audio, file, analysis, general)
        """
        if self._multi_agent_cache:
            mapping = self._multi_agent_cache.get("category_mapping", {})
            return mapping.get(tool_name, "general")
        return "general"
    
    def determine_agent_type_from_context(self, file_context: Optional[Dict[str, Any]]) -> str:
        """
        تعیین نوع agent بر اساس file context.
        
        Args:
            file_context: اطلاعات فایل
        
        Returns:
            نوع agent مناسب
        """
        if not file_context:
            return "general"
        
        file_type = file_context.get("type", "").lower()
        mime_type = file_context.get("mime_type", "").lower()
        
        # Image types
        if file_type == "image" or mime_type.startswith("image/"):
            return "vision"
        
        # Audio types
        if file_type == "audio" or mime_type.startswith("audio/"):
            return "audio"
        
        # Video types
        if file_type == "video" or mime_type.startswith("video/"):
            return "vision"  # Video analysis uses vision agent
        
        # Text types
        if file_type in ["text", "document"] or mime_type.startswith("text/"):
            return "text"
        
        # File types
        if file_type == "file":
            return "file"
        
        return "general"
    
    async def get_openai_tools_for_ids(
        self,
        tool_ids: List[str],
        channel: str = "llm",
    ) -> List[Dict[str, Any]]:
        """OpenAI tool defs from tool-runner (canonical JSON Schema)."""
        wanted = {str(t).strip() for t in tool_ids if str(t).strip()}
        if not wanted:
            return []
        try:
            all_tools = await self._get_openai_tools_cached(channel)
        except Exception:
            all_tools = []
        out: List[Dict[str, Any]] = []
        seen: set = set()
        for item in all_tools or []:
            fn = item.get("function") if isinstance(item, dict) else None
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name") or "").strip()
            if name in wanted and name not in seen:
                out.append(item)
                seen.add(name)
        return out

    @staticmethod
    def _declared_choices(prop: Any) -> List[str]:
        """Allowed values of a string parameter: JSON ``enum`` or a bare ``a|b|c`` description."""
        if not isinstance(prop, dict):
            return []
        enum = prop.get("enum")
        if isinstance(enum, list) and enum:
            return [str(v) for v in enum]
        desc = str(prop.get("description") or "").strip()
        if _CHOICE_LIST_RE.fullmatch(desc):
            return [v.strip() for v in desc.split("|")]
        return []

    async def _choice_argument_error(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        """Normalise case of choice arguments in place; describe any value outside the choices."""
        try:
            specs = await self.get_openai_tools_for_ids([tool_name])
        except Exception:
            return None
        params = ((specs[0].get("function") or {}).get("parameters") or {}) if specs else {}
        props = params.get("properties") if isinstance(params, dict) else None
        if not isinstance(props, dict):
            return None
        problems: List[str] = []
        for name, prop in props.items():
            value = args.get(name)
            choices = self._declared_choices(prop)
            if not choices or value is None or isinstance(value, (dict, list, bool)):
                continue
            by_lower = {c.lower(): c for c in choices}
            match = by_lower.get(str(value).strip().lower())
            if match is not None:
                if isinstance(value, str):
                    args[name] = match
                continue
            problems.append(f"{name}={value!r} is not one of {'|'.join(choices)}")
        if not problems:
            return None
        return f"invalid_argument: {'; '.join(problems)}. Retry with an allowed value or omit the argument."

    def get_openai_tools_format(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """تبدیل ابزارها به فرمت OpenAI function calling."""
        openai_tools = []
        for tool in tools:
            params = tool.get("parameters")
            if isinstance(params, dict) and isinstance(params.get("properties"), dict):
                parameters = dict(params)
            else:
                arguments = tool.get("arguments") or tool.get("args") or {}
                parameters = {
                    "type": "object",
                    "properties": self._convert_args_to_properties(arguments),
                    "required": self._get_required_args(arguments),
                }
            if parameters.get("required") == []:
                parameters.pop("required", None)
            desc = (
                (tool.get("llm_description") or tool.get("description") or "")
            ).strip()
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name") or tool.get("id"),
                    "description": desc[:1200],
                    "parameters": parameters,
                }
            })
        return openai_tools
    
    def _convert_args_to_properties(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """تبدیل آرگومان‌ها به فرمت JSON Schema."""
        properties = {}
        for name, spec in args.items():
            if isinstance(spec, str):
                # Parse string format like "string (required)"
                prop_type = "string"
                if "int" in spec.lower():
                    prop_type = "integer"
                elif "float" in spec.lower() or "number" in spec.lower():
                    prop_type = "number"
                elif "bool" in spec.lower():
                    prop_type = "boolean"
                properties[name] = {"type": prop_type, "description": spec}
            elif isinstance(spec, dict):
                properties[name] = spec
            else:
                properties[name] = {"type": "string"}
        return properties
    
    def _get_required_args(self, args: Dict[str, Any]) -> List[str]:
        """استخراج آرگومان‌های الزامی (فقط وقتی صریحاً optional/required در prefix باشد)."""
        import re

        required: List[str] = []
        for name, spec in args.items():
            if isinstance(spec, dict) and spec.get("required"):
                required.append(name)
                continue
            if not isinstance(spec, str):
                continue
            lower = spec.lower()
            if re.match(r"^\s*\w+\s*\(\s*optional\b", lower):
                continue
            if re.match(r"^\s*\w+\s*\(\s*required\b", lower):
                required.append(name)
                continue
            if "required" in lower and "optional" not in lower:
                required.append(name)
        return required
    
    async def tool_meta(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Manifest entry of ``tool_name`` (cached manifest; None when unknown)."""
        try:
            manifest = await self.get_tools_manifest()
        except Exception:  # noqa: BLE001
            return None
        for row in manifest or []:
            if isinstance(row, dict) and tool_name in (row.get("id"), row.get("name")):
                return row
        return None

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        file_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        discourse_focus: Optional[Dict[str, Any]] = None,
        file_registry: Optional[List[Dict[str, Any]]] = None,
        user_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a tool through the side-effect policy (src/tools/side_effects.py)."""
        from .side_effects import guarded

        async def run() -> Dict[str, Any]:
            return await self._execute_unguarded(
                tool_name, arguments, file_context=file_context, session_id=session_id,
                user_id=user_id, group_id=group_id, discourse_focus=discourse_focus,
                file_registry=file_registry, user_message=user_message,
            )

        meta = await self.tool_meta(tool_name)
        return await guarded(tool_name, dict(arguments or {}), meta, user_message or "", run)

    async def _execute_unguarded(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        file_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
        discourse_focus: Optional[Dict[str, Any]] = None,
        file_registry: Optional[List[Dict[str, Any]]] = None,
        user_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        اجرای یک ابزار.
        
        Args:
            tool_name: نام ابزار
            arguments: آرگومان‌های ابزار
            file_context: context فایل (برای ابزارهای file-based)
            session_id: شناسه سشن
        
        Returns:
            نتیجه اجرای ابزار
        """
        try:
            original_tool = tool_name
            arguments = dict(arguments or {})
            if original_tool in {
                "session_read_file",
                "session_read",
                "plugin_session_read",
            }:
                from ..session_tool_resolve import hydrate_read_args_from_focus

                arguments = hydrate_read_args_from_focus(
                    original_tool,
                    arguments,
                    discourse_focus=discourse_focus,
                    file_registry=file_registry,
                    user_message=user_message or "",
                )

            tool_name, arguments = self._remap_legacy_session_tools(
                original_tool, arguments
            )

            # نرمال‌سازی آرگومان‌ها
            normalized_args = self._normalize_arguments(
                tool_name, arguments, file_context, session_id
            )
            normalized_args = await self._resolve_session_file_args(
                tool_name,
                normalized_args,
                session_id=session_id,
                user_id=user_id,
                group_id=group_id,
            )
            choice_error = await self._choice_argument_error(tool_name, normalized_args)
            if choice_error:
                log.warning("Tool %s rejected before execute: %s", tool_name, choice_error)
                return {"ok": False, "error": choice_error, "tool_id": tool_name}
            
            log.info(f"🔧 Executing tool: {tool_name}")
            log.info(f"   Original args: {arguments}")
            log.info(f"   File context: {file_context}")
            log.info(f"   Normalized args: {normalized_args}")
            
            client = await self._get_client()
            # tool-runner expects {"args": {...}} format
            response = await client.post(
                f"{self.tool_runner_url}/tools/{tool_name}/execute",
                json={"args": normalized_args, "session_id": session_id},
                headers=self._tool_runner_headers(user_id=user_id, group_id=group_id),
            )
            
            result = response.json()
            
            if response.status_code >= 400:
                log.warning(f"⚠️ Tool {tool_name} returned error: {result}")
                failed = {
                    "ok": False,
                    "error": result.get("error") or result.get("detail") or f"HTTP {response.status_code}",
                    "tool_id": tool_name
                }
                recovered = await self._recover_after_tool_failure(
                    tool_name=tool_name,
                    failed_result=failed,
                    normalized_args=normalized_args,
                    session_id=session_id,
                    user_id=user_id,
                    group_id=group_id,
                )
                if recovered:
                    log.info(
                        "session_attach recovery applied after HTTP %s (session=%s)",
                        response.status_code,
                        session_id,
                    )
                return recovered or failed

            # tool-runner returns HTTP 200 with body {ok: false, error: ...} on validation failure
            if not result.get("ok", True) or result.get("error"):
                err = result.get("error") or "tool_failed"
                log.warning(f"⚠️ Tool {tool_name} reported failure: {err}")
                failed = {
                    "ok": False,
                    "error": err,
                    "data": result,
                    "tool_id": tool_name,
                }
                recovered = await self._recover_after_tool_failure(
                    tool_name=tool_name,
                    failed_result=failed,
                    normalized_args=normalized_args,
                    session_id=session_id,
                    user_id=user_id,
                    group_id=group_id,
                )
                if recovered:
                    log.info(
                        "session_attach recovery applied after tool failure (session=%s)",
                        session_id,
                    )
                return recovered or failed
            
            log.info(f"✅ Tool {tool_name} executed successfully")
            if tool_name == "session_files" and file_registry:
                result = self._enrich_session_files_from_registry(result, file_registry)
            # Tools that mutate the session artifact set invalidate the cache.
            if tool_name in {"session_attach"}:
                sid = (
                    session_id
                    or normalized_args.get("session_id")
                    or arguments.get("session_id")
                )
                if sid:
                    self.invalidate_session_artifacts_cache(sid)
            # tool-runner wraps plugin output as {ok, tool_id, result:{ok,data,channels}, error}.
            # Flatten so host-effect extractors see channels.ui / map_action at a stable path.
            return self._flatten_tool_runner_result(result, tool_name=tool_name)
            
        except Exception as exc:
            log.error(f"🔴 Tool execution failed: {tool_name} - {exc}", exc_info=True)
            return {
                "ok": False,
                "error": f"execution_failed: {type(exc).__name__}: {str(exc)}",
                "tool_id": tool_name
            }

    @staticmethod
    def _flatten_tool_runner_result(result: Dict[str, Any], *, tool_name: str) -> Dict[str, Any]:
        """Normalize tool-runner `{ok, result:{ok,data,channels}}` into a stable host-effect shape.

        Prefer top-level ``channels`` (especially ``channels.ui.map_action``) so the
        floating dock can apply map overlays without deep envelope walks.
        """
        if not isinstance(result, dict):
            return {"ok": True, "data": result, "tool_id": tool_name}

        inner = result.get("result") if isinstance(result.get("result"), dict) else None
        if isinstance(inner, dict) and (
            "channels" in inner or "data" in inner or "ok" in inner
        ):
            out: Dict[str, Any] = {
                "ok": bool(inner.get("ok", result.get("ok", True))),
                "data": inner.get("data", inner),
                "tool_id": tool_name,
            }
            if isinstance(inner.get("channels"), dict):
                out["channels"] = inner["channels"]
            elif isinstance(result.get("channels"), dict):
                out["channels"] = result["channels"]
            # Preserve nested envelope for legacy readers.
            out["raw_envelope"] = result
            return out

        if isinstance(result.get("channels"), dict):
            return {
                "ok": bool(result.get("ok", True)),
                "data": result.get("data", result),
                "channels": result["channels"],
                "tool_id": tool_name,
            }

        return {
            "ok": True,
            "data": result,
            "tool_id": tool_name,
        }

    async def ensure_session_attach_recovered(
        self,
        result: Dict[str, Any],
        *,
        tool_args: Dict[str, Any],
        session_id: Optional[str],
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Idempotent recovery wrapper for session_attach (engine + history repair)."""
        if not isinstance(result, dict):
            return result
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        if data.get("recovered"):
            return result
        probe = {
            "ok": result.get("ok", True),
            "error": result.get("error"),
            "data": result.get("data"),
        }
        if not _is_session_attach_not_found(probe):
            return result
        if not session_id:
            return result
        failed = {
            "ok": False,
            "error": result.get("error") or "artifact_not_found",
            "data": result.get("data"),
            "tool_id": "session_attach",
        }
        recovered = await self._recover_after_tool_failure(
            tool_name="session_attach",
            failed_result=failed,
            normalized_args=dict(tool_args or {}),
            session_id=session_id,
            user_id=user_id,
            group_id=group_id,
        )
        if recovered:
            log.info("session_attach recovery applied via ensure_* (session=%s)", session_id)
        return recovered or result

    async def _recover_after_tool_failure(
        self,
        *,
        tool_name: str,
        failed_result: Dict[str, Any],
        normalized_args: Dict[str, Any],
        session_id: Optional[str],
        user_id: Optional[str],
        group_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Deterministic bounded fallback for known recoverable tool failures."""
        if tool_name != "session_attach":
            return None

        if not _is_session_attach_not_found(failed_result):
            return None
        if not session_id:
            return None

        filename_hint = str(
            normalized_args.get("filename")
            or normalized_args.get("name")
            or normalized_args.get("path")
            or ""
        ).strip()
        if "/" in filename_hint:
            filename_hint = filename_hint.rsplit("/", 1)[-1]

        guidance_lines = [
            "session_attach could not find an accessible storage file.",
            f"attempted_locator={filename_hint or 'unknown'}",
            "This tool looks up storage/cases — not the chat workspace listing.",
            "If the locator is incomplete: ask the user for case_id, artifact_id, or full path, then retry session_attach.",
            "If files are already in this chat: use session_search_workspace(query=…) or session_read_file.",
            "Do not call session_files repeatedly when the workspace is empty.",
        ]

        return {
            "ok": True,
            "tool_id": "session_attach",
            "data": {
                "ok": True,
                "recovered": True,
                "recovery_strategy": "session_attach_not_found_guidance_v2",
                "original_failure": failed_result,
                "attempted_locator": filename_hint or None,
                "channels": {
                    "llm": "\n".join(guidance_lines),
                },
            },
        }

    async def _resolve_session_file_args(
        self,
        tool_name: str,
        args: Dict[str, Any],
        *,
        session_id: Optional[str],
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Hydrate file-oriented tool args from the session artifact registry.

        If the model provides only a filename, basename, artifact id, or a
        placeholder path, resolve it against the current session and fill in the
        concrete fields the tool actually needs (`path`, `file_id`,
        `artifact_id`, `filename`, ...).

        The session-artifacts listing is cached per session for a short TTL —
        when the chat loop calls 5 tools back-to-back we no longer issue 5
        identical ``GET /chat/sessions/.../artifacts`` round-trips.
        """
        if not session_id:
            return args

        candidate_keys = ("file_id", "artifact_id", "path", "file_path", "filename", "name")
        raw_candidates = []
        for key in candidate_keys:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                raw_candidates.append(value.strip())
        if not raw_candidates:
            return args

        artifacts = await self._cached_session_artifacts(
            session_id, user_id=user_id, group_id=group_id
        )
        if artifacts is None:
            log.warning(
                "⚠️ Could not list session artifacts to resolve %r for %s",
                raw_candidates,
                tool_name,
            )
            return args

        if not isinstance(artifacts, list):
            return args

        def _artifact_names(artifact: Dict[str, Any]) -> set[str]:
            names = set()
            for value in (
                artifact.get("id"),
                artifact.get("file_id"),
                artifact.get("artifact_id"),
                artifact.get("filename"),
                artifact.get("name"),
                artifact.get("original_filename"),
                artifact.get("storage_path"),
                artifact.get("path"),
                artifact.get("file_path"),
            ):
                if not value:
                    continue
                text = str(value).strip()
                if not text:
                    continue
                names.add(text.lower())
                names.add(os.path.basename(text).lower())
            return names

        wanted = set()
        for raw in raw_candidates:
            wanted.add(raw.lower())
            wanted.add(os.path.basename(raw).lower())

        matched_artifact: Optional[Dict[str, Any]] = None
        for art in artifacts:
            if isinstance(art, dict) and wanted & _artifact_names(art):
                matched_artifact = art
                break

        if not matched_artifact:
            return args

        out = dict(args)
        session_artifact_tools = {
            "session_read",
            "plugin_session_read",
            "session_fileinfo",
            "vision_info",
            "plugin_vision_info",
        }
        artifact_id = (
            matched_artifact.get("id")
            or matched_artifact.get("file_id")
            or matched_artifact.get("artifact_id")
        )
        filename = (
            matched_artifact.get("filename")
            or matched_artifact.get("name")
            or matched_artifact.get("original_filename")
        )
        storage_path = (
            matched_artifact.get("storage_path")
            or matched_artifact.get("path")
            or matched_artifact.get("file_path")
        )
        mime_type = matched_artifact.get("mime_type") or matched_artifact.get("content_type")
        media_type = matched_artifact.get("type") or matched_artifact.get("media_type")

        if artifact_id:
            out["file_id"] = str(artifact_id)
            out["artifact_id"] = str(artifact_id)
        if filename and not out.get("filename"):
            out["filename"] = str(filename)
        if filename and not out.get("name"):
            out["name"] = str(filename)
        if mime_type and not out.get("mime_type"):
            out["mime_type"] = str(mime_type)
        if media_type and not out.get("type"):
            out["type"] = str(media_type)
        if (
            storage_path
            and self._is_file_based_tool(tool_name)
            and tool_name not in session_artifact_tools
        ):
            out["path"] = str(storage_path)
        if storage_path and not out.get("storage_path"):
            out["storage_path"] = str(storage_path)

        log.info(
            "📂 Hydrated session file args for %s: input=%r artifact_id=%r path=%r",
            tool_name,
            raw_candidates,
            artifact_id,
            storage_path,
        )
        return out

        return args
    
    async def execute_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
        file_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        اجرای موازی چند ابزار.
        
        Args:
            tool_calls: لیست فراخوانی‌های ابزار [{name, arguments}, ...]
            file_context: context فایل
            session_id: شناسه سشن
        
        Returns:
            لیست نتایج
        """
        if not tool_calls:
            return []
        
        max_parallel = settings.MAX_PARALLEL_TOOLS
        
        # تقسیم به batch‌ها
        batches = [tool_calls[i:i+max_parallel] for i in range(0, len(tool_calls), max_parallel)]
        
        all_results = []
        for batch in batches:
            tasks = [
                self.execute(
                    tool_name=call.get("name"),
                    arguments=call.get("arguments", {}),
                    file_context=file_context,
                    session_id=session_id,
                    user_id=user_id,
                    group_id=group_id,
                )
                for call in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    all_results.append({
                        "ok": False,
                        "error": str(result),
                        "tool_id": batch[i].get("name")
                    })
                else:
                    all_results.append(result)
        
        return all_results

    @staticmethod
    def _enrich_session_files_from_registry(
        http_result: Dict[str, Any],
        file_registry: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """When Storage artifacts are empty but the workspace registry has files,
        merge registry rows so the model is not told the session is empty."""
        tool_body = http_result.get("result")
        if not isinstance(tool_body, dict) or not tool_body.get("ok"):
            return http_result
        inner = tool_body.get("data") if isinstance(tool_body.get("data"), dict) else tool_body
        files = inner.get("files") if isinstance(inner.get("files"), list) else []
        if files:
            return http_result

        synth: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for entry in file_registry:
            if not isinstance(entry, dict):
                continue
            fid = str(entry.get("artifact_id") or entry.get("id") or entry.get("file_id") or "")
            name = (
                entry.get("name")
                or entry.get("filename")
                or entry.get("original_filename")
                or "unnamed"
            )
            key = fid or str(name)
            if key in seen:
                continue
            seen.add(key)
            synth.append(
                {
                    "id": fid or None,
                    "name": name,
                    "type": entry.get("type") or "file",
                    "size_bytes": entry.get("size_bytes") or 0,
                    "path": entry.get("path") or entry.get("storage_path"),
                }
            )
        if not synth:
            return http_result

        lines = [
            f"📂 Session workspace — {len(synth)} فایل از registry سشن (Storage artifacts خالی بود):",
            "",
            "### session_file_index (file_id|name|type|size_bytes)",
        ]
        for f in synth:
            lines.append(
                f"{f.get('id') or ''}|{f.get('name')}|{f.get('type')}|{f.get('size_bytes', 0)}"
            )
        lines.append("")
        lines.append(
            "⚠️ لیست Storage خالی برگشت؛ از file_idهای بالا برای session_read_file استفاده کن "
            "یا برای جستجو در دیتابیس session_attach."
        )
        llm_text = "\n".join(lines)
        if isinstance(tool_body.get("data"), dict):
            tool_body["data"]["files"] = synth
            tool_body["data"]["total"] = len(synth)
            tool_body["data"]["registry_fallback"] = True
        channels = tool_body.get("channels")
        if isinstance(channels, dict):
            channels["llm"] = llm_text
        log.info(
            "session_files enriched from workspace registry (%d files); storage list was empty",
            len(synth),
        )
        return http_result

    @staticmethod
    def _remap_legacy_session_tools(
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Map split/legacy session tool names to session_read implementation."""
        if tool_name == "session_read_file":
            args = dict(arguments)
            args.setdefault("scope", "file")
            return "session_read", args
        if tool_name == "session_search_workspace":
            # Owns semantic (→ text_search scope=session) and keyword (→ session_read).
            return tool_name, dict(arguments)
        if tool_name == "session_read":
            args = dict(arguments)
            scope = str(args.get("scope") or "file").strip().lower()
            if args.get("info_only"):
                return "session_read", args
            if scope == "session":
                return "session_read", args
            args.setdefault("scope", "file")
            return "session_read", args
        if tool_name == "session_search":
            args = dict(arguments)
            old_scope = str(args.get("scope") or "all").strip().lower()
            content_scopes = {
                "all", "text_content", "ocr", "descriptions", "transcripts",
                "filenames", "text", "code", "files",
            }
            if old_scope in content_scopes:
                args["content_scope"] = args.pop("scope")
            elif old_scope == "session":
                args.pop("scope", None)
            args["scope"] = "session"
            return "session_read", args
        if tool_name == "session_fileinfo":
            args = dict(arguments)
            args["info_only"] = True
            args.setdefault("scope", "file")
            return "session_read", args
        return tool_name, arguments
    
    def _normalize_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        file_context: Optional[Dict[str, Any]],
        session_id: Optional[str]
    ) -> Dict[str, Any]:
        """نرمال‌سازی آرگومان‌های ابزار."""
        args = dict(arguments or {})
        session_read_tools = {"session_read", "plugin_session_read"}
        session_artifact_tools = {
            "session_read", "vision_info",
            "plugin_session_read", "plugin_vision_info",
        }
        is_session_read_tool = tool_name in session_read_tools
        
        # ==========================================
        # Resolve filename → path
        # LLM sends filename, we resolve to full path
        # ==========================================
        if "filename" in args and "path" not in args and not is_session_read_tool:
            filename = args.pop("filename")
            resolved_path = self._resolve_filename_to_path(filename, file_context)
            if resolved_path:
                args["path"] = resolved_path
                log.info(f"📂 Resolved filename '{filename}' → '{resolved_path}'")
            else:
                # Fallback: assume it's in /data/uploads
                args["path"] = f"/data/uploads/{filename}"
                log.warning(f"⚠️ Could not resolve filename '{filename}', using fallback path")
        
        # تبدیل file_path به path (برخی LLM‌ها file_path می‌فرستند)
        if "file_path" in args and "path" not in args:
            args["path"] = args.pop("file_path")
        
        # ==========================================
        # Resolve path that looks like a filename
        # LLM sometimes sends just the filename in "path"
        # ==========================================
        if "path" in args and not is_session_read_tool:
            path_value = args["path"]
            # If path doesn't look like a full path, try to resolve it
            if path_value and not path_value.startswith(("/", "minio://")):
                resolved_path = self._resolve_filename_to_path(path_value, file_context)
                if resolved_path:
                    args["path"] = resolved_path
                    log.info(f"📂 Resolved path '{path_value}' → '{resolved_path}'")
            elif (
                path_value
                and isinstance(path_value, str)
                and path_value.startswith("/")
                and file_context
                and self._is_file_based_tool(tool_name)
            ):
                # The LLM is instructed to provide a filename, but may still
                # invent placeholder absolute paths such as
                # /absolute/path/photo.jpg. Resolve by basename against the
                # session files before the tool sees a path that cannot exist
                # inside the tool-runner pod.
                basename = path_value.rsplit("/", 1)[-1]
                resolved_path = self._resolve_filename_to_path(basename, file_context)
                if resolved_path:
                    args["path"] = resolved_path
                    log.info(
                        "📂 Replaced absolute path %r with session file %r for %s",
                        path_value, resolved_path, tool_name,
                    )
                elif (
                    path_value.startswith("/absolute/")
                    or path_value.startswith("/absolute/path/")
                    or path_value.startswith("/path/")
                ) and file_context.get("path"):
                    args["path"] = file_context["path"]
                    log.info(
                        "📂 Replaced placeholder absolute path %r with current session file %r for %s",
                        path_value, file_context["path"], tool_name,
                    )
        
        # اضافه کردن path از file_context برای ابزارهای file-based (اگه هنوز path نداریم)
        if file_context and self._is_file_based_tool(tool_name) and not is_session_read_tool:
            if "path" not in args and file_context.get("path"):
                args["path"] = file_context["path"]
        
        # اضافه کردن session_id (یا پاکسازی placeholder های رایج که LLM می‌سازد)
        _SESSION_PLACEHOLDERS = {
            "", "current", "this", "...", "…", "this_session",
            "current_session", "session", "default", "uuid", "null", "none",
        }
        if session_id:
            current_sid = args.get("session_id")
            if (
                "session_id" not in args
                or not isinstance(current_sid, str)
                or current_sid.strip().lower() in _SESSION_PLACEHOLDERS
            ):
                if "session_id" in args and current_sid != session_id:
                    log.info(
                        "🩹 Replacing bogus session_id %r with real %r for tool %s",
                        current_sid, session_id, tool_name,
                    )
                args["session_id"] = session_id

        # graph_search (core + plugin): schema requires ``queries: string[]``; models/recovery
        # often send a lone ``query`` string instead.
        if tool_name in ("graph_search", "plugin_graph_explorer_graph_search"):
            if not args.get("queries"):
                raw_q = args.pop("query", None)
                if raw_q is None:
                    raw_q = args.pop("q", None)
                if raw_q is None:
                    raw_q = args.pop("text", None)
                if raw_q is None:
                    raw_q = args.pop("inputs", None)
                if raw_q is None:
                    raw_q = args.pop("input", None)
                if raw_q is not None:
                    if isinstance(raw_q, (list, tuple)):
                        args["queries"] = [str(x).strip() for x in raw_q if x is not None and str(x).strip()]
                    elif isinstance(raw_q, str) and raw_q.strip():
                        args["queries"] = [raw_q.strip()]

        # text_search: schema requires ``text: string``; LLMs love sending
        # ``query`` / ``q`` / ``queries=[...]`` instead.
        if tool_name == "text_search" and not args.get("text"):
            raw_t = args.pop("query", None)
            if raw_t is None:
                raw_t = args.pop("q", None)
            if raw_t is None:
                raw_t = args.pop("queries", None)
            if isinstance(raw_t, (list, tuple)):
                # Take the first non-empty; the schema is single-string.
                for x in raw_t:
                    if x is not None and str(x).strip():
                        args["text"] = str(x).strip()
                        break
            elif isinstance(raw_t, str) and raw_t.strip():
                args["text"] = raw_t.strip()

        # plugin_smart_search: requires ``query`` (single string) and/or
        # ``path``. Some models send ``queries: [...]`` / ``text: "..."`` /
        # ``q: "..."`` — coerce to a single ``query`` string.
        if tool_name == "plugin_smart_search" and not args.get("query") and not args.get("path"):
            raw_q = args.pop("queries", None)
            if raw_q is None:
                raw_q = args.pop("text", None)
            if raw_q is None:
                raw_q = args.pop("q", None)
            if isinstance(raw_q, (list, tuple)):
                # Join non-empty entries with " ; " — smart_search treats this
                # as a single semantic query (RRF fuses anyway).
                pieces = [str(x).strip() for x in raw_q if x is not None and str(x).strip()]
                if pieces:
                    args["query"] = " ; ".join(pieces)
            elif isinstance(raw_q, str) and raw_q.strip():
                args["query"] = raw_q.strip()

        # session_read / session_fileinfo / vision_info on session artifacts:
        # These tools look up the artifact by ``file_id`` (an UUID). Three
        # common LLM mistakes we fix here:
        #
        #   1. The model passed a ``filename`` instead of ``file_id``.
        #   2. ``file_id`` itself contains a filename (e.g. "report.txt")
        #      — i.e. the model hallucinated a file_id from a filename it
        #      saw earlier or just remembered.
        #   3. The generic path-injection block above gave the tool a
        #      ``path`` pointing at a DIFFERENT artifact (because the
        #      session-level file_context.path is just one representative
        #      file). For session_* tools the only thing that matters is
        #      ``file_id``; ``path`` would confuse the tool.
        if tool_name in session_artifact_tools:
            if not is_session_read_tool:
                # Drop the path that was injected by the generic file_context
                # branch — these tools don't accept it, and a stale path makes
                # logs misleading (executor "resolved" path A while the tool
                # actually loaded artifact B via file_id). session_read is the
                # exception: it now accepts filename/path/name directly and
                # resolves them inside the tool from the session artifact list.
                args.pop("path", None)

            all_files: List[Dict[str, Any]] = []
            if isinstance(file_context, dict):
                af = file_context.get("all_files") or []
                if isinstance(af, list):
                    all_files = af

            def _looks_like_filename(s: str) -> bool:
                if not isinstance(s, str) or not s:
                    return False
                # A real artifact UUID has no slashes and usually contains
                # hyphens / hex; filenames typically have a dot+extension
                # and almost never look like a UUID. We use "has a dot OR
                # has a slash OR matches a known artifact name" as the
                # filename heuristic.
                if "/" in s:
                    return True
                if "." in s and not _UUID_LIKE_RE.match(s):
                    return True
                for f in all_files:
                    if isinstance(f, dict) and f.get("name") == s:
                        return True
                    if isinstance(f, dict) and f.get("original_filename") == s:
                        return True
                return False

            def _lookup_file_id_by_name(name: str) -> Optional[str]:
                if not name:
                    return None
                for f in all_files:
                    if not isinstance(f, dict):
                        continue
                    fname = f.get("name") or f.get("filename") or f.get("original_filename")
                    fid = f.get("file_id") or f.get("artifact_id") or f.get("id")
                    if fname == name and fid:
                        return fid
                # Case-insensitive fallback
                low = name.lower()
                for f in all_files:
                    if not isinstance(f, dict):
                        continue
                    fname = f.get("name") or f.get("filename") or f.get("original_filename") or ""
                    fid = f.get("file_id") or f.get("artifact_id") or f.get("id")
                    if fname.lower() == low and fid:
                        return fid
                # Basename fallback (model may have given "subdir/foo.txt")
                base = name.rsplit("/", 1)[-1]
                if base != name:
                    return _lookup_file_id_by_name(base)
                return None

            # Case 1+2: resolve a filename → file_id.
            current_fid = args.get("file_id")
            stray_fname = args.get("filename") or args.get("name") or args.get("path")
            if isinstance(current_fid, str) and _looks_like_filename(current_fid):
                resolved = _lookup_file_id_by_name(current_fid)
                if resolved:
                    log.info(
                        "🩹 file_id %r looked like a filename → resolved to %r for %s",
                        current_fid, resolved, tool_name,
                    )
                    args["file_id"] = resolved
                else:
                    log.warning(
                        "⚠️ file_id %r looks like a filename but no matching artifact "
                        "in session for tool %s — leaving as-is (tool will surface "
                        "file_not_found).",
                        current_fid, tool_name,
                    )
            elif not current_fid and stray_fname:
                resolved = _lookup_file_id_by_name(stray_fname)
                if resolved:
                    log.info(
                        "🩹 Resolved filename %r → file_id %r for tool %s",
                        stray_fname, resolved, tool_name,
                    )
                    args["file_id"] = resolved
                elif not is_session_read_tool:
                    args.pop("filename", None)

        return args
    
    def _resolve_filename_to_path(
        self,
        filename: str,
        file_context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Resolve a filename to its full path using file_context.
        
        Args:
            filename: The filename to resolve (e.g., "photo1.jpg")
            file_context: Context containing all_files list
            
        Returns:
            Full path or None if not found
        """
        if not file_context:
            return None
        
        # Check in all_files list
        all_files = file_context.get("all_files", [])
        base_filename = str(filename).rsplit("/", 1)[-1]
        for f in all_files:
            fname = f.get("name") or f.get("filename") or f.get("original_filename")
            if fname == filename or fname == base_filename:
                return f.get("path")
        
        # Check primary file
        primary_name = file_context.get("name") or file_context.get("filename") or file_context.get("original_filename")
        if primary_name == filename or primary_name == base_filename:
            return file_context.get("path")
        
        # Partial match (case-insensitive)
        filename_lower = str(filename).lower()
        base_lower = base_filename.lower()
        for f in all_files:
            name_lower = (f.get("name") or f.get("filename") or f.get("original_filename") or "").lower()
            if name_lower == filename_lower or name_lower == base_lower:
                return f.get("path")
        
        return None
    
    def _is_file_based_tool(self, tool_name: str) -> bool:
        """بررسی اینکه آیا ابزار file-based است."""
        file_based_prefixes = ("image.", "audio.", "video.", "file.", "ocr.", "face.", "session_")
        return tool_name.lower().startswith(file_based_prefixes)


# Global executor instance
_executor: Optional[ToolExecutor] = None


def get_tool_executor() -> ToolExecutor:
    """Get or create the global tool executor."""
    global _executor
    if _executor is None:
        _executor = ToolExecutor()
    return _executor
