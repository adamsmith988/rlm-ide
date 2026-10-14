"""
Registry Wrapper for Original AI Tools Module
لایه پوشش برای استفاده از ماژول tools اصلی بدون تغییر
"""
import os
import re
import sys
import asyncio
import logging
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

import httpx

from ..config import settings
from ..storage_context import (
    patch_storage_http_clients,
    reset_storage_context,
    set_storage_context,
)

logger = logging.getLogger(__name__)
patch_storage_http_clients()

# Storage service URL for MinIO downloads
STORAGE_URL = os.environ.get("STORAGE_URL", "http://storage:8004")

# Singleton registry instance
_registry_instance: Optional["ToolRegistryWrapper"] = None


# ===========================================================================
# MinIO Path Resolver - Middleware for downloading MinIO files before tool execution
# ===========================================================================

class MinIOPathResolver:
    """
    Resolve minio:// paths to local temp files before tool execution.
    
    This middleware:
    1. Scans tool args for minio:// paths (and mangled minio:/)
    2. Downloads files to temp directory
    3. Replaces paths with local temp paths
    4. Cleans up after tool execution
    """
    
    PATH_ARGS = {"path", "file_path", "image_path", "audio_path", "video_path", "input_path"}
    LIST_PATH_ARGS = {"paths", "file_paths", "image_paths"}

    @staticmethod
    def _normalize_uri(value: str) -> str:
        """Repair Path()-collapsed minio:/ → minio://."""
        p = str(value or "").strip()
        if p.startswith("minio://"):
            return p
        if p.startswith("minio:/"):
            return "minio://" + p[len("minio:/") :]
        return p

    @classmethod
    def _is_minio_uri(cls, value: str) -> bool:
        p = str(value or "")
        return p.startswith("minio://") or p.startswith("minio:/")

    @staticmethod
    def _storage_download_headers(auth_context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Build storage download headers.

        Case-importer pipeline: prefer case-scoped internal auth (X-Case-Id only)
        so storage does not reject downloads when user_id does not match artifact row.
        """
        headers = {
            "X-Internal-Key": settings.INTERNAL_API_KEY,
            "X-Service-Name": settings.SERVICE_NAME,
        }
        if not auth_context:
            return headers
        case_id = auth_context.get("case_id")
        artifact_id = auth_context.get("artifact_id")
        if case_id and artifact_id:
            headers["X-Case-Id"] = str(case_id)
            return headers
        if auth_context.get("user_id"):
            headers["X-User-Id"] = str(auth_context["user_id"])
        if auth_context.get("group_id"):
            headers["X-Group-Id"] = str(auth_context["group_id"])
        if case_id:
            headers["X-Case-Id"] = str(case_id)
        return headers
    
    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self._temp_files: List[str] = []
    
    def _extract_session_id(self, minio_path: str) -> Optional[str]:
        """Extract session_id from MinIO path."""
        match = re.search(r'session_([a-f0-9-]+)', minio_path)
        return match.group(1) if match else None

    def _artifact_id_for_path(
        self, minio_path: str, auth_context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Pick the right artifact for this URI (batch embeds pass a path→id map)."""
        if not auth_context:
            return None
        norm = self._normalize_uri(minio_path)
        mapping = auth_context.get("path_artifact_ids")
        if isinstance(mapping, dict) and mapping:
            for key in (norm, minio_path, norm.replace("minio://", "minio:/", 1)):
                aid = mapping.get(key)
                if aid:
                    return str(aid)
            # Batch mode: never reuse another file's artifact_id.
            return None
        sp = auth_context.get("storage_path")
        if isinstance(sp, str) and sp and self._normalize_uri(sp) == norm:
            aid = auth_context.get("artifact_id")
            return str(aid) if aid else None
        # Single-file calls: auth artifact_id is for this path.
        aid = auth_context.get("artifact_id")
        return str(aid) if aid else None
    
    async def _download_file(self, minio_path: str, auth_context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Download file from MinIO via storage service.
        Supports chat session paths and case_importer paths (via artifact_id).
        """
        minio_path = self._normalize_uri(minio_path)
        aid = self._artifact_id_for_path(minio_path, auth_context)
        if "case_importer/" in minio_path and aid:
            return await self._download_by_artifact_id(aid, minio_path, auth_context)

        session_id = self._extract_session_id(minio_path)
        if not session_id:
            if aid:
                return await self._download_by_artifact_id(aid, minio_path, auth_context)
            # Batch / multi-file: resolve artifact id from storage_path.
            looked = await self._lookup_artifact_id_by_path(minio_path, auth_context)
            if looked:
                return await self._download_by_artifact_id(looked, minio_path, auth_context)
            logger.warning(f"Could not extract session_id from: {minio_path}")
            return None
        
        try:
            headers = self._storage_download_headers(auth_context)
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Get all artifacts for this session
                resp = await client.get(
                    f"{STORAGE_URL}/chat/sessions/{session_id}/artifacts",
                    headers=headers,
                )
                if resp.status_code != 200:
                    logger.warning(f"Failed to get session artifacts: {resp.status_code}")
                    return None
                
                artifacts = resp.json()
                
                # Find artifact with matching storage_path
                artifact_id = None
                original_filename = "tempfile"
                for art in artifacts:
                    sp = self._normalize_uri(str(art.get("storage_path") or ""))
                    if sp == minio_path or art.get("storage_path") == minio_path:
                        artifact_id = art.get("id")
                        original_filename = art.get("original_filename", "tempfile")
                        break
                
                if not artifact_id:
                    logger.warning(f"Artifact not found for path: {minio_path}")
                    return None
                
                # Download file content (storage streams for X-Service-Name=tool-runner;
                # follow_redirects covers legacy 302→presigned if still returned).
                resp = await client.get(
                    f"{STORAGE_URL}/artifacts/{artifact_id}/download",
                    headers=headers,
                    follow_redirects=True,
                )
                if resp.status_code != 200:
                    logger.warning(f"Download failed: {resp.status_code}")
                    return None
                
                # Write to temp file with original extension
                _, ext = os.path.splitext(original_filename)
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext or "") as tmp:
                    tmp.write(resp.content)
                    temp_path = tmp.name
                
                self._temp_files.append(temp_path)
                logger.info(f"📥 Downloaded MinIO file to: {temp_path} ({len(resp.content)} bytes)")
                return temp_path
                
        except Exception as e:
            logger.error(f"MinIO download error: {e}")
            return None

    async def _lookup_artifact_id_by_path(
        self,
        minio_path: str,
        auth_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Resolve artifact UUID for a minio:// storage_path (batch embeds)."""
        import urllib.parse

        try:
            headers = self._storage_download_headers(auth_context)
            # Prefer query form so :// is not mangled by URL path parsing.
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    f"{STORAGE_URL}/artifacts/by-path/{urllib.parse.quote(minio_path, safe='')}",
                    headers=headers,
                )
                if resp.status_code != 200:
                    return None
                body = resp.json() if resp.content else {}
                aid = body.get("id") or body.get("artifact_id")
                return str(aid) if aid else None
        except Exception as e:
            logger.warning(f"artifact by-path lookup failed for {minio_path[:80]}: {e}")
            return None

    async def _download_by_artifact_id(
        self,
        artifact_id: str,
        minio_path: str,
        auth_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        try:
            headers = self._storage_download_headers(auth_context)
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    f"{STORAGE_URL}/artifacts/{artifact_id}/download",
                    headers=headers,
                    follow_redirects=True,
                )
                if resp.status_code != 200:
                    logger.warning(f"Artifact download failed: {resp.status_code}")
                    return None
                if not resp.content:
                    logger.warning(f"Artifact download empty body: {artifact_id}")
                    return None
                _, ext = os.path.splitext(minio_path.split("/")[-1])
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext or "") as tmp:
                    tmp.write(resp.content)
                    temp_path = tmp.name
                self._temp_files.append(temp_path)
                logger.info(
                    f"📥 Downloaded case_importer artifact to: {temp_path} ({len(resp.content)} bytes)"
                )
                return temp_path
        except Exception as e:
            logger.error(f"Artifact download error: {e}")
            return None

    async def _hydrate_one(
        self,
        value: str,
        auth_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bool]:
        """Return (local_or_original, was_minio_attempt)."""
        if not isinstance(value, str) or not value:
            return value, False
        normalized = self._normalize_uri(value)
        needs_download = self._is_minio_uri(value)
        if not needs_download and auth_context and (
            auth_context.get("artifact_id") or auth_context.get("path_artifact_ids")
        ):
            needs_download = not os.path.isfile(value)
        if not needs_download:
            return value, False
        local_path = await self._download_file(normalized, auth_context=auth_context)
        if local_path:
            logger.info(f"🔄 Resolved path: {normalized[:48]}… → {local_path}")
            return local_path, True
        logger.warning(f"⚠️ Failed to resolve path: {normalized}")
        return normalized, True
    
    async def resolve_args(
        self,
        args: Dict[str, Any],
        auth_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Resolve any minio:// paths in args to local temp files.
        
        Returns:
            Tuple of (resolved_args, had_minio_paths)
        """
        resolved = dict(args)
        had_minio = False
        
        for key in self.PATH_ARGS:
            if key not in resolved:
                continue
            value = resolved[key]
            if not isinstance(value, str) or not value:
                continue
            new_val, attempted = await self._hydrate_one(value, auth_context)
            if attempted:
                had_minio = True
                resolved[key] = new_val

        for key in self.LIST_PATH_ARGS:
            if key not in resolved:
                continue
            values = resolved[key]
            if not isinstance(values, list):
                continue
            new_list: List[Any] = []
            for value in values:
                if not isinstance(value, str) or not value:
                    new_list.append(value)
                    continue
                new_val, attempted = await self._hydrate_one(value, auth_context)
                if attempted:
                    had_minio = True
                new_list.append(new_val)
            resolved[key] = new_list
        
        return resolved, had_minio
    
    def cleanup(self):
        """Remove all temp files created during resolution."""
        for temp_path in self._temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    logger.debug(f"🗑️ Cleaned up temp file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_path}: {e}")
        self._temp_files.clear()


# ===========================================================================


@dataclass
class ToolInfo:
    """Simplified tool info for API responses."""
    id: str
    description: str
    category: str
    capabilities: List[str]
    args: Dict[str, Any]
    parameters: Dict[str, Any]  # OpenAI-compatible schema
    enabled: bool
    version: Optional[str]
    timeout_sec: Optional[int]
    supports_persist: bool
    has_ui: bool
    channels: List[str] = field(default_factory=list)


class ToolRegistryWrapper:
    """
    Wrapper around the original ToolRegistry from AI/modules/tools.
    
    This class:
    1. Adds the original tools module to Python path
    2. Imports and initializes ToolRegistry
    3. Provides a clean API for the FastAPI service
    4. Handles tool execution with proper error handling
    """
    
    def __init__(self, tools_path: str = None, plugins_path: str = None, app=None):
        self.tools_path = tools_path or settings.TOOLS_MODULE_PATH
        self.plugins_path = plugins_path if plugins_path is not None else settings.PLUGINS_MODULE_PATH
        self.app = app
        self._registry = None
        self._initialized = False
        # list_tools() snapshot; rebuilt lazily, dropped on reload().
        self._tools_cache: Optional[Dict[str, ToolInfo]] = None
        
    def initialize(self) -> bool:
        """
        Initialize the registry by importing original module.
        Must be called during startup.
        """
        if self._initialized:
            return True
            
        try:
            # Add modules path to Python path
            modules_path = os.path.dirname(self.tools_path)
            if modules_path not in sys.path:
                sys.path.insert(0, modules_path)
                logger.info(f"Added to path: {modules_path}")
            
            # Also add parent (for relative imports in tools)
            parent_path = os.path.dirname(modules_path)
            if parent_path not in sys.path:
                sys.path.insert(0, parent_path)
            
            # Import the original registry
            from tools.registry import ToolRegistry
            
            # Create registry instance
            manifest_path = settings.TOOLS_MANIFEST_PATH
            self._registry = ToolRegistry(app=self.app, manifest_path=manifest_path)
            
            # Discover tools from tools/ and plugins/
            self._registry.discover_dir(root=self.tools_path, recursive=True)
            if self.plugins_path and os.path.isdir(self.plugins_path):
                self._registry.discover_dir(root=self.plugins_path, recursive=True)
                logger.info(f"📦 Also discovered plugins under {self.plugins_path}")

            # Workload behavior is declared by each discovered tool/plugin.
            # Rebuild the generic runtime registry only after all roots load.
            from ..workloads import load_from_tool_registry

            load_from_tool_registry(self._registry)

            tool_count = len(self._registry.list())
            logger.info(f"✅ Initialized ToolRegistry with {tool_count} tools from {self.tools_path}")
            
            self._tools_cache = None
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ToolRegistry: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @property
    def is_ready(self) -> bool:
        """Check if registry is ready."""
        return self._initialized and self._registry is not None

    @property
    def tool_registry(self):
        """Underlying modules ToolRegistry (for app.state.registry in tool-runner)."""
        return self._registry
    
    def invalidate_tools_cache(self) -> None:
        """Drop the cached ``list_tools()`` snapshot (after reload / re-register)."""
        self._tools_cache = None

    def list_tools(self) -> Dict[str, ToolInfo]:
        """Get all available tools with their metadata.

        The ``ToolInfo`` map is built once from the underlying registry and
        reused until :meth:`reload` (or :meth:`invalidate_tools_cache`); the
        hot ``get_tool`` path on every execute no longer rebuilds it. Callers
        must treat the returned dict as read-only.
        """
        if not self.is_ready:
            return {}

        cached = self._tools_cache
        if cached is not None:
            return cached
        
        raw_list = self._registry.list()
        result = {}
        
        for tool_id, info in raw_list.items():
            result[tool_id] = ToolInfo(
                id=tool_id,
                description=info.get("description", ""),
                category=info.get("category", "general"),
                capabilities=info.get("capabilities", []),
                args=info.get("args", {}),
                parameters=info.get("parameters", {}),
                enabled=info.get("enabled", True),
                version=info.get("version"),
                timeout_sec=info.get("timeout_sec"),
                supports_persist=info.get("supports_persist", False),
                has_ui=info.get("has_ui", False),
                channels=info.get("channels", []),
            )
        
        self._tools_cache = result
        return result
    
    def get_tool(self, tool_id: str) -> Optional[ToolInfo]:
        """Get info for a specific tool."""
        tools = self.list_tools()
        return tools.get(tool_id)
    
    def get_openai_tools(self, channel: str = "llm") -> List[Dict[str, Any]]:
        """
        Get OpenAI-compatible tool definitions.
        
        Args:
            channel: فیلتر بر اساس کانال (default: llm)
        """
        if not self.is_ready:
            return []
        return self._registry.as_openai_tools(channel=channel)
    
    def get_manifest(self, channel: str = None) -> List[Dict[str, Any]]:
        """
        Get categorized manifest for orchestrator.
        
        Args:
            channel: فیلتر بر اساس کانال (اختیاری)
        """
        if not self.is_ready:
            return []
        return self._registry.categorized_manifest(channel=channel)
    
    def get_multi_agent_manifest(self, channel: str = None) -> Dict[str, Any]:
        """
        Get multi-agent categorized manifest.
        
        Args:
            channel: فیلتر بر اساس کانال (اختیاری)
        """
        if not self.is_ready:
            return {"agents": {}, "all_tools": [], "category_mapping": {}}
        return self._registry.multi_agent_manifest(channel=channel)
    
    def get_agent_tools(self, agent_type: str) -> List[Dict[str, Any]]:
        """Get tools for a specific agent type."""
        if not self.is_ready:
            return []
        return self._registry.get_agent_tools(agent_type)
    
    def get_tools_by_file_requirement(self, channel: str = None) -> Dict[str, Any]:
        """
        Get tools categorized by file requirement.
        
        Args:
            channel: فیلتر بر اساس کانال (اختیاری)
        """
        if not self.is_ready:
            return {"file_required": [], "no_file_required": [], "file_optional": []}
        return self._registry.get_tools_by_file_requirement(channel=channel)
    
    def get_prompt_friendly_tools_info(self, channel: str = "llm") -> str:
        """
        Get dynamic prompt-friendly tools description.
        
        Args:
            channel: فیلتر بر اساس کانال (default: llm)
        """
        if not self.is_ready:
            return ""
        return self._registry.get_prompt_friendly_tools_info(channel=channel)
    
    def get_schema(self) -> Dict[str, Any]:
        """Get full schema for all tools."""
        if not self.is_ready:
            return {}
        return self._registry.schema()
    
    def get_tools_by_category_and_channel(
        self, 
        category: str = None, 
        channel: str = "llm",
        include_general: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get tools filtered by category and channel.
        For file-type based filtering.
        
        Args:
            category: Tool category (image, audio, text, face, file, embed, entity, geo)
            channel: Channel (llm, ui, metadata, rawdata, embed)
            include_general: Include general tools (general, file, text, entity)?
        
        Returns:
            List of tools in OpenAI tools format
        """
        if not self.is_ready:
            return []
        return self._registry.get_tools_by_category_and_channel(
            category=category, 
            channel=channel, 
            include_general=include_general
        )
    
    def get_available_categories(self) -> List[str]:
        """Get list of all available tool categories."""
        if not self.is_ready:
            return []
        return self._registry.get_available_categories()
    
    def map_file_type_to_category(self, file_type: str) -> str:
        """Map file type (MIME or extension) to tool category."""
        if not self.is_ready:
            return "file"
        return self._registry.map_file_type_to_category(file_type)
    
    def validate_args(self, tool_id: str, args: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
        """Validate and coerce arguments for a tool."""
        if not self.is_ready:
            return {}, ["registry_not_initialized"]
        return self._registry.validate_args(tool_id, args)
    
    async def execute(
        self, 
        tool_id: str, 
        args: Dict[str, Any],
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a tool with given arguments.
        
        Args:
            tool_id: Tool identifier
            args: Tool arguments
            validate: Whether to validate args first
            
        Returns:
            Tool execution result
        """
        if not self.is_ready:
            return {"ok": False, "error": "registry_not_initialized"}

        auth_context = (
            args.get("_auth_context")
            if isinstance(args.get("_auth_context"), dict)
            else None
        )
        
        # Validate args if requested
        if validate:
            validated_args, errors = self.validate_args(tool_id, args)
            if errors:
                return {"ok": False, "error": f"validation_failed: {', '.join(errors)}"}
            args = validated_args
        args = dict(args)
        args.pop("_auth_context", None)
        
        # =================================================================
        # MinIO Path Resolution Middleware
        # Download minio:// files to local temp files before execution
        # =================================================================
        resolver = MinIOPathResolver()
        storage_token = set_storage_context(auth_context)
        storage_api_token = None
        try:
            from modules.tools import storage_api

            headers = {
                "X-Internal-Key": settings.INTERNAL_API_KEY,
                "X-Service-Name": settings.SERVICE_NAME,
            }
            if auth_context:
                if auth_context.get("user_id"):
                    headers["X-User-Id"] = str(auth_context["user_id"])
                if auth_context.get("user_roles"):
                    headers["X-User-Roles"] = str(auth_context["user_roles"])
                if auth_context.get("group_id"):
                    headers["X-Group-Id"] = str(auth_context["group_id"])
                if auth_context.get("case_id"):
                    headers["X-Case-Id"] = str(auth_context["case_id"])
            storage_api_token = storage_api.set_auth_headers(headers)
        except Exception:
            storage_api_token = None
        try:
            resolved_args, had_minio = await resolver.resolve_args(
                args,
                auth_context=auth_context,
            )
            if had_minio:
                logger.info(f"🌐 MinIO middleware: resolved paths for {tool_id}")

            # Only require a local file when the arg was a MinIO URI we hydrated.
            # Map/layer tools pass relative directory paths (e.g. sas-both) — not files.
            for key in resolver.PATH_ARGS:
                val = resolved_args.get(key)
                orig = args.get(key)
                if not isinstance(val, str) or not val:
                    continue
                if isinstance(orig, str) and MinIOPathResolver._is_minio_uri(orig):
                    if not os.path.isfile(val):
                        logger.warning(
                            "Path arg %s not found after hydration: %s", key, val[:120]
                        )
                        return {
                            "ok": False,
                            "error": "file_not_found",
                            "detail": f"{key}={val}",
                        }
            for key in resolver.LIST_PATH_ARGS:
                vals = resolved_args.get(key)
                origs = args.get(key)
                if not isinstance(vals, list) or not isinstance(origs, list):
                    continue
                for i, (val, orig) in enumerate(zip(vals, origs)):
                    if not isinstance(val, str) or not val:
                        continue
                    if isinstance(orig, str) and MinIOPathResolver._is_minio_uri(orig):
                        if not os.path.isfile(val):
                            logger.warning(
                                "Path list %s[%s] not found after hydration: %s",
                                key,
                                i,
                                val[:120],
                            )
                            return {
                                "ok": False,
                                "error": "file_not_found",
                                "detail": f"{key}[{i}]={val}",
                            }

            # Tell internal LLM gateway who is calling, so binding lookups
            # (tool.<id>.<purpose>) can override the model per admin policy.
            # Nested tools bind as the callee, not as their parent workload.
            from modules.tools.tool_runner_llm_gateway import (
                llm_binding_ids,
                tool_call_context,
            )
            bind_meta = None
            if self._registry is not None:
                try:
                    adapter = self._registry.get(tool_id)
                    bind_meta = getattr(adapter, "meta", None) if adapter else None
                except Exception:
                    bind_meta = None
            bind_tool, bind_plugin = llm_binding_ids(tool_id, meta=bind_meta)
            with tool_call_context(bind_tool, bind_plugin):
                result = await self._registry.run(tool_id, **resolved_args)
            
            # Ensure ok field exists
            if "ok" not in result:
                result["ok"] = "error" not in result

            # Central durable persistence: tools declare output_intents only.
            # Persist whenever intents are present, including failed runs that
            # still emit a session envelope.
            if isinstance(result, dict):
                intents = result.get("output_intents")
                if result.get("ok") and (not isinstance(intents, list) or not intents):
                    # Legacy analysis tools: build intents via persist() adapter
                    # when an artifact_id is known from the call context.
                    tool_info = self.get_tool(tool_id)
                    artifact_hint = (
                        (auth_context or {}).get("artifact_id")
                        or resolved_args.get("artifact_id")
                    )
                    if (
                        tool_info
                        and getattr(tool_info, "supports_persist", False)
                        and artifact_hint
                    ):
                        try:
                            data = result.get("data")
                            if isinstance(data, dict) and not data.get("artifact_id"):
                                data["artifact_id"] = artifact_hint
                            else:
                                result.setdefault("artifact_id", artifact_hint)
                            legacy = await self._registry.persist(
                                tool_id,
                                result=result,
                                folder_id=resolved_args.get("folder_id"),
                            )
                            if isinstance(legacy, dict) and isinstance(
                                legacy.get("output_intents"), list
                            ):
                                result["output_intents"] = legacy["output_intents"]
                        except Exception as legacy_exc:
                            logger.debug(
                                "Legacy persist intent build skipped for %s: %s",
                                tool_id,
                                legacy_exc,
                            )

                # Strip legacy session.json blob intents — sessions are Postgres-only.
                if isinstance(result.get("output_intents"), list):
                    try:
                        from modules.tools.tool_sessions import strip_session_blob_intents

                        result["output_intents"] = strip_session_blob_intents(
                            result.get("output_intents")
                        )
                    except Exception as strip_exc:
                        logger.debug(
                            "strip_session_blob_intents skipped for %s: %s",
                            tool_id,
                            strip_exc,
                        )

                # Durable plugin/tool session → Storage tool_sessions (no MinIO file).
                plugin_session = result.get("plugin_session")
                if isinstance(plugin_session, dict) and plugin_session:
                    try:
                        from modules.tools.tool_sessions import upsert_tool_session

                        persist_auth = dict(auth_context or {})
                        plugin_seg = str(
                            plugin_session.get("plugin_id")
                            or persist_auth.get("plugin_id")
                            or ""
                        ).strip()
                        if not plugin_seg:
                            raise ValueError(
                                f"explicit_plugin_id_required:{tool_id}"
                            )
                        row = await upsert_tool_session(
                            envelope=plugin_session,
                            auth_context=persist_auth,
                            plugin_id=str(
                                plugin_session.get("plugin_id") or plugin_seg
                            ),
                            tool_id=tool_id,
                            record_id=str(
                                plugin_session.get("record_id")
                                or plugin_session.get("flow_id")
                                or ""
                            ),
                            run_id=str(plugin_session.get("run_id") or ""),
                            internal_api_key=settings.INTERNAL_API_KEY,
                        )
                        if isinstance(row, dict) and (
                            row.get("session_id") or row.get("id")
                        ):
                            sid = str(row.get("session_id") or row.get("id") or "")
                            result["plugin_session_id"] = sid
                            plugin_session["session_id"] = sid
                    except Exception as sess_exc:
                        logger.error(
                            "tool_session upsert failed for %s: %s", tool_id, sess_exc
                        )
                        result["plugin_session_report"] = {
                            "ok": False,
                            "error": str(sess_exc),
                        }

                if isinstance(result.get("output_intents"), list) and result.get(
                    "output_intents"
                ):
                    try:
                        from modules.tools.output_persistence import persist_outputs

                        persist_auth = dict(auth_context or {})
                        if resolved_args.get("session_id") and not persist_auth.get(
                            "session_id"
                        ):
                            persist_auth["session_id"] = resolved_args.get("session_id")
                        if resolved_args.get("run_id") and not persist_auth.get("run_id"):
                            persist_auth["run_id"] = resolved_args.get("run_id")
                        persist_profile = None
                        try:
                            adapter = (
                                self._registry.get_adapter(tool_id)
                                if self._registry
                                else None
                            )
                            meta = getattr(adapter, "meta", None) or {}
                            if isinstance(meta.get("persist_profile"), dict):
                                persist_profile = meta.get("persist_profile")
                            # Allow tool.json top-level plugin_id / profile.plugin_id
                            # into auth so OPS taxonomy is data-driven.
                            for key in ("plugin_id",):
                                val = meta.get(key) or (
                                    persist_profile.get(key)
                                    if isinstance(persist_profile, dict)
                                    else None
                                )
                                if val and not persist_auth.get(key):
                                    persist_auth[key] = val
                        except Exception:
                            persist_profile = None
                        await persist_outputs(
                            tool_id=tool_id,
                            result=result,
                            auth_context=persist_auth,
                            folder_id=resolved_args.get("folder_id"),
                            persist_profile=persist_profile,
                            strict=False,
                            internal_api_key=settings.INTERNAL_API_KEY,
                            service_name=settings.SERVICE_NAME,
                        )
                        try:
                            from ..workloads import workload_registry

                            await workload_registry.stamp_persisted(tool_id, result)
                        except Exception as stamp_exc:
                            logger.debug(
                                "stamp_persisted_artifacts skipped for %s: %s",
                                tool_id,
                                stamp_exc,
                            )
                        plugin_session = result.get("plugin_session")
                        if isinstance(plugin_session, dict) and plugin_session:
                            try:
                                from modules.tools.plugin_session import (
                                    collect_deliverable_files,
                                    normalize_case_visible_envelope,
                                )
                                from modules.tools.tool_sessions import upsert_tool_session

                                # After persist, refresh files roster from stamped
                                # output when Case-visible (soft-link only).
                                files = plugin_session.get("files")
                                if (
                                    plugin_session.get("expose_to_case")
                                    and (
                                        not isinstance(files, list) or not files
                                    )
                                    and plugin_session.get("output") is not None
                                ):
                                    filled = collect_deliverable_files(
                                        plugin_session.get("output")
                                    )
                                    if filled:
                                        plugin_session["files"] = filled
                                plugin_session = normalize_case_visible_envelope(
                                    plugin_session
                                )
                                result["plugin_session"] = plugin_session

                                persist_auth = dict(auth_context or {})
                                plugin_seg = str(
                                    plugin_session.get("plugin_id")
                                    or persist_auth.get("plugin_id")
                                    or ""
                                ).strip()
                                if not plugin_seg:
                                    raise ValueError(
                                        f"explicit_plugin_id_required:{tool_id}"
                                    )
                                await upsert_tool_session(
                                    envelope=plugin_session,
                                    auth_context=persist_auth,
                                    plugin_id=str(
                                        plugin_session.get("plugin_id") or plugin_seg
                                    ),
                                    tool_id=tool_id,
                                    record_id=str(
                                        plugin_session.get("record_id")
                                        or plugin_session.get("flow_id")
                                        or ""
                                    ),
                                    run_id=str(plugin_session.get("run_id") or ""),
                                    internal_api_key=settings.INTERNAL_API_KEY,
                                )
                            except Exception as sess_exc:
                                logger.debug(
                                    "plugin_session restamp skipped for %s: %s",
                                    tool_id,
                                    sess_exc,
                                )
                    except Exception as persist_exc:
                        logger.error(
                            "Auto-persist failed for %s: %s", tool_id, persist_exc
                        )
                        result["persist_report"] = {
                            "ok": False,
                            "error": str(persist_exc),
                            "fatal": True,
                        }
                
            return result
            
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_id} - {e}")
            return {"ok": False, "error": f"execution_failed: {str(e)}"}
        finally:
            if storage_api_token is not None:
                try:
                    from modules.tools import storage_api

                    storage_api.reset_auth_headers(storage_api_token)
                except Exception:
                    pass
            reset_storage_context(storage_token)
            # Always cleanup temp files
            resolver.cleanup()
    
    async def persist(
        self,
        tool_id: str,
        result: Dict[str, Any],
        folder_id: Optional[str] = None,
        auth_context: Optional[Dict[str, Any]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Persist tool result via OutputPersistenceService.

        Prefer ``output_intents`` on the result. If missing and the tool still
        exposes a legacy ``persist()`` hook, call it only to obtain intents
        (hooks must not write Storage themselves).
        """
        if not self.is_ready:
            return {"ok": False, "error": "registry_not_initialized"}

        from modules.tools.output_persistence import persist_outputs
        from modules.tools.persistence import ToolPersistenceError

        payload: Dict[str, Any] = dict(result or {})
        intents = payload.get("output_intents")
        if not isinstance(intents, list) or not intents:
            tool = self.get_tool(tool_id)
            if tool and getattr(tool, "supports_persist", False):
                try:
                    legacy = await self._registry.persist(
                        tool_id,
                        result=payload,
                        folder_id=folder_id,
                    )
                    if isinstance(legacy, dict):
                        legacy_intents = legacy.get("output_intents")
                        if isinstance(legacy_intents, list) and legacy_intents:
                            payload["output_intents"] = legacy_intents
                        elif legacy.get("artifact_id") and legacy.get("metadata"):
                            payload["output_intents"] = [
                                {
                                    "op": "merge_metadata",
                                    "artifact_id": legacy["artifact_id"],
                                    "metadata": legacy["metadata"],
                                }
                            ]
                        payload.setdefault("legacy_persist", legacy)
                except Exception as e:
                    logger.error("Legacy persist adapter failed: %s - %s", tool_id, e)
                    return {"ok": False, "error": f"persist_failed: {str(e)}"}

        storage_token = set_storage_context(auth_context)
        try:
            persist_auth = dict(auth_context or {})
            persist_profile = None
            try:
                adapter = self._registry.get_adapter(tool_id) if self._registry else None
                meta = getattr(adapter, "meta", None) or {}
                if isinstance(meta.get("persist_profile"), dict):
                    persist_profile = meta.get("persist_profile")
                for key in ("plugin_id",):
                    val = meta.get(key) or (
                        persist_profile.get(key)
                        if isinstance(persist_profile, dict)
                        else None
                    )
                    if val and not persist_auth.get(key):
                        persist_auth[key] = val
            except Exception:
                persist_profile = None
            report = await persist_outputs(
                tool_id=tool_id,
                result=payload,
                auth_context=persist_auth,
                folder_id=folder_id,
                persist_profile=persist_profile,
                strict=strict,
                internal_api_key=settings.INTERNAL_API_KEY,
                service_name=settings.SERVICE_NAME,
            )
            from ..workloads import workload_registry

            await workload_registry.stamp_persisted(tool_id, payload)
            return {"ok": bool(report.get("ok", True)), **report, "result": payload}
        except ToolPersistenceError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Tool persistence failed: {tool_id} - {e}")
            return {"ok": False, "error": f"persist_failed: {str(e)}"}
        finally:
            reset_storage_context(storage_token)

    def get_vram_estimates(self) -> Dict[str, Any]:
        """Get VRAM estimates from registry."""
        if not self.is_ready:
            return {"tools": {}, "cpu_only": [], "default_vram_mb": 500}
        return self._registry.get_vram_estimates()

    def get_tools_for_filetype(self, mime: str, channel: Optional[str] = None) -> List[str]:
        """Get tools for given file type."""
        if not self.is_ready:
            return []
        return self._registry.get_tools_for_filetype(mime, channel=channel)

    def get_category_map(self, channel: Optional[str] = None) -> Dict[str, List[str]]:
        """Get category to tool_id mapping."""
        if not self.is_ready:
            return {}
        return self._registry.get_category_map(channel=channel)

    def get_file_based_tools(self, channel: Optional[str] = None) -> List[str]:
        """Get list of tools requiring file input."""
        if not self.is_ready:
            return []
        return self._registry.get_file_based_tools(channel=channel)

    def get_agent_prompt_for_category(self, category: str, channel: Optional[str] = None) -> Optional[str]:
        """Get agent prompt for category."""
        if not self.is_ready:
            return None
        return self._registry.get_agent_prompt_for_category(category, channel=channel)

    def get_all_agent_prompts(self, channel: Optional[str] = None) -> Dict[str, str]:
        """Get all agent prompts."""
        if not self.is_ready:
            return {}
        return self._registry.get_all_agent_prompts(channel=channel)
    
    def cleanup_resources(self) -> bool:
        """
        Cleanup tool resources (ML models, etc).
        Called on shutdown or idle timeout.
        """
        if not self.is_ready:
            return False
        
        try:
            return self._registry.cleanup_resources()
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False

    def cleanup_tool(self, tool_id: str) -> bool:
        """
        Cleanup resources for a specific tool (GPU model eviction).
        Called by GPU Manager when evicting individual models.
        """
        if not self.is_ready:
            return False

        try:
            # Try registry-level per-tool cleanup
            if hasattr(self._registry, "cleanup_tool"):
                return self._registry.cleanup_tool(tool_id)

            # Fallback: access tool module directly
            if hasattr(self._registry, "_tools"):
                tool_entry = self._registry._tools.get(tool_id)
                mod = getattr(tool_entry, "module", None) if tool_entry else None
                if mod and hasattr(mod, "cleanup_model"):
                    mod.cleanup_model()
                    logger.info(f"🗑️ Cleaned up tool model: {tool_id}")
                    return True

            # Fallback 2: try importing the tool module by convention
            try:
                import importlib
                mod = importlib.import_module(f"tools.{tool_id.replace('.', '/')}.tool")
                if hasattr(mod, "cleanup_model"):
                    mod.cleanup_model()
                    logger.info(f"🗑️ Cleaned up tool model (import): {tool_id}")
                    return True
            except (ImportError, ModuleNotFoundError):
                pass

            logger.warning(f"No cleanup_model found for tool: {tool_id}")
            return False
        except Exception as e:
            logger.error(f"Cleanup failed for {tool_id}: {e}")
            return False
    
    def reload(self) -> bool:
        """Reload all tools."""
        if not self._registry:
            return False
        
        try:
            extra = []
            if self.plugins_path and os.path.isdir(self.plugins_path):
                extra.append(self.plugins_path)
            self._registry.reload(root=self.tools_path, recursive=True, extra_roots=extra)
            from ..workloads import load_from_tool_registry

            load_from_tool_registry(self._registry)
            tool_count = len(self._registry.list())
            logger.info(f"♻️ Reloaded {tool_count} tools")
            return True
        except Exception as e:
            logger.error(f"Reload failed: {e}")
            return False
        finally:
            # Registry contents changed (or partially changed on failure).
            self.invalidate_tools_cache()

    def list_plugins(self, *, include_tools: bool = False) -> List[Dict[str, Any]]:
        """plugin.json manifests + tool_ids (for gateway / UI)."""
        if not self.plugins_path or not os.path.isdir(self.plugins_path):
            return []
        try:
            from modules.plugins.discovery import build_plugin_payloads
        except ImportError:
            logger.warning("modules.plugins.discovery not importable; plugins list empty")
            return []
        reg = self._registry if self.is_ready else None
        return build_plugin_payloads(self.plugins_path, registry=reg, include_tools=include_tools)

    def list_plugins_swagger(self) -> List[Dict[str, Any]]:
        """Compact per-plugin tool list for API-Gateway Swagger injection."""
        if not self.plugins_path or not os.path.isdir(self.plugins_path):
            return []
        try:
            from modules.plugins.discovery import build_plugin_swagger_index
        except ImportError:
            logger.warning("modules.plugins.discovery not importable; swagger index empty")
            return []
        return build_plugin_swagger_index(self.plugins_path)


def get_registry() -> ToolRegistryWrapper:
    """Get the singleton registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ToolRegistryWrapper()
    return _registry_instance


def init_registry(app=None) -> ToolRegistryWrapper:
    """Initialize and return the registry."""
    global _registry_instance
    _registry_instance = ToolRegistryWrapper(app=app)
    _registry_instance.initialize()
    return _registry_instance
