"""
Central Storage Service
سرویس مرکزی ذخیره‌سازی - هماهنگ‌کننده همه بک‌اندها
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Union, BinaryIO, Sequence

from .config import settings
from .backends import PostgresBackend, MongoBackend, QdrantBackend, Neo4jBackend, MinioBackend
from .repositories import (
    ImageRepository, AudioRepository, TextRepository,
    UserRepository, RoleRepository, ChatRepository, TraceRepository,
    UserMessengerRepository, ToolSessionRepository,
)

logger = logging.getLogger(__name__)


class Storage:
    """
    Central storage orchestrator
    هماهنگ‌کننده PostgreSQL + MongoDB + Qdrant + Neo4j + MinIO
    
    ارائه‌دهنده یک API یکپارچه برای:
    - مدیریت آرتیفکت‌ها (فایل‌ها)
    - مدیریت افراد
    - مدیریت پوشه‌ها
    - ذخیره نتایج ابزارها
    - جستجوی برداری
    - عملیات گراف
    - مدیریت کاربران و چت‌ها
    - ذخیره‌سازی فایل‌ها در MinIO (S3)
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        mongodb_url: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        minio_enabled: Optional[bool] = None,
    ):
        self.database_url = database_url or settings.DATABASE_URL
        self.mongodb_url = mongodb_url or settings.MONGODB_URL
        self.qdrant_url = qdrant_url or settings.QDRANT_URL
        self.neo4j_uri = neo4j_uri or settings.NEO4J_URI
        self.neo4j_username = settings.NEO4J_USERNAME
        self.neo4j_password = settings.NEO4J_PASSWORD
        
        # MinIO settings
        self.minio_enabled = minio_enabled if minio_enabled is not None else settings.MINIO_ENABLED
        
        # Backends
        self.postgres: Optional[PostgresBackend] = None
        self.mongo: Optional[MongoBackend] = None
        self.qdrant: Optional[QdrantBackend] = None
        self.neo4j: Optional[Neo4jBackend] = None
        self.minio: Optional[MinioBackend] = None
        
        # Domain repositories
        self.images: Optional[ImageRepository] = None
        self.audio: Optional[AudioRepository] = None
        self.texts: Optional[TextRepository] = None
        
        # Auth & Chat repositories
        self.users: Optional[UserRepository] = None
        self.roles: Optional[RoleRepository] = None
        self.chats: Optional[ChatRepository] = None
        self.messenger: Optional[UserMessengerRepository] = None
        self.tool_sessions: Optional[ToolSessionRepository] = None
        
        # Agent tracing repository
        self.traces: Optional[TraceRepository] = None
        
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to all backends"""
        if self._connected:
            return True
        
        try:
            # PostgreSQL (required)
            if self.database_url:
                self.postgres = PostgresBackend(self.database_url)
                await self.postgres.connect()
                logger.info("✅ PostgreSQL connected")
            else:
                logger.warning("⚠️ No DATABASE_URL configured")
            
            # MongoDB (optional)
            if self.mongodb_url:
                self.mongo = MongoBackend(self.mongodb_url)
                await self.mongo.connect()
                logger.info("✅ MongoDB connected")
            else:
                logger.info("ℹ️ MongoDB not configured, skipping")
            
            # Qdrant (optional)
            if self.qdrant_url:
                self.qdrant = QdrantBackend(self.qdrant_url)
                try:
                    await self.qdrant.connect()
                    logger.info("✅ Qdrant connected")
                except Exception as exc:
                    logger.error(f"❌ Qdrant connection failed: {exc}")
                    self.qdrant = None
            else:
                logger.info("ℹ️ Qdrant not configured, skipping")
            
            # Neo4j (optional)
            if self.neo4j_uri:
                self.neo4j = Neo4jBackend(
                    uri=self.neo4j_uri,
                    username=self.neo4j_username,
                    password=self.neo4j_password,
                )
                await self.neo4j.connect()
                logger.info("✅ Neo4j connected")
            else:
                logger.info("ℹ️ Neo4j not configured, skipping")
            
            # MinIO (optional, for S3-compatible storage)
            if self.minio_enabled:
                self.minio = MinioBackend(
                    endpoint=settings.MINIO_ENDPOINT,
                    access_key=settings.MINIO_ACCESS_KEY,
                    secret_key=settings.MINIO_SECRET_KEY,
                    bucket=settings.MINIO_BUCKET,
                    secure=settings.MINIO_SECURE,
                    public_endpoint=settings.MINIO_PUBLIC_ENDPOINT,  # For presigned URLs
                )
                # MinIO may not be ready at the exact second we start (NP / pod race).
                for attempt in range(1, 6):
                    await self.minio.connect()
                    if self.minio.is_connected:
                        logger.info(
                            f"✅ MinIO connected (public: {settings.MINIO_PUBLIC_ENDPOINT})"
                        )
                        break
                    logger.warning(
                        "MinIO not ready (attempt %s/5); retrying…", attempt
                    )
                    await asyncio.sleep(min(2 * attempt, 8))
                if not self.minio.is_connected:
                    logger.error(
                        "❌ MinIO still unavailable after retries — downloads will 503 until reconnect"
                    )
            else:
                logger.info("ℹ️ MinIO not enabled, using local filesystem")
            
            # Initialize repositories
            if self.postgres:
                self.images = ImageRepository(self.postgres)
                self.audio = AudioRepository(self.postgres)
                self.texts = TextRepository(self.postgres, self.qdrant)
                
                # Auth & Chat repositories
                self.users = UserRepository(self.postgres)
                self.roles = RoleRepository(self.postgres)
                self.chats = ChatRepository(self.postgres)
                self.messenger = UserMessengerRepository(self.postgres)
                self.tool_sessions = ToolSessionRepository(self.postgres)
                self.traces = TraceRepository(self.postgres)
                logger.info("✅ Auth & Chat & Messenger & Trace & ToolSession repositories initialized")
            
            self._connected = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Storage connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from all backends"""
        if self.postgres:
            await self.postgres.disconnect()
        if self.mongo:
            await self.mongo.disconnect()
        if self.neo4j:
            await self.neo4j.disconnect()
        if self.minio:
            await self.minio.disconnect()
        self._connected = False
        logger.info("Storage disconnected")
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    # ==================== MinIO File Operations ====================
    
    async def file_upload(
        self,
        file_data: Union[bytes, BinaryIO],
        object_name: str,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Upload file to MinIO (if enabled)."""
        if self.minio and self.minio.is_connected:
            return await self.minio.upload_file(
                file_data=file_data,
                object_name=object_name,
                content_type=content_type,
                metadata=metadata,
            )
        return None
    
    async def file_download(self, object_name: str) -> Optional[bytes]:
        """Download file from MinIO (if enabled)."""
        if self.minio and self.minio.is_connected:
            return await self.minio.download_file(object_name)
        return None
    
    async def file_get_url(self, object_name: str, expires_seconds: int = 3600) -> Optional[str]:
        """Get presigned URL for file download."""
        if self.minio and self.minio.is_connected:
            return await self.minio.get_presigned_url(object_name, expires_seconds=expires_seconds)
        return None

    async def file_get_put_url(
        self,
        object_name: str,
        expires_seconds: int = 3600,
        content_type: Optional[str] = None,
    ) -> Optional[str]:
        """Presigned PUT URL for direct client upload."""
        if self.minio and self.minio.is_connected:
            return await self.minio.get_presigned_put_url(
                object_name,
                expires_seconds=expires_seconds,
                content_type=content_type,
            )
        return None

    async def file_copy(self, src_object: str, dest_object: str) -> bool:
        if self.minio and self.minio.is_connected:
            return await self.minio.copy_object(src_object, dest_object)
        return False

    async def file_stat(self, object_name: str) -> Optional[Dict[str, Any]]:
        if self.minio and self.minio.is_connected:
            return await self.minio.get_file_info(object_name)
        return None
    
    async def file_delete(self, object_name: str) -> bool:
        """Delete file from MinIO."""
        if self.minio and self.minio.is_connected:
            return await self.minio.delete_file(object_name)
        return False
    
    async def file_exists(self, object_name: str) -> bool:
        """Check if file exists in MinIO."""
        if self.minio and self.minio.is_connected:
            return await self.minio.file_exists(object_name)
        return False
    
    # ==================== Person Operations ====================
    
    async def person_create(
        self,
        full_name: str,
        external_id: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Create a new person record"""
        if not self.postgres:
            return None
        return await self.postgres.person_create(
            full_name=full_name,
            external_id=external_id,
            aliases=aliases,
            metadata=metadata
        )
    
    async def person_get(self, person_id: str) -> Optional[Dict]:
        """Get a person by ID"""
        if not self.postgres:
            return None
        return await self.postgres.person_get(person_id)
    
    async def person_list(
        self,
        limit: int = 100,
        offset: int = 0,
        search: Optional[str] = None
    ) -> List[Dict]:
        """List persons with optional search"""
        if not self.postgres:
            return []
        return await self.postgres.person_list(limit=limit, offset=offset, search=search)
    
    # ==================== Folder Operations ====================
    
    async def folder_get_or_create(
        self,
        name: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Get or create a folder"""
        if not self.postgres:
            return None
        return await self.postgres.folder_get_or_create(
            name=name,
            parent_id=parent_id,
            metadata=metadata
        )
    
    # ==================== Artifact Operations ====================
    
    async def artifact_register(
        self,
        storage_path: str,
        media_type: str,
        original_filename: Optional[str] = None,
        folder_id: Optional[str] = None,
        checksum: Optional[str] = None,
        file_size_bytes: Optional[int] = None,
        mime_type: Optional[str] = None,
        metadata: Optional[Dict] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """Register a new artifact"""
        if not self.postgres:
            return None
        # artifact_upsert expects 'path' as first positional arg
        artifact_id = await self.postgres.artifact_upsert(
            path=storage_path,
            media_type=media_type,
            original_filename=original_filename,
            folder_id=folder_id,
            checksum=checksum,
            file_size_bytes=file_size_bytes,
            mime_type=mime_type,
            metadata=metadata,
            user_id=user_id,
            session_id=session_id,
        )
        return {
            "id": str(artifact_id),
            "artifact_id": str(artifact_id),
            "storage_path": storage_path,
        }
    
    async def artifact_get(self, artifact_id: str) -> Optional[Dict]:
        """Get an artifact by ID"""
        if not self.postgres:
            return None
        return await self.postgres.artifact_get(artifact_id)
    
    async def artifact_get_by_path(self, storage_path: str) -> Optional[Dict]:
        """Get an artifact by storage path"""
        if not self.postgres:
            return None
        return await self.postgres.artifact_get_by_path(storage_path)
    
    async def artifact_list(
        self,
        folder_id: Optional[str] = None,
        media_type: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """List artifacts with filters"""
        if not self.postgres:
            return []
        return await self.postgres.artifact_list(
            folder_id=folder_id,
            media_type=media_type,
            user_id=user_id,
            session_id=session_id,
            limit=limit,
            offset=offset
        )

    async def artifact_is_accessible(
        self, *, user_id: str, artifact_id: str, action: str = "read"
    ) -> Dict[str, Any]:
        if not self.postgres:
            return {"accessible": False, "access": {"is_owner": False, "is_group": False, "is_override": False}}
        return await self.postgres.artifact_is_accessible(
            user_id=user_id, artifact_id=artifact_id, action=action
        )

    async def case_is_accessible(self, *, user_id: str, case_id: str) -> Dict[str, Any]:
        if not self.postgres:
            return {
                "accessible": False,
                "access": {
                    "is_owner": False,
                    "is_group": False,
                    "is_group_assigned": False,
                    "is_group_member": False,
                    "is_override": False,
                },
            }
        return await self.postgres.case_is_accessible(user_id=user_id, case_id=case_id)

    async def case_list_artifacts(
        self,
        case_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """List artifacts belonging to a case-importer case.

        ``limit=None`` returns the full roster (no server-side cap).
        """
        if not self.postgres:
            return []
        return await self.postgres.list_artifacts_for_case(case_id, limit=limit)

    async def case_accessible_list(
        self,
        *,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        session_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.postgres:
            return []
        return await self.postgres.case_accessible_list(
            user_id=user_id,
            limit=limit,
            offset=offset,
            session_id=session_id,
            status=status,
        )

    async def artifact_accessible_list(
        self,
        *,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        sort_by: Optional[str] = None,
        sort_dir: Optional[str] = None,
        folder_id: Optional[str] = None,
        media_type: Optional[str] = None,
        session_id: Optional[str] = None,
        search: Optional[str] = None,
        ownership: str = "any",
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
        size_min: Optional[int] = None,
        size_max: Optional[int] = None,
        mime_type: Optional[str] = None,
        mime_types: Optional[List[str]] = None,
        tags_any: Optional[List[str]] = None,
        tags_all: Optional[List[str]] = None,
        folder_prefix: Optional[str] = None,
        folder_prefixes: Optional[List[str]] = None,
        uploaded_by: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not self.postgres:
            return {"items": [], "total_count": 0}
        return await self.postgres.artifact_accessible_list(
            user_id=user_id,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_dir=sort_dir,
            folder_id=folder_id,
            media_type=media_type,
            session_id=session_id,
            search=search,
            ownership=ownership,
            created_from=created_from,
            created_to=created_to,
            size_min=size_min,
            size_max=size_max,
            mime_type=mime_type,
            mime_types=mime_types,
            tags_any=tags_any,
            tags_all=tags_all,
            folder_prefix=folder_prefix,
            folder_prefixes=folder_prefixes,
            uploaded_by=uploaded_by,
        )

    async def artifact_accessible_stats(
        self,
        *,
        user_id: str,
        folder_id: Optional[str] = None,
        media_type: Optional[str] = None,
        session_id: Optional[str] = None,
        search: Optional[str] = None,
        ownership: str = "any",
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
        size_min: Optional[int] = None,
        size_max: Optional[int] = None,
        mime_type: Optional[str] = None,
        mime_types: Optional[List[str]] = None,
        tags_any: Optional[List[str]] = None,
        tags_all: Optional[List[str]] = None,
        folder_prefix: Optional[str] = None,
        folder_prefixes: Optional[List[str]] = None,
        uploaded_by: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not self.postgres:
            return {"total": {"bytes": 0, "count": 0}, "by_type": {}}
        return await self.postgres.artifact_accessible_stats(
            user_id=user_id,
            folder_id=folder_id,
            media_type=media_type,
            session_id=session_id,
            search=search,
            ownership=ownership,
            created_from=created_from,
            created_to=created_to,
            size_min=size_min,
            size_max=size_max,
            mime_type=mime_type,
            mime_types=mime_types,
            tags_any=tags_any,
            tags_all=tags_all,
            folder_prefix=folder_prefix,
            folder_prefixes=folder_prefixes,
            uploaded_by=uploaded_by,
        )

    async def accessible_folders(
        self,
        *,
        user_id: str,
        prefix: str = "",
        delimiter: str = "/",
        limit: int = 200,
        offset: int = 0,
        ownership: str = "any",
        folder_id: Optional[str] = None,
        media_type: Optional[str] = None,
        session_id: Optional[str] = None,
        search: Optional[str] = None,
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
        size_min: Optional[int] = None,
        size_max: Optional[int] = None,
        mime_type: Optional[str] = None,
        mime_types: Optional[List[str]] = None,
        tags_any: Optional[List[str]] = None,
        tags_all: Optional[List[str]] = None,
        uploaded_by: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not self.postgres:
            return {"prefix": prefix, "delimiter": delimiter, "folders": []}
        return await self.postgres.accessible_folders(
            user_id=user_id,
            prefix=prefix,
            delimiter=delimiter,
            limit=limit,
            offset=offset,
            ownership=ownership,
            folder_id=folder_id,
            media_type=media_type,
            session_id=session_id,
            search=search,
            created_from=created_from,
            created_to=created_to,
            size_min=size_min,
            size_max=size_max,
            mime_type=mime_type,
            mime_types=mime_types,
            tags_any=tags_any,
            tags_all=tags_all,
            uploaded_by=uploaded_by,
        )

    async def accessible_facets(
        self,
        *,
        user_id: str,
        search: Optional[str] = None,
        ownership: str = "any",
        top_mimes: int = 30,
        top_tags: int = 30,
        top_uploaders: int = 10,
        top_sessions: int = 10,
        date_buckets: str = "month",
        folder_id: Optional[str] = None,
        media_type: Optional[str] = None,
        session_id: Optional[str] = None,
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
        size_min: Optional[int] = None,
        size_max: Optional[int] = None,
        mime_type: Optional[str] = None,
        mime_types: Optional[List[str]] = None,
        tags_any: Optional[List[str]] = None,
        tags_all: Optional[List[str]] = None,
        folder_prefix: Optional[str] = None,
        folder_prefixes: Optional[List[str]] = None,
        uploaded_by: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not self.postgres:
            return {"total": {"count": 0, "bytes": 0}, "facets": {"mime_type": []}}
        return await self.postgres.accessible_facets(
            user_id=user_id,
            search=search,
            ownership=ownership,
            top_mimes=top_mimes,
            top_tags=top_tags,
            top_uploaders=top_uploaders,
            top_sessions=top_sessions,
            date_buckets=date_buckets,
            folder_id=folder_id,
            media_type=media_type,
            session_id=session_id,
            created_from=created_from,
            created_to=created_to,
            size_min=size_min,
            size_max=size_max,
            mime_type=mime_type,
            mime_types=mime_types,
            tags_any=tags_any,
            tags_all=tags_all,
            folder_prefix=folder_prefix,
            folder_prefixes=folder_prefixes,
            uploaded_by=uploaded_by,
        )

    async def share_create(
        self,
        *,
        artifact_id: str,
        created_by: str,
        expires_sec: int = 3600,
        scope: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.postgres:
            raise RuntimeError("PostgreSQL backend not available")
        return await self.postgres.share_token_create(
            artifact_id=artifact_id,
            created_by=created_by,
            expires_sec=expires_sec,
            scope=scope,
        )

    async def share_get_by_token(self, *, token: str) -> Optional[Dict[str, Any]]:
        if not self.postgres:
            raise RuntimeError("PostgreSQL backend not available")
        return await self.postgres.share_token_get_by_hash(token=token)

    async def share_revoke(self, *, token: str) -> bool:
        if not self.postgres:
            raise RuntimeError("PostgreSQL backend not available")
        return await self.postgres.share_token_revoke(token=token)

    async def share_audit(
        self,
        *,
        token_id: Optional[str],
        action: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.postgres:
            raise RuntimeError("PostgreSQL backend not available")
        return await self.postgres.share_audit_log(
            token_id=token_id,
            action=action,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata,
        )
    
    async def artifact_update_metadata(
        self,
        artifact_id: str,
        metadata: Dict
    ) -> Optional[Dict]:
        """Merge metadata into an artifact"""
        if not self.postgres:
            return None
        return await self.postgres.artifact_merge_metadata(artifact_id, metadata)

    async def artifact_update_storage_path(
        self,
        artifact_id: str,
        storage_path: str,
    ) -> bool:
        """Update the artifact's storage_path (after MinIO/local promotion)."""
        if not self.postgres:
            return False
        return await self.postgres.artifact_update_storage_path(artifact_id, storage_path)

    async def artifact_update_storage_identity(
        self,
        artifact_id: str,
        *,
        storage_path: Optional[str] = None,
        original_filename: Optional[str] = None,
        media_type: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> bool:
        if not self.postgres:
            return False
        return await self.postgres.artifact_update_storage_identity(
            artifact_id,
            storage_path=storage_path,
            original_filename=original_filename,
            media_type=media_type,
            mime_type=mime_type,
        )

    async def artifact_attach_to_session(
        self,
        *,
        artifact_id: str,
        session_id: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Attach an existing artifact to a chat session workspace."""
        if not self.postgres:
            return None
        return await self.postgres.artifact_attach_to_session(
            artifact_id=artifact_id,
            session_id=session_id,
            user_id=user_id,
        )

    @staticmethod
    def _guess_media_kind(path: str) -> str:
        ext = (path.rsplit(".", 1)[-1] if "." in path else "").lower()
        if ext in {"jpg", "jpeg", "png", "gif", "webp", "bmp", "tif", "tiff", "heic"}:
            return "image"
        if ext in {"mp3", "wav", "flac", "ogg", "m4a", "aac"}:
            return "audio"
        if ext in {"mp4", "mkv", "avi", "mov", "webm"}:
            return "video"
        if ext in {"txt", "md", "csv", "json", "xml", "html", "log", "py", "js", "ts"}:
            return "text"
        if ext in {"pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx"}:
            return "document"
        return "other"

    @staticmethod
    def _artifact_case_id(meta: Any) -> Optional[str]:
        if isinstance(meta, str):
            try:
                import json as _json

                meta = _json.loads(meta)
            except Exception:
                return None
        if not isinstance(meta, dict):
            return None
        for key in ("case_id", "case_importer_last_case_id"):
            val = meta.get(key)
            if val:
                return str(val)
        return None

    @staticmethod
    def summarize_artifact_candidate(art: Dict[str, Any]) -> Dict[str, Any]:
        """Compact row for LLM disambiguation lists."""
        name = art.get("original_filename") or "unnamed"
        return {
            "artifact_id": str(art.get("id") or ""),
            "filename": name,
            "name": name,
            "path": art.get("storage_path"),
            "storage_path": art.get("storage_path"),
            "case_id": Storage._artifact_case_id(art.get("metadata")),
            "media_type": art.get("media_type"),
            "mime_type": art.get("mime_type"),
            "size_bytes": art.get("file_size_bytes"),
        }

    async def artifact_search_accessible(
        self,
        *,
        user_id: str,
        artifact_id: Optional[str] = None,
        case_id: Optional[str] = None,
        path: Optional[str] = None,
        filename: Optional[str] = None,
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        """Return accessible artifacts matching any provided filter."""
        if not self.postgres:
            return []
        uid = str(user_id or "").strip()
        if not uid:
            return []

        rows = await self.postgres.artifact_search_by_filters(
            artifact_id=artifact_id,
            case_id=case_id,
            path=path,
            filename=filename,
            limit=limit,
        )
        accessible: List[Dict[str, Any]] = []
        for row in rows:
            aid = str(row.get("id") or "")
            if not aid:
                continue
            access = await self.artifact_is_accessible(user_id=uid, artifact_id=aid)
            if access.get("accessible"):
                accessible.append(row)
        return accessible

    async def artifact_prepare_attach(
        self,
        *,
        user_id: str,
        artifact_id: Optional[str] = None,
        case_id: Optional[str] = None,
        path: Optional[str] = None,
        filename: Optional[str] = None,
        register_if_local_file: bool = True,
        search_limit: int = 25,
    ) -> Dict[str, Any]:
        """
        Resolve zero/one/many artifacts for session attach.

        Returns:
            status: attached_candidate | ambiguous | not_found
            candidates: list of summaries
            artifact: single row when status is attached_candidate
        """
        uid = str(user_id or "").strip()
        if not uid:
            return {"status": "not_found", "candidates": [], "artifact": None}

        path_hint = (path or "").strip()
        fn = (filename or "").strip()
        cid = (case_id or "").strip()
        aid = (artifact_id or "").strip()

        if not any([aid, path_hint, fn, cid]):
            return {"status": "not_found", "candidates": [], "artifact": None}

        matches = await self.artifact_search_accessible(
            user_id=uid,
            artifact_id=aid or None,
            case_id=cid or None,
            path=path_hint or None,
            filename=fn or None,
            limit=search_limit,
        )

        # Path-style locators (e.g. /data/uploads/جعفر) often mean "file named X" in storage.
        if not matches and path_hint and not fn:
            basename = os.path.basename(path_hint.rstrip("/"))
            if basename:
                matches = await self.artifact_search_accessible(
                    user_id=uid,
                    artifact_id=aid or None,
                    case_id=cid or None,
                    path=None,
                    filename=basename,
                    limit=search_limit,
                )

        if not matches and register_if_local_file and path_hint and os.path.isfile(path_hint):
            try:
                size = os.path.getsize(path_hint)
            except OSError:
                size = None
            media = self._guess_media_kind(path_hint)
            basename = os.path.basename(path_hint)
            meta: Dict[str, Any] = {"attached_from_path": path_hint}
            if cid:
                meta["case_id"] = cid
            rel_candidates = self.postgres._import_locator_candidates(path_hint)
            if rel_candidates:
                rel, fn_meta = rel_candidates[0]
                meta["case_importer_rel_path"] = rel
                meta["case_importer_filename"] = fn_meta
            registered = await self.artifact_register(
                storage_path=path_hint,
                media_type=media,
                original_filename=basename,
                file_size_bytes=size,
                user_id=uid,
                metadata=meta,
            )
            if registered and registered.get("id"):
                row = await self.artifact_get(str(registered["id"]))
                if row:
                    matches = [row]

        summaries = [self.summarize_artifact_candidate(r) for r in matches]

        if len(matches) == 1:
            return {
                "status": "attached_candidate",
                "candidates": summaries,
                "artifact": matches[0],
            }
        if len(matches) > 1:
            return {
                "status": "ambiguous",
                "candidates": summaries,
                "artifact": None,
            }
        return {"status": "not_found", "candidates": [], "artifact": None}

    async def artifact_resolve_for_attach(
        self,
        *,
        user_id: str,
        artifact_id: Optional[str] = None,
        path: Optional[str] = None,
        filename: Optional[str] = None,
        case_id: Optional[str] = None,
        register_if_local_file: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Backward-compatible: return the single match only (first if many)."""
        prep = await self.artifact_prepare_attach(
            user_id=user_id,
            artifact_id=artifact_id,
            case_id=case_id,
            path=path,
            filename=filename,
            register_if_local_file=register_if_local_file,
            search_limit=12,
        )
        if prep.get("status") == "attached_candidate":
            art = prep.get("artifact")
            return art if isinstance(art, dict) else None
        if prep.get("status") == "ambiguous":
            cands = prep.get("candidates") or []
            if cands and cands[0].get("artifact_id"):
                return await self.artifact_get(str(cands[0]["artifact_id"]))
        return None

    async def case_update_metadata(
        self,
        case_id: str,
        metadata: Dict,
    ) -> bool:
        """Merge metadata jsonb into a case row."""
        if not self.postgres:
            return False
        return await self.postgres.case_merge_metadata(case_id, metadata)

    async def artifact_find_by_checksum(
        self,
        checksum: str,
        *,
        user_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """First artifact matching SHA-256 (optional same-user scope)."""
        if not self.postgres:
            return None
        return await self.postgres.artifact_find_first_by_checksum(
            checksum, user_id=user_id
        )

    async def artifact_find_by_import_locator(
        self,
        relative_path: str,
        filename: str,
        *,
        user_id: Optional[str] = None,
    ) -> Optional[Dict]:
        if not self.postgres:
            return None
        return await self.postgres.artifact_find_first_by_import_locator(
            relative_path, filename, user_id=user_id
        )

    async def case_importer_prefs_get(self, user_id: str) -> Optional[Dict]:
        if not self.postgres:
            return None
        return await self.postgres.case_importer_prefs_get(user_id)

    async def case_importer_prefs_upsert(
        self,
        user_id: str,
        tool_allowlist: Optional[Any],
    ) -> Optional[Dict]:
        if not self.postgres:
            return None
        return await self.postgres.case_importer_prefs_upsert(user_id, tool_allowlist)

    # ==================== LLM Control Plane ====================

    async def llm_external_endpoint_upsert(self, **kwargs) -> Dict[str, Any]:
        if not self.postgres:
            raise RuntimeError("PostgreSQL backend not available")
        return await self.postgres.llm_external_endpoint_upsert(**kwargs)

    async def llm_external_endpoints_list(self) -> List[Dict[str, Any]]:
        if not self.postgres:
            return []
        return await self.postgres.llm_external_endpoints_list()

    async def llm_external_endpoint_delete(self, endpoint_id: str) -> bool:
        if not self.postgres:
            return False
        return await self.postgres.llm_external_endpoint_delete(endpoint_id)

    async def llm_external_endpoint_find_by_address(self, **kwargs) -> Optional[Dict[str, Any]]:
        if not self.postgres:
            return None
        return await self.postgres.llm_external_endpoint_find_by_address(**kwargs)

    async def llm_external_endpoint_merge_into(self, **kwargs) -> Dict[str, Any]:
        if not self.postgres:
            raise RuntimeError("PostgreSQL backend not available")
        return await self.postgres.llm_external_endpoint_merge_into(**kwargs)

    async def llm_external_endpoint_update(self, endpoint_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        if not self.postgres:
            return None
        return await self.postgres.llm_external_endpoint_update(endpoint_id, **kwargs)

    async def llm_endpoint_credentials_list(self, endpoint_id: str) -> List[Dict[str, Any]]:
        if not self.postgres:
            return []
        return await self.postgres.llm_endpoint_credentials_list(endpoint_id)

    async def llm_endpoint_credentials_list_all(self) -> List[Dict[str, Any]]:
        if not self.postgres:
            return []
        return await self.postgres.llm_endpoint_credentials_list_all()

    async def llm_endpoint_credential_get(
        self, endpoint_id: str, cred_id: str
    ) -> Optional[Dict[str, Any]]:
        if not self.postgres:
            return None
        return await self.postgres.llm_endpoint_credential_get(endpoint_id, cred_id)

    async def llm_endpoint_credential_create(self, endpoint_id: str, **kwargs) -> Dict[str, Any]:
        if not self.postgres:
            raise RuntimeError("PostgreSQL backend not available")
        return await self.postgres.llm_endpoint_credential_create(endpoint_id, **kwargs)

    async def llm_endpoint_credential_update(
        self, endpoint_id: str, cred_id: str, **kwargs
    ) -> Optional[Dict[str, Any]]:
        if not self.postgres:
            return None
        return await self.postgres.llm_endpoint_credential_update(endpoint_id, cred_id, **kwargs)

    async def llm_endpoint_credentials_set_rpm_for_model(
        self, endpoint_id: str, **kwargs
    ) -> List[Dict[str, Any]]:
        if not self.postgres:
            raise RuntimeError("PostgreSQL backend not available")
        return await self.postgres.llm_endpoint_credentials_set_rpm_for_model(
            endpoint_id, **kwargs
        )

    async def llm_endpoint_credential_delete(self, endpoint_id: str, cred_id: str) -> bool:
        if not self.postgres:
            return False
        return await self.postgres.llm_endpoint_credential_delete(endpoint_id, cred_id)

    async def llm_endpoint_credentials_ensure_default_from_auth(
        self, endpoint_id: str
    ) -> Optional[Dict[str, Any]]:
        if not self.postgres:
            return None
        return await self.postgres.llm_endpoint_credentials_ensure_default_from_auth(endpoint_id)

    async def llm_local_model_upsert(self, **kwargs) -> Dict[str, Any]:
        if not self.postgres:
            raise RuntimeError("PostgreSQL backend not available")
        return await self.postgres.llm_local_model_upsert(**kwargs)

    async def llm_local_models_list(self) -> List[Dict[str, Any]]:
        if not self.postgres:
            return []
        return await self.postgres.llm_local_models_list()

    async def llm_local_model_delete(
        self, model_name: str, *, node_id: Optional[str] = None
    ) -> bool:
        if not self.postgres:
            return False
        return await self.postgres.llm_local_model_delete(model_name, node_id=node_id)

    async def llm_local_model_patch(self, **kwargs) -> Optional[Dict[str, Any]]:
        if not self.postgres:
            return None
        return await self.postgres.llm_local_model_patch(**kwargs)

    async def llm_local_model_get(self, **kwargs) -> Optional[Dict[str, Any]]:
        if not self.postgres:
            return None
        return await self.postgres.llm_local_model_get(**kwargs)

    async def llm_nodes_list(self) -> List[Dict[str, Any]]:
        if not self.postgres:
            return []
        return await self.postgres.llm_nodes_list()

    async def llm_node_upsert(self, **kwargs) -> Dict[str, Any]:
        if not self.postgres:
            raise RuntimeError("PostgreSQL backend not available")
        return await self.postgres.llm_node_upsert(**kwargs)

    async def llm_node_patch(self, **kwargs) -> Optional[Dict[str, Any]]:
        if not self.postgres:
            raise RuntimeError("PostgreSQL backend not available")
        return await self.postgres.llm_node_patch(**kwargs)

    async def llm_node_delete(self, *, id: str) -> bool:
        if not self.postgres:
            return False
        return await self.postgres.llm_node_delete(id=id)

    async def llm_local_models_by_node(self, node_id: str) -> List[Dict[str, Any]]:
        if not self.postgres:
            return []
        return await self.postgres.llm_local_models_by_node(node_id)

    async def llm_route_upsert(self, **kwargs) -> Dict[str, Any]:
        if not self.postgres:
            raise RuntimeError("PostgreSQL backend not available")
        return await self.postgres.llm_route_upsert(**kwargs)

    async def llm_routes_list(self) -> List[Dict[str, Any]]:
        if not self.postgres:
            return []
        return await self.postgres.llm_routes_list()

    async def llm_route_delete(self, route_key: str) -> bool:
        if not self.postgres:
            return False
        return await self.postgres.llm_route_delete(route_key)

    async def llm_routes_purge_logical_ids(self, logical_ids: List[str]) -> Dict[str, Any]:
        if not self.postgres:
            return {"deleted_routes": [], "cleared_fallbacks": []}
        return await self.postgres.llm_routes_purge_logical_ids(logical_ids)

    async def llm_routes_enforce_catalog_bindings(self) -> Dict[str, Any]:
        if not self.postgres:
            return {"deleted_routes": [], "cleared_fallbacks": [], "orphan_routes": []}
        return await self.postgres.llm_routes_enforce_catalog_bindings()

    async def llm_pool_policies_list(self) -> List[Dict[str, Any]]:
        if not self.postgres:
            return []
        return await self.postgres.llm_pool_policies_list()

    async def llm_pool_policy_upsert(self, **kwargs) -> Dict[str, Any]:
        if not self.postgres:
            raise RuntimeError("PostgreSQL backend not available")
        return await self.postgres.llm_pool_policy_upsert(**kwargs)

    async def llm_pool_policy_delete(self, logical_id: str) -> bool:
        if not self.postgres:
            return False
        return await self.postgres.llm_pool_policy_delete(logical_id)

    async def llm_local_model_touch_usage(self, **kwargs) -> bool:
        if not self.postgres:
            return False
        return await self.postgres.llm_local_model_touch_usage(**kwargs)

    async def llm_model_candidate_upsert(self, **kwargs) -> Dict[str, Any]:
        if not self.postgres:
            raise RuntimeError("PostgreSQL backend not available")
        return await self.postgres.llm_model_candidate_upsert(**kwargs)

    async def llm_model_candidates_list(self) -> List[Dict[str, Any]]:
        if not self.postgres:
            return []
        return await self.postgres.llm_model_candidates_list()

    async def llm_gpu_budget(self, *, node: Optional[str] = None, max_fraction: float = 0.95) -> Dict[str, Any]:
        if not self.postgres:
            return {
                "max_fraction": max_fraction,
                "allocated_fraction": 0.0,
                "remaining_fraction": max_fraction,
                "over_budget": False,
                "models": [],
                "node": node,
            }
        return await self.postgres.llm_gpu_budget(node=node, max_fraction=max_fraction)

    async def tool_metadata_rows_for_artifact(
        self,
        tool_id: str,
        artifact_id: str,
    ) -> List[Dict]:
        if not self.postgres:
            return []
        return await self.postgres.tool_metadata_rows_for_artifact(tool_id, artifact_id)

    async def tools_for_artifact(self, artifact_id: str) -> List[Dict]:
        """Discover which dynamic tool_* tables have rows for an artifact_id."""
        if not self.postgres:
            return []
        return await self.postgres.tools_for_artifact(artifact_id)

    async def search_tool_outputs_text(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        channel: Optional[str] = None,
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        """Mongo text search over tool_outputs, RBAC-filtered when user_id is set."""
        if not self.mongo:
            return []
        raw = await self.mongo.search_tool_outputs_text(
            query,
            tool_name=tool_name,
            channel=channel,
            limit=max(int(limit or 30) * 3, 30),
        )
        if not user_id:
            return raw[:limit]
        allowed: List[Dict[str, Any]] = []
        for row in raw:
            aid = str(row.get("artifact_id") or "").strip()
            if not aid:
                continue
            try:
                access = await self.artifact_is_accessible(
                    user_id=str(user_id), artifact_id=aid, action="read"
                )
            except Exception:
                continue
            if (access or {}).get("accessible"):
                allowed.append(row)
                if len(allowed) >= limit:
                    break
        return allowed

    # ==================== Generic cross-DB discovery ====================

    async def artifact_everywhere(
        self,
        artifact_id: str,
        *,
        include: Optional[Sequence[str]] = None,
        per_table_limit: int = 200,
        per_collection_limit: int = 200,
        per_qdrant_limit: int = 100,
        max_relationships: int = 200,
    ) -> Dict[str, Any]:
        """
        For a given `artifact_id`, search EVERY backend for any record/document/
        vector/node that references it, and return a single aggregated payload.

        Discovery is generic — no hard-coded list of tables/collections/labels.
        The contract is simply: "anything that references this artifact has a
        field/property named `artifact_id`".

        Args:
            include: subset of {'minio','postgres','mongo','qdrant','neo4j'}.
                     Defaults to all available backends.
            per_table_limit: cap rows per Postgres table.
            per_collection_limit: cap docs per Mongo collection.
            per_qdrant_limit: cap points per Qdrant collection.
            max_relationships: cap edges returned from Neo4j.

        Returns:
            {
              "artifact_id": "...",
              "minio":    { ... },                # live S3 stat or null
              "postgres": { "<table>": { row_count, returned, truncated, rows: [...] } },
              "mongo":    { "<coll>":  { count, returned, truncated, docs: [...] } },
              "qdrant":   { "<coll>":  { count, truncated, points: [...] } },
              "neo4j":    { "nodes": [...], "relationships": [...] },
              "errors":   { "<backend>": "<message>" },   # only when something failed
            }
        """
        aid = (artifact_id or "").strip()
        if not aid:
            return {"artifact_id": aid, "errors": {"input": "missing_artifact_id"}}

        wanted = {str(x).strip().lower() for x in (include or []) if str(x).strip()}
        if not wanted:
            wanted = {"minio", "postgres", "mongo", "qdrant", "neo4j"}

        out: Dict[str, Any] = {"artifact_id": aid}
        errors: Dict[str, str] = {}

        # --- Build coroutines (skip backends that aren't connected/enabled) ---

        async def _pg() -> Dict[str, Any]:
            if not self.postgres or not self.postgres.pool:
                return {}
            return await self.postgres.find_artifact_everywhere(
                aid, per_table_limit=per_table_limit
            )

        async def _mongo() -> Dict[str, Any]:
            if not self.mongo or self.mongo.db is None:
                return {}
            return await self.mongo.find_artifact_in_all_collections(
                aid, per_collection_limit=per_collection_limit
            )

        async def _qdrant() -> Dict[str, Any]:
            if not self.qdrant or getattr(self.qdrant, "client", None) is None:
                return {}
            return await self.qdrant.scroll_artifact_all_collections(
                aid, per_collection_limit=per_qdrant_limit
            )

        async def _neo4j() -> Dict[str, Any]:
            if not self.neo4j or self.neo4j.driver is None:
                return {}
            return await self.neo4j.find_artifact_subgraph(
                aid, max_relationships=max_relationships
            )

        async def _minio() -> Optional[Dict[str, Any]]:
            if "minio" not in wanted:
                return None
            if not self.postgres:
                return None
            artifact = await self.postgres.artifact_get(aid)
            storage_path = (artifact or {}).get("storage_path") or ""
            if not storage_path:
                return None
            if not (self.minio and self.minio.is_connected):
                return None
            try:
                return await self.minio.stat_storage_path(storage_path)
            except Exception as exc:
                errors["minio"] = str(exc)
                return None

        tasks: Dict[str, Any] = {}
        if "postgres" in wanted:
            tasks["postgres"] = _pg()
        if "mongo" in wanted:
            tasks["mongo"] = _mongo()
        if "qdrant" in wanted:
            tasks["qdrant"] = _qdrant()
        if "neo4j" in wanted:
            tasks["neo4j"] = _neo4j()
        if "minio" in wanted:
            tasks["minio"] = _minio()

        if tasks:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for name, res in zip(tasks.keys(), results):
                if isinstance(res, Exception):
                    errors[name] = f"{type(res).__name__}: {res}"
                    out[name] = None
                else:
                    out[name] = res

        if errors:
            out["errors"] = errors
        return out

    async def artifact_delete(self, artifact_id: str, *, purge_minio: bool = True) -> Dict[str, Any]:
        """Delete an artifact (DB + optional MinIO purge)."""
        if not self.postgres:
            return {"ok": False, "error": "postgres_not_available"}
        if not self.postgres.pool:
            return {"ok": False, "error": "postgres_not_connected"}

        record = await self.postgres.artifact_get(artifact_id)
        if not record:
            return {"ok": False, "error": "artifact_not_found"}

        from .base import to_uuid as _to_uuid

        storage_path = record.get("storage_path") or ""
        async with self.postgres.pool.acquire() as conn:
            res = await conn.execute(
                "DELETE FROM artifacts WHERE id = $1",
                _to_uuid(artifact_id),
            )
        deleted = res.startswith("DELETE ") and not res.endswith(" 0")

        purged = False
        if deleted and purge_minio and self.minio and storage_path.startswith("minio://"):
            try:
                rest = storage_path[len("minio://"):]
                if "/" in rest:
                    _, object_name = rest.split("/", 1)
                    await self.file_delete(object_name)
                    purged = True
            except Exception as exc:
                logger.warning("Failed to purge MinIO for %s: %s", artifact_id, exc)

        return {
            "ok": deleted,
            "artifact_id": artifact_id,
            "deleted": deleted,
            "purged_minio": purged,
            "storage_path": storage_path,
        }

    async def artifact_purge_full(
        self,
        artifact_id: str,
        *,
        purge_minio: bool = True,
    ) -> Dict[str, Any]:
        """Remove artifact row and all cross-DB references (Mongo, PG tool_*, Qdrant, Neo4j)."""
        aid = (artifact_id or "").strip()
        summary: Dict[str, Any] = {"artifact_id": aid}
        errors: List[str] = []

        if self.neo4j and self.neo4j.driver:
            try:
                summary["neo4j"] = await self.neo4j.purge_by_property("artifact_id", aid)
            except Exception as exc:
                errors.append(f"neo4j:{exc}")

        if self.qdrant and getattr(self.qdrant, "client", None):
            try:
                summary["qdrant"] = await self.qdrant.purge_by_payload_field("artifact_id", aid)
            except Exception as exc:
                errors.append(f"qdrant:{exc}")

        if self.mongo and self.mongo.db is not None:
            try:
                summary["mongo"] = await self.mongo.purge_by_field("artifact_id", aid)
            except Exception as exc:
                errors.append(f"mongo:{exc}")

        if self.postgres and self.postgres.pool:
            try:
                summary["postgres_refs"] = await self.postgres.purge_column_references(
                    "artifact_id",
                    aid,
                    skip_tables={"artifacts"},
                )
            except Exception as exc:
                errors.append(f"postgres_refs:{exc}")

        art_del = await self.artifact_delete(aid, purge_minio=purge_minio)
        summary["artifact"] = art_del
        if errors:
            summary["errors"] = errors
        summary["ok"] = bool(art_del.get("deleted")) or art_del.get("error") == "artifact_not_found"
        return summary

    async def case_purge_full(self, case_id: str) -> Dict[str, Any]:
        """
        Delete a case-importer case and all persisted data owned by that case:
        case evidence artifacts, importer tool metadata keyed by case_id,
        Mongo/Qdrant/Neo4j case refs, MinIO case_importer prefix.

        Does **not** delete soft-linked plugin/tool sessions (``tool_sessions``
        with ``payload.case_id`` / ``expose_to_case``) or their user-scoped
        deliverables (``bind_case: false``) — those belong to other plugins.
        """
        cid = (case_id or "").strip()
        summary: Dict[str, Any] = {"case_id": cid}
        errors: List[str] = []

        artifact_ids: List[str] = []
        if self.postgres and self.postgres.pool:
            try:
                artifact_ids = await self.postgres.list_artifact_ids_for_case(cid)
            except Exception as exc:
                errors.append(f"list_artifacts:{exc}")
        summary["artifact_ids"] = artifact_ids

        if self.neo4j and self.neo4j.driver:
            try:
                summary["neo4j"] = await self.neo4j.purge_by_property("case_id", cid)
            except Exception as exc:
                errors.append(f"neo4j:{exc}")

        if self.qdrant and getattr(self.qdrant, "client", None):
            try:
                summary["qdrant"] = await self.qdrant.purge_by_payload_field("case_id", cid)
            except Exception as exc:
                errors.append(f"qdrant:{exc}")

        if self.mongo and self.mongo.db is not None:
            try:
                summary["mongo"] = await self.mongo.purge_by_field("case_id", cid)
            except Exception as exc:
                errors.append(f"mongo:{exc}")

        if self.postgres and self.postgres.pool:
            try:
                summary["postgres_case_refs"] = await self.postgres.purge_column_references(
                    "case_id",
                    cid,
                    skip_tables={"cases"},
                )
            except Exception as exc:
                errors.append(f"postgres_case_refs:{exc}")

        purged_artifacts: List[Dict[str, Any]] = []
        for aid in artifact_ids:
            try:
                purged_artifacts.append(await self.artifact_purge_full(aid, purge_minio=True))
            except Exception as exc:
                errors.append(f"artifact:{aid}:{exc}")
        summary["artifacts_purged"] = len(purged_artifacts)

        minio_deleted = 0
        if self.minio and self.minio.is_connected:
            prefix = f"case_importer/{cid}/"
            try:
                objects = await self.minio.list_objects(prefix=prefix, recursive=True)
                for obj in objects:
                    name = obj.get("name")
                    if name and await self.file_delete(name):
                        minio_deleted += 1
            except Exception as exc:
                errors.append(f"minio_prefix:{exc}")
        summary["minio_objects_deleted"] = minio_deleted

        if self.postgres and self.postgres.pool:
            try:
                summary["case_row_deleted"] = await self.postgres.delete_case_row(cid)
            except Exception as exc:
                errors.append(f"case_row:{exc}")

        if errors:
            summary["errors"] = errors
        summary["ok"] = not errors or bool(summary.get("case_row_deleted") or artifact_ids)
        return summary
    
    # ==================== Tool Run Tracking ====================
    
    async def save_tool_run(
        self,
        session_id: str,
        tool_id: str,
        inputs: Dict,
        output: Dict,
        trace_id: Optional[str] = None,
        step: Optional[str] = None,
        elapsed_ms: Optional[float] = None
    ) -> Optional[Dict]:
        """Save a tool execution record"""
        # Save to PostgreSQL
        pg_result = None
        if self.postgres:
            pg_result = await self.postgres.save_tool_run(
                session_id=session_id,
                tool_id=tool_id,
                inputs=inputs,
                output=output,
                trace_id=trace_id,
                step=step,
                elapsed_ms=elapsed_ms
            )
        
        # Also save to MongoDB for flexible querying
        if self.mongo:
            await self.mongo.append_tool_output(
                session_id=session_id,
                trace_id=trace_id or "",
                tool_id=tool_id,
                output=output
            )
        
        return pg_result
    
    async def get_tool_runs(
        self,
        session_id: str,
        tool_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get tool runs for a session from PostgreSQL"""
        if self.postgres:
            return await self.postgres.get_tool_runs(
                session_id=session_id,
                tool_id=tool_id,
                limit=limit
            )
        return []
    
    # ==================== Observations & Insights ====================
    
    async def save_observation(
        self,
        session_id: str,
        observation_type: str,
        content: str,
        artifact_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """Save an observation from analysis"""
        if not self.mongo:
            return None
        return await self.mongo.save_observation(
            session_id=session_id,
            observation_type=observation_type,
            content=content,
            artifact_id=artifact_id,
            metadata=metadata
        )
    
    async def list_observations(
        self,
        session_id: str,
        observation_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """List observations for a session"""
        if not self.mongo:
            return []
        return await self.mongo.list_observations(
            session_id=session_id,
            observation_type=observation_type,
            limit=limit
        )
    
    async def save_insight(
        self,
        session_id: str,
        insight_type: str,
        title: str,
        content: str,
        confidence: Optional[float] = None,
        sources: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """Save an insight derived from analysis"""
        if not self.mongo:
            return None
        return await self.mongo.save_insight(
            session_id=session_id,
            insight_type=insight_type,
            title=title,
            content=content,
            confidence=confidence,
            sources=sources,
            metadata=metadata
        )
    
    async def list_insights(
        self,
        session_id: str,
        insight_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """List insights for a session"""
        if not self.mongo:
            return []
        return await self.mongo.list_insights(
            session_id=session_id,
            insight_type=insight_type,
            limit=limit
        )
    
    # ==================== Vector Search ====================
    
    async def vector_upsert(
        self,
        collection_name: str,
        vectors: List[List[float]],
        ids: List[str],
        payloads: Optional[List[Dict]] = None,
        dimension: Optional[int] = None
    ) -> bool:
        """Upsert vectors to Qdrant"""
        if not self.qdrant:
            logger.warning("Qdrant not configured")
            return False
        if getattr(self.qdrant, "client", None) is None:
            logger.error("Qdrant backend has no active client (connect() failed)")
            return False

        if not vectors or not ids:
            return True

        if len(vectors) != len(ids):
            logger.error(
                "vector_upsert: len(vectors)=%d != len(ids)=%d",
                len(vectors), len(ids),
            )
            return False

        dim = dimension or (len(vectors[0]) if vectors and vectors[0] else 512)
        await self.qdrant.ensure_collection(collection_name, dimension=dim)

        points: List[Dict[str, Any]] = []
        for idx, (vid, vec) in enumerate(zip(ids, vectors)):
            payload = {}
            if payloads and idx < len(payloads) and isinstance(payloads[idx], dict):
                payload = payloads[idx]
            points.append({"id": vid, "vector": vec, "payload": payload})

        try:
            written = await self.qdrant.upsert_vectors(collection_name, points)
        except Exception as exc:
            logger.exception(
                "vector_upsert failed (collection=%s, count=%d): %s",
                collection_name, len(points), exc,
            )
            return False
        if not written:
            logger.error(
                "vector_upsert wrote 0 points to collection %s (expected %d)",
                collection_name, len(points),
            )
            return False
        return True
    
    async def vector_search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filter_dict: Optional[Dict] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """Search vectors in Qdrant"""
        if not self.qdrant or getattr(self.qdrant, "client", None) is None:
            return []
        return await self.qdrant.search(
            collection=collection_name,
            query_vector=query_vector,
            limit=limit,
            filter_dict=filter_dict,
            score_threshold=score_threshold,
        )

    async def vector_count_by_case(self, case_id: str) -> Dict[str, Any]:
        """Exact point counts in Qdrant for a case_id across all collections."""
        cid = str(case_id or "").strip()
        if not cid or not self.qdrant or getattr(self.qdrant, "client", None) is None:
            return {"ok": False, "case_id": cid, "total": 0, "by_collection": {}}
        by_coll = await self.qdrant.count_by_payload_field("case_id", cid)
        total = int(sum(int(v) for v in by_coll.values()))
        return {
            "ok": True,
            "case_id": cid,
            "total": total,
            "by_collection": by_coll,
        }
    
    # ==================== Graph (Neo4j) ====================
    
    async def graph_query(
        self,
        query: str,
        parameters: Optional[Dict] = None
    ) -> List[Dict]:
        """Execute a read Cypher query on Neo4j"""
        if not self.neo4j:
            return []
        return await self.neo4j.run_query(query, parameters)
    
    async def graph_write(
        self,
        query: str,
        parameters: Optional[Dict] = None
    ) -> Dict:
        """Execute a write Cypher query on Neo4j"""
        if not self.neo4j:
            return {"records": [], "counters": {}}
        return await self.neo4j.run_write(query, parameters)
    
    async def graph_write_batch(
        self,
        queries: List[Dict]
    ) -> Dict:
        """Execute multiple write Cypher queries in a single transaction"""
        if not self.neo4j:
            return {"count": 0, "counters": {}}
        return await self.neo4j.run_write_batch(queries)
    
    # ==================== Feedback ====================
    
    async def save_feedback(
        self,
        session_id: str,
        rating: str,
        comment: Optional[str] = None,
        message_text: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Save user feedback"""
        if not self.postgres:
            return None
        return await self.postgres.feedback_save(
            session_id=session_id,
            rating=rating,
            comment=comment,
            message_text=message_text,
            metadata=metadata
        )
    
    # ==================== Health Check ====================
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all backends"""
        status = {
            "healthy": True,
            "backends": {}
        }
        
        # PostgreSQL
        if self.postgres:
            try:
                if not getattr(self.postgres, "pool", None):
                    raise RuntimeError("pool not initialized")
                async with self.postgres.pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                status["backends"]["postgres"] = "ok"
            except Exception as e:
                status["backends"]["postgres"] = f"error: {e}"
                status["healthy"] = False
        else:
            status["backends"]["postgres"] = "not_configured"
        
        # MongoDB
        if self.mongo:
            try:
                await self.mongo.db.command("ping")
                status["backends"]["mongodb"] = "ok"
            except Exception as e:
                status["backends"]["mongodb"] = f"error: {e}"
                status["healthy"] = False
        else:
            status["backends"]["mongodb"] = "not_configured"
        
        # Qdrant
        if self.qdrant:
            try:
                self.qdrant.client.get_collections()
                status["backends"]["qdrant"] = "ok"
            except Exception as e:
                status["backends"]["qdrant"] = f"error: {e}"
                status["healthy"] = False
        else:
            status["backends"]["qdrant"] = "not_configured"
        
        # Neo4j
        if self.neo4j:
            try:
                ok = await self.neo4j.health_check()
                status["backends"]["neo4j"] = "ok" if ok else "error"
                if not ok:
                    status["healthy"] = False
            except Exception as e:
                status["backends"]["neo4j"] = f"error: {e}"
                status["healthy"] = False
        else:
            status["backends"]["neo4j"] = "not_configured"
        
        return status


# Singleton instance
_storage: Optional[Storage] = None


async def get_storage() -> Storage:
    """Get or create storage singleton"""
    global _storage
    if _storage is None:
        _storage = Storage()
        await _storage.connect()
    return _storage


async def close_storage():
    """Close storage connection"""
    global _storage
    if _storage:
        await _storage.disconnect()
        _storage = None
