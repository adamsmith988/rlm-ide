"""
MinIO backend for S3-compatible object storage.
ذخیره‌سازی فایل‌ها برای مقیاس پتابایت
"""
from __future__ import annotations

import asyncio
import io
import logging
from typing import Any, BinaryIO, Dict, List, Optional, Union

from ..base import BaseBackend

try:
    from minio import Minio
    from minio.error import S3Error
    import urllib3
except ImportError:
    Minio = None
    S3Error = None
    urllib3 = None

log = logging.getLogger(__name__)

# MinIO SDK is synchronous. Keep connect/read short so a dead public IP
# cannot stall the storage event loop (readiness /health/live).
_MINIO_REGION = "us-east-1"


def _ascii_safe_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """MinIO/S3 user metadata values must be US-ASCII; quote anything else.

    Object keys are already ASCII-safe; callers still pass display names like
    Persian/Arabic filenames into metadata and blow up with ValueError.
    """
    if not metadata:
        return None
    from urllib.parse import quote

    out: Dict[str, str] = {}
    for raw_key, raw_val in metadata.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        val = "" if raw_val is None else str(raw_val)
        try:
            val.encode("ascii")
        except UnicodeEncodeError:
            val = quote(val, safe="")
        try:
            key.encode("ascii")
        except UnicodeEncodeError:
            key = quote(key, safe="")
        out[key] = val
    return out or None


def _minio_http_client(*, connect: float = 5.0, read: float = 30.0):
    if urllib3 is None:
        return None
    return urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=connect, read=read),
        retries=False,
        maxsize=10,
    )


class MinioBackend(BaseBackend):
    """
    MinIO S3-compatible object storage backend.
    
    Features:
    - Upload/download files
    - Presigned URLs for direct access
    - Bucket management
    - Object metadata
    """

    def __init__(
        self,
        endpoint: str = "localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin123",
        bucket: str = "uploads",
        secure: bool = False,
        public_endpoint: str = None,  # For presigned URLs - if different from internal endpoint
    ):
        super().__init__()
        self.endpoint = endpoint
        self.public_endpoint = public_endpoint or endpoint  # Use internal endpoint if not specified
        self.access_key = access_key
        self.secret_key = secret_key
        self.default_bucket = bucket
        self.secure = secure
        self.client: Optional[Minio] = None
        self.public_client: Optional[Minio] = None

    async def connect(self) -> None:
        """Initialize MinIO connection and ensure bucket exists."""
        if Minio is None:
            log.warning("⚠️ minio package not installed, MinIO disabled")
            return

        if self.client is not None:
            return

        try:
            log.info(f"Connecting to MinIO at {self.endpoint}...")
            http_client = _minio_http_client()
            kwargs: Dict[str, Any] = {
                "access_key": self.access_key,
                "secret_key": self.secret_key,
                "secure": self.secure,
                "region": _MINIO_REGION,
            }
            if http_client is not None:
                kwargs["http_client"] = http_client
            self.client = Minio(self.endpoint, **kwargs)
            if self.public_endpoint != self.endpoint:
                pub_http = _minio_http_client(connect=2.0, read=5.0)
                pub_kwargs = dict(kwargs)
                if pub_http is not None:
                    pub_kwargs["http_client"] = pub_http
                # region is set so presign does not GetBucketLocation on the public IP.
                self.public_client = Minio(self.public_endpoint, **pub_kwargs)

            # Ensure default bucket exists
            await self._ensure_bucket(self.default_bucket)
            
            self._connected = True
            log.info(f"✅ MinIO connected, bucket '{self.default_bucket}' ready")
            
        except Exception as e:
            log.error(f"❌ MinIO connection failed: {e}")
            self.client = None
            self._connected = False

    async def disconnect(self) -> None:
        """Close MinIO connection."""
        self.client = None
        self.public_client = None
        self._connected = False
        log.info("MinIO disconnected")

    async def _ensure_bucket(self, bucket_name: str) -> bool:
        """Ensure bucket exists, create if not."""
        if not self.client:
            return False
        
        try:
            exists = await asyncio.to_thread(self.client.bucket_exists, bucket_name)
            if not exists:
                await asyncio.to_thread(self.client.make_bucket, bucket_name)
                log.info(f"📦 Created bucket: {bucket_name}")
            return True
        except S3Error as e:
            log.error(f"Bucket operation failed: {e}")
            return False

    async def upload_file(
        self,
        file_data: Union[bytes, BinaryIO],
        object_name: str,
        bucket: Optional[str] = None,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Upload file to MinIO.
        
        Args:
            file_data: File content as bytes or file-like object
            object_name: Object path in bucket (e.g., "session_id/artifact_id.jpg")
            bucket: Target bucket (default: self.default_bucket)
            content_type: MIME type
            metadata: Custom metadata dict
            
        Returns:
            Object path on success, None on failure
        """
        if not self.client:
            log.warning("MinIO not connected, cannot upload")
            return None

        bucket = bucket or self.default_bucket
        
        try:
            # Convert bytes to BytesIO if needed
            if isinstance(file_data, bytes):
                data_stream = io.BytesIO(file_data)
                length = len(file_data)
            else:
                # Assume file-like object
                data_stream = file_data
                # Get length
                data_stream.seek(0, 2)  # Seek to end
                length = data_stream.tell()
                data_stream.seek(0)  # Seek back to start

            await asyncio.to_thread(
                self.client.put_object,
                bucket_name=bucket,
                object_name=object_name,
                data=data_stream,
                length=length,
                content_type=content_type,
                metadata=_ascii_safe_metadata(metadata),
            )
            
            log.info(f"📤 Uploaded to MinIO: {bucket}/{object_name} ({length} bytes)")
            return f"{bucket}/{object_name}"
            
        except S3Error as e:
            log.error(f"Upload failed: {e}")
            return None
        except (ValueError, UnicodeEncodeError) as e:
            # Non-ASCII metadata / other client-side validation — treat as soft fail
            # so callers can fall back to local disk instead of a bare 500.
            log.error(f"Upload failed (metadata/client): {e}")
            return None

    async def download_file(
        self,
        object_name: str,
        bucket: Optional[str] = None,
    ) -> Optional[bytes]:
        """
        Download file from MinIO.
        
        Args:
            object_name: Object path in bucket
            bucket: Source bucket
            
        Returns:
            File content as bytes, None on failure
        """
        if not self.client:
            return None

        bucket = bucket or self.default_bucket
        
        try:
            def _read() -> bytes:
                response = self.client.get_object(bucket, object_name)
                try:
                    return response.read()
                finally:
                    response.close()
                    response.release_conn()

            data = await asyncio.to_thread(_read)
            log.info(f"📥 Downloaded from MinIO: {bucket}/{object_name}")
            return data
            
        except S3Error as e:
            log.error(f"Download failed: {e}")
            return None

    async def get_presigned_url(
        self,
        object_name: str,
        bucket: Optional[str] = None,
        expires_seconds: int = 3600,
    ) -> Optional[str]:
        """
        Generate presigned URL for direct download.
        
        Uses public_endpoint if configured, otherwise uses internal endpoint.
        
        Args:
            object_name: Object path
            bucket: Source bucket
            expires_seconds: URL validity in seconds
            
        Returns:
            Presigned URL string
        """
        if not self.client:
            return None

        bucket = bucket or self.default_bucket
        
        try:
            from datetime import timedelta
            
            # Sign with the browser-reachable endpoint when configured.
            signing_client = self.public_client or self.client
            url = await asyncio.to_thread(
                signing_client.presigned_get_object,
                bucket,
                object_name,
                timedelta(seconds=expires_seconds),
            )
            return url
        except S3Error as e:
            log.error(f"Presigned URL generation failed: {e}")
            return None

    async def get_presigned_put_url(
        self,
        object_name: str,
        bucket: Optional[str] = None,
        expires_seconds: int = 3600,
        content_type: Optional[str] = None,
    ) -> Optional[str]:
        """Presigned PUT for direct browser → MinIO upload."""
        if not self.client:
            return None
        bucket = bucket or self.default_bucket
        try:
            from datetime import timedelta

            signing_client = self.public_client or self.client

            def _sign():
                return signing_client.presigned_put_object(
                    bucket,
                    object_name,
                    expires=timedelta(seconds=expires_seconds),
                )

            url = await asyncio.to_thread(_sign)
            return url
        except Exception as e:
            log.error(f"Presigned PUT generation failed: {e}")
            return None

    async def create_multipart_upload(
        self,
        object_name: str,
        bucket: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> Optional[str]:
        """Start S3 multipart upload; returns upload_id."""
        if not self.client:
            return None
        bucket = bucket or self.default_bucket
        headers: Dict[str, str] = {}
        if content_type:
            headers["Content-Type"] = content_type
        try:
            upload_id = await asyncio.to_thread(
                self.client._create_multipart_upload,
                bucket,
                object_name,
                headers,
            )
            return str(upload_id)
        except Exception as e:
            log.error(f"create_multipart_upload failed: {e}")
            return None

    async def get_presigned_upload_part_url(
        self,
        object_name: str,
        upload_id: str,
        part_number: int,
        bucket: Optional[str] = None,
        expires_seconds: int = 3600,
    ) -> Optional[str]:
        """Presigned PUT for one multipart part (via public endpoint when set)."""
        if not self.client:
            return None
        bucket = bucket or self.default_bucket
        try:
            from datetime import timedelta

            signing_client = self.public_client or self.client

            def _sign():
                return signing_client.get_presigned_url(
                    "PUT",
                    bucket,
                    object_name,
                    expires=timedelta(seconds=expires_seconds),
                    extra_query_params={
                        "partNumber": str(part_number),
                        "uploadId": upload_id,
                    },
                )

            return await asyncio.to_thread(_sign)
        except Exception as e:
            log.error(f"presigned upload_part failed: {e}")
            return None

    async def list_multipart_parts(
        self,
        object_name: str,
        upload_id: str,
        bucket: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.client:
            return []
        bucket = bucket or self.default_bucket
        try:

            def _list():
                out: List[Dict[str, Any]] = []
                result = self.client._list_parts(bucket, object_name, upload_id)
                parts = getattr(result, "parts", None) or []
                for part in parts:
                    out.append(
                        {
                            "part_number": int(getattr(part, "part_number", 0) or 0),
                            "etag": str(getattr(part, "etag", "") or "").strip('"'),
                            "size": int(getattr(part, "size", 0) or 0),
                        }
                    )
                return out

            return await asyncio.to_thread(_list)
        except Exception as e:
            log.warning(f"list_multipart_parts failed: {e}")
            return []

    async def complete_multipart_upload(
        self,
        object_name: str,
        upload_id: str,
        parts: List[Dict[str, Any]],
        bucket: Optional[str] = None,
    ) -> bool:
        if not self.client:
            return False
        bucket = bucket or self.default_bucket
        try:
            from minio.datatypes import Part

            ordered = sorted(parts, key=lambda p: int(p.get("part_number") or 0))
            part_objs = [
                Part(int(p["part_number"]), str(p["etag"]).strip('"'))
                for p in ordered
                if p.get("part_number") and p.get("etag")
            ]
            if not part_objs:
                return False
            await asyncio.to_thread(
                self.client._complete_multipart_upload,
                bucket,
                object_name,
                upload_id,
                part_objs,
            )
            return True
        except Exception as e:
            log.error(f"complete_multipart_upload failed: {e}")
            return False

    async def abort_multipart_upload(
        self,
        object_name: str,
        upload_id: str,
        bucket: Optional[str] = None,
    ) -> bool:
        if not self.client:
            return False
        bucket = bucket or self.default_bucket
        try:
            await asyncio.to_thread(
                self.client._abort_multipart_upload,
                bucket,
                object_name,
                upload_id,
            )
            return True
        except Exception as e:
            log.warning(f"abort_multipart_upload failed: {e}")
            return False

    async def copy_object(
        self,
        src_object: str,
        dest_object: str,
        *,
        src_bucket: Optional[str] = None,
        dest_bucket: Optional[str] = None,
    ) -> bool:
        """Server-side copy within MinIO (no download through app)."""
        if not self.client:
            return False
        src_bucket = src_bucket or self.default_bucket
        dest_bucket = dest_bucket or self.default_bucket
        try:
            from minio.commonconfig import CopySource

            await asyncio.to_thread(
                self.client.copy_object,
                dest_bucket,
                dest_object,
                CopySource(src_bucket, src_object),
            )
            return True
        except Exception as e:
            log.error(f"MinIO copy failed {src_object} → {dest_object}: {e}")
            return False

    async def ensure_browser_cors(self, origins: Optional[list] = None) -> bool:
        """Best-effort CORS for browser PUT/GET against the bucket."""
        if not self.client:
            return False
        try:
            # Prefer XML config via put_bucket_cors when available
            allow = origins or ["*"]
            from urllib.request import Request, urlopen

            # Use minio client's private API if present
            if hasattr(self.client, "set_bucket_cors"):
                from minio.datatypes import CORSConfig, CORSRule

                rule = CORSRule(
                    allowed_methods=["GET", "PUT", "HEAD", "POST"],
                    allowed_origins=list(allow),
                    allowed_headers=["*"],
                    expose_headers=["ETag", "Content-Length", "Content-Type"],
                    max_age_seconds=3600,
                )
                await asyncio.to_thread(
                    self.client.set_bucket_cors,
                    self.default_bucket,
                    CORSConfig([rule]),
                )
                return True
            log.warning("MinIO client has no set_bucket_cors; configure CORS externally")
            return False
        except Exception as e:
            log.warning(f"MinIO CORS setup skipped/failed: {e}")
            return False

    async def delete_file(
        self,
        object_name: str,
        bucket: Optional[str] = None,
    ) -> bool:
        """Delete file from MinIO."""
        if not self.client:
            return False

        bucket = bucket or self.default_bucket
        
        try:
            self.client.remove_object(bucket, object_name)
            log.info(f"🗑️ Deleted from MinIO: {bucket}/{object_name}")
            return True
        except S3Error as e:
            log.error(f"Delete failed: {e}")
            return False

    async def file_exists(
        self,
        object_name: str,
        bucket: Optional[str] = None,
    ) -> bool:
        """Check if file exists in MinIO."""
        if not self.client:
            return False

        bucket = bucket or self.default_bucket
        
        try:
            await asyncio.to_thread(self.client.stat_object, bucket, object_name)
            return True
        except S3Error:
            return False

    async def get_file_info(
        self,
        object_name: str,
        bucket: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get file metadata."""
        if not self.client:
            return None

        bucket = bucket or self.default_bucket
        
        try:
            stat = await asyncio.to_thread(self.client.stat_object, bucket, object_name)
            return {
                "size": stat.size,
                "content_type": stat.content_type,
                "last_modified": stat.last_modified.isoformat() if stat.last_modified else None,
                "etag": stat.etag,
                "metadata": dict(stat.metadata) if stat.metadata else {},
            }
        except S3Error as e:
            log.error(f"Stat failed: {e}")
            return None

    async def stat_storage_path(self, storage_path: str) -> Optional[Dict[str, Any]]:
        """
        Resolve a stored `storage_path` (e.g. `minio://<bucket>/<object>` or just
        `<bucket>/<object>` or a bare `<object>` in default bucket) and return
        live S3 stat info — size, etag, content_type, last_modified.

        Returns None if the path can't be parsed or stat fails.
        """
        if not self.client or not storage_path:
            return None

        path = storage_path.strip()
        if path.startswith("minio://"):
            path = path[len("minio://"):]

        if "/" in path:
            head, _, rest = path.partition("/")
            if rest:
                bucket = head
                object_name = rest
            else:
                bucket = self.default_bucket
                object_name = head
        else:
            bucket = self.default_bucket
            object_name = path

        return await self.get_file_info(object_name, bucket=bucket)

    async def list_objects(
        self,
        prefix: str = "",
        bucket: Optional[str] = None,
        recursive: bool = True,
    ) -> List[Dict[str, Any]]:
        """List objects in bucket with optional prefix."""
        if not self.client:
            return []

        bucket = bucket or self.default_bucket
        
        try:
            objects = self.client.list_objects(bucket, prefix=prefix, recursive=recursive)
            return [
                {
                    "name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                    "is_dir": obj.is_dir,
                }
                for obj in objects
            ]
        except S3Error as e:
            log.error(f"List objects failed: {e}")
            return []

    async def copy_object(
        self,
        source_object: str,
        dest_object: str,
        source_bucket: Optional[str] = None,
        dest_bucket: Optional[str] = None,
    ) -> bool:
        """Copy object within MinIO."""
        if not self.client:
            return False

        source_bucket = source_bucket or self.default_bucket
        dest_bucket = dest_bucket or self.default_bucket
        
        try:
            from minio.commonconfig import CopySource
            self.client.copy_object(
                dest_bucket,
                dest_object,
                CopySource(source_bucket, source_object),
            )
            log.info(f"📋 Copied: {source_bucket}/{source_object} -> {dest_bucket}/{dest_object}")
            return True
        except S3Error as e:
            log.error(f"Copy failed: {e}")
            return False

    async def get_bucket_size(self, bucket: Optional[str] = None) -> int:
        """Get total size of all objects in bucket."""
        if not self.client:
            return 0

        bucket = bucket or self.default_bucket
        total_size = 0
        
        try:
            objects = self.client.list_objects(bucket, recursive=True)
            for obj in objects:
                total_size += obj.size or 0
            return total_size
        except S3Error:
            return 0

    async def create_bucket(self, bucket_name: str) -> bool:
        """Create a new bucket."""
        return await self._ensure_bucket(bucket_name)

    async def delete_bucket(self, bucket_name: str, force: bool = False) -> bool:
        """Delete a bucket. Use force=True to delete non-empty bucket."""
        if not self.client:
            return False

        try:
            if force:
                # Delete all objects first
                objects = self.client.list_objects(bucket_name, recursive=True)
                for obj in objects:
                    self.client.remove_object(bucket_name, obj.object_name)
            
            self.client.remove_bucket(bucket_name)
            log.info(f"🗑️ Deleted bucket: {bucket_name}")
            return True
        except S3Error as e:
            log.error(f"Delete bucket failed: {e}")
            return False
