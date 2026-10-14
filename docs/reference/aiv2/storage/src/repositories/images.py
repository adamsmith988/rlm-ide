"""
Image Repository
ریپوزیتوری مدیریت تصاویر و چهره‌ها
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..base import to_uuid

logger = logging.getLogger(__name__)


class ImageRepository:
    """
    Repository for image-specific operations
    عملیات تخصصی تصاویر: اطلاعات EXIF، چهره‌ها، توضیحات
    """
    
    def __init__(self, postgres_backend):
        self.pg = postgres_backend
    
    # --- artifact_images ---
    
    async def upsert_image_meta(
        self,
        artifact_id: str,
        width_px: Optional[int] = None,
        height_px: Optional[int] = None,
        dpi: Optional[int] = None,
        color_mode: Optional[str] = None,
        camera_model: Optional[str] = None,
        lens: Optional[str] = None,
        iso: Optional[int] = None,
        exposure: Optional[str] = None,
        focal_length: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Upsert image metadata for an artifact"""
        aid = to_uuid(artifact_id)
        if not aid:
            return None
        
        sql = """
            INSERT INTO artifact_images (
                artifact_id, width_px, height_px, dpi, color_mode,
                camera_model, lens, iso, exposure, focal_length, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (artifact_id) DO UPDATE SET
                width_px = COALESCE(EXCLUDED.width_px, artifact_images.width_px),
                height_px = COALESCE(EXCLUDED.height_px, artifact_images.height_px),
                dpi = COALESCE(EXCLUDED.dpi, artifact_images.dpi),
                color_mode = COALESCE(EXCLUDED.color_mode, artifact_images.color_mode),
                camera_model = COALESCE(EXCLUDED.camera_model, artifact_images.camera_model),
                lens = COALESCE(EXCLUDED.lens, artifact_images.lens),
                iso = COALESCE(EXCLUDED.iso, artifact_images.iso),
                exposure = COALESCE(EXCLUDED.exposure, artifact_images.exposure),
                focal_length = COALESCE(EXCLUDED.focal_length, artifact_images.focal_length),
                metadata = COALESCE(EXCLUDED.metadata, artifact_images.metadata),
                updated_at = now()
            RETURNING *
        """
        row = await self.pg.executor.fetchrow(
            sql, aid, width_px, height_px, dpi, color_mode,
            camera_model, lens, iso, exposure, focal_length, 
            metadata or {}
        )
        return dict(row) if row else None
    
    async def get_image_meta(self, artifact_id: str) -> Optional[Dict]:
        """Get image metadata by artifact ID"""
        aid = to_uuid(artifact_id)
        if not aid:
            return None
        
        row = await self.pg.executor.fetchrow(
            "SELECT * FROM artifact_images WHERE artifact_id = $1", aid
        )
        return dict(row) if row else None
    
    # --- artifact_faces ---
    
    async def upsert_face(
        self,
        artifact_id: str,
        face_key: str,
        bbox: Optional[Dict] = None,
        confidence: Optional[float] = None,
        person_id: Optional[str] = None,
        tag: Optional[str] = None,
        thumbnail_base64: Optional[str] = None,
        vector_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Upsert a face detection result"""
        aid = to_uuid(artifact_id)
        pid = to_uuid(person_id) if person_id else None
        
        if not aid:
            return None
        
        sql = """
            INSERT INTO artifact_faces (
                artifact_id, face_key, bbox, confidence, person_id,
                tag, thumbnail_base64, vector_id, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (artifact_id, face_key) DO UPDATE SET
                bbox = COALESCE(EXCLUDED.bbox, artifact_faces.bbox),
                confidence = COALESCE(EXCLUDED.confidence, artifact_faces.confidence),
                person_id = COALESCE(EXCLUDED.person_id, artifact_faces.person_id),
                tag = COALESCE(EXCLUDED.tag, artifact_faces.tag),
                thumbnail_base64 = COALESCE(EXCLUDED.thumbnail_base64, artifact_faces.thumbnail_base64),
                vector_id = COALESCE(EXCLUDED.vector_id, artifact_faces.vector_id),
                metadata = COALESCE(EXCLUDED.metadata, artifact_faces.metadata)
            RETURNING *
        """
        row = await self.pg.executor.fetchrow(
            sql, aid, face_key, bbox, confidence, pid,
            tag, thumbnail_base64, vector_id, metadata or {}
        )
        return dict(row) if row else None
    
    async def list_faces(
        self,
        artifact_id: Optional[str] = None,
        person_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """List faces, optionally filtered by artifact or person"""
        conditions = []
        params = []
        idx = 1
        
        if artifact_id:
            aid = to_uuid(artifact_id)
            if aid:
                conditions.append(f"artifact_id = ${idx}")
                params.append(aid)
                idx += 1
        
        if person_id:
            pid = to_uuid(person_id)
            if pid:
                conditions.append(f"person_id = ${idx}")
                params.append(pid)
                idx += 1
        
        where = " AND ".join(conditions) if conditions else "TRUE"
        params.append(limit)
        
        sql = f"""
            SELECT * FROM artifact_faces 
            WHERE {where} 
            ORDER BY created_at DESC 
            LIMIT ${idx}
        """
        rows = await self.pg.executor.fetch(sql, *params)
        return [dict(r) for r in rows]
    
    async def get_face(self, artifact_id: str, face_key: str) -> Optional[Dict]:
        """Get a specific face by artifact_id and face_key"""
        aid = to_uuid(artifact_id)
        if not aid:
            return None
        
        row = await self.pg.executor.fetchrow(
            "SELECT * FROM artifact_faces WHERE artifact_id = $1 AND face_key = $2",
            aid, face_key
        )
        return dict(row) if row else None
    
    async def link_face_to_person(
        self,
        artifact_id: str,
        face_key: str,
        person_id: str
    ) -> bool:
        """Link a face to a person"""
        aid = to_uuid(artifact_id)
        pid = to_uuid(person_id)
        
        if not aid or not pid:
            return False
        
        result = await self.pg.executor.execute(
            """
            UPDATE artifact_faces 
            SET person_id = $3 
            WHERE artifact_id = $1 AND face_key = $2
            """,
            aid, face_key, pid
        )
        return "UPDATE 1" in result
    
    async def get_faces_by_person(self, person_id: str, limit: int = 100) -> List[Dict]:
        """Get all faces linked to a person"""
        pid = to_uuid(person_id)
        if not pid:
            return []
        
        rows = await self.pg.executor.fetch(
            """
            SELECT af.*, a.storage_path, a.original_filename
            FROM artifact_faces af
            JOIN artifacts a ON a.id = af.artifact_id
            WHERE af.person_id = $1
            ORDER BY af.created_at DESC
            LIMIT $2
            """,
            pid, limit
        )
        return [dict(r) for r in rows]
    
    async def delete_face(self, artifact_id: str, face_key: str) -> bool:
        """Delete a face record"""
        aid = to_uuid(artifact_id)
        if not aid:
            return False
        
        result = await self.pg.executor.execute(
            "DELETE FROM artifact_faces WHERE artifact_id = $1 AND face_key = $2",
            aid, face_key
        )
        return "DELETE 1" in result
