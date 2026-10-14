"""
Text Repository
ریپوزیتوری مدیریت متون، OCR و جستجو
"""

import logging
from typing import Any, Dict, List, Optional

from ..base import to_uuid

logger = logging.getLogger(__name__)


class TextRepository:
    """
    Repository for text-specific operations
    عملیات تخصصی متن: OCR، جستجوی متنی، ایندکس‌گذاری
    """
    
    def __init__(self, postgres_backend, qdrant_backend=None):
        self.pg = postgres_backend
        self.qdrant = qdrant_backend
    
    async def save_ocr_result(
        self,
        artifact_id: str,
        text: str,
        blocks: Optional[List[Dict]] = None,
        language: Optional[str] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Save OCR result to artifact metadata"""
        aid = to_uuid(artifact_id)
        if not aid:
            return False
        
        ocr_data = {
            "text": text,
            "blocks": blocks or [],
            "language": language,
            "confidence": confidence,
            **(metadata or {})
        }
        
        # Merge into artifact metadata
        result = await self.pg.artifact_merge_metadata(
            artifact_id,
            {"ocr": ocr_data}
        )
        return result is not None
    
    async def get_ocr_result(self, artifact_id: str) -> Optional[Dict]:
        """Get OCR result from artifact metadata"""
        aid = to_uuid(artifact_id)
        if not aid:
            return None
        
        artifact = await self.pg.artifact_get(artifact_id)
        if not artifact:
            return None
        
        metadata = artifact.get("metadata", {})
        return metadata.get("ocr")
    
    async def save_description(
        self,
        artifact_id: str,
        description: str,
        model: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Save LLM-generated description to artifact metadata"""
        aid = to_uuid(artifact_id)
        if not aid:
            return False
        
        desc_data = {
            "text": description,
            "model": model,
            **(metadata or {})
        }
        
        result = await self.pg.artifact_merge_metadata(
            artifact_id,
            {"description": desc_data}
        )
        return result is not None
    
    async def get_description(self, artifact_id: str) -> Optional[str]:
        """Get description from artifact metadata"""
        aid = to_uuid(artifact_id)
        if not aid:
            return None
        
        artifact = await self.pg.artifact_get(artifact_id)
        if not artifact:
            return None
        
        metadata = artifact.get("metadata", {})
        desc = metadata.get("description")
        if isinstance(desc, dict):
            return desc.get("text")
        return desc
    
    async def search_text_in_artifacts(
        self,
        query: str,
        media_types: Optional[List[str]] = None,
        folder_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Search for text within artifact metadata (OCR, transcripts, descriptions)
        جستجوی متنی در متادیتای آرتیفکت‌ها
        """
        conditions = ["metadata::text ILIKE $1"]
        params = [f"%{query}%"]
        idx = 2
        
        if media_types:
            placeholders = ", ".join(f"${i}" for i in range(idx, idx + len(media_types)))
            conditions.append(f"media_type::text IN ({placeholders})")
            params.extend(media_types)
            idx += len(media_types)
        
        if folder_id:
            fid = to_uuid(folder_id)
            if fid:
                conditions.append(f"folder_id = ${idx}")
                params.append(fid)
                idx += 1
        
        params.append(limit)
        where = " AND ".join(conditions)
        
        sql = f"""
            SELECT id, media_type, storage_path, original_filename, 
                   metadata, created_at
            FROM artifacts
            WHERE {where}
            ORDER BY created_at DESC
            LIMIT ${idx}
        """
        
        rows = await self.pg.executor.fetch(sql, *params)
        return [dict(r) for r in rows]
    
    async def index_text_embedding(
        self,
        artifact_id: str,
        text: str,
        embedding: List[float],
        collection_name: str = "text_embeddings"
    ) -> bool:
        """Store text embedding in Qdrant for semantic search"""
        if not self.qdrant:
            logger.warning("Qdrant backend not configured")
            return False
        
        aid = to_uuid(artifact_id)
        if not aid:
            return False
        
        # Ensure collection exists
        await self.qdrant.ensure_collection(
            collection_name,
            dimension=len(embedding)
        )
        
        # Upsert vector
        await self.qdrant.upsert_vectors(
            collection_name,
            vectors=[embedding],
            ids=[str(aid)],
            payloads=[{
                "artifact_id": str(aid),
                "text_preview": text[:500] if text else ""
            }]
        )
        return True
    
    async def search_similar_texts(
        self,
        embedding: List[float],
        collection_name: str = "text_embeddings",
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """Search for semantically similar texts using Qdrant"""
        if not self.qdrant:
            logger.warning("Qdrant backend not configured")
            return []
        
        results = await self.qdrant.search(
            collection_name,
            query_vector=embedding,
            limit=limit,
            score_threshold=score_threshold
        )
        
        # Enrich with artifact data
        enriched = []
        for r in results:
            artifact_id = r.get("payload", {}).get("artifact_id")
            if artifact_id:
                artifact = await self.pg.artifact_get(artifact_id)
                if artifact:
                    enriched.append({
                        **r,
                        "artifact": artifact
                    })
                else:
                    enriched.append(r)
            else:
                enriched.append(r)
        
        return enriched
    
    async def get_extractable_text(self, artifact_id: str) -> str:
        """
        Get all extractable text from an artifact
        (OCR + description + transcription)
        """
        aid = to_uuid(artifact_id)
        if not aid:
            return ""
        
        artifact = await self.pg.artifact_get(artifact_id)
        if not artifact:
            return ""
        
        texts = []
        metadata = artifact.get("metadata", {})
        
        # OCR text
        ocr = metadata.get("ocr", {})
        if isinstance(ocr, dict) and ocr.get("text"):
            texts.append(ocr["text"])
        
        # Description
        desc = metadata.get("description")
        if isinstance(desc, dict) and desc.get("text"):
            texts.append(desc["text"])
        elif isinstance(desc, str):
            texts.append(desc)
        
        # Transcription (for audio)
        trans = metadata.get("transcription")
        if trans:
            texts.append(trans)
        
        return "\n\n".join(texts)
