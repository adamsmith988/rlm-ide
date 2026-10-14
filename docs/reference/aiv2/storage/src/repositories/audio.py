"""
Audio Repository
ریپوزیتوری مدیریت فایل‌های صوتی و ترنسکریپت
"""

import logging
from typing import Any, Dict, List, Optional

from ..base import json_dumps, to_uuid

logger = logging.getLogger(__name__)


class AudioRepository:
    """
    Repository for audio-specific operations
    عملیات تخصصی صوت: متادیتا، ترنسکریپت، سگمنت‌ها، گوینده‌ها
    """
    
    def __init__(self, postgres_backend):
        self.pg = postgres_backend
    
    # --- artifact_audio ---
    
    async def upsert_audio_meta(
        self,
        artifact_id: str,
        duration_sec: Optional[float] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        codec: Optional[str] = None,
        bitrate: Optional[int] = None,
        transcription: Optional[str] = None,
        language: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Upsert audio metadata for an artifact"""
        aid = to_uuid(artifact_id)
        if not aid:
            return None
        
        sql = """
            INSERT INTO artifact_audio (
                artifact_id, duration_sec, sample_rate, channels,
                codec, bitrate, transcription, language, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
            ON CONFLICT (artifact_id) DO UPDATE SET
                duration_sec = COALESCE(EXCLUDED.duration_sec, artifact_audio.duration_sec),
                sample_rate = COALESCE(EXCLUDED.sample_rate, artifact_audio.sample_rate),
                channels = COALESCE(EXCLUDED.channels, artifact_audio.channels),
                codec = COALESCE(EXCLUDED.codec, artifact_audio.codec),
                bitrate = COALESCE(EXCLUDED.bitrate, artifact_audio.bitrate),
                transcription = COALESCE(EXCLUDED.transcription, artifact_audio.transcription),
                language = COALESCE(EXCLUDED.language, artifact_audio.language),
                metadata = COALESCE(EXCLUDED.metadata, artifact_audio.metadata),
                updated_at = now()
            RETURNING *
        """
        row = await self.pg.executor.fetchrow(
            sql, aid, duration_sec, sample_rate, channels,
            codec, bitrate, transcription, language, json_dumps(metadata or {})
        )
        return dict(row) if row else None
    
    async def get_audio_meta(self, artifact_id: str) -> Optional[Dict]:
        """Get audio metadata by artifact ID"""
        aid = to_uuid(artifact_id)
        if not aid:
            return None
        
        row = await self.pg.executor.fetchrow(
            "SELECT * FROM artifact_audio WHERE artifact_id = $1", aid
        )
        return dict(row) if row else None
    
    async def update_transcription(
        self,
        artifact_id: str,
        transcription: str,
        language: Optional[str] = None
    ) -> bool:
        """Update transcription for an audio artifact"""
        aid = to_uuid(artifact_id)
        if not aid:
            return False
        
        if language:
            result = await self.pg.executor.execute(
                """
                UPDATE artifact_audio 
                SET transcription = $2, language = $3, updated_at = now()
                WHERE artifact_id = $1
                """,
                aid, transcription, language
            )
        else:
            result = await self.pg.executor.execute(
                """
                UPDATE artifact_audio 
                SET transcription = $2, updated_at = now()
                WHERE artifact_id = $1
                """,
                aid, transcription
            )
        return "UPDATE 1" in result
    
    # --- artifact_audio_segments ---
    
    async def add_segment(
        self,
        artifact_id: str,
        speaker_id: Optional[str] = None,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        text: Optional[str] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Add an audio segment (e.g., from diarization or transcription)"""
        aid = to_uuid(artifact_id)
        if not aid:
            return None
        
        sql = """
            INSERT INTO artifact_audio_segments (
                artifact_id, speaker_id, start_sec, end_sec,
                text, confidence, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
            RETURNING *
        """
        row = await self.pg.executor.fetchrow(
            sql, aid, speaker_id, start_sec, end_sec,
            text, confidence, json_dumps(metadata or {})
        )
        return dict(row) if row else None
    
    async def batch_add_segments(
        self,
        artifact_id: str,
        segments: List[Dict]
    ) -> int:
        """Add multiple segments in batch"""
        aid = to_uuid(artifact_id)
        if not aid:
            return 0
        
        count = 0
        for seg in segments:
            result = await self.add_segment(
                artifact_id=artifact_id,
                speaker_id=seg.get("speaker_id") or seg.get("speaker"),
                start_sec=seg.get("start_sec") or seg.get("start"),
                end_sec=seg.get("end_sec") or seg.get("end"),
                text=seg.get("text"),
                confidence=seg.get("confidence"),
                metadata=seg.get("metadata")
            )
            if result:
                count += 1
        return count
    
    async def list_segments(
        self,
        artifact_id: str,
        speaker_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """List segments for an artifact, optionally filtered by speaker"""
        aid = to_uuid(artifact_id)
        if not aid:
            return []
        
        if speaker_id:
            rows = await self.pg.executor.fetch(
                """
                SELECT * FROM artifact_audio_segments
                WHERE artifact_id = $1 AND speaker_id = $2
                ORDER BY start_sec
                LIMIT $3
                """,
                aid, speaker_id, limit
            )
        else:
            rows = await self.pg.executor.fetch(
                """
                SELECT * FROM artifact_audio_segments
                WHERE artifact_id = $1
                ORDER BY start_sec
                LIMIT $2
                """,
                aid, limit
            )
        return [dict(r) for r in rows]
    
    async def get_speakers(self, artifact_id: str) -> List[str]:
        """Get unique speaker IDs for an artifact"""
        aid = to_uuid(artifact_id)
        if not aid:
            return []
        
        rows = await self.pg.executor.fetch(
            """
            SELECT DISTINCT speaker_id FROM artifact_audio_segments
            WHERE artifact_id = $1 AND speaker_id IS NOT NULL
            ORDER BY speaker_id
            """,
            aid
        )
        return [r["speaker_id"] for r in rows]
    
    async def delete_segments(self, artifact_id: str) -> int:
        """Delete all segments for an artifact"""
        aid = to_uuid(artifact_id)
        if not aid:
            return 0
        
        result = await self.pg.executor.execute(
            "DELETE FROM artifact_audio_segments WHERE artifact_id = $1",
            aid
        )
        # Extract count from "DELETE N"
        try:
            return int(result.split()[-1])
        except:
            return 0
    
    async def get_full_transcript(self, artifact_id: str) -> str:
        """Get full transcript by joining all segments"""
        segments = await self.list_segments(artifact_id, limit=10000)
        if not segments:
            # Fallback to stored transcription
            meta = await self.get_audio_meta(artifact_id)
            return meta.get("transcription", "") if meta else ""
        
        # Join segment texts in order
        return " ".join(
            seg.get("text", "") 
            for seg in segments 
            if seg.get("text")
        )
