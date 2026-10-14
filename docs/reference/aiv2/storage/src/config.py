"""
Storage Service Configuration
تنظیمات سرویس ذخیره‌سازی
"""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """تنظیمات سرویس Storage."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Debug
    DEBUG: bool = False
    
    # PostgreSQL
    DATABASE_URL: str = "postgresql://localai:localai_pass@localhost:5432/localai"
    PG_POOL_MIN: int = 2
    PG_POOL_MAX: int = 10
    
    # MongoDB
    MONGODB_URL: Optional[str] = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "localai"
    
    # Qdrant
    QDRANT_URL: Optional[str] = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    
    # Neo4j (optional)
    NEO4J_URI: Optional[str] = "bolt://localhost:7687"
    NEO4J_USERNAME: Optional[str] = "neo4j"
    NEO4J_PASSWORD: Optional[str] = "neo4j_pass"
    
    # MinIO (S3-compatible object storage)
    MINIO_ENABLED: bool = False
    MINIO_ENDPOINT: str = "localhost:9000"  # Internal endpoint (Docker network)
    MINIO_PUBLIC_ENDPOINT: str = "localhost:9000"  # Public endpoint for presigned URLs
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin123"
    MINIO_BUCKET: str = "uploads"
    MINIO_SECURE: bool = False
    # Comma-separated origins for browser PUT/GET (presigned). "*" for open CORS.
    MINIO_CORS_ORIGINS: str = "*"
    # Staging / upload-session multipart (0 disables multipart — single PUT only)
    UPLOAD_MULTIPART_THRESHOLD_BYTES: int = 32 * 1024 * 1024
    UPLOAD_MULTIPART_PART_SIZE_BYTES: int = 16 * 1024 * 1024
    
    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    
    # File Storage (fallback when MinIO disabled)
    UPLOAD_DIR: str = "/data/uploads"
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB

    # Internal RBAC context propagated from tool-runner/api-gateway
    INTERNAL_API_KEY: str = "aiv2-internal-service-key-change-in-production"
    STORAGE_RBAC_ENABLED: bool = True
    AUTH_SERVICE_URL: str = "http://auth-service:8003"


settings = Settings()


DEFAULT_INTERNAL_API_KEY = "aiv2-internal-service-key-change-in-production"


def assert_secure_secrets() -> None:
    """Fail startup if security-critical secrets are left at their known defaults.

    Override only for local development by setting ALLOW_INSECURE_DEFAULTS=true.
    """
    import os

    if os.getenv("ALLOW_INSECURE_DEFAULTS", "").lower() in ("1", "true", "yes"):
        return

    insecure = []
    if settings.INTERNAL_API_KEY == DEFAULT_INTERNAL_API_KEY:
        insecure.append("INTERNAL_API_KEY")
    if insecure:
        raise RuntimeError(
            "Refusing to start with default secret(s): "
            + ", ".join(insecure)
            + ". Set them via environment variables, or set "
            "ALLOW_INSECURE_DEFAULTS=true for local development."
        )
