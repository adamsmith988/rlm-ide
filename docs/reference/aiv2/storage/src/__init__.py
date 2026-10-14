"""
Storage Service
سرویس ذخیره‌سازی AIv2

Multi-backend storage orchestrator:
- PostgreSQL: Structured data (artifacts, persons, folders)
- MongoDB: Unstructured data (tool outputs, observations, insights)
- Qdrant: Vector embeddings (faces, voices, text)
"""

from .storage import Storage, get_storage, close_storage
from .config import settings

__all__ = [
    "Storage",
    "get_storage", 
    "close_storage",
    "settings"
]
