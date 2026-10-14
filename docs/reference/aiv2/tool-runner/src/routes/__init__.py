from .tools import router as tools_router
from .health import router as health_router
from .internal import router as internal_router
from .plugins import router as plugins_router
from .jobs import router as jobs_router

__all__ = [
    "tools_router",
    "health_router",
    "internal_router",
    "plugins_router",
    "jobs_router",
]
