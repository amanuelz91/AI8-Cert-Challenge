"""
API endpoints module initialization.

Exports all endpoint routers.
"""

from .health import router as health_router
from .query import router as query_router
from .system import router as system_router
from .streaming import router as streaming_router

__all__ = [
    "health_router",
    "query_router", 
    "system_router",
    "streaming_router"
]
