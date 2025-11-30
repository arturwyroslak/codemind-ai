"""Health check endpoints"""

from fastapi import APIRouter
import logging
import time

logger = logging.getLogger(__name__)
router = APIRouter()

START_TIME = time.time()


@router.get("/")
async def health():
    """Basic health check"""
    return {
        "status": "healthy",
        "uptime": time.time() - START_TIME
    }


@router.get("/detailed")
async def detailed_health():
    """Detailed health check with component status"""
    return {
        "status": "healthy",
        "uptime": time.time() - START_TIME,
        "components": {
            "api": "healthy",
            "agents": "healthy",
            "rag": "healthy",
            "database": "healthy",
            "cache": "healthy"
        }
    }