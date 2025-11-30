"""API v1 Router"""

from fastapi import APIRouter
from app.api.v1.endpoints import analyze, generate, query, health

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(analyze.router, prefix="/analyze", tags=["analysis"])
api_router.include_router(generate.router, prefix="/generate", tags=["generation"])
api_router.include_router(query.router, prefix="/query", tags=["rag"])
api_router.include_router(health.router, prefix="/health", tags=["health"])