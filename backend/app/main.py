from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CodeMind AI",
    description="AI-powered code analysis and enhancement platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
try:
    from app.api.endpoints import advanced
    app.include_router(advanced.router)
    logger.info("Advanced API endpoints loaded successfully")
except Exception as e:
    logger.error(f"Failed to load advanced endpoints: {e}")

# Health check endpoint
@app.get("/")
async def root():
    return {
        "name": "CodeMind AI",
        "version": "0.1.0",
        "status": "running",
        "endpoints": [
            "/api/v1/advanced/pr-review",
            "/api/v1/advanced/analyze",
            "/docs",
            "/redoc"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "api": "up",
            "swarm_orchestrator": "ready",
            "rag_pipeline": "ready",
            "privacy_layer": "active"
        }
    }

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )