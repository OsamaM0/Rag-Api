# app/routes/health_routes.py
import traceback
from fastapi import APIRouter, Request
from app.config import logger
from app.utils.health import is_health_ok

router = APIRouter(tags=["System Health"])


@router.get("/health")
async def health_check(request: Request):
    """System health check endpoint."""
    try:
        if await is_health_ok():
            return {"status": "UP", "message": "System is healthy"}
        else:
            logger.error("Health check failed")
            return {"status": "DOWN", "message": "System health check failed"}, 503
    except Exception as e:
        logger.error(
            "Error during health check | Error: %s | Traceback: %s",
            str(e),
            traceback.format_exc(),
        )
        return {"status": "DOWN", "error": str(e), "message": "Health check error"}, 503


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG API Server",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc",
            "collections": "/collections",
            "documents": "/documents",
            "document_blocks": "/document-blocks",
            "document_content": "/document-content", 
            "embeddings": "/embeddings",
            "queries": "/queries",
            "database": "/database"
        }
    }
