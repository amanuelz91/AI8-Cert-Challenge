"""
Main FastAPI application.

Orchestrates all API components and middleware.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.api.endpoints import health_router, query_router, system_router, streaming_router
from src.api.services import rag_service
from src.core.system import create_production_rag_system
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Production RAG API",
    description="Production-grade RAG system with multiple retrieval strategies",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(query_router)
app.include_router(system_router)
app.include_router(streaming_router)


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    try:
        logger.info("🚀 Starting RAG API server")
        
        # Initialize RAG system
        rag_system = create_production_rag_system()
        
        # Set in service
        rag_service.set_rag_system(rag_system)
        
        logger.info("✅ RAG system initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {str(e)}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Production RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "stats": "/system/stats"
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"❌ Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
