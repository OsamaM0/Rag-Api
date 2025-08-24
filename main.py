# main.py
import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from starlette.responses import JSONResponse

from app.config import (
    VectorDBType,
    debug_mode,
    RAG_HOST,
    RAG_PORT,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PDF_EXTRACT_IMAGES,
    VECTOR_DB_TYPE,
    LogMiddleware,
    logger,
)
from app.middleware import security_middleware
from app.routes import (
    document_routes, 
    document_block_routes,
    document_image_routes,
    pgvector_routes, 
    collection_routes,
    query_routes,
    embedding_routes,
    database_routes,
    health_routes
)
from app.services.database import PSQLDatabase, ensure_vector_indexes, ensure_document_blocks_schema, ensure_document_images_schema


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic goes here
    # Create bounded thread pool executor based on CPU cores
    max_workers = min(
        int(os.getenv("RAG_THREAD_POOL_SIZE", str(os.cpu_count()))), 8
    )  # Cap at 8
    app.state.thread_pool = ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="rag-worker"
    )
    logger.info(
        f"Initialized thread pool with {max_workers} workers (CPU cores: {os.cpu_count()})"
    )

    if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
        await PSQLDatabase.get_pool()  # Initialize the pool
        await ensure_vector_indexes()
        await ensure_document_blocks_schema()  # Initialize document blocks schema
        await ensure_document_images_schema()  # Initialize document images schema

    yield

    # Cleanup logic
    logger.info("Shutting down thread pool")
    app.state.thread_pool.shutdown(wait=True)
    logger.info("Thread pool shutdown complete")


app = FastAPI(
    title="RAG API Server",
    description="A comprehensive RAG (Retrieval-Augmented Generation) API with document management, embeddings, and vector search capabilities",
    version="1.0.0",
    lifespan=lifespan, 
    debug=debug_mode,
    contact={
        "name": "RAG API Support",
        "email": "support@ragapi.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LogMiddleware)

app.middleware("http")(security_middleware)

# Set state variables for use in routes
app.state.CHUNK_SIZE = CHUNK_SIZE
app.state.CHUNK_OVERLAP = CHUNK_OVERLAP
app.state.PDF_EXTRACT_IMAGES = PDF_EXTRACT_IMAGES

# Include routers with organized groups
# System Health
app.include_router(health_routes.router)

# Collections - CRUD operations for document collections
app.include_router(collection_routes.router)

# Documents - File upload, processing, and document management
app.include_router(document_routes.router)

# Document Blocks - Document block management and JSON parsing
app.include_router(document_block_routes.router)

# Document Images - Document image management (read-only for viewing uploaded images)
if debug_mode:
    app.include_router(document_image_routes.router)

# Embeddings - Vector embedding operations
app.include_router(embedding_routes.router)

# Queries - Search and retrieval operations
app.include_router(query_routes.router)

# Database - Database management and maintenance
app.include_router(database_routes.router)

# PgVector - Advanced PostgreSQL vector operations (debug mode only)
if debug_mode:
    app.include_router(router=pgvector_routes.router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.debug(f"Validation error occurred")
    logger.debug(f"Raw request body: {body.decode()}")
    logger.debug(f"Validation errors: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": body.decode(),
            "message": "Request validation failed",
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host=RAG_HOST, port=RAG_PORT, log_config=None)
