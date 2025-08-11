# app/routes/__init__.py
"""
Route modules for the RAG API.

This package contains all the API route definitions organized by functional groups:
- Collections: Document collection management
- Documents: Document upload, processing, and CRUD operations  
- Document Blocks: Document block management and JSON parsing
- Embeddings: Vector embedding operations (supports both PG Vector and Atlas Mongo)
- Queries: Search and retrieval operations (supports both PG Vector and Atlas Mongo)
- Database: Database management and maintenance
- Health: System health checks
- PgVector: Advanced PostgreSQL vector operations (debug mode)
"""

from . import (
    collection_routes,
    document_routes,
    document_block_routes,
    embedding_routes,
    query_routes,
    database_routes,
    health_routes,
    pgvector_routes
)

__all__ = [
    "collection_routes",
    "document_routes",
    "document_block_routes",
    "embedding_routes",
    "query_routes",
    "database_routes",
    "health_routes",
    "pgvector_routes"
]