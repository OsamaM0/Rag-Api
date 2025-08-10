"""
Asynchronous PostgreSQL Vector Store wrapper.

This module provides an async wrapper around the ExtendedPgVector class,
allowing for non-blocking vector operations in async environments.
"""

from typing import Optional, List, Tuple, Dict, Any
import asyncio
from langchain_core.documents import Document
from langchain_core.runnables.config import run_in_executor
from .extended_pg_vector import ExtendedPgVector


class AsyncPgVector(ExtendedPgVector):
    """
    Asynchronous wrapper for ExtendedPgVector.
    
    This class provides async methods for all vector operations,
    allowing for non-blocking execution in async environments.
    """
    
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the async PGVector store."""
        super().__init__(*args, **kwargs)
        self._thread_pool = None
    
    def _get_thread_pool(self):
        """Get the thread pool executor for async operations."""
        if self._thread_pool is None:
            try:
                # Try to get the thread pool from FastAPI app state
                import contextvars
                from fastapi import Request
                # This is a fallback - in practice, we'll pass the executor explicitly
                loop = asyncio.get_running_loop()
                self._thread_pool = getattr(loop, '_default_executor', None)
            except Exception:
                pass
        return self._thread_pool
    
    async def get_all_ids(self, executor=None) -> List[str]:
        """Asynchronously retrieve all custom IDs from the vector store."""
        executor = executor or self._get_thread_pool()
        return await run_in_executor(executor, super().get_all_ids)
    
    async def get_filtered_ids(self, ids: List[str], executor=None) -> List[str]:
        """Asynchronously filter the provided IDs to only include those that exist."""
        executor = executor or self._get_thread_pool()
        return await run_in_executor(executor, super().get_filtered_ids, ids)

    async def get_documents_by_ids(self, ids: List[str], executor=None) -> List[Document]:
        """Asynchronously retrieve documents by their custom IDs."""
        executor = executor or self._get_thread_pool()
        return await run_in_executor(executor, super().get_documents_by_ids, ids)

    async def delete(
        self, ids: Optional[List[str]] = None, collection_only: bool = False, executor=None
    ) -> None:
        """Asynchronously delete multiple embeddings by their custom IDs."""
        executor = executor or self._get_thread_pool()
        await run_in_executor(executor, self._delete_multiple, ids, collection_only)
    
    async def asimilarity_search_with_score_by_vector(
        self, 
        embedding: List[float], 
        k: int = 4, 
        filter: Optional[Dict[str, Any]] = None,
        executor=None
    ) -> List[Tuple[Document, float]]:
        """Asynchronous similarity search with score by vector."""
        executor = executor or self._get_thread_pool()
        return await run_in_executor(
            executor, 
            super().similarity_search_with_score_by_vector, 
            embedding, 
            k, 
            filter
        )
    
    async def aadd_documents(
        self, 
        documents: List[Document], 
        ids: Optional[List[str]] = None,
        executor=None,
        **kwargs
    ) -> List[str]:
        """Asynchronously add documents to the vector store."""
        executor = executor or self._get_thread_pool()
        return await run_in_executor(
            executor, 
            super().add_documents, 
            documents, 
            ids=ids,
            **kwargs
        )