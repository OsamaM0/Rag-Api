# app/routes/query_routes.py
import traceback
from typing import List
from functools import lru_cache
from fastapi import APIRouter, HTTPException, Request
from app.config import logger, vector_store
from app.models import (
    QueryRequestBody,
    QueryMultipleBody,
    SimilaritySearchRequest,
    SimilaritySearchResponse
)
from app.services.vector_store.async_pg_vector import AsyncPgVector

router = APIRouter(prefix="/queries", tags=["Queries"])


# Cache the embedding function with LRU cache
@lru_cache(maxsize=128)
def get_cached_query_embedding(query: str):
    return vector_store.embedding_function.embed_query(query)


@router.post("/", response_model=List[SimilaritySearchResponse])
async def query_embeddings_by_file_id(
    body: QueryRequestBody,
    request: Request,
):
    """Query embeddings by file ID with authorization."""
    if not hasattr(request.state, "user"):
        user_authorized = body.entity_id if body.entity_id else "public"
    else:
        user_authorized = (
            body.entity_id if body.entity_id else request.state.user.get("id")
        )

    authorized_documents = []

    try:
        embedding = get_cached_query_embedding(body.query)

        if isinstance(vector_store, AsyncPgVector):
            documents = await vector_store.asimilarity_search_with_score_by_vector(
                embedding,
                k=body.k,
                filter={"file_id": body.file_id},
                executor=request.app.state.thread_pool,
            )
        else:
            documents = vector_store.similarity_search_with_score_by_vector(
                embedding, k=body.k, filter={"file_id": body.file_id}
            )

        if not documents:
            return authorized_documents

        document, score = documents[0]
        doc_metadata = document.metadata
        doc_user_id = doc_metadata.get("user_id")

        if doc_user_id is None or doc_user_id == user_authorized:
            authorized_documents = documents
        else:
            # If using entity_id and access denied, try again with user's actual ID
            if body.entity_id and hasattr(request.state, "user"):
                user_authorized = request.state.user.get("id")
                if doc_user_id == user_authorized:
                    authorized_documents = documents
                else:
                    if body.entity_id == doc_user_id:
                        logger.warning(
                            f"Entity ID {body.entity_id} matches document user_id but user {user_authorized} is not authorized"
                        )
                    else:
                        logger.warning(
                            f"Access denied for both entity ID {body.entity_id} and user {user_authorized} to document with user_id {doc_user_id}"
                        )
            else:
                logger.warning(
                    f"Unauthorized access attempt by user {user_authorized} to a document with user_id {doc_user_id}"
                )

        return authorized_documents

    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in query_embeddings_by_file_id | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in query embeddings | File ID: %s | Query: %s | Error: %s | Traceback: %s",
            body.file_id,
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multiple", response_model=List[SimilaritySearchResponse])
async def query_embeddings_by_file_ids(request: Request, body: QueryMultipleBody):
    """Query embeddings across multiple file IDs."""
    try:
        # Get the embedding of the query text
        embedding = get_cached_query_embedding(body.query)

        # Perform similarity search with the query embedding and filter by the file_ids in metadata
        if isinstance(vector_store, AsyncPgVector):
            documents = await vector_store.asimilarity_search_with_score_by_vector(
                embedding,
                k=body.k,
                filter={"file_id": {"$in": body.file_ids}},
                executor=request.app.state.thread_pool,
            )
        else:
            documents = vector_store.similarity_search_with_score_by_vector(
                embedding, k=body.k, filter={"file_id": {"$in": body.file_ids}}
            )

        # Ensure documents list is not empty
        if not documents:
            raise HTTPException(
                status_code=404, detail="No documents found for the given query"
            )

        return documents
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in query_embeddings_by_file_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in query multiple embeddings | File IDs: %s | Query: %s | Error: %s | Traceback: %s",
            body.file_ids,
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/similarity-search", response_model=List[SimilaritySearchResponse])
async def similarity_search(
    search_request: SimilaritySearchRequest,
    request: Request
):
    """Perform similarity search with advanced filtering."""
    try:
        embedding = get_cached_query_embedding(search_request.query)

        if isinstance(vector_store, AsyncPgVector):
            documents = await vector_store.asimilarity_search_with_score_by_vector(
                embedding,
                k=search_request.k,
                filter=search_request.filter,
                executor=request.app.state.thread_pool,
            )
        else:
            documents = vector_store.similarity_search_with_score_by_vector(
                embedding, 
                k=search_request.k, 
                filter=search_request.filter
            )

        # Apply score threshold filtering if specified
        if search_request.score_threshold is not None:
            documents = [
                (doc, score) for doc, score in documents 
                if score >= search_request.score_threshold
            ]

        if not documents:
            return []

        # Convert to response format
        results = []
        for doc, score in documents:
            results.append(SimilaritySearchResponse(
                document={
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                },
                score=score
            ))

        return results

    except Exception as e:
        logger.error(
            f"Error in similarity search | Query: {search_request.query} | Error: {str(e)} | Traceback: {traceback.format_exc()}"
        )
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")


@router.post("/semantic-search", response_model=List[SimilaritySearchResponse])
async def semantic_search(
    query: str,
    k: int = 4,
    collection_name: str = None,
    user_id: str = None,
    request: Request = None
):
    """Perform semantic search across documents."""
    try:
        embedding = get_cached_query_embedding(query)
        
        # Build filter based on parameters
        filter_dict = {}
        if collection_name:
            filter_dict["collection"] = collection_name
        if user_id:
            filter_dict["user_id"] = user_id

        if isinstance(vector_store, AsyncPgVector):
            documents = await vector_store.asimilarity_search_with_score_by_vector(
                embedding,
                k=k,
                filter=filter_dict if filter_dict else None,
                executor=request.app.state.thread_pool,
            )
        else:
            documents = vector_store.similarity_search_with_score_by_vector(
                embedding, 
                k=k, 
                filter=filter_dict if filter_dict else None
            )

        if not documents:
            return []

        # Convert to response format
        results = []
        for doc, score in documents:
            results.append(SimilaritySearchResponse(
                document={
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                },
                score=score
            ))

        return results

    except Exception as e:
        logger.error(f"Error in semantic search | Query: {query} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


@router.get("/history")
async def get_query_history(
    request: Request,
    user_id: str = None,
    limit: int = 50
):
    """Get query history for a user (mock implementation)."""
    try:
        # In a real implementation, you would fetch this from a database
        mock_history = [
            {
                "id": "query_1",
                "query": "What is machine learning?",
                "timestamp": "2024-08-09T10:00:00Z",
                "results_count": 5,
                "user_id": user_id or "public"
            },
            {
                "id": "query_2", 
                "query": "Deep learning algorithms",
                "timestamp": "2024-08-09T09:30:00Z",
                "results_count": 3,
                "user_id": user_id or "public"
            }
        ]
        
        return mock_history[:limit]
        
    except Exception as e:
        logger.error(f"Error getting query history | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get query history: {str(e)}")


@router.delete("/history/{query_id}")
async def delete_query_from_history(query_id: str, request: Request):
    """Delete a query from history (mock implementation)."""
    try:
        # In a real implementation, you would delete from database
        logger.info(f"Query deleted from history: {query_id}")
        return {"message": f"Query {query_id} deleted from history"}
        
    except Exception as e:
        logger.error(f"Error deleting query from history | Query ID: {query_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to delete query from history: {str(e)}")
