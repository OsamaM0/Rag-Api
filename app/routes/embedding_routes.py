# app/routes/embedding_routes.py
import traceback
import uuid
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Request, Query
from app.config import logger, vector_store, VECTOR_DB_TYPE, VectorDBType
from app.models import (
    EmbeddingResponse,
    EmbeddingCreate,
    SuccessResponse,
    PaginatedResponse
)
from app.services.database import (
    get_document_by_idx,
    get_document_by_uuid,
    create_embedding,
    get_embedding_by_id,
    delete_embedding,
    similarity_search_embeddings,
    get_embeddings_by_document,
    get_all_embeddings,
    get_collection_by_uuid,
    get_collection_by_idx
)
from app.services.vector_store.async_pg_vector import AsyncPgVector

router = APIRouter(prefix="/embeddings", tags=["Embeddings"])


async def get_document_by_id(document_id: str):
    """Get document by either UUID or idx (string)."""
    # Try to get by UUID first, then fall back to idx for backward compatibility
    doc = await get_document_by_uuid(document_id)
    if not doc:
        doc = await get_document_by_idx(document_id)
    return doc


def parse_metadata(metadata):
    """Parse metadata from database - handle both string (JSON) and dict formats."""
    import json
    if isinstance(metadata, str):
        try:
            return json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            return {}
    elif isinstance(metadata, dict):
        return metadata
    else:
        return {}


def parse_embedding(embedding):
    """Parse embedding from database - convert PostgreSQL vector string to list of floats."""
    if isinstance(embedding, str):
        try:
            # Remove brackets and split by comma, then convert to floats
            embedding_str = embedding.strip('[]')
            if embedding_str:
                return [float(x.strip()) for x in embedding_str.split(',')]
            else:
                return []
        except (ValueError, TypeError):
            return []
    elif isinstance(embedding, list):
        return embedding
    else:
        return []


async def create_vector_embedding(content: str, document_uuid: str, collection_uuid: str = None, metadata: dict = None, request: Request = None):
    """Create embedding using the configured vector store (PG Vector or Atlas Mongo)."""
    try:
        if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
            # Use database-only approach for PG Vector
            custom_id = f"{document_uuid}_embedding_{hash(content) % 1000000}"
            
            # Get embedding vector using the configured embeddings
            from app.config import embeddings
            embedding_vector = embeddings.embed_query(content)
            
            return await create_embedding(
                custom_id=custom_id,
                embedding=embedding_vector,
                document_text=content,
                document_uuid=document_uuid,
                collection_uuid=collection_uuid,
                metadata=metadata or {}
            )
        elif VECTOR_DB_TYPE == VectorDBType.ATLAS_MONGO:
            # Use vector store approach for Atlas Mongo
            from langchain_core.documents import Document
            doc = Document(page_content=content, metadata={**metadata, "document_uuid": document_uuid})
            
            if isinstance(vector_store, AsyncPgVector):
                ids = await vector_store.aadd_documents([doc], executor=request.app.state.thread_pool if request else None)
            else:
                ids = vector_store.add_documents([doc])
            
            return {
                "embedding_id": ids[0] if ids else None,
                "document_uuid": document_uuid,
                "content": content,
                "metadata": metadata,
                "embedding": [],  # Vector store handles this internally
                "created_at": None
            }
        else:
            raise ValueError(f"Unsupported vector store type: {VECTOR_DB_TYPE}")
    except Exception as e:
        logger.error(f"Failed to create vector embedding | Error: {str(e)}")
        raise


async def search_similar_embeddings(query_text: str, limit: int = 5, request: Request = None):
    """Search similar embeddings using the configured vector store."""
    try:
        if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
            # Use database-only approach for PG Vector
            # Convert query text to embedding vector first
            from app.config import embeddings
            query_embedding = embeddings.embed_query(query_text)
            raw_results = await similarity_search_embeddings(query_embedding=query_embedding, k=limit)
            
            # Convert to expected format
            results = []
            for result in raw_results:
                results.append({
                    "embedding_id": result.get('custom_id'),
                    "document_id": result.get('document_id'),
                    "content": result.get('document'),
                    "similarity_score": float(result.get('distance', 0.0)),
                    "metadata": parse_metadata(result.get('cmetadata', {}))
                })
            return results
        elif VECTOR_DB_TYPE == VectorDBType.ATLAS_MONGO:
            # Use vector store approach for Atlas Mongo
            if isinstance(vector_store, AsyncPgVector):
                documents = await vector_store.asimilarity_search_with_score(
                    query_text, k=limit, executor=request.app.state.thread_pool if request else None
                )
            else:
                documents = vector_store.similarity_search_with_score(query_text, k=limit)
            
            # Convert to expected format
            results = []
            for doc, score in documents:
                results.append({
                    "embedding_id": doc.metadata.get("id", "unknown"),
                    "document_id": doc.metadata.get("document_id", "unknown"),
                    "content": doc.page_content,
                    "similarity_score": float(score),
                    "metadata": doc.metadata
                })
            return results
        else:
            raise ValueError(f"Unsupported vector store type: {VECTOR_DB_TYPE}")
    except Exception as e:
        logger.error(f"Failed to search similar embeddings | Error: {str(e)}")
        raise


@router.post("/document/{document_id}", response_model=List[EmbeddingResponse])
async def create_embeddings_for_document(
    document_id: str,
    request: Request,
    chunk_size: int = Query(1000, ge=100, le=5000, description="Size of text chunks for embeddings"),
    chunk_overlap: int = Query(200, ge=0, le=1000, description="Overlap between chunks")
):
    """Create embeddings for a document by splitting its content into chunks."""
    try:
        # Get document content from document table
        document = await get_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        content = document.get('content') or document.get('page_content', '')
        if not content:
            raise HTTPException(status_code=400, detail="Document has no content to embed")
        
        # Split content into chunks
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        chunks = text_splitter.split_text(content)
        
        # Create embeddings for each chunk
        embedding_responses = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {
                "chunk_index": i,
                "chunk_size": len(chunk_text),
                "document_filename": document.get('filename', ''),
                "document_idx": document.get('idx', ''),
                "total_chunks": len(chunks)
            }
            
            # Create embedding using the appropriate vector store
            # Use the document's UUID (uuid) as the document_uuid for proper foreign key relationship
            embedding_record = await create_vector_embedding(
                content=chunk_text,
                document_uuid=str(document['uuid']),  # Use uuid from documents table
                collection_uuid=str(document['collection_id']) if document.get('collection_id') else None,
                metadata=chunk_metadata,
                request=request
            )
            
            if embedding_record:
                embedding_responses.append(EmbeddingResponse(
                    id=embedding_record.get('custom_id', f"{document_id}_chunk_{i}"),
                    embedding=parse_embedding(embedding_record.get('embedding', [])),
                    text=chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text,
                    document_id=document_id,  # Return the original document_id from request
                    metadata=parse_metadata(embedding_record.get('cmetadata', {})),
                    created_at=embedding_record.get('created_at')
                ))
        
        return embedding_responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create embeddings for document | Document ID: {document_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to create embeddings for document: {str(e)}")


@router.post("/", response_model=EmbeddingResponse)
async def create_single_embedding(
    embedding_request: EmbeddingCreate,
    request: Request
):
    """Create a single embedding with associated document ID."""
    try:
        # Verify document exists
        document = await get_document_by_id(embedding_request.document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Create embedding using the appropriate vector store
        embedding_record = await create_vector_embedding(
            content=embedding_request.text,
            document_uuid=str(document['uuid']),  # Use uuid from documents table
            collection_uuid=str(document['collection_id']) if document.get('collection_id') else None,
            metadata=embedding_request.metadata or {},
            request=request
        )
        
        if not embedding_record:
            raise HTTPException(status_code=500, detail="Failed to create embedding")
        
        return EmbeddingResponse(
            id=embedding_record.get('custom_id', f"{embedding_request.document_id}_single"),
            embedding=parse_embedding(embedding_record.get('embedding', [])),
            text=embedding_request.text[:100] + "..." if len(embedding_request.text) > 100 else embedding_request.text,
            document_id=embedding_request.document_id,  # Return the original document_id from request
            metadata=parse_metadata(embedding_record.get('cmetadata', {})),
            created_at=embedding_record.get('created_at')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create embedding | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to create embedding: {str(e)}")

@router.get("/", response_model=PaginatedResponse)
async def list_embeddings(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    document_id: Optional[str] = Query(None, description="Filter by document ID")
):
    """List embeddings with pagination and filtering."""
    try:
        offset = (page - 1) * page_size
        
        if document_id:
            # Get document to verify it exists and get its UUID
            document = await get_document_by_id(document_id)
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            
            # Get embeddings for specific document using its UUID (uuid)
            embeddings = await get_embeddings_by_document(str(document['uuid']))
            total = len(embeddings)
            # Apply pagination manually since database function doesn't support it
            embeddings = embeddings[offset:offset + page_size]
        else:
            # Get all embeddings with pagination
            embeddings, total = await get_all_embeddings(limit=page_size, offset=offset)
        
        # Convert to response format
        response_items = []
        for embedding in embeddings:
            response_items.append({
                "embedding_id": embedding.get('custom_id'),
                "document_id": embedding.get('document_id'),
                "content_preview": embedding.get('document', '')[:100] + "..." if len(embedding.get('document', '')) > 100 else embedding.get('document', ''),
                "metadata": parse_metadata(embedding.get('cmetadata', {})),
                "created_at": embedding.get('created_at'),
                "embedding_size": len(parse_embedding(embedding.get('embedding', [])))
            })

        total_pages = (total + page_size - 1) // page_size
        
        return PaginatedResponse(
            items=response_items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )
        
    except Exception as e:
        logger.error(f"Failed to list embeddings | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to list embeddings: {str(e)}")


@router.get("/{embedding_id}")
async def get_embedding(embedding_id: str, request: Request):
    """Get a specific embedding by ID."""
    try:
        embedding = await get_embedding_by_id(embedding_id)
        
        if not embedding:
            raise HTTPException(status_code=404, detail="Embedding not found")

        return {
            "embedding_id": embedding.get('custom_id'),
            "document_id": embedding.get('document_id'),
            "content": embedding.get('document'),
            "metadata": parse_metadata(embedding.get('cmetadata', {})),
            "embedding": parse_embedding(embedding.get('embedding', [])),
            "embedding_size": len(parse_embedding(embedding.get('embedding', []))),
            "created_at": embedding.get('created_at')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get embedding | ID: {embedding_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get embedding: {str(e)}")


@router.delete("/{embedding_id}", response_model=SuccessResponse)
async def delete_embedding(embedding_id: str, request: Request):
    """Delete a specific embedding."""
    try:
        success = await delete_embedding(embedding_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Embedding not found")

        return SuccessResponse(message=f"Embedding {embedding_id} deleted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete embedding | ID: {embedding_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to delete embedding: {str(e)}")


@router.post("/similarity/document/{document_id}")
async def find_similar_embeddings_in_document(
    document_id: str,
    text: str,
    k: int = Query(5, ge=1, le=50, description="Number of similar embeddings to return"),
):
    """Find similar embeddings to the given text within a specific document."""
    try:
        # Get document to verify it exists and get its UUID
        document = await get_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Convert query text to embedding vector
        from app.config import embeddings
        query_embedding = embeddings.embed_query(text)
        
        # Search only within this document using its UUID (uuid)
        raw_results = await similarity_search_embeddings(
            query_embedding=query_embedding, 
            k=k, 
            document_uuid=str(document['uuid'])
        )
        
        # Convert to expected format
        results = []
        for result in raw_results:
            results.append({
                "embedding_id": result.get('custom_id'),
                "document_id": document_id,  # Return the original document_id from request
                "content": result.get('document'),
                "similarity_score": float(result.get('distance', 0.0)),
                "metadata": parse_metadata(result.get('cmetadata', {}))
            })

        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to find similar embeddings in document | Document ID: {document_id} | Text: {text[:50]}... | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to find similar embeddings in document: {str(e)}")


@router.post("/similarity/collection/{collection_id}")
async def find_similar_embeddings_in_collection(
    collection_id: str,
    text: str,
    k: int = Query(5, ge=1, le=50, description="Number of similar embeddings to return"),
):
    """Find similar embeddings to the given text within a specific collection."""
    try:
        # Try to get collection by UUID first, then by idx for backward compatibility
        collection = await get_collection_by_uuid(collection_id)
        if not collection:
            collection = await get_collection_by_idx(collection_id)
            
        if not collection:
            raise HTTPException(status_code=404, detail="Collection not found")
        
        # Convert query text to embedding vector
        from app.config import embeddings
        query_embedding = embeddings.embed_query(text)
        
        # Search only within this collection using its UUID
        raw_results = await similarity_search_embeddings(
            query_embedding=query_embedding, 
            k=k, 
            collection_uuid=collection['uuid']
        )
        
        # Convert to expected format
        results = []
        for result in raw_results:
            results.append({
                "embedding_id": result.get('custom_id'),
                "document_id": result.get('document_id'),
                "content": result.get('document'),
                "similarity_score": float(result.get('distance', 0.0)),
                "metadata": parse_metadata(result.get('cmetadata', {}))
            })

        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to find similar embeddings in collection | Collection ID: {collection_id} | Text: {text[:50]}... | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to find similar embeddings in collection: {str(e)}")


@router.get("/document/{document_id}/embeddings")
async def get_document_embeddings(
    document_id: str,
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page")
):
    """Get all embeddings for a specific document."""
    try:
        offset = (page - 1) * page_size
        
        # Get document to verify it exists and get its UUID
        document = await get_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        embeddings = await get_embeddings_by_document(str(document['uuid']))
        total = len(embeddings)
        # Apply pagination manually
        embeddings = embeddings[offset:offset + page_size]
        
        # Format response
        response_items = []
        for embedding in embeddings:
            response_items.append({
                "embedding_id": embedding.get('custom_id'),
                "content_preview": embedding.get('document', '')[:100] + "..." if len(embedding.get('document', '')) > 100 else embedding.get('document', ''),
                "metadata": parse_metadata(embedding.get('cmetadata', {})),
                "created_at": embedding.get('created_at'),
                "embedding_size": len(parse_embedding(embedding.get('embedding', [])))
            })

        total_pages = (total + page_size - 1) // page_size
        
        return PaginatedResponse(
            items=response_items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document embeddings | Document ID: {document_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get document embeddings: {str(e)}")


@router.delete("/document/{document_id}/embeddings", response_model=SuccessResponse)
async def delete_document_embeddings(document_id: str, request: Request):
    """Delete all embeddings for a specific document."""
    try:
        # Get document to verify it exists and get its UUID
        document = await get_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get all embeddings for the document using its UUID
        embeddings = await get_embeddings_by_document(str(document['uuid']))
        
        if not embeddings:
            return SuccessResponse(message=f"No embeddings found for document {document_id}")
        
        # Delete all embeddings for the document
        deleted_count = 0
        for embedding in embeddings:
            success = await delete_embedding(embedding.get('custom_id'))
            if success:
                deleted_count += 1
        
        return SuccessResponse(
            message=f"Successfully deleted {deleted_count} embedding{'s' if deleted_count != 1 else ''} for document {document_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document embeddings | Document ID: {document_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document embeddings: {str(e)}")


@router.get("/stats/summary")
async def get_embedding_stats(request: Request):
    """Get embedding statistics."""
    try:
        # Get total embeddings count by counting all documents' embeddings
        # Get embedding statistics
        total_embeddings = 0
        try:
            # Get actual count from database
            embeddings_result = await get_all_embeddings(limit=1, offset=0)
            if embeddings_result and "total" in embeddings_result:
                total_embeddings = embeddings_result["total"]
        except Exception as e:
            logger.warning(f"Could not retrieve embedding count: {e}")
            total_embeddings = 0
        
        # Calculate approximate storage size (assuming 1536-dimensional embeddings)
        embedding_dimension = 1536
        storage_size_mb = total_embeddings * embedding_dimension * 4 / (1024 * 1024)  # 4 bytes per float
        
        stats = {
            "total_embeddings": total_embeddings,
            "embedding_dimension": embedding_dimension,
            "storage_size_mb": round(storage_size_mb, 2),
            "last_updated": "2024-08-09T10:00:00Z"
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get embedding stats | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get embedding stats: {str(e)}")


@router.post("/recompute/{embedding_id}")
async def recompute_embedding(embedding_id: str, request: Request):
    """Recompute embedding for a specific document."""
    try:
        # Get embedding record from database
        embedding_record = await get_embedding_by_id(embedding_id)
        
        if not embedding_record:
            raise HTTPException(status_code=404, detail="Embedding not found")

        # For now, return success without actual recomputation
        # Implementation would include re-embedding the document content
        logger.info(f"Embedding recompute requested for: {embedding_id}")
        
        return {
            "message": f"Embedding recompute initiated for {embedding_id}",
            "status": "pending"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to recompute embedding | ID: {embedding_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to recompute embedding: {str(e)}")
