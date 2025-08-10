# app/routes/collection_routes.py
import traceback
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Request, Query, status
from pydantic import BaseModel
from app.config import logger
from app.models import (
    CollectionCreate,
    CollectionUpdate,
    CollectionResponse,
    SuccessResponse,
    BulkOperationResponse,
    PaginatedResponse
)
from app.services import database

router = APIRouter(prefix="/collections", tags=["Collections"])


def filter_none_values(data: dict) -> dict:
    """Remove None values from dictionary to support partial updates."""
    return {k: v for k, v in data.items() if v is not None}


# Define a specific response model for paginated collections
class PaginatedCollectionsResponse(BaseModel):
    items: List[CollectionResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


@router.post("/", response_model=CollectionResponse, status_code=status.HTTP_201_CREATED)
async def create_collection(
    collection: CollectionCreate,
    request: Request
):
    """Create a new collection."""
    try:
        result = await database.create_collection(
            name=collection.name,
            description=collection.description,
            idx=collection.idx  # User-defined identifier (optional)
        )
        
        response = CollectionResponse(
            id=result['uuid'],  # UUID as primary identifier
            idx=result.get('idx'),  # User-defined identifier
            name=result['name'],
            description=result['description'],
            metadata=collection.metadata or {},
            created_at=result['created_at'],
            updated_at=result['updated_at']
        )
        
        logger.info(f"Collection created: {collection.name} with UUID: {result['uuid']}")
        return response
        
    except Exception as e:
        logger.error(f"Failed to create collection | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to create collection: {str(e)}")


@router.get("/", response_model=PaginatedCollectionsResponse)
async def list_collections(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search collections by name")
):
    """List all collections with pagination."""
    try:
        offset = (page - 1) * page_size
        collections, total = await database.get_all_collections(limit=page_size, offset=offset)
        
        # Apply search filter if provided
        if search:
            collections = [c for c in collections if search.lower() in c['name'].lower()]
            total = len(collections)  # Update total for filtered results
        
        # Convert to response format
        collection_responses = []
        for collection in collections:
            collection_responses.append(CollectionResponse(
                id=collection['uuid'],  # UUID as primary identifier
                idx=collection.get('idx'),  # User-defined identifier
                name=collection['name'],
                description=collection['description'],
                metadata={},  # Collections don't store complex metadata in this schema
                created_at=collection['created_at'],
                updated_at=collection['updated_at']
            ))
        
        total_pages = (total + page_size - 1) // page_size
        
        return PaginatedCollectionsResponse(
            items=collection_responses,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )
        
    except Exception as e:
        logger.error(f"Failed to list collections | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")


@router.get("/{collection_id}", response_model=CollectionResponse)
async def get_collection(collection_id: str, request: Request):
    """Get a specific collection by ID (accepts both UUID and idx)."""
    try:
        # Try to get by UUID first, then fall back to idx for backward compatibility
        collection = await database.get_collection_by_uuid(collection_id)
        if not collection:
            collection = await database.get_collection_by_idx(collection_id)
            
        if not collection:
            raise HTTPException(status_code=404, detail="Collection not found")
            
        return CollectionResponse(
            id=collection['uuid'],  # UUID as primary identifier
            idx=collection.get('idx'),  # User-defined identifier
            name=collection['name'],
            description=collection['description'],
            metadata={},
            created_at=collection['created_at'],
            updated_at=collection['updated_at']
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get collection | ID: {collection_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get collection: {str(e)}")


@router.put("/{collection_id}", response_model=CollectionResponse)
@router.patch("/{collection_id}", response_model=CollectionResponse)
async def update_collection(
    collection_id: str,
    collection_update: CollectionUpdate,
    request: Request
):
    """Update a collection (supports both PUT and PATCH for partial updates)."""
    try:
        # Build update parameters - only include fields that are not None
        update_data = collection_update.model_dump(exclude_unset=True, exclude_none=True)
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No valid fields provided for update")
        
        # Try to get collection by UUID first, then by idx for backward compatibility
        collection = await database.get_collection_by_uuid(collection_id)
        if not collection:
            collection = await database.get_collection_by_idx(collection_id)
            
        if not collection:
            raise HTTPException(status_code=404, detail="Collection not found")
        
        # Update using UUID
        result = await database.update_collection(
            collection_uuid=collection['uuid'],
            **update_data
        )
        
        return CollectionResponse(
            id=result['uuid'],  # UUID as primary identifier
            idx=result.get('idx'),  # User-defined identifier
            name=result['name'],
            description=result['description'],
            metadata=collection_update.metadata or {},
            created_at=result['created_at'],
            updated_at=result['updated_at']
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update collection | ID: {collection_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to update collection: {str(e)}")


@router.delete("/{collection_id}", response_model=SuccessResponse)
async def delete_collection(collection_id: str, request: Request):
    """Delete a collection and all its documents and embeddings."""
    try:
        # Try to get collection by UUID first, then by idx for backward compatibility
        collection = await database.get_collection_by_uuid(collection_id)
        if not collection:
            collection = await database.get_collection_by_idx(collection_id)
            
        if not collection:
            raise HTTPException(status_code=404, detail="Collection not found")
        
        # Delete using UUID
        await database.delete_collection(collection['uuid'])
        logger.info(f"Collection deleted: {collection['name']} (UUID: {collection['uuid']})")
        return SuccessResponse(message=f"Collection {collection['name']} deleted successfully")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete collection | ID: {collection_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")


@router.post("/bulk-delete", response_model=BulkOperationResponse)
async def bulk_delete_collections(
    collection_ids: List[str],
    request: Request
):
    """Delete multiple collections and their associated data."""
    try:
        success_count = 0
        failed_count = 0
        success_ids = []
        failed_ids = []
        
        for collection_id in collection_ids:
            try:
                # Try to get collection by UUID first, then by idx for backward compatibility
                collection = await database.get_collection_by_uuid(collection_id)
                if not collection:
                    collection = await database.get_collection_by_idx(collection_id)
                    
                if not collection:
                    failed_count += 1
                    failed_ids.append(collection_id)
                    continue
                
                # Delete using UUID
                await database.delete_collection(collection['uuid'])
                success_count += 1
                success_ids.append(collection_id)
                logger.info(f"Collection deleted: {collection['name']} (UUID: {collection['uuid']})")
            except Exception as e:
                failed_count += 1
                failed_ids.append(collection_id)
                logger.error(f"Failed to delete collection {collection_id}: {str(e)}")
        
        return BulkOperationResponse(
            success_count=success_count,
            failed_count=failed_count,
            total_count=len(collection_ids),
            success_ids=success_ids,
            failed_ids=failed_ids,
            message=f"Bulk delete completed: {success_count} successful, {failed_count} failed"
        )
        
    except Exception as e:
        logger.error(f"Failed bulk delete collections | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to bulk delete collections: {str(e)}")


@router.get("/{collection_id}/documents")
async def get_collection_documents(
    collection_id: str,
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100)
):
    """Get all documents in a collection."""
    try:
        # Try to get collection by UUID first, then by idx for backward compatibility
        collection = await database.get_collection_by_uuid(collection_id)
        if not collection:
            collection = await database.get_collection_by_idx(collection_id)
            
        if not collection:
            raise HTTPException(status_code=404, detail=f"Collection {collection_id} not found")
        
        offset = (page - 1) * page_size
        documents, total = await database.get_documents_by_collection(
            collection['uuid'], limit=page_size, offset=offset
        )
        
        total_pages = (total + page_size - 1) // page_size
        
        return PaginatedResponse(
            items=documents,
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
        logger.error(f"Failed to get collection documents | Collection ID: {collection_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get collection documents: {str(e)}")
