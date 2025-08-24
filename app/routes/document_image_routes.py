# app/routes/document_image_routes.py
import traceback
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import JSONResponse

from app.config import logger
from app.models import (
    DocumentImageResponse,
    PaginatedResponse,
    SuccessResponse
)
from app.services.database import (
    get_document_images_by_document,
    get_document_image_by_id,
    delete_document_image,
    delete_document_images_by_document,
    get_document_by_uuid
)

router = APIRouter(prefix="/document-images", tags=["Document Images"])


@router.get("/", response_model=PaginatedResponse)
async def list_document_images(
    request: Request,
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page")
):
    """List document images with pagination and filtering."""
    try:
        if not document_id:
            raise HTTPException(status_code=400, detail="document_id parameter is required")
        
        offset = (page - 1) * page_size
        
        images, total = await get_document_images_by_document(
            document_id=document_id,
            limit=page_size,
            offset=offset
        )
        
        # Convert to response format
        response_items = []
        for image in images:
            response_items.append(DocumentImageResponse(
                id=str(image['id']),
                document_id=str(image['document_id']),
                page_no=image['page_no'],
                mimetype=image['mimetype'],
                dpi=image.get('dpi'),
                width=image.get('width'),
                height=image.get('height'),
                page_width=image.get('page_width'),
                page_height=image.get('page_height'),
                uri=image['uri'],
                created_at=image.get('created_at')
            ))
        
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
        logger.error(f"Failed to list document images | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to list document images: {str(e)}")


@router.get("/document/{document_id}", response_model=PaginatedResponse)
async def get_document_images(
    document_id: str,
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Items per page")
):
    """Get all images for a specific document."""
    try:
        # Validate document exists
        document = await get_document_by_uuid(document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        offset = (page - 1) * page_size
        images, total = await get_document_images_by_document(document_id, limit=page_size, offset=offset)
        
        # Convert to response format
        response_items = []
        for image in images:
            response_items.append(DocumentImageResponse(
                id=str(image['id']),
                document_id=str(image['document_id']),
                page_no=image['page_no'],
                mimetype=image['mimetype'],
                dpi=image.get('dpi'),
                width=image.get('width'),
                height=image.get('height'),
                page_width=image.get('page_width'),
                page_height=image.get('page_height'),
                uri=image['uri'],
                created_at=image.get('created_at')
            ))
        
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
        logger.error(f"Failed to get document images | Document ID: {document_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get document images: {str(e)}")


@router.get("/{image_id}", response_model=DocumentImageResponse)
async def get_document_image_endpoint(image_id: str, request: Request):
    """Get a specific document image by ID."""
    try:
        image = await get_document_image_by_id(image_id)
        if not image:
            raise HTTPException(status_code=404, detail=f"Document image {image_id} not found")
        
        return DocumentImageResponse(
            id=str(image['id']),
            document_id=str(image['document_id']),
            page_no=image['page_no'],
            mimetype=image['mimetype'],
            dpi=image.get('dpi'),
            width=image.get('width'),
            height=image.get('height'),
            page_width=image.get('page_width'),
            page_height=image.get('page_height'),
            uri=image['uri'],
            created_at=image.get('created_at')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document image | ID: {image_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get document image: {str(e)}")


@router.delete("/{image_id}", response_model=SuccessResponse)
async def delete_document_image_endpoint(image_id: str, request: Request):
    """Delete a specific document image."""
    try:
        # Check if image exists
        existing_image = await get_document_image_by_id(image_id)
        if not existing_image:
            raise HTTPException(status_code=404, detail=f"Document image {image_id} not found")
        
        success = await delete_document_image(image_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete document image")
        
        return SuccessResponse(
            message=f"Successfully deleted document image {image_id}",
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document image | ID: {image_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document image: {str(e)}")


@router.delete("/document/{document_id}", response_model=SuccessResponse)
async def delete_document_images_endpoint(document_id: str, request: Request):
    """Delete all images for a specific document."""
    try:
        # Validate document exists
        document = await get_document_by_uuid(document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        deleted_count = await delete_document_images_by_document(document_id)
        
        return SuccessResponse(
            message=f"Successfully deleted {deleted_count} document images for document {document_id}",
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document images | Document ID: {document_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document images: {str(e)}")
