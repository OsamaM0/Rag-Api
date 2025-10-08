# app/routes/document_routes.py
import traceback
import uuid
import os
import hashlib
import json
from typing import List, Optional

import aiofiles
import aiofiles.os
from fastapi import APIRouter, HTTPException, Query, Request, Body, File, UploadFile, Form
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import RAG_UPLOAD_DIR, logger, VECTOR_DB_TYPE, VectorDBType, vector_store
from app.models import (
    DocumentResponse, DocumentCreate, DocumentUpdate, PaginatedResponse, SuccessResponse,
    HashSearchRequest, HashSearchResponse, DuplicateDocumentsResponse,
    HashGenerationRequest, HashGenerationResponse
)
from app.services.database import (
    get_all_documents,
    get_documents_by_collection,
    get_document_by_idx,
    get_document_by_uuid,
    create_document,
    update_document_by_idx,
    update_document,
    delete_document_by_idx,
    delete_document,
    create_embedding,
    get_collection_by_uuid,
    get_collection_by_idx,
    get_documents_by_binary_hash,
    get_documents_by_source_binary_hash,
    search_documents_by_hashes,
    get_document_by_binary_hash_single,
    get_document_by_source_binary_hash_single,
    find_duplicate_documents_by_hash,
    generate_blake2b_hash,
    generate_blake2b_hash_from_bytes
)
from app.utils.document_loader import get_loader, cleanup_temp_encoding_file

router = APIRouter(prefix="/documents", tags=["Documents"])


def filter_none_values(data: dict) -> dict:
    """Remove None values from dictionary to support partial updates."""
    return {k: v for k, v in data.items() if v is not None}


def parse_metadata(metadata_value):
    """Helper function to parse metadata from database (could be string or dict)."""
    if isinstance(metadata_value, str):
        try:
            return json.loads(metadata_value)
        except (json.JSONDecodeError, TypeError):
            return {}
    elif isinstance(metadata_value, dict):
        return metadata_value
    else:
        return {}


async def create_document_embeddings(document_id: str, content: str, request: Request = None):
    """Create embeddings for document content using the configured vector store."""
    try:
        if not content:
            return 0
            
        # Split content into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(content)
        
        created_count = 0
        for i, chunk_text in enumerate(chunks):
            try:
                if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
                    # Use database-only approach for PG Vector
                    custom_id = f"{document_id}_chunk_{i}"
                    from app.config import embeddings
                    embedding_vector = embeddings.embed_query(chunk_text)
                    
                    await create_embedding(
                        custom_id=custom_id,
                        embedding=embedding_vector,
                        document_text=chunk_text,
                        document_id=document_id,
                        metadata={
                            "chunk_index": i,
                            "chunk_size": len(chunk_text),
                            "total_chunks": len(chunks)
                        }
                    )
                elif VECTOR_DB_TYPE == VectorDBType.ATLAS_MONGO:
                    # Use vector store approach for Atlas Mongo
                    from langchain_core.documents import Document
                    from app.services.vector_store.async_pg_vector import AsyncPgVector
                    
                    doc = Document(
                        page_content=chunk_text, 
                        metadata={
                            "document_id": document_id,
                            "chunk_index": i,
                            "chunk_size": len(chunk_text),
                            "total_chunks": len(chunks)
                        }
                    )
                    
                    if isinstance(vector_store, AsyncPgVector):
                        await vector_store.aadd_documents([doc], executor=request.app.state.thread_pool if request else None)
                    else:
                        vector_store.add_documents([doc])
                
                created_count += 1
            except Exception as e:
                logger.warning(f"Failed to create embedding for chunk {i} of document {document_id}: {str(e)}")
                
        return created_count
    except Exception as e:
        logger.error(f"Failed to create document embeddings | Document ID: {document_id} | Error: {str(e)}")
        return 0


@router.get("/", response_model=PaginatedResponse)
async def list_documents(
    request: Request,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    file_id: Optional[str] = Query(None, description="Filter by file ID"),
    collection_id: Optional[str] = Query(None, description="Filter by collection ID")
):
    """List documents with pagination and filtering."""
    try:
        offset = (page - 1) * page_size
        
        if collection_id:
            # Try to get collection by UUID first, then by idx and custom_id for backward compatibility
            collection = await get_collection_by_uuid(collection_id)
            if not collection:
                collection = await get_collection_by_idx(collection_id)
            if not collection:
                from app.services.database import get_collection_by_custom_id
                collection = await get_collection_by_custom_id(collection_id)
                
            if not collection:
                raise HTTPException(status_code=404, detail=f"Collection {collection_id} not found")
                
            documents, total = await get_documents_by_collection(
                collection['uuid'], limit=page_size, offset=offset
            )
        else:
            documents, total = await get_all_documents(
                limit=page_size, offset=offset, user_id=user_id, file_id=file_id
            )
        
        # Convert to response format
        response_items = []
        for doc in documents:
            response_items.append(DocumentResponse(
                uuid=str(doc['uuid']),  # UUID as primary identifier
                idx=doc.get('idx'),  # User-defined identifier
                custom_id=doc.get('custom_id'),  # User-defined custom ID
                filename=doc.get('filename', ''),
                content=doc.get('content'),
                page_content=doc.get('page_content'),
                content_with_image=doc.get('content_with_image'),
                mimetype=doc.get('mimetype'),
                binary_hash=doc.get('binary_hash'),
                source_binary_hash=doc.get('source_binary_hash'),
                description=doc.get('description'),
                keywords=doc.get('keywords'),
                page_number=doc.get('page_number'),
                document_path=doc.get('document_path'),
                collection_id=str(doc.get('collection_id')) if doc.get('collection_id') else None,
                collection_name=doc.get('collection_name'),
                metadata=parse_metadata(doc.get('metadata')),
                created_at=doc.get('created_at'),
                updated_at=doc.get('updated_at')
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
        logger.error(f"Failed to list documents | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.post("/", response_model=DocumentResponse)
async def create_document_endpoint(
    request: Request,
    document: DocumentCreate
):
    """Create a new document."""
    try:
        # Validate collection if provided
        collection_uuid = None
        if document.collection_id:
            # Try to get collection by UUID first, then by idx and custom_id for backward compatibility
            collection = await get_collection_by_uuid(document.collection_id)
            if not collection:
                collection = await get_collection_by_idx(document.collection_id)
            if not collection:
                from app.services.database import get_collection_by_custom_id
                collection = await get_collection_by_custom_id(document.collection_id)
                
            if not collection:
                raise HTTPException(status_code=404, detail=f"Collection {document.collection_id} not found")
            collection_uuid = collection['uuid']
        
        # Create document in database
        created_doc = await create_document(
            idx=document.idx,
            custom_id=document.custom_id,
            collection_uuid=collection_uuid,
            filename=document.filename,
            content=document.content,
            page_content=document.page_content,
            mimetype=document.mimetype,
            source_binary_hash=document.source_binary_hash,
            description=document.description,
            keywords=document.keywords,
            page_number=document.page_number,
            document_path=document.document_path,
            metadata=document.metadata or {}
        )
        
        if not created_doc:
            raise HTTPException(status_code=500, detail="Failed to create document in database")
        
        return DocumentResponse(
            uuid=str(created_doc['uuid']),  # UUID as primary identifier
            idx=created_doc.get('idx'),  # User-defined identifier
            custom_id=created_doc.get('custom_id'),  # User-defined custom ID
            filename=created_doc.get('filename', ''),
            content=created_doc.get('content'),
            page_content=created_doc.get('page_content'),
            content_with_image=created_doc.get('content_with_image'),
            mimetype=created_doc.get('mimetype'),
            binary_hash=created_doc.get('binary_hash'),
            source_binary_hash=created_doc.get('source_binary_hash'),
            description=created_doc.get('description'),
            keywords=created_doc.get('keywords'),
            page_number=created_doc.get('page_number'),
            document_path=created_doc.get('document_path'),
            collection_id=str(created_doc.get('collection_id')) if created_doc.get('collection_id') else None,
            collection_name=created_doc.get('collection_name'),
            metadata=parse_metadata(created_doc.get('metadata')),
            created_at=created_doc.get('created_at'),
            updated_at=created_doc.get('updated_at')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create document | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to create document: {str(e)}")


@router.post("/upload", response_model=DocumentResponse)
async def create_document_with_upload(
    request: Request,
    file: UploadFile = File(...),
    collection_id: Optional[str] = Form(None),
    idx: Optional[str] = Form(None),
    custom_id: Optional[str] = Form(None),
    source_binary_hash: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    keywords: Optional[str] = Form(None),
    create_embeddings: bool = Form(True)
):
    """Create a new document by uploading a file with optional embedding generation."""
    try:
        # Validate collection if provided
        collection_uuid = None
        if collection_id:
            # Try to get collection by UUID first, then by idx and custom_id for backward compatibility
            collection = await get_collection_by_uuid(collection_id)
            if not collection:
                collection = await get_collection_by_idx(collection_id)
            if not collection:
                from app.services.database import get_collection_by_custom_id
                collection = await get_collection_by_custom_id(collection_id)
                
            if not collection:
                raise HTTPException(status_code=404, detail=f"Collection {collection_id} not found")
            collection_uuid = collection['uuid']
        
        # Generate unique ID for the document
        document_id = str(uuid.uuid4())
        
        # Read file content
        file_content = await file.read()
        file_hash = hashlib.blake2b(file_content).hexdigest()
        
        # Save file temporarily for processing
        temp_file_path = os.path.join(RAG_UPLOAD_DIR, f"{document_id}_{file.filename}")
        
        try:
            async with aiofiles.open(temp_file_path, 'wb') as f:
                await f.write(file_content)
            
            # Get loader for the file type
            loader, known_type, file_ext = get_loader(
                file.filename, file.content_type, temp_file_path
            )
            
            # Load document data
            from langchain_core.runnables import run_in_executor
            data = await run_in_executor(request.app.state.thread_pool, loader.load)
            
            # Clean up temporary UTF-8 file if it was created
            cleanup_temp_encoding_file(loader)
            
            # Extract content from loaded data
            full_content = ""
            page_content = ""
            if data:
                full_content = "\n".join([doc.page_content for doc in data])
                page_content = data[0].page_content if data else ""
            
            # Create document in database
            created_doc = await create_document(
                idx=idx or document_id,
                custom_id=custom_id,
                collection_uuid=collection_uuid,
                filename=file.filename,
                content=full_content,
                page_content=page_content,
                mimetype=file.content_type,
                source_binary_hash=source_binary_hash,
                description=description,
                keywords=keywords,
                document_path=temp_file_path if file_ext == "pdf" else None,
                metadata=json.dumps({"file_size": len(file_content), "file_ext": file_ext})
            )
            
            if not created_doc:
                raise HTTPException(status_code=500, detail="Failed to create document in database")
            
            # Create embeddings if requested
            if create_embeddings and full_content:
                try:
                    created_count = await create_document_embeddings(document_id, full_content, request)
                    logger.info(f"Created {created_count} embeddings for document {document_id}")
                except Exception as embed_error:
                    logger.warning(f"Failed to create embeddings for document {document_id}: {embed_error}")
                    # Don't fail the whole operation if embedding creation fails
            
            return DocumentResponse(
                uuid=str(created_doc['uuid']),  # UUID as primary identifier
                idx=created_doc.get('idx'),  # User-defined identifier
                custom_id=created_doc.get('custom_id'),  # User-defined custom ID
                filename=created_doc.get('filename', ''),
                content=created_doc.get('content'),
                page_content=created_doc.get('page_content'),
                content_with_image=created_doc.get('content_with_image'),
                mimetype=created_doc.get('mimetype'),
                binary_hash=created_doc.get('binary_hash'),
                source_binary_hash=created_doc.get('source_binary_hash'),
                description=created_doc.get('description'),
                keywords=created_doc.get('keywords'),
                page_number=created_doc.get('page_number'),
                document_path=created_doc.get('document_path'),
                collection_id=str(created_doc.get('collection_id')) if created_doc.get('collection_id') else None,
                collection_name=created_doc.get('collection_name'),
                metadata=parse_metadata(created_doc.get('metadata')),
                created_at=created_doc.get('created_at'),
                updated_at=created_doc.get('updated_at')
            )
            
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_file_path):
                    await aiofiles.os.remove(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary file {temp_file_path}: {cleanup_error}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create document with upload | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to create document with upload: {str(e)}")


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document_endpoint(document_id: str, request: Request):
    """Get a specific document by ID (accepts UUID, idx, or custom_id)."""
    try:
        # Try to get by UUID first, then fall back to idx and custom_id for backward compatibility
        doc = await get_document_by_uuid(document_id)
        if not doc:
            doc = await get_document_by_idx(document_id)
        if not doc:
            from app.services.database import get_document_by_custom_id
            doc = await get_document_by_custom_id(document_id)
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        return DocumentResponse(
            uuid=str(doc['uuid']),  # UUID as primary identifier
            idx=doc.get('idx'),  # User-defined identifier
            custom_id=doc.get('custom_id'),  # User-defined custom ID
            filename=doc.get('filename', ''),
            content=doc.get('content'),
            page_content=doc.get('page_content'),
            content_with_image=doc.get('content_with_image'),
            mimetype=doc.get('mimetype'),
            binary_hash=doc.get('binary_hash'),
            source_binary_hash=doc.get('source_binary_hash'),
            description=doc.get('description'),
            keywords=doc.get('keywords'),
            page_number=doc.get('page_number'),
            document_path=doc.get('document_path'),
            collection_id=str(doc.get('collection_id')) if doc.get('collection_id') else None,
            collection_name=doc.get('collection_name'),
            metadata=parse_metadata(doc.get('metadata')),
            created_at=doc.get('created_at'),
            updated_at=doc.get('updated_at')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document | ID: {document_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@router.put("/{document_id}", response_model=DocumentResponse)
@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document_endpoint(
    document_id: str,
    document_update: DocumentUpdate,
    request: Request
):
    """Update a document (supports both PUT and PATCH for partial updates)."""
    try:
        # Use Pydantic's model_dump to get only set fields, excluding None values
        update_data = document_update.model_dump(exclude_unset=True, exclude_none=True)
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No valid fields provided for update")
        
        # Convert metadata to JSON string if it exists
        if 'metadata' in update_data and update_data['metadata'] is not None:
            update_data['metadata'] = json.dumps(update_data['metadata'])
        
        # Try to get document by UUID first, then by idx and custom_id for backward compatibility
        doc = await get_document_by_uuid(document_id)
        if not doc:
            doc = await get_document_by_idx(document_id)
        if not doc:
            from app.services.database import get_document_by_custom_id
            doc = await get_document_by_custom_id(document_id)
            
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Update document using UUID (uuid)
        updated_doc = await update_document(doc['uuid'], **update_data)
        
        return DocumentResponse(
            uuid=str(updated_doc['uuid']),  # UUID as primary identifier
            idx=updated_doc.get('idx'),  # User-defined identifier
            custom_id=updated_doc.get('custom_id'),  # User-defined custom ID
            filename=updated_doc.get('filename', ''),
            content=updated_doc.get('content'),
            page_content=updated_doc.get('page_content'),
            content_with_image=updated_doc.get('content_with_image'),
            mimetype=updated_doc.get('mimetype'),
            binary_hash=updated_doc.get('binary_hash'),
            source_binary_hash=updated_doc.get('source_binary_hash'),
            description=updated_doc.get('description'),
            keywords=updated_doc.get('keywords'),
            page_number=updated_doc.get('page_number'),
            document_path=updated_doc.get('document_path'),
            collection_id=str(updated_doc.get('collection_id')) if updated_doc.get('collection_id') else None,
            collection_name=updated_doc.get('collection_name'),
            metadata=parse_metadata(updated_doc.get('metadata', {})),
            created_at=updated_doc.get('created_at'),
            updated_at=updated_doc.get('updated_at')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update document | ID: {document_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to update document: {str(e)}")


@router.delete("/{document_id}", response_model=SuccessResponse)
async def delete_document_endpoint(document_id: str, request: Request):
    """Delete a specific document."""
    try:
        # Try to get document by UUID first, then by idx and custom_id for backward compatibility
        doc = await get_document_by_uuid(document_id)
        if not doc:
            doc = await get_document_by_idx(document_id)
        if not doc:
            from app.services.database import get_document_by_custom_id
            doc = await get_document_by_custom_id(document_id)
            
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete document using UUID (uuid)
        success = await delete_document(doc['uuid'])
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete document")

        return SuccessResponse(message=f"Document {doc.get('filename', document_id)} deleted successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document | ID: {document_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.delete("/", response_model=SuccessResponse)
async def delete_documents(request: Request, document_ids: List[str] = Body(...)):
    """Delete multiple documents."""
    try:
        deleted_count = 0
        failed_ids = []
        
        for doc_id in document_ids:
            try:
                # Try to get document by UUID first, then by idx and custom_id for backward compatibility
                doc = await get_document_by_uuid(doc_id)
                if not doc:
                    doc = await get_document_by_idx(doc_id)
                if not doc:
                    from app.services.database import get_document_by_custom_id
                    doc = await get_document_by_custom_id(doc_id)
                    
                if not doc:
                    failed_ids.append(doc_id)
                    continue
                
                # Delete using UUID (uuid)
                success = await delete_document(doc['uuid'])
                if success:
                    deleted_count += 1
                else:
                    failed_ids.append(doc_id)
            except Exception as e:
                failed_ids.append(doc_id)
                logger.error(f"Failed to delete document {doc_id}: {str(e)}")
        
        if failed_ids:
            logger.warning(f"Failed to delete documents: {failed_ids}")
        
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="No documents found to delete")

        message = f"Successfully deleted {deleted_count} document{'s' if deleted_count > 1 else ''}"
        if failed_ids:
            message += f". Failed to delete {len(failed_ids)} document{'s' if len(failed_ids) > 1 else ''}"
            
        return SuccessResponse(message=message)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete documents | IDs: {document_ids} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to delete documents: {str(e)}")


# Hash-based Search Endpoints
@router.post("/search/hash", response_model=HashSearchResponse)
async def search_documents_by_hash(
    request: Request,
    search_request: HashSearchRequest
):
    """Search documents by binary_hash or source_binary_hash with fast indexing."""
    try:
        if not search_request.binary_hash and not search_request.source_binary_hash:
            raise HTTPException(
                status_code=400, 
                detail="At least one hash parameter (binary_hash or source_binary_hash) is required"
            )
        
        offset = (search_request.page - 1) * search_request.page_size
        
        # Validate collection if provided
        collection_uuid = None
        if search_request.collection_id:
            collection = await get_collection_by_uuid(search_request.collection_id)
            if not collection:
                collection = await get_collection_by_idx(search_request.collection_id)
            if not collection:
                from app.services.database import get_collection_by_custom_id
                collection = await get_collection_by_custom_id(search_request.collection_id)
                
            if not collection:
                raise HTTPException(status_code=404, detail=f"Collection {search_request.collection_id} not found")
            collection_uuid = collection['uuid']
        
        # Perform search
        documents, total = await search_documents_by_hashes(
            binary_hash=search_request.binary_hash,
            source_binary_hash=search_request.source_binary_hash,
            collection_uuid=collection_uuid,
            limit=search_request.page_size,
            offset=offset
        )
        
        # Convert to response format
        response_documents = []
        for doc in documents:
            response_documents.append(DocumentResponse(
                uuid=str(doc['uuid']),
                idx=doc.get('idx'),
                custom_id=doc.get('custom_id'),
                filename=doc.get('filename', ''),
                content=doc.get('content'),
                page_content=doc.get('page_content'),
                content_with_image=doc.get('content_with_image'),
                mimetype=doc.get('mimetype'),
                binary_hash=doc.get('binary_hash'),
                source_binary_hash=doc.get('source_binary_hash'),
                description=doc.get('description'),
                keywords=doc.get('keywords'),
                page_number=doc.get('page_number'),
                document_path=doc.get('document_path'),
                collection_id=str(doc.get('collection_id')) if doc.get('collection_id') else None,
                collection_name=doc.get('collection_name'),
                metadata=parse_metadata(doc.get('metadata')),
                created_at=doc.get('created_at'),
                updated_at=doc.get('updated_at')
            ))
        
        total_pages = (total + search_request.page_size - 1) // search_request.page_size
        
        return HashSearchResponse(
            documents=response_documents,
            total=total,
            page=search_request.page,
            page_size=search_request.page_size,
            total_pages=total_pages,
            search_params=search_request
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search documents by hash | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to search documents by hash: {str(e)}")


@router.get("/search/binary-hash/{binary_hash}", response_model=DocumentResponse)
async def get_document_by_binary_hash(binary_hash: str, request: Request):
    """Get a single document by binary_hash (fast lookup)."""
    try:
        doc = await get_document_by_binary_hash_single(binary_hash)
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found with the specified binary_hash")
        
        return DocumentResponse(
            uuid=str(doc['uuid']),
            idx=doc.get('idx'),
            custom_id=doc.get('custom_id'),
            filename=doc.get('filename', ''),
            content=doc.get('content'),
            page_content=doc.get('page_content'),
            content_with_image=doc.get('content_with_image'),
            mimetype=doc.get('mimetype'),
            binary_hash=doc.get('binary_hash'),
            source_binary_hash=doc.get('source_binary_hash'),
            description=doc.get('description'),
            keywords=doc.get('keywords'),
            page_number=doc.get('page_number'),
            document_path=doc.get('document_path'),
            collection_id=str(doc.get('collection_id')) if doc.get('collection_id') else None,
            collection_name=doc.get('collection_name'),
            metadata=parse_metadata(doc.get('metadata')),
            created_at=doc.get('created_at'),
            updated_at=doc.get('updated_at')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document by binary_hash | Hash: {binary_hash} | Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get document by binary_hash: {str(e)}")


@router.get("/search/source-binary-hash/{source_binary_hash}", response_model=DocumentResponse)
async def get_document_by_source_binary_hash(source_binary_hash: str, request: Request):
    """Get a single document by source_binary_hash (fast lookup)."""
    try:
        doc = await get_document_by_source_binary_hash_single(source_binary_hash)
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found with the specified source_binary_hash")
        
        return DocumentResponse(
            uuid=str(doc['uuid']),
            idx=doc.get('idx'),
            custom_id=doc.get('custom_id'),
            filename=doc.get('filename', ''),
            content=doc.get('content'),
            page_content=doc.get('page_content'),
            mimetype=doc.get('mimetype'),
            binary_hash=doc.get('binary_hash'),
            source_binary_hash=doc.get('source_binary_hash'),
            description=doc.get('description'),
            keywords=doc.get('keywords'),
            page_number=doc.get('page_number'),
            document_path=doc.get('document_path'),
            collection_id=str(doc.get('collection_id')) if doc.get('collection_id') else None,
            collection_name=doc.get('collection_name'),
            metadata=parse_metadata(doc.get('metadata')),
            created_at=doc.get('created_at'),
            updated_at=doc.get('updated_at')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document by source_binary_hash | Hash: {source_binary_hash} | Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get document by source_binary_hash: {str(e)}")


@router.get("/duplicates/{hash_type}", response_model=List[DuplicateDocumentsResponse])
async def find_duplicate_documents(
    hash_type: str,
    request: Request
):
    """Find documents with duplicate hashes for deduplication."""
    try:
        if hash_type not in ["binary_hash", "source_binary_hash"]:
            raise HTTPException(
                status_code=400,
                detail="hash_type must be 'binary_hash' or 'source_binary_hash'"
            )
        
        duplicates = await find_duplicate_documents_by_hash(hash_type)
        
        response = []
        for duplicate in duplicates:
            response.append(DuplicateDocumentsResponse(
                hash_value=duplicate[hash_type],
                count=duplicate['count'],
                document_ids=[str(uuid) for uuid in duplicate['document_ids']],
                hash_type=hash_type
            ))
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to find duplicate documents | Hash type: {hash_type} | Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to find duplicate documents: {str(e)}")


@router.post("/generate-hash", response_model=HashGenerationResponse)
async def generate_hash_for_content(
    request: Request,
    hash_request: HashGenerationRequest
):
    """Generate blake2b hash for given content."""
    try:
        if not hash_request.content:
            raise HTTPException(status_code=400, detail="Content cannot be empty")
        
        if hash_request.hash_type != "blake2b":
            raise HTTPException(status_code=400, detail="Only blake2b hash type is currently supported")
        
        hash_value = generate_blake2b_hash(hash_request.content)
        
        return HashGenerationResponse(
            hash_value=hash_value,
            hash_type=hash_request.hash_type,
            content_length=len(hash_request.content)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate hash | Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate hash: {str(e)}")
