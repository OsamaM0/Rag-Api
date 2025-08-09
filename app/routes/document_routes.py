# app/routes/document_routes.py
import traceback
import uuid
import os
import hashlib
import json
from shutil import copyfileobj
from typing import List, Optional

import aiofiles
import aiofiles.os
from fastapi import APIRouter, HTTPException, Query, Request, Body, File, UploadFile, Form, status
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import RAG_UPLOAD_DIR, logger, VECTOR_DB_TYPE, VectorDBType, vector_store
from app.models import (
    DocumentResponse, DocumentCreate, DocumentUpdate, DocumentUploadRequest,
    PaginatedResponse, SuccessResponse
)
from app.services.database import (
    get_all_documents,
    get_documents_by_collection,
    get_document_by_idx,
    get_document_by_serial_id,
    create_document,
    update_document,
    delete_document,
    create_embedding
)
from app.utils.document_loader import get_loader, cleanup_temp_encoding_file

router = APIRouter(prefix="/documents", tags=["Documents"])


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
            documents, total = await get_documents_by_collection(
                collection_id, limit=page_size, offset=offset
            )
            if documents is None:
                raise HTTPException(status_code=404, detail=f"Collection {collection_id} not found")
        else:
            documents, total = await get_all_documents(
                limit=page_size, offset=offset, user_id=user_id, file_id=file_id
            )
        
        # Convert to response format
        response_items = []
        for doc in documents:
            response_items.append(DocumentResponse(
                serial_id=str(doc['serial_id']),
                idx=doc.get('idx'),
                filename=doc.get('filename', ''),
                content=doc.get('content'),
                page_content=doc.get('page_content'),
                mimetype=doc.get('mimetype'),
                binary_hash=doc.get('binary_hash'),
                description=doc.get('description'),
                keywords=doc.get('keywords'),
                page_number=doc.get('page_number'),
                pdf_path=doc.get('pdf_path'),
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
        # Generate a unique ID for the document
        document_id = str(uuid.uuid4())
        
        # Create document in database
        created_doc = await create_document(
            idx=document.idx,
            collection_id=document.collection_id,
            filename=document.filename,
            content=document.content,
            page_content=document.page_content,
            mimetype=document.mimetype,
            description=document.description,
            keywords=document.keywords,
            page_number=document.page_number,
            pdf_path=document.pdf_path,
            metadata=document.metadata or {}
        )
        
        if not created_doc:
            raise HTTPException(status_code=500, detail="Failed to create document in database")
        
        return DocumentResponse(
            serial_id=str(created_doc['serial_id']),
            idx=created_doc.get('idx'),
            filename=created_doc.get('filename', ''),
            content=created_doc.get('content'),
            page_content=created_doc.get('page_content'),
            mimetype=created_doc.get('mimetype'),
            binary_hash=created_doc.get('binary_hash'),
            description=created_doc.get('description'),
            keywords=created_doc.get('keywords'),
            page_number=created_doc.get('page_number'),
            pdf_path=created_doc.get('pdf_path'),
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
    description: Optional[str] = Form(None),
    keywords: Optional[str] = Form(None),
    create_embeddings: bool = Form(True)
):
    """Create a new document by uploading a file with optional embedding generation."""
    try:
        # Generate unique ID for the document
        document_id = str(uuid.uuid4())
        
        # Read file content
        file_content = await file.read()
        file_hash = hashlib.md5(file_content).hexdigest()
        
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
                collection_id=collection_id,
                filename=file.filename,
                content=full_content,
                page_content=page_content,
                mimetype=file.content_type,
                description=description,
                save_pdf_path=(file_ext == "pdf"),
                keywords=keywords,
                pdf_path=temp_file_path if file_ext == "pdf" else None,
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
                serial_id=str(created_doc['serial_id']),
                idx=created_doc.get('idx'),
                filename=created_doc.get('filename', ''),
                content=created_doc.get('content'),
                page_content=created_doc.get('page_content'),
                mimetype=created_doc.get('mimetype'),
                binary_hash=created_doc.get('binary_hash'),
                description=created_doc.get('description'),
                keywords=created_doc.get('keywords'),
                page_number=created_doc.get('page_number'),
                pdf_path=created_doc.get('pdf_path'),
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
    """Get a specific document by ID."""
    try:
        # Try to parse as UUID first (serial_id), if it fails, treat as idx
        try:
            uuid.UUID(document_id)
            # It's a valid UUID, so it's a serial_id
            doc = await get_document_by_serial_id(document_id)
        except ValueError:
            # It's not a UUID, so it's an idx
            doc = await get_document_by_idx(document_id)
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        return DocumentResponse(
            serial_id=str(doc['serial_id']),
            idx=doc.get('idx'),
            filename=doc.get('filename', ''),
            content=doc.get('content'),
            page_content=doc.get('page_content'),
            mimetype=doc.get('mimetype'),
            binary_hash=doc.get('binary_hash'),
            description=doc.get('description'),
            keywords=doc.get('keywords'),
            page_number=doc.get('page_number'),
            pdf_path=doc.get('pdf_path'),
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
async def update_document_endpoint(
    document_id: str,
    document_update: DocumentUpdate,
    request: Request
):
    """Update a document."""
    try:
        # Build update parameters from the DocumentUpdate model
        update_params = {}
        
        if document_update.filename is not None:
            update_params['filename'] = document_update.filename
        if document_update.content is not None:
            update_params['content'] = document_update.content
        if document_update.page_content is not None:
            update_params['page_content'] = document_update.page_content
        if document_update.description is not None:
            update_params['description'] = document_update.description
        if document_update.keywords is not None:
            update_params['keywords'] = document_update.keywords
        if document_update.metadata is not None:
            # Convert metadata to JSON string for database storage
            update_params['metadata'] = json.dumps(document_update.metadata)
        
        # Try to parse as UUID first (serial_id), if it fails, treat as idx
        try:
            uuid.UUID(document_id)
            # For serial_id, we need to get the idx first, then update by idx
            doc = await get_document_by_serial_id(document_id)
            if not doc:
                raise HTTPException(status_code=404, detail="Document not found")
            document_idx = doc['idx']
        except ValueError:
            # It's an idx
            document_idx = document_id
        
        # Update document in database
        updated_doc = await update_document(document_idx, **update_params)
        
        if not updated_doc:
            raise HTTPException(status_code=404, detail="Document not found")

        return DocumentResponse(
            serial_id=str(updated_doc['serial_id']),
            idx=updated_doc.get('idx'),
            filename=updated_doc.get('filename', ''),
            content=updated_doc.get('content'),
            page_content=updated_doc.get('page_content'),
            mimetype=updated_doc.get('mimetype'),
            binary_hash=updated_doc.get('binary_hash'),
            description=updated_doc.get('description'),
            keywords=updated_doc.get('keywords'),
            page_number=updated_doc.get('page_number'),
            pdf_path=updated_doc.get('pdf_path'),
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
        # Delete document and associated embeddings from database
        success = await delete_document(document_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")

        return SuccessResponse(message=f"Document {document_id} deleted successfully")
        
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
            success = await delete_document(doc_id)
            if success:
                deleted_count += 1
            else:
                failed_ids.append(doc_id)
        
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
