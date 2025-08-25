# app/models.py
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from pydantic import ConfigDict
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import json


# Document Models - Using Proper PK/FK Relationships
class DocumentResponse(BaseModel):
    uuid: str  # UUID primary key (system identifier)
    idx: Optional[str] = None  # User-defined identifier (optional)
    custom_id: Optional[str] = None  # User-defined custom ID (optional)
    filename: str
    content: Optional[str] = None
    page_content: Optional[str] = None  # For vector chunks
    mimetype: Optional[str] = None
    binary_hash: Optional[str] = None
    source_binary_hash: Optional[str] = None  # Hash for source content identification
    description: Optional[str] = None
    keywords: Optional[str] = None
    page_number: Optional[int] = None
    document_path: Optional[str] = None
    collection_id: Optional[str] = None  # UUID foreign key to collection
    collection_name: Optional[str] = None
    metadata: Optional[dict] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DocumentCreate(BaseModel):
    filename: str = Field(..., description="Original filename")
    content: Optional[str] = Field(None, description="Full text content of the document")
    page_content: Optional[str] = Field(None, description="Chunk content for vector search")
    mimetype: str = Field(..., description="MIME type of the document")
    source_binary_hash: Optional[str] = Field(None, description="Blake2b hash for source content identification")
    description: Optional[str] = Field(None, description="Document description")
    keywords: Optional[str] = Field(None, description="Document keywords")
    page_number: Optional[int] = Field(None, description="Page number if applicable")
    collection_id: str = Field(..., description="Collection UUID this document belongs to")
    metadata: Optional[Union[dict, str]] = Field(default_factory=dict, description="Document metadata as dict or JSON string")
    file_id: Optional[str] = Field(None, description="Legacy file ID for compatibility")
    user_id: Optional[str] = Field(None, description="User ID")
    idx: Optional[str] = Field(None, description="User-defined document identifier (optional)")
    custom_id: Optional[str] = Field(None, description="User-defined custom ID (optional)")
    document_path: Optional[str] = Field(None, description="Path to the document file")

    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v):
        """Convert string metadata to dict if needed."""
        if v is None:
            return {}
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("metadata string must be valid JSON")
        if isinstance(v, dict):
            return v
        raise ValueError("metadata must be a dict or valid JSON string")



class DocumentUpdate(BaseModel):
    """Model for partial document updates. All fields are optional to support PATCH operations."""
    filename: Optional[str] = Field(None, description="Original filename")
    content: Optional[str] = Field(None, description="Full text content of the document")
    page_content: Optional[str] = Field(None, description="Chunk content for vector search")
    source_binary_hash: Optional[str] = Field(None, description="Blake2b hash for source content identification")
    description: Optional[str] = Field(None, description="Document description")
    keywords: Optional[str] = Field(None, description="Document keywords")
    custom_id: Optional[str] = Field(None, description="User-defined custom ID")
    metadata: Optional[Union[dict, str]] = Field(None, description="Document metadata as dict or JSON string")

    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v):
        """Convert string metadata to dict if needed."""
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("metadata string must be valid JSON")
        if isinstance(v, dict):
            return v
        raise ValueError("metadata must be a dict or valid JSON string")

    model_config = ConfigDict(extra="forbid")


class DocumentUploadRequest(BaseModel):
    description: Optional[str] = Field(None, description="Document description")
    keywords: Optional[str] = Field(None, description="Document keywords")
    collection_id: str = Field(..., description="Collection UUID this document belongs to")
    auto_embed: bool = Field(True, description="Automatically create embeddings for document")


# Legacy models for backward compatibility
class StoreDocument(BaseModel):
    filepath: str
    filename: str
    file_content_type: str
    file_id: str


# Collection Models - Using Proper PK/FK Relationships  
class CollectionCreate(BaseModel):
    name: str = Field(..., description="Collection name")
    description: Optional[str] = Field(None, description="Collection description")
    idx: Optional[str] = Field(None, description="User-defined collection identifier (optional)")
    custom_id: Optional[str] = Field(None, description="User-defined custom ID (optional)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class CollectionUpdate(BaseModel):
    """Model for partial collection updates. All fields are optional to support PATCH operations."""
    name: Optional[str] = Field(None, description="Collection name")
    description: Optional[str] = Field(None, description="Collection description")
    idx: Optional[str] = Field(None, description="User-defined collection identifier")
    custom_id: Optional[str] = Field(None, description="User-defined custom ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Collection metadata")

    model_config = ConfigDict(extra="forbid")


class CollectionResponse(BaseModel):
    id: str = Field(..., description="Collection UUID (primary key)")
    idx: Optional[str] = Field(None, description="User-defined collection identifier")
    custom_id: Optional[str] = Field(None, description="User-defined custom ID")
    name: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# Query Models
class QueryRequestBody(BaseModel):
    query: str
    file_id: str
    k: int = 4
    entity_id: Optional[str] = None


class QueryMultipleBody(BaseModel):
    query: str
    file_ids: List[str]
    k: int = 4


class SimilaritySearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    k: int = Field(default=4, description="Number of results to return")
    filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filter")
    score_threshold: Optional[float] = Field(None, description="Minimum similarity score")


class SimilaritySearchResponse(BaseModel):
    document: DocumentResponse
    score: float


# Hash-based Search Models
class HashSearchRequest(BaseModel):
    binary_hash: Optional[str] = Field(None, description="Binary hash to search for")
    source_binary_hash: Optional[str] = Field(None, description="Source binary hash to search for")
    collection_id: Optional[str] = Field(None, description="Collection ID to filter by")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=10, ge=1, le=100, description="Items per page")


class HashSearchResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
    search_params: HashSearchRequest


class DuplicateDocumentsResponse(BaseModel):
    hash_value: str
    count: int
    document_ids: List[str]
    hash_type: str = Field(description="Type of hash: binary_hash or source_binary_hash")


# Hash Generation Models
class HashGenerationRequest(BaseModel):
    content: str = Field(..., description="Content to generate hash for")
    hash_type: str = Field(default="blake2b", description="Hash type (currently only blake2b)")


class HashGenerationResponse(BaseModel):
    hash_value: str
    hash_type: str
    content_length: int

class EmbeddingResponse(BaseModel):
    id: str
    embedding: List[float]
    text: str
    document_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


class EmbeddingCreate(BaseModel):
    text: str = Field(..., description="Text to embed")
    document_id: str = Field(..., description="Document ID this embedding belongs to")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    chunk_index: Optional[int] = Field(None, description="Index of this chunk within the document")


class BatchEmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed")
    document_id: Optional[str] = Field(None, description="Document ID to associate with all embeddings")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class BatchEmbeddingResponse(BaseModel):
    embeddings: List[EmbeddingResponse]
    total_created: int


# Database Models
class DatabaseStatsResponse(BaseModel):
    total_documents: int
    total_collections: int
    total_embeddings: int
    database_size: Optional[str] = None


class DatabaseHealthResponse(BaseModel):
    status: str
    database_connected: bool
    vector_store_ready: bool
    last_check: datetime


# Generic Response Models
class SuccessResponse(BaseModel):
    message: str
    success: bool = True


class ErrorResponse(BaseModel):
    message: str
    error: str
    success: bool = False


class BulkOperationResponse(BaseModel):
    success_count: int
    failed_count: int
    total_count: int
    success_ids: List[str] = []
    failed_ids: List[str] = []
    message: str


# Enums
class CleanupMethod(str, Enum):
    incremental = "incremental"
    full = "full"


class SortOrder(str, Enum):
    asc = "asc"
    desc = "desc"


# Document Blocks Models
class DocumentBlockCreate(BaseModel):
    block_idx: int = Field(..., description="Block index (order)")
    name: str = Field(..., description="Block name/identifier")
    content: Optional[str] = Field(None, description="Block content")
    level: int = Field(..., description="Hierarchical level")
    page_idx: int = Field(..., description="Page index")
    tag: str = Field(..., description="Block tag (para, header, etc.)")
    block_class: Optional[str] = Field(None, description="Block class")
    x0: Optional[float] = Field(None, description="Left coordinate")
    y0: Optional[float] = Field(None, description="Bottom coordinate")
    x1: Optional[float] = Field(None, description="Right coordinate")
    y1: Optional[float] = Field(None, description="Top coordinate")
    parent_idx: Optional[int] = Field(None, description="Parent block index")
    content_type: str = Field(default="regular", description="Content type")
    section_type: Optional[str] = Field(None, description="Section type")
    demand_priority: Optional[int] = Field(None, description="Demand priority")


class DocumentBlockResponse(BaseModel):
    id: str = Field(..., description="Block UUID")
    block_idx: int = Field(..., description="Block index (order)")
    document_id: str = Field(..., description="Document UUID")
    name: str = Field(..., description="Block name/identifier")
    content: Optional[str] = Field(None, description="Block content")
    level: int = Field(..., description="Hierarchical level")
    page_idx: int = Field(..., description="Page index")
    tag: str = Field(..., description="Block tag")
    block_class: Optional[str] = Field(None, description="Block class")
    x0: Optional[float] = Field(None, description="Left coordinate")
    y0: Optional[float] = Field(None, description="Bottom coordinate")
    x1: Optional[float] = Field(None, description="Right coordinate")
    y1: Optional[float] = Field(None, description="Top coordinate")
    parent_idx: Optional[int] = Field(None, description="Parent block index")
    content_type: str = Field(default="regular", description="Content type")
    section_type: Optional[str] = Field(None, description="Section type")
    demand_priority: Optional[int] = Field(None, description="Demand priority")
    created_at: Optional[datetime] = None


class DocumentBlockUpdate(BaseModel):
    """Model for partial document block updates."""
    name: Optional[str] = Field(None, description="Block name/identifier")
    content: Optional[str] = Field(None, description="Block content")
    level: Optional[int] = Field(None, description="Hierarchical level")
    tag: Optional[str] = Field(None, description="Block tag")
    block_class: Optional[str] = Field(None, description="Block class")
    content_type: Optional[str] = Field(None, description="Content type")
    section_type: Optional[str] = Field(None, description="Section type")
    demand_priority: Optional[int] = Field(None, description="Demand priority")

    model_config = ConfigDict(extra="forbid")


class DocumentBlocksBulkCreate(BaseModel):
    document_id: str = Field(..., description="Document UUID")
    blocks: List[DocumentBlockCreate] = Field(..., description="List of blocks to create")


# Document Images Models
class DocumentImageCreate(BaseModel):
    page_no: int = Field(..., description="Page number")
    mimetype: str = Field(..., description="Image MIME type")
    dpi: Optional[int] = Field(None, description="Image DPI")
    width: Optional[float] = Field(None, description="Image width")
    height: Optional[float] = Field(None, description="Image height")
    page_width: Optional[float] = Field(None, description="Page width")
    page_height: Optional[float] = Field(None, description="Page height")
    uri: str = Field(..., description="Base64 encoded image data URI")
    document_path: Optional[str] = Field(None, description="Path to the document file")


class DocumentImageResponse(BaseModel):
    id: str = Field(..., description="Image UUID")
    document_id: str = Field(..., description="Document UUID")
    page_no: int = Field(..., description="Page number")
    mimetype: str = Field(..., description="Image MIME type")
    dpi: Optional[int] = Field(None, description="Image DPI")
    width: Optional[float] = Field(None, description="Image width")
    height: Optional[float] = Field(None, description="Image height")
    page_width: Optional[float] = Field(None, description="Page width")
    page_height: Optional[float] = Field(None, description="Page height")
    uri: str = Field(..., description="Base64 encoded image data URI")
    created_at: Optional[datetime] = None


class DocumentImageUpdate(BaseModel):
    """Model for partial document image updates."""
    mimetype: Optional[str] = Field(None, description="Image MIME type")
    dpi: Optional[int] = Field(None, description="Image DPI")
    width: Optional[float] = Field(None, description="Image width")
    height: Optional[float] = Field(None, description="Image height")
    page_width: Optional[float] = Field(None, description="Page width")
    page_height: Optional[float] = Field(None, description="Page height")
    uri: Optional[str] = Field(None, description="Base64 encoded image data URI")

    model_config = ConfigDict(extra="forbid")


# Pagination Models
class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=10, ge=1, le=100, description="Items per page")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: SortOrder = Field(default=SortOrder.desc, description="Sort order")


class PaginatedResponse(BaseModel):
    items: List[Any]  # Allow any type of items for flexibility
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool

    model_config = ConfigDict(arbitrary_types_allowed=True)
