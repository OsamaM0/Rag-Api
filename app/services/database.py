# app/services/database.py
import asyncpg
import hashlib
import uuid
from app.config import DSN, logger


class PSQLDatabase:
    pool = None

    @classmethod
    async def get_pool(cls):
        if cls.pool is None:
            cls.pool = await asyncpg.create_pool(dsn=DSN)
        return cls.pool

    @classmethod
    async def close_pool(cls):
        if cls.pool is not None:
            await cls.pool.close()
            cls.pool = None


async def ensure_three_table_schema():
    """Ensure the consolidated three-table schema: Collection, Document, Embedding (PG Vector)."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        # 1. Collection table - langchain_pg_collection (enhanced)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS langchain_pg_collection (
                uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                uuid SERIAL UNIQUE,
                idx VARCHAR UNIQUE,
                custom_id VARCHAR UNIQUE,
                name VARCHAR NOT NULL,
                description TEXT,
                cmetadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Add missing columns if they don't exist
        await conn.execute("""
            ALTER TABLE langchain_pg_collection 
            ADD COLUMN IF NOT EXISTS uuid SERIAL UNIQUE,
            ADD COLUMN IF NOT EXISTS idx VARCHAR UNIQUE,
            ADD COLUMN IF NOT EXISTS custom_id VARCHAR UNIQUE,
            ADD COLUMN IF NOT EXISTS description TEXT,
            ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW(),
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
        """)
        
        # Update existing collection records to have idx values if they don't exist
        await conn.execute("""
            UPDATE langchain_pg_collection 
            SET idx = COALESCE(idx, uuid::text)
            WHERE idx IS NULL;
        """)
        
        # 2. Document table - consolidated for full document storage and vector chunks
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                idx VARCHAR UNIQUE,
                custom_id VARCHAR UNIQUE,
                filename VARCHAR NOT NULL,
                content TEXT,
                page_content TEXT,  -- For vector search chunks
                mimetype VARCHAR,
                binary_hash VARCHAR,
                description TEXT,
                keywords TEXT,
                page_number INTEGER,
                pdf_path TEXT,
                collection_id UUID REFERENCES langchain_pg_collection(uuid) ON DELETE CASCADE,
                metadata JSONB DEFAULT '{}',
                file_id VARCHAR,  -- Legacy compatibility
                user_id VARCHAR,  -- Legacy compatibility
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # 3. Embedding table - langchain_pg_embedding (ensure vector extension and structure)
        # Enable pgvector extension
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception as e:
            logger.warning(f"Could not create vector extension: {e}")
        
        # Ensure embedding table exists with proper structure
        # First create table without foreign key constraints that might fail
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
                uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                collection_id UUID,
                document_id UUID,
                custom_id VARCHAR,
                embedding VECTOR,
                document TEXT,
                cmetadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        # Add missing columns to embedding table if they don't exist
        await conn.execute("""
            ALTER TABLE langchain_pg_embedding 
            ADD COLUMN IF NOT EXISTS document_id UUID,
            ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();
        """)
        
        # Add missing columns to documents table if they don't exist
        await conn.execute("""
            ALTER TABLE documents 
            ADD COLUMN IF NOT EXISTS custom_id VARCHAR UNIQUE;
        """)
        
        # Add foreign key constraints if they don't exist (after all tables are created)
        try:
            # Check if collection foreign key exists
            collection_fk_exists = await conn.fetchval("""
                SELECT COUNT(*) FROM information_schema.table_constraints 
                WHERE constraint_name = 'fk_embedding_collection' 
                AND table_name = 'langchain_pg_embedding'
            """)
            if not collection_fk_exists:
                await conn.execute("""
                    ALTER TABLE langchain_pg_embedding 
                    ADD CONSTRAINT fk_embedding_collection 
                    FOREIGN KEY (collection_id) REFERENCES langchain_pg_collection(uuid) ON DELETE CASCADE;
                """)
        except Exception as e:
            logger.warning(f"Could not add collection foreign key constraint: {e}")
            
        try:
            # Check if document foreign key exists
            document_fk_exists = await conn.fetchval("""
                SELECT COUNT(*) FROM information_schema.table_constraints 
                WHERE constraint_name = 'fk_embedding_document' 
                AND table_name = 'langchain_pg_embedding'
            """)
            if not document_fk_exists:
                await conn.execute("""
                    ALTER TABLE langchain_pg_embedding 
                    ADD CONSTRAINT fk_embedding_document 
                    FOREIGN KEY (document_id) REFERENCES documents(uuid) ON DELETE CASCADE;
                """)
        except Exception as e:
            logger.warning(f"Could not add document foreign key constraint: {e}")
        
        # Create optimized indexes
        indexes_to_create = [
            # Collection indexes
            ("idx_collections_idx", "langchain_pg_collection", "idx"),
            ("idx_collections_custom_id", "langchain_pg_collection", "custom_id"),
            ("idx_collections_name", "langchain_pg_collection", "name"),
            
            # Document indexes
            ("idx_documents_idx", "documents", "idx"),
            ("idx_documents_custom_id", "documents", "custom_id"),
            ("idx_documents_collection_id", "documents", "collection_id"),
            ("idx_documents_filename", "documents", "filename"),
            ("idx_documents_mimetype", "documents", "mimetype"),
            ("idx_documents_binary_hash", "documents", "binary_hash"),
            ("idx_documents_file_id", "documents", "file_id"),
            ("idx_documents_user_id", "documents", "user_id"),
            
            # Embedding indexes
            ("idx_embeddings_document_id", "langchain_pg_embedding", "document_id"),
            ("idx_embeddings_custom_id", "langchain_pg_embedding", "custom_id"),
            ("idx_embeddings_collection_id", "langchain_pg_embedding", "collection_id"),
        ]
        
        for index_name, table_name, column_name in indexes_to_create:
            try:
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({column_name});
                """)
            except Exception as e:
                logger.warning(f"Could not create index {index_name}: {e}")
        
        # Create vector similarity search index
        try:
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_vector 
                ON langchain_pg_embedding USING ivfflat (embedding vector_cosine_ops);
            """)
        except Exception as e:
            logger.warning(f"Could not create vector index: {e}")
        
        logger.info("Three-table consolidated schema ensured: Collection, Document, Embedding")


async def ensure_vector_indexes():
    """Legacy function - now calls ensure_three_table_schema for backward compatibility."""
    await ensure_three_table_schema()


# Collection Operations - Using Proper PK/FK Relationships
async def create_collection(name: str, description: str = None, idx: str = None, custom_id: str = None):
    """Create a new collection using UUID as primary key, idx and custom_id as optional user metadata."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        # Generate UUID explicitly for primary key
        collection_uuid = str(uuid.uuid4())
        
        # idx and custom_id are now optional user-defined metadata, not system identifiers
        result = await conn.fetchrow("""
            INSERT INTO langchain_pg_collection (uuid, name, idx, custom_id, description, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
            RETURNING uuid, idx, custom_id, name, description, created_at, updated_at
        """, collection_uuid, name, idx, custom_id, description)
        
        # Convert the result to a dict and ensure UUID is string
        result_dict = dict(result)
        result_dict['uuid'] = str(result_dict['uuid'])
        return result_dict


async def get_collection_by_uuid(collection_uuid: str):
    """Get collection by its UUID primary key."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        try: 
            result = await conn.fetchrow("""
                SELECT uuid, idx, custom_id, name, description, created_at, updated_at
                FROM langchain_pg_collection 
                WHERE uuid = $1
            """, collection_uuid)
            if result:
                result_dict = dict(result)
                result_dict['uuid'] = str(result_dict['uuid'])
                return result_dict
            return None
        except Exception as e:
            logger.error(f"Failed to get collection by UUID | ID: {collection_uuid} | Error: {str(e)}")
            return None


async def get_collection_by_idx(idx: str):
    """Get collection by its idx (user-defined identifier) - DEPRECATED."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        result = await conn.fetchrow("""
            SELECT uuid, idx, custom_id, name, description, created_at, updated_at
            FROM langchain_pg_collection 
            WHERE idx = $1
        """, idx)
        if result:
            result_dict = dict(result)
            result_dict['uuid'] = str(result_dict['uuid'])
            return result_dict
        return None


async def get_collection_by_custom_id(custom_id: str):
    """Get collection by its custom_id (user-defined identifier)."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        result = await conn.fetchrow("""
            SELECT uuid, idx, custom_id, name, description, created_at, updated_at
            FROM langchain_pg_collection 
            WHERE custom_id = $1
        """, custom_id)
        if result:
            result_dict = dict(result)
            result_dict['uuid'] = str(result_dict['uuid'])
            return result_dict
        return None


async def get_all_collections(limit: int = 10, offset: int = 0):
    """Get all collections with pagination."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        results = await conn.fetch("""
            SELECT uuid, idx, custom_id, name, description, created_at, updated_at
            FROM langchain_pg_collection
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
        """, limit, offset)
        
        total = await conn.fetchval("SELECT COUNT(*) FROM langchain_pg_collection")
        
        # Convert UUIDs to strings for all results
        collections = []
        for row in results:
            row_dict = dict(row)
            row_dict['uuid'] = str(row_dict['uuid'])
            collections.append(row_dict)
        
        return collections, total

async def update_collection(collection_uuid: str, name: str = None, description: str = None, idx: str = None, custom_id: str = None):
    """Update collection by UUID primary key."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        updates = []
        params = []
        param_count = 0
        
        if name is not None:
            param_count += 1
            updates.append(f"name = ${param_count}")
            params.append(name)
            
        if description is not None:
            param_count += 1
            updates.append(f"description = ${param_count}")
            params.append(description)
            
        if idx is not None:
            param_count += 1
            updates.append(f"idx = ${param_count}")
            params.append(idx)
            
        if custom_id is not None:
            param_count += 1
            updates.append(f"custom_id = ${param_count}")
            params.append(custom_id)
            
        if not updates:
            return None
            
        param_count += 1
        updates.append(f"updated_at = NOW()")
        params.append(collection_uuid)
        
        query = f"""
            UPDATE langchain_pg_collection 
            SET {', '.join(updates)}
            WHERE uuid = ${param_count}
            RETURNING uuid, idx, custom_id, name, description, created_at, updated_at
        """
        result = await conn.fetchrow(query, *params)
        if result:
            result_dict = dict(result)
            result_dict['uuid'] = str(result_dict['uuid'])
            return result_dict
        return None


async def update_collection_by_idx(idx: str, name: str = None, description: str = None):
    """Update collection by idx (user identifier) - DEPRECATED, use update_collection."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        # First get the UUID for the collection
        collection = await get_collection_by_idx(idx)
        if not collection:
            return None
        return await update_collection(collection['uuid'], name=name, description=description)


async def delete_collection(collection_uuid: str):
    """Delete collection by UUID primary key - cascades to documents and embeddings."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("""
            DELETE FROM langchain_pg_collection WHERE uuid = $1
        """, collection_uuid)
        return result


async def delete_collection_by_idx(idx: str):
    """Delete collection by idx (user identifier) - DEPRECATED, use delete_collection."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("""
            DELETE FROM langchain_pg_collection WHERE idx = $1
        """, idx)
        return result


# Document Operations - Using Proper PK/FK Relationships
async def create_document(
    collection_uuid: str, filename: str = None, idx: str = None, custom_id: str = None,
    content: str = None, page_content: str = None, mimetype: str = None,
    description: str = None, save_pdf_path: bool = False, 
    auto_embed: bool = True, **kwargs
):
    """Create a new document using UUID foreign key to collection."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        # Auto-detect binary_hash if content provided
        binary_hash = None
        if content:
            binary_hash = hashlib.md5(content.encode()).hexdigest()
            
        page_number = kwargs.get('page_number')
        pdf_path = kwargs.get('pdf_path') if save_pdf_path else None
        keywords = kwargs.get('keywords')
        metadata = kwargs.get('metadata', {})
        file_id = kwargs.get('file_id')  # Legacy compatibility
        user_id = kwargs.get('user_id')  # Legacy compatibility
        
        result = await conn.fetchrow("""
            INSERT INTO documents (
                idx, custom_id, collection_id, filename, content, page_content, mimetype, 
                binary_hash, description, page_number, pdf_path, keywords,
                metadata, file_id, user_id, created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, NOW(), NOW())
            RETURNING uuid, idx, custom_id, collection_id, filename, content, page_content, 
                     mimetype, binary_hash, description, page_number, pdf_path, 
                     keywords, metadata, file_id, user_id, created_at, updated_at
        """, idx, custom_id, collection_uuid, filename, content, page_content, mimetype, 
             binary_hash, description, page_number, pdf_path, keywords,
             metadata, file_id, user_id)
        return dict(result)


async def get_document_by_uuid(document_uuid: str):
    """Get document by its UUID primary key (uuid)."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        try:
            result = await conn.fetchrow("""
                SELECT d.*, c.name as collection_name
                FROM documents d
                LEFT JOIN langchain_pg_collection c ON d.collection_id = c.uuid
                WHERE d.uuid = $1
            """, document_uuid)
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Failed to get document by UUID | Error: {str(e)}")
            return None


async def get_document_by_idx(idx: str):
    """Get document by its idx (user-defined identifier) - DEPRECATED."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        result = await conn.fetchrow("""
            SELECT d.*, c.name as collection_name
            FROM documents d
            LEFT JOIN langchain_pg_collection c ON d.collection_id = c.uuid
            WHERE d.idx = $1
        """, idx)
        return dict(result) if result else None


async def get_document_by_custom_id(custom_id: str):
    """Get document by its custom_id (user-defined identifier)."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        result = await conn.fetchrow("""
            SELECT d.*, c.name as collection_name
            FROM documents d
            LEFT JOIN langchain_pg_collection c ON d.collection_id = c.uuid
            WHERE d.custom_id = $1
        """, custom_id)
        return dict(result) if result else None

async def update_document(document_uuid: str, **kwargs):
    """Update document by UUID primary key (uuid)."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        updates = []
        params = []
        param_count = 0
        
        # Handle binary_hash update if content is provided
        if 'content' in kwargs and kwargs['content'] is not None:
            kwargs['binary_hash'] = hashlib.md5(kwargs['content'].encode()).hexdigest()
        
        for field in ['idx', 'custom_id', 'filename', 'content', 'page_content', 'mimetype', 'binary_hash', 
                     'description', 'page_number', 'pdf_path', 'keywords', 'metadata']:
            if field in kwargs and kwargs[field] is not None:
                param_count += 1
                updates.append(f"{field} = ${param_count}")
                params.append(kwargs[field])
        
        if not updates:
            return None
            
        param_count += 1
        updates.append(f"updated_at = NOW()")
        params.append(document_uuid)
        
        query = f"""
            UPDATE documents 
            SET {', '.join(updates)}
            WHERE uuid = ${param_count}
            RETURNING uuid, idx, custom_id, collection_id, filename, content, page_content,
                     mimetype, binary_hash, description, page_number, pdf_path, 
                     keywords, metadata, file_id, user_id, created_at, updated_at
        """
        result = await conn.fetchrow(query, *params)
        return dict(result) if result else None


async def update_document_by_idx(idx: str, **kwargs):
    """Update document by idx (user identifier) - DEPRECATED, use update_document."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        # First get the document UUID
        document = await get_document_by_idx(idx)
        if not document:
            return None
        return await update_document(document['uuid'], **kwargs)


async def delete_document(document_uuid: str):
    """Delete document by UUID primary key (uuid) and cascade to embeddings."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("""
            DELETE FROM documents WHERE uuid = $1
        """, document_uuid)
        return result


async def delete_document_by_idx(idx: str):
    """Delete document by idx (user identifier) - DEPRECATED, use delete_document."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("""
            DELETE FROM documents WHERE idx = $1
        """, idx)
        return result


async def get_documents_by_collection(collection_uuid: str, limit: int = 10, offset: int = 0):
    """Get documents by collection UUID with pagination."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        # Get documents
        results = await conn.fetch("""
            SELECT d.*, c.name as collection_name
            FROM documents d
            LEFT JOIN langchain_pg_collection c ON d.collection_id = c.uuid
            WHERE d.collection_id = $1
            ORDER BY d.created_at DESC
            LIMIT $2 OFFSET $3
        """, collection_uuid, limit, offset)
        
        # Get total count
        total = await conn.fetchval("""
            SELECT COUNT(*) FROM documents WHERE collection_id = $1
        """, collection_uuid)
        
        return [dict(row) for row in results], total


async def get_documents_by_collection_idx(collection_idx: str, limit: int = 10, offset: int = 0):
    """Get documents by collection idx (user identifier) - DEPRECATED."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        # Get collection first
        collection = await conn.fetchrow("""
            SELECT uuid FROM langchain_pg_collection WHERE idx = $1
        """, collection_idx)
        
        if not collection:
            return None, 0
            
        return await get_documents_by_collection(collection['uuid'], limit, offset)


async def get_all_documents(limit: int = 10, offset: int = 0, user_id: str = None, file_id: str = None):
    """Get all documents with pagination and optional filtering."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        # Build filter conditions
        where_conditions = []
        params = [limit, offset]
        param_count = 2
        
        if user_id:
            param_count += 1
            where_conditions.append(f"d.user_id = ${param_count}")
            params.append(user_id)
            
        if file_id:
            param_count += 1
            where_conditions.append(f"d.file_id = ${param_count}")
            params.append(file_id)
        
        where_clause = " AND ".join(where_conditions)
        if where_clause:
            where_clause = f"WHERE {where_clause}"
        
        # Get documents
        results = await conn.fetch(f"""
            SELECT d.*, c.name as collection_name
            FROM documents d
            LEFT JOIN langchain_pg_collection c ON d.collection_id = c.uuid
            {where_clause}
            ORDER BY d.created_at DESC
            LIMIT $1 OFFSET $2
        """, *params)
        
        # Get total count with same filters
        count_params = params[2:]  # Remove limit and offset
        count_results = await conn.fetchval(f"""
            SELECT COUNT(*) FROM documents d {where_clause}
        """, *count_params)
        
        documents = [dict(row) for row in results]
        return documents, count_results


# Embedding Operations - Using Proper PK/FK Relationships
async def create_embedding(
    custom_id: str, embedding: list, document_text: str, 
    document_uuid: str = None, collection_uuid: str = None, metadata: dict = None
):
    """Create an embedding using UUID foreign keys."""
    import json
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        # Convert embedding list to string format for PostgreSQL vector type
        embedding_str = str(embedding)
        
        # Convert metadata dict to JSON string for PostgreSQL JSONB
        metadata_json = json.dumps(metadata or {})
        
        result = await conn.fetchrow("""
            INSERT INTO langchain_pg_embedding (
                uuid, custom_id, embedding, document, document_id, collection_id, cmetadata, created_at
            )
            VALUES (gen_random_uuid(), $1, $2::vector, $3, $4, $5, $6::jsonb, NOW())
            RETURNING uuid, custom_id, embedding, document, document_id, collection_id, cmetadata, created_at
        """, custom_id, embedding_str, document_text, document_uuid, collection_uuid, metadata_json)
        return dict(result)


async def get_embedding_by_id(embedding_id: str):
    """Get embedding by custom_id."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        result = await conn.fetchrow("""
            SELECT uuid, custom_id, embedding, document, document_id, collection_id, cmetadata, created_at
            FROM langchain_pg_embedding 
            WHERE custom_id = $1
        """, embedding_id)
        return dict(result) if result else None


async def get_embeddings_by_document(document_uuid: str):
    """Get all embeddings for a specific document UUID."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        results = await conn.fetch("""
            SELECT uuid, custom_id, embedding, document, document_id, collection_id, cmetadata, created_at
            FROM langchain_pg_embedding 
            WHERE document_id = $1
            ORDER BY created_at ASC
        """, document_uuid)
        return [dict(row) for row in results]


async def get_all_embeddings(limit: int = 100, offset: int = 0):
    """Get all embeddings with pagination."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        # Get total count
        total_count = await conn.fetchval("""
            SELECT COUNT(*) FROM langchain_pg_embedding
        """)
        
        # Get paginated results
        results = await conn.fetch("""
            SELECT uuid, custom_id, embedding, document, document_id, collection_id, cmetadata, created_at
            FROM langchain_pg_embedding 
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
        """, limit, offset)
        
        return [dict(row) for row in results], total_count


async def delete_embedding(embedding_id: str):
    """Delete embedding by custom_id."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("""
            DELETE FROM langchain_pg_embedding WHERE custom_id = $1
        """, embedding_id)
        return result

from typing import List, Dict, Optional
import asyncpg
import json
import numpy as np
from functools import lru_cache
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import List, Dict, Optional
import asyncpg
import json
import numpy as np
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def similarity_search_embeddings(
    query_embedding: List[float],
    k: int = 4,
    filter_metadata: Optional[Dict] = None,
    collection_uuid: Optional[str] = None,
    document_uuid: Optional[str] = None,
    use_hybrid_search: bool = False,
    keyword_query: Optional[str] = None,
    distance_threshold: Optional[float] = None,
    batch_size: int = 1000,
) -> List[Dict]:
    """
    Perform an optimized similarity search on embeddings with optional hybrid search.
    
    Args:
        query_embedding: Embedding vector for the query
        k: Number of results to return (default: 4)
        filter_metadata: Metadata filter as a dictionary
        collection_uuid: UUID of the collection to filter by
        document_uuid: UUID of the document to filter by
        use_hybrid_search: Enable hybrid search with keywords
        keyword_query: Keyword query for hybrid search
        distance_threshold: Maximum distance for results
        batch_size: Batch size for processing large datasets
    
    Returns:
        List of dictionaries containing search results with metadata and distances
    
    Raises:
        ValueError: If input parameters are invalid
        asyncpg.exceptions.PostgresError: If database query fails
    """
    try:
        # Input validation
        if not query_embedding or not isinstance(query_embedding, list):
            raise ValueError("query_embedding must be a non-empty list of floats")
        if k < 1:
            raise ValueError("k must be a positive integer")
        if use_hybrid_search and not keyword_query:
            raise ValueError("keyword_query is required when use_hybrid_search is True")

        # Normalize query embedding
        query_embedding = np.array(query_embedding, dtype=np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding_str = str(query_embedding.tolist())

        # Get database connection pool
        pool = await PSQLDatabase.get_pool()
        
        async with pool.acquire() as conn:
            # Build WHERE conditions
            where_conditions = []
            params = [query_embedding_str]
            param_count = 1

            if collection_uuid:
                param_count += 1
                where_conditions.append(f"collection_id = ${param_count}")
                params.append(collection_uuid)
                
            if document_uuid:
                param_count += 1
                where_conditions.append(f"document_id = ${param_count}")
                params.append(document_uuid)
                
            if filter_metadata:
                param_count += 1
                where_conditions.append(f"cmetadata @> ${param_count}::jsonb")
                params.append(json.dumps(filter_metadata))
                
            if distance_threshold is not None:
                param_count += 1
                where_conditions.append(f"(embedding <=> ${param_count}::vector) < ${param_count}")
                params.append(distance_threshold)
            
            # Add full-text search for hybrid search
            if use_hybrid_search and keyword_query:
                param_count += 1
                where_conditions.append(f"to_tsvector('english', document) @@ to_tsquery(${param_count})")
                params.append(keyword_query)

            where_clause = " AND ".join(where_conditions) if where_conditions else ""
            if where_clause:
                where_clause = f"WHERE {where_clause}"

            # Base query
            query = f"""
                SELECT 
                    uuid, 
                    custom_id, 
                    document, 
                    document_id, 
                    collection_id, 
                    cmetadata,
                    (embedding <=> $1::vector) as distance
                    {', ts_rank(to_tsvector(''english'', document), to_tsquery(${param_count})) as keyword_rank'
                     if use_hybrid_search else ''}
                FROM langchain_pg_embedding
                {where_clause}
                ORDER BY 
                    {'0.7 * (embedding <=> $1::vector) + 0.3 * keyword_rank'
                     if use_hybrid_search else '(embedding <=> $1::vector)'}
            """

            # Execute query with batching
            results = []
            offset = 0
            limit_param = k if not batch_size else min(batch_size, k)
            
            while len(results) < k:
                # Append LIMIT and OFFSET to query
                batch_query = f"{query} LIMIT ${param_count + 1} OFFSET ${param_count + 2}"
                params_batch = params + [limit_param, offset]
                
                batch_results = await conn.fetch(batch_query, *params_batch)
                results.extend([dict(row) for row in batch_results])
                
                if len(batch_results) < limit_param:
                    break
                offset += limit_param

            # Log query performance
            logger.info(f"Similarity search completed. Results: {len(results)}, Time: {datetime.now()}")
            
            return results[:k]

    except asyncpg.exceptions.PostgresSyntaxError as e:
        logger.error(f"Query syntax error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in similarity search: {str(e)}")
        raise


# Document Blocks functionality
async def ensure_document_blocks_schema():
    """Ensure the document blocks table schema exists."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS rag_document_blocks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                block_idx INTEGER NOT NULL,
                document_id UUID REFERENCES documents(uuid) ON DELETE CASCADE,
                name TEXT NOT NULL,
                content TEXT,
                level INTEGER NOT NULL,
                page_idx INTEGER NOT NULL,
                tag TEXT NOT NULL,
                block_class TEXT,
                x0 FLOAT,
                y0 FLOAT,
                x1 FLOAT,
                y1 FLOAT,
                parent_idx INTEGER,
                content_type TEXT DEFAULT 'regular',
                section_type TEXT,
                demand_priority INTEGER,
                content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', COALESCE(content, ''))) STORED,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(document_id, block_idx)
            );
        """)
        
        # Create indexes
        indexes_to_create = [
            "CREATE INDEX IF NOT EXISTS idx_document_blocks_content_tsv ON rag_document_blocks USING gin(content_tsv);",
            "CREATE INDEX IF NOT EXISTS idx_rag_blocks_content_type ON rag_document_blocks(content_type);",
            "CREATE INDEX IF NOT EXISTS idx_rag_blocks_section_type ON rag_document_blocks(section_type);",
            "CREATE INDEX IF NOT EXISTS idx_rag_blocks_demand_priority ON rag_document_blocks(demand_priority);",
            "CREATE INDEX IF NOT EXISTS idx_rag_blocks_document_id ON rag_document_blocks(document_id);",
            "CREATE INDEX IF NOT EXISTS idx_rag_blocks_block_idx ON rag_document_blocks(block_idx);",
            "CREATE INDEX IF NOT EXISTS idx_rag_blocks_parent_idx ON rag_document_blocks(parent_idx);",
            "CREATE INDEX IF NOT EXISTS idx_rag_blocks_page_idx ON rag_document_blocks(page_idx);",
        ]
        
        for index_sql in indexes_to_create:
            try:
                await conn.execute(index_sql)
            except Exception as e:
                logger.warning(f"Could not create index: {e}")
        
        logger.info("Document blocks schema ensured")


async def create_document_block(
    document_id: str,
    block_idx: int,
    name: str,
    content: str = None,
    level: int = 0,
    page_idx: int = 0,
    tag: str = "para",
    block_class: str = None,
    x0: float = None,
    y0: float = None,
    x1: float = None,
    y1: float = None,
    parent_idx: int = None,
    content_type: str = "regular",
    section_type: str = None,
    demand_priority: int = None
) -> dict:
    """Create a new document block."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        try:
            # First ensure the document exists
            doc_exists = await conn.fetchval(
                "SELECT uuid FROM documents WHERE uuid = $1", 
                uuid.UUID(document_id)
            )
            if not doc_exists:
                raise ValueError(f"Document with ID {document_id} not found")
            
            result = await conn.fetchrow("""
                INSERT INTO rag_document_blocks 
                (document_id, block_idx, name, content, level, page_idx, tag, block_class,
                 x0, y0, x1, y1, parent_idx, content_type, section_type, demand_priority)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                RETURNING *
            """, 
            uuid.UUID(document_id), block_idx, name, content, level, page_idx, tag, block_class,
            x0, y0, x1, y1, parent_idx, content_type, section_type, demand_priority)
            
            if result:
                return dict(result)
            return None
            
        except Exception as e:
            logger.error(f"Error creating document block: {str(e)}")
            raise


async def create_document_blocks_bulk(document_id: str, blocks: list) -> list:
    """Create multiple document blocks in a single transaction."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        try:
            # First ensure the document exists
            doc_exists = await conn.fetchval(
                "SELECT uuid FROM documents WHERE uuid = $1", 
                uuid.UUID(document_id)
            )
            if not doc_exists:
                raise ValueError(f"Document with ID {document_id} not found")
            
            # Prepare bulk insert data
            bulk_data = []
            for block in blocks:
                bulk_data.append((
                    uuid.UUID(document_id),
                    block.get("block_idx"),
                    block.get("name"),
                    block.get("content"),
                    block.get("level", 0),
                    block.get("page_idx", 0),
                    block.get("tag", "para"),
                    block.get("block_class"),
                    block.get("x0"),
                    block.get("y0"),
                    block.get("x1"),
                    block.get("y1"),
                    block.get("parent_idx"),
                    block.get("content_type", "regular"),
                    block.get("section_type"),
                    block.get("demand_priority")
                ))
            
            # Use executemany for bulk insert
            await conn.executemany("""
                INSERT INTO rag_document_blocks 
                (document_id, block_idx, name, content, level, page_idx, tag, block_class,
                 x0, y0, x1, y1, parent_idx, content_type, section_type, demand_priority)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                ON CONFLICT (document_id, block_idx) DO UPDATE SET
                name = EXCLUDED.name,
                content = EXCLUDED.content,
                level = EXCLUDED.level,
                page_idx = EXCLUDED.page_idx,
                tag = EXCLUDED.tag,
                block_class = EXCLUDED.block_class,
                x0 = EXCLUDED.x0,
                y0 = EXCLUDED.y0,
                x1 = EXCLUDED.x1,
                y1 = EXCLUDED.y1,
                parent_idx = EXCLUDED.parent_idx,
                content_type = EXCLUDED.content_type,
                section_type = EXCLUDED.section_type,
                demand_priority = EXCLUDED.demand_priority
            """, bulk_data)
            
            # Return the created blocks
            created_blocks = await conn.fetch("""
                SELECT * FROM rag_document_blocks 
                WHERE document_id = $1
                ORDER BY block_idx
            """, uuid.UUID(document_id))
            
            return [dict(block) for block in created_blocks]
            
        except Exception as e:
            logger.error(f"Error creating document blocks bulk: {str(e)}")
            raise


async def get_document_blocks_by_document(
    document_id: str, 
    limit: int = 100, 
    offset: int = 0
) -> tuple:
    """Get all blocks for a document with pagination."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        try:
            # Get total count
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM rag_document_blocks WHERE document_id = $1",
                uuid.UUID(document_id)
            )
            
            # Get blocks with pagination
            blocks = await conn.fetch("""
                SELECT * FROM rag_document_blocks 
                WHERE document_id = $1
                ORDER BY block_idx
                LIMIT $2 OFFSET $3
            """, uuid.UUID(document_id), limit, offset)
            
            return [dict(block) for block in blocks], total
            
        except Exception as e:
            logger.error(f"Error getting document blocks: {str(e)}")
            raise


async def get_document_block_by_id(block_id: str) -> dict:
    """Get a document block by its ID."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        try:
            result = await conn.fetchrow(
                "SELECT * FROM rag_document_blocks WHERE id = $1",
                uuid.UUID(block_id)
            )
            return dict(result) if result else None
            
        except Exception as e:
            logger.error(f"Error getting document block by ID: {str(e)}")
            raise


async def update_document_block(block_id: str, **updates) -> dict:
    """Update a document block."""
    if not updates:
        raise ValueError("No updates provided")
        
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        try:
            # Build dynamic update query
            set_clauses = []
            params = []
            param_idx = 1
            
            for field, value in updates.items():
                if field in ['name', 'content', 'level', 'tag', 'block_class', 
                           'content_type', 'section_type', 'demand_priority']:
                    set_clauses.append(f"{field} = ${param_idx}")
                    params.append(value)
                    param_idx += 1
            
            if not set_clauses:
                raise ValueError("No valid fields to update")
            
            params.append(uuid.UUID(block_id))
            
            query = f"""
                UPDATE rag_document_blocks 
                SET {', '.join(set_clauses)}
                WHERE id = ${param_idx}
                RETURNING *
            """
            
            result = await conn.fetchrow(query, *params)
            return dict(result) if result else None
            
        except Exception as e:
            logger.error(f"Error updating document block: {str(e)}")
            raise


async def delete_document_block(block_id: str) -> bool:
    """Delete a document block."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        try:
            result = await conn.execute(
                "DELETE FROM rag_document_blocks WHERE id = $1",
                uuid.UUID(block_id)
            )
            return result == "DELETE 1"
            
        except Exception as e:
            logger.error(f"Error deleting document block: {str(e)}")
            raise


async def delete_document_blocks_by_document(document_id: str) -> int:
    """Delete all blocks for a document."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        try:
            result = await conn.execute(
                "DELETE FROM rag_document_blocks WHERE document_id = $1",
                uuid.UUID(document_id)
            )
            # Extract number from result string "DELETE X"
            return int(result.split()[-1]) if result.split()[-1].isdigit() else 0
            
        except Exception as e:
            logger.error(f"Error deleting document blocks: {str(e)}")
            raise


async def search_document_blocks(
    document_id: str = None,
    query: str = None,
    content_type: str = None,
    section_type: str = None,
    limit: int = 20,
    offset: int = 0
) -> tuple:
    """Search document blocks with various filters."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        try:
            where_clauses = []
            params = []
            param_idx = 1
            
            if document_id:
                where_clauses.append(f"document_id = ${param_idx}")
                params.append(uuid.UUID(document_id))
                param_idx += 1
            
            if query:
                where_clauses.append(f"content_tsv @@ to_tsquery('english', ${param_idx})")
                params.append(query)
                param_idx += 1
            
            if content_type:
                where_clauses.append(f"content_type = ${param_idx}")
                params.append(content_type)
                param_idx += 1
            
            if section_type:
                where_clauses.append(f"section_type = ${param_idx}")
                params.append(section_type)
                param_idx += 1
            
            where_clause = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
            
            # Count query
            count_query = f"SELECT COUNT(*) FROM rag_document_blocks{where_clause}"
            total = await conn.fetchval(count_query, *params)
            
            # Search query
            params.extend([limit, offset])
            search_query = f"""
                SELECT * FROM rag_document_blocks
                {where_clause}
                ORDER BY block_idx
                LIMIT ${param_idx} OFFSET ${param_idx + 1}
            """
            
            blocks = await conn.fetch(search_query, *params)
            
            return [dict(block) for block in blocks], total
            
        except Exception as e:
            logger.error(f"Error searching document blocks: {str(e)}")
            raise


async def pg_health_check() -> bool:
    """Check if PostgreSQL database is healthy."""
    try:
        pool = await PSQLDatabase.get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


# For backwards compatibility and easier imports  
database = {
    # New UUID-based methods (recommended)
    'create_collection': create_collection,
    'get_collection': get_collection_by_uuid,
    'get_collection_by_uuid': get_collection_by_uuid,
    'get_collection_by_custom_id': get_collection_by_custom_id,
    'update_collection': update_collection,
    'delete_collection': delete_collection,
    'create_document': create_document,
    'get_document': get_document_by_uuid,
    'get_document_by_uuid': get_document_by_uuid,
    'get_document_by_custom_id': get_document_by_custom_id,
    'update_document': update_document,
    'delete_document': delete_document,
    'get_documents_by_collection': get_documents_by_collection,
    'create_embedding': create_embedding,
    'similarity_search_embeddings': similarity_search_embeddings,
    
    # Document Blocks methods
    'create_document_block': create_document_block,
    'create_document_blocks_bulk': create_document_blocks_bulk,
    'get_document_blocks_by_document': get_document_blocks_by_document,
    'get_document_block_by_id': get_document_block_by_id,
    'update_document_block': update_document_block,
    'delete_document_block': delete_document_block,
    'delete_document_blocks_by_document': delete_document_blocks_by_document,
    'search_document_blocks': search_document_blocks,
    'ensure_document_blocks_schema': ensure_document_blocks_schema,
    
    # Legacy idx-based methods (deprecated)
    'get_collection_by_idx': get_collection_by_idx,
    'update_collection_by_idx': update_collection_by_idx,
    'delete_collection_by_idx': delete_collection_by_idx,
    'get_document_by_idx': get_document_by_idx,
    'get_document_by_uuid': get_document_by_uuid,
    'update_document_by_idx': update_document_by_idx,
    'delete_document_by_idx': delete_document_by_idx,
    'get_documents_by_collection_idx': get_documents_by_collection_idx,
    
    # Other operations
    'get_all_collections': get_all_collections,
    'get_all_documents': get_all_documents,
}
