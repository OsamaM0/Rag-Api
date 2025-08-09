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
                serial_id SERIAL UNIQUE,
                idx VARCHAR UNIQUE,
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
            ADD COLUMN IF NOT EXISTS serial_id SERIAL UNIQUE,
            ADD COLUMN IF NOT EXISTS idx VARCHAR UNIQUE,
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
                serial_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                idx VARCHAR UNIQUE,
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
                    FOREIGN KEY (document_id) REFERENCES documents(serial_id) ON DELETE CASCADE;
                """)
        except Exception as e:
            logger.warning(f"Could not add document foreign key constraint: {e}")
        
        # Create optimized indexes
        indexes_to_create = [
            # Collection indexes
            ("idx_collections_idx", "langchain_pg_collection", "idx"),
            ("idx_collections_name", "langchain_pg_collection", "name"),
            
            # Document indexes
            ("idx_documents_idx", "documents", "idx"),
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


# Collection Operations - Streamlined
async def create_collection(name: str, description: str = None, idx: str = None):
    """Create a new collection with the three-table schema."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        if not idx:
            idx = str(uuid.uuid4())
        
        # Generate UUID explicitly since the table might not have a default
        collection_uuid = str(uuid.uuid4())
        
        result = await conn.fetchrow("""
            INSERT INTO langchain_pg_collection (uuid, name, idx, description, created_at, updated_at)
            VALUES ($1, $2, $3, $4, NOW(), NOW())
            RETURNING uuid, idx, name, description, created_at, updated_at
        """, collection_uuid, name, idx, description)
        return dict(result)


async def get_collection_by_idx(idx: str):
    """Get collection by its idx."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        result = await conn.fetchrow("""
            SELECT uuid, idx, name, description, created_at, updated_at
            FROM langchain_pg_collection 
            WHERE idx = $1
        """, idx)
        return dict(result) if result else None


async def get_all_collections(limit: int = 10, offset: int = 0):
    """Get all collections with pagination."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        results = await conn.fetch("""
            SELECT uuid, idx, name, description, created_at, updated_at
            FROM langchain_pg_collection
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
        """, limit, offset)
        
        total = await conn.fetchval("SELECT COUNT(*) FROM langchain_pg_collection")
        
        return [dict(row) for row in results], total


async def update_collection(idx: str, name: str = None, description: str = None):
    """Update collection by idx."""
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
            
        if not updates:
            return None
            
        param_count += 1
        updates.append(f"updated_at = NOW()")
        params.append(idx)
        
        query = f"""
            UPDATE langchain_pg_collection 
            SET {', '.join(updates)}
            WHERE idx = ${param_count}
            RETURNING uuid, idx, name, description, created_at, updated_at
        """
        result = await conn.fetchrow(query, *params)
        return dict(result) if result else None


async def delete_collection(idx: str):
    """Delete collection by idx - cascades to documents and embeddings."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("""
            DELETE FROM langchain_pg_collection WHERE idx = $1
        """, idx)
        return result


# Document Operations - Unified for full documents and vector chunks
async def create_document(
    idx: str, collection_id: str, filename: str = None, 
    content: str = None, page_content: str = None, mimetype: str = None,
    description: str = None, save_pdf_path: bool = False, 
    auto_embed: bool = True, **kwargs
):
    """Create a new document in the unified documents table."""
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
                idx, collection_id, filename, content, page_content, mimetype, 
                binary_hash, description, page_number, pdf_path, keywords,
                metadata, file_id, user_id, created_at, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, NOW(), NOW())
            RETURNING serial_id, idx, collection_id, filename, content, page_content, 
                     mimetype, binary_hash, description, page_number, pdf_path, 
                     keywords, metadata, file_id, user_id, created_at, updated_at
        """, idx, collection_id, filename, content, page_content, mimetype, 
             binary_hash, description, page_number, pdf_path, keywords,
             metadata, file_id, user_id)
        return dict(result)


async def get_document_by_idx(idx: str):
    """Get document by its idx."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        result = await conn.fetchrow("""
            SELECT d.*, c.name as collection_name
            FROM documents d
            LEFT JOIN langchain_pg_collection c ON d.collection_id = c.uuid
            WHERE d.idx = $1
        """, idx)
        return dict(result) if result else None


async def get_document_by_serial_id(serial_id: str):
    """Get document by its serial_id (UUID)."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        result = await conn.fetchrow("""
            SELECT d.*, c.name as collection_name
            FROM documents d
            LEFT JOIN langchain_pg_collection c ON d.collection_id = c.uuid
            WHERE d.serial_id = $1
        """, serial_id)
        return dict(result) if result else None


async def update_document(idx: str, **kwargs):
    """Update document by idx."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        updates = []
        params = []
        param_count = 0
        
        # Handle binary_hash update if content is provided
        if 'content' in kwargs and kwargs['content'] is not None:
            kwargs['binary_hash'] = hashlib.md5(kwargs['content'].encode()).hexdigest()
        
        for field in ['filename', 'content', 'page_content', 'mimetype', 'binary_hash', 
                     'description', 'page_number', 'pdf_path', 'keywords', 'metadata']:
            if field in kwargs and kwargs[field] is not None:
                param_count += 1
                updates.append(f"{field} = ${param_count}")
                params.append(kwargs[field])
        
        if not updates:
            return None
            
        param_count += 1
        updates.append(f"updated_at = NOW()")
        params.append(idx)
        
        query = f"""
            UPDATE documents 
            SET {', '.join(updates)}
            WHERE idx = ${param_count}
            RETURNING serial_id, idx, collection_id, filename, content, page_content,
                     mimetype, binary_hash, description, page_number, pdf_path, 
                     keywords, metadata, file_id, user_id, created_at, updated_at
        """
        result = await conn.fetchrow(query, *params)
        return dict(result) if result else None


async def delete_document(idx: str):
    """Delete document by idx and cascade to embeddings."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("""
            DELETE FROM documents WHERE idx = $1
        """, idx)
        return result


async def get_documents_by_collection(collection_idx: str, limit: int = 10, offset: int = 0):
    """Get documents by collection idx with pagination."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        # Get collection first
        collection = await conn.fetchrow("""
            SELECT uuid FROM langchain_pg_collection WHERE idx = $1
        """, collection_idx)
        
        if not collection:
            return None, 0
            
        # Get documents
        results = await conn.fetch("""
            SELECT d.*, c.name as collection_name
            FROM documents d
            LEFT JOIN langchain_pg_collection c ON d.collection_id = c.uuid
            WHERE d.collection_id = $1
            ORDER BY d.created_at DESC
            LIMIT $2 OFFSET $3
        """, collection['uuid'], limit, offset)
        
        # Get total count
        total = await conn.fetchval("""
            SELECT COUNT(*) FROM documents WHERE collection_id = $1
        """, collection['uuid'])
        
        return [dict(row) for row in results], total


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


# Embedding Operations - Direct PG Vector table operations
async def create_embedding(
    custom_id: str, embedding: list, document_text: str, 
    document_id: str = None, collection_id: str = None, metadata: dict = None
):
    """Create an embedding directly in the langchain_pg_embedding table."""
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
        """, custom_id, embedding_str, document_text, document_id, collection_id, metadata_json)
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


async def get_embeddings_by_document(document_id: str):
    """Get all embeddings for a specific document."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        results = await conn.fetch("""
            SELECT uuid, custom_id, embedding, document, document_id, collection_id, cmetadata, created_at
            FROM langchain_pg_embedding 
            WHERE document_id = $1
            ORDER BY created_at ASC
        """, document_id)
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


async def similarity_search_embeddings(
    query_embedding: list, k: int = 4, filter_metadata: dict = None,
    collection_id: str = None, document_id: str = None
):
    """Perform similarity search on embeddings with optional filters."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        # Convert embedding list to string format for PostgreSQL vector type
        query_embedding_str = str(query_embedding)
        
        # Build filter conditions
        where_conditions = []
        params = [query_embedding_str, k]
        param_count = 2
        
        if collection_id:
            param_count += 1
            where_conditions.append(f"collection_id = ${param_count}")
            params.append(collection_id)
            
        if document_id:
            param_count += 1
            where_conditions.append(f"document_id = ${param_count}")
            params.append(document_id)
            
        if filter_metadata:
            param_count += 1
            where_conditions.append(f"cmetadata @> ${param_count}::jsonb")
            params.append(filter_metadata)
        
        where_clause = " AND ".join(where_conditions)
        if where_clause:
            where_clause = f"WHERE {where_clause}"
        
        results = await conn.fetch(f"""
            SELECT 
                uuid, custom_id, document, document_id, collection_id, cmetadata,
                embedding <-> $1::vector as distance
            FROM langchain_pg_embedding
            {where_clause}
            ORDER BY embedding <-> $1::vector
            LIMIT $2
        """, *params)
        
        return [dict(row) for row in results]


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
    'create_collection': create_collection,
    'get_collection': get_collection_by_idx,
    'get_all_collections': get_all_collections,
    'update_collection': update_collection,
    'delete_collection': delete_collection,
    'create_document': create_document,
    'get_document': get_document_by_idx,
    'get_all_documents': get_all_documents,
    'get_documents_by_collection': get_documents_by_collection,
    'update_document': update_document,
    'delete_document': delete_document,
    'create_embedding': create_embedding,
    'similarity_search_embeddings': similarity_search_embeddings,
}
