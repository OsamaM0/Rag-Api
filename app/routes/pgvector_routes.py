# app/routes/pgvector_routes.py
import traceback
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Request, Query
from app.config import logger
from app.services.database import PSQLDatabase
from app.models import SuccessResponse

router = APIRouter(prefix="/pgvector", tags=["Database"])


async def check_index_exists(table_name: str, column_name: str) -> bool:
    """Check if an index exists on a specific column."""
    pool = await PSQLDatabase.get_pool()
    async with pool.acquire() as conn:
        result = await conn.fetch(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_indexes
                WHERE tablename = $1 
                AND indexdef LIKE '%' || $2 || '%'
            );
            """,
            table_name,
            column_name,
        )
    return result[0]['exists']


@router.get("/test/check_index")
async def check_file_id_index(
    table_name: str, 
    column_name: str,
    request: Request
):
    """Check if an index exists on a specific column."""
    try:
        if await check_index_exists(table_name, column_name):
            return {"message": f"Index on {column_name} exists in the table {table_name}."}
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"No index on {column_name} found in the table {table_name}."
            )
    except Exception as e:
        logger.error(f"Failed to check index | Table: {table_name}, Column: {column_name} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to check index: {str(e)}")


@router.get("/records/all")
async def get_all_records(table_name: str, request: Request):
    """Get all records from a specific table."""
    # Validate that the table name is one of the expected ones to prevent SQL injection
    allowed_tables = ["langchain_pg_collection", "langchain_pg_embedding"]
    if table_name not in allowed_tables:
        raise HTTPException(status_code=400, detail=f"Invalid table name. Allowed tables: {allowed_tables}")

    try:
        pool = await PSQLDatabase.get_pool()
        async with pool.acquire() as conn:
            # Use parameterized queries to prevent SQL injection
            records = await conn.fetch(f"SELECT * FROM {table_name};")

        # Convert records to JSON serializable format
        records_json = [dict(record) for record in records]

        return {
            "table_name": table_name,
            "record_count": len(records_json),
            "records": records_json
        }
        
    except Exception as e:
        logger.error(f"Failed to get all records | Table: {table_name} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get records: {str(e)}")


@router.get("/records")
async def get_records_filtered_by_custom_id(
    custom_id: str, 
    table_name: str = Query("langchain_pg_embedding", description="Table name"),
    request: Request = None
):
    """Get records filtered by custom ID."""
    # Validate that the table name is one of the expected ones to prevent SQL injection
    allowed_tables = ["langchain_pg_collection", "langchain_pg_embedding"]
    if table_name not in allowed_tables:
        raise HTTPException(status_code=400, detail=f"Invalid table name. Allowed tables: {allowed_tables}")

    try:
        pool = await PSQLDatabase.get_pool()
        async with pool.acquire() as conn:
            # Use parameterized queries to prevent SQL injection
            query = f"SELECT * FROM {table_name} WHERE custom_id=$1;"
            records = await conn.fetch(query, custom_id)

        # Convert records to JSON serializable format
        records_json = [dict(record) for record in records]

        return {
            "table_name": table_name,
            "custom_id": custom_id,
            "record_count": len(records_json),
            "records": records_json
        }
        
    except Exception as e:
        logger.error(f"Failed to get filtered records | Table: {table_name}, Custom ID: {custom_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get filtered records: {str(e)}")


@router.delete("/records/{custom_id}", response_model=SuccessResponse)
async def delete_record_by_custom_id(
    custom_id: str,
    table_name: str = Query("langchain_pg_embedding", description="Table name"),
    request: Request = None
):
    """Delete records by custom ID."""
    allowed_tables = ["langchain_pg_collection", "langchain_pg_embedding"]
    if table_name not in allowed_tables:
        raise HTTPException(status_code=400, detail=f"Invalid table name. Allowed tables: {allowed_tables}")

    try:
        pool = await PSQLDatabase.get_pool()
        async with pool.acquire() as conn:
            # First check if record exists
            check_query = f"SELECT COUNT(*) as count FROM {table_name} WHERE custom_id=$1;"
            result = await conn.fetchrow(check_query, custom_id)
            
            if result['count'] == 0:
                raise HTTPException(status_code=404, detail=f"No records found with custom_id: {custom_id}")
            
            # Delete the record(s)
            delete_query = f"DELETE FROM {table_name} WHERE custom_id=$1;"
            deleted = await conn.execute(delete_query, custom_id)
            
            # Extract number of deleted rows from the result
            deleted_count = int(deleted.split()[-1])
            
        return SuccessResponse(
            message=f"Successfully deleted {deleted_count} record(s) with custom_id: {custom_id} from {table_name}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete record | Table: {table_name}, Custom ID: {custom_id} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to delete record: {str(e)}")


@router.post("/indexes/create", response_model=SuccessResponse)
async def create_index(
    table_name: str,
    column_name: str,
    index_name: Optional[str] = None,
    index_type: str = Query("btree", description="Index type (btree, hash, gin, gist, etc.)"),
    request: Request = None
):
    """Create an index on a specific column."""
    allowed_tables = ["langchain_pg_collection", "langchain_pg_embedding"]
    if table_name not in allowed_tables:
        raise HTTPException(status_code=400, detail=f"Invalid table name. Allowed tables: {allowed_tables}")

    try:
        # Generate index name if not provided
        if not index_name:
            index_name = f"idx_{table_name}_{column_name}"

        pool = await PSQLDatabase.get_pool()
        async with pool.acquire() as conn:
            # Check if index already exists
            check_query = """
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes 
                    WHERE indexname = $1
                );
            """
            exists = await conn.fetchrow(check_query, index_name)
            
            if exists['exists']:
                raise HTTPException(status_code=409, detail=f"Index {index_name} already exists")
            
            # Create the index
            create_query = f"CREATE INDEX {index_name} ON {table_name} USING {index_type} ({column_name});"
            await conn.execute(create_query)
            
        return SuccessResponse(
            message=f"Index {index_name} created successfully on {table_name}.{column_name}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create index | Table: {table_name}, Column: {column_name} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to create index: {str(e)}")


@router.delete("/indexes/{index_name}", response_model=SuccessResponse)
async def drop_index(index_name: str, request: Request):
    """Drop an index."""
    try:
        pool = await PSQLDatabase.get_pool()
        async with pool.acquire() as conn:
            # Check if index exists
            check_query = """
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes 
                    WHERE indexname = $1
                );
            """
            exists = await conn.fetchrow(check_query, index_name)
            
            if not exists['exists']:
                raise HTTPException(status_code=404, detail=f"Index {index_name} not found")
            
            # Drop the index
            drop_query = f"DROP INDEX {index_name};"
            await conn.execute(drop_query)
            
        return SuccessResponse(message=f"Index {index_name} dropped successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to drop index | Index: {index_name} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to drop index: {str(e)}")


@router.get("/vector/similarity-search")
async def vector_similarity_search(
    vector: List[float] = Query(..., description="Query vector values (repeat 'vector' param)"),
    table_name: str = Query("langchain_pg_embedding", description="Table name"),
    limit: int = Query(10, ge=1, le=100, description="Number of results"),
    threshold: Optional[float] = Query(None, description="Similarity threshold"),
    request: Request = None
):
    """Perform vector similarity search using pgvector."""
    allowed_tables = ["langchain_pg_embedding"]
    if table_name not in allowed_tables:
        raise HTTPException(status_code=400, detail=f"Invalid table name for vector search. Allowed: {allowed_tables}")

    try:
        pool = await PSQLDatabase.get_pool()
        async with pool.acquire() as conn:
            # Convert vector to pgvector format
            vector_str = f"[{','.join(map(str, vector))}]"
            
            if threshold is not None:
                query = f"""
                    SELECT custom_id, document, cmetadata, 
                           embedding <-> $1::vector as distance
                    FROM {table_name}
                    WHERE embedding <-> $1::vector < $2
                    ORDER BY embedding <-> $1::vector
                    LIMIT $3;
                """
                results = await conn.fetch(query, vector_str, threshold, limit)
            else:
                query = f"""
                    SELECT custom_id, document, cmetadata,
                           embedding <-> $1::vector as distance
                    FROM {table_name}
                    ORDER BY embedding <-> $1::vector
                    LIMIT $2;
                """
                results = await conn.fetch(query, vector_str, limit)

        # Convert results to JSON
        search_results = []
        for row in results:
            search_results.append({
                "custom_id": row['custom_id'],
                "document": row['document'],
                "metadata": row['cmetadata'],
                "distance": float(row['distance']),
                "similarity": 1 - float(row['distance'])  # Convert distance to similarity
            })

        return {
            "query_vector_dimension": len(vector),
            "results_count": len(search_results),
            "threshold": threshold,
            "results": search_results
        }
        
    except Exception as e:
        logger.error(f"Failed vector similarity search | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Vector similarity search failed: {str(e)}")


@router.get("/maintenance/stats")
async def get_maintenance_stats(request: Request):
    """Get database maintenance statistics."""
    try:
        pool = await PSQLDatabase.get_pool()
        async with pool.acquire() as conn:
            # Get table sizes
            size_query = """
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                    pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
            """
            table_sizes = await conn.fetch(size_query)
            
            # Get index usage stats
            index_query = """
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch
                FROM pg_stat_user_indexes
                ORDER BY idx_scan DESC;
            """
            index_stats = await conn.fetch(index_query)

        return {
            "table_sizes": [dict(row) for row in table_sizes],
            "index_usage": [dict(row) for row in index_stats],
            "timestamp": "2024-08-09T10:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Failed to get maintenance stats | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get maintenance stats: {str(e)}")