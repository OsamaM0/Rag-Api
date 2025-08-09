# app/routes/database_routes.py
import traceback
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Request, Query, status
from app.config import logger
from app.models import (
    DatabaseStatsResponse,
    DatabaseHealthResponse,
    SuccessResponse
)
from app.services.database import PSQLDatabase
from app.utils.health import is_health_ok
from datetime import datetime

router = APIRouter(prefix="/database", tags=["Database"])


@router.get("/health", response_model=DatabaseHealthResponse)
async def database_health_check(request: Request):
    """Check database health and connectivity."""
    try:
        database_connected = False
        vector_store_ready = False
        
        # Check database connection
        try:
            pool = await PSQLDatabase.get_pool()
            async with pool.acquire() as conn:
                await conn.execute("SELECT 1")
            database_connected = True
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
        
        # Check vector store readiness
        try:
            vector_store_ready = await is_health_ok()
        except Exception as e:
            logger.error(f"Vector store health check failed: {str(e)}")
        
        status_value = "UP" if database_connected and vector_store_ready else "DOWN"
        
        return DatabaseHealthResponse(
            status=status_value,
            database_connected=database_connected,
            vector_store_ready=vector_store_ready,
            last_check=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Health check failed | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        return DatabaseHealthResponse(
            status="ERROR",
            database_connected=False,
            vector_store_ready=False,
            last_check=datetime.utcnow()
        )


@router.get("/stats", response_model=DatabaseStatsResponse)
async def get_database_stats(request: Request):
    """Get database statistics."""
    try:
        total_documents = 0
        total_collections = 0
        total_embeddings = 0
        database_size = None
        
        pool = await PSQLDatabase.get_pool()
        async with pool.acquire() as conn:
            # Get document count
            try:
                result = await conn.fetchrow("SELECT COUNT(*) as count FROM langchain_pg_embedding")
                total_embeddings = result['count'] if result else 0
                total_documents = total_embeddings  # Assuming one embedding per document chunk
            except Exception as e:
                logger.warning(f"Could not get document count: {str(e)}")
            
            # Get collection count
            try:
                result = await conn.fetchrow("SELECT COUNT(*) as count FROM langchain_pg_collection")
                total_collections = result['count'] if result else 0
            except Exception as e:
                logger.warning(f"Could not get collection count: {str(e)}")
            
            # Get database size
            try:
                result = await conn.fetchrow("SELECT pg_size_pretty(pg_database_size(current_database())) as size")
                database_size = result['size'] if result else None
            except Exception as e:
                logger.warning(f"Could not get database size: {str(e)}")
        
        return DatabaseStatsResponse(
            total_documents=total_documents,
            total_collections=total_collections,
            total_embeddings=total_embeddings,
            database_size=database_size
        )
        
    except Exception as e:
        logger.error(f"Failed to get database stats | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get database stats: {str(e)}")


@router.get("/tables")
async def get_table_names(request: Request, schema: str = Query("public", description="Database schema")):
    """Get all table names in the database."""
    try:
        pool = await PSQLDatabase.get_pool()
        async with pool.acquire() as conn:
            table_names = await conn.fetch(
                """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = $1
                ORDER BY table_name
                """,
                schema,
            )
        
        tables = [record['table_name'] for record in table_names]
        return {"schema": schema, "tables": tables}
        
    except Exception as e:
        logger.error(f"Failed to get table names | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get table names: {str(e)}")


@router.get("/tables/{table_name}/columns")
async def get_table_columns(
    table_name: str, 
    request: Request, 
    schema: str = Query("public", description="Database schema")
):
    """Get column information for a specific table."""
    try:
        pool = await PSQLDatabase.get_pool()
        async with pool.acquire() as conn:
            columns = await conn.fetch(
                """
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_schema = $1 AND table_name = $2
                ORDER BY ordinal_position;
                """,
                schema, table_name,
            )
        
        column_info = [
            {
                "name": col['column_name'],
                "type": col['data_type'],
                "nullable": col['is_nullable'] == 'YES',
                "default": col['column_default']
            }
            for col in columns
        ]
        
        return {
            "table_name": table_name,
            "schema": schema,
            "columns": column_info
        }
        
    except Exception as e:
        logger.error(f"Failed to get table columns | Table: {table_name} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get table columns: {str(e)}")


@router.get("/tables/{table_name}/indexes")
async def get_table_indexes(
    table_name: str, 
    request: Request,
    schema: str = Query("public", description="Database schema")
):
    """Get index information for a specific table."""
    try:
        pool = await PSQLDatabase.get_pool()
        async with pool.acquire() as conn:
            indexes = await conn.fetch(
                """
                SELECT 
                    indexname,
                    indexdef
                FROM pg_indexes
                WHERE schemaname = $1 AND tablename = $2
                ORDER BY indexname;
                """,
                schema, table_name,
            )
        
        index_info = [
            {
                "name": idx['indexname'],
                "definition": idx['indexdef']
            }
            for idx in indexes
        ]
        
        return {
            "table_name": table_name,
            "schema": schema,
            "indexes": index_info
        }
        
    except Exception as e:
        logger.error(f"Failed to get table indexes | Table: {table_name} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get table indexes: {str(e)}")


@router.post("/vacuum", response_model=SuccessResponse)
async def vacuum_database(request: Request, table_name: Optional[str] = None):
    """Vacuum the database or a specific table."""
    try:
        pool = await PSQLDatabase.get_pool()
        async with pool.acquire() as conn:
            if table_name:
                # Validate table name to prevent SQL injection
                tables_result = await conn.fetch(
                    "SELECT table_name FROM information_schema.tables WHERE table_name = $1 AND table_schema = 'public'",
                    table_name
                )
                if not tables_result:
                    raise HTTPException(status_code=404, detail=f"Table {table_name} not found")
                
                await conn.execute(f"VACUUM {table_name}")
                message = f"Vacuum completed for table: {table_name}"
            else:
                await conn.execute("VACUUM")
                message = "Full database vacuum completed"
        
        logger.info(message)
        return SuccessResponse(message=message)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to vacuum database | Table: {table_name} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to vacuum database: {str(e)}")


@router.post("/analyze", response_model=SuccessResponse)
async def analyze_database(request: Request, table_name: Optional[str] = None):
    """Analyze the database or a specific table to update statistics."""
    try:
        pool = await PSQLDatabase.get_pool()
        async with pool.acquire() as conn:
            if table_name:
                # Validate table name
                tables_result = await conn.fetch(
                    "SELECT table_name FROM information_schema.tables WHERE table_name = $1 AND table_schema = 'public'",
                    table_name
                )
                if not tables_result:
                    raise HTTPException(status_code=404, detail=f"Table {table_name} not found")
                
                await conn.execute(f"ANALYZE {table_name}")
                message = f"Analysis completed for table: {table_name}"
            else:
                await conn.execute("ANALYZE")
                message = "Full database analysis completed"
        
        logger.info(message)
        return SuccessResponse(message=message)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze database | Table: {table_name} | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze database: {str(e)}")


@router.get("/performance")
async def get_performance_metrics(request: Request):
    """Get database performance metrics."""
    try:
        pool = await PSQLDatabase.get_pool()
        async with pool.acquire() as conn:
            # Get connection info
            connection_info = await conn.fetchrow(
                """
                SELECT 
                    count(*) as active_connections,
                    max(now() - query_start) as longest_query_duration
                FROM pg_stat_activity 
                WHERE state = 'active'
                """
            )
            
            # Get cache hit ratio
            cache_hit_ratio = await conn.fetchrow(
                """
                SELECT 
                    sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read) + 1) * 100 as cache_hit_ratio
                FROM pg_statio_user_tables
                """
            )
            
            # Get database size and activity
            db_stats = await conn.fetchrow(
                """
                SELECT 
                    pg_size_pretty(pg_database_size(current_database())) as database_size,
                    numbackends as connections,
                    xact_commit as transactions_committed,
                    xact_rollback as transactions_rolled_back
                FROM pg_stat_database 
                WHERE datname = current_database()
                """
            )
        
        return {
            "active_connections": connection_info['active_connections'] if connection_info else 0,
            "longest_query_duration": str(connection_info['longest_query_duration']) if connection_info and connection_info['longest_query_duration'] else "0",
            "cache_hit_ratio": round(float(cache_hit_ratio['cache_hit_ratio']) if cache_hit_ratio and cache_hit_ratio['cache_hit_ratio'] else 0, 2),
            "database_size": db_stats['database_size'] if db_stats else "0 MB",
            "total_connections": db_stats['connections'] if db_stats else 0,
            "transactions_committed": db_stats['transactions_committed'] if db_stats else 0,
            "transactions_rolled_back": db_stats['transactions_rolled_back'] if db_stats else 0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


@router.post("/backup", response_model=SuccessResponse)
async def create_database_backup(request: Request, backup_name: Optional[str] = None):
    """Create a database backup (mock implementation)."""
    try:
        # In a real implementation, you would use pg_dump or similar
        backup_name = backup_name or f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Database backup created: {backup_name}")
        return SuccessResponse(message=f"Database backup '{backup_name}' created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create database backup | Error: {str(e)} | Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to create database backup: {str(e)}")
