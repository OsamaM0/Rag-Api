"""
Vector Store Factory Module.

This module provides a factory function for creating different types of vector stores
based on the specified mode and configuration parameters.
"""

from typing import Optional, Union
from pymongo import MongoClient
from langchain_core.embeddings import Embeddings

from .async_pg_vector import AsyncPgVector
from .atlas_mongo_vector import AtlasMongoVector
from .extended_pg_vector import ExtendedPgVector


def get_vector_store(
    connection_string: str,
    embeddings: Embeddings,
    collection_name: str,
    mode: str = "sync",
    search_index: Optional[str] = None
) -> Union[ExtendedPgVector, AsyncPgVector, AtlasMongoVector]:
    """
    Create a vector store instance based on the specified mode.
    
    Args:
        connection_string: Database connection string.
        embeddings: Embeddings instance to use for vector operations.
        collection_name: Name of the collection/table to use.
        mode: Vector store mode - 'sync', 'async', or 'atlas-mongo'.
        search_index: Search index name (required for atlas-mongo mode).
        
    Returns:
        Configured vector store instance.
        
    Raises:
        ValueError: If an invalid mode is specified.
    """
    if mode == "sync":
        return ExtendedPgVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    elif mode == "async":
        return AsyncPgVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
    elif mode == "atlas-mongo":
        mongo_db = MongoClient(connection_string).get_database()
        mongo_collection = mongo_db[collection_name]
        return AtlasMongoVector(
            collection=mongo_collection, 
            embedding=embeddings, 
            index_name=search_index
        )
    else:
        raise ValueError(
            f"Invalid mode '{mode}' specified. Choose 'sync', 'async', or 'atlas-mongo'."
        )