"""
MongoDB Atlas Vector Store implementation.

This module provides a MongoDB Atlas-specific implementation of vector storage
with support for similarity search, document management, and metadata handling.
"""

import copy
from typing import Any, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_mongodb import MongoDBAtlasVectorSearch


class AtlasMongoVector(MongoDBAtlasVectorSearch):
    """
    MongoDB Atlas Vector Store implementation.
    
    This class extends MongoDBAtlasVectorSearch to provide enhanced functionality
    for document management, similarity search, and ID-based operations.
    """

    @property
    def embedding_function(self) -> Embeddings:
        """Get the embedding function used by this vector store."""
        return self.embeddings

    def add_documents(self, docs: List[Document], ids: List[str]) -> List[str]:
        """
        Add documents to the vector store with custom IDs.
        
        Args:
            docs: List of documents to add.
            ids: List of custom IDs for the documents.
            
        Returns:
            List of generated IDs in the format {file_id}_{idx}.
        """
        # Generate new IDs in format {file_id}_{idx}
        new_ids = [id for id in range(len(ids))]
        file_id = docs[0].metadata['file_id']
        f_ids = [f'{file_id}_{id}' for id in new_ids]
        return super().add_documents(docs, f_ids)

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with scores using a vector.
        
        Args:
            embedding: Query embedding vector.
            k: Number of results to return.
            filter: Optional filter criteria.
            **kwargs: Additional search parameters.
            
        Returns:
            List of (Document, score) tuples.
        """
        docs = self._similarity_search_with_score(
            embedding,
            k=k,
            pre_filter=filter,
            post_filter_pipeline=None,
            **kwargs,
        )
        processed_documents: List[Tuple[Document, float]] = []
        for document, score in docs:
            # Make a deep copy to avoid mutating the original document
            doc_copy = copy.deepcopy(document.__dict__)
            # Remove _id field from metadata if it exists to avoid serialization issues
            if "metadata" in doc_copy and "_id" in doc_copy["metadata"]:
                del doc_copy["metadata"]["_id"]
            new_document = Document(**doc_copy)
            processed_documents.append((new_document, score))
        return processed_documents

    def get_all_ids(self) -> List[str]:
        """
        Get all unique file IDs from the collection.
        
        Returns:
            List of unique file IDs.
        """
        return self._collection.distinct("file_id")
    
    def get_filtered_ids(self, ids: List[str]) -> List[str]:
        """
        Filter the provided IDs to only include those that exist in the collection.
        
        Args:
            ids: List of IDs to filter.
            
        Returns:
            List of IDs that exist in the collection.
        """
        return self._collection.distinct("file_id", {"file_id": {"$in": ids}})

    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        """
        Retrieve documents by their file IDs.
        
        Args:
            ids: List of file IDs to retrieve.
            
        Returns:
            List of Document objects.
        """
        return [
            Document(
                page_content=doc["text"],
                metadata={
                    "file_id": doc["file_id"],
                    "user_id": doc["user_id"],
                    "digest": doc["digest"],
                    "source": doc["source"],
                    "page": int(doc.get("page", 0)),
                },
            )
            for doc in self._collection.find({"file_id": {"$in": ids}})
        ]

    def delete(self, ids: Optional[List[str]] = None) -> None:
        """
        Delete documents by their file IDs.
        
        Args:
            ids: List of file IDs to delete. If None, no deletion occurs.
        """
        if ids is not None:
            self._collection.delete_many({"file_id": {"$in": ids}})