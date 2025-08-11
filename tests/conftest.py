"""Pytest configuration and shared fixtures."""
import os
import types

from app.services.vector_store.async_pg_vector import AsyncPgVector

# Set environment variables early so config picks up test settings.
os.environ["TESTING"] = "1"
# Set DB_HOST (and DSN) to dummy values to avoid real connection attempts.
os.environ["DB_HOST"] = "localhost"
os.environ["DSN"] = "dummy://"
# Enable debug mode so optional debug routes (e.g., /pgvector) are registered
os.environ["DEBUG_RAG_API"] = "true"

# -- Patch the vector store classes to bypass DB connection --
# Do this before importing app modules that might instantiate vector stores.
from langchain_community.vectorstores.pgvector import PGVector


def dummy_post_init(self):
    # Skip extension creation or any heavy initialization in tests
    pass


AsyncPgVector.__post_init__ = dummy_post_init
PGVector.__post_init__ = dummy_post_init

from langchain_core.documents import Document


class _DummyEmbedder:
    def embed_query(self, query: str):
        # Return a deterministic small vector regardless of input
        return [0.1, 0.2, 0.3]


class DummyVectorStore:
    """A lightweight in-memory stand-in for the vector store."""

    def __init__(self):
        self.embedding_function = _DummyEmbedder()

    def get_all_ids(self) -> list[str]:
        return ["testid1", "testid2"]

    def get_filtered_ids(self, ids) -> list[str]:
        dummy_ids = ["testid1", "testid2"]
        return [id for id in dummy_ids if id in ids]

    async def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        return [Document(page_content="Test content", metadata={"file_id": id}) for id in ids]

    def similarity_search_with_score_by_vector(self, embedding, k: int, filter: dict | None):
        file_id = None
        if isinstance(filter, dict):
            if isinstance(filter.get("file_id"), dict) and "$in" in filter["file_id"]:
                file_id = filter["file_id"]["$in"][0]
            else:
                file_id = filter.get("file_id", "testid1")
        doc = Document(
            page_content="Queried content",
            metadata={"file_id": file_id or "testid1", "user_id": "testuser"},
        )
        return [(doc, 0.9)]

    def add_documents(self, docs, ids=None):
        return ids or ["id" for _ in docs]

    async def aadd_documents(self, docs, ids=None):
        return ids or ["id" for _ in docs]

    async def delete(self, ids=None, collection_only: bool = False):
        return None

    def as_retriever(self):
        return self


# Replace the global vector_store with our dummy to avoid DB/network access in tests
import app.config as app_config  # noqa: E402

app_config.vector_store = DummyVectorStore()
# Keep retriever consistent with the dummy store
app_config.retriever = app_config.vector_store.as_retriever()


# ---------------------------------------------------------------------------
# Common test fixtures
# ---------------------------------------------------------------------------
import datetime
import jwt
import pytest
from fastapi.testclient import TestClient
import main as main_module


@pytest.fixture
def auth_headers():
    jwt_secret = "testsecret"
    os.environ["JWT_SECRET"] = jwt_secret
    payload = {
        "id": "testuser",
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1),
    }
    token = jwt.encode(payload, jwt_secret, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client():
    """Shared FastAPI TestClient for tests."""
    return TestClient(main_module.app)

