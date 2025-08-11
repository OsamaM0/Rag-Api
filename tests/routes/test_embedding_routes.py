import pytest


def test_list_embeddings(monkeypatch, client, auth_headers):
    async def fake_get_all_embeddings(limit, offset):
        return ([{
            "custom_id": "e1",
            "document_id": "d1",
            "document": "hello world",
            "cmetadata": {},
            "embedding": [0.1, 0.2],
            "created_at": None
        }], 1)

    monkeypatch.setattr("app.routes.embedding_routes.db_get_all_embeddings", fake_get_all_embeddings)

    res = client.get("/embeddings/?page=1&page_size=10", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1


def test_get_embedding(monkeypatch, client, auth_headers):
    async def fake_get_embedding_by_id(eid):
        return {
            "custom_id": eid,
            "document_id": "d1",
            "document": "content",
            "embedding": [0.5, 0.5],
            "cmetadata": {},
            "created_at": None
        }

    monkeypatch.setattr("app.routes.embedding_routes.db_get_embedding_by_id", fake_get_embedding_by_id)

    res = client.get("/embeddings/e1", headers=auth_headers)
    assert res.status_code == 200
    body = res.json()
    assert body["embedding_id"] == "e1"


def test_delete_embedding(monkeypatch, client, auth_headers):
    async def fake_delete_embedding(eid):
        return True

    monkeypatch.setattr("app.routes.embedding_routes.db_delete_embedding", fake_delete_embedding)
    res = client.delete("/embeddings/e1", headers=auth_headers)
    assert res.status_code == 200


def test_find_similar_embeddings_in_document(monkeypatch, client, auth_headers):
    async def fake_get_document_by_uuid(doc_id):
        return {"uuid": doc_id, "content": "abc", "filename": "f.txt", "collection_id": None}

    async def fake_similarity_search_embeddings(query_embedding, k, document_uuid=None, collection_uuid=None):
        return [{"custom_id": "e1", "document": "text", "cmetadata": {}, "distance": 0.1}]

    class FakeEmb:
        def embed_query(self, t):
            return [0.1, 0.2]

    monkeypatch.setattr("app.routes.embedding_routes.get_document_by_uuid", fake_get_document_by_uuid)
    monkeypatch.setattr("app.routes.embedding_routes.similarity_search_embeddings", fake_similarity_search_embeddings)
    import app.config as cfg
    monkeypatch.setattr(cfg, "embeddings", FakeEmb(), raising=False)

    res = client.post("/embeddings/similarity/document/d1", params={"text": "hi", "k": 1}, headers=auth_headers)
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_find_similar_embeddings_in_collection(monkeypatch, client, auth_headers):
    async def fake_get_collection_by_uuid(cid):
        return {"uuid": cid, "name": "C"}

    async def fake_similarity_search_embeddings(query_embedding, k, document_uuid=None, collection_uuid=None):
        return [{"custom_id": "e1", "document": "text", "cmetadata": {}, "distance": 0.1, "document_id": "d1"}]

    class FakeEmb:
        def embed_query(self, t):
            return [0.1, 0.2]

    monkeypatch.setattr("app.routes.embedding_routes.get_collection_by_uuid", fake_get_collection_by_uuid)
    monkeypatch.setattr("app.routes.embedding_routes.similarity_search_embeddings", fake_similarity_search_embeddings)
    import app.config as cfg
    monkeypatch.setattr(cfg, "embeddings", FakeEmb(), raising=False)

    res = client.post("/embeddings/similarity/collection/c1", params={"text": "hi", "k": 1}, headers=auth_headers)
    assert res.status_code == 200
    assert isinstance(res.json(), list)
