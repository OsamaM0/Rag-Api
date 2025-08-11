import pytest


def test_pgvector_check_index(monkeypatch, client, auth_headers):
    class DummyConn:
        async def fetch(self, q, *args):
            return [{"exists": True}]

    class DummyAcquire:
        async def __aenter__(self):
            return DummyConn()
        async def __aexit__(self, a,b,c):
            pass

    class DummyPool:
        def acquire(self):
            return DummyAcquire()

    async def fake_get_pool():
        return DummyPool()

    monkeypatch.setattr("app.services.database.PSQLDatabase.get_pool", fake_get_pool)

    res = client.get("/pgvector/test/check_index", params={"table_name": "langchain_pg_embedding", "column_name": "custom_id"}, headers=auth_headers)
    assert res.status_code == 200


def test_pgvector_records(monkeypatch, client, auth_headers):
    class DummyConn:
        async def fetch(self, q, *args):
            return [{"custom_id": "x"}]

    class DummyAcquire:
        async def __aenter__(self):
            return DummyConn()
        async def __aexit__(self, a,b,c):
            pass

    class DummyPool:
        def acquire(self):
            return DummyAcquire()

    async def fake_get_pool():
        return DummyPool()

    monkeypatch.setattr("app.services.database.PSQLDatabase.get_pool", fake_get_pool)

    assert client.get("/pgvector/records/all", params={"table_name": "langchain_pg_embedding"}, headers=auth_headers).status_code == 200
    assert client.get("/pgvector/records", params={"table_name": "langchain_pg_embedding", "custom_id": "x"}, headers=auth_headers).status_code == 200


def test_pgvector_delete_and_indexes(monkeypatch, client, auth_headers):
    class DummyConn:
        async def fetchrow(self, q, *args):
            if "EXISTS" in q:
                return {"exists": True}
            # For delete path count check
            if "COUNT(*)" in q:
                return {"count": 1}
            return {"count": 1}
        async def execute(self, q, *args):
            return "DELETE 1"

    class DummyAcquire:
        async def __aenter__(self):
            return DummyConn()
        async def __aexit__(self, a,b,c):
            pass

    class DummyPool:
        def acquire(self):
            return DummyAcquire()

    async def fake_get_pool():
        return DummyPool()

    monkeypatch.setattr("app.services.database.PSQLDatabase.get_pool", fake_get_pool)

    assert client.delete("/pgvector/records/x", params={"table_name": "langchain_pg_embedding"}, headers=auth_headers).status_code == 200

    # create index 409 path is also covered by exists -> True
    res = client.post("/pgvector/indexes/create", params={"table_name": "langchain_pg_embedding", "column_name": "custom_id"}, headers=auth_headers)
    assert res.status_code in (200, 409)

    # drop index path
    res = client.delete("/pgvector/indexes/idx_langchain_pg_embedding_custom_id", headers=auth_headers)
    assert res.status_code in (200, 404)


def test_pgvector_vector_similarity(monkeypatch, client, auth_headers):
    class DummyConn:
        async def fetch(self, q, *args):
            return [{"custom_id": "x", "document": "txt", "cmetadata": {}, "distance": 0.1}]

    class DummyAcquire:
        async def __aenter__(self):
            return DummyConn()
        async def __aexit__(self, a,b,c):
            pass

    class DummyPool:
        def acquire(self):
            return DummyAcquire()

    async def fake_get_pool():
        return DummyPool()

    monkeypatch.setattr("app.services.database.PSQLDatabase.get_pool", fake_get_pool)

    # Some frameworks parse repeated params inconsistently; send a JSON body instead via POST
    res = client.get("/pgvector/vector/similarity-search", params=[("vector", 0.1), ("vector", 0.2)], headers=auth_headers)
    assert res.status_code == 200


def test_pgvector_maintenance_stats(monkeypatch, client, auth_headers):
    class DummyConn:
        async def fetch(self, q):
            return [{"schemaname": "public", "tablename": "t", "size": "1 MB", "size_bytes": 1024}]

    class DummyAcquire:
        async def __aenter__(self):
            return DummyConn()
        async def __aexit__(self, a,b,c):
            pass

    class DummyPool:
        def acquire(self):
            return DummyAcquire()

    async def fake_get_pool():
        return DummyPool()

    monkeypatch.setattr("app.services.database.PSQLDatabase.get_pool", fake_get_pool)

    res = client.get("/pgvector/maintenance/stats", headers=auth_headers)
    assert res.status_code == 200
