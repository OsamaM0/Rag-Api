import pytest


def test_database_health_check(monkeypatch, client, auth_headers):
    class DummyConn:
        async def execute(self, q):
            return "OK"

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

    async def fake_is_health_ok():
        return True

    monkeypatch.setattr("app.services.database.PSQLDatabase.get_pool", fake_get_pool)
    monkeypatch.setattr("app.utils.health.is_health_ok", fake_is_health_ok)

    res = client.get("/database/health", headers=auth_headers)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] in ("UP", "DOWN", "ERROR")


def test_database_stats(monkeypatch, client, auth_headers):
    class DummyConn:
        async def fetchrow(self, q):
            return {"count": 1, "size": "1 MB"}

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

    res = client.get("/database/stats", headers=auth_headers)
    assert res.status_code == 200
    body = res.json()
    assert "total_documents" in body


def test_database_tables_and_columns_and_indexes(monkeypatch, client, auth_headers):
    class DummyConn:
        async def fetch(self, q, *args):
            if "information_schema.tables" in q:
                return [{"table_name": "t1"}]
            if "information_schema.columns" in q:
                return [{"column_name": "c1", "data_type": "text", "is_nullable": "YES", "column_default": None}]
            if "pg_indexes" in q:
                return [{"indexname": "i1", "indexdef": "def"}]
            return []

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

    assert client.get("/database/tables?schema=public", headers=auth_headers).status_code == 200
    assert client.get("/database/tables/t1/columns?schema=public", headers=auth_headers).status_code == 200
    assert client.get("/database/tables/t1/indexes?schema=public", headers=auth_headers).status_code == 200


def test_database_vacuum_analyze_backup(monkeypatch, client, auth_headers):
    class DummyConn:
        async def execute(self, q):
            return "OK"
        async def fetch(self, q, *args):
            return [{"table_name": "t1"}]

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

    assert client.post("/database/vacuum", headers=auth_headers).status_code == 200
    assert client.post("/database/analyze", headers=auth_headers).status_code == 200
    assert client.post("/database/backup", headers=auth_headers).status_code == 200
