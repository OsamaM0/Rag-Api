import pytest


def test_create_collection_minimal(monkeypatch, client, auth_headers):
    async def fake_create_collection(name, description=None, idx=None, custom_id=None):
        return {
            "uuid": "11111111-1111-1111-1111-111111111111",
            "idx": idx,
            "custom_id": custom_id,
            "name": name,
            "description": description,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        }

    monkeypatch.setattr("app.services.database.create_collection", fake_create_collection)

    payload = {"name": "My Coll"}
    res = client.post("/collections/", json=payload, headers=auth_headers)
    assert res.status_code == 201
    data = res.json()
    assert data["id"] == "11111111-1111-1111-1111-111111111111"
    assert data["name"] == "My Coll"


def test_list_collections_basic(monkeypatch, client, auth_headers):
    async def fake_get_all_collections(limit, offset):
        return ([
            {"uuid": "c1", "name": "A", "description": "d", "created_at": None, "updated_at": None},
            {"uuid": "c2", "name": "B", "description": "d", "created_at": None, "updated_at": None},
        ], 2)

    monkeypatch.setattr("app.services.database.get_all_collections", fake_get_all_collections)

    res = client.get("/collections/?page=1&page_size=2", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2


def test_get_update_delete_collection(monkeypatch, client, auth_headers):
    async def fake_get_by_uuid(cid):
        if cid in ("c1", "111"):  # simple allow
            return {"uuid": cid, "name": "C", "description": None, "idx": None, "custom_id": None, "created_at": None, "updated_at": None}
        return None

    async def fake_get_by_idx(cid):
        return None

    async def fake_get_by_custom_id(cid):
        return None

    async def fake_update_collection(collection_uuid, **kwargs):
        return {"uuid": collection_uuid, "name": kwargs.get("name", "C"), "description": kwargs.get("description"), "created_at": None, "updated_at": None}

    async def fake_delete_collection(uuid):
        return True

    monkeypatch.setattr("app.services.database.get_collection_by_uuid", fake_get_by_uuid)
    monkeypatch.setattr("app.services.database.get_collection_by_idx", fake_get_by_idx)
    monkeypatch.setattr("app.services.database.get_collection_by_custom_id", fake_get_by_custom_id)
    monkeypatch.setattr("app.services.database.update_collection", fake_update_collection)
    monkeypatch.setattr("app.services.database.delete_collection", fake_delete_collection)

    # get
    res = client.get("/collections/c1", headers=auth_headers)
    assert res.status_code == 200
    # update
    res = client.patch("/collections/c1", json={"name": "New"}, headers=auth_headers)
    assert res.status_code == 200
    assert res.json()["name"] == "New"
    # delete
    res = client.delete("/collections/c1", headers=auth_headers)
    assert res.status_code == 200


def test_bulk_delete_collections(monkeypatch, client, auth_headers):
    async def fake_get_by_uuid(cid):
        return {"uuid": cid, "name": "C", "description": None}

    async def fake_get_by_idx(cid):
        return None

    async def fake_get_by_custom_id(cid):
        return None

    async def fake_delete_collection(uuid):
        return True

    monkeypatch.setattr("app.services.database.get_collection_by_uuid", fake_get_by_uuid)
    monkeypatch.setattr("app.services.database.get_collection_by_idx", fake_get_by_idx)
    monkeypatch.setattr("app.services.database.get_collection_by_custom_id", fake_get_by_custom_id)
    monkeypatch.setattr("app.services.database.delete_collection", fake_delete_collection)

    res = client.post("/collections/bulk-delete", json=["c1", "c2"], headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["total_count"] == 2
    assert data["success_count"] == 2
