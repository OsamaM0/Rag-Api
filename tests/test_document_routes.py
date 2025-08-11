import uuid as _uuid
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def _fake_doc(uuid_str: str):
    return {
        "uuid": uuid_str,
        "idx": "idx1",
        "custom_id": "c1",
        "filename": "f.txt",
        "content": "hello",
        "page_content": "hello",
        "mimetype": "text/plain",
        "binary_hash": None,
        "description": None,
        "keywords": None,
        "page_number": None,
        "pdf_path": None,
        "collection_id": None,
        "collection_name": None,
        "metadata": {},
        "created_at": None,
        "updated_at": None,
    }


def test_list_documents_minimal(monkeypatch, auth_headers):
    async def fake_get_all_documents(limit: int, offset: int, user_id=None, file_id=None):
        return ([_fake_doc(str(_uuid.uuid4()))], 1)

    import app.routes.document_routes as doc_routes

    monkeypatch.setattr(doc_routes, "get_all_documents", fake_get_all_documents)

    res = client.get("/documents", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1


def test_get_document_by_id(monkeypatch, auth_headers):
    uid = str(_uuid.uuid4())

    async def fake_get_document_by_uuid(document_id: str):
        return _fake_doc(uid) if document_id == uid else None

    import app.routes.document_routes as doc_routes

    monkeypatch.setattr(doc_routes, "get_document_by_uuid", fake_get_document_by_uuid)

    res = client.get(f"/documents/{uid}", headers=auth_headers)
    assert res.status_code == 200
    assert res.json()["uuid"] == uid


def test_delete_document(monkeypatch, auth_headers):
    uid = str(_uuid.uuid4())

    async def fake_get_document_by_uuid(document_id: str):
        return _fake_doc(uid)

    async def fake_delete_document(document_uuid: str):
        return True

    import app.routes.document_routes as doc_routes

    monkeypatch.setattr(doc_routes, "get_document_by_uuid", fake_get_document_by_uuid)
    monkeypatch.setattr(doc_routes, "delete_document", fake_delete_document)

    res = client.delete(f"/documents/{uid}", headers=auth_headers)
    assert res.status_code == 200
    assert res.json()["success"] is True


def test_bulk_delete_documents(monkeypatch, auth_headers):
    ids = [str(_uuid.uuid4()) for _ in range(2)]

    async def fake_get_document_by_uuid(document_id: str):
        # Return doc for first id only
        if document_id == ids[0]:
            return _fake_doc(ids[0])
        return None

    async def fake_delete_document(document_uuid: str):
        return True

    import app.routes.document_routes as doc_routes

    monkeypatch.setattr(doc_routes, "get_document_by_uuid", fake_get_document_by_uuid)
    monkeypatch.setattr(doc_routes, "delete_document", fake_delete_document)

    # Call exact route path with trailing slash to avoid redirect
    res = client.request("DELETE", "/documents/", json=ids, headers=auth_headers)
    assert res.status_code == 200
    msg = res.json()["message"].lower()
    assert "successfully deleted" in msg
