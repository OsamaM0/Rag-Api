import pytest
import uuid as _uuid
from unittest.mock import AsyncMock

from tests.conftest import client


def _fake_document_block(block_id: str = None, document_id: str = None, block_idx: int = 0, name: str = None):
    """Create a fake document block for testing."""
    return {
        "id": block_id or str(_uuid.uuid4()),
        "block_idx": block_idx,
        "document_id": document_id or str(_uuid.uuid4()),
        "name": name or f"block_{block_idx}",
        "content": f"Sample content for block {block_idx}",
        "level": 1,
        "page_idx": 0,
        "tag": "para",
        "block_class": "text",
        "x0": 10.0,
        "y0": 20.0,
        "x1": 100.0,
        "y1": 50.0,
        "parent_idx": None,
        "content_type": "regular",
        "section_type": None,
        "demand_priority": None,
        "created_at": None
    }


def _fake_document(doc_id: str = None):
    """Create a fake document for testing."""
    return {
        "uuid": doc_id or str(_uuid.uuid4()),
        "filename": "test.pdf",
        "content": "Test document content",
        "collection_id": None
    }


def test_list_document_blocks(monkeypatch, client, auth_headers):
    """Test listing document blocks with pagination."""
    async def fake_search_document_blocks(document_id=None, query=None, content_type=None, 
                                        section_type=None, limit=20, offset=0):
        return ([_fake_document_block()], 1)

    import app.routes.document_block_routes as routes
    monkeypatch.setattr(routes, "search_document_blocks", fake_search_document_blocks)

    res = client.get("/document-blocks/?page=1&page_size=10", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 1
    assert len(data["items"]) == 1


def test_create_document_block(monkeypatch, client, auth_headers):
    """Test creating a single document block."""
    doc_id = str(_uuid.uuid4())
    
    async def fake_get_document_by_uuid(document_id):
        return _fake_document(doc_id) if document_id == doc_id else None
    
    async def fake_create_document_block(document_id, **kwargs):
        return _fake_document_block(
            document_id=document_id, 
            block_idx=kwargs.get('block_idx', 0),
            name=kwargs.get('name', f"block_{kwargs.get('block_idx', 0)}")
        )

    import app.routes.document_block_routes as routes
    monkeypatch.setattr(routes, "get_document_by_uuid", fake_get_document_by_uuid)
    monkeypatch.setattr(routes, "create_document_block", fake_create_document_block)

    block_data = {
        "block_idx": 1,
        "name": "test_block",
        "content": "Test content",
        "level": 1,
        "page_idx": 0,
        "tag": "para",
        "content_type": "regular"
    }

    res = client.post(f"/document-blocks/?document_id={doc_id}", json=block_data, headers=auth_headers)
    assert res.status_code == 200
    result = res.json()
    assert result["block_idx"] == 1
    assert result["name"] == "test_block"


def test_create_document_blocks_bulk(monkeypatch, client, auth_headers):
    """Test creating multiple document blocks in bulk."""
    doc_id = str(_uuid.uuid4())
    
    async def fake_get_document_by_uuid(document_id):
        return _fake_document(doc_id) if document_id == doc_id else None
    
    async def fake_create_document_blocks_bulk(document_id, blocks_data):
        return [_fake_document_block(document_id=document_id, block_idx=i) for i in range(len(blocks_data))]

    import app.routes.document_block_routes as routes
    monkeypatch.setattr(routes, "get_document_by_uuid", fake_get_document_by_uuid)
    monkeypatch.setattr(routes, "create_document_blocks_bulk", fake_create_document_blocks_bulk)

    bulk_data = {
        "document_id": doc_id,
        "blocks": [
            {
                "block_idx": 1,
                "name": "block_1",
                "content": "Content 1",
                "level": 1,
                "page_idx": 0,
                "tag": "para",
                "content_type": "regular"
            },
            {
                "block_idx": 2,
                "name": "block_2",
                "content": "Content 2",
                "level": 1,
                "page_idx": 0,
                "tag": "header",
                "content_type": "header"
            }
        ]
    }

    res = client.post("/document-blocks/bulk", json=bulk_data, headers=auth_headers)
    assert res.status_code == 200
    result = res.json()
    assert result["success"] is True
    assert "2 document blocks" in result["message"]


def test_get_document_blocks(monkeypatch, client, auth_headers):
    """Test getting all blocks for a specific document."""
    doc_id = str(_uuid.uuid4())
    
    async def fake_get_document_by_uuid(document_id):
        return _fake_document(doc_id) if document_id == doc_id else None
    
    async def fake_get_document_blocks_by_document(document_id, limit=100, offset=0):
        return ([_fake_document_block(document_id=document_id, block_idx=i) for i in range(3)], 3)

    import app.routes.document_block_routes as routes
    monkeypatch.setattr(routes, "get_document_by_uuid", fake_get_document_by_uuid)
    monkeypatch.setattr(routes, "get_document_blocks_by_document", fake_get_document_blocks_by_document)

    res = client.get(f"/document-blocks/document/{doc_id}", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 3
    assert len(data["items"]) == 3


def test_get_document_block_by_id(monkeypatch, client, auth_headers):
    """Test getting a specific document block by ID."""
    block_id = str(_uuid.uuid4())
    
    async def fake_get_document_block_by_id(block_id_param):
        return _fake_document_block(block_id) if block_id_param == block_id else None

    import app.routes.document_block_routes as routes
    monkeypatch.setattr(routes, "get_document_block_by_id", fake_get_document_block_by_id)

    res = client.get(f"/document-blocks/{block_id}", headers=auth_headers)
    assert res.status_code == 200
    result = res.json()
    assert result["id"] == block_id


def test_update_document_block(monkeypatch, client, auth_headers):
    """Test updating a document block."""
    block_id = str(_uuid.uuid4())
    fake_block = _fake_document_block(block_id)
    
    async def fake_get_document_block_by_id(block_id_param):
        return fake_block if block_id_param == block_id else None
    
    async def fake_update_document_block(block_id_param, **updates):
        updated_block = fake_block.copy()
        updated_block.update(updates)
        return updated_block

    import app.routes.document_block_routes as routes
    monkeypatch.setattr(routes, "get_document_block_by_id", fake_get_document_block_by_id)
    monkeypatch.setattr(routes, "update_document_block", fake_update_document_block)

    update_data = {
        "name": "updated_block",
        "content": "Updated content"
    }

    res = client.patch(f"/document-blocks/{block_id}", json=update_data, headers=auth_headers)
    assert res.status_code == 200
    result = res.json()
    assert result["name"] == "updated_block"
    assert result["content"] == "Updated content"


def test_delete_document_block(monkeypatch, client, auth_headers):
    """Test deleting a document block."""
    block_id = str(_uuid.uuid4())
    
    async def fake_get_document_block_by_id(block_id_param):
        return _fake_document_block(block_id) if block_id_param == block_id else None
    
    async def fake_delete_document_block(block_id_param):
        return True

    import app.routes.document_block_routes as routes
    monkeypatch.setattr(routes, "get_document_block_by_id", fake_get_document_block_by_id)
    monkeypatch.setattr(routes, "delete_document_block", fake_delete_document_block)

    res = client.delete(f"/document-blocks/{block_id}", headers=auth_headers)
    assert res.status_code == 200
    result = res.json()
    assert result["success"] is True


def test_delete_document_blocks_by_document(monkeypatch, client, auth_headers):
    """Test deleting all blocks for a document."""
    doc_id = str(_uuid.uuid4())
    
    async def fake_get_document_by_uuid(document_id):
        return _fake_document(doc_id) if document_id == doc_id else None
    
    async def fake_delete_document_blocks_by_document(document_id):
        return 5  # Number of deleted blocks

    import app.routes.document_block_routes as routes
    monkeypatch.setattr(routes, "get_document_by_uuid", fake_get_document_by_uuid)
    monkeypatch.setattr(routes, "delete_document_blocks_by_document", fake_delete_document_blocks_by_document)

    res = client.delete(f"/document-blocks/document/{doc_id}", headers=auth_headers)
    assert res.status_code == 200
    result = res.json()
    assert result["success"] is True
    assert "5 document blocks" in result["message"]


def test_upload_json_and_create_blocks(monkeypatch, client, auth_headers):
    """Test uploading JSON file and creating blocks."""
    doc_id = str(_uuid.uuid4())
    
    async def fake_get_document_by_uuid(document_id):
        return _fake_document(doc_id) if document_id == doc_id else None
    
    async def fake_create_document_blocks_bulk(document_id, blocks_data):
        return [_fake_document_block(document_id=document_id, block_idx=i) for i in range(len(blocks_data))]

    import app.routes.document_block_routes as routes
    monkeypatch.setattr(routes, "get_document_by_uuid", fake_get_document_by_uuid)
    monkeypatch.setattr(routes, "create_document_blocks_bulk", fake_create_document_blocks_bulk)

    # Sample JSON data
    json_data = [
        {
            "block_idx": 0,
            "sentences": [{"text": "This is a test sentence."}],
            "level": 0,
            "page_idx": 0,
            "tag": "para",
            "bbox": [10.0, 20.0, 100.0, 50.0]
        },
        {
            "block_idx": 1,
            "sentences": [{"text": "Another test sentence."}],
            "level": 1,
            "page_idx": 0,
            "tag": "header",
            "bbox": [10.0, 60.0, 200.0, 90.0]
        }
    ]

    # Create a mock file
    import io
    import json
    json_content = json.dumps(json_data).encode('utf-8')
    files = {"file": ("test.json", io.BytesIO(json_content), "application/json")}
    data = {"document_id": doc_id, "custom_id": "test_custom"}

    res = client.post("/document-blocks/upload-json", files=files, data=data, headers=auth_headers)
    assert res.status_code == 200
    result = res.json()
    assert result["success"] is True
    assert result["blocks_created"] == 2


def test_upload_invalid_json(monkeypatch, client, auth_headers):
    """Test uploading invalid JSON file."""
    doc_id = str(_uuid.uuid4())
    
    async def fake_get_document_by_uuid(document_id):
        return _fake_document(doc_id)

    import app.routes.document_block_routes as routes
    monkeypatch.setattr(routes, "get_document_by_uuid", fake_get_document_by_uuid)

    # Create invalid JSON file
    import io
    invalid_json = b"{ invalid json content"
    files = {"file": ("test.json", io.BytesIO(invalid_json), "application/json")}
    data = {"document_id": doc_id}

    res = client.post("/document-blocks/upload-json", files=files, data=data, headers=auth_headers)
    assert res.status_code == 400
    assert "Invalid JSON file" in res.json()["detail"]


def test_create_block_document_not_found(monkeypatch, client, auth_headers):
    """Test creating block for non-existent document."""
    doc_id = str(_uuid.uuid4())
    
    async def fake_get_document_by_uuid(document_id):
        return None  # Document not found

    import app.routes.document_block_routes as routes
    monkeypatch.setattr(routes, "get_document_by_uuid", fake_get_document_by_uuid)

    block_data = {
        "block_idx": 1,
        "name": "test_block",
        "content": "Test content",
        "level": 1,
        "page_idx": 0,
        "tag": "para",
        "content_type": "regular"
    }

    res = client.post(f"/document-blocks/?document_id={doc_id}", json=block_data, headers=auth_headers)
    assert res.status_code == 404
    assert "not found" in res.json()["detail"]


def test_get_block_not_found(monkeypatch, client, auth_headers):
    """Test getting non-existent document block."""
    block_id = str(_uuid.uuid4())
    
    async def fake_get_document_block_by_id(block_id_param):
        return None  # Block not found

    import app.routes.document_block_routes as routes
    monkeypatch.setattr(routes, "get_document_block_by_id", fake_get_document_block_by_id)

    res = client.get(f"/document-blocks/{block_id}", headers=auth_headers)
    assert res.status_code == 404
    assert "not found" in res.json()["detail"]


def test_search_document_blocks_with_filters(monkeypatch, client, auth_headers):
    """Test searching document blocks with various filters."""
    doc_id = str(_uuid.uuid4())
    
    async def fake_search_document_blocks(document_id=None, query=None, content_type=None, 
                                        section_type=None, limit=20, offset=0):
        # Return filtered results based on parameters
        blocks = []
        if content_type == "header":
            blocks = [_fake_document_block(document_id=document_id, block_idx=0)]
        elif query == "test":
            blocks = [_fake_document_block(document_id=document_id, block_idx=1)]
        else:
            blocks = [_fake_document_block(document_id=document_id, block_idx=i) for i in range(2)]
        return blocks, len(blocks)

    import app.routes.document_block_routes as routes
    monkeypatch.setattr(routes, "search_document_blocks", fake_search_document_blocks)

    # Test with content_type filter
    res = client.get(f"/document-blocks/?content_type=header&document_id={doc_id}", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 1

    # Test with query filter
    res = client.get(f"/document-blocks/?query=test&document_id={doc_id}", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 1
