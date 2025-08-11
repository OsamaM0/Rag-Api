from app.models import DocumentResponse, QueryRequestBody, SimilaritySearchRequest


def test_document_response_model_minimal():
    data = {
        "uuid": "123",
        "filename": "f.txt",
    }
    m = DocumentResponse(**data)
    assert m.uuid == "123"
    assert m.filename == "f.txt"
    assert m.metadata is None or isinstance(m.metadata, dict)


def test_query_request_body_defaults():
    req = QueryRequestBody(query="q", file_id="fid")
    assert req.k == 4


def test_similarity_search_request_model():
    req = SimilaritySearchRequest(query="q", k=5, filter={"a": 1}, score_threshold=0.1)
    assert req.k == 5 and req.filter["a"] == 1 and req.score_threshold == 0.1