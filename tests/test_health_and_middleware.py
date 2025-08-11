import os
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_root_unauthorized_when_jwt_secret_set():
    os.environ["JWT_SECRET"] = "secret"
    res = client.get("/")
    assert res.status_code == 401
    assert res.json()["detail"]


def test_root_ok_when_no_jwt_secret():
    # Unset JWT_SECRET; middleware should allow through
    if "JWT_SECRET" in os.environ:
        del os.environ["JWT_SECRET"]
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["message"] == "RAG API Server"
