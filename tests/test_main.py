import os
import jwt
import datetime
import pytest
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


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


def test_root_with_auth(auth_headers):
    # Root endpoint requires auth when JWT_SECRET is set
    res = client.get("/", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["message"] == "RAG API Server"


def test_health_ok(monkeypatch):
    # Force health ok without DB access
    async def ok():
        return True

    import app.utils.health as health_util

    monkeypatch.setattr(health_util, "pg_health_check", ok)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "UP"
