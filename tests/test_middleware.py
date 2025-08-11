import os
import jwt
import pytest
from app.middleware import security_middleware


class DummyRequest:
    def __init__(self, path, headers):
        self.url = type("URL", (), {"path": path})
        self.headers = headers
        self.state = type("State", (), {})()


async def dummy_call_next(request):
    return type("DummyResponse", (), {"status_code": 200})()


@pytest.mark.asyncio
async def test_security_middleware_valid():
    jwt_secret = "testsecret"
    os.environ["JWT_SECRET"] = jwt_secret
    payload = {"id": "testuser", "exp": 9999999999}
    token = jwt.encode(payload, jwt_secret, algorithm="HS256")
    headers = {"Authorization": f"Bearer {token}"}
    request = DummyRequest("/protected", headers)
    response = await security_middleware(request, dummy_call_next)
    assert response.status_code == 200
    assert hasattr(request.state, "user")
    assert request.state.user["id"] == "testuser"


@pytest.mark.asyncio
async def test_security_middleware_invalid_token():
    os.environ["JWT_SECRET"] = "testsecret"
    headers = {"Authorization": "Bearer invalidtoken"}
    request = DummyRequest("/protected", headers)
    response = await security_middleware(request, dummy_call_next)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_security_middleware_allows_health_without_auth():
    # Even with JWT configured, /health should bypass
    os.environ["JWT_SECRET"] = "testsecret"
    request = DummyRequest("/health", {})
    response = await security_middleware(request, dummy_call_next)
    assert response.status_code == 200