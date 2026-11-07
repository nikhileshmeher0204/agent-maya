import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from agent.api.routes import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    """Verify healthcheck endpoint returns 200."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "service": "agent-synapse"}


def test_chat_endpoint_requires_auth(client):
    """Verify that requests without Authorization header are rejected with 401/422."""
    res = client.post(
        "/api/v1/agent/chat",
        json={"conversation_id": "test_conv", "message": "hello"},
    )
    assert res.status_code in (401, 422)


def test_chat_endpoint_streaming_offline(client):
    """Verify end-to-end SSE chat stream in local mode without live API keys."""
    headers = {"Authorization": "Bearer mock_token_usr_123"}
    res = client.post(
        "/api/v1/agent/chat",
        json={"conversation_id": "conv_local_offline", "message": "Hello!"},
        headers=headers,
    )
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    assert "event: token" in res.text
    assert "event: done" in res.text


def test_chat_endpoint_streaming(client, monkeypatch):
    """Verify SSE streaming format and sequential IDs."""
    from tests.test_graph_cycle import MockLLM

    mock_llm = MockLLM([AIMessage(content="Hello from Agent Synapse")])
    monkeypatch.setattr("agent.graph.nodes.get_llm", lambda: mock_llm)

    headers = {"Authorization": "Bearer mock_token_usr_123"}
    res = client.post(
        "/api/v1/agent/chat",
        json={"conversation_id": "conv_sse_test", "message": "Hi"},
        headers=headers,
    )
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    text = res.text
    assert "event: " in text
    assert "event: done" in text
    assert "seq" in text


def test_session_messages_reconnect_recovery(client):
    """Safeguard 4: Verify message history retrieval since seq."""
    headers = {"Authorization": "Bearer mock_token_usr_123"}
    res = client.get(
        "/api/v1/agent/sessions/conv_sse_test/messages?since_seq=0",
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert "messages" in data
