import time
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.graph.nodes import (
    context_trimmer,
    execute_tool,
    hydrate_context,
    is_tool_mutating,
    route_after_reasoner,
)
from agent.graph.state import AgentState, PendingAction


@pytest.mark.asyncio
async def test_safeguard_1_stale_consent_ttl_expiration():
    """Safeguard 1: Action pending over 24 hours is automatically expired."""
    expired_time = int(time.time() * 1000) - (25 * 3600 * 1000)  # 25 hours ago
    state: AgentState = {
        "app_id": "test_app",
        "user_id": "usr_1",
        "session_id": "thread_1",
        "messages": [HumanMessage(content="Hello")],
        "user_profile": {},
        "available_tools": [],
        "pending_action": {
            "action_id": "act_1",
            "tool_name": "create_item",
            "tool_call_id": "call_1",
            "args": {},
            "preview_diff": {},
            "created_at_epoch_ms": expired_time,
        },
        "consent_status": "PENDING",
        "step_count": 0,
    }

    updates = await hydrate_context(state)
    assert updates.get("pending_action") is None
    assert updates.get("consent_status") == "EXPIRED"
    assert any("expired" in m.content.lower() for m in updates.get("messages", []))


@pytest.mark.asyncio
async def test_safeguard_1_interruption_cleansing_on_new_message():
    """Safeguard 1: Interrupted pending action is discarded when user sends a new message."""
    recent_time = int(time.time() * 1000) - 1000  # 1 second ago
    state: AgentState = {
        "app_id": "test_app",
        "user_id": "usr_1",
        "session_id": "thread_1",
        "messages": [HumanMessage(content="Discard that, do something else")],
        "user_profile": {},
        "available_tools": [],
        "pending_action": {
            "action_id": "act_1",
            "tool_name": "delete_item",
            "tool_call_id": "call_1",
            "args": {},
            "preview_diff": {},
            "created_at_epoch_ms": recent_time,
        },
        "consent_status": "PENDING",
        "step_count": 0,
    }

    updates = await hydrate_context(state)
    assert updates.get("pending_action") is None
    assert updates.get("consent_status") == "EXPIRED"
    assert any("discarded" in m.content.lower() for m in updates.get("messages", []))


def test_safeguard_2_multi_tool_partitioning():
    """Safeguard 2: When both read-only and mutating tools are called, route to execute read-only first."""
    state: AgentState = {
        "app_id": "test_app",
        "user_id": "usr_1",
        "session_id": "thread_1",
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search_items", "args": {"q": "test"}, "id": "tc_read"},
                    {"name": "create_collection", "args": {"name": "favorites"}, "id": "tc_write"},
                ],
            )
        ],
        "user_profile": {},
        "available_tools": [],
        "pending_action": None,
        "consent_status": None,
        "step_count": 1,
    }

    route = route_after_reasoner(state)
    # Read-only must execute first to provide accurate preview diff
    assert route == "execute_tool"


@pytest.mark.asyncio
async def test_safeguard_5_context_trimmer():
    """Safeguard 5: Trims messages exceeding 6 turns and retains system prompt."""
    many_messages = [HumanMessage(content=f"Turn {i}") for i in range(12)]
    state: AgentState = {
        "app_id": "test_app",
        "user_id": "usr_1",
        "session_id": "thread_1",
        "messages": many_messages,
        "user_profile": {"genre": "Sci-Fi"},
        "available_tools": [],
        "pending_action": None,
        "consent_status": None,
        "step_count": 0,
    }

    result = await context_trimmer(state)
    messages = result["messages"]
    assert isinstance(messages[0], SystemMessage)
    assert "assistant" in messages[0].content
    # Preserves summary + 6 recent turns = 8 messages
    assert len(messages) <= 8


@pytest.mark.asyncio
async def test_safeguard_6_tool_exception_containment(monkeypatch):
    """Safeguard 6: Tool failure is caught and returned as ToolMessage(status='error') without crashing."""
    async def failing_tool(*args, **kwargs):
        raise ConnectionResetError("Remote MCP service timed out")

    monkeypatch.setattr("agent.mcp.client.nango_mcp_client.call_tool", failing_tool)

    state: AgentState = {
        "app_id": "test_app",
        "user_id": "usr_1",
        "session_id": "thread_1",
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_items", "args": {}, "id": "call_fail_1"}],
            )
        ],
        "user_profile": {},
        "available_tools": [],
        "pending_action": None,
        "consent_status": None,
        "step_count": 1,
    }

    result = await execute_tool(state)
    assert "messages" in result
    tool_msg = result["messages"][0]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.status == "error"
    assert "timed out" in tool_msg.content
