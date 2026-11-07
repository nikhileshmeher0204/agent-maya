from langgraph.checkpoint.memory import MemorySaver
from agent.graph.graph import create_agent_graph
from agent.graph.state import AgentState, PendingAction


def test_graph_compilation():
    """Verify that Agent Synapse StateGraph compiles cleanly with MemorySaver."""
    checkpointer = MemorySaver()
    graph = create_agent_graph(checkpointer=checkpointer)
    assert graph is not None
    assert "hydrate_context" in graph.nodes
    assert "context_trimmer" in graph.nodes
    assert "reasoner" in graph.nodes
    assert "consent_gate" in graph.nodes
    assert "execute_tool" in graph.nodes
    assert "async_memory" in graph.nodes


def test_pending_action_schema():
    """Verify TypedDict structure for pending actions."""
    action: PendingAction = {
        "action_id": "test-uuid-1234",
        "tool_name": "create_personal_collection",
        "tool_call_id": "call_abc",
        "args": {"name": "Test List"},
        "preview_diff": {"summary": "Create collection"},
        "created_at_epoch_ms": 1700000000000,
    }
    assert action["action_id"] == "test-uuid-1234"
    assert action["tool_name"] == "create_personal_collection"
