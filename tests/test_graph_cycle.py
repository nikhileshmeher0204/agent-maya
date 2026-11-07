import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent.graph.graph import create_agent_graph


class MockLLM:
    """Mock LLM to control outputs in graph cycle tests."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return AIMessage(content="Default response")


@pytest.mark.asyncio
async def test_graph_conversational_cycle_without_tools(monkeypatch):
    """Test standard conversational turn without tool calls."""
    mock_llm = MockLLM([AIMessage(content="I am your assistant.")])
    monkeypatch.setattr("agent.graph.nodes.get_llm", lambda: mock_llm)

    checkpointer = MemorySaver()
    graph = create_agent_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "test_conv_1"}}
    input_state = {
        "app_id": "test_app",
        "user_id": "usr_test",
        "session_id": "test_conv_1",
        "messages": [HumanMessage(content="Who are you?")],
        "step_count": 0,
    }

    final_state = await graph.ainvoke(input_state, config=config)
    messages = final_state["messages"]
    last_msg = messages[-1]
    assert isinstance(last_msg, AIMessage)
    assert last_msg.content == "I am your assistant."


@pytest.mark.asyncio
async def test_graph_read_only_tool_cycle(monkeypatch):
    """Test cycle with read-only tool execution and follow-up response."""
    # First turn calls tool, second turn responds to user
    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "search_data", "args": {"q": "movies"}, "id": "tc_read_1"}],
        ),
        AIMessage(content="I found 3 items matching your search."),
    ]
    mock_llm = MockLLM(responses)
    monkeypatch.setattr("agent.graph.nodes.get_llm", lambda: mock_llm)

    # Mock tool dispatcher
    async def mock_call_tool(*args, **kwargs):
        return [{"id": "1", "title": "Inception"}]

    monkeypatch.setattr("agent.mcp.client.nango_mcp_client.call_tool", mock_call_tool)

    checkpointer = MemorySaver()
    graph = create_agent_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "test_conv_2"}}
    input_state = {
        "app_id": "test_app",
        "user_id": "usr_test",
        "session_id": "test_conv_2",
        "messages": [HumanMessage(content="Search my movies")],
        "step_count": 0,
    }

    final_state = await graph.ainvoke(input_state, config=config)
    messages = final_state["messages"]
    # Check that tool output was captured
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) >= 1
    assert "Inception" in tool_msgs[0].content

    # Check final LLM response
    assert messages[-1].content == "I found 3 items matching your search."


@pytest.mark.asyncio
async def test_graph_hitl_consent_pause_and_resume_approval(monkeypatch):
    """Test HITL interrupt pause on mutating tool call and resumption upon approval."""
    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "create_personal_collection", "args": {"name": "SciFi"}, "id": "tc_mut_1"}],
        ),
        AIMessage(content="Collection 'SciFi' has been created successfully."),
    ]
    mock_llm = MockLLM(responses)
    monkeypatch.setattr("agent.graph.nodes.get_llm", lambda: mock_llm)

    executed_actions = []

    async def mock_call_tool(user_id, tool_name, arguments, idempotency_key=None):
        executed_actions.append({"tool": tool_name, "args": arguments, "key": idempotency_key})
        return {"collection_id": "col_123", "status": "CREATED"}

    monkeypatch.setattr("agent.mcp.client.nango_mcp_client.call_tool", mock_call_tool)

    checkpointer = MemorySaver()
    graph = create_agent_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "test_conv_hitl"}}
    input_state = {
        "app_id": "test_app",
        "user_id": "usr_test",
        "session_id": "test_conv_hitl",
        "messages": [HumanMessage(content="Create SciFi collection")],
        "step_count": 0,
    }

    # 1. Run until interrupt
    state_after_pause = await graph.ainvoke(input_state, config=config)
    curr = await graph.aget_state(config)

    # Verify that the graph paused at consent_gate with interrupt
    assert curr.tasks
    assert any(t.interrupts for t in curr.tasks)
    assert len(executed_actions) == 0  # Tool must NOT have run yet

    # 2. Resume graph with user approval Command(resume=...)
    resume_command = Command(resume={"approved": True, "action_id": "test-action-uuid"})
    resumed_state = await graph.ainvoke(resume_command, config=config)

    # Tool should now have executed with idempotency key
    assert len(executed_actions) == 1
    assert executed_actions[0]["tool"] == "create_personal_collection"
    assert resumed_state["messages"][-1].content == "Collection 'SciFi' has been created successfully."


@pytest.mark.asyncio
async def test_graph_hitl_consent_rejection(monkeypatch):
    """Test HITL interrupt rejection cancelling mutating tool execution."""
    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "delete_personal_collection", "args": {"collection_id": "123"}, "id": "tc_mut_2"}],
        ),
        AIMessage(content="Understood, I cancelled the deletion."),
    ]
    mock_llm = MockLLM(responses)
    monkeypatch.setattr("agent.graph.nodes.get_llm", lambda: mock_llm)

    executed_actions = []

    async def mock_call_tool(*args, **kwargs):
        executed_actions.append(args)
        return {"status": "DELETED"}

    monkeypatch.setattr("agent.mcp.client.nango_mcp_client.call_tool", mock_call_tool)

    checkpointer = MemorySaver()
    graph = create_agent_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "test_conv_reject"}}
    input_state = {
        "app_id": "test_app",
        "user_id": "usr_test",
        "session_id": "test_conv_reject",
        "messages": [HumanMessage(content="Delete collection 123")],
        "step_count": 0,
    }

    # 1. Run until interrupt
    await graph.ainvoke(input_state, config=config)

    # 2. Resume with rejection
    resume_command = Command(resume={"approved": False, "action_id": "test-action-uuid"})
    final_state = await graph.ainvoke(resume_command, config=config)

    # Tool must NEVER have executed
    assert len(executed_actions) == 0
    # ToolMessage should indicate cancellation
    tool_msgs = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    assert any("cancelled" in m.content.lower() for m in tool_msgs)
