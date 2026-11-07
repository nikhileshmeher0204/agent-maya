from typing import Optional
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.graph.nodes import (
    async_memory,
    consent_gate,
    context_trimmer,
    execute_tool,
    hydrate_context,
    reasoner,
    route_after_execute,
    route_after_reasoner,
)
from agent.graph.state import AgentState


def create_agent_graph(checkpointer: Optional[BaseCheckpointSaver] = None):
    """Build and compile the cyclic Agent Synapse LangGraph state machine."""
    builder = StateGraph(AgentState)

    # 1. Register Nodes
    builder.add_node("hydrate_context", hydrate_context)
    builder.add_node("context_trimmer", context_trimmer)
    builder.add_node("reasoner", reasoner)
    builder.add_node("consent_gate", consent_gate)
    builder.add_node("execute_tool", execute_tool)
    builder.add_node("async_memory", async_memory)

    # 2. Register Edges
    builder.add_edge(START, "hydrate_context")
    builder.add_edge("hydrate_context", "context_trimmer")
    builder.add_edge("context_trimmer", "reasoner")

    builder.add_conditional_edges(
        "reasoner",
        route_after_reasoner,
        {
            "execute_tool": "execute_tool",
            "consent_gate": "consent_gate",
            "async_memory": "async_memory",
        },
    )

    builder.add_edge("consent_gate", "execute_tool")

    builder.add_conditional_edges(
        "execute_tool",
        route_after_execute,
        {
            "reasoner": "reasoner",
            "consent_gate": "consent_gate",
            "async_memory": "async_memory",
        },
    )

    builder.add_edge("async_memory", END)

    # In-memory saver as default if no external checkpointer is passed
    saver = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(checkpointer=saver)


# Default in-memory compiled graph for immediate execution / tests
agent_graph = create_agent_graph()
