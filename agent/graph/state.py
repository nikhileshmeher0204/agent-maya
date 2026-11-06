from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class PendingAction(TypedDict):
    action_id: str                      # Unique UUID for idempotency
    tool_name: str
    tool_call_id: str
    args: Dict[str, Any]
    preview_diff: Dict[str, Any]        # Visual diff payload rendered by UI
    created_at_epoch_ms: int


class AgentState(TypedDict):
    # App & Session Identity (Multi-tenant)
    app_id: str                         # Dynamic calling application id
    user_id: str                        # External user ID (e.g. Firebase UID)
    session_id: str                     # Conversation thread UUID

    # Message History
    messages: Annotated[List[BaseMessage], add_messages]

    # Global User Profile Memory
    user_profile: Dict[str, Any]        # Dynamic learned preferences

    # Active Scoped Tools (Discovered dynamically via MCP)
    available_tools: List[Dict[str, Any]]

    # Human-in-the-Loop Interception State
    pending_action: Optional[PendingAction]
    consent_status: Optional[str]       # "PENDING" | "APPROVED" | "REJECTED" | "EXPIRED"

    # Guardrail Step Counter (Hard bound: max 5 iterations per turn)
    step_count: int
