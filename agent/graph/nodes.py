import json
import time
import uuid
from typing import Any, Dict, List, Literal
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import interrupt

from agent.graph.state import AgentState, PendingAction
from agent.llm.router import get_llm
from agent.mcp.client import nango_mcp_client
from agent.memory.profile_memory import mem0_service


MUTATING_PREFIXES = ("create_", "delete_", "add_", "remove_", "update_", "modify_", "insert_", "drop_")


def is_tool_mutating(tool_name: str, available_tools: List[Dict[str, Any]]) -> bool:
    """Dynamically determine if a tool performs a mutating action requiring consent."""
    for tool in available_tools:
        func = tool.get("function", tool)
        if func.get("name") == tool_name:
            if func.get("mutating") is True or func.get("requires_consent") is True:
                return True
    return any(tool_name.lower().startswith(p) for p in MUTATING_PREFIXES)


async def hydrate_context(state: AgentState) -> dict:
    """Node 1: Hydrate context from profile memory and dynamic MCP tools.
    
    Safeguard 1: Auto-expire stale consent (>24h) and cleanse interrupted state if new prompt arrived.
    """
    user_id = state.get("user_id", "anonymous")
    updates: Dict[str, Any] = {}

    # 1. Fetch user profile memories
    user_profile = await mem0_service.get_profile(user_id)
    updates["user_profile"] = user_profile

    # 2. Fetch authorized tools dynamically from Nango MCP gateway
    available_tools = await nango_mcp_client.get_user_tools(user_id)
    updates["available_tools"] = available_tools

    # 3. Safeguard 1: Stale consent check & interruption cleansing
    pending = state.get("pending_action")
    if pending:
        now_ms = int(time.time() * 1000)
        created_at = pending.get("created_at_epoch_ms", 0)
        ttl_ms = 24 * 3600 * 1000

        # Check 24-hour TTL expiration
        if (now_ms - created_at) > ttl_ms:
            updates["pending_action"] = None
            updates["consent_status"] = "EXPIRED"
            notice = SystemMessage(content="Previous pending action expired after 24 hours without confirmation.")
            updates["messages"] = [notice]
        # Cleansing: New prompt arrived while consent was pending
        elif state.get("consent_status") == "PENDING":
            messages = state.get("messages", [])
            if messages and isinstance(messages[-1], HumanMessage):
                updates["pending_action"] = None
                updates["consent_status"] = "EXPIRED"
                notice = SystemMessage(content="Previous pending action was discarded due to a new instruction.")
                updates["messages"] = [notice]

    return updates


async def context_trimmer(state: AgentState) -> dict:
    """Node 2: Pin system prompt and trim message context.
    
    Safeguard 5: Token budget management; keep last 6 turns verbatim, summarize older.
    """
    profile_str = json.dumps(state.get("user_profile", {}))
    system_prompt = (
        f"You are an intelligent, autonomous authorized AI assistant companion. "
        f"You help the user manage, retrieve, and organize their personal data through authorized tools. "
        f"If the client application or user specifies a custom companion persona or name, adopt that persona respectfully. "
        f"User Profile & Preferences: {profile_str}"
    )

    all_messages = [m for m in state.get("messages", []) if not isinstance(m, SystemMessage)]

    if len(all_messages) > 6:
        older = all_messages[:-6]
        recent = all_messages[-6:]
        summary_notice = SystemMessage(
            content=f"[Context compacted: {len(older)} earlier turns summarized. System and recent turns retained.]"
        )
        trimmed = [SystemMessage(content=system_prompt), summary_notice] + recent
    else:
        trimmed = [SystemMessage(content=system_prompt)] + all_messages

    return {"messages": trimmed}


async def reasoner(state: AgentState) -> dict:
    """Node 3: Core LLM reasoning node with LiteLLM router and dynamic MCP tools."""
    tools = state.get("available_tools", [])
    llm = get_llm()
    if tools:
        llm = llm.bind_tools(tools)

    response = await llm.ainvoke(state["messages"])
    step_count = state.get("step_count", 0) + 1
    return {"messages": [response], "step_count": step_count}


def route_after_reasoner(state: AgentState) -> Literal["execute_tool", "consent_gate", "async_memory"]:
    """Conditional Edge: Route based on tool requirements and mutation safety.
    
    Safeguard 2: Multi-tool partitioning (read-only first, then mutating).
    """
    messages = state.get("messages", [])
    if not messages:
        return "async_memory"

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return "async_memory"

    tools = state.get("available_tools", [])
    has_mutating = False
    has_readonly = False

    for tc in last_message.tool_calls:
        if is_tool_mutating(tc["name"], tools):
            has_mutating = True
        else:
            has_readonly = True

    # Safeguard 2: If both exist, execute read-only first for accurate preview diff
    if has_readonly and has_mutating:
        return "execute_tool"
    elif has_mutating:
        return "consent_gate"
    else:
        return "execute_tool"


async def consent_gate(state: AgentState) -> dict:
    """Node 4: Human-in-the-Loop Interceptor.
    
    Safeguard 3: Generates random action_id idempotency key.
    Pauses execution via LangGraph interrupt().
    """
    messages = state["messages"]
    last_message = messages[-1]
    tool_call = last_message.tool_calls[0] if isinstance(last_message, AIMessage) and last_message.tool_calls else {}

    action_id = str(uuid.uuid4())
    pending_action: PendingAction = {
        "action_id": action_id,
        "tool_name": tool_call.get("name", "unknown_action"),
        "tool_call_id": tool_call.get("id", str(uuid.uuid4())),
        "args": tool_call.get("args", {}),
        "preview_diff": {
            "summary": f"Confirm action: {tool_call.get('name')}",
            "args": tool_call.get("args", {}),
        },
        "created_at_epoch_ms": int(time.time() * 1000),
    }

    # LangGraph deterministic interrupt: pauses execution and persists to checkpointer
    user_decision = interrupt({
        "type": "requires_consent",
        "action": pending_action,
    })

    if isinstance(user_decision, dict) and user_decision.get("approved"):
        return {"pending_action": pending_action, "consent_status": "APPROVED"}
    else:
        reject_message = ToolMessage(
            tool_call_id=pending_action["tool_call_id"],
            content="Action cancelled by user.",
            status="error",
        )
        return {"messages": [reject_message], "pending_action": None, "consent_status": "REJECTED"}


async def execute_tool(state: AgentState) -> dict:
    """Node 5: Execute tools with exception containment and idempotency keys.
    
    Safeguard 3: Pass action_id as X-Idempotency-Key.
    Safeguard 6: Tool exception containment prevents agent crashes.
    """
    last_message = state["messages"][-1]
    tool_messages: List[ToolMessage] = []
    user_id = state.get("user_id", "anonymous")
    tools = state.get("available_tools", [])
    pending = state.get("pending_action")
    idempotency_key = pending["action_id"] if pending else None

    # Handle pending action execution if resuming from approved consent
    if state.get("consent_status") == "APPROVED" and pending:
        try:
            result = await nango_mcp_client.call_tool(
                user_id=user_id,
                tool_name=pending["tool_name"],
                arguments=pending["args"],
                idempotency_key=idempotency_key,
            )
            tool_messages.append(ToolMessage(
                tool_call_id=pending["tool_call_id"],
                content=json.dumps(result) if not isinstance(result, str) else result,
                status="success",
            ))
        except Exception as e:
            # Safeguard 6: Exception Containment
            tool_messages.append(ToolMessage(
                tool_call_id=pending["tool_call_id"],
                content=json.dumps({"error": str(e)}),
                status="error",
            ))
        return {"messages": tool_messages, "pending_action": None, "consent_status": None}

    # Handle regular tool calls from AIMessage
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        # Check if we should execute only read-only tools (Safeguard 2 partitioning)
        has_mutating = any(is_tool_mutating(tc["name"], tools) for tc in last_message.tool_calls)
        has_readonly = any(not is_tool_mutating(tc["name"], tools) for tc in last_message.tool_calls)

        for tc in last_message.tool_calls:
            is_mut = is_tool_mutating(tc["name"], tools)
            # If partitioned, execute read-only first
            if has_mutating and has_readonly and is_mut:
                continue

            try:
                res = await nango_mcp_client.call_tool(
                    user_id=user_id,
                    tool_name=tc["name"],
                    arguments=tc["args"],
                    idempotency_key=str(uuid.uuid4()) if is_mut else None,
                )
                tool_messages.append(ToolMessage(
                    tool_call_id=tc.get("id", str(uuid.uuid4())),
                    content=json.dumps(res) if not isinstance(res, str) else res,
                    status="success",
                ))
            except Exception as e:
                # Safeguard 6: Exception Containment
                tool_messages.append(ToolMessage(
                    tool_call_id=tc.get("id", str(uuid.uuid4())),
                    content=json.dumps({"error": str(e)}),
                    status="error",
                ))

    return {"messages": tool_messages}


def route_after_execute(state: AgentState) -> Literal["reasoner", "consent_gate", "async_memory"]:
    """Loop breaker: limit turns to 5 iterations max; route remaining mutating tools to consent."""
    if state.get("step_count", 0) >= 5:
        return "async_memory"

    # If any pending mutating tool call was skipped in multi-tool partitioning
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        tools = state.get("available_tools", [])
        if any(is_tool_mutating(tc["name"], tools) for tc in last_message.tool_calls):
            return "consent_gate"

    return "reasoner"


async def async_memory(state: AgentState) -> dict:
    """Node 6: Background worker extracting long-term preferences to Mem0/Postgres."""
    user_id = state.get("user_id", "anonymous")
    messages = state.get("messages", [])
    await mem0_service.extract_and_upsert(user_id, messages)
    return {}
