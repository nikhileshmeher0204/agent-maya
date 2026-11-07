import json
from typing import AsyncGenerator
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, status

load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.api.auth import verify_firebase_jwt
from agent.db.init import get_messages_since, save_message_record
from agent.graph.graph import agent_graph

app = FastAPI(
    title="Agent Synapse Microservice",
    description="Autonomous, plug-and-play multi-tenant agentic microservice",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    conversation_id: str = Field(..., description="UUID of the chat session")
    message: str = Field(..., description="User prompt text")
    app_id: str = Field(default="default", description="Calling client application identifier")


class ConsentRequest(BaseModel):
    conversation_id: str = Field(..., description="UUID of the chat session")
    action_id: str = Field(..., description="UUID of the intercepted action")
    approved: bool = Field(..., description="True to execute, False to reject")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "agent-synapse"}


def extract_tool_description(tool_name: str, args: dict) -> tuple[str, str, str]:
    """Return (in_progress_status, step_log, summary) for a tool call."""
    if tool_name == "search_personal_library":
        q = args.get("query", "").strip()
        status = f"Searching library for '{q}'..." if q else "Searching personal library..."
        log = f"Searched library for '{q}'" if q else "Searched personal library"
        summary = f"Searched library for '{q}'" if q else "Searched personal library"
        return status, log, summary
    elif tool_name == "list_user_collections":
        return "Inspecting collections...", "Inspected personal collections", "Inspected personal collections"
    elif tool_name == "create_personal_collection":
        name = args.get("name", "").strip()
        status = f"Preparing collection '{name}'..." if name else "Preparing new collection..."
        log = f"Created collection '{name}'" if name else "Created collection"
        summary = f"Created collection '{name}'" if name else "Created collection"
        return status, log, summary
    elif tool_name == "add_entities_to_collection":
        return "Adding item to collection...", "Added item to collection", "Added item to collection"
    elif tool_name == "delete_personal_collection":
        return "Deleting collection...", "Deleted collection", "Deleted collection"
    else:
        clean_name = tool_name.replace("_", " ").title()
        return f"Executing {clean_name}...", f"Executed {clean_name}", f"Executed {clean_name}"


@app.post("/api/v1/agent/chat")
async def chat_endpoint(
    req: ChatRequest,
    user: dict = Depends(verify_firebase_jwt),
):
    """Server-Sent Events (SSE) streaming chat endpoint.
    
    Adheres to:
    - Invariant 1: Application-agnostic, zero hardcoded app logic.
    - Invariant 4: Zero tokens in request body (auth strictly in transport header).
    - Safeguard 4: Monotonic sequence IDs for reconnection hydration.
    """
    user_id = user["uid"]
    thread_id = req.conversation_id
    config = {"configurable": {"thread_id": thread_id}}

    initial_input = {
        "app_id": req.app_id,
        "user_id": user_id,
        "session_id": thread_id,
        "messages": [HumanMessage(content=req.message)],
        "step_count": 0,
    }

    async def sse_event_stream() -> AsyncGenerator[str, None]:
        seq_id = 0
        full_content = ""
        current_status = "Analyzing request..."
        step_logs: List[str] = []
        tool_summary: Optional[str] = None
        executed_tools = False

        # Emits progressive stream events
        async for event in agent_graph.astream_events(initial_input, config=config, version="v2"):
            kind = event.get("event")
            node = event.get("metadata", {}).get("langgraph_node")

            # 1. Real LangGraph node state transitions
            if kind == "on_chain_start" and node:
                if node == "hydrate_context":
                    current_status = "Retrieving user context & authorized tools..."
                    if "Retrieved user profile & authorized tools" not in step_logs:
                        step_logs.append("Retrieved user profile & authorized tools")
                elif node == "context_trimmer":
                    current_status = "Analyzing conversation memory..."
                    if "Analyzed conversation memory" not in step_logs:
                        step_logs.append("Analyzed conversation memory")
                elif node == "reasoner":
                    if executed_tools:
                        current_status = "Synthesizing response from findings..."
                    else:
                        current_status = "Reasoning over request..."
                elif node == "consent_gate":
                    current_status = "Evaluating action authorization & safety..."
                    if "Evaluated action authorization" not in step_logs:
                        step_logs.append("Evaluated action authorization")
                elif node == "execute_tool":
                    current_status = "Executing authorized tools..."
                    executed_tools = True
                elif node == "async_memory":
                    current_status = "Updating knowledge graph..."
                    if "Updated knowledge graph" not in step_logs:
                        step_logs.append("Updated knowledge graph")

                seq_id += 1
                yield f"event: thought\nid: {seq_id}\ndata: {json.dumps({'status': current_status, 'logs': step_logs, 'summary': tool_summary, 'seq': seq_id})}\n\n"

            # 2. Tool calls planned by the LLM
            elif kind == "on_chat_model_end":
                output = event.get("data", {}).get("output")
                if output and hasattr(output, "tool_calls") and output.tool_calls:
                    for tc in output.tool_calls:
                        t_name = tc.get("name", "")
                        t_args = tc.get("args", {})
                        status_str, log_str, summ_str = extract_tool_description(t_name, t_args)
                        current_status = status_str
                        if log_str not in step_logs:
                            step_logs.append(log_str)
                        tool_summary = summ_str
                        seq_id += 1
                        yield f"event: thought\nid: {seq_id}\ndata: {json.dumps({'status': current_status, 'logs': step_logs, 'summary': tool_summary, 'seq': seq_id})}\n\n"

            # 3. Token text streaming
            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                token = getattr(chunk, "content", "") if chunk else ""
                if token:
                    seq_id += 1
                    full_content += token
                    yield f"event: token\nid: {seq_id}\ndata: {json.dumps({'text': token, 'seq': seq_id})}\n\n"

            # 4. Custom events (A2UI GenUI surface or consent card)
            elif kind == "on_custom_event":
                event_name = event.get("name")
                if event_name in ("surface_added", "requires_consent"):
                    seq_id += 1
                    payload = event.get("data", {})
                    payload["seq"] = seq_id
                    yield f"event: {event_name}\nid: {seq_id}\ndata: {json.dumps(payload)}\n\n"

        # Check if the graph execution ended at an interrupt (e.g. ConsentGate)
        curr_state = await agent_graph.aget_state(config)
        if curr_state.tasks and any(t.interrupts for t in curr_state.tasks):
            for task in curr_state.tasks:
                for intr in task.interrupts:
                    seq_id += 1
                    yield f"event: requires_consent\nid: {seq_id}\ndata: {json.dumps({'interrupt': intr.value, 'seq': seq_id})}\n\n"

        # Check final state messages to capture non-streamed assistant response
        if not full_content and curr_state.values.get("messages"):
            last_msg = curr_state.values["messages"][-1]
            if isinstance(last_msg, AIMessage) and last_msg.content:
                seq_id += 1
                full_content = str(last_msg.content)
                yield f"event: token\nid: {seq_id}\ndata: {json.dumps({'text': full_content, 'seq': seq_id})}\n\n"

        # Final thought state
        final_summary = tool_summary or ("Completed reasoning" if step_logs else "Completed")
        seq_id += 1
        yield f"event: thought\nid: {seq_id}\ndata: {json.dumps({'status': final_summary, 'logs': step_logs, 'summary': final_summary, 'seq': seq_id})}\n\n"

        # Save message record for Safeguard 4 reconnect hydration
        save_message_record(
            conversation_id=thread_id,
            role="assistant",
            content=full_content,
            sequence_id=seq_id,
            thought_process=final_summary,
        )

        seq_id += 1
        yield f"event: done\nid: {seq_id}\ndata: {json.dumps({'seq': seq_id})}\n\n"

    return StreamingResponse(sse_event_stream(), media_type="text/event-stream")


@app.post("/api/v1/agent/consent")
async def consent_endpoint(
    req: ConsentRequest,
    user: dict = Depends(verify_firebase_jwt),
):
    """Resume an interrupted LangGraph state after Human-in-the-Loop decision with SSE streaming."""
    thread_id = req.conversation_id
    config = {"configurable": {"thread_id": thread_id}}
    state = await agent_graph.aget_state(config)

    if not state.next:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active pending action awaiting consent for this conversation.",
        )

    resume_command = Command(resume={"approved": req.approved, "action_id": req.action_id})

    async def sse_event_stream() -> AsyncGenerator[str, None]:
        seq_id = 0
        full_content = ""
        current_status = "Executing approved action..." if req.approved else "Action cancelled by user"
        step_logs: List[str] = [f"Action {'approved' if req.approved else 'declined'} by user"]
        tool_summary: Optional[str] = f"{'Executed' if req.approved else 'Cancelled'} action"
        executed_tools = req.approved

        # Initial thought for resumed action
        seq_id += 1
        yield f"event: thought\nid: {seq_id}\ndata: {json.dumps({'status': current_status, 'logs': step_logs, 'summary': tool_summary, 'seq': seq_id})}\n\n"

        async for event in agent_graph.astream_events(resume_command, config=config, version="v2"):
            kind = event.get("event")
            node = event.get("metadata", {}).get("langgraph_node")

            # 1. Real LangGraph node state transitions
            if kind == "on_chain_start" and node:
                if node == "execute_tool":
                    current_status = "Executing authorized tool..."
                    executed_tools = True
                elif node == "reasoner":
                    current_status = "Synthesizing response from findings..."
                elif node == "async_memory":
                    current_status = "Updating knowledge graph..."
                    if "Updated knowledge graph" not in step_logs:
                        step_logs.append("Updated knowledge graph")

                seq_id += 1
                yield f"event: thought\nid: {seq_id}\ndata: {json.dumps({'status': current_status, 'logs': step_logs, 'summary': tool_summary, 'seq': seq_id})}\n\n"

            # 2. Tool calls if any
            elif kind == "on_chat_model_end":
                output = event.get("data", {}).get("output")
                if output and hasattr(output, "tool_calls") and output.tool_calls:
                    for tc in output.tool_calls:
                        t_name = tc.get("name", "")
                        t_args = tc.get("args", {})
                        status_str, log_str, summ_str = extract_tool_description(t_name, t_args)
                        current_status = status_str
                        if log_str not in step_logs:
                            step_logs.append(log_str)
                        tool_summary = summ_str
                        seq_id += 1
                        yield f"event: thought\nid: {seq_id}\ndata: {json.dumps({'status': current_status, 'logs': step_logs, 'summary': tool_summary, 'seq': seq_id})}\n\n"

            # 3. Token text streaming
            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                token = getattr(chunk, "content", "") if chunk else ""
                if token:
                    seq_id += 1
                    full_content += token
                    yield f"event: token\nid: {seq_id}\ndata: {json.dumps({'text': token, 'seq': seq_id})}\n\n"

            # 4. Custom events
            elif kind == "on_custom_event":
                event_name = event.get("name")
                if event_name in ("surface_added", "requires_consent"):
                    seq_id += 1
                    payload = event.get("data", {})
                    payload["seq"] = seq_id
                    yield f"event: {event_name}\nid: {seq_id}\ndata: {json.dumps(payload)}\n\n"

        curr_state = await agent_graph.aget_state(config)
        if not full_content and curr_state.values.get("messages"):
            last_msg = curr_state.values["messages"][-1]
            if isinstance(last_msg, AIMessage) and last_msg.content:
                seq_id += 1
                full_content = str(last_msg.content)
                yield f"event: token\nid: {seq_id}\ndata: {json.dumps({'text': full_content, 'seq': seq_id})}\n\n"

        final_summary = tool_summary or ("Completed reasoning" if step_logs else "Completed")
        seq_id += 1
        yield f"event: thought\nid: {seq_id}\ndata: {json.dumps({'status': final_summary, 'logs': step_logs, 'summary': final_summary, 'seq': seq_id})}\n\n"

        save_message_record(
            conversation_id=thread_id,
            role="assistant",
            content=full_content,
            sequence_id=seq_id,
            thought_process=final_summary,
        )

        seq_id += 1
        yield f"event: done\nid: {seq_id}\ndata: {json.dumps({'seq': seq_id})}\n\n"

    return StreamingResponse(sse_event_stream(), media_type="text/event-stream")


@app.get("/api/v1/agent/sessions/{conversation_id}/messages")
async def get_session_messages(
    conversation_id: str,
    since_seq: int = Query(default=0, ge=0),
    user: dict = Depends(verify_firebase_jwt),
):
    """Safeguard 4: Hydrate missed messages on reconnection."""
    messages = get_messages_since(conversation_id, since_seq=since_seq)
    return {"conversation_id": conversation_id, "messages": messages}
