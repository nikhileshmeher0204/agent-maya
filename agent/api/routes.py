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
    title="Agent Maya Microservice",
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
    return {"status": "ok", "service": "agent-maya"}


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
        thought_process = ""

        # Emits progressive stream events
        async for event in agent_graph.astream_events(initial_input, config=config, version="v2"):
            kind = event.get("event")

            # 1. Thought progress stream
            if kind == "on_chat_model_start":
                seq_id += 1
                thought_process = "Thinking..."
                yield f"event: thought\nid: {seq_id}\ndata: {json.dumps({'status': thought_process, 'seq': seq_id})}\n\n"

            # 2. Token text streaming
            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                token = getattr(chunk, "content", "") if chunk else ""
                if token:
                    seq_id += 1
                    full_content += token
                    yield f"event: token\nid: {seq_id}\ndata: {json.dumps({'text': token, 'seq': seq_id})}\n\n"

            # 3. Custom events (A2UI GenUI surface or consent card)
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

        # Save message record for Safeguard 4 reconnect hydration
        save_message_record(
            conversation_id=thread_id,
            role="assistant",
            content=full_content,
            sequence_id=seq_id,
            thought_process=thought_process,
        )

        seq_id += 1
        yield f"event: done\nid: {seq_id}\ndata: {json.dumps({'seq': seq_id})}\n\n"

    return StreamingResponse(sse_event_stream(), media_type="text/event-stream")


@app.post("/api/v1/agent/consent")
async def consent_endpoint(
    req: ConsentRequest,
    user: dict = Depends(verify_firebase_jwt),
):
    """Resume an interrupted LangGraph state after Human-in-the-Loop decision."""
    config = {"configurable": {"thread_id": req.conversation_id}}
    state = await agent_graph.aget_state(config)

    if not state.next:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active pending action awaiting consent for this conversation.",
        )

    # Resume the interrupted graph using Command(resume=...)
    resume_payload = {"approved": req.approved, "action_id": req.action_id}
    await agent_graph.aupdate_state(config, Command(resume=resume_payload))

    return {
        "status": "RESUMED",
        "action_id": req.action_id,
        "approved": req.approved,
    }


@app.get("/api/v1/agent/sessions/{conversation_id}/messages")
async def get_session_messages(
    conversation_id: str,
    since_seq: int = Query(default=0, ge=0),
    user: dict = Depends(verify_firebase_jwt),
):
    """Safeguard 4: Hydrate missed messages on reconnection."""
    messages = get_messages_since(conversation_id, since_seq=since_seq)
    return {"conversation_id": conversation_id, "messages": messages}
