# Agent Maya (`agent-maya`)
## Autonomous, Plug-and-Play Multi-Tenant Agentic Microservice ("Brain-as-a-Service")

Agent Maya is an application-agnostic, multi-tenant AI companion microservice built with **FastAPI**, **LangGraph**, **LiteLLM**, and the **Model Context Protocol (MCP)**. It serves as an authorized reasoning engine for connected client ecosystems while remaining 100% decoupled from underlying domain backends.

---

## 🏛️ Ecosystem Invariants

1. **100% Plug-and-Play Agent Microservice**: Zero application-specific models, imports, or hardcoded logic. Tools are dynamically discovered at runtime via MCP (`tools/list` and `tools/call`).
2. **Strict Application Decoupling**: Peer applications remain completely isolated. Maya interacts with external tools via standard JSON-RPC 2.0 MCP interfaces.
3. **Zero Agent Involvement in Linking or Auth**: Maya is strictly a stateless reasoning engine and tool executor. OAuth redirect flows, token encryption (AES-256-GCM), and token refresh loops are delegated to **Nango**.
4. **Zero Tokens / Credentials in Request Payloads**: Chat payloads are strictly `{"conversation_id": "...", "message": "..."}`. Authentication is handled statelessly via transport headers.

---

## 🚀 Architecture & Tech Stack

- **Framework**: Python 3.12, FastAPI, LangGraph (cyclic graph with deterministic `interrupt()` checkpointing).
- **LLM Routing**: Primary **Gemini 2.5 Flash** with automatic failover to **Claude 3.5 Haiku** via LiteLLM.
- **Persistence**: PostgreSQL 16 + `pgvector` with `langgraph-checkpoint-postgres`.
- **Memory**: Long-term user preference memory profile graph.
- **Streaming & UI Protocol**: SSE (Server-Sent Events) emitting A2UI (Agent-to-UI) surface mounting envelopes for dynamic native client rendering.

---

## 🛡️ Production Safeguards

1. **Stale Consent Auto-Expiration**: 24-hour TTL on pending mutating actions; automatic clearing upon incoming user prompts.
2. **Multi-Tool Execution Partitioning**: Read-only tools execute first; mutating tools pause for explicit user consent with visual diff previews.
3. **Mutation Idempotency Keys**: Cryptographically random `action_id` forwarded as `X-Idempotency-Key` to prevent duplicate writes.
4. **SSE Monotonic Sequence IDs**: Integer sequence tracking (`seq: N`) with reconnection catchup via `GET /messages?since_seq=N`.
5. **Token Budget & Context Trimming**: Preserves pinned system prompts and the latest 6 turns; condenses older messages into rolling summaries.
6. **Tool Exception Containment**: Downstream timeouts and 5xx errors are caught and surfaced to the LLM as observation errors rather than crashing the turn.

---

## 📦 Getting Started

### Local Development

```bash
# Clone the repository
git clone https://github.com/nikhileshmeher0204/agent-maya.git
cd agent-maya

# Install dependencies using uv
uv venv
uv pip install -e ".[dev]"

# Run tests
uv run pytest
```

### Docker Deployment

```bash
docker compose up -d
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
