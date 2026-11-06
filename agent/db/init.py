import os
from typing import Any, Dict, List, Optional
import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

INIT_DDL = """
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(128) NOT NULL,
    app_id VARCHAR(64) NOT NULL DEFAULT 'default',
    title VARCHAR(255) NOT NULL DEFAULT 'New Conversation',
    summary TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversations_user_app ON conversations(user_id, app_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL,
    role VARCHAR(32) NOT NULL,
    content TEXT NOT NULL,
    thought_process TEXT,
    a2ui_surfaces JSONB,
    tool_calls JSONB,
    sequence_id INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_seq ON messages(conversation_id, sequence_id ASC);

CREATE TABLE IF NOT EXISTS user_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(128) NOT NULL,
    memory_type VARCHAR(64) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_memories_user_id ON user_memories(user_id);
"""


async def init_db(db_url: Optional[str] = None) -> None:
    """Run DDL migrations for conversations, messages, and user_memories."""
    url = db_url or os.getenv("DATABASE_URL")
    if not url:
        return
    async with await psycopg.AsyncConnection.connect(url) as aconn:
        async with aconn.cursor() as cur:
            await cur.execute(INIT_DDL)
        await aconn.commit()


# In-memory session message store for local dev / tests when Postgres is not running
_memory_messages_store: Dict[str, List[Dict[str, Any]]] = {}


def save_message_record(
    conversation_id: str,
    role: str,
    content: str,
    sequence_id: int,
    thought_process: Optional[str] = None,
    a2ui_surfaces: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Store message record in memory and/or database."""
    record = {
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "thought_process": thought_process,
        "a2ui_surfaces": a2ui_surfaces or [],
        "sequence_id": sequence_id,
    }
    _memory_messages_store.setdefault(conversation_id, []).append(record)


def get_messages_since(conversation_id: str, since_seq: int = 0) -> List[Dict[str, Any]]:
    """Retrieve message records for reconnect recovery (Safeguard 4)."""
    records = _memory_messages_store.get(conversation_id, [])
    return [r for r in records if r["sequence_id"] > since_seq]
