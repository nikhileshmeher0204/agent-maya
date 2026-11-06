import os
from typing import Any, Dict, List
from langchain_core.messages import BaseMessage


class ProfileMemoryService:
    """Manages long-term user profile memories (preferences, facts, constraints)."""

    def __init__(self, db_url: str | None = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")

    async def get_profile(self, user_id: str) -> Dict[str, Any]:
        """Fetch consolidated user preference memory profile."""
        if not self.db_url:
            return {}
        try:
            import psycopg
            async with await psycopg.AsyncConnection.connect(self.db_url) as aconn:
                async with aconn.cursor() as cur:
                    await cur.execute(
                        "SELECT memory_type, content FROM user_memories WHERE user_id = %s ORDER BY updated_at DESC LIMIT 50",
                        (user_id,),
                    )
                    rows = await cur.fetchall()
                    profile: Dict[str, Any] = {}
                    for m_type, content in rows:
                        profile.setdefault(m_type, []).append(content)
                    return profile
        except Exception:
            return {}

    async def extract_and_upsert(self, user_id: str, messages: List[BaseMessage]) -> None:
        """Asynchronously extract facts or preferences from recent conversation and upsert."""
        if not self.db_url or not messages:
            return
        # Background memory extraction can be processed via lightweight rule or LLM
        # Stored in postgres user_memories


mem0_service = ProfileMemoryService()
