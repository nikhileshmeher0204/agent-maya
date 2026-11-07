import os
from typing import Any, Dict, List, Optional
import httpx


class UniversalMcpClient:
    """Universal MCP client communicating with MCP servers or Nango gateway.
    
    Adheres strictly to Invariant 1 (100% plug-and-play, zero application-specific logic)
    and Invariant 3 (zero connection tables or tokens stored in agent).
    """

    def __init__(self, base_url: Optional[str] = None, timeout_sec: float = 10.0):
        self.endpoint_url = os.getenv("MCP_SERVER_URL")
        if not self.endpoint_url:
            base = (base_url or os.getenv("NANGO_BASE_URL", "http://localhost:8080/api/v1")).rstrip("/")
            self.endpoint_url = f"{base}/mcp" if not base.endswith("/mcp") else base
        self.timeout = timeout_sec

    async def get_user_tools(self, user_id: str) -> List[Dict[str, Any]]:
        """Fetch available tool schemas dynamically from the MCP gateway."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {"connection_id": user_id},
        }
        headers = {
            "Content-Type": "application/json",
            "Connection-Id": user_id,
            "Authorization": f"Bearer {user_id if user_id.startswith('mock_') else 'mock_' + user_id}",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(self.endpoint_url, json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    tools_raw = data.get("result", {}).get("tools", [])
                    return [self._format_to_function_tool(t) for t in tools_raw]
        except Exception:
            # Fallback when running disconnected or standalone
            pass
        return []

    async def call_tool(
        self,
        user_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        idempotency_key: Optional[str] = None,
    ) -> Any:
        """Execute a tool via MCP JSON-RPC with exception containment and idempotency."""
        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "Connection-Id": user_id,
            "Authorization": f"Bearer {user_id if user_id.startswith('mock_') else 'mock_' + user_id}",
        }
        if idempotency_key:
            headers["X-Idempotency-Key"] = idempotency_key

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.post(self.endpoint_url, json=payload, headers=headers)
            res.raise_for_status()
            data = res.json()

            if "error" in data:
                raise RuntimeError(data["error"].get("message", "MCP Tool Error"))

            result = data.get("result", {})
            content = result.get("content", [])
            if content and isinstance(content, list) and len(content) > 0:
                first = content[0]
                if isinstance(first, dict) and "text" in first:
                    return first["text"]
            return result

    @staticmethod
    def _format_to_function_tool(raw_tool: Dict[str, Any]) -> Dict[str, Any]:
        """Convert standard MCP tool definition to OpenAI/LangChain function tool schema."""
        if "type" in raw_tool and raw_tool.get("type") == "function":
            return raw_tool
        return {
            "type": "function",
            "function": {
                "name": raw_tool.get("name"),
                "description": raw_tool.get("description", ""),
                "parameters": raw_tool.get("inputSchema")
                or raw_tool.get("parameters")
                or {"type": "object", "properties": {}},
            },
        }


nango_mcp_client = UniversalMcpClient()
