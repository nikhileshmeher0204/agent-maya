import json
import os
from typing import Any, Dict, List, Optional
import litellm
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult


def _convert_message_to_dict(message: BaseMessage) -> Dict[str, Any]:
    """Convert LangChain message to LiteLLM/OpenAI standard format."""
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": str(message.content),
        }
    elif isinstance(message, AIMessage):
        msg_dict: Dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.get("id", f"call_{i}"),
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"]) if isinstance(tc["args"], dict) else tc["args"],
                    },
                }
                for i, tc in enumerate(message.tool_calls)
            ]
        return msg_dict
    else:
        return {"role": "user", "content": str(message.content)}


class ChatLiteLLMRouted(BaseChatModel):
    """LiteLLM routed chat model with automatic 429 failover.
    
    Primary: Gemini 2.5 Flash
    Standby: Claude 3.5 Haiku
    """

    model: str = "gemini/gemini-2.5-flash"
    fallback_models: List[str] = ["anthropic/claude-3-5-haiku-20241022"]
    bound_tools: List[Dict[str, Any]] = []
    temperature: float = 0.2

    @property
    def _llm_type(self) -> str:
        return "litellm-routed"

    def bind_tools(self, tools: List[Dict[str, Any]], **kwargs: Any) -> "ChatLiteLLMRouted":
        """Return a clone of this model with tools bound."""
        return ChatLiteLLMRouted(
            model=self.model,
            fallback_models=self.fallback_models,
            bound_tools=tools,
            temperature=self.temperature,
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        import asyncio
        return asyncio.run(self._agenerate(messages, stop=stop, run_manager=None, **kwargs))

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        formatted = [_convert_message_to_dict(m) for m in messages]
        call_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": formatted,
            "fallbacks": self.fallback_models,
            "temperature": self.temperature,
        }
        if self.bound_tools:
            call_kwargs["tools"] = self.bound_tools
            call_kwargs["tool_choice"] = "auto"
        if stop:
            call_kwargs["stop"] = stop

        # Execute via LiteLLM with automatic quota/rate-limit fallback
        response = await litellm.acompletion(**call_kwargs)
        choice = response.choices[0].message
        content = choice.content or ""
        tool_calls: List[Dict[str, Any]] = []

        if hasattr(choice, "tool_calls") and choice.tool_calls:
            for tc in choice.tool_calls:
                func = tc.function
                args = func.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                tool_calls.append({
                    "id": tc.id,
                    "name": func.name,
                    "args": args,
                })

        ai_msg = AIMessage(content=content, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])


def get_llm() -> ChatLiteLLMRouted:
    """Factory providing the configured routed LLM instance."""
    return ChatLiteLLMRouted()
