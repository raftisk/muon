"""A scripted stand-in for `ChatOpenAI`: no network, replies in a fixed order."""

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, PrivateAttr

DEFAULT_USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def build_usage(
    input_tokens: int, output_tokens: int, reasoning: int = 0, cache_read: int = 0
) -> dict[str, Any]:
    """A `usage_metadata` dict with reasoning and cache-read token details."""
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_token_details": {"cache_read": cache_read},
        "output_token_details": {"reasoning": reasoning},
    }


def build_tool_call_message(
    query: str, call_id: str, purpose: str = "test", is_final: bool | None = None
) -> AIMessage:
    """An AI message that calls `run_cypher` once; `is_final` is sent only when it is set."""
    args: dict[str, Any] = {"query": query, "purpose": purpose}
    if is_final is not None:
        args["is_final"] = is_final
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "run_cypher",
                "args": args,
                "id": call_id,
                "type": "tool_call",
            }
        ],
        usage_metadata=DEFAULT_USAGE,  # type: ignore[arg-type]
    )


def build_final_message(text: str = "done") -> AIMessage:
    return AIMessage(content=text, usage_metadata=DEFAULT_USAGE)  # type: ignore[arg-type]


class ScriptedChatModel(BaseChatModel):
    """Returns `replies` in order from `invoke`, and `drafts` in order from structured output.

    `calls` records the messages of every call, so tests can read what the model was sent.
    """

    replies: list[AIMessage] = []
    drafts: list[BaseModel] = []
    _reply_index: int = PrivateAttr(default=0)
    _draft_index: int = PrivateAttr(default=0)
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def calls(self) -> list[list[BaseMessage]]:
        return self._calls

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._calls.append(list(messages))
        reply = self.replies[self._reply_index]
        self._reply_index += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable[Any, AIMessage]:  # type: ignore[override]
        return self  # type: ignore[return-value]

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:  # type: ignore[override]
        def answer(messages: list[BaseMessage]) -> dict[str, Any]:
            self._calls.append(list(messages))
            draft = self.drafts[self._draft_index]
            self._draft_index += 1
            return {
                "raw": AIMessage(content="", usage_metadata=DEFAULT_USAGE),  # type: ignore[arg-type]
                "parsed": draft,
                "parsing_error": None,
            }

        return RunnableLambda(answer)
