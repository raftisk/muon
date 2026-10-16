"""Mock HTTP transport and canned provider responses for the llm tests.

openai 3.x sends requests through `httpx2`, so the mocks use `httpx2.MockTransport`.
"""

from collections.abc import Callable
from typing import Any

import httpx2
from openai import AsyncOpenAI

MOCK_BASE_URL = "https://provider.test/v1"
MOCK_API_KEY = "test-key"
MOCK_MODEL = "test-model"

Handler = Callable[[httpx2.Request], httpx2.Response]


def build_mock_http_client(handler: Handler) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


def build_mock_client(handler: Handler) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=MOCK_BASE_URL,
        api_key=MOCK_API_KEY,
        max_retries=0,
        http_client=build_mock_http_client(handler),
    )


def chat_completion_body(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": MOCK_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "refusal": None},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
    }


def embedding_body(vectors: list[list[float]]) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
        "model": MOCK_MODEL,
        "usage": {"prompt_tokens": len(vectors), "total_tokens": len(vectors)},
    }
