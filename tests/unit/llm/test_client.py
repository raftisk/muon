import json

import httpx2
import openai
import pytest
from pydantic import BaseModel, SecretStr

from muon.config import EmbeddingSettings, LLMSettings
from muon.llm import create_client
from tests.unit.llm.fakes import build_mock_http_client, chat_completion_body, embedding_body

CHAT_URL = "https://chat.test/v1"
EMBEDDING_URL = "https://embed.test/v1"
CHAT_KEY = "chat-key"
EMBEDDING_KEY = "embed-key"
CHAT_MODEL = "chat-model"
EMBEDDING_MODEL = "embed-model"
TIMEOUT_SECONDS = 12.5
MESSAGES: list[openai.types.chat.ChatCompletionMessageParam] = [
    {"role": "user", "content": "What is the capital of France?"}
]


class CountryCapital(BaseModel):
    country: str
    capital: str


def build_llm_settings(max_retries: int = 0) -> LLMSettings:
    return LLMSettings(
        _env_file=None,
        base_url=CHAT_URL,
        api_key=SecretStr(CHAT_KEY),
        model=CHAT_MODEL,
        timeout=TIMEOUT_SECONDS,
        max_retries=max_retries,
    )


def build_embedding_settings() -> EmbeddingSettings:
    return EmbeddingSettings(
        _env_file=None,
        base_url=EMBEDDING_URL,
        api_key=SecretStr(EMBEDDING_KEY),
        model=EMBEDDING_MODEL,
        timeout=TIMEOUT_SECONDS,
        max_retries=0,
        dimensions=4,
    )


def test_client_applies_settings() -> None:
    client = create_client(build_llm_settings(max_retries=1))

    assert str(client.base_url) == f"{CHAT_URL}/"
    assert client.api_key == CHAT_KEY
    assert client.timeout == TIMEOUT_SECONDS
    assert client.max_retries == 1


def test_settings_groups_give_separate_clients() -> None:
    chat = create_client(build_llm_settings())
    embedding = create_client(build_embedding_settings())

    assert (str(chat.base_url), chat.api_key) == (f"{CHAT_URL}/", CHAT_KEY)
    assert (str(embedding.base_url), embedding.api_key) == (f"{EMBEDDING_URL}/", EMBEDDING_KEY)


async def test_text_call_reaches_configured_endpoint() -> None:
    requests: list[httpx2.Request] = []

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=chat_completion_body("Paris"))

    settings = build_llm_settings()
    client = create_client(settings).copy(http_client=build_mock_http_client(handle_request))
    async with client:
        completion = await client.chat.completions.create(model=settings.model, messages=MESSAGES)

    assert completion.choices[0].message.content == "Paris"
    assert str(requests[0].url) == f"{CHAT_URL}/chat/completions"
    assert requests[0].headers["Authorization"] == f"Bearer {CHAT_KEY}"
    assert json.loads(requests[0].content)["model"] == CHAT_MODEL


async def test_parse_returns_validated_model() -> None:
    requests: list[httpx2.Request] = []
    answer = CountryCapital(country="France", capital="Paris")

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=chat_completion_body(answer.model_dump_json()))

    settings = build_llm_settings()
    client = create_client(settings).copy(http_client=build_mock_http_client(handle_request))
    async with client:
        completion = await client.chat.completions.parse(
            model=settings.model, messages=MESSAGES, response_format=CountryCapital
        )

    response_format = json.loads(requests[0].content)["response_format"]
    assert completion.choices[0].message.parsed == answer
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True


async def test_json_schema_rejection_raises_sdk_error() -> None:
    def handle_request(request: httpx2.Request) -> httpx2.Response:
        error = {"message": "response_format json_schema is not supported", "type": "invalid"}
        return httpx2.Response(400, json={"error": error})

    settings = build_llm_settings()
    client = create_client(settings).copy(http_client=build_mock_http_client(handle_request))
    async with client:
        with pytest.raises(openai.BadRequestError, match="json_schema is not supported"):
            await client.chat.completions.parse(
                model=settings.model, messages=MESSAGES, response_format=CountryCapital
            )


async def test_rate_limit_is_retried_by_sdk() -> None:
    responses = [
        httpx2.Response(429, headers={"retry-after-ms": "1"}, json={"error": {"message": "slow"}}),
        httpx2.Response(200, json=chat_completion_body("Paris")),
    ]
    requests: list[httpx2.Request] = []

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return responses[len(requests) - 1]

    settings = build_llm_settings(max_retries=1)
    client = create_client(settings).copy(http_client=build_mock_http_client(handle_request))
    async with client:
        completion = await client.chat.completions.create(model=settings.model, messages=MESSAGES)

    assert completion.choices[0].message.content == "Paris"
    assert len(requests) == 2


async def test_embeddings_call_returns_indexed_vectors() -> None:
    requests: list[httpx2.Request] = []
    vectors = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=embedding_body(vectors))

    settings = build_embedding_settings()
    client = create_client(settings).copy(http_client=build_mock_http_client(handle_request))
    async with client:
        response = await client.embeddings.create(
            model=settings.model, input=["first", "second"], dimensions=settings.dimensions
        )

    assert [item.index for item in response.data] == [0, 1]
    assert [item.embedding for item in response.data] == vectors
    assert str(requests[0].url) == f"{EMBEDDING_URL}/embeddings"
    assert json.loads(requests[0].content)["dimensions"] == 4
