import json
from collections.abc import Callable, Mapping
from typing import Any, Literal

import httpx2
import pytest
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming
from pydantic import BaseModel

from muon.llm.batch import (
    BATCH_INPUT_FILENAME,
    CHAT_ENDPOINT,
    EMBEDDING_ENDPOINT,
    BatchLine,
    BatchSubmitError,
    build_chat_line,
    build_embedding_line,
    submit_batch,
)
from tests.unit.llm.fakes import (
    MOCK_BASE_URL,
    MOCK_MODEL,
    batch_body,
    build_mock_client,
    chat_completion_body,
    file_body,
)

FILES_PATH = "/v1/files"
BATCHES_PATH = "/v1/batches"
INPUT_FILE_ID = "file-input"
BATCH_ID = "batch-1"
MESSAGES: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "How does Diglett evolve?"}
]


class EvolutionTrigger(BaseModel):
    kind: Literal["level", "item", "trade"]
    level: int | None


class EvolutionDraft(BaseModel):
    target: str
    triggers: list[EvolutionTrigger]


def chat_params() -> CompletionCreateParamsNonStreaming:
    return {"model": MOCK_MODEL, "messages": MESSAGES, "temperature": 0}


def route_requests(
    responses: Mapping[str, httpx2.Response], requests: list[httpx2.Request]
) -> Callable[[httpx2.Request], httpx2.Response]:
    def handle_request(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return responses[request.url.path]

    return handle_request


def accept_batch(endpoint: str) -> dict[str, httpx2.Response]:
    return {
        FILES_PATH: httpx2.Response(200, json=file_body(INPUT_FILE_ID)),
        BATCHES_PATH: httpx2.Response(
            200, json=batch_body(BATCH_ID, "validating", endpoint=endpoint)
        ),
    }


def decode_json(request: httpx2.Request) -> Any:
    return json.loads(request.content)


async def test_chat_line_response_format_matches_parse() -> None:
    requests: list[httpx2.Request] = []

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        draft = EvolutionDraft(target="dugtrio", triggers=[{"kind": "level", "level": 26}])
        return httpx2.Response(200, json=chat_completion_body(draft.model_dump_json()))

    async with build_mock_client(handle_request) as client:
        await client.chat.completions.parse(**chat_params(), response_format=EvolutionDraft)

    line = build_chat_line("diglett", chat_params(), EvolutionDraft)

    sent_by_parse = decode_json(requests[0])["response_format"]
    assert json.loads(line.model_dump_json())["body"]["response_format"] == sent_by_parse
    assert sent_by_parse["json_schema"]["strict"] is True


def test_chat_line_without_model_carries_params() -> None:
    line = build_chat_line("diglett", chat_params())

    assert line.method == "POST"
    assert line.url == CHAT_ENDPOINT
    assert line.body == dict(chat_params())


def test_embedding_line_targets_embeddings() -> None:
    line = build_embedding_line("chunk-1", {"model": MOCK_MODEL, "input": ["text"]})

    assert line.url == EMBEDDING_ENDPOINT
    assert line.body == {"model": MOCK_MODEL, "input": ["text"]}


async def test_submit_uploads_jsonl_and_creates_batch() -> None:
    requests: list[httpx2.Request] = []
    lines = [
        build_chat_line("diglett", chat_params(), EvolutionDraft),
        build_chat_line("meowth", chat_params(), EvolutionDraft),
    ]

    handler = route_requests(accept_batch(CHAT_ENDPOINT), requests)
    async with build_mock_client(handler) as client:
        batch = await submit_batch(client, lines)

    upload, create = requests
    assert batch.id == BATCH_ID
    assert upload.url.path == FILES_PATH
    assert BATCH_INPUT_FILENAME.encode() in upload.content
    assert b'name="purpose"\r\n\r\nbatch' in upload.content
    for line in lines:
        assert f"{line.model_dump_json()}\n".encode() in upload.content
    assert decode_json(create) == {
        "input_file_id": INPUT_FILE_ID,
        "endpoint": CHAT_ENDPOINT,
        "completion_window": "24h",
    }


async def test_submit_embedding_lines_targets_embeddings() -> None:
    requests: list[httpx2.Request] = []
    lines = [
        build_embedding_line(f"chunk-{index}", {"model": MOCK_MODEL, "input": [f"text {index}"]})
        for index in range(2)
    ]

    handler = route_requests(accept_batch(EMBEDDING_ENDPOINT), requests)
    async with build_mock_client(handler) as client:
        await submit_batch(client, lines)

    assert decode_json(requests[1])["endpoint"] == EMBEDDING_ENDPOINT


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([], "no lines"),
        (
            [build_chat_line("diglett", chat_params()), build_chat_line("diglett", chat_params())],
            "duplicate custom ids: \\['diglett'\\]",
        ),
        (
            [
                build_chat_line("diglett", chat_params()),
                build_embedding_line("chunk-1", {"model": MOCK_MODEL, "input": ["text"]}),
            ],
            "mixes endpoints",
        ),
    ],
)
async def test_invalid_lines_fail_before_any_request(lines: list[BatchLine], message: str) -> None:
    requests: list[httpx2.Request] = []

    handler = route_requests(accept_batch(CHAT_ENDPOINT), requests)
    async with build_mock_client(handler) as client:
        with pytest.raises(BatchSubmitError, match=message):
            await submit_batch(client, lines)

    assert requests == []


async def test_server_without_files_route_fails_naming_base_url() -> None:
    requests: list[httpx2.Request] = []
    responses = {FILES_PATH: httpx2.Response(404, json={"error": {"message": "not found"}})}

    async with build_mock_client(route_requests(responses, requests)) as client:
        with pytest.raises(BatchSubmitError, match="upload batch input of 1 lines") as caught:
            await submit_batch(client, [build_chat_line("diglett", chat_params())])

    assert MOCK_BASE_URL in str(caught.value)
    assert "404" in str(caught.value)


async def test_rejected_embedding_batch_fails_naming_endpoint() -> None:
    requests: list[httpx2.Request] = []
    responses = {
        FILES_PATH: httpx2.Response(200, json=file_body(INPUT_FILE_ID)),
        BATCHES_PATH: httpx2.Response(400, json={"error": {"message": "endpoint not supported"}}),
    }
    lines = [build_embedding_line("chunk-1", {"model": MOCK_MODEL, "input": ["text"]})]

    async with build_mock_client(route_requests(responses, requests)) as client:
        with pytest.raises(BatchSubmitError, match=EMBEDDING_ENDPOINT) as caught:
            await submit_batch(client, lines)

    assert INPUT_FILE_ID in str(caught.value)
    assert "endpoint not supported" in str(caught.value)
