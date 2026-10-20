import json
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Literal

import httpx2
import pytest
from openai.types import Batch, CreateEmbeddingResponse
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import BaseModel

from muon.llm import (
    BatchOutput,
    BatchOutputError,
    BatchStateError,
    LLMBatchError,
    fetch_batch_results,
    parse_chat_output,
    parse_embedding_output,
    wait_for_batch,
)
from tests.unit.llm.fakes import (
    MOCK_BASE_URL,
    MOCK_MODEL,
    batch_body,
    build_mock_client,
    chat_completion_body,
    embedding_body,
    error_line,
    output_line,
)

BATCH_ID = "batch-1"
OUTPUT_FILE_ID = "file-output"
ERROR_FILE_ID = "file-error"
BATCH_PATH = f"/v1/batches/{BATCH_ID}"
OUTPUT_PATH = f"/v1/files/{OUTPUT_FILE_ID}/content"
ERROR_PATH = f"/v1/files/{ERROR_FILE_ID}/content"
MESSAGES: list[ChatCompletionMessageParam] = [
    {"role": "user", "content": "How does Diglett evolve?"}
]
BAD_REQUEST = {"error": {"message": "max_tokens is too large", "type": "invalid_request_error"}}


class EvolutionTrigger(BaseModel):
    kind: Literal["level", "item", "trade"]
    level: int | None


class EvolutionDraft(BaseModel):
    target: str
    triggers: list[EvolutionTrigger]


DIGLETT_DRAFT = EvolutionDraft(
    target="dugtrio", triggers=[EvolutionTrigger(kind="level", level=26)]
)


def build_batch(status: str, **fields: Any) -> Batch:
    return Batch.model_validate(batch_body(BATCH_ID, status, **fields))


def serve_files(
    files: Mapping[str, str], requests: list[httpx2.Request]
) -> Callable[[httpx2.Request], httpx2.Response]:
    def handle_request(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, text=files[request.url.path])

    return handle_request


def serve_statuses(
    statuses: Iterator[str], requests: list[httpx2.Request]
) -> Callable[[httpx2.Request], httpx2.Response]:
    def handle_request(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=batch_body(BATCH_ID, next(statuses)))

    return handle_request


def draft_output(custom_id: str) -> BatchOutput:
    body = chat_completion_body(DIGLETT_DRAFT.model_dump_json())
    return BatchOutput.model_validate_json(output_line(custom_id, 200, body))


async def test_wait_polls_until_completed() -> None:
    requests: list[httpx2.Request] = []
    statuses = iter(["in_progress", "finalizing", "completed"])

    async with build_mock_client(serve_statuses(statuses, requests)) as client:
        batch = await wait_for_batch(client, BATCH_ID, poll_interval=0)

    assert batch.status == "completed"
    assert [request.url.path for request in requests] == [BATCH_PATH] * 3


async def test_wait_returns_a_failed_batch() -> None:
    requests: list[httpx2.Request] = []

    async with build_mock_client(
        serve_statuses(iter(["validating", "failed"]), requests)
    ) as client:
        batch = await wait_for_batch(client, BATCH_ID, poll_interval=0)

    assert batch.status == "failed"


async def test_wait_wraps_a_read_failure() -> None:
    def handle_request(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(500, json={"error": {"message": "down"}})

    async with build_mock_client(handle_request) as client:
        with pytest.raises(LLMBatchError, match=f"read batch {BATCH_ID} from {MOCK_BASE_URL}"):
            await wait_for_batch(client, BATCH_ID, poll_interval=0)


async def test_completed_batch_gives_one_output_per_line() -> None:
    requests: list[httpx2.Request] = []
    body = chat_completion_body(DIGLETT_DRAFT.model_dump_json())
    files = {
        OUTPUT_PATH: f"{output_line('diglett', 200, body)}\n{output_line('meowth', 200, body)}\n",
        ERROR_PATH: f"{output_line('rotom', 400, BAD_REQUEST)}\n",
    }
    batch = build_batch("completed", output_file_id=OUTPUT_FILE_ID, error_file_id=ERROR_FILE_ID)

    async with build_mock_client(serve_files(files, requests)) as client:
        outputs = await fetch_batch_results(client, batch)

    assert [output.custom_id for output in outputs] == ["diglett", "meowth", "rotom"]
    assert [output.is_success for output in outputs] == [True, True, False]
    assert [request.url.path for request in requests] == [OUTPUT_PATH, ERROR_PATH]


async def test_outputs_map_by_custom_id_in_any_order() -> None:
    requests: list[httpx2.Request] = []
    custom_ids = ["meowth", "diglett", "tauros"]
    lines = [
        output_line(custom_id, 200, chat_completion_body(custom_id)) for custom_id in custom_ids
    ]
    batch = build_batch("completed", output_file_id=OUTPUT_FILE_ID)

    async with build_mock_client(serve_files({OUTPUT_PATH: "\n".join(lines)}, requests)) as client:
        outputs = await fetch_batch_results(client, batch)

    by_custom_id = {output.custom_id: parse_chat_output(output) for output in outputs}
    for custom_id in sorted(custom_ids):
        assert by_custom_id[custom_id].choices[0].message.content == custom_id


async def test_parsed_output_equals_parse_result() -> None:
    body = chat_completion_body(DIGLETT_DRAFT.model_dump_json())

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=body)

    async with build_mock_client(handle_request) as client:
        direct = await client.chat.completions.parse(
            model=MOCK_MODEL, messages=MESSAGES, response_format=EvolutionDraft
        )

    from_batch = parse_chat_output(draft_output("diglett"), EvolutionDraft)

    assert from_batch.choices[0].message.parsed == DIGLETT_DRAFT
    assert direct.choices[0].message.parsed == DIGLETT_DRAFT
    assert type(from_batch) is type(direct)
    # The SDK's generic ParsedChatCompletion warns when it serializes `parsed`.
    assert from_batch.model_dump(warnings=False) == direct.model_dump(warnings=False)


def test_output_without_model_parses_to_chat_completion() -> None:
    completion = parse_chat_output(draft_output("diglett"))

    assert isinstance(completion, ChatCompletion)
    assert completion.choices[0].message.content == DIGLETT_DRAFT.model_dump_json()


def test_failed_line_raises_on_parse() -> None:
    output = BatchOutput.model_validate_json(output_line("rotom", 400, BAD_REQUEST))

    with pytest.raises(BatchOutputError, match="rotom failed with status 400: max_tokens is too"):
        parse_chat_output(output, EvolutionDraft)


async def test_failed_batch_raises_with_its_errors() -> None:
    requests: list[httpx2.Request] = []
    errors = {
        "object": "list",
        "data": [{"code": "invalid_json_line", "message": "line 2 is not JSON", "line": 2}],
    }
    batch = build_batch("failed", errors=errors)

    async with build_mock_client(serve_files({}, requests)) as client:
        with pytest.raises(BatchStateError, match="invalid_json_line: line 2 is not JSON"):
            await fetch_batch_results(client, batch)

    assert requests == []


async def test_expired_batch_gives_finished_and_expired_lines() -> None:
    requests: list[httpx2.Request] = []
    files = {
        OUTPUT_PATH: output_line("diglett", 200, chat_completion_body("done")),
        ERROR_PATH: error_line("meowth", "batch_expired", "This request could not be executed"),
    }
    batch = build_batch("expired", output_file_id=OUTPUT_FILE_ID, error_file_id=ERROR_FILE_ID)

    async with build_mock_client(serve_files(files, requests)) as client:
        finished, expired = await fetch_batch_results(client, batch)

    assert finished.is_success
    assert not expired.is_success
    with pytest.raises(BatchOutputError, match="meowth failed with status None: batch_expired"):
        parse_chat_output(expired)


@pytest.mark.parametrize("status", ["validating", "in_progress", "finalizing", "cancelling"])
async def test_unfinished_batch_raises_without_download(status: str) -> None:
    requests: list[httpx2.Request] = []
    batch = build_batch(status, output_file_id=OUTPUT_FILE_ID)

    async with build_mock_client(serve_files({}, requests)) as client:
        with pytest.raises(BatchStateError, match=f"{BATCH_ID} is still {status}"):
            await fetch_batch_results(client, batch)

    assert requests == []


async def test_malformed_line_raises_naming_its_position() -> None:
    requests: list[httpx2.Request] = []
    text = f"{output_line('diglett', 200, chat_completion_body('done'))}\nnot json\n"
    batch = build_batch("completed", output_file_id=OUTPUT_FILE_ID)

    async with build_mock_client(serve_files({OUTPUT_PATH: text}, requests)) as client:
        with pytest.raises(
            LLMBatchError, match=f"line 2 in file {OUTPUT_FILE_ID} of batch {BATCH_ID}"
        ):
            await fetch_batch_results(client, batch)


def test_embedding_output_parses_to_response() -> None:
    vectors = [[0.1, 0.2], [0.3, 0.4]]
    output = BatchOutput.model_validate_json(output_line("chunk-1", 200, embedding_body(vectors)))

    response = parse_embedding_output(output)

    assert isinstance(response, CreateEmbeddingResponse)
    assert [item.embedding for item in response.data] == vectors
    assert json.loads(response.model_dump_json())["model"] == MOCK_MODEL
