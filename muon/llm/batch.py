"""Provider Batch API helpers over `AsyncOpenAI`.

Map: `build_chat_line` and `build_embedding_line` turn the params of a direct call into
`BatchLine`s, and `submit_batch` uploads them as 1 JSONL file and creates 1 batch.
`wait_for_batch` polls a batch id until the batch ends, `fetch_batch_results` reads 1
`BatchOutput` per line, and `parse_chat_output` and `parse_embedding_output` turn an
output into the SDK type a direct call returns.

Hazard: structured lines and outputs use the private SDK functions behind
`chat.completions.parse`, so a batch sends the same `response_format` and parses the
same way as a direct call. The parity tests in `tests/unit/llm/` fail when an SDK
upgrade changes either one.
"""

import asyncio
from collections import Counter
from collections.abc import Sequence
from typing import Any, Final, Literal, overload

import openai
from openai import AsyncOpenAI, omit
from openai.lib._parsing._completions import parse_chat_completion, type_to_response_format_param
from openai.types import Batch, CreateEmbeddingResponse, EmbeddingCreateParams
from openai.types.chat import ChatCompletion, ParsedChatCompletion
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming
from pydantic import BaseModel, Field, ValidationError

BatchEndpoint = Literal["/v1/chat/completions", "/v1/embeddings"]

CHAT_ENDPOINT: Final = "/v1/chat/completions"
EMBEDDING_ENDPOINT: Final = "/v1/embeddings"
COMPLETION_WINDOW: Final = "24h"
BATCH_FILE_PURPOSE: Final = "batch"
BATCH_INPUT_FILENAME = "batch-input.jsonl"

TERMINAL_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})
FAILED_STATUS = "failed"
DEFAULT_POLL_INTERVAL_SECONDS = 30.0
SUCCESS_STATUS_CODES = range(200, 300)
UNKNOWN_FAILURE = "no message from the provider"


class LLMBatchError(Exception):
    """Base for every error the batch helpers raise."""


class BatchSubmitError(LLMBatchError):
    """The lines cannot form 1 batch, or the provider rejected the upload or the batch."""


class BatchStateError(LLMBatchError):
    """The batch is not in a state its results can be read from."""


class BatchOutputError(LLMBatchError):
    """A batch line failed, so it has no response to parse."""


class BatchLine(BaseModel):
    """1 request of a batch input file."""

    custom_id: str = Field(min_length=1)
    method: Literal["POST"] = "POST"
    url: BatchEndpoint
    body: dict[str, Any]


class BatchOutputResponse(BaseModel):
    status_code: int
    request_id: str | None = None
    body: dict[str, Any]


class BatchOutputFailure(BaseModel):
    code: str | None = None
    message: str | None = None


class BatchOutput(BaseModel):
    """1 line of a batch output or error file.

    A line that reached the model has a `response`, with an HTTP status that may be an
    error. A line that never ran, such as one cut by expiry, has an `error` instead.
    """

    custom_id: str
    response: BatchOutputResponse | None = None
    error: BatchOutputFailure | None = None

    @property
    def is_success(self) -> bool:
        return (
            self.error is None
            and self.response is not None
            and self.response.status_code in SUCCESS_STATUS_CODES
        )


def build_chat_line(
    custom_id: str,
    params: CompletionCreateParamsNonStreaming,
    response_model: type[BaseModel] | None = None,
) -> BatchLine:
    """Return the batch form of `chat.completions.create(**params)`.

    With `response_model`, the line asks for the strict `response_format` that
    `chat.completions.parse(response_format=response_model)` sends.
    """
    body: dict[str, Any] = dict(params)
    if response_model is not None:
        body["response_format"] = type_to_response_format_param(response_model)
    return BatchLine(custom_id=custom_id, url=CHAT_ENDPOINT, body=body)


def build_embedding_line(custom_id: str, params: EmbeddingCreateParams) -> BatchLine:
    """Return the batch form of `embeddings.create(**params)`."""
    return BatchLine(custom_id=custom_id, url=EMBEDDING_ENDPOINT, body=dict(params))


def check_batch_lines(lines: Sequence[BatchLine]) -> BatchEndpoint:
    """Return the endpoint shared by `lines`, or raise if they cannot form 1 batch."""
    if not lines:
        raise BatchSubmitError("Cannot submit a batch with no lines")
    counts = Counter(line.custom_id for line in lines)
    duplicates = sorted(custom_id for custom_id, count in counts.items() if count > 1)
    if duplicates:
        raise BatchSubmitError(f"Cannot submit a batch with duplicate custom ids: {duplicates}")
    endpoints = sorted({line.url for line in lines})
    if len(endpoints) > 1:
        raise BatchSubmitError(f"Cannot submit a batch that mixes endpoints: {endpoints}")
    return lines[0].url


def encode_batch_lines(lines: Sequence[BatchLine]) -> bytes:
    return "".join(f"{line.model_dump_json()}\n" for line in lines).encode()


async def submit_batch(client: AsyncOpenAI, lines: Sequence[BatchLine]) -> Batch:
    """Upload `lines` as 1 JSONL file and create 1 batch over it.

    The caller keeps the returned batch id to collect the results, possibly from
    another process.
    """
    endpoint = check_batch_lines(lines)
    payload = encode_batch_lines(lines)
    try:
        input_file = await client.files.create(
            file=(BATCH_INPUT_FILENAME, payload), purpose=BATCH_FILE_PURPOSE
        )
    except openai.APIError as error:
        raise BatchSubmitError(
            f"Failed to upload batch input of {len(lines)} lines to {client.base_url}: {error}"
        ) from error
    try:
        return await client.batches.create(
            input_file_id=input_file.id, endpoint=endpoint, completion_window=COMPLETION_WINDOW
        )
    except openai.APIError as error:
        raise BatchSubmitError(
            f"Failed to create a batch on {endpoint} at {client.base_url} "
            f"from file {input_file.id}: {error}"
        ) from error


async def wait_for_batch(
    client: AsyncOpenAI, batch_id: str, poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS
) -> Batch:
    """Poll `batch_id` every `poll_interval` seconds and return it once it has ended.

    A batch ends as `completed`, `failed`, `expired` or `cancelled`. There is no own
    deadline, because the provider ends every batch within its completion window; wrap
    the call in `asyncio.timeout` for a shorter one.
    """
    while True:
        try:
            batch = await client.batches.retrieve(batch_id)
        except openai.APIError as error:
            raise LLMBatchError(
                f"Failed to read batch {batch_id} from {client.base_url}: {error}"
            ) from error
        if batch.status in TERMINAL_STATUSES:
            return batch
        await asyncio.sleep(poll_interval)


async def fetch_batch_results(client: AsyncOpenAI, batch: Batch) -> list[BatchOutput]:
    """Return 1 output per line of an ended batch, from its output file then its error file.

    Outputs keep the provider's order, which need not match the input; `custom_id` is
    the join key. An expired or cancelled batch gives its finished lines as successes
    and the others as error outputs.
    """
    check_batch_readable(batch)
    file_ids = [file_id for file_id in (batch.output_file_id, batch.error_file_id) if file_id]
    texts = [await read_batch_file(client, batch.id, file_id) for file_id in file_ids]
    return [
        output
        for file_id, text in zip(file_ids, texts, strict=True)
        for output in decode_batch_file(batch.id, file_id, text)
    ]


def check_batch_readable(batch: Batch) -> None:
    if batch.status not in TERMINAL_STATUSES:
        raise BatchStateError(f"Batch {batch.id} is still {batch.status}; no results yet")
    if batch.status == FAILED_STATUS:
        errors = batch.errors.data if batch.errors and batch.errors.data else []
        messages = [f"{error.code}: {error.message}" for error in errors]
        raise BatchStateError(f"Batch {batch.id} failed: {messages}")


async def read_batch_file(client: AsyncOpenAI, batch_id: str, file_id: str) -> str:
    try:
        content = await client.files.content(file_id)
    except openai.APIError as error:
        raise LLMBatchError(
            f"Failed to download file {file_id} of batch {batch_id} from {client.base_url}: {error}"
        ) from error
    return content.text


def decode_batch_file(batch_id: str, file_id: str, text: str) -> list[BatchOutput]:
    outputs = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            outputs.append(BatchOutput.model_validate_json(line))
        except ValidationError as error:
            raise LLMBatchError(
                f"Malformed line {line_number} in file {file_id} of batch {batch_id}: {error}"
            ) from error
    return outputs


def require_success(output: BatchOutput) -> dict[str, Any]:
    """Return the response body of a successful output, or raise `BatchOutputError`."""
    if output.is_success and output.response is not None:
        return output.response.body
    status = output.response.status_code if output.response is not None else None
    raise BatchOutputError(
        f"Batch line {output.custom_id} failed with status {status}: {describe_failure(output)}"
    )


def describe_failure(output: BatchOutput) -> str:
    if output.error is not None:
        return f"{output.error.code}: {output.error.message or UNKNOWN_FAILURE}"
    body_error = output.response.body.get("error") if output.response is not None else None
    if isinstance(body_error, dict) and isinstance(body_error.get("message"), str):
        return str(body_error["message"])
    return UNKNOWN_FAILURE


@overload
def parse_chat_output(output: BatchOutput) -> ChatCompletion: ...


@overload
def parse_chat_output[ModelT: BaseModel](
    output: BatchOutput, response_model: type[ModelT]
) -> ParsedChatCompletion[ModelT]: ...


def parse_chat_output[ModelT: BaseModel](
    output: BatchOutput, response_model: type[ModelT] | None = None
) -> ChatCompletion | ParsedChatCompletion[ModelT]:
    """Return the chat completion of a successful output.

    With `response_model`, the result is the `ParsedChatCompletion` that
    `chat.completions.parse` returns for the same response body.
    """
    completion = ChatCompletion.model_validate(require_success(output))
    if response_model is None:
        return completion
    return parse_chat_completion(
        response_format=response_model, input_tools=omit, chat_completion=completion
    )


def parse_embedding_output(output: BatchOutput) -> CreateEmbeddingResponse:
    """Return the embedding response of a successful output."""
    return CreateEmbeddingResponse.model_validate(require_success(output))
