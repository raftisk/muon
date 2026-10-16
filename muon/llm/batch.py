"""Provider Batch API helpers over `AsyncOpenAI`.

Map: `build_chat_line` and `build_embedding_line` turn the params of a direct call into
`BatchLine`s, and `submit_batch` uploads them as 1 JSONL file and creates 1 batch.

Hazard: structured lines use the private SDK function behind `chat.completions.parse`,
so a batch sends the same `response_format` as a direct call. The parity tests in
`tests/unit/llm/` fail when an SDK upgrade changes it.
"""

from collections import Counter
from collections.abc import Sequence
from typing import Any, Final, Literal

import openai
from openai import AsyncOpenAI
from openai.lib._parsing._completions import type_to_response_format_param
from openai.types import Batch, EmbeddingCreateParams
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming
from pydantic import BaseModel, Field

BatchEndpoint = Literal["/v1/chat/completions", "/v1/embeddings"]

CHAT_ENDPOINT: Final = "/v1/chat/completions"
EMBEDDING_ENDPOINT: Final = "/v1/embeddings"
COMPLETION_WINDOW: Final = "24h"
BATCH_FILE_PURPOSE: Final = "batch"
BATCH_INPUT_FILENAME = "batch-input.jsonl"


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
