"""Model access layer: `AsyncOpenAI` clients built from settings and Batch API helpers.

Callers use the openai SDK directly for chat and embedding calls. The agent uses the
LangChain chat model from `create_chat_model`.
"""

from muon.llm.batch import (
    BatchLine,
    BatchOutput,
    BatchOutputError,
    BatchStateError,
    BatchSubmitError,
    LLMBatchError,
    build_chat_line,
    build_embedding_line,
    fetch_batch_results,
    parse_chat_output,
    parse_embedding_output,
    submit_batch,
    wait_for_batch,
)
from muon.llm.client import ModelCallError, create_chat_model, create_client

__all__ = [
    "BatchLine",
    "BatchOutput",
    "BatchOutputError",
    "BatchStateError",
    "BatchSubmitError",
    "LLMBatchError",
    "ModelCallError",
    "build_chat_line",
    "build_embedding_line",
    "create_chat_model",
    "create_client",
    "fetch_batch_results",
    "parse_chat_output",
    "parse_embedding_output",
    "submit_batch",
    "wait_for_batch",
]
