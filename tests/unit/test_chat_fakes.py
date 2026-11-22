from contextlib import aclosing

import pytest
from langchain_core.messages import AIMessageChunk

from tests.unit.chat_fakes import DEFAULT_USAGE, ScriptedChatModel

STREAM_FAILURE = RuntimeError("stream broke")


async def test_stream_yields_the_scripted_chunks() -> None:
    model = ScriptedChatModel(streams=[["a", "b", "c"]])

    texts = [chunk.content async for chunk in model.astream("hi") if chunk.content]

    assert texts == ["a", "b", "c"]
    assert model.stream_progress == [1, 2, 3]
    assert len(model.calls) == 1
    assert model.is_stream_closed


async def test_summed_chunks_carry_the_usage() -> None:
    model = ScriptedChatModel(streams=[["a", "b"]])
    total: AIMessageChunk | None = None

    async for chunk in model.astream("hi"):
        total = chunk if total is None else total + chunk

    assert total is not None
    assert total.content == "ab"
    assert total.usage_metadata is not None
    assert total.usage_metadata["total_tokens"] == DEFAULT_USAGE["total_tokens"]


async def test_stream_failure_is_raised_after_the_set_chunks() -> None:
    model = ScriptedChatModel(
        streams=[["a", "b", "c"]], stream_failure=STREAM_FAILURE, fail_after=2
    )
    texts: list[str] = []

    with pytest.raises(RuntimeError, match="stream broke"):
        async for chunk in model.astream("hi"):
            texts.append(str(chunk.content))

    assert texts == ["a", "b"]
    assert model.is_stream_closed


async def test_closing_the_stream_early_marks_it_closed() -> None:
    model = ScriptedChatModel(streams=[["a", "b", "c"]])

    async with aclosing(model.astream("hi")) as stream:
        await anext(stream)
        assert not model.is_stream_closed

    assert model.is_stream_closed
