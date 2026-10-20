"""Live checks of direct calls against the providers configured in `.env`.

Run once with OpenAI values, then once with Qwen values exported inline and `-k chat`.
A test skips when its settings group is missing.
"""

import pytest
from openai import omit
from pydantic import BaseModel, ValidationError

from muon.config import EmbeddingSettings, LLMSettings
from muon.llm import create_client

pytestmark = pytest.mark.network

QUESTION = "What is the capital of France? Answer with the city name only."
EMBEDDING_INPUTS = ["Diglett lives underground.", "Meowth loves shiny coins."]


class CountryCapital(BaseModel):
    country: str
    capital: str


def describe_missing(error: ValidationError) -> str:
    return ", ".join(str(detail["loc"][0]) for detail in error.errors())


@pytest.fixture
def llm_settings() -> LLMSettings:
    try:
        return LLMSettings()
    except ValidationError as error:
        pytest.skip(f"MUON_LLM_* settings missing: {describe_missing(error)}")


@pytest.fixture
def embedding_settings() -> EmbeddingSettings:
    try:
        return EmbeddingSettings()
    except ValidationError as error:
        pytest.skip(f"MUON_EMBEDDING_* settings missing: {describe_missing(error)}")


async def test_live_chat_text(llm_settings: LLMSettings) -> None:
    async with create_client(llm_settings) as client:
        completion = await client.chat.completions.create(
            model=llm_settings.model, messages=[{"role": "user", "content": QUESTION}]
        )

    assert completion.choices[0].message.content


async def test_live_chat_structured(llm_settings: LLMSettings) -> None:
    async with create_client(llm_settings) as client:
        completion = await client.chat.completions.parse(
            model=llm_settings.model,
            messages=[{"role": "user", "content": QUESTION}],
            response_format=CountryCapital,
        )

    parsed = completion.choices[0].message.parsed
    assert parsed is not None
    assert parsed.capital == "Paris"


async def test_live_embeddings(embedding_settings: EmbeddingSettings) -> None:
    dimensions = embedding_settings.dimensions
    async with create_client(embedding_settings) as client:
        response = await client.embeddings.create(
            model=embedding_settings.model,
            input=EMBEDDING_INPUTS,
            dimensions=omit if dimensions is None else dimensions,
        )

    assert [item.index for item in response.data] == [0, 1]
    if dimensions is not None:
        assert all(len(item.embedding) == dimensions for item in response.data)
