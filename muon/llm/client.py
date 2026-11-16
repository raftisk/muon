"""Builds `AsyncOpenAI` and LangChain chat clients from provider settings."""

from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from muon.config import ProviderSettings, ReasoningLevel


class ModelCallError(Exception):
    """A model call failed after the SDK retries; names the step and the model."""

    def __init__(self, step: str, model: str, reason: str) -> None:
        super().__init__(f"Model call failed in the {step} step (model {model}): {reason}")
        self.step = step
        self.model = model


def create_client(settings: ProviderSettings) -> AsyncOpenAI:
    """Return a client for the endpoint, key, timeout and retries in `settings`.

    The caller owns the client's lifetime and closes it, usually with `async with`.
    The model is not bound to the client: each call passes `model=settings.model`.
    """
    return AsyncOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key.get_secret_value(),
        timeout=settings.timeout,
        max_retries=settings.max_retries,
    )


def create_chat_model(
    settings: ProviderSettings, reasoning: ReasoningLevel | None = None
) -> ChatOpenAI:
    """Return a LangChain chat model for the endpoint, key, model, timeout and retries.

    `reasoning` is sent as `reasoning_effort` only when it is set.
    """
    return ChatOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        model=settings.model,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
        reasoning_effort=reasoning,
    )
