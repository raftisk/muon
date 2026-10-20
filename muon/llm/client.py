"""Builds `AsyncOpenAI` clients from provider settings."""

from openai import AsyncOpenAI

from muon.config import ProviderSettings


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
