"""Model access layer: `AsyncOpenAI` clients built from settings.

Callers use the openai SDK directly for chat and embedding calls.
"""

from muon.llm.client import create_client

__all__ = ["create_client"]
