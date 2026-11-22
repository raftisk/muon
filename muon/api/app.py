"""The app factory and the production answer stream."""

from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from muon.agent.ask import AskRequest, ask
from muon.agent.events import AskEvent
from muon.agent.routing import create_agent_runtime
from muon.api.agent import create_agent_router
from muon.api.agui import AnswerStream
from muon.api.auth import create_auth_router
from muon.api.errors import install_error_handling
from muon.api.store import ThreadRepository
from muon.api.threads import create_threads_router
from muon.config import AgentSettings, ApiSettings, Settings
from muon.graph import Neo4jGraphClient
from muon.plugins import UniverseRegistry

CORS_METHODS = ["GET", "POST", "PATCH", "DELETE", "OPTIONS"]
CORS_HEADERS = ["Accept", "Content-Type"]
APP_TITLE = "muon chat server"

OpenAnswerStream = Callable[[], AbstractAsyncContextManager[AnswerStream]]


def create_app(
    api_settings: ApiSettings, open_answer_stream: OpenAnswerStream, store: ThreadRepository
) -> FastAPI:
    """Build the app. `open_answer_stream` is entered at startup and left at shutdown."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with open_answer_stream() as answer_stream:
            app.state.answer_stream = answer_stream
            yield

    app = FastAPI(title=APP_TITLE, lifespan=lifespan)
    install_error_handling(app)
    # CORS goes on after the error middleware so it wraps it and a 500 carries CORS headers.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_settings.cors_origins,
        allow_credentials=True,
        allow_methods=CORS_METHODS,
        allow_headers=CORS_HEADERS,
    )
    app.include_router(create_auth_router())
    app.include_router(create_threads_router(store))
    app.include_router(create_agent_router(store))
    return app


def open_ask_stream(
    settings: Settings, agent_settings: AgentSettings, registry: UniverseRegistry
) -> OpenAnswerStream:
    """Return the factory of the production stream: 1 Neo4j client and runtime per server."""

    @asynccontextmanager
    async def open_stream() -> AsyncIterator[AnswerStream]:
        async with Neo4jGraphClient(settings.neo4j) as client:
            runtime = create_agent_runtime(client, agent_settings)

            def answer(request: AskRequest) -> AsyncGenerator[AskEvent]:
                return ask(request, registry, settings, runtime)

            yield answer

    return open_stream
