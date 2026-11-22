"""`POST /agent`: answer the last user message as an AG-UI event stream."""

from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from http import HTTPStatus

from ag_ui.core import BaseEvent, Message, RunAgentInput, TextPart, UserMessage
from ag_ui.encoder import EventEncoder
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from muon.api.agui import AnswerStream, stream_run
from muon.api.store import ThreadNotFoundError, ThreadRepository

DEFAULT_ACCEPT = "text/event-stream"
NO_CACHE_HEADERS = {"Cache-Control": "no-cache"}


class EmptyQuestionError(Exception):
    """The request has no user message, or its text is blank."""


def read_message_text(message: UserMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(part.text for part in message.content if isinstance(part, TextPart))


def extract_question(messages: Sequence[Message]) -> tuple[UserMessage, str]:
    """Return the last user message and its text. Raise `EmptyQuestionError` if it has none."""
    last_user = next(
        (message for message in reversed(messages) if isinstance(message, UserMessage)), None
    )
    if last_user is None:
        raise EmptyQuestionError("The request has no user message")
    question = read_message_text(last_user)
    if not question.strip():
        raise EmptyQuestionError(f"User message {last_user.id} is blank")
    return last_user, question


def read_answer_stream(request: Request) -> AnswerStream:
    answer_stream: AnswerStream = request.app.state.answer_stream
    return answer_stream


async def encode_events(
    encoder: EventEncoder, events: AsyncGenerator[BaseEvent]
) -> AsyncIterator[str]:
    """Encode `events` as SSE lines, closing the run when the response stops reading."""
    try:
        async for event in events:
            yield encoder.encode(event)
    finally:
        await events.aclose()


def create_agent_router(store: ThreadRepository) -> APIRouter:
    router = APIRouter()

    @router.post("/agent")
    async def run_agent(run_input: RunAgentInput, request: Request) -> StreamingResponse:
        try:
            store.get_thread(run_input.thread_id)
        except ThreadNotFoundError as error:
            raise HTTPException(HTTPStatus.NOT_FOUND, str(error)) from error
        try:
            user_message, question = extract_question(run_input.messages)
        except EmptyQuestionError as error:
            raise HTTPException(HTTPStatus.UNPROCESSABLE_ENTITY, str(error)) from error
        store.append_message(run_input.thread_id, user_message)
        events = stream_run(run_input, question, read_answer_stream(request), store)
        encoder = EventEncoder(accept=request.headers.get("accept", DEFAULT_ACCEPT))
        return StreamingResponse(
            encode_events(encoder, events),
            media_type=encoder.get_content_type(),
            headers=NO_CACHE_HEADERS,
        )

    return router
