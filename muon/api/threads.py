"""The thread routes of the Toki contract."""

from http import HTTPStatus

from ag_ui.core import Message
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from muon.api.store import (
    FeedbackTargetError,
    MessageNotFoundError,
    Rating,
    ThreadNotFoundError,
    ThreadRecord,
    ThreadRepository,
)


class RenameThreadRequest(BaseModel):
    title: str

    @field_validator("title")
    @classmethod
    def reject_blank_title(cls, title: str) -> str:
        if not title.strip():
            raise ValueError("title must not be blank")
        return title


class FeedbackRequest(BaseModel):
    rating: Rating


def create_threads_router(store: ThreadRepository) -> APIRouter:
    router = APIRouter()

    @router.get("/threads")
    async def list_threads() -> list[ThreadRecord]:
        return store.list_threads()

    @router.post("/threads", status_code=HTTPStatus.CREATED)
    async def create_thread() -> ThreadRecord:
        return store.create_thread()

    @router.patch("/threads/{thread_id}")
    async def rename_thread(thread_id: str, request: RenameThreadRequest) -> ThreadRecord:
        try:
            return store.rename_thread(thread_id, request.title)
        except ThreadNotFoundError as error:
            raise HTTPException(HTTPStatus.NOT_FOUND, str(error)) from error

    @router.delete("/threads/{thread_id}", status_code=HTTPStatus.NO_CONTENT)
    async def delete_thread(thread_id: str) -> None:
        try:
            store.delete_thread(thread_id)
        except ThreadNotFoundError as error:
            raise HTTPException(HTTPStatus.NOT_FOUND, str(error)) from error

    @router.get("/threads/{thread_id}/messages", response_model_exclude_none=True)
    async def list_messages(thread_id: str) -> list[Message]:
        try:
            return store.list_messages(thread_id)
        except ThreadNotFoundError as error:
            raise HTTPException(HTTPStatus.NOT_FOUND, str(error)) from error

    @router.post(
        "/threads/{thread_id}/messages/{message_id}/feedback", status_code=HTTPStatus.NO_CONTENT
    )
    async def submit_feedback(thread_id: str, message_id: str, request: FeedbackRequest) -> None:
        try:
            store.set_feedback(thread_id, message_id, request.rating)
        except (ThreadNotFoundError, MessageNotFoundError) as error:
            raise HTTPException(HTTPStatus.NOT_FOUND, str(error)) from error
        except FeedbackTargetError as error:
            raise HTTPException(HTTPStatus.UNPROCESSABLE_ENTITY, str(error)) from error

    return router
