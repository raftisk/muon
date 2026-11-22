"""Thread state of the chat server.

`ThreadRepository` is the interface the routes and the run coordinator use. `ThreadStore`
keeps everything in memory and loses it when the process stops.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal, Protocol
from uuid import uuid4

from ag_ui.core import Message
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

DEFAULT_THREAD_TITLE = "New thread"
TITLE_MAX_LENGTH = 40
THREAD_ID_PREFIX = "thr_"
ASSISTANT_ROLE = "assistant"

Rating = Literal["up", "down"]


class ThreadRecord(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    id: str
    title: str
    updated_at: datetime


class ThreadNotFoundError(Exception):
    def __init__(self, thread_id: str) -> None:
        super().__init__(f"Thread {thread_id} not found")


class MessageNotFoundError(Exception):
    def __init__(self, thread_id: str, message_id: str) -> None:
        super().__init__(f"Message {message_id} not found in thread {thread_id}")


class FeedbackTargetError(Exception):
    def __init__(self, message_id: str, role: str) -> None:
        super().__init__(
            f"Feedback needs an assistant message, but message {message_id} has role {role}"
        )


class ThreadRepository(Protocol):
    """Everything the routes and runs need from thread storage.

    Every method except `create_thread` and `list_threads` raises `ThreadNotFoundError` for an
    unknown thread. Returned values are copies.
    """

    def create_thread(self) -> ThreadRecord: ...

    def list_threads(self) -> list[ThreadRecord]:
        """Return the threads, newest update first."""
        ...

    def get_thread(self, thread_id: str) -> ThreadRecord: ...

    def rename_thread(self, thread_id: str, title: str) -> ThreadRecord: ...

    def delete_thread(self, thread_id: str) -> None:
        """Remove the thread with its messages and ratings."""
        ...

    def list_messages(self, thread_id: str) -> list[Message]: ...

    def append_message(self, thread_id: str, message: Message) -> None: ...

    def apply_first_run_title(self, thread_id: str, question: str) -> None:
        """Set the title from `question`, only while the thread still has the default title."""
        ...

    def set_feedback(self, thread_id: str, message_id: str, rating: Rating) -> None:
        """Rate an assistant message.

        Raises `MessageNotFoundError` for an unknown message and `FeedbackTargetError` for a
        message that is not from the assistant.
        """
        ...


def utc_now() -> datetime:
    return datetime.now(UTC)


class ThreadStore:
    """In-memory `ThreadRepository`. Callers get copies, never the internal structures."""

    def __init__(self, now: Callable[[], datetime] = utc_now) -> None:
        self.now = now
        self.threads: dict[str, ThreadRecord] = {}
        self.messages: dict[str, list[Message]] = {}
        self.ratings: dict[tuple[str, str], Rating] = {}

    def create_thread(self) -> ThreadRecord:
        thread = ThreadRecord(
            id=f"{THREAD_ID_PREFIX}{uuid4().hex}",
            title=DEFAULT_THREAD_TITLE,
            updated_at=self.now(),
        )
        self.threads[thread.id] = thread
        self.messages[thread.id] = []
        return thread.model_copy()

    def list_threads(self) -> list[ThreadRecord]:
        newest_first = sorted(self.threads.values(), key=lambda t: t.updated_at, reverse=True)
        return [thread.model_copy() for thread in newest_first]

    def get_thread(self, thread_id: str) -> ThreadRecord:
        return self.find_thread(thread_id).model_copy()

    def rename_thread(self, thread_id: str, title: str) -> ThreadRecord:
        thread = self.find_thread(thread_id)
        renamed = thread.model_copy(update={"title": title, "updated_at": self.now()})
        self.threads[thread_id] = renamed
        return renamed.model_copy()

    def delete_thread(self, thread_id: str) -> None:
        self.find_thread(thread_id)
        del self.threads[thread_id]
        del self.messages[thread_id]
        self.ratings = {key: rating for key, rating in self.ratings.items() if key[0] != thread_id}

    def list_messages(self, thread_id: str) -> list[Message]:
        self.find_thread(thread_id)
        return list(self.messages[thread_id])

    def append_message(self, thread_id: str, message: Message) -> None:
        thread = self.find_thread(thread_id)
        self.messages[thread_id].append(message)
        self.threads[thread_id] = thread.model_copy(update={"updated_at": self.now()})

    def apply_first_run_title(self, thread_id: str, question: str) -> None:
        thread = self.find_thread(thread_id)
        if thread.title != DEFAULT_THREAD_TITLE:
            return
        title = question.strip()[:TITLE_MAX_LENGTH]
        self.threads[thread_id] = thread.model_copy(update={"title": title})

    def set_feedback(self, thread_id: str, message_id: str, rating: Rating) -> None:
        self.find_thread(thread_id)
        message = next((m for m in self.messages[thread_id] if m.id == message_id), None)
        if message is None:
            raise MessageNotFoundError(thread_id, message_id)
        if message.role != ASSISTANT_ROLE:
            raise FeedbackTargetError(message_id, message.role)
        self.ratings[(thread_id, message_id)] = rating

    def find_thread(self, thread_id: str) -> ThreadRecord:
        thread = self.threads.get(thread_id)
        if thread is None:
            raise ThreadNotFoundError(thread_id)
        return thread
