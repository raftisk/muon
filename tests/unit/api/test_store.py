from datetime import UTC, datetime, timedelta

import pytest
from ag_ui.core import AssistantMessage, UserMessage

from muon.api.store import (
    DEFAULT_THREAD_TITLE,
    TITLE_MAX_LENGTH,
    FeedbackTargetError,
    MessageNotFoundError,
    ThreadNotFoundError,
    ThreadStore,
)

START = datetime(2026, 1, 1, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.current = START

    def __call__(self) -> datetime:
        self.current += timedelta(seconds=1)
        return self.current


def build_store() -> ThreadStore:
    return ThreadStore(now=Clock())


def test_threads_list_newest_first() -> None:
    store = build_store()
    first = store.create_thread()
    second = store.create_thread()

    assert [t.id for t in store.list_threads()] == [second.id, first.id]
    store.rename_thread(first.id, "later")
    assert [t.id for t in store.list_threads()] == [first.id, second.id]


def test_the_first_title_call_cuts_the_stripped_text_and_later_calls_keep_it() -> None:
    store = build_store()
    thread = store.create_thread()
    assert thread.title == DEFAULT_THREAD_TITLE

    store.apply_first_run_title(thread.id, "  " + "q" * 60 + "  ")
    store.apply_first_run_title(thread.id, "another question")

    assert store.get_thread(thread.id).title == "q" * TITLE_MAX_LENGTH


def test_delete_removes_the_messages_and_ratings() -> None:
    store = build_store()
    thread = store.create_thread()
    store.append_message(thread.id, AssistantMessage(id="a1", content="hi"))
    store.set_feedback(thread.id, "a1", "up")

    store.delete_thread(thread.id)

    assert store.messages == {}
    assert store.ratings == {}
    with pytest.raises(ThreadNotFoundError):
        store.list_messages(thread.id)


def test_unknown_ids_raise() -> None:
    store = build_store()
    thread = store.create_thread()

    with pytest.raises(ThreadNotFoundError):
        store.get_thread("thr_missing")
    with pytest.raises(MessageNotFoundError):
        store.set_feedback(thread.id, "missing", "up")


def test_feedback_on_a_user_message_raises() -> None:
    store = build_store()
    thread = store.create_thread()
    store.append_message(thread.id, UserMessage(id="u1", content="hi"))

    with pytest.raises(FeedbackTargetError):
        store.set_feedback(thread.id, "u1", "down")


def test_returned_records_are_copies() -> None:
    store = build_store()
    thread = store.create_thread()
    store.append_message(thread.id, UserMessage(id="u1", content="hi"))

    store.list_messages(thread.id).clear()
    thread.title = "changed"

    assert len(store.list_messages(thread.id)) == 1
    assert store.get_thread(thread.id).title == DEFAULT_THREAD_TITLE
