import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from muon.agent.ask import ANSWER_STEP, QUERY_STEP, AskRequest
from muon.agent.events import AnswerFinished, AskEvent, StepFinished, StepStarted, TextDelta
from muon.api.agui import AnswerStream
from muon.api.app import create_app
from muon.api.store import DEFAULT_THREAD_TITLE, ThreadStore
from muon.config import ApiSettings
from tests.unit.api.test_agui import build_result

ORIGIN = "http://localhost:5173"
QUESTION = "What does the widget do?"


async def play_answer(request: AskRequest) -> AsyncIterator[AskEvent]:
    yield StepStarted(name=QUERY_STEP)
    yield StepFinished(name=QUERY_STEP)
    yield StepStarted(name=ANSWER_STEP)
    yield TextDelta(text="Alpha is a widget.")
    yield StepFinished(name=ANSWER_STEP)
    yield AnswerFinished(result=build_result(has_artifacts=True))


@asynccontextmanager
async def open_fake_stream() -> AsyncIterator[AnswerStream]:
    yield play_answer  # type: ignore[misc]


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app(ApiSettings(_env_file=None), open_fake_stream, ThreadStore())
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def build_run_body(thread_id: str, messages: list[dict[str, object]]) -> dict[str, object]:
    return {
        "threadId": thread_id,
        "runId": "run_1",
        "state": {},
        "messages": messages,
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }


def build_question_body(thread_id: str) -> dict[str, object]:
    return build_run_body(thread_id, [{"id": "u1", "role": "user", "content": QUESTION}])


def create_thread(client: TestClient) -> str:
    return str(client.post("/threads").json()["id"])


def test_cors_preflight_allows_only_the_configured_origin(client: TestClient) -> None:
    headers = {
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    }

    allowed = client.options("/threads", headers={"Origin": ORIGIN, **headers})
    other = client.options("/threads", headers={"Origin": "http://evil.test", **headers})

    assert allowed.headers["access-control-allow-origin"] == ORIGIN
    assert allowed.headers["access-control-allow-credentials"] == "true"
    assert allowed.status_code == 200
    assert "access-control-allow-origin" not in other.headers


def test_auth_is_anonymous(client: TestClient) -> None:
    assert client.get("/auth/me").json() == {"id": "anonymous", "name": "Anonymous"}
    assert client.post("/auth/login", json={"username": "a", "password": "b"}).status_code == 204
    assert client.post("/auth/logout").status_code == 204


def test_threads_can_be_created_listed_renamed_and_deleted(client: TestClient) -> None:
    created = client.post("/threads")
    thread_id = created.json()["id"]

    assert created.status_code == 201
    assert created.json()["title"] == DEFAULT_THREAD_TITLE
    assert [t["id"] for t in client.get("/threads").json()] == [thread_id]
    renamed = client.patch(f"/threads/{thread_id}", json={"title": "Renamed"})
    assert renamed.json()["title"] == "Renamed"
    blank = client.patch(f"/threads/{thread_id}", json={"title": "  "})
    assert blank.status_code == 422
    assert blank.json()["error"]["code"] == "validation_error"
    assert client.delete(f"/threads/{thread_id}").status_code == 204
    assert client.get(f"/threads/{thread_id}/messages").status_code == 404


def test_feedback_follows_the_contract(client: TestClient) -> None:
    thread_id = create_thread(client)
    with client.stream("POST", "/agent", json=build_question_body(thread_id)) as response:
        response.read()
    messages = client.get(f"/threads/{thread_id}/messages").json()
    user_id = messages[0]["id"]
    assistant_id = messages[1]["id"]
    path = f"/threads/{thread_id}/messages"

    assert client.post(f"{path}/{assistant_id}/feedback", json={"rating": "up"}).status_code == 204
    assert client.post(f"{path}/{user_id}/feedback", json={"rating": "up"}).status_code == 422
    assert client.post(f"{path}/missing/feedback", json={"rating": "up"}).status_code == 404


def test_agent_rejects_an_unknown_thread_and_a_missing_user_message(client: TestClient) -> None:
    unknown = client.post("/agent", json=build_question_body("thr_missing"))
    thread_id = create_thread(client)
    no_user = client.post("/agent", json=build_run_body(thread_id, []))

    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "not_found"
    assert no_user.status_code == 422


def test_agent_streams_events_and_stores_the_turn(client: TestClient) -> None:
    thread_id = create_thread(client)

    with client.stream(
        "POST",
        "/agent",
        json=build_question_body(thread_id),
        headers={"Accept": "text/event-stream"},
    ) as response:
        lines = [line for line in response.iter_lines() if line.startswith("data:")]
        content_type = response.headers["content-type"]

    types = [json.loads(line.removeprefix("data:"))["type"] for line in lines]
    assert content_type.startswith("text/event-stream")
    assert types[0] == "RUN_STARTED"
    assert types[-1] == "RUN_FINISHED"
    assert types.index("TEXT_MESSAGE_END") < types.index("TOOL_CALL_START")
    messages = client.get(f"/threads/{thread_id}/messages").json()
    assert [m["role"] for m in messages] == [
        "user",
        "assistant",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "activity",
    ]
    assert "toolCalls" in messages[2]
    assert all(value is not None for message in messages for value in message.values())
    assert client.get("/threads").json()[0]["title"] == QUESTION


def test_an_unhandled_route_error_returns_the_500_envelope_with_cors() -> None:
    app = create_app(ApiSettings(_env_file=None), open_fake_stream, ThreadStore())

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("boom")

    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.get("/boom", headers={"Origin": ORIGIN})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_server_error"
    assert response.headers["access-control-allow-origin"] == ORIGIN
