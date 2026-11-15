from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from muon.agent.routing import AgentRuntime, UniverseScope
from muon.config import AgentSettings
from muon.retrieval.cypher import TOOL_NAME
from muon.retrieval.evidence import EvidenceStatus
from muon.retrieval.retrieve import PROMPT_PATH, PromptError, fill_prompt, retrieve
from tests.unit.chat_fakes import ScriptedChatModel, build_final_message, build_tool_call_message
from tests.unit.retrieval.scope import build_toy_scope
from tests.unit.retrieval.test_cypher import StubReadClient

QUESTION = "What is W1?"
WIDGET_QUERY = "MATCH (w:ToyWidget:muontoy) RETURN w.name AS name"


@pytest.fixture
async def scope(tmp_path: Path) -> UniverseScope:
    return await build_toy_scope(tmp_path)


def build_runtime(
    replies: list[Any], client: StubReadClient, **limits: int
) -> tuple[AgentRuntime, ScriptedChatModel]:
    model = ScriptedChatModel(replies=replies)
    settings = AgentSettings(
        _env_file=None, base_url="https://a.test/v1", api_key=SecretStr("k"), model="m", **limits
    )
    return AgentRuntime(client=client, chat_model=model, agent_settings=settings), model  # type: ignore[arg-type]


async def test_two_tool_calls_give_two_evidence_items_in_order(scope: UniverseScope) -> None:
    client = StubReadClient([{"name": "W1"}])
    runtime, _ = build_runtime(
        [
            build_tool_call_message(WIDGET_QUERY, "c1"),
            build_tool_call_message(WIDGET_QUERY, "c2"),
            build_final_message(),
        ],
        client,
    )

    result = await retrieve(scope, QUESTION, runtime)

    assert [item.id for item in result.bundle.items] == ["e1", "e2"]
    assert result.bundle.universe == scope.universe_label
    assert result.bundle.question == QUESTION
    assert (result.tool_calls, result.cypher_calls) == (2, 2)
    assert len(client.reads) == 2


async def test_a_third_cypher_call_is_blocked_and_adds_no_evidence(scope: UniverseScope) -> None:
    client = StubReadClient([{"name": "W1"}])
    replies = [build_tool_call_message(WIDGET_QUERY, f"c{number}") for number in (1, 2, 3)]
    runtime, _ = build_runtime([*replies, build_final_message()], client)

    result = await retrieve(scope, QUESTION, runtime)

    assert len(result.bundle.items) == 2
    assert len(client.reads) == 2
    assert result.cypher_calls == 2
    assert result.tool_calls == 2


async def test_a_higher_cypher_limit_allows_four_calls(scope: UniverseScope) -> None:
    client = StubReadClient([{"name": "W1"}])
    replies = [build_tool_call_message(WIDGET_QUERY, f"c{number}") for number in (1, 2, 3, 4)]
    runtime, _ = build_runtime([*replies, build_final_message()], client, max_cypher_calls=4)

    result = await retrieve(scope, QUESTION, runtime)

    assert len(result.bundle.items) == 4


async def test_tokens_are_summed_over_the_model_calls(scope: UniverseScope) -> None:
    runtime, _ = build_runtime(
        [build_tool_call_message(WIDGET_QUERY, "c1"), build_final_message()],
        StubReadClient([{"name": "W1"}]),
    )

    result = await retrieve(scope, QUESTION, runtime)

    assert result.stats.step == "retrieve"
    assert result.stats.input_tokens == 20
    assert result.stats.output_tokens == 10


async def test_the_system_prompt_holds_the_schema_and_the_question_is_the_human_message(
    scope: UniverseScope,
) -> None:
    runtime, model = build_runtime([build_final_message()], StubReadClient([]))

    await retrieve(scope, QUESTION, runtime)

    system, human = model.calls[0][0], model.calls[0][1]
    assert scope.universe_label in str(system.content)
    assert "# Graph schema" in str(system.content)
    assert human.content == QUESTION
    assert TOOL_NAME in PROMPT_PATH.read_text(encoding="utf-8")


async def test_a_rejected_call_is_kept_as_evidence(scope: UniverseScope) -> None:
    runtime, _ = build_runtime(
        [build_tool_call_message("MATCH (p:Pokemon) RETURN p", "c1"), build_final_message()],
        StubReadClient([]),
    )

    result = await retrieve(scope, QUESTION, runtime)

    assert result.bundle.items[0].status is EvidenceStatus.REJECTED


def test_fill_prompt_replaces_markers_only(tmp_path: Path) -> None:
    path = tmp_path / "p.md"
    path.write_text("A {{one}} and {x: $y} and {{two}}", encoding="utf-8")

    assert fill_prompt(path, {"one": "1", "two": "{{one}}"}) == "A 1 and {x: $y} and {{one}}"


def test_fill_prompt_rejects_a_missing_and_an_extra_value(tmp_path: Path) -> None:
    path = tmp_path / "p.md"
    path.write_text("A {{one}}", encoding="utf-8")

    with pytest.raises(PromptError, match="two"):
        fill_prompt(path, {"one": "1", "two": "2"})
    with pytest.raises(PromptError, match="one"):
        fill_prompt(path, {})


def test_retrieve_prompt_fills_with_every_marker() -> None:
    text = fill_prompt(
        PROMPT_PATH, {"universe_name": "U", "universe_label": "ulabel", "schema": "SCHEMA"}
    )

    assert "{{" not in text
    assert "SCHEMA" in text
