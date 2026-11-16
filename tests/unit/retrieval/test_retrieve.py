from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from pydantic import SecretStr

from muon.agent.routing import AgentRuntime, UniverseScope
from muon.config import AgentSettings
from muon.retrieval.cypher import TOOL_NAME
from muon.retrieval.evidence import Evidence, EvidenceStatus, NodeKey
from muon.retrieval.resolved_names import NameCandidate, ResolvedName, render_resolved_names
from muon.retrieval.retrieve import (
    PROMPT_PATH,
    PromptError,
    TokenUsage,
    build_question_message,
    fill_prompt,
    is_done,
    read_step_usage,
    retrieve,
)
from tests.unit.chat_fakes import (
    ScriptedChatModel,
    build_final_message,
    build_tool_call_message,
    build_usage,
)
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
        _env_file=None,
        base_url="https://a.test/v1",
        api_key=SecretStr("k"),
        default_model="m",
        **limits,
    )
    return AgentRuntime(
        client=client,
        resolve_model=model,
        retrieve_model=model,
        generate_model=model,
        agent_settings=settings,
    ), model  # type: ignore[arg-type]


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

    result = await retrieve(scope, QUESTION, runtime, [])

    assert [item.id for item in result.bundle.items] == ["e1", "e2"]
    assert result.bundle.universe == scope.universe_label
    assert result.bundle.question == QUESTION
    assert (result.tool_calls, result.cypher_calls) == (2, 2)
    assert len(client.reads) == 2


async def test_the_loop_ends_when_the_cypher_budget_is_spent(scope: UniverseScope) -> None:
    client = StubReadClient([{"name": "W1"}])
    runtime, model = build_runtime(
        [build_tool_call_message(WIDGET_QUERY, "c1"), build_tool_call_message(WIDGET_QUERY, "c2")],
        client,
    )

    result = await retrieve(scope, QUESTION, runtime, [])

    assert result.stats.model_calls == 2
    assert len(model.calls) == 2
    assert len(result.bundle.items) == 2
    assert len(client.reads) == 2


async def test_the_loop_ends_after_a_rejected_last_query(scope: UniverseScope) -> None:
    runtime, model = build_runtime(
        [
            build_tool_call_message(WIDGET_QUERY, "c1"),
            build_tool_call_message("MATCH (p:Pokemon) RETURN p", "c2"),
        ],
        StubReadClient([{"name": "W1"}]),
    )

    result = await retrieve(scope, QUESTION, runtime, [])

    assert len(model.calls) == 2
    assert result.bundle.items[1].status is EvidenceStatus.REJECTED


async def test_a_parallel_call_past_the_budget_is_blocked_and_ends_the_loop(
    scope: UniverseScope,
) -> None:
    client = StubReadClient([{"name": "W1"}])
    parallel = build_tool_call_message(WIDGET_QUERY, "c1")
    parallel.tool_calls.append({**parallel.tool_calls[0], "id": "c2"})
    runtime, model = build_runtime([parallel], client, max_cypher_calls=1)

    result = await retrieve(scope, QUESTION, runtime, [])

    assert len(result.bundle.items) == 1
    assert len(client.reads) == 1
    assert len(model.calls) == 1


async def test_a_final_text_reply_before_the_budget_is_spent_costs_one_more_call(
    scope: UniverseScope,
) -> None:
    runtime, model = build_runtime(
        [build_tool_call_message(WIDGET_QUERY, "c1"), build_final_message()],
        StubReadClient([{"name": "W1"}]),
    )

    result = await retrieve(scope, QUESTION, runtime, [])

    assert result.stats.model_calls == 2
    assert len(model.calls) == 2


def build_item(
    number: int, status: EvidenceStatus = EvidenceStatus.OK, is_final: bool = False
) -> Evidence:
    return Evidence(
        id=f"e{number}",
        universe="toy",
        tool=TOOL_NAME,
        query="q",
        purpose="p",
        status=status,
        row_count=1 if status is EvidenceStatus.OK else 0,
        is_final=is_final,
    )


@pytest.mark.parametrize("status", list(EvidenceStatus))
def test_a_final_item_ends_the_loop_only_when_it_has_rows(status: EvidenceStatus) -> None:
    items = [build_item(1, status, is_final=True)]

    assert is_done(items, max_cypher_calls=2) is (status is EvidenceStatus.OK)


def test_a_non_final_item_with_rows_and_budget_left_continues() -> None:
    assert is_done([build_item(1)], max_cypher_calls=2) is False


def test_only_the_latest_item_decides_the_final_rule() -> None:
    items = [build_item(1, is_final=True), build_item(2)]

    assert is_done(items, max_cypher_calls=3) is False


@pytest.mark.parametrize("status", list(EvidenceStatus))
def test_a_spent_budget_ends_the_loop_for_every_status(status: EvidenceStatus) -> None:
    assert is_done([build_item(1), build_item(2, status)], max_cypher_calls=2) is True


def test_no_items_never_end_the_loop() -> None:
    assert is_done([], max_cypher_calls=2) is False


async def test_a_final_query_with_rows_ends_the_loop_after_one_model_call(
    scope: UniverseScope,
) -> None:
    runtime, model = build_runtime(
        [build_tool_call_message(WIDGET_QUERY, "c1", is_final=True)],
        StubReadClient([{"name": "W1"}]),
    )

    result = await retrieve(scope, QUESTION, runtime, [])

    assert result.stats.model_calls == 1
    assert len(model.calls) == 1
    assert result.bundle.items[0].is_final is True


async def test_a_final_query_with_no_rows_continues_to_a_second_model_call(
    scope: UniverseScope,
) -> None:
    runtime, model = build_runtime(
        [build_tool_call_message(WIDGET_QUERY, "c1", is_final=True), build_final_message()],
        StubReadClient([]),
    )

    result = await retrieve(scope, QUESTION, runtime, [])

    assert result.bundle.items[0].status is EvidenceStatus.EMPTY
    assert result.stats.model_calls == 2
    assert len(model.calls) == 2


async def test_a_higher_cypher_limit_allows_four_calls(scope: UniverseScope) -> None:
    client = StubReadClient([{"name": "W1"}])
    replies = [build_tool_call_message(WIDGET_QUERY, f"c{number}") for number in (1, 2, 3, 4)]
    runtime, _ = build_runtime([*replies, build_final_message()], client, max_cypher_calls=4)

    result = await retrieve(scope, QUESTION, runtime, [])

    assert len(result.bundle.items) == 4


async def test_tokens_are_summed_over_the_model_calls(scope: UniverseScope) -> None:
    runtime, _ = build_runtime(
        [build_tool_call_message(WIDGET_QUERY, "c1"), build_final_message()],
        StubReadClient([{"name": "W1"}]),
    )

    result = await retrieve(scope, QUESTION, runtime, [])

    assert result.stats.step == "retrieve"
    assert result.stats.input_tokens == 20
    assert result.stats.output_tokens == 10


async def test_reasoning_and_cached_tokens_are_summed_over_the_model_calls(
    scope: UniverseScope,
) -> None:
    first = build_tool_call_message(WIDGET_QUERY, "c1")
    first.usage_metadata = build_usage(100, 40, reasoning=30, cache_read=64)  # type: ignore[assignment]
    last = AIMessage(
        content="done", usage_metadata=build_usage(150, 20, reasoning=10, cache_read=96)
    )  # type: ignore[arg-type]
    runtime, _ = build_runtime([first, last], StubReadClient([{"name": "W1"}]))

    result = await retrieve(scope, QUESTION, runtime, [])

    stats = result.stats
    assert (stats.input_tokens, stats.output_tokens) == (250, 60)
    assert (stats.reasoning_tokens, stats.cached_input_tokens) == (40, 160)


async def test_missing_token_details_count_as_zero(scope: UniverseScope) -> None:
    runtime, _ = build_runtime(
        [build_tool_call_message(WIDGET_QUERY, "c1"), build_final_message()],
        StubReadClient([{"name": "W1"}]),
    )

    result = await retrieve(scope, QUESTION, runtime, [])

    assert (result.stats.reasoning_tokens, result.stats.cached_input_tokens) == (0, 0)


async def test_a_two_turn_loop_makes_two_model_calls(scope: UniverseScope) -> None:
    runtime, _ = build_runtime(
        [build_tool_call_message(WIDGET_QUERY, "c1"), build_final_message()],
        StubReadClient([{"name": "W1"}]),
    )

    result = await retrieve(scope, QUESTION, runtime, [])

    assert result.stats.model_calls == 2


def test_read_step_usage_skips_messages_that_are_not_ai_messages() -> None:
    usage = read_step_usage(["text", AIMessage(content="x")])

    assert usage == TokenUsage(0, 0, 0, 0, 1)


async def test_the_system_prompt_holds_the_schema_and_the_question_is_the_human_message(
    scope: UniverseScope,
) -> None:
    runtime, model = build_runtime([build_final_message()], StubReadClient([]))

    await retrieve(scope, QUESTION, runtime, [])

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

    result = await retrieve(scope, QUESTION, runtime, [])

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


def build_resolved(candidates: list[NameCandidate]) -> ResolvedName:
    return ResolvedName(mention="Widget One", label="ToyWidget", candidates=candidates)


def test_the_question_message_is_the_bare_question_without_names() -> None:
    assert build_question_message(QUESTION, []).content == QUESTION


def test_the_question_message_puts_the_block_before_the_question() -> None:
    candidate = NameCandidate(key=NodeKey(label="ToyWidget", id="w1"), name="W1", score=6.34)

    message = build_question_message(QUESTION, [build_resolved([candidate])])

    assert message.content == (
        f"{render_resolved_names([build_resolved([candidate])])}\n\nQuestion: {QUESTION}"
    )
    assert str(message.content).startswith("Resolved names")


def test_a_name_without_a_match_still_gets_a_block() -> None:
    message = build_question_message(QUESTION, [build_resolved([])])

    assert "no node matched" in str(message.content)


async def test_the_system_prompt_is_the_same_with_and_without_names(scope: UniverseScope) -> None:
    runtime, model = build_runtime(
        [build_final_message(), build_final_message()], StubReadClient([])
    )
    candidate = NameCandidate(key=NodeKey(label="ToyWidget", id="w1"), name="W1", score=6.3)

    await retrieve(scope, QUESTION, runtime, [])
    await retrieve(scope, QUESTION, runtime, [build_resolved([candidate])])

    assert model.calls[0][0].content == model.calls[1][0].content
    assert model.calls[0][1].content == QUESTION
    assert "w1" in str(model.calls[1][1].content)
