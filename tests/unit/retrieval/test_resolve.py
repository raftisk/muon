from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import SecretStr

from muon.agent.routing import AgentRuntime, UniverseScope
from muon.config import AgentSettings
from muon.graph import GraphNode, GraphQueryError
from muon.llm.client import ModelCallError
from muon.retrieval.evidence import NodeKey
from muon.retrieval.resolve import (
    FETCH_LIMIT,
    LOOK_UP_NAMES,
    MAX_CANDIDATES,
    MIN_NAME_SCORE,
    SHOW_NAME_INDEX,
    NameIndexError,
    NameMention,
    NameMentions,
    build_name_query,
    find_names,
    look_up_names,
    rank_candidates,
    resolve_names,
)
from muon.retrieval.resolved_names import NameCandidate, ResolvedName, render_resolved_names
from tests.unit.chat_fakes import ScriptedChatModel
from tests.unit.retrieval.fakes import QueryGraphClient
from tests.unit.retrieval.scope import build_toy_scope
from tests.unit.retrieval.toy_graph import GADGET_SET, UNIVERSE, WIDGET_SET

QUESTION = "Which colors does Widget One have?"


@pytest.fixture
async def scope(tmp_path: Path) -> UniverseScope:
    return await build_toy_scope(tmp_path)


def build_candidate(label: str, node_id: str, score: float) -> NameCandidate:
    return NameCandidate(key=NodeKey(label=label, id=node_id), name=node_id.title(), score=score)


def build_row(position: int, labels: list[str], node_id: str, score: float) -> dict[str, Any]:
    node = GraphNode(f"el-{node_id}", frozenset(labels), {"id": node_id, "name": node_id.title()})
    return {"position": position, "node": node, "score": score}


def build_client(rows: list[dict[str, Any]]) -> QueryGraphClient:
    return QueryGraphClient({LOOK_UP_NAMES: lambda parameters: rows})


def build_runtime(model: BaseChatModel, client: QueryGraphClient | None = None) -> AgentRuntime:
    settings = AgentSettings(
        _env_file=None, base_url="https://a.test/v1", api_key=SecretStr("k"), default_model="m"
    )
    return AgentRuntime(
        client=client or build_client([]),  # type: ignore[arg-type]
        resolve_model=model,
        retrieve_model=model,
        generate_model=model,
        agent_settings=settings,
    )


class FailingStructuredModel(ScriptedChatModel):
    """Structured output raises a provider error, or returns an unparsed reply."""

    error: Exception | None = None
    parsing_error: str = "bad json"

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:  # type: ignore[override]
        def answer(messages: Any) -> dict[str, Any]:
            if self.error is not None:
                raise self.error
            return {"raw": None, "parsed": None, "parsing_error": self.parsing_error}

        return RunnableLambda(answer)


def test_the_query_escapes_lucene_characters() -> None:
    assert build_name_query("Type: Null") == "name:(Type\\:~1 Null~1)^2 aliases:(Type\\:~1 Null~1)"
    assert build_name_query("ab+cd") == "name:(ab\\+cd~1)^2 aliases:(ab\\+cd~1)"
    assert build_name_query("Farfetch'd") == "name:(Farfetch'd~1)^2 aliases:(Farfetch'd~1)"


def test_the_query_keeps_symbols_and_dots_in_a_word() -> None:
    assert build_name_query("Nidoran♀") == "name:(Nidoran♀~1)^2 aliases:(Nidoran♀~1)"
    assert build_name_query("Mt.") == "name:(Mt.)^2 aliases:(Mt.)"


def test_only_words_of_4_or_more_characters_are_fuzzy() -> None:
    assert build_name_query("Route 1 Alola") == (
        "name:(Route~1 1 Alola~1)^2 aliases:(Route~1 1 Alola~1)"
    )
    assert build_name_query("Mr Mime") == "name:(Mr Mime~1)^2 aliases:(Mr Mime~1)"


def test_the_name_field_weighs_more_than_the_aliases() -> None:
    query = build_name_query("Garchomp")

    assert query.startswith("name:(") and ")^2 aliases:(" in query


def test_a_mention_without_a_letter_or_digit_has_no_query() -> None:
    assert build_name_query("- & ?") == ""
    assert build_name_query("") == ""


def test_ranking_puts_the_guessed_label_first_then_orders_by_score() -> None:
    candidates = [
        build_candidate("Move", "dragon-claw", 11.0),
        build_candidate("EggGroup", "dragon", 7.1),
        build_candidate("Type", "dragon", 7.1),
    ]

    ranked = rank_candidates(candidates, "EggGroup")

    assert [(item.key.label, item.key.id) for item in ranked] == [
        ("EggGroup", "dragon"),
        ("Move", "dragon-claw"),
        ("Type", "dragon"),
    ]


def test_a_score_tie_goes_to_the_shorter_name_then_the_lower_id() -> None:
    long_name = NameCandidate(
        key=NodeKey(label="Place", id="a"), name="Coronet Highlands", score=8.1
    )
    short_name = NameCandidate(key=NodeKey(label="Place", id="z"), name="Mount Coronet", score=8.1)
    same_length = NameCandidate(key=NodeKey(label="Place", id="b"), name="Mount Coronet", score=8.1)

    ranked = rank_candidates([long_name, short_name, same_length], "Place")

    assert [item.key.id for item in ranked] == ["b", "z", "a"]


def test_a_guess_that_matches_nothing_keeps_the_score_order() -> None:
    candidates = [build_candidate("Move", "a", 5.0), build_candidate("Type", "b", 9.0)]

    assert [item.key.id for item in rank_candidates(candidates, "Nope")] == ["b", "a"]
    assert [item.key.id for item in rank_candidates(candidates, "")] == ["b", "a"]


def test_the_cap_keeps_the_best_candidates_and_drops_none_for_its_label() -> None:
    moves = [build_candidate("Move", f"m{number}", 20.0 - number) for number in range(14)]
    wrong_guess_target = build_candidate("Type", "dragon", 7.1)
    lowest = build_candidate("Move", "last", 1.0)

    ranked = rank_candidates([*moves, wrong_guess_target, lowest], "Nope")

    assert len(ranked) == MAX_CANDIDATES
    assert wrong_guess_target in ranked
    assert lowest not in ranked


async def test_the_lookup_sends_one_query_with_the_limits_as_parameters(
    scope: UniverseScope,
) -> None:
    client = build_client(
        [build_row(0, WIDGET_SET, "w1", 6.3), build_row(1, GADGET_SET, "g1", 5.0)]
    )
    mentions = [
        NameMention(mention="Widget One", label="ToyWidget"),
        NameMention(mention="G", label=""),
    ]

    names = await look_up_names(mentions, scope, client)

    assert len(client.reads) == 1
    parameters = client.reads[0][1]
    assert parameters["min_score"] == MIN_NAME_SCORE
    assert parameters["fetch_limit"] == FETCH_LIMIT
    assert parameters["universe"] == UNIVERSE
    assert [query["position"] for query in parameters["queries"]] == [0, 1]
    assert [item.mention for item in names] == ["Widget One", "G"]
    assert names[0].candidates[0].key == NodeKey(label="ToyWidget", id="w1")
    assert names[1].candidates[0].key == NodeKey(label="ToyGadget", id="g1")


async def test_a_mention_without_rows_has_no_candidates(scope: UniverseScope) -> None:
    client = build_client([build_row(0, WIDGET_SET, "w1", 6.3)])
    mentions = [NameMention(mention="Widget", label=""), NameMention(mention="Nothing", label="")]

    names = await look_up_names(mentions, scope, client)

    assert [len(item.candidates) for item in names] == [1, 0]


async def test_a_mention_without_a_query_is_not_sent_and_has_no_candidates(
    scope: UniverseScope,
) -> None:
    client = build_client([])

    names = await look_up_names([NameMention(mention="- &", label="")], scope, client)

    assert client.reads == []
    assert names[0].candidates == []


async def test_no_mentions_send_no_query(scope: UniverseScope) -> None:
    client = QueryGraphClient({})

    assert await look_up_names([], scope, client) == []
    assert client.reads == []


async def test_a_missing_index_raises_a_clear_error(scope: UniverseScope) -> None:
    def fail(parameters: Mapping[str, Any]) -> list[dict[str, Any]]:
        raise GraphQueryError("There is no such fulltext schema index")

    client = QueryGraphClient({LOOK_UP_NAMES: fail, SHOW_NAME_INDEX: lambda parameters: []})

    with pytest.raises(NameIndexError, match=r"entity_names.*scripts/create_name_index\.py"):
        await look_up_names([NameMention(mention="Widget", label="")], scope, client)


async def test_another_query_failure_is_re_raised(scope: UniverseScope) -> None:
    def fail(parameters: Mapping[str, Any]) -> list[dict[str, Any]]:
        raise GraphQueryError("syntax error")

    def present(parameters: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [{"name": "entity_names"}]

    client = QueryGraphClient({LOOK_UP_NAMES: fail, SHOW_NAME_INDEX: present})

    with pytest.raises(GraphQueryError, match="syntax error"):
        await look_up_names([NameMention(mention="Widget", label="")], scope, client)


async def test_find_names_returns_the_mentions_and_the_step_stats(scope: UniverseScope) -> None:
    reply = NameMentions(names=[NameMention(mention="Widget One", label="ToyWidget")])
    model = ScriptedChatModel(drafts=[reply])

    mentions, stats = await find_names(QUESTION, scope, build_runtime(model))

    assert mentions == [NameMention(mention="Widget One", label="ToyWidget")]
    assert (stats.step, stats.model_calls, stats.input_tokens) == ("resolve", 1, 10)
    system, human = model.calls[0]
    assert "ToyWidget" in str(system.content)
    assert human.content == QUESTION


async def test_a_label_outside_the_scope_becomes_empty(scope: UniverseScope) -> None:
    reply = NameMentions(names=[NameMention(mention="Widget One", label="Nonsense")])

    mentions, _ = await find_names(
        QUESTION, scope, build_runtime(ScriptedChatModel(drafts=[reply]))
    )

    assert mentions == [NameMention(mention="Widget One", label="")]


async def test_a_provider_error_becomes_a_model_call_error(scope: UniverseScope) -> None:
    request = httpx.Request("POST", "https://a.test/v1")
    model = FailingStructuredModel(error=openai.APIConnectionError(request=request))

    with pytest.raises(ModelCallError, match="resolve step"):
        await find_names(QUESTION, scope, build_runtime(model))


async def test_an_unparsable_reply_becomes_a_model_call_error(scope: UniverseScope) -> None:
    with pytest.raises(ModelCallError, match="not a valid name list: bad json"):
        await find_names(QUESTION, scope, build_runtime(FailingStructuredModel()))


async def test_resolve_names_chains_the_model_and_the_lookup(scope: UniverseScope) -> None:
    reply = NameMentions(names=[NameMention(mention="Widget One", label="ToyWidget")])
    client = build_client([build_row(0, WIDGET_SET, "w1", 6.3)])
    runtime = build_runtime(ScriptedChatModel(drafts=[reply]), client)

    result = await resolve_names(QUESTION, scope, runtime)

    assert result.names[0].candidates[0].key.id == "w1"
    assert result.stats.step == "resolve"


async def test_a_question_without_names_runs_no_lookup(scope: UniverseScope) -> None:
    client = QueryGraphClient({})
    runtime = build_runtime(ScriptedChatModel(drafts=[NameMentions(names=[])]), client)

    result = await resolve_names("What are the 10 fastest ones?", scope, runtime)

    assert result.names == []
    assert client.reads == []


def test_the_block_lists_each_name_with_its_candidates_and_scores() -> None:
    names = [
        ResolvedName(
            mention="Mt. Coronet",
            label="Place",
            candidates=[
                build_candidate("Place", "mount-coronet", 6.34),
                build_candidate("Place", "coronet-highlands", 4.1),
            ],
        ),
        ResolvedName(mention="Pikablu", label="Creature", candidates=[]),
        ResolvedName(
            mention="Spikes", label="", candidates=[build_candidate("Move", "spikes", 11.0)]
        ),
    ]

    assert render_resolved_names(names).splitlines() == [
        "Resolved names (full-text match on name and aliases, best first):",
        '- "Mt. Coronet" (guess: Place): Mount-Coronet [Place mount-coronet, score 6.3]; '
        "Coronet-Highlands [Place coronet-highlands, score 4.1]",
        '- "Pikablu" (guess: Creature): no node matched',
        '- "Spikes": Spikes [Move spikes, score 11.0]',
    ]


def test_no_names_render_an_empty_block() -> None:
    assert render_resolved_names([]) == ""
