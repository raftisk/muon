from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from muon.agent.routing import UniverseScope
from muon.evaluation.eval_scores import ScoredNode, compare_nodes, resolve_gold, score_answer
from muon.evaluation.questions import EvalError, EvalQuestion
from muon.graph import GraphQueryError
from tests.unit.evaluation.fakes import (
    GENERATE_TOKENS,
    RETRIEVE_TOKENS,
    TOTAL_LATENCY_SECONDS,
    WIDGET_LABEL,
    WIDGET_QUERY,
    build_ask_result,
    build_eval_client,
    build_evidence,
    build_widget_key,
    build_widget_node,
)
from tests.unit.retrieval.scope import build_toy_scope


def build_gold(*widget_ids: str) -> list[ScoredNode]:
    return [ScoredNode(key=build_widget_key(widget_id), name=widget_id) for widget_id in widget_ids]


def build_question(gold_query: str = WIDGET_QUERY, has_empty_gold: bool = False) -> EvalQuestion:
    return EvalQuestion(
        id="Q01",
        family="S1",
        question="Q?",
        reference="A.",
        gold_query=gold_query,
        has_empty_gold=has_empty_gold,
    )


def test_precision_and_recall_count_shared_nodes() -> None:
    result = build_ask_result(cited_widget_ids=["a", "b", "c"], items=[build_evidence("e1", ["a"])])

    comparison = compare_nodes(result, build_gold("b", "c", "d", "e"))
    score = score_answer(result, comparison)

    assert score.precision == pytest.approx(2 / 3)
    assert score.recall == pytest.approx(2 / 4)
    assert [node.key.id for node in comparison.missed] == ["d", "e"]
    assert [node.key.id for node in comparison.extra] == ["a"]


def test_cited_nodes_hold_handle_nodes_and_the_nodes_of_cited_tables() -> None:
    rows = build_evidence("e2", ["x", "y", "z"])
    result = build_ask_result(
        cited_widget_ids=["a"],
        table_evidence_ids=["e2"],
        items=[build_evidence("e1", ["a"]), rows],
    )

    comparison = compare_nodes(result, build_gold("a", "x", "y", "z"))

    assert [node.key.id for node in comparison.cited] == ["a", "x", "y", "z"]
    assert comparison.missed == ()
    assert comparison.extra == ()


def test_a_table_id_that_the_bundle_lacks_adds_no_node() -> None:
    result = build_ask_result(table_evidence_ids=["e9"], items=[build_evidence("e1", ["a"])])

    comparison = compare_nodes(result, build_gold("a"))

    assert comparison.cited == ()


def test_names_come_from_the_slices() -> None:
    result = build_ask_result(cited_widget_ids=["a", "unseen"], items=[build_evidence("e1", ["a"])])

    comparison = compare_nodes(result, build_gold("a"))

    assert [(node.key.id, node.name) for node in comparison.cited] == [
        ("a", "a"),
        ("unseen", None),
    ]


def test_citation_is_valid_only_without_warnings() -> None:
    clean = build_ask_result()
    warned = build_ask_result(warnings=["[n99] names no node in the evidence"])

    assert score_answer(clean, compare_nodes(clean, [])).is_citation_valid is True
    assert score_answer(warned, compare_nodes(warned, [])).is_citation_valid is False


def test_counters_and_tokens_copy_the_stats() -> None:
    result = build_ask_result()

    score = score_answer(result, compare_nodes(result, build_gold("a")))

    assert (score.tool_calls, score.cypher_calls) == (1, 1)
    assert score.latency_seconds == TOTAL_LATENCY_SECONDS
    assert score.input_tokens == RETRIEVE_TOKENS[0] + GENERATE_TOKENS[0]
    assert score.output_tokens == RETRIEVE_TOKENS[1] + GENERATE_TOKENS[1]
    assert (score.reasoning_tokens, score.cached_input_tokens, score.model_calls) == (0, 0, 2)


def test_nothing_cited_against_an_empty_gold_set_scores_1_and_1() -> None:
    result = build_ask_result()

    score = score_answer(result, compare_nodes(result, []))

    assert (score.precision, score.recall) == (1.0, 1.0)


def test_something_cited_against_an_empty_gold_set_scores_0_and_1() -> None:
    result = build_ask_result(cited_widget_ids=["a"], items=[build_evidence("e1", ["a"])])

    score = score_answer(result, compare_nodes(result, []))

    assert (score.precision, score.recall) == (0.0, 1.0)


def test_nothing_cited_against_a_gold_set_scores_0_and_0() -> None:
    result = build_ask_result()

    score = score_answer(result, compare_nodes(result, build_gold("a")))

    assert (score.precision, score.recall) == (0.0, 0.0)


@pytest.fixture
async def scope(tmp_path: Path) -> UniverseScope:
    return await build_toy_scope(tmp_path)


def answer_rows(rows: list[dict[str, Any]]) -> Mapping[str, Any]:
    return {WIDGET_QUERY: lambda parameters: rows}


async def test_resolve_gold_returns_the_sorted_nodes_of_the_rows(scope: UniverseScope) -> None:
    rows = [{"w": build_widget_node("b")}, {"w": build_widget_node("a")}]
    client = build_eval_client(answer_rows(rows))

    gold = await resolve_gold(build_question(), scope, client)

    assert [node.key.id for node in gold] == ["a", "b"]
    assert gold[0].key.label == WIDGET_LABEL


async def test_resolve_gold_rejects_an_empty_result(scope: UniverseScope) -> None:
    client = build_eval_client(answer_rows([]))

    with pytest.raises(EvalError, match="Gold query of Q01 returned no rows"):
        await resolve_gold(build_question(), scope, client)


async def test_resolve_gold_accepts_an_empty_result_when_the_question_expects_it(
    scope: UniverseScope,
) -> None:
    client = build_eval_client(answer_rows([]))

    gold = await resolve_gold(build_question(has_empty_gold=True), scope, client)

    assert gold == ()


async def test_resolve_gold_rejects_rows_when_the_question_expects_none(
    scope: UniverseScope,
) -> None:
    client = build_eval_client(answer_rows([{"w": build_widget_node("a")}]))

    with pytest.raises(EvalError, match="Gold query of Q01 returned rows, but .*has_empty_gold"):
        await resolve_gold(build_question(has_empty_gold=True), scope, client)


async def test_resolve_gold_rejects_rows_without_a_node(scope: UniverseScope) -> None:
    client = build_eval_client(answer_rows([{"name": "a"}]))

    with pytest.raises(EvalError, match="returned rows with no node"):
        await resolve_gold(build_question(), scope, client)


async def test_resolve_gold_rejects_a_term_outside_the_schema(scope: UniverseScope) -> None:
    client = build_eval_client({})

    with pytest.raises(EvalError, match="Gold query of Q01 is rejected: .*Nope"):
        await resolve_gold(build_question("MATCH (n:Nope) RETURN n"), scope, client)


async def test_resolve_gold_reports_a_failed_query(scope: UniverseScope) -> None:
    def fail(parameters: Mapping[str, Any]) -> list[dict[str, Any]]:
        raise GraphQueryError("syntax error")

    client = build_eval_client({WIDGET_QUERY: fail})

    with pytest.raises(EvalError, match="Gold query of Q01 is failed: .*syntax error"):
        await resolve_gold(build_question(), scope, client)
