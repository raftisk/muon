import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import SecretStr

from muon.agent.ask import AskRequest, AskResult
from muon.agent.routing import AgentRuntime
from muon.agent.universes import UniverseRegistry, discover_universes
from muon.config import AgentSettings, Settings
from muon.evaluation.eval_scores import compare_nodes, score_answer
from muon.evaluation.questions import EvalError, EvalSet, load_questions
from muon.evaluation.run import (
    EvalRequest,
    EvalRun,
    RunRecord,
    RunStatus,
    format_row,
    format_summary,
    run_eval,
    summarize,
    write_run,
)
from muon.graph import GraphConnectionError
from muon.llm.client import ModelCallError
from tests.fixtures.toy_ontology import TOY_UNIVERSE, copy_toy_ontology
from tests.fixtures.toy_plugin import write_toy_plugin
from tests.unit.chat_fakes import ScriptedChatModel
from tests.unit.evaluation.fakes import (
    EVALS_FILE_NAME,
    WIDGET_QUERY,
    build_ask_result,
    build_claim,
    build_eval_client,
    build_evidence,
    build_question_toml,
    build_widget_node,
)
from tests.unit.retrieval.fakes import QueryGraphClient
from tests.unit.retrieval.scope import build_toy_settings

STARTED_AT = datetime(2026, 11, 16, 14, 30, 5, tzinfo=UTC)
RUN_ID = "20261116-143005"
SECOND_QUERY = "MATCH (w:ToyWidget:muontoy) RETURN w LIMIT 1"
GOLD_QUERIES = {"Q01": WIDGET_QUERY, "Q02": SECOND_QUERY, "Q03": WIDGET_QUERY}


@dataclass
class Setup:
    registry: UniverseRegistry
    settings: Settings
    eval_set: EvalSet
    client: QueryGraphClient
    runtime: AgentRuntime
    notes_path: Path


@dataclass
class FakeAsk:
    """Stands in for `ask`: answers by question text, or raises the error set for it."""

    outcomes: dict[str, AskResult | Exception]
    asked: list[str] = field(default_factory=list)

    async def __call__(
        self,
        request: AskRequest,
        registry: UniverseRegistry,
        settings: Settings,
        runtime: AgentRuntime,
    ) -> AskResult:
        self.asked.append(request.question)
        outcome = self.outcomes[request.question]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def build_setup(tmp_path: Path) -> Setup:
    ontology_dir = copy_toy_ontology(tmp_path / "ontology")
    universes_dir = tmp_path / "universes"
    universes_dir.mkdir()
    folder = write_toy_plugin(universes_dir, evals=EVALS_FILE_NAME)
    text = "".join(
        build_question_toml(question_id, gold_query=query)
        for question_id, query in GOLD_QUERIES.items()
    )
    (folder / EVALS_FILE_NAME).write_text(text, encoding="utf-8")
    registry = discover_universes(universes_dir, ontology_dir)
    rows = [{"w": build_widget_node("a")}, {"w": build_widget_node("b")}]
    client = build_eval_client(
        {WIDGET_QUERY: lambda parameters: rows, SECOND_QUERY: lambda parameters: rows[:1]}
    )
    agent_settings = AgentSettings(
        _env_file=None, base_url="https://a.test/v1", api_key=SecretStr("k"), model="model-x"
    )
    plugin = registry.get(TOY_UNIVERSE)
    model = ScriptedChatModel(replies=[], drafts=[])
    return Setup(
        registry=registry,
        settings=build_toy_settings(tmp_path, ontology_dir),
        eval_set=load_questions(plugin),
        client=client,
        runtime=AgentRuntime(client=client, chat_model=model, agent_settings=agent_settings),
        notes_path=plugin.notes_path,
    )


def build_request(
    setup: Setup, ids: list[str] | None = None, records: list[RunRecord] | None = None
) -> EvalRequest:
    questions = setup.eval_set.questions if ids is None else setup.eval_set.select(ids)
    return EvalRequest(
        prefix=TOY_UNIVERSE,
        questions=questions,
        source_hash=setup.eval_set.source_hash,
        settings=setup.settings,
        started_at=STARTED_AT,
        on_result=None if records is None else records.append,
    )


def build_good_result() -> AskResult:
    return build_ask_result(
        claims=[build_claim("c1", "e1", ["a"])], items=[build_evidence("e1", ["a"])]
    )


def build_outcomes(**overrides: AskResult | Exception) -> dict[str, AskResult | Exception]:
    outcomes: dict[str, AskResult | Exception] = {
        "Question Q01?": build_good_result(),
        "Question Q02?": build_good_result(),
        "Question Q03?": build_good_result(),
    }
    return {**outcomes, **{f"Question {key}?": value for key, value in overrides.items()}}


async def run_with(setup: Setup, outcomes: dict[str, AskResult | Exception]) -> EvalRun:
    return await run_eval(build_request(setup), setup.registry, setup.runtime, FakeAsk(outcomes))


async def test_questions_run_in_file_order_with_scores(tmp_path: Path) -> None:
    setup = build_setup(tmp_path)
    fake = FakeAsk(build_outcomes())

    run = await run_eval(build_request(setup), setup.registry, setup.runtime, fake)

    assert fake.asked == ["Question Q01?", "Question Q02?", "Question Q03?"]
    assert [record.question_id for record in run.records] == ["Q01", "Q02", "Q03"]
    first = run.records[0]
    assert first.status is RunStatus.OK
    assert first.score is not None
    assert (first.score.precision, first.score.recall) == (1.0, 0.5)
    assert run.records[1].score is not None
    assert run.records[1].score.recall == 1.0
    assert run.stop is None


async def test_a_gold_failure_stops_before_any_question_is_asked(tmp_path: Path) -> None:
    setup = build_setup(tmp_path)
    setup.client.answers[SECOND_QUERY] = lambda parameters: []
    fake = FakeAsk(build_outcomes())

    with pytest.raises(EvalError, match="Gold query of Q02 returned no rows"):
        await run_eval(build_request(setup), setup.registry, setup.runtime, fake)

    assert fake.asked == []


async def test_a_model_error_is_recorded_and_the_run_goes_on(tmp_path: Path) -> None:
    setup = build_setup(tmp_path)
    error = ModelCallError("retrieve", "model-x", "rate limited")

    run = await run_with(setup, build_outcomes(Q02=error))

    statuses = [record.status for record in run.records]
    assert statuses == [RunStatus.OK, RunStatus.ERROR, RunStatus.OK]
    failed = run.records[1]
    assert failed.error is not None
    assert "rate limited" in failed.error
    assert (failed.result, failed.score, failed.comparison) == (None, None, None)
    assert summarize(run).error_count == 1


async def test_a_lost_connection_keeps_the_finished_questions(tmp_path: Path) -> None:
    setup = build_setup(tmp_path)

    run = await run_with(setup, build_outcomes(Q02=GraphConnectionError("Neo4j is down")))

    assert [record.question_id for record in run.records] == ["Q01"]
    assert run.stop == "Q02: Neo4j is down"
    assert "stopped: Q02: Neo4j is down" in format_summary(run)


async def test_on_result_fires_once_per_question(tmp_path: Path) -> None:
    setup = build_setup(tmp_path)
    seen: list[RunRecord] = []

    run = await run_eval(
        build_request(setup, records=seen),
        setup.registry,
        setup.runtime,
        FakeAsk(build_outcomes()),
    )

    assert seen == list(run.records)


async def test_a_subset_resolves_and_asks_only_its_questions(tmp_path: Path) -> None:
    setup = build_setup(tmp_path)
    fake = FakeAsk(build_outcomes())

    run = await run_eval(build_request(setup, ["Q02"]), setup.registry, setup.runtime, fake)

    assert fake.asked == ["Question Q02?"]
    assert [record.question_id for record in run.records] == ["Q02"]
    assert setup.client.list_queries().count(SECOND_QUERY) == 1
    assert setup.client.list_queries().count(WIDGET_QUERY) == 0


async def test_meta_holds_the_run_identity_limits_hashes_and_census(tmp_path: Path) -> None:
    setup = build_setup(tmp_path)

    run = await run_with(setup, build_outcomes())

    meta = run.meta
    assert meta.run_id == RUN_ID
    assert (meta.universe, meta.model) == (TOY_UNIVERSE, "model-x")
    assert (meta.max_tool_calls, meta.max_cypher_calls, meta.max_check_retries) == (10, 2, 1)
    assert set(meta.hashes) == {"questions", "retrieve_prompt", "generate_prompt", "notes"}
    assert meta.hashes["questions"] == setup.eval_set.source_hash
    assert meta.census.node_counts == {"ToyColor": 2, "ToyGadget": 1, "ToyWidget": 3}
    assert meta.census.relationship_counts == {"HAS_COLOR": 3, "PAIRS_WITH": 4}


async def test_write_run_creates_the_folder_files(tmp_path: Path) -> None:
    setup = build_setup(tmp_path)
    run = await run_with(setup, build_outcomes(Q02=ModelCallError("retrieve", "m", "boom")))

    folder = write_run(run, tmp_path / "out")

    assert folder == tmp_path / "out" / RUN_ID
    answers = sorted(path.name for path in (folder / "answers").iterdir())
    assert answers == ["Q01.json", "Q02.json", "Q03.json"]
    document = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert set(document) == {"meta", "totals", "stop", "questions"}
    first = document["questions"][0]
    assert first["score"]["precision"] == 1.0
    assert "comparison" not in first
    assert document["questions"][1]["error"]
    assert document["totals"]["error_count"] == 1


async def test_stored_answers_re_score_to_the_stored_scores(tmp_path: Path) -> None:
    setup = build_setup(tmp_path)
    run = await run_with(setup, build_outcomes())
    folder = write_run(run, tmp_path / "out")

    for record in run.records:
        stored = RunRecord.model_validate_json(
            (folder / "answers" / f"{record.question_id}.json").read_text(encoding="utf-8")
        )
        assert stored.result is not None
        assert stored.comparison is not None
        comparison = compare_nodes(stored.result, stored.comparison.gold)
        assert comparison == stored.comparison
        assert score_answer(stored.result, comparison) == stored.score


async def test_review_sheet_has_a_section_per_question(tmp_path: Path) -> None:
    setup = build_setup(tmp_path)
    run = await run_with(setup, build_outcomes(Q02=ModelCallError("retrieve", "m", "boom")))

    sheet = (write_run(run, tmp_path / "out") / "review.md").read_text(encoding="utf-8")

    assert sheet.count("Verdict: [ ] correct  [ ] partial  [ ] wrong") == 3
    assert "## Q01 (S1)" in sheet
    assert "Reference: Answer Q01." in sheet
    assert "Answer: Answer text." in sheet
    assert f"```cypher\n{WIDGET_QUERY}\n```" in sheet
    assert "Missed gold nodes: b (ToyWidget b)" in sheet
    assert "Extra cited nodes: none" in sheet
    assert "Error: " in sheet
    assert "boom" in sheet


async def test_row_and_summary_lines(tmp_path: Path) -> None:
    setup = build_setup(tmp_path)
    run = await run_with(setup, build_outcomes(Q03=ModelCallError("retrieve", "m", "boom")))

    first, failed = format_row(run.records[0]), format_row(run.records[2])
    summary = format_summary(run)

    assert first.startswith("Q01   S1")
    assert "P 1.00  R 0.50  valid yes" in first
    assert "tools  1" in first
    assert "180 tokens" in first
    assert failed.endswith("error: " + str(run.records[2].error))
    assert summary.startswith("questions 3  errors 1  P 1.00  R 0.75  citations valid 2/2")
