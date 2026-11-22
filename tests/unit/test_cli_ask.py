import json
from functools import partial

import pytest
from pydantic import SecretStr

from muon import cli
from muon.agent import routing
from muon.agent.ask import AskRequest
from muon.agent.events import AskResult, AskStats
from muon.config import AgentSettings, Neo4jSettings, Settings
from muon.generation.contract import CitedAnswer, CitedHandles, Source
from muon.graph import GraphConnectionError
from muon.llm.client import ModelCallError
from muon.retrieval.evidence import (
    Evidence,
    EvidenceBundle,
    EvidenceStatus,
    GraphSlice,
    NodeCell,
    NodeKey,
    SliceNode,
)
from muon.retrieval.resolve import NameIndexError
from muon.retrieval.resolved_names import NameCandidate, ResolvedName
from muon.retrieval.retrieve import StepStats

UNIVERSE = "pkmn"
QUESTION = "What type is Diglett?"
SPECIES = NodeKey(label="PokemonSpecies", id="diglett")
TYPE = NodeKey(label="Type", id="ground")
PAGE_URL = "https://bulbapedia.test/w/index.php?title=Diglett&oldid=1"
LINKED_TEXT = (
    f'Diglett is Ground type [[1]]({PAGE_URL} "Diglett") and Ground [[2]]({PAGE_URL} "Ground").'
)
QUERY = "MATCH (s:pkmn:PokemonSpecies {id: 'diglett'}) RETURN s"
SECRET = SecretStr("secret")


@pytest.fixture(autouse=True)
def agent_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("BASE_URL", "API_KEY", "DEFAULT_MODEL"):
        monkeypatch.setenv(f"MUON_AGENT_{name}", "x")
    monkeypatch.setattr(cli, "AgentSettings", partial(AgentSettings, _env_file=None))


@pytest.fixture(autouse=True)
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    loaded = Settings(
        neo4j=Neo4jSettings(uri="neo4j://x", username="neo4j", password=SECRET, _env_file=None)
    )
    monkeypatch.setattr(cli, "get_settings", lambda: loaded)
    return loaded


def build_slice() -> GraphSlice:
    nodes = (
        SliceNode(key=SPECIES, name="Diglett", labels=("PokemonSpecies",), source_url=None),
        SliceNode(key=TYPE, name="Ground", labels=("Type",), source_url=None),
    )
    return GraphSlice(nodes=nodes)


def build_evidence(evidence_id: str, row_count: int = 1) -> Evidence:
    rows = tuple({"species": NodeCell(key=SPECIES), "n": index} for index in range(row_count))
    return Evidence(
        id=evidence_id,
        universe=UNIVERSE,
        tool="run_cypher",
        query=QUERY,
        purpose="type of Diglett",
        status=EvidenceStatus.OK,
        columns=("species", "n"),
        rows=rows,
        row_count=row_count,
        graph_slice=build_slice(),
    )


def build_result(
    text: str = LINKED_TEXT,
    table_ids: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
    resolved_names: list[ResolvedName] | None = None,
) -> AskResult:
    sources = (
        Source(number=1, url=PAGE_URL, title="Diglett"),
        Source(number=2, url=PAGE_URL, title="Ground"),
    )
    cited = CitedHandles(
        node_keys=(SPECIES, TYPE), table_ids=table_ids, sources=sources, warnings=warnings
    )
    answer = CitedAnswer(
        text=text, sources=sources, graph=None, tables=(), cited=cited, warnings=warnings
    )
    return AskResult(
        answer=answer,
        universe=UNIVERSE,
        bundle=EvidenceBundle(
            universe=UNIVERSE, question=QUESTION, items=(build_evidence("e1", row_count=3),)
        ),
        resolved_names=resolved_names or [],
        stats=AskStats(
            tool_calls=1,
            cypher_calls=1,
            steps=[
                StepStats(
                    step="retrieve",
                    input_tokens=7,
                    output_tokens=3,
                    reasoning_tokens=0,
                    cached_input_tokens=0,
                    model_calls=2,
                    latency_seconds=1,
                )
            ],
            total_latency_seconds=2.0,
        ),
    )


class RecordingAskRun:
    """Stands in for `cli.run_ask`."""

    def __init__(self, outcome: AskResult | Exception) -> None:
        self.outcome = outcome
        self.requests: list[AskRequest] = []

    async def __call__(self, context: cli.AgentContext, request: AskRequest) -> AskResult:
        self.requests.append(request)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def run_ask_command(
    monkeypatch: pytest.MonkeyPatch, outcome: AskResult | Exception, *flags: str
) -> RecordingAskRun:
    run = RecordingAskRun(outcome)
    monkeypatch.setattr(cli, "run_ask", run)
    cli.main(["ask", QUESTION, *flags])
    return run


def test_ask_parser_accepts_its_options() -> None:
    arguments = cli.build_parser().parse_args(
        ["ask", QUESTION, "--universe", UNIVERSE, "--json", "--show-cypher"]
    )

    assert (arguments.question, arguments.universe) == (QUESTION, UNIVERSE)
    assert (arguments.is_json, arguments.should_show_cypher) == (True, True)


def test_json_stdout_is_the_model_dump_byte_for_byte(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = build_result()

    run_ask_command(monkeypatch, result, "--json")

    expected = result.model_dump_json(indent=cli.ASK_JSON_INDENT, by_alias=True) + "\n"
    assert capsys.readouterr().out == expected


def test_ask_prints_the_raw_markdown_text_and_never_ids(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = run_ask_command(monkeypatch, build_result())

    captured = capsys.readouterr()
    assert run.requests == [AskRequest(question=QUESTION, universe=UNIVERSE)]
    assert LINKED_TEXT in captured.out
    for hidden in ("e1", "PokemonSpecies", "diglett", "MATCH"):
        assert hidden not in captured.out
    assert "1 queries" in captured.err


def test_a_table_prints_all_its_rows(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = build_result(table_ids=("e1",))
    items = (build_evidence("e1", row_count=120),)
    result = result.model_copy(update={"bundle": result.bundle.model_copy(update={"items": items})})
    run_ask_command(monkeypatch, result)

    lines = capsys.readouterr().out.splitlines()

    header = lines.index("species | n")
    table = lines[header : header + 121]
    assert table[1] == "Diglett | 0"
    assert table[-1] == "Diglett | 119"


def test_json_prints_one_document_with_camel_case_contract_keys(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_ask_command(monkeypatch, build_result(table_ids=("e1",)), "--json", "--show-cypher")

    captured = capsys.readouterr()
    document = json.loads(captured.out)
    assert set(document) == {"answer", "universe", "bundle", "resolved_names", "stats"}
    assert {"sources", "graph", "tables", "warnings", "cited"} <= set(document["answer"])
    assert document["answer"]["sources"][0] == {"number": 1, "url": PAGE_URL, "title": "Diglett"}
    assert document["answer"]["cited"]["tableIds"] == ["e1"]
    assert document["universe"] == UNIVERSE
    assert document["bundle"]["items"][0]["query"] == QUERY
    assert captured.err == ""


def test_show_cypher_lists_the_queries_after_the_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_ask_command(monkeypatch, build_result(), "--show-cypher")

    out = capsys.readouterr().out
    assert out.index("Diglett is Ground") < out.index("Queries")
    assert f"1. type of Diglett [ok, 3 rows]\n   {QUERY}" in out


def test_the_table_and_the_warnings_follow_the_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = build_result(table_ids=("e1",), warnings=("[n99] names no node in the evidence",))

    run_ask_command(monkeypatch, result)

    out = capsys.readouterr().out
    assert (
        out.index("Diglett is Ground")
        < out.index("species | n")
        < out.index("Warnings:\n- [n99] names no node in the evidence")
    )


def test_a_result_with_no_warning_prints_no_warnings_section(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_ask_command(monkeypatch, build_result())

    assert "Warnings:" not in capsys.readouterr().out


def test_missing_agent_settings_exit_1_before_any_connection(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("MUON_AGENT_API_KEY")

    def fail_if_built(*arguments: object) -> None:
        raise AssertionError("the graph client was built")

    monkeypatch.setattr(cli, "Neo4jGraphClient", fail_if_built)

    exit_code = cli.main(["ask", QUESTION])

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_FAILURE
    assert "MUON_AGENT_API_KEY" in captured.err
    assert captured.out == ""


def test_unknown_universe_exits_1_before_any_connection(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = RecordingAskRun(build_result())
    monkeypatch.setattr(cli, "run_ask", run)

    exit_code = cli.main(["ask", QUESTION, "--universe", "zelda"])

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_FAILURE
    assert "'zelda'" in captured.err
    assert "pkmn" in captured.err
    assert run.requests == []


@pytest.mark.parametrize(
    "error",
    [
        GraphConnectionError("Neo4j is down"),
        ModelCallError("retrieve", "m", "rate limited"),
        NameIndexError("The full-text index 'entity_names' does not exist"),
    ],
)
def test_run_failures_exit_1_with_the_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: Exception
) -> None:
    run_ask_command(monkeypatch, error)

    captured = capsys.readouterr()
    assert str(error) in captured.err
    assert captured.out == ""


def test_the_runtime_builds_each_chat_model_with_its_step_model_and_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple[str, str | None]] = []

    def record_chat_model(settings: AgentSettings, model: str, reasoning: str) -> object:
        created.append((model, reasoning))
        return object()

    monkeypatch.setattr(routing, "create_chat_model", record_chat_model)
    agent_settings = AgentSettings(
        _env_file=None,
        base_url="https://a.test/v1",
        api_key=SECRET,
        default_model="base",
        generate_model="writer",
        resolve_reasoning="high",
        retrieve_reasoning="low",
        generate_reasoning="medium",
    )
    routing.create_agent_runtime(object(), agent_settings)  # type: ignore[arg-type]

    assert created == [("base", "high"), ("base", "low"), ("writer", "medium")]


def test_the_status_line_shows_model_calls_and_token_detail() -> None:
    steps = [
        StepStats(
            step="retrieve",
            input_tokens=14_000,
            output_tokens=400,
            reasoning_tokens=300,
            cached_input_tokens=12_000,
            model_calls=2,
            latency_seconds=1,
        ),
        StepStats(
            step="generate",
            input_tokens=2_706,
            output_tokens=89,
            reasoning_tokens=102,
            cached_input_tokens=288,
            model_calls=1,
            latency_seconds=1,
        ),
    ]
    base = build_result()
    result = base.model_copy(
        update={
            "stats": base.stats.model_copy(update={"steps": steps, "total_latency_seconds": 34.1})
        }
    )

    assert cli.format_ask_status(result) == (
        "ask pkmn: 1 queries, 3 model calls, 16,706 input (12,288 cached) and "
        "489 output (402 reasoning) tokens, 34.1 s"
    )


def test_json_holds_the_resolved_names(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    candidate = NameCandidate(key=SPECIES, name="Diglett", score=8.2)
    names = [ResolvedName(mention="Diglet", label="", candidates=[candidate])]

    run_ask_command(monkeypatch, build_result(resolved_names=names), "--json")

    document = json.loads(capsys.readouterr().out)
    assert document["resolved_names"][0]["mention"] == "Diglet"
    assert document["resolved_names"][0]["candidates"][0]["key"]["id"] == SPECIES.id
