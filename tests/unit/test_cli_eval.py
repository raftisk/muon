import json
from functools import partial
from pathlib import Path

import pytest
from pydantic import SecretStr

from muon import cli
from muon.config import (
    AgentSettings,
    Neo4jSettings,
    OntologySettings,
    Settings,
    UniverseSettings,
)
from muon.evaluation.run import EvalRequest, EvalRun
from tests.fixtures.toy_ontology import TOY_UNIVERSE, copy_toy_ontology
from tests.fixtures.toy_plugin import write_toy_plugin
from tests.unit.evaluation.fakes import (
    EVALS_FILE_NAME,
    build_question_toml,
    build_run_meta,
    build_run_record,
)

QUESTION_IDS = ("Q01", "Q02", "Q03")


class RecordingEvalSession:
    """Stands in for `cli.run_eval_session`: answers every question, or fails the chosen ones."""

    def __init__(self, failing_ids: tuple[str, ...] = (), stop: str | None = None) -> None:
        self.failing_ids = failing_ids
        self.stop = stop
        self.requests: list[EvalRequest] = []

    async def __call__(self, context: cli.AgentContext, request: EvalRequest) -> EvalRun:
        self.requests.append(request)
        records = []
        for question in request.questions:
            error = "rate limited" if question.id in self.failing_ids else None
            record = build_run_record(question, error)
            records.append(record)
            if request.on_result is not None:
                request.on_result(record)
        return EvalRun(meta=build_run_meta(), records=tuple(records), stop=self.stop)


@pytest.fixture(autouse=True)
def agent_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("BASE_URL", "API_KEY", "MODEL"):
        monkeypatch.setenv(f"MUON_AGENT_{name}", "x")
    monkeypatch.setattr(cli, "AgentSettings", partial(AgentSettings, _env_file=None))


def install_settings(
    monkeypatch: pytest.MonkeyPatch, root: Path, has_evals: bool = True
) -> Settings:
    ontology_dir = copy_toy_ontology(root / "ontology")
    universes_dir = root / "universes"
    universes_dir.mkdir()
    overrides = {"evals": EVALS_FILE_NAME} if has_evals else {}
    folder = write_toy_plugin(universes_dir, **overrides)
    if has_evals:
        text = "".join(build_question_toml(question_id) for question_id in QUESTION_IDS)
        (folder / EVALS_FILE_NAME).write_text(text, encoding="utf-8")
    loaded = Settings(
        neo4j=Neo4jSettings(
            uri="neo4j://x", username="neo4j", password=SecretStr("p"), _env_file=None
        ),
        ontology=OntologySettings(ontology_dir=ontology_dir),
        universes=UniverseSettings(universes_dir=universes_dir, schema_cache_dir=root / "cache"),
    )
    monkeypatch.setattr(cli, "get_settings", lambda: loaded)
    return loaded


RESULTS_TABLE_WIDTH = 120


def run_eval_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    session: RecordingEvalSession,
    *flags: str,
) -> int:
    install_settings(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "run_eval_session", session)
    monkeypatch.setattr(cli.STDOUT, "width", RESULTS_TABLE_WIDTH)
    return cli.main(["eval", TOY_UNIVERSE, "--output-dir", str(tmp_path / "out"), *flags])


def test_eval_parser_accepts_its_options() -> None:
    arguments = cli.build_parser().parse_args(
        ["eval", "pkmn", "--question", "Q01", "--question", "Q07", "--output-dir", "runs"]
    )

    assert arguments.prefix == "pkmn"
    assert arguments.question_ids == ["Q01", "Q07"]
    assert arguments.output_dir == Path("runs")


def test_eval_parser_defaults_to_every_question_and_the_data_folder() -> None:
    arguments = cli.build_parser().parse_args(["eval", "pkmn"])

    assert arguments.question_ids == []
    assert arguments.output_dir.parts[-2:] == ("data", "evals")


def test_a_full_run_prints_rows_and_the_summary_and_writes_the_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    session = RecordingEvalSession()

    exit_code = run_eval_command(monkeypatch, tmp_path, session)

    out = capsys.readouterr().out
    assert exit_code == cli.EXIT_SUCCESS
    assert [question.id for question in session.requests[0].questions] == list(QUESTION_IDS)
    assert out.splitlines()[0] == f"eval {TOY_UNIVERSE}: 3 questions"
    for question_id in QUESTION_IDS:
        assert any(question_id in line and "S1" in line for line in out.splitlines())
    assert any("total" in line for line in out.splitlines())
    assert session.requests[0].on_result is not None
    folder = tmp_path / "out" / "20261116-143005"
    assert f"run folder: {folder}" in out
    assert json.loads((folder / "run.json").read_text(encoding="utf-8"))["totals"]


def test_question_flags_select_only_those_questions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = RecordingEvalSession()

    run_eval_command(monkeypatch, tmp_path, session, "--question", "Q03", "--question", "Q01")

    assert [question.id for question in session.requests[0].questions] == ["Q01", "Q03"]


def test_an_unknown_question_id_exits_1_before_any_connection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    session = RecordingEvalSession()

    exit_code = run_eval_command(monkeypatch, tmp_path, session, "--question", "Q99")

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_FAILURE
    assert "Q99" in captured.err
    assert "Q01, Q02, Q03" in captured.err
    assert session.requests == []
    assert not (tmp_path / "out").exists()


def test_a_plugin_without_evals_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    install_settings(monkeypatch, tmp_path, has_evals=False)
    session = RecordingEvalSession()
    monkeypatch.setattr(cli, "run_eval_session", session)

    exit_code = cli.main(["eval", TOY_UNIVERSE, "--output-dir", str(tmp_path / "out")])

    assert exit_code == cli.EXIT_FAILURE
    assert "has no eval set" in capsys.readouterr().err
    assert session.requests == []


def test_an_unknown_universe_exits_1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    install_settings(monkeypatch, tmp_path)
    session = RecordingEvalSession()
    monkeypatch.setattr(cli, "run_eval_session", session)

    exit_code = cli.main(["eval", "zelda"])

    assert exit_code == cli.EXIT_FAILURE
    assert "'zelda'" in capsys.readouterr().err
    assert session.requests == []


def test_missing_agent_settings_exit_1_before_any_connection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    install_settings(monkeypatch, tmp_path)
    monkeypatch.delenv("MUON_AGENT_API_KEY")
    session = RecordingEvalSession()
    monkeypatch.setattr(cli, "run_eval_session", session)

    exit_code = cli.main(["eval", TOY_UNIVERSE])

    captured = capsys.readouterr()
    assert exit_code == cli.EXIT_FAILURE
    assert "MUON_AGENT_API_KEY" in captured.err
    assert captured.out == ""
    assert session.requests == []


def test_an_error_record_exits_1_after_writing_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    session = RecordingEvalSession(failing_ids=("Q02",))

    exit_code = run_eval_command(monkeypatch, tmp_path, session)

    out = capsys.readouterr().out
    assert exit_code == cli.EXIT_FAILURE
    assert "rate limited" in out
    assert "errors: 1" in out
    assert (tmp_path / "out" / "20261116-143005" / "run.json").is_file()


def test_a_stopped_run_exits_1_and_says_why(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    session = RecordingEvalSession(stop="Q03: Neo4j is down")

    exit_code = run_eval_command(monkeypatch, tmp_path, session)

    assert exit_code == cli.EXIT_FAILURE
    assert "stopped: Q03: Neo4j is down" in capsys.readouterr().out
