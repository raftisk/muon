from pathlib import Path

import pytest

from muon.agent.universes import UniversePlugin, discover_universes
from muon.evaluation.questions import EvalError, load_questions
from tests.fixtures.toy_ontology import TOY_UNIVERSE, copy_toy_ontology
from tests.fixtures.toy_plugin import write_toy_plugin
from tests.unit.evaluation.fakes import EVALS_FILE_NAME, WIDGET_QUERY, build_question_toml


def build_plugin(tmp_path: Path, eval_text: str | None) -> UniversePlugin:
    ontology_dir = copy_toy_ontology(tmp_path / "ontology")
    universes_dir = tmp_path / "universes"
    universes_dir.mkdir()
    overrides = {} if eval_text is None else {"evals": EVALS_FILE_NAME}
    folder = write_toy_plugin(universes_dir, **overrides)
    if eval_text is not None:
        (folder / EVALS_FILE_NAME).write_text(eval_text, encoding="utf-8")
    return discover_universes(universes_dir, ontology_dir).get(TOY_UNIVERSE)


def test_questions_load_in_file_order(tmp_path: Path) -> None:
    text = build_question_toml("Q02") + build_question_toml("Q01", runbook="R05")
    plugin = build_plugin(tmp_path, text)

    eval_set = load_questions(plugin)

    assert [question.id for question in eval_set.questions] == ["Q02", "Q01"]
    assert eval_set.questions[1].runbook == "R05"
    assert eval_set.questions[0].runbook is None
    assert eval_set.questions[0].gold_query.strip() == WIDGET_QUERY


def test_the_hash_follows_the_file_bytes(tmp_path: Path) -> None:
    first = load_questions(build_plugin(tmp_path / "a", build_question_toml("Q01")))
    same = load_questions(build_plugin(tmp_path / "b", build_question_toml("Q01")))
    changed = load_questions(build_plugin(tmp_path / "c", build_question_toml("Q01", family="S2")))

    assert first.source_hash == same.source_hash
    assert first.source_hash != changed.source_hash


def test_duplicate_ids_raise(tmp_path: Path) -> None:
    plugin = build_plugin(tmp_path, build_question_toml("Q01") + build_question_toml("Q01"))

    with pytest.raises(EvalError, match="repeats question ids: Q01"):
        load_questions(plugin)


def test_an_unknown_field_raises(tmp_path: Path) -> None:
    plugin = build_plugin(tmp_path, build_question_toml("Q01", gold="x"))

    with pytest.raises(EvalError, match="gold"):
        load_questions(plugin)


def test_a_missing_field_raises(tmp_path: Path) -> None:
    text = build_question_toml("Q01").replace('reference = """Answer Q01."""\n', "")
    plugin = build_plugin(tmp_path, text)

    with pytest.raises(EvalError, match="reference"):
        load_questions(plugin)


def test_an_id_with_a_path_character_raises(tmp_path: Path) -> None:
    plugin = build_plugin(tmp_path, build_question_toml("../Q01"))

    with pytest.raises(EvalError, match="id"):
        load_questions(plugin)


def test_an_empty_file_raises(tmp_path: Path) -> None:
    plugin = build_plugin(tmp_path, "")

    with pytest.raises(EvalError, match="question"):
        load_questions(plugin)


def test_invalid_toml_names_the_file(tmp_path: Path) -> None:
    plugin = build_plugin(tmp_path, "[[question]\nid = ")

    with pytest.raises(EvalError, match=f"Invalid TOML in .*{EVALS_FILE_NAME}"):
        load_questions(plugin)


def test_a_plugin_without_evals_raises(tmp_path: Path) -> None:
    plugin = build_plugin(tmp_path, None)

    with pytest.raises(EvalError, match=f"Universe '{TOY_UNIVERSE}' has no eval set"):
        load_questions(plugin)


def test_select_keeps_file_order(tmp_path: Path) -> None:
    text = "".join(build_question_toml(f"Q0{number}") for number in (1, 2, 3))
    eval_set = load_questions(build_plugin(tmp_path, text))

    selected = eval_set.select(["Q03", "Q01"])

    assert [question.id for question in selected] == ["Q01", "Q03"]


def test_select_names_the_unknown_and_the_known_ids(tmp_path: Path) -> None:
    eval_set = load_questions(build_plugin(tmp_path, build_question_toml("Q01")))

    with pytest.raises(EvalError, match="Unknown question ids: Q99; known: Q01"):
        eval_set.select(["Q01", "Q99"])
