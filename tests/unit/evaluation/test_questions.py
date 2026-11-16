from pathlib import Path

import pytest

from muon.agent.universes import UniversePlugin, discover_universes
from muon.evaluation.questions import (
    EvalError,
    EvalSet,
    SelectedQuestion,
    load_eval_set,
    load_eval_sets,
    select_questions,
)
from tests.fixtures.toy_ontology import TOY_UNIVERSE, copy_toy_ontology
from tests.fixtures.toy_plugin import write_toy_plugin
from tests.unit.evaluation.fakes import EVAL_SET_NAME, WIDGET_QUERY, build_question_toml

SECOND_SET_NAME = "name-resolution"


def build_plugin(tmp_path: Path, eval_texts: dict[str, str] | None) -> UniversePlugin:
    """A toy plugin whose sets are `eval_texts`: set name to the text of its question file."""
    ontology_dir = copy_toy_ontology(tmp_path / "ontology")
    universes_dir = tmp_path / "universes"
    universes_dir.mkdir()
    evals = {name: f"{name}.toml" for name in eval_texts or {}}
    overrides = {"evals": evals} if evals else {}
    folder = write_toy_plugin(universes_dir, **overrides)
    for name, text in (eval_texts or {}).items():
        (folder / f"{name}.toml").write_text(text, encoding="utf-8")
    return discover_universes(universes_dir, ontology_dir).get(TOY_UNIVERSE)


def build_single(tmp_path: Path, text: str | None) -> UniversePlugin:
    return build_plugin(tmp_path, None if text is None else {EVAL_SET_NAME: text})


def load_single(plugin: UniversePlugin) -> EvalSet:
    return load_eval_sets(plugin, [])[0]


def test_questions_load_in_file_order(tmp_path: Path) -> None:
    text = build_question_toml("Q02") + build_question_toml("Q01", runbook="R05")
    plugin = build_single(tmp_path, text)

    eval_set = load_single(plugin)

    assert eval_set.name == EVAL_SET_NAME
    assert [question.id for question in eval_set.questions] == ["Q02", "Q01"]
    assert eval_set.questions[1].runbook == "R05"
    assert eval_set.questions[0].runbook is None
    assert eval_set.questions[0].gold_query.strip() == WIDGET_QUERY


def test_the_hash_follows_the_file_bytes(tmp_path: Path) -> None:
    first = load_single(build_single(tmp_path / "a", build_question_toml("Q01")))
    same = load_single(build_single(tmp_path / "b", build_question_toml("Q01")))
    changed = load_single(build_single(tmp_path / "c", build_question_toml("Q01", family="S2")))

    assert first.source_hash == same.source_hash
    assert first.source_hash != changed.source_hash


def test_duplicate_ids_in_a_file_raise(tmp_path: Path) -> None:
    plugin = build_single(tmp_path, build_question_toml("Q01") + build_question_toml("Q01"))

    with pytest.raises(EvalError, match="repeats question ids: Q01"):
        load_eval_sets(plugin, [])


def test_an_unknown_field_raises(tmp_path: Path) -> None:
    plugin = build_single(tmp_path, build_question_toml("Q01", gold="x"))

    with pytest.raises(EvalError, match="gold"):
        load_eval_sets(plugin, [])


def test_a_missing_field_raises(tmp_path: Path) -> None:
    text = build_question_toml("Q01").replace('reference = """Answer Q01."""\n', "")
    plugin = build_single(tmp_path, text)

    with pytest.raises(EvalError, match="reference"):
        load_eval_sets(plugin, [])


def test_an_id_with_a_path_character_raises(tmp_path: Path) -> None:
    plugin = build_single(tmp_path, build_question_toml("../Q01"))

    with pytest.raises(EvalError, match="id"):
        load_eval_sets(plugin, [])


def test_an_empty_file_raises(tmp_path: Path) -> None:
    plugin = build_single(tmp_path, "")

    with pytest.raises(EvalError, match="question"):
        load_eval_sets(plugin, [])


def test_invalid_toml_names_the_file(tmp_path: Path) -> None:
    plugin = build_single(tmp_path, "[[question]\nid = ")

    with pytest.raises(EvalError, match=f"Invalid TOML in .*{EVAL_SET_NAME}.toml"):
        load_eval_sets(plugin, [])


def test_an_unreadable_file_raises(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="Cannot read the eval file"):
        load_eval_set(EVAL_SET_NAME, tmp_path / "missing.toml")


def test_a_plugin_without_evals_raises(tmp_path: Path) -> None:
    plugin = build_single(tmp_path, None)

    with pytest.raises(EvalError, match=f"Universe '{TOY_UNIVERSE}' has no eval set"):
        load_eval_sets(plugin, [])


def two_sets(tmp_path: Path, second: str | None = None) -> UniversePlugin:
    return build_plugin(
        tmp_path,
        {
            EVAL_SET_NAME: build_question_toml("Q01") + build_question_toml("Q02"),
            SECOND_SET_NAME: second or build_question_toml("N01"),
        },
    )


def test_no_names_load_every_set_in_manifest_order(tmp_path: Path) -> None:
    sets = load_eval_sets(two_sets(tmp_path), [])

    assert [eval_set.name for eval_set in sets] == [EVAL_SET_NAME, SECOND_SET_NAME]


def test_names_load_only_those_sets(tmp_path: Path) -> None:
    sets = load_eval_sets(two_sets(tmp_path), [SECOND_SET_NAME])

    assert [eval_set.name for eval_set in sets] == [SECOND_SET_NAME]
    assert [question.id for question in sets[0].questions] == ["N01"]


def test_an_unknown_set_lists_the_declared_ones(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match=f"Unknown eval sets: nope; declared: {EVAL_SET_NAME}, "):
        load_eval_sets(two_sets(tmp_path), ["nope"])


def test_a_question_id_shared_by_two_sets_names_both_sets(tmp_path: Path) -> None:
    plugin = two_sets(tmp_path, second=build_question_toml("Q02"))

    with pytest.raises(EvalError, match=f"Question id Q02 is in the sets '{EVAL_SET_NAME}' and"):
        load_eval_sets(plugin, [])


def test_select_without_ids_returns_every_question_with_its_set(tmp_path: Path) -> None:
    sets = load_eval_sets(two_sets(tmp_path), [])

    selected = select_questions(sets, [])

    assert [(item.set_name, item.question.id) for item in selected] == [
        (EVAL_SET_NAME, "Q01"),
        (EVAL_SET_NAME, "Q02"),
        (SECOND_SET_NAME, "N01"),
    ]
    assert all(isinstance(item, SelectedQuestion) for item in selected)


def test_select_keeps_set_and_file_order(tmp_path: Path) -> None:
    sets = load_eval_sets(two_sets(tmp_path), [])

    selected = select_questions(sets, ["N01", "Q02"])

    assert [item.question.id for item in selected] == ["Q02", "N01"]


def test_select_names_the_unknown_and_the_known_ids(tmp_path: Path) -> None:
    sets = load_eval_sets(build_single(tmp_path, build_question_toml("Q01")), [])

    with pytest.raises(EvalError, match="Unknown question ids: Q99; known: Q01"):
        select_questions(sets, ["Q01", "Q99"])
