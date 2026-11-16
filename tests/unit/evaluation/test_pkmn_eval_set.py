import re

import pytest

from muon.agent.universes import discover_universes
from muon.config import DEFAULT_ONTOLOGY_DIR, DEFAULT_UNIVERSES_DIR
from muon.evaluation.questions import EvalQuestion, load_questions

QUESTION_COUNT = 30
FAMILIES = frozenset({"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S9", "S11", "S18"})
RUNBOOK_PATTERN = re.compile(r"R\d\d")
WRITE_WORD_PATTERN = re.compile(r"\b(SET|CREATE|MERGE|DELETE|REMOVE|DROP|CALL)\b", re.IGNORECASE)
LIMIT_PATTERN = re.compile(r"\bLIMIT\b", re.IGNORECASE)
LIMITED_IDS = frozenset({"Q08"})

QUESTIONS = load_questions(
    discover_universes(DEFAULT_UNIVERSES_DIR, DEFAULT_ONTOLOGY_DIR).get("pkmn")
).questions


def test_the_set_holds_30_questions_in_id_order() -> None:
    expected = [f"Q{number:02d}" for number in range(1, QUESTION_COUNT + 1)]

    assert [question.id for question in QUESTIONS] == expected


@pytest.mark.parametrize("question", QUESTIONS, ids=[question.id for question in QUESTIONS])
class TestEachQuestion:
    def test_it_has_a_family_of_the_v0_set(self, question: EvalQuestion) -> None:
        tokens = [token.strip() for token in question.family.split(",")]

        assert tokens
        assert set(tokens) <= FAMILIES

    def test_its_runbook_entry_has_the_runbook_form(self, question: EvalQuestion) -> None:
        assert question.runbook is None or RUNBOOK_PATTERN.fullmatch(question.runbook)

    def test_its_gold_query_only_reads_and_returns_nodes(self, question: EvalQuestion) -> None:
        assert not WRITE_WORD_PATTERN.search(question.gold_query)
        assert "RETURN" in question.gold_query

    def test_only_a_ranking_limits_its_gold_query(self, question: EvalQuestion) -> None:
        has_limit = LIMIT_PATTERN.search(question.gold_query) is not None

        assert has_limit == (question.id in LIMITED_IDS)
