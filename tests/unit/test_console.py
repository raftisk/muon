import io

import pytest
from rich.console import Console
from rich.spinner import Spinner

from muon.console import (
    DONE_MARK,
    FAILED_MARK,
    SPINNER_NAME,
    EvalDisplay,
    StepDisplay,
    StepLine,
    build_eval_summary,
    build_results_table,
    render_steps,
)
from muon.evaluation.questions import EvalQuestion, SelectedQuestion
from muon.evaluation.run import GOLD_STEP, EvalRun, RunRecord
from tests.unit.evaluation.fakes import build_run_meta, build_run_record

WIDTH = 100


def build_console() -> Console:
    return Console(file=io.StringIO(), width=WIDTH, markup=False, highlight=False, emoji=False)


def render_to_text(*renderables: object) -> str:
    console = build_console()
    for renderable in renderables:
        console.print(renderable)
    assert isinstance(console.file, io.StringIO)
    return console.file.getvalue()


def build_question(question_id: str) -> EvalQuestion:
    return EvalQuestion(
        id=question_id,
        family="S1",
        question=f"Question {question_id}?",
        reference="Answer.",
        gold_query="MATCH (n) RETURN n",
    )


def build_selected(question_id: str) -> SelectedQuestion:
    return SelectedQuestion("general", build_question(question_id))


def build_scored_record(question_id: str, tokens: int) -> RunRecord:
    record = build_run_record(build_question(question_id))
    assert record.score is not None
    score = record.score.model_copy(
        update={
            "precision": 0.5,
            "recall": 1.0,
            "input_tokens": tokens,
            "output_tokens": 0,
            "cached_input_tokens": 7,
            "reasoning_tokens": 3,
            "model_calls": 2,
        }
    )
    return record.model_copy(update={"score": score})


def test_render_steps_marks_finished_and_failed_lines() -> None:
    lines = [
        StepLine("reading schema", 0.0, 1.5, False),
        StepLine("querying graph", 1.5, 2.0, True),
    ]

    text = render_to_text(render_steps(lines, Spinner(SPINNER_NAME), now=3.0))

    assert f"{DONE_MARK} reading schema" in text
    assert "1.5 s" in text
    assert f"{FAILED_MARK} querying graph" in text


def test_step_reraises_the_same_error_and_marks_the_line_failed() -> None:
    display = StepDisplay(build_console())
    error = ValueError("boom")

    with pytest.raises(ValueError) as raised, display.step("querying graph"):
        raise error

    assert raised.value is error
    assert [line.has_failed for line in display.snapshot()] == [True]


def test_a_step_that_ends_normally_is_finished() -> None:
    display = StepDisplay(build_console())

    with display.step("reading schema"):
        pass

    (line,) = display.snapshot()
    assert line.finished_at is not None
    assert not line.has_failed


def test_step_display_writes_nothing_on_a_console_that_is_not_a_terminal() -> None:
    console = build_console()

    with StepDisplay(console) as display, display.step("reading schema"):
        pass

    assert isinstance(console.file, io.StringIO)
    assert console.file.getvalue() == ""


def test_handle_result_adds_a_row_and_clears_the_step_lines() -> None:
    questions = [build_selected("Q01"), build_selected("Q02")]
    display = EvalDisplay(build_console(), questions)
    with display.step("reading schema"):
        pass

    display.handle_result(build_scored_record("Q01", 10))

    assert display.snapshot() == ()
    text = render_to_text(display.render())
    assert "Q01" in text
    assert "reading schema" not in text


def test_eval_display_shows_the_header_of_the_current_question() -> None:
    questions = [build_selected("Q01"), build_selected("Q02")]
    display = EvalDisplay(build_console(), questions)
    display.handle_result(build_scored_record("Q01", 10))
    with display.step("querying graph"):
        pass

    assert "Q02 (2/2)" in render_to_text(display.render())


def test_displays_build_on_a_terminal() -> None:
    console = Console(file=io.StringIO(), width=WIDTH, force_terminal=True)

    with EvalDisplay(console, [build_selected("Q01")]) as display, display.step("reading schema"):
        pass
    with StepDisplay(console) as step_display, step_display.step("reading schema"):
        pass

    assert display.live is not None
    assert "reading schema" in console.file.getvalue()  # type: ignore[attr-defined]


def test_the_gold_step_shows_without_a_header() -> None:
    display = EvalDisplay(build_console(), [build_selected("Q01")])
    with display.step(GOLD_STEP):
        pass

    text = render_to_text(display.render())

    assert GOLD_STEP in text
    assert "Q01 (1/1)" not in text


def test_results_table_has_the_count_columns() -> None:
    text = render_to_text(build_results_table([], None))

    header = next(line for line in text.splitlines() if "family" in line)
    names = [cell.strip() for cell in header.strip("│").split("│")]
    assert names == [
        "id", "family", "P", "R", "valid", "tools", "calls", "s", "in", "cached", "out", "reason"
    ]  # fmt: skip


def test_results_table_shows_scores_and_the_error_cells() -> None:
    records = [build_scored_record("Q01", 1234), build_run_record(build_question("Q02"), "boom")]

    text = render_to_text(build_results_table(records, None))

    assert "0.50" in text
    assert "1,234" in text
    error_row = next(line for line in text.splitlines() if "Q02" in line)
    assert "error" in error_row
    assert error_row.count("-") == 9


def test_eval_summary_holds_totals_errors_and_the_stop_reason() -> None:
    records = (
        build_scored_record("Q01", 1000),
        build_scored_record("Q02", 234),
        build_run_record(build_question("Q03"), "rate limited"),
    )
    run = EvalRun(meta=build_run_meta(), records=records, stop="Q02: Neo4j is down")

    text = render_to_text(build_eval_summary(run))

    total_row = next(line for line in text.splitlines() if "total" in line)
    assert "0.50" in total_row
    assert "1.00" in total_row
    assert "2/2" in total_row
    assert "1,234" in total_row
    assert [cell.strip() for cell in total_row.strip("│").split("│")][6:] == [
        "4", "5.0", "1,234", "14", "0", "6"
    ]  # fmt: skip
    assert "errors: 1" in text
    assert "Q03: rate limited" in text
    assert "stopped: Q02: Neo4j is down" in text


def test_eval_summary_adds_a_totals_line_per_set_when_the_run_holds_several() -> None:
    records = (
        build_scored_record("Q01", 10),
        build_run_record(build_question("N01"), None, "name-resolution"),
    )
    run = EvalRun(meta=build_run_meta(), records=records, stop=None)

    lines = render_to_text(build_eval_summary(run)).splitlines()

    assert any(line.startswith("general: 1 questions, precision 0.50") for line in lines)
    assert any(line.startswith("name-resolution: 1 questions") for line in lines)


def test_eval_summary_has_no_set_lines_for_a_single_set() -> None:
    run = EvalRun(meta=build_run_meta(), records=(build_scored_record("Q01", 10),), stop=None)

    assert "general: " not in render_to_text(build_eval_summary(run))
