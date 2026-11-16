"""Terminal output of `muon ask` and `muon eval`. The only module that imports Rich.

Map: `STDOUT` and `STDERR` are the 2 consoles every command prints through. `StepDisplay`
and `EvalDisplay` show the steps live on a terminal and implement `StepTracker` of
`muon.agent.routing`. `build_results_table` and `build_eval_summary` render a run.
"""

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Self

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from muon.evaluation.questions import EvalQuestion
from muon.evaluation.run import GOLD_STEP, EvalRun, RunRecord, RunTotals, summarize

SPINNER_NAME = "dots"
REFRESH_PER_SECOND = 10
DONE_MARK = Text("✓", style="green")
FAILED_MARK = Text("✗", style="red")
MISSING_CELL = "-"
ERROR_CELL = "error"
TOTAL_LABEL = "total"
HEADER_STYLE = "bold"

STDOUT = Console(soft_wrap=True, markup=False, highlight=False, emoji=False)
STDERR = Console(soft_wrap=True, markup=False, highlight=False, emoji=False, stderr=True)


@dataclass(frozen=True)
class StepLine:
    """One step of a run: running while `finished_at` is `None`."""

    name: str
    started_at: float
    finished_at: float | None
    has_failed: bool


def render_steps(lines: Sequence[StepLine], spinner: Spinner, now: float) -> Table:
    """Render 1 row per step: the mark (the spinner while running), the name and the time."""
    grid = Table.grid(padding=(0, 0, 0, 1))
    grid.add_column()
    grid.add_column()
    grid.add_column(justify="right")
    for line in lines:
        done_mark = FAILED_MARK if line.has_failed else DONE_MARK
        mark = spinner if line.finished_at is None else done_mark
        elapsed = (now if line.finished_at is None else line.finished_at) - line.started_at
        grid.add_row(mark, Text(line.name), Text(f"{elapsed:.1f} s"))
    return grid


class LiveDisplay:
    """The step lines of 1 run in a `Live` area that runs only on a terminal.

    Subclasses set `is_transient` and may override `render`. `Live` calls `render` when it is
    built, so a subclass sets the fields `render` reads before `super().__init__`.
    """

    is_transient = False

    def __init__(self, console: Console) -> None:
        self.entries: list[StepLine] = []
        self.spinner = Spinner(SPINNER_NAME)
        self.live: Live | None = None
        if console.is_terminal:
            self.live = Live(
                console=console,
                get_renderable=self.render,
                transient=self.is_transient,
                refresh_per_second=REFRESH_PER_SECOND,
            )

    def render(self) -> RenderableType:
        return render_steps(self.snapshot(), self.spinner, time.monotonic())

    def snapshot(self) -> tuple[StepLine, ...]:
        """Return a copy that stays valid while the refresh thread renders it."""
        return tuple(self.entries)

    def clear(self) -> None:
        self.entries.clear()

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        """Show `name` as running until the block ends. An exception marks it failed."""
        index = len(self.entries)
        started_at = time.monotonic()
        self.entries.append(StepLine(name, started_at, None, False))
        is_done = False
        try:
            yield
            is_done = True
        finally:
            self.entries[index] = StepLine(name, started_at, time.monotonic(), not is_done)

    def __enter__(self) -> Self:
        if self.live is not None:
            self.live.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self.live is not None:
            self.live.stop()


class StepDisplay(LiveDisplay):
    """The step list of `muon ask`, left on screen when the run ends."""


def build_results_table(records: Sequence[RunRecord], totals: RunTotals | None) -> Table:
    """One row per record, and a total row when `totals` is given."""
    table = Table(box=None, pad_edge=False)
    table.add_column("id")
    table.add_column("family")
    for name in ("P", "R", "valid", "tools", "s", "tokens"):
        table.add_column(name, justify="right")
    for record in records:
        score = record.score
        if score is None:
            missing = [MISSING_CELL] * 5
            table.add_row(record.question_id, record.family, *missing[:2], ERROR_CELL, *missing[2:])
            continue
        tokens = score.input_tokens + score.output_tokens
        table.add_row(
            record.question_id,
            record.family,
            f"{score.precision:.2f}",
            f"{score.recall:.2f}",
            "yes" if score.is_citation_valid else "no",
            str(score.tool_calls),
            f"{score.latency_seconds:.1f}",
            f"{tokens:,}",
        )
    if totals is not None:
        answered = totals.question_count - totals.error_count
        table.add_section()
        table.add_row(
            TOTAL_LABEL,
            "",
            f"{totals.mean_precision:.2f}",
            f"{totals.mean_recall:.2f}",
            f"{totals.valid_count}/{answered}",
            str(totals.tool_calls),
            f"{totals.latency_seconds:.1f}",
            f"{totals.total_tokens:,}",
        )
    return table


def build_eval_summary(run: EvalRun) -> Group:
    """The results table with totals, then the errors and the reason a run stopped early."""
    totals = summarize(run)
    parts: list[RenderableType] = [build_results_table(run.records, totals)]
    if totals.error_count:
        parts.append(Text(f"errors: {totals.error_count}"))
        parts.extend(
            Text(f"{record.question_id}: {record.error}")
            for record in run.records
            if record.score is None
        )
    if run.stop is not None:
        parts.append(Text(f"stopped: {run.stop}"))
    return Group(*parts)


class EvalDisplay(LiveDisplay):
    """The finished rows and the current question of `muon eval`. The area clears at the end."""

    is_transient = True

    def __init__(self, console: Console, questions: Sequence[EvalQuestion]) -> None:
        self.questions = tuple(questions)
        self.records: list[RunRecord] = []
        super().__init__(console)

    def render(self) -> Group:
        lines = self.snapshot()
        parts: list[RenderableType] = [build_results_table(tuple(self.records), None)]
        question_lines = [line for line in lines if line.name != GOLD_STEP]
        position = len(self.records)
        if question_lines and position < len(self.questions):
            header = f"{self.questions[position].id} ({position + 1}/{len(self.questions)})"
            parts.append(Text(header, style=HEADER_STYLE))
        parts.append(render_steps(question_lines or lines, self.spinner, time.monotonic()))
        return Group(*parts)

    def handle_result(self, record: RunRecord) -> None:
        self.records.append(record)
        self.clear()
