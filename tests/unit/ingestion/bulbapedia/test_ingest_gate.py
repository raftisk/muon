"""`write_pass` skips the batches the Validator rejects and writes the others, and hands every
batch of a planned run to the planner."""

from enum import StrEnum
from pathlib import Path

import pytest

from muon.ingestion.batch import GraphBatch, NodeRecord
from muon.ingestion.bulbapedia.batching import PKMN, UNIVERSE
from muon.ingestion.bulbapedia.ingest import (
    IngestTarget,
    PassItem,
    PassProgress,
    should_count,
    write_pass,
    write_passes,
)
from muon.ingestion.plan import DiffTotals, ItemState, PlanHeader, PlannedItem
from muon.ingestion.policy import WritePolicy
from muon.ingestion.runner import Runner
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import load_ontology_model
from muon.validation import ValidationReport, Violation
from tests.unit.ingestion.recording_client import RecordingGraphClient
from tests.unit.validation.conftest import ONTOLOGY_DIR

SPECIES = PKMN + "PokemonSpecies"
REJECTED_ID = "bad"
RULE_ID = "S1"


class GatePass(StrEnum):
    SPECIES = "species"
    CHUNK = "chunk"


class RejectingValidator:
    """Rejects every batch whose first node has the id `REJECTED_ID`."""

    def __init__(self) -> None:
        self.validated: list[GraphBatch] = []

    async def validate(self, batch: GraphBatch) -> ValidationReport:
        self.validated.append(batch)
        if batch.nodes[0].id != REJECTED_ID:
            return ValidationReport()
        return ValidationReport(violations=(Violation(rule_id=RULE_ID, message="no number"),))


def build_item(node_id: str) -> PassItem:
    node = NodeRecord(class_iri=SPECIES, id=node_id, name=node_id)
    batch = GraphBatch(universe=UNIVERSE, scope=frozenset({SPECIES}), nodes=(node,), edges=())
    return PassItem(page_title=f"{node_id} (Pokémon)", batch=batch)


def build_target(client: RecordingGraphClient, validator: RejectingValidator) -> IngestTarget:
    model = load_ontology_model(ONTOLOGY_DIR, UNIVERSE)
    return IngestTarget(runner=Runner(client, LabelMaterializer(model)), validator=validator)


async def test_rejected_batch_is_skipped_and_the_pass_continues() -> None:
    client = RecordingGraphClient()
    validator = RejectingValidator()
    progress: list[PassProgress] = []
    items = [build_item("alpha"), build_item(REJECTED_ID), build_item("beta")]

    result = await write_pass(
        build_target(client, validator), items, GatePass.SPECIES, progress.append
    )

    assert len(client.batches) == 2
    assert len(validator.validated) == 3
    assert result.stop is None
    assert result.totals.batches_written == 2
    assert result.totals.batches_rejected == 1
    (rejection,) = result.rejections
    assert rejection.page_title == "bad (Pokémon)"
    assert rejection.ingest_pass == "species"
    assert [violation.rule_id for violation in rejection.violations] == [RULE_ID]
    assert [(entry.done, entry.total) for entry in progress] == [(3, 3)]


class PlannerFake:
    """Stands in for a `PlanWriter`: gives each batch the state `states` maps its id to."""

    def __init__(self, states: dict[str, ItemState]) -> None:
        self.states = states
        self.planned: list[tuple[str, WritePolicy | None]] = []
        self.is_finished = False
        self.plan_dir = Path("plans/run")

    async def plan_item(
        self, source: str, batch: GraphBatch, policy: WritePolicy | None
    ) -> PlannedItem:
        self.planned.append((source, policy))
        state = self.states[batch.nodes[0].id]
        violations = (
            (Violation(rule_id=RULE_ID, message="no number"),)
            if state is ItemState.REJECTED
            else ()
        )
        return PlannedItem(
            number=len(self.planned),
            source=source,
            state=state,
            policy=policy,
            batch=batch,
            violations=violations,
        )

    def finish(self) -> PlanHeader:
        self.is_finished = True
        return PlanHeader(
            universe=UNIVERSE,
            run_id="run",
            run_label="label",
            tbox_hash="t",
            shapes_hash="s",
            counts={},
            totals=DiffTotals(),
        )


STATES = {
    "alpha": ItemState.PLANNED,
    "beta": ItemState.NO_CHANGE,
    REJECTED_ID: ItemState.REJECTED,
}


def build_planned_target(client: RecordingGraphClient, planner: PlannerFake) -> IngestTarget:
    target = build_target(client, RejectingValidator())
    return IngestTarget(runner=target.runner, validator=target.validator, planner=planner)


async def test_planned_run_gives_every_batch_to_the_planner_and_writes_nothing() -> None:
    client, planner = RecordingGraphClient(), PlannerFake(STATES)
    items = [build_item("alpha"), build_item("beta"), build_item(REJECTED_ID)]

    result = await write_pass(
        build_planned_target(client, planner), items, GatePass.SPECIES, lambda _: None
    )

    assert planner.planned == [(item.page_title, None) for item in items]
    assert client.calls == []
    assert result.totals.batches_written == 2
    assert result.totals.batches_rejected == 1
    assert result.totals.counters == {}
    assert [violation.rule_id for violation in result.rejections[0].violations] == [RULE_ID]


@pytest.mark.parametrize(
    ("first_id", "last_id", "is_flagged"),
    [("alpha", "beta", True), ("beta", "alpha", False)],
    ids=["changes-before-a-later-pass", "changes-in-the-last-pass"],
)
async def test_planned_run_finishes_the_plan_and_flags_unapplied_earlier_passes(
    first_id: str, last_id: str, is_flagged: bool
) -> None:
    planner = PlannerFake(STATES)
    passes = [
        (GatePass.SPECIES, [build_item(first_id)]),
        (GatePass.CHUNK, [build_item(last_id)]),
    ]

    pass_run = await write_passes(
        build_planned_target(RecordingGraphClient(), planner), passes, lambda _: None
    )

    assert planner.is_finished
    assert pass_run.plan is not None
    assert pass_run.plan.plan_dir == planner.plan_dir
    assert pass_run.plan.has_unapplied_earlier_pass is is_flagged
    assert not should_count(pass_run)


async def test_stopped_planned_run_leaves_the_plan_unfinished() -> None:
    planner = PlannerFake({})

    async def fail(*_: object) -> PlannedItem:
        raise ValueError("boom")

    planner.plan_item = fail  # type: ignore[method-assign]
    passes = [(GatePass.SPECIES, [build_item("alpha")])]

    pass_run = await write_passes(
        build_planned_target(RecordingGraphClient(), planner), passes, lambda _: None
    )

    assert pass_run.stop is not None
    assert pass_run.plan is None
    assert not planner.is_finished
    assert not should_count(pass_run)
