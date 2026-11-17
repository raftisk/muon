"""`write_pass` skips the batches the Validator rejects and writes the others."""

from enum import StrEnum

from muon.ingestion.batch import GraphBatch, NodeRecord
from muon.ingestion.bulbapedia.batching import PKMN, UNIVERSE
from muon.ingestion.bulbapedia.ingest import IngestTarget, PassItem, PassProgress, write_pass
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
