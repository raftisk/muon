import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, LiteralString

from muon.ingestion.batch import GraphBatch, NodeRef
from muon.ingestion.seed import FIND_MISSING_NODES
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import OntologyModel
from muon.validation.report import EdgeTypingFinding, LabelSetFinding, SweepReport
from muon.validation.sweep import (
    FIND_EDGE_TYPING_FAULTS,
    LIST_LABEL_SETS,
    LIST_ORPHAN_STUBS,
    LIST_STUBS,
    find_edge_typing_findings,
    find_label_set_findings,
    find_stub_findings,
    run_sweep,
    write_sweep_report,
)
from tests.unit.ingestion.recording_client import RecordingGraphClient
from tests.unit.validation.conftest import PKMN

NO_FAULTS = {"start_count": 0, "start_ids": [], "end_count": 0, "end_ids": []}
TYPE_LABELS = ["Type", "Concept", "Entity", "pkmn"]
LOCATION_LABELS = ["Location", "Entity", "pkmn"]


class SweepGraphClient(RecordingGraphClient):
    """Answers each sweep query from `answers`, keyed by the query constant."""

    def __init__(self, answers: Mapping[LiteralString, list[dict[str, Any]]]) -> None:
        super().__init__()
        self.answers = answers

    async def execute_read(
        self,
        query: LiteralString,
        parameters: Mapping[str, Any] | None = None,
        max_records: int | None = None,
    ) -> list[dict[str, Any]]:
        await super().execute_read(query, parameters, max_records)
        default = [NO_FAULTS] if query is FIND_EDGE_TYPING_FAULTS else []
        return self.answers.get(query, default)


def build_materializer(model: OntologyModel) -> LabelMaterializer:
    return LabelMaterializer(model)


async def test_label_set_that_matches_a_class_is_not_a_finding(pkmn_model: OntologyModel) -> None:
    client = SweepGraphClient(
        {LIST_LABEL_SETS: [{"labels": TYPE_LABELS, "count": 19, "sample_ids": ["fire"]}]}
    )

    findings = await find_label_set_findings(client, build_materializer(pkmn_model), pkmn_model)

    assert findings == ()


async def test_unmatched_label_sets_merge_whatever_the_label_order(
    pkmn_model: OntologyModel,
) -> None:
    client = SweepGraphClient(
        {
            LIST_LABEL_SETS: [
                {"labels": ["Move", "PokemonSpecies", "pkmn"], "count": 2, "sample_ids": ["b"]},
                {"labels": ["pkmn", "PokemonSpecies", "Move"], "count": 1, "sample_ids": ["a"]},
            ]
        }
    )

    findings = await find_label_set_findings(client, build_materializer(pkmn_model), pkmn_model)

    assert findings == (
        LabelSetFinding(labels=("Move", "PokemonSpecies"), count=3, sample_ids=("a", "b")),
    )


async def test_edge_typing_fault_names_the_type_and_the_end(pkmn_model: OntologyModel) -> None:
    class FaultyClient(SweepGraphClient):
        async def execute_read(
            self,
            query: LiteralString,
            parameters: Mapping[str, Any] | None = None,
            max_records: int | None = None,
        ) -> list[dict[str, Any]]:
            await RecordingGraphClient.execute_read(self, query, parameters, max_records)
            assert parameters is not None
            if parameters["type"] == "HAS_ABILITY":
                return [{**NO_FAULTS, "end_count": 3, "end_ids": ["fire"]}]
            return [NO_FAULTS]

    client = FaultyClient({})

    findings = await find_edge_typing_findings(client, build_materializer(pkmn_model), pkmn_model)

    assert findings == (
        EdgeTypingFinding(
            relationship_type="HAS_ABILITY", end="end", count=3, sample_ids=("fire",)
        ),
    )
    has_ability = next(
        parameters
        for query, parameters in client.reads
        if query is FIND_EDGE_TYPING_FAULTS and parameters and parameters["type"] == "HAS_ABILITY"
    )
    assert has_ability["domain_labels"] == ["PokemonSpecies"]
    assert has_ability["range_labels"] == ["Ability"]


async def test_stubs_leave_out_the_abox_and_orphans_list_node_keys(
    pkmn_model: OntologyModel, pkmn_abox: GraphBatch
) -> None:
    client = SweepGraphClient(
        {
            LIST_STUBS: [
                {"labels": TYPE_LABELS, "id": "fire"},
                {"labels": LOCATION_LABELS, "id": "digletts-cave"},
            ],
            LIST_ORPHAN_STUBS: [
                {"labels": TYPE_LABELS, "id": "fire"},
                {"labels": LOCATION_LABELS, "id": "lonely-road"},
            ],
        }
    )
    materializer = build_materializer(pkmn_model)
    abox_keys = frozenset(
        (materializer.map_class_label(node.class_iri), node.id) for node in pkmn_abox.nodes
    )

    stubs, orphans = await find_stub_findings(client, materializer, pkmn_model, abox_keys)

    assert stubs == {"Location": ("digletts-cave",)}
    assert orphans == (NodeRef(class_iri="https://muon.dev/ns/core#Location", id="lonely-road"),)


async def test_missing_abox_node_makes_the_sweep_fail(
    pkmn_model: OntologyModel, pkmn_abox: GraphBatch
) -> None:
    ghost = pkmn_abox.nodes[0]
    client = SweepGraphClient(
        {FIND_MISSING_NODES: [{"class_iri": ghost.class_iri, "id": ghost.id}]}
    )

    report = await run_sweep(client, build_materializer(pkmn_model), pkmn_model, pkmn_abox)

    assert report.missing_abox == (ghost.node_ref,)
    assert report.is_failing


def test_stubs_and_orphans_alone_do_not_fail_the_sweep() -> None:
    report = SweepReport(
        stubs={"Location": ("a",)}, orphan_stubs=(NodeRef(class_iri=PKMN + "Item", id="b"),)
    )

    assert not report.is_failing


def test_sweep_report_is_written_as_json_under_the_root(tmp_path: Path) -> None:
    report = SweepReport(stubs={"Location": ("a",)})

    path = write_sweep_report(report, "pkmn", tmp_path)

    assert path == tmp_path / "pkmn-sweep-report.json"
    assert json.loads(path.read_text())["stubs"] == {"Location": ["a"]}
