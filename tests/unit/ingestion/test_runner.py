from pathlib import Path

import pytest

from muon.graph import WriteSummary
from muon.ingestion import runner
from muon.ingestion.batch import EdgeRecord, GraphBatch, NodeRecord
from muon.ingestion.runner import (
    Runner,
    WriteMode,
    build_edge_rows,
    build_ids_by_label,
    build_node_rows,
    build_statements,
)
from muon.ingestion.seed import read_abox
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import load_ontology_model
from tests.fixtures.toy_ontology import TOY_UNIVERSE, copy_toy_ontology
from tests.unit.ingestion.recording_client import RecordingGraphClient

TOY_SUMMARY = WriteSummary(
    nodes_created=1,
    nodes_deleted=2,
    relationships_created=3,
    relationships_deleted=4,
    labels_added=5,
    labels_removed=6,
    properties_set=7,
)
SYNC_STATEMENT_COUNT = 4
TOY = "https://muon.dev/ns/muontoy#"
TOY_GADGET = TOY + "ToyGadget"
TOY_COLOR = TOY + "ToyColor"
TOY_HAS_COLOR = TOY + "hasColor"


@pytest.fixture(scope="module")
def toy_ontology_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return copy_toy_ontology(tmp_path_factory.mktemp("ontology"))


@pytest.fixture(scope="module")
def materializer(toy_ontology_dir: Path) -> LabelMaterializer:
    return LabelMaterializer(load_ontology_model(toy_ontology_dir, TOY_UNIVERSE))


@pytest.fixture(scope="module")
def toy_batch(materializer: LabelMaterializer, toy_ontology_dir: Path) -> GraphBatch:
    return read_abox(materializer.model, toy_ontology_dir)


def test_sync_statements_in_order(toy_batch: GraphBatch, materializer: LabelMaterializer) -> None:
    statements = build_statements(toy_batch, WriteMode.SYNC, materializer)

    assert [statement.query for statement in statements] == [
        runner.PRUNE_NODES,
        runner.PRUNE_EDGES,
        runner.MERGE_NODES_SYNC,
        runner.MERGE_EDGES_SYNC,
    ]
    assert all(
        statement.query is constant
        for statement, constant in zip(
            statements,
            (
                runner.PRUNE_NODES,
                runner.PRUNE_EDGES,
                runner.MERGE_NODES_SYNC,
                runner.MERGE_EDGES_SYNC,
            ),
            strict=True,
        )
    )


def test_upsert_statements_without_stubs(
    toy_batch: GraphBatch, materializer: LabelMaterializer
) -> None:
    statements = build_statements(toy_batch, WriteMode.UPSERT, materializer)

    assert [statement.query for statement in statements] == [
        runner.MERGE_NODES_UPSERT,
        runner.MERGE_EDGES_UPSERT,
    ]


def test_statements_get_only_their_parameters(
    toy_batch: GraphBatch, materializer: LabelMaterializer
) -> None:
    parameter_names = [
        set(statement.parameters)
        for statement in build_statements(toy_batch, WriteMode.SYNC, materializer)
    ]

    assert parameter_names == [
        {"universe", "scope_labels", "ids_by_label"},
        {"universe", "scope_labels", "edges"},
        {"nodes", "class_labels"},
        {"universe", "edges"},
    ]


def test_node_row(toy_batch: GraphBatch, materializer: LabelMaterializer) -> None:
    rows = {row["id"]: row for row in build_node_rows(toy_batch.nodes, materializer)}
    alpha = rows["alpha"]

    assert alpha["key_labels"] == ["ToyGadget", TOY_UNIVERSE]
    assert alpha["labels"] == ["ToyGadget", "Entity", "Object", TOY_UNIVERSE]
    assert alpha["properties"] == {
        "id": "alpha",
        "name": "Alpha",
        "weight": 1.5,
        "serial": 7,
        "isBroken": False,
        "nickname": ["a-one", "a-prime"],
    }


def test_edge_row(toy_batch: GraphBatch, materializer: LabelMaterializer) -> None:
    rows = build_edge_rows(toy_batch, materializer)

    assert {
        "type": "PAIRS_WITH",
        "start_label": "ToyGadget",
        "start_id": "alpha",
        "end_label": "ToyGadget",
        "end_id": "beta-unit",
        "properties": {},
    } in rows


def test_ids_by_label_covers_every_scope_label(materializer: LabelMaterializer) -> None:
    scope = frozenset(
        {"https://muon.dev/ns/muontoy#ToyGadget", "https://muon.dev/ns/muontoy#ToyColor"}
    )
    empty_batch = GraphBatch(universe=TOY_UNIVERSE, scope=scope, nodes=(), edges=())

    assert build_ids_by_label(empty_batch, materializer) == {"ToyGadget": [], "ToyColor": []}


def test_class_labels_cover_the_ontology(
    toy_batch: GraphBatch, materializer: LabelMaterializer
) -> None:
    merge_nodes = build_statements(toy_batch, WriteMode.SYNC, materializer)[2]

    assert set(merge_nodes.parameters["class_labels"]) == materializer.list_ontology_labels()


async def test_sync_write_ensures_constraints_first(
    toy_batch: GraphBatch, materializer: LabelMaterializer
) -> None:
    client = RecordingGraphClient(TOY_SUMMARY)

    report = await Runner(client, materializer).write(toy_batch, WriteMode.SYNC)

    assert [type(call) for call in client.calls] == [str, str, list]
    assert len(client.batches[0]) == SYNC_STATEMENT_COUNT
    assert report.constraints_ensured == 2
    assert report.nodes_created == TOY_SUMMARY.nodes_created * SYNC_STATEMENT_COUNT
    assert report.labels_removed == TOY_SUMMARY.labels_removed * SYNC_STATEMENT_COUNT
    assert report.properties_set == TOY_SUMMARY.properties_set * SYNC_STATEMENT_COUNT


async def test_other_universe_batch_raises(materializer: LabelMaterializer) -> None:
    client = RecordingGraphClient()
    other_batch = GraphBatch(universe="other", scope=frozenset(), nodes=(), edges=())

    with pytest.raises(ValueError, match="'other' does not match"):
        await Runner(client, materializer).write(other_batch, WriteMode.SYNC)

    assert client.calls == []


async def test_purge_sends_the_universe_label(materializer: LabelMaterializer) -> None:
    client = RecordingGraphClient(TOY_SUMMARY)

    summary = await Runner(client, materializer).purge()

    assert client.writes == [runner.PURGE_UNIVERSE]
    assert client.batches == []
    assert client.write_parameters == [{"universe": TOY_UNIVERSE}]
    assert summary == TOY_SUMMARY


def test_purge_statement_takes_the_label_as_parameter_and_keeps_the_schema() -> None:
    assert "$universe" in runner.PURGE_UNIVERSE
    assert "DROP" not in runner.PURGE_UNIVERSE
    assert "CONSTRAINT" not in runner.PURGE_UNIVERSE


def build_stub_batch() -> GraphBatch:
    """Return a batch that owns the gadget `alpha` and refers to the color `red` as a stub."""
    alpha = NodeRecord(
        class_iri=TOY_GADGET, id="alpha", name="Alpha", properties={TOY + "serial": 7}
    )
    red = NodeRecord(class_iri=TOY_COLOR, id="red", name="Red")
    has_color = EdgeRecord(
        property_iri=TOY_HAS_COLOR,
        start=alpha.node_ref,
        end=red.node_ref,
        properties={"shade": "dark", "tags": ("a", "b")},
    )
    return GraphBatch(
        universe=TOY_UNIVERSE,
        scope=frozenset({TOY_GADGET}),
        nodes=(alpha, red),
        edges=(has_color,),
    )


def test_upsert_statements_with_stubs(materializer: LabelMaterializer) -> None:
    statements = build_statements(build_stub_batch(), WriteMode.UPSERT, materializer)
    merge_nodes, merge_stubs, _ = statements

    assert [statement.query for statement in statements] == [
        runner.MERGE_NODES_UPSERT,
        runner.MERGE_STUBS,
        runner.MERGE_EDGES_UPSERT,
    ]
    assert set(merge_stubs.parameters) == {"stubs"}
    assert merge_stubs.parameters["stubs"] == [
        {
            "key_labels": ["ToyColor", TOY_UNIVERSE],
            "labels": ["ToyColor", "Concept", "Entity", TOY_UNIVERSE],
            "id": "red",
            "properties": {"id": "red", "name": "Red"},
        }
    ]
    assert [row["id"] for row in merge_nodes.parameters["nodes"]] == ["alpha"]


def test_stub_statement_sets_only_on_create() -> None:
    set_lines = [line for line in runner.MERGE_STUBS.splitlines() if "SET" in line]

    assert "REMOVE" not in runner.MERGE_STUBS
    assert set_lines
    assert all(line.startswith("ON CREATE SET") for line in set_lines)


def test_edge_statements_set_properties_per_mode() -> None:
    assert runner.MERGE_EDGES_UPSERT.endswith("SET r += row.properties\n")
    assert runner.MERGE_EDGES_SYNC.endswith("SET r = row.properties\n")


def test_edge_row_carries_properties(materializer: LabelMaterializer) -> None:
    rows = build_edge_rows(build_stub_batch(), materializer)

    assert rows[0]["properties"] == {"shade": "dark", "tags": ["a", "b"]}


async def test_sync_write_with_stub_raises(materializer: LabelMaterializer) -> None:
    client = RecordingGraphClient()

    with pytest.raises(ValueError, match="Sync mode takes no stubs: .*ToyColor 'red'"):
        await Runner(client, materializer).write(build_stub_batch(), WriteMode.SYNC)

    assert client.calls == []


async def test_upsert_write_ensures_stub_class_constraints(
    materializer: LabelMaterializer,
) -> None:
    client = RecordingGraphClient()

    report = await Runner(client, materializer).write(build_stub_batch(), WriteMode.UPSERT)

    assert report.constraints_ensured == 2
    assert any("id_unique_ToyColor" in write for write in client.writes)
