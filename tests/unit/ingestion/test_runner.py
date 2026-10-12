from pathlib import Path

import pytest

from muon.graph import WriteSummary
from muon.ingestion import runner
from muon.ingestion.batch import ClaimDirection, EdgeClaim, EdgeRecord, GraphBatch, NodeRecord
from muon.ingestion.runner import (
    Runner,
    WriteMode,
    build_claim_rows,
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
TOY_WIDGET = TOY + "ToyWidget"
TOY_COLOR = TOY + "ToyColor"
TOY_HAS_COLOR = TOY + "hasColor"
TOY_PAIRS_WITH = TOY + "pairsWith"


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


def build_claim_batch(
    *claims: EdgeClaim, scope: frozenset[str] = frozenset({TOY_WIDGET})
) -> GraphBatch:
    """Return a batch of gadget `alpha`, widget `w1`, gadget `beta` and 2 colors.

    The default scope owns the widget only, so the rest are stubs.

    Edges: alpha has red and blue, alpha pairs with w1 and with beta, w1 has red.
    """
    alpha = NodeRecord(class_iri=TOY_GADGET, id="alpha", name="Alpha")
    widget = NodeRecord(class_iri=TOY_WIDGET, id="w1", name="W1")
    beta = NodeRecord(class_iri=TOY_GADGET, id="beta", name="Beta")
    red = NodeRecord(class_iri=TOY_COLOR, id="red", name="Red")
    blue = NodeRecord(class_iri=TOY_COLOR, id="blue", name="Blue")
    edges = (
        EdgeRecord(property_iri=TOY_HAS_COLOR, start=alpha.node_ref, end=red.node_ref),
        EdgeRecord(property_iri=TOY_HAS_COLOR, start=alpha.node_ref, end=blue.node_ref),
        EdgeRecord(property_iri=TOY_PAIRS_WITH, start=alpha.node_ref, end=widget.node_ref),
        EdgeRecord(property_iri=TOY_PAIRS_WITH, start=alpha.node_ref, end=beta.node_ref),
        EdgeRecord(property_iri=TOY_HAS_COLOR, start=widget.node_ref, end=red.node_ref),
    )
    return GraphBatch(
        universe=TOY_UNIVERSE,
        scope=scope,
        nodes=(alpha, widget, beta, red, blue),
        edges=edges,
        claims=claims,
    )


ALPHA = NodeRecord(class_iri=TOY_GADGET, id="alpha", name="Alpha").node_ref
COLOR_CLAIM = EdgeClaim(
    anchor=ALPHA,
    property_iri=TOY_HAS_COLOR,
    direction=ClaimDirection.OUT,
    other_class_iri=TOY_COLOR,
)
WIDGET_CLAIM = EdgeClaim(
    anchor=ALPHA,
    property_iri=TOY_PAIRS_WITH,
    direction=ClaimDirection.IN,
    other_class_iri=TOY_WIDGET,
    owns_other_end=True,
)


def test_prune_statements_come_before_merges(materializer: LabelMaterializer) -> None:
    batch = build_claim_batch(WIDGET_CLAIM, COLOR_CLAIM)

    statements = build_statements(batch, WriteMode.UPSERT, materializer)

    assert [statement.query for statement in statements] == [
        runner.PRUNE_CLAIMED_OUT,
        runner.PRUNE_CLAIMED_IN,
        runner.MERGE_NODES_UPSERT,
        runner.MERGE_STUBS,
        runner.MERGE_EDGES_UPSERT,
    ]


def test_prune_statements_get_only_their_parameters(materializer: LabelMaterializer) -> None:
    batch = build_claim_batch(COLOR_CLAIM)

    prune = build_statements(batch, WriteMode.UPSERT, materializer)[0]

    assert set(prune.parameters) == {"universe", "claims", "owner_ids_by_label"}
    assert prune.parameters["universe"] == TOY_UNIVERSE
    assert prune.parameters["owner_ids_by_label"] == {"ToyWidget": ["w1"]}


def test_direction_without_claims_sends_no_statement(materializer: LabelMaterializer) -> None:
    only_in = build_statements(build_claim_batch(WIDGET_CLAIM), WriteMode.UPSERT, materializer)
    only_out = build_statements(build_claim_batch(COLOR_CLAIM), WriteMode.UPSERT, materializer)

    assert only_in[0].query is runner.PRUNE_CLAIMED_IN
    assert runner.PRUNE_CLAIMED_OUT not in [statement.query for statement in only_in]
    assert only_out[0].query is runner.PRUNE_CLAIMED_OUT
    assert runner.PRUNE_CLAIMED_IN not in [statement.query for statement in only_out]


def test_batch_without_claims_compiles_as_before(materializer: LabelMaterializer) -> None:
    statements = build_statements(build_claim_batch(), WriteMode.UPSERT, materializer)

    assert [statement.query for statement in statements] == [
        runner.MERGE_NODES_UPSERT,
        runner.MERGE_STUBS,
        runner.MERGE_EDGES_UPSERT,
    ]


def test_claim_row_keeps_the_other_ends_of_matching_edges(materializer: LabelMaterializer) -> None:
    batch = build_claim_batch(COLOR_CLAIM)

    rows = build_claim_rows(batch, materializer, ClaimDirection.OUT)

    assert rows == [
        {
            "anchor_label": "ToyGadget",
            "anchor_id": "alpha",
            "type": "HAS_COLOR",
            "other_label": "ToyColor",
            "owns_other_end": False,
            "kept": [{"label": "ToyColor", "id": "red"}, {"label": "ToyColor", "id": "blue"}],
        }
    ]


def test_claim_row_for_any_class_has_null_other_label(materializer: LabelMaterializer) -> None:
    claim = COLOR_CLAIM.model_copy(update={"other_class_iri": None})

    rows = build_claim_rows(build_claim_batch(claim), materializer, ClaimDirection.OUT)

    assert rows[0]["other_label"] is None
    assert rows[0]["kept"] == [
        {"label": "ToyColor", "id": "red"},
        {"label": "ToyColor", "id": "blue"},
    ]


def test_claim_row_kept_follows_the_class_closure(materializer: LabelMaterializer) -> None:
    claim = EdgeClaim(
        anchor=ALPHA,
        property_iri=TOY_PAIRS_WITH,
        direction=ClaimDirection.OUT,
        other_class_iri=TOY_GADGET,
    )

    rows = build_claim_rows(build_claim_batch(claim), materializer, ClaimDirection.OUT)

    # w1 is a ToyWidget, which carries the ToyGadget label; beta is a ToyGadget.
    assert rows[0]["kept"] == [
        {"label": "ToyWidget", "id": "w1"},
        {"label": "ToyGadget", "id": "beta"},
    ]


def test_claim_row_in_direction_keeps_the_start_nodes(materializer: LabelMaterializer) -> None:
    claim = EdgeClaim(
        anchor=NodeRecord(class_iri=TOY_COLOR, id="red", name="Red").node_ref,
        property_iri=TOY_HAS_COLOR,
        direction=ClaimDirection.IN,
        other_class_iri=TOY_GADGET,
    )

    rows = build_claim_rows(build_claim_batch(claim), materializer, ClaimDirection.IN)

    assert rows[0]["kept"] == [
        {"label": "ToyGadget", "id": "alpha"},
        {"label": "ToyWidget", "id": "w1"},
    ]


def test_claim_row_without_matching_edges_keeps_nothing(materializer: LabelMaterializer) -> None:
    claim = EdgeClaim(
        anchor=NodeRecord(class_iri=TOY_GADGET, id="beta", name="Beta").node_ref,
        property_iri=TOY_HAS_COLOR,
        direction=ClaimDirection.OUT,
        other_class_iri=TOY_COLOR,
    )

    rows = build_claim_rows(build_claim_batch(claim), materializer, ClaimDirection.OUT)

    assert rows[0]["kept"] == []


def test_prune_statements_differ_only_in_the_arrow() -> None:
    assert "]->(other" in runner.PRUNE_CLAIMED_OUT
    assert "<-[r" in runner.PRUNE_CLAIMED_IN
    assert runner.PRUNE_CLAIMED_OUT.endswith(runner.PRUNE_CLAIMED_TAIL)
    assert runner.PRUNE_CLAIMED_IN.endswith(runner.PRUNE_CLAIMED_TAIL)


async def test_sync_write_with_claims_raises_before_any_call(
    materializer: LabelMaterializer,
) -> None:
    client = RecordingGraphClient()
    scope = frozenset({TOY_GADGET, TOY_WIDGET, TOY_COLOR})
    batch = build_claim_batch(COLOR_CLAIM, scope=scope)
    message = f"Sync mode takes no claims: {TOY_HAS_COLOR} at {TOY_GADGET} 'alpha'"

    with pytest.raises(ValueError, match=message):
        await Runner(client, materializer).write(batch, WriteMode.SYNC)

    assert client.calls == []


async def test_write_can_skip_the_constraint_step(materializer: LabelMaterializer) -> None:
    client = RecordingGraphClient(TOY_SUMMARY)

    report = await Runner(client, materializer).write(
        build_claim_batch(COLOR_CLAIM), WriteMode.UPSERT, should_ensure_constraints=False
    )

    assert client.writes == []
    assert len(client.batches) == 1
    assert report.constraints_ensured == 0
    assert report.nodes_created == TOY_SUMMARY.nodes_created * len(client.batches[0])


async def test_ensure_constraints_returns_the_class_count(materializer: LabelMaterializer) -> None:
    client = RecordingGraphClient()

    ensured = await Runner(client, materializer).ensure_constraints(
        [TOY_GADGET, TOY_COLOR, TOY_GADGET]
    )

    assert ensured == 2
    assert any("id_unique_ToyGadget" in write for write in client.writes)
    assert any("id_unique_ToyColor" in write for write in client.writes)
