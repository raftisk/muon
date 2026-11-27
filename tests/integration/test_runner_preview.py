"""The delete preview against the live Neo4j in `.env`.

Every test writes `muontoy` nodes only, and the shared `toy_runner` fixture purges that
universe afterwards. A test writes a batch, previews an edited copy and then writes the
copy to see what it deletes.
"""

import pytest

from muon.graph import Neo4jGraphClient
from muon.ingestion.batch import ClaimDirection, GraphBatch
from muon.ingestion.runner import DeletePreview, Runner, WriteMode
from tests.integration.test_runner_claims import (
    ALPHA,
    BETA,
    BLUE,
    FIRST_WIDGET,
    RED,
    SECOND_WIDGET,
    TOY_COLOR,
    TOY_GADGET,
    TOY_HAS_COLOR,
    TOY_PAIRS_WITH,
    TOY_WIDGET,
    build_batch,
    build_claim,
    build_edge,
    read_edges,
    read_node_ids,
)

pytestmark = pytest.mark.integration

SCOPE = frozenset({TOY_GADGET, TOY_WIDGET})
CLAIMS = (
    build_claim(ALPHA, TOY_PAIRS_WITH, ClaimDirection.OUT, TOY_WIDGET, True),
    build_claim(ALPHA, TOY_HAS_COLOR, ClaimDirection.OUT, TOY_COLOR),
    build_claim(RED, TOY_HAS_COLOR, ClaimDirection.IN, TOY_GADGET),
)


def build_first_batch() -> GraphBatch:
    edges = (
        build_edge(TOY_PAIRS_WITH, ALPHA, FIRST_WIDGET),
        build_edge(TOY_PAIRS_WITH, ALPHA, SECOND_WIDGET),
        build_edge(TOY_HAS_COLOR, ALPHA, RED),
        build_edge(TOY_HAS_COLOR, ALPHA, BLUE),
        build_edge(TOY_HAS_COLOR, BETA, RED),
        # Not claimed: it goes only because its widget is detach-deleted.
        build_edge(TOY_HAS_COLOR, SECOND_WIDGET, BLUE),
    )
    nodes = (ALPHA, BETA, FIRST_WIDGET, SECOND_WIDGET, RED, BLUE)
    return build_batch(SCOPE, nodes, edges, CLAIMS)


def build_edited_batch() -> GraphBatch:
    edges = (
        build_edge(TOY_PAIRS_WITH, ALPHA, FIRST_WIDGET),
        build_edge(TOY_HAS_COLOR, ALPHA, RED),
    )
    return build_batch(SCOPE, (ALPHA, BETA, FIRST_WIDGET, RED), edges, CLAIMS)


def list_edge_keys(preview: DeletePreview) -> set[tuple[str, str, str]]:
    return {(edge.start_id, edge.type, edge.end_id) for edge in preview.edges}


async def test_preview_lists_what_the_write_deletes(
    toy_runner: Runner, graph_client: Neo4jGraphClient
) -> None:
    await toy_runner.write(build_first_batch(), WriteMode.UPSERT)
    nodes_before = await read_node_ids(graph_client)
    edges_before = await read_edges(graph_client)

    preview = await toy_runner.preview_deletes(build_edited_batch())
    await toy_runner.write(build_edited_batch(), WriteMode.UPSERT)

    deleted_nodes = nodes_before - await read_node_ids(graph_client)
    deleted_edges = edges_before - await read_edges(graph_client)
    detached = {edge for edge in edges_before if {edge[0], edge[2]} & deleted_nodes}
    assert {node.id for node in preview.nodes} == deleted_nodes == {"w2"}
    assert list_edge_keys(preview) == {
        ("alpha", "PAIRS_WITH", "w2"),
        ("alpha", "HAS_COLOR", "blue"),
        ("beta", "HAS_COLOR", "red"),
    }
    assert deleted_edges == list_edge_keys(preview) | detached
    widget_labels = toy_runner.materializer.map_class_labels(TOY_WIDGET)
    assert preview.nodes[0].labels == tuple(sorted(widget_labels))


async def test_preview_writes_nothing(toy_runner: Runner) -> None:
    await toy_runner.write(build_first_batch(), WriteMode.UPSERT)
    census_before = await toy_runner.count_universe()

    preview = await toy_runner.preview_deletes(build_edited_batch())

    assert not preview.is_empty
    assert await toy_runner.count_universe() == census_before


async def test_preview_of_the_written_batch_is_empty(toy_runner: Runner) -> None:
    await toy_runner.write(build_first_batch(), WriteMode.UPSERT)

    preview = await toy_runner.preview_deletes(build_first_batch())

    assert preview.is_empty
