"""The sweep against the live Neo4j in `.env`.

The tests write `muontoy` nodes only, and the shared `toy_runner` fixture purges that
universe afterwards.
"""

from pathlib import Path

import pytest

from muon.graph import Neo4jGraphClient
from muon.ingestion.runner import Runner
from muon.ingestion.seed import read_abox, seed_universe
from muon.validation.sweep import run_sweep
from tests.fixtures.toy_ontology import TOY_UNIVERSE

pytestmark = pytest.mark.integration

GADGET_LABELS = "ToyGadget:Object:Entity"
CREATE_NODE = f"CREATE (:{GADGET_LABELS}:{TOY_UNIVERSE} {{id: $id, name: $id}})"


async def seed_toy_universe(graph_client: Neo4jGraphClient, ontology_dir: Path) -> None:
    await seed_universe(graph_client, ontology_dir, TOY_UNIVERSE)


async def sweep(toy_runner: Runner, ontology_dir: Path):
    model = toy_runner.materializer.model
    abox = read_abox(model, ontology_dir)
    return await run_sweep(toy_runner.client, toy_runner.materializer, model, abox)


async def test_clean_seeded_universe_does_not_fail(
    toy_runner: Runner, graph_client: Neo4jGraphClient, tmp_path: Path
) -> None:
    await seed_toy_universe(graph_client, tmp_path)

    report = await sweep(toy_runner, tmp_path)

    assert not report.is_failing
    assert report.stubs == {}
    assert report.orphan_stubs == ()


async def test_node_with_an_incomplete_label_set_is_an_l1_finding(
    toy_runner: Runner, graph_client: Neo4jGraphClient, tmp_path: Path
) -> None:
    await seed_toy_universe(graph_client, tmp_path)
    await graph_client.execute_write(
        f"CREATE (:ToyWidget:{TOY_UNIVERSE} {{id: 'bare', name: 'Bare'}})"
    )

    report = await sweep(toy_runner, tmp_path)

    assert report.is_failing
    (finding,) = report.label_sets
    assert finding.labels == ("ToyWidget",)
    assert finding.sample_ids == ("bare",)


async def test_has_color_edge_ending_at_a_gadget_is_a_t1_finding(
    toy_runner: Runner, graph_client: Neo4jGraphClient, tmp_path: Path
) -> None:
    await seed_toy_universe(graph_client, tmp_path)
    await graph_client.execute_write(
        f"MATCH (a:{TOY_UNIVERSE} {{id: 'alpha'}}), (b:{TOY_UNIVERSE} {{id: 'beta-unit'}}) "
        "CREATE (a)-[:HAS_COLOR]->(b)"
    )

    report = await sweep(toy_runner, tmp_path)

    assert report.is_failing
    finding = next(item for item in report.edge_typing if item.end == "end")
    assert finding.relationship_type == "HAS_COLOR"
    assert finding.sample_ids == ("beta-unit",)


async def test_stub_with_no_relationship_is_an_orphan_and_does_not_fail(
    toy_runner: Runner, graph_client: Neo4jGraphClient, tmp_path: Path
) -> None:
    await seed_toy_universe(graph_client, tmp_path)
    await graph_client.execute_write(CREATE_NODE, {"id": "lonely"})

    report = await sweep(toy_runner, tmp_path)

    assert not report.is_failing
    assert report.stubs == {"ToyGadget": ("lonely",)}
    assert [orphan.id for orphan in report.orphan_stubs] == ["lonely"]


async def test_missing_abox_node_is_reported(
    toy_runner: Runner, graph_client: Neo4jGraphClient, tmp_path: Path
) -> None:
    await seed_toy_universe(graph_client, tmp_path)
    await graph_client.execute_write(f"MATCH (n:{TOY_UNIVERSE} {{id: 'red'}}) DETACH DELETE n")

    report = await sweep(toy_runner, tmp_path)

    assert report.is_failing
    assert [node.id for node in report.missing_abox] == ["red"]
