import dataclasses
from pathlib import Path
from typing import Any

import pytest

import muon.retrieval.context as context_module
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import load_ontology_model
from muon.plugins import UniversePlugin, discover_universes
from muon.retrieval.context import (
    READ_GRAPH_COUNTS,
    CacheStatus,
    SchemaCacheError,
    SchemaContext,
    SchemaRequest,
    compute_fingerprint,
    load_schema_context,
    render_schema_text,
)
from muon.retrieval.schema import COUNT_LABEL_SETS
from tests.fixtures.toy_ontology import TOY_UNIVERSE, copy_toy_ontology
from tests.fixtures.toy_plugin import write_toy_plugin
from tests.unit.retrieval.fakes import QueryGraphClient
from tests.unit.retrieval.toy_graph import build_toy_answers

GRAPH_COUNTS = {
    "nodes": [{"count": 10}, {"count": 3, "label": "ToyWidget"}],
    "relationships": [{"count": 5, "type": "HAS_COLOR"}, {"count": 2, "type": "PAIRS_WITH"}],
}
GOLDEN_TEXT = """\
# Graph schema: Toy universe (muontoy)

## Node labels

ToyColor (2 nodes; also Concept, Entity)
- id: STRING
- hue: STRING
- tags: LIST<STRING>

ToyGadget (1 node; also Entity, Object)
- id: STRING
- serial: INTEGER

ToyWidget (3 nodes; also Entity, Object, ToyGadget)
- id: STRING
- name?: STRING
- nickname?: STRING, values: a, b
- serial?: INTEGER | STRING

## Relationships

(:ToyWidget)-[:HAS_COLOR]->(:ToyColor) 3
(:ToyGadget)-[:PAIRS_WITH {strength?: STRING, values: high}]->(:ToyGadget) 4

## Notes

### Widgets

A widget is a gadget with a serial number.
"""


@dataclasses.dataclass
class ToySetup:
    plugin: UniversePlugin
    materializer: LabelMaterializer
    ontology_dir: Path
    cache_dir: Path
    counts: dict[str, Any]

    def build_request(self, should_refresh: bool = False) -> SchemaRequest:
        return SchemaRequest(self.plugin, self.materializer, should_refresh)

    def build_client(self) -> QueryGraphClient:
        answers = build_toy_answers()
        answers[READ_GRAPH_COUNTS] = lambda parameters: [self.counts]
        return QueryGraphClient(answers)


@pytest.fixture
def toy_setup(tmp_path: Path) -> ToySetup:
    ontology_dir = copy_toy_ontology(tmp_path / "ontology")
    universes_dir = tmp_path / "universes"
    universes_dir.mkdir()
    write_toy_plugin(universes_dir)
    plugin = discover_universes(universes_dir, ontology_dir).get(TOY_UNIVERSE)
    materializer = LabelMaterializer(load_ontology_model(ontology_dir, TOY_UNIVERSE))
    return ToySetup(plugin, materializer, ontology_dir, tmp_path / "cache", dict(GRAPH_COUNTS))


async def load(toy_setup: ToySetup, client: QueryGraphClient, **options: Any) -> SchemaContext:
    request = toy_setup.build_request(**options)
    return await load_schema_context(client, request, toy_setup.cache_dir)


async def test_missing_cache_introspects_and_writes_the_file(toy_setup: ToySetup) -> None:
    client = toy_setup.build_client()

    context = await load(toy_setup, client)

    assert context.cache_status is CacheStatus.MISSING
    assert context.cache_path == toy_setup.cache_dir / "muontoy.json"
    assert context.cache_path.is_file()
    assert COUNT_LABEL_SETS in client.list_queries()


async def test_second_load_is_a_hit_and_reads_only_the_counts(toy_setup: ToySetup) -> None:
    first = await load(toy_setup, toy_setup.build_client())
    client = toy_setup.build_client()

    second = await load(toy_setup, client)

    assert second.cache_status is CacheStatus.HIT
    assert client.list_queries() == [READ_GRAPH_COUNTS]
    assert second.graph_schema == first.graph_schema
    assert second.text == first.text


async def test_changed_graph_counts_make_the_cache_stale(toy_setup: ToySetup) -> None:
    await load(toy_setup, toy_setup.build_client())
    toy_setup.counts["nodes"] = [{"count": 11}]

    context = await load(toy_setup, toy_setup.build_client())

    assert context.cache_status is CacheStatus.STALE


async def test_changed_t_box_byte_makes_the_cache_stale(toy_setup: ToySetup) -> None:
    await load(toy_setup, toy_setup.build_client())
    t_box = toy_setup.ontology_dir / "universes" / f"{TOY_UNIVERSE}.ttl"
    t_box.write_text(t_box.read_text() + "\n# edited\n")

    context = await load(toy_setup, toy_setup.build_client())

    assert context.cache_status is CacheStatus.STALE


async def test_changed_format_version_makes_the_cache_stale(
    toy_setup: ToySetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    await load(toy_setup, toy_setup.build_client())
    monkeypatch.setattr(
        context_module, "SCHEMA_FORMAT_VERSION", context_module.SCHEMA_FORMAT_VERSION + 1
    )

    context = await load(toy_setup, toy_setup.build_client())

    assert context.cache_status is CacheStatus.STALE


async def test_refresh_introspects_although_the_cache_is_current(toy_setup: ToySetup) -> None:
    await load(toy_setup, toy_setup.build_client())
    client = toy_setup.build_client()

    context = await load(toy_setup, client, should_refresh=True)

    assert context.cache_status is CacheStatus.REFRESHED
    assert COUNT_LABEL_SETS in client.list_queries()


async def test_truncated_cache_file_is_invalid_and_rewritten(toy_setup: ToySetup) -> None:
    first = await load(toy_setup, toy_setup.build_client())
    first.cache_path.write_text(first.cache_path.read_text()[:40])

    second = await load(toy_setup, toy_setup.build_client())
    third = await load(toy_setup, toy_setup.build_client())

    assert second.cache_status is CacheStatus.INVALID
    assert third.cache_status is CacheStatus.HIT


async def test_two_writes_of_the_same_schema_are_byte_identical(toy_setup: ToySetup) -> None:
    first = await load(toy_setup, toy_setup.build_client(), should_refresh=True)
    first_bytes = first.cache_path.read_bytes()

    second = await load(toy_setup, toy_setup.build_client(), should_refresh=True)

    assert second.cache_path.read_bytes() == first_bytes


async def test_fingerprint_ignores_the_order_of_count_entries(toy_setup: ToySetup) -> None:
    paths = toy_setup.plugin.ontology_paths
    reordered = {name: list(reversed(entries)) for name, entries in GRAPH_COUNTS.items()}
    first = QueryGraphClient({READ_GRAPH_COUNTS: lambda parameters: [GRAPH_COUNTS]})
    second = QueryGraphClient({READ_GRAPH_COUNTS: lambda parameters: [reordered]})

    assert await compute_fingerprint(first, paths) == await compute_fingerprint(second, paths)


async def test_rendered_text_matches_the_golden_text(toy_setup: ToySetup) -> None:
    context = await load(toy_setup, toy_setup.build_client())

    assert context.text == GOLDEN_TEXT
    assert render_schema_text(context.graph_schema, toy_setup.plugin) == GOLDEN_TEXT


async def test_a_notes_edit_shows_on_a_cache_hit(toy_setup: ToySetup) -> None:
    await load(toy_setup, toy_setup.build_client())
    edited_plugin = dataclasses.replace(toy_setup.plugin, notes_text="Edited notes.\n")
    request = SchemaRequest(edited_plugin, toy_setup.materializer, should_refresh=False)

    context = await load_schema_context(toy_setup.build_client(), request, toy_setup.cache_dir)

    assert context.cache_status is CacheStatus.HIT
    assert context.notes == "Edited notes.\n"
    assert context.text.endswith("## Notes\n\nEdited notes.\n")


async def test_a_materializer_of_another_universe_raises(toy_setup: ToySetup) -> None:
    other_model = dataclasses.replace(toy_setup.materializer.model, universe="muonalt")
    request = SchemaRequest(toy_setup.plugin, LabelMaterializer(other_model), should_refresh=False)

    with pytest.raises(ValueError, match="muonalt"):
        await load_schema_context(toy_setup.build_client(), request, toy_setup.cache_dir)


async def test_a_cache_folder_that_cannot_be_created_raises(
    toy_setup: ToySetup, tmp_path: Path
) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("a file, not a folder")
    request = toy_setup.build_request()

    with pytest.raises(SchemaCacheError, match="blocked"):
        await load_schema_context(toy_setup.build_client(), request, blocked)
