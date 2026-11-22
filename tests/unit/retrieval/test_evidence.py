from muon.retrieval.evidence import (
    MAX_CONTENT_ROWS,
    Cell,
    Evidence,
    EvidenceBundle,
    EvidenceStatus,
    GraphSlice,
    NodeCell,
    NodeKey,
    PathCell,
    RelationshipCell,
    SliceNode,
    SliceRelationship,
    render_evidence_content,
)

SPECIES_KEY = NodeKey(label="PokemonSpecies", id="regieleki")
TYPE_KEY = NodeKey(label="Type", id="electric")


def build_slice() -> GraphSlice:
    return GraphSlice(
        nodes=(
            SliceNode(
                key=SPECIES_KEY, name="Regieleki", labels=("PokemonSpecies",), source_url=None
            ),
            SliceNode(key=TYPE_KEY, name="Electric", labels=("Type",), source_url=None),
        ),
        relationships=(
            SliceRelationship(
                type="HAS_TYPE", start=SPECIES_KEY, end=TYPE_KEY, properties={"slot": 1}
            ),
        ),
    )


def build_evidence(rows: tuple[dict[str, Cell], ...], **overrides: object) -> Evidence:
    fields: dict[str, object] = {
        "id": "e1",
        "universe": "pkmn",
        "tool": "run_cypher",
        "query": "MATCH (n) RETURN n",
        "purpose": "fastest species",
        "status": EvidenceStatus.OK,
        "columns": tuple(rows[0]) if rows else (),
        "rows": rows,
        "row_count": len(rows),
        "graph_slice": build_slice(),
    }
    return Evidence.model_validate({**fields, **overrides})


def test_content_shows_a_node_and_a_scalar() -> None:
    evidence = build_evidence(({"s": NodeCell(key=SPECIES_KEY), "speed": 200},))

    assert render_evidence_content(evidence) == (
        "e1 ok, 1 row: fastest species\n"
        "columns: s, speed\n"
        '(PokemonSpecies regieleki "Regieleki") | 200'
    )


def test_content_cuts_120_rows_to_50_and_states_the_count() -> None:
    rows = tuple({"i": index} for index in range(120))

    content = render_evidence_content(build_evidence(rows))

    assert content.splitlines()[0] == "e1 ok, 120 rows, first 50 shown: fastest species"
    assert len(content.splitlines()) == 2 + MAX_CONTENT_ROWS


def test_content_shows_relationship_and_path_forms() -> None:
    relationship = RelationshipCell(type="HAS_TYPE", start=SPECIES_KEY, end=TYPE_KEY)
    path = PathCell(nodes=(SPECIES_KEY, TYPE_KEY), relationships=(relationship,))

    content = render_evidence_content(build_evidence(({"r": relationship, "p": path},)))

    assert content.splitlines()[2] == (
        '[:HAS_TYPE] | (PokemonSpecies regieleki "Regieleki")'
        '-[:HAS_TYPE]-(Type electric "Electric")'
    )


def test_content_shows_a_list_of_nodes_and_a_missing_name() -> None:
    unnamed = NodeKey(label="Type", id="shadow")
    cells: list[Cell] = [NodeCell(key=TYPE_KEY), NodeCell(key=unnamed)]

    content = render_evidence_content(build_evidence(({"types": cells},)))

    assert content.splitlines()[2] == '[(Type electric "Electric"), (Type shadow)]'


def test_content_of_rejected_and_failed_items_shows_the_message() -> None:
    for status in (EvidenceStatus.REJECTED, EvidenceStatus.FAILED):
        evidence = build_evidence((), status=status, message="unknown label Pokemon")

        assert render_evidence_content(evidence) == (
            f"e1 {status}: fastest species\nunknown label Pokemon"
        )


def test_content_is_the_same_for_a_final_and_a_non_final_item() -> None:
    rows = ({"s": NodeCell(key=SPECIES_KEY), "speed": 200},)

    final = build_evidence(rows, is_final=True)
    other = build_evidence(rows, is_final=False)

    assert render_evidence_content(final) == render_evidence_content(other)


def test_content_of_an_empty_item_has_no_columns_line() -> None:
    evidence = build_evidence((), status=EvidenceStatus.EMPTY)

    assert render_evidence_content(evidence) == "e1 empty, 0 rows: fastest species"


def test_bundle_round_trips_through_json_with_nested_cells() -> None:
    relationship = RelationshipCell(type="HAS_TYPE", start=SPECIES_KEY, end=TYPE_KEY)
    cells: list[Cell] = [NodeCell(key=TYPE_KEY), {"inner": [relationship, 1.5, None, True]}]
    row: dict[str, Cell] = {
        "path": PathCell(nodes=(SPECIES_KEY, TYPE_KEY), relationships=(relationship,)),
        "nested": cells,
        "plain": {"kind": "other"},
    }
    bundle = EvidenceBundle(universe="pkmn", question="q", items=(build_evidence((row,)),))

    restored = EvidenceBundle.model_validate_json(bundle.model_dump_json())

    assert restored == bundle
    assert restored.get("e1") is not None
    assert restored.get("e9") is None


def test_slice_node_has_no_properties_by_default() -> None:
    node = SliceNode(key=SPECIES_KEY, name="Regieleki", labels=("PokemonSpecies",), source_url=None)

    assert node.properties == {}
