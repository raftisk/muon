from collections.abc import Mapping
from typing import Any

import pytest

from muon.ontology.materializer import LabelMaterializer
from muon.retrieval.schema import (
    COUNT_LABEL_SETS,
    COUNT_PATTERNS,
    MAX_ENUM_VALUES,
    READ_NODE_PROPERTIES,
    READ_RELATIONSHIP_PROPERTIES,
    SAMPLE_NODE_VALUES,
    SAMPLE_RELATIONSHIP_VALUES,
    GraphSchema,
    NodeLabelSchema,
    PropertySchema,
    RelationshipPattern,
    RelationshipTypeSchema,
    SchemaIntrospectionError,
    find_leaf_label,
    introspect_schema,
    list_descendant_labels,
    normalize_property_type,
)
from tests.unit.retrieval.fakes import Answer, QueryGraphClient

UNIVERSE = "muontoy"
WIDGET_SET = ["ToyWidget", "ToyGadget", "Object", "Entity", UNIVERSE]
WIDGET_EXTRA_SET = [*WIDGET_SET, "Extra"]
GADGET_SET = ["ToyGadget", "Object", "Entity", UNIVERSE]
COLOR_SET = ["ToyColor", "Concept", "Entity", UNIVERSE]
STRING_ROW = "STRING NOT NULL"


def build_property_row(
    owner: list[str] | str, name: str, raw_type: str, is_mandatory: bool
) -> dict[str, Any]:
    return {"owner": owner, "name": name, "types": [raw_type], "is_mandatory": is_mandatory}


def answer(records: list[dict[str, Any]]) -> Answer:
    return lambda parameters: records


def sample_node_values(parameters: Mapping[str, Any]) -> list[dict[str, Any]]:
    if parameters["label"] == "ToyWidget":
        return [{"property": "nickname", "value": value} for value in ("b", "a")]
    too_many = MAX_ENUM_VALUES + 1
    return [{"property": "hue", "value": f"hue-{index}"} for index in range(too_many)]


def build_client() -> QueryGraphClient:
    node_properties = [
        build_property_row(WIDGET_SET, "id", STRING_ROW, True),
        build_property_row(WIDGET_SET, "name", STRING_ROW, True),
        build_property_row(WIDGET_SET, "nickname", STRING_ROW, False),
        build_property_row(WIDGET_SET, "serial", "INTEGER NOT NULL", False),
        build_property_row(WIDGET_EXTRA_SET, "id", STRING_ROW, True),
        build_property_row(WIDGET_EXTRA_SET, "serial", STRING_ROW, True),
        build_property_row(GADGET_SET, "id", STRING_ROW, True),
        build_property_row(GADGET_SET, "serial", "INTEGER NOT NULL", True),
        build_property_row(COLOR_SET, "id", STRING_ROW, True),
        build_property_row(COLOR_SET, "hue", STRING_ROW, True),
        build_property_row(COLOR_SET, "tags", "LIST<STRING NOT NULL> NOT NULL", True),
    ]
    relationship_properties = [
        build_property_row(":`PAIRS_WITH`", "strength", STRING_ROW, False),
        build_property_row(":`OTHER_UNIVERSE_TYPE`", "ignored", STRING_ROW, False),
    ]
    return QueryGraphClient(
        {
            COUNT_LABEL_SETS: answer(
                [
                    {"labels": WIDGET_SET, "count": 2},
                    {"labels": WIDGET_EXTRA_SET, "count": 1},
                    {"labels": GADGET_SET, "count": 1},
                    {"labels": COLOR_SET, "count": 2},
                ]
            ),
            READ_NODE_PROPERTIES: answer(node_properties),
            COUNT_PATTERNS: answer(
                [
                    {
                        "start_labels": WIDGET_SET,
                        "type": "HAS_COLOR",
                        "end_labels": COLOR_SET,
                        "count": 2,
                    },
                    {
                        "start_labels": WIDGET_EXTRA_SET,
                        "type": "HAS_COLOR",
                        "end_labels": COLOR_SET,
                        "count": 1,
                    },
                    {
                        "start_labels": GADGET_SET,
                        "type": "PAIRS_WITH",
                        "end_labels": GADGET_SET,
                        "count": 4,
                    },
                ]
            ),
            READ_RELATIONSHIP_PROPERTIES: answer(relationship_properties),
            SAMPLE_NODE_VALUES: sample_node_values,
            SAMPLE_RELATIONSHIP_VALUES: answer([{"property": "strength", "value": "high"}]),
        }
    )


def test_find_leaf_label_picks_the_most_specific_class(toy_materializer: LabelMaterializer) -> None:
    ancestors = toy_materializer.map_ancestor_labels()

    leaf = find_leaf_label(
        ["ToyWidget", "ToyGadget", "Object", "Entity", UNIVERSE], ancestors, UNIVERSE
    )

    assert leaf == "ToyWidget"


def test_find_leaf_label_ignores_labels_that_are_not_classes(
    toy_materializer: LabelMaterializer,
) -> None:
    ancestors = toy_materializer.map_ancestor_labels()

    assert (
        find_leaf_label(["MuonTest", "ToyColor", "Concept", UNIVERSE], ancestors, UNIVERSE)
        == "ToyColor"
    )


@pytest.mark.parametrize(
    "labels",
    [[UNIVERSE, "MuonTest"], ["ToyColor", "ToyGadget", "Entity", UNIVERSE]],
    ids=["no class label", "2 unrelated leaves"],
)
def test_find_leaf_label_raises_naming_the_label_set(
    toy_materializer: LabelMaterializer, labels: list[str]
) -> None:
    with pytest.raises(SchemaIntrospectionError) as error_info:
        find_leaf_label(labels, toy_materializer.map_ancestor_labels(), UNIVERSE)

    assert str(sorted(labels)) in str(error_info.value)


@pytest.mark.parametrize(
    ("raw_type", "expected"),
    [
        ("STRING NOT NULL", "STRING"),
        ("INTEGER", "INTEGER"),
        ("LIST<STRING NOT NULL> NOT NULL", "LIST<STRING>"),
        ("LIST<FLOAT NOT NULL>", "LIST<FLOAT>"),
    ],
)
def test_normalize_property_type(raw_type: str, expected: str) -> None:
    assert normalize_property_type(raw_type) == expected


def test_list_descendant_labels(toy_materializer: LabelMaterializer) -> None:
    ancestors = toy_materializer.map_ancestor_labels()

    assert list_descendant_labels("ToyGadget", ancestors) == ["ToyWidget"]
    assert list_descendant_labels("ToyWidget", ancestors) == []


async def test_introspect_schema_builds_the_schema_from_canned_rows(
    toy_materializer: LabelMaterializer,
) -> None:
    client = build_client()

    schema = await introspect_schema(client, toy_materializer)

    assert schema == GraphSchema(
        universe=UNIVERSE,
        node_labels=(
            NodeLabelSchema(
                label="ToyColor",
                count=2,
                ancestor_labels=("Concept", "Entity"),
                properties=(
                    PropertySchema(
                        name="id", types=("STRING",), is_mandatory=True, enum_values=None
                    ),
                    PropertySchema(
                        name="hue", types=("STRING",), is_mandatory=True, enum_values=None
                    ),
                    PropertySchema(
                        name="tags", types=("LIST<STRING>",), is_mandatory=True, enum_values=None
                    ),
                ),
            ),
            NodeLabelSchema(
                label="ToyGadget",
                count=1,
                ancestor_labels=("Entity", "Object"),
                properties=(
                    PropertySchema(
                        name="id", types=("STRING",), is_mandatory=True, enum_values=None
                    ),
                    PropertySchema(
                        name="serial", types=("INTEGER",), is_mandatory=True, enum_values=None
                    ),
                ),
            ),
            NodeLabelSchema(
                label="ToyWidget",
                count=3,
                ancestor_labels=("Entity", "Object", "ToyGadget"),
                properties=(
                    PropertySchema(
                        name="id", types=("STRING",), is_mandatory=True, enum_values=None
                    ),
                    PropertySchema(
                        name="name", types=("STRING",), is_mandatory=False, enum_values=None
                    ),
                    PropertySchema(
                        name="nickname",
                        types=("STRING",),
                        is_mandatory=False,
                        enum_values=("a", "b"),
                    ),
                    PropertySchema(
                        name="serial",
                        types=("INTEGER", "STRING"),
                        is_mandatory=False,
                        enum_values=None,
                    ),
                ),
            ),
        ),
        relationship_types=(
            RelationshipTypeSchema(type="HAS_COLOR", properties=()),
            RelationshipTypeSchema(
                type="PAIRS_WITH",
                properties=(
                    PropertySchema(
                        name="strength",
                        types=("STRING",),
                        is_mandatory=False,
                        enum_values=("high",),
                    ),
                ),
            ),
        ),
        patterns=(
            RelationshipPattern(
                start_label="ToyWidget", type="HAS_COLOR", end_label="ToyColor", count=3
            ),
            RelationshipPattern(
                start_label="ToyGadget", type="PAIRS_WITH", end_label="ToyGadget", count=4
            ),
        ),
    )


async def test_introspection_scopes_every_read_and_never_writes(
    toy_materializer: LabelMaterializer,
) -> None:
    client = build_client()

    await introspect_schema(client, toy_materializer)

    unscoped = [query for query, parameters in client.reads if "universe" not in parameters]
    assert unscoped == [READ_RELATIONSHIP_PROPERTIES]


async def test_node_enum_query_gets_the_label_and_the_descendants(
    toy_materializer: LabelMaterializer,
) -> None:
    client = build_client()

    await introspect_schema(client, toy_materializer)

    widget_parameters = next(
        parameters
        for query, parameters in client.reads
        if query == SAMPLE_NODE_VALUES and parameters["label"] == "ToyWidget"
    )
    assert widget_parameters["properties"] == ["nickname"]
    assert widget_parameters["descendant_labels"] == []
    assert widget_parameters["limit"] == MAX_ENUM_VALUES + 1


async def test_a_leaf_with_no_enum_candidate_runs_no_sample_query(
    toy_materializer: LabelMaterializer,
) -> None:
    client = build_client()

    await introspect_schema(client, toy_materializer)

    sampled_labels = [
        parameters["label"] for query, parameters in client.reads if query == SAMPLE_NODE_VALUES
    ]
    assert sorted(sampled_labels) == ["ToyColor", "ToyWidget"]


async def test_an_empty_universe_raises_and_names_the_seed_command(
    toy_materializer: LabelMaterializer,
) -> None:
    client = QueryGraphClient({COUNT_LABEL_SETS: answer([])})

    with pytest.raises(SchemaIntrospectionError, match="muon seed muontoy"):
        await introspect_schema(client, toy_materializer)


async def test_a_label_set_with_no_class_raises(toy_materializer: LabelMaterializer) -> None:
    client = QueryGraphClient({COUNT_LABEL_SETS: answer([{"labels": [UNIVERSE], "count": 1}])})

    with pytest.raises(SchemaIntrospectionError, match=UNIVERSE):
        await introspect_schema(client, toy_materializer)
