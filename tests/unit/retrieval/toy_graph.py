"""Canned graph rows of a toy universe, as the introspection queries answer them."""

from collections.abc import Mapping
from typing import Any

from muon.retrieval.schema import (
    COUNT_LABEL_SETS,
    COUNT_PATTERNS,
    MAX_ENUM_VALUES,
    READ_NODE_PROPERTIES,
    READ_RELATIONSHIP_PROPERTIES,
    SAMPLE_NODE_VALUES,
    SAMPLE_RELATIONSHIP_VALUES,
)
from tests.unit.retrieval.fakes import Answer

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


def build_toy_answers() -> dict[str, Answer]:
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
    return {
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
