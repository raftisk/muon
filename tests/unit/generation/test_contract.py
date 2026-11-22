import json

import pytest
from pydantic import ValidationError

from muon.generation.contract import (
    GRAPH_ARTIFACT_TYPE,
    TABLE_ARTIFACT_TYPE,
    Artifact,
    ArtifactEdge,
    ArtifactEnvelope,
    ArtifactNode,
    GraphData,
    Source,
    TableColumn,
    TableData,
)

SOURCE = Source(number=1, url="https://example.org/a", title="A")


def build_graph_data() -> GraphData:
    node = ArtifactNode(
        key="ToyWidget:w1",
        handle="n1",
        label="ToyWidget",
        name="Alpha",
        is_cited=True,
        source_number=1,
        properties={"serial": 1},
    )
    edge = ArtifactEdge(type="HAS_COLOR", start="ToyWidget:w1", end="ToyColor:red", properties={})
    return GraphData(nodes=(node,), edges=(edge,), sources=(SOURCE,), is_truncated=False)


def build_table_data() -> TableData:
    return TableData(
        columns=(TableColumn(key="w", label="w"),),
        rows=({"w": "Alpha", "tags": ["a", {"b": None}]},),
    )


def build_envelope(data: GraphData | TableData, artifact_type: str) -> ArtifactEnvelope:
    return ArtifactEnvelope(
        artifact=Artifact(id="art_1", type=artifact_type, title="Widgets", data=data)
    )


def test_graph_envelope_uses_camel_case_keys() -> None:
    envelope = build_envelope(build_graph_data(), GRAPH_ARTIFACT_TYPE)

    data = json.loads(envelope.to_json())["artifact"]["data"]

    assert data["isTruncated"] is False
    assert data["nodes"][0]["isCited"] is True
    assert data["nodes"][0]["sourceNumber"] == 1
    assert "is_cited" not in data["nodes"][0]


@pytest.mark.parametrize(
    ("data", "artifact_type"),
    [
        (build_graph_data(), GRAPH_ARTIFACT_TYPE),
        (build_table_data(), TABLE_ARTIFACT_TYPE),
    ],
)
def test_envelope_round_trips_through_json(data: GraphData | TableData, artifact_type: str) -> None:
    envelope = build_envelope(data, artifact_type)

    assert ArtifactEnvelope.model_validate_json(envelope.to_json()) == envelope


@pytest.mark.parametrize("field", ["id", "type", "title"])
def test_an_empty_envelope_field_fails_validation(field: str) -> None:
    fields = {"id": "art_1", "type": GRAPH_ARTIFACT_TYPE, "title": "Widgets", field: ""}

    with pytest.raises(ValidationError, match=field):
        Artifact(data=build_graph_data(), **fields)


def test_an_unknown_key_fails_validation() -> None:
    with pytest.raises(ValidationError, match="extra"):
        Source.model_validate({"number": 1, "url": "u", "title": "t", "extra": 1})
