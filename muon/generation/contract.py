"""The answer contract: the models an answer is built from and serialized with.

Map: `Source` is a numbered page link. `ArtifactNode`, `ArtifactEdge` and `GraphData` describe
the graph artifact; `TableColumn` and `TableData` describe a table artifact. `Artifact` wraps
either as `data` and `ArtifactEnvelope` is the JSON object a client receives. `CitedHandles`
is what the marker rewriter recorded and `CitedAnswer` is the finished answer. Every model
serializes with camelCase keys.
"""

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from muon.retrieval.evidence import Cell, NodeKey

GRAPH_ARTIFACT_TYPE = "application/vnd.muon.graph"
TABLE_ARTIFACT_TYPE = "application/vnd.toki.table"
ARTIFACT_ID_PREFIX = "art_"

type TableCell = str | int | float | bool | None | list[TableCell] | dict[str, TableCell]


class ContractModel(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", alias_generator=to_camel, populate_by_name=True
    )


class Source(ContractModel):
    """A page that the answer cites, numbered from 1 in order of first citation."""

    number: int
    url: str
    title: str


class ArtifactNode(ContractModel):
    """A graph node. `key` is `label:id`; `handle` is `None` for a node the model never saw."""

    key: str
    handle: str | None
    label: str
    name: str | None
    is_cited: bool
    source_number: int | None
    properties: dict[str, Cell]


class ArtifactEdge(ContractModel):
    """A relationship between 2 node keys."""

    type: str
    start: str
    end: str
    properties: dict[str, Cell]


class GraphData(ContractModel):
    """The data of a graph artifact. `is_truncated` is true when nodes were dropped for size."""

    nodes: tuple[ArtifactNode, ...]
    edges: tuple[ArtifactEdge, ...]
    sources: tuple[Source, ...]
    is_truncated: bool


class TableColumn(ContractModel):
    key: str
    label: str


class TableData(ContractModel):
    """The data of a table artifact: 1 mapping per row, keyed by column key."""

    columns: tuple[TableColumn, ...]
    rows: tuple[dict[str, TableCell], ...]


class Artifact(ContractModel):
    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    title: str = Field(min_length=1)
    data: GraphData | TableData


class ArtifactEnvelope(ContractModel):
    """The JSON object a client receives for 1 artifact."""

    artifact: Artifact

    def to_json(self) -> str:
        return self.model_dump_json(by_alias=True)


class CitedHandles(ContractModel):
    """What the marker rewriter recorded.

    `node_keys` and `table_ids` are in citation order and `sources` in number order.
    """

    node_keys: tuple[NodeKey, ...]
    table_ids: tuple[str, ...]
    sources: tuple[Source, ...]
    warnings: tuple[str, ...]


class CitedAnswer(ContractModel):
    """The finished answer: linked text with its attribution, sources and artifacts."""

    text: str
    sources: tuple[Source, ...]
    graph: ArtifactEnvelope | None
    tables: tuple[ArtifactEnvelope, ...]
    cited: CitedHandles
    warnings: tuple[str, ...]
