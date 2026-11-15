"""Graph access layer: the client protocol, its Neo4j implementation and error types."""

from muon.graph.client import Neo4jGraphClient
from muon.graph.errors import GraphConnectionError, GraphError, GraphQueryError
from muon.graph.protocol import (
    GraphClient,
    GraphNode,
    GraphPath,
    GraphRelationship,
    Statement,
    WriteSummary,
)

__all__ = [
    "GraphClient",
    "GraphConnectionError",
    "GraphError",
    "GraphNode",
    "GraphPath",
    "GraphQueryError",
    "GraphRelationship",
    "Neo4jGraphClient",
    "Statement",
    "WriteSummary",
]
