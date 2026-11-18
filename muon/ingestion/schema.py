"""Schema applier: uniqueness constraints on the node key and the name index.

Map: `ensure_id_constraints` creates one `id` constraint per class label and per key label, and
`ensure_name_index` creates the full-text index that retrieval uses to resolve names.

Neo4j accepts no label parameter in schema statements, so this module is the only
place where a label enters query text. It does so only for a label that matches
`CONSTRAINT_LABEL_PATTERN`, quoted with backticks.
"""

import re
from collections.abc import Iterable
from typing import LiteralString

from muon.graph import GraphClient
from muon.graph.indexes import ALIASES_FIELD, NAME_FIELD, NAME_INDEX, NAME_INDEX_LABELS
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import OntologyError

CONSTRAINT_LABEL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
CONSTRAINT_NAME_PREFIX = "id_unique_"


def build_constraint_name(label: str) -> str:
    return f"{CONSTRAINT_NAME_PREFIX}{label}"


def list_constraint_labels(
    materializer: LabelMaterializer, class_iris: Iterable[str]
) -> tuple[str, ...]:
    """Return the sorted labels that need an `id` constraint: each class and its key label."""
    labels: set[str] = set()
    for class_iri in set(class_iris):
        if class_iri not in materializer.model.classes:
            raise OntologyError(f"Cannot constrain {class_iri}: it is not a named class")
        labels.add(materializer.map_class_label(class_iri))
        labels.add(materializer.map_key_label(class_iri))
    return tuple(sorted(labels))


def build_id_constraint(label: str) -> LiteralString:
    """Return the statement that makes `id` unique for `label`."""
    if not CONSTRAINT_LABEL_PATTERN.fullmatch(label):
        raise OntologyError(
            f"Cannot constrain label {label!r}, "
            f"which does not match {CONSTRAINT_LABEL_PATTERN.pattern}"
        )
    name = build_constraint_name(label)
    # Safe as query text: `label` and `name` passed the pattern check above.
    statement: LiteralString = (
        f"CREATE CONSTRAINT `{name}` IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.id IS UNIQUE"
    )
    return statement


async def ensure_id_constraints(
    client: GraphClient, materializer: LabelMaterializer, class_iris: Iterable[str]
) -> int:
    """Create the missing constraints, 1 transaction each; return how many labels were ensured."""
    labels = list_constraint_labels(materializer, class_iris)
    for label in labels:
        await client.execute_write(build_id_constraint(label))
    return len(labels)


def build_name_index_statement() -> LiteralString:
    """Return the statement that creates the full-text index over names and aliases."""
    for label in NAME_INDEX_LABELS:
        if not CONSTRAINT_LABEL_PATTERN.fullmatch(label):
            raise OntologyError(
                f"Cannot index label {label!r}: "
                f"it does not match {CONSTRAINT_LABEL_PATTERN.pattern}"
            )
    labels = "|".join(f"`{label}`" for label in NAME_INDEX_LABELS)
    # Safe as query text: the labels passed the pattern check and the rest are constants.
    statement: LiteralString = (
        f"CREATE FULLTEXT INDEX `{NAME_INDEX}` IF NOT EXISTS FOR (n:{labels}) "
        f"ON EACH [n.`{NAME_FIELD}`, n.`{ALIASES_FIELD}`]"
    )
    return statement


async def ensure_name_index(client: GraphClient) -> None:
    """Create the name index when it is missing. A second call changes nothing."""
    await client.execute_write(build_name_index_statement())
