"""Schema applier: uniqueness constraints on the node key.

Neo4j accepts no label parameter in schema statements, so this module is the only
place where a label enters query text. It does so only for the label of a named
class whose local name matches `CONSTRAINT_LABEL_PATTERN`, quoted with backticks.
"""

import re
from collections.abc import Iterable
from typing import LiteralString

from muon.graph import GraphClient
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import OntologyError

CONSTRAINT_LABEL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
CONSTRAINT_NAME_PREFIX = "id_unique_"


def build_constraint_name(label: str) -> str:
    return f"{CONSTRAINT_NAME_PREFIX}{label}"


def build_id_constraint(materializer: LabelMaterializer, class_iri: str) -> LiteralString:
    """Return the statement that makes `id` unique for the label of `class_iri`."""
    if class_iri not in materializer.model.classes:
        raise OntologyError(f"Cannot constrain {class_iri}: it is not a named class")
    label = materializer.map_class_label(class_iri)
    if not CONSTRAINT_LABEL_PATTERN.fullmatch(label):
        raise OntologyError(
            f"Cannot constrain {class_iri}: label {label!r} does not match "
            f"{CONSTRAINT_LABEL_PATTERN.pattern}"
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
    """Create the missing constraints, 1 transaction each, and return how many were ensured."""
    statements = sorted(
        (build_id_constraint(materializer, class_iri) for class_iri in set(class_iris)),
    )
    for statement in statements:
        await client.execute_write(statement)
    return len(statements)
