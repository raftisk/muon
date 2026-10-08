"""Maps ontology terms to Neo4j names.

This is the only module that turns a class or property IRI into a label, a
relationship type or a node property name.
"""

import re

from rdflib.namespace import RDFS

from muon.ontology.model import OntologyModel, extract_local_name

NAME_PROPERTY = "name"
LABEL_PROPERTY_IRI = str(RDFS.label)
WORD_BOUNDARY_PATTERN = re.compile(r"(?<=[a-z])(?=[A-Z])")
WORD_SEPARATOR = "_"


def format_relationship_type(local_name: str) -> str:
    """Upper-case `local_name` with an underscore at each lowercase-to-uppercase step."""
    return WORD_BOUNDARY_PATTERN.sub(WORD_SEPARATOR, local_name).upper()


class LabelMaterializer:
    """Neo4j names for the terms of one ontology model.

    Every class in the subclass closure becomes a label, plus the universe label.
    """

    def __init__(self, model: OntologyModel) -> None:
        self.model = model

    @property
    def universe_label(self) -> str:
        return self.model.universe

    def map_class_label(self, class_iri: str) -> str:
        return extract_local_name(class_iri)

    def map_class_labels(self, class_iri: str) -> tuple[str, ...]:
        """Return the class label first, its ancestor labels sorted, the universe label last."""
        ancestor_labels = sorted(
            extract_local_name(ancestor) for ancestor in self.model.ancestors[class_iri]
        )
        return (self.map_class_label(class_iri), *ancestor_labels, self.universe_label)

    def list_ontology_labels(self) -> frozenset[str]:
        """Return the label of every named class, the set a node's ontology labels come from."""
        return frozenset(extract_local_name(class_iri) for class_iri in self.model.classes)

    def map_relationship_type(self, property_iri: str) -> str:
        return format_relationship_type(extract_local_name(property_iri))

    def map_property_name(self, property_iri: str) -> str:
        if property_iri == LABEL_PROPERTY_IRI:
            return NAME_PROPERTY
        return extract_local_name(property_iri)
