"""Maps ontology terms to Neo4j names.

This is the only module that turns a class or property IRI into a label, a
relationship type or a node property name.
"""

import re
from collections.abc import Collection, Mapping

from rdflib.namespace import RDFS

from muon.ontology.model import OntologyError, OntologyModel, extract_local_name
from muon.plugins import UniverseManifest

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
    `key_classes` are local names of classes that key a class family: a node is found by
    `id` on the label of its nearest key class, see `map_key_label`. A key class must not
    have a key class ancestor.
    """

    def __init__(self, model: OntologyModel, key_classes: Collection[str] = ()) -> None:
        self.model = model
        self.key_class_iris = self.resolve_key_classes(key_classes)

    def resolve_key_classes(self, key_classes: Collection[str]) -> frozenset[str]:
        iris_by_name = {extract_local_name(iri): iri for iri in self.model.classes}
        resolved: set[str] = set()
        for name in sorted(key_classes):
            if name not in iris_by_name:
                raise OntologyError(
                    f"Key class {name!r} is not a class of the {self.model.universe} model"
                )
            resolved.add(iris_by_name[name])
        for iri in sorted(resolved):
            nested = sorted(self.model.ancestors[iri] & resolved)
            if nested:
                raise OntologyError(
                    f"Key class {extract_local_name(iri)!r} has the key class "
                    f"{extract_local_name(nested[0])!r} as an ancestor"
                )
        return frozenset(resolved)

    @property
    def universe_label(self) -> str:
        return self.model.universe

    def map_class_label(self, class_iri: str) -> str:
        return extract_local_name(class_iri)

    def map_key_label(self, class_iri: str) -> str:
        """Return the label of the nearest key class of `class_iri`, itself included.

        A class with no key class in its ancestry is its own key class.
        """
        if class_iri in self.key_class_iris:
            return self.map_class_label(class_iri)
        key_ancestors = self.model.ancestors[class_iri] & self.key_class_iris
        if not key_ancestors:
            return self.map_class_label(class_iri)
        nearest = max(
            key_ancestors, key=lambda ancestor: (len(self.model.ancestors[ancestor]), ancestor)
        )
        return self.map_class_label(nearest)

    def map_class_labels(self, class_iri: str) -> tuple[str, ...]:
        """Return the class label first, its ancestor labels sorted, the universe label last."""
        ancestor_labels = sorted(
            extract_local_name(ancestor) for ancestor in self.model.ancestors[class_iri]
        )
        return (self.map_class_label(class_iri), *ancestor_labels, self.universe_label)

    def list_ontology_labels(self) -> frozenset[str]:
        """Return the label of every named class, the set a node's ontology labels come from."""
        return frozenset(extract_local_name(class_iri) for class_iri in self.model.classes)

    def map_ancestor_labels(self) -> Mapping[str, frozenset[str]]:
        """Map every class label to the labels of its strict ancestors.

        The universe label is not included.
        """
        return {
            self.map_class_label(class_iri): frozenset(
                extract_local_name(ancestor) for ancestor in ancestors
            )
            for class_iri, ancestors in self.model.ancestors.items()
        }

    def map_relationship_type(self, property_iri: str) -> str:
        return format_relationship_type(extract_local_name(property_iri))

    def map_property_name(self, property_iri: str) -> str:
        if property_iri == LABEL_PROPERTY_IRI:
            return NAME_PROPERTY
        return extract_local_name(property_iri)


def build_plugin_materializer(
    model: OntologyModel, manifest: UniverseManifest
) -> LabelMaterializer:
    """Return the materializer of a universe, with the key classes of its manifest."""
    return LabelMaterializer(model, manifest.key_classes)


def check_artifact_properties(
    materializer: LabelMaterializer, properties: Mapping[str, tuple[str, ...]]
) -> None:
    """Raise `OntologyError` when a class or property of `properties` is not in the model.

    Classes are matched by local name and properties by their Neo4j name.
    """
    model = materializer.model
    class_names = {extract_local_name(iri) for iri in model.classes}
    property_names = {materializer.map_property_name(iri) for iri in model.datatype_properties}
    for class_name, names in properties.items():
        if class_name not in class_names:
            raise OntologyError(
                f"Artifact class {class_name!r} is not a class of the {model.universe} model"
            )
        for name in names:
            if name not in property_names:
                raise OntologyError(
                    f"Artifact property {name!r} of {class_name!r} is not a datatype "
                    f"property of the {model.universe} model"
                )
