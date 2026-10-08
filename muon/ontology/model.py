"""Read-only model of one universe's T-Box merged with the core T-Box.

The ontology directory holds `core/*.ttl` and `universes/<prefix>.ttl`. Every IRI
in the model is a plain `str`: rdflib terms hash and compare by type, so mixing
`URIRef` and `str` breaks set lookups.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS
from rdflib.plugins.parsers.notation3 import BadSyntax

CORE_DIR_NAME = "core"
UNIVERSES_DIR_NAME = "universes"
TTL_SUFFIX = ".ttl"
TURTLE_FORMAT = "turtle"
UNIVERSE_PREFIX_PATTERN = re.compile(r"^[a-z][a-z0-9]*$")
FRAGMENT_SEPARATOR = "#"
PATH_SEPARATOR = "/"


class OntologyError(Exception):
    """The ontology files of a universe cannot be loaded or used."""


@dataclass(frozen=True)
class OntologyModel:
    """Named classes and properties of one universe plus the core.

    `ancestors` maps every named class to its strict transitive superclasses
    among named classes.
    """

    universe: str
    classes: frozenset[str]
    ancestors: Mapping[str, frozenset[str]]
    object_properties: frozenset[str]
    datatype_properties: frozenset[str]


def extract_local_name(iri: str) -> str:
    """Return the part after `#`, or after the last `/` when there is no `#`."""
    if FRAGMENT_SEPARATOR in iri:
        return iri.rsplit(FRAGMENT_SEPARATOR, 1)[1]
    return iri.rsplit(PATH_SEPARATOR, 1)[-1]


def build_ontology_model(tbox: Graph, universe: str) -> OntologyModel:
    """Build the model from a parsed T-Box. Blank-node classes are skipped."""
    classes = read_named_subjects(tbox, OWL.Class)
    return OntologyModel(
        universe=universe,
        classes=classes,
        ancestors=compute_ancestors(tbox, classes),
        object_properties=read_named_subjects(tbox, OWL.ObjectProperty),
        datatype_properties=read_named_subjects(tbox, OWL.DatatypeProperty),
    )


def read_named_subjects(tbox: Graph, declared_type: URIRef) -> frozenset[str]:
    subjects = tbox.subjects(RDF.type, declared_type)
    return frozenset(str(subject) for subject in subjects if isinstance(subject, URIRef))


def compute_ancestors(tbox: Graph, classes: frozenset[str]) -> dict[str, frozenset[str]]:
    """Walk `rdfs:subClassOf` upwards from each class. Cycles end the walk."""
    parents = {
        class_iri: frozenset(
            str(parent)
            for parent in tbox.objects(URIRef(class_iri), RDFS.subClassOf)
            if isinstance(parent, URIRef) and str(parent) in classes
        )
        for class_iri in classes
    }
    ancestors: dict[str, frozenset[str]] = {}
    for class_iri in classes:
        seen: set[str] = set()
        pending = list(parents[class_iri])
        while pending:
            parent = pending.pop()
            if parent in seen:
                continue
            seen.add(parent)
            pending.extend(parents[parent])
        seen.discard(class_iri)
        ancestors[class_iri] = frozenset(seen)
    return ancestors


def load_ontology_model(ontology_dir: Path, universe: str) -> OntologyModel:
    """Parse `core/*.ttl` and `universes/<universe>.ttl` under `ontology_dir`."""
    if not UNIVERSE_PREFIX_PATTERN.fullmatch(universe):
        raise OntologyError(
            f"Invalid universe prefix {universe!r}: expected {UNIVERSE_PREFIX_PATTERN.pattern}"
        )
    core_paths = sorted((ontology_dir / CORE_DIR_NAME).glob(f"*{TTL_SUFFIX}"))
    if not core_paths:
        raise OntologyError(f"No core ttl files in {ontology_dir / CORE_DIR_NAME}")
    universe_path = ontology_dir / UNIVERSES_DIR_NAME / f"{universe}{TTL_SUFFIX}"
    if not universe_path.is_file():
        raise OntologyError(f"Missing universe T-Box {universe_path}")

    tbox = Graph()
    for path in [*core_paths, universe_path]:
        try:
            tbox.parse(path, format=TURTLE_FORMAT)
        except BadSyntax as error:
            raise OntologyError(f"Failed to parse {path}: {error}") from error
    return build_ontology_model(tbox, universe)
