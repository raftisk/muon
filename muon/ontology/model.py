"""Read-only model of one universe's T-Box merged with the core T-Box.

The ontology directory holds `core/*.ttl` and `universes/<prefix>.ttl`. Every IRI
in the model is a plain `str`: rdflib terms hash and compare by type, so mixing
`URIRef` and `str` breaks set lookups.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from rdflib import BNode, Graph, URIRef
from rdflib.collection import Collection
from rdflib.namespace import OWL, RDF, RDFS, XSD
from rdflib.plugins.parsers.notation3 import BadSyntax
from rdflib.term import Node

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
    among named classes. `domains` and `ranges` map an object property to the classes
    its `rdfs:domain` and `rdfs:range` allow, with `owl:unionOf` lists expanded and a
    property that declares none taking the nearest entry up its `rdfs:subPropertyOf`
    chain; a property with no entry is unconstrained on that end. `datatype_ranges` maps
    a datatype property to its `xsd:` range.
    """

    universe: str
    classes: frozenset[str]
    ancestors: Mapping[str, frozenset[str]]
    object_properties: frozenset[str]
    datatype_properties: frozenset[str]
    domains: Mapping[str, frozenset[str]]
    ranges: Mapping[str, frozenset[str]]
    datatype_ranges: Mapping[str, str]


def extract_local_name(iri: str) -> str:
    """Return the part after `#`, or after the last `/` when there is no `#`."""
    if FRAGMENT_SEPARATOR in iri:
        return iri.rsplit(FRAGMENT_SEPARATOR, 1)[1]
    return iri.rsplit(PATH_SEPARATOR, 1)[-1]


def build_ontology_model(tbox: Graph, universe: str) -> OntologyModel:
    """Build the model from a parsed T-Box. Blank-node classes are skipped."""
    classes = read_named_subjects(tbox, OWL.Class)
    object_properties = read_named_subjects(tbox, OWL.ObjectProperty)
    datatype_properties = read_named_subjects(tbox, OWL.DatatypeProperty)
    return OntologyModel(
        universe=universe,
        classes=classes,
        ancestors=compute_ancestors(tbox, classes),
        object_properties=object_properties,
        datatype_properties=datatype_properties,
        domains=inherit_property_classes(
            tbox, read_property_classes(tbox, RDFS.domain, object_properties), object_properties
        ),
        ranges=inherit_property_classes(
            tbox, read_property_classes(tbox, RDFS.range, object_properties), object_properties
        ),
        datatype_ranges=read_datatype_ranges(tbox, datatype_properties),
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


def read_union_members(tbox: Graph, node: Node) -> frozenset[str]:
    """Return the class IRIs of a named class or of an `owl:unionOf` list, nested ones included."""
    if isinstance(node, URIRef):
        return frozenset({str(node)})
    if not isinstance(node, BNode):
        return frozenset()
    members: set[str] = set()
    for union_list in tbox.objects(node, OWL.unionOf):
        for member in Collection(tbox, union_list):
            members |= read_union_members(tbox, member)
    return frozenset(members)


def read_property_classes(
    tbox: Graph, predicate: URIRef, properties: frozenset[str]
) -> dict[str, frozenset[str]]:
    """Read `rdfs:domain` or `rdfs:range` per property; a property with no value is absent."""
    classes_by_property: dict[str, frozenset[str]] = {}
    for property_iri in properties:
        members: frozenset[str] = frozenset()
        for node in tbox.objects(URIRef(property_iri), predicate):
            members |= read_union_members(tbox, node)
        if members:
            classes_by_property[property_iri] = members
    return classes_by_property


def inherit_property_classes(
    tbox: Graph, own: Mapping[str, frozenset[str]], properties: frozenset[str]
) -> dict[str, frozenset[str]]:
    """Give a property with no own entry the nearest entry up its `rdfs:subPropertyOf` chain.

    The walk goes one level at a time and stops at the first level with an entry, so a
    nearer super-property wins. Cycles end the walk.
    """
    resolved = dict(own)
    for property_iri in sorted(properties - own.keys()):
        seen = {property_iri}
        level = {property_iri}
        while level:
            parents = {
                str(parent)
                for child in level
                for parent in tbox.objects(URIRef(child), RDFS.subPropertyOf)
                if isinstance(parent, URIRef)
            } - seen
            inherited = frozenset().union(*(own[parent] for parent in parents if parent in own))
            if inherited:
                resolved[property_iri] = inherited
                break
            seen |= parents
            level = parents
    return resolved


def read_datatype_ranges(tbox: Graph, datatype_properties: frozenset[str]) -> dict[str, str]:
    """Keep the `xsd:` range of each datatype property; other ranges are skipped."""
    xsd_namespace = str(XSD)
    ranges: dict[str, str] = {}
    for property_iri in datatype_properties:
        xsd_ranges = sorted(
            str(range_iri)
            for range_iri in tbox.objects(URIRef(property_iri), RDFS.range)
            if isinstance(range_iri, URIRef) and str(range_iri).startswith(xsd_namespace)
        )
        if xsd_ranges:
            ranges[property_iri] = xsd_ranges[0]
    return ranges


def list_ontology_paths(ontology_dir: Path, universe: str) -> tuple[Path, ...]:
    """Return the sorted `core/*.ttl` paths followed by `universes/<universe>.ttl`."""
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
    return (*core_paths, universe_path)


def load_tbox(ontology_dir: Path, universe: str) -> Graph:
    """Parse `core/*.ttl` and `universes/<universe>.ttl` under `ontology_dir` into 1 graph."""
    tbox = Graph()
    for path in list_ontology_paths(ontology_dir, universe):
        try:
            tbox.parse(path, format=TURTLE_FORMAT)
        except BadSyntax as error:
            raise OntologyError(f"Failed to parse {path}: {error}") from error
    return tbox


def load_ontology_model(ontology_dir: Path, universe: str) -> OntologyModel:
    """Build the model from `load_tbox`."""
    return build_ontology_model(load_tbox(ontology_dir, universe), universe)
