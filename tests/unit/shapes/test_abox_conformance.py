import re
import unicodedata

from pyshacl import validate
from rdflib import RDF, RDFS, Graph, Literal

from tests.unit.shapes.conftest import ONTOLOGY_DIR, TURTLE_FORMAT
from tests.unit.shapes.vocabulary import PKMN

ABOX_PATH = ONTOLOGY_DIR / "individuals" / "pkmn.ttl"
NON_SLUG_RUN = re.compile(r"[^a-z0-9]+")


def slugify(label: str) -> str:
    """Lowercase hyphenated slug of a label, the way the loader builds a node key."""
    ascii_label = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode()
    return NON_SLUG_RUN.sub("-", ascii_label.lower()).strip("-")


def add_node_keys(graph: Graph) -> Graph:
    """Return a copy of `graph` where each labelled individual carries its `pkmn:id`."""
    keyed = Graph()
    for triple in graph:
        keyed.add(triple)
    for subject, label in graph.subject_objects(RDFS.label):
        if (subject, RDF.type, None) in graph:
            keyed.add((subject, PKMN.id, Literal(slugify(str(label)))))
    return keyed


def test_slugify_strips_accents_and_punctuation():
    assert slugify("Pokémon") == "pokemon"
    assert slugify("Critical Hit Ratio") == "critical-hit-ratio"
    assert slugify("Mr. Mime") == "mr-mime"


def test_abox_conforms_to_the_shapes(shapes_graph, tbox_graph):
    abox = add_node_keys(Graph().parse(ABOX_PATH, format=TURTLE_FORMAT))
    conforms, _, report_text = validate(
        abox,
        shacl_graph=shapes_graph,
        ont_graph=tbox_graph,
        advanced=True,
        inference="none",
    )
    assert conforms, report_text
