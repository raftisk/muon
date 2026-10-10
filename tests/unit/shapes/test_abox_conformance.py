from pyshacl import validate
from rdflib import RDF, RDFS, Graph, Literal

from muon.ingestion.batch import build_node_id
from tests.unit.shapes.conftest import ONTOLOGY_DIR, TURTLE_FORMAT
from tests.unit.shapes.vocabulary import PKMN

ABOX_PATH = ONTOLOGY_DIR / "individuals" / "pkmn.ttl"


def add_node_keys(graph: Graph) -> Graph:
    """Return a copy of `graph` where each labelled individual carries its `pkmn:id`."""
    keyed = Graph()
    for triple in graph:
        keyed.add(triple)
    for subject, label in graph.subject_objects(RDFS.label):
        if (subject, RDF.type, None) in graph:
            keyed.add((subject, PKMN.id, Literal(build_node_id(str(label)))))
    return keyed


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
