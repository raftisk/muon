from rdflib import Graph, URIRef
from rdflib.namespace import RDFS, SH, SKOS

from tests.unit.shapes.vocabulary import PKMN

# `role` is an edge property with no declaration of its own; the ttl lists its values in the
# comment of the `requires` property it qualifies.
DOCUMENTING_PROPERTY = {PKMN.role: PKMN.requires}


def read_documentation(tbox: Graph, property_iri: URIRef) -> str:
    texts = [
        *tbox.objects(property_iri, SKOS.definition),
        *tbox.objects(property_iri, RDFS.comment),
    ]
    return " ".join(str(text) for text in texts)


def list_vocabularies(shapes: Graph) -> list[tuple[URIRef, list[str]]]:
    vocabularies = []
    for shape, value_list in shapes.subject_objects(SH["in"]):
        path = shapes.value(shape, SH.path)
        if isinstance(path, URIRef):
            vocabularies.append((path, [str(value) for value in shapes.items(value_list)]))
    return vocabularies


def test_shapes_define_vocabularies(shapes_graph):
    assert list_vocabularies(shapes_graph)


def test_every_sh_in_value_appears_in_the_ttl(shapes_graph, tbox_graph):
    for path, values in list_vocabularies(shapes_graph):
        documentation = read_documentation(tbox_graph, DOCUMENTING_PROPERTY.get(path, path))
        for value in values:
            assert value in documentation, f"{value!r} for {path} is not in its ttl documentation"
