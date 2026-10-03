from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, SH


def list_top_level_shapes(shapes: Graph) -> set:
    """Node shapes plus the property shapes they declare directly.

    Shapes nested inside `sh:or` branches never appear in a report, so they carry no rule id.
    """
    node_shapes = set(shapes.subjects(RDF.type, SH.NodeShape))
    property_shapes = {
        property_shape
        for node_shape in node_shapes
        for property_shape in shapes.objects(node_shape, SH.property)
    }
    return node_shapes | property_shapes


def test_shapes_file_parses(shapes_graph):
    assert len(shapes_graph) > 0


def test_empty_data_conforms(validate_fixture):
    assert validate_fixture("").conforms


def test_every_shape_has_rule_id_and_message(shapes_graph):
    for shape in list_top_level_shapes(shapes_graph):
        assert (shape, SH.name, None) in shapes_graph, f"shape {shape} has no sh:name"
        assert (shape, SH.message, None) in shapes_graph, f"shape {shape} has no sh:message"
        severities = set(shapes_graph.objects(shape, SH.severity))
        assert severities == {SH.Violation}, f"shape {shape} severity is {severities}"


ALWAYS = URIRef("https://muon.dev/ns/pkmn-shapes#Always")
SCOPE = URIRef("https://muon.dev/ns/pkmn-shapes#scope")


def test_every_shape_has_a_scope(shapes_graph):
    for shape in shapes_graph.subjects(RDF.type, SH.NodeShape):
        assert (shape, SCOPE, None) in shapes_graph, f"shape {shape} has no pkmn-shape:scope"


def test_every_scope_is_always_or_a_tbox_class(shapes_graph, tbox_graph):
    for shape, scope in shapes_graph.subject_objects(SCOPE):
        if scope == ALWAYS:
            continue
        assert (scope, RDF.type, OWL.Class) in tbox_graph, f"{shape} scopes unknown class {scope}"


def test_scope_matches_target_class(shapes_graph):
    for shape in shapes_graph.subjects(RDF.type, SH.NodeShape):
        target_classes = set(shapes_graph.objects(shape, SH.targetClass))
        scopes = set(shapes_graph.objects(shape, SCOPE))
        if target_classes and scopes != {ALWAYS}:
            assert scopes == target_classes, f"{shape} scope {scopes} differs from {target_classes}"


def test_always_shapes_stand_alone(shapes_graph):
    for shape in shapes_graph.subjects(SCOPE, ALWAYS):
        assert set(shapes_graph.objects(shape, SCOPE)) == {ALWAYS}, f"{shape} mixes Always"
