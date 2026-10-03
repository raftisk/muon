from rdflib import Graph
from rdflib.namespace import RDF, SH


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
