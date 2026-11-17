"""The sweep: checks that span batches, run on the live graph. Read-only.

Every query goes through `execute_read`, so the sweep reports and never changes the graph.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, LiteralString

from muon.config import PROJECT_ROOT
from muon.files import write_text_atomically
from muon.graph import GraphClient
from muon.ingestion.batch import GraphBatch, NodeRef
from muon.ingestion.runner import STUB_KEYS
from muon.ingestion.seed import find_missing_nodes
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import OntologyModel
from muon.validation.report import EdgeTypingFinding, LabelSetFinding, SweepReport

SWEEP_OUTPUT_ROOT = PROJECT_ROOT / "data" / "validation"
SWEEP_REPORT_SUFFIX = "-sweep-report.json"
SWEEP_REPORT_INDENT = 2
SAMPLE_SIZE = 5

LIST_LABEL_SETS: LiteralString = """
MATCH (n:$($universe))
WITH labels(n) AS labels, count(*) AS count, collect(n.id)[..$sample_size] AS sample_ids
RETURN labels, count, sample_ids
"""

# Counts relationships whose start or end carries none of the allowed labels. An empty
# label list leaves that end unchecked.
FIND_EDGE_TYPING_FAULTS: LiteralString = """
MATCH (a:$($universe))-[r:$($type)]->(b:$($universe))
WITH a, b,
     (size($domain_labels) = 0
      OR any(label IN labels(a) WHERE label IN $domain_labels)) AS start_ok,
     (size($range_labels) = 0
      OR any(label IN labels(b) WHERE label IN $range_labels)) AS end_ok
RETURN count(CASE WHEN NOT start_ok THEN 1 END) AS start_count,
       collect(CASE WHEN NOT start_ok THEN a.id END)[..$sample_size] AS start_ids,
       count(CASE WHEN NOT end_ok THEN 1 END) AS end_count,
       collect(CASE WHEN NOT end_ok THEN b.id END)[..$sample_size] AS end_ids
"""

LIST_STUBS: LiteralString = """
MATCH (n:$($universe))
WHERE all(key IN keys(n) WHERE key IN $stub_keys)
RETURN labels(n) AS labels, n.id AS id
"""

LIST_ORPHAN_STUBS: LiteralString = """
MATCH (n:$($universe))
WHERE all(key IN keys(n) WHERE key IN $stub_keys) AND NOT (n)--()
RETURN labels(n) AS labels, n.id AS id
"""


async def run_sweep(
    client: GraphClient,
    materializer: LabelMaterializer,
    model: OntologyModel,
    abox: GraphBatch,
) -> SweepReport:
    """Run the 5 sweep checks on the universe of `materializer`."""
    abox_keys = frozenset(
        (materializer.map_class_label(node.class_iri), node.id) for node in abox.nodes
    )
    stubs, orphan_stubs = await find_stub_findings(client, materializer, model, abox_keys)
    return SweepReport(
        label_sets=await find_label_set_findings(client, materializer, model),
        edge_typing=await find_edge_typing_findings(client, materializer, model),
        stubs=stubs,
        orphan_stubs=orphan_stubs,
        missing_abox=tuple(await find_missing_nodes(client, materializer, abox.nodes)),
    )


def build_label_sets(
    materializer: LabelMaterializer, model: OntologyModel
) -> dict[frozenset[str], str]:
    """Map the ontology labels a node of each class carries to the class IRI."""
    universe_label = materializer.universe_label
    return {
        frozenset(materializer.map_class_labels(class_iri)) - {universe_label}: class_iri
        for class_iri in model.classes
    }


def read_ontology_labels(labels: Iterable[str], materializer: LabelMaterializer) -> frozenset[str]:
    """Keep the labels that name an ontology class; the universe label and others drop out."""
    return frozenset(labels) & materializer.list_ontology_labels()


async def find_label_set_findings(
    client: GraphClient, materializer: LabelMaterializer, model: OntologyModel
) -> tuple[LabelSetFinding, ...]:
    """L1: list the ontology label sets that match no class, with counts and sample ids."""
    expected = build_label_sets(materializer, model)
    records = await client.execute_read(
        LIST_LABEL_SETS, {"universe": materializer.universe_label, "sample_size": SAMPLE_SIZE}
    )
    counts: dict[frozenset[str], int] = defaultdict(int)
    samples: dict[frozenset[str], list[str]] = defaultdict(list)
    for record in records:
        label_set = read_ontology_labels(record["labels"], materializer)
        counts[label_set] += record["count"]
        samples[label_set].extend(record["sample_ids"])
    return tuple(
        LabelSetFinding(
            labels=tuple(sorted(label_set)),
            count=counts[label_set],
            sample_ids=tuple(sorted(samples[label_set])[:SAMPLE_SIZE]),
        )
        for label_set in sorted(counts, key=sorted)
        if label_set not in expected
    )


async def find_edge_typing_findings(
    client: GraphClient, materializer: LabelMaterializer, model: OntologyModel
) -> tuple[EdgeTypingFinding, ...]:
    """T1: per object property with a domain or range, count the relationships that break it."""
    findings: list[EdgeTypingFinding] = []
    for property_iri in sorted(model.domains.keys() | model.ranges.keys()):
        parameters = {
            "universe": materializer.universe_label,
            "type": materializer.map_relationship_type(property_iri),
            "domain_labels": list_class_labels(model.domains.get(property_iri, ()), materializer),
            "range_labels": list_class_labels(model.ranges.get(property_iri, ()), materializer),
            "sample_size": SAMPLE_SIZE,
        }
        (record,) = await client.execute_read(FIND_EDGE_TYPING_FAULTS, parameters)
        relationship_type = str(parameters["type"])
        if record["start_count"]:
            findings.append(
                EdgeTypingFinding(
                    relationship_type=relationship_type,
                    end="start",
                    count=record["start_count"],
                    sample_ids=tuple(record["start_ids"]),
                )
            )
        if record["end_count"]:
            findings.append(
                EdgeTypingFinding(
                    relationship_type=relationship_type,
                    end="end",
                    count=record["end_count"],
                    sample_ids=tuple(record["end_ids"]),
                )
            )
    return tuple(findings)


def list_class_labels(class_iris: Iterable[str], materializer: LabelMaterializer) -> list[str]:
    return sorted(materializer.map_class_label(class_iri) for class_iri in class_iris)


async def find_stub_findings(
    client: GraphClient,
    materializer: LabelMaterializer,
    model: OntologyModel,
    abox_keys: frozenset[tuple[str, str]],
) -> tuple[Mapping[str, tuple[str, ...]], tuple[NodeRef, ...]]:
    """List the stubs per class label and the stubs with no relationship.

    A-Box nodes hold only the stub keys too, so `abox_keys` (class label, id) leaves them out.
    """
    class_by_label_set = build_label_sets(materializer, model)
    parameters = {"universe": materializer.universe_label, "stub_keys": list(STUB_KEYS)}
    stub_records = await client.execute_read(LIST_STUBS, parameters)
    orphan_records = await client.execute_read(LIST_ORPHAN_STUBS, parameters)

    stub_ids: dict[str, list[str]] = defaultdict(list)
    for class_iri, node_id in read_stub_keys(stub_records, materializer, class_by_label_set):
        class_label = materializer.map_class_label(class_iri)
        if (class_label, node_id) not in abox_keys:
            stub_ids[class_label].append(node_id)
    orphans = [
        NodeRef(class_iri=class_iri, id=node_id)
        for class_iri, node_id in read_stub_keys(orphan_records, materializer, class_by_label_set)
        if (materializer.map_class_label(class_iri), node_id) not in abox_keys
    ]
    return (
        {label: tuple(sorted(ids)) for label, ids in sorted(stub_ids.items())},
        tuple(sorted(orphans, key=lambda node_ref: (node_ref.class_iri, node_ref.id))),
    )


def read_stub_keys(
    records: Iterable[Mapping[str, Any]],
    materializer: LabelMaterializer,
    class_by_label_set: Mapping[frozenset[str], str],
) -> list[tuple[str, str]]:
    """Return (class IRI, id) per record; a label set with no class gives its joined labels."""
    keys: list[tuple[str, str]] = []
    for record in records:
        label_set = read_ontology_labels(record["labels"], materializer)
        class_iri = class_by_label_set.get(label_set, ",".join(sorted(label_set)))
        keys.append((class_iri, str(record["id"])))
    return keys


def write_sweep_report(report: SweepReport, universe: str, root: Path = SWEEP_OUTPUT_ROOT) -> Path:
    """Write the report under `root`, atomically, and return its path."""
    path = root / f"{universe}{SWEEP_REPORT_SUFFIX}"
    write_text_atomically(path, report.model_dump_json(indent=SWEEP_REPORT_INDENT) + "\n")
    return path
