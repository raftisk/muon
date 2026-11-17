"""Gate step 5: the stubs of a strict batch must already be in the graph."""

from muon.graph import GraphClient
from muon.ingestion.batch import GraphBatch
from muon.ingestion.seed import find_missing_nodes
from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import extract_local_name
from muon.validation.report import RULE_REFERENCE, Violation


async def find_unresolved_stubs(
    client: GraphClient, materializer: LabelMaterializer, batch: GraphBatch
) -> tuple[Violation, ...]:
    """Return 1 violation per stub that matches no node of its class in the graph.

    A batch with `can_create_stubs` set, or with no stubs, returns `()` and issues no query.
    """
    if batch.can_create_stubs or not batch.stub_nodes:
        return ()
    missing = await find_missing_nodes(client, materializer, batch.stub_nodes)
    return tuple(
        Violation(
            rule_id=RULE_REFERENCE,
            node=node_ref,
            value=node_ref.id,
            message=f"Stub {extract_local_name(node_ref.class_iri)} {node_ref.id!r} "
            f"is not in the graph and the batch cannot create stubs",
        )
        for node_ref in missing
    )
