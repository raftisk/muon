from pathlib import Path

import pytest

from muon.ontology.model import OntologyModel, load_ontology_model

REAL_ONTOLOGY_DIR = Path(__file__).resolve().parents[3] / "ontology"
REAL_UNIVERSE = "pkmn"


@pytest.fixture(scope="session")
def real_model() -> OntologyModel:
    return load_ontology_model(REAL_ONTOLOGY_DIR, REAL_UNIVERSE)
