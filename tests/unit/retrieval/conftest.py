from pathlib import Path

import pytest

from muon.ontology.materializer import LabelMaterializer
from muon.ontology.model import load_ontology_model
from tests.fixtures.toy_ontology import TOY_UNIVERSE, copy_toy_ontology


@pytest.fixture(scope="module")
def toy_ontology_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return copy_toy_ontology(tmp_path_factory.mktemp("toy-ontology"))


@pytest.fixture(scope="module")
def toy_materializer(toy_ontology_dir: Path) -> LabelMaterializer:
    return LabelMaterializer(load_ontology_model(toy_ontology_dir, TOY_UNIVERSE))
