"""Toy universes for seed loader tests.

The fixture universes assert only `Toy*` classes, so dropping their constraints
never touches a constraint of a real universe.
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_CORE_DIR = PROJECT_ROOT / "ontology" / "core"
FIXTURE_ONTOLOGY_DIR = Path(__file__).resolve().parent / "ontology"
FIXTURE_SUBDIRS = ("universes", "individuals")
TOY_UNIVERSE = "muontoy"
ALT_UNIVERSE = "muonalt"


def copy_toy_ontology(target_dir: Path) -> Path:
    """Build an ontology dir from the real core and every fixture universe."""
    shutil.copytree(REAL_CORE_DIR, target_dir / "core")
    for subdir in FIXTURE_SUBDIRS:
        shutil.copytree(FIXTURE_ONTOLOGY_DIR / subdir, target_dir / subdir)
    return target_dir
