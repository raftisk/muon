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


def add_alt_universe_t_box(ontology_dir: Path) -> None:
    """Write a T-Box for `ALT_UNIVERSE` into `ontology_dir`, a renamed copy of the toy one."""
    toy_t_box = (ontology_dir / "universes" / f"{TOY_UNIVERSE}.ttl").read_text(encoding="utf-8")
    alt_t_box = toy_t_box.replace(TOY_UNIVERSE, ALT_UNIVERSE).replace("Toy", "Alt")
    (ontology_dir / "universes" / f"{ALT_UNIVERSE}.ttl").write_text(alt_t_box, encoding="utf-8")
