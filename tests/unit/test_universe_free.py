"""The agent and retrieval layers name no universe (spec scenario 6.1)."""

import re
from pathlib import Path

from rdflib import Graph
from rdflib.namespace import OWL, RDF

from muon.config import PROJECT_ROOT
from muon.ontology.model import extract_local_name

UNIVERSE_PREFIX = "pkmn"
UNIVERSE_NAMESPACE = f"https://muon.dev/ns/{UNIVERSE_PREFIX}#"
UNIVERSE_TBOX = PROJECT_ROOT / "ontology" / "universes" / f"{UNIVERSE_PREFIX}.ttl"
UNIVERSE_FREE_PACKAGES = ("agent", "retrieval")


def list_universe_class_names() -> list[str]:
    tbox = Graph().parse(UNIVERSE_TBOX, format="turtle")
    return sorted(
        extract_local_name(str(subject))
        for subject in tbox.subjects(RDF.type, OWL.Class)
        if str(subject).startswith(UNIVERSE_NAMESPACE)
    )


def find_violations(source_path: Path, patterns: dict[str, re.Pattern[str]]) -> list[str]:
    violations = []
    for line_number, line in enumerate(source_path.read_text().splitlines(), start=1):
        for name, pattern in patterns.items():
            if pattern.search(line):
                relative_path = source_path.relative_to(PROJECT_ROOT)
                violations.append(f"{relative_path}:{line_number} names {name!r}")
    return violations


def test_agent_and_retrieval_name_no_universe_or_universe_class() -> None:
    names = [UNIVERSE_PREFIX, *list_universe_class_names()]
    patterns = {name: re.compile(rf"\b{re.escape(name)}\b") for name in names}
    source_paths = [
        path
        for package in UNIVERSE_FREE_PACKAGES
        for path in sorted((PROJECT_ROOT / "muon" / package).rglob("*.py"))
    ]

    violations = [
        violation for path in source_paths for violation in find_violations(path, patterns)
    ]

    assert source_paths
    assert violations == []
