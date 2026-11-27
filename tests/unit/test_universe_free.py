"""The agent, retrieval, generation, evaluation, validation and write gate code name no universe."""

import re
from pathlib import Path

from rdflib import Graph
from rdflib.namespace import OWL, RDF

from muon.config import PROJECT_ROOT
from muon.ontology.model import extract_local_name

UNIVERSE_PREFIX = "pkmn"
UNIVERSE_NAMESPACE = f"https://muon.dev/ns/{UNIVERSE_PREFIX}#"
UNIVERSE_TBOX = PROJECT_ROOT / "ontology" / "universes" / f"{UNIVERSE_PREFIX}.ttl"
UNIVERSE_FREE_PACKAGES = ("agent", "retrieval", "generation", "evaluation", "validation", "api")
UNIVERSE_FREE_MODULES = ("plugins.py", "ingestion/policy.py")
SOURCE_PATTERNS = ("*.py", "prompts/*.md")
READ_ONLY_PACKAGES = ("retrieval", "generation", "evaluation")
WRITE_METHOD_PATTERN = re.compile(r"\bexecute_write(_batch)?\b")


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


def list_source_paths(packages: tuple[str, ...], modules: tuple[str, ...] = ()) -> list[Path]:
    package_paths = [
        path
        for package in packages
        for pattern in SOURCE_PATTERNS
        for path in sorted((PROJECT_ROOT / "muon" / package).rglob(pattern))
    ]
    return [*package_paths, *(PROJECT_ROOT / "muon" / module for module in modules)]


def test_engine_packages_name_no_universe_or_universe_class() -> None:
    names = [UNIVERSE_PREFIX, *list_universe_class_names()]
    patterns = {name: re.compile(rf"\b{re.escape(name)}\b") for name in names}
    source_paths = list_source_paths(UNIVERSE_FREE_PACKAGES, UNIVERSE_FREE_MODULES)

    violations = [
        violation for path in source_paths for violation in find_violations(path, patterns)
    ]

    assert source_paths
    assert violations == []


def test_retrieval_and_generation_never_call_a_write_method() -> None:
    source_paths = list_source_paths(READ_ONLY_PACKAGES)

    violations = [
        violation
        for path in source_paths
        for violation in find_violations(path, {"a write method": WRITE_METHOD_PATTERN})
    ]

    assert source_paths
    assert violations == []


def test_api_imports_no_rich() -> None:
    pattern = re.compile(r"^\s*(import|from)\s+rich\b")
    source_paths = list_source_paths(("api",))

    violations = [
        violation for path in source_paths for violation in find_violations(path, {"rich": pattern})
    ]

    assert source_paths
    assert violations == []
