"""Synthetic species pages and a page store for the run and ingest tests.

The pages use the real pkmn A-Box, so a batch built from them compiles and ingests like a
pinned page. `ALPHA` evolves into `BETA`, `GAMMA` fails the A-Box check and `DELTA` has no
infobox.
"""

from datetime import UTC, datetime
from pathlib import Path

from muon.ingestion.bulbapedia.pages import Manifest, Page, PageStore, compute_sha1
from muon.ingestion.bulbapedia.species.run import (
    ParseRequest,
    ParseRun,
    RunInputs,
    parse_species_pages,
)

ONTOLOGY_DIR = Path(__file__).resolve().parents[5] / "ontology"
PAGE_TYPE = "species"
TIMESTAMP = datetime(2026, 10, 10, tzinfo=UTC)
ALPHA = "Alpha (Pokémon)"
BETA = "Beta (Pokémon)"
GAMMA = "Gamma (Pokémon)"
DELTA = "Delta (Pokémon)"
PAGE_IDS = {ALPHA: 101, BETA: 102, GAMMA: 103, DELTA: 104}
LEVEL_16 = "{{Bag/s|Rare Candy|SV}}<br>{{color2|000|Level|Level 16}}"
STATS = "{{Stats|HP=10|Attack=55|Defense=25|SpAtk=35|SpDef=45|Speed=95}}"


def species_wikitext(
    name: str, ndex: int, evolution: str = "", egg_group: str = "Field", extra: str = ""
) -> str:
    return f"""{{{{Pokémon Infobox
|name={name}
|ndex={ndex:04d}
|type1=Ground
|egggroup1={egg_group}
|eggcycles=20
|catchrate=255
|expyield=53
|lv100exp=1,000,000
|gendercode=127
|friendship=70
|color=Brown
|body=05
|generation=1
|height-m=0.2
|weight-kg=0.8
|category=Mole
|evsp=1
}}}}
==Game data==
===Stats===
====Base stats====
{STATS}
===Learnset===
====By leveling up====
{{{{learnlist/level9|1|Sand Attack|Ground|Status|—|100|15}}}}
===Evolution data===
{evolution}
{extra}
"""


def evolution_box(first: str, second: str, method: str = LEVEL_16) -> str:
    return f"{{{{Evobox-2|no1=0001|name1={first}|evo1={method}|no2=0002|name2={second}}}}}"


def build_page(title: str, wikitext: str) -> Page:
    return Page(
        title=title,
        page_id=PAGE_IDS[title],
        revision_id=PAGE_IDS[title] * 10,
        timestamp=TIMESTAMP,
        sha1=compute_sha1(wikitext),
        page_type=PAGE_TYPE,
        wikitext=wikitext,
    )


def build_store(root: Path, wikitexts: dict[str, str]) -> PageStore:
    store = PageStore(root)
    pages = [build_page(title, wikitext) for title, wikitext in wikitexts.items()]
    for page in pages:
        store.write_page(page)
    store.write_manifest(Manifest().replace_pins(PAGE_TYPE, [page.pin for page in pages]))
    return store


def default_wikitexts() -> dict[str, str]:
    return {
        ALPHA: species_wikitext(
            "Alpha",
            1,
            evolution_box("Alpha", "Beta") + "\n{{Evobox/Setup|1Alpha|Alpha}}",
        ),
        BETA: species_wikitext("Beta", 2, evolution_box("Alpha", "Beta")),
        GAMMA: species_wikitext("Gamma", 3, egg_group="Imaginary"),
        DELTA: "{{New Pokémon Infobox|name=Delta}}\n==Game data==\n",
    }


def run_parse(store: PageStore, *titles: str) -> ParseRun:
    return parse_species_pages(RunInputs(ParseRequest(titles), store, ONTOLOGY_DIR))
