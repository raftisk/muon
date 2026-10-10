import pytest

from muon.ingestion.bulbapedia.api import BulbapediaClient, build_http_client

pytestmark = pytest.mark.network

DIGLETT = "Diglett (Pokémon)"


def test_live_api_serves_a_species_page_and_category() -> None:
    with build_http_client() as http_client:
        client = BulbapediaClient(http_client)

        lookup = client.fetch_content_by_titles([DIGLETT])
        abilities = client.list_category_members("Category:Abilities")

    assert [revision.title for revision in lookup.revisions] == [DIGLETT]
    assert "{{Pokémon Infobox" in lookup.revisions[0].wikitext
    assert "Natural Cure (Ability)" in abilities
