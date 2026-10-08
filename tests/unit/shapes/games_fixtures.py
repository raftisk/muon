"""Turtle snippets for the chain Version -> VersionGroup -> Generation.

Shape tests that need a version or a version group build it here, so each one carries the
`partOf` edges the games shapes require. The snippets use the prefixes of `validate_fixture`.
"""

GENERATION_NODE = 'ex:gen1 a pkmn:Generation ; pkmn:id "gen1" .'


def build_version_group(local_name: str, node_id: str) -> str:
    return f'ex:{local_name} a pkmn:VersionGroup ; pkmn:id "{node_id}" ; mo:partOf ex:gen1 .'


def build_version(local_name: str, node_id: str, group_local_name: str) -> str:
    return (
        f'ex:{local_name} a pkmn:Version ; pkmn:id "{node_id}" ; mo:partOf ex:{group_local_name} .'
    )
