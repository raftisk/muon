"""A complete Move as a Turtle snippet, for shape tests that need one.

The move points at `ex:gen1`, so the caller also includes `GENERATION_NODE` from
`games_fixtures.py`, and the type and damage class nodes it names. The snippet uses the
prefixes of `validate_fixture`.
"""

MOVE_PP = 35
MOVE_PRIORITY = 0
MOVE_TARGET = "any_adjacent"


def build_move(
    local_name: str,
    *,
    type_local_name: str | None = "fire",
    damage_class_local_name: str | None = "special",
    extra: str = "",
) -> str:
    """A Move with id `local_name` and every value the move shapes require.

    `None` leaves out the type or damage class edge. `extra` holds more predicate-object
    pairs, separated by ` ; `, appended before the closing dot.
    """
    pairs = [
        "a pkmn:Move",
        f'pkmn:id "{local_name}"',
        f"pkmn:pp {MOVE_PP}",
        f"pkmn:priority {MOVE_PRIORITY}",
        f'pkmn:target "{MOVE_TARGET}"',
        "pkmn:introducedIn ex:gen1",
    ]
    if type_local_name is not None:
        pairs.append(f"pkmn:hasType ex:{type_local_name}")
    if damage_class_local_name is not None:
        pairs.append(f"pkmn:hasDamageClass ex:{damage_class_local_name}")
    if extra:
        pairs.append(extra)
    return f"ex:{local_name} {' ; '.join(pairs)} ."
