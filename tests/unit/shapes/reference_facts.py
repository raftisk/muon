"""Expected reference facts, copied from the pkmn reference facts spec tables."""

EFFECTIVENESS_PROPERTIES = ("superEffectiveAgainst", "notVeryEffectiveAgainst", "noEffectAgainst")

TYPE_CHART: dict[str, dict[str, tuple[str, ...]]] = {
    "Normal": {
        "superEffectiveAgainst": (),
        "notVeryEffectiveAgainst": ("Rock", "Steel"),
        "noEffectAgainst": ("Ghost",),
    },
    "Fire": {
        "superEffectiveAgainst": ("Grass", "Ice", "Bug", "Steel"),
        "notVeryEffectiveAgainst": ("Fire", "Water", "Rock", "Dragon"),
        "noEffectAgainst": (),
    },
    "Water": {
        "superEffectiveAgainst": ("Fire", "Ground", "Rock"),
        "notVeryEffectiveAgainst": ("Water", "Grass", "Dragon"),
        "noEffectAgainst": (),
    },
    "Electric": {
        "superEffectiveAgainst": ("Water", "Flying"),
        "notVeryEffectiveAgainst": ("Electric", "Grass", "Dragon"),
        "noEffectAgainst": ("Ground",),
    },
    "Grass": {
        "superEffectiveAgainst": ("Water", "Ground", "Rock"),
        "notVeryEffectiveAgainst": ("Fire", "Grass", "Poison", "Flying", "Bug", "Dragon", "Steel"),
        "noEffectAgainst": (),
    },
    "Ice": {
        "superEffectiveAgainst": ("Grass", "Ground", "Flying", "Dragon"),
        "notVeryEffectiveAgainst": ("Fire", "Water", "Ice", "Steel"),
        "noEffectAgainst": (),
    },
    "Fighting": {
        "superEffectiveAgainst": ("Normal", "Ice", "Rock", "Dark", "Steel"),
        "notVeryEffectiveAgainst": ("Poison", "Flying", "Psychic", "Bug", "Fairy"),
        "noEffectAgainst": ("Ghost",),
    },
    "Poison": {
        "superEffectiveAgainst": ("Grass", "Fairy"),
        "notVeryEffectiveAgainst": ("Poison", "Ground", "Rock", "Ghost"),
        "noEffectAgainst": ("Steel",),
    },
    "Ground": {
        "superEffectiveAgainst": ("Fire", "Electric", "Poison", "Rock", "Steel"),
        "notVeryEffectiveAgainst": ("Grass", "Bug"),
        "noEffectAgainst": ("Flying",),
    },
    "Flying": {
        "superEffectiveAgainst": ("Grass", "Fighting", "Bug"),
        "notVeryEffectiveAgainst": ("Electric", "Rock", "Steel"),
        "noEffectAgainst": (),
    },
    "Psychic": {
        "superEffectiveAgainst": ("Fighting", "Poison"),
        "notVeryEffectiveAgainst": ("Psychic", "Steel"),
        "noEffectAgainst": ("Dark",),
    },
    "Bug": {
        "superEffectiveAgainst": ("Grass", "Psychic", "Dark"),
        "notVeryEffectiveAgainst": (
            "Fire",
            "Fighting",
            "Poison",
            "Flying",
            "Ghost",
            "Steel",
            "Fairy",
        ),
        "noEffectAgainst": (),
    },
    "Rock": {
        "superEffectiveAgainst": ("Fire", "Ice", "Flying", "Bug"),
        "notVeryEffectiveAgainst": ("Fighting", "Ground", "Steel"),
        "noEffectAgainst": (),
    },
    "Ghost": {
        "superEffectiveAgainst": ("Psychic", "Ghost"),
        "notVeryEffectiveAgainst": ("Dark",),
        "noEffectAgainst": ("Normal",),
    },
    "Dragon": {
        "superEffectiveAgainst": ("Dragon",),
        "notVeryEffectiveAgainst": ("Steel",),
        "noEffectAgainst": ("Fairy",),
    },
    "Dark": {
        "superEffectiveAgainst": ("Psychic", "Ghost"),
        "notVeryEffectiveAgainst": ("Fighting", "Dark", "Fairy"),
        "noEffectAgainst": (),
    },
    "Steel": {
        "superEffectiveAgainst": ("Ice", "Rock", "Fairy"),
        "notVeryEffectiveAgainst": ("Fire", "Water", "Electric", "Steel"),
        "noEffectAgainst": (),
    },
    "Fairy": {
        "superEffectiveAgainst": ("Fighting", "Dragon", "Dark"),
        "notVeryEffectiveAgainst": ("Fire", "Poison", "Steel"),
        "noEffectAgainst": (),
    },
}
