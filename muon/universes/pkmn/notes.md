### Node keys

`id` is a lowercase hyphenated slug of the English name, for example `diglett-alola`, `u-turn`, `scarlet-violet` and `generation-iv`. `name` is the display name, for example `Mr. Mime` or `U-turn`. `aliases` holds other names of the entity, mostly its names in other languages. Match an entity by `id` when the question names it.

### Stubs

A node with only `id` and `name` is a stub, a reference created by one page before the page of that entity was ingested. A stub has no other fact to give.

- Every `Location` is a stub (906 nodes), so a location has no region, description or coordinates.
- Every `Item` that is not a `Machine` is a stub, because item pages are not ingested. `HeldItem` nodes belong to them.
- 90 `Move` nodes are stubs. Most are Z-moves, Max and G-Max moves and Shadow moves. A few come from mistaken ids, such as `baton-passed` (the move is `baton-pass`).

### Forms

Each form is its own `PokemonSpecies` node, and `isDefaultForm` is true on the 1,025 default forms. The default form links to its other forms through `HAS_FORM`. On regional forms the edge carries `region` (`Alola`, `Galar`, `Hisui` or `Paldea`), and other forms have none. A question that names a species without a form means the default form, for example `diglett`, while `diglett-alola` is a separate node.

### Edge properties

- `HAS_TYPE.slot` is 1 for the primary type and 2 for the secondary type. A move has 1 type and its edge has no slot.
- `HAS_ABILITY.isHidden` is true for the hidden ability of a species.
- `REQUIRES.role` says how an evolution or form change uses its target: `hold`, `use`, `knownMove`, `tradedFor`, `partySpecies` or `partyType`.
- `YIELDS_EV.amount` is the number of effort points.

### Evolutions and form changes

An `Evolution` node holds 1 evolution rule. `FROM_SPECIES` and `TO_SPECIES` point at the 2 species, and `REQUIRES` points at what the rule needs. `EVOLVES_TO` is a direct edge between the same 2 species. Some species have no evolution data yet. Eevee has no `Evolution` node and no `EVOLVES_TO` edge, so the graph cannot say what Eevee evolves into.

`FormChange` nodes cover Mega Evolution (91) and Primal Reversion (2). Other form changes are not ingested.

### Games

A `Version` is 1 game, a `VersionGroup` is the group of games that share data (`scarlet-violet`), and a `Generation` holds the groups of 1 generation. `PART_OF` links `Version` to `VersionGroup` and `VersionGroup` to `Generation`. A filter by generation goes through that chain, for example `(:Encounter)-[:IN_VERSION]->(:Version)-[:PART_OF*]->(:Generation)`. `INTRODUCED_IN` points at the `Generation` where an entity first appeared.

### Learnsets

`CAN_LEARN` runs from a species to a move. The detail sits on `LearnsetEntry` nodes, reached through `HAS_LEARNSET_ENTRY`. An entry gives the `method`, the `level` when it has one, the move (`TEACHES_MOVE`), the game (`IN_VERSION_GROUP`) and the machine (`VIA_MACHINE`).

Every entry belongs to `scarlet-violet`. Older games have no learnset data, so the graph cannot say what a species learned in them. 292 default forms have no learnset at all.

### Encounters

An `Encounter` is reached from a species through `HAS_ENCOUNTER`. It gives the game (`IN_VERSION`), the place (`AT_LOCATION`), the `method` and the `encounterKind` (`wild`, `raid`, `migration`, `static`, `snag` or `gift`). It has no level, rate or area. `FOUND_IN` is a separate edge from a species to a location and carries no version, so it cannot say in which game a species is found.

### Type chart

`SUPER_EFFECTIVE_AGAINST`, `NOT_VERY_EFFECTIVE_AGAINST` and `NO_EFFECT_AGAINST` run from the attacking type to the defending type. For example `water` has `SUPER_EFFECTIVE_AGAINST` edges to `fire`, `ground` and `rock`, and `ghost` has a `NO_EFFECT_AGAINST` edge to `normal`. A pair with no edge is neutral. The edges carry no property, so the graph holds the current chart and no earlier one.

### Text

A `TextChunk` is a passage of prose with `text`, `aspect` (`biology`, `origin`, `pokedex`, `trivia`, `effect` or `flavor`), `section` (the heading path, for example `Biology > Forms`) and `sourceUrl`. Each chunk belongs to 1 species, move, ability or item through `DESCRIBES`. Other entities named in a chunk point at it through `MENTIONED_IN`. `FROM_WORK` points at the `Version` or `VersionGroup` the passage is about. There is no vector or full-text index yet, so text is found by starting from an entity.

### Gaps

- Moves and abilities carry their effect as `effectText` and as chunks with the aspect `effect`. There is no `MoveEffect`, `AbilityEffect` or `ItemEffect` node.
- Item pages are ingested for the core series games. An item has `category`, the Scarlet/Violet `buyPrice` and `sellPrice` when a shop trades it, `flingPower` when it can be thrown and, for a berry, `HAS_FLAVOR` with a `potency`. Items have no `isConsumable` and no `effectText`.
