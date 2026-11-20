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
- `REQUIRES.role` says how an evolution, a form change or a location uses its target: `hold`, `use`, `knownMove`, `tradedFor`, `partySpecies` or `partyType` for evolutions and form changes, `fieldMove` or `fieldItem` for locations.
- `ADJACENT_TO.direction` is `north`, `south`, `east` or `west`: the direction in which the target lies from the start place.
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

An `Encounter` is reached from a species through `HAS_ENCOUNTER`. It gives the game (`IN_VERSION`), the place (`AT_LOCATION`), the `method` and the `encounterKind` (`wild`, `raid`, `migration`, `static`, `snag` or `gift`). `FOUND_IN` is a separate edge from a species to a location and carries no version, so it cannot say in which game a species is found.

There are 2 kinds of encounter node. A detailed encounter comes from a location page: its id starts with the place (`mount-coronet-2f:diamond:golbat`), `AT_LOCATION` is the location or one of its `LocationArea` nodes, and it has `minLevel` and `maxLevel` when the table gives a level, `chance` (a percentage from 1 to 100) when it gives a rate, and `conditions` (`time-morning`, `time-day`, `time-night`, `season-spring` and so on, or `weather-rain`) when the rate depends on the time, the season or the weather. A coarse encounter comes from a species page and has none of these: its id starts with the species (`diglett:vermilion-city:heartgold`) and its place is a whole location.

For a place, version and encounter kind that a location page covers, only the detailed encounters exist. A kind that the location pages do not read stays coarse: the Max Raid Battle dens of Sword and Shield and the Pal Park migrations are coarse only. A species that a location table leaves out (a form row, or a method cell no rule maps, such as `Rippling`) has no encounter at that place and version at all, so a missing species is not proof that it is absent from the game.

### Locations

A place is a `Location` node with the label of its class: `Settlement` (cities and towns), `Route`, `GeoFeature` (caves, mountains, islands, lakes and named areas such as the Obsidian Fieldlands), `Structure` (towers, ruins, shops and gyms), or only `Location` when no rule fits its page. A floor or a sector is a `LocationArea` with the id `<location id>-<area>` (for example `mt-moon-b1f`), joined to its place by `LOCATED_IN`. `LOCATED_IN` also joins a place to the region and to the places around it, and a `Region` is never a start node of it.

`ADJACENT_TO` is written from each place page and is not symmetric, so a question about neighbours should read both directions. `APPEARS_IN` links a place to each `Version` of the games it appears in. A route that needs a move or an item to be explored fully has `REQUIRES` with `role` `fieldMove` (to a `Move`, for example Surf) or `fieldItem` (to an `Item`, for example Bicycle). A short name such as Mt. Coronet is in `aliases` of Mount Coronet.

An `ItemFind` is 1 item lying at 1 place in 1 version. It is reached from the item through `HAS_FIND` and points at the place with `AT_LOCATION` (the location or one of its areas) and at the game with `IN_VERSION`. `isHidden` is true for a hidden item and absent otherwise, and `method` is `rock-smash`, `fishing`, `berry-tree` or `gift` when the item comes that way. `FOUND_IN` runs from the item to the location of the page and never to an area. A heading that names another place does not move a find: the find is at the location of the page. Finds carry no rate and no map text.

### Type chart

`SUPER_EFFECTIVE_AGAINST`, `NOT_VERY_EFFECTIVE_AGAINST` and `NO_EFFECT_AGAINST` run from the attacking type to the defending type. For example `water` has `SUPER_EFFECTIVE_AGAINST` edges to `fire`, `ground` and `rock`, and `ghost` has a `NO_EFFECT_AGAINST` edge to `normal`. A pair with no edge is neutral. The edges carry no property, so the graph holds the current chart and no earlier one.

### Text

A `TextChunk` is a passage of prose with `text`, `aspect` (`biology`, `origin`, `pokedex`, `trivia`, `effect`, `flavor` or `geography`), `section` (the heading path, for example `Biology > Forms`) and `sourceUrl`. Each chunk belongs to 1 species, move, ability, item or location through `DESCRIBES`. A `geography` chunk comes from the lead, the Geography or Description section and the Places of interest of a location page. Other entities named in a chunk point at it through `MENTIONED_IN`. `FROM_WORK` points at the `Version` or `VersionGroup` the passage is about. There is no vector or full-text index yet, so text is found by starting from an entity.

### Gaps

- Moves and abilities carry their effect as `effectText` and as chunks with the aspect `effect`. There is no `MoveEffect`, `AbilityEffect` or `ItemEffect` node.
- Item pages are ingested for the core series games. An item has `category`, the Scarlet/Violet `buyPrice` and `sellPrice` when a shop trades it, `flingPower` when it can be thrown and, for a berry, `HAS_FLAVOR` with a `potency`. Items have no `isConsumable` and no `effectText`.
- Location pages are ingested for the core series games. A place has a class, a region, neighbours, versions and the first generation it appeared in, when its page names them. A page with no core game category is read only when it is in a small table of places (the Sevii islands, Hammerlocke and Wyndon Stadium, Naranja and Uva Academy). Other places that a page names without a page of their own are `Location` nodes with only a name.
- Detailed encounters cover the wild tables of the core series games. The rows of other forms (Alolan, Galarian and the like), in-game trades, curry encounters and Dynamax Adventures are not read, and neither are the star columns of the dens. A rate or a level that a table writes as text is left out.
- A floor or sector area comes from the headings of the Items and Pokémon sections, so a few areas are item or encounter groups that the heading table does not know yet (for example `Mt. Moon Walking Pokémon`).
