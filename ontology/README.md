# muon ontologies

This directory holds the schema of the muon knowledge graph as RDF/OWL Turtle. The ttl files are the source of truth for every label, relationship and property in the graph. The graph itself lives in Neo4j as a labeled property graph (LPG), derived from the ttl by the rules in [LPG realization](#lpg-realization).

| File | Namespace | Content |
| --- | --- | --- |
| `core/muon.ttl` | `mo:` `https://muon.dev/ns/core#` | Core T-Box, shared by every universe. 34 classes, 20 object properties, 6 datatype properties |
| `universes/pkmn.ttl` | `pkmn:` `https://muon.dev/ns/pkmn#` | Pokemon T-Box. Imports the core. 37 classes, 47 object properties, 62 datatype properties |
| `individuals/pkmn.ttl` | `pkmn:` | Pokemon A-Box: 163 enumerable individuals in 17 classes |
| `shapes/pkmn.shacl.ttl` | `pkmn-shape:` `https://muon.dev/ns/pkmn-shapes#` | Pokemon SHACL shapes. Imports the pkmn T-Box. 16 rules, listed in [Shapes](#shapes) |

A universe gets a short lowercase prefix (`pkmn`). Its T-Box goes in `universes/<prefix>.ttl` and declares `owl:imports <https://muon.dev/ns/core>`. Its closed vocabularies go in `individuals/<prefix>.ttl` under the same namespace.

## Vocabularies and annotations

- `rdfs:` carries the hierarchy (`subClassOf`, `subPropertyOf`) and `domain`, `range`, `label`, `comment`.
- `owl:` declares terms (`Class`, `ObjectProperty`, `DatatypeProperty`) and their traits (`TransitiveProperty`, `inverseOf`, `disjointWith`, `unionOf`, `propertyChainAxiom`).
- `skos:definition` states what a term means. `rdfs:comment` holds modelling notes, such as scope limits and "not to be confused with".
- `xsd:` types every datatype property range.
- `rdfs:label` is the display name, always tagged `@en`.

Individuals are typed by their class only (no `owl:NamedIndividual`). Cardinality and value constraints are not expressed in OWL.

## Naming

| Kind | RDF | Example |
| --- | --- | --- |
| Class | `UpperCamelCase`, singular | `pkmn:PokemonSpecies` |
| Reified fact | Noun for the fact | `pkmn:Evolution`, `pkmn:MoveEffect` |
| Object property | `lowerCamelCase`, verb first | `pkmn:inflictsCondition` |
| State or membership | Preposition form | `mo:locatedIn`, `pkmn:inEggGroup` |
| Datatype property | `lowerCamelCase` noun | `pkmn:catchRate` |
| Boolean | `is` or `has` prefix | `pkmn:isLegendary` |
| Individual | `lowerCamelCase`, class prefix on a name clash | `pkmn:adamant`, `pkmn:eggGroupBug` |

A property shared by unrelated classes declares `rdfs:range` only, with no `rdfs:domain`.

## Core ontology (mo:)

The core is universe-agnostic and splits into 2 planes.

The diegetic plane holds everything that exists inside a fictional universe. Its root is `mo:Entity`:

```
Entity
  Agent: Person, Creature, Deity            (Person and Creature are disjoint)
  Location: World, Region, GeoFeature, Settlement, Structure, Route
  Group: Organization, Kin, Community, Party, Collective
  Object
  Event
  TimePeriod
  Concept: Species, Language, Power, System
```

The meta plane holds the real-world sources that describe a universe. Its root is `mo:Meta`:

```
Meta
  Universe
  Source: Series, Work, Chapter, TextChunk
  Creator
```

### Core object properties

| Property | Domain -> Range | Notes |
| --- | --- | --- |
| `locatedIn` | Location -> Location | Transitive |
| `foundIn` | Agent, Object or Species -> Location | Chain `foundIn o locatedIn` |
| `occursAt` | Event -> Location | |
| `memberOf`, `affiliatedWith` | Agent -> Group | Formal membership, looser association |
| `parentOf`, `childOf` | Agent -> Agent | Inverses |
| `owns` | Agent -> Entity | Open range |
| `participatesIn` | Agent -> Event | |
| `hasSpecies`, `speaks`, `hasPower` | Agent -> Species, Language, Power | |
| `partOf` | Source -> Source | Transitive |
| `appearsIn`, `mentionedIn` | Entity -> Source | Depiction, reference |
| `createdBy` | Series, Work or Universe -> Creator | |
| `fromWork` | TextChunk -> Work | |
| `describes` | TextChunk -> Entity | Exactly 1 per chunk |
| `ofUniverse` | Source -> Universe | RDF only |
| `inUniverse` | Entity -> Universe | RDF only |

### Core datatype properties

- `aliases` (Entity, repeated string): other names, such as other languages, trademark and beta names.
- `text` (TextChunk): the verbatim chunk text.
- `aspect` (TextChunk): the kind of content, one of biology, effect, flavor, trivia, origin, profile, anime, manga, side_game. pkmn adds pokedex.
- `source` (TextChunk): the non-canon origin of a chunk. Canon origins use `fromWork`.
- `section` (TextChunk): the heading path inside the source page, e.g. `Biology > Forms`.
- `sourceUrl` (TextChunk): permalink to the source page revision.

## Pokemon ontology (pkmn:)

The pkmn ontology models species and game mechanics. It holds no individual Pokemon. Structured facts describe the Scarlet/Violet snapshot, except learnset entries, machines and encounters, which are scoped to a version group or version.

### Classes

| Core parent | pkmn classes |
| --- | --- |
| `mo:Species` | `PokemonSpecies` (forms are `PokemonSpecies` nodes too) |
| `mo:Power` | `Move` |
| `mo:Concept` | `Type`, `Stat`, `BattleStat`, `Nature`, `EggGroup`, `Ability`, `MoveDamageClass`, `MoveFlag` |
| `mo:Concept` (reified) | `MoveEffect`, `AbilityEffect`, `ItemEffect`, `LearnsetEntry`, `Evolution`, `FormChange`, `Encounter` |
| `mo:Concept` | `BattleCondition` > `StatusCondition`, `VolatileCondition`, `FieldCondition` > `Weather`, `Terrain`, `EntryHazard`, `FieldEffect` |
| `mo:Object` | `Item` > `Berry`, `Machine`, `HeldItem` |
| `mo:Group` | `SpeciesGroup` > `LegendaryGroup`, `StarterTrio` |
| `mo:Location` | `LocationArea` (regions use `mo:Region`) |
| `mo:Series` | `Generation` |
| `mo:Work` | `VersionGroup`, `Version` |
| `mo:Meta` | `GenerationMechanic` |

### Individuals

`individuals/pkmn.ttl` enumerates the closed vocabularies that edges point at: 18 Types, 6 Stats, 3 BattleStats (Accuracy, Evasion, Critical Hit Ratio), 3 MoveDamageClasses, 12 EggGroups, 25 Natures, 6 StatusConditions, 20 VolatileConditions, 9 Weathers, 4 Terrains, 4 EntryHazards, 8 FieldEffects, 19 MoveFlags, 9 Generations, 11 Regions, 5 GenerationMechanics and the `pkmn:pokemon` Universe. Regions and mechanics carry `introducedIn`. Every other instance is created by ingestion.

### Species, types and natures

- A species has a type through `primaryType` and `secondaryType`, both sub-properties of `hasType`. A move has exactly 1 `hasType`.
- Type effectiveness uses sub-properties of `effectiveAgainst`, so the multiplier is in the property name: `superEffectiveAgainst` (x2), `notVeryEffectiveAgainst` (x0.5), `noEffectAgainst` (x0).
- A nature points at stats with `boostsStat` and `reducesStat`, both sub-properties of `affectsStat`.
- Other species edges: `hasAbility`, `inEggGroup`, `yieldsEv` (to Stat), `inGroup`, `designedBy` (to `mo:Creator`), `introducedIn`.
- Species properties: `pokedexNumber`, the 6 base stats (`baseHp` to `baseSpeed`), `height` (m), `weight` (kg), `femaleRate`, `isGenderless`, `catchRate`, `baseExpYield`, `shape`, `genus`, `eggCycles`, `baseFriendship`, `growthRate`, `color`, `isLegendary`, `isMythical`, `isStarter`, `isDefaultForm`.

### Moves and learnsets

- Move properties: `power`, `accuracy`, `pp`, `priority`, `target` (targeting code), `critStage`, `minHits`, `maxHits`, `effectText`.
- Move edges: `hasType`, `hasDamageClass`, `hasFlag`, `hasEffect`, `introducedIn`.
- `LearnsetEntry` reifies 1 (species, move, version group, method, level) fact. It links `teachesMove`, `inVersionGroup` and, where relevant, `viaMachine` or `viaParent`. `method` is one of level, machine, egg, tutor, evolution, special.
- `Machine` is 1 TM, HM or TR in 1 version group, with `machineKind`, `machineNumber`, `isReusable`, `teachesMove` and `inVersionGroup`.
- `canLearn` is a species-to-move shortcut over the Scarlet/Violet learnset entries.

### Effects

Moves, abilities and items share 1 effect model. `hasEffect` links the owner to reified `MoveEffect`, `AbilityEffect` or `ItemEffect` nodes, one node per effect.

Action edges say what the effect does:

- `inflictsCondition`, `curesCondition`, `blocksCondition` (status, volatile or weather)
- `immuneTo` (Type, MoveFlag or StatusCondition)
- `setsField`, `removesField` (FieldCondition)
- `setsType` (Type)
- `affectsStat`, `blocksStatDrop` (Stat or BattleStat), `restoresStat` (Stat)

Filter edges say when it applies. `triggeredBy` points at a MoveFlag, MoveDamageClass, Type, FieldCondition or (from a FormChange) Ability. Edges to the same class mean any of them; edges to different classes must all match. `restrictedTo` limits an item effect to the listed species.

Magnitudes are node properties: `probability`, `statStages`, `statMultiplier`, `powerMultiplier`, `damageMultiplier`, `hpChange` with `hpBasis` (max_hp, current_hp, damage_dealt), `hpThreshold`, `amount`.

`target` names the recipient (self, ally, foe, field, move). `trigger` names the moment an ability effect, item effect or form change fires: switch_in, switch_out, on_attack, on_hit, on_knock_out, low_hp, end_of_turn, passive.

Weather and terrain individuals point at types with `amplifiesType` and `dampensType`, both sub-properties of `modifiesType`.

### Evolution and forms

- `Evolution` reifies 1 evolution with `fromSpecies`, `toSpecies`, `requires` and the properties `evoTrigger`, `minLevel`, `timeOfDay`, `gender`, `condition`. `evolvesTo` is the species-to-species shortcut.
- `hasForm` links a species to a form node that differs in battle (stats, type, ability or learnset).
- `FormChange` reifies how a species reaches a form: `fromSpecies`, `toSpecies`, `requires`, `triggeredBy`, `trigger`, `formKind` (alternate, battle, mega, primal, ultra_burst, gigantamax, terastal) and `isBattleOnly`.
- `requires` points at an Item, Move, Location, PokemonSpecies or Type.

### Items

- `Item` properties: `category`, `price`, `isConsumable`, `effectText`. `Berry` adds `flavour`.
- Items with a battle effect use the effect model. Items that only gate something (evolution stones, Mega Stones) are reached through `requires`.

### Generations, versions and encounters

- `Version` is part of a `VersionGroup`, which is part of a `Generation`, through `mo:partOf`.
- `GenerationMechanic` records a battle mechanic (Mega Evolution, Z-Move, Dynamax, Gigantamax, Terastallization) and its generation.
- `Encounter` reifies 1 wild encounter. It links `atLocation` (a `LocationArea` or a location) and `inVersion`, with `method`, `conditions`, `minLevel`, `maxLevel` and `chance`. A species links to it with `hasEncounter`.
- `LocationArea` is a subdivision of a location (a floor, a sector) and chains to it with `mo:locatedIn`.

## Shapes

`shapes/pkmn.shacl.ttl` holds the constraints a pkmn node or edge must meet. The Validator loads it with `pyshacl` and checks the post-state of every write transaction, using the T-Box as the ontology graph. Its sections follow `universes/pkmn.ttl`. A shape's `sh:name` is its rule id and `sh:message` names the values involved.

| Section | Rules |
| --- | --- |
| Node keys and edge typing | L2 node key pattern, T1 edge range and domain read from the T-Box |
| Species | S1 dex number, S2 types, S3 female rate and growth rate, S4 EV amount |
| Moves and types | M1 one type and one damage class |
| Items | C2 machine links |
| Meta and text | X1 text chunk completeness, X2 `fromWork` by aspect |
| Moves: flags, effects, learnset | C2 learnset entry, E1 effect values, X3 `effectText` owner |
| Evolution and forms | V1 species links, V2 form kind, R1 `REQUIRES` role |
| Encounters and locations | C1 encounter links, level order and chance |

Every shape carries `pkmn-shape:scope`, the classes it checks, or `pkmn-shape:Always` for the shapes that run on every touched node (L2 and T1). `validate(batch, scope)` runs a shape when its scope class is in the requested scope or a subclass of it. A node outside the scope may be a stub with only `id`, `name` and labels, and the Always shapes still apply to it. `scope=None` runs every shape and is the final sweep after all ingestion passes.

The shapes check a projection of the Neo4j transaction, not the Neo4j data directly:

- The node key `id` becomes `pkmn:id` and `name` becomes `rdfs:label`. Neither is declared in the ttl.
- Each label becomes an `rdf:type`, and the T-Box subclass closure resolves `sh:class`.
- `HAS_TYPE {slot:1}` becomes `pkmn:primaryType` and `{slot:2}` becomes `pkmn:secondaryType`.
- An edge with properties (`REQUIRES {role}`, `YIELDS_EV {amount}`) becomes a reified edge node with `rdf:subject`, `rdf:predicate`, `rdf:object` and the edge properties.

The label hierarchy check (L1) and `id` uniqueness are not SHACL. The Validator runs them in code. Every `sh:in` list repeats values from a ttl definition, and `tests/unit/shapes/test_vocabularies.py` checks that they agree.

## LPG realization

The Neo4j graph follows the ttl through these rules.

| RDF | Neo4j | Example |
| --- | --- | --- |
| Class | Label with the same local name | `pkmn:Move` -> `:Move` |
| Object property | Relationship in `SCREAMING_SNAKE_CASE`, subject -> object | `hasDamageClass` -> `HAS_DAMAGE_CLASS` |
| Datatype property | Node property with the same name | `baseSpeed` |
| Universe prefix | Lowercase label on every node | `:pkmn` |
| `rdfs:label` of an individual | `name` | `"Fire"` |

- Labels. A node carries its class and every ancestor class as co-labels, so `pkmn:Berry` becomes `:Berry:Item:Object:pkmn`. Abstract classes never become labels: `mo:Entity`, `mo:Concept`, `mo:Meta`, `mo:Source`, `pkmn:BattleCondition`, `pkmn:FieldCondition`.
- Universe. Every node of a universe carries its prefix label, meta nodes included. `mo:inUniverse` and `mo:ofUniverse` have no relationship in Neo4j.
- Sub-properties. Only the most specific property becomes a relationship. An effect node may write the general `AFFECTS_STAT`, because its magnitude lives on the node.
- Exception: `pkmn:primaryType` and `pkmn:secondaryType` both become `HAS_TYPE`, with the edge property `slot` 1 or 2. A move has 1 `HAS_TYPE` and no `slot`.
- Shortcuts. `EVOLVES_TO`, `CAN_LEARN` and `FOUND_IN` are derived from reified nodes and exist next to them.
- Edge properties exist in Neo4j only: `isHidden` on `HAS_ABILITY`, `amount` on `YIELDS_EV` and `role` on `REQUIRES` (use, hold, knownMove, knownMoveType, atLocation, partySpecies, partyType, tradedFor, fusesWith).
- Individuals from `individuals/` become nodes with their class labels, the universe label and `name`.
- Node key. `id` is the lowercase hyphenated English name (`shadow-ball`, `diglett-alola`), unique per label. `name` is not a key, since names repeat across labels ("Psychic" is a Type and a Move).
- Datatypes. `xsd:string` -> `STRING`, `xsd:integer` -> `INTEGER`, `xsd:decimal` -> `FLOAT`, `xsd:boolean` -> `BOOLEAN`. A repeated string (`aliases`, `conditions`) becomes `LIST<STRING>`.
- Text. Only `TextChunk` nodes carry an `embedding` (`VECTOR`), which is not declared in the ttl.

```cypher
(:PokemonSpecies:Species:pkmn {id:"charmander"})-[:EVOLVES_TO]->(:PokemonSpecies:Species:pkmn {id:"charmeleon"})
(:Type:pkmn {name:"Fire"})-[:SUPER_EFFECTIVE_AGAINST]->(:Type:pkmn {name:"Grass"})
(:Move:Power:pkmn {id:"flamethrower", power:90})-[:HAS_EFFECT]->(:MoveEffect:pkmn {probability:10})-[:INFLICTS_CONDITION]->(:StatusCondition:pkmn {name:"Burn"})
(:Ability:pkmn {id:"iron-fist"})-[:HAS_EFFECT]->(:AbilityEffect:pkmn {trigger:"on_attack", powerMultiplier:1.2})-[:TRIGGERED_BY]->(:MoveFlag:pkmn {name:"punch"})
(:Evolution:pkmn {evoTrigger:"trade"})-[:REQUIRES {role:"hold"}]->(:HeldItem:Item:Object:pkmn {name:"Metal Coat"})
(:Version:Work:pkmn {id:"diamond"})-[:PART_OF]->(:VersionGroup:Work:pkmn)-[:PART_OF]->(:Generation:Series:pkmn)
(:TextChunk:pkmn {aspect:"biology"})-[:DESCRIBES]->(:PokemonSpecies:Species:pkmn {id:"diglett"})
```
