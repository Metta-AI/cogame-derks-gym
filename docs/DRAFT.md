# The loadout draft

One turn, before tick 0. **Simultaneous and hidden**: all six seats decide
at the same time and no seat sees any other seat's pick before committing.
That hiding is the metagame — a counter-draft must be a *prediction*, not a
reaction.

There is **no shared pool and no exclusivity**: two heroes, even on the same
team, may take the same item. Nothing is contended, so there is no pick order
and no tie-break, and the whole phase is one parallel batch instead of a
six-round snake draft.

Each seat fills **three slots — ARM, TAIL, MISC — with exactly one item
each**. Deltas are **additive** to the hero's own base stats.

## Catalog v1

`catalog_version: "v1"`, 12 items. Canonical sha256 of the catalog + clamp
table: `7c80ff58b617e148de4bfddfbfcaf8c336512123de6bfd582351de35ccefaef7`
(`cogame_derks_gym.catalog.catalog_sha256()`, also baked into the manifest
`config_schema`).

### Slot ARM (weapon)

| id | display name | deltas |
|---|---|---|
| `arm_none` | Bare Claws | *(none — all deltas 0)* |
| `arm_blaster` | Blaster | `base_damage +15` |
| `arm_cleaver` | Cleaver | `base_damage +35`, `basic_attack_cd +3` |
| `arm_needler` | Needler | `basic_attack_cd −3`, `base_damage −10` |

### Slot TAIL

| id | display name | deltas |
|---|---|---|
| `tail_none` | Stub Tail | *(none)* |
| `tail_plate` | Iron Plate | `base_health +200`, `basic_attack_cd +2` |
| `tail_stinger` | Stinger | `damage_gain_per_level +8` |
| `tail_rotor` | Rotor Tail | `move_speed +0.15`, `base_health −100` |

### Slot MISC (ability)

| id | display name | deltas |
|---|---|---|
| `misc_none` | Nothing | *(none)* |
| `misc_regen` | Regen Cell | `hp_gain_per_level +60` |
| `misc_battery` | Mana Battery | `base_mana +150`, `mana_gain_per_level +30` |
| `misc_focus` | Focus Chip | `base_damage +10`, `hp_gain_per_level −25` |

The `*_none` items are zero-delta on purpose: the **neutral loadout**
`{arm_none, tail_none, misc_none}` is both the house heroes' loadout and the
server's fallback, and it makes the un-drafted variant bit-identical to
upstream by construction rather than by a code path.

## Application order and clamps

Deltas are summed in the fixed order **ARM → TAIL → MISC** (addition
commutes, but the order is pinned so `applied` is reproducible bit-for-bit),
then each field is clamped:

| field | default | clamp |
|---|---|---|
| `base_health` | 300–700 by role | `[150, 1200]` |
| `base_mana` | 200–300 by role | `[100, 600]` |
| `base_damage` | 50 | `[20, 120]` |
| `basic_attack_cd` | 8 | `[3, 16]` (integer ticks) |
| `move_speed` | 1.0 | `[1.0, 1.5]` |
| `hp_gain_per_level` | 50–150 by role | `[0, 300]` |
| `mana_gain_per_level` | 50–90 by role | `[0, 200]` |
| `damage_gain_per_level` | 10–25 by role | `[0, 60]` |

`move_speed` is clamped at a **floor of 1.0** and no item lowers it, for a
fidelity reason: `compute_observations` writes
`obs_extra[6] = player->move_speed` as an `unsigned char`
(`vendor/upstream/moba.h:486`), so 0.9 would emit a `0` the pretrained
policies never saw in training, whereas 1.15 still casts to `1`. For the
same reason no item can push `base_damage/50` (`obs_extra[5]`) or
`basic_attack_cd` (`obs_extra[14]`) outside the small-integer band those
bytes already occupy. Loadouts are therefore **physically real but only
weakly encoded in the obs bytes** — the trained networks keep consuming
in-distribution observations and feel the items through the physics, which is
exactly the property that makes this mod safe.

Nothing else changes: no cost, no budget, no per-team uniqueness, no
in-match purchases, and no item touches skills, cooldown lengths, mana
costs, vision range or `agent_speed`.

## How a loadout becomes permanent

Every field above is already a member of upstream's `struct Entity`
(`moba.h:162-204`) that `spawn_player` (`moba.h:641-680`) and the level-up
path (`moba.h:791-793`) re-read on **every respawn and every level-up**.
Writing the summed, clamped block once — after `c_reset`, before the first
`c_step` — therefore makes the loadout permanent for the whole match for
free, with **no upstream patch** (the patch set stays at four).

`apply_loadout` (`sim/shim.c`, shared logic in `sim/loadout_common.h`) does
exactly this, in this order:

1. bounds-check the pid;
2. write the eight base fields;
3. `max_health = base_health + level*hp_gain_per_level` (same for mana and
   damage) — `spawn_player`'s derivation, reproduced **in place**, because a
   re-spawn would draw `rand()` and desync the seeded stream;
4. `health = max_health`, `mana = max_mana` (the hero is at full health at
   tick 0 anyway, so a neutral block is a byte-identical no-op);
5. record the block in the table `loadout_digest()` hashes.

It touches no RNG, no map cell and no allocation. One consequence worth
knowing: the tick-0 observation row was computed at the end of `c_reset`, so
it still carries the pre-loadout `damage/50`, `move_speed` and
`basic_attack_cd` bytes; from tick 1 on (`compute_observations` runs inside
`c_step`) it is the applied block. The server and the viewer apply loadouts
at the same point, so this changes nothing about determinism.

## Protocol

Server → player, once, before tick 0 (all six seats in one parallel batch
under one shared `draft_deadline_ms`): a `{"phase": "draft", ...}` object
carrying the seat's own hero (identity, role, lane, skills, exact base
stats), the aliases and roles of the five other seats and the four house
heroes, the entire catalog with exact deltas, the clamp table, the match
constants and its own deadline. Player → server, at most one message:

```json
{"phase": "draft",
 "picks": [{"arm": "arm_cleaver", "tail": "tail_plate",
            "misc": "misc_regen", "note": "tanky mid, out-scale the carry"}]}
```

Field caps: `phase` must equal `"draft"`; `picks` length exactly 1;
`arm`/`tail`/`misc` ≤ 24 chars each; `note` ≤ 120 characters truncated on
**Unicode-scalar (rune) boundaries**; whole frame ≤ 4096 bytes. `note` is
the only free-text field in the entire protocol.

## Resolution order, per seat

| # | condition | result |
|---|---|---|
| 1 | no reply by the shared deadline | neutral, `fallback_cause: "timeout"` |
| 2 | never connected / socket closed | neutral, `"disconnected"` |
| 3 | frame larger than 4096 bytes (dropped before the JSON parse) | neutral, `"oversize"` |
| 4 | not valid JSON, not an object, or `picks` not a 1-element array of objects | neutral, `"wrong_shape"` |
| 5 | any of `arm`/`tail`/`misc` missing, not a string, > 24 chars, or not an id of the **matching slot** (case-sensitive, exact after stripping ASCII spaces) | neutral **for the whole seat**, `"unknown_item"` |
| 6 | otherwise | accepted, `"none"`, `fallback: false` |

Partial acceptance in (5) is deliberately not allowed: it would let a seat
launder a typo into a free reroll of one slot. A bad `note` never
invalidates the picks. A `{"tick": ...}` reply (or any unrecognised `phase`)
arriving during the draft turn is ignored and does **not** consume the turn —
which is why an existing cogame-moba policy container plays this game
unmodified: it eats one draft timeout, gets the neutral loadout, and then
plays every tick normally.

House heroes (pids 3, 4, 8, 9) always get the neutral loadout with
`source: "house"`, `player_name: null` and `decision_ms: 0`.

A draft failure is **never fatal**: `end_reason` stays the same closed
four-value enum and the match starts on time on the neutral loadout.

## Two kinds of fallback (and where each one is counted)

These are different events with different records. Do not read one for the
other.

**Server-side substitution** — the table above. The server could not use
what the seat sent (or the seat sent nothing), so it substituted the
neutral loadout. Recorded in the replay header's draft record and in
`results.json`: `fallback: true`, `fallback_cause: <one of the six>`, and
`results.draft_fallbacks[seat] = true`. **`results.draft_fallbacks` counts
exactly this and nothing else.**

**Player-side LLM → scripted fallback** — inside a `PLAYER_PROMPT`
champion (`players/derk_player.py`). The model did not answer usefully, so
the player sends its *scripted* draft rule's pick instead. That pick is
legal, so the server accepts it: `fallback: false`, `fallback_cause:
"none"`, `draft_fallbacks[seat] = false`. The protocol's reply schema is
closed (`arm`/`tail`/`misc`/`note` only) and is deliberately **not**
widened for this, so the record is the player's own stderr line:

```
draft_fallback=scripted reason=<reason> picks={...}
```

with `reason` from this closed set (`derk_player.FALLBACK_REASONS`):

| reason | meaning |
|---|---|
| `no_key` | `ANTHROPIC_API_KEY` unset — no call was made at all |
| `no_time` | the observation's `deadline_ms` left less than 1 s for a call, so none was made (`derk_player.call_timeout`) |
| `timeout` | a call did not answer inside its budget (min of 20 s and what is left of `deadline_ms`) |
| `parse` | the reply contained no extractable JSON object |
| `illegal` | an object was extracted, but a slot's id was not in this seat's catalog |
| `transport` | HTTP/network failure (the exception type is logged on its own line) |

A successful model draft logs `draft=<prompt name> attempt=<1\|2>
picks={...}` instead, and never the `draft_fallback=` line. So:
**count LLM usage from the player logs — `draft_fallback=scripted
reason=…` per champion container — not from `results.draft_fallbacks`,
which is the server-side number.**

## Closed contract

`server/cogame_derks_gym/catalog.py`, the manifest `config_schema`
(`catalog_version` enum + the sha above), this page, and the
`item-<catalog id>` symbols in `viewer/derk_items.svg` change **together**
or `tests/test_catalog.py` fails (AGENTS.md rule 3).
