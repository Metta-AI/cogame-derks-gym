# cogame-derks-gym wire protocol

The protocol a policy container speaks to play an episode, plus the
spectator and replay surfaces. The observation and action *encodings* are
deliberately opaque here: they are the upstream PufferLib Ocean MOBA
encodings, transported verbatim (see "Observations" and "Actions" below
for the upstream ground truth).

## Seats and heroes

Upstream Puffer MOBA is 5v5 (`NUM_PLAYERS` 10). This game seats **six** of
those ten heroes, three per team, one hero per seat:

```
seat  0            1            2            3            4            5
pid   0            1            2            5            6            7
alias Cog-Alpha    Cog-Bravo    Cog-Charlie  Cog-Delta    Cog-Echo     Cog-Foxtrot
team  radiant      radiant      radiant      dire         dire         dire
role  support      assassin     burst        support      assassin     burst
```

Pids **3, 4, 8, 9** (both teams' tank and carry) are **house heroes**: the
game server drives them itself, in-process, off the vendored pretrained
network on the neutral loadout. They are not seats and cannot be played.

Two name spaces: a policy only ever sees the **aliases** above (teams are
server-assigned; a seat can neither choose nor infer its opponents'
identities). The **real** policy names live in `results.names` and in the
replay header's `config.players` — spectator side only. The LLM request body
is asserted by test never to contain a real player name.

## Player websocket (`GET /player?slot=N&token=T`)

A player container receives its fully-formed connection URL in the
`COWORLD_PLAYER_WS_URL` environment variable (legacy alias
`COGAMES_ENGINE_WS_URL`), e.g.
`ws://game-host:8080/player?slot=3&token=abc123`. Connect, answer the one
draft turn (below), then speak one JSON text message per tick each way:

```
server -> player   {"phase": "draft", ...}                 once, before tick 0
player -> server   {"phase": "draft", "picks": [{...}]}     at most one reply
server -> player   {"phase": "draft_result", "loadouts": [...]}  no reply

server -> player   {"tick": t, "obs": ["<base64>"]}         one base64 blob
player -> server   {"tick": t, "actions": [[a0..a5]]}       one 6-int row
server -> player   {"done": true, "result": {...}}          episode end, then close
```

- `obs` has one entry per hero this seat controls (always 1), decoding to
  exactly 510 bytes.
- The reply must echo the same `tick`. Wrong-tick, malformed, late
  (past `tick_deadline_ms`), or missing replies play the no-op action
  `[3, 3, 0, 0, 0, 0]` for that seat's heroes — the episode never stalls
  and never crashes on bad input.
- Seat `i` controls exactly one hero, pid `[0, 1, 2, 5, 6, 7][i]` (the
  fixed table above). Pids 0-4 are radiant, 5-9 dire; within each team the
  role order is support, assassin, burst, tank, carry.
- Strike rule: 10 consecutive no-op fallbacks mark a seat dead — the
  server stops waiting out the deadline for it, but keeps probing with
  each tick's obs; the first valid reply revives the seat. On the
  transition to dead the server also force-closes the seat's websocket:
  a connection that missed 10 straight ticks is treated as stale, and
  closing it lets the client observe the drop and reconnect.
- Bad slot/token is rejected with HTTP 403 — fatal, retrying can never
  succeed. A connection to a slot that already has a live connection is
  rejected with HTTP 409 — retryable: the server heartbeats player
  sockets (websocket ping/pong, ~20s) and strike-closes dead seats, so
  a half-open stale connection clears within seconds and a retried
  reconnect then succeeds. A seat that disconnects may reconnect (any
  number of times) and resume at whatever tick the server sends next.

## The draft turn (one turn, simultaneous, hidden)

Before tick 0 the server sends every seat one `{"phase": "draft", ...}`
observation — **all six as a single parallel batch under one shared
`draft_deadline_ms`** — and consumes at most one reply per seat:

```json
{"phase": "draft",
 "picks": [{"arm": "arm_cleaver", "tail": "tail_plate",
            "misc": "misc_regen", "note": "tanky mid, out-scale the carry"}]}
```

| field | cap |
|---|---|
| `phase` | ≤ 16 chars, must equal `"draft"` |
| `picks` | array, length exactly 1, of objects |
| `arm` / `tail` / `misc` | ≤ 24 chars each, an id of the **matching** catalog slot |
| `note` | ≤ 120 characters, truncated on Unicode-scalar (rune) boundaries, C0/C1 controls stripped; the only free-text field in the protocol |
| whole frame | ≤ 4096 bytes (a larger frame is dropped before the JSON parse) |

The resolution order (timeout / disconnected / oversize / wrong_shape /
unknown_item / accepted) and the full catalog with its deltas and clamps are
in [DRAFT.md](DRAFT.md). Every failure resolves to the neutral loadout for
that seat: a draft failure is never fatal and adds no `end_reason`.

Afterwards the server pushes `{"phase": "draft_result", "loadouts": [...]}`
— the ten draft-reveal records, **alias-only** — to every seat and to the
`/global` feed. No reply is expected.

Messages with an unrecognised `phase` are ignored by both sides, and a
`{"tick": ...}` reply arriving during the draft turn is ignored without
consuming the turn. A policy that never answers the draft simply plays the
neutral loadout — so an existing cogame-moba policy container plays this
game unmodified.

## Observations (510 bytes per hero, opaque)

The exact byte layout produced by upstream `compute_observations` in
`vendor/upstream/moba.h` (PufferAI/PufferLib @ `c5d3c637`): an 11x11
map crop around the hero (121 x 4 bytes) plus scalar hero state. Policies
trained on upstream Puffer MOBA consume these bytes unchanged — that is
the point of this port. Decode guidance for hand-written policies:
`players/scripted_player.py` documents the reliably-decodable fields.

## Actions (6 values per hero)

Upstream MultiDiscrete `[7, 7, 3, 2, 2, 2]`: `vel_y`, `vel_x` (0-6,
center 3 = zero velocity), target filter (0-2), and three skill buttons
(0/1). Values are truncated toward zero and clamped into range at the
sim boundary, exactly like upstream's C cast.

## Global viewer (`GET /global`, `GET /client/global`)

`/global` is a broadcast-only websocket: an initial
`{"type": "status", ...}` snapshot on connect (carrying `phase`
`"draft" | "play" | "done"`, the seat names, the aliases, and — once the
draft resolves — the alias-only `loadouts`), throttled `{"tick": t}`
progress messages while the episode runs, and the final
`{"done": true, "result": {...}}`. `/client/global` serves a minimal
HTML page over that feed. `GET /client/player?slot=N&token=T` serves a
token-checked seat page (play happens over the websocket, not the page).

## Runtime contract (Coworld)

The game container reads `COGAME_CONFIG_URI` (game config JSON, see the
manifest `config_schema`), writes `COGAME_RESULTS_URI` (results JSON,
see `results_schema`) and `COGAME_SAVE_REPLAY_URI` (binary replay), and
reports never-connected seats to `COGAME_PLAYER_FAILURE_URI`. It binds
`COGAME_HOST`:`COGAME_PORT` (default `0.0.0.0:8080`) and serves
`GET /healthz`. With `COGAME_LOAD_REPLAY_URI` set it runs in replay mode
instead: raw replay bytes at `GET /replay-data` and the wasm re-sim
viewer at `GET /client/replay`.

Wall-clock budget: the worst-case episode (`max_ticks x
tick_deadline_ms`) can far exceed the platform's
`episode_timeout_minutes` container kill, which would lose results and
replay. The engine therefore hard-stops a slow episode at
`wall_clock_budget_seconds` (config; default `min(0.9 x
episode_timeout_minutes x 60, max_ticks x tick_deadline_ms / 1000)`)
with `end_reason: "wall_clock"`, the same Ancient-health tiebreak as
`tick_cap`, and artifacts written normally.

## Replay format (binary, v2)

`DERK` magic, u8 version = 2, u32le header length, header JSON, then
`tick_count * 60` bytes of packed post-clamp actions (10 heroes x 6 uint8
per tick). The header is **self-sufficient** — everything the static viewer
needs is in these bytes and nothing but the replay file is ever fetched:

| key | what |
|---|---|
| `format_version`, `sim_wasm_sha256` | the format and the sim build that recorded it |
| `catalog_version`, `catalog` | the pinned catalog, with item names and exact deltas |
| `config` | the fully-resolved config incl. `seed` and the real player names (tokens excluded) |
| `aliases`, `seat_hero_pids`, `house_hero_pids` | the seat topology |
| `draft` | the ten draft-reveal records, incl. each hero's `applied` stat block |
| `loadout_digest` | u32 FNV-1a over the applied table; the viewer re-derives it and warns on a mismatch |
| `events` | ≤ 400 events (`draft`, `first_blood`, `kill`, `tower`, `level_spike`, `ancient`, `end`) for the feed and the scrubber beats |
| `result` | the full `results.json` document |
| `tick_count`, `final_state_digest` | body length / 60, and a u32 digest of hero x/y/health + both Ancients |

Seed + loadouts + actions fully determine the episode; the viewer
re-simulates it with the same wasm sim and the digests prove the equality
rather than assuming it. Replay v1 (`MOBA`) is not read — old cogame-moba
replays are watched in cogame-moba. Ground truth:
`server/cogame_derks_gym/replay.py`.
