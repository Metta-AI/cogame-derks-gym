# cogame-derks-gym

A [Coworld](https://softmax.com) game: **PufferLib's Ocean MOBA with a
pre-match loadout draft**. Before tick 0, all six seats simultaneously and
blindly pick one arm, one tail and one misc item from a 12-item catalog; the
summed, clamped stat deltas are written into their heroes for the whole
match. Then it is the upstream 5v5 MOBA, bit-exact with the environment the
pretrained policies were trained on.

The idea in one line: **the policy is metagame plus micro**. A MOBA tick is
100 ms and an episode is 6000 ticks, so no language model plays the ticks —
but 4³ = 64 loadouts per hero, chosen blind against an unseen opponent, is
exactly the kind of one-shot decision a prompt is good at. So an LLM drafts,
and the vendored pretrained network fights.

Forked from [`Metta-AI/cogame-moba`](https://github.com/Metta-AI/cogame-moba):
the vendored upstream C sim ([PufferAI/PufferLib](https://github.com/PufferAI/PufferLib)
@ `c5d3c637`, MIT) is still byte-pristine under `vendor/upstream/`, still
patched at build time by the same four patches, still compiled to
WebAssembly with emscripten and hosted by `wasmtime`. **The draft needed no
new patch** — see below.

## The fidelity guarantee (two gates)

1. `tests/test_fidelity.py`, inherited and inviolable: the patched
   production sim and a pristine build of the vendored source run side by
   side for thousands of ticks of identical random actions, and every
   observation and reward byte must match.
2. `tests/test_loadout.py`'s **zero-loadout identity**: 500 ticks with an
   all-neutral `apply_loadout` for all ten heroes must produce byte-identical
   observations, rewards and state digests to the same run with no
   `apply_loadout` call at all. That is what "keep the Puffer fidelity gate
   for the un-drafted mode" means operationally, and it is what would catch
   an `apply_loadout` that accidentally spawned, allocated or drew RNG.

Sim-touching changes must keep both green. The tests are inviolable — fix
the code, never the test.

## Why the draft needs no patch

Every stat the catalog touches (`base_health`, `base_mana`, `base_damage`,
`basic_attack_cd`, `move_speed`, and the three per-level gains) is already a
field of upstream's `struct Entity` that `spawn_player` and the level-up path
re-read on **every respawn and every level-up**. Writing the block once,
after `c_reset` and before the first `c_step`, therefore makes a loadout
permanent for the whole match for free. Details, and the obs-byte reason
`move_speed` is floored at 1.0, in [docs/DRAFT.md](docs/DRAFT.md).

## The game

- **Seats**: 6, three per team, one hero each — pids 0, 1, 2 (radiant
  support/assassin/burst) and 5, 6, 7 (dire ditto). Pids 3, 4, 8, 9 (both
  tanks and carries) are **house heroes** the server drives itself off the
  vendored pretrained network, so all ten heroes stay alive and the map plays
  like a MOBA instead of a four-hole skirmish. `num_agents` inside the wasm
  stays 10; nothing about the sim changes.
- **Map**: the upstream 128×128 Dota-shaped map, six creep lanes, 24 towers,
  18 neutral camps. Lane towers hit for 110–175 per shot at a hero's own
  scan radius; both Ancients have 4500 HP and deal no damage.
- **End**: an Ancient falls, or the 6000-tick cap with a remaining
  Ancient-health tiebreak (equal health is a draw). `scores` are 1.0 / 0.5 /
  0.0 per seat, zero-sum, higher is better.
- **Variants**: `draft` (the game) and `nodraft` (the Puffer-fidelity mode:
  no draft turn, no loadout applied at all).

## Quickstart

```sh
uv sync
bash sim/build_sim.sh      # sim wasm (requires emcc; brew install emscripten)
bash sim/build_brain.sh    # pretrained-brain wasm (requires xxd)
bash sim/build_viewer.sh   # browser replay viewer (downloads pinned raylib)
uv run pytest              # full suite, includes both fidelity gates
```

Run a local containerized episode (Docker required):

```sh
docker build --platform=linux/amd64 -t cogame-derks-gym:local .
bash tools/ci/docker_smoke.sh cogame-derks-gym:local
```

Watch a recorded replay: open the static bundle
(`viewer/dist/index.html?replay=<url>`), or
`COGAME_LOAD_REPLAY_URI=file://<replay> python -m cogame_derks_gym.server`
and browse to `/client/replay`.

## Policies

One entrypoint, one image, env-switched:

```
python -m players.derk_player
  PLAYER_PROMPT=derk-drafter-v1     LLM draft + pretrained micro   (champion)
  PLAYER_PROMPT=derk-metagamer-v1   ditto, counter-drafting prompt (champion)
  PLAYER_SCRIPTED=puffer-forge      fixed draft table + pretrained micro (default)
  PLAYER_SCRIPTED=lane-brawler      stat-derived draft + the FSM lane-push bot
```

Both unset plays `puffer-forge`, so a bare `docker run` works. Both set:
`PLAYER_PROMPT` wins and says so. An unknown name exits 2 with the legal
list — a typo must fail loudly, not silently ship a different policy.

The LLM path is **degrade-never-hang**: one Anthropic call per episode with a
20 s timeout, one retry at temperature 0, then the `puffer-forge` draft rule.
Worst case 40 s, inside the server's 45 s draft deadline. No
`ANTHROPIC_API_KEY` means no call at all.

Also inherited, for reference and tests: `players/baseline_player.py` (the
raw pretrained policy — its `MobaBrain` is the micro layer and the house
heroes' brain), `players/scripted_player.py`, `players/random_player.py`.

## Protocol (for policy authors)

[docs/PROTOCOL.md](docs/PROTOCOL.md) is the contract;
[docs/DRAFT.md](docs/DRAFT.md) is the catalog. Short version: connect to
`COWORLD_PLAYER_WS_URL`, answer one `{"phase": "draft"}` message with
`{"phase": "draft", "picks": [{"arm": ..., "tail": ..., "misc": ...}]}`, then
answer `{"tick", "obs": [base64 510B]}` with `{"tick", "actions": [[6 ints]]}`
in the upstream MultiDiscrete `[7,7,3,2,2,2]` space. Late, missing or
malformed replies play no-op; a missing draft reply plays the neutral
loadout. The observation and action encodings are transported verbatim from
upstream.

## Repo layout

- `vendor/upstream/` — byte-pristine vendored upstream source (never edit;
  see `vendor/UPSTREAM.md`); changes are patch files in `sim/patches/`
  (rationale in `vendor/PATCHES.md`)
- `sim/` — patches, wasm shims (`shim.c`, `brain_shim.c`, `viewer_main.c`),
  the shared headers `shim_common.h` (env config) and `loadout_common.h`
  (the draft's stat application, shared by the server shim and the viewer so
  they cannot drift), and the build scripts
- `server/cogame_derks_gym/` — config, the catalog, the draft turn, the
  event vocabulary, the lockstep engine, the house heroes, the websocket
  server, the v2 replay writer/reader, the wasmtime sim host; entry
  `python -m cogame_derks_gym.server`
- `players/` — see above
- `viewer/` — the starter's `index.html` plus the appended `#derk-*` game
  block (`derk_chrome.css`, `derk_chrome.js`, `derk_items.svg`)
- `tests/` — the full suite; `test_fidelity.py` and `test_loadout.py` are
  the gates
- `Dockerfile`, `compose.yaml`, `coworld_manifest_template.json` — Coworld
  packaging

## Porting other PufferLib envs

[docs/PORTING.md](docs/PORTING.md) is the recipe, inherited from
cogame-moba; this repo is now also a worked example of *modding* such a port
without breaking its fidelity gate.

## Attribution

The simulation, renderer, map assets, network weights and puffernet
inference library are from
[PufferAI/PufferLib](https://github.com/PufferAI/PufferLib), pinned at commit
`c5d3c637446047a6efbcaa74c039c5295d201ab0`, MIT license
(`vendor/LICENSE-pufferlib`). The draft, the Coworld packaging and the viewer
chrome are this repo's.
