# Working in this repo

Conventions for agents (and humans) making changes here. The design and
implementation history live in `docs/plans/`; the porting recipe this repo
demonstrates is `docs/PORTING.md`.

## The three inviolable rules

1. **`vendor/upstream/` is byte-pristine.** It is the vendored PufferLib
   source at the pinned commit (`vendor/UPSTREAM.md` records the commit and
   per-file sha256s). Never edit anything under it. All source changes are
   patch files in `sim/patches/`, applied at build time into `build/src-*`
   by `sim/apply_patches.sh`, each documented in `vendor/PATCHES.md`.
2. **The fidelity gates are inviolable.** `tests/test_fidelity.py` proves
   the patched production sim is byte-identical (obs + rewards, thousands of
   ticks) to a pristine build of the vendored source.
   `tests/test_loadout.py`'s zero-loadout identity proves an all-neutral
   `apply_loadout` for all ten heroes is a byte-identical no-op (obs,
   rewards and `state_digest`) against a run with no `apply_loadout` call at
   all — that is the un-drafted variant's guarantee, and it is what catches
   an `apply_loadout` that starts spawning, allocating or drawing RNG. Both
   must pass after every sim-touching change. If one fails, the code changed
   physics — fix the code, never the test. Weakening or skipping either is a
   failed task, not a passing build.
3. **The catalog and its deltas are a closed contract.**
   `server/cogame_derks_gym/catalog.py`, the manifest `config_schema`
   (the `catalog_version` enum and the baked catalog sha256), `docs/DRAFT.md`
   and the `item-<catalog id>` symbols in `viewer/derk_items.svg` change
   **together** or the tripwire test (`tests/test_catalog.py`) fails. A
   catalog edit that lands in one place only silently re-interprets every
   existing replay.

## Where things live

- Env-physics config values (vision_range, agent_speed, reward weights)
  mirror upstream `config/moba.ini` + `binding.c` and live in
  `sim/shim_common.h` (`moba_configure` — shared by the server shim and
  the viewer so they can never drift). The draft's stat application lives in
  `sim/loadout_common.h` for the same reason: `sim/shim.c`'s `apply_loadout`
  and `sim/viewer_main.c`'s `sim_fresh()` call the SAME function at the same
  point (after `c_reset`, before the first `c_step`). Server-contract
  defaults (max_ticks, the no-op action, the seat -> hero map, each hero's
  base stat block) live in `server/cogame_derks_gym/defaults.py`; the items,
  deltas and clamps live in `server/cogame_derks_gym/catalog.py`. Keep the
  upstream citations next to the values.
- The seat -> hero map is FIXED: `SEAT_HERO_PIDS = (0, 1, 2, 5, 6, 7)` and
  `HOUSE_HERO_PIDS = (3, 4, 8, 9)`. `num_agents` is 6 everywhere (config,
  every manifest variant's `game_config`, the certification fixture and
  `tools/ci/docker_smoke.sh`); the sim's own hero count stays 10.
- The 510-byte obs and `[7,7,3,2,2,2]` action encodings are opaque
  contracts — transport them verbatim, never re-encode.
- Results keys are a CLOSED schema: `server/cogame_derks_gym/server.py`
  `_results_doc` and the manifest template `results_schema` must list
  exactly the same keys. Adding a results field means updating both (and
  `tools/ci/docker_smoke.sh`'s expected-keys set). Same rule for the
  `end_reason` enum (four values) and the draft's `fallback_cause` enum
  (six values -- every one of them reachable; see draft.py).
- Every string that lands in the replay (a seat's `note` above all) is
  truncated on **Unicode-scalar (rune) boundaries**, never bytes: a
  byte-boundary truncation produces a replay that renders in a browser and
  fails a strict JSON parser.

## Build pipeline

```sh
bash sim/apply_patches.sh   # vendor + patches -> build/src-{pristine,patched}
bash sim/build_sim.sh       # -> build/derk_sim.wasm, build/derk_sim_pristine.wasm
bash sim/build_brain.sh     # -> build/derk_brain.wasm (needs xxd)
bash sim/build_viewer.sh    # -> viewer/dist/ + build/viewer_core.* (downloads pinned raylib)
                            #    (also copies derk_chrome.css/.js + derk_items.svg)
```

`build/`, `dist/`, and `viewer/dist/` are gitignored build outputs. The
Dockerfile runs the three build scripts in its wasm-builder stage
(`apply_patches.sh` runs inside `build_sim.sh` and `build_viewer.sh`;
`build_brain.sh` compiles the pristine vendor tree directly); the emcc
pin (6.0.5) is recorded in `vendor/UPSTREAM.md` and must stay in sync
across the Dockerfile and `.github/workflows/ci.yml`.

## Testing and review discipline

- `uv run pytest` runs the full suite (fast — slow-marked tests are
  included in CI too). Run it before every commit that touches
  sim/server/players.
- TDD for behavior changes: failing test first, then the implementation.
- Commit in small, single-purpose units with pathspec `git add` (never
  `git add -A` in a shared tree).
- Packaging changes (Dockerfile, compose, manifest template) must keep
  `docker build` + `tools/ci/docker_smoke.sh` and
  `uv run coworld build --project . --version <v>` +
  `uv run coworld certify dist/coworld_manifest.json` green.

## Coworld platform contract

The server implements the Coworld runtime contract (`COGAME_*` env vars,
`/player` + `/global` websockets, `/client/*` pages, replay mode) — see
`docs/PROTOCOL.md` and the certifier probes in
`coworld.runner.runner.run_episode_containers`. The manifest template
declares a static replay viewer bundle (`static-replay-viewer`, built by
`tools/build_replay_viewer.sh` from the Dockerfile's wasm-builder stage),
which replaces the legacy replay-route certification probes; the server
still serves `/client/replay` for local viewing.

Uploads: **publishing is `.github/workflows/coworld-release.yml`'s job,
not CI's.** The full release — build -> certify -> upload policies ->
upload-coworld -> secret put, in that load-bearing order — is dispatched
manually (`gh workflow run coworld-release.yml -f version=X.Y.Z`), and
`tools/ci/policies.json` is the default policy set it uploads. League
submission is `.github/workflows/coworld-submit.yml`.

`ci.yml` still carries the starter's `upload-coworld` job, but it is now
**off unless the `UPLOAD_REQUIRED` repo variable is exactly `true`**: a
push-published version raced the release chain, took the number the
release then 409ed on, and became canonical while never having been
locally certified, having no policy versions and no coworld secret
(derks-gym 0.1.1, 2026-08-28). If it is ever turned back on, its version
is the highest existing registry row patch-bumped via
`tools/ci/next_coworld_version.py` — never `coworld next-version`, see
that picker's docstring.
