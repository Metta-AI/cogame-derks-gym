# Patch set applied to the vendored sim

Files under `vendor/upstream/` are byte-pristine (see `vendor/UPSTREAM.md`).
All modifications are the patch files in `sim/patches/`, applied at build time
by `sim/apply_patches.sh` into `build/`:

- `build/src-pristine/` gets **0001 only** — the minimum required to compile
  the sim without raylib. This tree is the reference build for the fidelity
  test (`tests/test_fidelity.py`).
- `build/src-patched/` gets **all** patches. This is the production sim.

**Invariant:** patches must keep in-episode physics byte-identical. The
fidelity test drives both builds with identical seed + action logs and
requires byte-identical obs and reward streams. A patch that breaks it is
rejected.

## 0001-render-guard.patch

Guards `#include "raylib.h"` (+ the render-only `GLSL_VERSION` block) and the
entire renderer section of `moba.h` (everything from the `// Raylib client`
comment at former line 1970 through end of file: `COLORS[]`, `MapRenderer`,
`GameRenderer`, `c_render`, `close_game_renderer`) behind `#ifdef MOBA_RENDER`.

- Rationale: the server-side sim build must compile to STANDALONE_WASM with no
  raylib. Upstream's sim half (lines 1–1969) uses only libc/libm; the raylib
  include is unconditional, so without this guard nothing compiles headless.
- No sim lines change. Upstream's `c_close`/`free_allocated_moba` free only
  sim allocations (no renderer frees), so no in-function splits were needed.
  The `GameRenderer* client` pointer in `struct MOBA` stays: `GameRenderer` is
  forward-declared in the sim half and only ever dereferenced in the renderer.
- The viewer build (Phase 4) compiles the same tree with `-DMOBA_RENDER`.

## 0002-seed.patch

Adds `unsigned int seed;` to `struct MOBA` and calls `srand(env->seed);` at
the top of `init_moba()` — before the `CachedRNG` table is filled from
`rand()` and before any spawn jitter draws.

- Rationale: upstream 4.0 never calls `srand`, so every process replays the
  libc default stream (equivalent to seed 1). Upstream 3.0's
  `env_binding.h` did `srand(seed)` at env init; this restores that behavior
  so a league can run varied episodes. The seed value is provided by
  `sim/shim.c`'s `moba_init(seed, ...)`.
- In-episode physics are unchanged: for seed == 1 the stream is identical to
  the unseeded default (musl `srand(s)` stores `s - 1`; initial state is 0),
  which is exactly what the fidelity test relies on.

## 0003-done-flag.patch

Adds `int done; int winner;` to `struct MOBA`. In `c_step`'s win branch
(ancient death → `do_reset`), sets `env->done = 1` and `env->winner`
(0 = radiant, 1 = dire), keeps `add_log`, and **skips the internal
`c_reset(env)` auto-reset**.

- Rationale: upstream never writes `terminals` and silently restarts the
  episode inside `c_step` when an ancient dies. The Coworld server needs to
  detect the end of the episode and score it; after `done`, the server stops
  stepping. Affects only post-win behavior — every tick up to and including
  the winning tick is byte-identical (rewards for the winning tick are
  computed in `step_players`, before the win check).
- `done`/`winner` are cleared by the shim's `moba_reset()`, not by `c_reset`
  (keeps the patch surface minimal).
- If both ancients die on the same tick (upstream sets both victory flags),
  the tie deliberately goes to dire (`winner = dire_victory ? 1 : 0`).

## 0004-fault-flag.patch

Converts upstream's four **in-episode** debug-guard `exit()` calls into a
recorded fault: a file-scope `static int moba_fault_code` (site codes 1-4:
spawn_player move failure, scanned-target dist > 20, tower respawn move
failure, missed-reset invariant) set where upstream exited, bailing out of
the local operation only. Also adds `env->tick` to the three "glitch state"
printfs so the warnings are locatable in a replay.

- Rationale: `exit()` inside the wasm raises an ExitTrap in the host and
  kills the episode process — results and replay are lost. With the flag,
  `sim/shim.c` exports `moba_fault()`, the engine polls it every tick and
  ends the episode cleanly with `end_reason: "sim_fault"` (no winner, draw
  scores), writing results and the partial replay.
- The init-time guards (`game map load`, line ~1620) keep their `exit(1)`:
  failing to even construct the env is a startup failure, not an episode
  to salvage.
- In-episode physics are unchanged unless a guard trips — at which point
  upstream would have aborted the process entirely. The fidelity gate is
  unaffected: the pristine build keeps upstream's `exit()` calls, and the
  gate's action stream never trips a guard (`moba_fault()` stays 0, which
  `tests/test_engine.py::test_real_sim_fault_export_is_zero` pins).
- File-scope flag rather than a `MOBA` struct field because `spawn_player`
  receives no env pointer; `static` keeps it TU-local (each shim includes
  `moba.h` once).
