# Porting a PufferLib Ocean env to a Coworld

**Who this is for:** a coding agent starting fresh with this repository checked out, tasked with
turning another PufferLib Ocean environment (or any small C RL env with pretrained policies) into
a Coworld like this one. You are assumed to be able to read this repo's code and docs, run its
tests, and reach the public internet (GitHub, puffer.ai, softmax.com). Nothing else is assumed.

## Motivation

RL policies are trained against an exact environment: every byte of the observation encoding,
every quirk of the physics, every scripted-NPC behavior is baked into the weights. If you want
those policies to compete on the Coworld platform — Softmax's hosted tournament system, where
policies ship as Docker containers and play websocket-refereed episodes — the port must reproduce
the training environment *bit-exactly*, or the policies degrade in ways that are invisible until
they lose. Hand-porting a sim can never prove bit-exactness. This repo demonstrates a strategy
that can: vendor the original C source unmodified, compile it to WebAssembly, and prove with a
byte-comparison test that the served environment equals the trained-on environment. This document
is the recipe for repeating that against a different env.

## Introduction

The pieces you will build, and how they relate (each has a working example in this repo):

- **PufferLib Ocean env** — PufferLib (github.com/PufferAI/PufferLib, MIT) ships its "Ocean"
  suite of small C environments, each typically one header (`ocean/<env>/<env>.h`) containing
  BOTH the simulation and a raylib-based renderer, plus `binding.c` (Python glue),
  a `config/<env>.ini` with training hyperparameters, sometimes compiled-in assets
  (`xxd -i` dumps), pretrained weights (`resources/<env>/<env>_weights.bin`), and `src/puffernet.h`
  — a dependency-free C inference library that runs those weights.
- **Coworld** — a packaged game on the Softmax platform: a game Docker image implementing a
  websocket "referee" contract, player Docker images that connect to it, a manifest describing
  all of it, and platform machinery (certification, leagues, replays). The `coworld` Python
  package (`uv add coworld`) provides build/certify/upload/run tooling; its Cookbook documents
  the contract (fetch it via the package or softmax.com docs).
- **The wasm sim core** — the vendored env compiled with emscripten to a standalone
  WebAssembly module, hosted in-process by the Python game server through `wasmtime`. Wasm is
  what makes bit-exactness *portable*: one binary carries its own libc (musl `rand()`), one
  float semantics, identical on macOS dev machines, CI, hosted k8s, and in browsers.
- **The fidelity gate** — a test that compiles the env twice (untouched upstream vs. your
  patched copy), drives both with identical seeds and action logs, and asserts byte-identical
  observation and reward streams. It is the acceptance criterion for the whole port.
- **The lockstep server** — `server/cogame_derks_gym/` here: async episode engine + aiohttp
  websocket server implementing the Coworld game contract. Most of it is game-agnostic.
- **Replay-as-recording** — because the sim is deterministic, a replay is just
  header (config + seed + player names) + per-tick action log; any holder of the same wasm can
  re-simulate playback exactly. The browser viewer is the env's own raylib renderer compiled
  with emscripten, re-simulating from the action log.

## The process

Work the stages in order; each has a gate. The repo's own history followed exactly this shape,
and `docs/plans/` contains the design and implementation plans it was built from — read both
before starting, then write the equivalents for your env.

### Stage 1: Research the env until you can answer these questions

Read the actual upstream source (clone and pin a commit — you will vendor from it). Do not
trust env descriptions on puffer.ai or in READMEs; they are marketing-grade. Answer, with
file:line evidence:

1. **Spaces**: exact per-agent obs shape/dtype/layout and action space. Treat encodings as
   opaque byte contracts. Expect quirks (this env has an obs-write stride bug that the
   pretrained weights were *trained on* — see `docs/plans/2026-08-01-cogame-derks-gym-design.md`).
   **Never fix upstream quirks. Fidelity beats hygiene, always.**
2. **Config**: which env parameters the training config (`config/<env>.ini` + `binding.c`
   defaults) actually used. Your port must serve those exact values. Watch for flags that
   change who is policy-controlled (here: `script_opponents` — training used scripted
   opponents for half the agents; serving all seats required flipping it, a deliberate,
   documented decision).
3. **Sim/render coupling**: where the renderer starts in the header, whether the sim proper
   calls raylib or only libc/libm, and whether assets are compiled in or loaded from disk.
4. **RNG and determinism**: every randomness source (usually libc `rand()` — which is exactly
   why wasm matters: glibc/musl/macOS produce different streams from the same seed), whether
   anything reads clocks or uses threads, whether the env seeds itself (this one never called
   `srand` at all — every run replayed the default stream).
5. **Episode semantics**: how the env signals termination. Ocean envs often never write
   `terminals` and silently auto-reset inside `step` on the win condition; you will patch that
   into an exposed done/winner flag. Check for the absence of a tick limit (add truncation
   server-side, never in the sim).
6. **Weights and inference**: are pretrained weights shipped, what architecture, and does the
   demo binary (`ocean/<env>/<env>.c`) show the exact puffernet calls and obs preprocessing?
   That demo is your baseline player's specification.

### Stage 2: Decide, in writing, before building

Write a short design doc (pattern: `docs/plans/2026-08-01-cogame-derks-gym-design.md`) fixing:
seat mapping (N agents → how many Coworld seats; offer variants if both per-agent and per-team
seats make sense), pacing (this repo chose pure lockstep, as-fast-as-possible, browser is
replay-only — revisit only if the env is human-playable at real-time), repo name/org, and the
patch list with per-patch rationale. Get the human to confirm the decisions. Then write the
implementation plan (pattern: `docs/plans/2026-08-01-cogame-derks-gym-implementation.md`).

### Stage 3: Vendor + patch + fidelity gate (do this before ANY server code)

Mirror this repo's structure exactly; it encodes the discipline:

- `vendor/upstream/` — byte-pristine files from the pinned commit; `vendor/UPSTREAM.md` with
  commit hash + sha256 per file; `vendor/LICENSE-pufferlib`. Never edit these files.
- `sim/patches/NNNN-*.patch` — applied at build time by `sim/apply_patches.sh` into `build/`.
  The canonical minimal patch set, in order: **render guard** (wrap renderer + raylib include
  in `#ifdef <ENV>_RENDER`; needed even for the "pristine" build, so both builds share it),
  **seeding** (`srand(seed)` at init), **done flag** (expose done/winner, skip internal
  auto-reset). Every patch documented in `vendor/PATCHES.md`. Anything beyond these three
  demands suspicion.
- `sim/shim.c` — exports over the env struct: init (setting the *training-config* values,
  each with a citation comment), step, reset, buffer pointers, done/winner/tick, and stat
  accessors for scoring. Build with
  `emcc -O2 -sSTANDALONE_WASM --no-entry -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=<size> -sABORTING_MALLOC=1`
  (`ABORTING_MALLOC` is non-negotiable: under wasm, a failed malloc otherwise NULL-writes
  silently and corrupts state instead of crashing loudly). Size memory from the env's real
  allocations (this env lazily builds a 256 MB pathfinding cache).
- Python host (`server/cogame_derks_gym/sim.py` is the model): wasmtime + WASI, call the module's
  `_initialize` export before anything else, stub `env::emscripten_notify_memory_growth`,
  cache compiled modules per path, return obs/reward *copies*, and validate actions at this
  boundary: the host *raises* on NaN/Inf/bad shape and clamps finite out-of-range values
  (matching the sim's C `(int)` cast semantics); the graceful degrade-to-NOOP for misbehaving
  players lives one layer up, in the episode engine — keep that split.
- **The fidelity test** (`tests/test_fidelity.py` is the model): pristine build (render-guard
  patch only) vs. fully patched build, same seed, same multi-thousand-tick random action log,
  assert byte-identical obs + rewards every tick, and assert the tick count floor so the gate
  can't silently shrink if the action stream changes. This test is the permanent CI gate.
  If it fails, a patch changed physics: fix the patch, never the test.

Gate: fidelity test green. Nothing else starts before this.

### Stage 4: Server, players, replay, viewer

- **Server**: `server/cogame_derks_gym/{config,defaults,engine,server,replay,uris}.py` are written
  to be reused — the game-specific surface is: obs size, action dims and NOOP, agent count,
  seat→agent mapping, scoring/results fields, and the stat accessors. The engine (lockstep,
  per-tick deadline, degrade-to-NOOP, dead-seat strike rule so a silent client can't burn
  wall-clock) and the contract layer (`COGAME_CONFIG_URI`/`RESULTS_URI`/`SAVE_REPLAY_URI`/
  `PLAYER_FAILURE_URI`/`LOAD_REPLAY_URI`/`HOST`/`PORT`, `/player?slot=&token=` auth) transfer
  unchanged in shape. The certifier also probes browser surfaces the lockstep design doesn't
  otherwise need: `GET /client/player?slot=&token=` (token-checked), `GET /client/global`, a
  `/global` websocket that must emit a first message, and — on certifier versions predating
  the static-viewer skip — a legacy `/replay` websocket in replay mode. Budget for a minimal
  honest implementation of these (this repo's: broadcast-only status feed with fire-and-forget
  sends so a slow viewer can never stall the episode, and a bounded-send done message).
  Platform contract details that are easy to get wrong:
  - The player-failure payload is parsed by the platform with a **closed schema**: exactly
    `{"message", "failed_policy_index"}`, nothing else (see the docstring in
    `server/cogame_derks_gym/server.py`).
  - `results.json` should carry the columnar keys Coworld consumers expect (`names`,
    `scores`, ...) and any extra keys must be declared in the manifest's results schema —
    Coworld schemas are closed; undeclared keys are dropped.
  - Player containers read their websocket URL from `COWORLD_PLAYER_WS_URL` (legacy alias
    `COGAMES_ENGINE_WS_URL`).
- **Replay**: copy the format approach (`server/cogame_derks_gym/replay.py`): magic + version +
  JSON header (config incl. seed, **player names — the platform's static-viewer contract
  requires names to live in the replay bytes**, sim wasm hash, result) + packed per-tick
  actions. The mandatory test: record a real episode, re-simulate from the replay alone on a
  fresh sim, assert identical final tick/winner/obs bytes. Strengthen it with a cheap
  end-state digest (this repo: FNV-1a over entity positions/health, computed identically in
  `sim/shim_common.h` and exported by both the server sim and the viewer core) asserted
  equal between the live run and the re-sim — it catches divergences that happen to preserve
  the winner.
- **Baseline player**: if upstream ships weights, do not re-implement inference — compile
  upstream's own `puffernet.h` + the weights into a second wasm module (`sim/brain_shim.c`
  pattern), mirroring the demo binary's preprocessing and sampling exactly (puffernet
  *samples* from the softmax via `rand()`; it does not argmax). This is both your
  certification fixture player and the live proof that trained policies survive the port.
  Caveat to write down where the weights load: the wasm module's action stream matches a
  native upstream run only under the same libc — glibc/macOS `rand()` differ from musl, so a
  native reference trace diverges in sampled actions even with bit-exact logits; that is a
  libc difference, not a port bug.
- **Scripted reference bot** (optional but valuable — it exercises the obs as a *consumer*):
  before writing one, derive from `compute_observations` exactly which obs bytes are reliably
  decodable (this env's crop writes overlap; only the 121 tile ids survive) and prove it with
  a test against a live sim. Any static data you embed (map grid, waypoint tables, tower
  positions) must have a **tripwire test** that regex-parses the vendored C source and asserts
  byte-equality with the embedded constants — a future re-vendor must fail tests, never
  silently desync the bot (`tests/test_scripted.py` shows the pattern).
- **Viewer**: compile the env's own renderer with emscripten, driven by the replay action log
  instead of live input. Upstream's `build.sh --web` (in the pinned clone, not vendored) is
  the recipe to follow — note it links the **prebuilt** `raylib-X.Y_webassembly.zip` from
  raylib's GitHub releases rather than building raylib from source; sha256-pin that zip and
  stamp the pin into the build cache so a pin bump invalidates it. Declare the bundle in the
  manifest as a static replay viewer (`replay_viewer.bundle` + a build-hook script; see
  `coworld_manifest_template.json` and `tools/build_replay_viewer.sh`) so hosted replay views
  don't boot a container per view. Build the viewer core a second time without the renderer
  (`ENVIRONMENT=node`) so CI can assert the re-sim reaches the recorded winner and digest
  headlessly.

### Stage 5: Package, certify, publish

Single Docker image (game + players as different entrypoints), runtime `--platform=linux/amd64`.
Packaging facts that cost time to discover (all embodied in this repo's `Dockerfile`,
`coworld_manifest_template.json`, and `.github/workflows/`):

- Run the emscripten build stage on `$BUILDPLATFORM` — wasm output is arch-independent, so an
  ARM host needn't emulate x86 to compile it; only the runtime stage is amd64. The build stage
  needs `xxd` (weights embedding) and network for the sha-pinned raylib zip (its own cacheable
  layer).
- Manifest gotchas: version strings must be plain `\d+.\d+.\d+` semver (suffixes rejected);
  `episode_timeout_minutes` lives at the manifest top level, not under `game`; the
  certification fixture should cap `max_ticks` modestly — a self-play mirror match can
  stalemate, and a tick-cap draw with a score tiebreak is a valid certified outcome; declare
  the results schema closed and exactly matching what the server emits (a smoke script that
  asserts the exact key set is a cheap tripwire).
- Then locally: `uv run coworld build`, `uv run coworld certify`, and *watch one replay with
  your own eyes* — the certifier probes routes, it cannot tell you the viewer renders
  nonsense. Note the certifier's source-resolution step checks the manifest `source_url`
  exists on GitHub, so create/push the repo before the final certify.
- CI runs the fidelity gate on every push. For the upload workflow, mirror
  the `upload-coworld` job in `.github/workflows/ci.yml` (gated on the green
  test + docker-smoke jobs): version = highest existing registry row with the
  patch bumped (`tools/ci/next_coworld_version.py`) — **never** `coworld next-version`, whose
  canonical-based logic wedges on orphan rows from half-failed publishes; a guard step that
  warns-and-skips when the `SOFTMAX_TOKEN` secret is absent, so the workflow can merge before
  the credential exists; and know the bootstrap ordering — the version picker fails while the
  registry has zero rows, so the first upload is a one-time local `coworld upload-coworld`
  (reusing the certify proof) or a workflow_dispatch with an explicit version.
- Hosted upload to Softmax, league creation, and player submissions are a separate,
  human-approved step (this repo's plan records that sequence in its Phase 6 section:
  platform-ladder league seed → divisions → ladder settings → players).

### Verification discipline (applies to every stage)

Whatever process you use to write the code, verify it in two independent passes per stage:
one pass comparing implementation to the written plan (read the code, not the reports), and
one pass reviewing quality — with special attention to: every await on player input bounded
by a deadline, artifact writes independent and retried, and no path where a malicious or
crashed player can stall an episode. The failure mode these passes exist to catch is the
implementer's report sounding complete while the code is subtly not.

## Glossary

- **Coworld** — a packaged competitive game on the Softmax platform: game + player Docker
  images described by a manifest, refereed episodes, leagues, hosted replays.
- **Coworld game contract** — the runtime interface a game container implements: config in and
  results/replay/failure artifacts out via `COGAME_*` environment-variable URIs, player
  websockets at `/player?slot=&token=`, replay serving via `COGAME_LOAD_REPLAY_URI`.
- **PufferLib / Ocean** — PufferLib is an RL training library (github.com/PufferAI/PufferLib);
  Ocean is its suite of small single-header C environments.
- **puffernet** — PufferLib's dependency-free C inference library (`src/puffernet.h`) that runs
  the shipped pretrained weight files.
- **fidelity gate** — this port strategy's acceptance test: pristine vs. patched builds of the
  vendored sim must produce byte-identical obs/reward streams under identical seeds and actions.
- **pristine build** — the vendored env compiled with only the render-guard patch (the minimum
  that compiles headless); the comparison baseline for the fidelity gate.
- **lockstep** — episode pacing where the server waits (bounded by a deadline) for every seat's
  action each tick; no real-time pacing.
- **seat** — one player slot in a Coworld episode; a seat may control one or several env agents
  (variants can offer both mappings).
- **strike rule** — the engine policy that a seat failing to answer N consecutive ticks is
  marked dead and stops consuming the per-tick deadline (revivable on reconnect).
- **static replay viewer bundle** — a self-contained browser bundle declared in the manifest
  that renders replays client-side (here: the env's raylib renderer compiled to wasm,
  re-simulating from the replay's action log), replacing per-view server boots.
- **certification** — the Coworld tooling's automated local proof that a manifest's game runs an
  episode, writes valid artifacts, and serves replays (`uv run coworld certify`).
- **manifest** — `coworld_manifest.json`: the package descriptor listing images, runnables,
  variants, certification fixture, results schema, and viewer bundle.
- **STANDALONE_WASM** — emscripten mode producing a self-contained wasm module runnable under
  a non-browser host (wasmtime) via WASI, no JavaScript required.
- **wasmtime** — the WebAssembly runtime used by the Python server to host the sim in-process.
