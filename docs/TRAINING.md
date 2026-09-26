# Headless training

`server/cogame_derks_gym/training.py` exposes `TrainingMatch` over the same
compiled WebAssembly simulator as the hosted game. Build both artifacts first:

```sh
bash sim/build_sim.sh
bash sim/build_brain.sh
```

`TrainingMatch(seat, max_ticks=6000, draft_enabled=True)` trains one of the
six player seats. `reset(seed)` returns the first `TrainingStep`; `step(action)`
advances the draft or one simulator tick. A step contains a 513-float
observation, an 87-entry legal-action mask, the seat's dense reward, a terminal
flag, and a terminal score. It raises on invalid actions or a simulator fault.

The seven action heads have sizes `[64, 7, 7, 3, 2, 2, 2]`. Head zero chooses
the catalog loadout in base-four slot order: arm, tail, misc. During the draft,
the other six heads accept only `[3, 3, 0, 0, 0, 0]`. During play, head zero
accepts only zero and the other heads carry the upstream Puffer MOBA actions.
The first 510 observation values are the player-visible bytes divided by 255.
Three values give draft phase, seat index divided by five, and selected loadout
divided by 63. Draft observations zero the simulator bytes because the hosted
draft cannot inspect the unrevealed map. With `draft_enabled=False`, the match
starts directly in play and never calls `apply_loadout`, preserving the
upstream no-draft physics.

Other seats use the game's puffer-forge loadouts and isolated baseline brains;
the four house heroes use the hosted house policy. Terminal score is 1 for a
win, 0 for a loss, and 0.5 for a draw. At the configured tick cap, ancient
health breaks the tie, as in the hosted game. The adapter does not simulate
WebSocket deadlines, player disconnects, or LLM draft fallbacks.

Metta's native Puffer recipe is `recipes.external.derks.train`; the legacy
Metta RL recipe is `recipes.external.derks_metta_rl.train`. Both use this
headless adapter and fingerprint its Python source and WebAssembly artifacts.
The adapter runs on CPU; native Puffer training requires CUDA. The recipes
were merged through Metta #24628. Metta `fabric=true` exports a frozen policy
bundle. Build its ordinary player image from that bundle and the Metta checkout
used for training:

```sh
docker buildx build --platform linux/amd64 --load \
  --build-context fabric=/path/to/fabric-at-bea92fff \
  --build-context metta=/path/to/metta \
  --build-context bundle=/path/to/train_dir/derks_native/policy \
  -f Dockerfile.trained -t derks-frozen:local .
```

The image runs `players.trained_player` against the same `/player` WebSocket as
every other policy. It reads the game's accepted draft result before encoding
tick observations, so a timed-out draft uses the actual neutral loadout.
