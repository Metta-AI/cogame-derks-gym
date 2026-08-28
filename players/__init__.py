"""cogame-derks-gym player clients.

- ``players.client``: reusable async websocket harness (URL from env, the
  optional draft turn, obs decode, action send, bounded reconnects).
- ``python -m players.derk_player``: THE policy entrypoint — one image,
  env-switched (``PLAYER_PROMPT`` = LLM draft + puffernet micro,
  ``PLAYER_SCRIPTED`` = a scripted draft rule + its micro).
- ``python -m players.baseline_player``: the upstream pretrained policy
  via the wasm-compiled puffernet brain. Its ``MobaBrain`` is also the
  micro layer of ``derk_player`` and the game server's house heroes.
- ``python -m players.scripted_player``: hand-coded lane-push reference
  bot (finite-state, deterministic, no wasm dependency), the micro layer
  of the ``lane-brawler`` baseline.
- ``python -m players.random_player``: uniform-random policy.
"""
