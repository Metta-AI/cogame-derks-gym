"""The four house heroes, driven in-process off the vendored network.

Upstream Puffer MOBA spawns ten heroes; this game seats six of them
(``defaults.SEAT_HERO_PIDS``). The other four — radiant tank/carry and
dire tank/carry — are **house heroes**: the game server plays them itself
through ``players.baseline_player.MobaBrain`` (the vendored pretrained
puffernet compiled to wasm) on the neutral all-zero loadout, so all ten
heroes stay alive and the map still plays like a MOBA instead of a
four-hole skirmish.

Determinism: the brain module has ONE shared ``rand()`` stream and
puffernet *samples* from the softmax, so the forward calls must happen in
a fixed order. This class always iterates its pids in ascending order,
and the engine calls it once per tick before the seat batch.

Degrade, never hang: if the brain wasm is missing or a forward call
traps, construction / the call raises, the engine logs it once and the
house heroes play NOOP for the rest of the episode. The seats keep
playing.
"""

from __future__ import annotations

import sys

import numpy as np

from . import defaults

# Matches players.baseline_player.DEFAULT_SEED (upstream's unseeded libc
# rand stream). Fixed rather than configurable: the house heroes are part
# of the environment, not a policy anyone tunes.
DEFAULT_HOUSE_SEED = 1


class HouseHeroes:
    """``actions(obs) -> {pid: [6 ints]}`` for the four house heroes."""

    def __init__(self, pids: tuple[int, ...] = defaults.HOUSE_HERO_PIDS,
                 seed: int = DEFAULT_HOUSE_SEED, brain=None):
        self.pids = tuple(sorted(pids))
        if brain is None:
            # Imported lazily so the server package stays importable
            # without the brain wasm built (tests, dev).
            from players.baseline_player import MobaBrain
            brain = MobaBrain(seed=seed)
        self._brain = brain
        print(f"house heroes {self.pids} driven by the vendored pretrained "
              f"network (neutral loadout)", file=sys.stderr)

    def actions(self, obs: np.ndarray) -> dict[int, list[int]]:
        """One forward per house hero, ascending pid order.

        ``obs`` is the full (10, 510) uint8 matrix; each hero is fed its
        own row. Brain instance index == pid, so every house hero keeps
        its own MinGRU recurrent state.
        """
        out: dict[int, list[int]] = {}
        for pid in self.pids:
            row = self._brain.forward(pid, obs[pid].tobytes())
            out[pid] = defaults.clamp_actions(
                np.asarray([row], dtype=np.float64))[0].tolist()
        return out
