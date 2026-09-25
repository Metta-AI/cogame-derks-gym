from types import SimpleNamespace

import numpy as np

from cogame_derks_gym import catalog
from cogame_derks_gym.training import ACTION_SIZES, OBSERVATION_SIZE, numeric_action_mask
from players.trained_player import TrainedPolicy


class FrozenStub:
    spec = SimpleNamespace(observation_size=OBSERVATION_SIZE,
                           action_sizes=list(ACTION_SIZES))

    def __init__(self):
        self.seen = []
        self.actions = [[57, 3, 3, 0, 0, 0, 0],
                        [0, 1, 2, 0, 1, 0, 1]]

    def reset(self, seed):
        self.seed = seed

    def reset_seat(self, seat):
        self.seat = seat

    def predict(self, seat, observation):
        self.seen.append((seat, observation))
        return len(self.seen) - 1

    def sample(self, prediction):
        return self.actions[prediction]


def test_trained_player_uses_accepted_draft_and_training_codec():
    frozen = FrozenStub()
    player = TrainedPolicy(frozen, seat=2, seed=19)
    picks = player.on_draft({"seat": 2})
    assert frozen.seed == "19" and frozen.seat == 2
    assert picks == {
        slot: catalog.ITEMS[slot][(57 // (4 ** index)) % 4]["id"]
        for index, slot in enumerate(catalog.SLOTS)
    }
    draft_input = frozen.seen[0][1]
    assert draft_input.values[0][:510] == [0.0] * 510
    np.testing.assert_allclose(draft_input.values[0][-3:], [1.0, 2 / 5, 0.0])
    assert draft_input.action_masks[0] == list(numeric_action_mask(draft=True))

    # The game may reject a late draft reply. Its actual neutral result is
    # the feature the checkpoint must see on every subsequent tick.
    player.on_draft_result({"loadouts": [
        {"seat": 2, "picks": catalog.NEUTRAL_PICKS},
    ]})
    raw = bytes(index % 256 for index in range(510))
    assert player(0, [raw]) == [[1, 2, 0, 1, 0, 1]]
    micro_input = frozen.seen[1][1]
    np.testing.assert_allclose(micro_input.values[0][:510],
                               np.frombuffer(raw, np.uint8) / 255.0)
    np.testing.assert_allclose(micro_input.values[0][-3:], [0.0, 2 / 5, 0.0])
    assert micro_input.action_masks[0] == list(numeric_action_mask(draft=False))
