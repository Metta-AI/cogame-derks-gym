import numpy as np
import pytest

from cogame_derks_gym import catalog, defaults
from cogame_derks_gym.sim import HERO_STAT_CODES
from cogame_derks_gym.training import ACTION_SIZES, OBSERVATION_SIZE, TrainingMatch


def test_draft_hides_sim_state_and_applies_catalog_loadout():
    match = TrainingMatch(seat=1, max_ticks=4)
    draft = match.reset(seed=7)
    assert draft.observation.shape == (OBSERVATION_SIZE,)
    assert np.count_nonzero(draft.observation[:510]) == 0
    assert len(draft.action_mask) == sum(ACTION_SIZES)
    assert sum(draft.action_mask) == 64 + len(defaults.ACT_HIGH)

    # Base-4 digits pick blaster / plate / battery in the catalog's slot order.
    choice = 1 + 4 + 2 * 16
    first_tick = match.step([choice, *defaults.NOOP_ACTION])
    picks = {"arm": "arm_blaster", "tail": "tail_plate",
             "misc": "misc_battery"}
    expected = catalog.apply_picks(defaults.HERO_BASE[1], picks)
    for field in catalog.STAT_FIELDS:
        assert match.sim.hero_stat(1, HERO_STAT_CODES[field]) == expected[field]
    assert first_tick.observation[-3] == 0
    assert sum(first_tick.action_mask) == 1 + sum(defaults.ACT_HIGH)


def test_micro_uses_real_sim_and_tick_cap_scoring():
    def play(seed: int, action: tuple[int, ...]):
        match = TrainingMatch(seat=0, max_ticks=8, draft_enabled=False)
        observations = [match.reset(seed).observation]
        rewards = []
        for _ in range(8):
            step = match.step([0, *action])
            observations.append(step.observation)
            rewards.append(step.reward)
        assert step.done and match.sim.tick() == 8
        radiant, dire = (match.sim.ancient_health(team)
                         for team in range(defaults.NUM_TEAMS))
        expected = 0.5 if radiant == dire else float(radiant > dire)
        assert step.score == expected
        with pytest.raises(RuntimeError, match="ended"):
            match.step([0, *action])
        return np.stack(observations), rewards

    first = play(11, defaults.NOOP_ACTION)
    repeat = play(11, defaults.NOOP_ACTION)
    assert np.array_equal(first[0], repeat[0])
    assert first[1] == repeat[1]
    changed = play(11, (0, 0, 0, 0, 0, 0))
    assert not np.array_equal(first[0], changed[0])
