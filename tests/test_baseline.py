"""Tests for the baseline player (Task 3.2): wasm puffernet brain.

Unit tests exercise build/derk_brain.wasm directly (load, valid actions,
cross-instance determinism). The slow behavioral test plays a full
episode over the real websocket stack: baseline policy on team 0 vs the
random policy on team 1 — the pretrained policy must win.
"""

import asyncio
import json

import numpy as np
import pytest

from cogame_derks_gym import defaults
from cogame_derks_gym.sim import MobaSim
from players.baseline_player import (EXPECTED_PARAMS, NUM_ATNS, NUM_BRAINS,
                                     OBS_SIZE, BaselinePolicy, MobaBrain)
from players.client import play_episode
from players.random_player import RandomPolicy

from tests.test_server import SEATS, ServerHarness, make_config


# -- unit: wasm brain --------------------------------------------------------

def test_brain_loads_with_expected_param_count():
    # brain_init returns the embedded weight count; the constructor
    # raises if it isn't the 95,616 params of moba_weights.bin
    MobaBrain(seed=1)
    assert EXPECTED_PARAMS == 95_616


def test_forward_returns_in_range_actions():
    brain = MobaBrain(seed=1)
    obs = MobaSim(seed=3).observations()  # real first-tick obs
    for agent in range(NUM_BRAINS):
        acts = brain.forward(agent, obs[agent].tobytes())
        assert len(acts) == NUM_ATNS
        assert all(isinstance(a, int) for a in acts)
        assert all(0 <= a < high
                   for a, high in zip(acts, defaults.ACT_HIGH))


def test_brain_init_rejects_second_call():
    # re-init would leak nets and reset recurrent state; the shim
    # returns -1 (see brain_shim.c)
    brain = MobaBrain(seed=1)
    assert brain._exports["brain_init"](brain._store, 1) == -1


def test_forward_input_validation():
    brain = MobaBrain(seed=1)
    with pytest.raises(ValueError, match="agent_idx"):
        brain.forward(10, bytes(OBS_SIZE))
    with pytest.raises(ValueError, match="agent_idx"):
        brain.forward(-1, bytes(OBS_SIZE))
    with pytest.raises(ValueError, match="510 bytes"):
        brain.forward(0, bytes(12))


def test_two_fresh_instances_are_identical():
    """Same seed + same obs sequence in the same call order -> identical
    actions (recurrent state and the sampling RNG both live in-wasm)."""
    sim = MobaSim(seed=5)
    brain_a = MobaBrain(seed=1)

    # roll the sim with brain A driving all 10 heroes, recording the obs
    obs_history = []
    for _ in range(20):
        obs = sim.observations()
        obs_history.append(obs.copy())
        acts = np.array([brain_a.forward(p, obs[p].tobytes())
                         for p in range(10)], dtype=np.float32)
        sim.set_actions(acts)
        sim.step()

    # replay the recorded obs through a fresh instance
    brain_b = MobaBrain(seed=1)
    replayed, original = [], []
    brain_a2 = MobaBrain(seed=1)  # third instance to re-derive A's actions
    for obs in obs_history:
        original.append([brain_a2.forward(p, obs[p].tobytes())
                         for p in range(10)])
        replayed.append([brain_b.forward(p, obs[p].tobytes())
                         for p in range(10)])
    assert replayed == original


def test_different_seed_diverges():
    """Sampling is stochastic (softmax sampling, not argmax): different
    RNG seeds produce different action sequences on the same obs."""
    obs = MobaSim(seed=3).observations()
    a = MobaBrain(seed=1)
    b = MobaBrain(seed=1234)
    seq_a = [a.forward(p, obs[p].tobytes()) for _ in range(5)
             for p in range(10)]
    seq_b = [b.forward(p, obs[p].tobytes()) for _ in range(5)
             for p in range(10)]
    assert seq_a != seq_b


# -- behavioral: baseline must beat random -----------------------------------

@pytest.mark.slow
async def test_baseline_beats_random_full_episode(tmp_path):
    """Full ws episode: the pretrained policy on all three radiant seats
    against uniform-random on all three dire seats. The pretrained side
    must win — in calibration runs it destroys the dire ancient well
    inside the cap (also exercising patch 0003's done reporting). Both
    sides' house heroes (tank + carry) run the same vendored network, so
    the seats are what decides it.

    Toolchain-drift note: if this test starts failing right after an
    emcc/emsdk bump, suspect a musl rand()/libm stream change shifting
    the sampled-action trajectory — not a port bug. Re-calibrate seeds
    before digging into the sim or brain shim."""
    cfg = make_config(max_ticks=5000, seed=13, tick_deadline_ms=1000,
                      draft_enabled=False)
    async with ServerHarness(cfg, tmp_path) as h:
        done_msgs = await asyncio.gather(*(
            play_episode(
                BaselinePolicy(seed=1) if seat < 3 else RandomPolicy(seed=99),
                h.ws_url(seat, f"token-{seat}"))
            for seat in range(SEATS)))
        result = await h.episode_task

    results = json.loads(h.results_path.read_text())
    radiant_towers = sum(s["towers_killed"]
                         for s in results["agent_stats"][:5])
    dire_towers = sum(s["towers_killed"]
                      for s in results["agent_stats"][5:])
    print(f"\nbaseline-vs-random outcome: winner={result.winner} "
          f"end_reason={result.end_reason} final_tick={result.final_tick} "
          f"ancients={result.ancient_healths} "
          f"towers(baseline={radiant_towers}, random={dire_towers})")

    # the pretrained side (team 0) must win: outright ancient kill, or the
    # ancient-health tiebreak at the tick cap
    assert result.winner == 0
    if result.end_reason == "ancient":
        assert result.ancient_healths[1] == 0.0
    else:
        assert result.ancient_healths[0] > result.ancient_healths[1]
    assert results["scores"] == [1.0] * 3 + [0.0] * 3
    assert all(m["winner"] == 0 for m in done_msgs)
    # a healthy episode: no seat ever died, and NOOP fallbacks stay
    # negligible (load-tolerant: a deadline over thousands of real-time
    # ticks can drop a stray tick on a loaded CI machine)
    assert results["dead_seats"] == [False] * SEATS
    assert all(n <= 5 for n in results["noop_ticks"]), results["noop_ticks"]


# -- bounded orders / legality (both scripted baselines) ---------------------

@pytest.mark.slow
@pytest.mark.parametrize("name", ["puffer-forge", "lane-brawler"])
def test_baseline_orders_are_bounded_and_legal(name):
    """Against a live sim for 300 ticks, every row the baseline returns is
    exactly six ints in 0..ACT_HIGH[col]-1 and passes the engine's own
    sanitizer — never a NOOP fallback."""
    from cogame_derks_gym.engine import _sanitize
    from players.derk_player import _micro_for

    sim = MobaSim(seed=17)
    policies = {pid: _micro_for(name, 1)
                for pid in defaults.SEAT_HERO_PIDS}
    for tick in range(300):
        obs = sim.observations()
        actions = np.tile(np.asarray(defaults.NOOP_ACTION, dtype=np.float32),
                          (defaults.NUM_HEROES, 1))
        for pid, policy in policies.items():
            row = policy(tick, [obs[pid].tobytes()])
            assert len(row) == 1, (name, pid, row)
            assert len(row[0]) == 6, (name, pid, row)
            assert all(isinstance(v, (int, np.integer)) for v in row[0]), row
            for col, value in enumerate(row[0]):
                assert 0 <= value < defaults.ACT_HIGH[col], (name, col, value)
            sanitized = _sanitize(row)
            assert sanitized is not None, (name, pid, row)
            actions[pid] = sanitized[0]
        sim.set_actions(actions.astype(np.float32))
        sim.step()
    assert sim.tick() == 300


def test_baseline_draft_picks_are_legal_for_every_hero():
    """Every one of the ten role/hero combinations gets a legal id of the
    matching slot from both scripted rules."""
    from cogame_derks_gym import catalog, draft
    from cogame_derks_gym.config import GameConfig
    from players.derk_player import brawler_picks, forge_picks

    cfg = GameConfig.from_dict({
        "players": [{"name": f"p{i}"} for i in range(defaults.NUM_SEATS)],
        "tokens": [f"t{i}" for i in range(defaults.NUM_SEATS)],
        "seed": 1,
    })
    for pid in range(defaults.NUM_HEROES):
        hero = draft.hero_view(pid)
        observation = {"hero": hero}
        for rule in (forge_picks, brawler_picks):
            picks = rule(observation)
            assert catalog.normalized_picks(picks) is not None, (pid, picks)
            applied = catalog.apply_picks(defaults.HERO_BASE[pid], picks)
            for field, (low, high) in catalog.CLAMPS.items():
                assert low <= applied[field] <= high, (pid, field)
    # and through the real draft observation for the six seated heroes
    for seat in range(defaults.NUM_SEATS):
        observation = draft.draft_observation(seat, cfg)
        for rule in (forge_picks, brawler_picks):
            assert catalog.normalized_picks(rule(observation)) is not None
