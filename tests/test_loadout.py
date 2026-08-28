"""The mod's own fidelity gate, plus the loadout physics.

(a) **zero-loadout identity** — the production sim run with an all-neutral
    ``apply_loadout`` for all ten pids must be byte-identical (obs, rewards,
    state digest, every tick) to the same run with no ``apply_loadout`` call
    at all. This is what "keep the Puffer fidelity gate for the un-drafted
    mode" means operationally, and it is the test that would catch an
    ``apply_loadout`` that starts spawning, allocating or drawing RNG.
(b) each of the 12 items applies exactly its documented deltas, read back
    through ``hero_stat``;
(c) the clamp table holds for all 64 arm x tail x misc combinations on all
    five roles;
(d) deltas survive death and level-up;
(e) ``apply_loadout`` draws no RNG.

AGENTS.md rule 2: this file is inviolable. If it fails, the code changed
physics — fix the code, never the test.
"""

import os

import numpy as np
import pytest

from cogame_derks_gym import catalog, defaults
from cogame_derks_gym.sim import (ACT_HIGH, DEFAULT_WASM_PATH,
                                  HERO_STAT_CODES, MobaSim)

TICKS = 500


def require_built():
    """Same CI rule as the inherited fidelity gate: with
    COGAME_REQUIRE_WASM_BUILD set, a missing artifact is a failure, never
    a silent skip."""
    if DEFAULT_WASM_PATH.exists():
        return
    msg = f"{DEFAULT_WASM_PATH} missing - run sim/build_sim.sh first"
    if os.environ.get("COGAME_REQUIRE_WASM_BUILD"):
        pytest.fail(f"{msg} (COGAME_REQUIRE_WASM_BUILD is set: the "
                    f"loadout gate must not skip in CI)")
    pytest.skip(msg)


def neutral_blocks():
    return {pid: catalog.neutral_applied(defaults.HERO_BASE[pid])
            for pid in range(defaults.NUM_HEROES)}


def action_stream(seed=1234, ticks=TICKS):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, ACT_HIGH,
                         size=(defaults.NUM_HEROES, 6)).astype(np.float32)
            for _ in range(ticks)]


# -- (a) the gate ------------------------------------------------------------

def test_zero_loadout_identity():
    """500 ticks with an all-neutral apply_loadout vs no call at all."""
    require_built()
    plain = MobaSim(seed=7)
    drafted = MobaSim(seed=7)
    blocks = neutral_blocks()
    for pid in sorted(blocks):  # ascending pid order, as the server does
        drafted.apply_loadout(pid, blocks[pid])

    assert drafted.observations().tobytes() == plain.observations().tobytes(), \
        "a neutral apply_loadout changed the tick-0 observations"
    assert drafted.state_digest() == plain.state_digest()

    for tick, acts in enumerate(action_stream()):
        for sim in (plain, drafted):
            sim.set_actions(acts)
            sim.step()
        assert drafted.observations().tobytes() == \
            plain.observations().tobytes(), f"obs diverged at tick {tick}"
        assert drafted.rewards().tobytes() == plain.rewards().tobytes(), \
            f"rewards diverged at tick {tick}"
        assert drafted.state_digest() == plain.state_digest(), \
            f"state digest diverged at tick {tick}"
    assert plain.tick() == TICKS


def test_loadout_digest_matches_the_python_mirror():
    """The C digest (sim/loadout_common.h) and the Python one
    (catalog.loadout_digest) must agree: the viewer re-derives the C one
    and the replay header carries it."""
    require_built()
    sim = MobaSim(seed=7)
    # un-drafted: the digest OF the all-zero applied table (what a viewer
    # that pushes nothing reproduces)
    assert sim.loadout_digest() == catalog.loadout_digest()

    blocks = neutral_blocks()
    for pid in sorted(blocks):
        sim.apply_loadout(pid, blocks[pid])
    digest = sim.loadout_digest()
    assert digest == catalog.loadout_digest(blocks)
    assert digest != catalog.loadout_digest()

    again = MobaSim(seed=7)
    for pid in sorted(blocks):
        again.apply_loadout(pid, blocks[pid])
    assert again.loadout_digest() == digest

    # a single different pick changes it
    drafted = dict(blocks)
    drafted[2] = catalog.apply_picks(
        defaults.HERO_BASE[2],
        {"arm": "arm_cleaver", "tail": "tail_plate", "misc": "misc_regen"})
    third = MobaSim(seed=7)
    for pid in sorted(drafted):
        third.apply_loadout(pid, drafted[pid])
    assert third.loadout_digest() == catalog.loadout_digest(drafted)
    assert third.loadout_digest() != digest


# -- (b) every item's documented deltas --------------------------------------

@pytest.mark.parametrize("slot", catalog.SLOTS)
def test_each_item_applies_its_documented_deltas(slot):
    require_built()
    for item in catalog.ITEMS[slot]:
        sim = MobaSim(seed=3)
        pid = 2  # burst, radiant
        base = defaults.HERO_BASE[pid]
        picks = dict(catalog.NEUTRAL_PICKS, **{slot: item["id"]})
        applied = catalog.apply_picks(base, picks)
        sim.apply_loadout(pid, applied)
        for field in catalog.STAT_FIELDS:
            got = sim.hero_stat(pid, HERO_STAT_CODES[field])
            expected = base[field] + item["deltas"].get(field, 0)
            expected = min(max(expected, catalog.CLAMPS[field][0]),
                           catalog.CLAMPS[field][1])
            assert got == pytest.approx(expected), (item["id"], field)
            assert applied[field] == pytest.approx(expected), \
                (item["id"], field)


def test_level_dependent_stats_are_rederived():
    require_built()
    sim = MobaSim(seed=3)
    pid = 0  # support: 500 hp, 250 mana, +100 hp/level
    applied = catalog.apply_picks(
        defaults.HERO_BASE[pid],
        {"arm": "arm_cleaver", "tail": "tail_plate", "misc": "misc_regen"})
    sim.apply_loadout(pid, applied)
    level = int(sim.hero_stat(pid, HERO_STAT_CODES["level"]))
    assert level == 1  # c_reset spawns everyone at level 1
    for stat, base_field, gain_field in (
            ("max_health", "base_health", "hp_gain_per_level"),
            ("max_mana", "base_mana", "mana_gain_per_level"),
            ("damage", "base_damage", "damage_gain_per_level")):
        assert sim.hero_stat(pid, HERO_STAT_CODES[stat]) == pytest.approx(
            applied[base_field] + level * applied[gain_field]), stat
    # and the hero starts full
    assert sim.hero_stat(pid, HERO_STAT_CODES["health"]) == \
        sim.hero_stat(pid, HERO_STAT_CODES["max_health"])
    assert sim.hero_stat(pid, HERO_STAT_CODES["mana"]) == \
        sim.hero_stat(pid, HERO_STAT_CODES["max_mana"])


# -- (c) the clamp table, all 64 combinations x all five roles ---------------

def test_clamps_hold_for_every_combination_and_role():
    require_built()
    sim = MobaSim(seed=5)
    for pid in range(defaults.NUM_HEROES):
        base = defaults.HERO_BASE[pid]
        for arm in catalog.ITEMS["arm"]:
            for tail in catalog.ITEMS["tail"]:
                for misc in catalog.ITEMS["misc"]:
                    picks = {"arm": arm["id"], "tail": tail["id"],
                             "misc": misc["id"]}
                    applied = catalog.apply_picks(base, picks)
                    for field in catalog.STAT_FIELDS:
                        low, high = catalog.CLAMPS[field]
                        assert low <= applied[field] <= high, \
                            (pid, picks, field, applied[field])
                        if field in catalog.INT_FIELDS:
                            assert isinstance(applied[field], int), field
                    # and the sim accepts every one of them
                    sim.apply_loadout(pid, applied)


def test_apply_loadout_rejects_out_of_clamp_and_non_finite():
    require_built()
    sim = MobaSim(seed=5)
    good = catalog.neutral_applied(defaults.HERO_BASE[0])
    sim.apply_loadout(0, good)
    for field, value in (("base_health", 5000.0),
                         ("base_damage", float("nan")),
                         ("move_speed", 0.5),
                         ("basic_attack_cd", 99)):
        bad = dict(good, **{field: value})
        with pytest.raises(ValueError):
            sim.apply_loadout(0, bad)
    with pytest.raises(ValueError):
        sim.apply_loadout(0, {k: v for k, v in good.items()
                              if k != "base_mana"})
    with pytest.raises(ValueError):
        sim.apply_loadout(99, good)


# -- (d) deltas survive death and level-up -----------------------------------

@pytest.mark.slow
def test_deltas_survive_death_and_level_up():
    """The whole design rides on spawn_player and the level-up path
    re-reading the base fields, so a respawned or levelled hero must keep
    its drafted stats."""
    require_built()
    sim = MobaSim(seed=11)
    pid = 1  # assassin, radiant
    applied = catalog.apply_picks(
        defaults.HERO_BASE[pid],
        {"arm": "arm_cleaver", "tail": "tail_plate", "misc": "misc_regen"})
    sim.apply_loadout(pid, applied)

    saw_death = saw_level = False
    rng = np.random.default_rng(5)
    for _ in range(4000):
        sim.set_actions(rng.integers(
            0, ACT_HIGH, size=(defaults.NUM_HEROES, 6)).astype(np.float32))
        sim.step()
        level = int(sim.hero_stat(pid, HERO_STAT_CODES["level"]))
        deaths = int(sim.agent_stat(pid, 2))
        if level > 1:
            saw_level = True
        if deaths > 0:
            saw_death = True
        # the derivation must hold at every tick, whatever happened
        assert sim.hero_stat(pid, HERO_STAT_CODES["max_health"]) == \
            pytest.approx(applied["base_health"]
                          + level * applied["hp_gain_per_level"])
        assert sim.hero_stat(pid, HERO_STAT_CODES["max_mana"]) == \
            pytest.approx(applied["base_mana"]
                          + level * applied["mana_gain_per_level"])
        assert sim.hero_stat(pid, HERO_STAT_CODES["damage"]) == \
            pytest.approx(applied["base_damage"]
                          + level * applied["damage_gain_per_level"])
        assert sim.hero_stat(pid, HERO_STAT_CODES["base_health"]) == \
            pytest.approx(applied["base_health"])
        if saw_death and saw_level:
            break
    assert saw_level, "hero never levelled: cannot prove level-up survival"
    assert saw_death, "hero never died: cannot prove respawn survival"


# -- (e) apply_loadout draws no RNG ------------------------------------------

def test_apply_loadout_draws_no_rng():
    """Two sims with the same seed, one drafted and one not, must have
    IDENTICAL hero spawn positions: apply_loadout must not touch the
    seeded rand() stream (a re-spawn would).
    """
    require_built()
    plain = MobaSim(seed=99)
    drafted = MobaSim(seed=99)
    applied = catalog.apply_picks(
        defaults.HERO_BASE[6],
        {"arm": "arm_needler", "tail": "tail_rotor", "misc": "misc_focus"})
    drafted.apply_loadout(6, applied)

    # obs_extra[0]/[1] are the hero's x/y: spawn positions come straight
    # out of the seeded rand() stream in spawn_player.
    plain_positions = [tuple(plain.observations()[pid, 484:486])
                       for pid in range(defaults.NUM_HEROES)]
    drafted_positions = [tuple(drafted.observations()[pid, 484:486])
                         for pid in range(defaults.NUM_HEROES)]
    assert plain_positions == drafted_positions

    # At tick 0 EVERY row is still identical: compute_observations ran at
    # the end of c_reset, before apply_loadout, so the tick-0 row carries
    # the pre-loadout damage/speed/cooldown bytes (documented in
    # docs/DRAFT.md). The server and the viewer apply loadouts at the same
    # point, so this costs nothing in determinism.
    assert plain.observations().tobytes() == drafted.observations().tobytes()

    # From tick 1 on (compute_observations runs inside c_step) the drafted
    # hero's own stat bytes reflect the block, and nobody else's move.
    noop = np.tile(np.asarray(defaults.NOOP_ACTION, dtype=np.float32),
                   (defaults.NUM_HEROES, 1))
    for sim in (plain, drafted):
        sim.set_actions(noop)
        sim.step()
    # obs_extra[5] = damage/50, [6] = move_speed, [14] = basic_attack_cd
    plain_row = plain.observations()[6]
    drafted_row = drafted.observations()[6]
    assert drafted_row[484 + 14] < plain_row[484 + 14], \
        "the needler's -3 attack cooldown never reached the obs bytes"
    assert drafted_row[484 + 6] == plain_row[484 + 6] == 1, \
        "move_speed must still cast to 1 (the in-distribution band)"
