"""Tests for game config parsing/validation and seat/hero mapping helpers."""

import json

import pytest

from cogame_derks_gym import defaults
from cogame_derks_gym.config import ConfigError, GameConfig


SEATS = defaults.NUM_SEATS  # 6


def base_dict(**overrides):
    d = {
        "players": [{"name": f"player{i}"} for i in range(SEATS)],
        "tokens": [f"token-{i}" for i in range(SEATS)],
    }
    d.update(overrides)
    return d


# -- defaults + parsing ------------------------------------------------------

def test_defaults_applied():
    cfg = GameConfig.from_dict(base_dict())
    assert cfg.max_ticks == 6000
    assert cfg.tick_deadline_ms == 100
    assert cfg.player_connect_timeout_seconds == 60
    assert cfg.draft_enabled is True
    assert cfg.draft_deadline_ms == 45000
    assert cfg.catalog_version == "v1"
    assert cfg.num_seats == SEATS
    assert [p.name for p in cfg.players] == \
        [f"player{i}" for i in range(SEATS)]


def test_seed_derived_and_recorded_when_missing():
    cfg = GameConfig.from_dict(base_dict())
    assert isinstance(cfg.seed, int)
    assert 0 <= cfg.seed <= 0xFFFFFFFF
    # the derived seed must be recorded in the resolved config (replay header)
    assert cfg.to_dict()["seed"] == cfg.seed


def test_explicit_seed_preserved():
    cfg = GameConfig.from_dict(base_dict(seed=1234))
    assert cfg.seed == 1234
    assert cfg.to_dict()["seed"] == 1234


def test_wide_or_negative_seed_masked_to_u32():
    """The recorded seed must be the u32 the sim actually runs with."""
    cfg = GameConfig.from_dict(base_dict(seed=0x1_2345_6789))
    assert cfg.seed == 0x2345_6789
    assert cfg.to_dict()["seed"] == 0x2345_6789
    cfg = GameConfig.from_dict(base_dict(seed=-1))
    assert cfg.seed == 0xFFFF_FFFF


def test_explicit_values_override_defaults():
    cfg = GameConfig.from_dict(base_dict(
        max_ticks=500, tick_deadline_ms=50,
        player_connect_timeout_seconds=2))
    assert cfg.max_ticks == 500
    assert cfg.tick_deadline_ms == 50
    assert cfg.player_connect_timeout_seconds == 2


def test_nodraft_variant_parses():
    cfg = GameConfig.from_dict(base_dict(draft_enabled=False))
    assert cfg.draft_enabled is False
    assert cfg.to_dict()["draft_enabled"] is False


# -- validation --------------------------------------------------------------

@pytest.mark.parametrize("count", [1, 2, 5, 7, 10])
def test_wrong_player_count_rejected(count):
    """This game seats exactly six players: the other four heroes are
    house-driven, so any other count is a different game."""
    d = base_dict()
    d["players"] = [{"name": f"p{i}"} for i in range(count)]
    d["tokens"] = [f"t{i}" for i in range(count)]
    with pytest.raises(ConfigError):
        GameConfig.from_dict(d)


@pytest.mark.parametrize("num_agents", [2, 5, 7, 10])
def test_num_agents_must_be_six(num_agents):
    with pytest.raises(ConfigError):
        GameConfig.from_dict(base_dict(num_agents=num_agents))


def test_num_agents_six_accepted():
    assert GameConfig.from_dict(base_dict(num_agents=6)).num_seats == 6


@pytest.mark.parametrize("bad", [999, 0, -1, "45000"])
def test_bad_draft_deadline_rejected(bad):
    with pytest.raises(ConfigError):
        GameConfig.from_dict(base_dict(draft_deadline_ms=bad))


def test_draft_deadline_floor_accepted():
    cfg = GameConfig.from_dict(base_dict(draft_deadline_ms=1000))
    assert cfg.draft_deadline_ms == 1000


@pytest.mark.parametrize("bad", ["true", 1, None])
def test_bad_draft_enabled_rejected(bad):
    with pytest.raises(ConfigError):
        GameConfig.from_dict(base_dict(draft_enabled=bad))


@pytest.mark.parametrize("bad", ["v2", "V1", "", 1])
def test_unknown_catalog_version_rejected(bad):
    """Pinned so a future catalog cannot silently re-interpret an old
    replay."""
    with pytest.raises(ConfigError):
        GameConfig.from_dict(base_dict(catalog_version=bad))


def test_token_length_mismatch_rejected():
    d = base_dict()
    d["tokens"] = d["tokens"][:5]
    with pytest.raises(ConfigError):
        GameConfig.from_dict(d)


def test_missing_players_rejected():
    with pytest.raises(ConfigError):
        GameConfig.from_dict({"tokens": []})


def test_bad_max_ticks_rejected():
    with pytest.raises(ConfigError):
        GameConfig.from_dict(base_dict(max_ticks=0))


def test_bad_tick_deadline_rejected():
    with pytest.raises(ConfigError):
        GameConfig.from_dict(base_dict(tick_deadline_ms=-5))


def test_empty_player_name_rejected():
    d = base_dict()
    d["players"][3] = {"name": ""}
    with pytest.raises(ConfigError):
        GameConfig.from_dict(d)


# -- wall-clock budget -------------------------------------------------------

def test_wall_clock_budget_default_derived():
    # default: min(0.9 x platform episode timeout, max_ticks x deadline)
    cfg = GameConfig.from_dict(base_dict(max_ticks=100_000,
                                         tick_deadline_ms=1000))
    assert cfg.wall_clock_budget_seconds == pytest.approx(
        0.9 * defaults.PLATFORM_EPISODE_TIMEOUT_MINUTES * 60)
    # a short episode is capped by its own worst case instead
    cfg = GameConfig.from_dict(base_dict(max_ticks=100, tick_deadline_ms=500))
    assert cfg.wall_clock_budget_seconds == pytest.approx(50.0)


def test_default_episode_fits_inside_60_percent_of_the_platform_budget():
    """The design note's arithmetic, pinned: connect 60 + draft 45 + play
    6000 x 100 ms = 705 s < 720 s (0.60 x the 1200 s platform budget), and
    the engine hard-stops itself before that."""
    cfg = GameConfig.from_dict(base_dict())
    worst_case = (cfg.player_connect_timeout_seconds
                  + cfg.draft_deadline_ms / 1000
                  + cfg.max_ticks * cfg.tick_deadline_ms / 1000)
    assert worst_case <= 0.6 * 1200
    assert cfg.wall_clock_budget_seconds <= 645


def test_wall_clock_budget_explicit_override():
    cfg = GameConfig.from_dict(base_dict(wall_clock_budget_seconds=123.5))
    assert cfg.wall_clock_budget_seconds == 123.5
    assert cfg.to_dict()["wall_clock_budget_seconds"] == 123.5


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"),
                                 "60", True, None])
def test_bad_wall_clock_budget_rejected(bad):
    with pytest.raises(ConfigError):
        GameConfig.from_dict(base_dict(wall_clock_budget_seconds=bad))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_connect_timeout_rejected(bad):
    with pytest.raises(ConfigError):
        GameConfig.from_dict(base_dict(player_connect_timeout_seconds=bad))


# -- serialization -----------------------------------------------------------

def test_to_dict_excludes_tokens_by_default():
    cfg = GameConfig.from_dict(base_dict(seed=7))
    d = cfg.to_dict()
    assert "tokens" not in d
    assert d["players"] == [{"name": f"player{i}"} for i in range(SEATS)]
    # the draft fields reach the replay header too
    assert d["draft_enabled"] is True
    assert d["draft_deadline_ms"] == 45000
    assert d["catalog_version"] == "v1"
    # round-trips through from_dict (tokens re-supplied)
    d2 = dict(d, tokens=list(cfg.tokens))
    cfg2 = GameConfig.from_dict(d2)
    assert cfg2 == cfg


def test_from_file_uri(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(base_dict(seed=42)))
    for uri in (f"file://{path}", str(path)):
        cfg = GameConfig.from_file_uri(uri)
        assert cfg.seed == 42


# -- seat/hero/team mapping helpers -----------------------------------------

def test_noop_matches_sim_contract():
    from cogame_derks_gym import sim
    assert list(defaults.NOOP_ACTION) == list(sim.NOOP_ACTION)
    assert tuple(defaults.ACT_HIGH) == tuple(sim.ACT_HIGH)


def test_seat_hero_map_is_the_fixed_tuple():
    assert defaults.SEAT_HERO_PIDS == (0, 1, 2, 5, 6, 7)
    assert defaults.HOUSE_HERO_PIDS == (3, 4, 8, 9)
    assert set(defaults.SEAT_HERO_PIDS) | set(defaults.HOUSE_HERO_PIDS) == \
        set(range(defaults.NUM_HEROES))


def test_seat_for_pid_inverts_mapping():
    for seat in range(SEATS):
        assert defaults.seat_for_pid(defaults.pid_for_seat(seat)) == seat
    for pid in defaults.HOUSE_HERO_PIDS:
        assert defaults.seat_for_pid(pid) is None


def test_roles_and_lanes_match_upstream_init_moba():
    # moba.h:1666-1744: role order support, assassin, burst, tank, carry;
    # lanes 2, 1, 1, 0, 2 for radiant, +3 for dire.
    assert [defaults.role_for_pid(p) for p in range(5)] == \
        list(defaults.ROLE_NAMES)
    assert [defaults.lane_for_pid(p) for p in range(5)] == [2, 1, 1, 0, 2]
    assert [defaults.lane_for_pid(p) for p in range(5, 10)] == [5, 4, 4, 3, 5]


def test_aliases_are_anonymous_and_unique():
    aliases = list(defaults.SEAT_ALIASES) + list(
        defaults.HOUSE_ALIASES.values())
    assert len(set(aliases)) == len(aliases)
    assert defaults.alias_for_pid(0) == "Cog-Alpha"
    assert defaults.alias_for_pid(7) == "Cog-Foxtrot"
    assert defaults.alias_for_pid(3) == "House-Tank-R"


def test_team_for_pid():
    # moba.h init: pids 0-4 spawn as team 0 (radiant), 5-9 as team 1 (dire)
    for pid in range(5):
        assert defaults.team_for_pid(pid) == 0
    for pid in range(5, 10):
        assert defaults.team_for_pid(pid) == 1


def test_team_for_seat():
    assert [defaults.team_for_seat(s) for s in range(SEATS)] == \
        [0, 0, 0, 1, 1, 1]


def test_clamp_actions():
    import numpy as np
    raw = np.array([[6.9, -1.0, 2.0, 1.4, 0.0, 1.0],
                    [99.0, 3.0, -7.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    out = defaults.clamp_actions(raw)
    assert out.dtype == np.uint8
    assert out.tolist() == [[6, 0, 2, 1, 0, 1], [6, 3, 0, 0, 1, 0]]
