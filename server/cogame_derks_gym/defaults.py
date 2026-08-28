"""Server-level config defaults and the seat/hero/team topology.

Env-physics values that mirror upstream training defaults (vision_range,
agent_speed, reward weights) live in sim/shim_common.h
``moba_configure`` (shared by the server shim and the viewer) — not here.
This module owns the *server* contract: config defaults, the no-op
action, the seat -> hero map, and each hero's base stat block (the block
the draft's deltas are added to).

Seats and heroes
----------------
Upstream Puffer MOBA is 5v5: ``NUM_PLAYERS`` is 10 and ``init_moba``
spawns exactly five heroes per team in the fixed role order support,
assassin, burst, tank, carry (``vendor/upstream/moba.h:1636-1749``). This
game seats **six** of those ten heroes, three per team:

    seat  0  1  2   3  4  5
    pid   0  1  2   5  6  7

The remaining four — pids 3, 4 (radiant tank, carry) and 8, 9 (dire
tank, carry) — are **house heroes**: the game server drives them itself,
in-process, off the vendored pretrained network on the neutral loadout.
Nothing in the sim changes: ``num_agents`` inside the wasm stays 10 and
``script_opponents`` stays 0, so ``compute_observations`` writes all ten
510-byte rows and the replay body stays 60 bytes per tick.

Hero pid -> team (ground truth: init_moba's spawn loop,
``for (int team = 0; team < 2; team++) ... for (int pid = team*5; ...)``):
pids 0-4 are team 0 (radiant), pids 5-9 are team 1 (dire).
"""

from __future__ import annotations

import numpy as np

NUM_HEROES = 10
TEAM_SIZE = 5
NUM_TEAMS = 2
ACTIONS_PER_HERO = 6

# Seat -> hero pid, and the four in-process house heroes. Both are FIXED:
# ten heroes do not divide into six equal seats, and asymmetric seats
# would make league ranking unfair and the draft incoherent.
SEAT_HERO_PIDS: tuple[int, ...] = (0, 1, 2, 5, 6, 7)
HOUSE_HERO_PIDS: tuple[int, ...] = (3, 4, 8, 9)
NUM_SEATS = len(SEAT_HERO_PIDS)

# In-game names. A policy only ever sees these: teams are server-assigned
# and a seat can neither choose nor infer who it is playing. Real player
# names are spectator-side only (results.names, the replay header's
# config.players) — see docs/PROTOCOL.md "Two name spaces".
SEAT_ALIASES: tuple[str, ...] = (
    "Cog-Alpha", "Cog-Bravo", "Cog-Charlie",
    "Cog-Delta", "Cog-Echo", "Cog-Foxtrot",
)
HOUSE_ALIASES: dict[int, str] = {
    3: "House-Tank-R", 4: "House-Carry-R",
    8: "House-Tank-D", 9: "House-Carry-D",
}

TEAM_NAMES = ("radiant", "dire")

# hero_type -> role name (init_moba's per-team role blocks).
ROLE_NAMES = ("support", "assassin", "burst", "tank", "carry")

# Per-role skill names, in the engine's Q -> W -> E order
# (moba.h:1670-1748 env->skills[pid][0..2]).
ROLE_SKILLS: dict[str, tuple[str, str, str]] = {
    "support": ("support_hook", "support_aoe_heal", "support_stun"),
    "assassin": ("assassin_aoe_minions", "assassin_tp_damage",
                 "assassin_move_buff"),
    "burst": ("burst_nuke", "burst_aoe", "burst_aoe_stun"),
    "tank": ("tank_aoe_dot", "tank_self_heal", "tank_engage_aoe"),
    "carry": ("carry_retreat_slow", "carry_slow_damage", "carry_aoe"),
}

# Per-role base stat block, verbatim from init_moba (moba.h:1666-1744):
# base_health / base_mana / hp_gain_per_level / mana_gain_per_level /
# damage_gain_per_level per role, with base_damage 50, basic_attack_cd 8
# and move_speed = env->agent_speed (1.0) set for every hero
# (moba.h:1654-1656). This is the block the draft's deltas add to.
_ROLE_BASE: dict[str, dict] = {
    "support": {"base_health": 500.0, "base_mana": 250.0,
                "hp_gain_per_level": 100, "mana_gain_per_level": 50,
                "damage_gain_per_level": 10},
    "assassin": {"base_health": 400.0, "base_mana": 300.0,
                 "hp_gain_per_level": 100, "mana_gain_per_level": 65,
                 "damage_gain_per_level": 10},
    "burst": {"base_health": 400.0, "base_mana": 300.0,
              "hp_gain_per_level": 75, "mana_gain_per_level": 90,
              "damage_gain_per_level": 10},
    "tank": {"base_health": 700.0, "base_mana": 200.0,
             "hp_gain_per_level": 150, "mana_gain_per_level": 50,
             "damage_gain_per_level": 15},
    "carry": {"base_health": 300.0, "base_mana": 250.0,
              "hp_gain_per_level": 50, "mana_gain_per_level": 50,
              "damage_gain_per_level": 25},
}
# hero_type -> radiant lane; dire adds 3 (moba.h:1676,1693,1710,1727,1744).
_ROLE_LANE = {"support": 2, "assassin": 1, "burst": 1, "tank": 0, "carry": 2}


def role_for_pid(pid: int) -> str:
    """Role of hero ``pid`` (init_moba assigns hero_type = pid % 5)."""
    return ROLE_NAMES[pid % TEAM_SIZE]


def team_for_pid(pid: int) -> int:
    """0 = radiant (pids 0-4), 1 = dire (pids 5-9)."""
    return pid // TEAM_SIZE


def lane_for_pid(pid: int) -> int:
    return _ROLE_LANE[role_for_pid(pid)] + 3 * team_for_pid(pid)


def hero_base(pid: int) -> dict:
    """Hero ``pid``'s own base stat block (see _ROLE_BASE)."""
    base = dict(_ROLE_BASE[role_for_pid(pid)])
    base["base_damage"] = 50.0
    base["basic_attack_cd"] = 8
    base["move_speed"] = 1.0
    return base


HERO_BASE: dict[int, dict] = {pid: hero_base(pid) for pid in range(NUM_HEROES)}


def seat_for_pid(pid: int) -> int | None:
    """The seat controlling hero ``pid``, or None for a house hero."""
    try:
        return SEAT_HERO_PIDS.index(pid)
    except ValueError:
        return None


def pid_for_seat(seat: int) -> int:
    return SEAT_HERO_PIDS[seat]


def team_for_seat(seat: int) -> int:
    return team_for_pid(SEAT_HERO_PIDS[seat])


def alias_for_pid(pid: int) -> str:
    seat = seat_for_pid(pid)
    return SEAT_ALIASES[seat] if seat is not None else HOUSE_ALIASES[pid]


# MultiDiscrete action space highs (exclusive), per column: vel_y, vel_x,
# target-filter, use_q, use_w, use_e. Mirrors cogame_derks_gym.sim.ACT_HIGH
# (tested equal); duplicated so transport code needs no wasmtime import.
ACT_HIGH = (7, 7, 3, 2, 2, 2)
# No-op: center velocity (3,3 -> 0,0), scan-all filter, no skills.
NOOP_ACTION = (3, 3, 0, 0, 0, 0)

DEFAULT_MAX_TICKS = 6000
DEFAULT_TICK_DEADLINE_MS = 100
DEFAULT_PLAYER_CONNECT_TIMEOUT_SECONDS = 60
# The draft turn: one simultaneous, hidden turn, all six seats asked in a
# single parallel batch under this one shared deadline.
DEFAULT_DRAFT_DEADLINE_MS = 45000
MIN_DRAFT_DEADLINE_MS = 1000
DEFAULT_DRAFT_ENABLED = True

# Mirrors the manifest's top-level episode_timeout_minutes (the platform
# kills the container at that point, losing results and replay). The
# whole episode is sized to fit inside 60% of it:
#   connect 60 s + draft 45 s + play 6000 x 100 ms = 705 s < 720 s
# and the engine hard-stops itself at wall_clock_budget_seconds (645 s =
# draft 45 + play 600, timed from the start of the draft).
PLATFORM_EPISODE_TIMEOUT_MINUTES = 20


def derived_wall_clock_budget_seconds(max_ticks: int, tick_deadline_ms: int,
                                      draft_deadline_ms: int = 0) -> float:
    """Default wall-clock budget (see PLATFORM_EPISODE_TIMEOUT_MINUTES).

    The engine's clock starts at the DRAFT turn, so the draft's own
    deadline is part of the budget: draft 45 s + play 6000 x 100 ms =
    645 s, the design note's figure.
    """
    return min(0.9 * PLATFORM_EPISODE_TIMEOUT_MINUTES * 60,
               draft_deadline_ms / 1000.0
               + max_ticks * tick_deadline_ms / 1000.0)


def clamp_actions(actions: np.ndarray) -> np.ndarray:
    """Sanitize finite action values exactly as the sim boundary does.

    Truncate toward zero (the sim's C ``(int)`` cast) and clamp per column
    to ``0 .. ACT_HIGH[col]-1``. Returns uint8 (max action value is 6) —
    the post-clamp form stored in replays. Mirrors MobaSim.set_actions.
    """
    high = np.asarray(ACT_HIGH, dtype=np.float64) - 1.0
    return np.clip(np.trunc(np.asarray(actions, dtype=np.float64)),
                   0.0, high).astype(np.uint8)
