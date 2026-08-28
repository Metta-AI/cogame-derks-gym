"""Scripted (hand-coded) cogame-derks-gym player: ``python -m players.scripted_player``.

A per-hero finite-state lane-push bot, written as a legible reference
policy (Task 6.2b) — not a tuning contest entry. Structure borrows from
the coworld-ctf baseline bot: perception is parsed into an explicit
state struct, a small persistent world model sits on top of the raw
observation (static nav grid + cost fields, dead-tower memory), roles
are deterministic from the hero identity, and steering is separated
from mode selection.

Ground truth for every constant below is the vendored upstream source,
``vendor/upstream/moba.h`` (line numbers cited per table).

Observation contract (compute_observations, moba.h:460-533)
-----------------------------------------------------------
Each hero gets 510 bytes: a 484-byte "crop" region then 26 self bytes.

Self bytes (obs[484:510], moba.h:480-503): x, y, level, health*10/max,
mana*10/max, damage/50, move_speed, move_modifier (int-cast: slow 0.5
reads as 0), stun/move/q/w/e/basic-attack timers, basic cd, is_hit,
team, hero one-hot at 17+hero_type, and four reward-event flags.

Crop reliability — the stride-1 overlap quirk (moba.h:505-531): the
crop is written per cell at ``map_idx = (dy+5)*11 + (dx+5)`` — stride
1, not 4 — so indices run 0..120 while the buffer holds 484 bytes.
For each cell the tile id is written at ``map_idx`` and, if an entity
occupies the cell, health/mana/level at ``map_idx+1..3`` — which are
exactly the *tile* slots of the next three cells. Cells are visited in
increasing ``map_idx`` order, so every tile write at index i happens
AFTER any earlier cell's entity bytes landed on index i and clobbers
them. Conclusion, relied on throughout this module:

- bytes 0..120 are ALWAYS the 121 tile ids of the 11x11 crop
  (row-major, row = dy+5, col = dx+5, center = self);
- entity health/mana/level survive ONLY for the last cell (indices
  121..123, bottom-right corner) — useless, ignored;
- bytes 124..483 are never written (memset zero, moba.h:462).

So the crop yields tile-id presence only: which tile types sit where
around us. Tile ids (moba.h:28-43): EMPTY 0, WALL 1, TOWER 2 (team NOT
encoded — resolved against the static tower table via our absolute
position), RADIANT_CREEP 3, DIRE_CREEP 4, NEUTRAL 5, radiant heroes
6..10, dire heroes 11..15.

Action contract (step_players, moba.h:1503-1545)
------------------------------------------------
[vel_y, vel_x, attack_target, use_q, use_w, use_e], each 0..ACT_HIGH-1.
Velocity decodes as (a-3)/3, dest = pos + move_modifier*speed*vel with
speed 1.0 (sim/shim_common.h:17-18); move_to fails silently into
non-empty cells — players get NO pathfinding, hence the client-side
nav grid. attack_target 0/1 scan ALL hostiles (creeps, neutrals,
heroes, towers) in the vision square (radius 5); 2 scans heroes+towers
only (moba.h:1519-1525). The engine self-selects the NEAREST scanned
entity as the target for skills and basic attacks; skill flags are
tried Q, then W, then E, then basic attack (moba.h:1533-1541) and a
failed skill (cooldown/mana/target-type) falls through harmlessly.

Mode machine (per hero)
-----------------------
- PUSH: follow the hero's lane waypoints (WAYPOINTS, moba.h:71-72)
  toward the enemy ancient. Lanes mirror the engine's own hero lane
  assignment (init_moba, moba.h:1667-1744): support/carry mid-bot
  lane 2, assassin/burst lane 1, tank lane 0 (+3 for dire). Steering
  is a BFS cost field over the embedded static wall grid (derived from
  vendor/upstream/game_map.h; sha256 ec17403c...), descending one
  8-connected step per tick exactly like the engine's own bfs atn map.
- SIEGE HOLD (inside PUSH): towers scan at the same radius as heroes
  (TOWER_VISION 5, moba.h:45 — no out-ranging) but hit for 110-175 per
  shot (moba.h:65), an order of magnitude over a hero basic attack, so
  diving one without a creep wave soaking its shots is a fast death.
  When a live enemy tower is within its scan radius and no friendly
  creep wave is engaged on it, back off and wait for creeps. Ancients
  are exempt (TOWER_DAMAGE 0 for both, moba.h:65): they cannot shoot,
  so the endgame dive is free. Tower liveness is
  a small world model: a tower whose known cell is visible in the crop
  without a TOWER tile is remembered dead (kill_entity's body zeroes
  the grid cell, moba.h:632-633; called on tower death from attack,
  moba.h:735-737).
- RETREAT: health <= 3/10 sends the hero toward its own fountain
  (spawn, moba.h:1636-1643); passive +2/tick regen (moba.h:1468-1475)
  heals it; hysteresis exits at >= 8/10 so it cannot oscillate.
  Death needs no special mode: the engine respawns instantly at the
  fountain with full health (attack -> spawn_player, moba.h:723-728);
  a position teleport just resets waypoint tracking to the nearest
  waypoint.
- STUCK detour: creeps and heroes block cells; if position hasn't
  moved for a few ticks the desired step is rotated 90 degrees
  (alternating side) for a few ticks to slide around blockers.

Skills (moba.h:1151-1362) are fired by per-role rules on the engine's
Q->W->E priority chain; the target filter is 0 (everything) when
hostile creeps are visible — lane-clearing XP — else 2 (heroes+towers)
so sieging focuses towers and idle neutrals camps are not aggroed.

Determinism: pure function of (obs bytes, per-hero state). No RNG and
no clocks; COGAME_PLAYER_SEED is accepted for interface parity but
unused. Works for 1-hero and 5-hero seats: hero identity (team, hero
type) is read from each obs row, per-hero state is keyed by row index.
"""

from __future__ import annotations

import base64
import sys
import zlib
from collections import deque
from dataclasses import dataclass

from .client import run_policy_main, seed_from_env

# -- action space (mirrors players.random_player.ACT_HIGH) -------------------

ACT_HIGH = (7, 7, 3, 2, 2, 2)

# -- tile ids (moba.h:28-43) -------------------------------------------------

EMPTY, WALL, TOWER = 0, 1, 2
RADIANT_CREEP, DIRE_CREEP, NEUTRAL = 3, 4, 5
# heroes: 6+hero_type for radiant, 11+hero_type for dire

OBS_SIZE = 510
CROP_TILES = 121          # reliable prefix of the 484-byte crop region
CROP_W = 11
VIS = 5                   # vision_range (sim/shim_common.h:17)
SELF_OFF = 484            # 11*11*4

MAP_W = 128

# -- hero roles (init_moba, moba.h:1661-1744) --------------------------------

SUPPORT, ASSASSIN, BURST, TANK, CARRY = range(5)
# hero_type -> radiant lane (dire adds 3); moba.h:1676,1693,1710,1727,1744
LANE_FOR_HERO = (2, 1, 1, 0, 2)
# hero_type -> (base_mana, mana_gain_per_level); moba.h:1672-1743
MANA_TABLE = ((250, 50), (300, 65), (300, 90), (200, 50), (250, 50))

# team -> fountain/spawn cell (init_moba, moba.h:1636-1643)
SPAWN = ((113, 12), (15, 116))

TOWER_VISION = 5          # moba.h:45

# (y, x, team, tier) per tower index; 22 = dire ancient, 23 = radiant
# ancient (moba.h:65-70; ancient pids in c_step, moba.h:1909-1952)
TOWERS = (
    (112.015, 34.6, 0, 3),
    (96.877, 29.308, 0, 3),
    (91.215, 14.292, 0, 3),
    (113.615, 64.2, 0, 2),
    (112.138, 102.877, 0, 1),
    (85.431, 38.292, 0, 2),
    (71.708, 17.615, 0, 2),
    (51.031, 16.877, 0, 1),
    (102.415, 21.062, 0, 4),
    (28.262, 103.031, 1, 4),
    (18.723, 65.0, 1, 2),
    (18.723, 29.062, 1, 1),
    (48.754, 84.2, 1, 2),
    (59.985, 69.031, 1, 1),
    (62.046, 112.754, 1, 2),
    (41.677, 113.738, 1, 3),
    (20.569, 92.323, 1, 3),
    (36.085, 97.862, 1, 3),
    (30.908, 105.615, 1, 4),
    (104.938, 23.523, 0, 4),
    (75.831, 53.123, 0, 1),
    (78.3, 113.223, 1, 1),
    (26.538, 107.523, 1, 5),   # dire ancient
    (106.169, 19.462, 0, 5),   # radiant ancient
)
ANCIENT_IDX = (23, 22)        # per team: own ancient tower index

# (y, x) polylines, lanes 0-2 radiant->dire, 3-5 dire->radiant
# (WAYPOINTS/WAYPOINTS_N, moba.h:71-72)
WAYPOINTS = (
    (  # lane 0 (radiant top), 14 waypoints
        (96.262, 14.046), (93.615, 14.292), (58.231, 16.077),
        (43.4, 16.015), (32.108, 17.585), (24.938, 19.123),
        (22.969, 20.446), (21.062, 25.308), (20.2, 30.477),
        (19.4, 35.892), (18.477, 43.523), (18.046, 50.723),
        (20.508, 94.662), (27.677, 105.946),
    ),
    (  # lane 1 (radiant mid), 10 waypoints
        (99.523, 26.477), (98.662, 27.646), (86.046, 41.123),
        (71.338, 57.492), (66.969, 62.231), (64.077, 66.354),
        (60.2, 72.015), (51.892, 84.877), (34.662, 99.4),
        (28.169, 105.946),
    ),
    (  # lane 2 (radiant bot), 13 waypoints
        (112.015, 36.938), (112.015, 32.508), (116.2, 68.692),
        (113.738, 80.754), (113.554, 90.538), (113.215, 95.985),
        (111.492, 100.938), (109.062, 108.077), (107.185, 111.369),
        (101.738, 114.662), (90.538, 111.769), (39.4, 113.738),
        (28.169, 106.438),
    ),
    (  # lane 3 (dire top), 9 waypoints
        (20.446, 89.369), (20.631, 94.662), (18.785, 50.6),
        (22.292, 24.508), (22.815, 18.815), (27.523, 17.031),
        (35.031, 15.954), (93.615, 14.415), (103.646, 21.992),
    ),
    (  # lane 4 (dire mid), 9 waypoints
        (37.431, 96.508), (34.538, 99.277), (51.708, 84.385),
        (59.831, 71.646), (63.831, 66.108), (66.723, 61.985),
        (70.846, 57.246), (98.538, 27.523), (103.646, 22.485),
    ),
    (  # lane 5 (dire bot), 12 waypoints
        (36.938, 113.246), (39.4, 113.615), (62.292, 114.6),
        (83.646, 114.6), (90.662, 109.8), (94.877, 112.015),
        (104.262, 112.292), (109.308, 107.831), (112.231, 105.862),
        (114.446, 72.969), (111.892, 32.508), (104.138, 22.485),
    ),
)

# 128x128 wall mask (0 empty / 1 wall) from vendor/upstream/game_map.h,
# zlib-compressed + base64 (raw grid sha256
# ec17403cd2c93547f0d9da4da8d0d89ab08160153dc3a68e680e8dc9cc430478).
_WALLS_B64 = (
    "eNrtW+2SIyEIpN//pa+2NpOMCkgjmqu78cdWbda1/YCmQQM87WlPu5pYbd5D6VsH/x5UduJL"
    "TfsyfHYCUojvT032ws/xZSu8iy+/WHvgf0b//WH3ECg2InJs/aqVnsLHEXzeSb8M/3+tfy/5"
    "BJz0bzj/Inx0Tj3z/xr87j/7IWj85BHazLIJX+85DMLiM/FFH0lW8Cli1/q/REUSP3DSVyfE"
    "DofEV7tDA4waeWQCt/1H2NJ8H2gsYsv5FyiQNfsPzmMTfnx7Iz2qxSeoCVTGP3UQFOLDHxDp"
    "OcfwX0lLyC9r8fFheME1kXr8Ydz7p8CMsDMn4+KDG9E0DyTxOR50UUrwSd+O9KWLP6MKQTzO"
    "0ecPW2DdAibCtj707rE7+dIsuPEDZU+jZB/Lf9CLj0H+gPf/u8yY5V/TfKJTZbnwEbE8AwGf"
    "akvQCkn83sD7BKcxi+UE0OaX+wRm4nQth3PiXtyny/E9vl/MGcPRl4wr0RyTiP7ZwoDr14z6"
    "SIpRl1b2rn9uNXP8JqWmMgCoAoHEn2T0YGxd8SRG+aj2y+w+tD1x8TH8P5P/Lvv/IEfQqwAO"
    "n/V/eLPJ4FP86wuCSDFmzpcuNbq/1+TLNv6UEUAudQW/F8JdpSsfG3Pr7+txc/HI2z/M87d8"
    "JcILWFm/n/A0N4pSv/6mDuCRLMKrn+N3fxr6qicPUynS60e7nrEvDF5SFBzI8x+T3uvS3D0q"
    "m8xB4XN1lJDvYg0/HGxs6pKl9Wd0H47VXxFW/cFe3JQUsXJ9jBQ+f5k0zxmp9Qt3l+NuGBj8"
    "e6iJGoffz8+hFVZNWB8mDIEwvmRzPfpw6u4fYKUI7gSL4duwiajlX72xTn3ySU9CT/m0CiCx"
    "GwZxR2qC7SO0JPnBtOy3MvULGA75zHfBzwReT9x8bmAFY3yPuOqU1E9ACvAXGP8EPtbntu/t"
    "UR1+/CXFUO1CAb6pZnvN295PRSJp8P1J5G0GkCrPh84/xY+V9mdfBtlVbaoAweJr4SbFChl8"
    "rRwzPzvU4Qt/9wss8U+4+MZRGVF9TmXCZfjhZR3Dh6G55MT+3/3+XRIHf0u0tP+D/IIsvGEl"
    "vtli8lzmEVni/emeL6kcFF31+VchPr6CDpppNuELdh+67nv++rEdHxH8wmqhnX179Y3d+NNE"
    "1XkrKM0wThJtf/vNuQp+2tOe9i+2P3gjEM0="
)

# Engine step directions in engine order (ATN_MAP, moba.h:58-61, with
# move_towards mapping row 0 -> dy, row 1 -> dx, moba.h:615-617):
# S, N, E, W, SW, NW, NE, SE as (dy, dx). Reused for neighbor iteration
# so tie-breaks are deterministic and engine-like.
STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, -1), (-1, 1), (1, 1))

# -- tuning knobs ------------------------------------------------------------

RETREAT_ENTER_HP = 3      # health tenths; <= enters RETREAT
RETREAT_EXIT_HP = 8       # >= returns to PUSH (hysteresis)
WAYPOINT_REACHED = 2.5    # chebyshev cells; matches creep_ai's <2 spirit
TELEPORT_DIST = 4.0       # position jump > this = respawn/displacement
STUCK_TICKS = 4           # unmoved ticks before detouring
DETOUR_TICKS = 5          # how long one detour burst lasts
SIEGE_STANDOFF = TOWER_VISION + 1   # hold distance from a live tower
CREEP_ENGAGED_DIST = 5    # friendly creep within this of tower = engaged


def _decode_walls() -> bytes:
    grid = zlib.decompress(base64.b64decode("".join(_WALLS_B64)))
    if len(grid) != MAP_W * MAP_W:
        raise ValueError("embedded wall grid has wrong size")
    return grid


# -- perception --------------------------------------------------------------

@dataclass
class HeroObs:
    """Decoded 510-byte observation (see module docstring for layout)."""
    x: int
    y: int
    level: int
    health10: int          # health * 10 / max_health, 0..10
    mana10: int            # mana * 10 / max_mana, 0..10
    damage50: int
    move_speed: int
    move_modifier: int     # int-cast: 0 = slowed (0.5), 1 normal, 2 hasted
    stun_timer: int
    move_timer: int
    q_timer: int
    w_timer: int
    e_timer: int
    basic_timer: int
    basic_cd: int
    is_hit: int
    team: int              # 0 radiant, 1 dire
    hero_type: int         # 0..4 support/assassin/burst/tank/carry
    crop: bytes            # the 121 reliable tile ids, row-major

    def tile(self, dy: int, dx: int) -> int:
        """Crop tile id at offset (dy, dx) from self; -1 if outside."""
        if abs(dy) > VIS or abs(dx) > VIS:
            return -1
        return self.crop[(dy + VIS) * CROP_W + (dx + VIS)]

    def max_mana(self) -> int:
        base, gain = MANA_TABLE[self.hero_type]
        return base + self.level * gain

    def mana_at_least(self, cost: int) -> bool:
        """Conservative: mana10 floor-scaled, so mana >= mana10/10*max."""
        return self.mana10 * self.max_mana() >= cost * 10


def parse_obs(obs: bytes) -> HeroObs:
    """Decode one 510-byte observation row. Robust to garbage bytes:
    every field is taken as-is; hero_type falls back to 0 (support) when
    the one-hot region is empty/ambiguous, team is clamped to 0/1."""
    if len(obs) != OBS_SIZE:
        raise ValueError(f"obs must be {OBS_SIZE} bytes, got {len(obs)}")
    e = obs[SELF_OFF:]
    onehot = e[17:22]
    hero_type = max(range(5), key=lambda i: onehot[i]) if any(onehot) else 0
    return HeroObs(
        x=e[0], y=e[1], level=e[2], health10=e[3], mana10=e[4],
        damage50=e[5], move_speed=e[6], move_modifier=e[7],
        stun_timer=e[8], move_timer=e[9], q_timer=e[10], w_timer=e[11],
        e_timer=e[12], basic_timer=e[13], basic_cd=e[14], is_hit=e[15],
        team=1 if e[16] == 1 else 0, hero_type=hero_type,
        crop=obs[:CROP_TILES])


def hostile_tiles(team: int) -> tuple[frozenset, frozenset]:
    """(enemy hero tile ids, enemy creep tile id-set) for a team."""
    if team == 0:
        return frozenset(range(11, 16)), frozenset((DIRE_CREEP,))
    return frozenset(range(6, 11)), frozenset((RADIANT_CREEP,))


def friendly_creep_tile(team: int) -> int:
    return RADIANT_CREEP if team == 0 else DIRE_CREEP


# -- navigation --------------------------------------------------------------

class NavGrid:
    """BFS cost fields over the embedded static map.

    8-connected uniform-cost BFS from each goal (the same connectivity
    as the engine's bfs, moba.h:242-320), cached per goal cell. All
    goals are static (waypoints, spawns, ancients) so the cache stays
    small. Known tower cells are treated as walls (towers block
    move_to; dead towers leave the cell empty but avoiding it is
    harmless). A wall goal cell is force-opened so fields toward the
    enemy ancient still resolve.
    """

    UNREACHED = 0xFFFF

    def __init__(self):
        walls = bytearray(_decode_walls())
        for ty, tx, _team, _tier in TOWERS:
            walls[int(ty) * MAP_W + int(tx)] = 1
        self._walls = bytes(walls)
        self._fields: dict[tuple[int, int], list[int]] = {}

    def is_wall(self, y: int, x: int) -> bool:
        if not (0 <= y < MAP_W and 0 <= x < MAP_W):
            return True
        return self._walls[y * MAP_W + x] != 0

    def _field(self, goal: tuple[int, int]) -> list[int]:
        cached = self._fields.get(goal)
        if cached is not None:
            return cached
        dist = [self.UNREACHED] * (MAP_W * MAP_W)
        gy, gx = goal
        gy = min(max(gy, 0), MAP_W - 1)
        gx = min(max(gx, 0), MAP_W - 1)
        start = gy * MAP_W + gx
        dist[start] = 0
        queue = deque((start,))
        walls = self._walls
        while queue:
            adr = queue.popleft()
            d1 = dist[adr] + 1
            y, x = adr // MAP_W, adr % MAP_W
            for dy, dx in STEPS:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < MAP_W and 0 <= nx < MAP_W):
                    continue
                nadr = ny * MAP_W + nx
                if dist[nadr] != self.UNREACHED or walls[nadr]:
                    continue
                dist[nadr] = d1
                queue.append(nadr)
        self._fields[goal] = dist
        return dist

    def step_toward(self, y: int, x: int, goal: tuple[int, int],
                    ) -> tuple[int, int]:
        """Best (dy, dx) unit step from (y, x) toward goal.

        Descends the goal's cost field; deterministic tie-break in
        STEPS order. Falls back to the greedy sign step when the
        current cell is unreachable from the goal (e.g. standing on a
        cell our conservative wall model considers blocked)."""
        gy, gx = goal
        if abs(y - gy) <= 1 and abs(x - gx) <= 1:
            return (0, 0)
        field = self._field(goal)
        best, best_d = None, None
        if 0 <= y < MAP_W and 0 <= x < MAP_W:
            here = field[y * MAP_W + x]
        else:
            here = self.UNREACHED
        for dy, dx in STEPS:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < MAP_W and 0 <= nx < MAP_W):
                continue
            d = field[ny * MAP_W + nx]
            if d == self.UNREACHED:
                continue
            if best_d is None or d < best_d:
                best, best_d = (dy, dx), d
        if best is not None and (here == self.UNREACHED or best_d < here):
            return best
        # unreachable (or already at the local minimum): greedy fallback
        sign = lambda v: (v > 0) - (v < 0)  # noqa: E731
        return (sign(gy - y), sign(gx - x))


# -- per-hero state machine --------------------------------------------------

PUSH, RETREAT = "PUSH", "RETREAT"


class HeroState:
    def __init__(self):
        self.mode = PUSH
        self.wp_index: int | None = None    # next waypoint to visit
        self.last_pos: tuple[int, int] | None = None
        self.tried_move = False             # did LAST tick want to move?
        self.stuck_ticks = 0
        self.detour_left = 0
        self.detour_side = 1                # +1 / -1, alternates


def _rotate90(dy: int, dx: int, side: int) -> tuple[int, int]:
    """(dy, dx) rotated 90 degrees; side=+1 clockwise on screen."""
    return (side * dx, -side * dy)


class ScriptedPolicy:
    """policy(tick, obs_rows) -> [[6 ints] per hero].

    Deterministic; ``seed`` is accepted for parity with the other
    players but unused (no RNG anywhere). Per-hero state is keyed by
    row index within the seat; dead-tower memory is shared across the
    seat's heroes (their crops all feed it).
    """

    def __init__(self, seed: int | None = None):
        self.nav = NavGrid()
        self.heroes: dict[int, HeroState] = {}
        self.dead_towers: set[int] = set()

    # -- world model updates ------------------------------------------------

    def _update_tower_memory(self, s: HeroObs) -> None:
        for idx, (ty, tx, _team, _tier) in enumerate(TOWERS):
            if idx in self.dead_towers:
                continue
            dy, dx = int(ty) - s.y, int(tx) - s.x
            if abs(dy) <= VIS and abs(dx) <= VIS:
                if s.tile(dy, dx) != TOWER:
                    self.dead_towers.add(idx)

    def _nearest_live_enemy_tower(self, s: HeroObs,
                                  ) -> tuple[int, int] | None:
        """(tower index, chebyshev distance) of the closest live enemy
        tower that can actually shoot, or None. Ancients (tier 5) are
        exempt: TOWER_DAMAGE is 0 for both ancients (moba.h:65), so
        there is nothing to stand off from — diving the ancient is
        free and standing off would cede endgame time."""
        best = None
        for idx, (ty, tx, team, tier) in enumerate(TOWERS):
            if team == s.team or tier == 5 or idx in self.dead_towers:
                continue
            d = max(abs(int(ty) - s.y), abs(int(tx) - s.x))
            if best is None or d < best[1]:
                best = (idx, d)
        return best

    # -- steering -----------------------------------------------------------

    def _lane(self, s: HeroObs) -> tuple[tuple[float, float], ...]:
        return WAYPOINTS[LANE_FOR_HERO[s.hero_type] + 3 * s.team]

    def _nearest_wp(self, s: HeroObs, lane) -> int:
        return min(range(len(lane)),
                   key=lambda i: max(abs(lane[i][0] - s.y),
                                     abs(lane[i][1] - s.x)))

    def _push_goal(self, s: HeroObs, st: HeroState) -> tuple[int, int]:
        lane = self._lane(s)
        if st.wp_index is None:
            st.wp_index = self._nearest_wp(s, lane)
        while st.wp_index < len(lane):
            wy, wx = lane[st.wp_index]
            if max(abs(wy - s.y), abs(wx - s.x)) > WAYPOINT_REACHED:
                return (int(wy), int(wx))
            st.wp_index += 1
        ay, ax, _t, _tier = TOWERS[ANCIENT_IDX[1 - s.team]]
        return (int(ay), int(ax))

    # -- combat signals -----------------------------------------------------

    @staticmethod
    def _crop_scan(s: HeroObs):
        """(enemy hero dist, hostile creep dist, friendly creep cells)
        from the reliable crop tiles; dists are chebyshev or None."""
        heroes, creeps = hostile_tiles(s.team)
        friendly = friendly_creep_tile(s.team)
        hero_d = creep_d = None
        friendly_cells = []
        for dy in range(-VIS, VIS + 1):
            yy = s.y + dy
            if not (0 <= yy < MAP_W):
                continue
            for dx in range(-VIS, VIS + 1):
                xx = s.x + dx
                if not (0 <= xx < MAP_W):
                    continue
                t = s.crop[(dy + VIS) * CROP_W + (dx + VIS)]
                if t == EMPTY or t == WALL:
                    continue
                d = max(abs(dy), abs(dx))
                if t in heroes:
                    if hero_d is None or d < hero_d:
                        hero_d = d
                elif t in creeps:
                    if creep_d is None or d < creep_d:
                        creep_d = d
                elif t == friendly:
                    friendly_cells.append((yy, xx))
        return hero_d, creep_d, friendly_cells

    def _skill_flags(self, s: HeroObs, mode: str, hero_d, creep_d,
                     ) -> tuple[int, int, int]:
        """(use_q, use_w, use_e) per role. The engine re-checks
        cooldown/mana/target and fails skills harmlessly, so these are
        intent flags, gated just enough to avoid wasting mana."""
        enemy_hero = hero_d is not None
        hostile = enemy_hero or creep_d is not None
        h = s.hero_type
        if h == SUPPORT:      # hook / aoe heal / stun (moba.h:1151-1187)
            return (int(enemy_hero and s.mana_at_least(100)),
                    int(s.health10 <= 6 and s.mana_at_least(100)),
                    int(enemy_hero and s.mana_at_least(75)))
        if h == ASSASSIN:     # aoe minions / tp nuke / haste (1189-1233)
            return (int(creep_d is not None and s.mana_at_least(100)),
                    int(enemy_hero and s.health10 >= 5
                        and s.mana_at_least(150)),
                    int(mode == RETREAT and s.mana_at_least(100)))
        if h == BURST:        # nuke / aoe / aoe stun (1235-1272)
            return (int(enemy_hero and s.mana_at_least(200)),
                    int(enemy_hero and s.mana_at_least(100)),
                    int(enemy_hero and s.mana_at_least(75)))
        if h == TANK:         # aoe dot / self heal / engage (1274-1313)
            low = s.health10 <= 5
            return (int(hostile and not low and s.mana_at_least(5)),
                    int(low and s.mana_at_least(100)),
                    int(enemy_hero and s.health10 > 6
                        and s.mana_at_least(50)))
        # CARRY: retreat slow / slow nuke / aoe (1315-1362)
        return (int(mode == RETREAT and hostile and s.mana_at_least(25)),
                int(enemy_hero and s.mana_at_least(150)),
                int(creep_d is not None and s.mana_at_least(100)))

    # -- per-hero tick ------------------------------------------------------

    def _hero_action(self, idx: int, obs: bytes) -> list[int]:
        s = parse_obs(obs)
        st = self.heroes.setdefault(idx, HeroState())
        self._update_tower_memory(s)

        # respawn/teleport detection resets waypoint tracking
        pos = (s.y, s.x)
        if (st.last_pos is not None
                and max(abs(pos[0] - st.last_pos[0]),
                        abs(pos[1] - st.last_pos[1])) > TELEPORT_DIST):
            st.wp_index = None
            st.stuck_ticks = 0
            st.detour_left = 0

        # mode with hysteresis
        if st.mode == PUSH and s.health10 <= RETREAT_ENTER_HP:
            st.mode = RETREAT
        elif st.mode == RETREAT and s.health10 >= RETREAT_EXIT_HP:
            st.mode = PUSH
            st.wp_index = None      # re-anchor to the nearest waypoint

        hero_d, creep_d, friendly_creeps = self._crop_scan(s)

        # goal selection
        hold = False
        if st.mode == RETREAT:
            goal = SPAWN[s.team]
        else:
            goal = self._push_goal(s, st)
            tower = self._nearest_live_enemy_tower(s)
            if tower is not None and tower[1] <= SIEGE_STANDOFF:
                tidx, tdist = tower
                ty, tx = int(TOWERS[tidx][0]), int(TOWERS[tidx][1])
                engaged = any(
                    max(abs(cy - ty), abs(cx - tx)) <= CREEP_ENGAGED_DIST
                    for cy, cx in friendly_creeps)
                if engaged:
                    goal = (ty, tx)          # push onto the tower
                elif tdist <= TOWER_VISION:
                    goal = SPAWN[s.team]     # inside range, no wave: out
                else:
                    hold = True              # at standoff: wait for creeps

        # steering
        if hold:
            dy = dx = 0
        else:
            dy, dx = self.nav.step_toward(s.y, s.x, goal)

        # stuck detection -> deterministic 90-degree detour. A tick
        # only counts as stuck if the PREVIOUS tick actually tried to
        # move and the position stayed put; coming out of an
        # intentional hold must not count as a phantom stuck tick.
        if st.tried_move and st.last_pos == pos and (dy or dx):
            st.stuck_ticks += 1
        else:
            st.stuck_ticks = 0
        if st.detour_left > 0:
            st.detour_left -= 1
            dy, dx = _rotate90(dy, dx, st.detour_side)
        elif st.stuck_ticks >= STUCK_TICKS:
            st.detour_left = DETOUR_TICKS
            st.detour_side = -st.detour_side
            st.stuck_ticks = 0
            dy, dx = _rotate90(dy, dx, st.detour_side)
        st.tried_move = bool(dy or dx)
        st.last_pos = pos

        # target filter: everything while hostile creeps are around
        # (lane XP), heroes+towers otherwise (focus towers, don't
        # aggro neutral camps)
        target_filter = 0 if creep_d is not None else 2
        use_q, use_w, use_e = self._skill_flags(s, st.mode, hero_d, creep_d)

        return [3 + 3 * dy, 3 + 3 * dx, target_filter, use_q, use_w, use_e]

    def __call__(self, tick: int, obs_rows: list) -> list:
        return [self._hero_action(i, bytes(row))
                for i, row in enumerate(obs_rows)]


def policy_from_env() -> ScriptedPolicy:
    return ScriptedPolicy(seed_from_env(default=None))


def main() -> int:
    return run_policy_main(policy_from_env)


if __name__ == "__main__":
    sys.exit(main())
