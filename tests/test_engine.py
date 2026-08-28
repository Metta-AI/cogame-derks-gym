"""Tests for the transport-free lockstep episode engine."""

import asyncio

import numpy as np
import pytest

from cogame_derks_gym import defaults
from cogame_derks_gym.config import GameConfig
from cogame_derks_gym.engine import EpisodeResult, LockstepEngine

NOOP = list(defaults.NOOP_ACTION)


SEATS = defaults.NUM_SEATS  # 6
HEROES = defaults.NUM_HEROES  # 10 (six seats + four house heroes)
SEAT_PIDS = defaults.SEAT_HERO_PIDS  # (0, 1, 2, 5, 6, 7)
LAST_PID = SEAT_PIDS[-1]             # 7: the hero of the last seat
HOUSE_PIDS = defaults.HOUSE_HERO_PIDS


def make_config(**overrides):
    d = {
        "players": [{"name": f"p{i}"} for i in range(SEATS)],
        "tokens": [f"t{i}" for i in range(SEATS)],
        "seed": 5,
        "max_ticks": 10,
        "tick_deadline_ms": 200,
    }
    d.update(overrides)
    return GameConfig.from_dict(d)


class FakeSim:
    """Records actions fed per tick; obs row i is filled with byte value i."""

    def __init__(self, done_at=None, winner_team=0,
                 ancient_healths=(100.0, 100.0)):
        self._tick = 0
        self.done_at = done_at
        self.winner_team = winner_team
        self.ancient_healths = list(ancient_healths)
        self.fed_actions = []  # list of (10, 6) float32 arrays
        self.loadouts = {}     # pid -> applied stat block

    def observations(self):
        return np.tile(
            np.arange(10, dtype=np.uint8).reshape(10, 1), (1, 510))

    def set_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        assert actions.shape == (10, 6)
        self.fed_actions.append(actions.copy())

    def step(self):
        self._tick += 1

    def rewards(self):
        # hero pid p earns reward p+1 each tick
        return np.arange(1, 11, dtype=np.float32)

    def done(self):
        return int(self.done_at is not None and self._tick >= self.done_at)

    def winner(self):
        return self.winner_team

    def tick(self):
        return self._tick

    def agent_stat(self, pid, which):
        return pid * 100 + which

    def ancient_health(self, team):
        return self.ancient_healths[team]

    # the loadout half of the sim contract (sim/shim.c apply_loadout)
    def apply_loadout(self, pid, block):
        self.loadouts[pid] = dict(block)

    def loadout_digest(self):
        return len(self.loadouts)

    def state_digest(self):
        return 0xABCDEF


class ScriptedSource:
    """Returns a fixed per-hero action list every tick; records the obs seen."""

    def __init__(self, actions):
        self.actions = actions
        self.seen = []  # (tick, obs) pairs

    async def get_actions(self, tick, obs):
        self.seen.append((tick, obs.copy()))
        return self.actions


class SlowSource:
    async def get_actions(self, tick, obs):
        await asyncio.sleep(60)
        return [NOOP]


class NoneSource:
    async def get_actions(self, tick, obs):
        return None


class RaisingSource:
    async def get_actions(self, tick, obs):
        raise RuntimeError("player exploded")


class MalformedSource:
    def __init__(self, payload):
        self.payload = payload

    async def get_actions(self, tick, obs):
        return self.payload


# -- action routing ----------------------------------------------------------

async def test_scripted_actions_reach_seat_hero_rows():
    """Seat i's actions land on hero SEAT_HERO_PIDS[i]; the four house
    hero rows stay NOOP when no house source is given."""
    sim = FakeSim()
    sources = [ScriptedSource([[i, i % 7, 1, 0, 1, 0]]) for i in range(SEATS)]
    cfg = make_config(max_ticks=3)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.final_tick == 3
    assert len(sim.fed_actions) == 3
    for fed in sim.fed_actions:
        for seat, pid in enumerate(SEAT_PIDS):
            assert fed[pid].tolist() == [seat, seat % 7, 1, 0, 1, 0]
        for pid in HOUSE_PIDS:
            assert fed[pid].tolist() == NOOP


class FakeHouse:
    """A house source: fixed actions for the four house heroes, and it
    records the call order (which must be ascending pid: the brain has one
    shared rand() stream and puffernet SAMPLES from the softmax)."""

    def __init__(self):
        self.calls = []

    def actions(self, obs):
        self.calls.append([obs[pid][0] for pid in HOUSE_PIDS])
        return {pid: [4, 4, 2, 1, 0, 1] for pid in HOUSE_PIDS}


async def test_house_heroes_drive_their_rows_in_ascending_pid_order():
    sim = FakeSim()
    house = FakeHouse()
    sources = [ScriptedSource([[1, 1, 0, 0, 0, 0]]) for _ in range(SEATS)]
    cfg = make_config(max_ticks=2)
    await LockstepEngine(sim, cfg, sources, house_source=house).run()
    assert len(house.calls) == 2
    # obs row p is filled with byte p, so this asserts the pid order
    assert house.calls[0] == list(HOUSE_PIDS)
    fed = sim.fed_actions[0]
    for pid in HOUSE_PIDS:
        assert fed[pid].tolist() == [4, 4, 2, 1, 0, 1]
    for pid in SEAT_PIDS:
        assert fed[pid].tolist() == [1, 1, 0, 0, 0, 0]


async def test_house_source_failure_degrades_to_noop(capsys):
    """A house source that raises (missing brain wasm, a wasm trap) is
    logged once and its heroes play NOOP; the seats keep playing."""

    class Exploding:
        def actions(self, obs):
            raise RuntimeError("brain exploded")

    sim = FakeSim()
    sources = [ScriptedSource([[1, 1, 0, 0, 0, 0]]) for _ in range(SEATS)]
    cfg = make_config(max_ticks=3)
    result = await LockstepEngine(
        sim, cfg, sources, house_source=Exploding()).run()
    assert result.final_tick == 3
    for fed in sim.fed_actions:
        for pid in HOUSE_PIDS:
            assert fed[pid].tolist() == NOOP
        assert fed[SEAT_PIDS[0]].tolist() == [1, 1, 0, 0, 0, 0]
    err = capsys.readouterr().err
    assert err.count("house heroes: source raised") == 1


async def test_seat_obs_is_its_single_hero_row():
    sim = FakeSim()
    sources = [ScriptedSource([NOOP]) for _ in range(SEATS)]
    cfg = make_config(max_ticks=1)
    await LockstepEngine(sim, cfg, sources).run()
    for seat, src in enumerate(sources):
        (tick, obs), = src.seen
        assert tick == 0
        assert obs.shape == (1, 510)
        assert obs[0, 0] == SEAT_PIDS[seat]


# -- NOOP fallbacks ----------------------------------------------------------

@pytest.mark.parametrize("bad_source", [
    NoneSource(),
    RaisingSource(),
    MalformedSource("garbage"),
    MalformedSource([[1, 2, 3]]),                    # wrong shape
    MalformedSource([[1, 2, 3, 4, 5]]),              # 5 values not 6
    MalformedSource([[float("nan")] * 6]),           # non-finite
    MalformedSource([["a", "b", "c", "d", "e", "f"]]),
])
async def test_bad_sources_get_noop(bad_source):
    sim = FakeSim()
    sources = [ScriptedSource([[1, 1, 1, 1, 1, 1]])
               for _ in range(SEATS - 1)]
    sources.append(bad_source)
    cfg = make_config(max_ticks=2)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.final_tick == 2  # episode never crashes
    for fed in sim.fed_actions:
        assert fed[LAST_PID].tolist() == NOOP
        assert fed[SEAT_PIDS[0]].tolist() == [1, 1, 1, 1, 1, 1]


async def test_deadline_timeout_gets_noop():
    sim = FakeSim()
    sources = [ScriptedSource([[2, 2, 0, 0, 0, 0]])
               for _ in range(SEATS - 1)]
    sources.append(SlowSource())
    cfg = make_config(max_ticks=2, tick_deadline_ms=50)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.final_tick == 2
    for fed in sim.fed_actions:
        assert fed[LAST_PID].tolist() == NOOP


async def test_out_of_range_actions_clamped():
    sim = FakeSim()
    sources = [ScriptedSource([[99, -3, 7, 2, 1, -1]])] + \
        [ScriptedSource([NOOP]) for _ in range(SEATS - 1)]
    cfg = make_config(max_ticks=1)
    ticks = []
    engine = LockstepEngine(sim, cfg, sources,
                            on_tick=lambda t, a: ticks.append((t, a.copy())))
    await engine.run()
    assert sim.fed_actions[0][0].tolist() == [6, 0, 2, 1, 1, 0]
    # replay hook sees the same post-clamp values, as uint8
    (t0, acts0), = ticks
    assert t0 == 0
    assert acts0.dtype == np.uint8
    assert acts0[0].tolist() == [6, 0, 2, 1, 1, 0]
    assert acts0[LAST_PID].tolist() == NOOP


async def test_noop_causes_classified_per_seat():
    """Every NOOP fallback is attributed to a cause (results
    observability: noop_causes)."""
    sim = FakeSim()
    sources = [ScriptedSource([NOOP]), ScriptedSource([NOOP])] + [
        SlowSource(),                # -> timeout
        NoneSource(),                # -> disconnected (source had nothing)
        RaisingSource(),             # -> host_error
        MalformedSource("garbage"),  # -> malformed
    ]
    cfg = make_config(max_ticks=3, tick_deadline_ms=50)
    result = await LockstepEngine(sim, cfg, sources).run()
    causes = result.seat_noop_causes
    assert causes[2]["timeout"] == 3
    assert causes[3]["disconnected"] == 3
    assert causes[4]["host_error"] == 3
    assert causes[5]["malformed"] == 3
    for s in range(2):
        assert sum(causes[s].values()) == 0, (s, causes[s])
    # every noop tick is attributed
    for s in range(SEATS):
        assert sum(causes[s].values()) == result.seat_noop_ticks[s]


# -- termination + scoring ---------------------------------------------------

async def test_ancient_win_scores_by_team():
    sim = FakeSim(done_at=4, winner_team=1)
    sources = [ScriptedSource([NOOP]) for _ in range(SEATS)]
    cfg = make_config(max_ticks=100)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert isinstance(result, EpisodeResult)
    assert result.end_reason == "ancient"
    assert result.winner == 1
    assert result.final_tick == 4
    assert list(result.seat_scores) == [0.0] * 3 + [1.0] * 3


async def test_tick_cap_tiebreak_by_ancient_health():
    sim = FakeSim(ancient_healths=(50.0, 200.0))
    sources = [ScriptedSource([NOOP]) for _ in range(SEATS)]
    cfg = make_config(max_ticks=5)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.end_reason == "tick_cap"
    assert result.winner == 1
    assert result.final_tick == 5
    assert list(result.seat_scores) == [0.0] * 3 + [1.0] * 3
    assert result.ancient_healths == (50.0, 200.0)


async def test_tick_cap_equal_health_is_draw():
    sim = FakeSim(ancient_healths=(80.0, 80.0))
    sources = [ScriptedSource([NOOP]) for _ in range(SEATS)]
    cfg = make_config(max_ticks=5)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.winner is None
    assert list(result.seat_scores) == [0.5] * SEATS


async def test_reward_sums_and_stats():
    sim = FakeSim()
    sources = [ScriptedSource([NOOP]) for _ in range(SEATS)]
    cfg = make_config(max_ticks=3)
    result = await LockstepEngine(sim, cfg, sources).run()
    # hero pid p earns p+1 per tick, 3 ticks, one hero per seat
    assert list(result.seat_reward_sums) == pytest.approx(
        [(pid + 1) * 3 for pid in SEAT_PIDS])
    assert len(result.agent_stats) == HEROES
    assert result.agent_stats[2]["kills"] == 2 * 100 + 1
    assert result.agent_stats[7]["level"] == 7 * 100 + 0


async def test_wall_clock_budget_ends_episode():
    """A slow episode ends at the wall-clock budget with
    end_reason="wall_clock" and the usual ancient-health tiebreak, well
    before the platform's episode_timeout kill (which would lose the
    results and replay entirely)."""

    class Slowish:
        async def get_actions(self, tick, obs):
            await asyncio.sleep(0.02)
            return [NOOP]

    sim = FakeSim(ancient_healths=(50.0, 200.0))
    sources = [Slowish() for _ in range(SEATS)]
    cfg = make_config(max_ticks=10_000, tick_deadline_ms=1000,
                      wall_clock_budget_seconds=0.4)
    result = await asyncio.wait_for(
        LockstepEngine(sim, cfg, sources).run(), timeout=10)
    assert result.end_reason == "wall_clock"
    assert 0 < result.final_tick < 10_000
    assert result.winner == 1  # ancient-health tiebreak, like tick_cap
    assert list(result.seat_scores) == [0.0] * 3 + [1.0] * 3


# -- sim fault containment (patch 0004) --------------------------------------

class FaultingSim(FakeSim):
    """fault() goes nonzero after `fault_at` steps (patch-0004 flag)."""

    def __init__(self, fault_at, **kw):
        super().__init__(**kw)
        self.fault_at = fault_at

    def fault(self):
        return 2 if self._tick >= self.fault_at else 0


class TrapDeath(Exception):
    """Stands in for a wasmtime trap (e.g. an ExitTrap on exit())."""


class TrappingSim(FakeSim):
    """step() traps after `trap_at` steps and every call raises after —
    like a wasm instance whose execution trapped."""

    def __init__(self, trap_at, **kw):
        super().__init__(**kw)
        self.trap_at = trap_at
        self.dead = False

    def _check(self):
        if self.dead:
            raise TrapDeath("wasm instance is dead")

    def step(self):
        self._check()
        if self._tick >= self.trap_at:
            self.dead = True
            raise TrapDeath("wasm trapped in step()")
        super().step()

    def observations(self):
        self._check()
        return super().observations()

    def rewards(self):
        self._check()
        return super().rewards()

    def done(self):
        self._check()
        return super().done()

    def tick(self):
        self._check()
        return super().tick()

    def agent_stat(self, pid, which):
        self._check()
        return super().agent_stat(pid, which)

    def ancient_health(self, team):
        self._check()
        return super().ancient_health(team)


async def test_sim_fault_flag_ends_episode_as_sim_fault():
    sim = FaultingSim(fault_at=3)
    sources = [ScriptedSource([NOOP]) for _ in range(SEATS)]
    cfg = make_config(max_ticks=50)
    ticks = []
    result = await LockstepEngine(
        sim, cfg, sources, on_tick=lambda t, a: ticks.append(t)).run()
    assert result.end_reason == "sim_fault"
    assert result.final_tick == 3
    assert result.winner is None
    assert list(result.seat_scores) == [0.5] * SEATS
    # the faulting tick completed and is in the replay
    assert ticks == [0, 1, 2]


async def test_sim_trap_contained_as_sim_fault():
    """A hard wasm trap mid-step must not escape the engine: the episode
    ends as sim_fault with best-effort result fields, so the server can
    still write results and the partial replay."""
    sim = TrappingSim(trap_at=4)
    sources = [ScriptedSource([NOOP]) for _ in range(SEATS)]
    cfg = make_config(max_ticks=50)
    ticks = []
    result = await LockstepEngine(
        sim, cfg, sources, on_tick=lambda t, a: ticks.append(t)).run()
    assert result.end_reason == "sim_fault"
    assert result.winner is None
    assert list(result.seat_scores) == [0.5] * SEATS
    # 4 ticks completed before the trap; the trapped tick is not recorded
    assert ticks == [0, 1, 2, 3]
    assert result.final_tick == 4
    # dead-instance reads fall back instead of raising
    assert result.ancient_healths == (0.0, 0.0)
    assert all(v == 0 for stats in result.agent_stats
               for v in stats.values())


async def test_real_sim_fault_export_is_zero():
    """The patched wasm exports moba_fault(); a normal episode never
    trips it (the fidelity gate relies on exactly that)."""
    from cogame_derks_gym.sim import MobaSim

    sim = MobaSim(seed=11)
    assert sim.fault() == 0
    sources = [ScriptedSource([NOOP]) for _ in range(SEATS)]
    cfg = make_config(max_ticks=5, tick_deadline_ms=2000)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.end_reason == "tick_cap"
    assert sim.fault() == 0


# -- real wasm sim end-to-end ------------------------------------------------

async def test_real_sim_end_to_end():
    from cogame_derks_gym.sim import MobaSim

    class RngSource:
        def __init__(self, seat, heroes):
            self.rng = np.random.default_rng(seat)
            self.heroes = heroes

        async def get_actions(self, tick, obs):
            assert obs.shape == (self.heroes, 510)
            return self.rng.integers(
                0, defaults.ACT_HIGH, size=(self.heroes, 6)).tolist()

    sim = MobaSim(seed=11)
    cfg = make_config(max_ticks=40, tick_deadline_ms=2000)
    sources = [RngSource(seat, 1) for seat in range(SEATS)]
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.final_tick == 40
    assert result.end_reason == "tick_cap"
    assert sim.tick() == 40
    assert all(np.isfinite(result.seat_reward_sums))
    assert result.ancient_healths[0] > 0 and result.ancient_healths[1] > 0
    assert all(s["level"] >= 1 for s in result.agent_stats)


async def test_real_sim_seat_slicing():
    """Each seat sees exactly its own hero's obs row out of the real sim."""
    from cogame_derks_gym.sim import MobaSim

    seen = {}

    class CaptureSource:
        def __init__(self, seat):
            self.seat = seat

        async def get_actions(self, tick, obs):
            if tick == 0:
                seen[self.seat] = obs.copy()
            return [NOOP]

    sim = MobaSim(seed=11)
    full_obs = sim.observations()
    cfg = make_config(max_ticks=2, tick_deadline_ms=2000)
    await LockstepEngine(
        sim, cfg, [CaptureSource(s) for s in range(SEATS)]).run()
    for seat, pid in enumerate(SEAT_PIDS):
        np.testing.assert_array_equal(seen[seat], full_obs[pid:pid + 1])


# -- silent-seat strike rule -------------------------------------------------

async def test_silent_seat_goes_dead_and_episode_races():
    """After 10 consecutive timeouts a seat is marked dead: its deadline is
    no longer waited on, so a silent seat costs at most ~strike_limit x
    deadline of wall clock for the whole episode."""
    import time

    sim = FakeSim()
    sources = [ScriptedSource([NOOP])
               for _ in range(SEATS - 1)] + [SlowSource()]
    cfg = make_config(max_ticks=40, tick_deadline_ms=100)
    t0 = time.monotonic()
    result = await LockstepEngine(sim, cfg, sources).run()
    elapsed = time.monotonic() - t0
    assert result.final_tick == 40
    # 10 strike ticks x 100ms deadline ~= 1s; the other 30 ticks are instant.
    # A non-dead silent seat would cost 40 x 100ms = 4s+.
    assert elapsed < 3.0
    assert result.seat_dead[SEATS - 1] is True
    assert result.seat_noop_ticks[SEATS - 1] == 40
    assert result.seat_dead[0] is False
    assert result.seat_noop_ticks[0] == 0


async def test_dead_seat_revives_on_valid_action():
    """A dead seat keeps getting revival probes; the first valid reply
    resets its strikes and resumes normal play (late reconnects work)."""

    class WakingSource:
        """Times out for the first `sleep_asks` asks, then answers instantly."""

        def __init__(self, sleep_asks):
            self.asks = 0
            self.sleep_asks = sleep_asks

        async def get_actions(self, tick, obs):
            self.asks += 1
            if self.asks <= self.sleep_asks:
                await asyncio.sleep(60)
            return [[1, 1, 1, 1, 1, 1]]

    sim = FakeSim()
    waking = WakingSource(sleep_asks=10)
    sources = [ScriptedSource([NOOP]) for _ in range(SEATS - 1)] + [waking]
    cfg = make_config(max_ticks=20, tick_deadline_ms=50)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.final_tick == 20
    # seat went dead (>= 10 strikes) but revived and played real actions
    assert result.seat_dead[SEATS - 1] is False
    assert 10 <= result.seat_noop_ticks[SEATS - 1] < 20
    assert sim.fed_actions[-1][LAST_PID].tolist() == [1, 1, 1, 1, 1, 1]


async def test_dead_revive_dead_revive_cycle():
    """A second dead spell must be revivable: the stale probe left over
    from the first revival (which may hang forever, e.g. a ws waiter
    clobbered by the next tick's send) must not block re-probing.

    Ask timeline for the flaky seat (strike limit 10, see phases):
      asks 1-10   hang    -> 10 strikes, seat dead
      ask  11     answer  -> revival probe reply
      ask  12     hang forever -> the leftover probe created on the
                   revival tick (with the fix it is cancelled and may
                   even be cancelled before it starts; either way later
                   asks shift by at most one, which the phases absorb)
      asks 13-14  answer  -> live play
      asks 15-24  hang    -> 10 strikes, seat dead again
      asks 25+    answer  -> second revival must happen
    """

    class DoubleWaker:
        def __init__(self):
            self.asks = 0

        async def get_actions(self, tick, obs):
            self.asks += 1
            a = self.asks
            if a <= 10 or 15 <= a <= 24:
                await asyncio.sleep(60)   # deadline-cancelled: strike
            elif a == 12:
                await asyncio.sleep(600)  # leftover probe: hangs forever
            return [[1, 1, 1, 1, 1, 1]]

    sim = FakeSim()
    sources = [ScriptedSource([NOOP]) for _ in range(SEATS - 1)] + [DoubleWaker()]
    cfg = make_config(max_ticks=34, tick_deadline_ms=50)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.final_tick == 34
    # the seat died twice but ended the episode revived and playing
    assert result.seat_dead[SEATS - 1] is False
    assert result.seat_noop_ticks[SEATS - 1] >= 20
    assert sim.fed_actions[-1][LAST_PID].tolist() == [1, 1, 1, 1, 1, 1]


async def test_on_seat_dead_fires_once_per_death_transition():
    """The engine reports each strike-death transition exactly once (the
    server uses this to force-close the seat's stale websocket)."""

    class Phased:
        """Hangs for asks 1-10 (death), answers 11-13 (revive), hangs
        14-23 (second death), answers after."""

        def __init__(self):
            self.asks = 0

        async def get_actions(self, tick, obs):
            self.asks += 1
            a = self.asks
            if a <= 10 or 14 <= a <= 23:
                await asyncio.sleep(60)
            return [[1, 1, 1, 1, 1, 1]]

    deaths = []
    sim = FakeSim()
    sources = [ScriptedSource([NOOP]) for _ in range(SEATS - 1)] + [Phased()]
    cfg = make_config(max_ticks=40, tick_deadline_ms=30)
    result = await LockstepEngine(
        sim, cfg, sources, on_seat_dead=deaths.append).run()
    assert result.final_tick == 40
    assert deaths.count(SEATS - 1) == 2, deaths
    assert set(deaths) == {SEATS - 1}


async def test_on_seat_dead_exception_never_crashes_episode():
    def boom(seat):
        raise RuntimeError("callback exploded")

    sim = FakeSim()
    sources = [ScriptedSource([NOOP]) for _ in range(SEATS - 1)] + [NoneSource()]
    cfg = make_config(max_ticks=15, tick_deadline_ms=30)
    result = await LockstepEngine(
        sim, cfg, sources, on_seat_dead=boom).run()
    assert result.final_tick == 15
    assert result.seat_dead[SEATS - 1] is True


async def test_all_seats_dead_event_loop_keeps_yielding():
    """With EVERY seat dead the per-tick gather is empty and awaits
    nothing: without an explicit yield the while loop starves the event
    loop, stalling /healthz (liveness kill) and the revival probes
    themselves. The engine must yield each tick even with no live seats."""

    sim = FakeSim()
    seen_ticks = set()

    async def heartbeat():
        while True:
            seen_ticks.add(sim.tick())
            await asyncio.sleep(0)

    sources = [SlowSource() for _ in range(SEATS)]
    cfg = make_config(max_ticks=100, tick_deadline_ms=10)
    hb = asyncio.create_task(heartbeat())
    try:
        result = await asyncio.wait_for(
            LockstepEngine(sim, cfg, sources).run(), timeout=30)
    finally:
        hb.cancel()
    assert result.final_tick == 100
    assert all(result.seat_dead)
    # The heartbeat must observe the all-dead stretch tick by tick. A
    # starved loop lets it see only the pre-dead ticks (and possibly the
    # final tick, at the engine's post-loop probe cleanup await).
    dead_stretch = {t for t in seen_ticks if 15 <= t < 100}
    assert len(dead_stretch) > 50, sorted(seen_ticks)


async def test_all_seats_dead_then_one_revives():
    """Revival must work even from the all-dead state: the probe tasks
    only run if the engine yields to the event loop each tick."""

    class WakingSource:
        def __init__(self, sleep_asks):
            self.asks = 0
            self.sleep_asks = sleep_asks

        async def get_actions(self, tick, obs):
            self.asks += 1
            if self.asks <= self.sleep_asks:
                await asyncio.sleep(60)
            return [[1, 1, 1, 1, 1, 1]]

    sim = FakeSim()
    sources = [SlowSource() for _ in range(SEATS - 1)] + [WakingSource(10)]
    cfg = make_config(max_ticks=60, tick_deadline_ms=10)
    result = await asyncio.wait_for(
        LockstepEngine(sim, cfg, sources).run(), timeout=30)
    assert result.final_tick == 60
    assert result.seat_dead[SEATS - 1] is False
    assert sim.fed_actions[-1][LAST_PID].tolist() == [1, 1, 1, 1, 1, 1]


async def test_progress_heartbeat_and_revival_logged(capsys):
    """Observability: a slow episode emits throttled progress lines, and
    a revival logs the seat + tick."""

    class WakingSource:
        def __init__(self, sleep_asks):
            self.asks = 0
            self.sleep_asks = sleep_asks

        async def get_actions(self, tick, obs):
            self.asks += 1
            if self.asks <= self.sleep_asks:
                await asyncio.sleep(60)
            return [[1, 1, 1, 1, 1, 1]]

    sim = FakeSim()
    sources = [ScriptedSource([NOOP]) for _ in range(SEATS - 1)] + [WakingSource(10)]
    cfg = make_config(max_ticks=20, tick_deadline_ms=50)
    result = await LockstepEngine(
        sim, cfg, sources, progress_interval_seconds=0.05).run()
    assert result.final_tick == 20
    err = capsys.readouterr().err
    assert "progress: tick " in err
    assert "seat 5 revived at tick" in err


async def test_valid_action_resets_strike_counter():
    """Strikes are consecutive: a valid reply resets the count, so an
    intermittently slow seat is never marked dead."""

    class Intermittent:
        def __init__(self):
            self.asks = 0

        async def get_actions(self, tick, obs):
            self.asks += 1
            if self.asks % 3 == 0:
                return [[2, 2, 0, 0, 0, 0]]
            await asyncio.sleep(60)
            return None

    sim = FakeSim()
    sources = [ScriptedSource([NOOP]) for _ in range(SEATS - 1)] + [Intermittent()]
    cfg = make_config(max_ticks=12, tick_deadline_ms=30)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.seat_dead[SEATS - 1] is False
    assert result.final_tick == 12


# -- replay events (see cogame_derks_gym.events) ------------------------------

class StatSim(FakeSim):
    """Drives the four event-bearing stats from a per-tick script."""

    def __init__(self, script, **kw):
        super().__init__(**kw)
        self.script = script  # {tick: {pid: {stat: value}}}
        self.stats = [dict.fromkeys(
            ("level", "kills", "deaths", "towers_killed"), 0)
            for _ in range(HEROES)]
        for pid in range(HEROES):
            self.stats[pid]["level"] = 1

    def step(self):
        super().step()
        for pid, updates in self.script.get(self._tick, {}).items():
            self.stats[pid].update(updates)

    def agent_stat(self, pid, which):
        names = ("level", "kills", "deaths", "towers_killed")
        if which < len(names):
            return self.stats[pid][names[which]]
        return 0


async def test_events_extracted_from_stat_increases():
    from cogame_derks_gym.events import EventLog

    sim = StatSim({
        1: {0: {"kills": 1}, 6: {"deaths": 1}},          # first blood
        2: {2: {"towers_killed": 1}},                    # tower
        3: {1: {"level": 4}},                            # level_spike
        4: {5: {"kills": 1}, 1: {"deaths": 1}},          # a later kill
    })
    log = EventLog()
    log.add_draft()
    sources = [ScriptedSource([NOOP]) for _ in range(SEATS)]
    cfg = make_config(max_ticks=5)
    result = await LockstepEngine(
        sim, cfg, sources, event_log=log).run()
    kinds = [(e["tick"], e["kind"]) for e in result.events]
    assert kinds[0] == (0, "draft")
    assert (0, "first_blood") in kinds  # emitted on the tick-0 read of tick 1
    by_kind = {e["kind"]: e for e in result.events}
    assert by_kind["first_blood"]["pid"] == 0
    assert by_kind["first_blood"]["victim_pid"] == 6
    assert by_kind["tower"]["pid"] == 2
    assert by_kind["tower"]["team"] == 0
    assert by_kind["level_spike"]["pid"] == 1
    assert by_kind["level_spike"]["level"] == 4
    assert by_kind["kill"]["pid"] == 5
    assert by_kind["kill"]["victim_pid"] == 1


async def test_event_log_caps_at_400_dropping_level_spikes_first():
    from cogame_derks_gym.events import MAX_EVENTS, EventLog

    log = EventLog()
    log.add_draft()
    log.add(1, "first_blood", pid=0, victim_pid=5)
    for tick in range(2, 2 + MAX_EVENTS):
        log.add(tick, "level_spike", pid=0, level=2)
    for tick in range(1000, 1010):
        log.add(tick, "kill", pid=0, victim_pid=5)
    log.add_end(2000, "tick_cap")
    events = log.events()
    assert len(events) <= MAX_EVENTS
    kinds = [e["kind"] for e in events]
    # the undroppable records survive; level_spikes were shed first
    assert kinds[0] == "draft"
    assert "first_blood" in kinds
    assert kinds[-1] == "end"
    assert kinds.count("kill") == 10
