"""Headless training match using the same wasm simulator as hosted Derk's Gym."""

from dataclasses import dataclass

import numpy as np

from players.baseline_player import MobaBrain
from players.derk_player import FORGE_BY_ROLE

from . import catalog, defaults
from .house import HouseHeroes
from .sim import MobaSim, OBS_SIZE

ACTION_SIZES = (64, *defaults.ACT_HIGH)
OBSERVATION_SIZE = OBS_SIZE + 3  # phase, seat, selected loadout


@dataclass(frozen=True)
class TrainingStep:
    observation: np.ndarray
    action_mask: tuple[bool, ...]
    reward: float
    done: bool
    score: float | None


class TrainingMatch:
    """Train one seat against puffer-forge seats and production house heroes.

    Action head zero chooses one of the 64 catalog loadouts on the draft turn.
    Heads one through six carry the upstream MultiDiscrete micro action.
    Inactive heads admit only their no-op value.
    """

    def __init__(self, seat: int, max_ticks: int = defaults.DEFAULT_MAX_TICKS,
                 draft_enabled: bool = True):
        if not 0 <= seat < defaults.NUM_SEATS or max_ticks < 1:
            raise ValueError("need a valid seat and positive tick limit")
        self.seat = seat
        self.pid = defaults.pid_for_seat(seat)
        self.max_ticks = max_ticks
        self.draft_enabled = draft_enabled

    def reset(self, seed: int) -> TrainingStep:
        self.sim = MobaSim(seed=seed)
        self.house = HouseHeroes(seed=1)
        # Hosted seats each own a brain process and therefore a separate
        # sampling stream and recurrent state.
        self.opponents = {
            pid: MobaBrain(seed=1)
            for pid in defaults.SEAT_HERO_PIDS if pid != self.pid
        }
        self.phase = "draft" if self.draft_enabled else "micro"
        self.loadout = 0
        self.done = False
        return self._state(0.0, False)

    def _state(self, reward: float, done: bool) -> TrainingStep:
        obs = (np.zeros(OBS_SIZE, dtype=np.float32) if self.phase == "draft" else
               self.sim.observations()[self.pid].astype(np.float32) / 255.0)
        values = np.concatenate((obs, np.array([
            float(self.phase == "draft"), self.seat / (defaults.NUM_SEATS - 1),
            self.loadout / 63.0,
        ], dtype=np.float32)))
        if self.phase == "draft":
            mask = [True] * 64
            for size, noop in zip(defaults.ACT_HIGH, defaults.NOOP_ACTION, strict=True):
                mask.extend(index == noop for index in range(size))
        else:
            mask = [index == 0 for index in range(64)]
            for size in defaults.ACT_HIGH:
                mask.extend([True] * size)
        score = None
        if done:
            if self.sim.done():
                winner = self.sim.winner()
            else:
                radiant, dire = (self.sim.ancient_health(team)
                                 for team in range(defaults.NUM_TEAMS))
                winner = 0 if radiant > dire else 1 if dire > radiant else -1
            score = (0.5 if winner < 0 else
                     float(winner == defaults.team_for_pid(self.pid)))
        return TrainingStep(values, tuple(mask), reward, done, score)

    def step(self, action: list[int]) -> TrainingStep:
        if self.done:
            raise RuntimeError("training match has ended")
        if len(action) != len(ACTION_SIZES):
            raise ValueError("expected a loadout head and six micro heads")
        if self.phase == "draft":
            if not 0 <= action[0] < 64 or tuple(action[1:]) != defaults.NOOP_ACTION:
                raise ValueError("draft action has an invalid loadout or micro heads")
            self.loadout = action[0]
            choices = tuple(catalog.ITEMS[slot] for slot in catalog.SLOTS)
            picks = {
                slot: choices[index][(self.loadout // (4 ** index)) % 4]["id"]
                for index, slot in enumerate(catalog.SLOTS)
            }
            for pid in range(defaults.NUM_HEROES):
                if pid == self.pid:
                    selected = picks
                elif pid in defaults.HOUSE_HERO_PIDS:
                    selected = catalog.NEUTRAL_PICKS
                else:
                    selected = FORGE_BY_ROLE[defaults.role_for_pid(pid)]
                self.sim.apply_loadout(pid, catalog.apply_picks(
                    defaults.HERO_BASE[pid], selected))
            self.phase = "micro"
            return self._state(0.0, False)
        if action[0] != 0 or any(
            not 0 <= value < high
            for value, high in zip(action[1:], defaults.ACT_HIGH, strict=True)
        ):
            raise ValueError("micro action is outside the upstream action space")
        obs = self.sim.observations()
        actions = np.tile(np.asarray(defaults.NOOP_ACTION, dtype=np.float32),
                          (defaults.NUM_HEROES, 1))
        for pid, row in self.house.actions(obs).items():
            actions[pid] = row
        for pid in defaults.SEAT_HERO_PIDS:
            if pid != self.pid:
                actions[pid] = self.opponents[pid].forward(0, obs[pid].tobytes())
        actions[self.pid] = action[1:]
        self.sim.set_actions(actions)
        self.sim.step()
        if self.sim.fault():
            raise RuntimeError("Derk's Gym simulator faulted")
        done = bool(self.sim.done()) or self.sim.tick() >= self.max_ticks
        self.done = done
        return self._state(float(self.sim.rewards()[self.pid]), done)
