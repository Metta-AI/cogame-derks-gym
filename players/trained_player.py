"""Run a frozen Metta Fabric checkpoint as an ordinary Derk's Gym player."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from cogame_derks_gym import catalog, defaults
from cogame_derks_gym.training import (
    ACTION_SIZES, OBSERVATION_SIZE, numeric_action_mask, numeric_observation,
)

from .client import run_policy_main, seed_from_env, ws_url_from_env


@dataclass(frozen=True)
class NumericInput:
    values: list[list[float]]
    action_masks: list[list[bool]]


class TrainedPolicy:
    def __init__(self, frozen, *, seat: int, seed: int):
        if (frozen.spec.observation_size != OBSERVATION_SIZE
                or frozen.spec.action_sizes != list(ACTION_SIZES)):
            raise ValueError("checkpoint dimensions differ from Derk's Gym")
        if not 0 <= seat < defaults.NUM_SEATS:
            raise ValueError("invalid Derk's Gym seat")
        self.frozen = frozen
        self.seat = seat
        self.loadout = 0
        self.result_seen = False
        self.frozen.reset(str(seed))
        self.frozen.reset_seat(seat)

    def _predict(self, raw: bytes, *, draft: bool) -> list[int]:
        values = numeric_observation(raw, draft=draft,
                                     seat=self.seat, loadout=self.loadout)
        observed = NumericInput(values=[values.tolist()],
                                action_masks=[list(numeric_action_mask(draft=draft))])
        return self.frozen.sample(self.frozen.predict(self.seat, observed))

    def on_draft(self, observation: dict) -> dict[str, str]:
        if observation["seat"] != self.seat:
            raise ValueError("draft seat differs from player slot")
        action = self._predict(bytes(OBSERVATION_SIZE - 3), draft=True)
        loadout = action[0]
        return {
            slot: catalog.ITEMS[slot][(loadout // (4 ** index)) % 4]["id"]
            for index, slot in enumerate(catalog.SLOTS)
        }

    def on_draft_result(self, message: dict) -> None:
        record = next(rec for rec in message["loadouts"]
                      if rec["seat"] == self.seat)
        self.loadout = sum(
            next(index for index, item in enumerate(catalog.ITEMS[slot])
                 if item["id"] == record["picks"][slot]) * (4 ** digit)
            for digit, slot in enumerate(catalog.SLOTS)
        )
        self.result_seen = True

    def __call__(self, tick: int, obs_rows: list[bytes]) -> list[list[int]]:
        if not self.result_seen:
            raise RuntimeError("tick arrived before accepted draft result")
        if len(obs_rows) != 1:
            raise ValueError("Derk's Gym seat must control one hero")
        return [self._predict(obs_rows[0], draft=False)[1:]]


def main() -> int:
    from metta_training.inference import FrozenPolicy
    from metta_training.policy_bundle import load_frozen_policy_bundle

    query = parse_qs(urlparse(ws_url_from_env()).query)
    seat = int(query["slot"][0])
    bundle = Path(os.environ["DERKS_POLICY_BUNDLE"])
    frozen = FrozenPolicy(load_frozen_policy_bundle(bundle))
    # Compile inference before connecting: the game starts its shared draft
    # clock after seats connect. TrainedPolicy resets state and RNG afterward.
    frozen.predict(seat, NumericInput(
        values=[numeric_observation(bytes(OBSERVATION_SIZE - 3), draft=True,
                                    seat=seat, loadout=0).tolist()],
        action_masks=[list(numeric_action_mask(draft=True))],
    ))
    return run_policy_main(lambda: TrainedPolicy(
        frozen, seat=seat, seed=seed_from_env(0)))


if __name__ == "__main__":
    raise SystemExit(main())
