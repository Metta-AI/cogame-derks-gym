"""Game config model for the Coworld runtime contract.

The config JSON arrives via ``COGAME_CONFIG_URI``. Shape (paintarena/
coworld-ctf conventions: ``players`` and ``tokens`` are parallel arrays in
seat-slot order):

    {
      "seed": 1234,                          // optional; derived if absent
      "max_ticks": 6000,
      "tick_deadline_ms": 100,
      "player_connect_timeout_seconds": 60,
      "wall_clock_budget_seconds": 645,      // optional; derived if absent
      "draft_enabled": true,                 // false = Puffer-fidelity mode
      "draft_deadline_ms": 45000,
      "catalog_version": "v1",
      "num_agents": 6,                       // optional; must equal 6
      "players": [{"name": "..."}, ...],     // exactly 6
      "tokens": ["token-0", ...]             // exactly 6
    }

There is no ``heroes_per_seat``: the seat -> hero map is the fixed tuple
``defaults.SEAT_HERO_PIDS`` and the four house heroes are driven in
process (see :mod:`cogame_derks_gym.defaults`).

A missing seed is derived once at parse time and recorded on the resolved
config so it always reaches the replay header.
"""

from __future__ import annotations

import json
import math
import secrets
from dataclasses import dataclass
from pathlib import Path

from . import catalog, defaults


class ConfigError(ValueError):
    """Invalid or inconsistent game config."""


@dataclass(frozen=True)
class PlayerConfig:
    name: str


@dataclass(frozen=True)
class GameConfig:
    players: tuple[PlayerConfig, ...]
    tokens: tuple[str, ...]
    seed: int
    max_ticks: int
    tick_deadline_ms: int
    player_connect_timeout_seconds: float
    # Engine hard stop (end_reason="wall_clock"), timed from the start of
    # the draft turn: keeps the worst-case episode under the platform's
    # episode_timeout kill, so artifacts are always written.
    wall_clock_budget_seconds: float
    # The draft turn. draft_enabled False is the Puffer-fidelity mode: no
    # draft turn and no apply_loadout call at all, so the sim is
    # byte-identical to the upstream env the policies were trained on.
    draft_enabled: bool
    draft_deadline_ms: int
    catalog_version: str

    @property
    def num_seats(self) -> int:
        return len(self.players)

    @classmethod
    def from_dict(cls, data: dict) -> "GameConfig":
        if not isinstance(data, dict):
            raise ConfigError(f"config must be a JSON object, got {type(data).__name__}")

        players_raw = data.get("players")
        if not isinstance(players_raw, list) or not players_raw:
            raise ConfigError("config requires a non-empty 'players' array")
        players = []
        for i, entry in enumerate(players_raw):
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) \
                    or not entry["name"]:
                raise ConfigError(f"players[{i}] must be an object with a non-empty 'name'")
            players.append(PlayerConfig(name=entry["name"]))

        tokens_raw = data.get("tokens")
        if not isinstance(tokens_raw, list) or \
                not all(isinstance(t, str) and t for t in tokens_raw):
            raise ConfigError("config requires a 'tokens' array of non-empty strings")
        if len(tokens_raw) != len(players):
            raise ConfigError(
                f"tokens length {len(tokens_raw)} != players length {len(players)}")

        if len(players) != defaults.NUM_SEATS:
            raise ConfigError(
                f"this game seats exactly {defaults.NUM_SEATS} players "
                f"(one hero each; the other {len(defaults.HOUSE_HERO_PIDS)} "
                f"heroes are house-driven), got {len(players)}")

        # num_agents is what the platform ladder seats by; the game server
        # only cross-checks it (a variant edited without the design note
        # must fail loudly, not play a different game).
        num_agents = _int_field(data, "num_agents", defaults.NUM_SEATS)
        if num_agents != defaults.NUM_SEATS:
            raise ConfigError(
                f"num_agents must be {defaults.NUM_SEATS}, got {num_agents}")

        draft_enabled = data.get("draft_enabled", defaults.DEFAULT_DRAFT_ENABLED)
        if not isinstance(draft_enabled, bool):
            raise ConfigError(
                f"draft_enabled must be a boolean, got {draft_enabled!r}")

        draft_deadline_ms = _int_field(
            data, "draft_deadline_ms", defaults.DEFAULT_DRAFT_DEADLINE_MS)
        if draft_deadline_ms < defaults.MIN_DRAFT_DEADLINE_MS:
            raise ConfigError(
                f"draft_deadline_ms must be >= "
                f"{defaults.MIN_DRAFT_DEADLINE_MS}, got {draft_deadline_ms}")

        catalog_version = data.get("catalog_version", catalog.CATALOG_VERSION)
        if catalog_version != catalog.CATALOG_VERSION:
            # Pinned so a future catalog cannot silently re-interpret an
            # old replay.
            raise ConfigError(
                f"catalog_version must be {catalog.CATALOG_VERSION!r}, "
                f"got {catalog_version!r}")

        max_ticks = _int_field(data, "max_ticks", defaults.DEFAULT_MAX_TICKS)
        if max_ticks <= 0:
            raise ConfigError(f"max_ticks must be positive, got {max_ticks}")

        tick_deadline_ms = _int_field(
            data, "tick_deadline_ms", defaults.DEFAULT_TICK_DEADLINE_MS)
        if tick_deadline_ms <= 0:
            raise ConfigError(
                f"tick_deadline_ms must be positive, got {tick_deadline_ms}")

        timeout = data.get("player_connect_timeout_seconds",
                           defaults.DEFAULT_PLAYER_CONNECT_TIMEOUT_SECONDS)
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) \
                or not math.isfinite(timeout) or timeout < 0:
            raise ConfigError(
                "player_connect_timeout_seconds must be a finite non-negative "
                f"number, got {timeout!r}")

        budget = data.get(
            "wall_clock_budget_seconds",
            defaults.derived_wall_clock_budget_seconds(
                max_ticks, tick_deadline_ms))
        if not isinstance(budget, (int, float)) or isinstance(budget, bool) \
                or not math.isfinite(budget) or budget <= 0:
            raise ConfigError(
                "wall_clock_budget_seconds must be a finite positive number, "
                f"got {budget!r}")

        seed = data.get("seed")
        if seed is None:
            # Derive once and record it: the seed must reach the replay header.
            seed = secrets.randbits(32)
        elif not isinstance(seed, int) or isinstance(seed, bool):
            raise ConfigError(f"seed must be an integer, got {seed!r}")
        # The sim consumes a u32 (MobaSim masks likewise); mask HERE so
        # the canonical seed recorded in results and the replay header is
        # the value the sim actually ran with.
        seed &= 0xFFFFFFFF

        return cls(
            players=tuple(players),
            tokens=tuple(tokens_raw),
            seed=seed,
            max_ticks=max_ticks,
            tick_deadline_ms=tick_deadline_ms,
            player_connect_timeout_seconds=float(timeout),
            wall_clock_budget_seconds=float(budget),
            draft_enabled=draft_enabled,
            draft_deadline_ms=draft_deadline_ms,
            catalog_version=catalog_version,
        )

    @classmethod
    def from_file_uri(cls, uri: str) -> "GameConfig":
        """Parse a config from a local ``file://`` URI or plain path.

        Local-only convenience (tests, dev). The server reads
        ``COGAME_CONFIG_URI`` through :mod:`cogame_derks_gym.uris`, which
        also supports http(s).
        """
        path = uri.removeprefix("file://")
        try:
            raw = Path(path).read_text()
        except OSError as exc:
            raise ConfigError(f"cannot read config from {uri}: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"config at {uri} is not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Fully-resolved config for the replay header and results.

        Tokens are deliberately excluded: replays and results are public
        artifacts, tokens are per-episode player credentials.
        """
        return {
            "seed": self.seed,
            "max_ticks": self.max_ticks,
            "tick_deadline_ms": self.tick_deadline_ms,
            "player_connect_timeout_seconds": self.player_connect_timeout_seconds,
            "wall_clock_budget_seconds": self.wall_clock_budget_seconds,
            "draft_enabled": self.draft_enabled,
            "draft_deadline_ms": self.draft_deadline_ms,
            "catalog_version": self.catalog_version,
            "players": [{"name": p.name} for p in self.players],
        }


def _int_field(data: dict, key: str, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{key} must be an integer, got {value!r}")
    return value
