"""Binary replay format v2: header JSON + packed per-tick actions.

Layout:

    bytes 0-3   magic b"DERK"
    byte  4     format version, u8 = 2
    bytes 5-8   header_len, u32 little-endian
    ...         header JSON, utf-8, header_len bytes
    ...         body: tick_count * 60 bytes
                (per tick: 10 heroes x 6 uint8 action values, post-clamp,
                 exactly as fed to the sim)

The header is **self-sufficient**: everything the static viewer needs
lives in these bytes and nothing but the replay file is ever fetched.

    format_version      2
    sim_wasm_sha256     the sim build that recorded it
    catalog_version     pinned, so a future catalog cannot silently
                        re-interpret this replay
    catalog             the full catalog object (item names + deltas, so
                        the viewer renders them without the repo)
    config              fully resolved game config incl. seed and real
                        player names (names MUST live in the replay bytes
                        per the Coworld static-viewer contract; tokens
                        excluded)
    aliases             the six in-game seat aliases
    seat_hero_pids      seat -> hero pid
    house_hero_pids     the four in-process house heroes
    draft               the ten draft-reveal records (picks + the applied
                        stat blocks the viewer pushes into the sim)
    loadout_digest      u32 FNV-1a over the applied table; the viewer
                        re-derives it and warns on a mismatch
    events              <= 400 replay events (feed + scrubber beats)
    result              the full results.json document
    tick_count          body length / 60
    final_state_digest  u32 FNV-1a over hero x/y/health + ancients

A replay plus the pinned sim wasm fully determines the episode: re-run
the sim from ``config.seed``, apply ``draft[].applied``, feed each tick's
actions, and every obs/reward byte reproduces (proved by
tests/test_replay.py, not assumed — the digests are the cross-check).

Replay v1 (``MOBA``) is deliberately NOT read here: old cogame-moba
replays are watched in cogame-moba.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import defaults
from .config import GameConfig
from .sim import DEFAULT_WASM_PATH

MAGIC = b"DERK"
FORMAT_VERSION = 2
_PREFIX_LEN = len(MAGIC) + 1 + 4  # magic + version u8 + header_len u32le
BYTES_PER_TICK = defaults.NUM_HEROES * defaults.ACTIONS_PER_HERO  # 60


class ReplayError(ValueError):
    """Malformed or unsupported replay bytes."""


def sim_wasm_sha256(wasm_path: str | Path = DEFAULT_WASM_PATH) -> str:
    """Hex sha256 of the sim wasm binary (recorded in replay headers)."""
    return hashlib.sha256(Path(wasm_path).read_bytes()).hexdigest()


class ReplayWriter:
    """Accumulates per-tick actions; finalize() renders the full file.

    Body is buffered in memory: at the 6000-tick cap an episode is
    6000 x 60 B = 360 kB, so streaming to disk buys nothing.
    """

    def __init__(self, config: GameConfig, sim_wasm_sha256: str, *,
                 catalog_obj: dict | None = None):
        self._config = config
        self._sha = sim_wasm_sha256
        self._body = bytearray()
        self._tick_count = 0
        if catalog_obj is None:
            from . import catalog as _catalog
            catalog_obj = _catalog.catalog_dict()
        self._catalog = catalog_obj
        self._draft: list[dict] = []
        self._loadout_digest = 0
        self._events: list[dict] = []
        self._final_state_digest = 0

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def set_draft(self, records: list[dict], loadout_digest: int) -> None:
        """The ten draft-reveal records + the sim's loadout digest."""
        self._draft = list(records)
        self._loadout_digest = int(loadout_digest) & 0xFFFFFFFF

    def set_events(self, events: list[dict]) -> None:
        self._events = list(events)

    def set_final_state_digest(self, digest: int) -> None:
        self._final_state_digest = int(digest) & 0xFFFFFFFF

    def append_tick(self, tick: int, actions: np.ndarray) -> None:
        """Record one tick's (10, 6) post-clamp action matrix.

        Signature matches the engine's ``on_tick`` hook; ``tick`` must be
        the next sequential tick (catches skipped/duplicated ticks).
        """
        if tick != self._tick_count:
            raise ValueError(
                f"non-sequential tick {tick}, expected {self._tick_count}")
        actions = np.asarray(actions)
        if actions.shape != (defaults.NUM_HEROES, defaults.ACTIONS_PER_HERO):
            raise ValueError(
                f"actions must be ({defaults.NUM_HEROES}, "
                f"{defaults.ACTIONS_PER_HERO}), got {actions.shape}")
        # Replays store post-clamp values; anything outside the
        # MultiDiscrete range would silently wrap in the uint8 cast below
        # (e.g. 300 -> 44) and corrupt the re-simulation.
        high = np.asarray(defaults.ACT_HIGH)
        if (actions < 0).any() or (actions >= high).any():
            raise ValueError(
                f"action values out of range (per-column highs "
                f"{defaults.ACT_HIGH}, exclusive): {actions.tolist()}")
        self._body += actions.astype(np.uint8).tobytes()
        self._tick_count += 1

    def header(self, result: dict) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "sim_wasm_sha256": self._sha,
            "catalog_version": self._catalog["version"],
            "catalog": self._catalog,
            "config": self._config.to_dict(),
            "aliases": list(defaults.SEAT_ALIASES),
            "seat_hero_pids": list(defaults.SEAT_HERO_PIDS),
            "house_hero_pids": list(defaults.HOUSE_HERO_PIDS),
            "draft": self._draft,
            "loadout_digest": self._loadout_digest,
            "events": self._events,
            "result": result,
            "tick_count": self._tick_count,
            "final_state_digest": self._final_state_digest,
        }

    def finalize(self, result: dict) -> bytes:
        """Render the complete replay file with the episode result."""
        # ensure_ascii keeps the header bytes 7-bit even when a player's
        # `note` carries non-ASCII text, so the slice always decodes with
        # errors="strict" (see the strict-UTF-8 test in test_replay.py).
        header = json.dumps(
            self.header(result), separators=(",", ":")).encode("utf-8")
        return b"".join((
            MAGIC,
            bytes([FORMAT_VERSION]),
            len(header).to_bytes(4, "little"),
            header,
            bytes(self._body),
        ))


class Replay:
    """Parsed replay: validated header + per-tick action access."""

    def __init__(self, header: dict, body: bytes):
        self.header = header
        self._body = body
        self.tick_count = header["tick_count"]

    @classmethod
    def parse(cls, data: bytes) -> "Replay":
        if len(data) < _PREFIX_LEN:
            raise ReplayError(f"replay too short ({len(data)} bytes)")
        if data[:4] != MAGIC:
            raise ReplayError(f"bad magic {data[:4]!r}, expected {MAGIC!r}")
        if data[4] != FORMAT_VERSION:
            raise ReplayError(
                f"unsupported format version {data[4]}, "
                f"expected {FORMAT_VERSION}")
        header_len = int.from_bytes(data[5:9], "little")
        # No wrap: header_len is a u32 read out of the file and
        # _PREFIX_LEN + header_len is computed in Python ints, so a
        # 0xFFFFFFFF length is rejected here rather than slicing to the
        # end of the buffer.
        if _PREFIX_LEN + header_len > len(data):
            raise ReplayError("header extends past end of file")
        raw_header = data[_PREFIX_LEN:_PREFIX_LEN + header_len]
        try:
            # errors="strict": a byte-boundary truncation of a recorded
            # string must fail loudly here, not render "mostly fine" in a
            # browser and fail a strict JSON parser downstream.
            header = json.loads(raw_header.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ReplayError(f"header is not valid JSON: {exc}") from exc
        if not isinstance(header, dict) or \
                not isinstance(header.get("tick_count"), int):
            raise ReplayError("header missing integer tick_count")
        body = data[_PREFIX_LEN + header_len:]
        expected = header["tick_count"] * BYTES_PER_TICK
        if len(body) != expected:
            raise ReplayError(
                f"body is {len(body)} bytes, expected {expected} "
                f"({header['tick_count']} ticks x {BYTES_PER_TICK})")
        return cls(header, body)

    def actions(self, tick: int) -> np.ndarray:
        """The (10, 6) uint8 action matrix for one tick."""
        if not 0 <= tick < self.tick_count:
            raise IndexError(f"tick {tick} out of range 0..{self.tick_count - 1}")
        start = tick * BYTES_PER_TICK
        return np.frombuffer(
            self._body, dtype=np.uint8,
            count=BYTES_PER_TICK, offset=start,
        ).reshape(defaults.NUM_HEROES, defaults.ACTIONS_PER_HERO)

    def __iter__(self):
        for tick in range(self.tick_count):
            yield self.actions(tick)

    def __len__(self) -> int:
        return self.tick_count
