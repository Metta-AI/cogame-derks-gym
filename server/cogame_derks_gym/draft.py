"""The pre-match loadout draft: one turn, simultaneous, hidden.

All six seats decide at the same time and no seat sees any other seat's
pick before committing — that hiding IS the metagame, so a counter-draft
must be a prediction, not a reaction. There is no shared pool and no
exclusivity (two heroes may take the same item), so there is nothing to
contend for and therefore no pick-order tie-break to define: the whole
phase is one parallel batch under one shared deadline instead of a
six-round snake draft.

Degrade, never hang: every failure mode below resolves to the neutral
loadout for that seat and the match starts on time. A draft failure is
never fatal and never adds an ``end_reason``.

Resolution order per seat (docs/PROTOCOL.md carries the same table):

1. no reply by the shared deadline          -> neutral, "timeout"
2. never connected / socket closed          -> neutral, "disconnected"
3. frame larger than 4096 bytes             -> neutral, "oversize"
4. not JSON / not an object / picks not a
   1-element array of objects               -> neutral, "wrong_shape"
5. any of arm/tail/misc missing, not a
   string, > 24 chars, or not an id of the
   MATCHING slot in the catalog             -> neutral, "unknown_item"
6. otherwise                                -> accepted, "none"

Partial acceptance is deliberately not allowed in (5): it would let a
seat launder a typo into a free reroll of one slot.
"""

from __future__ import annotations

import asyncio
import sys
import time
import unicodedata
from typing import Protocol

from . import catalog, defaults
from .config import GameConfig

# Closed enum, triple-synced with the manifest results_schema and the
# replay header's draft records.
FALLBACK_CAUSES = ("none", "timeout", "malformed", "wrong_shape",
                   "unknown_item", "disconnected", "oversize")
SOURCES = ("seat", "house")

# Upstream match constants shown to a drafting seat (all from
# vendor/upstream/moba.h): Ancient health TOWER_HEALTH[22..23],
# lane-tower damage range TOWER_DAMAGE, creep wave period, passive regen.
ANCIENT_HEALTH = 4500
TOWER_DAMAGE_RANGE = (110, 175)
CREEP_WAVE_EVERY = 150
REGEN_PER_TICK = 2


class DraftSource(Protocol):
    """Per-seat draft provider (a websocket seat, or a test double)."""

    async def get_draft(self, observation: dict) -> tuple[object, str | None]:
        """``(reply, cause)`` for one seat's single draft message.

        ``reply`` is the decoded JSON object the seat sent, or None with
        a FALLBACK_CAUSES cause ("disconnected" / "oversize" /
        "wrong_shape"). Raising is allowed: the caller treats it as
        "wrong_shape". Being slow is allowed: the caller's shared
        deadline cancels the wait and the seat times out.
        """
        ...


def truncate_note(value: object) -> str:
    """The only free-text field in the protocol, made safe to record.

    - non-string / absent -> ""
    - C0/C1 control characters and lone surrogates removed (a replay
      header must decode with errors="strict")
    - truncated to 120 **Unicode scalars**, never mid-codepoint: a
      byte-boundary truncation is how a replay ends up failing a strict
      JSON parser while still rendering in a browser.
    """
    if not isinstance(value, str):
        return ""
    cleaned = "".join(
        ch for ch in value
        if not (0xD800 <= ord(ch) <= 0xDFFF)
        # category Cc covers both C0 (0x00-0x1F, 0x7F) and C1 (0x80-0x9F)
        and unicodedata.category(ch) != "Cc")
    return cleaned[:catalog.MAX_NOTE_RUNES]


def hero_view(pid: int) -> dict:
    """The drafting seat's own hero: identity, role, lane, skills, base
    stats. Everything else about the match is hidden (see docstring)."""
    role = defaults.role_for_pid(pid)
    base = defaults.HERO_BASE[pid]
    view = {
        "pid": pid,
        "role": role,
        "lane": defaults.lane_for_pid(pid),
        "skills": list(defaults.ROLE_SKILLS[role]),
    }
    for field in catalog.STAT_FIELDS:
        value = base[field]
        view[field] = int(value) if field in catalog.INT_FIELDS \
            else catalog.f32(value)
    return view


def match_view(cfg: GameConfig) -> dict:
    return {
        "max_ticks": cfg.max_ticks,
        "tick_deadline_ms": cfg.tick_deadline_ms,
        "ancient_health": ANCIENT_HEALTH,
        "creep_wave_every": CREEP_WAVE_EVERY,
        "tower_damage": list(TOWER_DAMAGE_RANGE),
        "regen_per_tick": REGEN_PER_TICK,
    }


def draft_observation(seat: int, cfg: GameConfig) -> dict:
    """The draft observation for one seat.

    Visible: its own hero's identity/role/lane/skills/base stats, the
    aliases and roles of the other five seats and the four house heroes,
    the whole catalog with exact deltas, the clamp table, the match
    constants, its own deadline.

    Hidden: every other seat's pick (the simultaneity), the sim seed, and
    every real player name — including its own (docs/PROTOCOL.md, "Two
    name spaces"). A test asserts no real name appears here.
    """
    pid = defaults.pid_for_seat(seat)
    team = defaults.team_for_pid(pid)
    teammates, opponents = [], []
    for other in range(defaults.NUM_SEATS):
        if other == seat:
            continue
        entry = {"alias": defaults.SEAT_ALIASES[other],
                 "role": defaults.role_for_pid(defaults.pid_for_seat(other))}
        (teammates if defaults.team_for_seat(other) == team
         else opponents).append(entry)
    return {
        "phase": "draft",
        "seat": seat,
        "alias": defaults.SEAT_ALIASES[seat],
        "team": defaults.TEAM_NAMES[team],
        "hero": hero_view(pid),
        "teammates": teammates,
        "opponents": opponents,
        "house_heroes": [
            {"team": defaults.TEAM_NAMES[defaults.team_for_pid(house_pid)],
             "role": defaults.role_for_pid(house_pid)}
            for house_pid in defaults.HOUSE_HERO_PIDS],
        "catalog": catalog.catalog_dict(),
        "clamps": catalog.clamps_dict(),
        "match": match_view(cfg),
        "deadline_ms": cfg.draft_deadline_ms,
    }


def resolve_reply(reply: object, cause: str | None,
                  ) -> tuple[dict[str, str], str, str]:
    """``(picks, note, fallback_cause)`` for one seat's reply.

    Implements steps 3-7 of the module docstring's resolution order;
    steps 1-2 arrive here as ``cause``.
    """
    if cause is not None:
        return dict(catalog.NEUTRAL_PICKS), "", cause
    if not isinstance(reply, dict) or reply.get("phase") != "draft":
        return dict(catalog.NEUTRAL_PICKS), "", "wrong_shape"
    picks_raw = reply.get("picks")
    if not isinstance(picks_raw, list) or len(picks_raw) != 1 \
            or not isinstance(picks_raw[0], dict):
        return dict(catalog.NEUTRAL_PICKS), "", "wrong_shape"
    entry = picks_raw[0]
    # A bad note never invalidates the picks.
    note = truncate_note(entry.get("note"))
    picks = catalog.normalized_picks(entry)
    if picks is None:
        return dict(catalog.NEUTRAL_PICKS), note, "unknown_item"
    return picks, note, "none"


def record(pid: int, *, picks: dict[str, str], note: str,
           source: str, fallback_cause: str, decision_ms: int,
           player_name: str | None) -> dict:
    """One draft-reveal record (replay header, results.draft, and the
    ``draft_result`` message — the last one alias-only)."""
    assert source in SOURCES, source
    assert fallback_cause in FALLBACK_CAUSES, fallback_cause
    seat = defaults.seat_for_pid(pid)
    return {
        "pid": pid,
        "seat": seat,
        "alias": defaults.alias_for_pid(pid),
        "player_name": player_name,
        "team": defaults.TEAM_NAMES[defaults.team_for_pid(pid)],
        "role": defaults.role_for_pid(pid),
        "picks": dict(picks),
        "note": note,
        "source": source,
        "fallback": fallback_cause != "none",
        "fallback_cause": fallback_cause,
        "decision_ms": decision_ms,
        "applied": catalog.apply_picks(defaults.HERO_BASE[pid], picks),
    }


def house_record(pid: int) -> dict:
    """A house hero: neutral loadout, no seat, no player name."""
    return record(pid, picks=dict(catalog.NEUTRAL_PICKS), note="",
                  source="house", fallback_cause="none", decision_ms=0,
                  player_name=None)


def neutral_records(cfg: GameConfig) -> list[dict]:
    """The ten records of an un-drafted (draft_enabled false) episode.

    Every hero runs the neutral loadout and the server applies nothing at
    all, so these records describe the sim exactly.
    """
    records = []
    for pid in range(defaults.NUM_HEROES):
        seat = defaults.seat_for_pid(pid)
        if seat is None:
            records.append(house_record(pid))
        else:
            records.append(record(
                pid, picks=dict(catalog.NEUTRAL_PICKS), note="",
                source="seat", fallback_cause="none", decision_ms=0,
                player_name=cfg.players[seat].name))
    return records


async def _one_seat(source: DraftSource, observation: dict,
                    ) -> tuple[object, str | None, int]:
    started = time.monotonic()
    try:
        reply, cause = await source.get_draft(observation)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # a source can never break the draft
        print(f"seat {observation['seat']}: draft source raised "
              f"{type(exc).__name__}: {exc} (neutral loadout)",
              file=sys.stderr)
        return None, "wrong_shape", int((time.monotonic() - started) * 1000)
    return reply, cause, int((time.monotonic() - started) * 1000)


async def run_draft(cfg: GameConfig, sources: list[DraftSource],
                    ) -> list[dict]:
    """Run the single draft turn and return the ten draft records.

    All six seats are asked as ONE parallel batch (a single
    ``asyncio.gather``) under one shared deadline: the turn costs
    ``draft_deadline_ms`` of wall clock no matter how many seats are slow,
    which is what keeps the episode inside 60% of the platform budget.
    """
    if len(sources) != defaults.NUM_SEATS:
        raise ValueError(
            f"need {defaults.NUM_SEATS} draft sources, got {len(sources)}")
    observations = [draft_observation(seat, cfg)
                    for seat in range(defaults.NUM_SEATS)]
    deadline = cfg.draft_deadline_ms / 1000.0

    async def batch():
        return await asyncio.gather(*(
            _one_seat(source, observation)
            for source, observation in zip(sources, observations)))

    started = time.monotonic()
    try:
        gathered = await asyncio.wait_for(batch(), deadline)
    except (asyncio.TimeoutError, TimeoutError):
        elapsed = int((time.monotonic() - started) * 1000)
        gathered = [(None, "timeout", elapsed)] * defaults.NUM_SEATS

    records: list[dict | None] = [None] * defaults.NUM_HEROES
    for seat, (reply, cause, decision_ms) in enumerate(gathered):
        picks, note, fallback_cause = resolve_reply(reply, cause)
        pid = defaults.pid_for_seat(seat)
        records[pid] = record(
            pid, picks=picks, note=note, source="seat",
            fallback_cause=fallback_cause, decision_ms=decision_ms,
            player_name=cfg.players[seat].name)
        if fallback_cause != "none":
            print(f"seat {seat} ({defaults.SEAT_ALIASES[seat]}): draft "
                  f"fallback to the neutral loadout "
                  f"(cause={fallback_cause})", file=sys.stderr)
    for pid in defaults.HOUSE_HERO_PIDS:
        records[pid] = house_record(pid)
    return [rec for rec in records if rec is not None]


def spectator_records(records: list[dict]) -> list[dict]:
    """The same records with real player names removed — what a playing
    seat and the /global feed are allowed to see."""
    return [{k: v for k, v in rec.items() if k != "player_name"}
            for rec in records]


def apply_to_sim(sim, records: list[dict]) -> None:
    """Write the ten applied blocks into the sim, ascending pid order.

    Called on the freshly initialised sim, after ``c_reset`` and before
    the first ``step()`` — the same placement the viewer uses in
    ``sim_fresh()`` (sim/loadout_common.h explains why re-spawning is
    forbidden here).
    """
    for rec in sorted(records, key=lambda r: r["pid"]):
        sim.apply_loadout(rec["pid"], rec["applied"])
