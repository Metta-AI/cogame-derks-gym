"""The loadout catalog: items, deltas, clamps, and stat application.

CLOSED CONTRACT (AGENTS.md rule 3): the ids and deltas here, the
``catalog_version`` enum + catalog sha256 in
``coworld_manifest_template.json``, ``docs/DRAFT.md`` and the
``item-<id>`` glyph symbols in ``viewer/derk_items.svg`` change together
or ``tests/test_catalog.py`` fails.

Every delta names a real field of upstream's ``struct Entity``
(``vendor/upstream/moba.h:162-204``) that ``spawn_player`` and the
level-up path re-read on every respawn and every level-up — which is why
writing the summed, clamped block once before tick 0 makes a loadout
permanent for the whole match (see ``sim/loadout_common.h``).

Nothing here touches skills, cooldown lengths, mana costs, vision range
or ``agent_speed``: those are physics, not stats (see the design note's
"Out of scope").
"""

from __future__ import annotations

import hashlib
import json
import struct

CATALOG_VERSION = "v1"

SLOTS = ("arm", "tail", "misc")

# Field order of a loadout block. Mirrors DERK_LOADOUT_FIELDS in
# sim/loadout_common.h (tested equal through the wasm read-back).
STAT_FIELDS = (
    "base_health", "base_mana", "base_damage", "basic_attack_cd",
    "move_speed", "hp_gain_per_level", "mana_gain_per_level",
    "damage_gain_per_level",
)
# Fields the sim stores as C ints (Entity.basic_attack_cd and the three
# per-level gains); the rest are float32.
INT_FIELDS = frozenset((
    "basic_attack_cd", "hp_gain_per_level", "mana_gain_per_level",
    "damage_gain_per_level",
))

# Deltas are ADDITIVE to the hero's own base stats and summed in the fixed
# order arm -> tail -> misc (addition commutes, but the order is pinned so
# `applied` is reproducible bit-for-bit), then clamped by CLAMPS.
ITEMS: dict[str, tuple[dict, ...]] = {
    "arm": (
        {"id": "arm_none", "name": "Bare Claws", "deltas": {}},
        {"id": "arm_blaster", "name": "Blaster",
         "deltas": {"base_damage": 15}},
        {"id": "arm_cleaver", "name": "Cleaver",
         "deltas": {"base_damage": 35, "basic_attack_cd": 3}},
        {"id": "arm_needler", "name": "Needler",
         "deltas": {"basic_attack_cd": -3, "base_damage": -10}},
    ),
    "tail": (
        {"id": "tail_none", "name": "Stub Tail", "deltas": {}},
        {"id": "tail_plate", "name": "Iron Plate",
         "deltas": {"base_health": 200, "basic_attack_cd": 2}},
        {"id": "tail_stinger", "name": "Stinger",
         "deltas": {"damage_gain_per_level": 8}},
        {"id": "tail_rotor", "name": "Rotor Tail",
         "deltas": {"move_speed": 0.15, "base_health": -100}},
    ),
    "misc": (
        {"id": "misc_none", "name": "Nothing", "deltas": {}},
        {"id": "misc_regen", "name": "Regen Cell",
         "deltas": {"hp_gain_per_level": 60}},
        {"id": "misc_battery", "name": "Mana Battery",
         "deltas": {"base_mana": 150, "mana_gain_per_level": 30}},
        {"id": "misc_focus", "name": "Focus Chip",
         "deltas": {"base_damage": 10, "hp_gain_per_level": -25}},
    ),
}

# Per-field clamps applied after summing.
#
# move_speed has a FLOOR of 1.0 and no item lowers it, for a fidelity
# reason: compute_observations writes obs_extra[6] = move_speed as an
# unsigned char (moba.h:486), so 0.9 would emit a 0 the pretrained
# policies never saw in training while 1.15 still casts to 1. For the
# same reason base_damage/50 (obs_extra[5]) and basic_attack_cd
# (obs_extra[14]) stay inside the small-integer band those bytes already
# occupy. Loadouts are physically real but only weakly encoded in the obs
# bytes — that is what makes this mod safe for the trained networks.
CLAMPS: dict[str, tuple[float, float]] = {
    "base_health": (150.0, 1200.0),
    "base_mana": (100.0, 600.0),
    "base_damage": (20.0, 120.0),
    "basic_attack_cd": (3, 16),
    "move_speed": (1.0, 1.5),
    "hp_gain_per_level": (0, 300),
    "mana_gain_per_level": (0, 200),
    "damage_gain_per_level": (0, 60),
}

# The zero-delta loadout: the house heroes' loadout, the server's
# fallback, and what makes the un-drafted variant bit-identical to
# upstream by construction rather than by a code path.
NEUTRAL_PICKS: dict[str, str] = {
    "arm": "arm_none", "tail": "tail_none", "misc": "misc_none"}

# Protocol caps (docs/PROTOCOL.md); the server enforces all of them.
MAX_ITEM_ID_CHARS = 24
MAX_NOTE_RUNES = 120
MAX_DRAFT_FRAME_BYTES = 4096

_ITEMS_BY_SLOT: dict[str, dict[str, dict]] = {
    slot: {item["id"]: item for item in items}
    for slot, items in ITEMS.items()
}


def f32(value: float) -> float:
    """The exact float32 value ``value`` becomes inside the sim.

    Applied stat blocks cross the JSON boundary (replay header ->
    viewer -> viewer_set_loadout), so recording the float32 value makes
    the float32 -> double -> float32 round trip lossless.
    """
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def catalog_dict() -> dict:
    """The catalog object as it appears in the draft observation and in
    the replay header (same shape in both, so a viewer needs no repo)."""
    return {
        "version": CATALOG_VERSION,
        **{slot: [dict(item, deltas=dict(item["deltas"]))
                  for item in ITEMS[slot]]
           for slot in SLOTS},
    }


def clamps_dict() -> dict:
    return {field: list(bounds) for field, bounds in CLAMPS.items()}


def catalog_sha256() -> str:
    """Hex sha256 of the canonical catalog + clamp table.

    Baked into the manifest ``config_schema`` and ``docs/DRAFT.md``; a
    silent catalog edit fails tests/test_catalog.py.
    """
    payload = json.dumps(
        {"catalog": catalog_dict(), "clamps": clamps_dict()},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def item(slot: str, item_id: str) -> dict | None:
    """The catalog entry for ``item_id`` in ``slot``, or None.

    Case-sensitive exact match after stripping leading/trailing ASCII
    spaces (the one tolerance, so it is testable).
    """
    if slot not in _ITEMS_BY_SLOT or not isinstance(item_id, str):
        return None
    return _ITEMS_BY_SLOT[slot].get(item_id.strip(" "))


def normalized_picks(raw: dict) -> dict[str, str] | None:
    """Validate one seat's three picks; None when any slot is illegal.

    Partial acceptance is deliberately NOT allowed: it would let a seat
    launder a typo into a free reroll of one slot.
    """
    picks: dict[str, str] = {}
    for slot in SLOTS:
        value = raw.get(slot)
        if not isinstance(value, str) or len(value) > MAX_ITEM_ID_CHARS:
            return None
        found = item(slot, value)
        if found is None:
            return None
        picks[slot] = found["id"]
    return picks


def apply_picks(base: dict, picks: dict[str, str]) -> dict:
    """Sum the picks' deltas onto ``base`` (arm -> tail -> misc), clamp.

    ``base`` is the hero's own base stat block (defaults.HERO_BASE).
    Returns a full block keyed by STAT_FIELDS: ints for INT_FIELDS,
    exact float32 values for the rest.
    """
    values = {field: float(base[field]) for field in STAT_FIELDS}
    for slot in SLOTS:  # pinned order
        entry = _ITEMS_BY_SLOT[slot][picks[slot]]
        for field, delta in entry["deltas"].items():
            values[field] += float(delta)
    applied: dict = {}
    for field in STAT_FIELDS:
        low, high = CLAMPS[field]
        value = min(max(values[field], float(low)), float(high))
        applied[field] = int(round(value)) if field in INT_FIELDS \
            else f32(value)
    return applied


def _fnv1a_f32(digest: int, value: float) -> int:
    for byte in struct.pack("<f", float(value)):
        digest ^= byte
        digest = (digest * 16777619) & 0xFFFFFFFF
    return digest


def loadout_digest(applied_by_pid: dict[int, dict] | None = None,
                   num_heroes: int = 10) -> int:
    """FNV-1a over the num_heroes x 8 applied float32 table, in pid then
    field order — the Python mirror of ``derk_loadout_digest`` in
    sim/loadout_common.h.

    Heroes with no applied block contribute a row of zeros, so an
    un-drafted sim (and a viewer that pushed nothing) both digest the
    all-zero table. Tests assert the C and Python values agree.
    """
    applied_by_pid = applied_by_pid or {}
    digest = 2166136261  # FNV-1a offset basis
    for pid in range(num_heroes):
        block = applied_by_pid.get(pid)
        for field in STAT_FIELDS:
            digest = _fnv1a_f32(digest, 0.0 if block is None
                                else block[field])
    return digest


def neutral_applied(base: dict) -> dict:
    """The base block itself: every ``*_none`` delta is zero."""
    return apply_picks(base, NEUTRAL_PICKS)
