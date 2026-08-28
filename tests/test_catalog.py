"""Tripwires for the CLOSED catalog contract (AGENTS.md rule 3).

`catalog.py`, the manifest `config_schema`, `docs/DRAFT.md` and
`viewer/derk_items.svg` change together or this file fails. A catalog edit
that lands in one place only silently re-interprets every existing replay.
"""

import json
import re
from pathlib import Path

import pytest

from cogame_derks_gym import catalog, defaults

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (REPO_ROOT / "coworld_manifest_template.json").read_text())
DRAFT_MD = (REPO_ROOT / "docs" / "DRAFT.md").read_text()
ITEMS_SVG = (REPO_ROOT / "viewer" / "derk_items.svg").read_text()
MOBA_H = (REPO_ROOT / "vendor" / "upstream" / "moba.h").read_text()

DOCUMENTED_IDS = {
    "arm": ["arm_none", "arm_blaster", "arm_cleaver", "arm_needler"],
    "tail": ["tail_none", "tail_plate", "tail_stinger", "tail_rotor"],
    "misc": ["misc_none", "misc_regen", "misc_battery", "misc_focus"],
}


def test_id_set_is_exactly_the_twelve_documented_ids():
    assert set(catalog.SLOTS) == set(DOCUMENTED_IDS)
    for slot, ids in DOCUMENTED_IDS.items():
        assert [item["id"] for item in catalog.ITEMS[slot]] == ids
    all_ids = [item["id"] for slot in catalog.SLOTS
               for item in catalog.ITEMS[slot]]
    assert len(all_ids) == 12
    assert len(set(all_ids)) == 12


def test_every_id_is_prefixed_by_its_slot():
    """The slot is derivable from the id, which is what makes a
    wrong-slot pick (arm_none in the tail slot) an unknown_item rather
    than a silently accepted one."""
    for slot in catalog.SLOTS:
        for item in catalog.ITEMS[slot]:
            assert item["id"].startswith(f"{slot}_"), item
            assert len(item["id"]) <= catalog.MAX_ITEM_ID_CHARS
            assert item["name"]


def test_every_delta_names_a_real_struct_entity_field():
    """Regex tripwire over the vendored source: a delta on a field that
    does not exist in upstream's struct Entity would write nothing (or,
    worse, the wrong offset if someone 'fixed' it later)."""
    fields = set(re.findall(r"^\s+(?:float|int)\s+(\w+);", MOBA_H,
                            re.MULTILINE))
    assert {"base_health", "base_damage", "basic_attack_cd",
            "move_speed"} <= fields, "moba.h struct parse broke"
    for slot in catalog.SLOTS:
        for item in catalog.ITEMS[slot]:
            for field in item["deltas"]:
                assert field in fields, (item["id"], field)
                assert field in catalog.STAT_FIELDS, (item["id"], field)


def test_none_items_are_zero_delta():
    """The neutral loadout must be a no-op by construction, not by a code
    path: that is what makes the un-drafted variant bit-identical."""
    for slot in catalog.SLOTS:
        none_item = catalog.ITEMS[slot][0]
        assert none_item["id"] == catalog.NEUTRAL_PICKS[slot]
        assert none_item["deltas"] == {}
    for pid in range(defaults.NUM_HEROES):
        base = defaults.HERO_BASE[pid]
        applied = catalog.neutral_applied(base)
        for field in catalog.STAT_FIELDS:
            assert applied[field] == pytest.approx(base[field]), (pid, field)


def test_no_item_lowers_move_speed_below_one():
    """obs_extra[6] is an unsigned char (moba.h:486): 0.9 would emit a 0
    the pretrained policies never saw in training."""
    assert catalog.CLAMPS["move_speed"][0] == 1.0
    for slot in catalog.SLOTS:
        for item in catalog.ITEMS[slot]:
            assert item["deltas"].get("move_speed", 0) >= 0, item["id"]


def test_every_id_has_a_glyph_symbol_in_the_sprite_sheet():
    for slot in catalog.SLOTS:
        for item in catalog.ITEMS[slot]:
            assert f'id="item-{item["id"]}"' in ITEMS_SVG, item["id"]
    # and no orphan symbols (a removed item must lose its glyph too)
    symbols = set(re.findall(r'<symbol id="item-([\w]+)"', ITEMS_SVG))
    assert symbols == {item["id"] for slot in catalog.SLOTS
                       for item in catalog.ITEMS[slot]}


def test_catalog_sha256_matches_the_manifest_and_the_docs():
    sha = catalog.catalog_sha256()
    description = (MANIFEST["game"]["config_schema"]["properties"]
                   ["catalog_version"]["description"])
    assert sha in description, \
        "the manifest config_schema does not carry this catalog's sha256"
    assert sha in DRAFT_MD, "docs/DRAFT.md does not carry this catalog's sha256"
    # the enum is pinned to exactly the shipped version
    assert (MANIFEST["game"]["config_schema"]["properties"]
            ["catalog_version"]["enum"]) == [catalog.CATALOG_VERSION]


def test_docs_draft_md_lists_every_id_and_delta():
    for slot in catalog.SLOTS:
        for item in catalog.ITEMS[slot]:
            assert f"`{item['id']}`" in DRAFT_MD, item["id"]
            assert item["name"] in DRAFT_MD, item["name"]
            for field, value in item["deltas"].items():
                # e.g. "base_damage +15" / "basic_attack_cd −3"
                sign = "+" if value > 0 else "\u2212"
                assert f"{field} {sign}{abs(value)}" in DRAFT_MD, \
                    (item["id"], field, value)


def test_clamp_table_is_documented_and_covers_every_stat_field():
    assert set(catalog.CLAMPS) == set(catalog.STAT_FIELDS)
    for field, (low, high) in catalog.CLAMPS.items():
        assert low <= high
        assert f"`{field}`" in DRAFT_MD
    for slot in catalog.SLOTS:
        for item in catalog.ITEMS[slot]:
            assert set(item["deltas"]) <= set(catalog.CLAMPS)


def test_catalog_dict_is_json_serialisable_and_self_describing():
    """The catalog travels in the replay header so the viewer can render
    item names and deltas without contacting the repo."""
    obj = catalog.catalog_dict()
    assert obj["version"] == catalog.CATALOG_VERSION
    round_trip = json.loads(json.dumps(obj))
    assert round_trip == obj
    for slot in catalog.SLOTS:
        assert len(round_trip[slot]) == 4
        for entry in round_trip[slot]:
            assert set(entry) == {"id", "name", "deltas"}


def test_catalog_dict_is_a_copy():
    obj = catalog.catalog_dict()
    obj["arm"][0]["deltas"]["base_damage"] = 9999
    assert catalog.ITEMS["arm"][0]["deltas"] == {}


def test_normalized_picks_accepts_and_rejects():
    good = {"arm": "arm_blaster", "tail": " tail_plate ",
            "misc": "misc_focus"}
    assert catalog.normalized_picks(good) == {
        "arm": "arm_blaster", "tail": "tail_plate", "misc": "misc_focus"}
    # wrong slot, unknown id, wrong case, non-string, too long, missing
    for bad in ({"arm": "tail_plate", "tail": "tail_plate",
                 "misc": "misc_focus"},
                {"arm": "arm_nope", "tail": "tail_plate",
                 "misc": "misc_focus"},
                {"arm": "ARM_BLASTER", "tail": "tail_plate",
                 "misc": "misc_focus"},
                {"arm": 1, "tail": "tail_plate", "misc": "misc_focus"},
                {"arm": "a" * 25, "tail": "tail_plate", "misc": "misc_focus"},
                {"arm": "arm_blaster", "misc": "misc_focus"}):
        assert catalog.normalized_picks(bad) is None, bad
