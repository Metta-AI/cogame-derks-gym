"""The draft turn: resolution order, caps, simultaneity, name spaces.

Every row of the table in docs/DRAFT.md gets a case here. The transport is
faked at the DraftSource boundary where it can be, and driven over a real
websocket where the case is about the wire (oversize frames, closed
sockets, wrong-phase messages).
"""

import asyncio
import json

import aiohttp
import pytest

from cogame_derks_gym import catalog, defaults, draft
from cogame_derks_gym.config import GameConfig

from tests.test_server import SEATS, ServerHarness, make_config

NEUTRAL = dict(catalog.NEUTRAL_PICKS)
LEGAL = {"arm": "arm_cleaver", "tail": "tail_plate", "misc": "misc_regen"}


def cfg(**overrides):
    d = {
        "players": [{"name": f"real-name-{i}"} for i in range(SEATS)],
        "tokens": [f"t{i}" for i in range(SEATS)],
        "seed": 3,
        "draft_deadline_ms": 1000,
    }
    d.update(overrides)
    return GameConfig.from_dict(d)


class Source:
    """A draft source that replies with a fixed frame, or stalls."""

    def __init__(self, reply=None, cause=None, delay=0.0, raises=False):
        self.reply = reply
        self.cause = cause
        self.delay = delay
        self.raises = raises
        self.observations = []

    async def get_draft(self, observation):
        self.observations.append(observation)
        if self.raises:
            raise RuntimeError("source exploded")
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.reply, self.cause


def frame(picks=None, **extra):
    """A well-formed draft reply; `extra` goes INSIDE the pick object
    (that is where `note` lives)."""
    body = dict(LEGAL if picks is None else picks, **extra)
    return {"phase": "draft", "picks": [body]}


async def run(sources, **config_overrides):
    return await draft.run_draft(cfg(**config_overrides), sources)


def by_pid(records):
    return {rec["pid"]: rec for rec in records}


# -- the happy path ----------------------------------------------------------

async def test_legal_picks_applied_to_every_seat():
    sources = [Source(frame()) for _ in range(SEATS)]
    records = await run(sources)
    assert len(records) == defaults.NUM_HEROES
    recs = by_pid(records)
    for seat, pid in enumerate(defaults.SEAT_HERO_PIDS):
        rec = recs[pid]
        assert rec["picks"] == LEGAL
        assert rec["fallback"] is False
        assert rec["fallback_cause"] == "none"
        assert rec["source"] == "seat"
        assert rec["seat"] == seat
        assert rec["alias"] == defaults.SEAT_ALIASES[seat]
        assert rec["player_name"] == f"real-name-{seat}"
        # applied = base + summed deltas, clamped
        base = defaults.HERO_BASE[pid]
        assert rec["applied"] == catalog.apply_picks(base, LEGAL)
    for pid in defaults.HOUSE_HERO_PIDS:
        rec = recs[pid]
        assert rec["source"] == "house"
        assert rec["seat"] is None
        assert rec["player_name"] is None
        assert rec["picks"] == NEUTRAL
        assert rec["decision_ms"] == 0
        assert rec["note"] == ""


async def test_note_is_kept_and_records_decision_time():
    sources = [Source(frame(note="tanky mid")) for _ in range(SEATS)]
    records = by_pid(await run(sources))
    assert records[0]["note"] == "tanky mid"
    assert records[0]["decision_ms"] >= 0


# -- the resolution order ----------------------------------------------------

@pytest.mark.parametrize("bad,cause", [
    (frame({"arm": "arm_nope", "tail": "tail_plate", "misc": "misc_regen"}),
     "unknown_item"),
    # a real id, but of the wrong slot
    (frame({"arm": "tail_plate", "tail": "tail_plate", "misc": "misc_regen"}),
     "unknown_item"),
    (frame({"arm": "ARM_CLEAVER", "tail": "tail_plate",
            "misc": "misc_regen"}), "unknown_item"),
    (frame({"arm": "a" * 25, "tail": "tail_plate", "misc": "misc_regen"}),
     "unknown_item"),
    (frame({"tail": "tail_plate", "misc": "misc_regen"}), "unknown_item"),
    ({"phase": "draft", "picks": []}, "wrong_shape"),
    ({"phase": "draft", "picks": [LEGAL, LEGAL]}, "wrong_shape"),
    ({"phase": "draft", "picks": LEGAL}, "wrong_shape"),
    ({"phase": "draft", "picks": ["arm_cleaver"]}, "wrong_shape"),
    ({"phase": "draft"}, "wrong_shape"),
    ({"picks": [LEGAL]}, "wrong_shape"),
    ("not an object", "wrong_shape"),
])
async def test_illegal_replies_fall_back_to_the_neutral_loadout(bad, cause):
    sources = [Source(bad)] + [Source(frame()) for _ in range(SEATS - 1)]
    records = by_pid(await run(sources))
    rec = records[defaults.SEAT_HERO_PIDS[0]]
    assert rec["picks"] == NEUTRAL
    assert rec["fallback"] is True
    assert rec["fallback_cause"] == cause
    # partial acceptance is NOT allowed: the whole seat goes neutral
    assert rec["applied"] == catalog.neutral_applied(defaults.HERO_BASE[0])
    # the other seats are unaffected
    assert records[defaults.SEAT_HERO_PIDS[1]]["picks"] == LEGAL


async def test_no_reply_by_the_deadline_is_a_timeout():
    """The deadline is shared, the resolution is PER SEAT: the seat that
    never answered times out, and every seat that answered legally inside
    the deadline keeps its picks."""
    sources = [Source(frame(), delay=5.0)] + \
        [Source(frame()) for _ in range(SEATS - 1)]
    records = by_pid(await run(sources, draft_deadline_ms=1000))
    slow = records[defaults.SEAT_HERO_PIDS[0]]
    assert slow["fallback_cause"] == "timeout"
    assert slow["fallback"] is True
    assert slow["picks"] == NEUTRAL
    for seat in range(1, SEATS):
        rec = records[defaults.SEAT_HERO_PIDS[seat]]
        assert rec["fallback_cause"] == "none", f"seat {seat}"
        assert rec["fallback"] is False, f"seat {seat}"
        assert rec["picks"] == LEGAL, f"seat {seat}"


async def test_disconnected_seat_is_reported_as_disconnected():
    sources = [Source(None, "disconnected")] + \
        [Source(frame()) for _ in range(SEATS - 1)]
    records = by_pid(await run(sources))
    assert records[0]["fallback_cause"] == "disconnected"


async def test_raising_source_never_breaks_the_draft():
    sources = [Source(raises=True)] + \
        [Source(frame()) for _ in range(SEATS - 1)]
    records = by_pid(await run(sources))
    assert records[0]["fallback_cause"] == "wrong_shape"
    assert records[defaults.SEAT_HERO_PIDS[1]]["picks"] == LEGAL


async def test_the_batch_is_one_shared_deadline_not_six():
    """All six seats are asked as ONE parallel batch: six slow seats cost
    one deadline, not six. That is what keeps the episode inside 60% of
    the platform budget."""
    import time

    sources = [Source(frame(), delay=5.0) for _ in range(SEATS)]
    started = time.monotonic()
    records = by_pid(await run(sources, draft_deadline_ms=1000))
    elapsed = time.monotonic() - started
    assert elapsed < 3.0, f"the draft turn took {elapsed:.1f}s"
    for pid in defaults.SEAT_HERO_PIDS:
        assert records[pid]["fallback_cause"] == "timeout"


# -- the note: rune-boundary truncation --------------------------------------

def test_note_truncated_on_rune_boundaries_with_a_straddling_emoji():
    """A 4-byte emoji straddling index 120 must never be split: a
    byte-boundary truncation produces a replay that renders in a browser
    and fails a strict JSON parser."""
    emoji = "\U0001F680"  # 4 UTF-8 bytes, one Unicode scalar
    note = "x" * 119 + emoji + "tail"
    out = draft.truncate_note(note)
    assert len(out) == 120
    assert out.endswith(emoji)
    out.encode("utf-8", "strict")  # no surrogate, no partial codepoint
    json.dumps(out)

    # one character later the emoji is dropped whole, not halved
    out = draft.truncate_note("x" * 120 + emoji)
    assert out == "x" * 120
    assert len(out.encode("utf-8")) == 120


def test_note_truncation_keeps_combining_sequences_valid_utf8():
    combining = "e\u0301"  # e + combining acute
    note = combining * 100  # 200 scalars
    out = draft.truncate_note(note)
    assert len(out) == 120
    out.encode("utf-8", "strict")


def test_note_strips_control_characters_and_surrogates():
    assert draft.truncate_note("a\x00b\x1fc\x7fd\x85e") == "abcde"
    assert draft.truncate_note("a\ud800b") == "ab"
    assert draft.truncate_note(None) == ""
    assert draft.truncate_note(42) == ""
    assert draft.truncate_note({"a": 1}) == ""


async def test_a_bad_note_never_invalidates_the_picks():
    sources = [Source(frame(note=12345))] + \
        [Source(frame()) for _ in range(SEATS - 1)]
    records = by_pid(await run(sources))
    assert records[0]["picks"] == LEGAL
    assert records[0]["note"] == ""
    assert records[0]["fallback"] is False


async def test_recorded_note_respects_the_cap():
    sources = [Source(frame(note="\U0001F680" * 300))] + \
        [Source(frame()) for _ in range(SEATS - 1)]
    records = by_pid(await run(sources))
    assert len(records[0]["note"]) == catalog.MAX_NOTE_RUNES


# -- the draft observation: simultaneity and the two name spaces -------------

async def test_no_draft_observation_contains_another_seats_pick():
    """Simultaneity: the observations are built BEFORE any reply is read,
    so a counter-draft must be a prediction, not a reaction."""
    sources = [Source(frame()) for _ in range(SEATS)]
    await run(sources)
    for source in sources:
        (observation,) = source.observations
        blob = json.dumps(observation)
        # nothing about anyone's picks, and no 'picks' key at all
        assert "picks" not in observation
        for other in sources:
            for other_observation in other.observations:
                assert "picks" not in other_observation
        assert "arm_cleaver" in blob  # the catalog IS visible
        assert "tail_plate" in blob


async def test_no_draft_observation_contains_a_real_player_name():
    """Two name spaces: a policy sees aliases only."""
    sources = [Source(frame()) for _ in range(SEATS)]
    await run(sources)
    for source in sources:
        blob = json.dumps(source.observations[0])
        for seat in range(SEATS):
            assert f"real-name-{seat}" not in blob


async def test_draft_observation_shape():
    sources = [Source(frame()) for _ in range(SEATS)]
    await run(sources)
    observation = sources[2].observations[0]
    assert observation["phase"] == "draft"
    assert observation["seat"] == 2
    assert observation["alias"] == "Cog-Charlie"
    assert observation["team"] == "radiant"
    assert observation["hero"] == {
        "pid": 2, "role": "burst", "lane": 1,
        "skills": ["burst_nuke", "burst_aoe", "burst_aoe_stun"],
        "base_health": 400.0, "base_mana": 300.0, "base_damage": 50.0,
        "basic_attack_cd": 8, "move_speed": 1.0,
        "hp_gain_per_level": 75, "mana_gain_per_level": 90,
        "damage_gain_per_level": 10}
    assert [t["alias"] for t in observation["teammates"]] == \
        ["Cog-Alpha", "Cog-Bravo"]
    assert [o["alias"] for o in observation["opponents"]] == \
        ["Cog-Delta", "Cog-Echo", "Cog-Foxtrot"]
    assert [h["role"] for h in observation["house_heroes"]] == \
        ["tank", "carry", "tank", "carry"]
    assert observation["catalog"] == catalog.catalog_dict()
    assert observation["clamps"] == catalog.clamps_dict()
    assert observation["match"]["ancient_health"] == 4500
    assert observation["deadline_ms"] == 1000
    assert "seed" not in json.dumps(observation)


# -- over the wire -----------------------------------------------------------

async def draft_client(harness, slot, send, *, close_after_draft=False):
    """Connects, sends `send` (a raw string) when the draft turn arrives,
    then plays NOOP until done."""
    seen = []
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(
                harness.ws_url(slot, f"token-{slot}")) as ws:
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    break
                data = json.loads(msg.data)
                if data.get("done"):
                    return seen
                phase = data.get("phase")
                if phase == "draft":
                    seen.append("draft")
                    if send is not None:
                        await ws.send_str(send)
                    if close_after_draft:
                        await ws.close()
                        return seen
                    continue
                if phase == "draft_result":
                    seen.append(("draft_result", data["loadouts"]))
                    continue
                if phase is not None:
                    continue
                await ws.send_str(json.dumps(
                    {"tick": data["tick"],
                     "actions": [list(defaults.NOOP_ACTION)]}))
    return seen


async def test_oversize_frame_is_dropped_before_the_json_parse(tmp_path):
    """A >4096-byte frame never reaches json.loads: oversize."""
    config = make_config(max_ticks=3, tick_deadline_ms=200,
                         draft_deadline_ms=1000)
    huge = json.dumps({"phase": "draft",
                       "picks": [dict(LEGAL, note="p" * 6000)]})
    assert len(huge.encode()) > catalog.MAX_DRAFT_FRAME_BYTES
    async with ServerHarness(config, tmp_path) as h:
        clients = [draft_client(h, 0, huge)] + [
            draft_client(h, s, json.dumps(frame()))
            for s in range(1, SEATS)]
        await asyncio.gather(*clients)
        await h.episode_task
    results = json.loads(h.results_path.read_text())
    records = by_pid(results["draft"])
    assert records[0]["fallback_cause"] == "oversize"
    assert records[0]["picks"] == NEUTRAL
    assert results["draft_fallbacks"] == [True] + [False] * (SEATS - 1)


async def test_non_json_frame_is_wrong_shape(tmp_path):
    config = make_config(max_ticks=3, tick_deadline_ms=200,
                         draft_deadline_ms=1000)
    async with ServerHarness(config, tmp_path) as h:
        clients = [draft_client(h, 0, "{not json")] + [
            draft_client(h, s, json.dumps(frame()))
            for s in range(1, SEATS)]
        await asyncio.gather(*clients)
        await h.episode_task
    records = by_pid(json.loads(h.results_path.read_text())["draft"])
    assert records[0]["fallback_cause"] == "wrong_shape"


async def test_closed_socket_during_the_draft_is_disconnected(tmp_path):
    config = make_config(max_ticks=3, tick_deadline_ms=200,
                         draft_deadline_ms=1000)
    async with ServerHarness(config, tmp_path) as h:
        clients = [draft_client(h, 0, None, close_after_draft=True)] + [
            draft_client(h, s, json.dumps(frame()))
            for s in range(1, SEATS)]
        await asyncio.gather(*clients)
        await h.episode_task
    records = by_pid(json.loads(h.results_path.read_text())["draft"])
    assert records[0]["fallback_cause"] in ("disconnected", "timeout")
    assert records[0]["picks"] == NEUTRAL


async def test_a_tick_reply_during_the_draft_does_not_consume_the_turn(
        tmp_path):
    """Compatibility: an unmodified cogame-moba policy answers ticks, not
    drafts. Its early {"tick": ...} message must be ignored WITHOUT
    consuming the draft turn, so a real draft reply still lands."""
    config = make_config(max_ticks=3, tick_deadline_ms=200,
                         draft_deadline_ms=2000)

    async def confused_then_correct(harness, slot):
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                    harness.ws_url(slot, f"token-{slot}")) as ws:
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        break
                    data = json.loads(msg.data)
                    if data.get("done"):
                        return
                    if data.get("phase") == "draft":
                        # a stale per-tick reply first, then the real one
                        await ws.send_str(json.dumps(
                            {"tick": 0, "actions": [[3, 3, 0, 0, 0, 0]]}))
                        await asyncio.sleep(0.05)
                        await ws.send_str(json.dumps(frame()))
                        continue
                    if data.get("phase") is not None:
                        continue
                    await ws.send_str(json.dumps(
                        {"tick": data["tick"],
                         "actions": [list(defaults.NOOP_ACTION)]}))

    async with ServerHarness(config, tmp_path) as h:
        await asyncio.gather(*(confused_then_correct(h, s)
                               for s in range(SEATS)))
        await h.episode_task
    results = json.loads(h.results_path.read_text())
    assert results["draft_fallbacks"] == [False] * SEATS
    records = by_pid(results["draft"])
    assert records[0]["picks"] == LEGAL


async def test_second_draft_message_is_ignored(tmp_path):
    """At most one reply is consumed per seat."""
    config = make_config(max_ticks=3, tick_deadline_ms=200,
                         draft_deadline_ms=1000)

    async def double_drafter(harness, slot):
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                    harness.ws_url(slot, f"token-{slot}")) as ws:
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        break
                    data = json.loads(msg.data)
                    if data.get("done"):
                        return
                    if data.get("phase") == "draft":
                        await ws.send_str(json.dumps(frame()))
                        await ws.send_str(json.dumps(frame(
                            {"arm": "arm_none", "tail": "tail_none",
                             "misc": "misc_none"})))
                        continue
                    if data.get("phase") is not None:
                        continue
                    await ws.send_str(json.dumps(
                        {"tick": data["tick"],
                         "actions": [list(defaults.NOOP_ACTION)]}))

    async with ServerHarness(config, tmp_path) as h:
        await asyncio.gather(*(double_drafter(h, s) for s in range(SEATS)))
        await h.episode_task
    records = by_pid(json.loads(h.results_path.read_text())["draft"])
    assert records[0]["picks"] == LEGAL, "the second message overrode the first"


async def test_draft_result_is_pushed_alias_only(tmp_path):
    config = make_config(max_ticks=3, tick_deadline_ms=200,
                         draft_deadline_ms=1000)
    async with ServerHarness(config, tmp_path) as h:
        seen = await asyncio.gather(*(
            draft_client(h, s, json.dumps(frame())) for s in range(SEATS)))
        await h.episode_task
    for events in seen:
        results = [e for e in events if isinstance(e, tuple)]
        assert results, "no draft_result message"
        _, loadouts = results[0]
        assert len(loadouts) == defaults.NUM_HEROES
        blob = json.dumps(loadouts)
        assert "player_name" not in blob
        for slot in range(SEATS):
            assert f"bot-{slot}" not in blob
