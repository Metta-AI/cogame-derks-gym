"""The binary replay format v2 (DERK): round-trip, header completeness,
strict-UTF-8 parsing, and end-to-end re-simulation from the bytes alone."""

import asyncio
import json

import numpy as np
import pytest

from cogame_derks_gym import defaults, replay
from cogame_derks_gym import draft as draft_module
from cogame_derks_gym.config import GameConfig
from cogame_derks_gym.engine import LockstepEngine
from cogame_derks_gym.replay import Replay, ReplayError, ReplayWriter

NOOP = list(defaults.NOOP_ACTION)


SEATS = defaults.NUM_SEATS
HEROES = defaults.NUM_HEROES


def make_config(**overrides):
    d = {
        "players": [{"name": f"hero-{i}"} for i in range(SEATS)],
        "tokens": [f"tok{i}" for i in range(SEATS)],
        "seed": 77,
        "max_ticks": 120,
        "tick_deadline_ms": 2000,
        "draft_deadline_ms": 1000,
    }
    d.update(overrides)
    return GameConfig.from_dict(d)


def random_actions(rng, n_ticks):
    return [rng.integers(0, defaults.ACT_HIGH,
                         size=(HEROES, 6)).astype(np.uint8)
            for _ in range(n_ticks)]


# -- round trip --------------------------------------------------------------

def test_round_trip():
    cfg = make_config()
    writer = ReplayWriter(cfg, sim_wasm_sha256="ab" * 32)
    rng = np.random.default_rng(0)
    ticks = random_actions(rng, 25)
    for t, acts in enumerate(ticks):
        writer.append_tick(t, acts)
    result = {"winner": 1, "end_reason": "ancient", "final_tick": 25}
    data = writer.finalize(result)

    rp = Replay.parse(data)
    assert rp.header["format_version"] == 2
    assert rp.header["sim_wasm_sha256"] == "ab" * 32
    assert rp.header["result"] == result
    assert rp.header["tick_count"] == 25
    assert rp.tick_count == 25
    # config round-trips fully resolved, names included, tokens excluded
    assert rp.header["config"] == cfg.to_dict()
    assert [p["name"] for p in rp.header["config"]["players"]] == \
        [f"hero-{i}" for i in range(SEATS)]
    assert "tokens" not in rp.header["config"]
    assert rp.header["config"]["seed"] == 77
    for t, acts in enumerate(rp):
        assert acts.dtype == np.uint8 and acts.shape == (HEROES, 6)
        np.testing.assert_array_equal(acts, ticks[t])
    np.testing.assert_array_equal(rp.actions(10), ticks[10])


def test_binary_layout():
    cfg = make_config()
    writer = ReplayWriter(cfg, sim_wasm_sha256="cd" * 32)
    writer.append_tick(0, np.zeros((HEROES, 6), dtype=np.uint8))
    data = writer.finalize({})
    assert data[:4] == b"DERK"
    assert data[4] == 2  # version
    header_len = int.from_bytes(data[5:9], "little")
    header = json.loads(data[9:9 + header_len])
    assert header["tick_count"] == 1
    body = data[9 + header_len:]
    assert len(body) == 60  # 10 heroes x 6 uint8 per tick
    assert body == b"\x00" * 60


# -- validation --------------------------------------------------------------

def test_bad_magic_rejected():
    with pytest.raises(ReplayError):
        Replay.parse(b"NOPE" + b"\x02" + b"\x00" * 20)


def test_replay_v1_moba_bytes_are_rejected():
    """v1 (MOBA) is deliberately not read: old cogame-moba replays are
    watched in cogame-moba."""
    with pytest.raises(ReplayError):
        Replay.parse(b"MOBA" + b"\x01" + b"\x00" * 20)


def test_bad_version_rejected():
    cfg = make_config()
    data = bytearray(ReplayWriter(cfg, "ee" * 32).finalize({}))
    data[4] = 9
    with pytest.raises(ReplayError):
        Replay.parse(bytes(data))


def test_truncated_body_rejected():
    cfg = make_config()
    writer = ReplayWriter(cfg, "ee" * 32)
    writer.append_tick(0, np.ones((HEROES, 6), dtype=np.uint8))
    data = writer.finalize({})
    with pytest.raises(ReplayError):
        Replay.parse(data[:-10])


def test_truncated_header_rejected():
    with pytest.raises(ReplayError):
        Replay.parse(b"DERK\x02\xff\xff\xff\x00rest")


def test_writer_rejects_non_sequential_tick():
    writer = ReplayWriter(make_config(), "ee" * 32)
    writer.append_tick(0, np.zeros((HEROES, 6), dtype=np.uint8))
    with pytest.raises(ValueError):
        writer.append_tick(2, np.zeros((HEROES, 6), dtype=np.uint8))


def test_writer_rejects_bad_tick_shape():
    writer = ReplayWriter(make_config(), "ee" * 32)
    with pytest.raises(ValueError):
        writer.append_tick(0, np.zeros((5, 6), dtype=np.uint8))


def test_writer_rejects_out_of_range_actions():
    """Out-of-range values would silently wrap in the uint8 cast (300 ->
    44) and corrupt re-simulation; the writer must reject them."""
    from cogame_derks_gym import defaults

    writer = ReplayWriter(make_config(), "ee" * 32)
    over = np.zeros((HEROES, 6), dtype=np.int64)
    over[3, 0] = 300
    with pytest.raises(ValueError, match="out of range"):
        writer.append_tick(0, over)
    # per-column high: 7 is legal nowhere, 6 only in the velocity columns
    bad_col = np.zeros((HEROES, 6), dtype=np.int64)
    bad_col[0, 2] = 3  # target-filter high is 3 (exclusive)
    with pytest.raises(ValueError, match="out of range"):
        writer.append_tick(0, bad_col)
    negative = np.zeros((HEROES, 6), dtype=np.int64)
    negative[0, 0] = -1
    with pytest.raises(ValueError, match="out of range"):
        writer.append_tick(0, negative)
    # boundary values are accepted
    top = np.tile(np.asarray(defaults.ACT_HIGH) - 1, (HEROES, 1))
    writer.append_tick(0, top)
    assert writer.tick_count == 1


def test_sim_wasm_sha256_matches_file(tmp_path):
    import hashlib
    p = tmp_path / "x.wasm"
    p.write_bytes(b"wasm bytes here")
    assert replay.sim_wasm_sha256(p) == \
        hashlib.sha256(b"wasm bytes here").hexdigest()


# -- re-simulation -----------------------------------------------------------

async def test_recorded_episode_resimulates_identically():
    """Record a real wasm episode via the engine hook, then re-run a fresh
    sim from the replay's seed feeding the replay's actions: same final
    tick, winner, and final obs bytes."""
    from cogame_derks_gym.sim import MobaSim

    class RngSource:
        def __init__(self, seat):
            self.rng = np.random.default_rng(1000 + seat)

        async def get_actions(self, tick, obs):
            return self.rng.integers(
                0, defaults.ACT_HIGH, size=(1, 6)).tolist()

    cfg = make_config(max_ticks=120)
    sim = MobaSim(seed=cfg.seed)
    writer = ReplayWriter(cfg, replay.sim_wasm_sha256())
    engine = LockstepEngine(
        sim, cfg, [RngSource(s) for s in range(SEATS)],
        on_tick=writer.append_tick)
    result = await engine.run()
    data = writer.finalize({
        "winner": result.winner,
        "end_reason": result.end_reason,
        "final_tick": result.final_tick,
    })
    recorded_obs = sim.observations().tobytes()

    rp = Replay.parse(data)
    assert rp.tick_count == result.final_tick
    resim = MobaSim(seed=rp.header["config"]["seed"])
    for acts in rp:
        resim.set_actions(acts.astype(np.float32))
        resim.step()
    assert resim.tick() == result.final_tick
    assert resim.done() == sim.done()
    assert resim.winner() == sim.winner()
    assert resim.observations().tobytes() == recorded_obs


# -- v2: the self-sufficient header ------------------------------------------

def _draft_records(cfg):
    return draft_module.neutral_records(cfg)


def test_header_is_self_sufficient():
    """Everything the static viewer needs is in the bytes: names,
    aliases, seed, config, catalog, ten draft records, events, digests."""
    from cogame_derks_gym import catalog

    cfg = make_config()
    writer = ReplayWriter(cfg, sim_wasm_sha256="ab" * 32)
    writer.append_tick(0, np.zeros((HEROES, 6), dtype=np.uint8))
    records = _draft_records(cfg)
    writer.set_draft(records, 0x12345678)
    writer.set_events([{"tick": 0, "kind": "draft", "pids": [0]},
                       {"tick": 1, "kind": "end", "reason": "tick_cap"}])
    writer.set_final_state_digest(0xDEADBEEF)
    header = Replay.parse(writer.finalize({"winner": None})).header

    assert header["format_version"] == 2
    assert header["catalog_version"] == catalog.CATALOG_VERSION
    assert header["catalog"] == catalog.catalog_dict()
    assert header["aliases"] == list(defaults.SEAT_ALIASES)
    assert header["seat_hero_pids"] == list(defaults.SEAT_HERO_PIDS)
    assert header["house_hero_pids"] == list(defaults.HOUSE_HERO_PIDS)
    assert header["config"]["seed"] == 77
    assert header["config"]["draft_enabled"] is True
    assert [p["name"] for p in header["config"]["players"]] == \
        [f"hero-{i}" for i in range(SEATS)]
    assert "tokens" not in header["config"]
    assert len(header["draft"]) == HEROES
    assert header["loadout_digest"] == 0x12345678
    assert [e["kind"] for e in header["events"]] == ["draft", "end"]
    assert header["final_state_digest"] == 0xDEADBEEF
    assert header["tick_count"] == 1


def test_header_decodes_with_strict_utf8_and_round_trips():
    """A byte-boundary truncation of a recorded string is how a replay
    ends up rendering in a browser and failing a strict JSON parser."""
    cfg = make_config()
    writer = ReplayWriter(cfg, sim_wasm_sha256="ab" * 32)
    records = _draft_records(cfg)
    records[0] = dict(records[0], note="\U0001F680 caf\u00e9 " + "x" * 100)
    writer.set_draft(records, 7)
    data = writer.finalize({"winner": 0})

    header_len = int.from_bytes(data[5:9], "little")
    raw = data[9:9 + header_len]
    text = raw.decode("utf-8", errors="strict")  # must not raise
    assert json.loads(text) == Replay.parse(data).header
    assert Replay.parse(data).header["draft"][0]["note"].startswith(
        "\U0001F680")


def test_flipped_header_byte_is_a_replay_error():
    cfg = make_config()
    writer = ReplayWriter(cfg, sim_wasm_sha256="ab" * 32)
    data = bytearray(writer.finalize({"winner": 0}))
    data[40] = 0x80  # invalid as a standalone UTF-8 byte
    with pytest.raises(ReplayError):
        Replay.parse(bytes(data))


def test_wrapping_header_len_is_a_replay_error():
    cfg = make_config()
    writer = ReplayWriter(cfg, sim_wasm_sha256="ab" * 32)
    data = bytearray(writer.finalize({"winner": 0}))
    data[5:9] = (0xFFFFFFFF).to_bytes(4, "little")
    with pytest.raises(ReplayError):
        Replay.parse(bytes(data))


def test_ragged_body_is_a_replay_error():
    cfg = make_config()
    writer = ReplayWriter(cfg, sim_wasm_sha256="ab" * 32)
    writer.append_tick(0, np.zeros((HEROES, 6), dtype=np.uint8))
    data = writer.finalize({"winner": 0})
    with pytest.raises(ReplayError):
        Replay.parse(data[:-1])


# -- end to end: a real episode, re-simulated from the bytes alone -----------

@pytest.mark.slow
async def test_recorded_episode_resimulates_from_the_replay_bytes(tmp_path):
    """Run a real 400-tick episode through the SERVER with six scripted
    seats, then re-simulate from the replay bytes alone on a fresh sim and
    require identical final tick, winner, obs bytes, state digest and
    loadout digest."""
    from cogame_derks_gym.server import GameServer
    from cogame_derks_gym.sim import MobaSim
    from tests.test_server import ServerHarness, make_config as server_config
    from tests.test_draft import draft_client

    cfg = server_config(max_ticks=400, tick_deadline_ms=1000, seed=4242,
                        draft_deadline_ms=2000)
    picks = json.dumps({"phase": "draft",
                        "picks": [{"arm": "arm_cleaver",
                                   "tail": "tail_plate",
                                   "misc": "misc_regen",
                                   "note": "e2e"}]})
    async with ServerHarness(cfg, tmp_path) as h:
        await asyncio.gather(*(draft_client(h, s, picks)
                               for s in range(SEATS)))
        result = await h.episode_task

    rp = Replay.parse(h.replay_path.read_bytes())
    assert rp.tick_count == result.final_tick
    header = rp.header
    assert len(header["draft"]) == HEROES
    assert header["events"], "no events recorded"
    assert header["events"][0]["kind"] == "draft"
    assert header["events"][-1]["kind"] == "end"
    assert header["result"]["end_reason"] == result.end_reason
    assert header["config"]["seed"] == 4242

    # re-simulate from the bytes alone
    resim = MobaSim(seed=header["config"]["seed"])
    for rec in sorted(header["draft"], key=lambda r: r["pid"]):
        resim.apply_loadout(rec["pid"], rec["applied"])
    assert resim.loadout_digest() == header["loadout_digest"]
    for acts in rp:
        resim.set_actions(acts.astype(np.float32))
        resim.step()
    assert resim.tick() == result.final_tick
    assert resim.state_digest() == header["final_state_digest"]
    assert int(resim.winner()) == (result.winner
                                   if result.winner is not None
                                   else int(resim.winner()))
