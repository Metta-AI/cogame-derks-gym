"""Tests for the player client library and random player (Task 3.1).

Real episodes against the in-process GameServer (fixtures reused from
tests.test_server) plus targeted reconnect/fatal-error tests against a
minimal protocol-speaking fake server.
"""

import asyncio
import base64
import json

import numpy as np
import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer

from cogame_derks_gym import defaults
from cogame_derks_gym.replay import Replay
from players import client, random_player
from players.client import PlayerError, play_episode

from tests.test_server import SEATS, ServerHarness, make_config


# -- full episodes driven by the client library ------------------------------

async def test_random_players_complete_six_seat_episode(tmp_path):
    cfg = make_config(max_ticks=12)
    async with ServerHarness(cfg, tmp_path) as h:
        results = await asyncio.gather(*(
            play_episode(random_player.RandomPolicy(seed=s),
                         h.ws_url(s, f"token-{s}"))
            for s in range(SEATS)))
        engine_result = await h.episode_task

    # every client got the same result doc from the done message
    assert all(isinstance(r, dict) for r in results)
    assert all(r["final_tick"] == engine_result.final_tick for r in results)
    assert sum(results[0]["scores"]) == SEATS / 2
    assert len(results[0]["scores"]) == SEATS

    # artifacts written: results + parseable replay
    written = json.loads(h.results_path.read_text())
    assert written["scores"] == results[0]["scores"]
    replay = Replay.parse(h.replay_path.read_bytes())
    assert replay.tick_count == engine_result.final_tick
    # nobody degraded to NOOP: the clients kept up every tick
    assert written["noop_ticks"] == [0] * SEATS
    # RandomPolicy has no on_draft: every seat times out of the draft turn
    # and plays the neutral loadout. That path must be legal and complete
    # (it is also how an unmodified cogame-moba policy plays this game).
    assert written["draft_fallbacks"] == [True] * SEATS
    for rec in written["draft"]:
        assert rec["picks"] == {"arm": "arm_none", "tail": "tail_none",
                                "misc": "misc_none"}, rec


async def test_one_obs_row_per_seat(tmp_path):
    """Each seat controls exactly one hero: one obs blob in, one action
    row out."""
    cfg = make_config(max_ticks=10)
    seen_obs_counts = set()

    class CountingRandomPolicy(random_player.RandomPolicy):
        def __call__(self, tick, obs_rows):
            seen_obs_counts.add(len(obs_rows))
            assert all(len(o) == 510 for o in obs_rows)
            return super().__call__(tick, obs_rows)

    async with ServerHarness(cfg, tmp_path) as h:
        results = await asyncio.gather(*(
            play_episode(CountingRandomPolicy(seed=s),
                         h.ws_url(s, f"token-{s}"))
            for s in range(SEATS)))
        engine_result = await h.episode_task

    assert seen_obs_counts == {1}
    assert sum(results[0]["scores"]) == SEATS / 2
    written = json.loads(h.results_path.read_text())
    assert written["team"] == ["radiant"] * 3 + ["dire"] * 3
    assert written["noop_ticks"] == [0] * SEATS
    assert engine_result.final_tick == 10


async def test_draft_answering_policy_is_recorded(tmp_path):
    """A policy carrying on_draft answers the single draft turn; the picks
    land in results.draft with fallback False."""
    cfg = make_config(max_ticks=6)

    class DraftingPolicy(random_player.RandomPolicy):
        def __init__(self, seed, picks):
            super().__init__(seed)
            self.picks = picks
            self.observations = []

        def on_draft(self, observation):
            self.observations.append(observation)
            return self.picks

    picks = {"arm": "arm_cleaver", "tail": "tail_rotor",
             "misc": "misc_regen", "note": "brawl"}
    policies = [DraftingPolicy(s, picks) for s in range(SEATS)]
    async with ServerHarness(cfg, tmp_path) as h:
        await asyncio.gather(*(
            play_episode(policies[s], h.ws_url(s, f"token-{s}"))
            for s in range(SEATS)))
        await h.episode_task

    written = json.loads(h.results_path.read_text())
    assert written["draft_fallbacks"] == [False] * SEATS
    for policy in policies:
        assert len(policy.observations) == 1  # exactly one draft turn
    for rec in written["draft"]:
        if rec["source"] != "seat":
            continue
        assert rec["picks"] == {"arm": "arm_cleaver", "tail": "tail_rotor",
                                "misc": "misc_regen"}
        assert rec["note"] == "brawl"
        assert rec["fallback"] is False


# -- fatal handshake errors --------------------------------------------------

async def test_bad_token_is_fatal_403(tmp_path):
    cfg = make_config(player_connect_timeout_seconds=5)
    async with ServerHarness(cfg, tmp_path) as h:
        with pytest.raises(PlayerError, match="403"):
            await play_episode(random_player.RandomPolicy(seed=0),
                               h.ws_url(0, "wrong-token"))


async def test_duplicate_seat_409_is_retried_then_succeeds():
    """PROTOCOL.md: a seat may reconnect. A 409 usually means the seat's
    previous (stale) connection has not been reaped yet — the server
    heartbeats and strike-closes it — so the client must retry, not die."""
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise web.HTTPConflict(text="slot already connected")
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(_obs_msg(0))
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                break
        await ws.send_str(json.dumps(
            {"done": True, "result": {"scores": [1.0]}}))
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/player", handler)
    server = TestServer(app)
    await server.start_server()
    try:
        result = await play_episode(
            random_player.RandomPolicy(seed=0),
            str(server.make_url("/player")),
            reconnect_delay_seconds=0.01)
    finally:
        await server.close()
    assert result == {"scores": [1.0]}
    assert calls == 3


async def test_duplicate_seat_409_retry_budget_is_bounded():
    """A slot that stays occupied exhausts the reconnect budget (bounded
    retry, not an infinite loop and not an instant fatal)."""
    calls = 0

    async def always_conflict(request):
        nonlocal calls
        calls += 1
        raise web.HTTPConflict(text="slot already connected")

    app = web.Application()
    app.router.add_get("/player", always_conflict)
    server = TestServer(app)
    await server.start_server()
    try:
        with pytest.raises(PlayerError, match="giving up"):
            await play_episode(
                random_player.RandomPolicy(seed=0),
                str(server.make_url("/player")),
                max_connect_attempts=3,
                reconnect_delay_seconds=0.01)
    finally:
        await server.close()
    assert calls == 3


# -- reconnect behavior ------------------------------------------------------

def _obs_msg(tick, heroes=1):
    return json.dumps({
        "tick": tick,
        "obs": [base64.b64encode(bytes(510)).decode("ascii")] * heroes,
    })


class FlakyProtocolServer:
    """Fake seat endpoint: drops the first connection mid-episode, then
    finishes the episode on the second connection."""

    def __init__(self):
        self.connections = 0
        self.received = []  # decoded action replies across connections

    async def handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.connections += 1

        async def collect(n):
            got = 0
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    break
                self.received.append(json.loads(msg.data))
                got += 1
                if got >= n:
                    return
            raise AssertionError("client closed early")

        if self.connections == 1:
            await ws.send_str(_obs_msg(0))
            await collect(1)
            await ws.send_str(_obs_msg(1))
            await collect(1)
            await ws.close()  # drop mid-episode
        else:
            await ws.send_str(_obs_msg(2))
            await collect(1)
            await ws.send_str(json.dumps(
                {"done": True, "result": {"scores": [1.0]}}))
            await ws.close()
        return ws


async def test_client_reconnects_after_midgame_drop():
    fake = FlakyProtocolServer()
    app = web.Application()
    app.router.add_get("/player", fake.handler)
    server = TestServer(app)
    await server.start_server()
    try:
        result = await play_episode(
            random_player.RandomPolicy(seed=3),
            str(server.make_url("/player")),
            reconnect_delay_seconds=0.01)
    finally:
        await server.close()

    assert result == {"scores": [1.0]}
    assert fake.connections == 2
    # all three ticks answered, echoing the server's tick numbers
    assert [m["tick"] for m in fake.received] == [0, 1, 2]
    assert all(len(m["actions"]) == 1 and len(m["actions"][0]) == 6
               for m in fake.received)


async def test_client_gives_up_after_bounded_reconnects():
    """Connections that never make progress exhaust the attempt budget."""
    drops = 0

    async def drop_immediately(request):
        nonlocal drops
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        drops += 1
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/player", drop_immediately)
    server = TestServer(app)
    await server.start_server()
    try:
        with pytest.raises(PlayerError, match="giving up"):
            await play_episode(
                random_player.RandomPolicy(seed=0),
                str(server.make_url("/player")),
                max_connect_attempts=3,
                reconnect_delay_seconds=0.01)
    finally:
        await server.close()
    assert drops == 3


async def test_unreachable_server_gives_up():
    with pytest.raises(PlayerError, match="giving up"):
        await play_episode(
            random_player.RandomPolicy(seed=0),
            "http://127.0.0.1:9/player",  # port 9 (discard): refused
            max_connect_attempts=2,
            reconnect_delay_seconds=0.01)


async def test_reconnect_attempts_are_logged(capsys):
    fake = FlakyProtocolServer()
    app = web.Application()
    app.router.add_get("/player", fake.handler)
    server = TestServer(app)
    await server.start_server()
    try:
        await play_episode(
            random_player.RandomPolicy(seed=3),
            str(server.make_url("/player")),
            reconnect_delay_seconds=0.01)
    finally:
        await server.close()
    err = capsys.readouterr().err
    assert "connection attempt failed" in err
    assert "2 ticks answered so far" in err


async def test_malformed_obs_is_clean_player_error():
    """A garbage obs payload raises PlayerError (clean exit-1 path in
    run_policy_main), not a raw TypeError/binascii traceback."""

    async def bad_obs_server(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps({"tick": 0, "obs": "not-a-list!!"}))
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/player", bad_obs_server)
    server = TestServer(app)
    await server.start_server()
    try:
        with pytest.raises(PlayerError, match="malformed obs"):
            await play_episode(
                random_player.RandomPolicy(seed=0),
                str(server.make_url("/player")))
    finally:
        await server.close()


async def test_policy_row_count_mismatch_is_fatal():
    """A policy returning the wrong number of action rows fails fast
    locally instead of degrading to silent server-side strikes."""

    async def one_hero_server(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(_obs_msg(0, heroes=1))
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/player", one_hero_server)
    server = TestServer(app)
    await server.start_server()
    try:
        with pytest.raises(PlayerError, match="2 action rows for 1 heroes"):
            await play_episode(
                lambda tick, obs_rows: [[3, 3, 0, 0, 0, 0]] * 2,
                str(server.make_url("/player")))
    finally:
        await server.close()


# -- env plumbing + policy properties ----------------------------------------

def test_bad_seed_env_is_fatal(monkeypatch):
    monkeypatch.setenv("COGAME_PLAYER_SEED", "not-a-number")
    with pytest.raises(PlayerError, match="COGAME_PLAYER_SEED"):
        client.seed_from_env()
    with pytest.raises(PlayerError, match="COGAME_PLAYER_SEED"):
        random_player.policy_from_env()
    monkeypatch.delenv("COGAME_PLAYER_SEED")
    assert client.seed_from_env(default=7) == 7


async def test_a_slow_draft_keeps_answering_the_ws_heartbeat():
    """A draft decision slower than the server's ping interval must not
    stall the read loop.

    aiohttp answers a peer PING only from inside ``receive()``, so a draft
    awaited inline in ``async for msg in ws`` stops the client from
    ponging: the server closes the socket at ping + ping/2 and the drafted
    reply is lost (the seat degrades to ``disconnected`` + the neutral
    loadout). Production numbers are a 20 s heartbeat
    (``server.PLAYER_WS_HEARTBEAT_SECONDS``) against an LLM draft path
    bounded at 2 x 20 s; this is the same shape at 0.4 s / 1.2 s.
    """
    heartbeat, block = 0.4, 1.2      # block > heartbeat * 1.5
    picks = {"arm": "arm_cleaver", "tail": "tail_plate", "misc": "misc_regen"}
    seen = {"draft": None, "ticks": [], "non_text": None}

    async def handler(request):
        ws = web.WebSocketResponse(heartbeat=heartbeat)
        await ws.prepare(request)
        await ws.send_str(json.dumps({"phase": "draft", "seat": 0}))
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                seen["non_text"] = str(msg.type)
                break
            data = json.loads(msg.data)
            if data.get("phase") == "draft":
                seen["draft"] = data
                await ws.send_str(_obs_msg(0))   # the socket survived: play on
                continue
            seen["ticks"].append(data)
            break
        await ws.send_str(json.dumps(
            {"done": True, "result": {"scores": [1.0]}}))
        await ws.close()
        return ws

    class SlowDraftPolicy(random_player.RandomPolicy):
        async def on_draft(self, observation):
            await asyncio.sleep(block)
            return picks

    app = web.Application()
    app.router.add_get("/player", handler)
    server = TestServer(app)
    await server.start_server()
    try:
        result = await play_episode(
            SlowDraftPolicy(seed=0), str(server.make_url("/player")),
            max_connect_attempts=1)   # no reconnect can paper this over
    finally:
        await server.close()

    assert seen["draft"] == {"phase": "draft", "picks": [picks]}
    assert seen["non_text"] is None      # the socket was never closed on us
    assert len(seen["ticks"]) == 1       # and it still played a tick after
    assert result == {"scores": [1.0]}


def test_ws_url_env_precedence(monkeypatch):
    monkeypatch.delenv("COWORLD_PLAYER_WS_URL", raising=False)
    monkeypatch.delenv("COGAMES_ENGINE_WS_URL", raising=False)
    with pytest.raises(PlayerError, match="COWORLD_PLAYER_WS_URL"):
        client.ws_url_from_env()
    monkeypatch.setenv("COGAMES_ENGINE_WS_URL", "ws://b/player")
    assert client.ws_url_from_env() == "ws://b/player"
    monkeypatch.setenv("COWORLD_PLAYER_WS_URL", "ws://a/player")
    assert client.ws_url_from_env() == "ws://a/player"


def test_random_policy_seeded_and_in_range(monkeypatch):
    # players/ deliberately duplicates ACT_HIGH; keep it in sync
    assert random_player.ACT_HIGH == defaults.ACT_HIGH

    a = random_player.RandomPolicy(seed=42)
    b = random_player.RandomPolicy(seed=42)
    obs_rows = [bytes(510)] * 5
    acts_a = [a(t, obs_rows) for t in range(20)]
    acts_b = [b(t, obs_rows) for t in range(20)]
    assert acts_a == acts_b  # deterministic under COGAME_PLAYER_SEED
    arr = np.asarray(acts_a)
    assert arr.shape == (20, 5, 6)
    assert (arr >= 0).all()
    assert (arr < np.asarray(defaults.ACT_HIGH)).all()

    monkeypatch.setenv("COGAME_PLAYER_SEED", "7")
    p1 = random_player.policy_from_env()
    p2 = random_player.policy_from_env()
    assert p1(0, obs_rows) == p2(0, obs_rows)
