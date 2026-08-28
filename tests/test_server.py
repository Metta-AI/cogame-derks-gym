"""End-to-end tests for the Coworld-contract websocket game server.

In-process aiohttp test server + real websocket clients.
"""

import asyncio
import base64
import json

import aiohttp
import numpy as np
import pytest
from aiohttp import WSMsgType
from aiohttp.test_utils import TestServer

from cogame_derks_gym import defaults, uris
from cogame_derks_gym.config import GameConfig
from cogame_derks_gym.replay import Replay
from cogame_derks_gym.server import GameServer


SEATS = defaults.NUM_SEATS  # 6


def make_config(**overrides):
    d = {
        "players": [{"name": f"bot-{i}"} for i in range(SEATS)],
        "tokens": [f"token-{i}" for i in range(SEATS)],
        "seed": 21,
        "max_ticks": 20,
        "tick_deadline_ms": 1000,
        "player_connect_timeout_seconds": 10,
        # The floor the config schema allows: a test client that does not
        # answer the draft turn should cost one second, not 45.
        "draft_deadline_ms": 1000,
        # Generous by default so no test trips the engine's hard stop by
        # accident; the wall-clock test sets its own.
        "wall_clock_budget_seconds": 120,
    }
    d.update(overrides)
    return GameConfig.from_dict(d)


class ServerHarness:
    def __init__(self, cfg, tmp_path):
        self.results_path = tmp_path / "results.json"
        self.replay_path = tmp_path / "replay.bin"
        self.failure_path = tmp_path / "player_failure.json"
        self.server = GameServer(
            cfg,
            results_uri=f"file://{self.results_path}",
            save_replay_uri=f"file://{self.replay_path}",
            player_failure_uri=f"file://{self.failure_path}",
        )
        self.test_server = TestServer(self.server.make_app())
        self.episode_task = None

    async def __aenter__(self):
        await self.test_server.start_server()
        self.episode_task = asyncio.create_task(self.server.run_episode())
        return self

    async def __aexit__(self, *exc):
        if not self.episode_task.done():
            self.episode_task.cancel()
        try:
            await self.episode_task
        except asyncio.CancelledError:
            pass
        await self.test_server.close()

    def ws_url(self, slot, token):
        return str(self.test_server.make_url(
            f"/player?slot={slot}&token={token}"))


# Every hand-written client below answers the single draft turn the same
# way; a client that ignored it would spend the shared draft deadline
# before every episode.
DRAFT_REPLY = json.dumps({
    "phase": "draft",
    "picks": [{"arm": "arm_blaster", "tail": "tail_plate",
               "misc": "misc_battery"}]})


async def handle_phase(ws, data, *, answer=True) -> bool:
    """True when `data` was a phase message (draft / draft_result / an
    unrecognised phase), in which case the caller must not treat it as a
    per-tick message."""
    phase = data.get("phase")
    if phase is None:
        return False
    if phase == "draft" and answer:
        await ws.send_str(DRAFT_REPLY)
    return True


async def play_random_client(harness, slot, token, heroes=1,
                             picks=None):
    """A well-behaved player: answers the draft turn, then random
    in-range actions until done."""
    rng = np.random.default_rng(slot)
    if picks is None:
        picks = {"arm": "arm_blaster", "tail": "tail_plate",
                 "misc": "misc_battery", "note": f"seat {slot}"}
    result = None
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(harness.ws_url(slot, token)) as ws:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    break
                data = json.loads(msg.data)
                if data.get("done"):
                    result = data["result"]
                    break
                if data.get("phase") == "draft":
                    assert "catalog" in data and "hero" in data
                    await ws.send_str(json.dumps(
                        {"phase": "draft", "picks": [picks]}))
                    continue
                if data.get("phase") is not None:
                    continue  # draft_result / anything unrecognised
                obs = [base64.b64decode(o) for o in data["obs"]]
                assert len(obs) == heroes
                assert all(len(o) == 510 for o in obs)
                acts = rng.integers(
                    0, defaults.ACT_HIGH, size=(heroes, 6)).tolist()
                await ws.send_str(json.dumps(
                    {"tick": data["tick"], "actions": acts}))
    return result


# -- full episodes -----------------------------------------------------------

async def test_full_episode_six_seats(tmp_path):
    cfg = make_config(max_ticks=15)
    async with ServerHarness(cfg, tmp_path) as h:
        clients = [play_random_client(h, s, f"token-{s}")
                   for s in range(SEATS)]
        done_msgs = await asyncio.gather(*clients)
        result = await h.episode_task

    # every client got the done message with the result
    assert all(m is not None for m in done_msgs)
    assert done_msgs[0]["final_tick"] == result.final_tick

    results = json.loads(h.results_path.read_text())
    assert results["names"] == [f"bot-{i}" for i in range(SEATS)]
    assert len(results["scores"]) == SEATS
    assert results["final_tick"] == result.final_tick
    assert results["end_reason"] in ("ancient", "tick_cap")
    assert results["seed"] == 21
    assert results["team"] == ["radiant"] * 3 + ["dire"] * 3
    assert len(results["agent_stats"]) == defaults.NUM_HEROES
    # scores consistent with winner
    if results["winner"] is None:
        assert results["scores"] == [0.5] * SEATS
    else:
        winners = [s for i, s in enumerate(results["scores"])
                   if defaults.team_for_seat(i) == results["winner"]]
        losers = [s for i, s in enumerate(results["scores"])
                  if defaults.team_for_seat(i) != results["winner"]]
        assert winners == [1.0] * 3 and losers == [0.0] * 3
    assert sum(results["scores"]) == SEATS / 2

    replay = Replay.parse(h.replay_path.read_bytes())
    assert replay.tick_count == result.final_tick
    assert replay.header["config"]["seed"] == 21
    assert [p["name"] for p in replay.header["config"]["players"]] == \
        results["names"]
    assert replay.header["result"]["winner"] == results["winner"]
    # no failures reported
    assert not h.failure_path.exists()


async def test_full_episode_nodraft_variant(tmp_path):
    """The Puffer-fidelity variant: no draft turn is sent at all, every
    hero runs the neutral loadout, and the ten records still describe
    the sim (so the viewer needs no special case)."""
    cfg = make_config(max_ticks=12, draft_enabled=False)
    saw_draft = []

    async def client(slot):
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                    h.ws_url(slot, f"token-{slot}")) as ws:
                async for msg in ws:
                    if msg.type != WSMsgType.TEXT:
                        break
                    data = json.loads(msg.data)
                    if data.get("done"):
                        return data["result"]
                    if data.get("phase") == "draft":
                        saw_draft.append(slot)
                        continue
                    if data.get("phase") is not None:
                        continue
                    await ws.send_str(json.dumps({
                        "tick": data["tick"],
                        "actions": [list(defaults.NOOP_ACTION)]}))
        return None

    async with ServerHarness(cfg, tmp_path) as h:
        done_msgs = await asyncio.gather(*(client(s) for s in range(SEATS)))
        result = await h.episode_task

    assert saw_draft == [], "no draft turn may be sent when disabled"
    assert all(m is not None for m in done_msgs)
    results = json.loads(h.results_path.read_text())
    assert len(results["scores"]) == SEATS
    assert results["draft_fallbacks"] == [False] * SEATS
    assert len(results["draft"]) == defaults.NUM_HEROES
    for rec in results["draft"]:
        assert rec["picks"] == {"arm": "arm_none", "tail": "tail_none",
                                "misc": "misc_none"}, rec
    replay = Replay.parse(h.replay_path.read_bytes())
    assert replay.tick_count == result.final_tick
    assert replay.header["config"]["draft_enabled"] is False
    # nothing applied: the digest is the digest OF the all-zero applied
    # table (which is what a viewer that pushes nothing reproduces)
    from cogame_derks_gym import catalog
    assert replay.header["loadout_digest"] == catalog.loadout_digest()


# -- degraded players --------------------------------------------------------

async def test_missing_player_noop_and_failure_report(tmp_path):
    cfg = make_config(max_ticks=6, tick_deadline_ms=200,
                      player_connect_timeout_seconds=0.4)
    async with ServerHarness(cfg, tmp_path) as h:
        clients = [play_random_client(h, s, f"token-{s}")
                   for s in range(SEATS - 1)]  # the last slot never connects
        await asyncio.gather(*clients)
        result = await h.episode_task

    assert result.final_tick > 0
    failure = json.loads(h.failure_path.read_text())
    assert failure["failed_policy_index"] == SEATS - 1
    assert f"bot-{SEATS - 1}" in failure["message"]
    assert set(failure) == {"failed_policy_index", "message"}
    assert h.results_path.exists()
    assert h.replay_path.exists()


async def test_two_no_shows_report_lowest_slot(tmp_path):
    """COGAME_PLAYER_FAILURE_URI holds ONE GamePlayerFailure doc; with
    several no-shows the LOWEST slot (first failure in seat order) is
    reported — pinned so the report is deterministic, not loop-order
    happenstance."""
    cfg = make_config(max_ticks=4, tick_deadline_ms=100,
                      player_connect_timeout_seconds=0.3)
    async with ServerHarness(cfg, tmp_path) as h:
        clients = [play_random_client(h, s, f"token-{s}")
                   for s in range(SEATS - 2)]  # the last two never connect
        await asyncio.gather(*clients)
        await h.episode_task

    failure = json.loads(h.failure_path.read_text())
    assert failure["failed_policy_index"] == SEATS - 2
    assert f"bot-{SEATS - 2}" in failure["message"]


async def test_malformed_messages_never_crash_episode(tmp_path):
    cfg = make_config(max_ticks=6, tick_deadline_ms=150)

    async def malformed_client(h, slot, token):
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(h.ws_url(slot, token)) as ws:
                garbage = iter([
                    "not json at all",
                    json.dumps({"tick": -99, "actions": [[0] * 6]}),
                    json.dumps({"nonsense": True}),
                    json.dumps({"tick": None, "actions": "x"}),
                ])
                async for msg in ws:
                    if msg.type != WSMsgType.TEXT:
                        break
                    data = json.loads(msg.data)
                    if data.get("done"):
                        return data["result"]
                    if await handle_phase(ws, data, answer=False):
                        continue  # this client never drafts either
                    try:
                        await ws.send_str(next(garbage))
                    except StopIteration:
                        # then wrong-shaped actions on the right tick
                        await ws.send_str(json.dumps(
                            {"tick": data["tick"], "actions": [[1, 2]]}))
        return None

    async with ServerHarness(cfg, tmp_path) as h:
        good = [play_random_client(h, s, f"token-{s}")
                for s in range(SEATS - 1)]
        results = await asyncio.gather(
            *good, malformed_client(h, SEATS - 1, f"token-{SEATS - 1}"))
        result = await h.episode_task

    assert result.final_tick == 6
    # the malformed client stayed connected and still got the done message
    assert results[-1] is not None
    assert h.results_path.exists()


async def test_results_report_noop_causes(tmp_path):
    """results.json noop_causes attributes every degrade: a seat that
    keeps answering the wrong tick shows wrong_tick message counts and
    per-tick timeouts; clean seats show all zeros."""
    cfg = make_config(max_ticks=4, tick_deadline_ms=150)

    async def wrong_tick_client(h, slot, token):
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(h.ws_url(slot, token)) as ws:
                async for msg in ws:
                    if msg.type != WSMsgType.TEXT:
                        break
                    data = json.loads(msg.data)
                    if data.get("done"):
                        return data["result"]
                    if await handle_phase(ws, data, answer=False):
                        continue
                    await ws.send_str(json.dumps({
                        "tick": data["tick"] + 1000,
                        "actions": [list(defaults.NOOP_ACTION)]}))
        return None

    async with ServerHarness(cfg, tmp_path) as h:
        good = [play_random_client(h, s, f"token-{s}")
                for s in range(SEATS - 1)]
        results_msgs = await asyncio.gather(
            *good, wrong_tick_client(h, SEATS - 1, f"token-{SEATS - 1}"))
        await h.episode_task

    assert results_msgs[-1] is not None
    results = json.loads(h.results_path.read_text())
    causes = results["noop_causes"]
    assert len(causes) == SEATS
    assert set(causes[0]) == {"timeout", "malformed", "wrong_tick",
                              "disconnected", "host_error"}
    for seat in range(SEATS - 1):
        assert all(v == 0 for v in causes[seat].values()), (seat, causes)
    assert causes[SEATS - 1]["timeout"] == 4
    assert causes[SEATS - 1]["wrong_tick"] >= 1
    assert results["noop_ticks"][SEATS - 1] == 4


async def test_dead_seat_disconnect_during_probe_then_reconnect_revives(
        tmp_path):
    """A seat that goes strike-dead while connected has a revival probe
    parked on its websocket. If that socket then drops, the probe's
    waiter must be failed (fail_waiter) so the engine can re-probe —
    otherwise a reconnecting player can never revive the seat."""
    cfg = make_config(max_ticks=80, tick_deadline_ms=100,
                      player_connect_timeout_seconds=2)

    async def paced_client(h, slot, token):
        """Well-behaved but slow (~25ms/tick): keeps the episode running
        long enough for the flaky seat's disconnect + reconnect."""
        rng = np.random.default_rng(slot)
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(h.ws_url(slot, token)) as ws:
                async for msg in ws:
                    if msg.type != WSMsgType.TEXT:
                        break
                    data = json.loads(msg.data)
                    if data.get("done"):
                        return data["result"]
                    if await handle_phase(ws, data):
                        continue
                    await asyncio.sleep(0.025)
                    acts = rng.integers(0, defaults.ACT_HIGH,
                                        size=(1, 6)).tolist()
                    await ws.send_str(json.dumps(
                        {"tick": data["tick"], "actions": acts}))
        return None

    async with ServerHarness(cfg, tmp_path) as h:
        gather = asyncio.gather(*(
            paced_client(h, s, f"token-{s}") for s in range(SEATS - 1)))

        async def flaky(h):
            # Phase 1: connect but never reply. The seat racks up strikes,
            # goes dead, and a revival probe parks on this socket (the obs
            # stream stops once the single outstanding probe is parked).
            async with aiohttp.ClientSession() as session:
                ws = await session.ws_connect(
                    h.ws_url(SEATS - 1, f"token-{SEATS - 1}"))
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.receive(), 0.5)
                    except (asyncio.TimeoutError, TimeoutError):
                        break  # probe parked: nothing more will arrive
                    if msg.type != WSMsgType.TEXT:
                        break
                    # answer the draft (this seat's failure is about
                    # TICKS), then go silent
                    await handle_phase(ws, json.loads(msg.data))
                await ws.close()  # drop with the probe still parked
            # Phase 2: reconnect and play properly; must revive the seat.
            return await play_random_client(
                h, SEATS - 1, f"token-{SEATS - 1}")

        flaky_result = await asyncio.wait_for(flaky(h), 30)
        await gather
        result = await asyncio.wait_for(h.episode_task, 30)

    assert flaky_result is not None
    assert result.seat_dead[SEATS - 1] is False, \
        "reconnected seat never revived (stuck probe waiter?)"
    assert 0 < result.seat_noop_ticks[SEATS - 1] < 80


async def test_strike_death_force_closes_stale_socket_then_revive(tmp_path):
    """When a seat goes strike-dead the server force-closes its (possibly
    half-open) websocket: the client sees the close, reconnects, and the
    seat revives. Without the close a client whose socket went stale
    server-side would keep feeding a black hole forever."""
    cfg = make_config(max_ticks=80, tick_deadline_ms=100,
                      player_connect_timeout_seconds=2)

    async def paced_client(h, slot, token):
        rng = np.random.default_rng(slot)
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(h.ws_url(slot, token)) as ws:
                async for msg in ws:
                    if msg.type != WSMsgType.TEXT:
                        break
                    data = json.loads(msg.data)
                    if data.get("done"):
                        return data["result"]
                    if await handle_phase(ws, data):
                        continue
                    await asyncio.sleep(0.025)
                    acts = rng.integers(0, defaults.ACT_HIGH,
                                        size=(1, 6)).tolist()
                    await ws.send_str(json.dumps(
                        {"tick": data["tick"], "actions": acts}))
        return None

    async with ServerHarness(cfg, tmp_path) as h:
        gather = asyncio.gather(*(
            paced_client(h, s, f"token-{s}") for s in range(SEATS - 1)))

        async def flaky(h):
            # Never reply; the server must close this socket when the
            # seat strikes out.
            async with aiohttp.ClientSession() as session:
                ws = await session.ws_connect(
                    h.ws_url(SEATS - 1, f"token-{SEATS - 1}"))
                while True:
                    msg = await ws.receive()  # no timeout: server closes
                    if msg.type != WSMsgType.TEXT:
                        break
                    await handle_phase(ws, json.loads(msg.data))
            # Reconnect and play properly; must revive the seat.
            return await play_random_client(
                h, SEATS - 1, f"token-{SEATS - 1}")

        flaky_result = await asyncio.wait_for(flaky(h), 30)
        await gather
        result = await asyncio.wait_for(h.episode_task, 30)

    assert flaky_result is not None
    assert result.seat_dead[SEATS - 1] is False
    assert 0 < result.seat_noop_ticks[SEATS - 1] < 80


async def test_wall_clock_budget_writes_artifacts(tmp_path):
    """Wall-clock expiry ends the episode normally: results.json says
    end_reason="wall_clock" and the partial replay is written."""
    cfg = make_config(max_ticks=5000, tick_deadline_ms=1000,
                      wall_clock_budget_seconds=0.5)

    async def paced_client(h, slot, token):
        rng = np.random.default_rng(slot)
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(h.ws_url(slot, token)) as ws:
                async for msg in ws:
                    if msg.type != WSMsgType.TEXT:
                        break
                    data = json.loads(msg.data)
                    if data.get("done"):
                        return data["result"]
                    if await handle_phase(ws, data):
                        continue
                    await asyncio.sleep(0.02)
                    acts = rng.integers(0, defaults.ACT_HIGH,
                                        size=(1, 6)).tolist()
                    await ws.send_str(json.dumps(
                        {"tick": data["tick"], "actions": acts}))
        return None

    async with ServerHarness(cfg, tmp_path) as h:
        done_msgs = await asyncio.gather(*(
            paced_client(h, s, f"token-{s}") for s in range(SEATS)))
        result = await asyncio.wait_for(h.episode_task, 30)

    assert all(m is not None for m in done_msgs)
    results = json.loads(h.results_path.read_text())
    assert results["end_reason"] == "wall_clock"
    assert 0 < results["final_tick"] < 5000
    replay = Replay.parse(h.replay_path.read_bytes())
    assert replay.tick_count == result.final_tick


# -- auth + connection management --------------------------------------------

async def test_bad_token_rejected(tmp_path):
    cfg = make_config(player_connect_timeout_seconds=5)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(aiohttp.WSServerHandshakeError) as exc:
                await session.ws_connect(h.ws_url(3, "wrong-token"))
            assert exc.value.status == 403


@pytest.mark.parametrize("slot", ["6", "17", "-1", "abc", ""])
async def test_bad_slot_rejected(tmp_path, slot):
    cfg = make_config(player_connect_timeout_seconds=5)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(aiohttp.WSServerHandshakeError) as exc:
                await session.ws_connect(
                    str(h.test_server.make_url(
                        f"/player?slot={slot}&token=token-0")))
            assert exc.value.status == 403


async def test_duplicate_slot_rejected_while_alive(tmp_path):
    cfg = make_config(player_connect_timeout_seconds=5)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            ws1 = await session.ws_connect(h.ws_url(0, "token-0"))
            with pytest.raises(aiohttp.WSServerHandshakeError):
                await session.ws_connect(h.ws_url(0, "token-0"))
            await ws1.close()
            # dead connection may be replaced; the server's handler may
            # not have observed the close yet, so retry briefly
            for _ in range(40):
                try:
                    ws2 = await session.ws_connect(h.ws_url(0, "token-0"))
                    break
                except aiohttp.WSServerHandshakeError:
                    await asyncio.sleep(0.05)
            else:
                pytest.fail("reconnect to a dead slot was never accepted")
            await ws2.close()


async def test_seat_lifecycle_logged(tmp_path, capsys):
    """One stderr line each for connect, disconnect, and 409-reject
    (strike-death and revival lines are covered by the engine tests)."""
    cfg = make_config(player_connect_timeout_seconds=5)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            ws1 = await session.ws_connect(h.ws_url(0, "token-0"))
            with pytest.raises(aiohttp.WSServerHandshakeError):
                await session.ws_connect(h.ws_url(0, "token-0"))
            await ws1.close()
            await asyncio.sleep(0.05)  # let the handler's finally run
    err = capsys.readouterr().err
    assert "seat 0 (bot-0) connected at tick 0" in err
    assert "rejected duplicate connection (409)" in err
    assert "seat 0 (bot-0) disconnected at tick 0" in err


async def test_healthz(tmp_path):
    cfg = make_config(player_connect_timeout_seconds=5)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            async with session.get(h.test_server.make_url("/healthz")) as resp:
                assert resp.status == 200
                assert (await resp.json())["status"] == "ok"


# -- platform browser/viewer contract (coworld GAME.md) ----------------------
# The local certifier probes GET /client/player?slot&token, GET
# /client/global, and requires the /global websocket to produce a first
# message (coworld.runner.runner.run_episode_containers).

async def test_client_global_page(tmp_path):
    cfg = make_config(player_connect_timeout_seconds=5)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                    h.test_server.make_url("/client/global")) as resp:
                assert resp.status == 200
                assert "text/html" in resp.headers["Content-Type"]
                assert "/global" in await resp.text()


async def test_client_player_page_token_checked(tmp_path):
    cfg = make_config(player_connect_timeout_seconds=5)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            async with session.get(h.test_server.make_url(
                    "/client/player?slot=0&token=token-0")) as resp:
                assert resp.status == 200
                assert "text/html" in resp.headers["Content-Type"]
            async with session.get(h.test_server.make_url(
                    "/client/player?slot=0&token=wrong")) as resp:
                assert resp.status == 403
            async with session.get(h.test_server.make_url(
                    "/client/player?slot=99&token=token-0")) as resp:
                assert resp.status == 403


async def test_global_ws_first_message_and_done(tmp_path):
    """Viewer gets a snapshot immediately, then the final done message."""
    cfg = make_config(max_ticks=10)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            global_ws = await session.ws_connect(
                str(h.test_server.make_url("/global")))
            first = json.loads((await asyncio.wait_for(
                global_ws.receive(), 5)).data)
            assert first["type"] == "status"
            assert first["players"] == [f"bot-{i}" for i in range(SEATS)]
            assert first["aliases"] == list(defaults.SEAT_ALIASES)
            assert first["phase"] in ("draft", "play")
            assert first["done"] is False

            clients = [play_random_client(h, s, f"token-{s}")
                       for s in range(SEATS)]
            gather = asyncio.gather(*clients)
            done_msg = None
            while True:
                msg = await asyncio.wait_for(global_ws.receive(), 30)
                if msg.type != WSMsgType.TEXT:
                    break
                data = json.loads(msg.data)
                if data.get("done"):
                    done_msg = data
                    break
            await gather
            assert done_msg is not None
            assert len(done_msg["result"]["scores"]) == SEATS


async def test_global_ws_late_viewer_snapshot_is_self_contained(tmp_path):
    """A viewer connecting after the episode ended gets done + result in
    the connect snapshot (no later message to wait for)."""
    cfg = make_config(max_ticks=10)
    async with ServerHarness(cfg, tmp_path) as h:
        clients = [play_random_client(h, s, f"token-{s}")
                   for s in range(SEATS)]
        await asyncio.gather(*clients)
        await h.episode_task  # episode fully finished
        async with aiohttp.ClientSession() as session:
            ws = await session.ws_connect(
                str(h.test_server.make_url("/global")))
            first = json.loads((await asyncio.wait_for(
                ws.receive(), 5)).data)
            assert first["type"] == "status"
            assert first["done"] is True
            assert first["phase"] == "done"
            assert len(first["result"]["scores"]) == SEATS
            # alias-only loadouts: /global is a spectator surface, but a
            # policy could read it too
            assert len(first["loadouts"]) == defaults.NUM_HEROES
            assert all("player_name" not in rec for rec in first["loadouts"])
            await ws.close()


async def test_global_ws_sender_never_crashes_episode(tmp_path):
    """A viewer that sends garbage and disconnects mid-episode is harmless."""
    cfg = make_config(max_ticks=60)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            global_ws = await session.ws_connect(
                str(h.test_server.make_url("/global")))
            await asyncio.wait_for(global_ws.receive(), 5)  # snapshot
            await global_ws.send_str("not json at all")
            clients = [play_random_client(h, s, f"token-{s}")
                       for s in range(SEATS)]
            gather = asyncio.gather(*clients)
            await asyncio.sleep(0.1)
            await global_ws.close()  # disconnect while episode is running
            results = await gather
        assert results[0] is not None
        assert h.results_path.exists()


# -- uris --------------------------------------------------------------------

async def test_file_uri_round_trip(tmp_path):
    target = tmp_path / "deep" / "nested" / "out.bin"
    await uris.write_uri(f"file://{target}", b"\x00\x01payload")
    assert await uris.read_uri(f"file://{target}") == b"\x00\x01payload"
    # plain paths (no scheme) also work, matching the runtime convention
    plain = tmp_path / "plain.txt"
    await uris.write_uri(str(plain), b"hello")
    assert await uris.read_uri(str(plain)) == b"hello"


async def test_coworld_mount_style_uri_path():
    # file:///coworld/out/results.json must resolve to /coworld/out/...
    assert uris.local_path("file:///coworld/out/results.json") is not None
    assert str(uris.local_path("file:///coworld/out/results.json")) == \
        "/coworld/out/results.json"


async def test_http_uri_read_write():
    from aiohttp import web

    stored = {}

    async def handle_get(request):
        return web.Response(body=stored.get("blob", b""))

    async def handle_put(request):
        stored["blob"] = await request.read()
        stored["content_type"] = request.content_type
        return web.Response(status=201)

    app = web.Application()
    app.router.add_get("/artifact", handle_get)
    app.router.add_put("/artifact", handle_put)
    server = TestServer(app)
    await server.start_server()
    try:
        url = str(server.make_url("/artifact"))
        await uris.write_uri(url, b"http-bytes", "application/json")
        assert stored["blob"] == b"http-bytes"
        assert stored["content_type"] == "application/json"
        assert await uris.read_uri(url) == b"http-bytes"
    finally:
        await server.close()


async def test_http_read_retries_then_succeeds(capsys):
    from aiohttp import web

    attempts = 0

    async def handle_get(request):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return web.Response(status=500, text="flaky")
        return web.Response(body=b"config-bytes")

    app = web.Application()
    app.router.add_get("/config", handle_get)
    server = TestServer(app)
    await server.start_server()
    try:
        data = await uris.read_uri(str(server.make_url("/config")),
                                   backoff_seconds=0.01)
        assert data == b"config-bytes"
        assert attempts == 3
    finally:
        await server.close()
    # per-attempt failures are logged
    err = capsys.readouterr().err
    assert "attempt 1/3 failed" in err
    assert "status 500" in err


async def test_http_read_raises_after_exhausted_retries():
    from aiohttp import web

    attempts = 0

    async def handle_get(request):
        nonlocal attempts
        attempts += 1
        return web.Response(status=503)

    app = web.Application()
    app.router.add_get("/config", handle_get)
    server = TestServer(app)
    await server.start_server()
    try:
        with pytest.raises(IOError, match="503"):
            await uris.read_uri(str(server.make_url("/config")),
                                backoff_seconds=0.01)
        assert attempts == 3
    finally:
        await server.close()


async def test_http_zero_attempts_rejected():
    # regression: attempts=0 used to fall through to `raise None`
    with pytest.raises(ValueError, match="attempts"):
        await uris.write_uri("http://127.0.0.1:9/x", b"x", attempts=0)
    with pytest.raises(ValueError, match="attempts"):
        await uris.read_uri("http://127.0.0.1:9/x", attempts=0)


async def test_unsupported_scheme_rejected():
    with pytest.raises(ValueError):
        await uris.read_uri("s3://bucket/key")
    with pytest.raises(ValueError):
        await uris.write_uri("ftp://host/file", b"x")


# -- replay mode (Task 2.5) --------------------------------------------------

def _write_replay_bytes():
    from cogame_derks_gym.replay import ReplayWriter

    cfg = make_config()
    writer = ReplayWriter(cfg, "aa" * 32)
    rng = np.random.default_rng(3)
    for t in range(8):
        writer.append_tick(
            t, rng.integers(0, defaults.ACT_HIGH,
                            size=(defaults.NUM_HEROES, 6)).astype(np.uint8))
    return writer.finalize({"winner": 0, "end_reason": "ancient",
                            "final_tick": 8})


async def test_replay_mode_serves_bytes_and_viewer():
    from cogame_derks_gym.server import make_replay_app

    data = _write_replay_bytes()
    server = TestServer(make_replay_app(data))
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(server.make_url("/replay-data")) as resp:
                assert resp.status == 200
                assert resp.content_type == "application/octet-stream"
                assert await resp.read() == data
            async with session.get(server.make_url("/client/replay")) as resp:
                assert resp.status == 200
                assert resp.content_type == "text/html"
                html = await resp.text()
                assert "/replay-data" in html
            async with session.get(server.make_url("/healthz")) as resp:
                assert resp.status == 200
    finally:
        await server.close()


async def test_replay_mode_legacy_replay_ws_first_message():
    """The certifier's replay-loadable probe (coworld<=0.1.34 runs it
    even with a static bundle declared) needs one non-empty message
    from the /replay websocket."""
    from cogame_derks_gym.server import make_replay_app

    data = _write_replay_bytes()
    server = TestServer(make_replay_app(data))
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as session:
            ws = await session.ws_connect(str(server.make_url("/replay")))
            msg = json.loads((await asyncio.wait_for(ws.receive(), 5)).data)
            assert msg["type"] == "replay_header"
            assert msg["header"]["tick_count"] == 8
            assert msg["header"]["result"]["winner"] == 0
            await ws.close()
    finally:
        await server.close()


async def test_replay_mode_serves_viewer_bundle_when_built(tmp_path):
    """With a viewer/dist bundle present, /client/replay serves the real
    viewer index and its static assets (Task 4.2)."""
    from cogame_derks_gym.server import make_replay_app

    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text(
        "<!DOCTYPE html><title>viewer</title>fetches /replay-data")
    (dist / "derk_viewer.js").write_text("// glue")
    (dist / "derk_viewer.wasm").write_bytes(b"\x00asm fake")

    data = _write_replay_bytes()
    server = TestServer(make_replay_app(data, viewer_dist=dist))
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(server.make_url("/client/replay")) as resp:
                assert resp.status == 200
                # pins the relative-asset regression: the slashless URL
                # must 302 to /client/replay/ so derk_viewer.js resolves
                # under /client/replay/, not /client/
                assert [r.status for r in resp.history] == [302]
                assert resp.url.path == "/client/replay/"
                assert resp.content_type == "text/html"
                html = await resp.text()
                assert "viewer" in html
                assert "Placeholder" not in html
            async with session.get(
                    server.make_url("/client/replay/derk_viewer.js")) as resp:
                assert resp.status == 200
                assert await resp.text() == "// glue"
            async with session.get(
                    server.make_url("/client/replay/derk_viewer.wasm")) as resp:
                assert resp.status == 200
                assert await resp.read() == b"\x00asm fake"
            # bundle mode keeps /replay-data intact
            async with session.get(server.make_url("/replay-data")) as resp:
                assert resp.status == 200
                assert await resp.read() == data
    finally:
        await server.close()


async def test_replay_mode_falls_back_to_placeholder_without_bundle(tmp_path):
    """No viewer/dist (emscripten build not run): the placeholder page
    keeps server tests and dev flows working."""
    from cogame_derks_gym.server import make_replay_app

    data = _write_replay_bytes()
    server = TestServer(
        make_replay_app(data, viewer_dist=tmp_path / "no-such-dist"))
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(server.make_url("/client/replay")) as resp:
                assert resp.status == 200
                assert resp.content_type == "text/html"
                html = await resp.text()
                assert "Placeholder" in html
                assert "/replay-data" in html
    finally:
        await server.close()


async def test_replay_mode_rejects_corrupt_replay():
    from cogame_derks_gym.replay import ReplayError
    from cogame_derks_gym.server import make_replay_app

    with pytest.raises(ReplayError):
        make_replay_app(b"not a replay")


# -- sim fault containment (patch 0004) --------------------------------------

async def test_sim_fault_writes_results_and_partial_replay(tmp_path):
    """A sim fault (patch-0004 flag) ends the episode with results.json
    (end_reason sim_fault, closed key set intact) and a parseable
    partial replay — instead of the pre-patch exit() losing both."""
    from tests.test_engine import FaultingSim

    cfg = make_config(max_ticks=50, tick_deadline_ms=50,
                      player_connect_timeout_seconds=0.1)
    results_path = tmp_path / "results.json"
    replay_path = tmp_path / "replay.bin"
    server = GameServer(
        cfg,
        results_uri=f"file://{results_path}",
        save_replay_uri=f"file://{replay_path}",
        sim_factory=lambda seed: FaultingSim(fault_at=2),
    )
    result = await asyncio.wait_for(server.run_episode(), 30)
    assert result.end_reason == "sim_fault"

    results = json.loads(results_path.read_text())
    assert results["end_reason"] == "sim_fault"
    assert results["winner"] is None
    assert results["scores"] == [0.5] * SEATS
    # same closed key set as a normal episode
    normal_keys = set(server._results_doc(result))
    assert set(results) == normal_keys
    replay = Replay.parse(replay_path.read_bytes())
    assert replay.tick_count == 2


async def test_engine_exception_still_writes_fault_artifacts(tmp_path):
    """Even an unexpected host failure (here: the sim factory raising)
    writes fault results + the (empty) replay before re-raising."""

    def exploding_factory(seed):
        raise RuntimeError("host exploded")

    cfg = make_config(max_ticks=10, player_connect_timeout_seconds=0.1)
    results_path = tmp_path / "results.json"
    replay_path = tmp_path / "replay.bin"
    server = GameServer(
        cfg,
        results_uri=f"file://{results_path}",
        save_replay_uri=f"file://{replay_path}",
        sim_factory=exploding_factory,
    )
    with pytest.raises(RuntimeError, match="host exploded"):
        await asyncio.wait_for(server.run_episode(), 30)

    results = json.loads(results_path.read_text())
    assert results["end_reason"] == "sim_fault"
    assert results["final_tick"] == 0
    assert results["names"] == [f"bot-{i}" for i in range(SEATS)]
    replay = Replay.parse(replay_path.read_bytes())
    assert replay.tick_count == 0


# -- shutdown robustness (quality review) ------------------------------------

async def test_unresponsive_client_never_blocks_episode_exit(tmp_path):
    """A connected client that never reads or replies must not prevent
    run_episode from returning (bounded done-broadcast, strike rule)."""
    cfg = make_config(max_ticks=5, tick_deadline_ms=100,
                      player_connect_timeout_seconds=2,
                      wall_clock_budget_seconds=60)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            silent_ws = await session.ws_connect(
                h.ws_url(SEATS - 1, f"token-{SEATS - 1}"))
            good = [play_random_client(h, s, f"token-{s}")
                    for s in range(SEATS - 1)]
            await asyncio.gather(*good)
            result = await asyncio.wait_for(h.episode_task, timeout=20)
            await silent_ws.close()
    assert result.final_tick == 5
    results = json.loads(h.results_path.read_text())
    assert results["noop_ticks"][SEATS - 1] == 5
    assert results["noop_ticks"][:SEATS - 1] == [0] * (SEATS - 1)


async def test_failing_results_uri_does_not_block_replay_write(tmp_path):
    """Artifact writes are independent: a failing results URI must not
    prevent the replay write; the aggregate error is raised after."""
    cfg = make_config(max_ticks=3, tick_deadline_ms=50,
                      player_connect_timeout_seconds=0.1)
    replay_path = tmp_path / "replay.bin"
    server = GameServer(
        cfg,
        results_uri="badscheme://results",
        save_replay_uri=f"file://{replay_path}",
        player_failure_uri=f"file://{tmp_path / 'failure.json'}",
    )
    with pytest.raises(IOError):
        await server.run_episode()
    replay = Replay.parse(replay_path.read_bytes())
    assert replay.tick_count == 3


async def test_http_write_retries_then_succeeds():
    from aiohttp import web

    attempts = 0
    stored = {}

    async def handle_put(request):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return web.Response(status=500)
        stored["blob"] = await request.read()
        return web.Response(status=200)

    app = web.Application()
    app.router.add_put("/artifact", handle_put)
    server = TestServer(app)
    await server.start_server()
    try:
        url = str(server.make_url("/artifact"))
        await uris.write_uri(url, b"retried", "application/json",
                             backoff_seconds=0.01)
        assert attempts == 3
        assert stored["blob"] == b"retried"
    finally:
        await server.close()


async def test_http_write_raises_after_exhausted_retries():
    from aiohttp import web

    attempts = 0

    async def handle_put(request):
        nonlocal attempts
        attempts += 1
        return web.Response(status=503)

    app = web.Application()
    app.router.add_put("/artifact", handle_put)
    server = TestServer(app)
    await server.start_server()
    try:
        with pytest.raises(IOError):
            await uris.write_uri(str(server.make_url("/artifact")),
                                 b"x", backoff_seconds=0.01)
        assert attempts == 3
    finally:
        await server.close()
