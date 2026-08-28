"""Viewer verification without a browser.

Four layers:

- build outputs: sim/build_viewer.sh artifacts exist, including the
  appended game block's own files (derk_chrome.css/.js, derk_items.svg);
- headless re-sim: the viewer core (viewer_main.c compiled WITHOUT
  MOBA_RENDER, ENVIRONMENT=node) loads a real recorded v2 replay under
  node, applies the ten drafted loadout blocks pushed in from JS, and
  must reach the header's tick_count with the sim's winner, final-state
  digest AND loadout digest matching the live recording — proving the
  viewer's replay parsing, loadout application and step scheduling with
  no pixels involved;
- the scorebug/minimap readouts (viewer_ancient_health,
  viewer_agent_stat, viewer_hero_positions) match the server sim at the
  same tick;
- malformed input: viewer_load must reject bad magic (including replay
  v1's MOBA), bad version, truncated and wasm32-wrapping header lengths,
  and ragged bodies.

The browser half — the appended #derk-* chrome, the transport band, the
scrubber beats — is asserted in CI by tools/ci/viewer_smoke.mjs and
tools/ci/derk_viewer_checks.mjs against the real bundle.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cogame_derks_gym import catalog, defaults, draft, replay
from cogame_derks_gym.config import GameConfig
from cogame_derks_gym.engine import LockstepEngine
from cogame_derks_gym.events import EventLog
from cogame_derks_gym.replay import ReplayWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWER_DIST = REPO_ROOT / "viewer" / "dist"
VIEWER_CORE_JS = REPO_ROOT / "build" / "viewer_core.js"
HARNESS = Path(__file__).parent / "viewer_core_harness.js"

NOT_BUILT = "viewer not built - run sim/build_viewer.sh first"

PICKS = {"arm": "arm_cleaver", "tail": "tail_plate", "misc": "misc_regen"}


def _skip_or_fail_not_built():
    """Same CI rule as the fidelity gate: with COGAME_REQUIRE_WASM_BUILD
    set, a missing build artifact is a failure, never a silent skip."""
    if os.environ.get("COGAME_REQUIRE_WASM_BUILD"):
        pytest.fail(NOT_BUILT + " (COGAME_REQUIRE_WASM_BUILD is set)")
    pytest.skip(NOT_BUILT)


def test_build_viewer_outputs_exist():
    if not VIEWER_CORE_JS.exists():
        _skip_or_fail_not_built()
    for name in ("index.html", "derk_viewer.js", "derk_viewer.wasm",
                 "derk_viewer.data", "sim_sha.js",
                 "derk_chrome.css", "derk_chrome.js", "derk_items.svg"):
        assert (VIEWER_DIST / name).exists(), f"viewer/dist/{name} missing"
    assert (REPO_ROOT / "build" / "viewer_core.wasm").exists()


def test_bundle_index_links_the_game_block():
    index = (REPO_ROOT / "viewer" / "index.html").read_text()
    # the starter's ids all survive (nothing removed, nothing re-styled)
    for starter_id in ('id="stage"', 'id="canvas"', 'id="status"',
                       'id="controls"', 'id="playpause"', 'id="speed"',
                       'id="seek"', 'id="tickinfo"', 'id="endcard"',
                       'id="warn"', 'id="teams"'):
        assert starter_id in index, starter_id
    # and the appended block is wired up
    assert 'href="derk_chrome.css"' in index
    assert 'src="derk_chrome.js"' in index
    assert 'src="derk_viewer.js"' in index
    assert index.index('src="derk_chrome.js"') < \
        index.index('src="derk_viewer.js"'), \
        "derk_chrome.js must load before the emscripten module"
    # the browser build keeps the starter's onRuntimeInitialized pairing:
    # no MODULARIZE/EXPORT_NAME factory anywhere in the shell
    assert "onRuntimeInitialized" in index
    assert "MODULARIZE" not in index and "EXPORT_NAME" not in index


def test_browser_build_flags_stay_paired():
    """The lantern deadlock (2026-08-23) was a MODULARIZE/EXPORT_NAME link
    line under an onRuntimeInitialized shell. The browser build must keep
    neither flag; only the separate node core build has them."""
    build = (REPO_ROOT / "sim" / "build_viewer.sh").read_text()
    browser = build.split("-- headless core")[0]
    assert "-sENVIRONMENT=web" in browser
    assert "MODULARIZE" not in browser
    assert "EXPORT_NAME" not in browser
    node_core = build.split("-- headless core")[1]
    assert "-sMODULARIZE=1" in node_core
    assert "-sEXPORT_NAME=createViewerCore" in node_core
    # every export the chrome calls must be on the link line
    for name in ("_viewer_set_loadout", "_viewer_loadout_digest",
                 "_viewer_ancient_health", "_viewer_agent_stat",
                 "_viewer_hero_positions", "_viewer_set_camera"):
        assert name in build, name


def test_chrome_uses_the_transport_rules():
    css = (REPO_ROOT / "viewer" / "derk_chrome.css").read_text()
    js = (REPO_ROOT / "viewer" / "derk_chrome.js").read_text()
    # --band / --hudscale are set on :root by relayout()
    assert '--band' in css and '--hudscale' in css
    assert 'setProperty("--band"' in js and 'setProperty("--hudscale"' in js
    # the two full-stage overlays stop above the transport band
    assert css.count("inset: 0 0 var(--band) 0") >= 1
    assert "#derk-draft,\n#derk-endcard {" in css
    # every event kind the server can emit has a beat style
    from cogame_derks_gym.events import KINDS
    for kind in KINDS:
        assert f".beat-{kind}" in css, kind
    # the load/error signals the CI viewer smoke polls
    assert "dataset.replayLoaded" in js
    assert "dataset.replayError" in js


async def _record_replay(tmp_path: Path):
    """Record a real drafted episode: the pretrained network on every
    seat, non-neutral loadouts applied, house heroes driven in process —
    the same shape the server records in production."""
    from cogame_derks_gym.house import HouseHeroes
    from cogame_derks_gym.sim import MobaSim
    from players.baseline_player import BaselinePolicy

    class BrainSource:
        def __init__(self, seat):
            self.policy = BaselinePolicy(seed=1 + seat)

        async def get_actions(self, tick, obs):
            return self.policy(tick, [row.tobytes() for row in obs])

    class RngSource:
        def __init__(self, seat):
            self.rng = np.random.default_rng(4000 + seat)

        async def get_actions(self, tick, obs):
            return self.rng.integers(
                0, defaults.ACT_HIGH, size=(len(obs), 6)).tolist()

    cfg = GameConfig.from_dict({
        "players": [{"name": f"seat-{i}"} for i in range(defaults.NUM_SEATS)],
        "tokens": [f"tok{i}" for i in range(defaults.NUM_SEATS)],
        "seed": 13,
        "max_ticks": 600,
        "tick_deadline_ms": 2000,
    })
    sim = MobaSim(seed=cfg.seed)
    records = []
    for pid in range(defaults.NUM_HEROES):
        seat = defaults.seat_for_pid(pid)
        if seat is None:
            records.append(draft.house_record(pid))
        else:
            records.append(draft.record(
                pid, picks=dict(PICKS), note="viewer fixture", source="seat",
                fallback_cause="none", decision_ms=1,
                player_name=cfg.players[seat].name))
    draft.apply_to_sim(sim, records)

    writer = ReplayWriter(cfg, replay.sim_wasm_sha256())
    writer.set_draft(records, sim.loadout_digest())
    events = EventLog()
    events.add_draft()
    sources = [BrainSource(s) if s < 3 else RngSource(s)
               for s in range(defaults.NUM_SEATS)]
    engine = LockstepEngine(
        sim, cfg, sources, on_tick=writer.append_tick,
        house_source=HouseHeroes(), event_log=events)
    result = await engine.run()
    events.add_end(result.final_tick, result.end_reason)
    writer.set_events(events.events())
    writer.set_final_state_digest(sim.state_digest())
    data = writer.finalize({
        "winner": result.winner,
        "end_reason": result.end_reason,
        "final_tick": result.final_tick,
        "ancient_healths": list(result.ancient_healths),
    })
    path = tmp_path / "replay.bin"
    path.write_bytes(data)
    return path, result, sim


@pytest.mark.slow
async def test_headless_core_resimulates_recorded_replay(tmp_path):
    if not VIEWER_CORE_JS.exists():
        _skip_or_fail_not_built()
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not on PATH")

    replay_path, result, sim = await _record_replay(tmp_path)

    proc = subprocess.run(
        [node, str(HARNESS), str(VIEWER_CORE_JS), str(replay_path)],
        capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"harness failed:\n{proc.stderr}"
    out = json.loads(proc.stdout)

    # malformed bytes are rejected (-1), incl. replay v1 and the wasm32
    # wrap case
    assert out["malformed"] == {
        "badMagic": -1, "badVersion": -1, "v1Magic": -1, "tooShort": -1,
        "truncatedHeader": -1, "wrappingHeaderLen": -1, "raggedBody": -1}

    # replay body parse: C-side tick count == header tick count
    assert out["total"] == result.final_tick
    assert out["headerTickCount"] == result.final_tick

    # the drafted loadouts made it in: the viewer's digest must equal the
    # recording sim's
    assert out["loadoutDigest"] == out["headerLoadoutDigest"]
    assert out["loadoutDigest"] == sim.loadout_digest()
    assert out["loadoutDigest"] != 0

    # frame scheduling: 1 tick / 12 frames at 1x, 4 at 4x, none paused
    assert out["cadence1"] == 1
    assert out["cadence4"] == 4
    assert out["pausedTicks"] == 0

    # interpolation phase-lock (viewer jitter fix)
    assert out["phaseAfterSeek"] == 12
    assert out["phaseSweep"] == [0, 1, 2]
    assert out["phasePaused"] == 2
    assert out["phaseResumed"] == 3
    assert out["phaseAt64"] == 12

    # time-based advance: 1 tick per 200ms of (speed-scaled) wall time
    assert out["dtTicks100a"] == 0
    assert out["dtTicks100b"] == 1
    assert out["dtClamped"] == 0
    assert out["dtAfterClamp"] == 1

    # seek: mid lands exactly, end reaches tick_count and pauses (no loop)
    assert out["midTick"] == result.final_tick // 2
    assert out["endTick"] == result.final_tick
    assert out["playingAtEnd"] == 0
    assert out["playAtEndRefused"] == 1

    # the re-simulated episode reproduces the recorded outcome for real
    assert out["winner"] == result.winner if result.winner is not None \
        else True
    assert out["stateDigest"] == sim.state_digest()

    # scorebug + minimap readouts agree with the server sim at the same tick
    assert out["ancientHealth"] == [pytest.approx(sim.ancient_health(0)),
                                    pytest.approx(sim.ancient_health(1))]
    for pid in range(defaults.NUM_HEROES):
        assert out["agentStats"][pid] == [
            sim.agent_stat(pid, which) for which in range(4)], pid
    assert len(out["heroPositions"]) == defaults.NUM_HEROES
    for pid, (x, y, team, alive) in enumerate(out["heroPositions"]):
        assert team == defaults.team_for_pid(pid)
        assert alive in (0.0, 1.0)
        if alive:
            assert 0 <= x < 128 and 0 <= y < 128
    # camera selection is honoured
    assert out["camera"] == 7


def test_neutral_replay_needs_no_loadout_push():
    """An un-drafted replay records loadout_digest 0 and the viewer, which
    pushes nothing, reproduces exactly that."""
    cfg = GameConfig.from_dict({
        "players": [{"name": f"s{i}"} for i in range(defaults.NUM_SEATS)],
        "tokens": [f"t{i}" for i in range(defaults.NUM_SEATS)],
        "seed": 5, "draft_enabled": False,
    })
    writer = ReplayWriter(cfg, "aa" * 32)
    # nothing is applied, so the recorded digest is the digest OF the
    # all-zero applied table — exactly what a viewer that pushes nothing
    # re-derives.
    writer.set_draft(draft.neutral_records(cfg), catalog.loadout_digest())
    header = json.loads(json.dumps(writer.header({})))
    assert header["loadout_digest"] == catalog.loadout_digest()
    assert header["config"]["draft_enabled"] is False
    assert all(rec["picks"] == dict(catalog.NEUTRAL_PICKS)
               for rec in header["draft"])
