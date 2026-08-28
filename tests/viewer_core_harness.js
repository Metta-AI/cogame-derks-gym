#!/usr/bin/env node
// Headless verification harness for the viewer core wasm (no raylib).
//
// Usage: node viewer_core_harness.js <build/viewer_core.js> <replay.bin>
//
// Loads the replay exactly like viewer/index.html does (header JSON
// parsed JS-side, bytes copied into the wasm heap, seed passed from
// header.config.seed, then the ten drafted loadout blocks pushed in with
// viewer_set_loadout before playback), exercises the core API —
// including malformed-bytes rejection — and prints one JSON object for
// tests/test_viewer.py to assert on. Exits non-zero on any failure.
"use strict";

const fs = require("fs");
const path = require("path");

const [, , coreJsPath, replayPath] = process.argv;
if (!coreJsPath || !replayPath) {
  console.error("usage: viewer_core_harness.js <viewer_core.js> <replay.bin>");
  process.exit(2);
}

const bytes = fs.readFileSync(replayPath);
if (bytes.toString("latin1", 0, 4) !== "DERK" || bytes[4] !== 2) {
  console.error("bad replay magic/version");
  process.exit(2);
}
const headerLen = bytes.readUInt32LE(5);
const header = JSON.parse(bytes.toString("utf-8", 9, 9 + headerLen));
if (!Number.isInteger(header.config.seed)) {
  console.error("header config.seed is not an integer:", header.config.seed);
  process.exit(2);
}

const createViewerCore = require(path.resolve(coreJsPath));

// viewer_load must return -1 for each of these, before any real load.
function malformedResults(M, call) {
  const tryLoad = (buf) => {
    const p = M._malloc(buf.length);
    M.HEAPU8.set(buf, p);
    const r = call("viewer_load", "number",
      ["number", "number", "number"], [p, buf.length, 1]);
    M._free(p);
    return r;
  };
  const goodPrefix = Buffer.from("DERK\x02", "latin1");
  const cases = {};
  cases.badMagic = tryLoad(Buffer.concat(
    [Buffer.from("NOPE\x02", "latin1"), Buffer.alloc(64)]));
  cases.badVersion = tryLoad(Buffer.concat(
    [Buffer.from("DERK\x09", "latin1"), Buffer.alloc(64)]));
  // replay v1 (MOBA) is deliberately not read by this viewer
  cases.v1Magic = tryLoad(Buffer.concat(
    [Buffer.from("MOBA\x01", "latin1"), Buffer.alloc(64)]));
  cases.tooShort = tryLoad(Buffer.from("DERK\x02\x00\x00", "latin1"));
  // header_len runs past end of buffer
  const truncated = Buffer.concat([goodPrefix, Buffer.alloc(4 + 8)]);
  truncated.writeUInt32LE(1000, 5);
  cases.truncatedHeader = tryLoad(truncated);
  // header_len near UINT32_MAX: 9 + header_len wraps on wasm32 — the
  // non-wrappable check must still reject it
  const wrapping = Buffer.concat([goodPrefix, Buffer.alloc(4 + 8)]);
  wrapping.writeUInt32LE(0xFFFFFFFF, 5);
  cases.wrappingHeaderLen = tryLoad(wrapping);
  // body not a multiple of 60
  const raggedHeader = Buffer.from("{}", "utf-8");
  const ragged = Buffer.concat(
    [goodPrefix, Buffer.alloc(4), raggedHeader, Buffer.alloc(61)]);
  ragged.writeUInt32LE(raggedHeader.length, 5);
  cases.raggedBody = tryLoad(ragged);
  return cases;
}

function run(M) {
  const call = (name, ret, args = [], vals = []) =>
    M.ccall(name, ret, args, vals);

  const malformed = malformedResults(M, call);

  const ptr = M._malloc(bytes.length);
  M.HEAPU8.set(bytes, ptr);
  const seed = header.config.seed >>> 0;  // & 0xFFFFFFFF, like the host
  const total = call("viewer_load", "number",
    ["number", "number", "number"], [ptr, bytes.length, seed]);

  // The draft: JS parses the header and pushes the ten applied stat
  // blocks in, exactly as viewer/derk_chrome.js does. Every later
  // sim_fresh() (i.e. every seek) re-applies them.
  const loadoutTypes = new Array(9).fill("number");
  for (const rec of (header.draft || []).slice().sort((a, b) => a.pid - b.pid)) {
    const a = rec.applied;
    call("viewer_set_loadout", null, loadoutTypes, [
      rec.pid, a.base_health, a.base_mana, a.base_damage,
      a.basic_attack_cd, a.move_speed, a.hp_gain_per_level,
      a.mana_gain_per_level, a.damage_gain_per_level]);
  }
  call("viewer_seek", null, ["number"], [0]);
  const loadoutDigest = call("viewer_loadout_digest", "number") >>> 0;

  // Frame cadence: at speed s, s ticks per 12 advance_frame calls.
  const ticksOver = (frames) => {
    let n = 0;
    for (let i = 0; i < frames; i++)
      n += call("viewer_advance_frame", "number");
    return n;
  };
  call("viewer_set_playing", null, ["number"], [1]);
  const cadence1 = ticksOver(12);
  call("viewer_set_speed", null, ["number"], [4]);
  const cadence4 = ticksOver(12);
  const pausedTicks = (() => {  // paused: advance_frame must be a no-op
    call("viewer_set_playing", null, ["number"], [0]);
    return ticksOver(24);
  })();

  const mid = Math.floor(total / 2);
  call("viewer_seek", null, ["number"], [mid]);
  const midTick = call("viewer_tick", "number");

  // Interpolation phase-lock (viewer_render_phase): at-tick (12) after
  // a seek; sweeps 0,1,2,... once ticks step at 1x; frozen across
  // pause/resume; pinned at-tick at multi-tick-per-frame speeds.
  const phase = () => call("viewer_render_phase", "number");
  const phaseAfterSeek = phase();
  call("viewer_set_speed", null, ["number"], [1]);
  call("viewer_set_playing", null, ["number"], [1]);
  let guard = 0;  // advance until the first tick fires (<= 12 frames)
  while (call("viewer_advance_frame", "number") === 0 && guard++ < 24) {}
  const phaseSweep = [phase()];
  call("viewer_advance_frame", "number");
  phaseSweep.push(phase());
  call("viewer_advance_frame", "number");
  phaseSweep.push(phase());
  call("viewer_set_playing", null, ["number"], [0]);
  ticksOver(5);  // paused: phase must freeze
  const phasePaused = phase();
  call("viewer_set_playing", null, ["number"], [1]);
  call("viewer_advance_frame", "number");
  const phaseResumed = phase();  // sweep continues, no backward reset
  call("viewer_set_speed", null, ["number"], [64]);
  call("viewer_advance_frame", "number");
  const phaseAt64 = phase();  // multi-tick frame: pinned at-tick

  // Time-based advance (viewer jitter fix): 1 tick per 200ms at 1x,
  // independent of callback count; a single callback's dt clamps to
  // 100ms so a backgrounded tab does not burst on return.
  call("viewer_seek", null, ["number"], [mid]);
  call("viewer_set_speed", null, ["number"], [1]);
  const dtTicks100a = call("viewer_advance", "number", ["number"], [100]);
  const dtTicks100b = call("viewer_advance", "number", ["number"], [100]);
  // 5000ms in one callback clamps to 100ms: half a tick, no burst
  const dtClamped = call("viewer_advance", "number", ["number"], [5000]);
  const dtAfterClamp = call("viewer_advance", "number", ["number"], [100]);

  call("viewer_seek", null, ["number"], [total]);
  const endTick = call("viewer_tick", "number");
  const playingAtEnd = call("viewer_playing", "number");
  // set_playing(1) at end must refuse (no silent restart/loop)
  call("viewer_set_playing", null, ["number"], [1]);
  const playAtEndRefused = call("viewer_playing", "number") === 0 ? 1 : 0;

  // Scorebug / minimap readouts at the final tick (the server records
  // its own values at the same tick; tests/test_viewer.py compares).
  const ancientHealth = [0, 1].map((team) =>
    call("viewer_ancient_health", "number", ["number"], [team]));
  const agentStats = [];
  for (let pid = 0; pid < 10; pid++) {
    agentStats.push([0, 1, 2, 3].map((which) =>
      call("viewer_agent_stat", "number", ["number", "number"],
           [pid, which])));
  }
  const positionsPtr = M._malloc(10 * 4 * 4);
  const heroCount = call("viewer_hero_positions", "number", ["number"],
                         [positionsPtr]);
  const view = new DataView(M.HEAPU8.buffer);
  const heroPositions = [];
  for (let i = 0; i < heroCount; i++) {
    const base = positionsPtr + i * 16;
    heroPositions.push([
      view.getFloat32(base, true), view.getFloat32(base + 4, true),
      view.getFloat32(base + 8, true), view.getFloat32(base + 12, true)]);
  }
  M._free(positionsPtr);

  // Camera selection (the #derk-viewpanel affordance).
  call("viewer_set_camera", null, ["number"], [7]);
  const camera = call("viewer_camera", "number");

  console.log(JSON.stringify({
    malformed,
    loadoutDigest, ancientHealth, agentStats, heroPositions, camera,
    headerLoadoutDigest: (header.loadout_digest || 0) >>> 0,
    total, cadence1, cadence4, pausedTicks, midTick, endTick,
    playingAtEnd, playAtEndRefused,
    phaseAfterSeek, phaseSweep, phasePaused, phaseResumed, phaseAt64,
    dtTicks100a, dtTicks100b, dtClamped, dtAfterClamp,
    done: call("viewer_done", "number"),
    winner: call("viewer_winner", "number"),
    // u32 digest (ccall returns the i32 bit pattern; normalize)
    stateDigest: call("viewer_state_digest", "number") >>> 0,
    headerTickCount: header.tick_count,
  }));
}

createViewerCore().then(run).catch((e) => {
  console.error("harness failed:", e);
  process.exit(1);
});
