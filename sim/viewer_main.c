// cogame-derks-gym replay viewer: re-simulates a recorded episode in the
// browser and renders it with the upstream raylib client.
//
// Built twice by sim/build_viewer.sh, always against build/src-patched:
//   - with -DMOBA_RENDER (raylib + emscripten main loop)
//       -> viewer/dist/derk_viewer.{js,wasm,data}      (browser bundle)
//   - without MOBA_RENDER (headless core, ENVIRONMENT=node, --no-entry)
//       -> build/viewer_core.{js,wasm}                 (node verification)
//
// The core API (viewer_load / viewer_seek / viewer_advance_frame / ...)
// is identical in both builds; only the raylib main loop is render-only.
// This is what lets tests/test_viewer.py prove the re-sim logic under
// node without pixels.
//
// Replay format v2 (server/cogame_derks_gym/replay.py is the authority):
//   bytes 0-3  magic "DERK"; byte 4 version u8 == 2;
//   bytes 5-8  header_len u32le; then header JSON (parsed JS-side);
//   then tick_count * 60 bytes (10 heroes x 6 uint8 actions, post-clamp).
// C never parses the JSON: tick_count == body_len / 60 by construction.
//
// Determinism: viewer_seek() rebuilds the sim through the exact fresh-
// instance path the server host uses (zeroed struct -> moba_configure ->
// allocate_moba -> c_reset, patch 0002 srand(seed) inside init_moba) and
// replays the recorded actions from tick 0. Replay values are stored
// post-clamp, so a direct uint8 -> float cast matches the server's
// set_actions byte-for-byte.
//
// The draft: C never parses the replay header JSON, so JS pushes the ten
// applied loadout blocks in with viewer_set_loadout() before the first
// seek/play. sim_fresh() re-applies them immediately after c_reset(),
// which is exactly where the server applies them (Phase B step 6), so
// every re-simulation runs the drafted heroes. With no block pushed
// (an un-drafted replay) nothing is applied and re-simulation is
// byte-identical to the un-drafted sim.

#include <stdlib.h>
#include <string.h>

#include "shim_common.h"    // moba_configure(): shared env defaults (+ moba.h)
#include "loadout_common.h"  // derk_apply_loadout(): shared with sim/shim.c

#ifdef MOBA_RENDER
#include <emscripten.h>
#endif

#define VIEWER_MAGIC "DERK"
#define VIEWER_FORMAT_VERSION 2
#define VIEWER_MAGIC_LEN 9        // magic(4) + version u8 + header_len u32le
#define VIEWER_BYTES_PER_TICK 60  // 10 heroes x 6 uint8
#define VIEWER_NUM_AGENTS 10      // replays always drive all 10 heroes
#define VIEWER_FRAMES_PER_TICK 12 // upstream demo cadence: 1 sim tick per
                                  // 12 frames at 60 fps. Kept as the
                                  // interpolation-phase granularity and
                                  // the meaning of one "60Hz-equivalent
                                  // frame" (viewer_advance_frame).

// Nominal playback: 5 sim ticks/s at 1x (the upstream demo cadence),
// advanced by WALL TIME, not render-callback count — rAF callbacks fire
// per display refresh, so counting them plays 2x too fast on a 120 Hz
// display and hitches on every dropped frame.
#define VIEWER_TICK_MS (1000.0 * VIEWER_FRAMES_PER_TICK / 60.0)  // 200 ms
// Per-callback dt clamp: a backgrounded tab can sit for seconds between
// callbacks; do not burst-run that gap on return.
#define VIEWER_MAX_DT_MS 100.0

static MOBA env;
static int g_allocated = 0;

// Drafted loadouts, pushed in from JS (the header JSON is parsed JS-side).
// g_loadout_set[pid] gates application, so an un-drafted replay — or the
// four house heroes of a drafted one, which run the neutral block — is
// re-simulated exactly as recorded.
static float g_loadout[VIEWER_NUM_AGENTS][DERK_LOADOUT_FIELDS];
static int g_loadout_set[VIEWER_NUM_AGENTS];

static const unsigned char* g_body = NULL;  // action log, inside JS-owned buf
static int g_total_ticks = 0;
static int g_tick = 0;         // ticks fed so far == current sim tick
static int g_playing = 0;
static int g_speed = 1;         // playback multiplier (5*speed ticks/s)
static double g_time_acc = 0.0; // speed-scaled ms toward the next tick
                                // (invariant: < VIEWER_TICK_MS between
                                // viewer_advance calls)
static unsigned int g_seed = 0;
static int g_loaded = 0;

// Interpolation phase for the render half, in units of
// VIEWER_FRAMES_PER_TICK: how far the display sits through the current
// [last, cur] entity-position interpolation window. 0..N-1 mid-sweep;
// == N means "render exactly at-tick" (tick_frac 1.0). Upstream's
// c_render interpolates from its OWN free-running renderer->frame
// counter, which assumes exactly one sim tick per 12 phase-locked
// render calls; seeks and speed changes break that assumption and
// cause per-tick lurching, so frame() overwrites the renderer counter
// from this value before every c_render call.
static int g_phase = 0;

// Rebuild the sim exactly like a fresh wasm instance's moba_init(): the
// static struct starts zeroed, moba_configure sets the trained-on
// defaults + seed, allocate_moba re-allocates everything (cold ai_paths
// cache included) and runs init_moba (srand(seed), CachedRNG refill),
// then c_reset spawns. The renderer client survives across rebuilds.
static void sim_fresh(void) {
    void* client = env.client;  // GameRenderer*, owned by the render loop
    if (g_allocated)
        free_allocated_moba(&env);
    memset(&env, 0, sizeof(env));
    env.client = client;
    moba_configure(&env, g_seed, VIEWER_NUM_AGENTS);
    allocate_moba(&env);
    c_reset(&env);
    // Identical placement to the server's Phase-B step 6: after c_reset,
    // before the first c_step, ascending pid order.
    for (int pid = 0; pid < VIEWER_NUM_AGENTS; pid++) {
        if (g_loadout_set[pid])
            derk_apply_loadout(&env, pid, g_loadout[pid]);
    }
    g_allocated = 1;
    g_tick = 0;
}

static void feed_and_step(void) {
    // MultiDiscrete highs (exclusive) per action column; replays are
    // written post-clamp, but a hand-crafted body byte >= high would
    // index out of bounds inside the sim — clamp defensively.
    static const unsigned char act_max[6] = {6, 6, 2, 1, 1, 1};
    const unsigned char* a = g_body + (size_t)g_tick * VIEWER_BYTES_PER_TICK;
    for (int i = 0; i < VIEWER_BYTES_PER_TICK; i++) {
        unsigned char hi = act_max[i % 6];
        env.actions[i] = (float)(a[i] > hi ? hi : a[i]);
    }
    c_step(&env);
    g_tick++;
}

// Parse replay bytes at ptr/len (JS-owned wasm heap memory that must stay
// alive while loaded) and start a fresh sim at tick 0, paused. The header
// JSON is parsed JS-side; JS passes the seed from header.config.seed.
// Returns total tick count, or -1 on malformed bytes.
int viewer_load(const unsigned char* data, int len, unsigned int seed) {
    if (data == NULL || len < VIEWER_MAGIC_LEN)
        return -1;
    if (memcmp(data, VIEWER_MAGIC, 4) != 0 || data[4] != VIEWER_FORMAT_VERSION)
        return -1;
    unsigned int header_len = (unsigned int)data[5]
        | ((unsigned int)data[6] << 8)
        | ((unsigned int)data[7] << 16)
        | ((unsigned int)data[8] << 24);
    // Non-wrappable on wasm32: len >= VIEWER_MAGIC_LEN is established
    // above, so the subtraction is safe; adding to header_len is not.
    if (header_len > (unsigned int)(len - VIEWER_MAGIC_LEN))
        return -1;
    size_t body_len = (size_t)len - VIEWER_MAGIC_LEN - header_len;
    if (body_len % VIEWER_BYTES_PER_TICK != 0)
        return -1;

    g_body = data + VIEWER_MAGIC_LEN + header_len;
    g_total_ticks = (int)(body_len / VIEWER_BYTES_PER_TICK);
    g_seed = seed;
    g_playing = 0;
    g_time_acc = 0.0;
    g_phase = VIEWER_FRAMES_PER_TICK;  // display tick 0 exactly
    sim_fresh();
    g_loaded = 1;
    return g_total_ticks;
}

// Re-sim from tick 0 to `tick` (clamped to 0..total), no rendering.
void viewer_seek(int tick) {
    if (!g_loaded)
        return;
    if (tick < 0) tick = 0;
    if (tick > g_total_ticks) tick = g_total_ticks;
    sim_fresh();
    while (g_tick < tick)
        feed_and_step();
    g_time_acc = 0.0;
    g_phase = VIEWER_FRAMES_PER_TICK;  // display the seek target exactly
    if (g_tick >= g_total_ticks)
        g_playing = 0;  // seek-to-end lands in the "ended" state
}

// Advance the simulation by dt_ms of wall time (clamped to
// VIEWER_MAX_DT_MS). At speed s, one sim tick per VIEWER_TICK_MS/s of
// wall time — 5*s ticks/s regardless of display refresh rate. Pauses at
// end of replay instead of looping. Returns sim ticks stepped.
int viewer_advance(double dt_ms) {
    if (!g_loaded || !g_playing)
        return 0;
    if (dt_ms < 0.0)
        dt_ms = 0.0;
    if (dt_ms > VIEWER_MAX_DT_MS)
        dt_ms = VIEWER_MAX_DT_MS;  // tab-switch gap: no burst on return
    int stepped = 0;
    g_time_acc += dt_ms * (double)g_speed;
    // Epsilon: accumulated 60Hz dts round to 199.99999999999997 over 12
    // frames; a sub-nanosecond shortfall must not defer the tick a
    // whole callback.
    while (g_time_acc >= VIEWER_TICK_MS - 1e-6) {
        g_time_acc -= VIEWER_TICK_MS;
        if (g_time_acc < 0.0)
            g_time_acc = 0.0;
        if (g_tick >= g_total_ticks) {
            g_playing = 0;   // ended: JS sees playing==0 && tick==total
            g_time_acc = 0.0;
            break;
        }
        feed_and_step();
        stepped++;
        if (g_tick >= g_total_ticks) {
            g_playing = 0;
            g_time_acc = 0.0;
            break;
        }
    }
    if (g_tick >= g_total_ticks || stepped > 1) {
        // Ended (show the final state exactly), or several ticks in one
        // callback: interpolating the last tick interval is
        // meaningless — render at-tick.
        g_phase = VIEWER_FRAMES_PER_TICK;
    } else if (stepped == 1 || g_phase != VIEWER_FRAMES_PER_TICK) {
        // Fresh interpolation window (a tick just stepped) or mid-sweep:
        // g_time_acc is wall-time progress toward the next tick, which
        // is exactly the progress through the current [last, cur]
        // window; quantize it to the renderer's 12ths. From an at-tick
        // display with no new tick (else-branch not taken) the phase
        // holds at-tick — sweeping backwards would lurch.
        g_phase = (int)(g_time_acc * VIEWER_FRAMES_PER_TICK / VIEWER_TICK_MS);
        if (g_phase < 0)
            g_phase = 0;
        if (g_phase >= VIEWER_FRAMES_PER_TICK)
            g_phase = VIEWER_FRAMES_PER_TICK - 1;
    }
    return stepped;
}

// One 60Hz-equivalent frame (the pre-time-based API, kept stable for
// the node harness): 12 calls == one tick at 1x.
int viewer_advance_frame(void) {
    return viewer_advance(1000.0 / 60.0);
}

// Current interpolation phase (see g_phase). Exported for the node
// harness so the phase-lock behavior is testable without pixels.
int viewer_render_phase(void) { return g_phase; }

int viewer_tick(void) { return g_tick; }

int viewer_total_ticks(void) { return g_total_ticks; }

void viewer_set_speed(int speed) {
    if (speed >= 1 && speed <= 1024)
        g_speed = speed;
}

int viewer_get_speed(void) { return g_speed; }

void viewer_set_playing(int playing) {
    if (!g_loaded)
        return;
    if (playing && g_tick >= g_total_ticks)
        return;  // ended: JS must seek first (no silent loop)
    g_playing = playing ? 1 : 0;
    // g_time_acc is deliberately kept: it is always < VIEWER_TICK_MS
    // here (the advance loop reduces it), so no tick burst is possible,
    // and preserving it resumes the interpolation sweep exactly where
    // the pause froze it (a reset would lurch the display backwards).
}

int viewer_playing(void) { return g_playing; }

// Patch-0003 episode state, for end-of-replay display and the headless
// verification (winner must match the replay header's result).
int viewer_done(void) { return g_allocated ? env.done : 0; }

int viewer_winner(void) { return g_allocated ? env.winner : 0; }

// Final-state digest at the current tick (see sim/shim_common.h). Must
// equal the recording host's state_digest() at the same tick — the
// headless verification (tests/test_viewer.py) and replay certification
// rely on this.
unsigned int viewer_state_digest(void) {
    return g_allocated ? moba_state_digest(&env) : 0;
}


// -- the draft, the scorebug and the minimap ---------------------------------
//
// viewer_set_loadout: one hero's absolute post-clamp stat block, taken
// from the replay header's draft[].applied (JS parses the JSON). Marks
// the pid so every later sim_fresh() applies it. Must be called for all
// ten pids of a drafted replay BEFORE the first seek/play, and the
// resulting viewer_loadout_digest() must equal the header's
// loadout_digest — index.html warns on screen when it does not.
void viewer_set_loadout(int pid, float base_health, float base_mana,
                        float base_damage, int basic_attack_cd,
                        float move_speed, int hp_gain_per_level,
                        int mana_gain_per_level, int damage_gain_per_level) {
    if (pid < 0 || pid >= VIEWER_NUM_AGENTS)
        return;
    const float block[DERK_LOADOUT_FIELDS] = {
        base_health, base_mana, base_damage, (float)basic_attack_cd,
        move_speed, (float)hp_gain_per_level, (float)mana_gain_per_level,
        (float)damage_gain_per_level};
    for (int f = 0; f < DERK_LOADOUT_FIELDS; f++)
        g_loadout[pid][f] = block[f];
    g_loadout_set[pid] = 1;
    // A loadout pushed after the sim was built applies from the next
    // sim_fresh(); apply it here too so the currently displayed tick 0
    // is already the drafted hero.
    if (g_allocated && g_tick == 0)
        derk_apply_loadout(&env, pid, g_loadout[pid]);
}

unsigned int viewer_loadout_digest(void) {
    return derk_loadout_digest(g_loadout);
}

// Ancient health for the scorebug bars. team 0 = radiant (entity
// TOWER_OFFSET+23), 1 = dire (TOWER_OFFSET+22) — the same mapping and
// dead-guard as sim/shim.c ancient_health().
float viewer_ancient_health(int team) {
    if (!g_allocated)
        return 0.0f;
    int idx = (team == 0) ? TOWER_OFFSET + 23 : TOWER_OFFSET + 22;
    Entity* ancient = &env.entities[idx];
    if (ancient->pid == -1)
        return 0.0f;
    return ancient->health;
}

// Per-hero scoreboard stats. `which` codes are sim/shim.c agent_stat()'s
// (0 level, 1 kills, 2 deaths, 3 towers_killed, ...) so the viewer's
// scorebug and the server's agent_stats can be compared directly —
// tests/test_viewer.py asserts they agree at the same tick.
int viewer_agent_stat(int pid, int which) {
    if (!g_allocated || pid < 0 || pid >= NUM_PLAYERS)
        return 0;
    Entity* e = &env.entities[pid];
    PlayerLog* pl = &env.player_logs[pid];
    switch (which) {
        case 0:  return e->level;
        case 1:  return (int)pl->kills;
        case 2:  return (int)pl->deaths;
        case 3:  return (int)pl->towers_killed;
        case 4:  return (int)pl->creeps_killed;
        case 5:  return (int)pl->neutrals_killed;
        case 6:  return e->xp;
        case 7:  return (int)pl->damage_dealt;
        case 8:  return (int)pl->damage_received;
        case 9:  return (int)pl->healing_dealt;
        case 10: return (int)pl->healing_received;
        default: return 0;
    }
}

// Ten heroes x (x, y, team, alive) into a JS-provided float buffer (40
// floats) for the minimap. Returns the hero count written, or 0 when the
// sim is not built yet. kill_entity() sets pid = -1 on death, which is
// also how the caller draws a hollow dot.
int viewer_hero_positions(float* out) {
    if (!g_allocated || out == NULL)
        return 0;
    for (int pid = 0; pid < VIEWER_NUM_AGENTS; pid++) {
        Entity* e = &env.entities[pid];
        out[pid * 4 + 0] = e->x;
        out[pid * 4 + 1] = e->y;
        out[pid * 4 + 2] = (float)e->team;
        out[pid * 4 + 3] = (e->pid == -1) ? 0.0f : 1.0f;
    }
    return VIEWER_NUM_AGENTS;
}

// Camera follow target (#derk-viewpanel). The upstream renderer is a
// 41x23-cell camera view over a 128x128 map that follows
// renderer->human_player (moba.h c_render), so selecting a hero IS the
// zoom affordance this renderer supports. Stored even before the
// renderer exists (first frame) and re-applied on every set.
static int g_camera_pid = 1;  // upstream's own default (init_game_renderer)

void viewer_set_camera(int pid) {
    if (pid < 0 || pid >= VIEWER_NUM_AGENTS)
        return;
    g_camera_pid = pid;
#ifdef MOBA_RENDER
    if (env.client != NULL) {
        GameRenderer* renderer = env.client;
        renderer->human_player = pid;
    }
#endif
}

int viewer_camera(void) { return g_camera_pid; }

#ifdef MOBA_RENDER
// Render loop: one callback per browser animation frame. Sim cadence is
// handled by viewer_advance_frame; c_render (upstream, unchanged) lazily
// creates its GameRenderer/window on first call and interpolates entity
// positions between sim ticks.
static void frame(void) {
    if (!g_loaded)
        return;  // window/canvas appear on first frame after load
    // Advance by measured wall time (emscripten_get_now, ms): rAF fires
    // per display refresh, so callback-counting would play 2x too fast
    // on 120 Hz displays and hitch on dropped frames. The dt clamp in
    // viewer_advance absorbs tab-switch gaps; last_now keeps updating
    // even while paused so resume sees a normal dt.
    static double last_now = -1.0;
    double now = emscripten_get_now();
    double dt_ms = (last_now < 0.0) ? 0.0 : now - last_now;
    last_now = now;
    viewer_advance(dt_ms);
    if (env.client != NULL) {
        // Phase-lock upstream's interpolation counter to the true
        // inter-tick progress (see g_phase): c_render reads
        // renderer->frame for tick_frac and assumes 1 sim tick per 12
        // phase-aligned render calls, which seeks/speeds/120Hz displays
        // all violate. renderer->frame's other uses are harmless here:
        // the HUMAN_CONTROL action-clear is replay-irrelevant (every
        // tick's actions come from the log), and the end-of-render
        // increment/wrap is superseded by this per-frame overwrite.
        // VIEWER_FRAMES_PER_TICK (== upstream FRAMES) yields
        // tick_frac 1.0: render exactly at-tick.
        GameRenderer* renderer = env.client;
        renderer->frame = viewer_render_phase();
        renderer->human_player = g_camera_pid;
    }
    c_render(&env);
}

int main(void) {
    // 0 fps == requestAnimationFrame; don't simulate an infinite loop —
    // main returns and the runtime stays alive (EXIT_RUNTIME=0) for the
    // viewer_* exports.
    emscripten_set_main_loop(frame, 0, 0);
    return 0;
}
#endif  // MOBA_RENDER
