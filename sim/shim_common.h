// Shared MOBA env configuration used by every host shim (sim/shim.c and
// sim/viewer_main.c). One definition so the viewer's re-simulation can
// never drift from the server sim's init path.
//
// Values are the trained-on env defaults, copied verbatim from upstream
// config/moba.ini [env] (fed through ocean/moba/binding.c my_vec_init).
// Do not change: policies were trained on these.
#ifndef COGAME_SHIM_COMMON_H
#define COGAME_SHIM_COMMON_H

#include "moba.h"  // vendored sim; includes game_map.h (game_map_npy)

static inline void moba_configure(MOBA* env, unsigned int seed,
                                  int num_agents) {
    env->num_agents = num_agents;             // 10 (all heroes seat-controlled)
    env->script_opponents = (num_agents == 5); // binding.c: agents_per_env = script_opponents ? 5 : 10
    env->vision_range = 5;                    // moba.ini [env] vision_range
    env->agent_speed = 1.0f;                  // moba.ini [env] agent_speed
    env->reward_death = -0.163764f;           // moba.ini [env] reward_death
    env->reward_xp = 0.00665677f;             // moba.ini [env] reward_xp
    env->reward_distance = 0.0f;              // moba.ini [env] reward_distance
    env->reward_tower = 0.642119f;            // moba.ini [env] reward_tower
#ifndef PRISTINE
    env->seed = seed;                         // patch 0002: init_moba() -> srand(seed)
#else
    (void)seed;                               // pristine build: libc default stream (== seed 1)
#endif
}

// Cheap final-state digest for replay certification: FNV-1a (32-bit) over
// each hero's (x, y, health) float bits plus both ancient healths
// (radiant then dire, ancient_health() dead-guard semantics: 0.0f when
// pid == -1). Pure read of entity state — never touches the obs/reward
// path. Shared so the server host (sim/shim.c) and the viewer
// (sim/viewer_main.c) can never diverge; a recorded episode's live
// digest must equal the viewer's re-sim digest at the same tick.
static inline unsigned int moba_fnv1a_f32(unsigned int h, float v) {
    unsigned char b[sizeof(float)];
    memcpy(b, &v, sizeof(float));
    for (unsigned int i = 0; i < sizeof(float); i++) {
        h ^= b[i];
        h *= 16777619u;
    }
    return h;
}

static inline unsigned int moba_state_digest(const MOBA* env) {
    unsigned int h = 2166136261u;  // FNV-1a offset basis
    for (int pid = 0; pid < NUM_PLAYERS; pid++) {
        const Entity* e = &env->entities[pid];
        h = moba_fnv1a_f32(h, e->x);
        h = moba_fnv1a_f32(h, e->y);
        h = moba_fnv1a_f32(h, e->health);
    }
    // radiant ancient = TOWER_OFFSET+23, dire = TOWER_OFFSET+22 (matching
    // c_step's radiant_pid/dire_pid and shim.c ancient_health()).
    const int ancients[2] = {TOWER_OFFSET + 23, TOWER_OFFSET + 22};
    for (int i = 0; i < 2; i++) {
        const Entity* a = &env->entities[ancients[i]];
        h = moba_fnv1a_f32(h, (a->pid == -1) ? 0.0f : a->health);
    }
    return h;
}

#endif  // COGAME_SHIM_COMMON_H
