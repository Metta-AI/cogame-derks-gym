// Shared post-draft loadout application, used by BOTH host shims
// (sim/shim.c apply_loadout and sim/viewer_main.c viewer_set_loadout /
// sim_fresh). One definition so the viewer's re-simulation can never
// drift from the server sim's loadout path — the same reason
// sim/shim_common.h exists for the env config.
//
// A loadout is an ABSOLUTE, post-clamp stat block for one hero: the
// draft's deltas are summed and clamped host-side (Python:
// cogame_derks_gym/catalog.py), so this layer only writes fields and
// re-derives the level-dependent values. It is called AFTER moba_init
// (allocate_moba + c_reset) and BEFORE the first moba_step, once per
// pid, in ascending pid order.
//
// It must NOT re-spawn the hero: spawn_player() draws rand() (moba.h
// spawn position loop), which would desync the seeded stream and break
// both the fidelity guarantee and replay re-simulation. Instead it
// reproduces spawn_player's derivation in place — the derivation the
// sim itself re-runs on every respawn and every level-up
// (moba.h spawn_player, and the level-up path in attack()), which is
// what makes a loadout permanent for the whole match for free.
#ifndef COGAME_LOADOUT_COMMON_H
#define COGAME_LOADOUT_COMMON_H

#include "shim_common.h"  // moba_fnv1a_f32() (+ moba.h)

// Field order of a loadout block (and of the g_applied digest table):
//   0 base_health   1 base_mana         2 base_damage
//   3 basic_attack_cd (int ticks)       4 move_speed
//   5 hp_gain_per_level (int)           6 mana_gain_per_level (int)
//   7 damage_gain_per_level (int)
#define DERK_LOADOUT_FIELDS 8

// hero_stat() / viewer_agent_stat() `which` codes.
#define DERK_STAT_BASE_HEALTH 0
#define DERK_STAT_BASE_MANA 1
#define DERK_STAT_BASE_DAMAGE 2
#define DERK_STAT_BASIC_ATTACK_CD 3
#define DERK_STAT_MOVE_SPEED 4
#define DERK_STAT_HP_GAIN 5
#define DERK_STAT_MANA_GAIN 6
#define DERK_STAT_DAMAGE_GAIN 7
#define DERK_STAT_MAX_HEALTH 8
#define DERK_STAT_MAX_MANA 9
#define DERK_STAT_DAMAGE 10
#define DERK_STAT_HEALTH 11
#define DERK_STAT_MANA 12
#define DERK_STAT_LEVEL 13

// Write one hero's absolute stat block, in the order documented in
// docs/DRAFT.md. Returns 0 on success, -1 on a bad pid.
static inline int derk_apply_loadout(MOBA* env, int pid,
                                    const float block[DERK_LOADOUT_FIELDS]) {
    if (pid < 0 || pid >= NUM_PLAYERS)
        return -1;
    Entity* e = &env->entities[pid];

    // 1. the eight base fields
    e->base_health = block[0];
    e->base_mana = block[1];
    e->base_damage = block[2];
    e->basic_attack_cd = (int)block[3];
    e->move_speed = block[4];
    e->hp_gain_per_level = (int)block[5];
    e->mana_gain_per_level = (int)block[6];
    e->damage_gain_per_level = (int)block[7];

    // 2. spawn_player's derivation, in place (moba.h spawn_player)
    e->max_health = e->base_health + e->level * e->hp_gain_per_level;
    e->max_mana = e->base_mana + e->level * e->mana_gain_per_level;
    e->damage = e->base_damage + e->level * e->damage_gain_per_level;

    // 3. the hero is at full health at tick 0 anyway, so with a
    //    zero-delta (neutral) block this whole call is a byte-identical
    //    no-op — asserted by tests/test_loadout.py.
    e->health = e->max_health;
    e->mana = e->max_mana;
    return 0;
}

// Read one hero's stats back (test/viewer readout; see the DERK_STAT_*
// codes above).
static inline float derk_hero_stat(const MOBA* env, int pid, int which) {
    if (pid < 0 || pid >= NUM_PLAYERS)
        return 0.0f;
    const Entity* e = &env->entities[pid];
    switch (which) {
        case DERK_STAT_BASE_HEALTH: return e->base_health;
        case DERK_STAT_BASE_MANA: return e->base_mana;
        case DERK_STAT_BASE_DAMAGE: return e->base_damage;
        case DERK_STAT_BASIC_ATTACK_CD: return (float)e->basic_attack_cd;
        case DERK_STAT_MOVE_SPEED: return e->move_speed;
        case DERK_STAT_HP_GAIN: return (float)e->hp_gain_per_level;
        case DERK_STAT_MANA_GAIN: return (float)e->mana_gain_per_level;
        case DERK_STAT_DAMAGE_GAIN: return (float)e->damage_gain_per_level;
        case DERK_STAT_MAX_HEALTH: return e->max_health;
        case DERK_STAT_MAX_MANA: return e->max_mana;
        case DERK_STAT_DAMAGE: return e->damage;
        case DERK_STAT_HEALTH: return e->health;
        case DERK_STAT_MANA: return e->mana;
        case DERK_STAT_LEVEL: return (float)e->level;
        default: return 0.0f;
    }
}

// FNV-1a (32-bit) over the NUM_PLAYERS x 8 applied float32 table, in pid
// then field order. Recorded in the replay header as loadout_digest and
// re-derived by the viewer after it applies the same table: a mismatch
// means the viewer is re-simulating different heroes than the server did.
static inline unsigned int derk_loadout_digest(
        const float applied[NUM_PLAYERS][DERK_LOADOUT_FIELDS]) {
    unsigned int h = 2166136261u;  // FNV-1a offset basis
    for (int pid = 0; pid < NUM_PLAYERS; pid++)
        for (int f = 0; f < DERK_LOADOUT_FIELDS; f++)
            h = moba_fnv1a_f32(h, applied[pid][f]);
    return h;
}

#endif  // COGAME_LOADOUT_COMMON_H
