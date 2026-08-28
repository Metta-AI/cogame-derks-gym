"use strict";
/* cogame-derks-gym additions to the inherited cogame-moba chrome.
 *
 * The starter's page and its inline transport script (fetch -> parse
 * header -> viewer_load -> play/pause/speed/seek) are untouched; they
 * call four additive hooks defined here:
 *
 *   derkOnLoad(header, totalTicks)  once, after viewer_load succeeded
 *   derkOnFrame(tick, ended)        every rAF, from refreshUi()
 *   derkDismissEndcard()            from every seek handler
 *   derkSetError(message)           from every failure path
 *
 * Everything drawn here is DOM chrome, never canvas text: the upstream
 * raylib renderer's in-canvas DrawText labels are 20px in a 1312px frame
 * and would shrink to 5.5px at a 360px viewport. All readouts are
 * therefore HTML, scaled by --hudscale (floored at 0.8 so nothing drops
 * below 10px).
 *
 * Every string that came from a player (names, notes) is written with
 * textContent, never innerHTML.
 */

const DERK_TEAM_NAMES = ["radiant", "dire"];
const DERK_ANCIENT_HEALTH = 4500;
const DERK_TICKS_PER_SECOND = 5;      // upstream demo cadence
const DERK_DRAFT_AUTOCLOSE_TICK = 30; // ~6 s of playback at 1x
const DERK_FEED_LINES = 6;
const DERK_BEAT_MIN_PX = 12;
const DERK_STAT_LEVEL = 0, DERK_STAT_KILLS = 1, DERK_STAT_DEATHS = 2,
      DERK_STAT_TOWERS = 3;

let derkHeader = null;
let derkTotal = 0;
let derkRecords = [];            // ten draft records, pid order
let derkEvents = [];             // header events (tick order)
let derkBeats = [];              // {tick, kind, label}
let derkDraftDismissed = false;
let derkEndcardDismissed = false;
let derkLoadedSignalled = false;
let derkCameraAuto = true;
let derkCameraPid = 0;
let derkPositions = 0;           // wasm heap pointer for viewer_hero_positions
let derkRosterRows = new Map();  // pid -> {level, kd, ...} elements

function derkEl(id) { return document.getElementById(id); }

function derkCall(name, ret, types, args) {
  return Module.ccall(name, ret, types || [], args || []);
}

function derkText(tag, cls, text) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text !== undefined) el.textContent = text;
  return el;
}

/* The item glyphs live in derk_items.svg (one hand-authored 24x24
 * <symbol> per catalog id). Chromium does not resolve EXTERNAL <use>
 * references, so the sheet is fetched once and injected into the page,
 * and the <use> elements point at the injected symbols by fragment. The
 * fetch starts at script load; derkRefreshGlyphs() re-touches every href
 * once it lands, so glyph rendering never depends on the race.
 */
const derkSpritesReady = (async () => {
  const resp = await fetch("derk_items.svg");
  if (!resp.ok) throw new Error("derk_items.svg: HTTP " + resp.status);
  const doc = new DOMParser().parseFromString(
    await resp.text(), "image/svg+xml");
  const sheet = document.importNode(doc.documentElement, true);
  sheet.setAttribute("aria-hidden", "true");
  sheet.setAttribute("id", "derk-sprites");
  sheet.style.display = "none";
  document.body.appendChild(sheet);
})().catch((e) => console.error(e));

function derkRefreshGlyphs() {
  for (const use of document.querySelectorAll(".derk-glyph use")) {
    use.setAttribute("href", use.getAttribute("href"));
  }
}

function derkGlyph(itemId) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "derk-glyph");
  svg.setAttribute("viewBox", "0 0 24 24");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", "#item-" + itemId);
  svg.appendChild(use);
  return svg;
}

function derkItemName(slot, id) {
  const cat = (derkHeader && derkHeader.catalog) || {};
  const entry = (cat[slot] || []).find((e) => e && e.id === id);
  return entry ? entry.name : id;
}

function derkPickDeltas(picks) {
  // The catalog travels in the replay header, so the exact deltas render
  // without contacting the repo.
  const cat = (derkHeader && derkHeader.catalog) || {};
  const totals = {};
  for (const slot of ["arm", "tail", "misc"]) {
    const entry = (cat[slot] || []).find((e) => e && e.id === picks[slot]);
    const deltas = (entry && entry.deltas) || {};
    for (const [field, value] of Object.entries(deltas)) {
      totals[field] = (totals[field] || 0) + value;
    }
  }
  return totals;
}

function derkDeltaSpans(parent, picks) {
  const totals = derkPickDeltas(picks);
  const fields = Object.keys(totals).sort();
  if (!fields.length) {
    parent.appendChild(derkText("span", "derk-deltas", "no deltas"));
    return;
  }
  for (const field of fields) {
    const value = totals[field];
    const sign = value > 0 ? "+" : "\u2212";
    parent.appendChild(derkText(
      "span", "derk-deltas " + (value > 0 ? "derk-up" : "derk-down"),
      field + " " + sign + Math.abs(value) + " "));
  }
}

/* -- loadouts: push the drafted stat blocks into the sim ------------- */

function derkPushLoadouts() {
  const drafted = !derkHeader.config || derkHeader.config.draft_enabled !== false;
  if (!drafted) {
    // Un-drafted (Puffer-fidelity) replay: the server applied nothing at
    // all, so the viewer must apply nothing either or the digests differ.
    return;
  }
  const types = ["number", "number", "number", "number", "number",
                 "number", "number", "number", "number"];
  for (const rec of derkRecords) {
    const a = rec.applied || {};
    derkCall("viewer_set_loadout", null, types, [
      rec.pid, a.base_health, a.base_mana, a.base_damage,
      a.basic_attack_cd, a.move_speed, a.hp_gain_per_level,
      a.mana_gain_per_level, a.damage_gain_per_level]);
  }
  // Re-simulate from tick 0 so the blocks are applied through the exact
  // sim_fresh() path every later seek uses.
  derkCall("viewer_seek", null, ["number"], [0]);

  const mine = derkCall("viewer_loadout_digest", "number") >>> 0;
  const recorded = (derkHeader.loadout_digest || 0) >>> 0;
  if (mine !== recorded) {
    // Same element and pattern as the starter's sim-sha mismatch warning.
    const warn = derkEl("warn");
    if (warn) {
      warn.appendChild(document.createElement("br"));
      warn.appendChild(document.createTextNode(
        "warning: loadout digest mismatch (replay " + recorded +
        " vs viewer " + mine + "); the drafted stats may differ"));
    }
  }
}

/* -- the draft-reveal screen ---------------------------------------- */

function derkDraftCard(rec) {
  const card = derkText("div", "derk-card derk-" + rec.team);
  const head = derkText("div", "derk-head");
  const glyphs = derkText("span", "derk-glyphs");
  for (const slot of ["arm", "tail", "misc"]) {
    glyphs.appendChild(derkGlyph(rec.picks[slot]));
  }
  head.appendChild(glyphs);
  head.appendChild(derkText("span", "derk-alias", rec.alias));
  head.appendChild(derkText("span", "derk-name", rec.player_name || "house"));
  head.appendChild(derkText("span", "derk-role", rec.role));
  card.appendChild(head);
  card.appendChild(derkText("div", "derk-items", [
    derkItemName("arm", rec.picks.arm),
    derkItemName("tail", rec.picks.tail),
    derkItemName("misc", rec.picks.misc)].join(" \u00b7 ")));
  const stats = derkText("div", "derk-stats");
  derkDeltaSpans(stats, rec.picks);
  card.appendChild(stats);
  const a = rec.applied || {};
  card.appendChild(derkText("div", "derk-stats",
    "hp " + a.base_health + " \u00b7 mana " + a.base_mana +
    " \u00b7 dmg " + a.base_damage + " \u00b7 cd " + a.basic_attack_cd +
    " \u00b7 spd " + a.move_speed +
    " \u00b7 /lvl " + a.hp_gain_per_level + "/" +
    a.mana_gain_per_level + "/" + a.damage_gain_per_level));
  if (rec.note) card.appendChild(derkText("div", "derk-note", rec.note));
  if (rec.fallback) {
    card.appendChild(derkText("div", "derk-fallback",
      "neutral loadout (" + rec.fallback_cause + ")"));
  }
  return card;
}

function derkBuildDraft() {
  const cols = derkEl("derk-draft-cols");
  cols.textContent = "";
  for (const team of DERK_TEAM_NAMES) {
    const col = derkText("div", "derk-col");
    col.appendChild(derkText("h3", null, team === "radiant" ? "Radiant" : "Dire"));
    for (const rec of derkRecords) {
      if (rec.team !== team || rec.source !== "seat") continue;
      col.appendChild(derkDraftCard(rec));
    }
    cols.appendChild(col);
  }
  derkEl("derk-draft-close").addEventListener("click", derkHideDraft);
  derkEl("derk-draft").addEventListener("click", derkHideDraft);
  derkEl("derk-draft-inner").addEventListener("click", (e) => {
    e.stopPropagation();  // a click on a card must not close the screen
  });
  // A labelled re-open button, appended to the starter's #controls.
  const button = derkText("button", null, "draft");
  button.id = "derk-draft-open";
  button.type = "button";
  button.addEventListener("click", () => {
    derkDraftDismissed = false;
    derkShowDraft();
  });
  derkEl("controls").appendChild(button);
  derkShowDraft();
}

function derkShowDraft() {
  derkEl("derk-draft").hidden = false;
  derkRelayout();
}

function derkHideDraft() {
  derkDraftDismissed = true;
  derkEl("derk-draft").hidden = true;
  derkRelayout();
}

/* -- roster --------------------------------------------------------- */

function derkBuildRoster() {
  const roster = derkEl("derk-roster");
  roster.textContent = "";
  derkRosterRows = new Map();
  const ordered = derkRecords.slice().sort((a, b) => {
    if (a.team !== b.team) return a.team === "radiant" ? -1 : 1;
    if ((a.source === "house") !== (b.source === "house")) {
      return a.source === "house" ? 1 : -1;
    }
    return a.pid - b.pid;
  });
  for (const rec of ordered) {
    const row = derkText("div",
      "derk-row derk-" + rec.team + (rec.source === "house" ? " derk-house" : ""));
    const glyphs = derkText("span", "derk-glyphs");
    for (const slot of ["arm", "tail", "misc"]) {
      glyphs.appendChild(derkGlyph(rec.picks[slot]));
    }
    row.appendChild(glyphs);
    row.appendChild(derkText("span", "derk-alias", rec.alias));
    row.appendChild(derkText("span", "derk-name",
      rec.source === "house" ? "house" : (rec.player_name || "")));
    row.appendChild(derkText("span", "derk-role", rec.role));
    const level = derkText("span", "derk-kd", "lvl -");
    const kd = derkText("span", "derk-kd", "0/0");
    row.appendChild(level);
    row.appendChild(kd);
    const deltas = derkText("span", "derk-deltas-wrap");
    derkDeltaSpans(deltas, rec.picks);
    row.appendChild(deltas);
    roster.appendChild(row);
    derkRosterRows.set(rec.pid, {level, kd});
  }
}

/* -- camera select + minimap ---------------------------------------- */

function derkBuildCameras() {
  const box = derkEl("derk-cameras");
  box.textContent = "";
  const make = (label, onclick) => {
    const b = derkText("button", null, label);
    b.type = "button";
    b.addEventListener("click", onclick);
    box.appendChild(b);
    return b;
  };
  for (const rec of derkRecords) {
    if (rec.source !== "seat") continue;
    const b = make(rec.alias, () => derkSelectCamera(rec.pid, false));
    b.dataset.pid = String(rec.pid);
  }
  const auto = make("auto", () => derkSelectCamera(derkCameraPid, true));
  auto.id = "derk-camera-auto";
  derkSelectCamera(derkRecords.length ? derkRecords[0].pid : 0, true);
}

function derkSelectCamera(pid, auto) {
  derkCameraAuto = !!auto;
  derkCameraPid = pid;
  derkCall("viewer_set_camera", null, ["number"], [pid]);
  for (const b of derkEl("derk-cameras").querySelectorAll("button")) {
    const isPid = b.dataset.pid !== undefined &&
      Number(b.dataset.pid) === pid && !derkCameraAuto;
    b.setAttribute("aria-pressed", String(
      b.id === "derk-camera-auto" ? derkCameraAuto : isPid));
  }
}

function derkDrawMinimap() {
  const canvas = derkEl("derk-minimap");
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  if (!derkPositions) derkPositions = Module._malloc(10 * 4 * 4);
  const count = derkCall("viewer_hero_positions", "number", ["number"],
                         [derkPositions]);
  ctx.clearRect(0, 0, 128, 128);
  ctx.fillStyle = "#06100f";
  ctx.fillRect(0, 0, 128, 128);
  if (!count) return;
  // HEAPU8 is the only exported heap view; read the floats through a
  // DataView (the buffer is re-created on memory growth, so never cache).
  const view = new DataView(Module.HEAPU8.buffer);
  const seatPids = (derkHeader && derkHeader.seat_hero_pids) || [];
  for (let i = 0; i < count; i++) {
    const base = derkPositions + i * 16;
    const x = view.getFloat32(base, true);
    const y = view.getFloat32(base + 4, true);
    const team = view.getFloat32(base + 8, true);
    const alive = view.getFloat32(base + 12, true);
    const colour = team === 0 ? "#4cf" : "#f66";
    const size = seatPids.indexOf(i) === -1 ? 2 : 3;
    if (alive) {
      ctx.fillStyle = colour;
      ctx.fillRect(x - size / 2, y - size / 2, size, size);
    } else {
      ctx.strokeStyle = colour;
      ctx.lineWidth = 1;
      ctx.strokeRect(x - size / 2, y - size / 2, size, size);
    }
    if (i === derkCameraPid) {
      // The camera's 41x23-cell window over the 128x128 board.
      ctx.strokeStyle = "#6db";
      ctx.lineWidth = 1;
      ctx.strokeRect(x - 20.5, y - 11.5, 41, 23);
    }
  }
}

function derkMinimapClick(event) {
  const canvas = derkEl("derk-minimap");
  const rect = canvas.getBoundingClientRect();
  const mx = (event.clientX - rect.left) * 128 / rect.width;
  const my = (event.clientY - rect.top) * 128 / rect.height;
  if (!derkPositions) return;
  const count = derkCall("viewer_hero_positions", "number", ["number"],
                         [derkPositions]);
  const view = new DataView(Module.HEAPU8.buffer);
  let best = null, bestD = 1e9;
  for (let i = 0; i < count; i++) {
    const base = derkPositions + i * 16;
    const dx = view.getFloat32(base, true) - mx;
    const dy = view.getFloat32(base + 4, true) - my;
    const d = dx * dx + dy * dy;
    if (d < bestD) { bestD = d; best = i; }
  }
  if (best !== null && bestD <= 100) derkSelectCamera(best, false);
}

/* -- scrubber beats -------------------------------------------------- */

const DERK_BEAT_LABELS = {
  draft: "draft", first_blood: "first", kill: "kill", tower: "tower",
  level_spike: "lvl", ancient: "ancient", end: "end",
};

function derkBuildBeats() {
  derkBeats = derkEvents.map((e) => ({
    tick: e.tick || 0,
    kind: e.kind,
    label: DERK_BEAT_LABELS[e.kind] || e.kind,
  }));
  derkEl("derk-beats").addEventListener("click", (e) => e.stopPropagation());
  derkLayoutBeats();
}

function derkSeekTo(tick) {
  const seek = derkEl("seek");
  seek.value = String(tick);
  // Reuse the starter's own seek path (its `change` handler re-simulates).
  seek.dispatchEvent(new Event("change"));
}

function derkLayoutBeats() {
  const box = derkEl("derk-beats");
  const seek = derkEl("seek");
  if (!seek || !derkTotal) { box.classList.remove("derk-on"); return; }
  const rect = seek.getBoundingClientRect();
  if (rect.width < 20) { box.classList.remove("derk-on"); return; }
  box.style.left = rect.left + "px";
  box.style.top = (rect.bottom + 1) + "px";
  box.style.width = rect.width + "px";
  box.style.height = "16px";
  box.classList.add("derk-on");
  box.textContent = "";

  // Collapse beats that land within DERK_BEAT_MIN_PX of each other into a
  // shared "+n" chip, so a 360px scrubber stays clickable.
  const groups = [];
  for (const beat of derkBeats) {
    const px = rect.width * (beat.tick / Math.max(1, derkTotal));
    const last = groups[groups.length - 1];
    if (last && px - last.px < DERK_BEAT_MIN_PX) {
      last.beats.push(beat);
    } else {
      groups.push({px, beats: [beat]});
    }
  }
  for (const group of groups) {
    const first = group.beats[0];
    const many = group.beats.length > 1;
    const button = derkText("button", null,
      many ? "+" + group.beats.length : first.label);
    button.type = "button";
    button.className = many ? "beat beat-more" : "beat beat-" + first.kind;
    button.style.left = group.px + "px";
    button.setAttribute("aria-label", many
      ? group.beats.length + " events at tick " + first.tick
      : first.kind + " at tick " + first.tick);
    button.addEventListener("click", () => derkSeekTo(first.tick));
    box.appendChild(button);
  }
}

/* -- endcard -------------------------------------------------------- */

function derkShowEndcard() {
  const card = derkEl("derk-endcard");
  if (card.hidden === false) return;
  const inner = derkEl("derk-endcard-inner");
  inner.textContent = "";
  const result = (derkHeader && derkHeader.result) || {};
  const winner = result.winner === 0 ? "Radiant wins"
    : result.winner === 1 ? "Dire wins" : "Draw";
  inner.appendChild(derkText("div", "derk-winner", winner));
  inner.appendChild(derkText("div", null,
    "end_reason " + (result.end_reason || "?") +
    " \u00b7 tick " + (result.final_tick !== undefined
      ? result.final_tick : derkTotal)));
  const healths = result.ancient_healths || [];
  inner.appendChild(derkText("div", null,
    "ancients \u00b7 radiant " + Math.round(healths[0] || 0) +
    " \u00b7 dire " + Math.round(healths[1] || 0)));
  const cards = derkText("div", "derk-cards");
  const team = result.winner === 1 ? "dire" : "radiant";
  for (const rec of derkRecords) {
    if (rec.source !== "seat" || rec.team !== team) continue;
    cards.appendChild(derkDraftCard(rec));
  }
  if (result.winner !== null && result.winner !== undefined) {
    inner.appendChild(derkText("div", null, "winning loadouts"));
    inner.appendChild(cards);
  }
  const close = derkText("button", "derk-close", "close");
  close.type = "button";
  close.addEventListener("click", derkDismissEndcard);
  inner.appendChild(close);
  card.hidden = false;
  derkRelayout();
}

function derkDismissEndcard() {
  derkEndcardDismissed = true;
  const card = derkEl("derk-endcard");
  if (card) card.hidden = true;
  const draft = derkEl("derk-draft");
  if (draft && !draft.hidden) derkHideDraft();
}

/* -- transport rules: --band and --hudscale ------------------------- */

function derkRelayout() {
  const controls = derkEl("controls");
  const box = controls ? controls.getBoundingClientRect() : null;
  const height = box ? box.height : 40;
  // --band is the space a fixed overlay must leave free at the BOTTOM of
  // the viewport so that the transport is never covered. On a desktop
  // page the controls sit at the bottom of the viewport and this is just
  // their height + 8 (48px on one row, 84px on two). On a narrow, tall
  // page they sit mid-viewport, so the band grows to reach them —
  // otherwise `inset: 0 0 var(--band) 0` would lie over the scrubber.
  // Capped at 70% of the viewport so an overlay never collapses to
  // nothing when the page is scrolled far down.
  let band = Math.round(height + 8);
  if (box) {
    band = Math.max(band, Math.round(window.innerHeight - box.top + 8));
    band = Math.min(band, Math.round(window.innerHeight * 0.7));
  }
  const root = document.documentElement;
  root.style.setProperty("--band", band + "px");
  const scale = Math.min(1, Math.max(0.8, window.innerWidth / 1100));
  root.style.setProperty("--hudscale", String(scale));
  derkLayoutBeats();
}

window.addEventListener("load", derkRelayout);
window.addEventListener("resize", derkRelayout);
// The transport moves under a fixed overlay when the page scrolls, so the
// band has to be re-measured (passive: never blocks scrolling).
window.addEventListener("scroll", derkRelayout, {passive: true});

/* -- per-frame readouts --------------------------------------------- */

function derkClockText(tick) {
  const seconds = Math.floor(tick / DERK_TICKS_PER_SECOND);
  const mm = Math.floor(seconds / 60);
  const ss = String(seconds % 60).padStart(2, "0");
  return "tick " + tick + " / " + derkTotal + " \u00b7 " + mm + ":" + ss;
}

function derkUpdateScorebug() {
  for (let team = 0; team < 2; team++) {
    const name = DERK_TEAM_NAMES[team];
    const health = derkCall("viewer_ancient_health", "number", ["number"],
                            [team]);
    derkEl("derk-hp-" + name).textContent = String(Math.round(health));
    derkEl("derk-bar-" + name).style.width =
      Math.max(0, Math.min(100, health / DERK_ANCIENT_HEALTH * 100)) + "%";
    let towers = 0, kills = 0;
    for (let pid = team * 5; pid < team * 5 + 5; pid++) {
      towers += derkCall("viewer_agent_stat", "number",
                         ["number", "number"], [pid, DERK_STAT_TOWERS]);
      kills += derkCall("viewer_agent_stat", "number",
                        ["number", "number"], [pid, DERK_STAT_KILLS]);
    }
    derkEl("derk-towers-" + name).textContent = towers + " towers";
    derkEl("derk-kills-" + name).textContent = kills + " kills";
  }
}

function derkUpdateRoster() {
  for (const [pid, cells] of derkRosterRows) {
    const level = derkCall("viewer_agent_stat", "number",
                           ["number", "number"], [pid, DERK_STAT_LEVEL]);
    const kills = derkCall("viewer_agent_stat", "number",
                           ["number", "number"], [pid, DERK_STAT_KILLS]);
    const deaths = derkCall("viewer_agent_stat", "number",
                            ["number", "number"], [pid, DERK_STAT_DEATHS]);
    cells.level.textContent = "lvl " + level;
    cells.kd.textContent = kills + "/" + deaths;
  }
}

function derkEventLine(e) {
  const who = e.pid !== undefined ? derkAliasFor(e.pid) : "";
  switch (e.kind) {
    case "draft": return "tick 0 \u00b7 draft resolved";
    case "first_blood":
      return "tick " + e.tick + " \u00b7 first blood: " + who +
        " killed " + derkAliasFor(e.victim_pid);
    case "kill":
      return "tick " + e.tick + " \u00b7 " + who + " killed " +
        derkAliasFor(e.victim_pid);
    case "tower":
      return "tick " + e.tick + " \u00b7 " + who + " took a tower";
    case "level_spike":
      return "tick " + e.tick + " \u00b7 " + who + " reached level " + e.level;
    case "ancient":
      return "tick " + e.tick + " \u00b7 " +
        DERK_TEAM_NAMES[e.team] + " ancient fell";
    case "end":
      return "tick " + e.tick + " \u00b7 end (" + e.reason + ")";
    default: return "tick " + e.tick + " \u00b7 " + e.kind;
  }
}

function derkAliasFor(pid) {
  const rec = derkRecords.find((r) => r.pid === pid);
  return rec ? rec.alias : ("pid " + pid);
}

function derkUpdateFeed(tick) {
  const feed = derkEl("derk-feed");
  const lines = derkEvents.filter((e) => (e.tick || 0) <= tick)
    .slice(-DERK_FEED_LINES);
  feed.textContent = "";
  for (const e of lines) {
    feed.appendChild(derkText("div", "derk-k-" + e.kind, derkEventLine(e)));
  }
  if (derkCameraAuto) {
    for (let i = lines.length - 1; i >= 0; i--) {
      if (lines[i].pid !== undefined) {
        if (lines[i].pid !== derkCameraPid) {
          derkSelectCamera(lines[i].pid, true);
        }
        break;
      }
    }
  }
}

function derkSignalLoaded() {
  if (derkLoadedSignalled) return;
  const canvas = Module && Module.canvas;
  if (!canvas || !canvas.width) return;
  derkLoadedSignalled = true;
  document.documentElement.dataset.replayLoaded = "true";
}

function derkSetError(message) {
  document.documentElement.dataset.replayError = String(message || "error");
}

/* -- the hooks ------------------------------------------------------ */

function derkOnLoad(header, totalTicks) {
  derkHeader = header;
  derkTotal = totalTicks;
  derkRecords = (header.draft || []).slice()
    .sort((a, b) => a.pid - b.pid);
  derkEvents = (header.events || []).slice()
    .sort((a, b) => (a.tick || 0) - (b.tick || 0));
  derkPushLoadouts();
  derkBuildRoster();
  derkBuildCameras();
  derkBuildBeats();
  derkBuildDraft();
  derkEl("derk-minimap").addEventListener("click", derkMinimapClick);
  derkSpritesReady.then(derkRefreshGlyphs);
  derkRelayout();
}

function derkOnFrame(tick, ended) {
  if (!derkHeader) return;
  derkSignalLoaded();
  derkEl("derk-clock").textContent = derkClockText(tick);
  derkUpdateScorebug();
  derkUpdateRoster();
  derkUpdateFeed(tick);
  derkDrawMinimap();
  if (!derkDraftDismissed) {
    if (tick >= DERK_DRAFT_AUTOCLOSE_TICK) {
      derkHideDraft();
    } else {
      derkEl("derk-draft-count").textContent =
        "(closes in " + Math.max(0, Math.ceil(
          (DERK_DRAFT_AUTOCLOSE_TICK - tick) / DERK_TICKS_PER_SECOND)) + "s)";
    }
  }
  if (ended && !derkEndcardDismissed) {
    derkShowEndcard();
  } else if (!ended) {
    // Playing again after a seek: the endcard may reappear at the end.
    const card = derkEl("derk-endcard");
    if (card && !card.hidden) card.hidden = true;
    derkEndcardDismissed = false;
  }
}
