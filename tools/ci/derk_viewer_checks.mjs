#!/usr/bin/env node
// cogame-derks-gym chrome checks: the game-specific half of the wasm-viewer job.
//
// tools/ci/viewer_smoke.mjs is copied VERBATIM from the coworld-builder
// templates and is deliberately game-agnostic (it proves the bundle boots,
// draws, and keeps advancing). This script is its sibling and asserts the
// things only this game's appended chrome can be wrong about — the design
// note's wasm-viewer list:
//
//   1. open index.html?replay=... at 1280x800
//   2. wait <= timeout for <html data-replay-loaded="true"> and assert
//      data-replay-error is absent
//   3. #derk-draft holds six alias rows x three item names, and each row's
//      real player name differs from its alias
//   4. #derk-clock reads `tick 0 / <total> - 0:00` and the scorebug shows
//      4500 for both Ancients. Asserted after an explicit pause + seek to
//      tick 0 rather than "at load": the starter's transport AUTOPLAYS at 5
//      ticks/s, so a bare "at load" read is a race against the first frame.
//      Same facts, deterministically.
//   5. clicking the last #derk-beats button changes #tickinfo and leaves
//      #derk-endcard out of the transport band (bounding-box check against
//      the #controls box)
//   6. at 360x640 the computed font size of #derk-scorebug is >= 10px and no
//      element overlaps the transport band
//   7. worst-case model text: the same bundle loaded against a replay whose
//      six seat records each carry a FULL-CAP (120-rune), unbreakable note --
//      the draft overlay must still lay out inside the transport band, must
//      not clip sideways, and must still hold the full-length strings
//   8. screenshots; exits non-zero on any failure
//
// usage: node tools/ci/derk_viewer_checks.mjs --bundle <dir> --replay <file>
//                                             [--timeout 90] [--out .]
"use strict";

import { createServer } from "node:http";
import { createReadStream, existsSync, readFileSync, statSync,
         writeFileSync } from "node:fs";
import { basename, extname, join, resolve, sep } from "node:path";
import process from "node:process";

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".wasm": "application/wasm",
  ".data": "application/octet-stream",
  ".replay": "application/octet-stream",
  ".png": "image/png",
};

function die(code, message) {
  console.error(message);
  process.exit(code);
}

function parseArgs(argv) {
  const out = { timeout: 90, outDir: process.cwd() };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      const value = argv[i + 1];
      if (value === undefined) die(2, `missing value for ${arg}`);
      i += 1;
      return value;
    };
    switch (arg) {
      case "--bundle": out.bundle = resolve(next()); break;
      case "--replay": out.replay = resolve(next()); break;
      case "--timeout": out.timeout = Number(next()); break;
      case "--out": out.outDir = resolve(next()); break;
      default: die(2, `unknown argument: ${arg}`);
    }
  }
  if (!out.bundle || !out.replay) die(2, "--bundle and --replay are required");
  return out;
}

const args = parseArgs(process.argv.slice(2));

function serve(bundleDir, replayPaths) {
  const root = resolve(bundleDir);
  const replays = new Map(replayPaths.map((p) => [`/${basename(p)}`, p]));
  const server = createServer((req, res) => {
    let pathname;
    try {
      pathname = decodeURIComponent(new URL(req.url, "http://127.0.0.1").pathname);
    } catch {
      res.writeHead(400).end("bad url");
      return;
    }
    if (pathname === "/") pathname = "/index.html";
    const mapped = replays.get(pathname);
    const target = mapped || resolve(join(root, pathname));
    if (!mapped && !(target === root || target.startsWith(root + sep))) {
      res.writeHead(403).end("forbidden");
      return;
    }
    if (!existsSync(target) || !statSync(target).isFile()) {
      res.writeHead(404).end("not found");
      return;
    }
    res.writeHead(200, {
      "content-type": MIME[extname(target).toLowerCase()] || "application/octet-stream",
      "content-length": String(statSync(target).size),
      "cache-control": "no-store",
    });
    createReadStream(target).pipe(res);
  });
  return new Promise((ok) => {
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      ok({
        server,
        urlFor: (p) => `http://127.0.0.1:${port}/index.html?replay=` +
          encodeURIComponent(`http://127.0.0.1:${port}/${basename(p)}`),
      });
    });
  });
}

// The only model-authored string that reaches this viewer is a seat's draft
// `note`, capped server-side at 120 Unicode scalars (draft.truncate_note).
// CI replays carry no model text at all -- every smoke seat is scripted and
// puffer-forge emits no note -- so the worst case has to be built here, from
// the real recorded replay, and then loaded through the real bundle.
// Unbroken Ws: no wrap opportunity anywhere, the widest ASCII glyph.
const WORST_NOTE = "W".repeat(120);

function writeWorstCaseNoteReplay(srcPath, outPath) {
  const buf = readFileSync(srcPath);
  if (buf.length < 9 || buf.toString("latin1", 0, 4) !== "DERK" || buf[4] !== 2) {
    throw new Error(`not a DERK v2 replay: ${srcPath}`);
  }
  const headerLen = buf.readUInt32LE(5);
  const header = JSON.parse(buf.toString("utf8", 9, 9 + headerLen));
  const body = buf.subarray(9 + headerLen);
  let seats = 0;
  for (const rec of header.draft || []) {
    if (rec.source === "seat") { rec.note = WORST_NOTE; seats += 1; }
  }
  if (seats !== 6) throw new Error(`expected 6 seat draft records, got ${seats}`);
  const headerBytes = Buffer.from(JSON.stringify(header), "utf8");
  const prefix = Buffer.alloc(9);
  prefix.write("DERK", 0, "latin1");
  prefix[4] = 2;
  prefix.writeUInt32LE(headerBytes.length, 5);
  writeFileSync(outPath, Buffer.concat([prefix, headerBytes, body]));
  return outPath;
}

async function loadChromium() {
  for (const candidate of [process.env.PLAYWRIGHT_MODULE, "playwright",
                           "playwright-core"].filter(Boolean)) {
    try {
      const mod = await import(candidate);
      if (mod.chromium) return mod.chromium;
      if (mod.default && mod.default.chromium) return mod.default.chromium;
    } catch { /* try the next candidate */ }
  }
  die(2, "could not load Playwright (npm install --no-save playwright@1.55.0)");
}

const failures = [];
function check(ok, message) {
  if (!ok) failures.push(message);
  console.log(`${ok ? "ok  " : "FAIL"} ${message}`);
}

async function waitLoaded(page, timeoutMs) {
  const started = Date.now();
  for (;;) {
    const state = await page.evaluate(() => ({
      loaded: document.documentElement.dataset.replayLoaded || "",
      error: document.documentElement.dataset.replayError || "",
      status: (document.getElementById("status") || {}).textContent || "",
    }));
    if (state.error) throw new Error(`data-replay-error: ${state.error}`);
    if (state.loaded === "true") return Date.now() - started;
    if (Date.now() - started > timeoutMs) {
      throw new Error(
        `no data-replay-loaded within ${timeoutMs} ms (status: ${state.status})`);
    }
    await page.waitForTimeout(250);
  }
}

const summary = {};

async function main() {
  const chromium = await loadChromium();
  const worstPath = writeWorstCaseNoteReplay(
    args.replay, join(args.outDir, "worst-case-notes.replay"));
  const { server, urlFor } = await serve(args.bundle, [args.replay, worstPath]);
  const url = urlFor(args.replay);
  const browser = await chromium.launch({
    headless: true,
    args: ["--enable-unsafe-swiftshader", "--use-gl=swiftshader",
           "--no-sandbox"],
  });
  try {
    // -- 1280x800 -----------------------------------------------------
    let page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
    const console_lines = [];
    page.on("console", (m) => console_lines.push(`${m.type()}: ${m.text()}`));
    page.on("pageerror", (e) => console_lines.push(`pageerror: ${e.message}`));
    await page.goto(url, { waitUntil: "domcontentloaded" });
    summary.load_ms = await waitLoaded(page, args.timeout * 1000);
    check(true, `loaded in ${summary.load_ms} ms at 1280x800`);

    // Pause and park on tick 0: the inherited transport autoplays, so the
    // tick-0 readouts below would otherwise race the first frames.
    await page.evaluate(() => {
      const play = document.getElementById("playpause");
      if (play && play.textContent === "pause") play.click();
      const seek = document.getElementById("seek");
      seek.value = "0";
      seek.dispatchEvent(new Event("change"));
    });
    await page.waitForTimeout(1200);
    // The draft screen is dismissed by every seek (by design), so re-open
    // it with its labelled #controls button for the checks below.
    await page.evaluate(() => {
      const open = document.getElementById("derk-draft-open");
      if (open) open.click();
    });
    await page.waitForTimeout(300);

    // -- 3. the draft-reveal screen -----------------------------------
    const draft = await page.evaluate(() => {
      const cards = [...document.querySelectorAll("#derk-draft .derk-card")];
      return {
        visible: !document.getElementById("derk-draft").hidden,
        rows: cards.map((c) => ({
          alias: (c.querySelector(".derk-alias") || {}).textContent || "",
          name: (c.querySelector(".derk-name") || {}).textContent || "",
          items: ((c.querySelector(".derk-items") || {}).textContent || "")
            .split("\u00b7").map((s) => s.trim()).filter(Boolean),
          glyphs: c.querySelectorAll(".derk-glyph use").length,
        })),
      };
    });
    summary.draft = draft;
    check(draft.visible, "#derk-draft is shown at load");
    check(draft.rows.length === 6,
      `#derk-draft holds 6 seat cards (got ${draft.rows.length})`);
    check(draft.rows.every((r) => r.items.length === 3),
      "every draft card names three items");
    check(draft.rows.every((r) => r.glyphs === 3),
      "every draft card carries three item glyphs");
    check(draft.rows.every((r) => r.alias && r.name && r.alias !== r.name),
      "every draft card's real player name differs from its alias");

    // -- 4. clock + scorebug ------------------------------------------
    const readouts = await page.evaluate(() => ({
      clock: (document.getElementById("derk-clock") || {}).textContent || "",
      radiant: (document.getElementById("derk-hp-radiant") || {}).textContent || "",
      dire: (document.getElementById("derk-hp-dire") || {}).textContent || "",
      total: Number((document.getElementById("seek") || {}).max || 0),
    }));
    summary.readouts = readouts;
    check(readouts.clock === `tick 0 / ${readouts.total} \u00b7 0:00`,
      `#derk-clock reads "tick 0 / ${readouts.total} \u00b7 0:00" ` +
      `(got "${readouts.clock}")`);
    check(readouts.radiant === "4500" && readouts.dire === "4500",
      `both Ancients read 4500 (got ${readouts.radiant}/${readouts.dire})`);

    // The draft screen must stop above the transport band while it is
    // actually visible.
    const draftBand = await page.evaluate(() => {
      const controls = document.getElementById("controls").getBoundingClientRect();
      const el = document.getElementById("derk-draft");
      return {
        hidden: el.hidden,
        bottom: el.getBoundingClientRect().bottom,
        controlsTop: controls.top,
      };
    });
    summary.draft_band = draftBand;
    check(!draftBand.hidden && draftBand.bottom <= draftBand.controlsTop + 1,
      "#derk-draft stops above the transport band while shown " +
      `(${draftBand.bottom} <= ${draftBand.controlsTop})`);

    // -- 5. the scrubber beats seek ------------------------------------
    await page.evaluate(() => {
      const b = document.getElementById("derk-draft-close");
      if (b) b.click();
    });
    const beats = await page.evaluate(
      () => [...document.querySelectorAll("#derk-beats button")].map((b) => ({
        label: b.textContent, aria: b.getAttribute("aria-label"),
        cls: b.className,
      })));
    summary.beats = beats;
    check(beats.length > 0, `#derk-beats rendered ${beats.length} buttons`);
    check(beats.every((b) => b.label && b.aria),
      "every beat is a labelled button with an aria-label");
    const before = await page.textContent("#tickinfo");
    if (beats.length > 0) {
      await page.evaluate(() => {
        const all = document.querySelectorAll("#derk-beats button");
        all[all.length - 1].click();
      });
      await page.waitForTimeout(2000);
      const after = await page.textContent("#tickinfo");
      summary.tickinfo = { before, after };
      check(after !== before,
        `clicking the last beat changed #tickinfo ("${before}" -> "${after}")`);
      const dismissed = await page.evaluate(
        () => document.getElementById("derk-endcard").hidden);
      check(dismissed, "the seek dismissed #derk-endcard");
    }

    // -- the endcard, played to naturally, still stops above the band ---
    await page.evaluate(() => {
      const seek = document.getElementById("seek");
      seek.value = "0";
      seek.dispatchEvent(new Event("change"));
    });
    await page.waitForTimeout(1500);
    await page.evaluate(() => {
      const speed = document.getElementById("speed");
      speed.value = "64";
      speed.dispatchEvent(new Event("change"));
      const play = document.getElementById("playpause");
      if (play.textContent !== "pause") play.click();
    });
    let endBand = null;
    for (let i = 0; i < 80; i++) {
      await page.waitForTimeout(500);
      endBand = await page.evaluate(() => {
        const controls = document.getElementById("controls")
          .getBoundingClientRect();
        const el = document.getElementById("derk-endcard");
        return {
          hidden: el.hidden,
          bottom: el.getBoundingClientRect().bottom,
          controlsTop: controls.top,
          tickinfo: document.getElementById("tickinfo").textContent,
        };
      });
      if (!endBand.hidden) break;
    }
    summary.end_band = endBand;
    check(endBand && !endBand.hidden,
      `#derk-endcard appears when playback reaches the end ` +
      `(${endBand && endBand.tickinfo})`);
    if (endBand && !endBand.hidden) {
      check(endBand.bottom <= endBand.controlsTop + 1,
        "#derk-endcard stops above the transport band " +
        `(${endBand.bottom} <= ${endBand.controlsTop})`);
    }
    await page.screenshot({ path: join(args.outDir, "derk-viewer-1280.png") });
    await page.close();

    // -- 6. 360x640 ---------------------------------------------------
    page = await browser.newPage({ viewport: { width: 360, height: 640 } });
    page.on("pageerror", (e) => console_lines.push(`pageerror(360): ${e.message}`));
    await page.goto(url, { waitUntil: "domcontentloaded" });
    await waitLoaded(page, args.timeout * 1000);
    const narrow = await page.evaluate(() => {
      const px = (el) => parseFloat(getComputedStyle(el).fontSize);
      const controls = document.getElementById("controls").getBoundingClientRect();
      const overlapping = [];
      for (const el of document.querySelectorAll(
             "#derk > *:not([hidden]), #derk-scorebug, #derk-roster, #derk-feed")) {
        if (el.id === "derk-beats") continue;  // the beats ARE the scrubber
        const box = el.getBoundingClientRect();
        if (box.height === 0) continue;
        if (box.bottom > controls.top && box.top < controls.bottom) {
          overlapping.push(el.id || el.className);
        }
      }
      return {
        scorebug: px(document.getElementById("derk-scorebug")),
        feed: px(document.getElementById("derk-feed")),
        band: getComputedStyle(document.documentElement)
          .getPropertyValue("--band").trim(),
        overlapping,
      };
    });
    summary.narrow = narrow;
    check(narrow.scorebug >= 10,
      `#derk-scorebug font-size at 360px is ${narrow.scorebug}px (>= 10)`);
    check(narrow.feed >= 10,
      `#derk-feed font-size at 360px is ${narrow.feed}px (>= 10)`);
    check(narrow.overlapping.length === 0,
      `no chrome element overlaps the transport band at 360px ` +
      `(${JSON.stringify(narrow.overlapping)})`);
    await page.screenshot({ path: join(args.outDir, "derk-viewer-360.png") });
    await page.close();

    // -- 7. worst case: a full-cap note on EVERY seat at once ---------
    // This viewer draws NO model text on the canvas (viewer_smoke reports
    // canvas_text total 0 because the renderer is WebGL/raylib and every
    // readout is DOM), so the checklist's worst-case renderer fixture is a
    // DOM fixture here: the same bundle, the same real replay, with all six
    // notes replaced by the 120-rune cap. Scrolling is fine (the overlay is
    // overflow:auto by design); clipping and a covered transport are not,
    // and a note that arrived shortened would mean the fixture tested
    // nothing.
    page = await browser.newPage({ viewport: { width: 360, height: 640 } });
    page.on("pageerror", (e) => console_lines.push(`pageerror(worst): ${e.message}`));
    await page.goto(urlFor(worstPath), { waitUntil: "domcontentloaded" });
    await waitLoaded(page, args.timeout * 1000);
    await page.evaluate(() => {
      const play = document.getElementById("playpause");
      if (play && play.textContent === "pause") play.click();
      const open = document.getElementById("derk-draft-open");
      if (open) open.click();
    });
    await page.waitForTimeout(500);
    const worst = await page.evaluate((cap) => {
      const overlay = document.getElementById("derk-draft");
      const controls = document.getElementById("controls").getBoundingClientRect();
      const box = overlay.getBoundingClientRect();
      const notes = [...overlay.querySelectorAll(".derk-card .derk-note")];
      return {
        hidden: overlay.hidden,
        notes: notes.length,
        fullLength: notes.filter((n) => n.textContent.length === cap).length,
        clipped: notes.filter((n) => {
          const nb = n.getBoundingClientRect();
          const cb = n.parentElement.getBoundingClientRect();
          return nb.height < 1 || nb.right > cb.right + 1 || nb.left < cb.left - 1;
        }).length,
        overflowX: overlay.scrollWidth - overlay.clientWidth,
        scrollable: overlay.scrollHeight > overlay.clientHeight,
        bottom: box.bottom,
        controlsTop: controls.top,
      };
    }, WORST_NOTE.length);
    summary.worst_case_notes = worst;
    check(!worst.hidden && worst.notes === 6,
      `worst case: 6 full-cap notes rendered in #derk-draft (got ${worst.notes})`);
    check(worst.fullLength === worst.notes,
      `worst case: every note is still ${WORST_NOTE.length} runes long ` +
      `(${worst.fullLength}/${worst.notes})`);
    check(worst.clipped === 0,
      `worst case: no note is clipped by its card (${worst.clipped} clipped)`);
    check(worst.overflowX <= 1,
      `worst case: #derk-draft does not overflow sideways ` +
      `(scrollWidth - clientWidth = ${worst.overflowX}px; vertical scroll ` +
      `is fine: scrollable=${worst.scrollable})`);
    check(worst.bottom <= worst.controlsTop + 1,
      `worst case: #derk-draft still stops above the transport band ` +
      `(${worst.bottom} <= ${worst.controlsTop})`);
    await page.screenshot({
      path: join(args.outDir, "derk-viewer-worst-notes-360.png") });

    summary.console_tail = console_lines.slice(-30);
  } finally {
    await browser.close();
    server.close();
  }
}

main().then(() => {
  summary.ok = failures.length === 0;
  summary.failures = failures;
  writeFileSync(join(args.outDir, "derk-viewer-checks.json"),
                JSON.stringify(summary, null, 2));
  if (failures.length) {
    console.error(`\n${failures.length} chrome check(s) failed:`);
    for (const f of failures) console.error(`  - ${f}`);
    process.exit(1);
  }
  console.log("\nall derks-gym chrome checks passed");
}).catch((error) => {
  summary.ok = false;
  summary.error = String((error && error.stack) || error);
  writeFileSync(join(args.outDir, "derk-viewer-checks.json"),
                JSON.stringify(summary, null, 2));
  console.error(summary.error);
  process.exit(1);
});
