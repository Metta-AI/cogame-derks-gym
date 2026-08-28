# Vendored upstream: PufferLib Ocean MOBA

- **Repo:** https://github.com/PufferAI/PufferLib
- **Commit:** `c5d3c637446047a6efbcaa74c039c5295d201ab0` (branch 4.0)
- **License:** MIT (see `vendor/LICENSE-pufferlib`)
- **Fetched:** 2026-08-01

## Rule: files under `vendor/upstream/` are byte-pristine

Never edit files in `vendor/upstream/`. All modifications are patch files in
`sim/patches/`, applied at build time into `build/` by `sim/apply_patches.sh`.
Pristineness is verifiable by diffing against the pinned upstream commit and
by the sha256 sums below.

## Files

| vendored file | upstream path | sha256 |
|---|---|---|
| `moba.h` | `ocean/moba/moba.h` | `3e705457337416c4390cb32ac5813e944ba7222010ad7a7070cc7af13a111c02` |
| `game_map.h` | `ocean/moba/game_map.h` | `e4cc1b91bc630a85120a3528b7a9ce60689eb3d40a187b7757a5bea9f16ae629` |
| `binding.c` | `ocean/moba/binding.c` | `7e795954b689fa7bcf37bcb8496384b68d2f4eb042e3a3887daf6327384bdbe0` |
| `moba.c` | `ocean/moba/moba.c` | `90ced112e4d381f19416e1b80e2179fa4955000cd1704190d9ae430a53292cce` |
| `puffernet.h` | `src/puffernet.h` | `f7f53ca1a1d1a56190bc8c73a099d5ac356013da3e4abdb0050342e33b88405b` |
| `moba_weights.bin` | `resources/moba/moba_weights.bin` | `394b19fa8e2894879e05f9bcdff1da78ae2e83ec161e16ca8ad4a5811674e896` |
| `moba.ini` | `config/moba.ini` | `38596e3630fe4758f784917c7ea27fcfbc5e0895973da6aceb465a1e530a5dfb` |
| `resources/moba/moba_assets.png` | `resources/moba/moba_assets.png` | `f03f8d627c5675bbe1c3186eae50555aa3e0f0fa07db1ed26761dfa67a6739bd` |
| `resources/moba/dota_map.png` | `resources/moba/dota_map.png` | `8075a144349c1f780f93b6fc313acfa9983c6b958ada1792c055ab1e030aec90` |
| `resources/moba/map_shader_100.fs` | `resources/moba/map_shader_100.fs` | `921351ca7d18af605a858e6631f70b579cea0db6623c477c59a3691e500b3652` |
| `resources/moba/map_shader_330.fs` | `resources/moba/map_shader_330.fs` | `cff087df5940f19e0271fa71a5aae8fc4b8c3ae9ae459b00728a1f72414d782d` |
| `resources/moba/bloom_shader_100.fs` | `resources/moba/bloom_shader_100.fs` | `8e75da8a71735ca8868c303c34557bbd016de4b4ef3a14b1a16d866838e2e6c9` |
| `resources/moba/bloom_shader_330.fs` | `resources/moba/bloom_shader_330.fs` | `3fdfb628f7b2da89025b82afe5e1573206d427d238cac4bbe3d2c637cea91d41` |

The `resources/moba/` files are the render assets the viewer build preloads
(`sim/build_viewer.sh`). The renderer (`moba.h` `init_game_renderer`) opens
exactly `dota_map.png`, `moba_assets.png`, `map_shader_<GLSL>.fs` and
`bloom_shader_<GLSL>.fs`; both GLSL 100 (web) and 330 (desktop) shader
variants are vendored. No fonts or audio: `DrawText`/`DrawFPS` use raylib's
built-in font, and the renderer loads no sound. `resources/shared/` is not
referenced by the moba renderer.

`vendor/LICENSE-pufferlib` is the upstream repo `LICENSE` file.

## Build toolchain

- emscripten 6.0.5, in two flavors:
  - **Docker/CI (authoritative for released artifacts):** the
    `emscripten/emsdk:6.0.5` image (Dockerfile wasm-builder stage) and the
    same 6.0.5 pin in `.github/workflows/ci.yml` — every shipped wasm is
    built with this toolchain.
  - Local dev: Homebrew `emcc 6.0.5-git`. Close enough for development and
    the fidelity gate, but it is a moving git build; when local and CI
    binaries differ, the Docker one is the reference.
  The wasm binaries are build outputs (gitignored), not committed.

### Build-time dependency: raylib 5.5 (web, prebuilt)

Not vendored into git; fetched by `sim/build_viewer.sh` into
`build/raylib-web/` (cached) exactly as upstream `build.sh --web` does:

- URL: https://github.com/raysan5/raylib/releases/download/5.5/raylib-5.5_webassembly.zip
- Upstream pins this same artifact (`build.sh`: `RAYLIB_URL=".../5.5"`,
  `RAYLIB_NAME='raylib-5.5_webassembly'`) — a prebuilt emscripten static
  library, no source build.
- zip sha256: recorded/verified by `sim/build_viewer.sh` (`RAYLIB_ZIP_SHA256`).
