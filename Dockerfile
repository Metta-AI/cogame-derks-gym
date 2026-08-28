# cogame-derks-gym Coworld image: game server + bundled players in one image.
#
# Stage 1 (wasm-builder) compiles the vendored PufferLib moba sim with
# emscripten: sim wasm (wasmtime host), brain wasm (baseline policy), and
# the browser replay-viewer bundle. Wasm artifacts are architecture-
# independent, so this stage runs on the build host's native platform
# ($BUILDPLATFORM) — no qemu emulation for the compile on ARM hosts.
#
# Stage 2 is the linux/amd64 runtime: python:3.11-slim + locked deps via
# uv, with the repo layout preserved at /workspace (server code resolves
# build/*.wasm and viewer/dist relative to the repo root, so the project
# is NOT pip-installed into site-packages).
#
# Entrypoints (Coworld manifest `run`):
#   game    python -m cogame_derks_gym.server   (also /bin/derks-gym)
#   player  python -m players.derk_player       (also /bin/derks-gym-player)
#           env-switched: PLAYER_PROMPT=<prompt> | PLAYER_SCRIPTED=<name>
# The /bin/<slug> wrappers exist because that is the entrypoint convention
# tools/ci/docker_smoke.sh and tools/ci/policies.json use.
#
# Build: docker build --platform=linux/amd64 -t cogame-derks-gym:local .

# Pin matches the emcc used for local builds (vendor/UPSTREAM.md: 6.0.5).
FROM --platform=$BUILDPLATFORM emscripten/emsdk:6.0.5 AS wasm-builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl libdigest-sha-perl patch unzip xxd && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Prefetch the raylib web build into the exact cache location
# sim/build_viewer.sh expects, as its own layer so source edits never
# re-download. URL/sha MUST stay in sync with sim/build_viewer.sh; if
# they drift, build_viewer.sh just re-fetches (correct, slower).
ARG RAYLIB_ZIP_URL="https://github.com/raysan5/raylib/releases/download/5.5/raylib-5.5_webassembly.zip"
ARG RAYLIB_ZIP_SHA256="798b6bea650e78a60fe49f106a15d92ea4e33efd3aa1b3efa34b0438a14bbf2c"
RUN mkdir -p build && \
    curl -fsSL --retry 3 "$RAYLIB_ZIP_URL" -o /tmp/raylib-web.zip && \
    echo "$RAYLIB_ZIP_SHA256  /tmp/raylib-web.zip" | shasum -a 256 -c - && \
    (cd build && unzip -q /tmp/raylib-web.zip && \
     mv raylib-5.5_webassembly raylib-web && \
     printf '%s\n' "$RAYLIB_ZIP_SHA256" > raylib-web/.zip-sha256) && \
    rm /tmp/raylib-web.zip

COPY vendor/ vendor/
COPY sim/ sim/
COPY viewer/index.html viewer/derk_chrome.css viewer/derk_chrome.js \
     viewer/derk_items.svg viewer/

# apply_patches.sh runs inside build_sim.sh and build_viewer.sh (and is
# idempotent); build_brain.sh compiles the pristine vendor tree directly
# (puffernet + weights need no patches).
RUN bash sim/build_sim.sh && \
    bash sim/build_brain.sh && \
    bash sim/build_viewer.sh


FROM python:3.11-slim

WORKDIR /workspace

# Locked runtime deps only (aiohttp/numpy/wasmtime): the project itself
# stays at /workspace via PYTHONPATH so repo-root-relative wasm/viewer
# paths keep working. uv is bind-mounted from its distribution image for
# this RUN only — a COPY'd binary would persist in its layer even after
# a later `rm`, so it never becomes a layer at all.
COPY pyproject.toml uv.lock ./
RUN --mount=from=ghcr.io/astral-sh/uv:0.9.18,source=/uv,target=/usr/local/bin/uv \
    uv sync --frozen --no-dev --no-install-project

ENV PATH="/workspace/.venv/bin:$PATH" \
    PYTHONPATH="/workspace/server:/workspace" \
    PYTHONUNBUFFERED=1

COPY server/ server/
COPY players/ players/

# /bin/<slug> and /bin/<slug>-player: the entrypoint names the CI smoke
# and the uploaded policy versions use (tools/ci/docker_smoke.sh,
# tools/ci/policies.json). Both exec the same modules the manifest `run`
# commands name, so there is exactly one code path.
RUN printf '#!/bin/sh\nexec python -m cogame_derks_gym.server "$@"\n' \
        > /bin/derks-gym && \
    printf '#!/bin/sh\nexec python -m players.derk_player "$@"\n' \
        > /bin/derks-gym-player && \
    chmod +x /bin/derks-gym /bin/derks-gym-player
COPY --from=wasm-builder /src/build/derk_sim.wasm /src/build/derk_brain.wasm build/
COPY --from=wasm-builder /src/viewer/dist/ viewer/dist/

CMD ["python", "-m", "cogame_derks_gym.server"]
