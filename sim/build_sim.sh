#!/usr/bin/env bash
# Build the sim wasm binaries:
#   build/derk_sim.wasm           patched sim (production)
#   build/derk_sim_pristine.wasm  0001-only sim (fidelity-test reference)
#
# Flags:
#   STANDALONE_WASM + --no-entry : WASI reactor module for wasmtime hosting
#   ALLOW_MEMORY_GROWTH, MAXIMUM_MEMORY=1gb : the sim's ai_paths BFS cache is
#     a 256 MB calloc; grow up to 1 GB
#   ABORTING_MALLOC=1 : fail loudly on OOM instead of NULL-write corruption
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v emcc >/dev/null 2>&1; then
    echo "error: emcc not found on PATH - install emscripten (brew install emscripten)" >&2
    exit 1
fi

./sim/apply_patches.sh

COMMON_FLAGS=(-O2 -sSTANDALONE_WASM --no-entry
              -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=1gb -sABORTING_MALLOC=1)

emcc "${COMMON_FLAGS[@]}" -I build/src-patched  sim/shim.c -o build/derk_sim.wasm
emcc "${COMMON_FLAGS[@]}" -DPRISTINE -I build/src-pristine sim/shim.c -o build/derk_sim_pristine.wasm

ls -la build/*.wasm
echo "build_sim: OK"
