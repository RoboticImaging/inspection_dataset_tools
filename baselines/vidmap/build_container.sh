#!/usr/bin/env bash
# Build the vidmap baseline image. Run this once (and again after editing the Dockerfile or
# bumping the colmap/vidmap submodules) — it does nothing else: no dataset paths, no running of
# the baseline.
#
# Before running this for the first time, make sure vidmap's own nested submodules are checked
# out (its Dockerfile COPYs them in as plain files, so an uninitialized submodule copies as an
# empty directory and breaks its cmake/pip build — all three are mandatory, not optional extras,
# see the Dockerfile's own header):
#   git -C vidmap submodule update --init --recursive
#
# Usage:
#   baselines/vidmap/build_container.sh
#
# CUDA_ARCHITECTURES defaults to "all-major" (colmap/docker/Dockerfile's own default, which asks
# nvcc to target every major architecture it knows about). Override it if you know your GPU's
# compute capability and want a faster build / smaller binary — see the colmap submodule's own
# docker/README.md, "Build from Scratch":
#   CUDA_ARCHITECTURES=89 baselines/vidmap/build_container.sh
#
# This is a from-source build of COLMAP, then vidmap's own C++ extension on top of pinned
# torch/xformers, one after another in a single stage (see the Dockerfile's own header for why no
# builder/runtime split) — expect this to take considerably longer than baselines/colmap's own
# build. This Dockerfile has not yet been build-tested: unlike baselines/pixel-perfect-sfm (which
# needed real build output to find and fix four separate COLMAP-version compatibility risks
# before it would compile at all), vidmap's own version target (COLMAP 4.1.0) is much closer to
# this repo's own colmap submodule and only trips a non-fatal CMake warning on a mismatch rather
# than refusing to build — but treat the first attempt here the same way: diagnostic, not final.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-inspection-baselines/vidmap}"

BUILD_ARGS=()
if [ -n "${CUDA_ARCHITECTURES:-}" ]; then
    BUILD_ARGS+=( --build-arg "CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" )
fi

# Build context is the repo root: the Dockerfile copies both the colmap and vidmap submodules in.
docker build "${BUILD_ARGS[@]}" -f "${REPO_ROOT}/baselines/vidmap/Dockerfile" \
    -t "$IMAGE" "$REPO_ROOT"

echo
echo "built $IMAGE — try it with, e.g.:"
echo "  docker run -it --rm --gpus all -v \"\$PWD\":/working -w /working $IMAGE bash"
