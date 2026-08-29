#!/usr/bin/env bash
# Build the pixel-perfect-sfm baseline image. Run this once (and again after editing the
# Dockerfile or bumping the pixel-perfect-sfm submodule) — it does nothing else: no dataset paths,
# no running of the baseline. Note COLMAP itself is NOT one of this repo's submodules for this
# baseline: the Dockerfile clones COLMAP fresh from upstream at git tag 3.8 (pinned exactly to what
# pycolmap/pyceres require — see the Dockerfile's own header for why this repo's much newer
# `colmap` submodule doesn't work here), so there's nothing to bump on the COLMAP side.
#
# Before running this for the first time, make sure pixel-perfect-sfm's own nested submodules are
# checked out (its Dockerfile COPYs them in as plain files, so an uninitialized submodule copies
# as an empty directory and breaks its cmake build):
#   git -C pixel-perfect-sfm submodule update --init --recursive
#
# Usage:
#   baselines/pixel-perfect-sfm/build_container.sh
#
# This is a from-source build of Eigen 3.4.0, Ceres 2.1.0, COLMAP 3.8, then pycolmap, pyceres, AND
# pixsfm's own C++ extension, one after another in a single stage on Ubuntu 20.04 / CUDA 11.1.1 —
# see the Dockerfile's own header for the full story of why this combination, its provenance (a
# real GitHub issue's own community-debugged recipe), and what it replaced (an earlier Ubuntu
# 22.04 / CUDA 11.8 build, preserved as the `inspection-baselines/pixel-perfect-sfm-legacy` image
# tag / `Dockerfile.legacy-ubuntu22-cuda11.8.bak` in this directory, which built and ran fine but
# hit an unresolved intermittent native crash this build doesn't reproduce). Expect this to take
# considerably longer than baselines/colmap's own build.
#
# CMAKE_CUDA_ARCHITECTURES is hardcoded in the Dockerfile to `89` (this repo's actual GPUs, 2x RTX
# 6000 Ada Generation) rather than exposed as a build-arg here — COLMAP 3.8's own CMakeLists.txt
# requires this be set explicitly at all (no silent default), and generalizing it to other GPUs is
# a deliberate, not-yet-done follow-up (see the Dockerfile's own "CUDA architecture" section) — if
# that follow-up re-adds an ARG for it, this script's own `--build-arg` passthrough should come
# back too.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-inspection-baselines/pixel-perfect-sfm}"

# Build context is the repo root: the Dockerfile COPYs in the pixel-perfect-sfm submodule and
# this repo's own top-level requirements.txt (COLMAP/Ceres/Eigen are all `git clone`d inside the
# Dockerfile, not copied from context — see this file's own header).
docker build -f "${REPO_ROOT}/baselines/pixel-perfect-sfm/Dockerfile" \
    -t "$IMAGE" "$REPO_ROOT"

echo
echo "built $IMAGE — try it with, e.g.:"
echo "  docker run -it --rm --gpus all -v \"\$PWD\":/working -w /working $IMAGE bash"
