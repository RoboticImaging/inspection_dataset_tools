#!/usr/bin/env bash
# Build the glomap baseline image. Run this once (and again after editing the Dockerfile or
# bumping the colmap submodule) — it does nothing else: no dataset paths, no running of the
# baseline.
#
# Usage:
#   baselines/glomap/build_container.sh
#
# Then: baselines/glomap/run_container.sh  (drops you into a shell in the image)
#
# CUDA_ARCHITECTURES defaults to "all-major" (colmap/docker/Dockerfile's own default, which asks
# nvcc to target every major architecture it knows about). Override it if you know your GPU's
# compute capability and want a faster build / smaller binary — see the submodule's own
# docker/README.md, "Build from Scratch":
#   CUDA_ARCHITECTURES=75 baselines/glomap/build_container.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-inspection-baselines/glomap}"

BUILD_ARGS=()
if [ -n "${CUDA_ARCHITECTURES:-}" ]; then
    BUILD_ARGS+=( --build-arg "CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" )
fi

# Build context is the repo root: the Dockerfile copies the colmap submodule in (glomap = colmap's
# own `global_mapper` command, same binary — see the Dockerfile's own header).
docker build "${BUILD_ARGS[@]}" -f "${REPO_ROOT}/baselines/glomap/Dockerfile" -t "$IMAGE" "$REPO_ROOT"

echo
echo "built $IMAGE — next: ${REPO_ROOT}/baselines/glomap/run_container.sh"
