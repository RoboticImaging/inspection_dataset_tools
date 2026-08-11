#!/usr/bin/env bash
# Build the Depth Anything 3 baseline image. Run this once (and again after editing the
# Dockerfile or bumping the depth-anything-3 submodule) — it does nothing else: no dataset
# paths, no running of the baseline.
#
# Usage:
#   baselines/depth_anything_3/build_container.sh
#
# Then: baselines/depth_anything_3/run_container.sh  (drops you into a shell in the image)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-inspection-baselines/da3}"

# Build context is the repo root: the Dockerfile copies the depth-anything-3 submodule in.
docker build -f "${REPO_ROOT}/baselines/depth_anything_3/Dockerfile" -t "$IMAGE" "$REPO_ROOT"

echo
echo "built $IMAGE — next: ${REPO_ROOT}/baselines/depth_anything_3/run_container.sh"
