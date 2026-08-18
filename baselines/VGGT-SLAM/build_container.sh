#!/usr/bin/env bash
# Build the VGGT-SLAM baseline image. Run this once (and again after editing the
# Dockerfile or bumping the VGGT-SLAM submodule) — it does nothing else: no dataset
# paths, no running of the baseline.
#
# Usage:
#   baselines/VGGT-SLAM/build_container.sh
#
# Then: baselines/VGGT-SLAM/run_container.sh  (drops you into a shell in the image)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-inspection-baselines/vggt-slam}"

# Build context is the repo root: the Dockerfile copies the VGGT-SLAM submodule in.
docker build -f "${REPO_ROOT}/baselines/VGGT-SLAM/Dockerfile" -t "$IMAGE" "$REPO_ROOT"
