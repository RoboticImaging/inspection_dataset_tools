#!/usr/bin/env bash
# Build the vggt-omega baseline image. Run this once (and again after editing the
# Dockerfile or bumping the vggt-omega submodule) — it does nothing else: no dataset
# paths, no running of the baseline.
#
# Usage:
#   baselines/vggt-omega/build_container.sh
#
# Then: baselines/vggt-omega/run_container.sh  (drops you into a shell in the image)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-inspection-baselines/vggt-omega}"

# Build context is the repo root: the Dockerfile copies the vggt-omega submodule in.
docker build -f "${REPO_ROOT}/baselines/vggt-omega/Dockerfile" -t "$IMAGE" "$REPO_ROOT"
