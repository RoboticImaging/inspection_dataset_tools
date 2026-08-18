#!/usr/bin/env bash
# Build the vggt baseline image. Run this once (and again after editing the
# Dockerfile or bumping the vggt submodule) — it does nothing else: no dataset
# paths, no running of the baseline.
#
# Usage:
#   baselines/vggt/build_container.sh
#
# Then: baselines/vggt/run_container.sh  (drops you into a shell in the image)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-inspection-baselines/vggt}"

# Build context is the repo root: the Dockerfile copies the vggt submodule in.
docker build -f "${REPO_ROOT}/baselines/vggt/Dockerfile" -t "$IMAGE" "$REPO_ROOT"
