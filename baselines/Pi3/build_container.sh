#!/usr/bin/env bash
# Build the Pi3 baseline image. Run this once (and again after editing the
# Dockerfile or bumping the Pi3 submodule) — it does nothing else: no dataset
# paths, no running of the baseline.
#
# Usage:
#   baselines/Pi3/build_container.sh
#
# Then: baselines/Pi3/run_container.sh  (drops you into a shell in the image)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-inspection-baselines/pi3}"

# Build context is the repo root: the Dockerfile copies the Pi3 submodule in.
docker build -f "${REPO_ROOT}/baselines/Pi3/Dockerfile" -t "$IMAGE" "$REPO_ROOT"
