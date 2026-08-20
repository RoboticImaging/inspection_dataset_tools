#!/usr/bin/env bash
# Build the pi-long baseline image. Run this once (and again after editing the
# Dockerfile or bumping the Pi-Long submodule) — it does nothing else: no dataset
# paths, no running of the baseline.
#
# Usage:
#   baselines/Pi-Long/build_container.sh
#
# Then: baselines/Pi-Long/run_container.sh  (drops you into a shell in the image)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-inspection-baselines/pi-long}"

# Build context is the repo root: the Dockerfile copies the Pi-Long submodule in.
docker build -f "${REPO_ROOT}/baselines/Pi-Long/Dockerfile" -t "$IMAGE" "$REPO_ROOT"
