#!/usr/bin/env bash
# Build the MP-SfM baseline image. Run this once (and again after editing the Dockerfile, or
# after `docker pull mpsfm/mpsfm:latest` picks up a new upstream base image) — it does nothing
# else: no dataset paths, no running of the baseline.
#
# Usage:
#   baselines/mpsfm/build_container.sh
#
# Then: baselines/mpsfm/run_container.sh  (drops you into a shell in the image)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-inspection-baselines/mpsfm}"

# The base image (mpsfm/mpsfm:latest, pulled fresh below) is what actually carries
# Ceres/pyceres/the COLMAP fork/pycolmap -- see the Dockerfile's own header for why this baseline
# builds on it instead of replaying that build from source itself. `docker build` alone won't
# re-pull an already-cached `:latest` tag even if upstream has moved on, so pull explicitly first.
docker pull mpsfm/mpsfm:latest

# Build context is the repo root: the Dockerfile copies the mpsfm submodule's requirements.txt in.
docker build -f "${REPO_ROOT}/baselines/mpsfm/Dockerfile" -t "$IMAGE" "$REPO_ROOT"

echo
echo "built $IMAGE — next: ${REPO_ROOT}/baselines/mpsfm/run_container.sh"
