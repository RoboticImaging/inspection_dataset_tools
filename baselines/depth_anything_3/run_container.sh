#!/usr/bin/env bash
# Run the Depth Anything 3 baseline on a sequence, inside its Docker image.
#
# Usage:
#   baselines/depth_anything_3/run_container.sh baselines/configs/da3_glass_house.yaml
#
# Everything the baseline needs (torch, xformers, depth-anything-3, inspection_viz's light
# deps) lives in the Docker image built from ../Dockerfile — this script only orchestrates
# `docker build`/`docker run` from the host and never touches the project's venv or any
# host Python install; do not run run_sequence.py outside the container.
#
# The repo is bind-mounted at /workspace so edits to run_sequence.py/inspection_viz take effect
# without a rebuild; the dataset root named in the config is bind-mounted read-write at the same
# path it has on the host, so config yaml paths need no translation. Downloaded HF weights are
# cached in .cache/huggingface/ at the repo root so repeat runs don't re-download.
set -euo pipefail

CONFIG="${1:?usage: run_container.sh <config.yaml>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE=inspection-baselines/da3

# Pull `dataset: { root: ... }` out of the yaml with awk (not python/PyYAML) so this script has
# no interpreter dependency of its own. Only handles our own config's simple `key: value` shape.
DATASET_ROOT="$(awk '
    /^dataset:/            { in_ds=1; next }
    in_ds && /^[^[:space:]]/ { in_ds=0 }
    in_ds && /^[[:space:]]*root:/ {
        sub(/^[[:space:]]*root:[[:space:]]*/, "")
        sub(/[[:space:]]*#.*/, "")
        gsub(/^[\"'"'"']|[\"'"'"']$/, "")
        print
        exit
    }
' "$CONFIG")"
[ -n "$DATASET_ROOT" ] || { echo "no dataset.root found in $CONFIG" >&2; exit 1; }
[ -d "$DATASET_ROOT" ] || {
    echo "dataset.root in $CONFIG doesn't exist on this host: $DATASET_ROOT" >&2
    echo "edit $CONFIG and point dataset.root at your asset folder" >&2
    exit 1
}
DATASET_ROOT="$(cd "$DATASET_ROOT" && pwd)"  # resolve to absolute, for a matching in-container path

docker build -f "${REPO_ROOT}/baselines/depth_anything_3/Dockerfile" -t "$IMAGE" "$REPO_ROOT"

docker run --rm --gpus '"device=1"' \
    -v "${REPO_ROOT}:/workspace" \
    -v "${DATASET_ROOT}:${DATASET_ROOT}" \
    -v "${REPO_ROOT}/.cache/huggingface:/workspace/.cache/huggingface" \
    -w /workspace \
    "$IMAGE" \
    --config "$CONFIG"
