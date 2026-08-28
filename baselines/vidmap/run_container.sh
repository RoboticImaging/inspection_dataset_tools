#!/usr/bin/env bash
# Drop into an interactive shell inside the vidmap baseline image. That's all it does — it runs
# no baseline itself. Once inside:
#
#   python run_sequence.py <config.yaml>          # cwd starts at this baseline's folder
#
# `vidmap` itself is also usable directly from that same shell, e.g.:
#   python -m vidmap.run --input_data ./images --output ./out
# or as separate frontend/mapping steps (`python -m vidmap.frontend`, `python -m vidmap.map`) --
# see vidmap/README.md for the full reference.
#
# Usage:
#   baselines/vidmap/run_container.sh          # shell
#   baselines/vidmap/run_container.sh <cmd...> # run one command, then exit
#
# Env overrides: IMAGE, GPUS (default all, e.g. GPUS='"device=1"'), MOUNTS (space-separated
# extra host paths), SHM_SIZE.
#
# The container sees the host's filesystem at the *same paths* it has outside (/home, /mnt,
# ... are bind-mounted onto themselves), so every absolute path — the repo, config yamls,
# dataset roots — is valid inside without translation, and no rebuild is needed after editing
# code. It runs as your uid/gid (with your supplementary groups, and /etc/passwd + /etc/group
# mounted read-only so the names resolve), so whatever vidmap/run_sequence.py writes is owned by
# you and is deletable from outside the container.
#
# BOTH HF_HOME and TORCH_HOME are mounted here, unlike this repo's other baselines (which only
# ever need one or the other): vidmap's own checkpoint downloads are split across two mechanisms
# -- Depth-Anything-3/MegaLoc go through huggingface_hub (HF_HOME), but ALIKED/RoMaV2/GeoCalib
# land under torch.hub.get_dir() instead (TORCH_HOME) -- confirmed directly from vidmap's own
# source, not assumed. Without both, the ~9 GB of first-run checkpoints (vidmap's own README)
# would partially re-download on every container run instead of persisting.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
IMAGE="${IMAGE:-inspection-baselines/vidmap}"
GPUS="${GPUS:-all}"
SHM_SIZE="${SHM_SIZE:-8g}"

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    echo "image $IMAGE isn't built yet — run ${HERE}/build_container.sh first" >&2
    exit 1
}

# Host trees to replicate. /home covers the repo and most datasets; /mnt and /media cover the
# extra data disks. Anything else (another mount, a nfs export) goes in $MOUNTS.
MOUNT_ARGS=()
for path in /home /mnt /media ${MOUNTS:-}; do
    [ -d "$path" ] && MOUNT_ARGS+=(-v "${path}:${path}")
done

# Your supplementary groups too, so group-readable dataset dirs stay readable in here.
GROUP_ARGS=()
for gid in $(id -G); do GROUP_ARGS+=(--group-add "$gid"); done

# -t only when there's a real terminal, so `run_container.sh <cmd>` also works from a script
# or a pipe (docker refuses to attach a TTY to a non-terminal stdin).
TTY_ARGS=(-i)
[ -t 0 ] && [ -t 1 ] && TTY_ARGS+=(-t)

exec docker run --rm "${TTY_ARGS[@]}" \
    --gpus "$GPUS" \
    --shm-size "$SHM_SIZE" \
    --hostname vidmap-container \
    --user "$(id -u):$(id -g)" \
    "${GROUP_ARGS[@]}" \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    "${MOUNT_ARGS[@]}" \
    -e HOME="$HOME" \
    -e TERM="${TERM:-xterm}" \
    -e HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}" \
    -e TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.cache/torch}" \
    -e PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    -w "$HERE" \
    "$IMAGE" "$@"
